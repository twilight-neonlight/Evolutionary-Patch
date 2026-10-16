"""파일럿 임시 전처리 `simple` -10-12 코드 입력 변경).

LLM 2회 호출 (설계 전제: 코드베이스 + 이슈 → 재현 테스트 + 위치):
  1단계: 이슈 + 파일 트리 → 후보 파일 최대 M개(`prep.simple.candidate_files.max_files`).
  2단계: 후보 파일의 base commit 코드(예산 안에서 전체, 넘으면 라인 번호가 붙은 골격) + 이슈
         → 한 호출에서 `### LOCATIONS`와 `### TEST` (출력 형식은 1회 호출 시절과 같음).
위치는 재현 테스트와 같은 호출의 부산물이다. 위치 해석(locate.resolve_symbol)과 PrepOutput은 그대로.
정렬 평가·재시도·SBFL 없음. 결과는 파일럿 점검용이며 논문 결과로 쓰지 않는다.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings

from .. import promptdoc as D
from ..config import UndecidedConfigError, sampling_params
from ..data_loader import InstanceView
from ..locate import defined_as, parse_identifier, resolve_symbol
from ..redact import TestFileMasker
from ..seeding import derive_seed
from ..types import LLMUsage, PrepFailure, PrepOutput, ReproTest
from .base import PrepProvider

DJANGO, SYMPY = "django/django", "sympy/sympy"

def files_prompt(view, file_list: str, max_files: int) -> str:
    """prep ①: 이슈 + 비테스트 소스 파일 목록 → 후보 파일(문서 PREP.FILES_USER)."""
    return D.fill("PREP.FILES_USER", REPO=view.repo, VERSION=view.version, BASE_COMMIT=view.base_commit,
                  FILE_LIST=file_list, ISSUE=view.problem_statement, MAX_FILES=max_files)


def locate_prompt(view, kind: str, code_context: str, max_locations: int, runner_rule_text: str, v2_block: str = "") -> str:
    """prep ②(위치 + 재현 테스트). v2_block이 있으면 `</issue>`와 `Task:` 사이에 끼운다(V2 위치 호출, 문서 PREP.LOCATE_USER)."""
    return D.fill("PREP.LOCATE_USER", REPO=view.repo, VERSION=view.version, BASE_COMMIT=view.base_commit,
                  CODE_NOTE=D.text(f"PREP.CODE_NOTE.{kind}"), CODE_CONTEXT=code_context, ISSUE=view.problem_statement,
                  V2_BLOCK_PART=D.fill("V2.BLOCK_PART", BLOCK=v2_block) if v2_block else "",
                  RUNNER_RULE=runner_rule_text, MAX_LOCATIONS=max_locations)


# 코드 입력이 예산을 넘을 때 줄이는 순서 (로그 `code_context_reduction`에 시도한 단계를 남긴다)
REDUCTION_ORDER = ("full", "skeleton", "skeleton_drop", "skeleton_truncated")

# prep 호출별 주 구간(기록용)
PREP_SECTIONS = {"files": ["PREP.FILES_USER", "PREP.FILE_LIST"], "locate_test": ["PREP.LOCATE_USER"],
                 "locate_test_V2": ["PREP.LOCATE_USER", "V2.BLOCK_PART", "V2.OUTPUT"]}

TEST_DIRS = {"tests", "test", "testing"}


def is_source_py(path: str) -> bool:
    """비테스트 .py (S1-d). 숨김 경로와 테스트 경로를 뺀다."""
    if not path.endswith(".py"):
        return False
    parts = path.split("/")
    if any(p.startswith(".") for p in parts) or any(p in TEST_DIRS for p in parts[:-1]):
        return False
    base = parts[-1]
    return not (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py")


def file_list_levels(paths: list[str]) -> list[tuple[str, str]]:
    """줄이는 단계: full(경로 전체) → grouped(디렉터리별 파일명) → dirs(디렉터리와 파일 수) → dirs2(깊이 2까지)."""
    src = sorted(p for p in paths if is_source_py(p))
    by_dir: dict[str, list[str]] = {}
    for p in src:
        d, _, b = p.rpartition("/")
        by_dir.setdefault(d or ".", []).append(b)
    full = "\n".join(src)
    grouped = "\n".join(D.fill("PREP.FILE_LIST.GROUPED_ITEM", DIR=d, FILES=" ".join(bs)) for d, bs in sorted(by_dir.items()))
    dirs = "\n".join(D.fill("PREP.FILE_LIST.DIR_ITEM", DIR=d, N=len(bs)) for d, bs in sorted(by_dir.items()))
    top: dict[str, int] = {}
    for d, bs in by_dir.items():
        k = "/".join(d.split("/")[:2])
        top[k] = top.get(k, 0) + len(bs)
    dirs2 = "\n".join(D.fill("PREP.FILE_LIST.DIR_ITEM", DIR=d, N=n) for d, n in sorted(top.items()))
    return [("full", full), ("grouped", grouped), ("dirs", dirs), ("dirs_depth2", dirs2)]


def runner_rule(repo: str, path: str, prefix: str) -> str:
    """재현 테스트 작성·실행 규칙(저장소 종류별 문서 구간)."""
    if repo == DJANGO:
        return D.text("PREP.RUNNER_RULE.django")
    if repo == SYMPY:
        return D.text("PREP.RUNNER_RULE.sympy")
    return D.fill("PREP.RUNNER_RULE.pytest", PATH=path, PREFIX=prefix)


def parse_response(text: str) -> tuple[list[str], str | None]:
    loc_part, test_part = "", ""
    m = re.search(r"###\s*LOCATIONS\s*\n(.*?)###\s*TEST", text, re.S)
    if m:
        loc_part = m.group(1)
    t = re.search(r"###\s*TEST\s*\n.*?```(?:python)?\s*\n(.*?)```", text, re.S)
    if t:
        test_part = t.group(1)
    return [l for l in loc_part.splitlines() if l.strip()], (test_part if test_part.strip() else None)


def parse_files_response(text: str) -> list[str]:
    """1단계 응답의 `### FILES` 아래 줄들(빈 줄 제외). 머리가 없으면 빈 목록."""
    m = re.search(r"###\s*FILES\s*\n(.*)", text, re.S)
    if not m:
        return []
    out = []
    for l in m.group(1).splitlines():
        if l.strip().startswith("###") or l.strip().startswith("```"):
            if out:
                break
            continue
        if l.strip():
            out.append(l)
    return out


def normalize_path(line: str) -> str:
    p = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip("`'\"").strip()
    for pre in ("/testbed/", "./"):
        if p.startswith(pre) and len(p) > len(pre):
            p = p[len(pre):]
            break
    return p


def select_candidate_files(lines: list[str], repo_files: set[str], test_files: set[str],
                           max_files: int) -> tuple[list[str], list[dict]]:
    """앞에서부터 max_files줄만 본다(위치 식별자의 [:max_locations]와 같은 규칙). 버린 줄은 사유와 함께 돌려준다.
    사유: over_limit / not_python_file / test_file / not_in_repo / duplicate."""
    chosen, failures = [], []
    for k, line in enumerate(lines):
        p = normalize_path(line)
        if k >= max_files:
            failures.append({"line": line.strip(), "path": p, "reason": "over_limit"})
            continue
        if not p.endswith(".py"):
            reason = "not_python_file"
        elif p in test_files or not is_source_py(p):
            reason = "test_file"
        elif p not in repo_files:
            reason = "not_in_repo"
        elif p in chosen:
            reason = "duplicate"
        else:
            chosen.append(p)
            continue
        failures.append({"line": line.strip(), "path": p, "reason": reason})
    return chosen, failures


def _header_end(node, lines: list[str]) -> int:
    """def/class 머리(시그니처)의 마지막 줄 번호. 본문 첫 문장 직전까지에서 빈 줄·주석을 뺀다."""
    first_body = node.body[0].lineno
    if first_body <= node.lineno:  # `def f(): pass`
        return node.lineno
    end = first_body - 1
    while end > node.lineno and (not lines[end - 1].strip() or lines[end - 1].strip().startswith("#")):
        end -= 1
    return end


def code_skeleton(source: str) -> str:
    """라인 번호가 붙은 골격: 모듈·클래스 수준의 클래스와 함수(데코레이터, 시그니처). 함수 본문은 생략하고 범위만 적는다.
    함수 안에 정의된 함수는 넣지 않는다(위치 해석 대상이 아님). 문법 오류 파일은 def/class/데코레이터 줄만."""
    lines = source.splitlines()

    def num(i: int) -> str:
        return f"{i:>5}| {lines[i - 1]}"

    try:
        with warnings.catch_warnings():  # 원본 코드의 SyntaxWarning(잘못된 escape 등)은 로그에 내지 않는다
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(num(i) for i, l in enumerate(lines, 1)
                         if re.match(r"\s*(@|def\s|async\s+def\s|class\s)", l))
    out: list[str] = []

    def walk(body):
        for n in body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            out.extend(num(i) for i in range(start, _header_end(n, lines) + 1))
            indent = re.match(r"\s*", lines[n.lineno - 1]).group(0) + "    "
            if isinstance(n, ast.ClassDef):
                walk(n.body)
                out.append(f"{'':>5}| {indent}" + D.fill("PREP.SKELETON.END_OF_CLASS", NAME=n.name, START=start, END=n.end_lineno))
            else:
                out.append(f"{'':>5}| {indent}" + D.fill("PREP.SKELETON.BODY_OMITTED", START=start, END=n.end_lineno))

    walk(tree.body)
    return "\n".join(out)


def render_code(files: list[tuple[str, str]], kind: str) -> str:
    """files: (경로, 내용). kind=full이면 원문, skeleton이면 내용이 이미 골격."""
    parts = []
    for path, text in files:
        n = text.count("\n") + 1 if text else 0
        parts.append(D.fill("PREP.CODE_FILE.full", PATH=path, N=n, TEXT=text) if kind == "full"
                     else D.fill("PREP.CODE_FILE.skeleton", PATH=path, TEXT=text))
    return D.text("PREP.CODE_FILE.SEPARATOR").join(parts)


class SimplePrep(PrepProvider):
    name = "simple"

    def prepare(self, view: InstanceView) -> PrepOutput:
        test, idents, sources, usage, path, run_target = self._llm_call(view)
        if test is None:
            raise PrepFailure("no_test_in_response", usage, self.meta)
        if not idents:
            raise PrepFailure("no_locations_parsed", usage, self.meta)
        locations = self._resolve_locations(view, idents, sources)
        if not locations:
            raise PrepFailure("locations_not_found", usage, self.meta)  # Q5: 미해결로 집계
        iid = view.instance_id
        return PrepOutput(
            instance_id=iid, repo=view.repo, base_commit=view.base_commit,
            problem_statement=view.problem_statement, locations=locations,
            repro_test=ReproTest(path=path, content=test, run_target=run_target),
            regression_tests=self.regression_tests(iid),
            test_files=self.test_files(iid), prep_usage=usage, prep_provider=self.name)

    def _llm_call(self, view: InstanceView):
        """LLM 2회 호출(1단계 후보 파일, 2단계 위치+테스트). (테스트 원문|None, 위치 식별자, 후보 원문, 사용량, 경로, 실행 대상).
        1단계 실패·코드 예산 초과는 PrepFailure. 테스트 없음·식별자 없음 판정은 호출한 쪽이 한다(2026-10-15 hybrid용 분리, 동작 불변)."""
        d = self.deps
        cfg = d.cfg
        iid = view.instance_id
        spec = d.exec_specs[iid]
        paths = cfg.require("prep.simple.repro_paths")
        if view.repo not in paths:  # 정해진 경로가 없는 저장소: 결정되지 않은 값 (무인 모드에서 skipped_undecided)
            raise UndecidedConfigError(f"prep.simple.repro_paths에 {view.repo}가 없습니다")
        path = paths[view.repo]
        run_target = "swe_repro" if view.repo == DJANGO else path
        max_loc = cfg.require("prep.simple.max_locations")
        max_files = cfg.require("prep.simple.candidate_files.max_files")
        selection = cfg.require("prep.simple.candidate_files.selection")
        if selection != "llm":
            raise ValueError(f"prep.simple.candidate_files.selection={selection!r}: 구현된 방식은 llm뿐입니다")
        code_budget = cfg.require("prep.simple.code_input.max_tokens")
        tf = set(self.test_files(iid))
        # B2: test_files 경로·파일명 마스킹. 두 호출이 같은 마스커를 쓴다(hits 합산)
        masker = TestFileMasker(self.test_files(iid))
        usage: list[LLMUsage] = []
        self.meta.update(llm_calls=0)

        # ---------- 1단계: 이슈 + 파일 트리 → 후보 파일 ----------
        repo_files = [p for p in d.executor.list_files(iid) if p not in tf]  # B2: test_files는 목록에서 뺀다
        file_list = ""
        if cfg.require("prep.simple.file_list.enabled"):
            file_list = self._file_list(repo_files, cfg.require("prep.simple.file_list.max_tokens"))
        user1 = files_prompt(view, file_list, max_files)
        user1 = masker(user1, [view.problem_statement, file_list])  # 파일 목록은 test_files를 뺀 base commit 경로
        res1 = self._chat(view, user1, step="files", max_tokens=cfg.require("llm.max_tokens.prep_files"),
                          seed_parts=("prep", "files"), usage=usage)
        lines = parse_files_response(res1.text)
        cands, cand_fail = select_candidate_files(lines, set(repo_files), tf, max_files)
        self.meta.update(finish_reason_files=res1.finish_reason, candidate_files_requested=len(lines))
        sources: dict[str, str] = {}
        if cands:
            read = d.executor.read_files(iid, cands)
            for p in list(cands):
                if read.get(p) is None:
                    cand_fail.append({"line": p, "path": p, "reason": "read_failed"})
                    cands.remove(p)
                else:
                    sources[p] = read[p]
        self.meta.update(candidate_files=list(cands), candidate_file_failures=cand_fail)
        if not lines:
            self._finish_meta(masker)
            raise PrepFailure("no_candidate_files_parsed", usage, self.meta)
        if not cands:
            self._finish_meta(masker)
            raise PrepFailure("candidate_files_not_found", usage, self.meta)

        # ---------- 2단계: 코드 + 이슈 → 위치 + 재현 테스트 ----------
        code_context, kind = self._code_context(cands, sources, code_budget, usage)
        user2 = locate_prompt(view, kind, code_context, max_loc, runner_rule(view.repo, path, spec.test_cmd_prefix))
        user2 = masker(user2, [view.problem_statement])  # A-2: 코드 입력도 가린다(이슈 본문만 보호 구간)
        res = self._chat(view, user2, step="locate_test", max_tokens=cfg.require("llm.max_tokens.prep"),
                         seed_parts=("prep",), usage=usage)
        self._finish_meta(masker)
        loc_lines, test = parse_response(res.text)
        idents = [p for p in (parse_identifier(l) for l in loc_lines) if p][:max_loc]
        self.meta.update(locations_requested=len(idents), finish_reason=res.finish_reason,
                         location_identifiers=[f"{f}::{s}" for f, s in idents],
                         locations_in_candidate_files=sum(f in sources for f, _ in idents))
        return test, idents, sources, usage, path, run_target

    def _resolve_locations(self, view: InstanceView, idents: list[tuple[str, str]], sources: dict[str, str]) -> list:
        d = self.deps
        cfg = d.cfg
        iid = view.instance_id
        need = sorted({f for f, _ in idents} - set(sources))
        files = {f: sources.get(f) for f, _ in idents}
        if need:
            files.update(d.executor.read_files(iid, need))
        allow_class = cfg.require("prep.simple.allow_class_locations")
        locations, missing, kinds, class_tokens, failures = [], [], [], [], []
        for f, s in idents:
            r = resolve_symbol(files[f], f, s, allow_class=allow_class) if files.get(f) is not None else None
            if r is None:
                missing.append(f"{f}::{s}")
                failures.append(self._failure_reason(files.get(f), f, s, allow_class))
                continue
            loc, kind_ = r
            if loc in locations:
                continue
            locations.append(loc)
            kinds.append(kind_)
            if kind_ == "class":  # S3: 클래스 본문 토큰 수 기록
                class_tokens.append(d.llm.count_tokens([{"role": "user", "content": loc.code}]))
        self.meta.update(location_range_rule="from_first_decorator",  # 결정 3: 함수·메서드 범위는 첫 데코레이터 줄부터
                         locations_found=len(locations), locations_missing=missing, location_failures=failures,
                         location_kinds=kinds,
                         class_locations=kinds.count("class"), class_location_tokens=class_tokens,
                         location_code_lines=[l.end_line - l.start_line + 1 for l in locations])
        return locations

    def _chat(self, view: InstanceView, user: str, *, step: str, max_tokens: int, seed_parts: tuple, usage: list):
        d = self.deps
        msgs = [{"role": "user", "content": user}]
        sha = hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()
        res = d.llm.chat(msgs, stage="prep", instance_id=view.instance_id,
                         generation=None, max_tokens=max_tokens,
                         seed=derive_seed(view.instance_id, d.run_index, "llm", *seed_parts),
                         extra_log={**d.extra_log, "prep_step": step, "prompt_sha256": sha,
                                    "prompt_sections": PREP_SECTIONS.get(step, [])}, **sampling_params(d.cfg, "prep"))
        usage.append(LLMUsage(model=d.llm.model, stage="prep", prompt_tokens=res.prompt_tokens,
                              completion_tokens=res.completion_tokens))
        self.meta["llm_calls"] = self.meta.get("llm_calls", 0) + 1
        return res

    def _finish_meta(self, masker: TestFileMasker) -> None:
        self.meta.update(prompt_mask_hits=dict(masker.hits))

    def _tokens(self, text: str) -> int:
        return self.deps.llm.count_tokens([{"role": "user", "content": text}])

    def _code_context(self, cands: list[str], sources: dict[str, str], budget: int,
                      usage: list[LLMUsage]) -> tuple[str, str]:
        """예산(토큰) 안의 코드 입력. 줄이는 순서(REDUCTION_ORDER):
        full(후보 전체 원문) → skeleton(후보 전체 골격) → skeleton_drop(우선순위가 낮은 후보부터 하나씩 빼되 1개는 남김)
        → skeleton_truncated(1순위 골격을 앞에서부터 예산까지만). 시도한 단계와 토큰 수를 meta에 남긴다."""
        tried: list[dict] = []

        def attempt(level: str, kind: str, files: list[tuple[str, str]]):
            text = render_code(files, kind)
            n = self._tokens(text)
            tried.append({"level": level, "files": [p for p, _ in files], "tokens": n, "fits": n <= budget})
            return text, n

        def done(level: str, kind: str, text: str, n: int, files: list[str]):
            self.meta.update(code_context_kind=kind, code_context_level=level, code_context_tokens=n,
                             code_context_budget=budget, code_context_files=files,
                             code_context_dropped_files=[p for p in cands if p not in files],
                             code_context_reduction=tried)
            return text, kind

        full = [(p, sources[p]) for p in cands]
        text, n = attempt("full", "full", full)
        if n <= budget:
            return done("full", "full", text, n, cands)
        skel = [(p, code_skeleton(sources[p])) for p in cands]
        text, n = attempt("skeleton", "skeleton", skel)
        if n <= budget:
            return done("skeleton", "skeleton", text, n, cands)
        for k in range(len(skel) - 1, 0, -1):
            text, n = attempt("skeleton_drop", "skeleton", skel[:k])
            if n <= budget:
                return done("skeleton_drop", "skeleton", text, n, cands[:k])
        p0, s0 = skel[0]
        sl = s0.splitlines()
        lo, hi = 0, len(sl)  # 예산 안에 드는 최대 줄 수(이분 탐색)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            t = render_code([(p0, "\n".join(sl[:mid]) + f"\n{'':>5}| " + D.text("PREP.SKELETON.TRUNCATED"))], "skeleton")
            if self._tokens(t) <= budget:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            self.meta.update(code_context_reduction=tried, code_context_budget=budget)
            raise PrepFailure("code_context_over_budget", usage, self.meta)
        text, n = attempt("skeleton_truncated", "skeleton",
                          [(p0, "\n".join(sl[:lo]) + f"\n{'':>5}| " + D.text("PREP.SKELETON.TRUNCATED"))])
        return done("skeleton_truncated", "skeleton", text, n, [p0])

    @staticmethod
    def _failure_reason(source, f: str, s: str, allow_class: bool) -> dict:
        """식별자를 못 찾은 원인 (로그 전용, 해석 규칙과 결과에는 영향 없음).
        file_not_found / parse_error / class_not_allowed / qualname_mismatch(같은 이름이 다른 자격으로 정의됨,
        1곳이면 found_as에 그 위치, 여러 곳이면 ambiguous=True) / symbol_not_in_file."""
        ident = f"{f}::{s}"
        if source is None:
            return {"identifier": ident, "reason": "file_not_found"}
        found = defined_as(source, s.split(".")[-1])
        if found is None:
            return {"identifier": ident, "reason": "parse_error"}
        if s in [q for q, kind in found] and not allow_class:
            return {"identifier": ident, "reason": "class_not_allowed"}
        if found:
            return {"identifier": ident, "reason": "qualname_mismatch", "found_as": [f"{q} ({k})" for q, k in found][:5],
                    "ambiguous": len(found) > 1}
        return {"identifier": ident, "reason": "symbol_not_in_file"}

    def _file_list(self, paths: list[str], budget: int) -> str:
        """base commit의 git ls-files(test_files 제외)에서 비테스트 .py 목록을 만들고, 예산(토큰) 안에 들 때까지 줄인다."""
        levels = file_list_levels(paths)
        n_src = sum(1 for p in paths if is_source_py(p))
        for name, text in levels:
            n = self.deps.llm.count_tokens([{"role": "user", "content": text}])
            if n <= budget:
                self.meta.update(file_list_level=name, file_list_tokens=n, file_list_n_files=n_src)
                return D.fill("PREP.FILE_LIST", HEADER=D.text(f"PREP.FILE_LIST.HEADER.{name}"), LIST=text)
        self.meta.update(file_list_level="none", file_list_tokens=0, file_list_n_files=n_src)
        return ""
