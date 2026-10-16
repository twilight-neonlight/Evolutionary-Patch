"""hybrid 전처리 (2026-10-15 사용자 지시): 제3자 연구 산출물(읽기 전용)을 먼저 쓰고, 쓸 수 없을 때만 simple prep LLM 호출로 대체한다.

재현 테스트
- 제3자 연구 산출물(complete/<이슈>/test.py)이 있고 "쓸 수 있으면" 그대로 쓴다(lab_fixed 어댑터의 연결 규칙 그대로).
  쓸 수 있음 = 기존 테스트 파일에 연결되고(split_lab_test·block_tests·run_target_for 통과), base commit에서 실행했을 때
  파싱된 테스트가 1개 이상. 정답 패치 정보는 쓰지 않는다. base에서 통과하는 제3자 연구 테스트도 그대로 쓴다.
  base 실행이 오류·시간 초과면 수집 0개로 본다(대체).
- 쓸 수 없거나 산출물이 없으면 simple prep의 재현 테스트(같은 호출, 같은 시드)를 쓴다.
위치 (prep.location_priority)
- simple_first: simple prep 위치 호출을 먼저 하고, 위치를 하나도 못 찾으면 제3자 연구 fault_locations를 쓴다.
- lab_first: 제3자 연구 fault_locations가 있으면 그것을 쓰고, 없으면 simple prep 위치 호출을 쓴다.
- 제3자 연구 위치 → 범위: 진단 스크립트 lab_location_info의 lab_locations와 같은 규칙(줄이 함수 안이면 그 함수 범위, 아니면 끝 이름이
  같은 함수 범위, 둘 다 없으면 그 줄 하나). 차이: 경로 대응에 정답 수정 파일 대신 base commit 파일 목록을 쓴다(같은 경로 또는
  제3자 연구 경로의 꼬리인 파일 중 가장 긴 것, 없으면 제3자 연구 경로를 꼬리로 가진 파일이 하나뿐일 때 그 파일, 여러 개면 버림). 소스 파일만(테스트 파일·저장소에 없는 파일 제외), 목록 순서 앞에서 최대 3개.
  candidate_source_files(파일 단위)는 쓰지 않는다.
실패와 기록
- prep 실패 = 테스트(제3자 연구→LLM) 또는 위치(우선순위대로 두 출처)를 끝까지 대체해도 얻지 못한 경우만.
- meta: test_source / location_source (lab | llm | None), fallback_reasons {test, location}.
- 제3자 연구 산출물은 읽기만 하고 복사하지 않는다(base 실행 로그는 실행기 출력 폴더에 남는다).
"""
from __future__ import annotations

import ast
import json
import tempfile
import warnings
from collections import Counter
from pathlib import Path

from ..data_loader import InstanceView
from ..types import Location, PrepFailure, PrepOutput, ReproTest
from .lab_fixed import LabUnlinkable, block_tests, load_verdicts, run_target_for, split_lab_test
from .simple import DJANGO, SYMPY, SimplePrep

PRIORITIES = ("simple_first", "lab_first")
# 제3자 연구 재현 테스트 패키지 형식(external.repro_tests_format)
#  new_standalone(기본): <이슈>/{generated_test.py, code_context.json, issue_clue.json}. 독립 테스트 파일을 LLM 재현 테스트와
#                        같은 저장소별 고정 경로에 두고 파일 전체를 실행한다.
#  old_appended: complete/<이슈>/{test.py, context.json, clue.json}. 기존 테스트 파일 + 덧붙인 블록(덧붙인 테스트만 실행).
FORMATS = ("new_standalone", "old_appended")
FILES = {"new_standalone": ("generated_test.py", "code_context.json", "issue_clue.json"),
         "old_appended": ("test.py", "context.json", "clue.json")}
PARSE_FAIL, BASE_ERROR, BASE_TIMEOUT, IMPORT_ERROR, ZERO_COLLECTED, NO_PATH = (
    "파싱 불가", "base 실행 오류", "base 시간 초과", "import 오류", "수집 0", "저장 경로 미정")
IMPORT_RE = __import__("re").compile(r"\b(ImportError|ModuleNotFoundError|cannot import name)\b")


def package_format(cfg) -> str:
    fmt = cfg.get("external.repro_tests_format", "new_standalone")
    if fmt not in FORMATS:
        raise ValueError(f"external.repro_tests_format={fmt!r}: {FORMATS} 중 하나")
    return fmt


def issue_dir(pkg: Path, iid: str, fmt: str) -> Path:
    return pkg / iid if fmt == "new_standalone" else pkg / "complete" / iid
NO_ARTIFACT, UNLINKABLE, ZERO_TESTS, LOC_NOT_FOUND = "산출물 없음", "연결 불가", "base에서 테스트 0개", "위치 못 찾음"
LAB_NO_LOC = "제3자 연구 위치 없음"
MAX_LAB_LOCATIONS = 3
TEST_DIRS = ("tests", "test", "testing")


def norm(p: str) -> str:
    p = p.strip()
    for pre in ("/testbed/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def same_file(p: str, g: str) -> bool:
    return p == g or p.endswith("/" + g) or g.endswith("/" + p)


def is_test(p: str) -> bool:
    parts = p.split("/")
    name = parts[-1]
    return any(x in TEST_DIRS for x in parts[:-1]) or name.startswith("test_") \
        or name.endswith("_test.py") or name == "conftest.py"


def map_repo_path(p: str, repo_files: list[str]) -> str | None:
    """제3자 연구 경로 → base commit 파일. 같은 경로 > p의 "/…" 꼬리인 파일 중 가장 긴 것 > p를 꼬리로 가진 파일(유일할 때만)."""
    if p in repo_files:
        return p
    tails = [f for f in repo_files if p.endswith("/" + f)]
    if tails:
        return max(tails, key=len)
    longer = [f for f in repo_files if f.endswith("/" + p)]
    return longer[0] if len(longer) == 1 else None


def function_ranges(source: str) -> list[tuple[str, int, int]] | None:
    """(식별자, 시작 줄, 끝 줄). 진단의 함수 범위 규칙과 같음(모듈 함수와 클래스 안 메서드, 첫 데코레이터 줄부터)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except SyntaxError:
        return None
    out = []

    def walk(nodes, path):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s = min([d.lineno for d in n.decorator_list] + [n.lineno])
                out.append((".".join(path + [n.name]), s, n.end_lineno))
            elif isinstance(n, ast.ClassDef):
                walk(n.body, path + [n.name])

    walk(tree.body, [])
    return out


def lab_fault_locations(fault: list[dict], repo_files: list[str], read) -> tuple[list[Location], dict]:
    """fault_locations → Location 목록(최대 3개)과 기록용 정보. read(paths) -> {path: source|None}."""
    picked, dropped = [], Counter()
    for f in fault:
        p = norm(f.get("file_path") or "")
        if not p.endswith(".py") or is_test(p):
            dropped["test_or_non_py"] += 1
            continue
        path = map_repo_path(p, repo_files)
        if path is None or is_test(path):
            dropped["not_in_repo"] += 1
            continue
        picked.append((f, path))
    srcs = read(sorted({p for _, p in picked})) if picked else {}
    out: list[Location] = []
    kinds, sources_used = [], []
    for f, path in picked:
        if len(out) >= MAX_LAB_LOCATIONS:
            break
        src = srcs.get(path)
        if src is None:
            dropped["read_failed"] += 1
            continue
        lines = src.splitlines()
        fr = function_ranges(src) or []
        ln = f.get("line_no") or 0
        name = (f.get("function_name") or "").split(".")[-1]
        hit = next(((i, s, e) for i, s, e in fr if s <= ln <= e), None)
        if hit is None and name:
            hit = next(((i, s, e) for i, s, e in fr if i.split(".")[-1] == name), None)
        if hit is not None:
            sym, s, e, kind = hit[0], hit[1], hit[2], "lab_function"
        elif 1 <= ln <= len(lines):
            sym, s, e, kind = f"<line {ln}>", ln, ln, "lab_line"
        else:
            dropped["no_range"] += 1
            continue
        loc = Location(file=path, symbol=sym, start_line=s, end_line=e, code="\n".join(lines[s - 1:e]))
        if any((l.file, l.start_line, l.end_line) == (loc.file, loc.start_line, loc.end_line) for l in out):
            dropped["duplicate"] += 1
            continue
        out.append(loc)
        kinds.append(kind)
        sources_used.append(f.get("source"))
    return out, {"lab_fault_locations_n": len(fault), "lab_locations_used": len(out), "lab_location_kinds": kinds,
                 "lab_location_sources": sources_used, "lab_location_dropped": dict(dropped)}


class HybridPrep(SimplePrep):
    name = "hybrid"

    def _work_dir(self, iid: str) -> Path:
        wd = getattr(self.deps, "work_dir", None)
        return Path(wd) if wd is not None else Path(tempfile.mkdtemp(prefix=f"hybrid_{iid}_"))

    def _lab_test(self, view: InstanceView, idir: Path) -> tuple[ReproTest | None, str | None]:
        """(제3자 연구 재현 테스트 | None, 대체 사유 | None). 패키지 형식에 따라 나눈다."""
        if package_format(self.deps.cfg) == "new_standalone":
            return self._standalone_test(view, idir)
        return self._appended_test(view, idir)

    def _standalone_test(self, view: InstanceView, idir: Path) -> tuple[ReproTest | None, str | None]:
        """새 형식: generated_test.py(독립 파일)를 LLM 재현 테스트와 같은 고정 경로에 두고 파일 전체를 base에서 실행한다.
        파싱이 되고 base에서 테스트가 1개 이상 수집되면 쓴다. 아니면 대체 사유를 남기고 None(LLM 테스트로 대체)."""
        d = self.deps
        iid = view.instance_id
        f = idir / FILES["new_standalone"][0]
        if not f.is_file():
            return None, NO_ARTIFACT
        test_py = f.read_text(encoding="utf-8")
        self.meta.update(lab_format="new_standalone", lab_test_lines=test_py.count("\n") + 1, lab_block_offset=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(test_py)
        except (SyntaxError, ValueError) as e:
            self.meta.update(lab_fallback_detail=f"{type(e).__name__}: {e}"[:200])
            return None, PARSE_FAIL
        paths = d.cfg.get("prep.simple.repro_paths") or {}
        if view.repo not in paths:
            return None, NO_PATH
        path = paths[view.repo]
        repro = ReproTest(path=path, content=test_py, run_target="swe_repro" if view.repo == DJANGO else path)
        out_dir = self._work_dir(iid) / "lab_test_base"
        b = d.executor.run_diff("", instance_id=iid, base_commit=view.base_commit, regression_tests=None,
                                repro=repro, out_dir=out_dir)
        self._lab_base = (b, out_dir)
        n = len(b.repro_statuses or {})
        self.meta.update(lab_base_n_tests=n, lab_base_error=b.error, lab_base_timed_out=bool(b.timed_out),
                         lab_base_pass=b.repro_pass, lab_run_scope="file")
        if b.timed_out:
            return None, BASE_TIMEOUT
        if b.error:
            return None, BASE_ERROR
        if n == 0:
            log = (out_dir / "repro.log").read_text(errors="replace") if (out_dir / "repro.log").exists() else ""
            m = IMPORT_RE.search(log)
            if m:
                line = next((l for l in log.splitlines() if IMPORT_RE.search(l)), "")
                self.meta.update(lab_fallback_detail=line.strip()[:200])
                return None, IMPORT_ERROR
            return None, ZERO_COLLECTED
        return repro, None

    def _appended_test(self, view: InstanceView, idir: Path) -> tuple[ReproTest | None, str | None]:
        """기존 형식: 기존 테스트 파일 + 덧붙인 블록."""
        d = self.deps
        iid = view.instance_id
        if not (idir / "test.py").is_file():
            return None, NO_ARTIFACT
        test_py = (idir / "test.py").read_text(encoding="utf-8")
        ctx = json.loads((idir / "context.json").read_text(encoding="utf-8")) if (idir / "context.json").is_file() else {}
        cands = [c["path"] for c in ctx.get("candidate_test_files") or [] if c.get("path")]
        try:
            originals = d.executor.read_files(iid, cands) if cands else {}
            target, block, rank = split_lab_test(test_py, cands, originals)
            funcs, classes = block_tests(originals[target], block)
            rt = run_target_for(view.repo, target, funcs, classes)
        except LabUnlinkable as e:
            self.meta.update(lab_unlinkable=e.reason)
            return None, UNLINKABLE
        repro = ReproTest(path=target, content=test_py, run_target=rt)
        self.meta.update(lab_format="old_appended", lab_target_rank=rank, lab_append_lines=block.count("\n"),
                         lab_block_offset=len(test_py) - len(block),  # 2026-10-15 피드백: 덧붙인 블록 시작 위치
                         lab_block_tests=len(funcs) + len(classes),
                         lab_run_scope="file" if view.repo == SYMPY else "block_tests")
        out_dir = self._work_dir(iid) / "lab_test_base"
        b = d.executor.run_diff("", instance_id=iid, base_commit=view.base_commit, regression_tests=None,
                                repro=repro, out_dir=out_dir)
        self._lab_base = (b, out_dir)  # 2026-10-16 hybrid_v2가 위치 호출 입력·repro_on_base로 다시 씀(hybrid 동작 불변)
        n = len(b.repro_statuses or {})
        self.meta.update(lab_base_n_tests=n, lab_base_error=b.error, lab_base_timed_out=bool(b.timed_out),
                         lab_base_pass=b.repro_pass)
        if b.error or b.timed_out or n == 0:
            return None, ZERO_TESTS
        return repro, None

    def _lab_locations(self, view: InstanceView, idir: Path) -> list[Location]:
        clue = idir / FILES[package_format(self.deps.cfg)][2]
        if not clue.is_file():
            return []
        fault = json.loads(clue.read_text(encoding="utf-8")).get("fault_locations") or []
        if not fault:
            self.meta.update(lab_fault_locations_n=0, lab_locations_used=0)
            return []
        iid = view.instance_id
        repo_files = self.deps.executor.list_files(iid)
        locs, info = lab_fault_locations(fault, repo_files, lambda ps: self.deps.executor.read_files(iid, ps))
        self.meta.update(info)
        return locs

    def prepare(self, view: InstanceView) -> PrepOutput:
        d = self.deps
        iid = view.instance_id
        priority = d.cfg.require("prep.location_priority")
        if priority not in PRIORITIES:
            raise ValueError(f"prep.location_priority={priority!r}: {PRIORITIES} 중 하나")
        pkg = Path(d.cfg.require("external.repro_tests_dir")).expanduser()
        idir = issue_dir(pkg, iid, package_format(d.cfg))
        has_artifact = idir.is_dir()
        self.meta.update(lab_verdict=load_verdicts(pkg).get(iid, "없음"), lab_artifact=has_artifact,
                         location_priority=priority)

        lab_test, test_reason = self._lab_test(view, idir) if has_artifact else (None, NO_ARTIFACT)
        lab_locs = self._lab_locations(view, idir) if has_artifact else []

        need_llm_loc = priority == "simple_first" or not lab_locs
        llm_test, llm_locs, usage, llm_fail = None, [], [], None
        llm_path = llm_rt = None
        if lab_test is None or need_llm_loc:
            try:
                llm_test, idents, sources, usage, llm_path, llm_rt = self._llm_call(view)
                if need_llm_loc and idents:
                    llm_locs = self._resolve_locations(view, idents, sources)
                if llm_test is None:
                    llm_fail = "no_test_in_response"
                elif need_llm_loc and not idents:
                    llm_fail = "no_locations_parsed"
                elif need_llm_loc and not llm_locs:
                    llm_fail = "locations_not_found"
            except PrepFailure as e:  # 1단계 실패·코드 예산 초과: 테스트와 위치 모두 LLM에서 못 얻음
                usage, llm_fail = e.usage, e.reason
        self.meta.update(llm_failure_reason=llm_fail)

        # 재현 테스트: 제3자 연구 → LLM
        if lab_test is not None:
            repro, test_source = lab_test, "lab"
        elif llm_test is not None:
            repro, test_source = ReproTest(path=llm_path, content=llm_test, run_target=llm_rt), "llm"
        else:
            repro, test_source = None, None
        # 위치: 우선순위대로
        order = [("llm", llm_locs), ("lab", lab_locs)] if priority == "simple_first" else [("lab", lab_locs), ("llm", llm_locs)]
        location_source, locations = next(((s, l) for s, l in order if l), (None, []))
        if location_source == "lab" and priority == "simple_first":
            loc_reason = LOC_NOT_FOUND
        elif location_source == "llm" and priority == "lab_first":
            loc_reason = NO_ARTIFACT if not has_artifact else LAB_NO_LOC
        else:
            loc_reason = None
        if location_source is None:
            loc_reason = LOC_NOT_FOUND
        self.meta.update(test_source=test_source, location_source=location_source,
                         fallback_reasons={"test": test_reason, "location": loc_reason})
        self.meta.update(locations_used=len(locations))

        if repro is None:
            raise PrepFailure(llm_fail or "no_test_in_response", usage, self.meta)
        if not locations:
            raise PrepFailure("locations_not_found", usage, self.meta)
        return PrepOutput(
            instance_id=iid, repo=view.repo, base_commit=view.base_commit,
            problem_statement=view.problem_statement, locations=locations, repro_test=repro,
            regression_tests=self.regression_tests(iid), test_files=self.test_files(iid),
            prep_usage=usage, prep_provider=self.name)
