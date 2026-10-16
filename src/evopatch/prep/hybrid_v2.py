"""hybrid_v2 전처리 (2026-10-16 최종 파이프라인 후보 지시 3·4): 재현 테스트 확정 → base 실행 1회 → V2 위치 호출 → 위치 확정.

재현 테스트(hybrid와 같음): 제3자 연구 test.py가 기존 테스트 파일에 연결되고 base에서 테스트가 1개 이상 수집되면 제3자 연구 테스트,
  아니면 simple prep(호출 ① 후보 파일 최대 3개, 호출 ② 위치 후보 + 재현 테스트)의 호출 ② 테스트.
V2 위치 호출: simple 호출 ②와 같은 프롬프트·파라미터·시드(prep)·출력 형식에서
  `</issue>`와 `Task:` 사이에 [재현 테스트 소스 + base 실행 출력] 블록만 끼운다. 출력 길이는 앞 3000자 + 뒤 6000자,
  컨텍스트에 안 맞으면 둘 다 절반씩. 응답의 TEST 부분은 쓰지 않고 LOCATIONS만 기존 해석 규칙으로 해석한다.
  - 블록 문장은 prompts/prompts.txt의 V2.* 구간. trim_log와 상수(HEAD_CHARS 등)는 위치 실험에서 정한 값.
base 실행 결과는 위치 호출 입력으로만 쓰고(테스트를 거르거나 바꾸지 않음), runner가 repro_on_base로 그대로 기록한다(base_result).
fallback:
  제3자 연구 테스트 경로: V2 위치 없음 → simple 호출 ②(위치용) → 제3자 연구 clue.json fault_locations(소스 파일만, 최대 3개).
  LLM 테스트 경로: base 실행 실패(실행 오류·시간 초과) 또는 V2 위치 없음 → 같은 이슈의 simple 호출 ② 위치.
  모두 실패 → prep 실패(locations_not_found).
기록(meta): test_source(lab|llm), location_source(V2|simple_prep|lab_fallback), base_exec, fallback_reasons{test, location}, v2_*.
정답 패치·정답 위치·평가용 테스트는 쓰지 않는다(test_files는 마스킹·목록 제외에만).
"""
from __future__ import annotations

from pathlib import Path

from ..config import UndecidedConfigError
from ..data_loader import InstanceView
from ..llm_client import ContextBudgetExceeded
from ..locate import parse_identifier
from ..redact import TestFileMasker
from ..types import PrepFailure, PrepOutput, ReproTest
from .hybrid import NO_ARTIFACT, HybridPrep, issue_dir, load_verdicts, package_format
from .. import promptdoc as D
from .simple import DJANGO, files_prompt, locate_prompt, parse_files_response, parse_response, runner_rule, \
    select_candidate_files

V2_FAIL_BASE, V2_NO_LOC, SIMPLE_NO_LOC, NO_STAGE1 = "base 실행 실패", "V2 위치 없음", "simple 위치 없음", "호출 ① 실패"
# ---- 위치 호출 입력 길이 상수와 출력 자르기 ----
HEAD_CHARS, TAIL_CHARS = 3000, 6000      # base 실행 출력: 앞 3000자 + 뒤 6000자
BLOCK_CHARS = 16000                      # 재현 테스트 블록 상한(넘으면 앞부분만)
CTX_MARGIN = 256                         # 입력 + max_tokens.prep ≤ context_window - 여유


def trim_log(log: str, head: int, tail: int) -> tuple[str, bool]:
    if len(log) <= head + tail:
        return log, False
    omitted = log[head:len(log) - tail].count("\n")
    return log[:head] + D.fill("V2.LOG_OMITTED", N=omitted) + log[len(log) - tail:], True
# ---- 옮긴 부분 끝 ----


def lab_block(target: str, block: str) -> str:
    """제3자 연구 테스트 블록(덧붙인 부분만)."""
    blk = block if len(block) <= BLOCK_CHARS else block[:BLOCK_CHARS] + D.text("V2.TEST_TRUNCATED")
    return D.fill("V2.LAB_TEST", TARGET=target, BLOCK=blk)


def llm_block(path: str, content: str) -> str:
    """LLM 테스트 블록(파일 전체)."""
    blk = content if len(content) <= BLOCK_CHARS else content[:BLOCK_CHARS] + D.text("V2.TEST_TRUNCATED")
    return D.fill("V2.LLM_TEST", PATH=path, BLOCK=blk)


def v2_block(v1: str, log: str, head: int, tail: int) -> tuple[str, bool]:
    """V2 블록 = 테스트 블록 + base 실행 출력(앞 head자 + 뒤 tail자)."""
    tl, cut = trim_log(log, head, tail)
    note = D.fill("V2.OUTPUT_NOTE", HEAD=head, TAIL=tail) if cut else ""
    return v1 + D.fill("V2.OUTPUT", NOTE=note, LOG=tl), cut


class HybridV2Prep(HybridPrep):
    name = "hybrid_v2"

    def __init__(self, deps):
        super().__init__(deps)
        self.base_result = None  # runner가 repro_on_base로 기록(다시 실행하지 않음)

    # ---------- simple 호출 ①·② (simple._llm_call과 같은 프롬프트·시드, 두 부분으로 나눔) ----------
    def _stage1(self, view: InstanceView, usage: list) -> dict:
        d = self.deps
        cfg = d.cfg
        iid = view.instance_id
        paths = cfg.require("prep.simple.repro_paths")
        if view.repo not in paths:
            raise UndecidedConfigError(f"prep.simple.repro_paths에 {view.repo}가 없습니다")
        path = paths[view.repo]
        max_files = cfg.require("prep.simple.candidate_files.max_files")
        if cfg.require("prep.simple.candidate_files.selection") != "llm":
            raise ValueError("prep.simple.candidate_files.selection: 구현된 방식은 llm뿐입니다")
        tf = set(self.test_files(iid))
        masker = TestFileMasker(self.test_files(iid))
        self.meta.update(llm_calls=0)
        repo_files = [p for p in d.executor.list_files(iid) if p not in tf]
        file_list = ""
        if cfg.require("prep.simple.file_list.enabled"):
            file_list = self._file_list(repo_files, cfg.require("prep.simple.file_list.max_tokens"))
        user1 = files_prompt(view, file_list, max_files)
        user1 = masker(user1, [view.problem_statement, file_list])
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
        code_context, kind = self._code_context(cands, sources, cfg.require("prep.simple.code_input.max_tokens"), usage)
        return dict(masker=masker, sources=sources, code_context=code_context, kind=kind, path=path,
                    run_target="swe_repro" if view.repo == DJANGO else path)

    def _user2(self, view: InstanceView, s1: dict, extra: str) -> str:
        """simple 호출 ② 프롬프트. extra가 있으면 `</issue>`와 `Task:` 사이에 끼운다."""
        d = self.deps
        cfg = d.cfg
        user2 = locate_prompt(view, s1["kind"], s1["code_context"], cfg.require("prep.simple.max_locations"),
                              runner_rule(view.repo, s1["path"], d.exec_specs[view.instance_id].test_cmd_prefix), extra)
        return s1["masker"](user2, [view.problem_statement])

    def _call2(self, view: InstanceView, s1: dict, extra: str, step: str, usage: list):
        cfg = self.deps.cfg
        res = self._chat(view, self._user2(view, s1, extra), step=step, max_tokens=cfg.require("llm.max_tokens.prep"),
                         seed_parts=("prep",), usage=usage)
        self._finish_meta(s1["masker"])
        loc_lines, test = parse_response(res.text)
        idents = [p for p in (parse_identifier(l) for l in loc_lines) if p][:cfg.require("prep.simple.max_locations")]
        key = "v2_" if step == "locate_test_V2" else ""
        self.meta.update({f"{key}locations_requested": len(idents), f"{key}finish_reason": res.finish_reason,
                          f"{key}location_identifiers": [f"{f}::{s}" for f, s in idents]})
        if not key:
            self.meta.update(locations_in_candidate_files=sum(f in s1["sources"] for f, _ in idents))
        return test, idents

    def _locs(self, view, idents, s1) -> list:
        return self._resolve_locations(view, idents, s1["sources"]) if idents else []

    # ---------- V2 위치 호출 ----------
    def _v2(self, view: InstanceView, s1: dict, v1: str, log: str, usage: list) -> list:
        cfg = self.deps.cfg
        ctx_window = cfg.require("llm.context_window")
        max_out = cfg.require("llm.max_tokens.prep")
        head, tail = HEAD_CHARS, TAIL_CHARS
        while True:
            v2, cut = v2_block(v1, log, head, tail)
            n = self._tokens(self._user2(view, s1, v2))
            if n + max_out + CTX_MARGIN <= ctx_window or head + tail <= 500:
                break
            head, tail = head // 2, tail // 2
        self.meta.update(v2_attempted=True, v2_input_tokens_counted=n, v2_log_truncated=cut,
                         v2_log_halved=head < HEAD_CHARS, v2_log_chars=len(log))
        try:
            _, idents = self._call2(view, s1, v2, "locate_test_V2", usage)
        except ContextBudgetExceeded as e:
            self.meta.update(v2_error=f"context_budget: {e}"[:200])
            return []
        locs = self._locs(view, idents, s1)
        self.meta.update(v2_locations_found=len(locs))
        return locs

    def _base_exec(self, b, ok: bool) -> dict:
        return {"ok": ok, "n_tests": len(b.repro_statuses or {}), "pass": b.repro_pass, "error": b.error,
                "timed_out": bool(b.timed_out)}

    # ---------- 전처리 ----------
    def prepare(self, view: InstanceView) -> PrepOutput:
        d = self.deps
        iid = view.instance_id
        priority = d.cfg.require("prep.location_priority")
        if priority != "simple_first":
            raise ValueError("hybrid_v2는 prep.location_priority: simple_first만 지원")
        pkg = Path(d.cfg.require("external.repro_tests_dir")).expanduser()
        fmt = package_format(d.cfg)
        idir = issue_dir(pkg, iid, fmt)
        has_artifact = idir.is_dir()
        self.meta.update(lab_verdict=load_verdicts(pkg).get(iid, "없음"), lab_artifact=has_artifact, lab_format=fmt,
                         location_priority=priority, v2_attempted=False)
        lab_test, test_reason = self._lab_test(view, idir) if has_artifact else (None, NO_ARTIFACT)
        usage: list = []
        s1, s1_fail = None, None
        try:
            s1 = self._stage1(view, usage)
        except PrepFailure as e:
            s1_fail = e.reason
            self.meta.update(llm_failure_reason=e.reason)
        locations, location_source, loc_reason = [], None, None

        if lab_test is not None:
            # ----- 제3자 연구 테스트 경로 -----
            repro, test_source = lab_test, "lab"
            b, out_dir = self._lab_base
            self.base_result = b
            self.meta.update(base_exec=self._base_exec(b, True))
            log = (out_dir / "repro.log").read_text(errors="replace") if (out_dir / "repro.log").exists() else ""
            off = self.meta.get("lab_block_offset") or 0
            if s1 is not None:
                block = (llm_block(repro.path, repro.content) if fmt == "new_standalone"  # 독립 파일: 파일 전체("saved as")
                         else lab_block(repro.path, repro.content[off:]))
                locations = self._v2(view, s1, block, log, usage)
                if locations:
                    location_source = "V2"
                else:
                    loc_reason = V2_NO_LOC
                    _, idents = self._call2(view, s1, "", "locate_test", usage)
                    locations = self._locs(view, idents, s1)
                    location_source = "simple_prep" if locations else None
            else:
                loc_reason = NO_STAGE1
            if not locations:
                if s1 is not None:
                    loc_reason = f"{V2_NO_LOC}·{SIMPLE_NO_LOC}"
                locations = self._lab_locations(view, idir)
                location_source = "lab_fallback" if locations else None
        else:
            # ----- LLM 테스트 경로 -----
            if s1 is None:
                self.meta.update(test_source=None, location_source=None, fallback_reasons={"test": test_reason, "location": None})
                raise PrepFailure(s1_fail, usage, self.meta)
            test, idents = self._call2(view, s1, "", "locate_test", usage)
            simple_locs = self._locs(view, idents, s1)
            if test is None:
                self.meta.update(test_source=None, location_source=None, llm_failure_reason="no_test_in_response",
                                 fallback_reasons={"test": test_reason, "location": None})
                raise PrepFailure("no_test_in_response", usage, self.meta)
            repro, test_source = ReproTest(path=s1["path"], content=test, run_target=s1["run_target"]), "llm"
            out_dir = self._work_dir(iid) / "base"
            b = d.executor.run_diff("", instance_id=iid, base_commit=view.base_commit, regression_tests=None,
                                    repro=repro, out_dir=out_dir)
            self.base_result = b
            ok = not (b.error or b.timed_out)
            self.meta.update(base_exec=self._base_exec(b, ok))
            if ok:
                log = (out_dir / "repro.log").read_text(errors="replace") if (out_dir / "repro.log").exists() else ""
                locations = self._v2(view, s1, llm_block(repro.path, repro.content), log, usage)
                if locations:
                    location_source = "V2"
                else:
                    loc_reason = V2_NO_LOC
            else:
                loc_reason = V2_FAIL_BASE
            if not locations:
                locations = simple_locs
                location_source = "simple_prep" if locations else None
                if not locations:
                    loc_reason = f"{loc_reason}·{SIMPLE_NO_LOC}"
                    if has_artifact:  # METHOD 2절: prep ② 위치도 없으면 제3자 연구 산출물의 위치 정보
                        locations = self._lab_locations(view, idir)
                        location_source = "lab_fallback" if locations else None

        self.meta.update(test_source=test_source, location_source=location_source, locations_used=len(locations),
                         fallback_reasons={"test": test_reason, "location": loc_reason})
        if not locations:
            raise PrepFailure("locations_not_found", usage, self.meta)
        return PrepOutput(
            instance_id=iid, repo=view.repo, base_commit=view.base_commit,
            problem_statement=view.problem_statement, locations=locations, repro_test=repro,
            regression_tests=self.regression_tests(iid), test_files=self.test_files(iid),
            prep_usage=usage, prep_provider=self.name)
