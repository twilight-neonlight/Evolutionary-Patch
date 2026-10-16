"""이슈 단위 실행: 전처리 → (repro_on_base) → 진화 → 최종 패치. 로그는 5절 + 보조 필드, 이어붙이기만.

- 이슈 하나가 끝날 때마다 issues.jsonl에 1줄과 최종 패치 파일을 저장한다(R1).
- 이슈 안의 예외는 잡아서 status=error로 기록하고 다음 이슈로 넘어간다(R2). --resume 때 다시 실행된다(Q8a).
- 모든 로그 행에 attempt_id를 붙여 시도를 구분한다(Q8a·Q8b).
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swebench.harness.grading import test_passed

from .config import Config
from .data_loader import InstanceView
from .config import UndecidedConfigError
from .errors import DiskGuardStop, ImageUnavailable, InfraError, IssueTimeout
from . import feedback as FB
from . import promptdoc
from . import prompts
from . import regmsg as RM
from .edits import SCOPES
from .scope import build_code_view
from .executor import END, START, Executor, cleanup_label, count_containers
from .llm_client import ContextBudgetExceeded, LLMClient
from .logs import JsonlLog, disk_free_row
from .loop import EvolveHooks, evolve
from .operators import OpContext, crossover, initial_population, mutate
from .population import DEFAULT_TIE_BREAK, Individual, check_tie_break
from .prep.base import PrepDeps, make_provider
from .redact import TestFileMasker
from .specs import ExecSpec
from .types import PrepFailure, PrepOutput

# 이슈 결과 상태: ok / prep_failed / budget_exceeded 는 방법의 결과(완료), error 는 재실행 대상
# 완료로 보는 상태(--resume 때 건너뜀). error와 infra_error는 다시 실행한다.
DONE_STATUSES = ("ok", "prep_failed", "budget_exceeded", "issue_timeout", "skipped_undecided", "image_unavailable")
DEF_RE = re.compile(r"^\s*(?:async\s+)?(def|class)\s+(\w+)", re.M)


def _tail_between_markers(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    t = path.read_text(errors="replace")
    t = t.split(START, 1)[-1].split(END, 1)[0]
    return "\n".join(t.strip().splitlines()[-n:])


class IssueRunner:
    def __init__(self, *, cfg: Config, run_dir: Path, run_id: str, llm: LLMClient, executor: Executor,
                 exec_specs: dict[str, ExecSpec], regression_pool: dict[str, list[str]]):
        self.cfg, self.run_dir, self.run_id = cfg, run_dir, run_id
        check_tie_break(cfg.get("algorithm.tie_break", DEFAULT_TIE_BREAK))  # D-8 스위치: 잘못된 값이면 시작 전에 오류
        self.feedback = cfg.get("loop.feedback", "off")  # 2026-10-15 루프 피드백(키 없음 = off, 기존 프롬프트)
        if self.feedback not in FB.MODES:
            raise ValueError(f"loop.feedback={self.feedback!r}: {FB.MODES} 중 하나")
        self.edit_scope = cfg.get("algorithm.edit_scope", "function")  # 수정 범위 S0(function) / S2(file), METHOD 12절
        if self.edit_scope not in SCOPES:
            raise ValueError(f"algorithm.edit_scope={self.edit_scope!r}: {SCOPES} 중 하나")
        self.prompt_doc = promptdoc.load(cfg.get("prompts.path"))  # 프롬프트 고정 문서(경로·SHA-256은 실행 메타에 기록)
        self.llm, self.executor = llm, executor
        self.exec_specs, self.regression_pool = exec_specs, regression_pool
        self.calls_log = llm.call_log
        # E-2: 이슈 시도(attempt_id)별 토큰을 호출 로그에 쓰는 순간 증분 집계한다(파일 전체 재읽기 없음).
        # 집계 규칙은 이전 방식(llm_calls.jsonl에서 attempt_id가 같은 행 전부)과 같다.
        self._token_acc: dict[str, dict] = {}
        self._token_lock = threading.Lock()
        self.calls_log.add_listener(self._on_call_row)
        self.cand_log = JsonlLog(run_dir / "candidates.jsonl")
        self.issue_log = JsonlLog(run_dir / "issues.jsonl")
        self.validity_log = JsonlLog(run_dir / "validity.jsonl")
        self.system_log = JsonlLog(run_dir / "system.jsonl")
        self.gen_log = JsonlLog(run_dir / "generations.jsonl")
        self.run_index = cfg.require("run.run_index")
        self.disk_guard_gb = cfg.get("unattended.disk_guard_gb")
        # 이슈 시도(attempt_id)별 공유 상태. 이슈 병렬 실행 때 첫 사용 시점 생성 경쟁을 막으려고 여기서 미리 만든다.
        self._deadlines: dict[str, float | None] = {}
        self._prep_meta: dict[str, dict] = {}
        self._mask_hits: dict[str, dict] = {}

    # ---------- 보조 ----------
    def disk(self, event: str, **kw) -> None:
        self.system_log.append({"run_id": self.run_id, "event": event, **kw,
                                **disk_free_row({"docker": "/var/lib/docker", "runs": self.run_dir})})

    def _exec_view(self, ind: Individual, reg: list[str], *, iid: str, attempt_id: str) -> dict:
        """변이 프롬프트용 실행 요약. A2: 실패한 회귀 테스트 이름은 요약에 넣지 않고 private/에만 저장한다."""
        ex = ind.exec
        out = Path(ex.output_path)
        failed = []
        sm_path = out / "regression_status.json"
        if sm_path.exists():
            sm = json.loads(sm_path.read_text())
            failed = [t for t in reg if not test_passed(t, sm)]
        if failed:
            priv = self.run_dir / "private" / iid / attempt_id
            priv.mkdir(parents=True, exist_ok=True)
            (priv / f"{ind.cid}.json").write_text(json.dumps({"failed_regressions": failed}, ensure_ascii=False))
        view = {"repro_pass": ex.repro_pass, "repro_statuses": ex.repro_statuses,
                "repro_log_tail": _tail_between_markers(out / "repro.log"),
                "reg_passed": ex.reg_passed, "reg_total": ex.reg_total, "n_failed_regressions": len(failed),
                "no_regression_signal": ex.no_regression_signal, "timed_out": ex.timed_out, "error": ex.error}
        if self.feedback != "off" and failed:  # 2026-10-16 12:30: 회귀 오류 내용(이름·테스트 프레임을 지운 것만, regmsg)
            log_path = out / "regression.log"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            spec = (self.exec_specs or {}).get(iid)
            view["reg_failure"] = RM.extract(log.split(START, 1)[-1].split(END, 1)[0], failed, reg,
                                             getattr(spec, "repo", ""), getattr(spec, "test_files", ()))
        return view

    def _repro_source(self, prep: PrepOutput, attempt_id: str):
        """피드백용 재현 테스트 소스 (경로, 소스, 덧붙인 부분만인지). 제3자 연구 테스트는 덧붙인 블록만(V1과 같은 이유: 기존 테스트 파일 원문 비노출)."""
        if self.feedback == "off":
            return None
        rt = prep.repro_test
        meta = self._prep_meta.get(attempt_id, {})
        off = meta.get("lab_block_offset")
        if meta.get("test_source") == "lab" and isinstance(off, int) and 0 < off < len(rt.content):
            return (rt.path, rt.content[off:], True)
        return (rt.path, rt.content, False)

    def _write_gen0_report(self, prep: PrepOutput, ctx, pop: list[Individual], attempt_id: str) -> str:
        """A6: 세대 0 유효 후보가 N개 미만일 때의 진단 보고서. 선택·종료 규칙과 무관(진단 전용)."""
        info = ctx.gen0_info
        loc_code = "\n".join(l.code for l in prep.locations)
        idents = {}
        for fp in info.get("failed_patches", []):
            if fp["reason"] == "search_not_found" and fp.get("failed_search"):
                m = DEF_RE.search(fp["failed_search"])
                fp["identifier"] = m.group(2) if m else None
                if fp["identifier"]:
                    idents[fp["identifier"]] = None
        shown = {n for n in idents if re.search(rf"^\s*(?:async\s+)?(?:def|class)\s+{n}\b", loc_code, re.M)}
        # D-6: 저장소 전체에서 정의를 찾는 git grep은 컨테이너를 하나 더 띄우므로 설정으로 켰을 때만(기본 off).
        # 끄면 프롬프트에 없던 식별자는 저장소 존재 여부를 따지지 않고 not_shown_unchecked로 둔다.
        find_defs_on = bool(self.cfg.get("diagnostics.gen0_find_defs", False))
        not_shown = [n for n in idents if n not in shown]
        in_repo = self.executor.find_defs(prep.instance_id, not_shown) if find_defs_on and not_shown else set()
        for fp in info.get("failed_patches", []):
            if fp["reason"] == "search_not_found":
                n = fp.get("identifier")
                fp["identifier_class"] = ("no_def_line" if not n else "shown_in_prompt" if n in shown
                                          else "not_shown_unchecked" if not find_defs_on
                                          else "in_repo_not_shown" if n in in_repo else "not_in_repo")
        reasons = Counter(fp["reason"] for fp in info.get("failed_patches", []))
        for c in info.get("calls", []):
            if c.get("missing_reason"):
                reasons[c["missing_reason"]] += c["missing_patches"]
        report = {
            "instance_id": prep.instance_id, "attempt_id": attempt_id,
            "requested": info.get("requested"),
            "parsed": sum(c["parsed"] for c in info.get("calls", [])),
            "valid": sum(i.valid for i in pop),
            "shortfall_retries": max(0, len(info.get("calls", [])) - 1),
            "calls": info.get("calls", []),
            "failure_reasons": dict(reasons),
            "search_not_found_identifier_class": dict(Counter(
                fp.get("identifier_class") for fp in info.get("failed_patches", []) if fp["reason"] == "search_not_found")),
            "failed_patches": info.get("failed_patches", []),
            "prep_locations": [{"file": l.file, "symbol": l.symbol, "lines": [l.start_line, l.end_line]} for l in prep.locations],
            "prep_locations_requested": getattr(self, "_prep_meta", {}).get(attempt_id, {}).get("locations_requested"),
            "prep_locations_found": getattr(self, "_prep_meta", {}).get(attempt_id, {}).get("locations_found", len(prep.locations)),
            "prep_location_kinds": getattr(self, "_prep_meta", {}).get(attempt_id, {}).get("location_kinds"),
            "prep_provider": prep.prep_provider,
            "find_defs_checked": find_defs_on,
            "prompt_tokens_init": next((c["prompt_tokens"] for c in info.get("calls", []) if c["stage"] == "init"), None),
        }
        d = self.run_dir / "gen0_failures"
        d.mkdir(exist_ok=True)
        path = d / f"{prep.instance_id}.{attempt_id}.json"
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        return str(path.relative_to(self.run_dir))

    def _check_deadline(self, attempt_id: str) -> None:
        d = getattr(self, "_deadlines", {}).get(attempt_id)
        if d is not None and time.monotonic() > d:
            raise IssueTimeout("이슈 wall-clock 상한 초과")

    @staticmethod
    def tokens_from_rows(rows, attempt_id: str) -> dict:
        """이전 방식(호출 로그 행 목록에서 attempt_id로 거름). 대조·검증용."""
        acc: dict[str, dict] = {}
        for r in rows:
            if r.get("attempt_id") == attempt_id:
                IssueRunner._add_row(acc, r)
        return IssueRunner._summarize(acc)

    @staticmethod
    def _add_row(by_stage: dict, r: dict) -> None:
        s = by_stage.setdefault(r["stage"], {"prompt": 0, "completion": 0, "calls": 0})
        s["prompt"] += r.get("prompt_tokens") or 0
        s["completion"] += r.get("completion_tokens") or 0
        s["calls"] += 1

    def _on_call_row(self, r: dict) -> None:
        aid = r.get("attempt_id")
        if aid is None:
            return
        with self._token_lock:
            self._add_row(self._token_acc.setdefault(aid, {}), r)

    def _tokens(self, attempt_id: str) -> dict:
        with self._token_lock:
            by_stage = self._token_acc.pop(attempt_id, {})
        return self._summarize(by_stage)

    @staticmethod
    def _summarize(by_stage: dict) -> dict:
        n = sum(v["calls"] for v in by_stage.values())
        return {"by_stage": by_stage, "n_calls": n,
                "prompt": sum(v["prompt"] for v in by_stage.values()),
                "completion": sum(v["completion"] for v in by_stage.values())}

    # ---------- 이슈 하나 ----------
    def run_issue(self, view: InstanceView) -> dict:
        iid = view.instance_id
        attempt_id = uuid.uuid4().hex[:12]
        extra = {"attempt_id": attempt_id}
        t0 = time.monotonic()
        limit = self.cfg.get("unattended.issue_timeout_s")
        self._deadlines[attempt_id] = t0 + limit if limit else None
        provider_name = self.cfg.require("prep.provider")
        row = {"run_id": self.run_id, "instance_id": iid, "prep_provider": provider_name,
               "final_candidate_id": None, "terminated_by": None, "total_prompt_tokens": 0,
               "total_completion_tokens": 0, "wall_clock_s": None, "budget_exceeded": False,
               "status": None, "attempt_id": attempt_id, "error": None, "final_patch_path": None,
               # 2026-10-15: 루프 피드백 방식(이전 런에는 이 필드가 없음 = 기존 코드). 2026-10-16: run.algo_version이 있으면 그 값
               "algo_version": self.cfg.get("run.algo_version") or f"feedback_{self.feedback}"}
        self.disk("issue_start", instance_id=iid, attempt_id=attempt_id)
        exec_root = self.run_dir / "exec" / iid / attempt_id
        provider = make_provider(provider_name, PrepDeps(
            cfg=self.cfg, llm=self.llm, executor=self.executor, exec_specs=self.exec_specs,
            regression_pool=self.regression_pool, run_index=self.run_index, extra_log=extra,
            work_dir=exec_root / "prep"))
        try:
            if self.disk_guard_gb is not None:
                free = disk_free_row({"d": "/var/lib/docker"}).get("d_free_gb", 0)
                if free < self.disk_guard_gb:
                    raise DiskGuardStop(f"디스크 여유 {free}GB < 가드 {self.disk_guard_gb}GB")
            prep = provider.prepare(view)
            self._prep_meta[attempt_id] = dict(provider.meta)  # A6 세대 0 보고서용
            self._check_deadline(attempt_id)
            row.update({f"prep_{k}": v for k, v in provider.meta.items()})
            (self.run_dir / "prep").mkdir(exist_ok=True)
            (self.run_dir / "prep" / f"{iid}.{attempt_id}.json").write_text(
                json.dumps(prep.to_json(), indent=1, ensure_ascii=False))
            row.update(regression_list_size=len(prep.regression_tests),
                       regression_list_empty=not prep.regression_tests,
                       no_regression_signal=not prep.test_files)
            if self.cfg.require("executor.repro_on_base"):
                # 2026-10-16 hybrid_v2: 전처리가 위치 호출 입력용으로 이미 base에서 같은 재현 테스트를 실행했으면 그 결과를 쓴다(다시 실행 안 함)
                b = getattr(provider, "base_result", None)
                from_prep = b is not None
                if b is None:
                    b = self.executor.run_base_repro(prep, exec_root / "base")
                row["repro_on_base"] = {"pass": b.repro_pass, "statuses": b.repro_statuses,
                                        "n_tests": len(b.repro_statuses or {}), "timed_out": b.repro_timed_out,
                                        "error": b.error, **({"from_prep": True} if from_prep else {})}
            res = self._evolve(prep, view, exec_root, extra)
            final = res.final
            all_invalid = not any(res.valid_by_gen)
            row.update(status="ok", final_candidate_id=final.cid,
                       terminated_by="all_invalid" if all_invalid else res.terminated_by,
                       terminated_by_loop=res.terminated_by,  # 루프의 종료 사유(perfect/K)
                       generations_run=res.generations_run, final_valid=final.valid, final_r=final.r,
                       final_g=final.g, final_g_total=final.g_total,
                       valid_by_gen=res.valid_by_gen, executed_by_gen=res.executed_by_gen,
                       cache_hits_by_gen=res.cache_hits_by_gen,
                       any_all_invalid_gen=any(v == 0 for v in res.valid_by_gen),
                       gen0_valid_count=res.gen0_valid_count, gen0_report=res.gen0_report)
            if self.edit_scope != "function":
                row.update(edit_scope=self.edit_scope, code_view=getattr(res, "scope_meta", None))
            patch = final.candidate.diff if final.valid else ""
            row["final_patch_empty"] = not patch.strip()
        except PrepFailure as e:
            row.update(status="prep_failed", prep_failure_reason=e.reason, final_patch_empty=True,
                       **{f"prep_{k}": v for k, v in provider.meta.items()})
            patch = ""
        except ContextBudgetExceeded as e:
            row.update(status="budget_exceeded", budget_exceeded=True, error=str(e))
            patch = ""
        except InfraError as e:  # B5: 풀이 실패가 아님. 재시도 대상
            row.update(status="infra_error", error=f"{type(e).__name__}: {e}"[:500])
            patch = ""
        except ImageUnavailable as e:  # A2: 이미지를 받을 수 없음 → 이 이슈만 빈 패치로 기록하고 다음 이슈로
            row.update(status="image_unavailable", image_unavailable_cause=e.cause, error=str(e)[:500],
                       final_patch_empty=True)
            patch = ""
        except IssueTimeout as e:  # B6
            row.update(status="issue_timeout", error=str(e))
            patch = ""
        except UndecidedConfigError as e:  # B4: 결정되지 않은 값 → 건너뜀
            row.update(status="skipped_undecided", error=str(e)[:500])
            patch = ""
        except Exception as e:  # R2: 이 이슈만 오류로 기록
            tb = traceback.format_exc()
            (self.run_dir / "errors").mkdir(exist_ok=True)
            p = self.run_dir / "errors" / f"{iid}.{attempt_id}.txt"
            p.write_text(tb)
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:500], traceback_path=str(p.relative_to(self.run_dir)))
            patch = ""
        for k in ("test_source", "location_source", "fallback_reasons", "base_exec"):  # hybrid 출처 기록(최상위 필드)
            if k in provider.meta:
                row[k] = provider.meta[k]
        (self.run_dir / "final").mkdir(exist_ok=True)
        fp = self.run_dir / "final" / f"{iid}.{attempt_id}.diff"
        fp.write_text(patch, encoding="utf-8")
        row["final_patch_path"] = str(fp.relative_to(self.run_dir))
        # B6: 이 이슈의 남은 후보 컨테이너 수를 남기고 정리
        row["leftover_containers"] = count_containers(self.executor.run_label, iid) if hasattr(self.executor, "run_label") else None
        if row["leftover_containers"]:
            cleanup_label(self.executor.run_label, iid)
        self._deadlines.pop(attempt_id, None)
        m = getattr(self, "_mask_hits", {}).pop(attempt_id, None)
        if m is not None:  # B2: 프롬프트에서 test_files 경로·파일명을 가린 횟수(형태별)
            row["prompt_mask_hits"] = dict(m)
        tok = self._tokens(attempt_id)
        row.update(total_prompt_tokens=tok["prompt"], total_completion_tokens=tok["completion"],
                   tokens_by_stage=tok["by_stage"], n_llm_calls=tok["n_calls"],
                   wall_clock_s=round(time.monotonic() - t0, 1))
        self.issue_log.append(row)
        self.disk("issue_end", instance_id=iid, attempt_id=attempt_id, status=row["status"])
        return row

    def _evolve(self, prep: PrepOutput, view: InstanceView, exec_root: Path, extra: dict):
        cfg = self.cfg
        iid = prep.instance_id
        files = sorted({l.file for l in prep.locations})
        originals = self.executor.read_files(iid, files)
        masker = TestFileMasker(prep.test_files)  # B2: test_files는 가리는 데만 쓴다(프롬프트에 넣지 않음)
        self._mask_hits[extra["attempt_id"]] = masker.hits
        orig_ok = {k: v for k, v in originals.items() if v is not None}
        code_view, scope_meta = None, None
        if self.edit_scope == "file":  # S2: 위치 파일을 넓게 보여 준다. init 출력 예산(N × init_per_patch)은 그대로 남긴다
            count = lambda t: self.llm.count_tokens([{"role": "user", "content": t}])  # noqa: E731
            empty = self.llm.count_tokens([{"role": "system", "content": prompts.system()}, {"role": "user", "content":
                                          prompts.init_prompt(prep.problem_statement, prep.locations, cfg.require("algorithm.N"),
                                                              "", "file")}])
            budget = (cfg.require("llm.context_window") - cfg.require("llm.max_tokens.init_margin")
                      - cfg.require("algorithm.N") * cfg.require("llm.max_tokens.init_per_patch") - empty)
            code_view, scope_meta = build_code_view(prep.locations, orig_ok, count, budget, prompts.code_context(prep.locations))
        ctx = OpContext(cfg=cfg, llm=self.llm, instance_id=iid, run_index=self.run_index,
                        problem_statement=prep.problem_statement, locations=prep.locations,
                        originals=orig_ok, edit_scope=self.edit_scope, code_view=code_view,
                        validity_log=self.validity_log, extra_log=extra,
                        deadline=getattr(self, "_deadlines", {}).get(extra["attempt_id"]), mask=masker,
                        feedback=self.feedback, repro_source=self._repro_source(prep, extra["attempt_id"]))
        N = cfg.require("algorithm.N")

        # 실행 캐시: 같은 patch_hash는 다시 실행하지 않고 이 이슈 시도 안의 이전 결과를 쓴다(알고리즘 불변).
        exec_cache: dict[str, tuple] = {}
        exec_count: dict[int, int] = {}  # 세대별 실제 실행 수
        hit_count: dict[int, int] = {}  # 세대별 캐시 적중 수

        def evaluate(pop: list[Individual], gen: int) -> None:
            todo = [i for i in pop if i.valid and i.exec is None]
            first: dict[str, Individual] = {}
            for ind in todo:
                if ind.candidate.patch_hash not in exec_cache:
                    first.setdefault(ind.candidate.patch_hash, ind)

            def run(ind):
                ctx.check_deadline()
                ind.exec = self.executor.run(ind.candidate, prep, exec_root / f"g{gen}" / ind.cid)
                ind.exec_view = self._exec_view(ind, prep.regression_tests, iid=iid, attempt_id=extra["attempt_id"])
                ind.meta["cache_hit"] = False
                exec_count[gen] = exec_count.get(gen, 0) + 1
                exec_cache[ind.candidate.patch_hash] = (ind.exec, ind.exec_view, ind.cid)

            with ThreadPoolExecutor(max(1, len(first))) as ex:
                list(ex.map(run, first.values()))
            for ind in todo:
                if ind.exec is None:
                    ex_res, view, src = exec_cache[ind.candidate.patch_hash]
                    ind.exec, ind.exec_view = ex_res, view
                    ind.meta.update(cache_hit=True, cache_hit_from=src)
                    hit_count[gen] = hit_count.get(gen, 0) + 1
            self.disk("generation_end", instance_id=iid, generation=gen, attempt_id=extra["attempt_id"])

        def on_generation(gen, ranked, elites):
            elite_ids = {e.cid for e in elites}
            for ind in ranked:
                ex = ind.exec
                self.cand_log.append({
                    "run_id": self.run_id, "instance_id": iid, "generation": gen, "candidate_id": ind.cid,
                    "parent_ids": list(ind.parent_ids),
                    "op": "elite" if ind.cid in elite_ids else ind.op,
                    "valid": ind.valid, "r": ind.r if ex else None, "g": ind.g if ex else None,
                    "g_total": ind.g_total if ex else None,
                    "exec_timeout": bool(ex and ex.timed_out),
                    "test_output_path": str(Path(ex.output_path).relative_to(self.run_dir)) if ex else None,
                    "patch_hash": ind.candidate.patch_hash,
                    # 보조
                    "attempt_id": extra["attempt_id"], "gen_born": ind.gen_born, "orig_op": ind.op,
                    "invalid_reason": ind.invalid_reason, "full_pass": ind.full_pass(),
                    "repro_statuses": ex.repro_statuses if ex else None,
                    "no_regression_signal": ex.no_regression_signal if ex else None,
                    "exec_error": ex.error if ex else None, "exec_duration_s": ex.duration_s if ex else None,
                    "cache_hit": ind.meta.get("cache_hit"),
                    "patch_lines": ind.candidate.diff.count("\n"), "meta": ind.meta,
                })

        gen0 = {}

        def initial():
            pop = initial_population(ctx)
            gen0["valid"] = sum(i.valid for i in pop)
            if gen0["valid"] < N:  # A6: 진화 루프 전에 세대 0 실패 보고서
                gen0["report"] = self._write_gen0_report(prep, ctx, pop, extra["attempt_id"])
            return pop

        def on_pairs(gen, pairs):  # A3: 같은 부모 비율 (관측용)
            self.gen_log.append({
                "run_id": self.run_id, "instance_id": iid, "attempt_id": extra["attempt_id"], "generation": gen,
                "n_pairs": len(pairs), "same_candidate": sum(1 for a, b, _ in pairs if a is b),
                "same_patch_hash": sum(1 for a, b, _ in pairs if a.candidate.patch_hash == b.candidate.patch_hash),
                "same_patch_hash_distinct": sum(1 for a, b, _ in pairs
                                                if a is not b and a.candidate.patch_hash == b.candidate.patch_hash),
                "redraws": sum(r for _, _, r in pairs)})

        hooks = EvolveHooks(
            initial=initial,
            evaluate=evaluate,
            crossover=lambda a, b, gen, j: crossover(ctx, a, b, gen=gen, idx=j),
            mutate=lambda child, a, b, gen, j: mutate(ctx, child, a, b, gen=gen, idx=j),
            on_generation=on_generation,
            child_workers=N,
            on_pairs=on_pairs,
        )
        res = evolve(instance_id=iid, run_index=self.run_index, N=N, K=cfg.require("algorithm.K"),
                     p_mut=cfg.require("algorithm.p_mut"),
                     max_redraws=cfg.require("algorithm.parent_redraw_max"), hooks=hooks,
                     min_valid_to_evolve=cfg.require("algorithm.min_valid_to_evolve"),
                     tie_break=cfg.get("algorithm.tie_break", DEFAULT_TIE_BREAK))  # D-8 스위치, 키 없음 = newer_first
        res.gen0_valid_count = gen0.get("valid")
        res.gen0_report = gen0.get("report")
        res.scope_meta = scope_meta
        # 보이기 위한 집계(선택·제출 규칙은 그대로)
        res.valid_by_gen = [sum(i.valid for i in ranked) for ranked in res.history]
        res.executed_by_gen = [exec_count.get(g, 0) for g in range(len(res.history))]
        res.cache_hits_by_gen = [hit_count.get(g, 0) for g in range(len(res.history))]
        return res
