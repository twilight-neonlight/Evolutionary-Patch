"""동등성 검증(재생): 파일럿 기준 실행 기록으로 Main의 IssueRunner를 다시 돌려, 보낸 프롬프트와 결정이 기준본과 같은지 본다.

- LLM: LLMClient는 그대로 쓰고 HTTP 요청만 바꾼다. /chat/completions 요청은 seed가 같은 기록 호출 중 메시지가 같은 것을 찾아
  기록된 응답·토큰·finish_reason을 돌려준다(메시지가 다르면 불일치로 세고, 같은 seed의 다음 기록 응답으로 계속).
  /tokenize는 실제 서버(llm.base_url)로 보낸다(토큰 수가 줄이기 단계·max_tokens를 정하므로).
- 실행기: 저장소 파일 읽기(list_files·read_files)는 실제 Docker 이미지에서, 테스트 실행(run·run_diff)은 기록된 실행 출력
  (result.json·로그)을 같은 상대 경로에서 복사해 재생한다. 후보 실행이면 patch.diff가 기록과 같은지도 본다.
- 비교: 호출 종류별 메시지 바이트 일치, 후보(세대·번호별 patch_hash·유효성·r·g·부모·연산자·무효 사유), 세대별 부모 쌍 기록,
  이슈 결과(상태·최종 후보·종료 사유·세대 수·세대별 유효 수·테스트·위치 출처와 위치 식별자).
결과는 집계만(이슈 ID 없음). 파일럿 기록 경로는 환경 변수 EVOPATCH_PILOT_RUN(또는 --pilot-run)으로 받는다.
실행: python tests/equivalence/replay.py --pilot-run <기준 실행 폴더> [--issues 12] [--out reports/equivalence.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from evopatch.config import Config  # noqa: E402
from evopatch.data_loader import load_instances  # noqa: E402
from evopatch.executor import Executor, cleanup_label  # noqa: E402
from evopatch.llm_client import LLMClient  # noqa: E402
from evopatch.logs import JsonlLog  # noqa: E402
from evopatch.runner import IssueRunner  # noqa: E402
from evopatch.specs import load_regression_pool, specs_from_config  # noqa: E402
from evopatch.types import ExecResult  # noqa: E402

ID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+")
CAND_KEYS = ("generation", "candidate_id", "parent_ids", "op", "orig_op", "gen_born", "valid", "r", "g", "g_total",
             "patch_hash", "invalid_reason", "full_pass")
ISSUE_KEYS = ("status", "terminated_by", "terminated_by_loop", "generations_run", "final_candidate_id", "final_valid",
              "final_r", "final_g", "final_g_total", "valid_by_gen", "gen0_valid_count", "test_source", "location_source",
              "prep_location_identifiers", "prep_v2_location_identifiers", "fallback_reasons", "prep_failure_reason")
GEN_KEYS = ("generation", "n_pairs", "same_candidate", "same_patch_hash", "same_patch_hash_distinct", "redraws")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def call_kind(row: dict) -> str:
    st = row["stage"]
    if st == "prep":
        return f"prep:{row.get('prep_step')}"
    if st == "retry":
        return f"retry:{row.get('retry_of')}"
    return st


class RefRun:
    def __init__(self, run: Path):
        self.run = run
        self.root = run.parent.parent  # 기준 실행 폴더 = <프로젝트>/runs/<이름>
        self.rows = {}
        for r in jl(run / "issues.jsonl"):
            self.rows[r["instance_id"]] = r  # 마지막 행
        self.calls = defaultdict(list)
        for r in jl(run / "llm_calls.jsonl"):
            self.calls[r["attempt_id"]].append(r)
        self._resp = None

    def responses(self) -> dict:
        if self._resp is None:
            need = {c["call_id"] for cs in self.calls.values() for c in cs}
            self._resp = {}
            for line in (self.run / "responses.jsonl").open():
                r = json.loads(line)
                if r["call_id"] in need:
                    m = r["messages"]
                    self._resp[r["call_id"]] = (ast.literal_eval(m) if isinstance(m, str) else m, r["response"])
        return self._resp

    def config(self) -> Config:
        d = Config.load(self.run / "config.snapshot.yaml").as_dict()
        pkg = (d.get("prep", {}).get("lab_fixed") or {}).get("package_dir")
        if pkg:  # 파일럿 설정 키 → Main 키
            d.setdefault("external", {})["repro_tests_dir"] = pkg
        return Config(d)


class Replay:
    """한 이슈의 LLM·실행기 재생 상태와 비교 기록."""

    def __init__(self, pilot: RefRun, iid: str, cfg: Config):
        self.p, self.iid, self.cfg = pilot, iid, cfg
        self.row = pilot.rows[iid]
        self.att = self.row["attempt_id"]
        resp = pilot.responses()
        self.by_seed = defaultdict(list)
        for c in pilot.calls[self.att]:
            if c["call_id"] in resp:
                self.by_seed[c["seed"]].append(c)
        self.used = set()
        self.lock = threading.Lock()
        self.msg = Counter()
        self.msg_bad = Counter()
        self.mismatch_examples = []
        self.exec_bad = []

    # ---- LLM ----
    def chat_response(self, payload: dict) -> httpx.Response:
        resp = self.p.responses()
        with self.lock:
            cands = [c for c in self.by_seed.get(payload["seed"], []) if c["call_id"] not in self.used]
            hit = next((c for c in cands if resp[c["call_id"]][0] == payload["messages"]), None)
            chosen = hit or (cands[0] if cands else None)
            if chosen is None:
                self.msg_bad["unrecorded_call"] += 1
                return httpx.Response(500, text="replay: 기록에 없는 호출")
            self.used.add(chosen["call_id"])
            kind = call_kind(chosen)
            self.msg[kind] += 1
            if hit is None:
                self.msg_bad[kind] += 1
                a, b = json.dumps(resp[chosen["call_id"]][0], ensure_ascii=False), json.dumps(payload["messages"], ensure_ascii=False)
                i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                if len(self.mismatch_examples) < 5:
                    self.mismatch_examples.append({"kind": kind, "offset": i, "recorded": a[max(0, i - 80):i + 80],
                                                   "main": b[max(0, i - 80):i + 80]})
        body = {"choices": [{"message": {"content": resp[chosen["call_id"]][1]}, "finish_reason": chosen.get("finish_reason")}],
                "usage": {"prompt_tokens": chosen.get("prompt_tokens"), "completion_tokens": chosen.get("completion_tokens")}}
        return httpx.Response(200, json=body)

    # ---- 실행기 ----
    def pilot_dir(self, out_dir: Path, run_dir: Path) -> Path:
        rel = Path(out_dir).relative_to(run_dir / "exec" / self.iid)
        return self.p.run / "exec" / self.iid / self.att / Path(*rel.parts[1:])

    def replay_exec(self, out_dir: Path, run_dir: Path, diff: str | None) -> ExecResult:
        src = self.pilot_dir(out_dir, run_dir)
        res_file = src / "result.json"
        if not res_file.exists():
            self.exec_bad.append({"kind": "missing_recorded_exec", "rel": str(Path(*src.parts[-2:]))})
            return ExecResult(repro_pass=False, reg_passed=0, reg_total=0, timed_out=False, output_path=str(out_dir),
                              error="replay: 기록 없음")
        shutil.copytree(src, out_dir, dirs_exist_ok=True)
        if diff is not None and (src / "patch.diff").exists() and (src / "patch.diff").read_text() != diff:
            self.exec_bad.append({"kind": "patch_diff_differs", "rel": str(Path(*src.parts[-2:]))})
        d = json.loads(res_file.read_text())
        d["output_path"] = str(out_dir)
        return ExecResult(**d)


class ReplayExecutor:
    def __init__(self, real: Executor, rp: Replay, run_dir: Path):
        self._real, self._rp, self._run_dir = real, rp, run_dir

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run(self, candidate, prep, out_dir):
        return self._rp.replay_exec(Path(out_dir), self._run_dir, candidate.diff)

    def run_base_repro(self, prep, out_dir):
        return self._rp.replay_exec(Path(out_dir), self._run_dir, None)

    def run_diff(self, diff, *, out_dir, **kw):
        return self._rp.replay_exec(Path(out_dir), self._run_dir, diff or None)


def compare_rows(a: list[dict], b: list[dict], keys, key_fn) -> tuple[int, int, list]:
    ma = {key_fn(r): r for r in a}
    mb = {key_fn(r): r for r in b}
    bad = []
    for k in sorted(set(ma) | set(mb), key=str):
        x, y = ma.get(k), mb.get(k)
        if x is None or y is None:
            bad.append({"key": str(k), "only": "recorded" if y is None else "main"})
            continue
        diff = [f for f in keys if json.dumps(x.get(f), sort_keys=True) != json.dumps(y.get(f), sort_keys=True)]
        if diff:
            bad.append({"key": str(k), "fields": diff})
    return len(set(ma) | set(mb)), len(bad), bad[:5]


def replay_issue(pilot: RefRun, iid: str, work: Path, cfg: Config, views, specs, pool) -> dict:
    rp = Replay(pilot, iid, cfg)
    run_dir = work / f"run_{len(list(work.iterdir()))}"
    run_dir.mkdir()
    llm = LLMClient.from_config(cfg, call_log=JsonlLog(run_dir / "llm_calls.jsonl"), run_id="replay",
                                response_log=JsonlLog(run_dir / "responses.jsonl"))
    real_post = llm._post

    def post(url, payload):
        if url.endswith("/chat/completions"):
            return rp.chat_response(payload), 1
        return real_post(url, payload)

    llm._post = post
    label = f"eqreplay_{int(time.time() * 1000) % 10**9}"
    real_ex = Executor.from_config(cfg, specs, label)
    runner = IssueRunner(cfg=cfg, run_dir=run_dir, run_id="replay", llm=llm,
                         executor=ReplayExecutor(real_ex, rp, run_dir), exec_specs=specs, regression_pool=pool)
    t0 = time.time()
    try:
        row = runner.run_issue(views[iid])
    finally:
        cleanup_label(label)
        llm.close()
    return {"rp": rp, "row": row, "run_dir": run_dir, "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-run", default=os.environ.get("EVOPATCH_PILOT_RUN"))
    ap.add_argument("--issues", type=int, default=12, help="최대 이슈 수(호출 종류마다 5건 이상이 되도록 고름)")
    ap.add_argument("--min-per-kind", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--positions", default=None, help="고른 순서에서 이 순번(1부터, 쉼표)만 실행(불일치 조사용)")
    a = ap.parse_args(argv)
    if not a.pilot_run:
        raise SystemExit("기준 실행 폴더(--pilot-run 또는 EVOPATCH_PILOT_RUN)가 필요합니다")
    pilot = RefRun(Path(a.pilot_run))
    cfg = pilot.config()
    specs = specs_from_config(cfg, pilot.root)
    pool = load_regression_pool(cfg.path_value("data.regression_pool"))
    # 고르기: 호출 종류별 기록 수가 min-per-kind에 모자란 종류를 채우는 이슈부터(이름순, 결정적)
    kinds_by_issue = {}
    for iid, r in sorted(pilot.rows.items()):
        kinds_by_issue[iid] = Counter(call_kind(c) for c in pilot.calls[r["attempt_id"]] if not c.get("error"))
    picked, have = [], Counter()
    all_kinds = sorted({k for c in kinds_by_issue.values() for k in c})
    while len(picked) < a.issues:
        need = [k for k in all_kinds if have[k] < a.min_per_kind]
        if not need:
            break
        best = max((i for i in kinds_by_issue if i not in picked),
                   key=lambda i: (sum(min(kinds_by_issue[i][k], a.min_per_kind - have[k]) for k in need), -len(picked)))
        if not any(kinds_by_issue[best][k] for k in need):
            break
        picked.append(best)
        have.update(kinds_by_issue[best])
    if a.positions:
        picked = [picked[int(x) - 1] for x in a.positions.split(",")]
    views = load_instances(cfg.path_value("data.parquet"), picked)
    work = Path(tempfile.mkdtemp(prefix="eqreplay_"))
    msg, msg_bad, examples, exec_bad = Counter(), Counter(), [], []
    cand_tot = cand_bad = issue_bad = gen_tot = gen_bad = 0
    detail = []
    try:
        for iid in picked:
            out = replay_issue(pilot, iid, work, cfg, views, specs, pool)
            rp, row, rd = out["rp"], out["row"], out["run_dir"]
            msg.update(rp.msg)
            msg_bad.update(rp.msg_bad)
            examples += rp.mismatch_examples
            exec_bad += rp.exec_bad
            unused = [c for cs in rp.by_seed.values() for c in cs if c["call_id"] not in rp.used]
            rec_c = [c for c in jl(pilot.run / "candidates.jsonl") if c["attempt_id"] == rp.att]
            new_c = jl(rd / "candidates.jsonl")
            n, b, ex = compare_rows(rec_c, new_c, CAND_KEYS, lambda r: (r["generation"], r["candidate_id"]))
            cand_tot += n
            cand_bad += b
            rec_g = [g for g in jl(pilot.run / "generations.jsonl") if g["attempt_id"] == rp.att]
            new_g = jl(rd / "generations.jsonl")
            gn, gb, gex = compare_rows(rec_g, new_g, GEN_KEYS, lambda r: r["generation"])
            gen_tot += gn
            gen_bad += gb
            ib = [k for k in ISSUE_KEYS if json.dumps(rp.row.get(k), sort_keys=True) != json.dumps(row.get(k), sort_keys=True)]
            issue_bad += bool(ib)
            detail.append({"index": len(detail), "repo": iid.split("__")[0], "seconds": out["seconds"],
                           "calls": dict(rp.msg), "calls_mismatch": dict(rp.msg_bad), "unused_recorded_calls": len(unused),
                           "candidates": n, "candidate_mismatch": b, "candidate_examples": ex,
                           "generations": gn, "generation_mismatch": gb, "issue_fields_mismatch": ib,
                           "exec_problems": rp.exec_bad[:5]})
            print(f"[{len(detail)}/{len(picked)}] calls {sum(rp.msg.values())} (불일치 {sum(rp.msg_bad.values())}), "
                  f"후보 {n} (불일치 {b}), 세대 {gn} (불일치 {gb}), 이슈 필드 불일치 {ib}, 미사용 기록 {len(unused)}, {out['seconds']}s",
                  flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "issues": len(picked),
           "calls_by_kind": dict(msg), "calls_mismatch_by_kind": dict(msg_bad),
           "candidates_compared": cand_tot, "candidates_mismatch": cand_bad,
           "generations_compared": gen_tot, "generations_mismatch": gen_bad,
           "issues_with_field_mismatch": issue_bad, "exec_problems": len(exec_bad),
           "unused_recorded_calls": sum(d["unused_recorded_calls"] for d in detail),
           "mismatch_examples": examples[:5], "per_issue": detail}
    res["all_equal"] = (not msg_bad and not cand_bad and not gen_bad and not issue_bad and not exec_bad
                        and res["unused_recorded_calls"] == 0)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    text = ID_RE.sub("<ID>", text)  # 결과 파일에 이슈 ID를 남기지 않는다
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_issue", "mismatch_examples")}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["all_equal"] else 1)
