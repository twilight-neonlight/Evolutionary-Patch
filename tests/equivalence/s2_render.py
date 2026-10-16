"""동등성 검증(S2 렌더링): S2(algorithm.edit_scope=file) 기준 실행 기록의 0세대 첫 호출 메시지와, Main이 같은 입력
(저장된 prep: 이슈 본문·위치·test_files, 원본 파일은 Docker 이미지에서 읽음, 토큰 수는 서버 /tokenize)으로 만든
S2 초기 생성 프롬프트(system + user, 마스킹 후)가 바이트 단위로 같은지 본다. 결과는 집계만(이슈 ID 없음).
기준 실행 폴더는 --s2-run 또는 환경 변수 EVOPATCH_PILOT_S2_RUN.
실행: python tests/equivalence/s2_render.py --s2-run <폴더> [--issues 10] [--out reports/equivalence_s2.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evopatch import promptdoc, prompts  # noqa: E402
from evopatch.config import Config  # noqa: E402
from evopatch.executor import Executor, cleanup_label  # noqa: E402
from evopatch.llm_client import LLMClient  # noqa: E402
from evopatch.logs import JsonlLog  # noqa: E402
from evopatch.redact import TestFileMasker  # noqa: E402
from evopatch.scope import build_code_view  # noqa: E402
from evopatch.specs import specs_from_config  # noqa: E402
from evopatch.types import Location  # noqa: E402

ID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-run", default=os.environ.get("EVOPATCH_PILOT_S2_RUN"))
    ap.add_argument("--issues", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if not a.s2_run:
        raise SystemExit("--s2-run 또는 EVOPATCH_PILOT_S2_RUN이 필요합니다")
    run = Path(a.s2_run)
    cfg = Config(Config.load(run / "config.snapshot.yaml").as_dict())
    promptdoc.load(cfg.get("prompts.path"))
    specs = specs_from_config(cfg, run.parent.parent)
    rows = {}
    for line in (run / "issues.jsonl").open():
        r = json.loads(line)
        rows[r["instance_id"]] = r
    init_calls = {}
    for line in (run / "llm_calls.jsonl").open():
        c = json.loads(line)
        if c["stage"] == "init" and c.get("generation") == 0 and not c.get("error"):
            init_calls[c["attempt_id"]] = c["call_id"]
    picked = []
    for iid, r in sorted(rows.items()):
        if r.get("status") == "ok" and r["attempt_id"] in init_calls and (run / "prep" / f"{iid}.{r['attempt_id']}.json").exists():
            picked.append(iid)
        if len(picked) >= a.issues:
            break
    want = {init_calls[rows[i]["attempt_id"]] for i in picked}
    sent = {}
    for line in (run / "responses.jsonl").open():
        r = json.loads(line)
        if r["call_id"] in want:
            sent[r["call_id"]] = ast.literal_eval(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
    llm = LLMClient.from_config(cfg, call_log=JsonlLog(Path("/dev/null")), run_id="s2render")
    label = f"eqs2_{int(time.time()) % 10**8}"
    ex = Executor.from_config(cfg, specs, label)
    N = cfg.require("algorithm.N")
    same, rows_out = 0, []
    try:
        for k, iid in enumerate(picked):
            d = json.loads((run / "prep" / f"{iid}.{rows[iid]['attempt_id']}.json").read_text())
            locs = [Location(**x) for x in d["locations"]]
            files = sorted({l.file for l in locs})
            originals = {p: v for p, v in ex.read_files(iid, files).items() if v is not None}
            count = lambda t: llm.count_tokens([{"role": "user", "content": t}])  # noqa: E731
            empty = llm.count_tokens([{"role": "system", "content": prompts.system()}, {"role": "user", "content":
                                     prompts.init_prompt(d["problem_statement"], locs, N, "", "file")}])
            budget = (cfg.require("llm.context_window") - cfg.require("llm.max_tokens.init_margin")
                      - N * cfg.require("llm.max_tokens.init_per_patch") - empty)
            view, meta = build_code_view(locs, originals, count, budget, prompts.code_context(locs))
            user = prompts.init_prompt(d["problem_statement"], locs, N, view, "file")
            protected = [d["problem_statement"]] + [l.code for l in locs] + [l.file for l in locs] + [view]
            user = TestFileMasker(d["test_files"])(user, protected, [])
            mine = [{"role": "system", "content": prompts.system()}, {"role": "user", "content": user}]
            rec = sent[init_calls[rows[iid]["attempt_id"]]]
            ok = mine == rec
            same += ok
            rows_out.append({"index": k, "repo": iid.split("__")[0], "equal": ok, "code_view_level": meta.get("level")})
            print(f"[{k + 1}/{len(picked)}] {'일치' if ok else '불일치'} ({meta.get('level')})", flush=True)
    finally:
        cleanup_label(label)
        llm.close()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "issues": len(picked), "equal": same,
           "all_equal": same == len(picked) and len(picked) > 0, "rows": rows_out}
    text = ID_RE.sub("<ID>", json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    sys.exit(0 if main()["all_equal"] else 1)
