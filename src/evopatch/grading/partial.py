"""완료분 집계 채점 (2026-10-13 사용자 GO S1-4: 시간 상자 안에서 끝나지 않은 런의 완료 이슈만). 집계만 남긴다.

grade_aggregate.py와 같은 하니스·배정·집계 함수를 쓰되, generation_complete.json을 요구하지 않고 완료 상태(DONE_STATUSES)인
이슈만 예측을 메모리에서 만들어 채점한다(predictions.jsonl을 쓰지 않음 → 이후 런 재개·완료에 영향 없음).
출력: <run-dir>/grading_partial_aggregate_<시각>.json (집계만, 이슈 ID 정규식 검사). 이슈별 결과는 저장하지 않는다.

실행: python -m evopatch.grading.partial --run-dir runs/<날짜>_<run_id> [--per-issue]
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq

from ..config import Config
from ..logs import JsonlLog
from ..predictions import build_predictions, latest_issue_rows, model_name
from ..runner import DONE_STATUSES
from .aggregate import aggregate, assert_no_ids, cleanup_containers, dev_set, run_harness, write_per_issue


def main(argv=None, harness=run_harness):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--per-issue", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir).resolve()
    cfg = Config.load(run_dir / "config.snapshot.yaml")
    ids_all = [x.strip() for x in (run_dir / "ids.txt").read_text().splitlines() if x.strip()]
    latest = latest_issue_rows(JsonlLog(run_dir / "issues.jsonl"))
    ids = [i for i in ids_all if latest.get(i, {}).get("status") in DONE_STATUSES]
    name = model_name(cfg.require("llm.model"), cfg.require("prep.provider"), cfg.require("algorithm.N"), cfg.require("algorithm.K"))
    preds = build_predictions(run_dir, ids, name)
    dev = dev_set(Path(a.registry) if a.registry else None)
    assigned, groups = {}, []
    for d in cfg.require("grading.datasets"):
        dids = set(pq.read_table(Path(d["parquet"]).expanduser(), columns=["instance_id"]).column(0).to_pylist())
        mine = [p for p in preds if p["instance_id"] in dids and p["instance_id"] not in assigned]
        for p in mine:
            assigned[p["instance_id"]] = d["name"]
        if mine:
            groups.append((d["name"], d["short"], mine))
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="evopatch_gradep_"))
    reports, left = {}, 0
    tag = f"part{int(t0)}"
    try:
        for dname, short, mine in groups:
            run_id = f"{tag}_{short}"
            try:
                reports[dname] = harness(cfg, dname, mine, tmp, run_id)
            finally:
                left += cleanup_containers(run_id)
        agg = aggregate(run_dir, reports, assigned, ids, dev)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    agg["partial"] = {"n_ids_total": len(ids_all), "n_done_graded": len(ids),
                      "generation_complete": (run_dir / "generation_complete.json").exists()}
    agg["grading"] = {"harness": "swebench 5.0.2 run_evaluation", "max_workers": cfg.require("grading.max_workers"),
                      "dataset_order": [d["name"] for d in cfg.require("grading.datasets")],
                      "elapsed_s": round(time.time() - t0, 1), "temp_dir_removed": not tmp.exists(),
                      "leftover_containers_removed": left}
    text = json.dumps(agg, indent=1, ensure_ascii=False)
    assert_no_ids(text)
    out = run_dir / f"grading_partial_aggregate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(text)
    if a.per_issue:
        write_per_issue(run_dir, reports, ids)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
