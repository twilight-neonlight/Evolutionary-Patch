"""K 어블레이션(METHOD 10절): K=5 실행 기록을 k세대(0..k-1)에서 끊었을 때의 제출 패치로 파생 실행 폴더를 만든다.

끊는 규칙(진화 루프와 같음): 세대 0부터 차례로 보며
  - 그 세대 순위에 완전 통과 후보가 있으면 그중 순위 최상위를 제출하고 멈춘다(조기 종료).
  - 세대 k-1에 이르면 그 세대 순위 최상위를 제출한다.
  - 0세대 분기(유효 0개·1개)로 진화하지 않은 이슈는 원래 제출 그대로.
candidates.jsonl은 세대마다 순위 순서로 기록되어 있다(첫 행 = 최상위). 제출 패치 diff는 그 후보의 실행 출력 폴더의 patch.diff.
출력: runs/<원래 이름>__K<k>/ (config.snapshot.yaml, ids.txt, issues.jsonl(제출·종료 필드만 바꿈), predictions.jsonl,
      generation_complete.json, ablation.json). 이 폴더를 그대로 채점한다:
      python -m evopatch.grading.aggregate --run-dir runs/<원래 이름>__K<k> --by-set --per-issue
실행: python scripts/ablation_k.py --run-dir runs/<K=5 실행> --k 1 [--k 3]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evopatch.config import Config  # noqa: E402
from evopatch.logs import JsonlLog  # noqa: E402
from evopatch.predictions import latest_issue_rows, model_name, write_predictions  # noqa: E402


def truncate(rows_by_gen: dict[int, list[dict]], k: int) -> tuple[dict | None, str]:
    for g in range(k):
        ranked = rows_by_gen.get(g)
        if not ranked:
            break
        perfect = [r for r in ranked if r.get("full_pass")]
        if perfect:
            return perfect[0], "perfect"
        if g == k - 1:
            return ranked[0], "K"
    return None, "ended_before_k"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, action="append", required=True)
    ap.add_argument("--out-root", default=None, help="파생 폴더를 만들 곳(기본: 원래 실행 폴더 옆)")
    a = ap.parse_args(argv)
    run = Path(a.run_dir).resolve()
    cfg = Config.load(run / "config.snapshot.yaml")
    K = cfg.require("algorithm.K")
    ids = [x.strip() for x in (run / "ids.txt").read_text().splitlines() if x.strip()]
    latest = latest_issue_rows(JsonlLog(run / "issues.jsonl"))
    att = {i: r.get("attempt_id") for i, r in latest.items()}
    gens: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in JsonlLog(run / "candidates.jsonl").read():
        if att.get(c["instance_id"]) == c.get("attempt_id"):
            gens[c["instance_id"]][c["generation"]].append(c)
    for k in a.k:
        if not 1 <= k <= K:
            raise SystemExit(f"k={k}: 1..{K}")
        out = (Path(a.out_root) if a.out_root else run.parent) / f"{run.name}__K{k}"
        if out.exists():
            raise SystemExit(f"{out}가 이미 있습니다")
        out.mkdir()
        d = cfg.as_dict()
        d["algorithm"]["K"] = k
        (out / "config.snapshot.yaml").write_text(__import__("yaml").safe_dump(d, allow_unicode=True, sort_keys=False))
        shutil.copy(run / "ids.txt", out / "ids.txt")
        name = model_name(cfg.require("llm.model"), cfg.require("prep.provider"), cfg.require("algorithm.N"), k)
        preds, rows, changed = [], [], 0
        for iid in ids:
            r = dict(latest.get(iid, {"instance_id": iid, "status": "missing"}))
            patch = ""
            evolved = r.get("status") == "ok" and r.get("terminated_by") in ("perfect", "K") and gens.get(iid)
            if evolved:
                cand, why = truncate(gens[iid], k)
                if cand is not None:
                    if cand["candidate_id"] != r.get("final_candidate_id"):
                        changed += 1
                    r.update(final_candidate_id=cand["candidate_id"], terminated_by=why, final_valid=cand["valid"],
                             final_r=cand.get("r"), final_g=cand.get("g"), final_g_total=cand.get("g_total"),
                             generations_run=min(k, r.get("generations_run") or k))
                    p = run / (cand.get("test_output_path") or "") / "patch.diff"
                    patch = p.read_text() if cand["valid"] and cand.get("test_output_path") and p.exists() else ""
                else:  # 원래 실행이 k보다 먼저 끝남(조기 종료) → 원래 제출
                    patch = (run / r["final_patch_path"]).read_text() if r.get("final_patch_path") else ""
            elif r.get("final_patch_path"):
                patch = (run / r["final_patch_path"]).read_text() if (run / r["final_patch_path"]).exists() else ""
            r["ablation_k"] = k
            rows.append(r)
            preds.append({"instance_id": iid, "model_name_or_path": name, "model_patch": patch})
        (out / "issues.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        write_predictions(out / "predictions.jsonl", preds)
        (out / "generation_complete.json").write_text(json.dumps({"derived_from": run.name, "k": k}))
        (out / "ablation.json").write_text(json.dumps({"source_run": run.name, "k": k, "K_source": K,
                                                       "submissions_changed": changed, "n": len(ids)}, indent=1))
        print(f"K={k}: {out} (제출이 바뀐 이슈 {changed}/{len(ids)})")


if __name__ == "__main__":
    main()
