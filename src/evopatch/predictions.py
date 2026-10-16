"""최종 패치 → 예측 JSONL . 필드: instance_id, model_name_or_path, model_patch."""
from __future__ import annotations

import json
from pathlib import Path

from .logs import JsonlLog


def model_name(model: str, prep_provider: str, N: int, K: int) -> str:
    return f"{model}__evopatch_{prep_provider}_N{N}K{K}".replace("/", "_")


def latest_issue_rows(issues_log: JsonlLog) -> dict[str, dict]:
    """이슈마다 마지막 줄 (Q8a)."""
    rows = {}
    for r in issues_log.read():
        rows[r["instance_id"]] = r
    return rows


def build_predictions(run_dir: Path, ids: list[str], name: str) -> list[dict]:
    rows = latest_issue_rows(JsonlLog(run_dir / "issues.jsonl"))
    preds = []
    for iid in ids:
        patch = ""
        r = rows.get(iid)
        if r and r.get("final_patch_path"):
            patch = (run_dir / r["final_patch_path"]).read_text(encoding="utf-8")
        preds.append({"instance_id": iid, "model_name_or_path": name, "model_patch": patch})
    return preds


def write_predictions(path: Path, preds: list[dict]) -> Path:
    text = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{path}가 이미 있고 내용이 다름 (덮어쓰지 않음)")
        return path
    path.write_text(text, encoding="utf-8")
    return path
