"""완료분 집계 채점(evopatch.grading.partial): 완료 상태 이슈만 채점, predictions.jsonl을 쓰지 않음, 출력에 이슈 ID 없음(모의 하니스)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_grade_aggregate import LITE, VER, make_run  # noqa: E402

pytestmark = pytest.mark.skipif(not (LITE.exists() and VER.exists()), reason="데이터 없음")


def test_partial_grades_done_only(tmp_path, capsys, monkeypatch):
    import evopatch.grading.partial as m
    run, ids = make_run(tmp_path)
    (run / "generation_complete.json").unlink()
    (run / "predictions.jsonl").unlink()
    with (run / "issues.jsonl").open("a") as f:  # 마지막 2개는 미완료(오류) 상태로 덮음
        for i in ids[-2:]:
            f.write(json.dumps({"instance_id": i, "status": "error"}) + "\n")
    seen = []

    def fake_harness(cfg, name, preds, tmp, run_id):
        seen.extend(p["instance_id"] for p in preds)
        return {"submitted_instances": len(preds), "completed_instances": 0, "resolved_instances": 0,
                "unresolved_instances": 0, "empty_patch_instances": len(preds), "error_instances": 0,
                "resolved_ids": [], "unresolved_ids": [], "_returncode": 0}

    monkeypatch.setattr(m, "cleanup_containers", lambda run_id: 0)
    assert m.main(["--run-dir", str(run)], harness=fake_harness) == 0
    assert sorted(seen) == sorted(ids[:-2]) and not (run / "predictions.jsonl").exists()
    out = next(run.glob("grading_partial_aggregate_*.json")).read_text()
    from evopatch.grading.aggregate import ID_RE
    assert not ID_RE.search(out) and not ID_RE.search(capsys.readouterr().out)
    agg = json.loads(out)
    assert agg["partial"] == {"n_ids_total": len(ids), "n_done_graded": len(ids) - 2, "generation_complete": False}
