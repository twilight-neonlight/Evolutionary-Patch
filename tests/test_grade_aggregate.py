"""집계 전용 채점(evopatch.grading.aggregate): 이슈별 결과를 남기지 않고, 출력·파일에 이슈 ID 패턴이 없다(모의 하니스)."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LITE = ROOT / "data" / "SWE-bench_Lite" / "test.parquet"
VER = ROOT / "data" / "SWE-bench_Verified" / "test.parquet"
pytestmark = pytest.mark.skipif(not (LITE.exists() and VER.exists()), reason="데이터 없음")


def load_mod():
    import evopatch.grading.aggregate as m
    return m


def make_run(tmp_path, n=12):
    import pyarrow.parquet as pq
    ids = sorted(pq.read_table(LITE, columns=["instance_id"]).column(0).to_pylist())[:n]
    run = tmp_path / "20990101_fake"
    run.mkdir()
    cfg = yaml.safe_load((ROOT / "tests" / "data" / "test_config.yaml").read_text())
    cfg["grading"]["datasets"] = [{"name": "SWE-bench/SWE-bench_Verified", "short": "verified", "parquet": str(VER)},
                                  {"name": "SWE-bench/SWE-bench_Lite", "short": "lite", "parquet": str(LITE)}]
    (run / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    (run / "ids.txt").write_text("\n".join(ids) + "\n")
    (run / "generation_complete.json").write_text("{}")
    rows, preds = [], []
    for k, i in enumerate(ids):
        st = "prep_failed" if k % 6 == 5 else "ok"
        rows.append({"instance_id": i, "status": st, "terminated_by": None if st != "ok" else ("perfect" if k % 4 == 0 else "K"),
                     "final_r": int(k % 4 == 0), "final_g": 3, "final_g_total": 3, "attempt_id": f"a{k}"})
        preds.append({"instance_id": i, "model_name_or_path": "m", "model_patch": "" if st != "ok" else "diff x\n"})
    (run / "issues.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    return run, ids


def test_aggregate_only_no_ids_and_temp_removed(tmp_path, capsys, monkeypatch):
    m = load_mod()
    run, ids = make_run(tmp_path)
    seen = {}

    def fake_harness(cfg, name, preds, tmp, run_id):
        seen.setdefault("tmp", tmp)
        (tmp / "logs").mkdir(exist_ok=True)
        (tmp / "logs" / f"{run_id}.log").write_text("\n".join(p["instance_id"] for p in preds))  # 이슈별 로그(임시)
        nonempty = [p["instance_id"] for p in preds if p["model_patch"]]
        res = nonempty[::2]
        return {"submitted_instances": len(preds), "completed_instances": len(nonempty), "resolved_instances": len(res),
                "unresolved_instances": len(nonempty) - len(res), "empty_patch_instances": len(preds) - len(nonempty),
                "error_instances": 0, "resolved_ids": res, "unresolved_ids": nonempty[1::2], "_returncode": 0}

    monkeypatch.setattr(m, "cleanup_containers", lambda run_id: 0)
    assert m.main(["--run-dir", str(run)], harness=fake_harness) == 0
    out = capsys.readouterr().out
    text = (run / "grading_aggregate.json").read_text()
    assert not m.ID_RE.search(out) and not m.ID_RE.search(text)
    assert not seen["tmp"].exists()  # 임시 디렉터리(이슈별 로그·report) 삭제
    assert not (run / "grading").exists() and not (run / "grading_private").exists()
    assert sorted(p.name for p in run.iterdir()) == sorted(
        ["config.snapshot.yaml", "ids.txt", "generation_complete.json", "issues.jsonl", "predictions.jsonl", "grading_aggregate.json"])
    agg = json.loads(text)
    assert agg["a_total"]["denominator"] == len(ids) and agg["grading"]["temp_dir_removed"] is True
    # 재채점 금지: 결과 파일이 있으면 다시 돌지 않는다
    with pytest.raises(SystemExit):
        m.main(["--run-dir", str(run)], harness=fake_harness)


def test_merge_small_and_two_by_two():
    m = load_mod()
    cells = m.merge_small([["a", 2, 1], ["b", 10, 4], ["c", 1, 0]])
    assert [c["n"] for c in cells] == [12, 1] or all(c["n"] >= 3 for c in cells)
    assert all(c["n"] >= 3 for c in cells) and sum(c["n"] for c in cells) == 13
    t = m.two_by_two("r=1", list("abcdefgh"), {x: x in "ab" for x in "abcdefgh"}, set("ac"))
    assert t["rows"][0]["label"].endswith("(n<3 합침)")  # 참 행 n=2 → 두 행 합침
    t = m.two_by_two("r=1", list("abcdefghijkl"), {x: x in "abcde" for x in "abcdefghijkl"}, set("afghi"))
    assert t["rows"][0]["resolved"] == "resolved/unresolved 칸 n<3 합침" and t["rows"][1]["resolved"] == 4


def test_assert_no_ids_blocks_output():
    m = load_mod()
    with pytest.raises(SystemExit):
        m.assert_no_ids('{"x": "owner__repo-12345"}')
    m.assert_no_ids('{"repo": "django/django", "n": 3}')
