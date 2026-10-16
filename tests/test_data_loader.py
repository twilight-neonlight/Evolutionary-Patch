"""8.1-1 검증: 로더 객체에서 정답 필드 접근 불가."""
import dataclasses
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evopatch.data_loader import ALLOWED_FIELDS, InstanceView, load_instances

FORBIDDEN = ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text")
REAL_PARQUET = Path(__file__).resolve().parents[1] / "data" / "SWE-bench_Verified" / "test.parquet"


@pytest.fixture
def fake_parquet(tmp_path):
    row = {k: f"v_{k}" for k in ALLOWED_FIELDS}
    row["instance_id"] = "x__y-1"
    row.update({k: f"SECRET_{k}" for k in FORBIDDEN})
    p = tmp_path / "d.parquet"
    pq.write_table(pa.Table.from_pylist([row]), p)
    return p


def test_view_has_only_allowed_fields():
    assert tuple(f.name for f in dataclasses.fields(InstanceView)) == ALLOWED_FIELDS


def test_forbidden_fields_inaccessible(fake_parquet):
    v = load_instances(fake_parquet)["x__y-1"]
    for name in FORBIDDEN:
        assert not hasattr(v, name)
        with pytest.raises(AttributeError):
            getattr(v, name)
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            setattr(v, name, "leak")
    assert not hasattr(v, "__dict__")
    assert "SECRET" not in repr(v)


def test_view_is_immutable(fake_parquet):
    v = load_instances(fake_parquet)["x__y-1"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.problem_statement = "x"


def test_only_allowed_columns_read(fake_parquet, monkeypatch):
    import evopatch.data_loader as dl
    seen = {}
    orig = dl.pq.read_table

    def spy(path, columns=None, **kw):
        seen["columns"] = columns
        return orig(path, columns=columns, **kw)

    monkeypatch.setattr(dl.pq, "read_table", spy)
    load_instances(fake_parquet)
    assert seen["columns"] == list(ALLOWED_FIELDS)


def test_unknown_id_raises(fake_parquet):
    with pytest.raises(KeyError):
        load_instances(fake_parquet, ["nope"])


@pytest.mark.skipif(not REAL_PARQUET.exists(), reason="실데이터 없음")
def test_real_verified():
    views = load_instances(REAL_PARQUET)
    assert len(views) == 500
    v = next(iter(views.values()))
    assert all(not hasattr(v, n) for n in FORBIDDEN)


def test_pipeline_source_has_no_forbidden_field_access():
    """src/evopatch 안에서 금지 필드를 키나 속성으로 쓰지 않는다(3절, 5a절)."""
    src = Path(__file__).resolve().parents[1] / "src" / "evopatch"
    pat = re.compile(r"""(\[\s*['"](%s)['"]\s*\])|(\.(%s)\b)|(['"](%s)['"])""" % (("|".join(FORBIDDEN),) * 3))
    hits = []
    for f in src.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f.name}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_pipeline_source_does_not_read_analysis_or_offline():
    """src/evopatch는 analysis/·offline/ 모듈을 import하지 않고, analysis/ 경로나 그 산출물을 읽지 않는다."""
    src = Path(__file__).resolve().parents[1] / "src" / "evopatch"
    imp = re.compile(r"^\s*(from|import)\s+(analysis|offline|grading)\b")
    path = re.compile(r"""["'][^"']*\banalysis\b[^"']*["']|/\s*["']analysis["']""")
    hits = []
    for f in src.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if imp.search(line) or path.search(line):
                hits.append(f"{f.name}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)
