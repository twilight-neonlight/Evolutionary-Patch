"""SWE-bench 로더 (3절 데이터 격리).

parquet을 읽을 때부터 허용 컬럼만 지정해 읽는다. 금지 필드는 메모리에도 올라오지 않으며,
InstanceView는 __slots__로 고정돼 다른 속성을 만들 수 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ALLOWED_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "version",
    "environment_setup_commit",
)


@dataclass(frozen=True, slots=True)
class InstanceView:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str
    environment_setup_commit: str


def load_instances(parquet_path, instance_ids: list[str] | None = None) -> dict[str, InstanceView]:
    """parquet 경로 하나 또는 목록(앞의 파일 우선, 예: Verified 다음 Lite). 허용 컬럼만 읽는다."""
    paths = parquet_path if isinstance(parquet_path, (list, tuple)) else [parquet_path]
    views: dict[str, InstanceView] = {}
    for path in paths:
        table = pq.read_table(Path(path).expanduser(), columns=list(ALLOWED_FIELDS))
        for r in table.to_pylist():
            views.setdefault(r["instance_id"], InstanceView(**{k: str(r[k]) for k in ALLOWED_FIELDS}))
    if instance_ids is None:
        return views
    missing = [i for i in instance_ids if i not in views]
    if missing:
        raise KeyError(f"데이터셋에 없는 instance_id: {missing}")
    return {i: views[i] for i in instance_ids}


def read_id_list(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]
