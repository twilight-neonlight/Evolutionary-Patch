"""오프라인 산출물 읽기: 실행 명세(exec_specs)와 Agentless 회귀 풀.

exec_specs는 offline/extract_exec_specs.py가 만든다. 이 모듈은 결과 JSON만 읽는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BASE_PLACEHOLDER = "<BASE_COMMIT>"


@dataclass(frozen=True)
class ExecSpec:
    instance_id: str
    repo: str
    version: str
    image: str
    test_cmd_prefix: str
    pre_commands: tuple[str, ...]
    log_parser: str
    test_files: tuple[str, ...]

    def pre_script(self, base_commit: str) -> str:
        return "\n".join(c.replace(BASE_PLACEHOLDER, base_commit) for c in self.pre_commands)


def load_exec_specs(path) -> dict[str, ExecSpec]:
    """경로 하나 또는 목록. 목록이면 앞의 파일이 우선이고, 겹치는 이슈의 명세가 다르면 중단한다."""
    if isinstance(path, (list, tuple)):
        merged: dict[str, ExecSpec] = {}
        for p in path:
            for iid, spec in load_exec_specs(p).items():
                if iid in merged and merged[iid] != spec:
                    raise ValueError(f"{iid}: exec_specs 파일 사이에 명세가 다름")
                merged.setdefault(iid, spec)
        return merged
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for iid, inst in data["instances"].items():
        rs = data["repo_specs"][inst["spec_key"]]
        out[iid] = ExecSpec(
            instance_id=iid,
            repo=inst["repo"],
            version=inst["version"],
            image=inst["image"],
            test_cmd_prefix=rs["test_cmd_prefix"],
            pre_commands=tuple(rs["pre_commands"]),
            log_parser=rs["log_parser"],
            test_files=tuple(inst["test_files"]),
        )
    return out


def load_regression_pool(path) -> dict[str, list[str]]:
    """이슈별 첫 줄의 tests_passing_in_original_repo . 경로 목록이면 앞의 풀이 우선."""
    if isinstance(path, (list, tuple)):
        merged: dict[str, list[str]] = {}
        for p in path:
            for iid, tests in load_regression_pool(p).items():
                merged.setdefault(iid, tests)
        return merged
    pool: dict[str, list[str]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row["instance_id"]
            if iid in pool:
                continue
            pool[iid] = list(row.get("tests_passing_in_original_repo") or [])
    return pool


def specs_from_config(cfg, root: Path) -> dict[str, ExecSpec]:
    """설정의 data.exec_specs(경로 하나 또는 목록, 프로젝트 루트 기준)를 읽는다."""
    es = cfg.require("data.exec_specs")
    return load_exec_specs([Path(root) / x for x in es] if isinstance(es, list) else Path(root) / es)
