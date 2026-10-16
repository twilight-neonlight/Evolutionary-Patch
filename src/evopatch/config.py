"""설정 로드. 실험 값은 설정 파일에만 두고, 미결정(null) 값은 읽는 순간 멈춘다."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class UndecidedConfigError(RuntimeError):
    """설정값이 null(미결정)인데 코드가 그 값을 필요로 할 때."""


class Config:
    def __init__(self, data: dict, path: Path | None = None):
        self._data = data
        self.path = path

    @classmethod
    def load(cls, path: str | Path, local: str | Path | None = None) -> "Config":
        """설정 파일을 읽고, local(서버별 값, git 제외)이 있으면 그 위에 깊게 덮어쓴다."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if local is not None:
            with Path(local).open(encoding="utf-8") as f:
                data = deep_merge(data, yaml.safe_load(f) or {})
        return cls(data, path)

    def require(self, dotted: str) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"설정에 '{dotted}'가 없습니다")
            node = node[key]
        if node is None:
            raise UndecidedConfigError(
                f"설정 '{dotted}'가 미결정(null)입니다. 사용자 결정 후 config에 채워야 합니다."
            )
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self.require(dotted)
        except (KeyError, UndecidedConfigError):
            return default

    def path_value(self, dotted: str):
        """경로 하나면 Path, 목록이면 Path 목록."""
        v = self.require(dotted)
        if isinstance(v, list):
            return [Path(str(x)).expanduser() for x in v]
        return Path(str(v)).expanduser()

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def digest(self) -> str:
        blob = json.dumps(self._data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()


def deep_merge(base: dict, over: dict) -> dict:
    """over의 값으로 base를 덮어쓴 새 dict(중첩 dict는 키별로 합침, 그 밖의 값은 통째로 바꿈)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def sampling_params(cfg: Config, stage: str, retry_of: str | None = None) -> dict:
    """단계별 temperature/top_p. retry는 원래 단계(retry_of)의 값을 그대로 쓴다(B3)."""
    key = retry_of if stage == "retry" else stage
    if stage == "retry" and retry_of is None:
        raise ValueError("retry에는 retry_of가 필요합니다")
    s = cfg.require(f"llm.sampling.{key}")
    return {"temperature": s["temperature"], "top_p": s["top_p"]}
