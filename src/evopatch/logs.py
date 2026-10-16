"""JSONL 로그. 항상 이어붙이기만 한다(5절: 덮어쓰지 않음)."""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path


class JsonlLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._listeners: list = []

    def add_listener(self, fn) -> None:
        """이 객체로 쓰는 행마다 fn(row)를 부른다(쓰기와 같은 잠금 안에서, 쓴 직후). 파일을 다시 읽지 않는 증분 집계용."""
        with self._lock:
            self._listeners.append(fn)

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            for fn in self._listeners:
                fn(row)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def disk_free_row(paths: dict[str, str | Path]) -> dict:
    """디스크 여유 기록용 행. paths: {이름: 경로}."""
    row: dict = {"ts": time.time()}
    for name, p in paths.items():
        try:
            u = shutil.disk_usage(p)
            row[f"{name}_free_gb"] = round(u.free / 1e9, 2)
            row[f"{name}_used_pct"] = round(100 * u.used / u.total, 1)
        except OSError as e:
            row[f"{name}_error"] = str(e)
    return row
