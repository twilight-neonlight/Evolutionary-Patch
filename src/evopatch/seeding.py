"""결정적 시드 (A2, : (이슈, 실행 번호, 용도...)에서 hashlib로 파생한다. 파이썬 hash()는 쓰지 않는다."""
from __future__ import annotations

import hashlib
import random


def derive_seed(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF  # vLLM seed는 32비트 양수


def rng(*parts) -> random.Random:
    return random.Random(derive_seed(*parts))
