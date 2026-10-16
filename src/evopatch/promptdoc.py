"""프롬프트 고정 문서(prompts/prompts.txt) 읽기·채우기.

코드는 프롬프트 문장을 갖지 않는다. 여기서는 구간을 이름으로 읽고({{자리표시자}}를 한 번에 채우고) 돌려줄 뿐이다.
문서 형식: 머리 주석(첫 구분선 앞의 # 줄) → `===== 이름 =====` 구간들 → `===== END =====`.
구간 내용 = 구분선 다음 줄부터 다음 구분선 바로 앞 줄까지를 "\n"으로 이은 것(마지막 줄 뒤 줄바꿈 없음).

호출 기록용: 스레드마다 지금 만드는 프롬프트에 쓴 구간 이름을 모은다(begin_record / end_record).
"""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "prompts" / "prompts.txt"
SEP = re.compile(r"^===== (\S+) =====$")
PH = re.compile(r"\{\{([A-Z_0-9]+)\}\}")


class PromptDoc:
    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.sections: dict[str, str] = {}
        name, buf = None, []
        for ln in raw.decode("utf-8").split("\n"):
            m = SEP.match(ln)
            if m:
                if name is not None:
                    if name in self.sections:
                        raise ValueError(f"프롬프트 문서에 같은 구간이 두 번: {name}")
                    self.sections[name] = "\n".join(buf)
                name, buf = m.group(1), []
                if name == "END":
                    break
            elif name is not None:
                buf.append(ln)
        if name != "END":
            raise ValueError("프롬프트 문서에 ===== END =====가 없음")


_doc: PromptDoc | None = None
_lock = threading.Lock()
_rec = threading.local()


def load(path: str | Path | None = None) -> PromptDoc:
    """문서를 (다시) 읽는다. path가 None이면 기본 위치(prompts/prompts.txt)."""
    global _doc
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_absolute():
        p = ROOT / p
    with _lock:
        _doc = PromptDoc(p)
    return _doc


def doc() -> PromptDoc:
    if _doc is None:
        load()
    return _doc


def text(name: str) -> str:
    """구간 원문(자리표시자 그대로)."""
    d = doc()
    if name not in d.sections:
        raise KeyError(f"프롬프트 문서에 구간이 없음: {name}")
    used = getattr(_rec, "names", None)
    if used is not None:
        used.append(name)
    return d.sections[name]


def has(name: str) -> bool:
    return name in doc().sections


def fill(name: str, **values) -> str:
    """구간의 {{자리표시자}}를 한 번에 채운다. 구간에 있는 자리표시자는 모두 값이 있어야 한다."""
    t = text(name)

    def rep(m):
        k = m.group(1)
        if k not in values:
            raise KeyError(f"구간 {name}: 자리표시자 {{{{{k}}}}} 값 없음")
        return str(values[k])
    return PH.sub(rep, t)


def prefix(name: str) -> str:
    """구간에서 첫 자리표시자 앞까지(문장 식별용, 예: 요약 블록 찾기)."""
    return PH.split(text(name))[0]


def begin_record() -> None:
    _rec.names = []


def end_record() -> list[str]:
    names = getattr(_rec, "names", None) or []
    _rec.names = None
    return list(dict.fromkeys(names))
