"""프롬프트 마스킹: test_files(test_patch에서 오프라인 추출) 경로·파일명을 모델 입력에서 가린다 (B2).

test_files는 실행기 안에서만 쓰는 값이다. 프롬프트에 들어가는 텍스트에서는 다음 형태를 MASK로 바꾼다.
- path:      전체 경로 (예: tests/admin_views/tests.py, /testbed/… 접두사가 붙어도 일치)
- module:    점 표기 모듈 (예: tests.admin_views.tests, Django 라벨형 admin_views.tests)
- basename:  파일명이 경로의 일부가 아닌 단독 토큰으로 나올 때 (예: test_card.py).
             단, 패치 텍스트(부모 diff·무효 부모 원문·이전 답변·실패 SEARCH 첫 줄) 안에서는 가리지 않는다(결정 2).
허용 입력(이슈 본문)과 base commit 코드 원문(위치 코드·위치 파일 경로)은 보호 구간으로 두고 가리지 않는다.
SEARCH가 보이는 코드와 한 글자도 다르지 않아야 하므로 코드 원문을 바꿀 수 없기 때문이다.
"""
from __future__ import annotations

import re
from collections import Counter



def mask_token() -> str:
    """test_files 경로·파일명을 바꿔 넣는 표시(문서 MASK.TEST_FILE)."""
    from . import promptdoc
    return promptdoc.text("MASK.TEST_FILE")


def mask_terms(test_files) -> list[tuple[str, re.Pattern]]:
    """(형태, 정규식) 목록. 긴 것부터 적용한다."""
    paths, modules, bases = set(), set(), set()
    for p in test_files or ():
        p = p.strip().lstrip("./")
        if not p:
            continue
        paths.add(p)
        stem = p[:-3] if p.endswith(".py") else p
        mod = stem.replace("/", ".")
        if "." in mod:
            modules.add(mod)
        if p.startswith("tests/") and "." in mod[len("tests."):]:  # Django runtests 라벨형
            modules.add(mod[len("tests."):])
        b = p.rsplit("/", 1)[-1]
        if b != p:
            bases.add(b)
    out = []
    for p in sorted(paths, key=len, reverse=True):
        out.append(("path", re.compile(r"(?<![\w.-])" + re.escape(p) + r"(?![\w])")))
    for m in sorted(modules, key=len, reverse=True):
        out.append(("module", re.compile(r"(?<![\w.])" + re.escape(m) + r"(?![\w])")))
    for b in sorted(bases, key=len, reverse=True):
        out.append(("basename", re.compile(r"(?<![\w./-])" + re.escape(b) + r"(?![\w])")))
    return out


class TestFileMasker:
    """호출 가능한 마스커. hits는 형태별로 가린 횟수(로그용)."""

    __test__ = False  # pytest 수집 대상 아님

    def __init__(self, test_files=()):
        self.terms = mask_terms(test_files)
        self.hits: Counter = Counter()

    def __call__(self, text: str, protected=(), basename_exempt=()) -> str:
        """protected: 아무것도 가리지 않는 구간(이슈 본문, 위치 코드 원문).
        basename_exempt: 경로·모듈만 가리고 단독 파일명은 가리지 않는 구간(패치 텍스트: 코드 안의 문자열 보존, 결정 2)."""
        if not self.terms or not text:
            return text
        keep = {}

        def hold(t, spans, tag):
            for i, p in enumerate(sorted({p for p in spans if p}, key=lambda x: (-len(x), x))):  # 결정적 순서
                if p in t:
                    tok = f"\x00{tag}{i}\x00"
                    keep[tok] = p
                    t = t.replace(p, tok)
            return t

        text = hold(text, protected, "P")
        full = [(k, pat) for k, pat in self.terms if k != "basename"]
        exempt = []
        for span in basename_exempt:  # 본문과 같은 변환(보호 구간 토큰 → 경로·모듈 가림)을 거친 형태로 맞춘다
            for tok, p in sorted(keep.items(), key=lambda kv: len(kv[1]), reverse=True):
                span = span.replace(p, tok)
            for _, pat in full:
                span = pat.sub(mask_token(), span)
            exempt.append(span)
        for kind, pat in full:
            text, n = pat.subn(mask_token(), text)
            if n:
                self.hits[kind] += n
        text = hold(text, exempt, "E")
        for kind, pat in self.terms:
            if kind == "basename":
                text, n = pat.subn(mask_token(), text)
                if n:
                    self.hits[kind] += n
        for tok, p in reversed(list(keep.items())):  # 나중 표시(E)부터: E 구간 안에 든 P 표시까지 모두 되돌린다
            text = text.replace(tok, p)
        return text


def identity(text: str, protected=(), basename_exempt=()) -> str:
    return text
