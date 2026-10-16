"""S2 코드 컨텍스트(algorithm.edit_scope=file일 때만, METHOD 12절 미결정 항목): 위치한 파일을 프롬프트에 넓게 보여 준다.
문장·머리글 형식은 prompts/prompts.txt의 S2.* 구간.

파일마다(위치 순서대로)
- 파일 토큰 수 ≤ full_file_max_tokens(6000): 파일 전체.
- 그보다 크면 발췌: 모듈 상단(첫 최상위 def/class 앞까지: docstring·import·상수) + 다른 최상위 import·대입 문장
  + 각 위치 범위 앞뒤 window(40)줄 + 위치를 감싼 클래스의 시그니처(데코레이터~머리 줄). 겹치는 구간은 합친다.
전체가 예산(토큰)을 넘으면 아래 순서로 줄이고(결정적), 줄인 단계를 기록한다:
  1 full_to_excerpt: 전체로 보여 준 파일을 큰 것부터 하나씩 발췌로
  2 window_20 / window_10 / window_0: 위치 앞뒤 줄 수 축소
  3 no_module_statements: 상단 이후의 최상위 import·대입 문장 제외
  4 header_30: 모듈 상단을 앞 30줄로
  5 function_only: 기존(S0) 위치 코드만 (그래도 넘으면 그대로 두고 기록)
"""
from __future__ import annotations

import ast
import warnings
from typing import Callable

from . import promptdoc as D
from .types import Location

FULL_FILE_MAX_TOKENS = 6000
WINDOW = 40


def _tree(src: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def _start(n) -> int:
    return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])


def _header_end(n, lines: list[str]) -> int:
    first_body = _start(n.body[0])  # 첫 문장이 데코레이터 달린 정의면 데코레이터 줄부터 본문
    if first_body <= n.lineno:
        return n.lineno
    end = first_body - 1
    while end > n.lineno and (not lines[end - 1].strip() or lines[end - 1].strip().startswith("#")):
        end -= 1
    return end


def enclosing_class_headers(src: str, line: int) -> list[tuple[int, int]]:
    """line을 감싸는 클래스들의 시그니처 구간(데코레이터~머리 줄)."""
    tree = _tree(src)
    if tree is None:
        return []
    lines = src.splitlines()
    out = []

    def walk(nodes):
        for n in nodes:
            if isinstance(n, ast.ClassDef) and _start(n) <= line <= n.end_lineno:
                out.append((_start(n), _header_end(n, lines)))
                walk(n.body)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno:
                walk(n.body)

    walk(tree.body)
    return out


def module_segments(src: str, header_max: int | None = None, statements: bool = True) -> list[tuple[int, int]]:
    """모듈 상단(첫 최상위 def/class 앞) + (statements면) 다른 최상위 import·대입 문장."""
    tree = _tree(src)
    n_lines = len(src.splitlines())
    if tree is None:
        return [(1, min(n_lines, header_max or 60))] if n_lines else []
    defs = [_start(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    top_end = (min(defs) - 1) if defs else n_lines
    segs = []
    if top_end >= 1:
        segs.append((1, min(top_end, header_max) if header_max else top_end))
    if statements:
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) and n.lineno > top_end:
                segs.append((n.lineno, n.end_lineno))
    return segs


def merge(segs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def excerpt_segments(src: str, locs: list[Location], window: int, statements: bool = True,
                     header_max: int | None = None) -> list[tuple[int, int]]:
    n = len(src.splitlines())
    segs = module_segments(src, header_max, statements)
    for l in locs:
        segs.append((max(1, l.start_line - window), min(n, l.end_line + window)))
        segs += enclosing_class_headers(src, l.start_line)
    return merge([(s, e) for s, e in segs if s <= e])


def render_full(path: str, src: str) -> str:
    return D.fill("S2.FULL_FILE", PATH=path, N=len(src.splitlines()), SOURCE=src.rstrip(chr(10)))


def render_excerpt(path: str, src: str, segs: list[tuple[int, int]]) -> str:
    lines = src.splitlines()
    parts = [D.fill("S2.EXCERPT_HEADER", PATH=path)]
    for s, e in segs:
        parts.append(D.fill("S2.EXCERPT", PATH=path, START=s, END=e, CODE="\n".join(lines[s - 1:e])))
    return "\n".join(parts)


def build_code_view(locations: list[Location], originals: dict[str, str], count: Callable[[str], int], budget: int,
                    function_view: str, full_max: int = FULL_FILE_MAX_TOKENS, window: int = WINDOW) -> tuple[str, dict]:
    """(코드 컨텍스트 문자열, 기록). count(text) = 토큰 수. function_view = 기존(S0) 위치 코드 렌더링(마지막 대체용)."""
    files = list(dict.fromkeys(l.file for l in locations if l.file in originals))
    by_file = {f: [l for l in locations if l.file == f] for f in files}
    full_tokens = {f: count(originals[f]) for f in files}
    mode = {f: "full" if full_tokens[f] <= full_max else "excerpt" for f in files}
    state = {"window": window, "statements": True, "header_max": None}

    def render():
        parts = []
        for f in files:
            if mode[f] == "full":
                parts.append(render_full(f, originals[f]))
            else:
                parts.append(render_excerpt(f, originals[f], excerpt_segments(
                    originals[f], by_file[f], state["window"], state["statements"], state["header_max"])))
        return D.text("S2.FILE_SEPARATOR").join(parts)

    steps = []
    text = render()
    n = count(text)
    meta = {"files": len(files), "full_files": sum(m == "full" for m in mode.values()),
            "file_tokens": sorted(full_tokens.values(), reverse=True), "budget": budget}
    if n <= budget:
        return text, {**meta, "level": "as_specified", "truncated": False, "tokens": n, "steps": steps}
    for f in sorted((f for f in files if mode[f] == "full"), key=lambda f: -full_tokens[f]):
        mode[f] = "excerpt"
        steps.append("full_to_excerpt")
        text = render()
        n = count(text)
        if n <= budget:
            return text, {**meta, "level": "full_to_excerpt", "truncated": True, "tokens": n, "steps": steps}
    for w in (20, 10, 0):
        state["window"] = w
        steps.append(f"window_{w}")
        text = render()
        n = count(text)
        if n <= budget:
            return text, {**meta, "level": f"window_{w}", "truncated": True, "tokens": n, "steps": steps}
    state["statements"] = False
    steps.append("no_module_statements")
    text = render()
    n = count(text)
    if n <= budget:
        return text, {**meta, "level": "no_module_statements", "truncated": True, "tokens": n, "steps": steps}
    state["header_max"] = 30
    steps.append("header_30")
    text = render()
    n = count(text)
    if n <= budget:
        return text, {**meta, "level": "header_30", "truncated": True, "tokens": n, "steps": steps}
    steps.append("function_only")
    n = count(function_view)
    return function_view, {**meta, "level": "function_only", "truncated": True, "tokens": n, "steps": steps,
                           "over_budget": n > budget}
