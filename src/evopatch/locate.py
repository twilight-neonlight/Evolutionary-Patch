"""함수 식별자(file::func 또는 file::Class.method)를 base commit 파일에서 찾아 Location으로 (4.1a)."""
from __future__ import annotations

import ast

from .types import Location


def parse_identifier(line: str) -> tuple[str, str] | None:
    line = line.strip().strip("`-* ").strip()
    if "::" not in line:
        return None
    f, sym = line.split("::", 1)
    f, sym = f.strip().removeprefix("./"), sym.strip().rstrip("()")
    if not f.endswith(".py") or not sym:
        return None
    return f, sym


def find_symbol(source: str, file: str, symbol: str) -> Location | None:
    """함수·메서드만 찾는다."""
    r = resolve_symbol(source, file, symbol, allow_class=False)
    return r[0] if r else None


def resolve_symbol(source: str, file: str, symbol: str, *, allow_class: bool) -> tuple[Location, str] | None:
    """식별자를 찾아 (Location, 종류)로. 종류: function / method / class.
    함수·메서드 범위는 첫 데코레이터 줄부터 끝 줄까지(데코레이터가 없으면 def 줄부터). 클래스는 class 줄부터.
    allow_class면 클래스 식별자는 클래스 본문 전체를 위치 범위로 쓴다()."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parts = symbol.split(".")
    nodes = tree.body
    target = None
    for k, name in enumerate(parts):
        found = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
                found = n
                break
        if found is None:
            return None
        if k < len(parts) - 1:
            if not isinstance(found, ast.ClassDef):
                return None
            nodes = found.body
        target = found
    if isinstance(target, ast.ClassDef):
        if not allow_class:
            return None
        kind = "class"
    else:
        kind = "method" if len(parts) > 1 else "function"
    lines = source.splitlines()
    s, e = target.lineno, target.end_lineno
    if kind != "class" and target.decorator_list:  # 결정 3(2026-10-12): 함수·메서드는 첫 데코레이터 줄부터 범위에 포함
        s = min(d.lineno for d in target.decorator_list)
    return Location(file=file, symbol=symbol, start_line=s, end_line=e, code="\n".join(lines[s - 1:e])), kind


def defined_as(source: str, name: str) -> list[tuple[str, str]] | None:
    """파일 안에서 name으로 정의된 모든 함수·클래스의 (자격 이름, 종류). 파싱 실패면 None. 진단·로그 전용."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    out: list[tuple[str, str]] = []

    def walk(node, path, in_func):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = ".".join(path + [n.name])
                if n.name == name:
                    if isinstance(n, ast.ClassDef):
                        kind = "class"
                    elif in_func:
                        kind = "nested_function"
                    else:
                        kind = "method" if path else "module_function"
                    out.append((q, kind))
                walk(n, path + [n.name], in_func or not isinstance(n, ast.ClassDef))
            else:
                walk(n, path, in_func)

    walk(tree, [], False)
    return out
