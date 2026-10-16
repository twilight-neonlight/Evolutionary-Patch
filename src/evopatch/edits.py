"""Search/Replace 편집: 파싱, 함수 범위 매칭(B7), 적용, 문법 검사, unified diff ."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

from .types import Candidate, Edit, Location

PATCH_HEADER = re.compile(r"^###\s*PATCH\s+(\d+)\s*$", re.M)
BLOCK = re.compile(
    r"(?P<file>[^\n]*?)\s*\n(?:```[^\n]*\n)?<<<<<<< SEARCH\n(?P<search>.*?)\n?=======\n(?P<replace>.*?)\n?>>>>>>> REPLACE",
    re.S,
)

# 무효 사유 (로그 집계용)
NO_EDITS = "no_edits"
FILE_NOT_IN_LOCATIONS = "file_not_in_locations"
EMPTY_SEARCH = "empty_search"
MATCH_OUTSIDE_LOCATION = "match_outside_location"  # B7
SEARCH_NOT_FOUND = "search_not_found"
SYNTAX_ERROR = "syntax_error"
EMPTY_DIFF = "empty_diff"
MISSING_FILE_PATH = "missing_file_path"  # 경로 줄이 없고 위치 범위에서 유일하게 매칭되지도 않음
SEARCH_NOT_UNIQUE = "search_not_unique"  # S2(파일 범위): SEARCH가 파일 안에서 두 곳 이상 매칭
SCOPES = ("function", "file")  # algorithm.edit_scope: function = 위치 함수 범위(S0), file = 위치한 파일 전체(S2)


PY_PATH = re.compile(r"([\w.\-/]+\.py)\b")


def _clean_path(s: str) -> str:
    """경로 줄에서 .py 경로만 뽑는다(`### a.py :: f (lines ..)`, `File: a.py`, `a/a.py` 등)."""
    m = PY_PATH.search(s)
    if not m:
        return ""  # 경로 줄 없음 → apply_edits에서 추론 시도 (S2)
    p = m.group(1)
    for pre in ("./", "a/", "b/"):
        if p.startswith(pre) and len(p) > len(pre):
            p = p[len(pre):]
            break
    return p


def parse_blocks(text: str) -> list[Edit]:
    edits = []
    for m in BLOCK.finditer(text):
        lines = [l for l in m.group("file").splitlines() if l.strip() and not l.strip().startswith("```")]
        path = _clean_path(lines[-1]) if lines else ""
        edits.append(Edit(file=path, search=m.group("search"), replace=m.group("replace")))
    return edits


def parse_patches(text: str) -> list[list[Edit]]:
    """`### PATCH <번호>` 머리로 나뉜 응답을 패치별 편집 목록으로 (4.2). 블록이 없는 패치는 빈 목록."""
    heads = list(PATCH_HEADER.finditer(text))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append(parse_blocks(text[h.end(): end]))
    return out


def edits_text(edits: list[Edit] | tuple[Edit, ...]) -> str:
    return "\n".join(f"{e.file}\n<<<<<<< SEARCH\n{e.search}\n=======\n{e.replace}\n>>>>>>> REPLACE" for e in edits)


@dataclass
class ApplyResult:
    valid: bool
    reason: str | None
    new_files: dict[str, str]
    detail: str = ""
    files_inferred: int = 0  # 경로 줄이 없어 위치 범위의 유일 매칭으로 파일을 정한 편집 수 (S2)
    failed_block: int | None = None  # 실패한 SEARCH 블록 번호(1부터)
    failed_first_line: str | None = None  # 그 SEARCH의 첫 비어 있지 않은 줄
    # D-3: 원본 매칭이 정확 일치가 아니라 줄 단위 strip 정규화로 이루어진 편집 수, 그중 REPLACE 들여쓰기를 고친 편집 수
    normalized_matches: int = 0
    reindented: int = 0


def _strip_lines(lines: list[str]) -> list[str]:
    return [l.strip() for l in lines]


def _trim_blank(lines: list[str]) -> list[str]:
    s, e = 0, len(lines)
    while s < e and not lines[s].strip():
        s += 1
    while e > s and not lines[e - 1].strip():
        e -= 1
    return lines[s:e]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find(region: list[str], search: list[str]) -> tuple[int, bool] | None:
    """region 안에서 search 줄들의 시작 위치. 정확 일치 우선, 그다음 공백·들여쓰기 정규화 일치."""
    n = len(search)
    for i in range(len(region) - n + 1):
        if region[i: i + n] == search:
            return i, True
    ss = _strip_lines(search)
    for i in range(len(region) - n + 1):
        if _strip_lines(region[i: i + n]) == ss:
            return i, False
    return None


def _reindent(replace: list[str], search_first: str, matched_first: str) -> list[str]:
    """정규화 일치일 때 들여쓰기 차이를 replace에도 반영한다(D-3, 승인).
    SEARCH 첫 줄의 들여쓰기(s_ind)와 원본에서 매칭된 첫 줄의 들여쓰기(m_ind)가 다를 때만 바뀐다:
    s_ind로 시작하는 비어 있지 않은 REPLACE 줄의 앞부분 s_ind를 m_ind로 바꾼다. 빈 줄과 s_ind로 시작하지 않는 줄은 그대로."""
    s_ind, m_ind = _indent(search_first), _indent(matched_first)
    out = []
    for l in replace:
        if not l.strip():
            out.append(l)
        elif l.startswith(s_ind):
            out.append(m_ind + l[len(s_ind):])
        else:
            out.append(l)
    return out


def _count_matches(region: list[str], search: list[str]) -> int:
    n = len(search)
    ss = _strip_lines(search)
    return sum(1 for i in range(len(region) - n + 1) if _strip_lines(region[i: i + n]) == ss)


def _infer_file(search_text: str, ranges: dict, files: dict) -> str | None:
    """경로 줄이 없는 편집: SEARCH가 위치 범위 전체에서 정확히 한 곳에서만 매칭되면 그 파일 (S2-b)."""
    search = _trim_blank(search_text.splitlines())
    if not search:
        return None
    hits = []
    for f, rgs in ranges.items():
        if f not in files:
            continue
        plain = [l.rstrip("\n").rstrip("\r") for l in files[f]]
        for rg in rgs:
            k = _count_matches(plain[rg[0] - 1: rg[1]], search)
            hits += [f] * k
    return hits[0] if len(hits) == 1 else None


def apply_edits(edits: list[Edit] | tuple[Edit, ...], locations: list[Location],
                originals: dict[str, str], scope: str = "function") -> ApplyResult:
    """편집을 위치 산출물의 함수 본문 범위 안에서만 적용한다(S0) → 문법 컴파일.
    scope="file"(S2): 위치한 파일 안 어디든(파일 전체를 한 범위로). 다른 파일은 그대로 거부하고, SEARCH가 파일 안에서
    두 곳 이상 매칭되면(공백 정규화 기준) search_not_unique로 거부한다."""
    if scope not in SCOPES:
        raise ValueError(f"edit scope {scope!r}: {SCOPES} 중 하나")
    if not edits:
        return ApplyResult(False, NO_EDITS, {})
    ranges: dict[str, list[list[int]]] = {}
    for loc in locations:
        ranges.setdefault(loc.file, []).append([loc.start_line, loc.end_line])
    if scope == "file":
        ranges = {f: [[1, max(1, len(originals[f].splitlines()))]] for f in ranges if f in originals}
    files = {p: c.splitlines(keepends=True) for p, c in originals.items() if p in ranges}
    inferred = 0
    normalized = reindented = 0

    for bi, e in enumerate(edits, 1):
        if not e.file:
            f = _infer_file(e.search, ranges, files)
            if f is None:
                return ApplyResult(False, MISSING_FILE_PATH, {}, "", inferred)
            e = Edit(f, e.search, e.replace)
            inferred += 1
        if e.file not in ranges or e.file not in files:
            return ApplyResult(False, FILE_NOT_IN_LOCATIONS, {}, e.file)
        search = _trim_blank(e.search.splitlines())
        if not search:
            return ApplyResult(False, EMPTY_SEARCH, {}, e.file)
        replace = e.replace.splitlines()
        lines = files[e.file]
        plain = [l.rstrip("\n").rstrip("\r") for l in lines]
        if scope == "file" and _count_matches(plain, search) > 1:
            return ApplyResult(False, SEARCH_NOT_UNIQUE, {}, e.file, inferred, failed_block=bi,
                               failed_first_line=next((l for l in search if l.strip()), "").strip())
        hit = None
        for rg in ranges[e.file]:
            s, t = rg[0] - 1, rg[1]
            f = _find(plain[s:t], search)
            if f is not None:
                hit = (rg, s + f[0], f[1])
                break
        if hit is None:
            where = MATCH_OUTSIDE_LOCATION if _find(plain, search) is not None else SEARCH_NOT_FOUND
            return ApplyResult(False, where, {}, e.file, inferred, failed_block=bi,
                               failed_first_line=next((l for l in search if l.strip()), "").strip())
        rg, start, exact = hit
        if not exact:
            normalized += 1
            fixed = _reindent(replace, search[0], plain[start])
            reindented += fixed != replace
            replace = fixed
        new = [l + "\n" for l in replace]
        end = start + len(search)
        # 원래 마지막 줄에 개행이 없었다면 유지
        if end == len(lines) and not lines[-1].endswith("\n") and new:
            new[-1] = new[-1].rstrip("\n")
        files[e.file] = lines[:start] + new + lines[end:]
        delta = len(new) - len(search)
        for other in ranges[e.file]:  # 뒤쪽 범위 줄 번호 보정
            if other is rg:
                other[1] += delta
            elif other[0] - 1 >= end:
                other[0] += delta
                other[1] += delta

    new_files = {p: "".join(ls) for p, ls in files.items() if "".join(ls) != originals[p]}
    for p, c in new_files.items():
        if p.endswith(".py"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compile(c, p, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as ex:
                return ApplyResult(False, SYNTAX_ERROR, {}, f"{p}: {ex}", inferred,
                                   normalized_matches=normalized, reindented=reindented)
    if not new_files:
        return ApplyResult(False, EMPTY_DIFF, {}, "", inferred, normalized_matches=normalized, reindented=reindented)
    return ApplyResult(True, None, new_files, "", inferred, normalized_matches=normalized, reindented=reindented)


def make_diff(originals: dict[str, str], new_files: dict[str, str]) -> str:
    """임시 git 저장소에서 git diff로 표준 unified diff를 만든다(원본 파일 전체 기준, 4.3)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        git = ["git", "-c", "user.email=x@x", "-c", "user.name=x", "-c", "core.autocrlf=false"]
        subprocess.run(git + ["init", "-q"], cwd=root, check=True)
        for p in new_files:
            fp = root / p
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(originals[p].encode())
        subprocess.run(git + ["add", "-A"], cwd=root, check=True)
        subprocess.run(git + ["commit", "-qm", "base"], cwd=root, check=True)
        for p, c in new_files.items():
            (root / p).write_bytes(c.encode())
        d = subprocess.run(git + ["diff", "--no-color", "--no-ext-diff"], cwd=root, check=True,
                           capture_output=True).stdout.decode()
    return d


def normalize_patch(diff: str) -> str:
    keep = [l.rstrip() for l in diff.splitlines() if not l.startswith("index ")]
    return "\n".join(keep).strip() + "\n"


def patch_hash(text: str) -> str:
    return hashlib.sha256(normalize_patch(text).encode()).hexdigest()[:16]


def build_candidate(edits: list[Edit], locations: list[Location], originals: dict[str, str],
                    scope: str = "function") -> tuple[Candidate, ApplyResult]:
    res = apply_edits(edits, locations, originals, scope)
    if res.valid:
        diff = make_diff(originals, res.new_files)
        return Candidate(edits=tuple(edits), patch_hash=patch_hash(diff), diff=diff), res
    # 무효 후보는 편집 원문으로 해시(중복 비율 집계용)
    return Candidate(edits=tuple(edits), patch_hash=patch_hash("INVALID\n" + edits_text(edits)), diff=""), res
