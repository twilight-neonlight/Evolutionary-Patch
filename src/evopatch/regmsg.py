"""회귀 실패 오류 내용 추출과 가림 (2026-10-16 최종 후보 지시 5, loop.feedback both 전용).

회귀 실행 출력(regression.log의 START~END 구간)에서 실패한 회귀 테스트의 오류 내용을 저장소 형식별로 뽑는다.
- pytest류(astropy·sklearn·matplotlib·xarray·pylint·pytest·sphinx·seaborn·flask·requests): `____ 이름 ____` 구간,
  프레임은 `_ _ _` 줄로 나뉘고 위치 줄 `경로:줄: in 함수`/`경로:줄: 예외`, 코드 줄 `>`, 예외 `E   ...`.
- Django(unittest): `FAIL:`/`ERROR: 이름 (라벨)` + `-----` 다음 Python 트레이스백, 다음 `=====`/`-----`까지.
- Sympy(bin/test): `____ 경로:이름 ____` 다음 Python 트레이스백, 다음 `____` 머리나 `=== tests finished`까지.
남기는 것: 예외 종류·메시지, 저장소 소스 파일 프레임(경로·줄 번호·함수 이름·코드 줄).
바꾸는 것: 테스트 파일 프레임 → `<test frame>` 한 줄. 테스트 ID·이름(회귀 목록 전체의 ID, 그 구성 이름, `test_*`/`Test*` 이름)
  → `<test>`. 그 위에 프롬프트 단계에서 TestFileMasker가 다시 걸린다.
2026-10-16 14:20 수정: 추출 전에 ANSI 색·제어 코드와 pytest `INTERNALERROR> ` 줄 머리를 지운다. pytest류 머리는 밑줄 한 개
  (`_ 긴 이름 _`)와 `ERROR at setup/teardown of`, `ERROR collecting 경로`(그 파일의 테스트 전부)를 인식하고, 본문은 다음 머리나
  다음 `=== … ===` 줄까지(테스트 안 pytest 세션 출력에서 영역이 끊기지 않게). Django는 머리 다음 독스트링 줄을 건너뛰고
  그 줄을 별칭으로 맞춘다. 테스트 자신의 구간이 없으면 테스트 구간 밖의 오류(실행 전체 중단·모듈 가져오기 오류·Sympy 실행기
  오류, 수집 오류로 세션이 멈춘 경우 그 수집 오류)를 대신 쓴다(같은 내용은 한 번만). 오류 출력이 아예 없으면 사유 no_error_output.
  가림 규칙(이름·테스트 프레임·저장소 밖 프레임·상한)은 그대로.
안전하게 뽑지 못하면(구간 없음·예외 줄 없음·가린 뒤에도 이름이 남음) 그 테스트 내용은 넣지 않고 사유만 센다.
같은 내용은 한 번만, 최대 3개, 부모당 합계 약 1,500자(넘으면 뒤쪽(예외 쪽)을 남기고 앞을 자름).
"""
from __future__ import annotations

import re
from collections import Counter

from . import promptdoc as D

MAX_MSGS = 3
TOTAL_CHARS = 1500
MAX_FRAMES = 8
MAX_EXC_LINES = 8


def test_frame() -> str:
    """테스트 파일 프레임을 바꿔 넣는 표시(문서 REGRESSION_MESSAGE.TEST_FRAME)."""
    return D.text("REGRESSION_MESSAGE.TEST_FRAME")


def name_mask() -> str:
    return D.text("REGRESSION_MESSAGE.NAME_MASK")


def frame_line(path: str, line, func: str, code: str | None) -> str:
    return D.fill("REGRESSION_MESSAGE.FRAME", PATH=path, LINE=line, FUNC=func) + (
        D.fill("REGRESSION_MESSAGE.FRAME_CODE", CODE=code) if code else "")
TEST_DIRS = ("tests", "test", "testing")
DJANGO, SYMPY = "django/django", "sympy/sympy"

_PY_FRAME = re.compile(r'^\s*File "(?:/testbed/)?([^"]+)", line (\d+), in (\S+)\s*$')
_PT_ABS = re.compile(r"^(/[\w./\-]+\.py):(\d+):(?: in (\S+)| (\w[\w.]*))?\s*$")
_PT_LOC = re.compile(r"^(?:/testbed/)?([\w./\-]+\.py):(\d+):(?: in (\S+)| (\w[\w.]*))?\s*$")
_GENERIC_NAMES = (re.compile(r"\btest_\w*"), re.compile(r"\bTest[A-Z0-9_]\w*"))
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")  # CSI(색 등)·기타 ESC 시퀀스
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control(text: str) -> str:
    """ANSI 색·제어 코드 제거(추출 전). 줄 끝 \\r\\n은 \\n으로, 남은 \\r은 줄바꿈으로."""
    text = _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return _CTRL.sub("", text)


def is_external(p: str) -> bool:
    """저장소 밖 파일(절대 경로: site-packages·표준 라이브러리 등, `<string>`처럼 파일이 없는 코드)."""
    return (p.startswith("/") and not p.startswith("/testbed/")) or p.startswith("<")


def is_test_path(p: str, test_files=()) -> bool:
    p = p.split("/testbed/")[-1]
    if p in set(test_files):
        return True
    parts = p.split("/")
    name = parts[-1]
    return any(x in TEST_DIRS for x in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py") \
        or name == "conftest.py" or name == "tests.py"


def name_parts(test_id: str) -> set[str]:
    """회귀 테스트 ID에서 가릴 이름들(ID 전체와 구성 이름)."""
    out = {test_id.strip()}
    t = test_id.strip()
    m = re.match(r"^(\S+) \(([\w.]+)\)$", t)  # Django: test_m (pkg.mod.Class) / (pkg.mod.Class.test_m)
    if m:
        out.add(m.group(1))
        out.add(m.group(2))
        out.update(x for x in m.group(2).split(".") if len(x) >= 4)
        return out
    if "::" in t:  # pytest node id
        parts = t.split("::")
        out.update(parts)
        for p in parts[1:]:
            out.add(p.split("[")[0])
        out.add(".".join(parts[1:]))
    elif ":" in t:  # sympy path:name
        path, nm = t.rsplit(":", 1)
        out.update({path, nm})
    else:
        out.add(t.split("[")[0])
    return {x for x in out if x and len(x) >= 4}


def scrub_names(text: str, names: set[str]) -> tuple[str, int]:
    n = 0
    for nm in sorted(names, key=lambda x: (-len(x), x)):  # 길이 같으면 이름순(프로세스마다 같은 결과)
        if nm in text:
            n += text.count(nm)
            text = text.replace(nm, name_mask())
    for rx in _GENERIC_NAMES:
        text, k = rx.subn(name_mask(), text)
        n += k
    return text, n


# ---------- 구간 찾기 ----------
# 구간 = {"head", "body", "start", "end", "alias"}. start~end는 정리된 log 안의 위치(구간 밖 오류 찾기에 씀).
_HEAD_PT = re.compile(r"^_+ (.+?) _+[ \t]*$", re.M)  # 이름이 길면 pytest가 밑줄을 한 개까지 줄임
_RESULT_LINE = re.compile(r"^=+ .+ =+[ \t]*$", re.M)
_PT_PHASE = re.compile(r"^ERROR at (?:setup|teardown) of ")
_PT_COLLECT = "ERROR collecting "
_DASH = re.compile(r"^-{5,}[ \t]*$", re.M)
_DJ_STOP = re.compile(r"^(?:={5,}|-{5,})[ \t]*$", re.M)
_TB = re.compile(r"^Traceback \(most recent call last\):", re.M)
_SEP_LINE = re.compile(r"^(?:[_=\-]{10,}|=+ .+ =+)[ \t]*$", re.M)  # 구분 줄·결과 줄


def _sec(head, log, start, body_start, end, alias=None):
    return {"head": head, "body": log[body_start:end], "start": start, "end": end, "alias": alias}


def _sections_pytest(log: str) -> list[dict]:
    """`_ 이름 _` 머리(FAILURES/ERRORS 머리 뒤). 본문은 다음 머리나 다음 `=== … ===` 줄까지.
    (트레이스백·E 줄은 캡처 출력보다 앞에 오므로, 테스트 안에서 돌린 pytest 세션 출력에서 끊겨도 오류 내용은 남는다.)"""
    start = re.search(r"^=+ (FAILURES|ERRORS) =+[ \t]*$", log, re.M)
    if not start:
        return []
    heads = [h for h in _HEAD_PT.finditer(log, start.end()) if h.group(1).strip("_ ")]  # `_ _ _` 구분 줄 제외
    out = []
    for k, h in enumerate(heads):
        nxt = heads[k + 1].start() if k + 1 < len(heads) else len(log)
        e = _RESULT_LINE.search(log, h.end(), nxt)
        stop = e.start() if e else nxt
        out.append(_sec(h.group(1).strip(), log, h.start(), h.end(), stop))
    return out


def _sections_django(log: str) -> list[dict]:
    """`FAIL:/ERROR: 이름 (라벨)` [+ 독스트링 첫 줄] + `-----` + 트레이스백, 다음 `=====`/`-----`까지.
    독스트링 줄은 alias(회귀 목록이 독스트링 첫 줄을 ID로 쓰는 경우)."""
    out = []
    for h in re.finditer(r"^(?:FAIL|ERROR): (.+)$", log, re.M):
        dash = _DASH.search(log, h.end())
        desc = log[h.end():dash.start()].strip() if dash else ""
        if dash and (not desc or desc.count("\n") < 3):
            body_start, alias = dash.end(), (desc.splitlines()[0].strip() if desc else None)
        else:
            body_start, alias = h.end(), None
        stop = _DJ_STOP.search(log, body_start)
        end = stop.start() if stop else len(log)
        out.append(_sec(h.group(1).strip(), log, h.start(), body_start, end, alias))
    return out


def _sections_sympy(log: str) -> list[dict]:
    heads = list(re.finditer(r"^_{3,} (\S+?:\S+?) _{3,}[ \t]*$", log, re.M))
    out = []
    for k, h in enumerate(heads):
        stop_h = heads[k + 1].start() if k + 1 < len(heads) else len(log)
        e = re.compile(r"^(?:=+ tests finished|_{20,}[ \t]*$)", re.M).search(log, h.end(), stop_h)
        out.append(_sec(h.group(1).strip(), log, h.start(), h.end(), e.start() if e else stop_h))
    return out


def _match(test_id: str, sec: dict, repo: str) -> bool:
    t = test_id.strip()
    head = sec["head"]
    if repo == DJANGO:
        if head == t or (sec.get("alias") and sec["alias"] == t):
            return True
        hm0 = re.match(r"^(\S+) \(", head)
        if " (" not in t and hm0:  # 짧은 이름(메서드 이름만)
            return hm0.group(1) == t
        m = re.match(r"^(\S+) \(([\w.]+)\)$", t)
        hm = re.match(r"^(\S+) \(([\w.]+)\)", head)
        return bool(m and hm and m.group(1) == hm.group(1) and (hm.group(2) == m.group(2)
                                                               or hm.group(2).startswith(m.group(2) + ".")))
    if repo == SYMPY:
        return head == t or head.endswith(":" + t)
    if head.startswith(_PT_COLLECT):  # 수집 오류: 그 파일(또는 디렉터리)의 테스트 전부
        path = head[len(_PT_COLLECT):].strip().rstrip("/")
        tp = t.split("::")[0]
        return tp == path or tp.startswith(path + "/")
    head = _PT_PHASE.sub("", head)
    if "::" in t:
        parts = t.split("::")
        return head in (".".join(parts[1:]), parts[-1])
    return head == t or head.split("[")[0] == t


def _outside_block(log: str, sections: list[dict]) -> str | None:
    """테스트 구간 밖의 마지막 Python 트레이스백(실행 전체 중단·모듈 가져오기 오류 등). 없으면 None."""
    spans = [(x["start"], x["end"]) for x in sections]
    tbs = [m.start() for m in _TB.finditer(log) if not any(a <= m.start() < b for a, b in spans)]
    if not tbs:
        return None
    s = tbs[-1]
    nxt = [a for a, _ in spans if a > s]
    sep = _SEP_LINE.search(log, s)
    return log[s:min(nxt + ([sep.start()] if sep else []) or [len(log)])]


# ---------- 본문 → 정리된 오류 내용 ----------
def _from_python_tb(body: str, test_files) -> tuple[list[str], list[str], int]:
    """(프레임 줄, 예외 줄, 테스트 프레임 수). 표준 Python 트레이스백."""
    lines = body.splitlines()
    frames, exc, n_test, last = [], [], 0, -1
    i = 0
    while i < len(lines):
        m = _PY_FRAME.match(lines[i])
        if m:
            code = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].startswith("    ") \
                and not _PY_FRAME.match(lines[i + 1]) else ""
            if is_test_path(m.group(1), test_files):
                frames.append(test_frame())
                n_test += 1
            elif is_external(m.group(1)):
                pass  # 저장소 소스 프레임만 남김
            else:
                frames.append(frame_line(m.group(1), m.group(2), m.group(3), code))
            last = i + (1 if code else 0)
            i += 2 if code else 1
            continue
        i += 1
    if last >= 0:
        for l in lines[last + 1:]:
            if (not l.strip() and exc) or _SEP_LINE.match(l):
                break
            if l.strip() and not l.startswith("Traceback (most recent call last)") and l.rstrip() not in exc:
                exc.append(l.rstrip())
    return frames, exc[:MAX_EXC_LINES], n_test


def _from_pytest(body: str, test_files) -> tuple[list[str], list[str], int]:
    segs = re.split(r"^(?:_ ){5,}_?\s*$", body, flags=re.M)
    frames, exc, n_test = [], [], 0
    for seg in segs:
        lines = seg.splitlines()
        loc = None
        for k, l in enumerate(lines):
            m = _PT_LOC.match(l) or _PT_ABS.match(l)
            if m:
                loc = (k, m)
        if loc is None:
            if "Traceback (most recent call last)" in seg:  # 수집 오류 등 Python 형식
                f, e, t = _from_python_tb(seg, test_files)
                frames += f
                exc += e
                n_test += t
            continue
        k, m = loc
        path, ln = m.group(1), m.group(2)
        func = m.group(3)
        code = next((l[1:].strip() for l in lines[:k] if l.startswith(">")), None)
        if code is None and k + 1 < len(lines) and lines[k + 1].startswith("    "):
            code = lines[k + 1].strip()
        if func is None:
            defs = [re.match(r"^\s*(?:async\s+)?def (\w+)", l) for l in lines[:k]]
            defs = [d.group(1) for d in defs if d]
            func = defs[-1] if defs else "?"
        if is_test_path(path, test_files):
            frames.append(test_frame())
            n_test += 1
        elif is_external(path):
            pass
        else:
            frames.append(frame_line(path, ln, func, code))
        e = []
        for l in lines:
            if l.startswith("E ") and l[1:].strip() not in e:
                e.append(l[1:].strip())
        if e:
            exc = e
    return frames, exc[:MAX_EXC_LINES], n_test


def clean_block(body: str, repo: str, test_files, python_tb: bool = False) -> tuple[str | None, str | None, int]:
    """(정리된 내용 | None, 실패 사유 | None, 테스트 프레임 수)."""
    if python_tb or repo in (DJANGO, SYMPY):
        frames, exc, n_test = _from_python_tb(body, test_files)
    else:
        frames, exc, n_test = _from_pytest(body, test_files)
    if not exc:
        return None, "no_exception_line", n_test
    kept = []
    for f in frames[-MAX_FRAMES:]:
        if f == test_frame() and kept and kept[-1] == test_frame():
            continue
        kept.append(f)
    return "\n".join(kept + exc), None, n_test


def extract(log: str, failed: list[str], all_ids: list[str], repo: str, test_files=()) -> dict:
    """실패한 회귀 테스트(순서대로)에서 서로 다른 오류 내용 최대 3개. 기록용 수치를 함께 돌려준다.
    테스트 자신의 구간이 없으면 테스트 구간 밖의 오류(실행 전체 중단·수집/가져오기 오류)를 대신 쓴다(같은 내용은 한 번만)."""
    log = re.sub(r"^INTERNALERROR> ?", "", strip_control(log), flags=re.M)  # pytest 내부 오류의 줄 머리
    sections = _sections_django(log) if repo == DJANGO else _sections_sympy(log) if repo == SYMPY else _sections_pytest(log)
    outside, outside_tb = _outside_block(log, sections), True
    if outside is None and re.search(r"^!+ Interrupted: \d+ errors? during collection", log, re.M):
        coll = [x for x in sections if x["head"].startswith(_PT_COLLECT)]  # 수집 오류로 세션 전체가 멈춤
        if coll:
            outside, outside_tb = coll[0]["body"], False
    has_error_output = bool(_TB.search(log) or re.search(r"^E ", log, re.M))
    names = set()
    for t in list(all_ids) + list(failed):
        names |= name_parts(t)
    msgs, reasons, scrub, dup, fallback = [], Counter(), Counter(), 0, 0
    cache = {}
    budget = TOTAL_CHARS
    for t in failed:
        if len(msgs) >= MAX_MSGS or budget < 100:
            break
        sec = next((x for x in sections if _match(t, x, repo)), None)
        if sec is not None:
            key = ("sec", sec["start"])
            if key not in cache:
                cache[key] = clean_block(sec["body"], repo, test_files)
        elif outside is not None:
            key = ("outside", 0)
            if key not in cache:
                cache[key] = clean_block(outside, repo, test_files, python_tb=outside_tb)
        else:
            reasons["section_not_found" if has_error_output else "no_error_output"] += 1
            continue
        text, why, n_test = cache[key]
        if text is None:
            reasons[why] += 1
            continue
        text, n_names = scrub_names(text, names)
        if any(nm in text for nm in names if len(nm) >= 6):
            reasons["name_residue"] += 1
            continue
        if text in msgs or (len(text) > budget and D.text("RESULTS.TRUNCATED") + "\n" + text[-(budget - 20):] in msgs):
            dup += 1
            continue
        fallback += key[0] == "outside"
        scrub["test_frames"] += n_test
        scrub["test_names"] += n_names
        if len(text) > budget:
            text = D.text("RESULTS.TRUNCATED") + "\n" + text[-(budget - 20):]
        msgs.append(text)
        budget -= len(text)
    return {"n_failed": len(failed), "messages": msgs, "fail_reasons": dict(reasons), "scrub": dict(scrub),
            "duplicates_skipped": dup, "outside_blocks": fallback}
