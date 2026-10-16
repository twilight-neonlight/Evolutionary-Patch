"""회귀 오류 내용 추출·가림(regmsg)과 both 회귀 결과 블록(2026-10-16 12:30 지시 5·7). 실제 회귀 출력 예시(pytest·Django·Sympy)
는 tests/data/regmsg/에 있다(이전 런의 regression.log에서 잘라 옴). GPU·도커 없음."""
import json
import re
from pathlib import Path

import pytest

from evopatch import feedback as FB
from evopatch import regmsg as M
from evopatch.executor import END, START

DATA = Path(__file__).resolve().parent / "data" / "regmsg"


def load(name):
    meta = json.loads((DATA / f"{name}.json").read_text())
    log = (DATA / f"{name}.log").read_text().split(START, 1)[-1].split(END, 1)[0]
    return log, meta


def names_of(meta):
    out = set()
    for t in meta["failed"] + meta["all"]:
        out |= {x for x in M.name_parts(t) if len(x) >= 6}
    return out


EXPECT = {
    "django": (["AssertionError: Lists differ"], []),
    "sympy": (["AssertionError"], []),
    "pytest_sklearn": (["NameError: name 'pd' is not defined"],
                       ['File "sklearn/utils/validation.py", line 460, in check_array',
                        "is_dataframe = isinstance(array, pd.DataFrame)"]),
    "pytest_pylint": (["NameError: name 're' is not defined"], ['File "pylint/lint/pylinter.py", line 615, in _discover_files']),
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_real_logs_keep_exception_and_source_frames_drop_test_ids(name):
    log, meta = load(name)
    r = M.extract(log, meta["failed"], meta["all"], meta["repo"], ())
    assert r["messages"] and r["n_failed"] == len(meta["failed"]) and len(r["messages"]) <= 3
    text = "\n".join(r["messages"])
    exc, frames = EXPECT[name]
    for e in exc + frames:
        assert e in text
    assert M.test_frame() in text and r["scrub"]["test_frames"] >= 1
    for nm in names_of(meta):  # 테스트 ID·이름 없음
        assert nm not in text, nm
    assert not re.search(r"\btest_\w+", text) and "::" not in text
    assert not re.search(r'File "[^"]*(?:/tests?/|test_)[^"]*"', text)  # 테스트 파일 프레임 없음
    assert "/opt/" not in text.split("\n", 1)[0] and not re.search(r'File "/opt/', text)  # 저장소 밖 프레임 없음
    assert sum(len(m) for m in r["messages"]) <= M.TOTAL_CHARS + 50


def test_unextractable_leaves_only_count():
    r = M.extract("random output without sections\n", ["tests/a.py::test_x", "tests/a.py::test_y"],
                  ["tests/a.py::test_x"], "scikit-learn/scikit-learn", ())
    assert r["messages"] == [] and r["fail_reasons"] == {"no_error_output": 2}  # 오류 출력 자체가 없음(14:20 수정)
    r2 = M.extract("=== FAILURES ===\n____ test_other ____\n\nE   ValueError: v\n\npkg/m.py:1: ValueError\n=== 1 failed ===\n",
                   ["tests/a.py::test_x"], [], "scikit-learn/scikit-learn", ())
    assert r2["messages"] == [] and r2["fail_reasons"] == {"section_not_found": 1}  # 오류 출력은 있으나 그 테스트 구간이 없음
    ex = {"repro_pass": False, "repro_statuses": {}, "repro_log_tail": "", "reg_passed": 3, "reg_total": 5,
          "no_regression_signal": False, "timed_out": False, "reg_failure": r}
    t, items, _ = FB.exec_summary({"valid": True, "exec": ex})
    assert "Regression tests that did not pass: 2" in t and "Regression failure #" not in t
    assert "regression_failure_messages" not in items


def test_no_exception_line_is_a_failure_reason():
    log = "=== FAILURES ===\n____ test_a ____\n\n    def test_a():\n>       x()\n\ntests/t.py:3: \n=== 1 failed in 0.1s ===\n"
    r = M.extract(log, ["tests/t.py::test_a"], [], "psf/requests", ())
    assert r["messages"] == [] and r["fail_reasons"] == {"no_exception_line": 1}


def test_duplicates_once_and_cap():
    body = "\n".join(f"____ test_{i} ____\n\n    def test_{i}():\n>       f()\n\npkg/t_{i}.py:3: \n"
                     f"_ _ _ _ _ _ _ _ _ _ _ _\n\n    def f():\n>       raise ValueError('same')\nE       ValueError: same\n\npkg/m.py:9: ValueError"
                     for i in range(5))
    log = "==== FAILURES ====\n" + body + "\n==== 5 failed in 1s ====\n"
    r = M.extract(log, [f"pkg/t_{i}.py::test_{i}" for i in range(5)], [], "psf/requests", ())
    assert len(r["messages"]) == 1 and r["duplicates_skipped"] == 4
    big = "\n".join(f"____ test_{i} ____\n\n    def g():\n>       raise ValueError('{i}' * 900)\nE       ValueError: {str(i) * 900}\n\npkg/m.py:9: ValueError"
                    for i in range(3))
    r2 = M.extract("==== FAILURES ====\n" + big + "\n==== 3 failed in 1s ====\n", [f"t.py::test_{i}" for i in range(3)], [],
                   "psf/requests", ())
    assert sum(len(m) for m in r2["messages"]) <= M.TOTAL_CHARS + 20 and r2["messages"][-1].startswith("... (truncated)")


# ---------- both 회귀 결과 블록: 세 가지 경우, 전체·통과 개수 없음 ----------
def _ex(**kw):
    return {"valid": True, "exec": {"repro_pass": False, "repro_statuses": {}, "repro_log_tail": "", "reg_passed": 7,
                                    "reg_total": 9, "no_regression_signal": False, "timed_out": False, **kw}}


def test_regression_block_three_cases_no_totals():
    fail = _ex(reg_failure={"messages": ["<test frame>\nValueError: x", "KeyError: 'k'"], "n_failed": 2})
    t, items, _ = FB.exec_summary(fail)
    assert "Regression tests that did not pass: 2" in t and "Regression failure #1:\n```\n<test frame>\nValueError: x\n```" in t
    assert "Regression failure #2:" in t and "Regression failure #3" not in t
    ok, _, _ = FB.exec_summary(_ex(reg_passed=9))
    assert "Regression tests: no failures." in ok
    none, _, _ = FB.exec_summary(_ex(no_regression_signal=True, reg_total=0, reg_passed=0))
    assert "Regression tests: none for this issue." in none
    for s in (t, ok, none):
        assert "7/9" not in s and "9 passed" not in s and " passed," not in s and "/9" not in s
    timed, _, _ = FB.exec_summary(_ex(timed_out=True))
    assert "(test run timed out)" in timed


def test_reduction_order_keeps_count_line():
    v = _ex(reg_failure={"messages": ["ValueError: x"], "n_failed": 2}, repro_log_tail="L\n" * 40,
            repro_statuses={"t": "FAILED"})
    t1, i1, _ = FB.exec_summary(v, 1)
    assert "Regression tests that did not pass: 2" in t1 and "Regression failure" not in t1 and "output (tail)" in t1
    assert FB.source_block(("p.py", "S", False), 1)[0] and not FB.source_block(("p.py", "S", False), 2)[0]
    t3, _, _ = FB.exec_summary(v, 3)
    assert t3.count("\nL") <= 10
    t4, _, _ = FB.exec_summary(v, 4)
    assert "output (tail)" not in t4 and "per-test status" in t4
    t5, _, _ = FB.exec_summary(v, 5)
    assert "per-test status" not in t5 and "did not pass: 2" in t5
    assert FB.LEVELS == ("full", "no_reg_messages", "no_source", "tail_10", "no_tail", "no_statuses")
