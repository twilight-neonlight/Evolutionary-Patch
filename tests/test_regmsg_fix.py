"""회귀 오류 내용 추출기 수정(ANSI 제거, 형식별 구간 인식, 테스트 구간 밖 오류 대체)의 단위 테스트. GPU·도커 없음.
예시 출력(tests/data/regmsg_fix)은 기준 실행에 저장된 회귀 출력에서 경우별로 고른 것이다.
가림 규칙이 바뀌지 않았는지(기존 예시 4개의 결과가 수정 전과 같음)도 함께 본다.
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from evopatch import regmsg as SRC
from evopatch.executor import END, START

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "evopatch").is_dir())
DATA = ROOT / "tests" / "data" / "regmsg_fix"
OLD_DATA = ROOT / "tests" / "data" / "regmsg"


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


M = SRC
BEFORE = None  # 수정 전 모듈과의 비교는 패치를 만들 때 했다(Main에는 수정 후 모듈만 있음)


def case(name):
    meta = json.loads((DATA / f"{name}.json").read_text())
    return (DATA / f"{name}.log").read_text(), meta


def run(name):
    log, m = case(name)
    return M.extract(log, m["failed"], m["all"], m["repo"], m["test_files"]), m


def assert_no_residue(msgs, m):
    names = set()
    for t in m["failed"] + m["all"]:
        names |= {x for x in M.name_parts(t) if len(x) >= 6}
        if t.split("::")[0].endswith(".py"):
            names.add(t.split("::")[0])
    text = "\n".join(msgs)
    for nm in names:
        assert nm not in text, nm
    assert not re.search(r"\btest_\w+", text) and not re.search(r"[\w/]+\.py::\w", text)
    for f in re.findall(r'File "([^"]+)"', text):
        assert not M.is_test_path(f, m["test_files"]) and not M.is_external(f), f
    assert "\x1b" not in text and "INTERNALERROR" not in text
    assert sum(len(x) for x in msgs) <= M.TOTAL_CHARS + 50 and len(msgs) <= M.MAX_MSGS


# 경우 → (마지막 예외 줄에 있어야 할 내용, 저장소 소스 프레임 일부 | None)
EXPECT = {
    "ansi_astropy": ("UnboundLocalError: local variable '_new_re' referenced before assignment", 'File "astropy/io/ascii/qdp.py"'),
    "ansi_sphinx-doc": ("AttributeError: module 'posixpath' has no attribute 'mkdir'", 'File "sphinx/'),
    "django_docstring": ("AssertionError:", None),
    "django_run_crash": ("NameError: name 'AutoField' is not defined", 'File "django/'),
    "pytest_collection_interrupted": ("AttributeError: '_MergeResult' object has no attribute '_variables'", 'File "xarray/core/'),
    "pytest_internalerror": ("AttributeError: 'Namespace' object has no attribute 'fixtures'", 'File "src/_pytest/'),
    "pytest_setup_error": ("OSError:", 'File "lib/matplotlib/'),
    "pytest_short_header": ("OverflowError: cannot convert float infinity to integer", 'File "sklearn/'),
    "sympy_import_error": ("NameError: name 'str_types' is not defined", 'File "sympy/core/sympify.py"'),
    "sympy_runner_crash": ("NameError: name 'MatrixKind' is not defined", 'File "sympy/'),
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_real_outputs_now_extracted_with_masking(name):
    r, m = run(name)
    assert r["messages"], r["fail_reasons"]
    assert r["n_failed"] == len(m["failed"]) and not r["fail_reasons"]
    exc, frame = EXPECT[name]
    assert any(x.splitlines()[-1].startswith(exc) for x in r["messages"])
    if frame:
        assert frame in "\n".join(r["messages"])
    assert_no_residue(r["messages"], m)


@pytest.mark.skipif(BEFORE is None, reason="패치 적용 뒤에는 수정 전 모듈이 없음")
@pytest.mark.parametrize("name", sorted(EXPECT))
def test_these_outputs_failed_before(name):
    log, m = case(name)
    b = BEFORE.extract(log, m["failed"], m["all"], m["repo"], m["test_files"])
    r, _ = run(name)
    assert b["fail_reasons"].get("section_not_found") and len(r["messages"]) >= len(b["messages"])


def test_no_error_output_is_its_own_reason():
    r, m = run("no_error_output")
    assert r["messages"] == [] and r["fail_reasons"] == {"no_error_output": len(m["failed"])}


def test_one_error_for_many_tests_is_counted_once():
    for name in ("pytest_collection_interrupted", "pytest_internalerror", "django_run_crash"):
        r, m = run(name)
        assert len(r["messages"]) == 1 and r["duplicates_skipped"] == len(m["failed"]) - 1


@pytest.mark.parametrize("name", ["django", "sympy", "pytest_sklearn", "pytest_pylint"])
def test_previous_examples_unchanged(name):
    meta = json.loads((OLD_DATA / f"{name}.json").read_text())
    log = (OLD_DATA / f"{name}.log").read_text().split(START, 1)[-1].split(END, 1)[0]
    r = M.extract(log, meta["failed"], meta["all"], meta["repo"], ())
    snap = json.loads((DATA / "previous_examples_messages.json").read_text())[name]
    assert r["messages"] == snap


# ---------- 작은 합성 예시 ----------
def test_strip_control():
    assert M.strip_control("\x1b[1m\x1b[31mE   ValueError: x\x1b[0m\r\nnext\x07") == "E   ValueError: x\nnext"


def test_ansi_pytest_block():
    log = ("\x1b[31m=== FAILURES ===\x1b[0m\n\x1b[31m\x1b[1m____ test_a ____\x1b[0m\n\n    def test_a():\n>       f()\n\n"
           "pkg/tests/t.py:3: \n_ _ _ _ _ _ _ _ _ _\n\n    def f():\n>       raise KeyError('k')\n"
           "\x1b[1m\x1b[31mE       KeyError: 'k'\x1b[0m\n\n\x1b[1m\x1b[31mpkg/m.py\x1b[0m:9: KeyError\n=== 1 failed in 1s ===\n")
    r = M.extract(log, ["pkg/tests/t.py::test_a"], [], "psf/requests", ())
    assert r["messages"] == [f'{M.test_frame()}\n  File "pkg/m.py", line 9, in f\n    raise KeyError(\'k\')\nKeyError: \'k\'']


def test_django_docstring_alias():
    log = ("ERROR: test_m (app.tests.C)\nDoc first line.\n" + "-" * 70 + "\nTraceback (most recent call last):\n"
           '  File "/testbed/tests/app/tests.py", line 3, in test_m\n    f()\n'
           '  File "/testbed/django/x.py", line 9, in f\n    raise ValueError("v")\nValueError: v\n\n' + "-" * 70 + "\nRan 1 test\n")
    r = M.extract(log, ["Doc first line."], ["Doc first line."], "django/django", ())
    assert r["messages"] == [f'{M.test_frame()}\n  File "django/x.py", line 9, in f\n    raise ValueError("v")\nValueError: v']


def test_pytest_short_header_and_setup_prefix():
    log = ("=== ERRORS ===\n_ ERROR at setup of test_b[p] _\n\n    def fx():\n>       g()\n\npkg/conftest.py:2: \n"
           "_ _ _ _ _ _ _\n\n    def g():\n>       raise OSError('o')\nE       OSError: o\n\npkg/m.py:5: OSError\n"
           "=== 1 error in 1s ===\n")
    r = M.extract(log, ["pkg/tests/t.py::test_b[p]"], [], "matplotlib/matplotlib", ())
    assert r["messages"] and r["messages"][0].endswith("OSError: o") and 'File "pkg/m.py", line 5, in g' in r["messages"][0]


def test_pseudo_file_frames_dropped():
    assert M.is_external("<string>") and M.is_external("/opt/x/site-packages/a.py") and not M.is_external("pkg/a.py")
