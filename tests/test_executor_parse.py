from evopatch.executor import SYMPY_FILE_MARK, repro_verdict, select_repro_statuses
from evopatch.types import ReproTest
from swebench.harness.log_parsers import PARSER_REGISTRY


def test_sympy_last_test_parsed_after_normalization():
    log = ("sympy/core/tests/test_swe_repro.py[2] \ntest_a ok\n"
           "test_b ok                                                   [OK]\n")
    raw = PARSER_REGISTRY["parse_log_sympy"](log, None)
    assert "test_b" not in raw  # 하네스 파서 한계
    fixed = PARSER_REGISTRY["parse_log_sympy"](SYMPY_FILE_MARK.sub("", log), None)
    assert fixed == {"test_a": "PASSED", "test_b": "PASSED"}
    fail = "test_a ok\ntest_b F                                                 [FAIL]\n"
    assert PARSER_REGISTRY["parse_log_sympy"](SYMPY_FILE_MARK.sub("", fail), None)["test_b"] == "FAILED"


def test_verdict_rules():
    r = ReproTest(path="pkg/tests/test_swe_repro.py", content="", run_target="pkg/tests/test_swe_repro.py")
    sm = {"pkg/tests/test_swe_repro.py::test_a": "PASSED", "other.py::t": "FAILED"}
    st = select_repro_statuses("astropy/astropy", r, sm)
    assert st == {"pkg/tests/test_swe_repro.py::test_a": "PASSED"} and repro_verdict(st)
    assert not repro_verdict({}) and not repro_verdict({"x": "ERROR"})
