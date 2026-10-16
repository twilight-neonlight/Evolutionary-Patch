"""lab_fixed 전처리(2026-10-14): 제3자 연구 test.py 분해, 연결 불가 경로, 실행 대상, simple 위치 재사용. 도커·LLM 없음(가짜 의존성)."""
import json

import pytest

from evopatch.config import Config
from evopatch.prep import lab_fixed as L
from evopatch.prep.base import PrepDeps, make_provider
from evopatch.prep.simple import SimplePrep
from evopatch.types import Location, PrepFailure, PrepOutput, ReproTest

ORIG = "import pytest\n\n\ndef test_old():\n    assert 1\n"
BLOCK = "\n\ndef test_new_issue():\n    assert f() == 2\n\n\nclass TestMore:\n    def test_x(self):\n        pass\n"


def test_split_prefix_and_rank():
    t, b, r = L.split_lab_test(ORIG + BLOCK, ["a/test_other.py", "a/test_x.py"],
                               {"a/test_other.py": "x = 1\n", "a/test_x.py": ORIG})
    assert (t, b, r) == ("a/test_x.py", BLOCK, 2)
    # 원본 끝 줄바꿈 없이 이어 붙인 경우도 앞부분 일치
    t, b, _ = L.split_lab_test(ORIG.rstrip("\n") + BLOCK, ["a/test_x.py"], {"a/test_x.py": ORIG})
    assert ORIG + b == ORIG.rstrip("\n") + BLOCK  # 원본 + 블록 = 제3자 연구 test.py 그대로
    t, b, _ = L.split_lab_test(ORIG.rstrip("\n") + "x" + BLOCK, ["a/test_x.py"], {"a/test_x.py": ORIG})
    assert ORIG.rstrip("\n") + b == ORIG.rstrip("\n") + "x" + BLOCK


@pytest.mark.parametrize("test_py,cands,orig,reason", [
    (ORIG + BLOCK, [], {}, "대상 파일을 못 찾음(후보 없음)"),
    (ORIG + BLOCK, ["a/t.py"], {"a/t.py": None}, "대상 파일을 못 찾음(후보 파일이 base commit에 없음)"),
    ("import pytest\n\n\ndef test_old():\n    assert 2\n" + BLOCK, ["a/t.py"], {"a/t.py": ORIG}, "원본과 앞부분 불일치"),
    ("def test_brand_new():\n    q = 1\n    assert q\n", ["a/t.py"], {"a/t.py": ORIG}, "새 파일"),
    (ORIG + "\n\n", ["a/t.py"], {"a/t.py": ORIG}, "덧붙인 블록 없음"),
])
def test_split_unlinkable(test_py, cands, orig, reason):
    with pytest.raises(L.LabUnlinkable) as e:
        L.split_lab_test(test_py, cands, orig)
    assert e.value.reason == reason


def test_block_tests_and_run_targets():
    funcs, classes = L.block_tests(ORIG, BLOCK)
    assert funcs == ["test_new_issue"] and classes == ["TestMore"]  # 원본의 test_old는 제외
    assert L.run_target_for("pydata/xarray", "xarray/tests/t.py", funcs, classes) == \
        "xarray/tests/t.py::TestMore xarray/tests/t.py::test_new_issue"
    assert L.run_target_for("sympy/sympy", "sympy/core/tests/t.py", funcs, classes) == "sympy/core/tests/t.py"
    dj_block = "\n\nclass IssueTests(SimpleTestCase):\n    def test_a(self):\n        pass\n"
    f, c = L.block_tests("from django.test import SimpleTestCase\n", dj_block)
    assert c == ["IssueTests"]
    assert L.run_target_for("django/django", "tests/model_fields/tests.py", f, c) == "model_fields.tests.IssueTests"
    with pytest.raises(L.LabUnlinkable):
        L.run_target_for("django/django", "tests/model_fields/tests.py", ["test_a"], [])
    with pytest.raises(L.LabUnlinkable):
        L.run_target_for("pydata/xarray", "x.py", [], [])
    with pytest.raises(L.LabUnlinkable):
        L.block_tests(ORIG, "\n\ndef test_bad(:\n")


# ---------- provider ----------
IID, IID_RR = "own__repo-1", "own__repo-2"


def make_pkg(tmp_path, test_py=ORIG + BLOCK):
    pkg = tmp_path / "pkg"
    (pkg / "manifests").mkdir(parents=True, exist_ok=True)
    (pkg / "manifests" / "final_authoritative_manifest.csv").write_text(
        "instance_id,final_status,raw_terminal_reason\n"
        f"{IID},COMPLETE,\n{IID_RR},REPAIR_REQUIRED,NO_ELIGIBLE_SCENARIO\n")
    d = pkg / "complete" / IID
    d.mkdir(parents=True, exist_ok=True)
    (d / "test.py").write_text(test_py)
    (d / "context.json").write_text(json.dumps({"candidate_test_files": [{"path": "pkg/tests/test_a.py"}]}))
    return pkg


class FakeExec:
    def __init__(self):
        self.calls = 0

    def read_files(self, iid, paths):
        self.calls += 1
        return {p: ORIG for p in paths}


def simple_out(view):
    loc = Location(file="pkg/mod.py", symbol="f", start_line=1, end_line=2, code="def f():\n    return 1\n")
    return PrepOutput(instance_id=view.instance_id, repo=view.repo, base_commit=view.base_commit,
                      problem_statement=view.problem_statement, locations=[loc],
                      repro_test=ReproTest(path="pkg/tests/test_swe_repro.py", content="simple", run_target="x"),
                      regression_tests=["t1"], test_files=["pkg/tests/test_a.py"], prep_usage=[], prep_provider="simple")


def provider(tmp_path, monkeypatch, pkg, calls):
    cfg = Config({"prep": {"provider": "lab_fixed", "lab_fixed": {"package_dir": str(pkg)}}})
    L._manifest_cache.clear()

    def fake_simple(self, view):
        calls.append(view.instance_id)
        return simple_out(view)
    monkeypatch.setattr(SimplePrep, "prepare", fake_simple)
    deps = PrepDeps(cfg=cfg, llm=None, executor=FakeExec(), exec_specs={}, regression_pool={}, run_index=0)
    return make_provider("lab_fixed", deps)


class View:
    def __init__(self, iid, repo="own/repo"):
        self.instance_id, self.repo, self.base_commit, self.problem_statement = iid, repo, "abc", "issue"








def test_simple_provider_unchanged():
    from evopatch.prep.simple import SimplePrep as SP
    deps = PrepDeps(cfg=Config({}), llm=None, executor=None, exec_specs={}, regression_pool={}, run_index=0)
    assert type(make_provider("simple", deps)) is SP and SP.name == "simple"
