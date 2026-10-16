"""2026-10-12 simple prep 코드 입력(2단계) 목 LLM 테스트: 계약 불변, 테스트 파일·정답 필드 비유입, 골격, 예산 축소, 1단계 실패."""
import dataclasses

import pytest

from .conftest import FakeLLM
from evopatch.data_loader import InstanceView
from evopatch.prep.base import PrepDeps
from evopatch.prep.simple import SimplePrep, code_skeleton, select_candidate_files
from evopatch.redact import mask_token

MASK = mask_token()
from evopatch.specs import ExecSpec
from evopatch.types import Location, PrepFailure, PrepOutput, ReproTest

TEST_FILES = ["pkg/tests/test_core.py"]
SRC = '''import os


@register
@cache(
    size=3,
)
def add(a, b):
    """Add."""
    def inner():
        return 0
    return a + b


class K(Base):
    """K."""

    x = 1

    @property
    def m(self,
          y):
        # see pkg/tests/test_core.py
        return y

    class Inner:
        def deep(self): return 1
'''
OTHER = "def helper():\n    return 2\n" + "\n".join(f"# filler {i}" for i in range(200)) + "\n"
TEST_BODY = "def test_secret():\n    assert 'SECRET_TEST_BODY'\n"
FILES = {"pkg/mod.py": SRC, "pkg/other.py": OTHER, "pkg/tests/test_core.py": TEST_BODY}
GOOD2 = "### LOCATIONS\npkg/mod.py::add\npkg/mod.py::K.m\n### TEST\n```python\ndef test_x():\n    assert 1\n```\n"


class LenLLM(FakeLLM):
    """토큰 수 = 글자 수 // 4 (예산 축소 테스트용)."""
    model = "m"

    def count_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 4


class Ex:
    def __init__(self, files=FILES):
        self.files = files
        self.reads = []

    def list_files(self, iid):
        return sorted(self.files) + ["README.rst"]

    def read_files(self, iid, paths, diff=""):
        self.reads.append(list(paths))
        return {p: self.files.get(p) for p in paths}


def make(cfg, responses, *, budget=12000, files=FILES, llm_cls=LenLLM):
    llm = llm_cls(responses)
    llm.model = "m"
    ex = Ex(files)
    spec = ExecSpec("x__y-1", "a/b", "1", "img", "pytest", (), "parse_log_pytest", tuple(TEST_FILES))
    cfg._data["prep"]["simple"]["repro_paths"]["a/b"] = "tests/test_swe_repro.py"
    cfg._data["prep"]["simple"]["code_input"]["max_tokens"] = budget
    deps = PrepDeps(cfg=cfg, llm=llm, executor=ex, exec_specs={"x__y-1": spec}, regression_pool={"x__y-1": ["r1"]},
                    run_index=0, extra_log={"attempt_id": "a1"})
    return SimplePrep(deps), llm, ex


VIEW = InstanceView("x__y-1", "a/b", "c", "add is wrong", "1", "e")


def test_contract_unchanged_and_two_calls(cfg):
    prov, llm, ex = make(cfg, ["### FILES\npkg/mod.py\n", GOOD2])
    out = prov.prepare(VIEW)
    assert [f.name for f in dataclasses.fields(PrepOutput)] == [
        "instance_id", "repo", "base_commit", "problem_statement", "locations", "repro_test",
        "regression_tests", "test_files", "prep_usage", "prep_provider"]
    assert isinstance(out.locations[0], Location) and isinstance(out.repro_test, ReproTest)
    assert [l.symbol for l in out.locations] == ["add", "K.m"]
    assert out.locations[0].start_line == 4  # 첫 데코레이터 줄부터(해석 규칙 그대로)
    assert out.regression_tests == ["r1"] and out.test_files == TEST_FILES and out.prep_provider == "simple"
    assert [u.stage for u in out.prep_usage] == ["prep", "prep"]
    assert [c["stage"] for c in llm.calls] == ["prep", "prep"]
    assert [c["extra_log"]["prep_step"] for c in llm.calls] == ["files", "locate_test"]
    assert llm.calls[0]["seed"] != llm.calls[1]["seed"]
    assert ex.reads == [["pkg/mod.py"]]  # 후보 파일은 한 번만 읽고 위치 해석에 재사용
    m = prov.meta
    assert m["llm_calls"] == 2 and m["candidate_files"] == ["pkg/mod.py"] and m["code_context_kind"] == "full"
    assert m["code_context_level"] == "full" and m["code_context_tokens"] > 0
    # 기존 meta 필드 유지(리포트·gen0 보고서가 읽는 값)
    for k in ("file_list_level", "file_list_tokens", "locations_requested", "locations_found", "location_kinds",
              "location_failures", "location_range_rule", "finish_reason", "prompt_mask_hits", "location_identifiers"):
        assert k in m, k


def test_code_prompt_has_code_but_no_test_files_or_gold(cfg):
    # 1단계에서 모델이 테스트 파일을 골라도 읽지 않고 버린다(사유 기록)
    prov, llm, ex = make(cfg, ["### FILES\npkg/tests/test_core.py\npkg/mod.py\n", GOOD2])
    prov.prepare(VIEW)
    code = llm.calls[1]["messages"][0]["content"]
    assert "def add(a, b):" in code and "SECRET_TEST_BODY" not in code
    assert "pkg/tests/test_core.py" not in code and f"# see {MASK}" in code  # 코드 입력에도 마스커 적용
    assert all("pkg/tests/test_core.py" not in r for r in ex.reads)
    assert prov.meta["candidate_file_failures"][0]["reason"] == "test_file"
    for c in llm.calls:
        text = c["messages"][0]["content"]
        for bad in ("FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", "hints_text", "SECRET_TEST_BODY"):
            assert bad not in text
    assert prov.meta["prompt_mask_hits"]


def test_skeleton_line_numbers_and_decorators():
    sk = code_skeleton(SRC)
    lines = sk.splitlines()
    assert "    4| @register" in lines and "    5| @cache(" in lines and "    8| def add(a, b):" in lines
    assert any("body omitted (lines 4-12)" in l for l in lines)  # 범위는 첫 데코레이터 줄부터
    assert "inner" not in sk and "return a + b" not in sk and '"""Add."""' not in sk  # 본문·중첩 함수 생략
    assert "   15| class K(Base):" in lines and "   20|     @property" in lines
    assert "   21|     def m(self," in lines and "   22|           y):" in lines  # 여러 줄 시그니처
    assert "# see" not in sk  # 시그니처 뒤 주석은 머리에 넣지 않음
    assert "   26|     class Inner:" in lines and "   27|         def deep(self): return 1" in lines
    assert any("end of class K (lines 15-27)" in l for l in lines)
    assert code_skeleton("def f(:\n  pass\n@d\nclass X:\n  pass\n").splitlines() == [
        "    1| def f(:", "    3| @d", "    4| class X:"]  # 문법 오류 파일: 정의 줄만


def _levels(prov):
    return [t["level"] for t in prov.meta["code_context_reduction"]]


def test_budget_reduction_order(cfg):
    two = "### FILES\npkg/mod.py\npkg/other.py\n"
    full = len("\n\n".join([SRC, OTHER])) // 4
    # 1) 전체가 예산 안
    prov, llm, _ = make(cfg, [two, GOOD2], budget=10 ** 6)
    prov.prepare(VIEW)
    assert _levels(prov) == ["full"] and prov.meta["code_context_kind"] == "full"
    # 2) 전체 초과 → 두 파일 골격
    prov, llm, _ = make(cfg, [two, GOOD2], budget=full)
    prov.prepare(VIEW)
    assert _levels(prov) == ["full", "skeleton"] and prov.meta["code_context_kind"] == "skeleton"
    assert "(skeleton)" in llm.calls[1]["messages"][0]["content"] and "line number" in llm.calls[1]["messages"][0]["content"]
    sk_two = prov.meta["code_context_tokens"]
    # 3) 골격 둘도 초과 → 우선순위가 낮은 후보(뒤)부터 뺀다
    prov, llm, _ = make(cfg, [two, GOOD2], budget=sk_two - 1)
    prov.prepare(VIEW)
    assert _levels(prov) == ["full", "skeleton", "skeleton_drop"]
    assert prov.meta["code_context_files"] == ["pkg/mod.py"] and prov.meta["code_context_dropped_files"] == ["pkg/other.py"]
    # 4) 1순위 골격도 초과 → 앞에서부터 잘라 예산에 맞춤
    prov, llm, _ = make(cfg, [two, GOOD2], budget=60)
    prov.prepare(VIEW)
    assert _levels(prov) == ["full", "skeleton", "skeleton_drop", "skeleton_truncated"]
    assert prov.meta["code_context_tokens"] <= 60 and "skeleton truncated" in llm.calls[1]["messages"][0]["content"]
    # 5) 한 줄도 못 넣으면 사유를 남기고 실패(2단계 호출 없음)
    prov, llm, _ = make(cfg, [two, GOOD2], budget=1)
    with pytest.raises(PrepFailure) as e:
        prov.prepare(VIEW)
    assert e.value.reason == "code_context_over_budget" and len(llm.calls) == 1 and len(e.value.usage) == 1


def test_stage1_failures_recorded(cfg):
    # 머리 없음 → 파싱 실패, 2단계 호출 없음
    prov, llm, _ = make(cfg, ["pkg/mod.py\n"])
    with pytest.raises(PrepFailure) as e:
        prov.prepare(VIEW)
    assert e.value.reason == "no_candidate_files_parsed" and len(llm.calls) == 1 and prov.meta["llm_calls"] == 1
    # 모두 무효 → 사유별 기록 후 실패
    prov, llm, _ = make(cfg, ["### FILES\n1. pkg/nope.py\n- README.rst\n`pkg/tests/test_core.py`\npkg/mod.py\n"])
    with pytest.raises(PrepFailure) as e:
        prov.prepare(VIEW)
    assert e.value.reason == "candidate_files_not_found" and len(llm.calls) == 1
    reasons = [(f["path"], f["reason"]) for f in prov.meta["candidate_file_failures"]]
    assert reasons == [("pkg/nope.py", "not_in_repo"), ("README.rst", "not_python_file"),
                       ("pkg/tests/test_core.py", "test_file"), ("pkg/mod.py", "over_limit")]
    assert prov.meta["candidate_files"] == [] and "prompt_mask_hits" in prov.meta
    # 읽기 실패도 사유로
    files = dict(FILES)
    prov, llm, ex = make(cfg, ["### FILES\npkg/mod.py\npkg/other.py\n", GOOD2], files=files)
    ex.files = {**files, "pkg/other.py": None}
    prov.prepare(VIEW)
    assert prov.meta["candidate_files"] == ["pkg/mod.py"]
    assert prov.meta["candidate_file_failures"] == [{"line": "pkg/other.py", "path": "pkg/other.py", "reason": "read_failed"}]


def test_select_candidate_files_rules():
    chosen, fails = select_candidate_files(["./a/x.py", "a/x.py", "b.py"], {"a/x.py", "b.py"}, set(), 2)
    assert chosen == ["a/x.py"] and [f["reason"] for f in fails] == ["duplicate", "over_limit"]


def test_locations_outside_candidates_still_resolved_and_failures_kept(cfg):
    """위치 해석 규칙은 그대로: 후보 밖 파일의 식별자도 읽어서 해석하고, 못 찾은 식별자는 사유만 남긴다(완화 없음)."""
    resp = "### LOCATIONS\npkg/other.py::helper\npkg/mod.py::m\n### TEST\n```python\ndef test_x():\n    assert 1\n```\n"
    prov, llm, ex = make(cfg, ["### FILES\npkg/mod.py\n", resp])
    out = prov.prepare(VIEW)
    assert [l.symbol for l in out.locations] == ["helper"] and ex.reads == [["pkg/mod.py"], ["pkg/other.py"]]
    assert prov.meta["location_failures"][0]["reason"] == "qualname_mismatch"
    assert prov.meta["locations_in_candidate_files"] == 1
