import ast
"""hybrid_v2 전처리(2026-10-16 최종 후보 지시 3·4): 위치 실험 V2 코드 대조, 경로별 위치 확정·fallback, base 실행 재사용, 기록 필드.
가짜 LLM·실행기만 쓴다(GPU·도커 없음)."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evopatch.config import Config
from evopatch.prep import hybrid_v2 as H
from evopatch.prep.base import PrepDeps, make_provider
from evopatch.types import ExecResult, PrepFailure, ReproTest

ROOT = Path(__file__).resolve().parents[1]
import os
PILOT = Path(os.environ["EVOPATCH_REF_HYBRID_V2"]) if os.environ.get("EVOPATCH_REF_HYBRID_V2") else None  # 기준본 hybrid_v2.py(선택)
PILOT_V2 = PILOT
needs_pilot = pytest.mark.skipif(PILOT is None, reason="EVOPATCH_REF_HYBRID_V2 없음(기준본 대조)")
IID = "own__repo-1"
SRC = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
TEST = "from pkg.m import f\n\ndef test_x():\n    assert f() == 2\n"


# ---------- 위치 실험 V2 코드와 글자 그대로 같은지 ----------




# ---------- 가짜 LLM·실행기 ----------
class LLM:
    model = "m"

    def __init__(self, v2_text):
        self.calls, self.v2_text = [], v2_text

    def count_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 4

    def chat(self, messages, *, stage, instance_id, generation, max_tokens, seed, extra_log, temperature, top_p, **k):
        step = extra_log["prep_step"]
        self.calls.append({"step": step, "user": messages[0]["content"], "seed": seed})
        text = {"files": "### FILES\npkg/m.py",
                "locate_test": f"### LOCATIONS\npkg/m.py::f\n### TEST\n```python\n{TEST}```",
                "locate_test_V2": self.v2_text}[step]
        return SimpleNamespace(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=5)


class Ex:
    def __init__(self, base_error=None):
        self.runs, self.base_error = [], base_error

    def list_files(self, iid):
        return ["pkg/m.py", "pkg/tests/test_m.py"]

    def read_files(self, iid, paths, diff=""):
        return {p: SRC if p == "pkg/m.py" else None for p in paths}

    def run_diff(self, diff, *, instance_id, base_commit, regression_tests, repro, out_dir):
        self.runs.append(repro)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "repro.log").write_text("FAILED test_x - assert 1 == 2\nTraceback (most recent call last):\n")
        return ExecResult(repro_pass=False, reg_passed=0, reg_total=0, timed_out=False, output_path=str(out_dir),
                          repro_statuses={"t::test_x": "FAILED"}, error=self.base_error)

    def run_base_repro(self, *a):
        raise AssertionError("base는 전처리에서 이미 실행")


def make(tmp_path, v2_text="### LOCATIONS\npkg/m.py::g\n### TEST\n```python\nx\n```", base_error=None, fmt="old_appended"):
    pkg = tmp_path / "pkg"
    (pkg / "manifests").mkdir(parents=True, exist_ok=True)
    (pkg / "manifests" / "final_authoritative_manifest.csv").write_text("instance_id,final_status,raw_terminal_reason\n")
    d = Config.load(ROOT / "config" / "main.yaml.example").as_dict()
    d["prep"]["provider"] = "hybrid_v2"
    d.setdefault("external", {})["repro_tests_dir"] = str(pkg)
    d["external"]["repro_tests_format"] = fmt
    d["prep"]["simple"]["repro_paths"] = {"own/repo": "pkg/tests/test_swe_repro.py"}
    from evopatch.prep import lab_fixed
    lab_fixed._manifest_cache.clear()
    llm, ex = LLM(v2_text), Ex(base_error)
    specs = {IID: SimpleNamespace(test_files=("pkg/tests/test_m.py",), test_cmd_prefix="pytest")}
    deps = PrepDeps(cfg=Config(d), llm=llm, executor=ex, exec_specs=specs, regression_pool={IID: ["r1"]}, run_index=0,
                    work_dir=tmp_path / "work")
    return make_provider("hybrid_v2", deps), llm, ex


class View:
    instance_id, repo, base_commit, version = IID, "own/repo", "abc", "1"
    problem_statement = "f returns 1 but should return 2"


def lab_ok(p, tmp_path):
    """제3자 연구 테스트를 쓸 수 있는 경우를 흉내(hybrid._lab_test의 base 실행 결과 저장 포함)."""
    def fake(view, idir):
        repro = ReproTest(path="pkg/tests/test_m.py", content="ORIG\n" + TEST, run_target="pkg/tests/test_m.py::test_x")
        p.meta.update(lab_block_offset=5)
        out = tmp_path / "work" / "lab_test_base"
        b = p.deps.executor.run_diff("", instance_id=IID, base_commit="abc", regression_tests=None, repro=repro, out_dir=out)
        p._lab_base = (b, out)
        return repro, None
    (tmp_path / "pkg" / "complete" / IID).mkdir(parents=True, exist_ok=True)
    p._lab_test = fake


# ---------- LLM 테스트 경로 ----------
def test_llm_path_v2_locations_and_prompt_is_call2_plus_block(tmp_path):
    p, llm, ex = make(tmp_path)
    out = p.prepare(View())
    assert [c["step"] for c in llm.calls] == ["files", "locate_test", "locate_test_V2"]
    assert [l.symbol for l in out.locations] == ["g"] and p.meta["location_source"] == "V2"
    assert out.repro_test.content == TEST and p.meta["test_source"] == "llm"
    assert len(ex.runs) == 1 and p.base_result is not None and p.meta["base_exec"]["ok"] is True
    c2, v2 = llm.calls[1], llm.calls[2]
    assert c2["seed"] == v2["seed"]  # 위치 실험와 같이 simple 호출 ②와 같은 시드
    block = v2["user"].split("</issue>\n\n", 1)[1].split("\n\nTask:", 1)[0]
    assert block.startswith("Reproduction test for this issue (saved as `pkg/tests/test_swe_repro.py`)")
    assert "Output of running this test on the current (unfixed) code:" in block and "assert 1 == 2" in block
    assert v2["user"].replace(block + "\n\n", "", 1) == c2["user"]  # 블록 말고는 호출 ②와 바이트 동일
    assert p.meta["fallback_reasons"] == {"test": "산출물 없음", "location": None}
    assert p.meta["v2_attempted"] and p.meta["v2_locations_found"] == 1


def test_llm_path_base_failure_uses_simple_locations(tmp_path):
    p, llm, _ = make(tmp_path, base_error="setup_failed")
    out = p.prepare(View())
    assert [c["step"] for c in llm.calls] == ["files", "locate_test"]
    assert [l.symbol for l in out.locations] == ["f"] and p.meta["location_source"] == "simple_prep"
    assert p.meta["fallback_reasons"]["location"] == H.V2_FAIL_BASE and p.meta["base_exec"]["ok"] is False


def test_llm_path_v2_no_location_uses_simple(tmp_path):
    p, llm, _ = make(tmp_path, v2_text="### LOCATIONS\npkg/m.py::nope\n### TEST\n```python\nx\n```")
    out = p.prepare(View())
    assert [l.symbol for l in out.locations] == ["f"] and p.meta["location_source"] == "simple_prep"
    assert p.meta["fallback_reasons"]["location"] == H.V2_NO_LOC


# ---------- 제3자 연구 테스트 경로 ----------
def test_lab_path_v2_only_no_call2(tmp_path):
    p, llm, ex = make(tmp_path)
    lab_ok(p, tmp_path)
    out = p.prepare(View())
    assert [c["step"] for c in llm.calls] == ["files", "locate_test_V2"]  # 위치가 나오면 호출 ② 없음
    assert p.meta["test_source"] == "lab" and p.meta["location_source"] == "V2" and len(ex.runs) == 1
    v2 = llm.calls[1]["user"]
    assert "appended to the existing test file `<test file>`" in v2 and "ORIG" not in v2  # 경로 마스킹, 덧붙인 부분만
    assert out.repro_test.path == "pkg/tests/test_m.py" and p.base_result.repro_statuses == {"t::test_x": "FAILED"}
    assert p.meta["base_exec"]["ok"] is True and p.meta["fallback_reasons"] == {"test": None, "location": None}


def test_lab_path_fallbacks_simple_then_lab_clue(tmp_path, monkeypatch):
    p, llm, _ = make(tmp_path, v2_text="nothing")
    lab_ok(p, tmp_path)
    out = p.prepare(View())
    assert [c["step"] for c in llm.calls] == ["files", "locate_test_V2", "locate_test"]
    assert p.meta["location_source"] == "simple_prep" and [l.symbol for l in out.locations] == ["f"]
    p2, llm2, _ = make(tmp_path, v2_text="nothing")
    lab_ok(p2, tmp_path)
    monkeypatch.setattr(p2, "_call2", lambda *a: (None, []) if a[3] == "locate_test" else H.HybridV2Prep._call2(p2, *a))
    from evopatch.types import Location
    monkeypatch.setattr(p2, "_lab_locations", lambda v, i: [Location("pkg/m.py", "g", 5, 6, "def g():\n    return 2")])
    out2 = p2.prepare(View())
    assert p2.meta["location_source"] == "lab_fallback" and out2.locations[0].symbol == "g"
    p3, _, _ = make(tmp_path, v2_text="nothing")
    lab_ok(p3, tmp_path)
    monkeypatch.setattr(p3, "_call2", lambda *a: (None, []) if a[3] == "locate_test" else H.HybridV2Prep._call2(p3, *a))
    monkeypatch.setattr(p3, "_lab_locations", lambda v, i: [])
    with pytest.raises(PrepFailure) as e:
        p3.prepare(View())
    assert e.value.reason == "locations_not_found" and p3.meta["location_source"] is None


# ---------- 러너: base 실행 재사용, 기록 필드 ----------
def test_runner_reuses_prep_base_run_and_records_fields(tmp_path, monkeypatch):
    import evopatch.runner as R
    from evopatch.logs import JsonlLog
    from evopatch.data_loader import InstanceView
    p, llm, ex = make(tmp_path)
    cfg = p.deps.cfg
    d = cfg.as_dict()
    d["run"]["algo_version"] = "final_candidate_v2"
    llm_obj = SimpleNamespace(run_id="t", model="m", call_log=JsonlLog(tmp_path / "c.jsonl"))
    r = R.IssueRunner(cfg=Config(d), run_dir=tmp_path / "run", run_id="t", llm=llm_obj, executor=ex, exec_specs={},
                      regression_pool={})
    monkeypatch.setattr(R, "make_provider", lambda n, deps: p)
    monkeypatch.setattr(R.IssueRunner, "_evolve", lambda self, *a: (_ for _ in ()).throw(RuntimeError("stop after prep")))
    row = r.run_issue(InstanceView(IID, "own/repo", "abc", "issue", "1", "e"))
    assert row["repro_on_base"]["from_prep"] is True and row["repro_on_base"]["n_tests"] == 1
    assert row["test_source"] == "llm" and row["location_source"] == "V2" and row["base_exec"]["ok"] is True
    assert row["fallback_reasons"] == {"test": "산출물 없음", "location": None}
    assert row["algo_version"] == "final_candidate_v2" and "n_llm_calls" in row and "wall_clock_s" in row
    assert "total_prompt_tokens" in row and "total_completion_tokens" in row


def test_hybrid_v2_source_uses_no_gold_fields():
    text = (ROOT / "src" / "evopatch" / "prep" / "hybrid_v2.py").read_text()
    for bad in ("test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", ".patch", "gold"):
        assert bad not in text



# ---------- 파일럿 기준본과 출력이 같은지(원본 함수 소스를 읽어 그 자리에서 실행, EVOPATCH_PILOT_ROOT가 있을 때만) ----------
def _pilot_v2_funcs():
    src = PILOT_V2.read_text()
    tree = ast.parse(src)
    ns = {}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("trim_log", "lab_block", "llm_block", "v2_block"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("BLOCK_CHARS",) or isinstance(t, ast.Tuple) for t in n.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "pilot_hybrid_v2", "exec"), ns)
    return ns


@needs_pilot
def test_v2_blocks_and_trim_log_same_output_as_pilot():
    o = _pilot_v2_funcs()
    assert (H.HEAD_CHARS, H.TAIL_CHARS, H.BLOCK_CHARS, H.CTX_MARGIN) == (3000, 6000, 16000, 256)
    log = "".join(f"line {i} " + "x" * 50 + "\n" for i in range(400))
    big = "def test_a():\n    assert 1\n" * 1200
    for head, tail in ((3000, 6000), (1500, 3000), (10, 20), (100000, 100000)):
        assert H.trim_log(log, head, tail) == o["trim_log"](log, head, tail)
    for blk in (TEST, big):
        assert H.lab_block("pkg/tests/test_m.py", blk) == o["lab_block"]("pkg/tests/test_m.py", blk)
        assert H.llm_block("pkg/tests/test_swe_repro.py", blk) == o["llm_block"]("pkg/tests/test_swe_repro.py", blk)
        v1 = H.lab_block("t.py", blk)
        for head, tail in ((3000, 6000), (10, 20)):
            assert H.v2_block(v1, log, head, tail) == o["v2_block"](v1, log, head, tail)



# ---------- 새 형식(new_standalone): 독립 generated_test.py ----------
class ExS(Ex):
    """base 실행 결과를 정할 수 있는 실행기(수집 수·로그)."""

    def __init__(self, statuses, log="", base_error=None, timed_out=False):
        super().__init__(base_error)
        self.statuses, self.log, self.to = statuses, log, timed_out

    def run_diff(self, diff, *, instance_id, base_commit, regression_tests, repro, out_dir):
        self.runs.append(repro)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "repro.log").write_text(self.log)
        return ExecResult(repro_pass=False, reg_passed=0, reg_total=0, timed_out=self.to, output_path=str(out_dir),
                          repro_statuses=self.statuses, error=self.base_error)


def make_new(tmp_path, test_text, ex):
    p, llm, _ = make(tmp_path, fmt="new_standalone")
    (tmp_path / "pkg" / IID).mkdir(parents=True, exist_ok=True)
    if test_text is not None:
        (tmp_path / "pkg" / IID / "generated_test.py").write_text(test_text)
    (tmp_path / "pkg" / IID / "issue_clue.json").write_text('{"fault_locations": []}')
    p.deps.executor = ex
    return p, llm


def test_standalone_used_whole_file_at_fixed_path(tmp_path):
    ex = ExS({"t::test_x": "FAILED"}, "FAILED test_x")
    p, llm = make_new(tmp_path, TEST, ex)
    out = p.prepare(View())
    assert p.meta["test_source"] == "lab" and p.meta["lab_format"] == "new_standalone"
    assert out.repro_test.path == "pkg/tests/test_swe_repro.py" and out.repro_test.content == TEST
    assert out.repro_test.run_target == "pkg/tests/test_swe_repro.py" and ex.runs[0].content == TEST
    v2 = next(c for c in llm.calls if c["step"] == "locate_test_V2")
    assert "Reproduction test for this issue (saved as `pkg/tests/test_swe_repro.py`)" in v2["user"]
    assert "appended to the existing test file" not in v2["user"]
    assert p.meta["lab_block_offset"] == 0  # 피드백 재현 테스트 소스 = 파일 전체(상한은 feedback이 적용)


@pytest.mark.parametrize("test_text,ex,reason", [
    ("def test_x(:\n", ExS({}, ""), "파싱 불가"),
    (TEST, ExS({}, "ImportError: cannot import name 'zz' from 'pkg.m'\n"), "import 오류"),
    (TEST, ExS({}, "collected 0 items\n"), "수집 0"),
    (TEST, ExS({}, "", base_error="setup_failed"), "base 실행 오류"),
    (TEST, ExS({}, "", timed_out=True), "base 시간 초과"),
    (None, ExS({}, ""), "산출물 없음"),
])
def test_standalone_fallback_reasons_then_llm_test(tmp_path, test_text, ex, reason):
    p, llm = make_new(tmp_path, test_text, ex)
    p.prepare(View())
    assert p.meta["test_source"] == "llm" and p.meta["fallback_reasons"]["test"] == reason


def test_runner_feedback_source_is_whole_standalone_file(tmp_path):
    import evopatch.runner as R
    from evopatch.types import PrepOutput
    r = R.IssueRunner.__new__(R.IssueRunner)
    r.feedback = "both"
    r._prep_meta = {"a": {"test_source": "lab", "lab_block_offset": 0}}
    prep = PrepOutput(instance_id=IID, repo="own/repo", base_commit="abc", problem_statement="x", locations=[],
                      repro_test=ReproTest(path="pkg/tests/test_swe_repro.py", content=TEST, run_target="x"),
                      regression_tests=[], test_files=[], prep_usage=[], prep_provider="hybrid_v2")
    assert r._repro_source(prep, "a") == ("pkg/tests/test_swe_repro.py", TEST, False)
