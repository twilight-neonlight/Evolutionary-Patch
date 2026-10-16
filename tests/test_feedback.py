"""루프 피드백(loop.feedback): off 바이트 동일성, 상한·길이 검사, 마스킹, 기록 필드, 출처 기록. GPU·도커 없음.
2026-10-16 최종 후보 지시: 실패 회귀 이름·회귀 실패 메시지는 넣지 않는다(A2 유지) — 관련 검사를 그에 맞게 바꿈."""
import importlib.util
import json
from pathlib import Path

import pytest

from evopatch import feedback as FB
from evopatch import operators as O
from evopatch import prompts
from evopatch.population import Individual
from evopatch.redact import TestFileMasker
from evopatch.types import Candidate, ExecResult

import os
ORIG_PROMPTS = Path(os.environ["EVOPATCH_REF_PROMPTS"]) if os.environ.get("EVOPATCH_REF_PROMPTS") else None  # 기준본 prompts.py(선택)
GOOD = "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = a + b + 1\n>>>>>>> REPLACE\n"


def orig_prompts():
    if ORIG_PROMPTS is None:
        pytest.skip("EVOPATCH_REF_PROMPTS 없음(기준본 대조)")
    spec = importlib.util.spec_from_file_location("evopatch._orig_prompts", ORIG_PROMPTS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def view(valid=True, rp=False, **kw):
    if not valid:
        return {"cid": "c", "valid": False, "diff": "", "raw": "raw edits", "exec": None}
    ex = {"repro_pass": rp, "repro_statuses": {"pkg/tests/test_r.py::test_a": "FAILED"}, "repro_log_tail": "E  assert 1 == 2",
          "reg_passed": 3, "reg_total": 5, "n_failed_regressions": 2, "no_regression_signal": False, "timed_out": False,
          "error": None, **kw}
    return {"cid": "c", "valid": True, "diff": "diff --git a/x b/x\n+y\n", "raw": "r", "exec": ex}


def parent(cid, v):
    ex = ExecResult(repro_pass=False, reg_passed=3, reg_total=5, timed_out=False, output_path="") if v["valid"] else None
    return Individual(cid=cid, gen_born=0, idx=0, op="init", parent_ids=(),
                      candidate=Candidate(edits=(), patch_hash="h", diff=v["diff"]), valid=v["valid"], raw=v["raw"],
                      exec=ex, exec_view=v["exec"])


# ---------- off: 기존 프롬프트와 바이트 동일 ----------
@pytest.mark.parametrize("a,b", [(view(), view(rp=True)), (view(valid=False), view()), (view(timed_out=True), view(valid=False))])
def test_off_prompts_byte_identical_to_original(locations, a, b):
    o = orig_prompts()
    assert prompts.crossover_prompt("bug text", locations, a, b) == o.crossover_prompt("bug text", locations, a, b)
    assert prompts.mutation_prompt("bug text", locations, a, b) == o.mutation_prompt("bug text", locations, a, b)
    assert prompts.init_prompt("bug text", locations, 10) == o.init_prompt("bug text", locations, 10)
    assert prompts.edit_format() == o.EDIT_FORMAT and prompts.system() == o.SYSTEM


def test_off_operators_send_original_prompts(make_ctx, locations):
    o = orig_prompts()
    a, b = view(), view(rp=True)
    ctx, llm = make_ctx([GOOD, GOOD])
    assert ctx.feedback == "off"
    O.crossover(ctx, parent("p0", a), parent("p1", b), gen=1, idx=0)
    O.mutate(ctx, parent("ch", a), parent("p0", a), parent("p1", b), gen=1, idx=0)
    assert llm.calls[0]["messages"][1]["content"] == o.crossover_prompt("bug", locations, a, b)
    assert llm.calls[1]["messages"][1]["content"] == o.mutation_prompt("bug", locations, a, b)
    assert llm.calls[0]["extra_log"]["feedback_items"] == [] and llm.calls[0]["extra_log"]["feedback_mode"] == "off"


def test_mutation_only_leaves_crossover_unchanged(make_ctx, locations):
    o = orig_prompts()
    a, b = view(), view()
    ctx, llm = make_ctx([GOOD, GOOD])
    ctx.feedback = "mutation_only"
    O.crossover(ctx, parent("p0", a), parent("p1", b), gen=1, idx=0)
    O.mutate(ctx, parent("ch", a), parent("p0", a), parent("p1", b), gen=1, idx=0)
    assert llm.calls[0]["messages"][1]["content"] == o.crossover_prompt("bug", locations, a, b)
    m = llm.calls[1]["messages"][1]["content"]
    assert "Regression tests that did not pass: 2" in m and llm.calls[1]["extra_log"]["feedback_mode"] == "mutation_only"
    assert "3/5" not in m


# ---------- both: 내용·문구 ----------
def test_both_crossover_and_mutation_content(make_ctx, locations):
    # 실행 요약에 이름·메시지 키가 있어도(옛 런 형식) 프롬프트에는 들어가지 않는다
    a = view(failed_regressions_head=["pkg/tests/test_x.py::LEAK_k"], first_failure_message="____ LEAK_k ____\nE  boom")
    b = view(valid=False)
    ctx, llm = make_ctx([GOOD, GOOD])
    ctx.feedback = "both"
    ctx.repro_source = ("pkg/tests/test_r.py", "def test_a():\n    assert f() == 2\n", True)
    O.crossover(ctx, parent("p0", a), parent("p1", b), gen=1, idx=0)
    O.mutate(ctx, parent("ch", a), parent("p0", a), parent("p1", b), gen=1, idx=0)
    x, m = (c["messages"][1]["content"] for c in llm.calls)
    for t in (x, m):
        assert "Reproduction test: FAILED" in t and "per-test status:" in t and "output (tail):" in t
        assert "Regression tests that did not pass: 2" in t and "3/5" not in t and "Execution results: NONE" in t
        assert "LEAK" not in t and "failed regression tests" not in t and "first failure message" not in t
        assert "only the appended part is shown" in t and "def test_a():" in t
    assert "Refer to the parent patches and their test results above." in x and "Write ONE new patch that semantically combines" in x
    assert "Using these failures as feedback, re-implement the fix at the same place with a DIFFERENT approach." in m
    assert "Refer to the parent patches and their test results above." not in m
    e = llm.calls[0]["extra_log"]
    assert e["feedback_mode"] == "both" and set(e["feedback_items"]) == {
        "repro_result", "repro_statuses", "repro_output_tail", "regression_result", "repro_test_source"}
    assert e["feedback_reg_msgs"] == 0 and "feedback_reg_extract_fail" in e and "feedback_reg_scrub" in e
    assert e["feedback_reduction"] == "full" and e["feedback_over_limit"] is False and "feedback_est_tokens" in e


# ---------- 상한 ----------
def test_item_caps():
    many = {f"t{i}": "FAILED" for i in range(50)}
    v = view(repro_statuses=many, repro_log_tail="\n".join(f"line {i} " + "x" * 300 for i in range(100)))
    t, items, capped = FB.exec_summary(v)
    assert "(first 20 of 50)" in t and t.count("=FAILED") == 20
    tail = t.split("output (tail):\n")[1].split("\nRegression")[0]
    assert len(tail) <= FB.TAIL_CHARS + 20 and "line 99" in tail and "line 59" not in tail
    assert capped == {"repro_statuses", "repro_output_tail"}
    long_status = view(repro_statuses={"t" + "k" * 400 + str(i): "FAILED" for i in range(10)})
    t2, _, c2 = FB.exec_summary(long_status)
    assert "repro_statuses" in c2 and len(t2.split("per-test status: ")[1].split("\n")[0]) <= FB.STATUS_CHARS + 40
    s, c = FB.source_block(("p.py", "q" * (FB.SOURCE_CHARS + 500), False))
    assert c and len(s) < FB.SOURCE_CHARS + 200


# ---------- 길이 검사: 피드백만 줄이고 이슈·코드는 그대로 ----------
def test_length_check_reduces_feedback_only(make_ctx, locations):
    big_issue = "I" * int(FB.LIMIT_TOKENS * FB.CHARS_PER_TOKEN * 0.98)
    a = view(repro_log_tail="L\n" * 40)
    ctx, llm = make_ctx([GOOD])
    ctx.feedback = "both"
    ctx.problem_statement = big_issue
    ctx.repro_source = ("p.py", "S" * 7000, False)
    O.crossover(ctx, parent("p0", a), parent("p1", a), gen=1, idx=0)
    e = llm.calls[0]["extra_log"]
    u = llm.calls[0]["messages"][1]["content"]
    assert big_issue in u and "pkg/mod.py ::" in u  # 이슈·코드는 줄지 않음
    assert e["feedback_reduction"] != "full" and e["feedback_truncated"] is True
    assert "repro_test_source" not in e["feedback_items"]
    # 이슈만으로 한도를 넘으면 모든 단계 뒤 over_limit 기록(그대로 보냄)
    ctx2, llm2 = make_ctx([GOOD])
    ctx2.feedback = "both"
    ctx2.problem_statement = "I" * int(FB.LIMIT_TOKENS * FB.CHARS_PER_TOKEN * 1.1)
    O.crossover(ctx2, parent("p0", a), parent("p1", a), gen=1, idx=0)
    e2 = llm2.calls[0]["extra_log"]
    assert e2["feedback_over_limit"] is True and e2["feedback_reduction"] == "no_statuses"


def test_retry_rechecks_length_with_suffix(make_ctx):
    a = view()
    BAD = "pkg/mod.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"
    ctx, llm = make_ctx([BAD, GOOD])
    ctx.feedback = "both"
    O.crossover(ctx, parent("p0", a), parent("p1", a), gen=1, idx=0)
    u2 = llm.calls[1]["messages"][1]["content"]
    assert "Your previous answer was rejected" in u2 and "Reproduction test: FAILED" in u2
    assert llm.calls[1]["extra_log"]["feedback_mode"] == "both"


# ---------- 마스킹 ----------
def test_masker_hides_test_paths_in_new_items(make_ctx, locations):
    """새 항목(테스트별 상태, 재현 출력, 재현 테스트 소스)에도 test_files 마스킹이 걸린다."""
    tf = ["tests/test_x.py", "tests/admin_views/tests.py"]
    a = view(repro_statuses={"tests/test_x.py::test_k": "FAILED", "test_m (admin_views.tests.AdminTests)": "PASSED"},
             repro_log_tail='File "/testbed/tests/test_x.py", line 3, in test_k\n  test_x.py failed',
             reg_failure={"n_failed": 2, "messages": ["OSError: cannot open tests/test_x.py"]})
    ctx, llm = make_ctx([GOOD, GOOD])
    ctx.feedback = "both"
    ctx.mask = TestFileMasker(tf)
    ctx.repro_source = ("tests/test_x.py", "from tests.test_x import helper\ndef test_new():\n    pass\n", True)
    O.crossover(ctx, parent("p0", a), parent("p1", a), gen=1, idx=0)
    O.mutate(ctx, parent("ch", a), parent("p0", a), parent("p1", a), gen=1, idx=0)
    for c in llm.calls:
        u = c["messages"][1]["content"]
        assert "tests/test_x.py" not in u and "admin_views.tests" not in u and " test_x.py" not in u
        assert "tests.test_x" not in u
        assert "<test file>::test_k" in u and "(<test file>.AdminTests)" in u
        assert "OSError: cannot open <test file>" in u  # 회귀 오류 내용에도 마스킹
        assert c["extra_log"]["prompt_mask_hits"] > 0
        assert all("tests/test_x.py" not in b for b in c["extra_log"]["feedback_reg_blocks"])


# ---------- 러너 기록 ----------
def _runner(tmp_path, mode):
    import evopatch.runner as R
    from types import SimpleNamespace
    from evopatch.config import Config
    from evopatch.logs import JsonlLog
    cfg = Config.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "test_config.yaml")
    d = cfg.as_dict()
    if mode is not None:
        d.setdefault("loop", {})["feedback"] = mode
    llm = SimpleNamespace(run_id="t", model="m", call_log=JsonlLog(tmp_path / "c.jsonl"))
    return R.IssueRunner(cfg=Config(d), run_dir=tmp_path, run_id="t", llm=llm, executor=None, exec_specs={}, regression_pool={})


def test_runner_mode_validation_and_algo_version(tmp_path, monkeypatch):
    import evopatch.runner as R
    from evopatch.data_loader import InstanceView
    assert _runner(tmp_path, None).feedback == "off"
    with pytest.raises(ValueError):
        _runner(tmp_path, "always")

    class Boom:
        meta = {}

        def prepare(self, view):
            raise RuntimeError("x")
    monkeypatch.setattr(R, "make_provider", lambda n, d: Boom())
    row = _runner(tmp_path, "both").run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))
    assert row["algo_version"] == "feedback_both"
    assert json.loads((tmp_path / "issues.jsonl").read_text().splitlines()[-1])["algo_version"] == "feedback_both"
    r2 = _runner(tmp_path / "v2", "both")
    r2.cfg = type(r2.cfg)({**r2.cfg.as_dict(), "run": {**r2.cfg.as_dict()["run"], "algo_version": "final_candidate_v2"}})
    (tmp_path / "v2").mkdir(exist_ok=True)
    assert r2.run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))["algo_version"] == "final_candidate_v2"


def test_exec_view_same_for_off_and_both_no_regression_names(tmp_path):
    from evopatch.executor import END, START
    out = tmp_path / "ex"
    out.mkdir()
    (out / "regression_status.json").write_text(json.dumps({"tests/t.py::test_a": "FAILED", "tests/t.py::test_b": "PASSED"}))
    (out / "regression.log").write_text(f"{START}\n____ test_a ____\nE  bad\n{END}\n")
    (out / "repro.log").write_text(f"{START}\nrepro out\n{END}\n")
    ex = ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path=str(out))
    ind = Individual(cid="c", gen_born=0, idx=0, op="init", parent_ids=(), candidate=Candidate(edits=(), patch_hash="h", diff="d"),
                     valid=True, raw="r", exec=ex)
    reg = ["tests/t.py::test_a", "tests/t.py::test_b"]
    v_off = _runner(tmp_path / "a", "off")._exec_view(ind, reg, iid="i", attempt_id="x")
    v = _runner(tmp_path / "b", "both")._exec_view(ind, reg, iid="i", attempt_id="x")
    assert "reg_failure" not in v_off and "test_a" not in json.dumps(v_off)
    assert {k: v[k] for k in v_off} == v_off and "test_a" not in json.dumps(v)  # 이름 없이 오류 내용만
    assert v["reg_failure"]["n_failed"] == 1


def test_regression_names_never_reach_prompts_with_feedback_both(tmp_path, make_ctx, locations):
    """A2 유지(2026-10-16): feedback both에서도 실패 회귀 이름·회귀 로그 내용이 교배·변이 프롬프트에 없다."""
    from evopatch.executor import END, START
    leak = "tests/test_mod.py::LEAKCHECK_xyz"
    out = tmp_path / "ex"
    out.mkdir()
    (out / "regression_status.json").write_text(json.dumps({leak: "FAILED", "tests/test_mod.py::ok": "PASSED"}))
    (out / "regression.log").write_text(
        f"{START}\n==== FAILURES ====\n____ LEAKCHECK_xyz ____\n\n    def LEAKCHECK_xyz():\n>       f()\n\ntests/test_mod.py:3: \n"
        f"_ _ _ _ _ _ _ _ _ _ _ _\n\n    def f():\n>       raise ValueError('boom')\nE       ValueError: boom\n\n"
        f"pkg/m.py:9: ValueError\n==== 1 failed, 1 passed in 0.1s ====\n{END}\n")
    (out / "repro.log").write_text(f"{START}\nrepro out\n{END}\n")
    ex = ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path=str(out))
    ind = Individual(cid="c", gen_born=0, idx=0, op="init", parent_ids=(), candidate=Candidate(edits=(), patch_hash="h", diff="d"),
                     valid=True, raw="r", exec=ex)
    ind.exec_view = _runner(tmp_path / "r", "both")._exec_view(ind, [leak, "tests/test_mod.py::ok"], iid="i", attempt_id="x")
    ctx, llm = make_ctx([GOOD, GOOD])
    ctx.feedback = "both"
    ctx.repro_source = ("t.py", "def test_a():\n    pass\n", False)
    O.crossover(ctx, ind, ind, gen=1, idx=0)
    O.mutate(ctx, ind, ind, ind, gen=1, idx=0)
    assert ind.exec_view["reg_failure"]["messages"] and "LEAK" not in json.dumps(ind.exec_view)
    for c in llm.calls:
        u = c["messages"][1]["content"]
        assert "LEAK" not in u and "test_mod" not in u and "1/2" not in u
        assert "Regression tests that did not pass: 1" in u and "ValueError: boom" in u and "<test frame>" in u
        assert 'File "pkg/m.py", line 9, in f' in u and c["extra_log"]["feedback_reg_msgs"] == 2


def test_repro_source_lab_block_only(tmp_path):
    from evopatch.types import PrepOutput, ReproTest
    r = _runner(tmp_path, "both")
    prep = PrepOutput(instance_id="i", repo="a/b", base_commit="c", problem_statement="p", locations=[],
                      repro_test=ReproTest(path="t/test_a.py", content="ORIG\nBLOCK\n", run_target="x"),
                      regression_tests=[], test_files=[], prep_usage=[], prep_provider="hybrid")
    r._prep_meta["a1"] = {"test_source": "lab", "lab_block_offset": 5}
    assert r._repro_source(prep, "a1") == ("t/test_a.py", "BLOCK\n", True)
    r._prep_meta["a2"] = {"test_source": "llm"}
    assert r._repro_source(prep, "a2") == ("t/test_a.py", "ORIG\nBLOCK\n", False)
    assert _runner(tmp_path / "o", "off")._repro_source(prep, "a1") is None


# ---------- 새 실행 출처 기록 ----------
def test_provenance_and_code_snapshot(tmp_path):
    from evopatch import cli
    from evopatch.config import Config
    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root / "config" / "main.yaml.example")
    p = cli.provenance(cfg)
    assert any(k.startswith("src/evopatch/") for k in p["modules_sha256"])
    assert p["env"]["n_packages"] > 0 and p["env"]["grading_code_sha256"]
    s = cli.save_code_snapshot(tmp_path)
    assert (tmp_path / "code_snapshot" / "src" / "evopatch" / "feedback.py").exists() and s["files"] > 10

