"""8.1-3 검증: 모의 LLM으로 파싱·검사·재시도 경로."""
import pytest

from evopatch import operators as O
from evopatch.llm_client import ContextBudgetExceeded
from evopatch.population import Individual
from evopatch.types import Candidate, ExecResult

GOOD = "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = a + b + {k}\n>>>>>>> REPLACE\n"
BAD_OUTSIDE = "pkg/mod.py\n<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n"


def patches(n_good, n_bad, start=1):
    out, k = [], start
    for _ in range(n_good):
        out.append(f"### PATCH {k}\n" + GOOD.format(k=k)); k += 1
    for _ in range(n_bad):
        out.append(f"### PATCH {k}\n" + BAD_OUTSIDE); k += 1
    return "\n".join(out)


def test_init_all_valid_single_call(make_ctx):
    ctx, llm = make_ctx([patches(10, 0)])
    pop = O.initial_population(ctx)
    assert len(pop) == 10 and all(i.valid for i in pop) and len(llm.calls) == 1
    c = llm.calls[0]
    assert c["stage"] == "init" and c["temperature"] == 0.7 and c["top_p"] == 1.0
    assert c["max_tokens"] == min(10 * 1500, 32768 - 100 - 512)
    assert c["extra_log"]["max_tokens_by_formula"] == 15000 and c["extra_log"]["max_tokens_capped_by_context"] is False
    assert [i.cid for i in pop] == [f"g0c{i}" for i in range(10)]


def test_init_shortfall_retry_then_fill(make_ctx):  # A4
    ctx, llm = make_ctx([patches(7, 3), patches(1, 2), patches(0, 2)])
    pop = O.initial_population(ctx)
    assert len(llm.calls) == 3
    assert [c["stage"] for c in llm.calls] == ["init", "retry", "retry"]
    assert all(c["retry_of"] == "init" and c["temperature"] == 0.7 for c in llm.calls[1:])
    assert llm.calls[1]["max_tokens"] == 3 * 1500 and llm.calls[2]["max_tokens"] == 2 * 1500
    assert len(pop) == 10 and sum(i.valid for i in pop) == 8
    assert all(not i.valid for i in pop[8:])


def test_init_shortfall_success_on_retry(make_ctx):
    ctx, llm = make_ctx([patches(9, 1), patches(1, 0)])
    pop = O.initial_population(ctx)
    assert len(llm.calls) == 2 and all(i.valid for i in pop)


def test_init_budget_exceeded(make_ctx):
    ctx, _ = make_ctx([patches(10, 0)], prompt_tokens=32500)
    with pytest.raises(ContextBudgetExceeded):
        O.initial_population(ctx)


def _parent(cid, valid=True, rp=False):
    ex = ExecResult(repro_pass=rp, reg_passed=1, reg_total=2, timed_out=False, output_path="") if valid else None
    view = {"repro_pass": rp, "repro_statuses": {"t": "FAILED"}, "reg_passed": 1, "reg_total": 2,
            "n_failed_regressions": 1, "no_regression_signal": False, "timed_out": False} if valid else None
    return Individual(cid=cid, gen_born=0, idx=0, op="init", parent_ids=(),
                      candidate=Candidate(edits=(), patch_hash="h", diff="diff --git a/x b/x\n" if valid else ""),
                      valid=valid, raw="raw-edits", exec=ex, exec_view=view)


def test_crossover_retry_then_valid(make_ctx):
    ctx, llm = make_ctx([BAD_OUTSIDE, "garbage", GOOD.format(k=9)])
    child = O.crossover(ctx, _parent("g0c0"), _parent("g0c1"), gen=1, idx=5)
    assert child.valid and child.cid == "g1c5" and child.parent_ids == ("g0c0", "g0c1")
    assert [c["stage"] for c in llm.calls] == ["crossover", "retry", "retry"]
    assert all(c["temperature"] == 0.5 for c in llm.calls)  # B3: retry는 crossover 값
    assert all(c["retry_of"] == "crossover" for c in llm.calls[1:])
    assert "match_outside_location" in llm.calls[1]["messages"][1]["content"]
    assert len({c["seed"] for c in llm.calls}) == 3


def test_crossover_all_fail_invalid(make_ctx):
    ctx, llm = make_ctx([BAD_OUTSIDE] * 3)
    child = O.crossover(ctx, _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert not child.valid and child.invalid_reason == "match_outside_location" and len(llm.calls) == 3


def test_mutation_invalid_parent_marked_and_revert(make_ctx):  # Q4b, A5: 재요청 없이 되돌림
    ctx, llm = make_ctx([BAD_OUTSIDE] * 3)
    base_child = _parent("g1c0")
    out = O.mutate(ctx, base_child, _parent("g0c0"), _parent("g0c1", valid=False), gen=1, idx=0)
    assert out is base_child and out.meta["mutation_reverted"] and out.meta["mutation_attempts"] == 1
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Execution results: NONE" in prompt and "INVALID patch" in prompt
    assert [c["stage"] for c in llm.calls] == ["mutation"]
    assert llm.calls[0]["temperature"] == 0.7


def test_mutation_retries_only_if_configured(make_ctx, cfg):
    cfg._data["algorithm"]["mutation_max_retries"] = 2
    ctx, llm = make_ctx([BAD_OUTSIDE] * 3)
    out = O.mutate(ctx, _parent("g1c0"), _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert out.meta["mutation_attempts"] == 3
    assert [c["stage"] for c in llm.calls] == ["mutation", "retry", "retry"]
    assert all(c["retry_of"] == "mutation" and c["temperature"] == 0.7 for c in llm.calls[1:])


def test_mutation_valid(make_ctx):
    ctx, llm = make_ctx([GOOD.format(k=3)])
    out = O.mutate(ctx, _parent("g1c0"), _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert out.valid and out.op == "mutation" and out.cid == "g1c0"
    assert "Reproduction test: FAILED" in llm.calls[0]["messages"][1]["content"]


SNF = "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = 0\n>>>>>>> REPLACE\n" \
      "pkg/mod.py\n<<<<<<< SEARCH\n    def made_up(self):\n        return 1\n=======\n    pass\n>>>>>>> REPLACE\n"


def test_crossover_retry_notice_for_search_not_found(make_ctx):  # D-2-1
    ctx, llm = make_ctx([SNF, GOOD.format(k=1)])
    child = O.crossover(ctx, _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert child.valid and len(llm.calls) == 2
    retry = llm.calls[1]["messages"][1]["content"]
    assert "SEARCH block #2 in pkg/mod.py (first line: `def made_up(self):`)" in retry
    assert "does not match the code shown above anywhere" in retry
    assert "character for character" in retry and "Functions that are not shown" in retry
    assert "Previous answer:" in retry  # 기존 동작 유지
    assert llm.calls[1]["retry_of"] == "crossover" and llm.calls[1]["temperature"] == 0.5


def test_mutation_no_retry_by_default_reverts(make_ctx):  # A5
    ctx, llm = make_ctx([SNF])
    child = _parent("g1c0")
    out = O.mutate(ctx, child, _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert out is child and out.meta["mutation_reverted"] and out.meta["mutation_attempts"] == 1 and len(llm.calls) == 1


def test_mutation_retry_notice_same_code_path_when_enabled(make_ctx, cfg):
    cfg._data["algorithm"]["mutation_max_retries"] = 1
    ctx, llm = make_ctx([SNF, GOOD.format(k=2)])
    out = O.mutate(ctx, _parent("g1c0"), _parent("g0c0"), _parent("g0c1"), gen=1, idx=0)
    assert out.valid and "SEARCH block #2" in llm.calls[1]["messages"][1]["content"]


def test_init_retry_notice_lists_search_not_found_patches(make_ctx):
    bad = "### PATCH 2\n" + SNF
    ctx, llm = make_ctx([patches(1, 0) + "\n" + bad + "\n" + "\n".join(
        f"### PATCH {k}\n" + GOOD.format(k=k) for k in range(3, 11)), patches(1, 0)])
    pop = O.initial_population(ctx)
    assert sum(i.valid for i in pop) == 10 and len(llm.calls) == 2
    retry = llm.calls[1]["messages"][1]["content"]
    assert "- PATCH 2: SEARCH block #2 in pkg/mod.py (first line: `def made_up(self):`)" in retry
    assert "Previous answer:" not in retry  # init 재요청은 이전 답변을 붙이지 않는 기존 동작
    assert llm.calls[1]["max_tokens"] == 1 * 1500


def test_init_max_tokens_capped_by_context_logged(make_ctx):  # A
    ctx, llm = make_ctx([patches(10, 0)], prompt_tokens=20000)
    O.initial_population(ctx)
    c = llm.calls[0]
    assert c["max_tokens"] == 32768 - 20000 - 512
    assert c["extra_log"]["max_tokens_capped_by_context"] is True and c["extra_log"]["max_tokens_by_formula"] == 15000


def test_validity_log_records_normalized_match(make_ctx, tmp_path):  # D-3
    import json
    norm = "pkg/mod.py\n<<<<<<< SEARCH\nif x:\n    return x\n=======\nif x is not None:\n    return x\n>>>>>>> REPLACE\n"
    text = "\n".join([f"### PATCH 1\n{norm}"] + [f"### PATCH {k}\n" + GOOD.format(k=k) for k in range(2, 11)])
    ctx, llm = make_ctx([text])
    pop = O.initial_population(ctx)
    rows = [json.loads(l) for l in ctx.validity_log.path.read_text().splitlines() if '"init"' in l]
    rows = [r for r in rows if r["stage"] == "init"]
    assert rows[0]["normalized_match_edits"] == 1 and rows[0]["reindented_edits"] == 1 and rows[0]["valid"]
    assert all(r["normalized_match_edits"] == 0 for r in rows[1:])
    assert pop[0].meta["reindented_edits"] == 1


# ---------- 2026-10-12: init 재요청 안내 확대(모든 실패 사유) ----------
FAILS = {
    "match_outside_location": BAD_OUTSIDE,
    "file_not_in_locations": "pkg/other.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n",
    "empty_diff": "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = a + b\n>>>>>>> REPLACE\n",
    "syntax_error": "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = (a +\n>>>>>>> REPLACE\n",
    "no_edits": "no blocks here\n",
    "search_not_found": "pkg/mod.py\n<<<<<<< SEARCH\n    def made_up(self):\n=======\n    pass\n>>>>>>> REPLACE\n",
}
EXPECT = {
    "match_outside_location": ("SEARCH block #1 in pkg/mod.py (first line: `return 1`) matches code outside",
                               "Edit only inside the function ranges shown above"),
    "file_not_in_locations": ("it edits `pkg/other.py`, which is not one of the files shown above",
                              "Edit only the files shown above"),
    "empty_diff": ("its REPLACE text leaves the code unchanged", "Each patch must actually change the code"),
    "syntax_error": ("the patched file does not compile (pkg/mod.py:", "valid Python with correct indentation"),
    "no_edits": ("no SEARCH/REPLACE block could be read", "one or more SEARCH/REPLACE blocks"),
    "search_not_found": ("SEARCH block #1 in pkg/mod.py (first line: `def made_up(self):`) does not match",
                         "copied character for character"),
}


@pytest.mark.parametrize("reason", sorted(FAILS))
def test_init_retry_notice_for_each_patch_reason(make_ctx, reason):
    first = "\n".join([f"### PATCH 1\n" + FAILS[reason]] + [f"### PATCH {k}\n" + GOOD.format(k=k) for k in range(2, 11)])
    ctx, llm = make_ctx([first, patches(1, 0)])
    pop = O.initial_population(ctx)
    assert len(llm.calls) == 2 and sum(i.valid for i in pop) == 10
    retry = llm.calls[1]["messages"][1]["content"]
    line, rule = EXPECT[reason]
    assert f"- PATCH 1: {line}" in retry and rule in retry
    assert "Previous answer:" not in retry  # 이전 답변은 붙이지 않음(기존 동작)


@pytest.mark.parametrize("finish,reason_text,rule", [
    ("length", "was cut off at the length limit; only 7 of 10", "Keep each patch short"),
    ("stop", "Only 7 of 10 requested patches could be read", "`### PATCH <number>` header")])
def test_init_retry_notice_for_missing_patches(cfg, originals, locations, tmp_path, finish, reason_text, rule):
    from evopatch.llm_client import LLMResult
    from evopatch.logs import JsonlLog
    from .conftest import FakeLLM

    class L(FakeLLM):
        def chat(self, messages, **kw):
            r = super().chat(messages, **kw)
            return LLMResult(r.text, 10, 5, finish if len(self.calls) == 1 else "stop", 0.0, {})

    llm = L([patches(7, 0), patches(3, 0, start=1)])
    ctx = O.OpContext(cfg=cfg, llm=llm, instance_id="x__y-1", run_index=0, problem_statement="bug",
                      locations=locations, originals=originals, validity_log=JsonlLog(tmp_path / "v.jsonl"))
    pop = O.initial_population(ctx)
    assert len(llm.calls) == 2 and sum(i.valid for i in pop) == 10
    retry = llm.calls[1]["messages"][1]["content"]
    assert reason_text in retry and rule in retry


def test_every_init_retry_has_notice(make_ctx):
    """재요청(attempt>0)에는 항상 안내가 붙는다(여러 사유가 섞인 경우 포함, 규칙은 사유당 한 번)."""
    first = "\n".join([f"### PATCH 1\n" + FAILS["empty_diff"], f"### PATCH 2\n" + FAILS["empty_diff"],
                       f"### PATCH 3\n" + FAILS["match_outside_location"]])
    ctx, llm = make_ctx([first, patches(0, 2), patches(0, 1)])
    O.initial_population(ctx)
    assert len(llm.calls) == 3
    for c in llm.calls[1:]:
        assert "did not give enough valid patches" in c["messages"][1]["content"]
    r1 = llm.calls[1]["messages"][1]["content"]
    assert r1.count("Each patch must actually change the code") == 1 and "Only 3 of 10 requested patches" in r1
    assert "- PATCH 1: its REPLACE" in r1 and "- PATCH 2: its REPLACE" in r1 and "- PATCH 3: SEARCH block #1" in r1


def test_retry_notice_has_no_test_or_gold_info():
    from evopatch import prompts
    fs = [{"patch_no": i + 1, "reason": r, "block": 1, "first_line": "x", "file": "pkg/mod.py",
           "detail": "pkg/mod.py: invalid syntax (pkg/mod.py, line 3)"} for i, r in enumerate([r for r in prompts.RETRY_REASONS if r != "search_not_unique"])]
    t = prompts.init_retry_notice(fs, {"reason": "truncated", "parsed": 1, "requested": 3})
    for bad in ("FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", "hints", "regression", "Reproduction"):
        assert bad not in t
