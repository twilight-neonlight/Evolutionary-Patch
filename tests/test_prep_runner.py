import json
import pytest

import evopatch.runner as R
from evopatch.config import Config
from evopatch.data_loader import InstanceView
from evopatch.logs import JsonlLog
from evopatch.prep.simple import parse_response


def test_simple_parse_response():
    text = ("### LOCATIONS\n`a/b.py::f`\nx/y.py::K.m\nnot an id\n### TEST\n```python\ndef test_x():\n    assert 1\n```\n")
    locs, test = parse_response(text)
    assert locs[:2] == ["`a/b.py::f`", "x/y.py::K.m"] and "def test_x" in test
    assert parse_response("### LOCATIONS\na.py::f\n")[1] is None


class StubLLM:
    run_id = "t"
    model = "m"

    def __init__(self, tmp):
        self.call_log = JsonlLog(tmp / "llm_calls.jsonl")


def test_issue_exception_recorded_as_error(tmp_path, monkeypatch, cfg):  # R2
    class Boom:
        meta = {}

        def prepare(self, view):
            raise RuntimeError("boom")

    monkeypatch.setattr(R, "make_provider", lambda name, deps: Boom())
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                           exec_specs={}, regression_pool={})
    v = InstanceView("x__y-1", "a/b", "c", "p", "1", "e")
    row = runner.run_issue(v)
    assert row["status"] == "error" and "boom" in row["error"]
    assert (tmp_path / row["traceback_path"]).exists()
    assert json.loads((tmp_path / "issues.jsonl").read_text().splitlines()[-1])["status"] == "error"
    # 디스크 여유 기록 (R3)
    sysrows = [json.loads(l) for l in (tmp_path / "system.jsonl").read_text().splitlines()]
    assert any("docker_free_gb" in r for r in sysrows)


def test_exec_cache_reuses_same_patch_hash(tmp_path, monkeypatch, cfg):
    """같은 patch_hash는 한 번만 실행하고 결과를 재사용한다. 알고리즘(개체 수, 세대 진행)은 그대로."""
    from evopatch.population import Individual
    from evopatch.types import Candidate, ExecResult, Location, PrepOutput, ReproTest

    def mk(cid, gen, idx, h):
        return Individual(cid=cid, gen_born=gen, idx=idx, op="init", parent_ids=(), valid=True,
                          candidate=Candidate(edits=(), patch_hash=h, diff=f"diff {h}\n"))

    monkeypatch.setattr(R, "initial_population", lambda ctx: [mk(f"g0c{i}", 0, i, f"h{i % 3}") for i in range(10)])
    monkeypatch.setattr(R, "crossover", lambda ctx, a, b, gen, idx: mk(f"g{gen}c{idx}", gen, idx, "h1"))
    monkeypatch.setattr(R, "mutate", lambda ctx, child, a, b, gen, idx: child)

    class Ex:
        def __init__(self):
            self.ran = []

        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def run(self, cand, prep, out):
            self.ran.append(cand.patch_hash)
            out.mkdir(parents=True, exist_ok=True)
            return ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path=str(out))

    ex = Ex()
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=ex,
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "x")],
                      ReproTest("t.py", "", "t.py"), ["t1", "t2"], ["f"], [], "simple")
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    assert sorted(ex.ran) == ["h0", "h1", "h2"]  # 5세대 동안 고유 해시 3개만 실행
    assert res.generations_run == 5
    rows = [json.loads(l) for l in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 50 and sum(1 for r in rows if r["cache_hit"]) > 0


def test_two_identical_patches_executed_once_and_fitness_reused(tmp_path, monkeypatch, cfg):
    """B-3: 같은 패치 두 개 → 실행 1회, 두 번째는 (r, g)를 재사용하고 cache_hit=True."""
    from evopatch.population import Individual
    from evopatch.types import Candidate, ExecResult, Location, PrepOutput, ReproTest

    def mk(cid, idx, h):
        return Individual(cid=cid, gen_born=0, idx=idx, op="init", parent_ids=(), valid=True,
                          candidate=Candidate(edits=(), patch_hash=h, diff=f"diff {h}\n"))

    cfg._data["algorithm"]["K"] = 1  # 세대 0만 (진화 없음 조건)
    cfg._data["algorithm"]["N"] = 2
    monkeypatch.setattr(R, "initial_population", lambda ctx: [mk("g0c0", 0, "same"), mk("g0c1", 1, "same")])

    class Ex:
        n = 0

        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def run(self, cand, prep, out):
            Ex.n += 1
            out.mkdir(parents=True, exist_ok=True)
            return ExecResult(repro_pass=True, reg_passed=3, reg_total=4, timed_out=False, output_path=str(out))

    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "x")],
                      ReproTest("t.py", "", "t.py"), ["a", "b", "c", "d"], ["f"], [], "simple")
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    assert Ex.n == 1
    a, b = res.history[0]
    assert (a.r, a.g, a.g_total) == (b.r, b.g, b.g_total) == (1, 3, 4)
    rows = [json.loads(l) for l in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert sorted(r["cache_hit"] for r in rows) == [False, True]
    assert res.executed_by_gen == [1] and res.cache_hits_by_gen == [1] and res.valid_by_gen == [2]


def test_prep_location_failure_reasons():
    from evopatch.prep.simple import SimplePrep
    src = "def _format_float(value):\n    return 1\n\nclass Card:\n    def __init__(self):\n        pass\n\nclass K:\n    pass\n"
    fr = SimplePrep._failure_reason
    assert fr(None, "a.py", "f", True)["reason"] == "file_not_found"
    r = fr(src, "a.py", "Card._format_float", True)
    assert r["reason"] == "qualname_mismatch" and r["found_as"] == ["_format_float (module_function)"] and not r["ambiguous"]
    assert fr(src, "a.py", "Card.nope", True)["reason"] == "symbol_not_in_file"
    assert fr(src, "a.py", "K", False)["reason"] == "class_not_allowed"
    assert fr("def (", "a.py", "f", True)["reason"] == "parse_error"


def test_runner_gen0_all_invalid_stops_without_exec(tmp_path, monkeypatch, cfg):  # D-2-2
    from evopatch.population import Individual
    from evopatch.types import Candidate, Location, PrepOutput, ReproTest
    monkeypatch.setattr(R, "initial_population", lambda ctx: [
        Individual(cid=f"g0c{i}", gen_born=0, idx=i, op="init", parent_ids=(), valid=False,
                   invalid_reason="search_not_found", candidate=Candidate(edits=(), patch_hash=f"h{i}"))
        for i in range(10)])
    monkeypatch.setattr(R, "crossover", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no crossover")))

    class Ex:
        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def run(self, *a):
            raise AssertionError("no exec")

    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "x")],
                      ReproTest("t.py", "", "t.py"), [], ["f"], [], "simple")
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    assert res.terminated_by == "all_invalid" and res.valid_by_gen == [0] and res.executed_by_gen == [0]
    assert len((tmp_path / "candidates.jsonl").read_text().splitlines()) == 10


def test_regression_names_never_reach_prompts(tmp_path, cfg):  # A2
    """가짜 실패 회귀 이름(LEAKCHECK_xyz)이 어떤 프롬프트에도 나타나지 않고 private/에만 저장된다."""
    from evopatch import prompts
    from evopatch.population import Individual
    from evopatch.types import Candidate, ExecResult, Location
    leak = "tests/test_mod.py::LEAKCHECK_xyz"
    out = tmp_path / "exec" / "g0c0"
    out.mkdir(parents=True)
    (out / "regression_status.json").write_text(json.dumps({leak: "FAILED", "tests/test_mod.py::ok": "PASSED"}))
    (out / "repro.log").write_text(">>>>> EVOPATCH START\nrepro output\n>>>>> EVOPATCH END\n")
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                           exec_specs={}, regression_pool={})
    ind = Individual(cid="g0c0", gen_born=0, idx=0, op="init", parent_ids=(), valid=True,
                     candidate=Candidate(edits=(), patch_hash="h", diff="diff --git a/x b/x\n"),
                     exec=ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path=str(out)))
    ind.exec_view = runner._exec_view(ind, [leak, "tests/test_mod.py::ok"], iid="x__y-1", attempt_id="a1")
    assert "LEAKCHECK" not in json.dumps(ind.exec_view) and ind.exec_view["n_failed_regressions"] == 1
    assert "LEAKCHECK_xyz" in (tmp_path / "private" / "x__y-1" / "a1" / "g0c0.json").read_text()
    locs = [Location("m.py", "f", 1, 1, "def f():\n    pass")]
    texts = [prompts.init_prompt("bug", locs, 10), prompts.crossover_prompt("bug", locs, ind.view(), ind.view()),
             prompts.mutation_prompt("bug", locs, ind.view(), ind.view()),
             prompts.retry_suffix("prev", "search_not_found", prompts.search_not_found_detail(1, "x", "m.py")),
             prompts.init_retry_notice([{"patch_no": 1, "reason": "search_not_found", "block": 1, "first_line": "x", "file": "m.py"}],
                                     {"reason": "truncated", "parsed": 1, "requested": 3})]
    for t in texts:
        assert "LEAKCHECK" not in t
    assert "1/2 passed, 1 failed" in texts[2]


def test_prompt_code_cannot_read_private_dir():  # A2-b
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "evopatch"
    for name in ("prompts.py", "operators.py"):
        text = (src / name).read_text()
        for bad in ("private", "open(", "read_text", "Path("):
            assert bad not in text, f"{name}: {bad}"


def test_runner_min_valid_zero_writes_gen0_report_without_llm_or_exec(tmp_path, monkeypatch, cfg):  # A6
    from evopatch.population import Individual
    from evopatch.types import Candidate, Location, PrepOutput, ReproTest
    events = []

    def init(ctx):
        ctx.gen0_info = {"requested": 10, "calls": [{"attempt": 0, "stage": "init", "requested": 10, "parsed": 10,
                         "valid": 0, "finish_reason": "stop", "max_tokens": 15000, "max_tokens_capped_by_context": False,
                         "prompt_tokens": 900, "completion_tokens": 100, "missing_patches": 0, "missing_reason": None}],
                         "failed_patches": [{"attempt": 0, "patch_no": i + 1, "reason": "no_edits"} for i in range(10)]}
        return [Individual(cid=f"g0c{i}", gen_born=0, idx=i, op="init", parent_ids=(), valid=False,
                           invalid_reason="no_edits", candidate=Candidate(edits=(), patch_hash=f"h{i}")) for i in range(10)]

    monkeypatch.setattr(R, "initial_population", init)
    monkeypatch.setattr(R, "crossover", lambda *a, **k: events.append("crossover"))
    orig = R.IssueRunner._write_gen0_report
    monkeypatch.setattr(R.IssueRunner, "_write_gen0_report", lambda self, *a: (events.append("report"), orig(self, *a))[1])

    class Ex:
        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def run(self, *a):
            events.append("exec")

    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "x")],
                      ReproTest("t.py", "", "t.py"), [], ["f"], [], "simple")
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    assert events == ["report"] and res.terminated_by == "all_invalid" and res.gen0_valid_count == 0
    rep = json.loads((tmp_path / res.gen0_report).read_text())
    assert rep["valid"] == 0 and rep["failure_reasons"] == {"no_edits": 10} and rep["calls"][0]["max_tokens"] == 15000
    assert {"prep_locations_requested", "prep_locations_found", "prep_location_kinds", "prompt_tokens_init"} <= set(rep)


def test_runner_min_valid_one_executes_once_and_submits(tmp_path, monkeypatch, cfg):
    from evopatch.population import Individual
    from evopatch.types import Candidate, ExecResult, Location, PrepOutput, ReproTest
    pops = [Individual(cid=f"g0c{i}", gen_born=0, idx=i, op="init", parent_ids=(), valid=(i == 0),
                       invalid_reason=None if i == 0 else "no_edits",
                       candidate=Candidate(edits=(), patch_hash=f"h{i}", diff="diff x\n" if i == 0 else "")) for i in range(10)]
    monkeypatch.setattr(R, "initial_population", lambda ctx: pops)
    monkeypatch.setattr(R, "crossover", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no crossover")))

    class Ex:
        n = 0

        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def run(self, cand, prep, out):
            Ex.n += 1
            out.mkdir(parents=True, exist_ok=True)
            return ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path=str(out))

    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "x")],
                      ReproTest("t.py", "", "t.py"), ["a", "b"], ["f"], [], "simple")
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    assert Ex.n == 1 and res.terminated_by == "insufficient_valid" and res.final.cid == "g0c0"
    assert res.gen0_valid_count == 1 and (tmp_path / res.gen0_report).exists()


def test_run_meta_records_hashes(tmp_path, cfg):  # B0
    from evopatch.cli import build_run_meta, ROOT
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n")
    meta = build_run_meta(ROOT / "tests" / "data" / "test_config.yaml", ids, cfg, fetch=lambda u: {"data": [{"id": "fake-model"}]})
    assert "cli.py" in meta["code_sha256"] and "prep/simple.py" in meta["code_sha256"]
    assert all(len(h) == 64 for h in meta["code_sha256"].values())
    import hashlib
    assert meta["ids_sha256"] == hashlib.sha256(b"a\nb\n").hexdigest()
    assert meta["served"] == {"models": ["fake-model"]} and meta["config_digest"] == cfg.digest()
    bad = build_run_meta(ROOT / "tests" / "data" / "test_config.yaml", None, cfg, fetch=lambda u: 1 / 0)
    assert "error" in bad["served"]  # 기록 실패가 실행을 막지 않음


def test_resume_allows_only_operational_changes(cfg):
    import copy
    from evopatch.cli import config_diff_operational
    from evopatch.config import Config
    c2 = Config(copy.deepcopy(cfg.as_dict()))
    c2._data["llm"]["base_url"] = "http://localhost:8999/v1"
    c2._data["unattended"]["infra_wait_max_s"] = 60
    same, changed = config_diff_operational(cfg, c2)
    assert same and set(changed) == {"llm.base_url", "unattended"}
    c3 = Config(copy.deepcopy(cfg.as_dict()))
    c3._data["algorithm"]["N"] = 20
    assert config_diff_operational(cfg, c3)[0] is False


def test_issue_timeout_status(tmp_path, monkeypatch, cfg):  # B6
    from evopatch.errors import IssueTimeout

    class Slow:
        meta = {}

        def prepare(self, view):
            raise IssueTimeout("이슈 wall-clock 상한 초과")

    monkeypatch.setattr(R, "make_provider", lambda name, deps: Slow())
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                           exec_specs={}, regression_pool={})
    row = runner.run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))
    assert row["status"] == "issue_timeout" and row["status"] in R.DONE_STATUSES


def test_infra_and_undecided_status(tmp_path, monkeypatch, cfg):  # B4, B5
    from evopatch.config import UndecidedConfigError
    from evopatch.llm_client import LLMConnectionError
    for exc, status in ((LLMConnectionError("down"), "infra_error"), (UndecidedConfigError("x"), "skipped_undecided")):
        class P:
            meta = {}

            def prepare(self, view, exc=exc):
                raise exc

        monkeypatch.setattr(R, "make_provider", lambda name, deps: P())
        runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                               exec_specs={}, regression_pool={})
        row = runner.run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))
        assert row["status"] == status
    assert "infra_error" not in R.DONE_STATUSES and "skipped_undecided" in R.DONE_STATUSES


def test_deadline_check_in_operators(make_ctx):
    import time
    from evopatch.errors import IssueTimeout
    ctx, llm = make_ctx(["x"])
    ctx.deadline = time.monotonic() - 1
    with pytest.raises(IssueTimeout):
        O_init(ctx)
    assert llm.calls == []


def O_init(ctx):
    from evopatch import operators
    return operators.initial_population(ctx)


def _gen0_sf_runner(tmp_path, monkeypatch, cfg, calls):
    """search_not_found 패치(프롬프트에 없던 def 이름) 10개로 세대 0 보고서를 쓰게 하는 준비."""
    from evopatch.population import Individual
    from evopatch.types import Candidate, Location, PrepOutput, ReproTest

    def init(ctx):
        ctx.gen0_info = {"requested": 10, "calls": [], "failed_patches": [
            {"attempt": 0, "patch_no": i + 1, "reason": "search_not_found", "failed_search": "    def hidden(self):\n"}
            for i in range(10)]}
        return [Individual(cid=f"g0c{i}", gen_born=0, idx=i, op="init", parent_ids=(), valid=False,
                           invalid_reason="search_not_found", candidate=Candidate(edits=(), patch_hash=f"h{i}"))
                for i in range(10)]

    monkeypatch.setattr(R, "initial_population", init)

    class Ex:
        def read_files(self, iid, files, diff=""):
            return {f: "x\n" for f in files}

        def find_defs(self, iid, names):
            calls.append(list(names))
            return {"hidden"}

    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    prep = PrepOutput("x__y-1", "a/b", "c", "p", [Location("m.py", "f", 1, 1, "def f():\n    pass")],
                      ReproTest("t.py", "", "t.py"), [], ["f"], [], "simple")
    return runner, prep


def test_gen0_report_find_defs_off_by_default(tmp_path, monkeypatch, cfg):  # D-6
    calls = []
    assert cfg.get("diagnostics.gen0_find_defs") is False
    runner, prep = _gen0_sf_runner(tmp_path, monkeypatch, cfg, calls)
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    rep = json.loads((tmp_path / res.gen0_report).read_text())
    assert calls == [] and rep["find_defs_checked"] is False
    assert rep["search_not_found_identifier_class"] == {"not_shown_unchecked": 10}
    # 키가 없는 이전 설정에서도 off
    del cfg._data["diagnostics"]
    calls2 = []
    runner, prep = _gen0_sf_runner(tmp_path / "b", monkeypatch, cfg, calls2)
    runner._evolve(prep, None, tmp_path / "b" / "exec", {"attempt_id": "a2"})
    assert calls2 == []


def test_gen0_report_find_defs_on(tmp_path, monkeypatch, cfg):  # D-6
    calls = []
    cfg._data["diagnostics"]["gen0_find_defs"] = True
    runner, prep = _gen0_sf_runner(tmp_path, monkeypatch, cfg, calls)
    res = runner._evolve(prep, None, tmp_path / "exec", {"attempt_id": "a1"})
    rep = json.loads((tmp_path / res.gen0_report).read_text())
    assert calls == [["hidden"]] and rep["search_not_found_identifier_class"] == {"in_repo_not_shown": 10}


def test_diagnostics_is_operational_for_resume(cfg):  # D-6
    import copy
    from evopatch.cli import config_diff_operational
    from evopatch.config import Config
    c2 = Config(copy.deepcopy(cfg.as_dict()))
    c2._data["diagnostics"]["gen0_find_defs"] = True
    assert config_diff_operational(cfg, c2) == (True, ["diagnostics"])


def test_incremental_tokens_match_full_reread(tmp_path, cfg):  # E-2
    llm = StubLLM(tmp_path)
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=llm, executor=None,
                           exec_specs={}, regression_pool={})
    rows = [{"attempt_id": a, "stage": s, "prompt_tokens": p, "completion_tokens": c}
            for a, s, p, c in [("a1", "prep", 10, 5), ("a2", "init", 7, None), ("a1", "init", 3, 4),
                               ("a1", "retry", None, None), (None, "prep", 99, 99), ("a2", "init", 1, 1)]]
    for r in rows:
        llm.call_log.append(r)
    for aid in ("a1", "a2"):
        assert runner._tokens(aid) == R.IssueRunner.tokens_from_rows(llm.call_log.read(), aid)
    assert runner._tokens("a1")["n_calls"] == 0  # 한 번 꺼내면 비워진다(이슈 끝에서 한 번만 호출)


def test_src_does_not_read_dev_registry_or_grading_outputs():
    """개발 이슈 등록부와 채점 결과는 생성 파이프라인(src, 채점 패키지 제외)이 읽지 않는다."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "evopatch"
    for p in src.rglob("*.py"):
        if "grading" in p.relative_to(src).parts or p.name == "cli.py":  # 채점 패키지와 후처리 호출부는 채점 결과를 쓰는 쪽
            continue
        t = p.read_text()
        for bad in ("dev_issue_registry", "grading_private", "grading_aggregate"):
            assert bad not in t, f"{p.name}: {bad}"
