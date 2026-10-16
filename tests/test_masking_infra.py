"""B2·B3: 프롬프트의 test_files 마스킹. A1·A2·A4: 디스크 가드, 이미지 없음, pull 간격."""
import json
import re

import pytest

import evopatch.runner as R
from evopatch import operators as O
from evopatch.data_loader import InstanceView
from evopatch.errors import DiskGuardStop, ImageUnavailable, InfraError
from evopatch.executor import Executor, image_unavailable_cause
from evopatch.logs import JsonlLog
from evopatch.operators import OpContext
from evopatch.population import Individual
from evopatch.redact import TestFileMasker, mask_terms, mask_token

MASK = mask_token()
from evopatch.specs import ExecSpec
from evopatch.types import Candidate, ExecResult

from .conftest import FakeLLM

TEST_FILES = ["tests/test_widget.py", "pkg/tests/test_core.py", "tests/admin_views/tests.py"]
# 가려야 하는 모든 형태: 전체 경로, 점 표기 모듈(Django 라벨형 포함), 단독 파일명
LEAK_FORMS = ["tests/test_widget.py", "pkg/tests/test_core.py", "tests/admin_views/tests.py",
              "pkg.tests.test_core", "tests.admin_views.tests", "admin_views.tests", "tests.test_widget"]
LEAK_BASENAMES = ["test_widget.py", "test_core.py"]


def leaked(text: str, basenames: bool = True) -> list[str]:
    found = [f for f in LEAK_FORMS if f in text]
    if basenames:
        found += [b for b in LEAK_BASENAMES if re.search(r"(?<![\w./-])" + re.escape(b), text)]
    return found


def test_mask_terms_forms():
    m = TestFileMasker(TEST_FILES)
    t = m("File \"/testbed/pkg/tests/test_core.py\", line 3\nERROR: test_x (admin_views.tests.T)\n"
          "see test_widget.py and pkg/tests/test_core.py::test_a; tests.admin_views.tests ok")
    assert leaked(t) == [] and t.count(MASK) == 5
    # 다른 소스 파일의 경로 일부인 파일명은 가리지 않는다(파일명만 같은 다른 파일)
    assert m("sympy/utilities/test_core.py") == "sympy/utilities/test_core.py"
    assert m("pkg/tests/test_core_extra.py") == "pkg/tests/test_core_extra.py"
    assert {k for k, _ in mask_terms(TEST_FILES)} == {"path", "module", "basename"}


def test_mask_protected_spans_kept():
    m = TestFileMasker(TEST_FILES)
    issue = "Running tests/test_widget.py fails"
    assert m(f"<issue>{issue}</issue> and tests/test_widget.py", [issue]) == f"<issue>{issue}</issue> and {MASK}"


LEAKY_LOG = ("Traceback (most recent call last):\n  File \"/testbed/pkg/tests/test_core.py\", line 9, in helper\n"
             "ImportError from tests.admin_views.tests; also test_widget.py")


def _leaky_parent(cid, valid=True):
    view = {"repro_pass": False, "repro_statuses": {"pkg/tests/test_core.py::test_a": "FAILED",
                                                    "test_x (admin_views.tests.T)": "FAILED"},
            "repro_log_tail": LEAKY_LOG, "reg_passed": 1, "reg_total": 2, "n_failed_regressions": 1,
            "no_regression_signal": False, "timed_out": False, "error": "see tests/test_widget.py"} if valid else None
    ex = ExecResult(repro_pass=False, reg_passed=1, reg_total=2, timed_out=False, output_path="") if valid else None
    raw = "tests/test_widget.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE"
    return Individual(cid=cid, gen_born=0, idx=0, op="init", parent_ids=(),
                      candidate=Candidate(edits=(), patch_hash=cid,
                                          diff="diff --git a/pkg/mod.py b/pkg/mod.py\n# test_core.py\n" if valid else ""),
                      valid=valid, raw=raw, exec=ex, exec_view=view)


# 테스트 파일을 편집하려는 무효 패치(file_not_in_locations), 테스트 파일 경로가 든 search_not_found
TEST_FILE_EDIT = "tests/test_widget.py\n<<<<<<< SEARCH\n    x = 1\n=======\n    x = 2\n>>>>>>> REPLACE\n"
SNF_TEST = ("pkg/mod.py\n<<<<<<< SEARCH\n    import pkg.tests.test_core  # test_core.py\n=======\n    pass\n"
            ">>>>>>> REPLACE\n")
GOOD = "pkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = a + b + 1\n>>>>>>> REPLACE\n"


def test_all_generated_prompts_hide_test_files(cfg, originals, locations, tmp_path):  # B3
    """init(+부족분 재요청 안내) / crossover(+재요청) / mutation의 모든 프롬프트에 test_files 경로·모듈명이 없다.
    단독 파일명은 패치 텍스트(부모 diff·원문, 이전 답변, 실패 SEARCH 첫 줄) 밖에서만 가린다(결정 2)."""
    init1 = "\n".join(f"### PATCH {k}\n" + (GOOD.replace("+ 1", f"+ {k}") if k <= 7 else
                                             TEST_FILE_EDIT if k == 8 else SNF_TEST) for k in range(1, 11))
    init2 = "\n".join(f"### PATCH {k}\n" + SNF_TEST for k in range(1, 4))
    init3 = "\n".join(f"### PATCH {k}\n" + GOOD.replace("+ 1", f"+ {k + 20}") for k in range(1, 4))
    responses = [init1, init2, init3,
                 TEST_FILE_EDIT + "\nsee tests/admin_views/tests.py", SNF_TEST, GOOD,  # crossover 3회
                 TEST_FILE_EDIT]  # mutation 1회(무효 → 되돌림)
    llm = FakeLLM(responses)
    masker = TestFileMasker(TEST_FILES)
    ctx = OpContext(cfg=cfg, llm=llm, instance_id="x__y-1", run_index=0, problem_statement="widget renders twice",
                    locations=locations, originals=originals, validity_log=JsonlLog(tmp_path / "v.jsonl"),
                    mask=masker)
    pop = O.initial_population(ctx)
    assert sum(i.valid for i in pop) == 10
    child = O.crossover(ctx, _leaky_parent("g0c0"), _leaky_parent("g0c1", valid=False), gen=1, idx=0)
    assert child.valid
    O.mutate(ctx, child, _leaky_parent("g0c0"), _leaky_parent("g0c1", valid=False), gen=1, idx=0)
    stages = [c["stage"] for c in llm.calls]
    assert stages == ["init", "retry", "retry", "crossover", "retry", "retry", "mutation"]
    for c in llm.calls:
        text = "\n".join(m["content"] for m in c["messages"])
        assert leaked(text, basenames=False) == [], (c["stage"], leaked(text))
    assert sum(c["extra_log"]["prompt_mask_hits"] for c in llm.calls) > 0
    # init 재요청 안내의 실패 SEARCH 첫 줄: 모듈은 가리고, 패치 텍스트 안의 단독 파일명은 둔다
    assert f"import {MASK}  # test_core.py" in llm.calls[1]["messages"][1]["content"]
    assert MASK in llm.calls[4]["messages"][1]["content"]  # crossover 재요청의 이전 답변(경로)
    cx = llm.calls[3]["messages"][1]["content"]
    assert "# test_core.py" in cx  # 부모 diff 안의 문자열은 그대로
    mut = llm.calls[6]["messages"][1]["content"]
    tail = mut.split("output (tail):", 1)[1].split("Regression tests:", 1)[0]
    assert f"also {MASK}" in tail and leaked(tail) == []  # 실행 출력(패치 텍스트 아님)의 단독 파일명은 가림
    assert masker.hits["path"] and masker.hits["module"] and masker.hits["basename"]


def test_masker_basename_exempt_spans():  # 결정 2
    m = TestFileMasker(TEST_FILES)
    patch = "-    path = 'test_core.py'\n+    see pkg/tests/test_core.py\n"
    out = m("diff:\n" + patch + "log: test_core.py failed", (), [patch])
    assert f"-    path = 'test_core.py'\n+    see {MASK}\n" in out and out.endswith(f"log: {MASK} failed")


def test_locations_and_issue_text_are_not_masked(cfg, originals, locations, tmp_path):
    """보호 구간: 위치 코드 원문은 SEARCH 대조 기준이라 바꾸지 않는다. 이슈 본문은 허용 입력이라 그대로 둔다."""
    llm = FakeLLM(["\n".join(f"### PATCH {k}\n" + GOOD.replace("+ 1", f"+ {k}") for k in range(1, 11))])
    ctx = OpContext(cfg=cfg, llm=llm, instance_id="x__y-1", run_index=0,
                    problem_statement="see tests/test_widget.py", locations=locations, originals=originals,
                    validity_log=JsonlLog(tmp_path / "v.jsonl"), mask=TestFileMasker(["pkg/mod.py", *TEST_FILES]))
    O.initial_population(ctx)
    text = llm.calls[0]["messages"][1]["content"]
    assert "see tests/test_widget.py" in text and "### pkg/mod.py :: add" in text
    assert locations[0].code in text


def test_prep_prompt_hides_test_files(cfg):  # B3 (prep)
    from evopatch.prep.base import PrepDeps
    from evopatch.prep.simple import SimplePrep

    class Ex:
        def list_files(self, iid):
            return ["pkg/mod.py", "pkg/tests/test_core.py", "tests/test_widget.py", "pkg/tests_core_helpers.py"]

        def read_files(self, iid, files, diff=""):
            return {f: "def add(a, b):\n    return a + b\n" for f in files}

    llm = FakeLLM(["### FILES\npkg/mod.py\n",  # 2026-10-12: prep 1단계(후보 파일) 응답
                   "### LOCATIONS\npkg/mod.py::add\n### TEST\n```python\ndef test_x():\n    assert 1\n```\n"])
    llm.model = "m"
    spec = ExecSpec("x__y-1", "a/b", "1", "img", "pytest", (), "parse_log_pytest", tuple(TEST_FILES))
    cfg._data["prep"]["simple"]["repro_paths"]["a/b"] = "tests/test_swe_repro.py"
    deps = PrepDeps(cfg=cfg, llm=llm, executor=Ex(), exec_specs={"x__y-1": spec}, regression_pool={}, run_index=0)
    prov = SimplePrep(deps)
    out = prov.prepare(InstanceView("x__y-1", "a/b", "c", "widget renders twice", "1", "e"))
    text = llm.calls[0]["messages"][0]["content"]
    assert leaked(text) == [] and "pkg/mod.py" in text
    code = llm.calls[1]["messages"][0]["content"]  # 2단계: 후보 파일 코드가 들어간 프롬프트
    assert leaked(code) == [] and "def add(a, b):" in code
    assert out.test_files == TEST_FILES  # 실행기에는 그대로 전달


# ---------- A2: 이미지 없음 ----------
def test_image_unavailable_cause_classification():
    assert image_unavailable_cause("Error response from daemon: pull access denied for swebench/x, repository "
                                   "does not exist or may require 'docker login'") == "repository_not_found"
    assert image_unavailable_cause("manifest for swebench/x:latest not found: manifest unknown") == "manifest_unknown"
    assert image_unavailable_cause('failed to resolve reference "docker.io/a/b:latest": '
                                   "docker.io/a/b:latest: not found") == "not_found"
    assert image_unavailable_cause("toomanyrequests: You have reached your pull rate limit") is None
    assert image_unavailable_cause("Cannot connect to the Docker daemon") is None


class _StubLLM:
    run_id, model = "t", "m"

    def __init__(self, tmp):
        self.call_log = JsonlLog(tmp / "llm_calls.jsonl")


def test_runner_image_unavailable_status(tmp_path, monkeypatch, cfg):
    class P:
        meta = {}

        def prepare(self, view):
            raise ImageUnavailable("manifest_unknown", "manifest unknown")

    monkeypatch.setattr(R, "make_provider", lambda name, deps: P())

    class Ex:
        run_label = "lbl"

    monkeypatch.setattr(R, "count_containers", lambda *a: 0)
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=_StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    row = runner.run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))
    assert row["status"] == "image_unavailable" and row["image_unavailable_cause"] == "manifest_unknown"
    assert row["final_patch_empty"] and (tmp_path / row["final_patch_path"]).read_text() == ""
    assert "image_unavailable" in R.DONE_STATUSES


def test_runner_disk_guard_raises_stop(tmp_path, monkeypatch, cfg):  # A1
    cfg._data["unattended"]["disk_guard_gb"] = 120
    monkeypatch.setattr(R, "disk_free_row", lambda d: {k + "_free_gb": 50.0 for k in d})

    class Ex:
        run_label = "lbl"

    monkeypatch.setattr(R, "count_containers", lambda *a: 0)
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=_StubLLM(tmp_path), executor=Ex(),
                           exec_specs={}, regression_pool={})
    row = runner.run_issue(InstanceView("x__y-1", "a/b", "c", "p", "1", "e"))
    assert row["status"] == "infra_error" and row["error"].startswith(DiskGuardStop.__name__)
    assert issubclass(DiskGuardStop, InfraError)


def test_configs_disk_guard_fixed_120():
    from pathlib import Path
    from evopatch.config import Config
    root = Path(__file__).resolve().parents[1]
    for p in (root / "tests" / "data" / "test_config.yaml",):
        c = Config.load(p)
        assert c.get("unattended.disk_guard_gb") == 120
        assert c.get("unattended.image_unavailable_escalate_after") == 5
        assert c.get("unattended.pull_min_interval_s") == 0


# ---------- A4: pull 간격 ----------
def _ex(interval):
    return Executor(specs={}, repro_timeout_s=1, regression_timeout_s=1, setup_timeout_s=1,
                    max_parallel_containers=1, regression_cap=None, run_label="t", pull_min_interval_s=interval)


def test_pull_interval_default_off(monkeypatch):
    import evopatch.executor as E
    calls = []
    monkeypatch.setattr(E.subprocess, "run", lambda *a, **k: calls.append(a) or None)
    _ex(0).ensure_image("img")
    assert calls == []  # 기본값 0: 추가 docker 호출 없음(기존 동작)


def test_pull_interval_spacing_and_404(monkeypatch):
    import evopatch.executor as E

    class P:
        def __init__(self, rc, err=""):
            self.returncode, self.stderr, self.stdout = rc, err, ""

    calls, sleeps = [], []

    def run(cmd, **k):
        calls.append(cmd[:3])
        if cmd[1] == "image":
            return P(1)  # 로컬에 없음
        return P(1, "manifest unknown") if cmd[-1] == "missing" else P(0)

    monkeypatch.setattr(E.subprocess, "run", run)
    monkeypatch.setattr(E.time, "sleep", lambda s: sleeps.append(s))
    ex = _ex(30)
    ex.ensure_image("a")
    ex.ensure_image("b")
    assert [c[1] for c in calls] == ["image", "pull", "image", "pull"] and len(sleeps) == 1 and 0 < sleeps[0] <= 30
    with pytest.raises(ImageUnavailable) as e:
        ex.ensure_image("missing")
    assert e.value.cause == "manifest_unknown"


# ---------- A2: cli 승격 ----------
def test_cli_escalates_five_consecutive_image_unavailable(tmp_path, monkeypatch, cfg):
    import time
    import yaml
    import evopatch.cli as C

    class FakeRunner:
        def __init__(self, **kw):
            self.run_dir = kw["run_dir"]
            self.issue_log = CountingLog(self.run_dir / "issues.jsonl")

        def run_issue(self, view):
            iid = view.instance_id
            bad = iid.startswith("m")
            r = {"instance_id": iid, "status": "image_unavailable" if bad else "ok", "attempt_id": "a" + iid,
                 "terminated_by": None, "generations_run": None, "total_prompt_tokens": 0,
                 "total_completion_tokens": 0, "wall_clock_s": 0.1, "final_patch_path": None,
                 "image_unavailable_cause": "manifest_unknown" if bad else None,
                 "error": "manifest unknown" if bad else None}
            self.issue_log.append(r)
            return r

    class CountingLog(JsonlLog):  # 승격 행도 이슈 실행기의 로그 객체(같은 잠금)로 쓰는지 확인
        escalated = 0

        def append(self, row):
            CountingLog.escalated += row.get("escalated_from") == "image_unavailable"
            super().append(row)

    class L:
        @classmethod
        def from_config(cls, *a, **k):
            return cls()

        def close(self):
            pass

    ids = ["ok1", "m1", "m2", "ok2", "m3", "m4", "m5", "m6", "m7", "ok3"]
    cfg._data["run"].update(issue_concurrency=1, runs_dir="runs")
    cfg._data["unattended"].update(disk_guard_gb=None, postprocess=False, infra_wait_max_s=0, status_interval_s=3600)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg._data))
    (tmp_path / "ids.txt").write_text("\n".join(ids))
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "IssueRunner", FakeRunner)
    monkeypatch.setattr(C, "specs_from_config", lambda cfg, root: {
        i: ExecSpec(i, "a/b", "1", "img-" + i, "", (), "p", ()) for i in ids})
    monkeypatch.setattr(C, "load_regression_pool", lambda p: {})
    monkeypatch.setattr(C, "load_instances", lambda p, todo: {i: InstanceView(i, "a/b", "c", "p", "1", "e") for i in todo})
    monkeypatch.setattr(C, "cleanup_label", lambda *a: 0)
    monkeypatch.setattr(C, "build_run_meta", lambda *a, **k: {})
    monkeypatch.setattr(C, "check_served_model", lambda cfg: {"models": ["m"]})
    probes = []
    monkeypatch.setattr(C, "infra_healthy", lambda cfg, guard, probe=None: probes.append(probe) or (False, "registry"))
    monkeypatch.setattr(C, "LLMClient", L)
    monkeypatch.setattr(C.Executor, "from_config", classmethod(lambda cls, *a: None))
    code = C.main(["--config", str(cfg_path), "--ids-file", str(tmp_path / "ids.txt"), "--run-id", "x"])
    assert code == C.EXIT_INFRA_STOP
    rd = tmp_path / "runs" / f"{time.strftime('%Y%m%d')}_x"
    latest = {}
    for r in JsonlLog(rd / "issues.jsonl").read():
        latest[r["instance_id"]] = r
    # m1, m2 다음 ok2가 연속을 끊었으므로 m3..m7 다섯 건이 승격된다. ok3는 시작하지 않는다.
    assert [i for i in ids if latest.get(i, {}).get("status") == "infra_error"] == ["m3", "m4", "m5", "m6", "m7"]
    assert latest["m1"]["status"] == latest["m2"]["status"] == "image_unavailable"
    assert latest["m7"]["escalated_from"] == "image_unavailable" and "ok3" not in latest
    assert CountingLog.escalated == 5
    assert probes and probes[0] == "img-m7"  # 승격 후 대기에서 레지스트리 확인
    ev = [json.loads(l) for l in (rd / "system.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "image_unavailable_escalated" and e["n"] == 5 for e in ev)
    assert "image_unavailable_issues: 2" in (rd / "status.txt").read_text()


def test_empty_regression_list_prompt_says_not_available():  # 결정 4
    from evopatch import prompts
    base = {"repro_pass": False, "repro_statuses": {}, "reg_passed": 0, "reg_total": 0, "no_regression_signal": False}
    assert "Regression tests: not available for this issue." in prompts._exec_summary({"valid": True, "exec": base})
    assert "0/0" not in prompts._exec_summary({"valid": True, "exec": base})
    assert "3/4 passed, 1 failed" in prompts._exec_summary({"valid": True, "exec": {**base, "reg_passed": 3, "reg_total": 4}})


def test_check_served_model():
    import evopatch.cli as C
    from evopatch.config import Config
    from pathlib import Path
    cfg = Config.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "test_config.yaml")
    assert C.check_served_model(cfg, fetch=lambda u: {"data": [{"id": "qwen3-coder"}]})["models"] == ["qwen3-coder"]
    with pytest.raises(SystemExit, match="llm.model"):
        C.check_served_model(cfg, fetch=lambda u: {"data": [{"id": "devstral"}]})
    with pytest.raises(SystemExit, match="확인할 수 없어"):
        C.check_served_model(cfg, fetch=lambda u: (_ for _ in ()).throw(ConnectionError("down")))


def test_resume_refuses_code_change_unless_allowed(tmp_path, monkeypatch, cfg):  # 결정 6
    import yaml
    import evopatch.cli as C
    rd = tmp_path / "runs" / "20990101_x"
    rd.mkdir(parents=True)
    cfg_path = rd / "config.snapshot.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg._data))
    (rd / "ids.txt").write_text("a\n")
    (rd / "run_meta.json").write_text(json.dumps({"run_id": "x", "code_sha256": {"cli.py": "old"}}))
    monkeypatch.setattr(C, "check_served_model", lambda cfg: {"models": ["m"]})
    with pytest.raises(SystemExit, match="allow-code-change"):
        C.main(["--config", str(cfg_path), "--run-dir", str(rd), "--resume"])

    class Stop(Exception):
        pass

    monkeypatch.setattr(C, "build_run_meta", lambda *a, **k: {})
    monkeypatch.setattr(C, "cleanup_label", lambda *a: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        C.main(["--config", str(cfg_path), "--run-dir", str(rd), "--resume", "--allow-code-change"])
    meta = json.loads((rd / "run_meta.json").read_text())
    assert meta["code_sha256"] == {"cli.py": "old"}  # 원래 기록은 그대로
    assert "cli.py" in meta["code_change_allowed"][0]["changed_files"]
    rm = json.loads(next(rd.glob("resume_meta_*.json")).read_text())
    assert "cli.py" in rm["code_change_allowed"]


def test_report_fitness_progress_counts_elites_per_generation(tmp_path):
    """세대 0 최고 후보가 엘리트로 남아도 세대 0 최고로 센다(후보별 마지막 행만 보면 개선으로 잘못 셈)."""
    from evopatch.report import build_report
    iss = {"run_id": "t", "instance_id": "a__b-1", "status": "ok", "attempt_id": "x", "total_prompt_tokens": 1,
           "total_completion_tokens": 1, "final_patch_path": None, "final_patch_empty": False,
           "terminated_by": "K", "regression_list_size": 3, "prep_location_range_rule": "from_first_decorator"}
    rows = []
    for g in range(3):
        rows.append({"instance_id": "a__b-1", "attempt_id": "x", "generation": g, "candidate_id": "g0c0",
                     "op": "init" if g == 0 else "elite", "valid": True, "r": 0, "g": 3, "g_total": 3,
                     "exec_timeout": False, "patch_hash": "h0", "meta": {}, "patch_lines": 1})
        rows.append({"instance_id": "a__b-1", "attempt_id": "x", "generation": g, "candidate_id": f"g{g}c1",
                     "op": "crossover", "valid": True, "r": 0, "g": 1, "g_total": 3,
                     "exec_timeout": False, "patch_hash": f"h{g}", "meta": {}, "patch_lines": 1})
    JsonlLog(tmp_path / "issues.jsonl").append(iss)
    for r in rows:
        JsonlLog(tmp_path / "candidates.jsonl").append(r)
    for n in ("llm_calls", "validity", "generations"):
        (tmp_path / f"{n}.jsonl").write_text("")
    rep = build_report(tmp_path)
    p = rep["fitness_progress_gen0_to_final"]
    assert (p["executed_issues"], p["improved"], p["no_improvement"]) == (1, 0, 1)
    assert rep["match_outside_location_after_decorator_fix"]["location_range_rule"] == {"from_first_decorator": 1}


def test_protected_path_inside_exempt_patch_text_is_restored():
    """보호 구간(위치 파일 경로)이 파일명 예외 구간(부모 diff) 안에 있어도 내부 표시가 남지 않는다(기준본 결함 수정)."""
    m = TestFileMasker(["pkg/tests/test_mod.py"])
    diff = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n+import test_mod\n"
    text = "Code:\npkg/mod.py\n" + diff + "see pkg/tests/test_mod.py"
    for _ in range(3):
        out = m(text, ["pkg/mod.py", "def f():\n    pass"], [diff])
        assert "\x00" not in out
        assert "diff --git a/pkg/mod.py b/pkg/mod.py" in out and "+import test_mod" in out
        assert "pkg/tests/test_mod.py" not in out
