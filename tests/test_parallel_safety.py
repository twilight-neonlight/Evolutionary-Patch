"""이슈 병렬 실행 안전성: 이슈 실행기의 시도별 공유 dict는 __init__에서 한 번만 만들고, 동시에 시작한 이슈들이 서로 덮어쓰지 않는다."""
import threading

import evopatch.runner as R
from evopatch.data_loader import InstanceView
from evopatch.logs import JsonlLog


class StubLLM:
    run_id = "t"
    model = "m"

    def __init__(self, tmp):
        self.call_log = JsonlLog(tmp / "llm_calls.jsonl")


def test_shared_dicts_created_in_init(tmp_path, cfg):
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                           exec_specs={}, regression_pool={})
    for name in ("_deadlines", "_prep_meta", "_mask_hits"):
        assert getattr(runner, name) == {}


def test_concurrent_issue_start_keeps_every_deadline(tmp_path, monkeypatch, cfg):
    n = 8
    cfg._data["unattended"].update(issue_timeout_s=7200, disk_guard_gb=None)
    barrier = threading.Barrier(n)
    seen = []

    class Prov:
        meta = {}

        def __init__(self, deps):
            self.deps = deps

        def prepare(self, view):
            barrier.wait(timeout=10)  # 모든 이슈가 시작 단계를 지난 뒤에 확인
            seen.append(runner._deadlines.get(self.deps.extra_log["attempt_id"]) is not None)
            raise RuntimeError("stop")

    monkeypatch.setattr(R, "make_provider", lambda name, deps: Prov(deps))
    runner = R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=StubLLM(tmp_path), executor=None,
                           exec_specs={}, regression_pool={})
    dicts = (id(runner._deadlines), id(runner._prep_meta), id(runner._mask_hits))
    ts = [threading.Thread(target=runner.run_issue, args=(InstanceView(f"x__y-{k}", "a/b", "c", "p", "1", "e"),))
          for k in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == [True] * n  # 이슈마다 시간 상한이 남아 있음
    assert (id(runner._deadlines), id(runner._prep_meta), id(runner._mask_hits)) == dicts
    assert runner._deadlines == {}  # 끝난 시도는 정리됨
    assert len(JsonlLog(tmp_path / "issues.jsonl").read()) == n
