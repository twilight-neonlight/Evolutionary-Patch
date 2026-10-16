"""운영 안전장치: 컨테이너 메모리 상한(executor.memory_limit_gb, 2026-10-14 사용자 GO).

- 기본값(null)이면 docker run 인자·종료 처리·실행 결과가 기존과 같다.
- 상한을 지정하면 --memory·--memory-swap 같은 값으로 띄우고, 컨테이너 안 OOM kill이 있으면 실행 결과 error="oom_killed".
- 실제 docker 확인(로컬 이미지가 있을 때만): 상한을 넘는 할당이 OOM kill로 감지된다.
"""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from evopatch import executor as exmod
from evopatch.config import Config
from evopatch.executor import Executor, container_oom_killed, memory_args
from evopatch.specs import ExecSpec

SPEC = ExecSpec(instance_id="o__r-1", repo="o/r", version="1", image="img:latest", test_cmd_prefix="pytest",
                pre_commands=("true",), log_parser="parse_log_pytest", test_files=("t.py",))


def make_ex(limit=None):
    return Executor(specs={SPEC.instance_id: SPEC}, repro_timeout_s=10, regression_timeout_s=10, setup_timeout_s=10,
                    max_parallel_containers=2, regression_cap=None, run_label="memtest", memory_limit_gb=limit)


def test_memory_args():
    assert memory_args(None) == []
    assert memory_args(16) == ["--memory", "16384m", "--memory-swap", "16384m"]
    assert memory_args(0.5) == ["--memory", "512m", "--memory-swap", "512m"]
    with pytest.raises(ValueError):
        make_ex(0)


def test_from_config_default_is_none(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("executor:\n  repro_timeout_s: 1\n  regression_timeout_s: 1\n  setup_timeout_s: 1\n"
                 "  max_parallel_containers: 1\n")
    assert Executor.from_config(Config.load(p), {}, "x").memory_limit_gb is None
    p.write_text(p.read_text() + "  memory_limit_gb: 16\n")
    assert Executor.from_config(Config.load(p), {}, "x").memory_limit_gb == 16


class FakeDocker:
    """subprocess.run 대역: docker run/exec/rm/inspect 호출을 기록하고 정해진 결과를 돌려준다."""

    def __init__(self, oom_events="oom 0\noom_kill 0\n"):
        self.calls, self.oom_events = [], oom_events

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return SimpleNamespace(returncode=0, stdout="cid123\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and cmd[-1] == "/sys/fs/cgroup/memory.events":
            return SimpleNamespace(returncode=0, stdout=self.oom_events, stderr="")
        if cmd[:2] == ["docker", "exec"]:  # 패치 적용·셋업·테스트
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def run_once(monkeypatch, tmp_path, limit, oom_events="oom 0\noom_kill 0\n"):
    fake = FakeDocker(oom_events)
    monkeypatch.setattr(exmod.subprocess, "run", fake)
    ex = make_ex(limit)
    monkeypatch.setattr(ex, "ensure_image", lambda image: None)
    monkeypatch.setattr(ex, "_run_tests", lambda cid, spec, targets, timeout: ("", False))
    monkeypatch.setattr(ex, "_parse", lambda spec, log, repro=False: {"t.py::a": "PASSED"})
    res = ex.run_diff("", instance_id=SPEC.instance_id, base_commit="abc", regression_tests=["t.py::a"],
                      repro=None, out_dir=tmp_path / "o")
    return res, fake.calls


def test_default_behavior_unchanged(monkeypatch, tmp_path):
    res, calls = run_once(monkeypatch, tmp_path, None, oom_events="oom_kill 3\n")
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    assert "--memory" not in run and "--memory-swap" not in run
    assert not any(c[-1] == "/sys/fs/cgroup/memory.events" for c in calls)  # 상한 없으면 OOM 확인도 안 함
    assert res.error is None and res.reg_passed == 1 and res.reg_total == 1


def test_limit_without_oom_is_normal(monkeypatch, tmp_path):
    res, calls = run_once(monkeypatch, tmp_path, 16)
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    i = run.index("--memory")
    assert run[i:i + 4] == ["--memory", "16384m", "--memory-swap", "16384m"]
    assert run.index("--memory") < run.index("img:latest")  # 이미지 이름 앞(docker run 옵션)
    assert res.error is None and res.reg_passed == 1


def test_limit_exceeded_recorded_as_error(monkeypatch, tmp_path):
    res, calls = run_once(monkeypatch, tmp_path, 16, oom_events="oom 1\noom_kill 1\n")
    assert res.error == "oom_killed"
    # 기존 경로: 실행 오류가 있는 후보는 무효 순위(population.Individual)로 간다
    import inspect
    from evopatch import population
    assert "self.exec.error" in inspect.getsource(population)


def test_oom_falls_back_to_inspect(monkeypatch):
    def fake(cmd, **kw):
        if cmd[:2] == ["docker", "exec"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="container not running")
        return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
    monkeypatch.setattr(exmod.subprocess, "run", fake)
    assert container_oom_killed("cid") is True


def _local_python_image():
    if not shutil.which("docker"):
        return None
    p = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True)
    imgs = [x for x in p.stdout.split() if "sweb.eval" in x]
    return imgs[0] if imgs else None


@pytest.mark.skipif(_local_python_image() is None, reason="로컬 docker 이미지 없음")
def test_real_docker_oom_detected():
    img = _local_python_image()
    ex = make_ex(0.25)
    spec = ExecSpec(instance_id="o__r-2", repo="o/r", version="1", image=img, test_cmd_prefix="true", pre_commands=(),
                    log_parser="parse_log_pytest", test_files=())
    ctr = exmod._Container(ex, spec)
    with ctr as cid:
        r = ex._exec(cid, "python3 -c 'x = bytearray(1024**3); print(len(x))'", timeout=120)
        assert r.returncode != 0  # 1GB 할당이 상한(256MB)에 걸려 죽음
    assert ctr.oom_killed is True
    ok = exmod._Container(ex, spec)
    with ok as cid:
        ex._exec(cid, "python3 -c 'x = bytearray(10 * 1024**2)'", timeout=120)
    assert ok.oom_killed is False


def _status_text(tmp_path, cands):
    import json
    from evopatch.cli import Status
    (tmp_path / "issues.jsonl").write_text(json.dumps({"instance_id": "o__r-1", "status": "ok", "wall_clock_s": 1.0}) + "\n")
    if cands is not None:
        (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cands))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("algorithm:\n  N: 10\n")
    st = Status(tmp_path, ["o__r-1"], Config.load(cfg), None)
    st.write()
    return (tmp_path / "status.txt").read_text()


def test_status_records_oom_count_zero(tmp_path):
    assert "oom_killed_execs: 0\n" in _status_text(tmp_path, None)  # candidates.jsonl 없음
    assert "oom_killed_execs: 0\n" in _status_text(tmp_path, [{"test_output_path": "a", "exec_error": None},
                                                                {"test_output_path": "b", "exec_error": "container_error: x"}])


def test_status_records_oom_count(tmp_path):
    c = [{"test_output_path": "a", "exec_error": "oom_killed"}, {"test_output_path": "a", "exec_error": "oom_killed"},
         {"test_output_path": "b", "exec_error": "oom_killed; setup_timeout"}, {"test_output_path": "c", "exec_error": None}]
    text = _status_text(tmp_path, c)
    assert "oom_killed_execs: 2\n" in text  # 같은 실행 재기록은 한 번만
    assert "o__r-1" not in text  # 상태 파일에 이슈 ID 없음


