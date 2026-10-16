"""후보 실행기 .

- 후보마다 공식 하네스 이미지로 새 컨테이너를 띄우고, 끝나면 반드시 제거한다(R3). 이미지는 지우지 않는다.
- 순서: 패치 적용(git apply --check → git apply) → 하네스 사전 명령 → 회귀(test_files 단위) → 재현 테스트.
- 판정은 하네스 코드를 그대로 쓴다: 로그 파서(PARSER_REGISTRY), 회귀 통과 판정(grading.test_passed).
- test_files는 오프라인 산출물(exec_specs)에서만 읽는다. test_patch는 여기 들어오지 않는다.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from swebench.harness.constants import TestStatus
from swebench.harness.grading import test_passed
from swebench.harness.log_parsers import PARSER_REGISTRY

from .errors import ImageUnavailable, InfraError
from .specs import ExecSpec
from .types import Candidate, ExecResult, PrepOutput, ReproTest

START = ">>>>> EVOPATCH START"
END = ">>>>> EVOPATCH END"
EXIT = ">>>>> EVOPATCH EXIT"
TIMEOUT_CODES = (124, 137)
SYMPY_FILE_MARK = re.compile(r"\s+\[(?:OK|FAIL)\]\s*$", re.M)
DJANGO = "django/django"


def _env_lines(pre_commands: tuple[str, ...]) -> list[str]:
    """사전 명령 중 셸 환경만 바꾸는 줄(activate, export, cd). 테스트 실행마다 다시 적용한다."""
    keep = []
    for c in pre_commands:
        s = c.strip()
        if s.startswith(("source ", "conda activate", "export ", "cd ")):
            keep.append(s)
    return keep


def repro_files(repo: str, repro: ReproTest) -> dict[str, str]:
    """컨테이너에 써 넣을 파일들. Django는 테스트 패키지라서 __init__.py가 함께 필요하다."""
    files = {repro.path: repro.content}
    if repo == DJANGO and repro.run_target == "swe_repro":
        files[str(Path(repro.path).parent / "__init__.py")] = ""
    return files


def select_repro_statuses(repo: str, repro: ReproTest, status_map: dict[str, str]) -> dict[str, str]:
    """하네스 파서 결과에서 재현 파일의 테스트만 고른다."""
    if repo in (DJANGO, "sympy/sympy"):
        # 재현 라벨/파일 하나만 실행하므로 파싱된 테스트가 전부 해당한다.
        # (Django 파서는 docstring이 있는 테스트를 docstring 첫 줄로 이름 붙이므로 라벨로 거를 수 없다)
        return dict(status_map)
    return {k: v for k, v in status_map.items() if k == repro.path or k.startswith(repro.path + "::")}


def repro_verdict(statuses: dict[str, str]) -> bool:
    """r=1: 재현 파일 테스트가 1개 이상 파싱되고 모두 PASSED (A6)."""
    return bool(statuses) and all(v == TestStatus.PASSED.value for v in statuses.values())


class ContainerError(RuntimeError):
    pass


# docker exec의 stderr에는 컨테이너 안 명령의 stderr도 섞이므로, 도커 자체의 오류 표시만 인프라로 본다.
DOCKER_INFRA_MARKERS = ("Cannot connect to the Docker daemon", "Is the docker daemon running", "No such container",
                        "error during connect")


# A2: 이미지를 받을 수 없음(레지스트리에 없음). 이슈 단위로 처리한다. 요청 제한·네트워크 오류는 여기 넣지 않는다(인프라).
IMAGE_UNAVAILABLE_MARKERS = (
    ("manifest_unknown", ("manifest unknown", "manifest for ")),
    ("repository_not_found", ("pull access denied", "repository does not exist")),
    ("not_found", ("failed to resolve reference",)),
)


def image_unavailable_cause(stderr: str) -> str | None:
    """docker run/pull의 stderr에서 '이미지 없음' 원인을 분류. 해당 없으면 None."""
    s = (stderr or "").lower()
    if "toomanyrequests" in s or "rate limit" in s:
        return None
    for cause, marks in IMAGE_UNAVAILABLE_MARKERS:
        if any(m in s for m in marks) and (cause != "not_found" or "not found" in s):
            return cause
    return None


def _docker_infra_failure(returncode: int, stderr: str) -> bool:
    """docker exec의 125는 도커 데몬 쪽 오류. 그 밖에는 도커 고유 문구가 있을 때만."""
    return returncode == 125 or any(m in (stderr or "") for m in DOCKER_INFRA_MARKERS)


class Executor:
    def __init__(
        self,
        *,
        specs: dict[str, ExecSpec],
        repro_timeout_s: int,
        regression_timeout_s: int,
        setup_timeout_s: int,
        max_parallel_containers: int,
        regression_cap: int | None,
        run_label: str,
        pull_min_interval_s: float = 0,
        memory_limit_gb: float | None = None,
    ):
        if max_parallel_containers > 48:
            raise ValueError("병렬 컨테이너는 48 이하 (4.6)")
        self.specs = specs
        self.repro_timeout_s = repro_timeout_s
        self.regression_timeout_s = regression_timeout_s
        self.setup_timeout_s = setup_timeout_s
        self.regression_cap = regression_cap
        self.run_label = run_label
        self._slots = threading.BoundedSemaphore(max_parallel_containers)
        # A4: 로컬에 없는 이미지를 받기 전 최소 간격(초). 0이면 기존 동작(docker run이 알아서 pull)
        self.pull_min_interval_s = pull_min_interval_s or 0
        self._pull_lock = threading.Lock()
        self._last_pull = 0.0
        # 운영 안전장치(2026-10-14): 컨테이너 메모리 상한(GB). None이면 상한 없음(기존 동작).
        # 지정하면 docker --memory·--memory-swap 같은 값(스왑 없음). 넘으면 컨테이너 안에서 OOM kill → 실행 결과 error="oom_killed"
        if memory_limit_gb is not None and memory_limit_gb <= 0:
            raise ValueError("executor.memory_limit_gb는 양수이거나 null")
        self.memory_limit_gb = memory_limit_gb

    @classmethod
    def from_config(cls, cfg, specs, run_label):
        return cls(
            specs=specs,
            repro_timeout_s=cfg.require("executor.repro_timeout_s"),
            regression_timeout_s=cfg.require("executor.regression_timeout_s"),
            setup_timeout_s=cfg.require("executor.setup_timeout_s"),
            max_parallel_containers=cfg.require("executor.max_parallel_containers"),
            regression_cap=cfg.get("executor.regression_cap"),
            run_label=run_label,
            pull_min_interval_s=cfg.get("unattended.pull_min_interval_s", 0),
            memory_limit_gb=cfg.get("executor.memory_limit_gb"),
        )

    def ensure_image(self, image: str) -> None:
        """A4: pull_min_interval_s > 0일 때만 동작. 로컬에 없으면 직전 pull과 간격을 두고 docker pull."""
        if not self.pull_min_interval_s:
            return
        with self._pull_lock:
            if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
                return
            wait = self._last_pull + self.pull_min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                p = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                raise InfraError("docker pull 시간 초과")
            finally:
                self._last_pull = time.monotonic()
            if p.returncode != 0:
                cause = image_unavailable_cause(p.stderr)
                if cause:
                    raise ImageUnavailable(cause, p.stderr.strip()[:300])
                raise InfraError(f"docker pull 실패: {p.stderr[:300]}")

    # ---------- 공개 API ----------
    def run(self, candidate: Candidate, prep: PrepOutput, out_dir: Path) -> ExecResult:
        """6.1 계약: 후보 하나를 실행해 (r, g)를 낸다."""
        return self.run_diff(
            candidate.diff,
            instance_id=prep.instance_id,
            base_commit=prep.base_commit,
            regression_tests=prep.regression_tests,
            repro=prep.repro_test,
            out_dir=out_dir,
        )

    def run_base_repro(self, prep: PrepOutput, out_dir: Path) -> ExecResult:
        """A5: 원본 코드에서 재현 테스트만 실행(관측용)."""
        return self.run_diff("", instance_id=prep.instance_id, base_commit=prep.base_commit,
                             regression_tests=None, repro=prep.repro_test, out_dir=out_dir)

    def read_files(self, instance_id: str, paths: list[str], diff: str = "") -> dict[str, str | None]:
        """저장소 파일 읽기(이미지의 /testbed, base commit). diff가 있으면 적용한 뒤 읽는다. 없으면 None."""
        spec = self.specs[instance_id]
        out: dict[str, str | None] = {}
        with self._container(spec) as cid:
            if diff.strip():
                self._copy_in(cid, {"/tmp/evopatch_read.diff": diff})
                r = self._exec(cid, "cd /testbed && git apply /tmp/evopatch_read.diff", timeout=120)
                if r.returncode != 0:
                    raise ContainerError(f"git apply 실패: {r.stderr[:300]}")
            for p in paths:
                r = self._exec(cid, f"cat /testbed/{shlex.quote(p)}", timeout=60)
                out[p] = r.stdout if r.returncode == 0 else None
        return out

    def find_defs(self, instance_id: str, names: list[str]) -> set[str]:
        """base commit 저장소의 .py 어딘가에 `def/class 이름`이 정의된 이름들 (진단 전용)."""
        names = sorted({n for n in names if n and n.isidentifier()})
        if not names:
            return set()
        spec = self.specs[instance_id]
        script = "cd /testbed\n" + "\n".join(
            f"git grep -qE '^\\s*(async\\s+)?(def|class) {n}\\b' -- '*.py' && echo {n}" for n in names)
        with self._container(spec) as cid:
            r = self._exec(cid, script, timeout=300)
        return set(r.stdout.split()) & set(names)

    def list_files(self, instance_id: str) -> list[str]:
        spec = self.specs[instance_id]
        with self._container(spec) as cid:
            r = self._exec(cid, "cd /testbed && git ls-files", timeout=120)
        return r.stdout.splitlines()

    def run_diff(
        self,
        diff: str,
        *,
        instance_id: str,
        base_commit: str,
        regression_tests: list[str] | None,
        repro: ReproTest | None,
        out_dir: Path,
    ) -> ExecResult:
        spec = self.specs[instance_id]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        res = ExecResult(repro_pass=False, reg_passed=0, reg_total=0, timed_out=False, output_path=str(out_dir))
        reg = None if regression_tests is None else list(regression_tests)
        if reg is not None and self.regression_cap is not None:
            reg = reg[: self.regression_cap]
        ctr = self._container(spec)
        try:
            with ctr as cid:
                (out_dir / "patch.diff").write_text(diff, encoding="utf-8")
                if diff.strip():
                    self._copy_in(cid, {"/tmp/evopatch.diff": diff})
                    r = self._exec(cid, "cd /testbed && git apply --check /tmp/evopatch.diff && git apply /tmp/evopatch.diff",
                                   timeout=120)
                    (out_dir / "apply.log").write_text(r.stdout + r.stderr, encoding="utf-8")
                    if r.returncode != 0:
                        res.apply_ok = False
                        res.error = "apply_failed"
                        return res

                setup = spec.pre_script(base_commit)
                r = self._exec(cid, setup, timeout=self.setup_timeout_s, inner_timeout=self.setup_timeout_s)
                (out_dir / "setup.log").write_text(r.stdout + r.stderr, encoding="utf-8")
                if r.timed_out:
                    res.timed_out = True
                    res.error = "setup_timeout"
                    return res

                # 회귀 (4.6a: 파일 단위 실행 → 파서 → 목록 대조)
                if reg is not None:
                    res.reg_total = len(reg)
                    if not spec.test_files:
                        res.no_regression_signal = True  # Q3b: 빈 test_files → r만으로
                        res.reg_total = 0
                    elif reg:
                        targets = " ".join(shlex.quote(t) for t in spec.test_files)
                        log, timed_out = self._run_tests(cid, spec, targets, self.regression_timeout_s)
                        (out_dir / "regression.log").write_text(log, encoding="utf-8")
                        res.reg_timed_out = timed_out
                        if not timed_out:
                            sm = self._parse(spec, log)
                            res.reg_passed = sum(1 for t in reg if test_passed(t, sm))
                            (out_dir / "regression_status.json").write_text(json.dumps(sm, indent=0))

                # 재현 테스트
                if repro is not None:
                    self._copy_in(cid, {f"/testbed/{p}": c for p, c in repro_files(spec.repo, repro).items()})
                    target = repro.run_target
                    log, timed_out = self._run_tests(cid, spec, target, self.repro_timeout_s)
                    (out_dir / "repro.log").write_text(log, encoding="utf-8")
                    res.repro_timed_out = timed_out
                    if not timed_out:
                        sm = self._parse(spec, log, repro=True)
                        res.repro_statuses = select_repro_statuses(spec.repo, repro, sm)
                        res.repro_pass = repro_verdict(res.repro_statuses)
                res.timed_out = res.reg_timed_out or res.repro_timed_out
        except ContainerError as e:
            res.error = f"container_error: {e}"[:500]
        finally:
            if getattr(ctr, "oom_killed", False):  # 메모리 상한 초과: 실행 오류로 기록(population: error → 무효 순위)
                res.error = "oom_killed" if res.error is None else f"oom_killed; {res.error}"[:500]
            res.duration_s = round(time.monotonic() - t0, 2)
            (out_dir / "result.json").write_text(json.dumps(res.to_json(), indent=1, ensure_ascii=False))
        return res

    # ---------- 내부 ----------
    def _parse(self, spec: ExecSpec, log: str, *, repro: bool = False) -> dict[str, str]:
        parser = PARSER_REGISTRY[spec.log_parser]
        if repro and spec.log_parser == "parse_log_sympy":
            # 하네스 sympy 파서는 파일의 마지막 테스트 줄(`test_x ok    [OK]`)을 놓친다.
            # 재현 판정에서만 줄 끝의 파일 요약 표시를 떼고 넘긴다(회귀는 공식 채점과 같게 그대로).
            log = SYMPY_FILE_MARK.sub("", log)
        sliced = log.split(START, 1)[-1].split(END, 1)[0]
        sm = parser(sliced, None)
        if not sm:
            sm = parser(log, None)  # 하네스와 같은 대체 경로(grading.get_logs_eval)
        return sm

    def _run_tests(self, cid: str, spec: ExecSpec, targets: str, timeout: int) -> tuple[str, bool]:
        env = "\n".join(_env_lines(spec.pre_commands))
        cmd = f"{spec.test_cmd_prefix} {targets}"
        script = (
            f"{env}\ncd /testbed\necho '{START}'\n"
            f"timeout -k 10 {timeout} bash -c {shlex.quote(cmd)} 2>&1\nrc=$?\n"
            f"echo '{END}'\necho \"{EXIT}: $rc\"\n"
        )
        r = self._exec(cid, script, timeout=timeout + 120)
        log = r.stdout + r.stderr
        rc = None
        for line in log.splitlines():
            if line.startswith(EXIT + ":"):
                rc = int(line.split(":", 1)[1].strip() or -1)
        timed_out = r.timed_out or rc in TIMEOUT_CODES
        return log, timed_out

    class _Proc:
        def __init__(self, returncode, stdout, stderr, timed_out):
            self.returncode, self.stdout, self.stderr, self.timed_out = returncode, stdout, stderr, timed_out

    def _exec(self, cid: str, script: str, *, timeout: int, inner_timeout: int | None = None):
        if inner_timeout:
            script = f"timeout -k 10 {inner_timeout} bash -c {shlex.quote(script)}"
        try:
            p = subprocess.run(["docker", "exec", "-i", cid, "bash", "-s"], input=script,
                               capture_output=True, text=True, timeout=timeout + 30, errors="replace")
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            return self._Proc(-1, out, "", True)
        if p.returncode != 0 and _docker_infra_failure(p.returncode, p.stderr):
            raise InfraError(f"docker exec: {p.stderr[:300]}")
        return self._Proc(p.returncode, p.stdout, p.stderr, inner_timeout is not None and p.returncode in TIMEOUT_CODES)

    def _copy_in(self, cid: str, files: dict[str, str]) -> None:
        for dest, content in files.items():
            d = shlex.quote(str(Path(dest).parent))
            p = subprocess.run(["docker", "exec", "-i", cid, "bash", "-c", f"mkdir -p {d} && cat > {shlex.quote(dest)}"],
                               input=content, capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                if _docker_infra_failure(p.returncode, p.stderr):
                    raise InfraError(f"docker copy: {p.stderr[:300]}")
                raise ContainerError(f"copy {dest}: {p.stderr[:200]}")

    def _container(self, spec: ExecSpec):
        return _Container(self, spec)


def memory_args(limit_gb: float | None) -> list[str]:
    """docker run 메모리 인자. None이면 빈 목록(기존 동작). 스왑을 허용하지 않도록 --memory-swap을 같은 값으로."""
    if limit_gb is None:
        return []
    m = f"{int(round(limit_gb * 1024))}m"
    return ["--memory", m, "--memory-swap", m]


def container_oom_killed(cid: str) -> bool:
    """컨테이너 cgroup에서 OOM kill이 있었는지. cgroup v2 memory.events의 oom_kill(컨테이너 안에서 읽음),
    읽지 못하면(주 프로세스까지 죽은 경우 등) docker inspect의 State.OOMKilled."""
    try:
        p = subprocess.run(["docker", "exec", cid, "cat", "/sys/fs/cgroup/memory.events"],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            for line in p.stdout.splitlines():
                k, _, v = line.partition(" ")
                if k == "oom_kill":
                    return int(v or 0) > 0
        q = subprocess.run(["docker", "inspect", "-f", "{{.State.OOMKilled}}", cid],
                           capture_output=True, text=True, timeout=30)
        return q.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, ValueError):
        return False


class _Container:
    """컨테이너 수명 관리. 어떤 경로로 끝나든 docker rm -f (R3)."""

    def __init__(self, ex: Executor, spec: ExecSpec):
        self.ex, self.spec, self.cid = ex, spec, None
        self.oom_killed = False

    def __enter__(self) -> str:
        self.ex.ensure_image(self.spec.image)
        self.ex._slots.acquire()
        name = f"evopatch-{self.ex.run_label}-{uuid.uuid4().hex[:10]}"
        try:
            p = subprocess.run(
                ["docker", "run", "-d", "--name", name, "--label", f"evopatch.run={self.ex.run_label}",
                 "--label", f"evopatch.issue={self.spec.instance_id}", *memory_args(self.ex.memory_limit_gb),
                 self.spec.image, "sleep", "infinity"],
                capture_output=True, text=True, timeout=1800)  # 처음 쓰는 이미지는 pull 시간 포함
        except subprocess.TimeoutExpired:
            self.ex._slots.release()
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            raise InfraError("docker run 시간 초과(이미지 pull 지연 등)")
        if p.returncode != 0:
            self.ex._slots.release()
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            cause = image_unavailable_cause(p.stderr)
            if cause:  # A2: 이미지가 레지스트리에 없음 → 이 이슈만 image_unavailable
                raise ImageUnavailable(cause, p.stderr.strip()[:300])
            # 그 밖에 컨테이너를 띄우지 못하면 풀이와 무관한 인프라 문제로 본다(B5)
            raise InfraError(f"docker run 실패: {p.stderr[:300]}")
        self.cid = p.stdout.strip()
        return self.cid

    def __exit__(self, *exc):
        try:
            if self.ex.memory_limit_gb is not None and self.cid:
                self.oom_killed = container_oom_killed(self.cid)
            subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True, timeout=120)
        finally:
            self.ex._slots.release()
        return False


def count_containers(run_label: str, instance_id: str | None = None) -> int:
    """남은 후보 컨테이너 수 (B6 로그용)."""
    cmd = ["docker", "ps", "-aq", "--filter", f"label=evopatch.run={run_label}"]
    if instance_id:
        cmd += ["--filter", f"label=evopatch.issue={instance_id}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return len(p.stdout.split())


def cleanup_label(run_label: str, instance_id: str | None = None) -> int:
    """이 실행 라벨(과 이슈)로 남은 컨테이너를 모두 제거(비정상 종료 대비). 이미지는 건드리지 않는다."""
    cmd = ["docker", "ps", "-aq", "--filter", f"label=evopatch.run={run_label}"]
    if instance_id:
        cmd += ["--filter", f"label=evopatch.issue={instance_id}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ids = p.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)
