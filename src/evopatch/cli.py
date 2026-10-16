"""파이프라인 실행 진입점 (무인 실행 지원).

새 실행:  .venv/bin/python -m evopatch.cli --config <설정> --ids-file <ID 목록> [--run-id 이름]
이어 실행: .venv/bin/python -m evopatch.cli --config <설정> --run-dir runs/<날짜>_<run_id> --resume
- 실행 폴더는 runs/<YYYYMMDD>_<run_id>. 새 실행인데 폴더가 있으면 중단(덮어쓰지 않음).
- --resume: 결과에 영향을 주는 설정이 스냅샷과 다르면 중단. 운영 값(OPERATIONAL_KEYS)만 다르면 허용하고 기록.
  완료된 이슈(DONE_STATUSES)는 건너뛰고, error·infra_error 이슈는 다시 실행.
- 무인 모드(B4~B9): 입력 대기 없음. 인프라 오류는 infra_wait_max_s까지 대기·재시도, 그래도 안 되면 깨끗이 중단(종료 코드 3).
  디스크 여유가 고정 하한(unattended.disk_guard_gb, 기본 설정 120GB) 미만이면 새 이슈를 시작하지 않고 중단(종료 코드 4).
  이미지를 받을 수 없는 이슈는 image_unavailable(빈 패치)로 넘어가되, 서로 다른 이슈에서 같은 원인이
  연속 image_unavailable_escalate_after(기본 5)건이면 그 이슈들을 infra_error로 승격하고 인프라 대기·중단 절차를 밟는다.
  status.txt를 주기적으로 갱신.
  모든 이슈가 끝나면 예측 JSONL → grade.py → report.py를 각각 독립적으로 실행.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

from .config import Config
from .data_loader import load_instances, read_id_list
from .errors import DiskGuardStop
from .executor import Executor, cleanup_label
from .llm_client import LLMClient
from .logs import JsonlLog, disk_free_row
from .predictions import build_predictions, latest_issue_rows, model_name, write_predictions
from .runner import DONE_STATUSES, IssueRunner
from .specs import load_regression_pool, specs_from_config

ROOT = Path(__file__).resolve().parents[2]
# 결과에 영향을 주지 않는 운영 값: --resume 때 달라도 허용 (차이는 resume 메타에 기록)
OPERATIONAL_KEYS = ("llm.base_url", "llm.http_timeout_s", "llm.conn_retries", "llm.conn_backoff_s",
                    "run.issue_concurrency", "executor.max_parallel_containers", "unattended", "grading",
                    "diagnostics")  # diagnostics: 진단 전용(선택·종료·적합도와 무관)
EXIT_OK, EXIT_ERRORS, EXIT_INFRA_STOP, EXIT_DISK_STOP = 0, 1, 3, 4


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def served_models(base_url: str, fetch=None) -> dict:
    try:
        data = fetch(base_url) if fetch else httpx.get(base_url.rstrip("/") + "/models", timeout=10).json()
        return {"models": [m["id"] for m in data.get("data", [])]}
    except Exception as e:  # 기록용: 실패해도 실행 동작에는 영향 없음
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def code_hashes() -> dict[str, str]:
    src = ROOT / "src" / "evopatch"
    return {str(p.relative_to(src)): sha256_file(p) for p in sorted(src.rglob("*.py"))}


def code_changes(recorded: dict[str, str], now: dict[str, str]) -> list[str]:
    """기록된 코드 해시와 지금 코드가 다른 파일(추가·삭제 포함)."""
    return sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k))


def prompt_doc_meta(cfg: Config) -> dict:
    """프롬프트 고정 문서의 경로·SHA-256·타입(설정 prompts.*)."""
    from . import promptdoc
    d = promptdoc.load(cfg.get("prompts.path"))
    return {"path": str(d.path.relative_to(ROOT)) if d.path.is_relative_to(ROOT) else str(d.path), "sha256": d.sha256,
            "type": cfg.get("prompts.type"), "sections": len(d.sections)}


def check_external_checksum(cfg: Config) -> dict | None:
    """제3자 연구 재현 테스트 패키지: 파일 SHA-256 목록(external.repro_tests_checksum_list)의 SHA-256이
    external.repro_tests_checksum과 같고, 목록의 모든 파일이 패키지와 같으면 기록용 dict. 다르면 시작하지 않는다."""
    pkg, lst, want = (cfg.get("external.repro_tests_dir"), cfg.get("external.repro_tests_checksum_list"),
                      cfg.get("external.repro_tests_checksum"))
    if not pkg:
        return None
    if not (lst and want):
        return {"dir_set": True, "checksum": None}
    lst = Path(str(lst)).expanduser()
    lst = lst if lst.is_absolute() else ROOT / lst
    got = sha256_file(lst)
    if got != want:
        raise SystemExit(f"제3자 연구 패키지 체크섬 목록 불일치: {got} != {want}")
    root = Path(str(pkg)).expanduser()
    root = root if root.is_absolute() else ROOT / root
    bad = []
    n = 0
    for line in lst.read_text().splitlines():
        h, rel = line.split("  ", 1)
        n += 1
        f = root / rel
        if not f.is_file() or sha256_file(f) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"제3자 연구 패키지 파일 {len(bad)}개가 목록과 다름(예: {bad[:3]})")
    return {"format": cfg.get("external.repro_tests_format", "new_standalone"), "files": n, "checksum": got}


def check_served_model(cfg: Config, fetch=None) -> dict:
    """시작 전 검증: 서버 /v1/models에 설정의 llm.model이 있어야 한다. 없거나 확인할 수 없으면 시작하지 않는다."""
    served = served_models(cfg.require("llm.base_url"), fetch)
    want = cfg.require("llm.model")
    if "error" in served:
        raise SystemExit(f"서빙 모델을 확인할 수 없어 시작하지 않습니다: {served['error']}")
    if want not in served["models"]:
        raise SystemExit(f"서빙 중인 모델 {served['models']}에 설정의 llm.model={want!r}이 없어 시작하지 않습니다.")
    return served


def build_run_meta(config_path: Path, ids_file: Path | None, cfg: Config, fetch=None) -> dict:
    """B0: 코드·설정·서브셋 해시와 서빙 중인 모델명 (기록 전용)."""
    return {
        "code_sha256": code_hashes(),
        "config_file": str(config_path), "config_sha256": sha256_file(config_path),
        "config_digest": cfg.digest(),
        "ids_file": str(ids_file) if ids_file else None,
        "ids_sha256": sha256_file(ids_file) if ids_file else None,
        "served": served_models(cfg.require("llm.base_url"), fetch),
        "python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


GRADING_DIR = ROOT / "src" / "evopatch" / "grading"  # 채점 코드(실행 메타에 해시 기록)


def provenance(cfg: Config) -> dict:
    """2026-10-15: 새 실행 시작 시 출처 기록(기록 전용, resume 검사에는 쓰지 않음).
    - modules: 지금 불러온 모듈 중 이 폴더(ROOT) 아래 파일의 sha256(src 밖 analysis 제공자 포함)
    - data: exec_specs, 회귀 풀, 데이터셋 parquet, 제3자 연구 산출물 manifest의 sha256
    - env: 파이썬·플랫폼, 설치 패키지 목록 해시와 주요 패키지 버전, 채점 환경의 swebench 버전, 채점 코드 sha256"""
    import importlib.metadata as md
    mods = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().is_relative_to(ROOT) and f.endswith(".py"):
            mods[str(Path(f).resolve().relative_to(ROOT))] = sha256_file(Path(f))
    data = {}

    def add(key, v):
        for x in (v if isinstance(v, list) else [v]):
            x = Path(str(x)).expanduser()
            x = x if x.is_absolute() else ROOT / x
            if x.is_file():
                data[f"{key}:{x}"] = sha256_file(x)
    for key in ("data.exec_specs", "data.regression_pool", "data.parquet"):
        v = cfg.get(key)
        if v:
            add(key, v)
    lst = cfg.get("external.repro_tests_checksum_list")
    if lst:  # 제3자 연구 재현 테스트 패키지: 파일 SHA-256 목록의 sha256만 기록
        add("external_repro_tests_list", lst)
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    sw = sorted(str(p.name) for p in Path(str(cfg.get("grading.python") or sys.executable)).expanduser()
                .parent.parent.glob("lib/python*/site-packages/swebench-*.dist-info"))
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages_sha256": hashlib.sha256("\n".join(dists).encode()).hexdigest(), "n_packages": len(dists),
           "key_packages": [d for d in dists if d.split("==")[0].lower() in ("httpx", "pyyaml", "pyarrow", "swebench")],
           "grading_swebench": sw,
           "grading_code_sha256": {p.name: sha256_file(p) for p in sorted(GRADING_DIR.glob("*.py"))}}
    return {"modules_sha256": dict(sorted(mods.items())), "data_sha256": data, "env": env}


def save_code_snapshot(run_dir: Path) -> dict:
    """runs/<run>/code_snapshot/ 에 src/evopatch의 .py를 저장한다(새 실행에만)."""
    src = ROOT / "src" / "evopatch"
    dst = run_dir / "code_snapshot" / "src" / "evopatch"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.py"))
    for f in files:
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    return {"files": len(files)}


def _strip_operational(d: dict) -> dict:
    d = copy.deepcopy(d)
    for k in OPERATIONAL_KEYS:
        parts = k.split(".")
        node = d
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return d


def config_diff_operational(snap: Config, cfg: Config) -> tuple[bool, list[str]]:
    """(결과 영향 설정이 같은가, 달라진 운영 키 목록)."""
    same = json.dumps(_strip_operational(snap.as_dict()), sort_keys=True) == \
        json.dumps(_strip_operational(cfg.as_dict()), sort_keys=True)
    changed = []
    for k in OPERATIONAL_KEYS:
        if json.dumps(snap.get(k), sort_keys=True) != json.dumps(cfg.get(k), sort_keys=True):
            changed.append(k)
    return same, changed


def infra_healthy(cfg: Config, guard_gb: float | None, probe_image: str | None = None) -> tuple[bool, str]:
    try:
        r = httpx.get(cfg.require("llm.base_url").rstrip("/") + "/models", timeout=10)
        if r.status_code != 200:
            return False, f"vLLM HTTP {r.status_code}"
    except Exception as e:
        return False, f"vLLM: {type(e).__name__}"
    d = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if d.returncode != 0:
        return False, "docker daemon"
    if guard_gb is not None:
        free = disk_free_row({"d": "/var/lib/docker"}).get("d_free_gb", 0)
        if free < guard_gb:
            return False, f"disk {free}GB < {guard_gb}GB"
    if probe_image:  # A2 승격 후: 레지스트리에서 이미지 정보를 다시 얻을 수 있는지(내려받지 않음)
        m = subprocess.run(["docker", "manifest", "inspect", probe_image], capture_output=True, text=True)
        if m.returncode != 0:
            return False, f"registry: {m.stderr.strip()[:200]}"
    return True, "ok"


def count_oom_killed(run_dir: Path) -> int:
    """메모리 상한 초과(executor error="oom_killed")로 끝난 후보 실행 수. 같은 실행(엘리트 재기록)은 한 번만 센다.
    docker 오류율(watchdog의 container_error)과는 별개로 센다."""
    seen = set()
    for c in JsonlLog(run_dir / "candidates.jsonl").read():
        if (c.get("exec_error") or "").startswith("oom_killed"):
            seen.add(c.get("test_output_path") or c.get("candidate_id"))
    return len(seen)


class Status:
    """B8: status.txt 주기 갱신 + 인프라·중단 상태."""

    def __init__(self, run_dir: Path, ids: list[str], cfg: Config, guard_gb):
        self.run_dir, self.ids, self.cfg, self.guard_gb = run_dir, ids, cfg, guard_gb
        self.stop_reason = None
        self.started = time.time()
        self._stop = threading.Event()

    def write(self) -> None:
        rows = latest_issue_rows(JsonlLog(self.run_dir / "issues.jsonl"))
        all_rows = JsonlLog(self.run_dir / "issues.jsonl").read()
        done = [r for i, r in rows.items() if r.get("status") in DONE_STATUSES and i in self.ids]
        wc = [r["wall_clock_s"] for r in done if r.get("wall_clock_s") is not None]
        N = self.cfg.require("algorithm.N")
        short = sum(1 for r in done if r.get("gen0_valid_count") is not None and r["gen0_valid_count"] < N)
        infra = sum(1 for r in all_rows if r.get("status") == "infra_error")
        img = sum(1 for i, r in rows.items() if r.get("status") == "image_unavailable" and i in self.ids)
        disk = disk_free_row({"d": "/var/lib/docker"})
        status_counts = {}
        for r in rows.values():
            status_counts[r.get("status")] = status_counts.get(r.get("status"), 0) + 1
        text = (f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"done/total: {len(done)}/{len(self.ids)}\n"
                f"mean_s_per_issue: {round(statistics.mean(wc), 1) if wc else None}\n"
                f"gen0_shortfall_issues: {short}\n"
                f"infra_error_events: {infra}\n"
                f"image_unavailable_issues: {img}\n"
                f"oom_killed_execs: {count_oom_killed(self.run_dir)}\n"
                f"disk_free_gb: {disk.get('d_free_gb')} (guard {self.guard_gb})\n"
                f"status_counts: {json.dumps(status_counts)}\n"
                f"elapsed_s: {round(time.time() - self.started)}\n"
                f"stop_reason: {self.stop_reason}\n")
        tmp = self.run_dir / "status.txt.tmp"
        tmp.write_text(text)
        tmp.replace(self.run_dir / "status.txt")

    def loop(self) -> None:
        interval = self.cfg.get("unattended.status_interval_s", 300)
        while not self._stop.wait(interval):
            try:
                self.write()
            except Exception:
                pass

    def start(self):
        self.write()
        threading.Thread(target=self.loop, daemon=True).start()

    def close(self):
        self._stop.set()
        self.write()


def postprocess(run_dir: Path, cfg: Config) -> dict:
    """B9: 채점 → 보고서. 단계마다 독립 실행(앞 단계가 실패해도 다음 단계 실행)."""
    results = {}
    steps = [("grade", [sys.executable, "-m", "evopatch.grading.aggregate", "--run-dir", str(run_dir), "--per-issue"]),
             ("report", [sys.executable, "-m", "evopatch.report", "--run-dir", str(run_dir)])]
    for name, cmd in steps:
        log = run_dir / f"postprocess_{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with log.open("w") as f:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}).returncode
        except Exception as e:
            rc = f"{type(e).__name__}: {e}"
        results[name] = {"returncode": rc, "log": log.name}
    (run_dir / f"postprocess_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids-file")
    ap.add_argument("--local", help="서버별 설정(기본: config/local.yaml이 있으면 그것)")
    ap.add_argument("--run-id")
    ap.add_argument("--run-dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="--resume 때 코드 해시가 run_meta와 달라도 이어 실행 (허용 기록을 run_meta에 남김)")
    a = ap.parse_args(argv)

    cfg_path = Path(a.config).resolve()
    local = Path(a.local).resolve() if a.local else (cfg_path.parent / "local.yaml")  # 설정 파일과 같은 폴더의 local.yaml
    local = local if local.exists() else None
    cfg = Config.load(cfg_path, local)
    external_check = check_external_checksum(cfg)
    if a.resume:
        if not a.run_dir:
            raise SystemExit("--resume에는 --run-dir가 필요합니다")
        run_dir = Path(a.run_dir).resolve()
        snap = Config.load(run_dir / "config.snapshot.yaml")
        same, changed = config_diff_operational(snap, cfg)
        if not same:
            raise SystemExit("결과에 영향을 주는 설정이 스냅샷과 다릅니다. 같은 설정으로만 이어 실행할 수 있습니다.")
        meta = json.loads((run_dir / "run_meta.json").read_text())
        run_id = meta["run_id"]
        changed_code = code_changes(meta.get("code_sha256", {}), code_hashes())
        if changed_code and not a.allow_code_change:
            raise SystemExit(f"코드가 run_meta 기록과 다릅니다({changed_code}). 같은 코드로만 이어 실행합니다. "
                             "의도한 변경이면 --allow-code-change")
        check_served_model(cfg)
        if changed_code:  # 결정 6: 허용한 경우 run_meta에 기록(기존 키는 그대로 두고 목록에 추가)
            meta.setdefault("code_change_allowed", []).append(
                {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "changed_files": changed_code,
                 "code_sha256_now": {k: v for k, v in code_hashes().items() if k in changed_code}})
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        ids = read_id_list(run_dir / "ids.txt")
        if a.ids_file and read_id_list(a.ids_file) != ids:
            raise SystemExit("--ids-file이 실행 폴더의 ids.txt와 다릅니다")
        (run_dir / f"resume_meta_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(
            {"config": str(cfg_path), "operational_changes": {k: {"snapshot": snap.get(k), "now": cfg.get(k)} for k in changed},
             "code_change_allowed": changed_code,
             **build_run_meta(cfg_path, None, cfg)}, indent=1, ensure_ascii=False))
    else:
        if not a.ids_file:
            raise SystemExit("--ids-file이 필요합니다")
        ids = read_id_list(a.ids_file)
        check_served_model(cfg)
        run_id = a.run_id or cfg.get("run.run_id") or \
            f"{cfg.require('llm.model')}_N{cfg.require('algorithm.N')}K{cfg.require('algorithm.K')}_{time.strftime('%H%M%S')}"
        run_dir = (ROOT / cfg.require("run.runs_dir") / f"{time.strftime('%Y%m%d')}_{run_id}").resolve()
        if run_dir.exists():
            raise SystemExit(f"{run_dir}가 이미 있습니다(덮어쓰지 않음). 이어 실행하려면 --resume")
        run_dir.mkdir(parents=True)
        (run_dir / "config.snapshot.yaml").write_text(  # 합친 설정(main + local)
            yaml.safe_dump(cfg.as_dict(), allow_unicode=True, sort_keys=False))
        (run_dir / "ids.txt").write_text("\n".join(ids) + "\n")
        snap = save_code_snapshot(run_dir)  # 2026-10-15: 새 실행에만(이어 실행에는 적용하지 않음)
        (run_dir / "run_meta.json").write_text(json.dumps(
            {"run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             **build_run_meta(cfg_path, Path(a.ids_file).resolve(), cfg),
             "local_config": {"path": str(local), "sha256": sha256_file(local)} if local else None,
             "prompt_doc": prompt_doc_meta(cfg),
             "external_repro_tests": external_check,
             "provenance": provenance(cfg), "code_snapshot": snap}, indent=1, ensure_ascii=False))

    # 이후 동작은 실행 시점 설정(운영 값 포함)을 쓰되, 결과 영향 설정은 스냅샷과 같음이 보장된다
    sys_log = JsonlLog(run_dir / "system.jsonl")
    guard = cfg.get("unattended.disk_guard_gb")
    done = {iid for iid, r in latest_issue_rows(JsonlLog(run_dir / "issues.jsonl")).items()
            if r.get("status") in DONE_STATUSES}
    todo = [i for i in ids if i not in done]
    sys_log.append({"run_id": run_id, "event": "run_start", "resume": a.resume, "n_ids": len(ids),
                    "n_done": len(done), "n_todo": len(todo), "ts": time.time()})
    print(f"run_dir={run_dir}\n완료 {len(done)} / 남은 {len(todo)}", flush=True)

    label = run_dir.name.replace(".", "_")
    cleanup_label(label)
    specs = specs_from_config(cfg, ROOT)
    llm = LLMClient.from_config(cfg, call_log=JsonlLog(run_dir / "llm_calls.jsonl"), run_id=run_id,
                                response_log=JsonlLog(run_dir / "responses.jsonl"))
    executor = Executor.from_config(cfg, specs, label)
    runner = IssueRunner(cfg=cfg, run_dir=run_dir, run_id=run_id, llm=llm, executor=executor,
                         exec_specs=specs, regression_pool=load_regression_pool(cfg.path_value("data.regression_pool")))
    views = load_instances(cfg.path_value("data.parquet"), todo) if todo else {}
    status = Status(run_dir, ids, cfg, guard)
    status.start()
    stop = threading.Event()
    infra_lock = threading.Lock()
    exit_code = [EXIT_OK]

    def wait_for_infra(reason: str, probe_image: str | None = None) -> bool:
        """B5: 최대 infra_wait_max_s 동안 주기적으로 건강 검사. 회복하면 True."""
        with infra_lock:
            deadline = time.time() + cfg.get("unattended.infra_wait_max_s", 1800)
            poll = cfg.get("unattended.infra_poll_s", 60)
            sys_log.append({"run_id": run_id, "event": "infra_wait_start", "reason": reason, "ts": time.time()})
            while True:
                ok, why = infra_healthy(cfg, guard, probe_image)
                if ok:
                    sys_log.append({"run_id": run_id, "event": "infra_recovered", "ts": time.time()})
                    return True
                if time.time() >= deadline or stop.is_set():
                    sys_log.append({"run_id": run_id, "event": "infra_wait_giveup", "reason": why, "ts": time.time()})
                    return False
                time.sleep(min(poll, max(1, deadline - time.time())))

    esc_after = cfg.get("unattended.image_unavailable_escalate_after", 5)
    img_lock = threading.Lock()
    img_streak = {"cause": None, "rows": []}  # A2: 서로 다른 이슈의 연속 image_unavailable (완료 순서)

    def track_image(r) -> list[dict]:
        """연속 건수를 갱신하고, 승격할 때가 되면 그 행들을 돌려준다."""
        with img_lock:
            if r["status"] != "image_unavailable":
                img_streak.update(cause=None, rows=[])
                return []
            c = r.get("image_unavailable_cause")
            if c != img_streak["cause"]:
                img_streak.update(cause=c, rows=[])
            if r["instance_id"] not in {x["instance_id"] for x in img_streak["rows"]}:
                img_streak["rows"].append(r)
            if esc_after and len(img_streak["rows"]) >= esc_after:
                rows, img_streak["rows"] = img_streak["rows"], []
                img_streak["cause"] = None
                return rows
            return []

    def escalate(rows: list[dict]) -> None:
        """A2: 승격. 각 이슈에 같은 attempt_id로 infra_error 행을 덧붙인다(--resume 때 다시 실행됨)."""
        for x in rows:  # 이슈 실행기와 같은 로그 객체(같은 잠금)로 쓴다
            runner.issue_log.append({**x, "status": "infra_error", "escalated_from": "image_unavailable",
                        "error": f"image_unavailable x{len(rows)} 연속 ({x.get('image_unavailable_cause')}): {x.get('error')}"[:500]})
        sys_log.append({"run_id": run_id, "event": "image_unavailable_escalated", "n": len(rows),
                        "cause": rows[0].get("image_unavailable_cause"),
                        "instance_ids": [x["instance_id"] for x in rows], "ts": time.time()})

    def one(iid):
        while not stop.is_set():
            if guard is not None:  # B7: 새 이슈 시작 전 디스크 가드
                free = disk_free_row({"d": "/var/lib/docker"}).get("d_free_gb", 0)
                if free < guard:
                    status.stop_reason = f"disk_guard: free {free}GB < {guard}GB"
                    sys_log.append({"run_id": run_id, "event": "disk_guard_stop", "free_gb": free, "guard_gb": guard,
                                    "ts": time.time()})
                    exit_code[0] = EXIT_DISK_STOP
                    stop.set()
                    return None
            r = runner.run_issue(views[iid])
            print(f"[{time.strftime('%H:%M:%S')}] {iid}: {r['status']} term={r.get('terminated_by')} "
                  f"gens={r.get('generations_run')} tok={r['total_prompt_tokens']}+{r['total_completion_tokens']} "
                  f"t={r['wall_clock_s']}s", flush=True)
            if r["status"] == "infra_error" and (r.get("error") or "").startswith(DiskGuardStop.__name__):
                status.stop_reason = f"disk_guard: {r.get('error')}"  # A1: 대기 없이 중단
                sys_log.append({"run_id": run_id, "event": "disk_guard_stop", "error": r.get("error"), "ts": time.time()})
                exit_code[0] = EXIT_DISK_STOP
                stop.set()
                return r
            esc = track_image(r)
            if esc:
                escalate(esc)
                if not wait_for_infra(f"image_unavailable x{len(esc)}", probe_image=specs[iid].image):
                    status.stop_reason = f"infra_error: image_unavailable x{len(esc)} ({esc[0].get('image_unavailable_cause')})"
                    exit_code[0] = EXIT_INFRA_STOP
                    stop.set()
                return r
            if r["status"] != "infra_error":
                return r
            if not wait_for_infra(r.get("error") or "infra_error"):
                status.stop_reason = f"infra_error: {r.get('error')}"
                exit_code[0] = EXIT_INFRA_STOP
                stop.set()
                return r
        return None

    try:
        with ThreadPoolExecutor(cfg.require("run.issue_concurrency")) as ex:
            list(ex.map(one, todo))
    finally:
        left = cleanup_label(label)
        sys_log.append({"run_id": run_id, "event": "run_end", "leftover_containers_removed": left,
                        "stop_reason": status.stop_reason, "ts": time.time()})
        llm.close()
        status.close()

    if stop.is_set():
        print(f"중단: {status.stop_reason}\n--resume으로 이어 실행하세요. 끝난 이슈는 건너뜁니다.", flush=True)
        return exit_code[0]
    latest = latest_issue_rows(JsonlLog(run_dir / "issues.jsonl"))
    errors = [i for i in ids if latest.get(i, {}).get("status") not in DONE_STATUSES]
    if errors:
        print(f"오류/미완료 이슈 {len(errors)}개: {errors}\n--resume로 다시 실행하세요. 예측 파일은 쓰지 않았습니다.")
        return EXIT_ERRORS
    name = model_name(cfg.require("llm.model"), cfg.require("prep.provider"), cfg.require("algorithm.N"),
                      cfg.require("algorithm.K"))
    write_predictions(run_dir / "predictions.jsonl", build_predictions(run_dir, ids, name))
    marker = run_dir / "generation_complete.json"
    if not marker.exists():
        marker.write_text(json.dumps({"completed": time.strftime("%Y-%m-%d %H:%M:%S"), "n": len(ids)}))
    print(f"생성 완료. 예측: {run_dir / 'predictions.jsonl'}", flush=True)
    if cfg.get("unattended.postprocess", False) and not a.no_postprocess:
        print("후처리:", postprocess(run_dir, cfg), flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
