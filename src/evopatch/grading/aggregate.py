"""집계 전용 공식 채점 (2026-10-12 사용자 지시, Lite 300 개발 런용). 이슈별 결과는 어디에도 저장하지 않는다.

- swebench 5.0.2 하니스를 임시 디렉터리(cwd, 로그, report_dir 모두 그 안)에서 실행한다. 하니스 stdout/stderr는 임시 파일로만.
- 각 예측은 처음 포함하는 데이터셋(설정 grading.datasets 순서: Verified 우선, 나머지 Lite)에 배정한다.
- 같은 프로세스 안에서 집계를 계산하고 임시 디렉터리를 지운 뒤 남은 하니스 컨테이너를 정리한다.
- 결과는 <run-dir>/grading_aggregate.json 하나(집계만). grading/·grading_private/ 폴더를 만들지 않는다.
- 저장·출력 전에 이슈 ID 패턴이 없는지 정규식으로 확인하고, 있으면 쓰지 않고 중단한다.
- 모든 칸은 n<3이면 인접 칸과 합친다. 저장소별은 n<5 저장소를 "기타"로 합친다.

실행: python -m evopatch.grading.aggregate --run-dir runs/<날짜>_<run_id> [--by-set] [--per-issue]
--per-issue: 이슈별 해결 여부를 <run-dir>/resolved_per_issue.json에 따로 저장(실행 폴더 안, git 제외). 집계 파일에는 이슈 ID를 쓰지 않는다.

--by-set(2026-10-14): 채점은 지금과 똑같이 한 번(예측마다 처음 포함하는 데이터셋 하나로 하니스 실행)만 하고, 집계 단계에서
설정 grading.datasets의 데이터셋마다 그 데이터셋 전체 이슈 목록(예: Lite 300, Verified 500)을 분모로 따로 집계한다.
두 데이터셋에 모두 있는 이슈(겹침)는 한 번 채점한 결과를 두 집계에 모두 넣는다. 실행하지 않았거나 예측이 없는 이슈도
분모에 남아 미해결로 셈한다. 결과 파일에 기존 집계(실행 이슈 전체)는 그대로 두고 "by_set"을 덧붙인다.
--by-set이 없으면 출력은 이전과 같다.
"""
import argparse
import hashlib
from collections import Counter
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq

from ..config import Config
from ..logs import JsonlLog
from ..predictions import latest_issue_rows

ID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+")


def read_ids(p: Path) -> set[str]:
    return {x.strip() for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")}


def dev_set(registry: Path | None) -> set[str]:
    """제외 집합(선택): 한 줄에 이슈 ID 하나인 목록 파일. 없으면 빈 집합(d_dev_set은 모두 "밖")."""
    return read_ids(registry) if registry else set()


def run_harness(cfg, name: str, preds: list[dict], tmp: Path, run_id: str) -> dict:
    """하니스 1회. stdout/stderr는 임시 파일로만 보내고 읽지 않는다. 반환: 하니스 report(dict)."""
    p = tmp / f"preds_{run_id}.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds))
    python = str(Path(cfg.get("grading.python") or sys.executable).expanduser())
    cmd = [python, "-m", "swebench.harness.run_evaluation", "-d", name, "-p", str(p),
           "-i", *[x["instance_id"] for x in preds], "--max_workers", str(cfg.require("grading.max_workers")),
           "-id", run_id, "--report_dir", str(tmp)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    with (tmp / f"harness_{run_id}.out").open("w") as f:
        rc = subprocess.run(cmd, cwd=tmp, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    reports = list(tmp.glob(f"*.{run_id}.json"))
    rep = json.loads(reports[0].read_text()) if reports else {}
    rep["_returncode"] = rc
    return rep


def cleanup_containers(run_id: str) -> int:
    p = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={run_id}"], capture_output=True, text=True)
    ids = p.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)


def merge_small(cells: list[list], minimum: int = 3) -> list[dict]:
    """cells: [[라벨, n, resolved], ...] 순서대로. n<minimum인 칸은 다음 칸(마지막이면 앞 칸)과 합친다."""
    cells = [list(c) for c in cells if c[1] > 0]
    i = 0
    while i < len(cells) and len(cells) > 1:
        if cells[i][1] < minimum:
            j = i + 1 if i + 1 < len(cells) else i - 1
            a, b = sorted((i, j))
            cells[a] = [f"{cells[a][0]} + {cells[b][0]} (n<{minimum} 합침)", cells[a][1] + cells[b][1], cells[a][2] + cells[b][2]]
            del cells[b]
            i = 0
            continue
        i += 1
    return [{"label": c[0], "n": c[1], "resolved": c[2], "rate": round(c[2] / c[1], 3) if c[1] else None} for c in cells]


def two_by_two(label: str, evaluated: list[str], cond: dict[str, bool], resolved: set[str]) -> dict:
    """조건 참/거짓 x resolved. 한 행에서 resolved·unresolved 칸 중 하나라도 n<3이면 그 행은 n만(칸 합침)."""
    rows = []
    for v, name in ((True, "참"), (False, "거짓")):
        xs = [i for i in evaluated if cond[i] is v]
        r = sum(i in resolved for i in xs)
        rows.append([f"{label}={name}", len(xs), r])
    out = []
    small_rows = [r for r in rows if r[1] < 3]
    if small_rows:  # 행 자체가 n<3이면 두 행을 합친다
        n, r = sum(x[1] for x in rows), sum(x[2] for x in rows)
        return {"rows": [{"label": f"{label}=참 + 거짓 (n<3 합침)", "n": n, "resolved": r}]}
    for name, n, r in rows:
        if min(r, n - r) < 3:
            out.append({"label": name, "n": n, "resolved": "resolved/unresolved 칸 n<3 합침"})
        else:
            out.append({"label": name, "n": n, "resolved": r, "rate": round(r / n, 3)})
    return {"rows": out}


def aggregate(run_dir: Path, reports: dict[str, dict], assigned: dict[str, str], ids: list[str], dev: set[str]) -> dict:
    latest = latest_issue_rows(JsonlLog(run_dir / "issues.jsonl"))
    resolved, evaluated = set(), set()
    harness = {}
    for name, rep in reports.items():
        resolved |= set(rep.get("resolved_ids") or [])
        evaluated |= set(rep.get("resolved_ids") or []) | set(rep.get("unresolved_ids") or [])
        harness[name] = {k: rep.get(k) for k in ("submitted_instances", "completed_instances", "resolved_instances",
                                                 "unresolved_instances", "empty_patch_instances", "error_instances")}
        harness[name]["returncode"] = rep.get("_returncode")
    tot = {k: sum((v.get(k) or 0) for v in harness.values()) for k in
           ("submitted_instances", "completed_instances", "resolved_instances", "empty_patch_instances", "error_instances")}
    n = len(ids)

    def term(i):
        r = latest.get(i, {})
        if r.get("status") == "ok":
            t = r.get("terminated_by")
            return "all_invalid+prep 실패" if t == "all_invalid" else t
        return "all_invalid+prep 실패" if r.get("status") == "prep_failed" else "기타 상태(시간 초과·예산 초과·이미지 없음·오류)"

    groups = {}
    for i in ids:
        groups.setdefault(term(i), []).append(i)
    order = ["perfect", "K", "insufficient_valid", "all_invalid+prep 실패", "기타 상태(시간 초과·예산 초과·이미지 없음·오류)"]
    by_term = merge_small([[k, len(groups.get(k, [])), sum(i in resolved for i in groups.get(k, []))] for k in order])
    ds = {}
    for i in ids:
        ds.setdefault(assigned.get(i, "unassigned"), []).append(i)
    by_ds = merge_small([[k, len(v), sum(i in resolved for i in v)] for k, v in ds.items()])
    ev = sorted(evaluated)
    r1 = {i: latest.get(i, {}).get("final_r") == 1 for i in ev}
    gfull = {i: latest.get(i, {}).get("final_g") is not None
             and latest[i].get("final_g") == latest[i].get("final_g_total") for i in ev}
    in_dev = [i for i in ids if i in dev]
    out_dev = [i for i in ids if i not in dev]
    by_dev = merge_small([["개발 집합에 속함", len(in_dev), sum(i in resolved for i in in_dev)],
                          ["개발 집합 밖", len(out_dev), sum(i in resolved for i in out_dev)]])
    repo = {}
    for i in ids:
        owner, rest = i.split("__", 1)
        repo.setdefault(f"{owner}/{rest.rsplit('-', 1)[0]}", []).append(i)
    big = sorted((k for k, v in repo.items() if len(v) >= 5), key=lambda k: -len(repo[k]))
    small = [i for k, v in repo.items() if len(v) < 5 for i in v]
    cells = [[k, len(repo[k]), sum(i in resolved for i in repo[k])] for k in big]
    if small:
        cells.append([f"기타(n<5 저장소 {sum(1 for v in repo.values() if len(v) < 5)}개)", len(small),
                      sum(i in resolved for i in small)])
    return {
        "run_dir": run_dir.name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_issues": n,
        "a_total": {"resolved": len(resolved), "rate": round(len(resolved) / n, 4) if n else None, "denominator": n,
                    "harness": tot, "harness_by_dataset": harness},
        "b_by_termination": by_term, "b_by_dataset": by_ds,
        "c_r1_x_resolved": two_by_two("r=1", ev, r1, resolved),
        "c_gfull_x_resolved": two_by_two("g=g_total", ev, gfull, resolved),
        "c_evaluated": len(ev),
        "d_dev_set": by_dev, "e_by_repo": merge_small(cells),
        "note": "집계만 저장. 이슈별 결과는 임시 디렉터리에서만 계산하고 삭제함. n<3 칸은 합침, 저장소는 n<5를 기타로 합침",
    }


REPORT_ID_KEYS = ("submitted", "completed", "resolved", "unresolved", "empty_patch", "error")


def filter_reports(reports: dict[str, dict], ids: set[str]) -> dict[str, dict]:
    """하니스 보고서를 ids에 속한 이슈로만 줄인다(세트별 집계용). 개수 칸은 걸러진 목록 길이로 다시 센다."""
    out = {}
    for name, rep in reports.items():
        r = dict(rep)
        for k in REPORT_ID_KEYS:
            if f"{k}_ids" in rep:
                r[f"{k}_ids"] = [i for i in rep.get(f"{k}_ids") or [] if i in ids]
                r[f"{k}_instances"] = len(r[f"{k}_ids"])
        out[name] = r
    return out


def aggregate_by_set(run_dir: Path, reports: dict[str, dict], assigned: dict[str, str], run_ids: list[str], dev: set[str],
                     sets: list[tuple[str, str, list[str]]]) -> dict:
    """sets: [(데이터셋 이름, 짧은 이름, 그 데이터셋 전체 이슈 목록)]. 세트마다 분모 = 그 목록 길이."""
    out, members = {}, {}
    run_set = set(run_ids)
    for name, short, set_ids in sets:
        ids = sorted(set_ids)
        members[short] = set(ids)
        agg = aggregate(run_dir, filter_reports(reports, set(ids)), assigned, ids, dev)
        agg["set"] = {"dataset": name, "denominator": len(ids), "in_run": len(run_set & set(ids)),
                      "not_in_run": len(set(ids) - run_set),
                      "graded_under": dict(Counter(assigned.get(i, "unassigned") for i in ids))}
        out[short] = agg
    shorts = [s for _, s, _ in sets]
    check = {"denominators": {s: len(members[s]) for s in shorts}}
    if len(shorts) == 2:
        a, b = shorts
        both = members[a] & members[b]
        check["overlap"] = len(both)
        check["overlap_in_both_aggregates"] = all(i in members[a] and i in members[b] for i in both)
        check["union"] = len(members[a] | members[b])
    return {"by_set": out, "by_set_check": check}


def assert_no_ids(text: str) -> None:
    if ID_RE.search(text):
        raise SystemExit("집계 출력에 이슈 ID 패턴이 있어 저장하지 않습니다")


def main(argv=None, harness=run_harness):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--registry", default=None, help="(선택) 따로 집계할 개발 집합 이슈 목록 파일")
    ap.add_argument("--by-set", action="store_true", help="데이터셋별(예: Lite 300 / Verified 500) 집계를 덧붙임")
    ap.add_argument("--per-issue", action="store_true", help="이슈별 해결 여부를 <run-dir>/resolved_per_issue.json에 저장")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir).resolve()
    out_path = run_dir / "grading_aggregate.json"
    if out_path.exists():
        raise SystemExit(f"{out_path}가 이미 있습니다(재채점하지 않음)")
    if not (run_dir / "generation_complete.json").exists():
        raise SystemExit("생성이 끝나지 않았습니다")
    cfg = Config.load(run_dir / "config.snapshot.yaml")
    ids = [x.strip() for x in (run_dir / "ids.txt").read_text().splitlines() if x.strip()]
    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines() if l.strip()]
    dev = dev_set(Path(a.registry) if a.registry else None)
    assigned, groups, sets = {}, [], []
    for d in cfg.require("grading.datasets"):
        dids = set(pq.read_table(Path(d["parquet"]).expanduser(), columns=["instance_id"]).column(0).to_pylist())
        sets.append((d["name"], d["short"], sorted(dids)))
        mine = [p for p in preds if p["instance_id"] in dids and p["instance_id"] not in assigned]
        for p in mine:
            assigned[p["instance_id"]] = d["name"]
        if mine:
            groups.append((d["name"], d["short"], mine))
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="evopatch_grade_"))
    reports, left = {}, 0
    tag = f"agg{int(t0)}"
    try:
        for name, short, mine in groups:
            run_id = f"{tag}_{short}"
            try:
                reports[name] = harness(cfg, name, mine, tmp, run_id)
            finally:
                left += cleanup_containers(run_id)
        agg = aggregate(run_dir, reports, assigned, ids, dev)
        if a.by_set:
            agg.update(aggregate_by_set(run_dir, reports, assigned, ids, dev, sets))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    agg["grading"] = {"harness": "swebench 5.0.2 run_evaluation", "max_workers": cfg.require("grading.max_workers"),
                      "dataset_order": [d["name"] for d in cfg.require("grading.datasets")],
                      "elapsed_s": round(time.time() - t0, 1), "temp_dir_removed": not tmp.exists(),
                      "leftover_containers_removed": left}
    text = json.dumps(agg, indent=1, ensure_ascii=False)
    assert_no_ids(text)
    out_path.write_text(text)
    if a.per_issue:
        write_per_issue(run_dir, reports, ids)
    print(text)
    return 0


def write_per_issue(run_dir: Path, reports: dict[str, dict], ids: list[str]) -> Path:
    """이슈별 해결 여부(실행 폴더 안, 이슈 ID 포함 → git 제외 대상). 하니스 보고서의 resolved_ids 기준, 나머지는 미해결."""
    resolved = sorted({i for rep in reports.values() for i in rep.get("resolved_ids") or []})
    completed = sum((rep.get("completed_instances") or 0) for rep in reports.values())
    out = run_dir / "resolved_per_issue.json"
    out.write_text(json.dumps({"n": len(ids), "resolved": resolved, "unresolved": sorted(set(ids) - set(resolved)),
                               "completed_instances": completed}, indent=1))
    return out


if __name__ == "__main__":
    sys.exit(main())
