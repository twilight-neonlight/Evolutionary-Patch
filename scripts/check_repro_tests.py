"""제3자 연구 재현 테스트 패키지 점검(LLM 없음): 이슈마다 어댑터의 채택 판정(파싱 → 저장소별 고정 경로에 두고 base에서 실행 →
수집 1개 이상이면 채택)을 실제 Docker로 돌려 저장소별 채택·대체 수와 대체 사유를 집계한다.
출력: <out>.json(집계만, 이슈 ID 없음)과 <out>_per_issue.json(이슈별, git 제외 위치에 둘 것).
실행: python scripts/check_repro_tests.py --config config/main.yaml --out reports/repro_tests_check [--workers 16] [--limit N]
"""
import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evopatch.config import Config  # noqa: E402
from evopatch.data_loader import load_instances  # noqa: E402
from evopatch.executor import Executor, cleanup_label  # noqa: E402
from evopatch.prep.base import PrepDeps  # noqa: E402
from evopatch.prep.hybrid import HybridPrep, issue_dir, package_format  # noqa: E402
from evopatch.specs import load_regression_pool, specs_from_config  # noqa: E402

ID_RE = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/main.yaml")
    ap.add_argument("--local", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfgp = Path(a.config)
    local = Path(a.local) if a.local else (cfgp.parent / "local.yaml" if (cfgp.parent / "local.yaml").exists() else None)
    cfg = Config.load(cfgp, local)
    pkg = Path(str(cfg.require("external.repro_tests_dir")))
    pkg = pkg if pkg.is_absolute() else ROOT / pkg
    fmt = package_format(cfg)
    ids = sorted(p.name for p in (pkg if fmt == "new_standalone" else pkg / "complete").iterdir() if p.is_dir() and "__" in p.name)
    if a.limit:
        ids = ids[:a.limit]
    specs = specs_from_config(cfg, ROOT)
    views = load_instances(cfg.path_value("data.parquet"), ids)
    label = f"repro_check_{int(time.time()) % 10**8}"
    ex = Executor.from_config(cfg, specs, label)
    work = Path(tempfile.mkdtemp(prefix="repro_check_"))
    rows = []

    def one(iid):
        deps = PrepDeps(cfg=cfg, llm=None, executor=ex, exec_specs=specs, regression_pool={}, run_index=0,
                        work_dir=work / iid)
        p = HybridPrep(deps)
        t0 = time.time()
        try:
            repro, reason = p._lab_test(views[iid], issue_dir(pkg, iid, fmt))
        except Exception as e:  # 이슈 하나의 오류는 기록하고 계속
            repro, reason = None, f"점검 오류: {type(e).__name__}"
        r = {"instance_id": iid, "repo": views[iid].repo, "adopted": repro is not None, "reason": reason,
             "n_tests": p.meta.get("lab_base_n_tests"), "detail": p.meta.get("lab_fallback_detail"),
             "seconds": round(time.time() - t0, 1)}
        rows.append(r)
        if len(rows) % 25 == 0:
            print(f"{len(rows)}/{len(ids)}", flush=True)
        return r

    try:
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(one, ids))
    finally:
        cleanup_label(label)
    by_repo = defaultdict(Counter)
    for r in rows:
        by_repo[r["repo"]]["adopted" if r["adopted"] else f"fallback:{r['reason']}"] += 1
    tot = Counter()
    for c in by_repo.values():
        tot.update(c)
    dj = by_repo.get("django/django", Counter())
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "format": fmt, "issues": len(rows),
           "adopted": sum(r["adopted"] for r in rows), "fallback": sum(not r["adopted"] for r in rows),
           "total": dict(tot), "by_repo": {k: dict(v) for k, v in sorted(by_repo.items())},
           "django": {"issues": sum(dj.values()), "adopted": dj.get("adopted", 0),
                      "zero_collected": dj.get("fallback:수집 0", 0), "import_error": dj.get("fallback:import 오류", 0),
                      "other_fallback": {k: v for k, v in dj.items() if k.startswith("fallback:") and k not in
                                         ("fallback:수집 0", "fallback:import 오류")}}}
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if ID_RE.search(text):
        raise SystemExit("집계에 이슈 ID 패턴")
    Path(a.out + ".json").write_text(text)
    Path(a.out + "_per_issue.json").write_text(json.dumps(sorted(rows, key=lambda r: r["instance_id"]), ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
