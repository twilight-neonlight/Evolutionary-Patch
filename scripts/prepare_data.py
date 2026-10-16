"""외부 데이터 받기·만들기(고정 버전, SHA-256 검증). 결과는 data/ 아래(git 제외), 출처·버전·해시는 data/MANIFEST(추적).

1 SWE-bench Lite·Verified: Hugging Face 데이터셋의 고정 revision에서 test 분할 parquet을 받는다.
  (실행 명세용 열 eval_script·image·log_parser·eval_type이 있는 revision. 파이프라인 로더는 허용 6필드만 읽는다.)
2 Agentless 회귀 목록: Agentless GitHub 릴리스(고정 태그)의 공개 산출물 zip에서 select_regression/output.jsonl만 꺼낸다
  (tests_passing_in_original_repo). zip은 data/downloads/에 둔다.
3 실행 명세: scripts/extract_exec_specs.py로 받은 parquet에서 만든다(test_files·명령·이미지; 정답 패치·평가 테스트 본문은 저장 안 함).
4 (선택 --hf-cache) 채점 하니스가 오프라인으로 데이터셋을 찾도록 같은 revision을 Hugging Face 캐시에 받는다.

받은 파일의 SHA-256이 기대값과 다르면 중단한다(--allow-mismatch면 기록만). --compare-dir를 주면 같은 이름의 다른 사본과
SHA-256을 비교해 data/compare.json에 기록한다(git 제외).
실행: python scripts/prepare_data.py [--only lite,verified,agentless_lite,agentless_verified,exec_specs] [--hf-cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}"
AGENTLESS = "https://github.com/OpenAutoCoder/Agentless/releases/download/{tag}/{file}"

SOURCES = {
    "lite": {"kind": "hf", "repo": "SWE-bench/SWE-bench_Lite", "revision": "b0dde1093fe417d83b7184254edf8199c1f0dff5",
             "file": "data/test-00000-of-00001.parquet", "out": "SWE-bench_Lite/test.parquet",
             "sha256": "438e281d80587aa7be470896ce410557002fde02d2ceee3e099331d308f62dd3"},
    "verified": {"kind": "hf", "repo": "SWE-bench/SWE-bench_Verified", "revision": "78f471bf655a3137b2e8a75af1501690ec009ec3",
                 "file": "data/test-00000-of-00001.parquet", "out": "SWE-bench_Verified/test.parquet",
                 "sha256": "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"},
    "agentless_lite": {"kind": "zip", "tag": "v1.5.0", "file": "agentless_swebench_lite.zip",
                       "zip_sha256": "a7787c0e1f0ed26fe68e938ef1f525b2b533d3b8d332adf9ab9a0ce5fe1faf52",
                       "member": "agentless_swebench_lite/select_regression/output.jsonl",
                       "out": "agentless/lite_select_regression.jsonl",
                       "sha256": "4cdc427b8008811989f32064c72659794264b00ddb602185aaf2e4130f39ec79"},
    "agentless_verified": {"kind": "zip", "tag": "v1.5.0", "file": "agentless_swebench_verified.zip",
                           "zip_sha256": "e8966cd3087a3f2d2e75a915959a1f007e0ea0cbfbef61ed23800053ce6784db",
                           "member": "agentless_swebench_verified/select_regression/output.jsonl",
                           "out": "agentless/verified_select_regression.jsonl",
                           "sha256": "959b2081422915f374d8489832893640a20f3264971740399aec4ac00a77fda0"},
}
EXEC_SPECS = {"lite": "exec_specs/lite.json", "verified": "exec_specs/verified.json"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def check(name: str, path: Path, want: str, allow: bool, log: dict) -> None:
    got = sha256(path)
    log[name] = {"path": str(path.relative_to(ROOT)), "sha256": got, "expected": want, "match": got == want}
    if got != want and not allow:
        raise SystemExit(f"{name}: SHA-256 불일치 {got} != {want}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="lite,verified,agentless_lite,agentless_verified,exec_specs")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--hf-cache", action="store_true", help="채점 하니스용으로 같은 revision을 Hugging Face 캐시에도 받음")
    ap.add_argument("--compare-dir", action="append", default=[], metavar="NAME=PATH",
                    help="다른 사본과 SHA-256 비교(예: lite=/path/test.parquet). 결과는 data/compare.json")
    a = ap.parse_args(argv)
    want = set(a.only.split(","))
    log, manifest = {}, []
    for name, s in SOURCES.items():
        if name not in want:
            continue
        out = DATA / s["out"]
        if s["kind"] == "hf":
            url = HF.format(repo=s["repo"], rev=s["revision"], file=s["file"])
            if not out.exists():
                download(url, out)
            check(name, out, s["sha256"], a.allow_mismatch, log)
            manifest.append(f"{s['out']}\t{s['sha256']}\thf:{s['repo']}@{s['revision']}/{s['file']}\t{url}")
        else:
            url = AGENTLESS.format(tag=s["tag"], file=s["file"])
            z = DATA / "downloads" / s["file"]
            if not z.exists():
                download(url, z)
            check(name + ":zip", z, s["zip_sha256"], a.allow_mismatch, log)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf, zf.open(s["member"]) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            check(name, out, s["sha256"], a.allow_mismatch, log)
            manifest.append(f"{s['out']}\t{s['sha256']}\tgithub:OpenAutoCoder/Agentless@{s['tag']}/{s['file']}:{s['member']}\t{url}"
                            f"\t(zip sha256 {s['zip_sha256']})")
    if "exec_specs" in want:
        for ds, rel in EXEC_SPECS.items():
            out = DATA / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_exec_specs.py"),
                                "--dataset", str(DATA / SOURCES[ds]["out"]), "--label", SOURCES[ds]["repo"] + "@" + SOURCES[ds]["revision"],
                                "--out", str(out)], check=True,
                               env={**__import__("os").environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
            got = sha256(out)
            log[f"exec_specs_{ds}"] = {"path": str(out.relative_to(ROOT)), "sha256": got}
            manifest.append(f"{rel}\t{got}\tgenerated: scripts/extract_exec_specs.py from {SOURCES[ds]['out']}\t-")
    if a.hf_cache:
        from huggingface_hub import snapshot_download
        for ds in ("lite", "verified"):
            if ds in want:
                snapshot_download(SOURCES[ds]["repo"], repo_type="dataset", revision=SOURCES[ds]["revision"])
                log[f"hf_cache_{ds}"] = {"revision": SOURCES[ds]["revision"]}
    if manifest:
        mf = DATA / "MANIFEST"
        old = [l for l in mf.read_text().splitlines()[3:] if l.strip()] if mf.exists() else []
        keep = {l.split("\t")[0]: l for l in old}
        keep.update({l.split("\t")[0]: l for l in manifest})
        mf.write_text("# data/MANIFEST — scripts/prepare_data.py가 쓴다. 이 파일만 git으로 추적하고 data/의 나머지는 제외한다.\n"
                      "# 형식: 경로(data/ 기준)<TAB>SHA-256<TAB>출처(고정 버전)<TAB>URL\n#\n"
                      + "\n".join(keep[k] for k in sorted(keep)) + "\n")
    cmp = {}
    for item in a.compare_dir:
        name, path = item.split("=", 1)
        p = Path(path)
        mine = log.get(name, {}).get("sha256")
        cmp[name] = {"other_sha256": sha256(p) if p.exists() else None, "main_sha256": mine,
                     "same": p.exists() and sha256(p) == mine}
    if cmp:
        (DATA / "compare.json").write_text(json.dumps(cmp, indent=1))
    (DATA / "prepare_log.json").write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **log}, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
