"""실행 명세 만들기(정답 쪽 오프라인 전처리, 데이터셋 eval_script·test_patch에서).

swebench 5.0.2에서는 테스트 명령이 이슈별 eval_script 안에만 있다. 이 스크립트는 다음을 만든다.
  repo_specs[repo@version]: eval_script에서 뽑은 (저장소, 버전)별 테스트 명령 앞부분, 사전 명령, log_parser.
                            같은 그룹 안의 모든 이슈에서 동일한지 확인하고, 다르면 중단한다.
  instances[iid]: image, repo, version, test_files(test_patch에서 뽑은 테스트 대상)
test_patch·eval_script 본문은 저장하지 않는다. 파이프라인 실행기는 출력 JSON만 읽는다(test_files는 마스킹·실행 대상에만).

실행: HF_HUB_OFFLINE=1 python scripts/extract_exec_specs.py --dataset data/SWE-bench_Lite/test.parquet \
        --label SWE-bench/SWE-bench_Lite@<revision> --out data/exec_specs/lite.json
"""
import argparse
import json
import re
from pathlib import Path

from swebench.harness.constants import END_TEST_OUTPUT, NON_TEST_EXTS, START_TEST_OUTPUT
from swebench.harness.utils import load_swebench_dataset

BASE = "<BASE_COMMIT>"


def test_directives(repo: str, test_patch: str) -> list[str]:
    """swebench 2.1.0 get_test_directives와 같은 규칙."""
    files = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    files = [f for f in files if not any(f.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":
        out = []
        for f in files:
            if f.endswith(".py"):
                f = f[: -len(".py")]
            if f.startswith("tests/"):
                f = f[len("tests/"):]
            out.append(f.replace("/", "."))
        files = out
    return files


def split_eval_script(inst: dict) -> tuple[str, list[str], str]:
    lines = inst["eval_script"].split("\n")
    i = next(k for k, l in enumerate(lines) if START_TEST_OUTPUT in l)
    j = next(k for k, l in enumerate(lines) if END_TEST_OUTPUT in l)
    cmd = " ".join(lines[i + 1 : j]).strip()
    # 사전 명령: shebang/set 다음부터 테스트 파일 checkout 직전까지 (테스트 패치 적용 이전)
    co = next(k for k, l in enumerate(lines) if l.startswith("git checkout " + inst["base_commit"]))
    pre = [l.replace(inst["base_commit"], BASE) for l in lines[:co]
           if l not in ("#!/bin/bash", "set -uxo pipefail")]
    return cmd, pre, inst["log_parser"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/SWE-bench_Verified/test.parquet", help="parquet 경로 또는 HF 이름")
    ap.add_argument("--label", default=None, help="출처 표시(예: HF 저장소@revision)")
    ap.add_argument("--out", default="data/exec_specs/verified.json")
    a = ap.parse_args()

    data = load_swebench_dataset(a.dataset, "test")
    repo_specs: dict[str, dict] = {}
    instances: dict[str, dict] = {}
    for inst in data:
        cmd, pre, parser = split_eval_script(inst)
        dirs = test_directives(inst["repo"], inst["test_patch"])
        if dirs:
            tail = " ".join(dirs)
            if not cmd.endswith(" " + tail):
                raise SystemExit(f"{inst['instance_id']}: eval_script 테스트 대상이 4.6a 추출과 다름")
            prefix = cmd[: -len(tail)].strip()
        else:
            prefix = cmd
        key = f"{inst['repo']}@{inst['version']}"
        spec = {"repo": inst["repo"], "version": inst["version"], "test_cmd_prefix": prefix,
                "pre_commands": pre, "log_parser": parser, "eval_type": inst["eval_type"]}
        if key in repo_specs and repo_specs[key] != spec:
            raise SystemExit(f"{key}: 같은 (저장소, 버전)인데 명령이 이슈마다 다름 ({inst['instance_id']})")
        repo_specs[key] = spec
        instances[inst["instance_id"]] = {
            "repo": inst["repo"], "version": inst["version"], "spec_key": key,
            "image": inst["image"], "test_files": dirs,
        }

    out = Path(a.out)
    payload = {
        "source": f"{a.label or a.dataset} eval_script + test_patch, swebench 5.0.2",
        "repo_specs": dict(sorted(repo_specs.items())),
        "instances": dict(sorted(instances.items())),
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    if out.exists() and out.read_text() != text:
        raise SystemExit(f"{out}가 이미 있고 내용이 다름 (덮어쓰지 않음)")
    out.write_text(text)
    empty = [k for k, v in instances.items() if not v["test_files"]]
    print(f"instances={len(instances)} repo_specs={len(repo_specs)} empty_test_files={empty}")


if __name__ == "__main__":
    main()
