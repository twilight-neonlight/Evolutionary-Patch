"""lab_fixed 전처리 (2026-10-14 사용자 결정): 제3자 연구 재현 테스트 결과물(handoff package, 읽기 전용)을 재현 테스트로 쓰고,
위치 등 나머지는 simple 전처리 로직을 그대로 쓴다.

- 결과물: <package_dir>/complete/<이슈>/test.py(기존 테스트 파일 + 덧붙인 블록), context.json(candidate_test_files 순위),
  manifests/final_authoritative_manifest.csv(제3자 연구 판정). 결과물은 읽기만 한다.
- 순서: (1) 제3자 연구 판정·test.py 확인 → 없으면 prep 실패(lab_no_test, LLM 호출 없음, 생성 실패 = prep 실패).
  (2) 후보 테스트 파일을 순위대로 base commit에서 읽어 test.py 앞부분과 일치하는 첫 파일을 대상 파일로 정하고,
  나머지를 덧붙인 블록(append_block)으로 뗀다. 분해가 안 되면 추측하지 않고 prep 실패(lab_unlinkable:<사유>).
  (3) simple 전처리를 그대로 실행해 위치를 얻는다(같은 호출이 내는 simple 재현 테스트는 버린다. 별도 생성 호출 없음).
  (4) 재현 테스트 = 대상 파일 경로에 제3자 연구 test.py 원문(= base commit 원본 + 덧붙인 블록)을 쓰고, 실행 대상은 덧붙인 블록의 테스트로 한정한다.
     pytest류: `경로::테스트` 나열, Django: `모듈.클래스` 라벨, sympy: 파일 전체(러너가 이름 선택을 지원하지 않음 → 기존 테스트 포함).
- 이슈마다 제3자 연구 판정(정합/비정합/오류/없음)을 meta(→ issues.jsonl의 prep_lab_verdict)에 남긴다.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import json
import threading
import warnings
from pathlib import Path

from ..data_loader import InstanceView
from ..types import PrepFailure, PrepOutput, ReproTest
from .simple import DJANGO, SYMPY, SimplePrep

VERDICTS = {"COMPLETE": "정합", "NO_ELIGIBLE_SCENARIO": "비정합", "M5_GENERATION_ERROR": "오류"}
_manifest_cache: dict[str, dict[str, str]] = {}
_manifest_lock = threading.Lock()


class LabUnlinkable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_verdicts(package_dir: Path) -> dict[str, str]:
    key = str(package_dir)
    with _manifest_lock:
        if key not in _manifest_cache:
            out = {}
            mf = package_dir / "manifests" / "final_authoritative_manifest.csv"
            if not mf.is_file():  # 새 형식(new_standalone) 패키지에는 판정 표가 없다
                _manifest_cache[key] = out
                return out
            with mf.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    k = r["final_status"] if r["final_status"] == "COMPLETE" else r.get("raw_terminal_reason", "")
                    out[r["instance_id"]] = VERDICTS.get(k, "기타")
            _manifest_cache[key] = out
        return _manifest_cache[key]


def split_lab_test(test_py: str, candidates: list[str], originals: dict[str, str | None]) -> tuple[str, str, int]:
    """(대상 파일, 덧붙인 블록, 후보 순위 1부터). 앞부분 일치는 원본 그대로 또는 끝 줄바꿈만 뺀 원본."""
    if not candidates:
        raise LabUnlinkable("대상 파일을 못 찾음(후보 없음)")
    readable = [p for p in candidates if originals.get(p) is not None]
    if not readable:
        raise LabUnlinkable("대상 파일을 못 찾음(후보 파일이 base commit에 없음)")
    for rank, p in enumerate(candidates, 1):
        o = originals.get(p)
        if o is None or not o.strip():
            continue
        for prefix in (o, o.rstrip("\n")):
            if test_py.startswith(prefix):
                block = test_py[len(prefix):]
                if not block.strip():
                    raise LabUnlinkable("덧붙인 블록 없음")
                return p, block, rank
    tl = test_py.splitlines()
    best = max(sum(1 for l in tl if l in set(originals[p].splitlines())) / max(1, len(tl)) for p in readable)
    raise LabUnlinkable("원본과 앞부분 불일치" if best > 0.2 else "새 파일")


def _is_test_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("Test"):
        return True
    return any("TestCase" in ast.unparse(b) for b in node.bases)


def block_tests(original: str, block: str) -> tuple[list[str], list[str]]:
    """덧붙인 블록에 새로 정의된 최상위 테스트 함수·테스트 클래스 이름 (함수, 클래스)."""
    full = original + block if original.endswith("\n") or block.startswith("\n") else original + block
    n_orig = original.count("\n") + (0 if original.endswith("\n") else 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(full)
    except SyntaxError:
        raise LabUnlinkable("연결한 파일 문법 오류")
    funcs, classes = [], []
    for n in tree.body:
        if n.lineno <= n_orig:
            continue
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            funcs.append(n.name)
        elif isinstance(n, ast.ClassDef) and _is_test_class(n):
            classes.append(n.name)
    return list(dict.fromkeys(funcs)), list(dict.fromkeys(classes))


def run_target_for(repo: str, path: str, funcs: list[str], classes: list[str]) -> str:
    if repo == DJANGO:
        if not (path.startswith("tests/") and path.endswith(".py")):
            raise LabUnlinkable("Django 테스트 라벨을 만들 수 없음(tests/ 밖)")
        if not classes:
            raise LabUnlinkable("Django 블록에 테스트 클래스 없음")
        module = path[len("tests/"):-3].replace("/", ".")
        return " ".join(f"{module}.{c}" for c in classes)
    if not funcs and not classes:
        raise LabUnlinkable("덧붙인 블록에 테스트 없음")
    if repo == SYMPY:
        return path  # sympy bin/test는 이름 선택 없이 파일 단위
    return " ".join(f"{path}::{n}" for n in classes + funcs)


class LabFixedPrep(SimplePrep):
    name = "lab_fixed"

    def prepare(self, view: InstanceView) -> PrepOutput:
        d = self.deps
        iid = view.instance_id
        pkg = Path(d.cfg.require("external.repro_tests_dir")).expanduser()
        verdict = load_verdicts(pkg).get(iid, "없음")
        self.meta.update(lab_verdict=verdict)
        idir = pkg / "complete" / iid
        if not (idir / "test.py").is_file():
            raise PrepFailure("lab_no_test", [], self.meta)  # 생성 실패 = prep 실패 (LLM 호출 없음)
        test_py = (idir / "test.py").read_text(encoding="utf-8")
        ctx = json.loads((idir / "context.json").read_text(encoding="utf-8")) if (idir / "context.json").is_file() else {}
        cands = [c["path"] for c in ctx.get("candidate_test_files") or [] if c.get("path")]
        try:
            originals = d.executor.read_files(iid, cands) if cands else {}
            target, block, rank = split_lab_test(test_py, cands, originals)
            funcs, classes = block_tests(originals[target], block)
            rt = run_target_for(view.repo, target, funcs, classes)
        except LabUnlinkable as e:
            self.meta.update(lab_unlinkable=e.reason)
            raise PrepFailure(f"lab_unlinkable:{e.reason}", [], self.meta)
        self.meta.update(lab_target_rank=rank, lab_append_lines=block.count("\n"),
                         lab_block_tests=len(funcs) + len(classes),
                         lab_run_scope="file" if view.repo == SYMPY else "block_tests",
                         lab_scenario=None)
        lab_meta = dict(self.meta)
        out = super().prepare(view)  # 위치 등: simple 로직 그대로 (실패하면 simple과 같은 prep 실패)
        self.meta = {**self.meta, **lab_meta}
        return dataclasses.replace(out, repro_test=ReproTest(path=target, content=test_py, run_target=rt),  # = 대상 원본 + 덧붙인 블록(제3자 연구 원문 그대로)
                                   prep_provider=self.name)
