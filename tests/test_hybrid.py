"""hybrid 전처리(2026-10-15): 제3자 연구 우선·LLM 대체, 두 우선순위, 출처 기록, 연결 불가·base 0개 경로, 제3자 연구 위치 변환. 도커·LLM 없음(가짜 의존성)."""
import json
from types import SimpleNamespace

import pytest

from evopatch.config import Config
from evopatch.prep import hybrid as H
from evopatch.prep import lab_fixed as L
from evopatch.prep.base import PrepDeps, make_provider
from evopatch.prep.simple import SimplePrep
from evopatch.types import Location, PrepFailure

ORIG = "import pytest\n\n\ndef test_old():\n    assert 1\n"
BLOCK = "\n\ndef test_new_issue():\n    assert f() == 2\n"
SRC = "import os\n\n\ndef helper():\n    return 1\n\n\nclass A:\n    @property\n    def meth(self):\n        x = 1\n        return x\n"
IID = "own__repo-1"
REPO_FILES = ["pkg/mod.py", "pkg/tests/test_a.py", "pkg/sub/mod2.py", "other/sub/mod2.py", "pkg/__init__.py"]
FAULT = [{"file_path": "/srv/proj/pkg/mod.py", "line_no": 11, "function_name": "meth", "source": "traceback"}]
LLM_LOC = Location(file="pkg/llm.py", symbol="g", start_line=1, end_line=2, code="def g():\n    pass")


def make_pkg(tmp_path, *, test_py=ORIG + BLOCK, fault=FAULT, artifact=True):
    pkg = tmp_path / "pkg"
    (pkg / "manifests").mkdir(parents=True, exist_ok=True)
    (pkg / "manifests" / "final_authoritative_manifest.csv").write_text(
        f"instance_id,final_status,raw_terminal_reason\n{IID},COMPLETE,\n")
    if artifact:
        d = pkg / "complete" / IID
        d.mkdir(parents=True, exist_ok=True)
        (d / "test.py").write_text(test_py)
        (d / "context.json").write_text(json.dumps({"candidate_test_files": [{"path": "pkg/tests/test_a.py"}]}))
        (d / "clue.json").write_text(json.dumps({"fault_locations": fault}))
    return pkg


class FakeExec:
    def __init__(self, statuses=None, error=None):
        self.statuses = {"pkg/tests/test_a.py::test_new_issue": "FAILED"} if statuses is None else statuses
        self.error = error
        self.runs = 0

    def read_files(self, iid, paths):
        return {p: (ORIG if "tests" in p else SRC) if p in REPO_FILES else None for p in paths}

    def list_files(self, iid):
        return list(REPO_FILES)

    def run_diff(self, diff, **kw):
        self.runs += 1
        assert diff == "" and kw["regression_tests"] is None  # base commit, 정답 패치 없음
        return SimpleNamespace(repro_statuses=self.statuses, error=self.error, timed_out=False,
                               repro_pass=False)


def provider(tmp_path, monkeypatch, pkg, *, priority="simple_first", ex=None, llm="ok", locs=(LLM_LOC,)):
    """llm: ok(테스트+식별자) / no_test / early(1단계 실패) / no_idents. locs: _resolve_locations 결과."""
    cfg = Config({"prep": {"provider": "hybrid", "location_priority": priority,
                           "lab_fixed": {}},
                   "external": {"repro_tests_dir": str(pkg)}})
    L._manifest_cache.clear()
    calls = {"llm": 0, "resolve": 0}

    def fake_call(self, view):
        calls["llm"] += 1
        if llm == "early":
            raise PrepFailure("candidate_files_not_found", ["u"], self.meta)
        test = None if llm == "no_test" else "def test_llm(): assert 0"
        idents = [] if llm == "no_idents" else [("pkg/llm.py", "g")]
        return test, idents, {}, ["u"], "pkg/tests/test_swe_repro.py", "pkg/tests/test_swe_repro.py"

    def fake_resolve(self, view, idents, sources):
        calls["resolve"] += 1
        return list(locs)
    monkeypatch.setattr(SimplePrep, "_llm_call", fake_call)
    monkeypatch.setattr(SimplePrep, "_resolve_locations", fake_resolve)
    deps = PrepDeps(cfg=cfg, llm=None, executor=ex or FakeExec(),
                    exec_specs={IID: SimpleNamespace(test_files=["pkg/tests/test_a.py"])}, regression_pool={IID: ["t1"]},
                    run_index=0, work_dir=tmp_path / "work")
    return make_provider("hybrid", deps), calls


class View:
    def __init__(self, iid=IID, repo="own/repo"):
        self.instance_id, self.repo, self.base_commit, self.problem_statement = iid, repo, "abc", "issue"




# ---------- simple_first ----------










# ---------- 제3자 연구 테스트를 쓸 수 없는 경로 ----------










# ---------- lab_first ----------










# ---------- 제3자 연구 위치 변환 ----------
def test_lab_fault_locations_rules():
    fault = [
        {"file_path": "repro.py", "line_no": 3, "function_name": "<module>"},  # 저장소에 없음
        {"file_path": "pkg/tests/test_a.py", "line_no": 4, "function_name": "test_old"},  # 테스트 파일
        {"file_path": "./pkg/mod.py", "line_no": 0, "function_name": "helper"},  # 이름으로 함수 범위
        {"file_path": "/x/y/pkg/mod.py", "line_no": 11, "function_name": "zzz"},  # 줄이 메서드 안(데코레이터부터)
        {"file_path": "pkg/mod.py", "line_no": 12, "function_name": "meth"},  # 같은 범위 중복
        {"file_path": "sub/mod2.py", "line_no": 1, "function_name": ""},  # 꼬리 일치 2개(같은 길이) → 버림
        {"file_path": "pkg/mod.py", "line_no": 1, "function_name": "<module>"},  # 함수 밖 → 그 줄 하나
        {"file_path": "pkg/mod.py", "line_no": 4, "function_name": "helper"},  # 4번째 위치(최대 3개)
    ]
    ex = FakeExec()
    locs, info = H.lab_fault_locations(fault, REPO_FILES, lambda ps: ex.read_files(IID, ps))
    assert [(l.file, l.symbol, l.start_line, l.end_line) for l in locs] == [
        ("pkg/mod.py", "helper", 4, 5), ("pkg/mod.py", "A.meth", 9, 12), ("pkg/mod.py", "<line 1>", 1, 1)]
    assert locs[1].code.startswith("    @property")
    assert info["lab_location_dropped"] == {"not_in_repo": 2, "test_or_non_py": 1, "duplicate": 1}
    assert info["lab_locations_used"] == 3


def test_map_repo_path():
    assert H.map_repo_path("pkg/mod.py", REPO_FILES) == "pkg/mod.py"
    assert H.map_repo_path("/srv/pkg/sub/mod2.py", REPO_FILES) == "pkg/sub/mod2.py"
    assert H.map_repo_path("sub/mod2.py", REPO_FILES) is None
    assert H.map_repo_path("nope.py", REPO_FILES) is None


# ---------- 러너 기록 ----------


