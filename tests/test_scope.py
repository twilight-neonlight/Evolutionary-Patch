"""S2(algorithm.edit_scope=file, 미결정 옵션): 파일 범위 편집 규칙, 코드 컨텍스트 확장·축소, 프롬프트 문구. 도커·LLM 없음."""
import pytest

from evopatch import prompts
from evopatch.edits import (FILE_NOT_IN_LOCATIONS, MATCH_OUTSIDE_LOCATION, SEARCH_NOT_UNIQUE, apply_edits,
                            build_candidate)
from evopatch.scope import build_code_view, enclosing_class_headers, excerpt_segments, module_segments
from evopatch.types import Edit, Location

SRC = ("import os\nX = 1\n\n\ndef f():\n    return X\n\n\nclass A:\n    @property\n    def g(self):\n        return 1\n\n"
       "    def h(self):\n        return 1\n")
LOC = Location(file="pkg/m.py", symbol="f", start_line=5, end_line=6, code="def f():\n    return X")
ORIG = {"pkg/m.py": SRC, "pkg/other.py": "Y = 2\n"}


def test_function_scope_rejects_module_level_edit_file_scope_accepts():
    e = [Edit("pkg/m.py", "X = 1", "X = 2")]
    assert apply_edits(e, [LOC], ORIG).reason == MATCH_OUTSIDE_LOCATION
    assert apply_edits(e, [LOC], ORIG, "function").reason == MATCH_OUTSIDE_LOCATION
    r = apply_edits(e, [LOC], ORIG, "file")
    assert r.valid and "X = 2\n" in r.new_files["pkg/m.py"]


def test_file_scope_still_rejects_other_files_and_requires_unique_match():
    assert apply_edits([Edit("pkg/other.py", "Y = 2", "Y = 3")], [LOC], ORIG, "file").reason == FILE_NOT_IN_LOCATIONS
    dup = [Edit("pkg/m.py", "        return 1", "        return 2")]
    r = apply_edits(dup, [LOC], ORIG, "file")
    assert r.reason == SEARCH_NOT_UNIQUE and r.failed_block == 1 and r.failed_first_line == "return 1"
    ok = [Edit("pkg/m.py", "    def h(self):\n        return 1", "    def h(self):\n        return 2")]
    assert apply_edits(ok, [LOC], ORIG, "file").valid


def test_file_scope_multiple_edits_and_line_shift():
    e = [Edit("pkg/m.py", "import os", "import os\nimport sys\nimport re"), Edit("pkg/m.py", "    def h(self):", "    def hh(self):")]
    r = apply_edits(e, [LOC], ORIG, "file")
    assert r.valid and "import re\n" in r.new_files["pkg/m.py"] and "def hh(self)" in r.new_files["pkg/m.py"]
    cand, res = build_candidate(e, [LOC], ORIG, "file")
    assert res.valid and cand.diff.count("+import") == 2


def test_function_scope_unchanged_and_invalid_scope():
    e = [Edit("pkg/m.py", "    return X", "    return X + 1")]
    assert apply_edits(e, [LOC], ORIG).valid
    with pytest.raises(ValueError):
        apply_edits(e, [LOC], ORIG, "repo")


def test_segments_and_class_headers():
    assert module_segments(SRC) == [(1, 4)]
    assert enclosing_class_headers(SRC, 12) == [(9, 9)]
    m = Location(file="pkg/m.py", symbol="A.g", start_line=10, end_line=12, code="")
    assert excerpt_segments(SRC, [m], 0) == [(1, 4), (9, 12)]
    assert excerpt_segments(SRC, [m], 40) == [(1, 15)]


def count_chars(t):
    return len(t)


def test_code_view_full_when_small():
    text, meta = build_code_view([LOC], ORIG, count_chars, 10_000, "FUNC", full_max=6000)
    assert text.startswith("### pkg/m.py (full file, 15 lines)") and SRC.rstrip("\n") in text
    assert meta["level"] == "as_specified" and not meta["truncated"] and meta["full_files"] == 1


def test_code_view_excerpt_when_file_large_and_reductions():
    big = "import os\n" + "".join(f"V{i} = {i}\n" for i in range(5)) + "\n\n" + "".join(
        f"def f{i}():\n    return {i}\n\n\n" for i in range(200))
    loc = Location(file="pkg/big.py", symbol="f100", start_line=0, end_line=0, code="")
    start = big.splitlines().index("def f100():") + 1
    loc = Location(file="pkg/big.py", symbol="f100", start_line=start, end_line=start + 1, code="def f100():\n    return 100")
    orig = {"pkg/big.py": big}
    text, meta = build_code_view([loc], orig, count_chars, 100_000, "FUNC", full_max=100)
    assert meta["level"] == "as_specified" and meta["full_files"] == 0
    assert "(excerpts of the file" in text and "lines 1-8" in text and "def f100():" in text and "def f150" not in text
    assert f"lines {start - 40}-{start + 41}" in text
    # 예산을 줄이면 창 축소 → 함수만
    t2, m2 = build_code_view([loc], orig, count_chars, 400, "FUNC", full_max=100)
    assert m2["truncated"] and m2["level"] in ("window_20", "window_10", "window_0", "no_module_statements", "header_30")
    t3, m3 = build_code_view([loc], orig, count_chars, 10, "FUNC", full_max=100)
    assert t3 == "FUNC" and m3["level"] == "function_only" and m3["over_budget"] is False


def test_code_view_full_to_excerpt_step():
    big = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(100))
    loc = Location(file="pkg/b.py", symbol="f1", start_line=5, end_line=6, code="def f1():\n    return 1")
    text, meta = build_code_view([loc], {"pkg/b.py": big}, count_chars, len(big) - 10, "FUNC", full_max=10**9)
    assert meta["level"] == "full_to_excerpt" and meta["truncated"] and "(excerpts of the file" in text


def test_prompts_scope_text():
    p0 = prompts.init_prompt("issue", [LOC], 3)
    p2 = prompts.init_prompt("issue", [LOC], 3, "CODE VIEW", "file")
    assert "Only edit inside the functions shown above" in p0 and "CODE VIEW" not in p0
    assert "CODE VIEW" in p2 and "Only edit the files shown above" in p2 and "inside the functions" not in p2
    assert prompts.init_prompt("issue", [LOC], 3, None, "function") == p0
    n = prompts.init_retry_notice([{"patch_no": 1, "reason": "search_not_unique", "block": 2, "first_line": "x", "file": "a.py"}],
                                  None, "file")
    assert "matches more than one place" in n and "exactly one place" in n
    assert "Files that are not shown" in prompts.search_not_found_detail(1, "x", "a.py", "file")
    assert "Functions that are not shown" in prompts.search_not_found_detail(1, "x", "a.py")
    c = prompts.crossover_prompt("i", [LOC], {"diff": "d"}, {"diff": "d"}, code="CV", scope="file")
    assert "CV" in c and "Only edit the files shown above" in c


def test_runner_rejects_bad_scope(tmp_path):
    import evopatch.runner as R
    from types import SimpleNamespace
    from evopatch.config import Config
    from evopatch.logs import JsonlLog
    from pathlib import Path
    cfg = Config.load(Path(__file__).resolve().parents[1] / "tests" / "data" / "test_config.yaml")
    d = cfg.as_dict()
    d["algorithm"]["edit_scope"] = "repo"
    llm = SimpleNamespace(run_id="t", model="m", call_log=JsonlLog(tmp_path / "c.jsonl"))
    with pytest.raises(ValueError):
        R.IssueRunner(cfg=Config(d), run_dir=tmp_path, run_id="t", llm=llm, executor=None, exec_specs={}, regression_pool={})
    d["algorithm"]["edit_scope"] = "file"
    assert R.IssueRunner(cfg=Config(d), run_dir=tmp_path, run_id="t", llm=llm, executor=None, exec_specs={},
                         regression_pool={}).edit_scope == "file"
    assert R.IssueRunner(cfg=cfg, run_dir=tmp_path, run_id="t", llm=llm, executor=None, exec_specs={},
                         regression_pool={}).edit_scope == "function"
