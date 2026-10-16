import subprocess

from evopatch import edits as E
from evopatch.types import Edit


def blk(search, replace, f="pkg/mod.py"):
    return f"{f}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_parse_patches_and_fences():
    text = ("### PATCH 1\n" + blk("    total = a + b", "    total = a - b") +
            "### PATCH 2\n```python\n" + blk("    return total", "    return -total") + "```\n" +
            "### PATCH 3\nno blocks here\n")
    ps = E.parse_patches(text)
    assert [len(p) for p in ps] == [1, 1, 0]
    assert ps[1][0] == Edit("pkg/mod.py", "    return total", "    return -total")


def test_exact_match_valid(originals, locations):
    r = E.apply_edits([Edit("pkg/mod.py", "    total = a + b", "    total = b + a")], locations, originals)
    assert r.valid and "total = b + a" in r.new_files["pkg/mod.py"]


def test_normalized_indent_match_reindents(originals, locations):
    r = E.apply_edits([Edit("pkg/mod.py", "if x:\n    return x", "if x is not None:\n    return x")], locations, originals)
    assert r.valid
    assert "        if x is not None:\n            return x\n" in r.new_files["pkg/mod.py"]
    assert (r.normalized_matches, r.reindented) == (1, 1)  # D-3: 정규화 매칭 + 들여쓰기 수정 기록


def test_match_kind_counters(originals, locations):  # D-3
    exact = E.apply_edits([Edit("pkg/mod.py", "    total = a + b", "    total = b + a")], locations, originals)
    assert exact.valid and (exact.normalized_matches, exact.reindented) == (0, 0)
    # 줄 끝 공백만 다름: 정규화 매칭이지만 들여쓰기가 같아 REPLACE는 그대로
    trail = E.apply_edits([Edit("pkg/mod.py", "    total = a + b   ", "    total = b + a")], locations, originals)
    assert trail.valid and (trail.normalized_matches, trail.reindented) == (1, 0)
    # 정규화 매칭 후 문법 오류여도 카운터는 남긴다
    bad = E.apply_edits([Edit("pkg/mod.py", "if x:\n    return x", "if x:\n  (")], locations, originals)
    assert bad.reason == E.SYNTAX_ERROR and bad.normalized_matches == 1


def test_match_outside_location_is_invalid(originals, locations):  # B7
    r = E.apply_edits([Edit("pkg/mod.py", "    return 1", "    return 2")], locations, originals)
    assert not r.valid and r.reason == E.MATCH_OUTSIDE_LOCATION


def test_not_found_syntax_file_empty(originals, locations):
    assert E.apply_edits([Edit("pkg/mod.py", "nope", "x")], locations, originals).reason == E.SEARCH_NOT_FOUND
    assert E.apply_edits([Edit("pkg/mod.py", "    total = a + b", "    total = (a +")], locations, originals).reason == E.SYNTAX_ERROR
    assert E.apply_edits([Edit("pkg/x.py", "a", "b")], locations, originals).reason == E.FILE_NOT_IN_LOCATIONS
    assert E.apply_edits([Edit("pkg/mod.py", "    return total", "    return total")], locations, originals).reason == E.EMPTY_DIFF
    assert E.apply_edits([], locations, originals).reason == E.NO_EDITS


def test_multi_edit_shifts_ranges(originals, locations):
    r = E.apply_edits([Edit("pkg/mod.py", "    total = a + b", "    a = int(a)\n    b = int(b)\n    total = a + b"),
                       Edit("pkg/mod.py", "        return None", "        return 0")], locations, originals)
    assert r.valid and "return 0" in r.new_files["pkg/mod.py"]


def test_diff_applies_with_git(tmp_path, originals, locations):
    orig = {"pkg/mod.py": originals["pkg/mod.py"].rstrip("\n")}  # 끝 개행 없는 파일
    cand, res = E.build_candidate([Edit("pkg/mod.py", "        return None", "        return 0")], locations, orig)
    assert res.valid and cand.diff.startswith("diff --git a/pkg/mod.py b/pkg/mod.py")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/mod.py").write_text(orig["pkg/mod.py"])
    (tmp_path / "p.diff").write_text(cand.diff)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", "--check", "p.diff"], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", "p.diff"], cwd=tmp_path, check=True)
    assert (tmp_path / "pkg/mod.py").read_text() == res.new_files["pkg/mod.py"]


def test_patch_hash_ignores_index_lines():
    d = "diff --git a/x b/x\nindex 1..2 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert E.patch_hash(d) == E.patch_hash(d.replace("index 1..2", "index 3..4"))


def test_path_line_variants():
    for head in ["pkg/mod.py", "### pkg/mod.py :: add (lines 4-7)", "File: `pkg/mod.py`", "a/pkg/mod.py", "./pkg/mod.py", "**pkg/mod.py**"]:
        es = E.parse_blocks(head + "\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n")
        assert es and es[0].file == "pkg/mod.py", head


def test_missing_path_inferred_when_unique(originals, locations):  # S2-b
    es = E.parse_blocks("Here's my fix:\n```python\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = b + a\n>>>>>>> REPLACE\n```\n")
    assert es[0].file == ""
    r = E.apply_edits(es, locations, originals)
    assert r.valid and r.files_inferred == 1


def test_missing_path_ambiguous_or_absent_invalid(originals, locations):
    # 'return' 한 줄은 add와 K.m 두 범위에서 매칭 → 유일하지 않음
    r = E.apply_edits([E.Edit("", "        return x", "        return 1")], locations, originals)
    assert r.valid  # K.m에만 있음
    r = E.apply_edits([E.Edit("", "return", "x")], locations, originals)
    assert not r.valid and r.reason == E.MISSING_FILE_PATH
    r = E.apply_edits([E.Edit("", "    return 1", "    return 2")], locations, originals)  # 범위 밖에만 있음
    assert not r.valid and r.reason == E.MISSING_FILE_PATH


def test_resolve_symbol_kinds():
    from evopatch.locate import find_symbol, resolve_symbol
    from tests.conftest import ORIG
    assert resolve_symbol(ORIG, "m.py", "add", allow_class=True)[1] == "function"
    assert resolve_symbol(ORIG, "m.py", "K.m", allow_class=True)[1] == "method"
    loc, kind = resolve_symbol(ORIG, "m.py", "K", allow_class=True)
    assert kind == "class" and loc.code.startswith("class K:") and loc.end_line == 18
    assert resolve_symbol(ORIG, "m.py", "K", allow_class=False) is None and find_symbol(ORIG, "m.py", "K") is None


def test_location_range_includes_decorators():  # 결정 3
    from evopatch.edits import apply_edits
    from evopatch.locate import resolve_symbol
    src = ("class K:\n    x = 1\n\n    @property\n    @other(1)\n    def m(self):\n        return 1\n\n"
           "@decorate\nclass D:\n    pass\n\n@cache\ndef f():\n    return 2\n")
    loc, kind = resolve_symbol(src, "a.py", "K.m", allow_class=True)
    assert kind == "method" and (loc.start_line, loc.end_line) == (4, 7) and loc.code.startswith("    @property")
    floc, _ = resolve_symbol(src, "a.py", "f", allow_class=True)
    assert (floc.start_line, floc.end_line) == (13, 15)
    cloc, _ = resolve_symbol(src, "a.py", "D", allow_class=True)  # 클래스는 class 줄부터(그대로)
    assert cloc.start_line == 10
    from evopatch.types import Edit
    e = Edit("a.py", "    @property\n    @other(1)\n    def m(self):", "    @other(1)\n    def m(self):")
    assert apply_edits([e], [loc], {"a.py": src}).valid
