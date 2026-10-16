"""프롬프트 고정 문서(prompts/prompts.txt)와 코드의 분리.

- src 안의 문자열 상수(독스트링 제외)에 문서 구간의 문장이 남아 있지 않다(자리표시자로 나눈 조각 중 12자 이상).
- 코드가 이름으로 부르는 구간은 문서에 모두 있다.
- 초기 생성 구간(SYSTEM, INIT_USER, LOCATION_CODE.*, INIT_RETRY_NOTICE.* 30개)은 고정본 v1과 바이트 단위로 같다
  (tests/data/initial_prompt_v1_section_sha256.json의 구간별 SHA-256).
- 줄바꿈·공백이 중요한 구간의 값.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

from evopatch import promptdoc as D

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "evopatch"
PH = re.compile(r"\{\{[A-Z_0-9]+\}\}")
# 응답 파서가 인식해야 하는 출력 형식 표시(프롬프트 문장이 아니라 파싱 규칙). 코드에 남는 것을 허용한다.
PARSER_MARKERS = {"<<<<<<< SEARCH", ">>>>>>> REPLACE"}


def string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            docs.add(id(n.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_doc_parses_and_has_sections():
    d = D.load()
    assert len(d.sections) >= 100 and d.sha256 == hashlib.sha256(d.path.read_bytes()).hexdigest()


def test_no_prompt_sentences_left_in_src():
    d = D.load()
    frags = set()
    for name, body in d.sections.items():
        for part in PH.split(body):
            for line in part.split("\n"):
                if len(line.strip()) >= 12 and line.strip() not in PARSER_MARKERS:
                    frags.add(line.strip())
    found = []
    for p in sorted(SRC.rglob("*.py")):
        consts = string_constants(p)
        for f in frags:
            if any(f in c for c in consts):
                found.append((p.relative_to(ROOT).as_posix(), f[:60]))
    assert not found, found[:10]


def test_section_names_used_in_code_exist():
    d = D.load()
    names = set()
    for p in SRC.rglob("*.py"):
        names |= set(re.findall(r'D\.(?:text|fill|prefix|has)\("([^"{]+)"', p.read_text()))
        names |= set(re.findall(r'promptdoc\.text\("([^"{]+)"', p.read_text()))
    missing = sorted(n for n in names if n not in d.sections)
    assert names and not missing, missing
    for reason in ("search_not_found", "match_outside_location", "file_not_in_locations", "missing_file_path", "empty_search",
                   "empty_diff", "syntax_error", "no_edits", "truncated", "parse_failure", "search_not_unique"):
        assert f"INIT_RETRY_NOTICE.{reason}.LINE" in d.sections and f"INIT_RETRY_NOTICE.{reason}.RULE" in d.sections
    for k in ("full", "grouped", "dirs", "dirs_depth2"):
        assert f"PREP.FILE_LIST.HEADER.{k}" in d.sections
    for k in ("full", "skeleton"):
        assert f"PREP.CODE_NOTE.{k}" in d.sections and f"PREP.CODE_FILE.{k}" in d.sections
    for op in ("CROSSOVER", "MUTATION"):
        assert f"{op}_USER" in d.sections and f"{op}_USER.FEEDBACK" in d.sections


def test_initial_sections_byte_identical_to_v1():
    d = D.load()
    want = json.loads((ROOT / "tests" / "data" / "initial_prompt_v1_section_sha256.json").read_text())
    assert len(want) == 30
    got = {k: hashlib.sha256(d.sections[k].encode()).hexdigest() for k in want}
    assert got == want


def test_edit_format_inline_in_init_matches_section():
    d = D.load()
    assert d.sections["EDIT_FORMAT"] in d.sections["INIT_USER"]
    assert d.sections["EDIT_FORMAT.FILE_SCOPE"] in d.sections["INIT_USER.FILE_SCOPE"]
    assert d.sections["INIT_USER"].replace(d.sections["EDIT_FORMAT"], d.sections["EDIT_FORMAT.FILE_SCOPE"]) == \
        d.sections["INIT_USER.FILE_SCOPE"]


def test_whitespace_sensitive_sections():
    d = D.load()
    want = {"LOCATION_CODE.SEPARATOR": "\n\n", "PREP.CODE_FILE.SEPARATOR": "\n\n", "S2.FILE_SEPARATOR": "\n\n",
            "SOURCE_PART": "\n\n{{SOURCE}}", "V2.BLOCK_PART": "{{BLOCK}}\n\n", "V2.LOG_OMITTED": "\n... [{{N}} lines omitted] ...\n",
            "V2.TEST_TRUNCATED": "\n# ... (truncated)", "PREP.FILE_LIST": "\nRepository {{HEADER}}:\n{{LIST}}\n",
            "RESULTS.STATUS_SEPARATOR": ", ", "RETRY_SUFFIX.DETAIL_PART": " ({{DETAIL}})",
            "REGRESSION_MESSAGE.FRAME_CODE": "\n    {{CODE}}", "SEARCH_NOT_FOUND_DETAIL": "{{LINE}} {{RULE}}",
            "INIT_RETRY_NOTICE.RULES_HEADER": "\nRules:\n"}
    assert {k: d.sections[k] for k in want} == want


def test_fill_is_single_pass_and_requires_all_values():
    D.load()
    assert D.fill("RESULTS.REPRO", RESULT="{{RESULT}}") == "Reproduction test: {{RESULT}}"
    try:
        D.fill("RESULTS.REPRO")
    except KeyError:
        pass
    else:
        raise AssertionError("값 없는 자리표시자")


def test_call_log_records_prompt_hash_sections_and_retry_info(make_ctx):
    """호출 기록: 렌더링된 프롬프트 SHA-256, 쓴 구간 이름, 재요청 안내에 쓴 정보(실패 블록 번호·첫 줄)."""
    import hashlib
    import json as _j
    from evopatch.operators import initial_population
    bad = "### PATCH 1\npkg/mod.py\n<<<<<<< SEARCH\n    nothing_here()\n=======\n    x = 1\n>>>>>>> REPLACE\n"
    good = "".join(f"### PATCH {i}\npkg/mod.py\n<<<<<<< SEARCH\n    total = a + b\n=======\n    total = a + b + {i}\n"
                   ">>>>>>> REPLACE\n" for i in range(1, 11))
    ctx, llm = make_ctx([bad, good])
    ctx.cfg._data["algorithm"]["N"] = 10
    initial_population(ctx)
    first, retry = llm.calls[0], llm.calls[1]
    for c in (first, retry):
        ex = c["extra_log"]
        assert ex["prompt_sha256"] == hashlib.sha256(_j.dumps(c["messages"], ensure_ascii=False).encode()).hexdigest()
        assert "INIT_USER" in ex["prompt_sections"] and "SYSTEM" in ex["prompt_sections"]
    assert "INIT_RETRY_NOTICE.HEADER" in retry["extra_log"]["prompt_sections"]
    f = retry["extra_log"]["retry_notice"]["failures"][0]
    assert f["reason"] == "search_not_found" and f["block"] == 1 and f["first_line"] == "nothing_here()"
