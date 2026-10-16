"""LLM 프롬프트 조립(초기 생성·재요청·교배·변이). 문장은 prompts/prompts.txt(promptdoc)에만 있다.

여기서는 구간을 고르고({{자리표시자}} 채우기), 정해진 순서로 잇기만 한다. 마스킹·길이 검사는 operators/feedback이 한다.
모델 입력에는 허용 필드(이슈 본문)와 위치 산출물, 재현·회귀 실행 결과만 들어간다. test_files는 넣지 않는다.
수정 범위(scope): "function"(S0, 위치 함수 범위) | "file"(S2, 위치 파일 전체). S2는 위치 코드 대신 code(S2 코드 문맥)를 쓴다.
"""
from __future__ import annotations

from . import promptdoc as D
from .types import Location


def system() -> str:
    return D.text("SYSTEM")


def edit_format(scope: str = "function") -> str:
    return D.text("EDIT_FORMAT.FILE_SCOPE" if scope == "file" else "EDIT_FORMAT")


def code_context(locations: list[Location]) -> str:
    return D.text("LOCATION_CODE.SEPARATOR").join(
        D.fill("LOCATION_CODE.ITEM", FILE=l.file, SYMBOL=l.symbol, START_LINE=l.start_line, END_LINE=l.end_line, CODE=l.code)
        for l in locations)


def init_prompt(problem_statement: str, locations: list[Location], n: int, code: str | None = None,
                scope: str = "function") -> str:
    return D.fill("INIT_USER.FILE_SCOPE" if scope == "file" else "INIT_USER", ISSUE=problem_statement,
                  LOCATION_CODE=code if code is not None else code_context(locations), N=n)


def _parent(label_section: str, diff: str | None, raw: str | None) -> str:
    label = D.text(label_section)
    if diff:
        return D.fill("PARENT.VALID", LABEL=label, DIFF=diff)
    return D.fill("PARENT.INVALID", LABEL=label, RAW=raw or D.text("PARENT.INVALID_EMPTY"))


def _source_part(fb: dict) -> str:
    return D.fill("SOURCE_PART", SOURCE=fb["source"]) if fb.get("source") else ""


def _evolution_prompt(op: str, problem_statement: str, locations: list[Location], a: dict, b: dict,
                      fb: dict | None, code: str | None, scope: str) -> str:
    """op: CROSSOVER | MUTATION. fb가 있으면 피드백판(부모별 확장 실행 요약 + 재현 테스트 블록), 없으면 기존판."""
    common = dict(ISSUE=problem_statement, LOCATION_CODE=code if code is not None else code_context(locations),
                  PARENT_A=_parent("PARENT.LABEL_A", a.get("diff"), a.get("raw")),
                  PARENT_B=_parent("PARENT.LABEL_B", b.get("diff"), b.get("raw")), EDIT_FORMAT=edit_format(scope))
    if fb is not None:
        return D.fill(f"{op}_USER.FEEDBACK", RESULTS_A=fb["a"], RESULTS_B=fb["b"], SOURCE_PART=_source_part(fb), **common)
    if op == "CROSSOVER":
        return D.fill("CROSSOVER_USER", **common)
    return D.fill("MUTATION_USER", RESULTS_A=_exec_summary(a), RESULTS_B=_exec_summary(b), **common)


def crossover_prompt(problem_statement: str, locations: list[Location], a: dict, b: dict, fb: dict | None = None,
                     code: str | None = None, scope: str = "function") -> str:
    """fb(loop.feedback=both): {"a": 부모 A 실행 요약, "b": 부모 B 실행 요약, "source": 재현 테스트 블록(빈 문자열 가능)}."""
    return _evolution_prompt("CROSSOVER", problem_statement, locations, a, b, fb, code, scope)


def mutation_prompt(problem_statement: str, locations: list[Location], a: dict, b: dict, fb: dict | None = None,
                    code: str | None = None, scope: str = "function") -> str:
    return _evolution_prompt("MUTATION", problem_statement, locations, a, b, fb, code, scope)


def _exec_summary(p: dict) -> str:
    """피드백을 끈(off) 변이 프롬프트의 부모 실행 요약(회귀는 개수만)."""
    ex = p.get("exec")
    if not p.get("valid") or ex is None:
        return D.text("RESULTS.INVALID")
    lines = [D.fill("RESULTS.REPRO", RESULT=D.text("RESULTS.PASSED" if ex["repro_pass"] else "RESULTS.FAILED"))]
    if ex.get("repro_statuses"):
        lines.append(D.fill("RESULTS.STATUS_LINE", STATUSES=D.text("RESULTS.STATUS_SEPARATOR").join(
            D.fill("RESULTS.STATUS_ITEM", TEST=k, STATUS=v) for k, v in list(ex["repro_statuses"].items())[:20])))
    if ex.get("repro_log_tail"):
        lines.append(D.fill("RESULTS.OUTPUT_TAIL", TAIL=ex["repro_log_tail"]))
    if ex.get("no_regression_signal") or not ex.get("reg_total"):
        lines.append(D.text("RESULTS.REGRESSION_NOT_AVAILABLE"))
    else:
        lines.append(D.fill("RESULTS.REGRESSION_COUNTS", PASSED=ex["reg_passed"], TOTAL=ex["reg_total"],
                            FAILED=ex["reg_total"] - ex["reg_passed"]))
    if ex.get("timed_out"):
        lines.append(D.text("RESULTS.TIMED_OUT"))
    return "\n".join(lines)


def _where(file: str | None) -> str:
    return f" in {file}" if file else ""


def _search_rule(scope: str) -> str:
    return D.text("INIT_RETRY_NOTICE.search_not_found.RULE.FILE_SCOPE" if scope == "file"
                  else "INIT_RETRY_NOTICE.search_not_found.RULE")


def _search_line(block, first_line, file) -> str:
    return D.fill("INIT_RETRY_NOTICE.search_not_found.LINE", BLOCK=block, IN_FILE=_where(file), FIRST_LINE=first_line)


def search_not_found_detail(block: int | None, first_line: str | None, file: str | None, scope: str = "function") -> str:
    """search_not_found 재시도 안내(블록 번호, 첫 줄, 제공 코드와 불일치, 복사 지시)."""
    return D.fill("SEARCH_NOT_FOUND_DETAIL", LINE=_search_line(block, first_line, file), RULE=_search_rule(scope))


# 재요청 규칙이 있는 사유(규칙 문장은 문서의 INIT_RETRY_NOTICE.<사유>.RULE). search_not_unique는 S2에서만 생긴다.
RETRY_REASONS = ("search_not_found", "match_outside_location", "file_not_in_locations", "missing_file_path", "empty_search",
                 "empty_diff", "syntax_error", "no_edits", "truncated", "parse_failure", "search_not_unique")


def _retry_rule(reason: str, scope: str) -> str | None:
    if reason == "search_not_found":
        return _search_rule(scope)
    if reason not in RETRY_REASONS or (reason == "search_not_unique" and scope != "file"):
        return None
    return D.text(f"INIT_RETRY_NOTICE.{reason}.RULE")


def _retry_line(f: dict) -> str:
    """패치 하나의 실패 내용. f: patch_no, reason, block, first_line, file, detail."""
    r, b, fl, path = f["reason"], f.get("block"), f.get("first_line"), f.get("file")
    if r == "search_not_found":
        what = _search_line(b, fl, path)
    elif r in ("match_outside_location", "search_not_unique"):
        what = D.fill(f"INIT_RETRY_NOTICE.{r}.LINE", BLOCK=b, IN_FILE=_where(path), FIRST_LINE=fl)
    elif r == "file_not_in_locations":
        what = D.fill("INIT_RETRY_NOTICE.file_not_in_locations.LINE", FILE_OR_DETAIL=f.get("detail") or path)
    elif r in ("missing_file_path", "empty_diff", "no_edits"):
        what = D.text(f"INIT_RETRY_NOTICE.{r}.LINE")
    elif r == "empty_search":
        what = D.fill("INIT_RETRY_NOTICE.empty_search.LINE", IN_FILE=_where(path))
    elif r == "syntax_error":
        what = D.fill("INIT_RETRY_NOTICE.syntax_error.LINE", DETAIL_160=(f.get("detail") or "").strip()[:160])
    else:
        what = D.fill("INIT_RETRY_NOTICE.OTHER.LINE", REASON=r)
    return D.fill("INIT_RETRY_NOTICE.PATCH_LINE", PATCH_NO=f["patch_no"], WHAT=what)


def init_retry_notice(failures: list[dict], missing: dict | None = None, scope: str = "function") -> str:
    """0세대 부족분 재요청 안내(모든 실패 사유). 이전 답변은 붙이지 않는다.
    failures: 직전 응답에서 무효였던 패치(사유별 줄). missing: 요청보다 적게 읽힌 경우
    {"reason": truncated|parse_failure, "parsed": int, "requested": int}."""
    lines = [_retry_line(f) for f in failures]
    reasons = [f["reason"] for f in failures]
    if missing:
        lines.append(D.fill(f"INIT_RETRY_NOTICE.{missing['reason']}.LINE", PARSED=missing["parsed"],
                            REQUESTED=missing["requested"]))
        reasons.append(missing["reason"])
    rules = [x for x in (_retry_rule(r, scope) for r in dict.fromkeys(reasons)) if x]
    return (D.text("INIT_RETRY_NOTICE.HEADER")
            + ("\n".join(lines) if lines else D.text("INIT_RETRY_NOTICE.NO_PROBLEM_LINES"))
            + (D.text("INIT_RETRY_NOTICE.RULES_HEADER") + "\n".join(D.fill("INIT_RETRY_NOTICE.RULE_ITEM", RULE=x) for x in rules)
               if rules else ""))


def retry_suffix(previous_output: str, reason: str, detail: str) -> str:
    """교배·변이 재요청 꼬리: 직전 답변(끝 6,000자)과 유효성 실패 원인."""
    return D.fill("RETRY_SUFFIX", REASON=reason,
                  DETAIL_PART=D.fill("RETRY_SUFFIX.DETAIL_PART", DETAIL=detail) if detail else "",
                  PREVIOUS=previous_output[-6000:])
