"""루프 피드백(2026-10-15 사용자 최종 통합 지시 1): 교배·변이 프롬프트에 넣는 부모별 실행 요약.

설정 loop.feedback: off(기존과 바이트 단위로 같은 프롬프트) / mutation_only(변이만 확장 요약) / both(교배·변이 모두).
부모별 항목(모두 문자 상한): 재현 통과/실패, 테스트별 상태(최대 20개), 재현 출력 마지막 40줄, 회귀 결과.
호출당 한 번: 재현 테스트 소스(상한 2,000토큰).
회귀 결과(2026-10-16 12:30 최종 후보 지시 5): 전체·통과 개수는 넣지 않는다.
  실패 있음 → 통과하지 못한 회귀 테스트 수 + 오류 내용 최대 3개(regmsg.extract가
  테스트 ID·이름을 지우고 테스트 파일 프레임을 바꾼 것, 부모당 약 1,500자). 실패 없음·회귀 없음은 각각의 문장.
  문장은 prompts/prompts.txt의 RESULTS.* 구간. 회귀 테스트 이름·경로는 넣지 않는다(test_regression_names_never_reach_prompts).
입력 길이 검사: 한도 28,672토큰(문자 수 / 3.87로 추정). 넘으면 피드백 항목만 아래 순서로 줄이고 이슈·코드는 줄이지 않는다.
  1 회귀 오류 내용 빼기("N개" 줄은 남김) → 2 재현 테스트 소스 빼기 → 3 재현 출력 10줄로 → 4 재현 출력 빼기
  → 5 테스트별 상태 빼기 → 그래도 넘으면 over_limit 기록(그대로 보냄).
"""
from __future__ import annotations

import math

from . import promptdoc as D

MODES = ("off", "mutation_only", "both")
CHARS_PER_TOKEN = 3.87
LIMIT_TOKENS = 28672  # 32,768 − single_patch 4,096
STATUS_MAX, STATUS_CHARS = 20, 1500
TAIL_LINES, TAIL_CHARS = 40, 4000
SOURCE_TOKENS = 2000
SOURCE_CHARS = int(SOURCE_TOKENS * CHARS_PER_TOKEN)
LEVELS = ("full", "no_reg_messages", "no_source", "tail_10", "no_tail", "no_statuses")

def est_tokens(*texts: str) -> int:
    return math.ceil(sum(len(t) for t in texts) / CHARS_PER_TOKEN)


def _cap(text: str, n: int, keep_tail: bool = False) -> tuple[str, bool]:
    if len(text) <= n:
        return text, False
    mark = D.text("RESULTS.TRUNCATED")
    return (mark + "\n" + text[-n:]) if keep_tail else (text[:n] + "\n" + mark), True


def exec_summary(p: dict, level: int = 0) -> tuple[str, set[str], set[str]]:
    """부모 하나의 확장 실행 요약. (텍스트, 들어간 항목, 문자 상한으로 잘린 항목). level은 LEVELS 순번."""
    ex = p.get("exec")
    if not p.get("valid") or ex is None:
        return D.text("RESULTS.INVALID"), set(), set()
    items, capped = set(), set()
    lines = [D.fill("RESULTS.REPRO", RESULT=D.text("RESULTS.PASSED" if ex["repro_pass"] else "RESULTS.FAILED"))]
    items.add("repro_result")
    st = list((ex.get("repro_statuses") or {}).items())
    if st and level < 5:
        s, c = _cap(D.text("RESULTS.STATUS_SEPARATOR").join(D.fill("RESULTS.STATUS_ITEM", TEST=k, STATUS=v)
                                                             for k, v in st[:STATUS_MAX]), STATUS_CHARS)
        more = D.fill("RESULTS.STATUS_MORE", SHOWN=STATUS_MAX, TOTAL=len(st)) if len(st) > STATUS_MAX else ""
        lines.append(D.fill("RESULTS.STATUS_LINE", STATUSES=s + more))
        items.add("repro_statuses")
        if c or len(st) > STATUS_MAX:
            capped.add("repro_statuses")
    tail = ex.get("repro_log_tail") or ""
    if tail and level < 4:
        tl = tail.splitlines()
        n = TAIL_LINES if level < 3 else 10
        if len(tl) > n:
            capped.add("repro_output_tail")
        t, c = _cap("\n".join(tl[-n:]), TAIL_CHARS, keep_tail=True)
        lines.append(D.fill("RESULTS.OUTPUT_TAIL", TAIL=t))
        items.add("repro_output_tail")
        if c:
            capped.add("repro_output_tail")
    if ex.get("no_regression_signal") or not ex.get("reg_total"):
        lines.append(D.text("RESULTS.REGRESSION_NONE"))
        items.add("regression_result")
    else:
        nf = ex["reg_total"] - ex["reg_passed"]
        items.add("regression_result")
        if nf == 0:
            lines.append(D.text("RESULTS.REGRESSION_NO_FAILURES"))
        else:
            lines.append(D.fill("RESULTS.REGRESSION_FAILED_COUNT", N=nf))
            msgs = (ex.get("reg_failure") or {}).get("messages") or []
            if msgs and level < 1:
                for k, m in enumerate(msgs, 1):
                    lines.append(D.fill("RESULTS.REGRESSION_FAILURE", K=k, MESSAGE=m))
                items.add("regression_failure_messages")
                if any(m.startswith(D.text("RESULTS.TRUNCATED")) for m in msgs):
                    capped.add("regression_failure_messages")
    if ex.get("timed_out"):
        lines.append(D.text("RESULTS.TIMED_OUT"))
    return "\n".join(lines), items, capped


def source_block(repro_source: tuple[str, str, bool] | None, level: int = 0) -> tuple[str, bool]:
    """(블록, 잘림). repro_source = (경로, 소스, 덧붙인 부분만인지). level ≥ 2이면 빼고 빈 문자열."""
    if not repro_source or level >= 2:
        return "", False
    path, src, appended = repro_source
    s, c = _cap(src, SOURCE_CHARS)
    return D.fill("RESULTS.SOURCE.APPENDED" if appended else "RESULTS.SOURCE.SAVED", PATH=path, SOURCE=s), c


def fit(build, system: str, extra_chars: int = 0) -> tuple[str, dict]:
    """build(level) -> (user, items, capped). 추정 토큰이 한도 안에 들 때까지 level을 올린다. 기록 dict를 돌려준다."""
    for level in range(len(LEVELS)):
        user, items, capped = build(level)
        n = est_tokens(system, user) + math.ceil(extra_chars / CHARS_PER_TOKEN)
        if n <= LIMIT_TOKENS:
            break
    return user, {"feedback_items": sorted(items), "feedback_capped": sorted(capped),
                  "feedback_reduction": LEVELS[level], "feedback_truncated": bool(capped) or level > 0,
                  "feedback_est_tokens": n, "feedback_over_limit": n > LIMIT_TOKENS}
