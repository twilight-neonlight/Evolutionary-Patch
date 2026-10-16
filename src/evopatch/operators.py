"""초기 생성, crossover, 유효성 검사·재요청, 변이 ."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from . import feedback as FB
from . import prompts
from .config import Config, sampling_params
from .errors import IssueTimeout
from . import promptdoc as D
from .edits import MATCH_OUTSIDE_LOCATION, SEARCH_NOT_FOUND, SEARCH_NOT_UNIQUE, build_candidate, edits_text, parse_blocks, parse_patches
from .llm_client import ContextBudgetExceeded, LLMClient
from .logs import JsonlLog
from .population import Individual
from .redact import identity
from .seeding import derive_seed
from .types import Candidate, Location

INIT_SHORTFALL = "init_shortfall"


@dataclass
class OpContext:
    cfg: Config
    llm: LLMClient
    instance_id: str
    run_index: int
    problem_statement: str
    locations: list[Location]
    originals: dict[str, str]  # 위치 파일의 base commit 원문
    validity_log: JsonlLog
    extra_log: dict = field(default_factory=dict)  # attempt_id 등
    gen0_info: dict = field(default_factory=dict)  # A6 세대 0 진단(호출별 결과, 실패 패치별 사유). 선택·종료와 무관
    deadline: float | None = None  # B6: 이슈 wall-clock 상한 (time.monotonic 기준)
    mask: Callable = identity  # B2: 프롬프트의 test_files 경로·파일명 마스커 (redact.TestFileMasker)
    feedback: str = "off"  # 2026-10-15 loop.feedback: off / mutation_only / both (feedback.py)
    repro_source: tuple | None = None  # (경로, 소스, 덧붙인 부분만인지). feedback이 off가 아닐 때만 채운다
    edit_scope: str = "function"  # algorithm.edit_scope: function(S0) / file(S2)
    code_view: str | None = None  # S2 코드 문맥(scope.build_code_view). None이면 위치 코드

    def masked(self, user: str, patch_texts=()) -> tuple[str, int]:
        """보내기 직전 프롬프트 전체를 가린다. 이슈 본문과 위치 코드·위치 파일 경로(base commit 원문)는 보호 구간.
        patch_texts(부모 diff·원문, 이전 답변, 실패 SEARCH 첫 줄)에서는 단독 파일명을 가리지 않는다(결정 2)."""
        hits = getattr(self.mask, "hits", None)
        before = sum(hits.values()) if hits is not None else 0
        protected = [self.problem_statement] + [l.code for l in self.locations] + [l.file for l in self.locations]
        if self.code_view is not None:  # S2: 위치 파일 원문(base commit)도 보호 구간
            protected.append(self.code_view)
        out = self.mask(user, protected, patch_texts)
        return out, (sum(hits.values()) - before if hits is not None else 0)

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise IssueTimeout("이슈 wall-clock 상한 초과")

    def seed(self, *parts) -> int:
        return derive_seed(self.instance_id, self.run_index, "llm", *parts)

    def messages(self, user: str) -> list[dict]:
        return [{"role": "system", "content": prompts.system()}, {"role": "user", "content": user}]

    def log_validity(self, **row) -> None:
        self.validity_log.append({"run_id": self.llm.run_id, "instance_id": self.instance_id,
                                  **self.extra_log, **row})


def _call(ctx: OpContext, user: str, *, stage: str, generation: int, max_tokens: int, seed: int,
          retry_index: int = 0, retry_of: str | None = None, extra: dict | None = None, patch_texts=()):
    ctx.check_deadline()
    sp = sampling_params(ctx.cfg, stage, retry_of)
    sections = D.end_record()
    user, n_masked = ctx.masked(user, patch_texts)
    msgs = ctx.messages(user)
    extra = {**(extra or {}), "prompt_mask_hits": n_masked, "prompt_sections": sections,
             "prompt_sha256": hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()}
    D.begin_record()  # 다음 호출의 구간 기록 시작
    return ctx.llm.chat(msgs, stage=stage, instance_id=ctx.instance_id, generation=generation,
                        max_tokens=max_tokens, seed=seed, retry_index=retry_index, retry_of=retry_of,
                        extra_log={**ctx.extra_log, **(extra or {})}, **sp)


def _make(ctx, edits, *, cid, gen, idx, op, parents) -> Individual:
    cand, res = build_candidate(edits, ctx.locations, ctx.originals, ctx.edit_scope)
    meta = {"files_inferred": res.files_inferred, "normalized_match_edits": res.normalized_matches,
            "reindented_edits": res.reindented}
    if not res.valid:
        meta["invalid_detail"] = res.detail
        if res.reason == SEARCH_NOT_FOUND:  # D-2-1: 재시도 안내에 블록 번호·첫 줄·지시를 넣는다
            meta["invalid_detail"] = prompts.search_not_found_detail(res.failed_block, res.failed_first_line, res.detail,
                                                                     ctx.edit_scope)
        if res.reason in (SEARCH_NOT_FOUND, MATCH_OUTSIDE_LOCATION, SEARCH_NOT_UNIQUE):  # init 재요청 안내용 블록 번호·첫 줄
            meta["failed_block"], meta["failed_first_line"] = res.failed_block, res.failed_first_line
    return Individual(cid=cid, gen_born=gen, idx=idx, op=op, parent_ids=parents, candidate=cand,
                      valid=res.valid, invalid_reason=res.reason, raw=edits_text(edits), meta=meta)


def _init_max_tokens(ctx: OpContext, user: str, k: int) -> tuple[int, dict]:
    """4.7·A: min(context_window − 입력 − init_margin, k × init_per_patch). 적용된 상한과 컨텍스트 때문에 줄었는지 반환."""
    n_in = ctx.llm.count_tokens(ctx.messages(user))
    per = ctx.cfg.require("llm.max_tokens.init_per_patch")
    margin = ctx.cfg.require("llm.max_tokens.init_margin")
    by_formula = k * per
    by_context = ctx.cfg.require("llm.context_window") - n_in - margin
    mt = min(by_formula, by_context)
    if mt <= 0:
        raise ContextBudgetExceeded(n_in, by_formula, ctx.cfg.require("llm.context_window"))
    return mt, {"max_tokens_by_formula": by_formula, "max_tokens_by_context": by_context,
                "max_tokens_capped_by_context": by_context < by_formula}


def initial_population(ctx: OpContext) -> list[Individual]:
    """세대 0: 한 응답에 N개 (4.2). 유효 패치가 N개 미만이면 부족분만 최대 2회 재요청(A4).
    "유효"는 파싱 + 유효성 검사(함수 범위 매칭·문법) 통과로 본다."""
    D.begin_record()
    n = ctx.cfg.require("algorithm.N")
    max_retries = ctx.cfg.require("algorithm.init_shortfall_max_retries")
    valid: list[Individual] = []
    invalid: list[Individual] = []
    attempt = 0
    need = n
    failures: list[dict] = []  # 직전 응답의 무효 패치(모든 사유) — 재요청 안내용
    missing: dict | None = None  # 직전 응답에서 요청보다 적게 읽힌 패치(truncated / parse_failure)
    ctx.gen0_info = {"requested": n, "calls": [], "failed_patches": []}
    while need > 0 and attempt <= max_retries:
        user = prompts.init_prompt(ctx.problem_statement, ctx.locations, need, ctx.code_view, ctx.edit_scope)
        if attempt > 0:
            user += prompts.init_retry_notice(failures, missing, ctx.edit_scope)
            cap_extra = {"retry_notice": {"failures": [{k: f.get(k) for k in ("patch_no", "reason", "block", "first_line")}
                                                       for f in failures], "missing": missing}}
        else:
            cap_extra = {}
        mt, cap = _init_max_tokens(ctx, user, need)
        cap = {**cap, **cap_extra}
        if attempt == 0:
            res = _call(ctx, user, stage="init", generation=0, max_tokens=mt, seed=ctx.seed(0, "init", 0), extra=cap)
        else:
            res = _call(ctx, user, stage="retry", retry_of="init", generation=0, max_tokens=mt,
                        seed=ctx.seed(0, "init", attempt), retry_index=attempt, extra=cap,
                        patch_texts=[f["first_line"] for f in failures if f.get("first_line")])
        patches = parse_patches(res.text)
        got = 0
        failures = []
        truncated = res.finish_reason == "length"
        for j, edits in enumerate(patches):
            ind = _make(ctx, edits, cid="", gen=0, idx=0, op="init", parents=())
            ctx.log_validity(generation=0, stage="init", attempt=attempt, patch_no=j + 1, valid=ind.valid,
                             reason=ind.invalid_reason, detail=ind.meta.get("invalid_detail"),
                             files_inferred=ind.meta.get("files_inferred", 0),
                             normalized_match_edits=ind.meta["normalized_match_edits"],
                             reindented_edits=ind.meta["reindented_edits"],
                             n_edits=len(edits), finish_reason=res.finish_reason, call_id=res.log_row.get("call_id"))
            (valid if ind.valid else invalid).append(ind)
            got += ind.valid
            if not ind.valid:
                fb = ind.meta.get("failed_block")
                failures.append({"patch_no": j + 1, "reason": ind.invalid_reason, "block": fb,
                                 "first_line": ind.meta.get("failed_first_line"),
                                 "file": (edits[fb - 1].file or None) if fb else None,
                                 "detail": ind.meta.get("invalid_detail") if ind.invalid_reason != SEARCH_NOT_FOUND else None})
                ctx.gen0_info["failed_patches"].append({
                    "attempt": attempt, "patch_no": j + 1, "reason": ind.invalid_reason,
                    "last_patch_of_truncated_response": truncated and j == len(patches) - 1,
                    "failed_block": fb, "failed_search": edits[fb - 1].search if fb else None,
                    "failed_file": edits[fb - 1].file if fb else None})
        ctx.log_validity(generation=0, stage="init_summary", attempt=attempt, requested=need,
                         parsed=len(patches), valid=got, finish_reason=res.finish_reason)
        ctx.gen0_info["calls"].append({
            "attempt": attempt, "stage": "init" if attempt == 0 else "retry", "requested": need,
            "parsed": len(patches), "valid": got, "finish_reason": res.finish_reason, "max_tokens": mt,
            "max_tokens_capped_by_context": cap["max_tokens_capped_by_context"],
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            # 요청 수보다 적게 파싱된 패치: 길이 상한에 걸린 응답이면 truncated, 아니면 parse_failure
            "missing_patches": max(0, need - len(patches)),
            "missing_reason": ("truncated" if truncated else "parse_failure") if need > len(patches) else None})
        missing = ({"reason": "truncated" if truncated else "parse_failure", "parsed": len(patches), "requested": need}
                   if need > len(patches) else None)
        need = n - len(valid)
        attempt += 1

    pop = valid[:n]
    for ind in invalid:  # 부족분: 파싱됐지만 무효인 패치 → 그래도 모자라면 빈 무효 후보
        if len(pop) >= n:
            break
        pop.append(ind)
    while len(pop) < n:
        pop.append(Individual(cid="", gen_born=0, idx=0, op="init", parent_ids=(),
                              candidate=Candidate(edits=(), patch_hash="EMPTY", diff=""),
                              valid=False, invalid_reason=INIT_SHORTFALL))
    for i, ind in enumerate(pop):
        ind.cid, ind.idx = f"g0c{i}", i
    return pop


def _patch_texts(a: Individual, b: Individual) -> list[str]:
    """결정 2: 단독 파일명 마스킹을 하지 않는 패치 텍스트(부모 diff, 무효 부모 편집 원문)."""
    return [t for p in (a, b) for t in (p.candidate.diff, p.raw) if t]


def _retry_texts(previous_output: str, ind: Individual) -> list[str]:
    """재요청에 붙는 이전 답변(retry_suffix와 같은 꼬리)과 실패 SEARCH 첫 줄."""
    return [previous_output[-6000:]] + ([ind.meta["failed_first_line"]] if ind.meta.get("failed_first_line") else [])


def _fb_prompt(ctx: OpContext, make, a: Individual, b: Individual, suffix: str = "") -> tuple[str, dict]:
    """2026-10-15 피드백 프롬프트: 부모별 확장 요약 + 재현 테스트 블록을 길이 한도 안에 맞춘다(feedback.fit). (user, 기록)."""
    va, vb = a.view(), b.view()

    def build(level):
        ta, ia, ca = FB.exec_summary(va, level)
        tb, ib, cb = FB.exec_summary(vb, level)
        src, cs = FB.source_block(ctx.repro_source, level)
        items, capped = ia | ib | ({"repro_test_source"} if src else set()), ca | cb | ({"repro_test_source"} if cs else set())
        return make(ctx.problem_statement, ctx.locations, va, vb, {"a": ta, "b": tb, "source": src},
                    ctx.code_view, ctx.edit_scope) + suffix, items, capped

    user, meta = FB.fit(build, prompts.system())
    # 2026-10-16 12:30: 호출마다 회귀 오류 내용이 들어간 테스트 수, 추출 실패 수·사유, 가림 횟수
    rf = [(v.get("exec") or {}).get("reg_failure") for v in (va, vb)]
    rf = [r for r in rf if r]
    reasons, scrub = {}, {}
    for r in rf:
        for k, n in (r.get("fail_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + n
        for k, n in (r.get("scrub") or {}).items():
            scrub[k] = scrub.get(k, 0) + n
    used = "regression_failure_messages" in meta["feedback_items"]
    blocks = []
    if used:  # 보고서 예시용: 실제로 들어간 회귀 결과 블록(프롬프트와 같은 가림 적용 후)
        lv = FB.LEVELS.index(meta["feedback_reduction"])
        for v in (va, vb):
            t = FB.exec_summary(v, lv)[0]
            if D.prefix("RESULTS.REGRESSION_FAILURE") in t:
                blk = t[t.index(D.prefix("RESULTS.REGRESSION_FAILED_COUNT")):]
                blocks.append(ctx.mask(blk) if ctx.mask else blk)
    return user, {**meta, "feedback_mode": ctx.feedback, "feedback_reg_blocks": blocks,
                  "feedback_reg_extracted": sum(len(r.get("messages") or []) for r in rf),
                  "feedback_reg_msgs": sum(len(r.get("messages") or []) for r in rf) if used else 0,
                  "feedback_reg_n_failed": sum(r.get("n_failed", 0) for r in rf),
                  "feedback_reg_extract_fail": sum(reasons.values()), "feedback_reg_fail_reasons": reasons,
                  "feedback_reg_scrub": scrub}


def crossover(ctx: OpContext, a: Individual, b: Individual, *, gen: int, idx: int) -> Individual:
    """부모 쌍마다 개별 호출(2절). 무효면 실패 정보를 붙여 최대 2회 재요청, 3회 모두 실패하면 무효(최하위).
    2026-10-15: loop.feedback=both이면 부모별 실행 요약을 넣는다(off·mutation_only는 기존 프롬프트 그대로)."""
    D.begin_record()
    use_fb = ctx.feedback == "both"
    fbmeta = {"feedback_mode": ctx.feedback, "feedback_items": []}
    if use_fb:
        base, fbmeta = _fb_prompt(ctx, prompts.crossover_prompt, a, b)
    else:
        base = prompts.crossover_prompt(ctx.problem_statement, ctx.locations, a.view(), b.view(),
                                        code=ctx.code_view, scope=ctx.edit_scope)
    max_retries = ctx.cfg.require("algorithm.crossover_max_retries")
    mt = ctx.cfg.require("llm.max_tokens.single_patch")
    user = base
    parent_texts = _patch_texts(a, b)
    patch_texts = parent_texts
    ind = None
    for k in range(max_retries + 1):
        stage = "crossover" if k == 0 else "retry"
        res = _call(ctx, user, stage=stage, retry_of=None if k == 0 else "crossover", generation=gen,
                    max_tokens=mt, seed=ctx.seed(gen, "crossover", idx, k), retry_index=k, patch_texts=patch_texts,
                    extra=fbmeta)
        ind = _make(ctx, parse_blocks(res.text), cid=f"g{gen}c{idx}", gen=gen, idx=idx, op="crossover",
                    parents=(a.cid, b.cid))
        ind.meta["crossover_attempts"] = k + 1
        ctx.log_validity(generation=gen, stage="crossover", child=idx, attempt=k, valid=ind.valid,
                         reason=ind.invalid_reason, detail=ind.meta.get("invalid_detail"),
                         files_inferred=ind.meta.get("files_inferred", 0),
                         normalized_match_edits=ind.meta["normalized_match_edits"],
                         reindented_edits=ind.meta["reindented_edits"],
                         finish_reason=res.finish_reason, call_id=res.log_row.get("call_id"))
        if ind.valid:
            return ind
        suffix = prompts.retry_suffix(res.text, ind.invalid_reason or "", ind.meta.get("invalid_detail", ""))
        if use_fb:  # 재요청 꼬리까지 넣고 다시 길이 검사(피드백 항목만 줄임)
            user, fbmeta = _fb_prompt(ctx, prompts.crossover_prompt, a, b, suffix)
        else:
            user = base + suffix
        patch_texts = parent_texts + _retry_texts(res.text, ind)
    return ind


def mutate(ctx: OpContext, child: Individual, a: Individual, b: Individual, *, gen: int, idx: int) -> Individual:
    """변이(2절): 입력은 두 부모 diff와 보관된 두 부모 실행 결과. 무효 부모는 '실행 결과 없음'(Q4b).
    무효면 mutation_max_retries회(기본 0회, A5)까지 재요청하고, 그래도 실패하면 변이 전 crossover 자식으로 되돌린다."""
    D.begin_record()
    use_fb = ctx.feedback in ("mutation_only", "both")
    fbmeta = {"feedback_mode": ctx.feedback, "feedback_items": []}
    if use_fb:
        base, fbmeta = _fb_prompt(ctx, prompts.mutation_prompt, a, b)
    else:
        base = prompts.mutation_prompt(ctx.problem_statement, ctx.locations, a.view(), b.view(),
                                       code=ctx.code_view, scope=ctx.edit_scope)
    max_retries = ctx.cfg.require("algorithm.mutation_max_retries")
    mt = ctx.cfg.require("llm.max_tokens.single_patch")
    user = base
    parent_texts = _patch_texts(a, b)
    patch_texts = parent_texts
    for k in range(max_retries + 1):
        stage = "mutation" if k == 0 else "retry"
        res = _call(ctx, user, stage=stage, retry_of=None if k == 0 else "mutation", generation=gen,
                    max_tokens=mt, seed=ctx.seed(gen, "mutation", idx, k), retry_index=k, patch_texts=patch_texts,
                    extra=fbmeta)
        m = _make(ctx, parse_blocks(res.text), cid=child.cid, gen=gen, idx=idx, op="mutation",
                  parents=(a.cid, b.cid))
        ctx.log_validity(generation=gen, stage="mutation", child=idx, attempt=k, valid=m.valid,
                         reason=m.invalid_reason, detail=m.meta.get("invalid_detail"),
                         files_inferred=m.meta.get("files_inferred", 0),
                         normalized_match_edits=m.meta["normalized_match_edits"],
                         reindented_edits=m.meta["reindented_edits"],
                         finish_reason=res.finish_reason, call_id=res.log_row.get("call_id"))
        if m.valid:
            m.meta.update(mutation_attempts=k + 1, crossover_attempts=child.meta.get("crossover_attempts"))
            return m
        suffix = prompts.retry_suffix(res.text, m.invalid_reason or "", m.meta.get("invalid_detail", ""))
        if use_fb:
            user, fbmeta = _fb_prompt(ctx, prompts.mutation_prompt, a, b, suffix)
        else:
            user = base + suffix
        patch_texts = parent_texts + _retry_texts(res.text, m)
    child.meta.update(mutation_attempts=max_retries + 1, mutation_reverted=True)
    return child
