"""8.2 관측 항목 집계 (생성 로그 + 채점 결과).

실행: .venv/bin/python -m evopatch.report --run-dir runs/<날짜>_<run_id>
출력: <run-dir>/report.json, report.md (매번 시각을 붙여 새 파일로 쓴다)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from .logs import JsonlLog
from .predictions import latest_issue_rows
from .runner import DONE_STATUSES


def _ratio(a, b):
    return round(a / b, 4) if b else None


def build_report(run_dir: Path) -> dict:
    issues_all = JsonlLog(run_dir / "issues.jsonl").read()
    latest = latest_issue_rows(JsonlLog(run_dir / "issues.jsonl"))
    final_attempts = {r["attempt_id"] for r in latest.values()}
    calls = JsonlLog(run_dir / "llm_calls.jsonl").read()
    cands = [c for c in JsonlLog(run_dir / "candidates.jsonl").read() if c.get("attempt_id") in final_attempts]
    validity = [v for v in JsonlLog(run_dir / "validity.jsonl").read() if v.get("attempt_id") in final_attempts]

    rep: dict = {"run_dir": str(run_dir), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "n_issues": len(latest), "status": dict(Counter(r["status"] for r in latest.values()))}

    # 이슈당 비용: 최종 시도 토큰만 (Q8b)
    per_issue = {}
    for iid, r in latest.items():
        per_issue[iid] = {"status": r["status"], "terminated_by": r.get("terminated_by"),
                          "generations_run": r.get("generations_run"), "wall_clock_s": r.get("wall_clock_s"),
                          "prompt_tokens": r["total_prompt_tokens"], "completion_tokens": r["total_completion_tokens"],
                          "tokens_by_stage": r.get("tokens_by_stage"), "final_r": r.get("final_r"),
                          "final_g": r.get("final_g"), "final_g_total": r.get("final_g_total"),
                          "prep_failure_reason": r.get("prep_failure_reason"),
                          "locations_requested": r.get("prep_locations_requested"),
                          "locations_found": r.get("prep_locations_found"),
                          "repro_on_base": r.get("repro_on_base"),
                          "regression_list_size": r.get("regression_list_size"),
                          "location_kinds": r.get("prep_location_kinds"),
                          "class_location_tokens": r.get("prep_class_location_tokens"),
                          "file_list_level": r.get("prep_file_list_level"),
                          "file_list_tokens": r.get("prep_file_list_tokens"),
                          "no_regression_signal": r.get("no_regression_signal")}
    # 세대별 유효 후보·실행 수, 빈 최종 패치, 전부 무효인 세대 (C, 보이기만 함)
    cand_by_issue = defaultdict(list)
    for c in cands:  # 위에서 최종 시도로 거른 후보 행(같은 필터)
        cand_by_issue[c["instance_id"]].append(c)
    for iid, r in latest.items():
        cs = cand_by_issue.get(iid, [])
        gens = sorted({c["generation"] for c in cs})
        vbg = r.get("valid_by_gen") or [sum(c["valid"] for c in cs if c["generation"] == g) for g in gens]
        ebg = r.get("executed_by_gen") or [sum(1 for c in cs if c["generation"] == g and c["valid"] and c["op"] != "elite")
                                           for g in gens]
        fp = run_dir / r["final_patch_path"] if r.get("final_patch_path") else None
        empty = r.get("final_patch_empty")
        if empty is None:
            empty = not (fp and fp.exists() and fp.read_text().strip())
        term = r.get("terminated_by")
        if r["status"] == "ok" and vbg and not any(vbg):
            term = "all_invalid"
        per_issue[iid].update(valid_by_gen=vbg, executed_by_gen=ebg, final_patch_empty=empty,
                              any_all_invalid_gen=any(v == 0 for v in vbg) if vbg else None, terminated_by=term,
                              location_failures=r.get("prep_location_failures"))
    rep["per_issue"] = per_issue
    tot = [v["prompt_tokens"] + v["completion_tokens"] for v in per_issue.values()]
    q = statistics.quantiles(tot, n=20, method="inclusive") if len(tot) >= 2 else []
    q1, q3 = (statistics.quantiles(tot, n=4, method="inclusive")[0], statistics.quantiles(tot, n=4, method="inclusive")[2]) if len(tot) >= 2 else (None, None)
    fence = q3 + 1.5 * (q3 - q1) if q1 is not None else None  # 이상치: Q3 + 1.5·IQR 초과(Tukey)
    rep["tokens_per_issue"] = {"mean": round(statistics.mean(tot), 1) if tot else None,
                               "median": statistics.median(tot) if tot else None,
                               "p90": round(q[17], 1) if q else None,
                               "outlier_fence": round(fence, 1) if fence is not None else None,
                               "outliers": {iid: v["prompt_tokens"] + v["completion_tokens"] for iid, v in per_issue.items()
                                            if fence is not None and v["prompt_tokens"] + v["completion_tokens"] > fence},
                               "mean_without_outliers": round(statistics.mean([t for t in tot if fence is None or t <= fence]), 1) if tot else None,
                               "sum_prompt": sum(v["prompt_tokens"] for v in per_issue.values()),
                               "sum_completion": sum(v["completion_tokens"] for v in per_issue.values())}
    stage_tot = defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
    for v in per_issue.values():
        for s, t in (v["tokens_by_stage"] or {}).items():
            for k in ("prompt", "completion", "calls"):
                stage_tot[s][k] += t[k]
    rep["tokens_by_stage_total"] = dict(stage_tot)
    # 중단·오류 시도 토큰 (Q8b)
    other = [c for c in calls if c.get("attempt_id") not in final_attempts]
    rep["non_final_attempt_tokens"] = {"calls": len(other),
                                       "prompt": sum(c.get("prompt_tokens") or 0 for c in other),
                                       "completion": sum(c.get("completion_tokens") or 0 for c in other),
                                       "attempts": len({c.get("attempt_id") for c in other})}
    wc = [v["wall_clock_s"] for v in per_issue.values() if v["wall_clock_s"] is not None]
    rep["wall_clock_s"] = {"mean": round(statistics.mean(wc), 1) if wc else None, "max": max(wc) if wc else None,
                           "sum": round(sum(wc), 1)}

    # finish_reason=length 비율 (최종 시도 호출)
    fc = [c for c in calls if c.get("attempt_id") in final_attempts and c.get("error") is None]
    rep["finish_reason"] = dict(Counter(c.get("finish_reason") for c in fc))
    rep["finish_reason_length_ratio"] = _ratio(sum(c.get("finish_reason") == "length" for c in fc), len(fc))
    rep["finish_reason_length_by_stage"] = {
        s: _ratio(sum(c.get("finish_reason") == "length" for c in fc if c["stage"] == s), sum(c["stage"] == s for c in fc))
        for s in sorted({c["stage"] for c in fc})}
    rep["llm_call_errors"] = dict(Counter(c["error"].split(":")[0] for c in calls if c.get("error")))

    # 유효성: 파싱 실패·무효 사유 (B7 함수 범위 밖 매칭 비율 포함)
    checks = [v for v in validity if v["stage"] in ("init", "crossover", "mutation")]
    by_stage = {}
    for s in ("init", "crossover", "mutation"):
        xs = [v for v in checks if v["stage"] == s]
        by_stage[s] = {"checks": len(xs), "valid": sum(v["valid"] for v in xs),
                       "invalid_ratio": _ratio(sum(not v["valid"] for v in xs), len(xs)),
                       "reasons": dict(Counter(v["reason"] for v in xs if not v["valid"])),
                       # D-3: 정규화 매칭(줄 단위 strip)을 포함한 유효 판정, 그중 REPLACE 들여쓰기를 고친 것.
                       # 필드가 없는 이전 로그는 None(기록 없음)
                       "valid_with_normalized_match": (sum(1 for v in xs if v["valid"] and v.get("normalized_match_edits"))
                                                       if any("normalized_match_edits" in v for v in xs) else None),
                       "valid_with_reindent": (sum(1 for v in xs if v["valid"] and v.get("reindented_edits"))
                                               if any("reindented_edits" in v for v in xs) else None)}
    rep["validity_by_stage"] = by_stage
    rep["empty_diff_ratio"] = _ratio(sum(v.get("reason") == "empty_diff" for v in checks), len(checks))
    rep["files_inferred"] = {"edits": sum(v.get("files_inferred") or 0 for v in checks),
                             "checks_with_inference": sum(1 for v in checks if v.get("files_inferred")),
                             "checks": len(checks)}
    rep["match_outside_location_ratio"] = _ratio(sum(v.get("reason") == "match_outside_location" for v in checks), len(checks))
    # 결정 3 이후(위치 범위 = 첫 데코레이터 줄부터) 함수 범위 밖 매칭 수. 규칙은 이슈별 prep 기록으로 확인
    rules = Counter(r.get("prep_location_range_rule") or "def_line(이전 규칙)" for r in latest.values()
                    if r["status"] == "ok")
    rep["match_outside_location_after_decorator_fix"] = {
        "location_range_rule": dict(rules),
        "by_stage": {s_: sum(1 for v in checks if v["stage"] == s_ and v.get("reason") == "match_outside_location")
                     for s_ in ("init", "crossover", "mutation")},
        "total": sum(v.get("reason") == "match_outside_location" for v in checks)}
    summ = [v for v in validity if v["stage"] == "init_summary"]
    rep["init_parse"] = {"requests": len(summ), "requested": sum(v["requested"] for v in summ),
                         "parsed": sum(v["parsed"] for v in summ), "valid": sum(v["valid"] for v in summ)}

    # 후보: 중복 비율(세대별 patch_hash), 시간 초과 비율, r·g 분포
    by_gen = defaultdict(list)
    for c in cands:
        by_gen[(c["instance_id"], c["generation"])].append(c)
    dup = defaultdict(list)
    for (iid, g), cs in by_gen.items():
        hs = [c["patch_hash"] for c in cs if c["valid"]]
        dup[g].append(_ratio(len(hs) - len(set(hs)), len(hs)) or 0.0)
    rep["duplicate_ratio_by_generation"] = {g: round(statistics.mean(v), 3) for g, v in sorted(dup.items())}
    # 정의 A(위): 세대 개체군의 유효 후보 전체(보존된 엘리트 포함) 중 (개수 − 고유 해시 수)/개수, 이슈 평균.
    # 정의 B: 그 세대에 새로 만든 유효 후보끼리만 (세대 0은 init 전체). 엘리트는 세지 않는다.
    # 정의 C: 새로 만든 유효 후보 중 같은 세대 엘리트와 해시가 같은 비율.
    dup_b, dup_c = defaultdict(list), defaultdict(list)
    for (iid, g), cs in by_gen.items():
        new = [c["patch_hash"] for c in cs if c["valid"] and c["op"] != "elite"]
        eli = {c["patch_hash"] for c in cs if c["valid"] and c["op"] == "elite"}
        if new:
            dup_b[g].append((len(new) - len(set(new))) / len(new))
            if g > 0:
                dup_c[g].append(sum(h in eli for h in new) / len(new))
    rep["duplicate_ratio_new_only_by_generation"] = {g: round(statistics.mean(v), 3) for g, v in sorted(dup_b.items())}
    rep["new_equal_to_elite_ratio_by_generation"] = {g: round(statistics.mean(v), 3) for g, v in sorted(dup_c.items())}
    executed = {(c["instance_id"], c["candidate_id"]): c for c in cands if c["r"] is not None}
    ex = list(executed.values())
    rep["executed_candidates"] = len(ex)
    # 실행 캐시: 실제 실행 수와 적중 수. cache_hit 필드가 없는 로그(캐시 도입 전 실행)는 기록 없음으로 표시하고,
    # 같은 이슈 안 patch_hash 중복으로 '캐시가 있었다면'의 적중 수를 따로 계산한다.
    new_valid = [c for c in cands if c["valid"] and c["op"] != "elite"]
    if any("cache_hit" in c for c in new_valid):
        rep["exec_cache"] = {"recorded": True, "executions": sum(1 for c in new_valid if c.get("cache_hit") is False),
                             "hits": sum(1 for c in new_valid if c.get("cache_hit") is True)}
    else:
        seen, hits = set(), 0
        for c in new_valid:
            k = (c["instance_id"], c["patch_hash"])
            hits += k in seen
            seen.add(k)
        rep["exec_cache"] = {"recorded": False, "executions": len(new_valid), "hits": None,
                             "hits_if_cache_had_been_on": hits}
    rep["exec_timeout_ratio"] = _ratio(sum(c["exec_timeout"] for c in ex), len(ex))
    rep["exec_timeout_count"] = sum(c["exec_timeout"] for c in ex)
    rep["exec_error"] = dict(Counter(c["exec_error"] for c in ex if c.get("exec_error")))
    rep["r_distribution"] = dict(Counter(c["r"] for c in ex))
    rep["g_fraction_distribution"] = dict(Counter(
        ("no_signal" if c.get("no_regression_signal") or not c["g_total"] else
         f"{min(9, int(10 * c['g'] / c['g_total'])) / 10:.1f}+") for c in ex))
    rep["r_all_same_per_issue"] = {iid: sorted({c["r"] for c in ex if c["instance_id"] == iid})
                                   for iid in sorted({c["instance_id"] for c in ex})}
    # 세대 0 최고 적합도 대비 최종 최고 적합도 (실행된 유효 후보 기준, 사전식 (r, g))
    best = defaultdict(dict)
    for c in cands:  # 세대별 개체군 전체(엘리트 포함). ex는 후보별 마지막 행만 남기므로 쓰지 않는다
        if not c["valid"] or c["r"] is None:
            continue
        k = (c["r"], c["g"])
        g_ = c["generation"]
        if k > best[c["instance_id"]].get(g_, (-1, -1)):
            best[c["instance_id"]][g_] = k
    prog = {"executed_issues": 0, "improved": 0, "r_up": 0, "g_only_up": 0, "no_improvement": 0,
            "improved_issue_ids": []}
    for iid, bg in best.items():
        first, last = bg[min(bg)], bg[max(bg)]
        prog["executed_issues"] += 1
        if last > first:
            prog["improved"] += 1
            prog["improved_issue_ids"].append(iid)
            prog["r_up" if last[0] > first[0] else "g_only_up"] += 1
        else:
            prog["no_improvement"] += 1
    prog["improved_issue_ids"].sort()
    rep["fitness_progress_gen0_to_final"] = prog
    rep["mutation"] = {"drawn": sum(1 for c in cands if c["meta"].get("mutation_drawn") and c["op"] != "elite"),
                       "reverted": sum(1 for c in cands if c["meta"].get("mutation_reverted") and c["op"] != "elite")}
    rep["patch_lines"] = {"median": statistics.median([c["patch_lines"] for c in ex]) if ex else None,
                          "max": max([c["patch_lines"] for c in ex]) if ex else None}
    rep["empty_regression_list_issues"] = {iid: per_issue[iid]["status"] for iid, v in per_issue.items()
                                           if v["regression_list_size"] == 0}
    rep["no_regression_signal_issues"] = [iid for iid, v in per_issue.items() if v["no_regression_signal"]]
    kinds = Counter(k for v in per_issue.values() for k in (v["location_kinds"] or []))
    ctoks = [t for v in per_issue.values() for t in (v["class_location_tokens"] or [])]
    rep["locations"] = {"kinds": dict(kinds),
                        "issues_using_class_location": sum(1 for v in per_issue.values() if "class" in (v["location_kinds"] or [])),
                        "class_location_tokens": {"n": len(ctoks), "median": statistics.median(ctoks) if ctoks else None,
                                                  "max": max(ctoks) if ctoks else None},
                        "prep_failure_reasons": dict(Counter(v["prep_failure_reason"] for v in per_issue.values() if v["prep_failure_reason"])),
                        "file_list_levels": dict(Counter(v["file_list_level"] for v in per_issue.values() if v["file_list_level"]))}
    # A4: 원본에서 이미 통과한 재현 테스트 (관측용)
    rep["weak_repro_signal"] = sorted(i for i, r in latest.items() if (r.get("repro_on_base") or {}).get("pass"))
    for iid in per_issue:
        per_issue[iid]["weak_repro_signal"] = iid in rep["weak_repro_signal"]
        per_issue[iid]["gen0_valid_count"] = latest[iid].get("gen0_valid_count")
    # A6: 세대 0 유효 후보 부족 (gen0_failures 보고서 요약)
    g0 = []
    for iid, r in latest.items():
        rp = r.get("gen0_report")
        if rp and (run_dir / rp).exists():
            g0.append(json.loads((run_dir / rp).read_text()))
    reasons, idc = Counter(), Counter()
    for x in g0:
        reasons.update(x.get("failure_reasons", {}))
        idc.update(x.get("search_not_found_identifier_class", {}))
    rep["gen0_shortfall"] = {"issues": len(g0), "issue_ids": sorted(x["instance_id"] for x in g0),
                             "failure_reasons": dict(reasons), "search_not_found_identifier_class": dict(idc),
                             "terminated_by": dict(Counter(per_issue[x["instance_id"]]["terminated_by"] for x in g0))}
    # A3: 한 쌍에 같은 후보 / 같은 patch_hash 후보가 뽑힌 비율 (세대별, 관측용)
    gl = [g for g in JsonlLog(run_dir / "generations.jsonl").read() if g.get("attempt_id") in final_attempts]
    by_g = defaultdict(lambda: [0, 0, 0])
    for g in gl:
        by_g[g["generation"]][0] += g["n_pairs"]
        by_g[g["generation"]][1] += g["same_candidate"]
        by_g[g["generation"]][2] += g["same_patch_hash_distinct"]
    rep["same_parent_pairs_by_generation"] = {
        k: {"pairs": v[0], "same_candidate_ratio": _ratio(v[1], v[0]), "same_hash_distinct_ratio": _ratio(v[2], v[0])}
        for k, v in sorted(by_g.items())} if gl else None
    rep["prep_location_failure_reasons"] = dict(Counter(
        f["reason"] for v in per_issue.values() for f in (v.get("location_failures") or [])))
    rep["final_patch_empty_issues"] = sorted(i for i, v in per_issue.items() if v.get("final_patch_empty"))
    rep["all_invalid_issues"] = sorted(i for i, v in per_issue.items() if v.get("terminated_by") == "all_invalid")
    rep["budget_exceeded_issues"] = [iid for iid, v in per_issue.items() if v["status"] == "budget_exceeded"]
    # A2: 이미지를 받을 수 없는 이슈(빈 패치)와 infra_error로 승격된 행
    img = {iid: r.get("image_unavailable_cause") for iid, r in latest.items() if r["status"] == "image_unavailable"}
    rep["image_unavailable"] = {"issues": len(img), "by_cause": dict(Counter(img.values())), "issue_ids": sorted(img),
                                "escalated_rows": sum(1 for r in issues_all if r.get("escalated_from") == "image_unavailable")}
    # B2: 프롬프트에서 test_files 경로·파일명을 가린 횟수 (진화 단계 + prep)
    mh = Counter()
    for r in latest.values():
        mh.update(r.get("prompt_mask_hits") or {})
        mh.update({f"prep_{k}": v for k, v in (r.get("prep_prompt_mask_hits") or {}).items()})
    rep["prompt_mask_hits"] = {"total": dict(mh), "issues": sorted(
        i for i, r in latest.items() if sum((r.get("prompt_mask_hits") or {}).values())
        + sum((r.get("prep_prompt_mask_hits") or {}).values()))}
    rep["errored_attempts_total"] = sum(1 for r in issues_all if r["status"] not in DONE_STATUSES)

    # 채점 결과(있으면 최신)
    sums = sorted(p for p in (run_dir / "grading").glob("*/summary.json")
                  if not p.parent.name.startswith("repro_diag_")) if (run_dir / "grading").exists() else []
    if sums:
        g = json.loads(sums[-1].read_text())
        rep["grading"] = {k: g.get(k) for k in ("report", "resolved_instances", "completed_instances",
                                                 "error_instances", "empty_patch_instances", "resolved_ids", "error_ids")}
        rep["pass_at_1"] = _ratio(g.get("resolved_instances") or 0, len(latest))
        resolved = set(g.get("resolved_ids") or [])
        self_pass = {iid for iid, v in per_issue.items()
                     if v["status"] == "ok" and v["final_r"] == 1 and v["final_g"] == v["final_g_total"]}
        rep["self_pass_vs_official"] = {
            "self_pass": len(self_pass),
            "self_pass_but_unresolved": sorted(self_pass - resolved),
            "resolved_without_self_pass": sorted(resolved - self_pass),
            "both": sorted(self_pass & resolved)}
        for iid in per_issue:
            per_issue[iid]["resolved"] = iid in resolved
    diags = sorted((run_dir / "grading").glob("repro_diag_*/summary.json")) if (run_dir / "grading").exists() else []
    if diags:
        rep["repro_validity"] = json.loads(diags[-1].read_text())
        rows = [json.loads(l) for l in (diags[-1].parent / "results.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:
            if r["instance_id"] in per_issue:
                per_issue[r["instance_id"]]["repro_diag"] = f"base r={r['base_r']} / gold r={r['gold_r']}"
    return rep


def to_markdown(rep: dict) -> str:
    L = [f"# Run report — {rep['run_dir']}", f"생성: {rep['generated']}", "",
         f"- 이슈 {rep['n_issues']}개, 상태 {rep['status']}"]
    if "grading" in rep:
        L.append(f"- 채점: resolved {rep['grading']['resolved_instances']} / {rep['n_issues']} "
                 f"(pass@1 {rep.get('pass_at_1')}), error {rep['grading']['error_instances']}")
    if "repro_validity" in rep:
        L.append(f"- 재현 테스트 유효율(원본 실패·정답 통과, 채점 쪽 진단): {rep['repro_validity']}")
    L.append(f"- 위치: {rep['locations']}, budget_exceeded: {rep['budget_exceeded_issues']}")
    if "self_pass_vs_official" in rep:
        L.append(f"- 자체 신호 완전 통과 vs 공식 채점: {rep['self_pass_vs_official']}")
    L += [f"- 이슈당 토큰(최종 시도): {rep['tokens_per_issue']}",
          f"- 단계별 토큰 합계: {json.dumps(rep['tokens_by_stage_total'])}",
          f"- 중단·오류 시도 토큰: {rep['non_final_attempt_tokens']}",
          f"- wall-clock(s): {rep['wall_clock_s']}",
          f"- finish_reason: {rep['finish_reason']} (length 비율 {rep['finish_reason_length_ratio']}, 단계별 {rep['finish_reason_length_by_stage']})",
          f"- 유효성(단계별): {json.dumps(rep['validity_by_stage'], ensure_ascii=False)}",
          f"- 함수 범위 밖 매칭 비율(B7): {rep['match_outside_location_ratio']}, empty_diff 비율: {rep['empty_diff_ratio']}, 경로 추론(S2): {rep['files_inferred']}",
          f"- 초기 개체군 파싱: {rep['init_parse']}",
          f"- 세대별 중복 비율 A(엘리트 포함 개체군): {rep['duplicate_ratio_by_generation']}",
          f"- 세대별 중복 비율 B(새 후보끼리만): {rep['duplicate_ratio_new_only_by_generation']}, C(새 후보가 엘리트와 같음): {rep['new_equal_to_elite_ratio_by_generation']}",
          f"- 실행 캐시: {rep['exec_cache']}",
          f"- 세대 0 부족(A6): {rep['gen0_shortfall']}",
          f"- 같은 부모 쌍 비율(A3, 세대별): {rep['same_parent_pairs_by_generation']}",
          f"- 세대 0 대비 최종 최고 적합도(r 상승/g만 상승/무개선): {rep['fitness_progress_gen0_to_final']}",
          f"- 함수 범위 밖 매칭 수(데코레이터 보정 이후): {rep['match_outside_location_after_decorator_fix']}",
          f"- 이미지 없음(image_unavailable): {rep['image_unavailable']}",
          f"- 프롬프트 test_files 마스킹: {rep['prompt_mask_hits']}",
          f"- weak_repro_signal(원본에서 재현 테스트 통과, A4): {rep['weak_repro_signal']}",
          f"- 빈 최종 패치: {rep['final_patch_empty_issues']}, 유효 후보 0(all_invalid): {rep['all_invalid_issues']}, "
          f"전처리 위치 실패 원인: {rep['prep_location_failure_reasons']}",
          f"- 실행 후보 {rep['executed_candidates']}개, 시간 초과 {rep['exec_timeout_count']}개(비율 {rep['exec_timeout_ratio']}), 실행 오류 {rep['exec_error']}",
          f"- r 분포 {rep['r_distribution']}, g 비율 분포 {rep['g_fraction_distribution']}",
          f"- 이슈별 r 값 집합: {rep['r_all_same_per_issue']}",
          f"- 변이: {rep['mutation']}, 패치 줄 수: {rep['patch_lines']}",
          f"- 빈 회귀 목록 이슈: {rep['empty_regression_list_issues']}, 회귀 신호 없음(빈 test_files): {rep['no_regression_signal_issues']}",
          "", "## 이슈별", "| 이슈 | 상태 | 종료 | 세대 | r | g/total | 공식 resolved | 토큰(p+c) | 시간(s) | 위치 찾음/요청 (종류) | base 재현 | 재현 진단 | 세대별 유효 | 세대별 실행 | 빈 최종 패치 | 세대 0 유효 | weak_repro_signal |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for iid, v in rep["per_issue"].items():
        b = v.get("repro_on_base") or {}
        st = b.get("statuses")
        base = (("PASS" if b.get("pass") else "FAIL") + " " + json.dumps(dict(Counter(st.values())))) if st is not None else "-"
        L.append(f"| {iid} | {v['status']}{' (' + v['prep_failure_reason'] + ')' if v['prep_failure_reason'] else ''} | "
                 f"{v['terminated_by'] or '-'} | {v['generations_run'] or '-'} | {v['final_r'] if v['final_r'] is not None else '-'} | "
                 f"{v['final_g'] if v['final_g'] is not None else '-'}/{v['final_g_total'] if v['final_g_total'] is not None else '-'} | "
                 f"{v.get('resolved', '-')} | {v['prompt_tokens']}+{v['completion_tokens']} | {v['wall_clock_s']} | "
                 f"{v['locations_found'] if v['locations_found'] is not None else 0}/{v['locations_requested'] or 0} "
                 f"({','.join(v['location_kinds'] or []) or '-'}) | {base} | {v.get('repro_diag', '-')} | "
                 f"{v.get('valid_by_gen') or '-'} | {v.get('executed_by_gen') or '-'} | {v.get('final_patch_empty')} | "
                 f"{v.get('gen0_valid_count') if v.get('gen0_valid_count') is not None else '-'} | {v.get('weak_repro_signal')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run_dir = Path(a.run_dir).resolve()
    rep = build_report(run_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    (run_dir / f"report_{ts}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (run_dir / f"report_{ts}.md").write_text(to_markdown(rep))
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
