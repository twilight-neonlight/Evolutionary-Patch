"""8.1-4 검증: 모의 적합도로 선택·엘리트 보존·종료(세대 0 포함 5세대)."""
import random
from collections import Counter

from evopatch.loop import EvolveHooks, evolve
from evopatch.population import (Individual, draw_parent_pairs, rank, rank_probabilities, select_elites)
from evopatch.types import Candidate, ExecResult

N, K = 10, 5


def ind(cid, gen, idx, valid=True, fit=None, op="init", g_total=10):
    i = Individual(cid=cid, gen_born=gen, idx=idx, op=op, parent_ids=(), valid=valid,
                   candidate=Candidate(edits=(), patch_hash=cid, diff="d" if valid else ""))
    if fit is not None:
        i.exec = ExecResult(repro_pass=bool(fit[0]), reg_passed=fit[1], reg_total=g_total, timed_out=False, output_path="")
    return i


class Mock:
    """적합도 함수를 주입하는 모의 훅. fitness(cid, gen) -> (r, g) 또는 None(무효)."""

    def __init__(self, fitness, init_invalid=0):
        self.fitness = fitness
        self.init_invalid = init_invalid
        self.gens = []
        self.evaluated = []

    def hooks(self):
        def initial():
            return [ind(f"g0c{i}", 0, i, valid=i >= self.init_invalid) for i in range(N)]

        def evaluate(pop, gen):
            for i in pop:
                if i.valid and i.exec is None:
                    r, g = self.fitness(i.cid, gen)
                    i.exec = ExecResult(repro_pass=bool(r), reg_passed=g, reg_total=10, timed_out=False, output_path="")
                    self.evaluated.append(i.cid)

        def crossover(a, b, gen, j):
            return ind(f"g{gen}c{j}", gen, j, op="crossover")

        def mutate(child, a, b, gen, j):
            child.op = "mutation"
            return child

        def on_gen(gen, ranked, elites):
            self.gens.append((gen, [i.cid for i in ranked], [e.cid for e in elites]))

        return EvolveHooks(initial, evaluate, crossover, mutate, on_gen)


def run(mock, p_mut=0.15):
    return evolve(instance_id="x", run_index=0, N=N, K=K, p_mut=p_mut, max_redraws=5, hooks=mock.hooks())


def test_no_perfect_runs_exactly_K_generations_including_gen0():
    m = Mock(lambda cid, gen: (0, int(cid.split("c")[1]) % 7))
    res = run(m)
    assert res.terminated_by == "K" and res.generations_run == 5
    assert [g for g, _, _ in m.gens] == [0, 1, 2, 3, 4]


def test_stops_on_first_perfect():
    m = Mock(lambda cid, gen: (1, 10) if (gen == 2 and cid == "g2c3") else (0, 3))
    res = run(m)
    assert res.terminated_by == "perfect" and res.generations_run == 3 and res.final.cid == "g2c3"


def test_perfect_in_gen0_no_evolution():
    m = Mock(lambda cid, gen: (1, 10) if cid == "g0c4" else (1, 9))
    res = run(m)
    assert res.generations_run == 1 and res.final.cid == "g0c4"


def test_elites_preserved_and_not_reevaluated():
    m = Mock(lambda cid, gen: (0, int(cid.split("c")[1])))  # 번호가 클수록 좋음
    run(m)
    gen0_ranked = m.gens[0][1]
    elites1 = m.gens[1][2]
    assert elites1 == gen0_ranked[:5]
    assert set(elites1) <= set(m.gens[1][1])  # 다음 세대에 그대로 남음
    assert len(m.evaluated) == len(set(m.evaluated))  # 같은 후보를 다시 실행하지 않음
    assert all(len(r) == N for _, r, _ in m.gens)


def test_survivors_are_top_half_regardless_of_validity_and_children_half():  # METHOD 6절 8
    m = Mock(lambda cid, gen: (0, 1), init_invalid=7)  # 세대 0: 유효 3개, 무효 7개
    run(m)
    surv = m.gens[1][2]
    assert len(surv) == N // 2 and sum(int(c.split("c")[1]) >= 7 for c in surv) == 3  # 상위 N/2 = 유효 3(g0c7..9) + 무효 2
    assert len(m.gens[1][1]) == N  # 자식 N/2로 N 유지
    assert all(len(r) == N for _, r, _ in m.gens)


def test_mutation_decided_after_all_crossovers():  # METHOD 6절 6
    import evopatch.loop as L
    order = []
    m = Mock(lambda cid, gen: (0, int(cid.split("c")[1])))
    h = m.hooks()
    cx, mu = h.crossover, h.mutate
    h.crossover = lambda a, b, gen, j: (order.append(("x", gen)), cx(a, b, gen, j))[1]
    h.mutate = lambda c, a, b, gen, j: (order.append(("m", gen)), mu(c, a, b, gen, j))[1]
    L.evolve(instance_id="x", run_index=0, N=N, K=3, p_mut=1.0, max_redraws=0, hooks=h)
    for gen in (1, 2):
        ops = [o for o, g in order if g == gen]
        assert ops == ["x"] * (N // 2) + ["m"] * (N // 2)


def test_rank_order_and_ties():  # D-8(B5 대체), B6
    a = ind("g0c3", 0, 3, fit=(1, 2))
    b = ind("g1c0", 1, 0, fit=(1, 2))
    c = ind("g1c1", 1, 1, fit=(1, 5))
    d = ind("g1c2", 1, 2, valid=False)
    e = ind("g1c3", 1, 3, fit=(0, 9))
    assert [i.cid for i in rank([d, e, b, a, c], current_gen=1)] == ["g1c1", "g1c0", "g0c3", "g1c3", "g1c2"]


def test_tie_prefers_later_generation():  # D-8: 같은 (r, g)면 후대 우선(생존자 우선 없음)
    old = [ind(f"g0c{i}", 0, i, fit=(0, 5)) for i in range(3)]
    new = [ind(f"g3c{i}", 3, i, fit=(0, 5)) for i in range(2)]
    mid = ind("g2c7", 2, 7, fit=(0, 5))
    got = [i.cid for i in rank(old + [mid] + new, current_gen=3)]
    assert got == ["g3c0", "g3c1", "g2c7", "g0c0", "g0c1", "g0c2"]
    assert rank(old + new, current_gen=3) == rank(old + new, current_gen=4)  # current_gen과 무관


def test_tie_same_generation_by_candidate_number():  # D-8
    xs = [ind(f"g2c{i}", 2, i, fit=(1, 3)) for i in (7, 2, 9, 0)]
    assert [i.cid for i in rank(xs, current_gen=2)] == ["g2c0", "g2c2", "g2c7", "g2c9"]
    inv = [ind(f"g{g}c{i}", g, i, valid=False) for g, i in ((0, 1), (2, 5), (2, 3))]
    assert [i.cid for i in rank(inv + [xs[0]], 2)] == ["g2c7", "g2c3", "g2c5", "g0c1"]  # 무효는 맨 아래, 같은 규칙


def test_multiple_perfect_picks_latest_generation():  # D-8: 완전 통과가 여럿이면 후대 → 번호
    def fit(cid, gen):
        return (1, 10) if cid in ("g1c3", "g1c1") else (0, 3)
    m = Mock(fit)
    res = run(m)
    assert res.terminated_by == "perfect" and res.final.cid == "g1c1"
    pop = [ind("g0c4", 0, 4, fit=(1, 10)), ind("g2c9", 2, 9, fit=(1, 10)), ind("g2c3", 2, 3, fit=(1, 10))]
    assert [i.cid for i in rank(pop, 2) if i.full_pass()][0] == "g2c3"


def test_elite_boundary_tie_prefers_offspring():  # D-8: 엘리트 경계 동점이면 후대 자손이 남는다
    pop = ([ind(f"g0c{i}", 0, i, fit=(0, 9)) for i in range(3)] + [ind(f"g0c{i}", 0, i, fit=(0, 5)) for i in range(3, 5)]
           + [ind(f"g1c{i}", 1, i, fit=(0, 5)) for i in range(5, 10)])
    elites = select_elites(rank(pop, 1), N)
    assert [e.cid for e in elites] == ["g0c0", "g0c1", "g0c2", "g1c5", "g1c6"]


def test_rank_probabilities_formula():
    p = rank_probabilities(10)
    assert abs(sum(p) - 1) < 1e-12 and abs(p[0] - 20 / 110) < 1e-12 and abs(p[-1] - 2 / 110) < 1e-12


def test_selection_frequencies_match_rank_probabilities():
    pop = [ind(f"g0c{i}", 0, i, fit=(0, 10 - i)) for i in range(N)]
    ranked = rank(pop, 0)
    cnt = Counter()
    pairs = draw_parent_pairs(ranked, 20000, random.Random(1), max_redraws=0)
    for a, _, _ in pairs:
        cnt[a.cid] += 1
    for r, p in enumerate(rank_probabilities(N)):
        assert abs(cnt[ranked[r].cid] / 20000 - p) < 0.01
    # 무효 후보도 부모로 뽑힐 수 있다(Q4b)
    pop2 = [ind(f"g0c{i}", 0, i, valid=(i != 0), fit=None if i == 0 else (0, 1)) for i in range(N)]
    got = {a.cid for a, _, _ in draw_parent_pairs(rank(pop2, 0), 5000, random.Random(2), 5)}
    assert "g0c0" in got


def test_same_parent_redraw_cap():  # A3
    pop = [ind("g0c0", 0, 0, fit=(1, 1))]
    pairs = draw_parent_pairs(pop, 3, random.Random(0), max_redraws=5)
    assert all(a is b and r == 5 for a, b, r in pairs)  # 5회 넘으면 그대로 허용
    pop = [ind(f"g0c{i}", 0, i, fit=(0, i)) for i in range(N)]
    pairs = draw_parent_pairs(rank(pop, 0), 2000, random.Random(3), max_redraws=5)
    assert sum(a is b for a, b, _ in pairs) / 2000 < 0.001


def test_deterministic_given_seed():
    f = lambda cid, gen: (0, (hash(cid) % 5) if False else int(cid[-1]))
    m1, m2 = Mock(f), Mock(f)
    run(m1); run(m2)
    assert m1.gens == m2.gens


def test_mutation_rate_close_to_p_mut():
    calls = Counter()
    m = Mock(lambda cid, gen: (0, 0))
    h = m.hooks()
    orig = h.mutate

    def mut(child, a, b, gen, j):
        calls["m"] += 1
        return orig(child, a, b, gen, j)

    h.mutate = mut
    total = 0
    for k in range(200):
        evolve(instance_id=f"i{k}", run_index=0, N=N, K=K, p_mut=0.15, max_redraws=5, hooks=h)
        total += 4 * N // 2
    assert abs(calls["m"] / total - 0.15) < 0.03


def test_select_elites_helper():
    pop = rank([ind(f"g0c{i}", 0, i, fit=(0, i)) for i in range(N)], 0)
    assert [e.cid for e in select_elites(pop, N)] == [f"g0c{i}" for i in (9, 8, 7, 6, 5)]


def _run_min_valid(init_invalid, min_valid):
    m = Mock(lambda cid, gen: (0, 0), init_invalid=init_invalid)
    calls = Counter()
    h = m.hooks()
    xo = h.crossover
    h.crossover = lambda *a: (calls.update(["x"]), xo(*a))[1]
    res = evolve(instance_id="x", run_index=0, N=N, K=K, p_mut=0.15, max_redraws=0, hooks=h,
                 min_valid_to_evolve=min_valid)
    return res, m, calls


def test_min_valid_zero_valid_all_invalid():  # A6
    res, m, calls = _run_min_valid(N, 2)
    assert res.terminated_by == "all_invalid" and res.generations_run == 1 and not res.final.valid
    assert calls["x"] == 0 and m.evaluated == []


def test_min_valid_one_valid_evaluated_and_submitted():
    res, m, calls = _run_min_valid(N - 1, 2)
    assert m.evaluated == ["g0c9"] and calls["x"] == 0
    assert res.terminated_by == "insufficient_valid" and res.final.cid == "g0c9" and res.generations_run == 1


def test_min_valid_one_valid_perfect():
    m = Mock(lambda cid, gen: (1, 10), init_invalid=N - 1)
    res = evolve(instance_id="x", run_index=0, N=N, K=K, p_mut=0.15, max_redraws=0, hooks=m.hooks(), min_valid_to_evolve=2)
    assert res.terminated_by == "perfect" and res.final.cid == "g0c9"


def test_min_valid_two_valid_evolves():
    res, m, calls = _run_min_valid(N - 2, 2)
    assert res.generations_run == 5 and calls["x"] > 0


def test_min_valid_setting_one_is_previous_behavior():
    res, _, _ = _run_min_valid(N, 1)
    assert res.terminated_by == "all_invalid"
    res, m, calls = _run_min_valid(N - 1, 1)  # 유효 1개면 이전처럼 진화
    assert res.generations_run == 5 and calls["x"] > 0


def test_tied_probabilities():  # A1
    from evopatch.population import tied_rank_probabilities
    fits = [(1, 5), (1, 5), (1, 3), (0, 9), (0, 9), (0, 9), None, None, (0, 1), (0, 0)]
    pop = [ind(f"g0c{i}", 0, i, valid=f is not None, fit=f) for i, f in enumerate(fits)]
    ranked = rank(pop, 0)
    p = tied_rank_probabilities(ranked)
    base = rank_probabilities(N)
    assert abs(sum(p) - 1) < 1e-12
    by = {}
    for i, q in zip(ranked, p):
        by.setdefault(i.fitness(), []).append(q)
    for k, qs in by.items():
        assert max(qs) - min(qs) < 1e-15  # 동점 후보는 같은 확률
    assert abs(by[(1, 5)][0] - (base[0] + base[1]) / 2) < 1e-15
    assert abs(by[None][0] - (base[8] + base[9]) / 2) < 1e-15  # 무효는 최하위 순위들의 평균
    nodup = rank([ind(f"g0c{i}", 0, i, fit=(0, i)) for i in range(N)], 0)
    assert tied_rank_probabilities(nodup) == base  # 동점 없으면 기존 p_r과 동일


def test_independent_draw_allows_same_parent():  # A3: 재추첨 0회
    pop = rank([ind(f"g0c{i}", 0, i, fit=(0, i)) for i in range(N)], 0)
    pairs = draw_parent_pairs(pop, 5000, random.Random(5), max_redraws=0)
    assert all(r == 0 for _, _, r in pairs)
    same = sum(a is b for a, b, _ in pairs) / 5000
    expected = sum(q * q for q in rank_probabilities(N))
    assert abs(same - expected) < 0.02
