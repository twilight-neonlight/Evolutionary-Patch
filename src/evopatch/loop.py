"""세대 진행·종료(METHOD 3·6절). K는 세대 0을 포함한다(세대 0..K-1)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .population import DEFAULT_TIE_BREAK, Individual, check_tie_break, draw_parent_pairs, rank, select_elites
from .seeding import rng


@dataclass
class EvolveResult:
    final: Individual
    terminated_by: str  # perfect / K
    generations_run: int
    history: list[list[Individual]] = field(default_factory=list)


@dataclass
class EvolveHooks:
    initial: Callable[[], list[Individual]]
    evaluate: Callable[[list[Individual], int], None]  # 유효하고 아직 실행 안 된 후보를 실행
    crossover: Callable[[Individual, Individual, int, int], Individual]
    mutate: Callable[[Individual, Individual, Individual, int, int], Individual]
    on_generation: Callable[[int, list[Individual], list[Individual]], None]  # (세대, 순위 매긴 개체군, 엘리트)
    child_workers: int = 1
    on_pairs: Callable[[int, list], None] | None = None  # (자손 세대, 부모 쌍 목록) 관측용


def evolve(*, instance_id: str, run_index: int, N: int, K: int, p_mut: float, max_redraws: int,
           hooks: EvolveHooks, min_valid_to_evolve: int = 0, tie_break: str = DEFAULT_TIE_BREAK) -> EvolveResult:
    check_tie_break(tie_break)  # 잘못된 값이면 LLM 호출 전에 오류
    pop = hooks.initial()
    assert len(pop) == N
    elites: list[Individual] = []
    history = []
    n_valid = sum(i.valid for i in pop)
    if n_valid < min_valid_to_evolve:
        # A6: 세대 0(부족분 재요청 후) 유효 후보가 최소 수 미만이면 진화하지 않는다. 추가 LLM 호출 없음.
        if n_valid == 0:  # 실행도 하지 않고 종료
            ranked = rank(pop, 0, tie_break)
            history.append(ranked)
            hooks.on_generation(0, ranked, [])
            return EvolveResult(ranked[0], "all_invalid", 1, history)
        hooks.evaluate(pop, 0)  # 유효 후보만 실행
        ranked = rank(pop, 0, tie_break)
        history.append(ranked)
        hooks.on_generation(0, ranked, [])
        perfect = [i for i in ranked if i.full_pass()]
        if perfect:
            return EvolveResult(perfect[0], "perfect", 1, history)
        return EvolveResult(ranked[0], "insufficient_valid", 1, history)  # 최상위 유효 후보 제출
    for gen in range(K):
        hooks.evaluate(pop, gen)
        ranked = rank(pop, gen, tie_break)
        history.append(ranked)
        hooks.on_generation(gen, ranked, elites)
        perfect = [i for i in ranked if i.full_pass()]
        if perfect:
            return EvolveResult(perfect[0], "perfect", gen + 1, history)
        if gen == K - 1:
            return EvolveResult(ranked[0], "K", gen + 1, history)

        nxt = gen + 1
        # 세대 교체(METHOD 6절 8): 다음 세대 = 현재 순위 상위 N/2(유효 여부와 무관) + 새 자식 N/2
        elites = select_elites(ranked, N)
        n_off = N - len(elites)
        sel = rng(instance_id, run_index, "selection", nxt)
        pairs = draw_parent_pairs(ranked, n_off, sel, max_redraws)
        if hooks.on_pairs:
            hooks.on_pairs(nxt, pairs)

        def cross(j):
            a, b, redraws = pairs[j]
            child = hooks.crossover(a, b, nxt, j)  # 재시도 포함
            child.meta["parent_redraws"] = redraws
            return child

        with ThreadPoolExecutor(max(1, hooks.child_workers)) as ex:
            children = list(ex.map(cross, range(n_off)))
        # 변이 판정(METHOD 6절 6): 교배와 재시도가 모두 끝난 뒤 자식마다 p_mut로 정한다(교배 자식이 무효여도 시도)
        mrng = rng(instance_id, run_index, "mutation", nxt)
        flags = [mrng.random() < p_mut for _ in children]

        def mut(j):
            child = children[j]
            child.meta["mutation_drawn"] = flags[j]
            if not flags[j]:
                return child
            a, b, redraws = pairs[j]
            m = hooks.mutate(child, a, b, nxt, j)  # 변이가 무효면 교배 자식을 돌려준다
            m.meta.update(parent_redraws=redraws, mutation_drawn=True)
            return m

        with ThreadPoolExecutor(max(1, hooks.child_workers)) as ex:
            children = list(ex.map(mut, range(n_off)))
        pop = elites + children
    raise AssertionError("unreachable")
