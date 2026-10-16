"""적합도 비교, 순위, 순위 비례 선택, 세대 교체에서 남는 후보(METHOD 4·6절)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .types import Candidate, ExecResult


@dataclass
class Individual:
    cid: str  # "g<세대>c<번호>"
    gen_born: int
    idx: int
    op: str  # init / crossover / mutation
    parent_ids: tuple[str, ...]
    candidate: Candidate
    valid: bool
    invalid_reason: str | None = None
    raw: str = ""  # 편집 원문 (무효 부모를 프롬프트에 보일 때 사용)
    exec: ExecResult | None = None
    exec_view: dict | None = None  # 변이 프롬프트용 실행 결과 요약 (보관, 2절)
    meta: dict = field(default_factory=dict)

    @property
    def r(self) -> int:
        return int(bool(self.exec and self.exec.repro_pass))

    @property
    def g(self) -> int:
        return self.exec.reg_passed if self.exec else 0

    @property
    def g_total(self) -> int:
        return self.exec.reg_total if self.exec else 0

    def fitness(self) -> tuple[int, int] | None:
        return (self.r, self.g) if self.valid and self.exec is not None else None

    def full_pass(self) -> bool:
        """완전 통과 = r 통과 + 회귀 전부 통과 (회귀 신호가 없으면 r만, 1.5·Q3b)."""
        if not self.valid or self.exec is None or self.exec.error or self.exec.timed_out:
            return False
        return self.r == 1 and self.g == self.g_total

    def view(self) -> dict:
        return {"cid": self.cid, "valid": self.valid, "diff": self.candidate.diff, "raw": self.raw,
                "exec": self.exec_view}


TIE_BREAKS = ("newer_first", "survivor_first")
DEFAULT_TIE_BREAK = "newer_first"  # D-8. 설정 algorithm.tie_break가 없을 때


def check_tie_break(tie_break: str) -> str:
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"algorithm.tie_break는 {TIE_BREAKS} 중 하나여야 합니다: {tie_break!r}")
    return tie_break


def rank_key(ind: Individual, current_gen: int, tie_break: str = DEFAULT_TIE_BREAK):
    """정렬 키(작을수록 상위). 무효는 모든 유효보다 아래(B6). (유효, r, g)가 같을 때:
    - newer_first(D-8, 기본): 태어난 세대가 큰(후대) 쪽 우선, 같은 세대면 후보 번호 오름차순. current_gen은 쓰지 않는다.
    - survivor_first(D-8 이전 규칙 B5, 이전 런 재현용): 이전 세대 생존자 우선 → 태어난 세대 오름차순 → 후보 번호."""
    fit = ind.fitness()
    head = (0 if fit is not None else 1, -(fit[0] if fit else 0), -(fit[1] if fit else 0))
    if check_tie_break(tie_break) == "newer_first":
        return head + (-ind.gen_born, ind.idx)
    return head + (0 if ind.gen_born < current_gen else 1, ind.gen_born, ind.idx)


def rank(pop: list[Individual], current_gen: int, tie_break: str = DEFAULT_TIE_BREAK) -> list[Individual]:
    check_tie_break(tie_break)
    return sorted(pop, key=lambda i: rank_key(i, current_gen, tie_break))


def rank_probabilities(n: int) -> list[float]:
    """p_r = 2(N−r+1)/(N(N+1)), r = 1..N."""
    return [2 * (n - r + 1) / (n * (n + 1)) for r in range(1, n + 1)]


def tie_key(ind: Individual) -> tuple:
    """부모 선택 확률의 동점 기준: (valid, r, g)가 모두 같으면 동점. 무효 후보끼리는 모두 동점."""
    fit = ind.fitness()
    return ("valid", fit[0], fit[1]) if fit is not None else ("invalid",)


def tied_rank_probabilities(ranked: list[Individual]) -> list[float]:
    """A1: p_r을 순위대로 주되, (valid, r, g)가 같은 후보들은 해당 순위들의 p_r 평균을 공유한다.
    rank()는 동점 후보를 인접하게 정렬하므로 연속 구간 단위로 평균한다. 동점이 없으면 p_r과 같다."""
    base = rank_probabilities(len(ranked))
    out = list(base)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and tie_key(ranked[j + 1]) == tie_key(ranked[i]):
            j += 1
        avg = sum(base[i:j + 1]) / (j + 1 - i)
        for k in range(i, j + 1):
            out[k] = avg
        i = j + 1
    return out


def select_elites(ranked: list[Individual], n: int) -> list[Individual]:
    """세대 교체에서 남는 후보: 순위 상위 N/2(유효 여부와 무관, METHOD 6절 8). 무효 후보는 순위상 유효 후보보다 아래."""
    return list(ranked[: n // 2])


def draw_parent_pairs(ranked: list[Individual], n_pairs: int, rng: random.Random,
                      max_redraws: int) -> list[tuple[Individual, Individual, int]]:
    """전체 개체군에서 순위 비례(동점은 평균 공유, A1)로 부모 쌍을 독립 추출.
    같은 후보가 두 번 뽑히면 두 번째를 최대 max_redraws회 재추출(기본 0회: 논문의 독립 추출, A3).
    반환: (부모1, 부모2, 재추출 횟수)."""
    probs = tied_rank_probabilities(ranked)
    pairs = []
    for _ in range(n_pairs):
        a = rng.choices(ranked, weights=probs, k=1)[0]
        b = rng.choices(ranked, weights=probs, k=1)[0]
        redraws = 0
        while b is a and redraws < max_redraws:
            b = rng.choices(ranked, weights=probs, k=1)[0]
            redraws += 1
        pairs.append((a, b, redraws))
    return pairs
