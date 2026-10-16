"""D-8 스위치 algorithm.tie_break: newer_first(기본) / survivor_first(D-8 이전 규칙 B5 재현) / 잘못된 값은 오류."""
import pytest

import evopatch.runner as R
from evopatch.config import Config
from evopatch.population import DEFAULT_TIE_BREAK, rank, select_elites

from .conftest import ROOT
from .test_loop import N, Mock, ind


def tied_pop():
    a = ind("g0c3", 0, 3, fit=(1, 2))
    b = ind("g1c0", 1, 0, fit=(1, 2))
    c = ind("g1c1", 1, 1, fit=(1, 5))
    d = ind("g1c2", 1, 2, valid=False)
    e = ind("g1c3", 1, 3, fit=(0, 9))
    f = ind("g2c4", 2, 4, fit=(1, 2))
    return [d, e, b, a, c, f]


def test_newer_first_order():
    got = [i.cid for i in rank(tied_pop(), 2, "newer_first")]
    assert got == ["g1c1", "g2c4", "g1c0", "g0c3", "g1c3", "g1c2"]


def test_survivor_first_reproduces_b5():
    # 세대 2 기준: 생존자(g0c3, g1c0) 먼저, 그 안에서는 태어난 세대 오름차순 → 번호, 그다음 이번 세대 자손
    got = [i.cid for i in rank(tied_pop(), 2, "survivor_first")]
    assert got == ["g1c1", "g0c3", "g1c0", "g2c4", "g1c3", "g1c2"]
    # D-8 이전 테스트(test_rank_order_and_ties의 이전 기대값)와 같음
    old = [x for x in tied_pop() if x.cid != "g2c4"]
    assert [i.cid for i in rank(old, 1, "survivor_first")] == ["g1c1", "g0c3", "g1c0", "g1c3", "g1c2"]
    pop = [ind(f"g0c{i}", 0, i, fit=(0, 5)) for i in range(5)] + [ind(f"g1c{i}", 1, i, fit=(0, 5)) for i in range(5)]
    assert [e.cid for e in select_elites(rank(pop, 1, "survivor_first"), N)] == [f"g0c{i}" for i in range(5)]
    assert [e.cid for e in select_elites(rank(pop, 1, "newer_first"), N)] == [f"g1c{i}" for i in range(5)]


def test_default_is_newer_first_and_configs_unset():
    assert DEFAULT_TIE_BREAK == "newer_first"
    assert rank(tied_pop(), 2) == rank(tied_pop(), 2, "newer_first")
    assert Config.load(ROOT / "tests" / "data" / "test_config.yaml").get("algorithm.tie_break") is None  # 키 없음 = 기본값


def test_evolve_uses_switch():
    from evopatch.loop import evolve
    finals = {}  # 모든 개체가 같은 적합도(정체): K 도달 시 제출은 동점 규칙만으로 정해진다
    for tb in ("newer_first", "survivor_first"):
        m = Mock(lambda cid, gen: (0, 3))
        res = evolve(instance_id="x", run_index=0, N=N, K=5, p_mut=0.15, max_redraws=5, hooks=m.hooks(), tie_break=tb)
        finals[tb] = res.final.cid
    assert finals["newer_first"] == "g4c0" and finals["survivor_first"] == "g0c0"


def test_invalid_value_errors(cfg):
    with pytest.raises(ValueError):
        rank(tied_pop(), 2, "oldest_first")
    from evopatch.loop import evolve
    with pytest.raises(ValueError):
        evolve(instance_id="x", run_index=0, N=N, K=5, p_mut=0.15, max_redraws=5, hooks=Mock(lambda c, g: (0, 1)).hooks(),
               tie_break="random")
    d = cfg.as_dict()
    d["algorithm"]["tie_break"] = "random"
    bad = Config(d)
    with pytest.raises(ValueError):
        R.IssueRunner(cfg=bad, run_dir=ROOT, run_id="t", llm=None, executor=None, exec_specs={}, regression_pool={})
