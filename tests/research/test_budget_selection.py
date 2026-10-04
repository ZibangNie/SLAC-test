"""Synthetic selection contracts; no tokenizer, model, dataset, or network."""

import math
import random
from dataclasses import replace

import pytest

from SLAC.retrieval.pack.budget_selection import (
    BudgetCandidate,
    select_budgeted,
    select_greedy,
)


def candidate(index, utility=1.0, *, duplicate=None, priority=None):
    return BudgetCandidate(
        f"synthetic-{index}", utility,
        str(index) if duplicate is None else duplicate,
        index if priority is None else priority,
    )


def test_exact_recovers_two_useful_units_blocked_by_greedy_choice():
    pool = [candidate(0, .9), candidate(1, .8), candidate(2, .8)]
    lengths = [8, 5, 5]
    result = select_budgeted(pool, lambda s: sum(lengths[i] for i in s), budget_tokens=10)
    assert result.greedy.selected_indexes == (0,)
    assert result.optimal.selected_indexes == (1, 2)
    assert result.optimal.utility == math.fsum([.8, .8])
    assert result.optimal.cost_tokens == 10
    assert not result.greedy.optimality_proven
    assert result.optimal.optimality_proven


def test_nonadditive_full_cost_disallows_pair_that_individually_fits():
    pool = [candidate(0, .8), candidate(1, .7)]
    costs = {(): 0, (0,): 2, (1,): 2, (0, 1): 9}
    result = select_budgeted(pool, costs.__getitem__, budget_tokens=4)
    assert result.greedy.selected_indexes == result.optimal.selected_indexes == (0,)
    assert result.greedy.optimality_proven
    assert result.feasible_subsets == 3


def test_nonmonotone_cost_does_not_prune_infeasible_singletons():
    pool = [candidate(0, .8), candidate(1, .7)]
    costs = {(): 0, (0,): 20, (1,): 20, (0, 1): 5}
    result = select_budgeted(pool, costs.__getitem__, budget_tokens=5)
    assert result.greedy.selected_indexes == ()
    assert result.optimal.selected_indexes == (0, 1)
    assert result.optimal.cost_tokens == 5
    assert result.enumerated_subsets == 4


def test_duplicate_locations_remain_alternatives_but_are_mutually_exclusive():
    pool = [candidate(0, .9, duplicate="same"), candidate(1, .9, duplicate="same"),
            candidate(2, .8)]
    lengths = [9, 1, 4]
    called = []

    def cost(indexes):
        called.append(indexes)
        assert not (0 in indexes and 1 in indexes)
        return sum(lengths[i] for i in indexes)

    result = select_budgeted(pool, cost, budget_tokens=10)
    assert result.greedy.selected_indexes == (0,)
    assert result.optimal.selected_indexes == (1, 2)
    assert result.duplicate_pruned_subsets == 2
    assert result.enumerated_subsets == result.cost_evaluations == len(called) == 6
    assert result.total_subsets == 8


def test_optimal_greedy_survives_equal_utility_lower_token_alternative():
    pool = [candidate(0, 2), candidate(1, 1), candidate(2, 1)]
    costs = {(): 0, (0,): 10, (1,): 2, (2,): 2,
             (0, 1): 12, (0, 2): 12, (1, 2): 4, (0, 1, 2): 14}
    result = select_budgeted(pool, costs.__getitem__, budget_tokens=10)
    assert result.greedy.selected_indexes == result.optimal.selected_indexes == (0,)
    assert result.optimal.cost_tokens == 10


def test_non_greedy_optimal_tie_follows_priority_membership_not_index_or_cost():
    pool = [candidate(0, 1), candidate(1, 2, priority=2), candidate(2, 2, priority=1)]
    costs = {(): 0, (0,): 10, (1,): 2, (2,): 8,
             (0, 1): 20, (0, 2): 20, (1, 2): 20, (0, 1, 2): 20}
    result = select_budgeted(pool, costs.__getitem__, budget_tokens=10)
    assert result.greedy.selected_indexes == (0,)
    assert result.optimal.selected_indexes == (2,)


def test_zero_utilities_keep_greedy_instead_of_preferring_empty_pack():
    pool = [candidate(i, 0) for i in range(4)]
    result = select_budgeted(pool, len, budget_tokens=3)
    assert result.greedy.selected_indexes == result.optimal.selected_indexes == (0, 1, 2)
    assert result.optimal.utility == 0.0


def test_priority_controls_scan_but_cost_receives_canonical_original_indexes():
    pool = [candidate(0, priority=2), candidate(1, priority=0), candidate(2, priority=0)]
    calls = []

    def cost(indexes):
        assert indexes == tuple(sorted(set(indexes)))
        calls.append(indexes)
        return len(indexes)

    result = select_budgeted(pool, cost, budget_tokens=2)
    assert result.greedy.selected_indexes == result.optimal.selected_indexes == (1, 2)
    assert calls[:3] == [(), (1,), (1, 2)]
    assert len(calls) == len(set(calls)) == result.cost_evaluations


@pytest.mark.parametrize("pool,max_units", [([], 3), ([candidate(0)], 0)])
def test_empty_domain_only_evaluates_empty_pack(pool, max_units):
    calls = []

    def cost(indexes):
        calls.append(indexes)
        return 0

    result = select_budgeted(pool, cost, budget_tokens=0, max_units=max_units)
    assert result.optimal.selected_indexes == ()
    assert result.greedy.optimality_proven
    assert result.total_subsets == result.enumerated_subsets == result.feasible_subsets == 1
    assert calls == [()]


def test_greedy_helper_uses_same_contract_without_enumerating():
    pool = [candidate(0, .9), candidate(1, .8), candidate(2, .8)]
    lengths = [8, 5, 5]
    calls = []

    def cost(indexes):
        calls.append(indexes)
        return sum(lengths[i] for i in indexes)

    result = select_greedy(pool, cost, budget_tokens=10)
    assert result.selected_indexes == (0,)
    assert not result.optimality_proven
    assert (1, 2) not in calls
    assert len(calls) == 4


def test_fixed_default_pool_cardinality_domain_is_697_subsets():
    pool = [candidate(i) for i in range(16)]
    result = select_budgeted(pool, len, budget_tokens=3)
    assert result.total_subsets == result.enumerated_subsets == result.cost_evaluations == 697
    assert result.feasible_subsets == 697


def test_subset_guard_rejects_before_any_callback_instead_of_approximating():
    def forbidden(_):
        pytest.fail("guard must run before any cost calls")

    with pytest.raises(ValueError, match="max_subsets"):
        select_budgeted([candidate(i) for i in range(16)], forbidden,
                        budget_tokens=100, max_units=16, max_subsets=10000)
    with pytest.raises(ValueError, match="max_candidates"):
        select_budgeted([candidate(i) for i in range(17)], forbidden, budget_tokens=100)


def test_explicit_small_subset_limit_and_max_units_above_pool_size():
    result = select_budgeted([candidate(0), candidate(1)], len,
                            budget_tokens=5, max_units=20, max_subsets=4)
    assert result.optimal.selected_indexes == (0, 1)
    assert result.total_subsets == 4
    with pytest.raises(ValueError, match="max_subsets"):
        select_budgeted([candidate(0), candidate(1)], len,
                        budget_tokens=5, max_units=20, max_subsets=3)


def test_resource_matched_selection_uses_actual_baseline_cost_and_cardinality():
    pool = [candidate(0, .9), candidate(1, .8), candidate(2, .8)]
    lengths = [8, 5, 5]
    cost = lambda s: sum(lengths[i] for i in s)
    original = select_budgeted(pool, cost, budget_tokens=10)
    matched = select_budgeted(pool, cost, budget_tokens=original.greedy.cost_tokens,
                              max_units=len(original.greedy.selected_indexes))
    assert matched.optimal.selected_indexes == (0,)
    assert matched.optimal.cost_tokens <= original.greedy.cost_tokens
    assert len(matched.optimal.selected_indexes) <= len(original.greedy.selected_indexes)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -.1,
                                True, False, "0.5", None, 10 ** 1000])
def test_invalid_utilities_are_rejected(bad):
    with pytest.raises((TypeError, ValueError), match="utility"):
        select_budgeted([candidate(0, bad)], len, budget_tokens=10)


@pytest.mark.parametrize("field,bad", [("candidate_id", ""), ("candidate_id", 1),
                                     ("duplicate_key", None), ("priority", True),
                                     ("priority", 1.0), ("priority", -1)])
def test_invalid_candidate_fields_are_rejected(field, bad):
    with pytest.raises((TypeError, ValueError)):
        select_budgeted([replace(candidate(0), **{field: bad})], len, budget_tokens=10)


def test_duplicate_ids_and_wrong_candidate_types_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_budgeted([candidate(0), candidate(0)], len, budget_tokens=10)
    with pytest.raises(TypeError, match="BudgetCandidate"):
        select_budgeted([{}], len, budget_tokens=10)
    with pytest.raises(TypeError, match="sequence"):
        select_budgeted(iter([candidate(0)]), len, budget_tokens=10)
    with pytest.raises(TypeError, match="callable"):
        select_budgeted([], None, budget_tokens=10)


@pytest.mark.parametrize("name", ["budget_tokens", "max_units", "max_candidates", "max_subsets"])
@pytest.mark.parametrize("bad", [True, 1.0, -1])
def test_invalid_limits_are_rejected(name, bad):
    kwargs = {"budget_tokens": 10, name: bad}
    with pytest.raises((TypeError, ValueError), match=name):
        select_budgeted([], len, **kwargs)


def test_zero_subset_cap_is_invalid():
    with pytest.raises(ValueError, match="max_subsets"):
        select_budgeted([], len, budget_tokens=0, max_subsets=0)


@pytest.mark.parametrize("bad", [True, False, 1.0, -1, float("nan"), float("inf"), None])
def test_invalid_costs_are_rejected_without_numeric_coercion(bad):
    with pytest.raises((TypeError, ValueError), match="pack cost"):
        select_budgeted([candidate(0)], lambda _: bad, budget_tokens=10)


def test_invalid_nonempty_cost_is_also_rejected():
    with pytest.raises(TypeError, match="pack cost"):
        select_budgeted([candidate(0)], lambda s: 1.0 if s else 0, budget_tokens=10)


def test_empty_cost_must_fit_and_callback_failures_propagate():
    with pytest.raises(ValueError, match="empty pack"):
        select_budgeted([], lambda _: 2, budget_tokens=1)

    def broken(_):
        raise RuntimeError("synthetic counter failed")

    with pytest.raises(RuntimeError, match="synthetic counter"):
        select_budgeted([], broken, budget_tokens=1)


def test_nonfinite_sum_is_rejected_even_when_individual_utilities_are_finite():
    with pytest.raises(ValueError, match="utility sum"):
        select_budgeted([candidate(0, 1e308), candidate(1, 1e308)], len, budget_tokens=2)


def test_no_tolerance_hides_a_real_fsum_objective_improvement():
    pool = [candidate(0, 1), candidate(1, math.nextafter(1.0, math.inf))]
    result = select_budgeted(pool, len, budget_tokens=1)
    assert result.greedy.selected_indexes == (0,)
    assert result.optimal.selected_indexes == (1,)
    assert result.optimal.utility > result.greedy.utility


def test_matches_independent_bitmask_oracle_on_small_nonmonotone_problems():
    rng = random.Random(20261004)
    for _ in range(40):
        size = rng.randrange(1, 7)
        cap = rng.randrange(0, 4)
        budget = rng.randrange(0, 8)
        pool = [candidate(i, rng.randrange(0, 5), duplicate=str(rng.randrange(0, size)),
                          priority=rng.randrange(0, size)) for i in range(size)]
        costs = {}
        for mask in range(1 << size):
            indexes = tuple(i for i in range(size) if mask & (1 << i))
            costs[indexes] = rng.randrange(0, 12) if mask else 0
        result = select_budgeted(pool, costs.__getitem__, budget_tokens=budget, max_units=cap)
        priority = sorted(range(size), key=lambda i: (pool[i].priority, i))
        feasible = []
        for mask in range(1 << size):
            indexes = tuple(i for i in range(size) if mask & (1 << i))
            if len(indexes) > cap or len({pool[i].duplicate_key for i in indexes}) != len(indexes):
                continue
            if costs[indexes] <= budget:
                feasible.append(indexes)
        values = {s: math.fsum(pool[i].utility for i in s) for s in feasible}
        optimum = max(values.values())
        if result.greedy.utility == optimum:
            expected = result.greedy.selected_indexes
        else:
            expected = max((s for s in feasible if values[s] == optimum),
                           key=lambda s: tuple(i in s for i in priority))
        assert result.optimal.selected_indexes == expected
        assert result.optimal.utility == optimum
        assert result.feasible_subsets == len(feasible)
        assert result.total_subsets == result.enumerated_subsets + result.duplicate_pruned_subsets
        assert result.cost_evaluations == result.enumerated_subsets
