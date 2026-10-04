"""Small-pool evidence selection with an exact, whole-pack token budget.

This optional component does not change the existing evidence packer. Utilities
are supplied by the caller; the module neither reads model scores nor accesses
reference answers. Maximizing their sum is a selection objective, not a claim
about calibrated probabilities or downstream answer quality.

The cost callback must be deterministic and side-effect free. It receives a
tuple of original candidate indexes, sorted in increasing order, including the
empty tuple. It must count the complete rendered pack, not sum independent
unit estimates. Costs need not be additive or monotone. The empty pack must be
feasible, so both methods have a valid baseline. A callback result is cached
only within one call to a public selection function.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Real


PackCost = Callable[[tuple[int, ...]], int]


@dataclass(frozen=True)
class BudgetCandidate:
    """One selectable unit; duplicate keys use exact, case-sensitive equality.

    Lower priority values are scanned first. Equal priorities preserve input
    order. IDs must be unique nonempty strings. Duplicate keys may be empty;
    all units with the same key are mutually exclusive. Utility must be finite
    and nonnegative; a zero-utility candidate remains eligible.
    """

    candidate_id: str
    utility: float
    duplicate_key: str
    priority: int


@dataclass(frozen=True)
class BudgetSelection:
    selected_indexes: tuple[int, ...]
    utility: float
    cost_tokens: int
    optimality_proven: bool


@dataclass(frozen=True)
class BudgetSelectionComparison:
    """Selections and exhaustive-search accounting.

    ``total_subsets`` includes duplicate-incompatible subsets and the empty
    subset. ``enumerated_subsets`` counts duplicate-compatible subsets whose
    full cost was checked; ``feasible_subsets`` additionally satisfy the token
    budget. Thus ``total_subsets == enumerated_subsets +
    duplicate_pruned_subsets``. Greedy and enumeration share the cost cache;
    ``cost_evaluations`` counts actual callback calls.

    Optimality covers only the supplied candidates, utility, duplicate rule,
    cardinality, budget, and deterministic callback. It says nothing about
    candidates outside that pool or about evidence/answer quality.
    """

    greedy: BudgetSelection
    optimal: BudgetSelection
    candidate_count: int
    total_subsets: int
    enumerated_subsets: int
    duplicate_pruned_subsets: int
    feasible_subsets: int
    cost_evaluations: int


@dataclass(frozen=True)
class _Problem:
    candidates: tuple[BudgetCandidate, ...]
    utilities: tuple[float, ...]
    priority_order: tuple[int, ...]
    budget: int
    max_units: int


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    # bool and floating token counts must not silently become integers.
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _prepare(
    candidates: Sequence[BudgetCandidate],
    cost: PackCost,
    budget_tokens: int,
    max_units: int,
    max_candidates: int,
) -> _Problem:
    budget = _integer(budget_tokens, "budget_tokens")
    limit = _integer(max_units, "max_units")
    pool_limit = _integer(max_candidates, "max_candidates")
    if not callable(cost):
        raise TypeError("cost must be callable")
    if not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    if len(candidates) > pool_limit:
        raise ValueError("candidate count exceeds max_candidates cap")

    pool = tuple(candidates)
    ids: set[str] = set()
    utilities: list[float] = []
    for index, candidate in enumerate(pool):
        if not isinstance(candidate, BudgetCandidate):
            raise TypeError(f"candidate {index} must be a BudgetCandidate")
        if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
            raise ValueError(f"candidate {index} ID must be a nonempty string")
        if candidate.candidate_id in ids:
            raise ValueError("candidate IDs must be unique")
        ids.add(candidate.candidate_id)
        if not isinstance(candidate.duplicate_key, str):
            raise TypeError(f"candidate {index} duplicate_key must be a string")
        _integer(candidate.priority, f"candidate {index} priority")
        if isinstance(candidate.utility, bool) or not isinstance(candidate.utility, Real):
            raise TypeError(f"candidate {index} utility must be a real number, excluding bool")
        try:
            utility = float(candidate.utility)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"candidate {index} utility must be finite") from exc
        if not math.isfinite(utility) or utility < 0:
            raise ValueError(f"candidate {index} utility must be finite and nonnegative")
        utilities.append(utility)

    return _Problem(
        candidates=pool,
        utilities=tuple(utilities),
        priority_order=tuple(sorted(range(len(pool)), key=lambda i: (pool[i].priority, i))),
        budget=budget,
        max_units=min(limit, len(pool)),
    )


class _Costs:
    def __init__(self, callback: PackCost, budget: int) -> None:
        self.callback = callback
        self.cache: dict[tuple[int, ...], int] = {}
        if self(()) > budget:
            raise ValueError("the empty pack cost must fit budget_tokens")

    def __call__(self, indexes: tuple[int, ...]) -> int:
        if indexes not in self.cache:
            self.cache[indexes] = _integer(self.callback(indexes), "pack cost")
        return self.cache[indexes]


def _utility(problem: _Problem, indexes: tuple[int, ...]) -> float:
    try:
        return math.fsum(problem.utilities[i] for i in indexes)
    except OverflowError as exc:
        raise ValueError("selected utility sum must be finite") from exc


def _greedy(problem: _Problem, costs: _Costs) -> BudgetSelection:
    selected: tuple[int, ...] = ()
    duplicate_keys: set[str] = set()
    for index in problem.priority_order:
        if len(selected) >= problem.max_units:
            break
        key = problem.candidates[index].duplicate_key
        if key in duplicate_keys:
            continue
        proposed = tuple(sorted((*selected, index)))
        if costs(proposed) <= problem.budget:
            selected = proposed
            duplicate_keys.add(key)
    return BudgetSelection(selected, _utility(problem, selected), costs(selected), False)


def select_greedy(
    candidates: Sequence[BudgetCandidate],
    cost: PackCost,
    *,
    budget_tokens: int,
    max_units: int = 3,
    max_candidates: int = 16,
) -> BudgetSelection:
    """Scan fixed priorities, skipping duplicates and over-budget additions.

    The returned indexes use original input order, regardless of scan order.
    This standalone method does not prove optimality. Failed additions are not
    revisited; ``select_budgeted`` can discover feasible combinations skipped
    by this rule when costs are nonmonotone.
    """
    problem = _prepare(candidates, cost, budget_tokens, max_units, max_candidates)
    return _greedy(problem, _Costs(cost, problem.budget))


def select_budgeted(
    candidates: Sequence[BudgetCandidate],
    cost: PackCost,
    *,
    budget_tokens: int,
    max_units: int = 3,
    max_candidates: int = 16,
    max_subsets: int = 10000,
) -> BudgetSelectionComparison:
    """Compare priority-greedy with complete constrained subset enumeration.

    The objective is ``math.fsum`` of selected utilities, compared exactly as
    returned (no epsilon). Keep the greedy pack if it attains the maximum.
    Otherwise, ties prefer membership of the earliest priority candidate, then
    the next, and so on. Token count does not break equal-utility ties.

    All subsets through ``max_units`` are considered, including the empty set;
    only duplicate-incompatible subsets are pruned without calling ``cost``.
    A singleton over budget is not removed: adding another unit may lower the
    full pack cost. The candidate and subset caps raise before any cost calls
    instead of silently falling back to approximate selection. At the default
    16-candidate/3-unit limits, at most 697 subsets are enumerated. Zero units
    and an empty pool are supported.
    """
    problem = _prepare(candidates, cost, budget_tokens, max_units, max_candidates)
    subset_limit = _integer(max_subsets, "max_subsets", minimum=1)
    total = 0
    for size in range(problem.max_units + 1):
        total += math.comb(len(problem.candidates), size)
        if total > subset_limit:
            raise ValueError("subset count exceeds max_subsets cap")

    costs = _Costs(cost, problem.budget)
    greedy = _greedy(problem, costs)
    best = greedy
    enumerated = duplicate_pruned = feasible = 0

    def tie_key(indexes: tuple[int, ...]) -> tuple[bool, ...]:
        included = set(indexes)
        return tuple(i in included for i in problem.priority_order)

    for size in range(problem.max_units + 1):
        for indexes in combinations(range(len(problem.candidates)), size):
            keys = {problem.candidates[i].duplicate_key for i in indexes}
            if len(keys) != size:
                duplicate_pruned += 1
                continue
            enumerated += 1
            tokens = costs(indexes)
            if tokens > problem.budget:
                continue
            feasible += 1
            utility = _utility(problem, indexes)
            improves = utility > best.utility
            breaks_non_greedy_tie = (
                utility == best.utility
                and best.utility > greedy.utility
                and tie_key(indexes) > tie_key(best.selected_indexes)
            )
            if improves or breaks_non_greedy_tie:
                best = BudgetSelection(indexes, utility, tokens, False)

    return BudgetSelectionComparison(
        greedy=replace(greedy, optimality_proven=greedy.utility == best.utility),
        optimal=replace(best, optimality_proven=True),
        candidate_count=len(problem.candidates),
        total_subsets=total,
        enumerated_subsets=enumerated,
        duplicate_pruned_subsets=duplicate_pruned,
        feasible_subsets=feasible,
        cost_evaluations=len(costs.cache),
    )
