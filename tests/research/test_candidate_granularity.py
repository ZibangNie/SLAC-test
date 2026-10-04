"""Six synthetic tests for interval merging and the declared single-pass policy.

Counters below are toy costs, not tokenizer measurements. Importing the module
does not prepare inputs or run its model/data analysis commands.
"""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "docs/research/probe_candidate_granularity.py"
_SPEC = importlib.util.spec_from_file_location("probe_candidate_granularity", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def candidate(name, span, dense_rank=0):
    return {"candidate_id": name, "task_id": name, "dense_rank": dense_rank, "span": span}


def toy_cost(runs):
    return sum(end - start for start, end in runs)


def test_merge_only_touching_spans_without_filling_gaps_or_mutating_input():
    spans = [[4, 6], [0, 2], [2, 3]]
    assert probe.merge_spans(spans) == [[0, 3], [4, 6]]
    assert spans == [[4, 6], [0, 2], [2, 3]]
    assert probe.merge_spans([]) == []


def test_merge_rejects_overlapping_and_duplicate_spans():
    for spans in ([(0, 2), (1, 3)], [(0, 2), (0, 2)], [(0, 4), (1, 2)]):
        with pytest.raises(ValueError, match="overlap"):
            probe.merge_spans(spans)


def test_reaching_three_runs_still_allows_later_adjacent_candidate():
    items = [candidate("a", [0, 1]), candidate("b", [3, 4]),
             candidate("c", [6, 7]), candidate("d", [1, 2])]
    selected, runs, trace = probe.select_ranked(items, dict(a=4, b=3, c=2, d=1), toy_cost)
    assert [item["candidate_id"] for item in selected] == ["a", "b", "c", "d"]
    assert runs == [[0, 2], [3, 4], [6, 7]]
    assert [row["action"] for row in trace] == ["admit"] * 4
    assert [row["proposed_runs"] for row in trace] == [1, 2, 3, 3]


def test_over_budget_candidate_is_skipped_and_later_feasible_candidates_are_used():
    items = [candidate("large", [0, 1025]), candidate("exact", [2000, 3024]),
             candidate("extra", [4000, 4001])]
    selected, runs, trace = probe.select_ranked(items, dict(large=3, exact=2, extra=1), toy_cost)
    assert [item["candidate_id"] for item in selected] == ["exact"]
    assert runs == [[2000, 3024]]
    assert [row["action"] for row in trace] == ["skip_tokens", "admit", "skip_tokens"]
    assert [row["proposed_tokens"] for row in trace] == [1025, 1024, 1025]


def test_run_limit_skip_is_not_revisited_after_later_bridge():
    items = [candidate("a", [0, 1]), candidate("b", [2, 3]),
             candidate("c", [4, 5]), candidate("skipped", [6, 7]),
             candidate("bridge", [1, 2])]
    calls = []

    def count(runs):
        calls.append(runs)
        return toy_cost(runs)

    scores = dict(a=5, b=4, c=3, skipped=2, bridge=1)
    selected, runs, trace = probe.select_ranked(items, scores, count)
    assert [item["candidate_id"] for item in selected] == ["a", "b", "c", "bridge"]
    assert runs == [[0, 3], [4, 5]]
    assert trace[3] == {"candidate_id": "skipped", "action": "skip_runs",
                        "proposed_runs": 4, "proposed_tokens": None}
    assert len(calls) == 4  # The four-run proposal is not token-counted.
    # A retry would now fit, but the declared policy deliberately does not retry.
    retry_runs = probe.merge_spans(runs + [[6, 7]])
    assert len(retry_runs) == 3 and toy_cost(retry_runs) <= 1024
    assert len(trace) == 5


def test_equal_scores_use_dense_parent_rank_then_source_order_deterministically():
    items = [candidate("later", [2, 3], 1), candidate("first_rank", [6, 7], 0),
             candidate("earlier", [0, 1], 1)]
    scores = {item["task_id"]: 0.5 for item in items}
    expected = ["first_rank", "earlier", "later"]
    for order in (items, list(reversed(items))):
        selected, runs, trace = probe.select_ranked(order, scores, toy_cost)
        assert [item["candidate_id"] for item in selected] == expected
        assert [row["candidate_id"] for row in trace] == expected
        assert runs == [[0, 1], [2, 3], [6, 7]]
