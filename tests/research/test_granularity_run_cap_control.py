"""Synthetic run-cap controls; costs are toy bytes, not tokenizer measurements."""

import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'docs/research'))
from granularity_run_cap_control import select_with_run_cap
from probe_candidate_granularity import merge_spans, select_ranked


def candidate(name, span, dense_rank=0):
    return {'candidate_id': name, 'task_id': name, 'dense_rank': dense_rank, 'span': span}


def toy_bytes(runs):
    return sum(b - a for a, b in runs)


def test_three_run_control_matches_all_legacy_fields_and_order():
    items = [candidate('a', [0, 1]), candidate('b', [2, 3]),
             candidate('c', [4, 5]), candidate('skipped', [6, 7]),
             candidate('bridge', [1, 2], 1), candidate('later', [3, 4], 1)]
    scores = dict(a=5, b=4, c=3, skipped=2, bridge=1, later=1)
    original = copy.deepcopy(items)
    for ordered in (items, list(reversed(items))):
        legacy_calls, control_calls = [], []

        def legacy_count(runs):
            legacy_calls.append(copy.deepcopy(runs))
            return toy_bytes(runs)

        def control_count(runs):
            control_calls.append(copy.deepcopy(runs))
            return toy_bytes(runs)

        assert select_with_run_cap(ordered, scores, control_count, 3) == select_ranked(ordered, scores, legacy_count)
        assert control_calls == legacy_calls
    assert items == original


def test_tiny_fourth_fragment_is_rejected_only_by_run_cap():
    items = [candidate(str(i), [2 * i, 2 * i + 1]) for i in range(4)]
    scores = {str(i): 4 - i for i in range(4)}
    capped = select_with_run_cap(items, scores, toy_bytes, 3)
    uncapped = select_with_run_cap(items, scores, toy_bytes, None)
    assert capped[2][-1] == {'candidate_id': '3', 'action': 'skip_runs',
                             'proposed_runs': 4, 'proposed_tokens': None}
    assert len(capped[0]) == 3 and len(uncapped[0]) == len(uncapped[1]) == 4
    assert uncapped[2][-1]['action'] == 'admit'
    assert uncapped[2][-1]['proposed_tokens'] == 4


def test_no_cap_still_measures_complete_render_including_headers_and_separators():
    source = 'a b c d e'
    items = [candidate(str(i), [2 * i, 2 * i + 1]) for i in range(5)]
    scores = {str(i): 5 - i for i in range(5)}
    calls = []

    def count(runs):
        rendered = 'H' * 1016 + '|' + '|'.join(source[a:b] for a, b in runs)
        calls.append(rendered)
        return len(rendered.encode('utf-8'))

    chosen, runs, trace = select_with_run_cap(items, scores, count, None)
    assert len(chosen) == len(runs) == 4  # More than three runs are allowed.
    assert len(calls) == 5
    assert [row['proposed_tokens'] for row in trace] == [1018, 1020, 1022, 1024, 1026]
    assert trace[-1]['action'] == 'skip_tokens'
    assert toy_bytes(runs) == 4  # Body-only addition would miss the actual limit.


def test_nonmonotone_budget_skip_continues_but_is_not_revisited_after_bridge():
    items = [candidate('a', [0, 1]), candidate('b', [2, 3]),
             candidate('c', [4, 5]), candidate('skipped', [6, 7]),
             candidate('bridge', [1, 2])]
    scores = dict(a=5, b=4, c=3, skipped=2, bridge=1)

    def count(runs):
        return 1025 if [6, 7] in runs and [0, 3] not in runs else toy_bytes(runs)

    chosen, runs, trace = select_with_run_cap(items, scores, count, None)
    assert [c['candidate_id'] for c in chosen] == ['a', 'b', 'c', 'bridge']
    assert [r['action'] for r in trace] == ['admit', 'admit', 'admit', 'skip_tokens', 'admit']
    assert runs == [[0, 3], [4, 5]]
    assert count(merge_spans(runs + [[6, 7]])) <= 1024
    assert len(trace) == len(items)  # Feasibility after the bridge does not trigger a retry.


def test_invalid_run_caps_are_rejected_before_counting():
    def never_count(_):
        raise AssertionError('invalid cap must fail before counting')

    for invalid in (True, False, 0, -1, 3.0, '3', [], {}):
        with pytest.raises(ValueError, match='positive integer or None'):
            select_with_run_cap([], {}, never_count, invalid)
