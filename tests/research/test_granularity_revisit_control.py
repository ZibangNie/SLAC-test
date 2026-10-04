"""Synthetic revisit checks. Toy costs below are not measured tokenizer facts."""

import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'docs/research'))
from granularity_revisit_control import select_with_revisits
from granularity_run_cap_control import select_with_run_cap
from probe_candidate_granularity import merge_spans


def candidate(name, span, *, task_id=None, rank=0):
    return {'candidate_id': name, 'task_id': task_id or name,
            'dense_rank': rank, 'span': span}


def length_cost(runs):
    return sum(b - a for a, b in runs)


def test_first_round_exactly_matches_legacy_and_later_bridge_enables_revisit():
    items = [candidate('a', [0, 1]), candidate('b', [2, 3]),
             candidate('c', [4, 5]), candidate('skipped', [6, 7]),
             candidate('bridge', [1, 2])]
    scores = dict(a=5, b=4, c=3, skipped=2, bridge=1)
    original = copy.deepcopy(items)
    legacy_chosen, legacy_runs, legacy_trace = select_with_run_cap(items, scores, length_cost, 3)
    chosen, runs, rounds = select_with_revisits(items, scores, length_cost)
    assert rounds[0] == {'round_index': 0, 'chosen': legacy_chosen,
                         'runs': legacy_runs, 'trace': legacy_trace}
    assert legacy_trace[3]['action'] == 'skip_runs'
    assert rounds[1]['trace'] == [{'candidate_id': 'skipped', 'action': 'admit',
                                   'proposed_runs': 3, 'proposed_tokens': 5}]
    assert [c['candidate_id'] for c in chosen] == ['a', 'b', 'c', 'bridge', 'skipped']
    assert runs == [[0, 3], [4, 5], [6, 7]]
    assert rounds[-1]['round_index'] == 2 and rounds[-1]['trace'] == []
    assert len(rounds[0]['chosen']) == 4 and items == original
    chosen[0]['span'][0] = 99
    assert rounds[0]['chosen'][0]['span'] == [0, 1]  # Saved round snapshots are independent.


def test_reused_model_task_ids_preserve_distinct_source_candidates_and_ties():
    items = [candidate('later', [4, 5], task_id='shared', rank=1),
             candidate('first_rank', [8, 9], task_id='shared', rank=0),
             candidate('earlier', [0, 1], task_id='shared', rank=1)]
    for ordered in (items, list(reversed(items))):
        chosen, runs, rounds = select_with_revisits(ordered, {'shared': 0.5}, length_cost)
        assert [c['candidate_id'] for c in chosen] == ['first_rank', 'earlier', 'later']
        assert runs == [[0, 1], [4, 5], [8, 9]]
        assert rounds[-1]['trace'] == []


def test_old_token_rejection_is_recomputed_after_nonmonotone_toy_cost_changes():
    items = [candidate('a', [0, 1]), candidate('b', [2, 3]),
             candidate('c', [4, 5]), candidate('old_token_skip', [1, 2]),
             candidate('bridge', [3, 4])]
    scores = dict(a=5, b=4, c=3, old_token_skip=2, bridge=1)

    def count(runs):
        # Deliberate nonmonotone toy counter, not a natural token measurement.
        return 1025 if runs == [[0, 3], [4, 5]] else length_cost(runs)

    chosen, runs, rounds = select_with_revisits(items, scores, count)
    assert rounds[0]['trace'][3]['action'] == 'skip_tokens'
    assert rounds[1]['trace'] == [{'candidate_id': 'old_token_skip', 'action': 'admit',
                                   'proposed_runs': 1, 'proposed_tokens': 5}]
    assert len(chosen) == 5 and runs == [[0, 5]] and rounds[-1]['trace'] == []


def test_terminal_single_item_maximality_does_not_imply_joint_optimality():
    items = [candidate('a', [0, 100]), candidate('b', [110, 210]),
             candidate('c', [220, 320]), candidate('x', [100, 105]),
             candidate('y', [105, 110])]
    scores = dict(a=5, b=4, c=3, x=2, y=1)

    def count(runs):
        return length_cost(runs) + 240 * len(runs)  # Toy run overhead, not tokens.

    chosen, runs, rounds = select_with_revisits(items, scores, count)
    assert [c['candidate_id'] for c in chosen] == ['a', 'b', 'c']
    assert count(runs) == 1020 and len(rounds) == 2
    assert [r['action'] for r in rounds[-1]['trace']] == ['skip_tokens', 'skip_tokens']
    assert all(r['proposed_tokens'] == 1025 for r in rounds[-1]['trace'])
    for remaining in items[3:]:
        assert count(merge_spans(runs + [remaining['span']])) > 1024
    joint_runs = merge_spans(runs + [c['span'] for c in items[3:]])
    assert len(joint_runs) == 2 and count(joint_runs) == 790


def test_invalid_candidate_domain_fails_before_count_and_empty_domain_terminates():
    def forbidden(_):
        raise AssertionError('invalid or empty domain must not invoke count')

    duplicate = [candidate('same', [0, 1]), candidate('same', [2, 3])]
    overlap = [candidate('a', [0, 2]), candidate('b', [1, 3])]
    for items, message in ((duplicate, 'unique'), (overlap, 'overlap')):
        with pytest.raises(ValueError, match=message):
            select_with_revisits(items, {}, forbidden)
    chosen, runs, rounds = select_with_revisits([], {}, forbidden)
    assert chosen == runs == []
    assert rounds[0] == {'round_index': 0, 'chosen': [], 'runs': [], 'trace': []}
    assert rounds[-1]['trace'] == []
