"""Fixed-order add-only revisits under the original three-run/1024 budget.

The final zero-admission round establishes only single-candidate maximality
for a deterministic full-render counter, not optimality over joint additions.
"""

from copy import deepcopy

from granularity_run_cap_control import select_with_run_cap
from probe_candidate_granularity import merge_spans


def select_with_revisits(candidates, scores, count):
    """Return selected candidates, source runs, and complete per-round snapshots."""
    candidates = list(candidates)
    candidate_ids = [c['candidate_id'] for c in candidates]
    if any(not isinstance(cid, str) or not cid for cid in candidate_ids):
        raise ValueError('candidate_id must be a nonempty string')
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('candidate_id must be unique')
    merge_spans([c['span'] for c in candidates])  # Validate the entire disjoint domain.
    ranked = sorted(candidates, key=lambda c: (-scores[c['task_id']], c['dense_rank'], *c['span']))
    chosen, runs, first_trace = select_with_run_cap(candidates, scores, count, 3)
    rounds = [{'round_index': 0, 'chosen': deepcopy(chosen),
               'runs': deepcopy(runs), 'trace': deepcopy(first_trace)}]
    selected_ids = {c['candidate_id'] for c in chosen}
    while True:
        trace = []
        added = False
        for candidate in ranked:
            if candidate['candidate_id'] in selected_ids:
                continue
            proposed = merge_spans([c['span'] for c in chosen] + [candidate['span']])
            tokens = count(proposed) if len(proposed) <= 3 else None
            action = 'admit' if tokens is not None and tokens <= 1024 else 'skip_runs' if tokens is None else 'skip_tokens'
            trace.append({'candidate_id': candidate['candidate_id'], 'action': action,
                          'proposed_runs': len(proposed), 'proposed_tokens': tokens})
            if action == 'admit':
                chosen.append(candidate)
                selected_ids.add(candidate['candidate_id'])
                added = True
        runs = merge_spans([c['span'] for c in chosen])
        rounds.append({'round_index': len(rounds), 'chosen': deepcopy(chosen),
                       'runs': deepcopy(runs), 'trace': trace})
        if not added:
            return chosen, runs, rounds
