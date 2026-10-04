"""Single-pass granularity selector with an optional output-run cap.

The callback measures each complete proposed render. Its cost need not be
additive or monotone. This control changes only the run cap, not ranking,
the 1024 budget, or the policy of never revisiting skipped candidates.
"""

from probe_candidate_granularity import merge_spans


def select_with_run_cap(candidates, scores, count, max_runs):
    """Return chosen candidates, merged source runs, and the existing trace schema."""
    if max_runs is not None and (type(max_runs) is not int or max_runs <= 0):
        raise ValueError("max_runs must be a positive integer or None")
    ranked = sorted(candidates, key=lambda c: (-scores[c['task_id']], c['dense_rank'], *c['span']))
    selected, trace = [], []
    for candidate in ranked:
        proposed = merge_spans([c['span'] for c in selected] + [candidate['span']])
        tokens = count(proposed) if max_runs is None or len(proposed) <= max_runs else None
        action = 'admit' if tokens is not None and tokens <= 1024 else 'skip_runs' if tokens is None else 'skip_tokens'
        trace.append({'candidate_id': candidate['candidate_id'], 'action': action,
                      'proposed_runs': len(proposed), 'proposed_tokens': tokens})
        if action == 'admit':
            selected.append(candidate)
    return selected, merge_spans([c['span'] for c in selected]), trace
