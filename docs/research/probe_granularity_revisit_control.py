"""Fixed zero-API revisit diagnostic, gated by six independent budget checks."""
from __future__ import annotations

from collections import Counter
import json

import analyze_granularity_three_arm as shared
import evaluate_granularity_jev_microdiagnostic as jev
import probe_granularity_run_cap_control as prior_control
from granularity_lift_control import prepare_parent_layout, lift_parent_max_scores
from granularity_revisit_control import select_with_revisits

ROOT = shared.ROOT
PHASE = ROOT / 'artifacts/research-foundation/offline-20261005/granularity-revisit-control-01'
OUTPUT = PHASE / 'revisit-01'
GATE_SHA = '19b6b23008f3ef10e85e82333a066b049e8d94441105ec0f29e6d8c218cf0c42'
GATE_FREEZE_SHA = '9a9cd3c2fa880c3cc6f586e844d1c64271c055e19079e13e4cae3d5687397af2'
POLICIES = ('single_pass', 'revisit')
SOURCE_NAMES = tuple(dict.fromkeys(shared.SOURCE_NAMES + (
    'docs/research/evaluate_granularity_jev_microdiagnostic.py',
    'docs/research/probe_granularity_run_cap_control.py',
    'docs/research/granularity_run_cap_control.py',
    'docs/research/GRANULARITY_REVISIT_PROTOCOL_20261005.md',
    'docs/research/probe_granularity_final_additions.py',
    'docs/research/granularity_revisit_control.py',
    'tests/research/test_granularity_revisit_control.py',
    'docs/research/probe_granularity_revisit_control.py',
)))
sha, require, write = shared.sha, shared.require, shared.probe.write


def source_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}


def load_gate():
    raw = (PHASE / 'additions-01/report.json').read_bytes()
    require(sha(raw) == GATE_SHA, 'fixed independent additions report changed')
    gate = json.loads(raw)
    require(gate['status'] == 'completed' and gate['counts']['independent_additions'] == 6,
            'six additions must be completed')
    require(gate['counts']['within_run_and_token_caps'] > 0, 'no feasible addition; revisit gate is closed')
    require(len(gate['rows']) == 6 and all(r['proposed_run_count'] <= 3 for r in gate['rows']), 'geometry gate differs')
    require(sum(r['within_1024'] for r in gate['rows']) == gate['counts']['within_run_and_token_caps'],
            'budget gate count differs')
    require(sha((PHASE / 'additions-01/freeze.json').read_bytes()) == gate['freeze_sha256'] == GATE_FREEZE_SHA,
            'independent additions freeze changed')
    require(sha((PHASE / 'additions-01/hypothetical_packs.json').read_bytes()) == gate['hypothetical_packs_sha256'],
            'independent additions packs changed')
    require(sha((ROOT / 'docs/research/probe_granularity_final_additions.py').read_bytes()) == gate['source_sha256'],
            'independent additions code changed')
    return gate


def compare_policies(packs, rows):
    index = {(p['policy'], p['ordinal'], p['arm']): p for p in packs}
    metric = {(r['policy'], r['ordinal'], r['annotation_index'], r['arm']): r for r in rows}
    comparisons = []
    for ordinal in (1, 2):
        for arm in shared.ARMS:
            a, b = (index[p, ordinal, arm] for p in POLICIES)
            left, right = (shared.probe.positions(p['runs']) for p in (a, b))
            require(left <= right, 'revisit removed source characters')
            changes = []
            for ai in (0, 1):
                old, new = (metric[p, ordinal, ai, arm] for p in POLICIES)
                require(new['overlap_chars'] >= old['overlap_chars'], 'coverage contradicts set inclusion')
                changes.append({'annotation_index': ai, 'single_pass_overlap_chars': old['overlap_chars'],
                    'revisit_overlap_chars': new['overlap_chars'],
                    'overlap_chars_delta': new['overlap_chars'] - old['overlap_chars'],
                    'complete_reference_units_delta': new['selected_complete_reference_units'] - old['selected_complete_reference_units']})
            comparisons.append({'ordinal': ordinal, 'arm': arm,
                'same_source_character_set': left == right, 'original_source_is_subset': True,
                'added_source_chars': len(right - left), 'removed_source_chars': 0,
                'proxy_tokens_delta': b['proxy_tokens'] - a['proxy_tokens'],
                'output_runs_delta': len(b['runs']) - len(a['runs']),
                'added_input_blocks': len(b['chosen']) - len(a['chosen']),
                'same_complete_render': a['rendered_text'] == b['rendered_text'], 'annotation_changes': changes})
    return comparisons


def run():
    require(not OUTPUT.exists(), 'refusing to overwrite an earlier attempt')
    gate = load_gate()
    sample, input_raw = shared.load_input()
    prior = prior_control.load_prior()
    scores = jev.validate_judgments(sample, prior['verified_judgments.json'])
    sources = source_hashes()
    OUTPUT.mkdir(exist_ok=False)
    render, runtime_count, tokenizer_hashes = shared.budget_runtime()
    counts = {}

    def count_text(text):
        if text not in counts:
            value = runtime_count(text)
            require(type(value) is int and value >= 0, 'invalid complete-render token count')
            counts[text] = value
        return counts[text]

    # Prove all six old packs first; no revisit outputs precede this check.
    baseline, max_audit, baseline_strings = shared.build_packs(sample, scores, render, count_text)
    require(json.loads(json.dumps(baseline)) == prior['frozen_packs.json'], 'original six-pack replay failed')
    require(baseline_strings == len(counts) == 55, 'original counting contract changed')
    original = {(p['ordinal'], p['arm']): p for p in baseline}
    packs = [{'policy': 'single_pass', **p} for p in baseline]
    round_summaries, maximality = [], []
    for case in sample['cases']:
        ordinal = case['ordinal']
        units, atoms = (case['candidates'][name] for name in ('source_units', 'source_atoms'))
        lifted, lifted_scores, _ = lift_parent_max_scores(prepare_parent_layout(units, atoms), scores)
        domain = shared.probe.positions([u['span'] for u in units])

        def count(runs):
            return count_text(render(case, runs))

        for arm, candidates, arm_scores in (
                ('source_units', units, scores), ('parent_max', lifted, lifted_scores), ('source_atoms', atoms, scores)):
            chosen, runs, rounds = select_with_revisits(candidates, arm_scores, count)
            first = rounds[0]
            first_pack = {'ordinal': ordinal, 'arm': arm, 'chosen': first['chosen'], 'runs': first['runs'],
                          'trace': first['trace'], 'rendered_text': render(case, first['runs']),
                          'proxy_tokens': count(first['runs'])}
            require(json.loads(json.dumps(first_pack)) == original[ordinal, arm], 'revisit first pass changed')
            require(len(runs) <= 3 and count(runs) <= 1024, 'final original budget failed')
            require(shared.probe.positions(original[ordinal, arm]['runs']) <= shared.probe.positions(runs) <= domain,
                    'source inclusion/domain failed')
            require(not any(t['action'] == 'admit' for t in rounds[-1]['trace']), 'missing zero-admission terminal pass')
            require(len(rounds) <= len(candidates) + 1, 'finite progress bound exceeded')
            for r in rounds:
                round_summaries.append({'ordinal': ordinal, 'arm': arm, 'round_index': r['round_index'],
                    'counts': dict(Counter(t['action'] for t in r['trace'])),
                    'selected_input_blocks_after': len(r['chosen']), 'output_runs_after': len(r['runs']),
                    'proxy_tokens_after': count(r['runs'])})
            chosen_ids = {c['candidate_id'] for c in chosen}
            require(len(chosen_ids) == len(chosen), 'candidate was selected twice')
            remaining = [c for c in candidates if c['candidate_id'] not in chosen_ids]
            independent_checks = []
            for candidate in remaining:
                proposed = shared.probe.merge_spans(runs + [candidate['span']])
                token_count = count(proposed) if len(proposed) <= 3 else None
                feasible = token_count is not None and token_count <= 1024
                require(not feasible, 'terminal pack still admits an individual candidate')
                independent_checks.append({'candidate_id': candidate['candidate_id'], 'proposed_runs': len(proposed),
                                           'proposed_tokens': token_count, 'feasible': feasible})
            maximality.append({'ordinal': ordinal, 'arm': arm, 'remaining_candidates': len(remaining),
                               'feasible_individual_additions': 0, 'checks': independent_checks})
            packs.append({'policy': 'revisit', 'ordinal': ordinal, 'arm': arm, 'chosen': chosen, 'runs': runs,
                          'rounds': rounds, 'rendered_text': render(case, runs), 'proxy_tokens': count(runs)})
    require(len(packs) == 12 and len(maximality) == 6, 'incomplete pack set')
    write(OUTPUT / 'frozen_packs.json', packs)
    write(OUTPUT / 'parent_max_audit.json', max_audit)
    write(OUTPUT / 'terminal_checks.json', maximality)
    write(OUTPUT / 'budget_counts.json', [{'rendered_sha256': sha(t.encode()), 'proxy_tokens': n} for t, n in counts.items()])
    freeze = {'schema': 'slac-granularity-revisit-freeze-v1', 'status': 'frozen_before_reference_read',
        'input_sha256': sha(input_raw), 'prior_sha256': prior_control.PINNED,
        'additions_gate_sha256': GATE_SHA, 'additions_freeze_sha256': GATE_FREEZE_SHA,
        'source_sha256': sources, 'tokenizer_sha256': tokenizer_hashes,
        'policies': list(POLICIES), 'max_output_runs': 3, 'max_complete_evidence_block_proxy_tokens': 1024,
        'all_original_six_packs_exact': True, 'all_revisit_first_passes_exact': True,
        'artifacts_sha256': {n: sha((OUTPUT / n).read_bytes()) for n in
                            ('frozen_packs.json', 'parent_max_audit.json', 'terminal_checks.json', 'budget_counts.json')},
        'budget_unique_strings': len(counts), 'references_read_before_freeze_in_this_run': False,
        'prior_reference_exposure': True, 'api_calls': 0, 'scoring_model_calls': 0}
    write(OUTPUT / 'pack_freeze.json', freeze)

    refs = shared.load_references(sample)
    prior = prior_control.load_prior(include_report=True)
    availability, rows, comparisons = None, [], []
    for policy in POLICIES:
        avail, new_rows, new_comp = shared.evaluate_packs(sample, [p for p in packs if p['policy'] == policy], refs)
        if availability is None:
            availability = avail
        require(availability == avail, 'candidate availability differs')
        if policy == 'single_pass':
            require(new_rows == prior['report.json']['rows'], 'original reference metrics differ')
        rows.extend({'policy': policy, **r} for r in new_rows)
        comparisons.extend({'policy': policy, **r} for r in new_comp)
    require(len(rows) == 24 and len(availability) == 4 and len(comparisons) == 8, 'incomplete reference denominators')
    report = {'schema': 'slac-granularity-revisit-result-v1', 'status': 'completed',
        'counts': {'questions': 2, 'original_judgments': 117, 'annotations': 4, 'representations': 3,
                   'policies': 2, 'packs': 12, 'evaluation_rows': 24, 'budget_strings_tokenized': len(counts)},
        'max_output_runs': 3, 'max_complete_evidence_block_proxy_tokens': 1024,
        'availability': availability, 'rows': rows, 'within_policy_comparisons': comparisons,
        'policy_comparisons': compare_policies(packs, rows), 'round_summaries': round_summaries,
        'terminal_maximality': [{k: v for k, v in m.items() if k != 'checks'} for m in maximality],
        'additions_gate': {'report_sha256': GATE_SHA, 'counts': gate['counts'], 'rows': gate['rows']},
        'input_sha256': sha(input_raw), 'prior_sha256': prior_control.PINNED,
        'source_sha256': sources, 'tokenizer_sha256': tokenizer_hashes,
        'pack_freeze_sha256': sha((OUTPUT / 'pack_freeze.json').read_bytes()),
        'reference_records_sha256': shared.probe.REFERENCE_RECORDS_SHA,
        'checks': {'original_six_packs_exact': True, 'all_revisit_first_passes_exact': True,
                   'original_reference_rows_exact': True, 'original_source_subset_all_arms': True,
                   'terminal_single_addition_maximality_all_arms': True},
        'resources': {'api_calls': 0, 'scoring_model_calls': 0, 'training_updates': 0,
                      'answer_generations': 0, 'new_questions': 0},
        'limitations': [
            'Two exposed questions and known references; post-hoc diagnosis, not independent confirmation.',
            'Only additions are allowed: reference overlap/recall non-decrease follows from set inclusion.',
            'No guarantee for precision, answer quality, global optimality, joint additions, or replacements.',
            'Identical q1 annotations are not independent successes; both q2 scopes are retained.',
            'No score change, choice filter, learned Refiner, answer generation, or novelty claim.',
            'Tokens cover only the complete evidence-block proxy; realized lengths may differ within the same cap.']}
    require(source_hashes() == sources and sha(shared.INPUT.read_bytes()) == sha(input_raw), 'source/input drift')
    require(all(sha((OUTPUT / n).read_bytes()) == h for n, h in freeze['artifacts_sha256'].items()), 'frozen artifact drift')
    load_gate()
    prior_control.load_prior(include_report=True)
    write(OUTPUT / 'report.json', report)
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'rows', 'policy_comparisons',
                                            'round_summaries', 'terminal_maximality')}, ensure_ascii=False))
