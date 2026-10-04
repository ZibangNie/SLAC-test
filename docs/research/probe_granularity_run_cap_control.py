"""Bounded, zero-network run-cap control on two frozen JEV-scored questions."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import analyze_granularity_three_arm as shared
import evaluate_granularity_jev_microdiagnostic as jev
from granularity_lift_control import prepare_parent_layout, lift_parent_max_scores
from granularity_run_cap_control import select_with_run_cap

ROOT = shared.ROOT
PRIOR = jev.LIVE_ROOT / 'analysis-01'
PHASE = ROOT / 'artifacts/research-foundation/offline-20261005/granularity-run-cap-control-01'
OUTPUT = PHASE / 'run-01'
PINNED = {
    'frozen_packs.json': '44b3d04204cca2473211d77a18134029c7889b964ae2857408d3decca6c5d580',
    'verified_judgments.json': '7e5111594b94dec71afec21fdeacf68520e170d60ccccc0603c69245a93a3659',
    'frozen_manifest.json': 'f8a46ebf7dfd211c24bc9842c360fd219330eaa6b4af3b9e580167c3cfeb502f',
    'report.json': '8eeddd40f2e68d591737c9f78f2201ba157ed39d750e0624ef94ba926dd0fc07',
}
GEOMETRY_SHA = 'cf8aabdc64f6c14f3875f809f33381aa486562970ee06f755b809a191933df2c'
SOURCE_NAMES = tuple(dict.fromkeys(shared.SOURCE_NAMES + (
    'docs/research/evaluate_granularity_jev_microdiagnostic.py',
    'docs/research/granularity_run_cap_control.py',
    'tests/research/test_granularity_run_cap_control.py',
    'docs/research/probe_granularity_run_cap_control.py',
    'docs/research/GRANULARITY_RUN_CAP_PROTOCOL_20261005.md',
)))
POLICIES = {'cap3': 3, 'no_cap': None}
sha, require, write = shared.sha, shared.require, shared.probe.write


def source_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}


def load_prior(*, include_report=False):
    names = [name for name in PINNED if include_report or name != 'report.json']
    raw = {name: (PRIOR / name).read_bytes() for name in names}
    require(all(sha(value) == PINNED[name] for name, value in raw.items()), 'prior frozen bytes changed')
    decoded = {name: json.loads(value) for name, value in raw.items()}
    require(decoded['frozen_manifest.json']['input_sha256'] == shared.INPUT_SHA256,
            'prior input binding differs')
    return decoded


def build_packs(sample, scores, original, render, count_text):
    """Finish and exactly check all cap3 packs before constructing no_cap packs."""
    packs, audits, count_cache, totals = [], [], {}, {}
    prepared = []
    for case in sample['cases']:
        units, atoms = (case['candidates'][arm] for arm in ('source_units', 'source_atoms'))
        layout = prepare_parent_layout(units, atoms)
        lifted, lifted_scores, audit = lift_parent_max_scores(layout, scores)
        audits.append({'ordinal': case['ordinal'], 'parents': audit})
        domain = shared.probe.positions([u['span'] for u in units])
        require(domain == shared.probe.positions([a['span'] for a in atoms]), 'candidate domains differ')
        require(all(c['span'][1] <= len(case['source_text']) for c in units + atoms), 'span exceeds source')
        prepared.append((case, domain, (
            ('source_units', units, scores), ('parent_max', lifted, lifted_scores), ('source_atoms', atoms, scores))))

    for policy, cap in POLICIES.items():
        for case, domain, arms in prepared:
            def count(runs):
                text = render(case, runs)
                if text not in count_cache:
                    value = count_text(text)
                    require(type(value) is int and value >= 0, 'invalid complete-render count')
                    count_cache[text] = value
                return count_cache[text]

            for arm, candidates, arm_scores in arms:
                chosen, runs, trace = select_with_run_cap(candidates, arm_scores, count, cap)
                require(cap is None or len(runs) <= cap, 'final run cap exceeded')
                require(count(runs) <= 1024, 'final token budget exceeded')
                require(shared.probe.positions(runs) <= domain, 'pack escapes candidate domain')
                pack = {'ordinal': case['ordinal'], 'arm': arm, 'chosen': chosen, 'runs': runs,
                        'trace': trace, 'rendered_text': render(case, runs), 'proxy_tokens': count(runs)}
                packs.append({'run_cap': policy, **pack})
        totals[policy] = len(count_cache)
        if policy == 'cap3':
            replay = [{k: v for k, v in p.items() if k != 'run_cap'} for p in packs]
            # JSON roundtrip normalizes tuple/list representation only, never text or values.
            require(json.loads(json.dumps(replay)) == original, 'cap3 exact replay failed')
            require(totals[policy] == 55, 'original complete-render count changed')
    return packs, audits, count_cache, totals


def cap_comparisons(packs, rows):
    pack_map = {(p['run_cap'], p['ordinal'], p['arm']): p for p in packs}
    row_map = {(r['run_cap'], r['ordinal'], r['annotation_index'], r['arm']): r for r in rows}
    changes = []
    for ordinal in (1, 2):
        for arm in shared.ARMS:
            old, new = (pack_map[policy, ordinal, arm] for policy in POLICIES)
            a, b = (shared.probe.positions(p['runs']) for p in (old, new))
            annotation_changes = []
            for ai in (0, 1):
                left, right = (row_map[policy, ordinal, ai, arm] for policy in POLICIES)
                annotation_changes.append({'annotation_index': ai,
                    'cap3_overlap_chars': left['overlap_chars'], 'no_cap_overlap_chars': right['overlap_chars'],
                    'overlap_chars_delta': right['overlap_chars'] - left['overlap_chars'],
                    'complete_reference_units_delta': right['selected_complete_reference_units'] - left['selected_complete_reference_units']})
            changes.append({'ordinal': ordinal, 'arm': arm, 'left_policy': 'cap3', 'right_policy': 'no_cap',
                'same_source_character_set': a == b, 'cap3_only_chars': len(a - b),
                'no_cap_only_chars': len(b - a), 'shared_chars': len(a & b),
                'same_complete_render': old['rendered_text'] == new['rendered_text'],
                'proxy_tokens_delta': new['proxy_tokens'] - old['proxy_tokens'],
                'output_runs_delta': len(new['runs']) - len(old['runs']),
                'annotation_changes': annotation_changes})
    return changes


def analyze():
    require(not OUTPUT.exists(), 'refusing to overwrite a prior attempt')
    sample, input_raw = shared.load_input()
    prior = load_prior()
    scores = jev.validate_judgments(sample, prior['verified_judgments.json'])
    geometry_raw = (PHASE / 'final_geometry.json').read_bytes()
    require(sha(geometry_raw) == GEOMETRY_SHA, 'independent geometry file changed')
    geometry = json.loads(geometry_raw)
    require(len(geometry['rows']) == 70, 'all original skip_runs required')
    sources = source_hashes()
    OUTPUT.mkdir(exist_ok=False)
    render, count_text, tokenizer_hashes = shared.budget_runtime()
    packs, audits, count_cache, totals = build_packs(sample, scores, prior['frozen_packs.json'], render, count_text)
    require(len(packs) == 12 and len(audits) == 2, 'incomplete pack generation')
    write(OUTPUT / 'frozen_packs.json', packs)
    write(OUTPUT / 'parent_max_audit.json', audits)
    write(OUTPUT / 'budget_counts.json', [{'rendered_sha256': sha(t.encode()), 'proxy_tokens': n}
                                         for t, n in count_cache.items()])
    freeze = {'schema': 'slac-granularity-run-cap-freeze-v1',
        'status': 'frozen_before_reference_read', 'input_sha256': sha(input_raw), 'prior_sha256': PINNED,
        'source_sha256': sources, 'tokenizer_sha256': tokenizer_hashes, 'policies': POLICIES,
        'complete_evidence_block_proxy_budget': 1024, 'original_cap3_replay_exact': True,
        'artifacts_sha256': {name: sha((OUTPUT / name).read_bytes()) for name in
                            ('frozen_packs.json', 'parent_max_audit.json', 'budget_counts.json')},
        'budget_unique_strings_cumulative_by_policy': totals,
        'references_read_before_freeze_in_this_run': False, 'prior_reference_exposure': True,
        'api_calls': 0, 'scoring_model_calls': 0}
    write(OUTPUT / 'pack_freeze.json', freeze)

    refs = shared.load_references(sample)
    prior = load_prior(include_report=True)
    availability, rows, comparisons = None, [], []
    for policy in POLICIES:
        policy_packs = [p for p in packs if p['run_cap'] == policy]
        avail, policy_rows, policy_comparisons = shared.evaluate_packs(sample, policy_packs, refs)
        if availability is None:
            availability = avail
        require(availability == avail, 'availability changed across policies')
        if policy == 'cap3':
            require(policy_rows == prior['report.json']['rows'], 'prior cap3 reference metrics differ')
        rows.extend({'run_cap': policy, **r} for r in policy_rows)
        comparisons.extend({'run_cap': policy, **c} for c in policy_comparisons)
    require(len(availability) == 4 and len(rows) == 24 and len(comparisons) == 8,
            'incomplete evaluation denominators')
    scan_counts = [{'ordinal': p['ordinal'], 'arm': p['arm'], 'run_cap': p['run_cap'],
                    'counts': dict(Counter(t['action'] for t in p['trace']))} for p in packs]
    geometry_summary = []
    for ordinal in (1, 2):
        subset = [r for r in geometry['rows'] if r['ordinal'] == ordinal]
        geometry_summary.append({'ordinal': ordinal, 'original_skip_runs': len(subset),
            'final_independent_addition_run_count_histogram': dict(Counter(r['proposed_run_count'] for r in subset)),
            'final_independent_additions_within_three_runs': sum(r['proposed_run_count'] <= 3 for r in subset)})
    report = {'schema': 'slac-granularity-run-cap-result-v1', 'status': 'completed',
        'counts': {'questions': 2, 'original_score_pairs': 117, 'representations': 3, 'policies': 2,
                   'packs': 12, 'annotations': 4, 'evaluation_rows': 24, 'budget_strings_tokenized': len(count_cache)},
        'policies': POLICIES, 'complete_evidence_block_proxy_budget': 1024,
        'original_cap3_replay_exact': True, 'original_cap3_reference_rows_exact': True,
        'availability': availability, 'rows': rows, 'within_policy_comparisons': comparisons,
        'cap_intervention_comparisons': cap_comparisons(packs, rows), 'scan_counts': scan_counts,
        'final_geometry_summary': geometry_summary, 'final_geometry_sha256': GEOMETRY_SHA,
        'input_sha256': sha(input_raw), 'prior_sha256': PINNED, 'source_sha256': sources,
        'tokenizer_sha256': tokenizer_hashes, 'reference_records_sha256': shared.probe.REFERENCE_RECORDS_SHA,
        'pack_freeze_sha256': sha((OUTPUT / 'pack_freeze.json').read_bytes()),
        'resources': {'api_calls': 0, 'scoring_model_calls': 0, 'training_updates': 0,
                      'answer_generations': 0, 'new_questions': 0, 'tokenizer': 'local bge-m3 proxy only'},
        'limitations': [
            'Two previously exposed questions; post-hoc mechanism diagnosis, not independent confirmation.',
            'no_cap does not satisfy the original three-run requirement; greedy paths also change.',
            'All four annotations retained; q1 annotations have identical evidence and are not independent.',
            'Character coverage is not semantic sufficiency, EvidenceF1, AnswerF1, or novelty evidence.',
            'Geometry additions are independent, not cumulative, and do not measure token feasibility.',
            'No learned Refiner, score model, API call, no-filter change, threshold tuning, or revisiting.',
            'Complete evidence block tokens exclude system/query message framing and answer output.']}
    require(source_hashes() == sources and sha(shared.INPUT.read_bytes()) == sha(input_raw), 'source/input drift')
    load_prior(include_report=True)
    require(sha((PHASE / 'final_geometry.json').read_bytes()) == GEOMETRY_SHA, 'geometry drift')
    require(all(sha((OUTPUT / name).read_bytes()) == value for name, value in freeze['artifacts_sha256'].items()),
            'frozen pack artifact drift')
    write(OUTPUT / 'report.json', report)
    return report


if __name__ == '__main__':
    result = analyze()
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'rows', 'cap_intervention_comparisons',
                                            'scan_counts', 'final_geometry_summary')}, ensure_ascii=False))
