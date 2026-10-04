"""Freeze complete JEV judgments into three arms before a separate reference read."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Real
from pathlib import Path

import analyze_granularity_three_arm as shared

ROOT, INPUT, INPUT_SHA256 = shared.ROOT, shared.INPUT, shared.INPUT_SHA256
LIVE_ROOT = ROOT / 'artifacts/research-foundation/offline-20261005/jev-granularity-live-01'
PAIR_COUNT = 117
CHOICES = {'yes', 'no', 'unknown'}
SCHEMA = 'slac-granularity-jev-evaluation-v1'
SOURCE_NAMES = tuple(dict.fromkeys(shared.SOURCE_NAMES + (
    'docs/research/evaluate_granularity_jev_microdiagnostic.py',
    'tests/research/test_granularity_jev_evaluation.py',
    'docs/research/run_granularity_jev_microdiagnostic.py',
    'tests/research/test_granularity_jev_microdiagnostic.py',
    'docs/research/GRANULARITY_JEV_MICRODIAGNOSTIC_PROTOCOL_20261005.md',
)))
ARMS = {
    'source_units': {'score': 'fresh whole-source-unit JEV reported yes score', 'selection': 'whole source units'},
    'parent_max': {'score': 'maximum fresh child-atom JEV reported yes score; not a parent probability', 'selection': 'whole source units'},
    'source_atoms': {'score': 'fresh atom JEV reported yes score', 'selection': 'individual source atoms'},
}
POLICY = {
    'score_kind': 'jev_reported_yes_score', 'provider': 'typesafe',
    'ranking': 'descending reported yes; original dense parent rank; source start; source end',
    'exclude_no': False, 'unknown_fallback': False, 'score_threshold': None,
    'parent_label_inheritance': False, 'max_output_runs': 3, 'max_evidence_block_proxy_tokens': 1024,
    'pack': 'original single-pass select_ranked; merge touching spans; skip violations without revisiting',
    'prior_reference_exposure': True, 'references_used_for_scoring_or_selection': False,
}
ACCOUNTING_FIELDS = ('attempts', 'completed_requests', 'reserved_usd', 'known_cost_usd',
                     'unknown_cost_attempts', 'automatic_retries')
sha, require, write = shared.sha, shared.require, shared.probe.write


def code_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}


def verify_run(plan_dir):
    # Importing the evaluator never imports a transport or starts a request.
    from run_granularity_jev_microdiagnostic import verify_run as verify
    return verify(plan_dir)


def child(path, *, existing):
    path = Path(path).resolve()
    require(path.parent == LIVE_ROOT.resolve(), 'directory must be a direct LIVE_ROOT child')
    require(path.is_dir() if existing else not path.exists(), 'directory existence contract violated')
    return path


def load_input():
    return shared.load_input()


def validate_judgments(sample, result):
    """Retain every task, including no/unknown labels and zero reported yes scores."""
    require(isinstance(result, dict) and result.get('complete') is True, 'complete verified judgments required')
    pairs = sample['pairs']
    require(len(pairs) == PAIR_COUNT, 'wrong prepared pair count')
    expected = {p['task_id'] for p in pairs}
    require(len(expected) == PAIR_COUNT, 'duplicate prepared task')
    labels, scores = result.get('labels'), result.get('reported_scores')
    require(isinstance(labels, dict) and isinstance(scores, dict), 'missing judgment maps')
    require(set(labels) == set(scores) == expected, 'complete judgment identity differs')
    yes = {}
    for task in expected:
        require(isinstance(labels[task], str) and labels[task] in CHOICES, 'invalid choice')
        dimensions = scores[task]
        require(isinstance(dimensions, dict) and set(dimensions) == CHOICES, 'exact yes/no/unknown scores required')
        for value in dimensions.values():
            require(not isinstance(value, bool) and isinstance(value, Real), 'score must be nonbool real')
            try:
                valid = math.isfinite(value) and 0 <= value <= 1
            except (OverflowError, ValueError, TypeError):
                valid = False
            require(valid, 'score must be finite and between zero and one')
        total = sum((Decimal(str(value)) for value in dimensions.values()), Decimal(0))
        require(Decimal('0.985') <= total <= Decimal('1.015'), 'reported score sum outside tolerance')
        yes[task] = dimensions['yes']
    accounting = result.get('accounting')
    require(isinstance(accounting, dict) and set(accounting) == set(ACCOUNTING_FIELDS), 'verified accounting fields required')
    for field, expected_value in (('attempts', 15), ('completed_requests', 15),
                                  ('unknown_cost_attempts', 0), ('automatic_retries', 0)):
        require(type(accounting[field]) is int and accounting[field] == expected_value, 'incomplete execution accounting')
    for field in ('reserved_usd', 'known_cost_usd'):
        value = accounting[field]
        require(not isinstance(value, bool) and isinstance(value, (str, int, float)), 'invalid cost type')
        try:
            cost = Decimal(str(value))
        except InvalidOperation:
            raise ValueError('invalid cost value') from None
        require(cost.is_finite() and Decimal(0) <= cost <= Decimal('0.10'), 'cost outside frozen budget')
    require(isinstance(result.get('bindings'), dict) and result['bindings'], 'verified execution bindings required')
    return yes


def verify_bindings(bindings):
    for filename, expected in bindings.items():
        require(isinstance(filename, str) and Path(filename).is_absolute(), 'binding path must be absolute')
        require(isinstance(expected, str) and len(expected) == 64, 'invalid binding hash')
        require(sha(Path(filename).read_bytes()) == expected, 'execution binding changed')


def tokenizer_bindings(hashes):
    from probe_natural_partition_coverage import TOKENIZER, TOKEN_HASHES
    require(hashes == TOKEN_HASHES, 'tokenizer contract differs')
    return {str((TOKENIZER / name).resolve()): expected for name, expected in hashes.items()}


def freeze(plan_dir, output):
    """Verify only saved outputs; no references, keys, API or inference are read."""
    plan_dir, output = child(plan_dir, existing=True), child(output, existing=False)
    sample, input_raw = load_input()
    result = verify_run(plan_dir)
    yes = validate_judgments(sample, result)
    require(result['bindings'].get(str(INPUT.resolve())) == sha(input_raw) == INPUT_SHA256,
            'verified run must bind the exact selected input')
    verify_bindings(result['bindings'])
    sources = code_hashes()
    render, count, token_hashes = shared.budget_runtime()
    token_files = tokenizer_bindings(token_hashes)
    packs, max_audit, string_count = shared.build_packs(sample, yes, render, count)
    require(len(packs) == 6 and len(max_audit) == 2, 'all six packs and two parent audits required')
    require({(p['ordinal'], p['arm']) for p in packs} == {(o, a) for o in (1, 2) for a in ARMS}, 'pack identities differ')
    require(code_hashes() == sources and sha(INPUT.read_bytes()) == sha(input_raw), 'source or input changed')
    verify_bindings(result['bindings'])
    verify_bindings(token_files)
    output.mkdir(exist_ok=False)
    write(output / 'verified_judgments.json', result)
    write(output / 'frozen_packs.json', packs)
    write(output / 'parent_max_audit.json', max_audit)
    manifest = {'schema': SCHEMA, 'status': 'frozen_before_reference_read',
        'plan_dir': str(plan_dir), 'input_sha256': sha(input_raw), 'arms': ARMS, 'policy': POLICY,
        'artifact_sha256': {name: sha((output / name).read_bytes()) for name in
                           ('verified_judgments.json', 'frozen_packs.json', 'parent_max_audit.json')},
        'source_sha256': sources, 'execution_bindings': result['bindings'], 'tokenizer_bindings': token_files,
        'reference_records_sha256': shared.probe.REFERENCE_RECORDS_SHA,
        'budget_strings_tokenized': string_count, 'references_read': False,
        'evaluation_api_calls': 0, 'evaluation_model_calls': 0}
    write(output / 'frozen_manifest.json', manifest)
    return {'status': manifest['status'], 'judgments': PAIR_COUNT, 'packs': 6,
            'references_read': False, 'frozen_manifest_sha256': sha((output / 'frozen_manifest.json').read_bytes())}


def score(output):
    """Read the four fixed reference records only after every frozen binding passes."""
    output = child(output, existing=True)
    require(not (output / 'report.json').exists(), 'existing evaluation is not overwritten')
    manifest_raw = (output / 'frozen_manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    require(manifest['schema'] == SCHEMA and manifest['status'] == 'frozen_before_reference_read'
            and manifest['references_read'] is False, 'reference-free pack freeze required')
    require(manifest['arms'] == ARMS and manifest['policy'] == POLICY, 'selection contract changed')
    require(code_hashes() == manifest['source_sha256'], 'evaluation source changed')
    sample, input_raw = load_input()
    require(sha(input_raw) == manifest['input_sha256'] == INPUT_SHA256, 'selected input changed')
    expected_names = {'verified_judgments.json', 'frozen_packs.json', 'parent_max_audit.json'}
    require(set(manifest['artifact_sha256']) == expected_names, 'incomplete frozen artifacts')
    blobs = {name: (output / name).read_bytes() for name in expected_names}
    require(all(sha(raw) == manifest['artifact_sha256'][name] for name, raw in blobs.items()), 'frozen artifact changed')
    result = json.loads(blobs['verified_judgments.json'])
    validate_judgments(sample, result)
    require(result['bindings'] == manifest['execution_bindings'], 'judgment provenance changed')
    verify_bindings(manifest['execution_bindings'])
    verify_bindings(manifest['tokenizer_bindings'])
    require(manifest['reference_records_sha256'] == shared.probe.REFERENCE_RECORDS_SHA, 'reference contract changed')
    packs = json.loads(blobs['frozen_packs.json'])
    require(len(packs) == 6 and {(p['ordinal'], p['arm']) for p in packs} == {(o, a) for o in (1, 2) for a in ARMS},
            'all six frozen packs required')
    refs = shared.load_references(sample)
    availability, rows, comparisons = shared.evaluate_packs(sample, packs, refs)
    require(len(rows) == 12 and len(availability) == 4, 'all annotation rows required')
    indexed = {(p['ordinal'], p['arm']): p for p in packs}
    for comparison in comparisons:
        left = indexed[comparison['ordinal'], comparison['left_arm']]['proxy_tokens']
        right = indexed[comparison['ordinal'], comparison['right_arm']]['proxy_tokens']
        comparison.update(left_proxy_tokens=left, right_proxy_tokens=right, delta_tokens_right_minus_left=right-left)
    require(code_hashes() == manifest['source_sha256'] and sha(INPUT.read_bytes()) == manifest['input_sha256'],
            'source or input changed during evaluation')
    verify_bindings(manifest['execution_bindings'])
    require((output / 'frozen_manifest.json').read_bytes() == manifest_raw
            and all((output / name).read_bytes() == raw for name, raw in blobs.items()), 'freeze changed during evaluation')
    report = {'schema': SCHEMA, 'status': 'completed', 'provider': 'typesafe', 'score_kind': 'jev_reported_yes_score',
        'arms': ARMS, 'policy': POLICY, 'availability': availability, 'rows': rows, 'comparisons': comparisons,
        'input_sha256': manifest['input_sha256'], 'source_sha256': manifest['source_sha256'],
        'frozen_manifest_sha256': sha(manifest_raw), 'artifact_sha256': manifest['artifact_sha256'],
        'reference_records_sha256': manifest['reference_records_sha256'],
        'execution_accounting': {key: result['accounting'][key] for key in ACCOUNTING_FIELDS},
        'counts': {'questions': 2, 'judgments': PAIR_COUNT, 'annotations': 4, 'methods': 3, 'packs': 6,
                   'evaluation_rows': 12, 'budget_strings_tokenized': manifest['budget_strings_tokenized']},
        'resources': {'evaluation_api_calls': 0, 'evaluation_model_calls': 0, 'training_updates': 0,
                      'new_documents': 0, 'answer_generations': 0},
        'limits': ['Already exposed development comparison, not independent confirmation.',
                   'JEV reported yes scores are not BGE logits; child maximum is not a parent probability.',
                   'Reported choice need not be the score argmax; neither choice nor unknown status filters candidates.',
                   'No-filter ranking was fixed in advance for this diagnostic; it is not claimed optimal.',
                   'A/B changes scores with whole-unit packing fixed; B/C jointly changes aggregation and selectable granularity.',
                   'Candidate counts, max aggregation multiplicity, greedy feasibility and realized token lengths differ.',
                   'Character coverage is not semantic sufficiency, official EvidenceF1 or AnswerF1.',
                   'No learned Refiner partition is evaluated; the source-unit and source-atom candidate pools are fixed.',
                   'All four annotations are retained; missing candidate evidence and duplicate profiles remain.',
                   'Budget covers the whole evidence-block proxy, excluding other messages and output.']}
    write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', nargs=2, type=Path, metavar=('PLAN', 'OUTPUT'))
    mode.add_argument('--score', type=Path, metavar='OUTPUT')
    args = parser.parse_args()
    result = freeze(*args.freeze) if args.freeze else score(args.score)
    print(json.dumps(result if args.freeze else {k: result[k] for k in ('status', 'counts', 'availability', 'rows', 'comparisons')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
