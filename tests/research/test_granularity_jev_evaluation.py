"""Fake verified judgments only; no runner, references, model or tokenizer calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

RESEARCH = Path(__file__).resolve().parents[2] / 'docs/research'
sys.path.insert(0, str(RESEARCH))
SPEC = importlib.util.spec_from_file_location('jev_granularity_evaluation', RESEARCH / 'evaluate_granularity_jev_microdiagnostic.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def blob(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def judgments(sample):
    return {'complete': True,
            'labels': {p['task_id']: 'unknown' for p in sample['pairs']},
            'reported_scores': {p['task_id']: {'yes': 0, 'no': 0, 'unknown': 1} for p in sample['pairs']},
            'accounting': {'attempts': 15, 'reserved_usd': '0.09', 'known_cost_usd': '0.01',
                           'unknown_cost_attempts': 0, 'completed_requests': 15, 'automatic_retries': 0},
            'bindings': {'synthetic': '0' * 64}}


def test_complete_117_judgments_keep_zero_unknown_and_nonargmax_choice_unchanged():
    sample = {'pairs': [{'task_id': f'task-{i}'} for i in range(117)]}
    result = judgments(sample)
    result['labels']['task-0'] = 'no'
    result['reported_scores']['task-0'] = {'yes': 1, 'no': 0, 'unknown': 0}
    before = deepcopy(result)
    yes = evaluation.validate_judgments(sample, result)
    assert len(yes) == 117 and yes['task-0'] == 1 and sum(v == 0 for v in yes.values()) == 116
    assert result == before


def test_incomplete_identity_invalid_choice_dimensions_and_score_domains_fail_closed():
    sample = {'pairs': [{'task_id': f'task-{i}'} for i in range(117)]}
    mutations = [
        lambda r: r.update(complete=False),
        lambda r: r['labels'].pop('task-0'),
        lambda r: r['reported_scores'].pop('task-0'),
        lambda r: r['labels'].update(extra='yes'),
        lambda r: r['labels'].update({'task-0': True}),
        lambda r: r['reported_scores']['task-0'].pop('no'),
        lambda r: r['reported_scores']['task-0'].update(extra=0),
        lambda r: r['reported_scores'].update({'task-0': {'yes': 0.984, 'no': 0, 'unknown': 0}}),
        lambda r: r['reported_scores'].update({'task-0': {'yes': 1, 'no': 0.016, 'unknown': 0}}),
    ]
    for value in (True, '0', None, float('nan'), float('inf'), -0.01, 1.01):
        mutations.append(lambda r, v=value: r['reported_scores']['task-0'].update(yes=v))
    for field, value in (('attempts', 14), ('completed_requests', True), ('unknown_cost_attempts', 1),
                         ('automatic_retries', 1), ('reserved_usd', '0.11'), ('known_cost_usd', '-0.01'),
                         ('known_cost_usd', 'C:/private/path'), ('known_cost_usd', 'NaN')):
        mutations.append(lambda r, f=field, v=value: r['accounting'].update({f: v}))
    for mutate in mutations:
        result = judgments(sample); mutate(result)
        with pytest.raises(ValueError):
            evaluation.validate_judgments(sample, result)
    for boundary in (0.985, 1.015):
        result = judgments(sample)
        result['reported_scores']['task-0'] = {'yes': 0.5, 'no': boundary - 0.5, 'unknown': 0}
        assert len(evaluation.validate_judgments(sample, result)) == 117


def fake_environment(tmp_path, monkeypatch):
    sample = {'cases': [], 'pairs': []}
    for ordinal, source in ((1, 'abcd  wxyz'), (2, 'ABCD  WXYZ')):
        case = {'ordinal': ordinal, 'doc_id': f'fake-doc-{ordinal}', 'question_id': f'fake-question-{ordinal}',
                'query': f'fake query {ordinal}', 'source_text': source,
                'candidates': {'source_units': [], 'source_atoms': []}}
        for arm, spans in (('source_units', [(0, 4), (6, 10)]), ('source_atoms', [(0, 2), (2, 4), (6, 8), (8, 10)])):
            for index, (a, b) in enumerate(spans):
                task = f'{ordinal}-{arm}-{index}'
                parent = 0 if a < 4 else 1
                case['candidates'][arm].append({'candidate_id': task, 'task_id': task,
                    'parent_native_unit_id': f'parent-{parent}', 'dense_rank': parent, 'span': [a, b]})
                sample['pairs'].append({'task_id': task, 'query': case['query'], 'passage': source[a:b]})
        sample['cases'].append(case)
    live = tmp_path / 'live'; live.mkdir()
    plan = live / 'plan'; plan.mkdir()
    input_file = tmp_path / 'input.json'; input_raw = blob(sample); input_file.write_bytes(input_raw)
    response = plan / 'response.json'; response.write_bytes(b'fake saved response')
    result = judgments(sample)
    result['bindings'] = {str(p.resolve()): evaluation.sha(p.read_bytes()) for p in (input_file, response)}
    source_hashes = {'synthetic-code': 'a' * 64}
    monkeypatch.setattr(evaluation, 'LIVE_ROOT', live)
    monkeypatch.setattr(evaluation, 'INPUT', input_file)
    monkeypatch.setattr(evaluation, 'INPUT_SHA256', evaluation.sha(input_raw))
    monkeypatch.setattr(evaluation, 'PAIR_COUNT', len(sample['pairs']))
    monkeypatch.setattr(evaluation, 'load_input', lambda: (sample, input_raw))
    monkeypatch.setattr(evaluation, 'verify_run', lambda directory: deepcopy(result))
    monkeypatch.setattr(evaluation, 'code_hashes', lambda: dict(source_hashes))
    monkeypatch.setattr(evaluation, 'tokenizer_bindings', lambda hashes: {})
    monkeypatch.setattr(evaluation.shared, 'budget_runtime', lambda: (
        lambda case, spans: ''.join(case['source_text'][a:b] for a, b in spans), lambda text: 256 * len(text), {}))
    reference_calls = []
    def references(given):
        reference_calls.append('read')
        return {(o, a): {'reference_intervals': [[0, 4]]} for o in (1, 2) for a in (0, 1)}
    monkeypatch.setattr(evaluation.shared, 'load_references', references)
    return live, plan, input_file, response, sample, result, source_hashes, reference_calls


def test_zero_no_unknown_candidates_reach_all_three_selectors_before_any_reference(tmp_path, monkeypatch):
    live, plan, _, _, sample, result, _, refs = fake_environment(tmp_path, monkeypatch)
    result['labels'][sample['pairs'][0]['task_id']] = 'no'
    output = live / 'evaluation'
    receipt = evaluation.freeze(plan, output)
    assert receipt['packs'] == 6 and receipt['references_read'] is False and refs == []
    packs = json.loads((output / 'frozen_packs.json').read_bytes())
    assert len(packs) == 6 and all(p['chosen'] and p['proxy_tokens'] == 1024 for p in packs)
    for p in packs:
        assert len(p['trace']) == (4 if p['arm'] == 'source_atoms' else 2)
        assert any(t['action'] == 'skip_tokens' for t in p['trace'])
    max_audit = json.loads((output / 'parent_max_audit.json').read_bytes())
    assert all(parent['max_score'] == 0 for row in max_audit for parent in row['parents'])
    assert 'label' not in json.dumps(max_audit)
    manifest = json.loads((output / 'frozen_manifest.json').read_bytes())
    assert manifest['policy']['exclude_no'] is False and manifest['policy']['unknown_fallback'] is False
    assert manifest['policy']['score_kind'] == 'jev_reported_yes_score'
    for name, expected in manifest['artifact_sha256'].items():
        assert evaluation.sha((output / name).read_bytes()) == expected


def test_failed_verifier_or_input_binding_never_reads_references_or_creates_output(tmp_path, monkeypatch):
    live, plan, input_file, response, _, result, _, refs = fake_environment(tmp_path, monkeypatch)
    output = live / 'evaluation'
    result['complete'] = False
    with pytest.raises(ValueError, match='complete'):
        evaluation.freeze(plan, output)
    result['complete'] = True
    result['bindings'][str(input_file.resolve())] = '0' * 64
    with pytest.raises(ValueError, match='exact selected input'):
        evaluation.freeze(plan, output)
    result['bindings'][str(input_file.resolve())] = evaluation.sha(input_file.read_bytes())
    response.write_bytes(b'changed fake response')
    with pytest.raises(ValueError, match='binding changed'):
        evaluation.freeze(plan, output)
    assert refs == [] and not output.exists()


def test_source_response_and_pack_drift_reject_before_reference_access(tmp_path, monkeypatch):
    live, plan, _, response, _, _, sources, refs = fake_environment(tmp_path, monkeypatch)
    output = live / 'evaluation'; evaluation.freeze(plan, output)
    sources['synthetic-code'] = 'b' * 64
    with pytest.raises(ValueError, match='source changed'):
        evaluation.score(output)
    sources['synthetic-code'] = 'a' * 64
    response.write_bytes(b'changed fake response')
    with pytest.raises(ValueError, match='binding changed'):
        evaluation.score(output)
    response.write_bytes(b'fake saved response')
    path = output / 'frozen_packs.json'; path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluation.score(output)
    assert refs == [] and not (output / 'report.json').exists()


def test_separate_score_requires_complete_freeze_then_keeps_all_12_rows_and_anonymous_report(tmp_path, monkeypatch):
    live, plan, _, _, _, _, _, refs = fake_environment(tmp_path, monkeypatch)
    for invalid in (tmp_path / 'outside', live / 'nested' / 'output'):
        with pytest.raises(ValueError, match='direct LIVE_ROOT'):
            evaluation.freeze(plan, invalid)
    output = live / 'evaluation'; receipt = evaluation.freeze(plan, output)
    assert refs == []
    def checked_reference_read(sample):
        manifest = json.loads((output / 'frozen_manifest.json').read_bytes())
        assert evaluation.sha((output / 'frozen_manifest.json').read_bytes()) == receipt['frozen_manifest_sha256']
        for name, expected in manifest['artifact_sha256'].items():
            assert evaluation.sha((output / name).read_bytes()) == expected
        assert len(json.loads((output / 'frozen_packs.json').read_bytes())) == 6
        assert len(json.loads((output / 'parent_max_audit.json').read_bytes())) == 2
        refs.append('read')
        return {(o, a): {'reference_intervals': [[0, 4]]} for o in (1, 2) for a in (0, 1)}
    monkeypatch.setattr(evaluation.shared, 'load_references', checked_reference_read)
    report = evaluation.score(output)
    assert refs == ['read'] and report['counts']['evaluation_rows'] == 12
    assert len(report['availability']) == 4 and len(report['comparisons']) == 4
    assert all(row['overlap_chars'] == 4 for row in report['rows'])
    assert all(row['delta_tokens_right_minus_left'] == 0 for row in report['comparisons'])
    assert report['provider'] == 'typesafe' and report['score_kind'] == 'jev_reported_yes_score'
    assert report['resources']['evaluation_api_calls'] == 0 and report['execution_accounting']['attempts'] == 15
    text = json.dumps(report)
    assert all(private not in text for private in ('fake-doc-', 'fake-question-', 'abcd', str(plan)))
    with pytest.raises(ValueError, match='not overwritten'):
        evaluation.score(output)
