"""Synthetic score binding and freeze-order tests; no natural input or tokenizer."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

RESEARCH = Path(__file__).resolve().parents[2] / 'docs/research'
sys.path.insert(0, str(RESEARCH))
SPEC = importlib.util.spec_from_file_location('granularity_three_arm', RESEARCH / 'analyze_granularity_three_arm.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def blob(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def score_bundle(sample, values=None):
    input_sha = analysis.sha(blob(sample))
    rows = [{'task_id': p['task_id'], 'query_sha256': analysis.sha(p['query'].encode()),
             'passage_sha256': analysis.sha(p['passage'].encode()),
             'raw_logit': values[p['task_id']] if values else i - 60.5}
            for i, p in enumerate(sample['pairs'])]
    scores = {'schema': analysis.SCORE_SCHEMA, 'input_sha256': input_sha, 'pair_scores': rows}
    summary = {'status': 'completed', 'input_sha256': input_sha, 'pair_count': len(rows), 'query_count': 2,
               'provider': 'local_bge', 'score_kind': 'raw_classification_logit',
               'references_read': False, 'api_calls': 0,
               'contract': {'model_id': analysis.MODEL_ID, 'revision': analysis.MODEL_REVISION,
                            'device': 'cpu', 'dtype': 'float32', 'truncation': False}}
    return scores, summary


def seal(scores, summary):
    scores_raw = blob(scores)
    summary = dict(summary, scores_sha256=analysis.sha(scores_raw))
    summary_raw = blob(summary)
    manifest = {'status': 'completed', 'scores_sha256': analysis.sha(scores_raw),
                'summary_sha256': analysis.sha(summary_raw)}
    return scores_raw, summary_raw, blob(manifest)


def synthetic_pairs():
    return {'pairs': [{'task_id': f'task-{i}', 'query': f'artificial question {i}',
                       'passage': f'artificial passage {i}'} for i in range(117)]}


def test_complete_117_bindings_preserve_inputs_and_report_actual_cpu_metadata():
    sample = synthetic_pairs()
    scores, summary = score_bundle(sample)
    before = deepcopy((sample, scores, summary))
    values, verified = analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))
    assert len(values) == 117 and values['task-0'] == -60.5 and values['task-116'] == 55.5
    assert analysis.public_scoring(verified)['device'] == 'cpu'
    assert (sample, scores, summary) == before


def test_task_and_text_binding_errors_cannot_hide_behind_valid_file_hashes():
    sample = synthetic_pairs()
    for edit in (
            lambda d: d['pair_scores'].pop(),
            lambda d: d['pair_scores'].append(deepcopy(d['pair_scores'][0])),
            lambda d: d['pair_scores'].__setitem__(1, deepcopy(d['pair_scores'][0])),
            lambda d: d['pair_scores'][0].update(task_id='unknown-task'),
            lambda d: d['pair_scores'][0].update(query_sha256='0' * 64),
            lambda d: d['pair_scores'][0].update(passage_sha256='0' * 64)):
        scores, summary = score_bundle(sample)
        edit(scores)
        with pytest.raises(ValueError):
            analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))


def test_nonfinite_scores_and_wrong_provider_status_or_reference_contract_rejected():
    sample = synthetic_pairs()
    for invalid in (True, None, '0.3', float('nan'), float('inf'), -float('inf'), 10 ** 1000):
        scores, summary = score_bundle(sample)
        scores['pair_scores'][0]['raw_logit'] = invalid
        with pytest.raises(ValueError):
            analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))
    for field, value in (('status', 'failed'), ('provider', 'jev'), ('score_kind', 'reported_yes'),
                         ('references_read', True), ('references_read', 0), ('api_calls', 1),
                         ('api_calls', False), ('pair_count', 116), ('query_count', 3),
                         ('input_sha256', 'wrong'), ('contract', None)):
        scores, summary = score_bundle(sample)
        summary[field] = value
        with pytest.raises(ValueError):
            analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))
    for field, value in (('model_id', 'other-model'), ('revision', 'unbound-revision'),
                         ('device', 'cuda:1'), ('dtype', 'float16'), ('truncation', True)):
        scores, summary = score_bundle(sample)
        summary['contract'][field] = value
        with pytest.raises(ValueError):
            analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))
    scores, summary = score_bundle(sample)
    scores['schema'] = 'old-jev-scores'
    with pytest.raises(ValueError, match='schema'):
        analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *seal(scores, summary))


def test_manifest_hash_chain_and_direct_child_directory_boundaries(tmp_path, monkeypatch):
    sample = synthetic_pairs()
    scores, summary = score_bundle(sample)
    valid = seal(scores, summary)
    for index, change in ((0, lambda d: d['pair_scores'][0].update(raw_logit=999)),
                          (1, lambda d: d.update(dtype='float16')),
                          (2, lambda d: d.update(status='failed')),
                          (2, lambda d: d.update(scores_sha256='0' * 64)),
                          (2, lambda d: d.update(summary_sha256='0' * 64))):
        altered = list(valid)
        value = json.loads(altered[index]); change(value); altered[index] = blob(value)
        with pytest.raises(ValueError):
            analysis.validate_score_bundle(sample, analysis.sha(blob(sample)), *altered)
    phase = tmp_path / 'phase'; phase.mkdir(); scores_dir = phase / 'scores'; scores_dir.mkdir()
    monkeypatch.setattr(analysis, 'PHASE', phase)
    assert analysis.phase_child(scores_dir, existing=True) == scores_dir
    assert analysis.phase_child(phase / 'output', existing=False) == phase / 'output'
    for path in (tmp_path / 'outside', phase / 'nested' / 'output', phase):
        with pytest.raises(ValueError):
            analysis.phase_child(path, existing=False)
    with pytest.raises(ValueError):
        analysis.phase_child(scores_dir, existing=False)


def two_cases():
    sample, values = {'cases': [], 'pairs': []}, {}
    for ordinal, text in ((1, 'abcd  wxyz'), (2, 'ABCD  WXYZ')):
        case = {'ordinal': ordinal, 'doc_id': f'synthetic-doc-{ordinal}',
                'question_id': f'synthetic-question-{ordinal}', 'query': f'query {ordinal}', 'source_text': text,
                'candidates': {'source_units': [], 'source_atoms': []}}
        for arm, spans, logits in (('source_units', [(0, 4), (6, 10)], [9, 0]),
                                    ('source_atoms', [(0, 2), (2, 4), (6, 8), (8, 10)], [7, 1, 8, 6])):
            for index, ((a, b), score) in enumerate(zip(spans, logits)):
                task = f'{ordinal}-{arm}-{index}'
                parent = 0 if a < 4 else 1
                case['candidates'][arm].append({'candidate_id': task, 'task_id': task,
                    'parent_native_unit_id': f'parent-{parent}', 'dense_rank': parent, 'span': [a, b]})
                sample['pairs'].append({'task_id': task, 'query': case['query'], 'passage': text[a:b]})
                values[task] = score
        sample['cases'].append(case)
    return sample, values


def fake_render(case, runs):
    return ''.join(case['source_text'][a:b] for a, b in runs)


def test_same_selector_separates_score_lift_from_atom_output_at_exact_budget():
    sample, values = two_cases()
    before = deepcopy((sample, values))
    packs, audit, _ = analysis.build_packs(sample, values, fake_render, lambda text: 256 * len(text))
    assert len(packs) == 6 and len(audit) == 2
    first = {p['arm']: p for p in packs if p['ordinal'] == 1}
    assert first['source_units']['runs'] == [[0, 4]]
    assert first['parent_max']['runs'] == [[6, 10]]
    assert first['source_atoms']['runs'] == [[0, 2], [6, 8]]
    assert all(p['proxy_tokens'] == 1024 for p in packs)
    assert all(any(t['action'] == 'skip_tokens' for t in p['trace']) for p in packs)
    assert audit[0]['parents'][1]['max_score'] == 8
    assert (sample, values) == before


def test_failure_prevents_reference_access_and_success_freezes_all_packs_before_read(tmp_path, monkeypatch):
    sample, values = two_cases()
    input_raw = blob(sample)
    phase = tmp_path / 'phase'; phase.mkdir()
    input_file = phase / 'input.json'; input_file.write_bytes(input_raw)
    score_dir = phase / 'scores'; score_dir.mkdir()
    scores, summary = score_bundle(sample, values)
    def save_bundle():
        for name, raw in zip(('scores.json', 'summary.json', 'run_manifest.json'), seal(scores, summary)):
            (score_dir / name).write_bytes(raw)
    save_bundle()
    monkeypatch.setattr(analysis, 'PHASE', phase)
    monkeypatch.setattr(analysis, 'INPUT', input_file)
    monkeypatch.setattr(analysis, 'PAIR_COUNT', len(sample['pairs']))
    monkeypatch.setattr(analysis, 'load_input', lambda: (sample, input_raw))
    monkeypatch.setattr(analysis, 'code_hashes', lambda: {'synthetic-source': 'fixed-hash'})
    monkeypatch.setattr(analysis, 'budget_runtime', lambda: (fake_render, lambda text: 256 * len(text), {}))
    calls = []
    output = phase / 'output'
    def references_after_freeze(given):
        calls.append('references')
        assert given is sample
        freeze = json.loads((output / 'pack_freeze.json').read_bytes())
        assert len(json.loads((output / 'frozen_packs.json').read_bytes())) == 6
        assert len(json.loads((output / 'parent_max_audit.json').read_bytes())) == 2
        assert freeze['pack_sha256'] == analysis.sha((output / 'frozen_packs.json').read_bytes())
        assert freeze['parent_max_audit_sha256'] == analysis.sha((output / 'parent_max_audit.json').read_bytes())
        assert freeze['reference_records_read_during_this_analysis_before_freeze'] is False
        return {(o, a): {'reference_intervals': [[6, 10]]} for o in (1, 2) for a in (0, 1)}
    monkeypatch.setattr(analysis, 'load_references', references_after_freeze)
    summary['status'] = 'failed'; save_bundle()
    with pytest.raises(ValueError, match='completed'):
        analysis.analyze(score_dir, output)
    assert calls == [] and not output.exists()
    summary['status'] = 'completed'; save_bundle()
    report = analysis.analyze(score_dir, output)
    assert calls == ['references'] and report['counts']['evaluation_rows'] == 12
    assert [r['overlap_chars'] for r in report['rows'][:3]] == [0, 4, 2]
    assert report['availability'][0] == report['availability'][1] | {'annotation_index': 0}
    public = json.dumps(report)
    assert 'synthetic-doc-' not in public and 'synthetic-question-' not in public and 'wxyz' not in public
    assert report['scoring']['device'] == 'cpu'
    with pytest.raises(ValueError, match='existence'):
        analysis.analyze(score_dir, output)
