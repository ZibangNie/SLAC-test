"""Freeze four JEV packs before decoding four selected Qasper references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/research-foundation'
PREP = ART / 'offline-20261004/score-transfer-prep-01'
REFERENCE = ART / 'qasper-confirmation-export-run-02/references.jsonl'
REFERENCE_SHA = '2c9b541a43109f5072ed7416411f8eaa9c7e14e7e6082ec4cfff1ef2831f9269'
METHODS = ('dense_k3', 'reranker_k3', 'p_yes_only_k3')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

def contained(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ART.resolve()) and path != ART.resolve(), 'output must be a private artifact subdirectory')
    return path

def frozen_inputs():
    handoff = read(PREP / 'handoff.json')
    for name, expected in handoff['sealed_inputs'].items():
        require(digest(PREP / name) == expected, 'preparation changed')
    return read(PREP / 'bounded_prepared.json'), read(PREP / 'baseline_replay.json'), read(PREP / 'selected_identities.json')

def freeze(plan_path, output):
    """No gold/reference file, key, network or inference is accessed here."""
    import run_score_transfer_microdiagnostic as runner
    from analyze_qasper_jev_probability import probability_ranking
    from prepare_qasper_confirmation_candidates import tokenizer_for, DEFAULT_MODEL, TOKENIZER_HASHES
    from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack
    result = runner.verify_run(plan_path)
    require(result['complete'], 'all eight requests and 64 judgments required before pack freezing')
    prepared, baselines, identities = frozen_inputs()
    expected = {task['id'] for task in prepared['support_tasks']}
    require(set(result['labels']) == set(result['reported_scores']) == expected and len(expected) == 64, 'complete judgment identity differs')
    for name, expected_sha in TOKENIZER_HASHES.items():
        require(digest(DEFAULT_MODEL / name) == expected_sha, 'evidence tokenizer changed')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    tokenizer = tokenizer_for(DEFAULT_MODEL)
    support = {(t['doc_id'], t['question_id'], t['unit_id']): t['id'] for t in prepared['support_tasks']}
    rows = []
    for ordinal, (query, baseline) in enumerate(zip(prepared['queries'], baselines, strict=True), 1):
        identity = {k: query[k] for k in ('doc_id', 'family_id', 'question_id')}
        require(identity == baseline['identity'], 'cached baseline identity differs')
        units = [Unit(**u) for u in prepared['documents'][query['doc_id']]]
        task_ids = {uid: support[(query['doc_id'], query['question_id'], uid)] for uid in query['candidate_ids']}
        relevance = {uid: result['labels'][task_id] for uid, task_id in task_ids.items()}
        probabilities = {uid: result['reported_scores'][task_id] for uid, task_id in task_ids.items()}
        ranking = probability_ranking(units, query['candidate_ids'], relevance, probabilities,
                                      query['ranked_ids'], 'p_yes_only', 'reported-scores')
        count = PackCounter(tokenizer, units)
        chosen = pack_ranked(units, ranking, 1024, count, max_units=3)
        rendered = render_pack(units, chosen)
        score_pack = identity | {'method': 'p_yes_only_k3', 'budget': 1024,
            'selected_ids': [units[i].unit_id for i in chosen], 'actual_evidence_tokens': count(chosen),
            'rendered_pack': rendered, 'pack_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
        packs = [baseline['dense'], baseline['bge'], score_pack]
        for pack in packs:
            require({k: pack[k] for k in identity} == identity and pack['budget'] == 1024, 'arm identity differs')
            require(len(pack['selected_ids']) <= 3 and pack['actual_evidence_tokens'] <= 1024, 'arm budget exceeded')
        rows.append({'ordinal': ordinal, 'identity': identity, 'packs': packs})
    require(len(rows) == 4, 'four fixed cases required')
    output = contained(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'frozen_packs.json', rows)
    manifest = {'schema': 'slac-score-transfer-pack-freeze-v1', 'status': 'frozen_before_reference_read',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_path': str(Path(plan_path).resolve()),
        'pack_sha256': digest(output / 'frozen_packs.json'), 'methods': list(METHODS),
        'execution_bindings': result['bindings'], 'reference_expected_sha256': REFERENCE_SHA,
        'selection_sha256': digest(PREP / 'selected_identities.json'),
        'preparation_sha256': digest(PREP / 'bounded_prepared.json'),
        'metric_source_sha256': digest(ROOT / 'docs/research/qasper_metrics.py'),
        'evaluator_sha256': digest(__file__), 'references_read': False, 'api_calls': 0}
    write(output / 'frozen_manifest.json', manifest)
    return {'status': manifest['status'], 'cases': 4, 'packs': 12, 'references_read': False}

def selected_references(identities):
    """Hash the saved bytes, decode only four fixed physical rows, then stop."""
    by_line = {s['question_zero_based_line']: s for s in identities['selected']}
    require(set(by_line) == {0, 4, 10, 17}, 'frozen reference positions differ')
    result, retained, hasher = {}, {}, hashlib.sha256()
    with REFERENCE.open('rb') as stream:
        for line in range(18):
            raw = stream.readline()
            require(bool(raw), 'missing reference prefix row')
            hasher.update(raw)
            if line in by_line:
                retained[line] = raw
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    require(hasher.hexdigest() == REFERENCE_SHA, 'reference source changed')
    for line, raw in retained.items():
        row = json.loads(raw)
        require(set(row) == {'doc_id', 'source_id', 'original_family_id', 'family_id', 'question_id', 'answer_annotations'}, 'reference schema differs')
        identity = by_line[line]
        require(all(row[k] == identity[k] for k in ('doc_id', 'source_id', 'original_family_id', 'question_id'))
                and row['family_id'] == identity['component_key'], 'reference identity differs')
        result[(row['doc_id'], row['question_id'])] = row['answer_annotations']
    require(len(result) == 4, 'reference coverage differs')
    return result

def score(output):
    from qasper_metrics import evidence_metrics
    output = contained(output)
    manifest = read(output / 'frozen_manifest.json')
    require(manifest['status'] == 'frozen_before_reference_read' and manifest['references_read'] is False, 'pack freeze required')
    require(manifest['evaluator_sha256'] == digest(__file__), 'evaluator changed after freezing')
    require(manifest['metric_source_sha256'] == digest(ROOT / 'docs/research/qasper_metrics.py'), 'metric changed after freezing')
    require(digest(output / 'frozen_packs.json') == manifest['pack_sha256'], 'packs changed after freezing')
    for path, expected_sha in manifest['execution_bindings'].items():
        require(digest(path) == expected_sha, 'execution evidence changed')
    prepared, _, identities = frozen_inputs()
    require(digest(PREP / 'selected_identities.json') == manifest['selection_sha256']
            and digest(PREP / 'bounded_prepared.json') == manifest['preparation_sha256'], 'inputs changed')
    packs = read(output / 'frozen_packs.json')
    # This marker is created before the only reference-reading function is called.
    write(output / 'reference_read_started.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
        'pack_sha256': manifest['pack_sha256'], 'expected_lines': [1, 5, 11, 18]})
    references = selected_references(identities)
    records, contrasts = [], []
    for case, query in zip(packs, prepared['queries'], strict=True):
        require(case['identity'] == {k: query[k] for k in case['identity']}, 'case identity mismatch')
        require([p['method'] for p in case['packs']] == list(METHODS), 'three fixed arms required')
        units = {u['unit_id']: u for u in prepared['documents'][query['doc_id']]}
        case_scores = {}
        for pack in case['packs']:
            prediction = [units[uid]['native_text'] for uid in pack['selected_ids']]
            metric = evidence_metrics(prediction, references[(query['doc_id'], query['question_id'])])
            case_scores[pack['method']] = metric['evidence_f1']
            records.append({'ordinal': case['ordinal'], **case['identity'], 'method': pack['method'],
                'official_evidence_f1': metric['evidence_f1'], 'actual_evidence_tokens': pack['actual_evidence_tokens'],
                'selected_units': len(pack['selected_ids']), 'selected_ids': pack['selected_ids'],
                'pack_sha256': pack['pack_sha256']})
        contrasts.append({'ordinal': case['ordinal'], 'score_minus_bge': case_scores[METHODS[2]] - case_scores[METHODS[1]],
                          'score_minus_dense': case_scores[METHODS[2]] - case_scores[METHODS[0]]})
    require(len(records) == 12 and len(contrasts) == 4, 'full planned denominator required')
    aggregate = {method: {field: sum(r[field] for r in records if r['method'] == method) / 4
                          for field in ('official_evidence_f1', 'actual_evidence_tokens', 'selected_units')}
                 for method in METHODS}
    result = {'schema': 'slac-score-transfer-evidence-readout-v1', 'status': 'complete_four_case_descriptive_readout',
        'records': records, 'per_case_contrasts': contrasts, 'aggregate': aggregate,
        'primary_mean_delta': sum(c['score_minus_bge'] for c in contrasts) / 4,
        'secondary_mean_delta': sum(c['score_minus_dense'] for c in contrasts) / 4,
        'reference_read_audit': {'source_sha256': REFERENCE_SHA, 'full_file_hashed_only': True,
            'prefix_lines_read': 18, 'rows_decoded': 4, 'decoded_one_based_lines': [1, 5, 11, 18],
            'unselected_rows_decoded': 0, 'original_archive_read': False},
        'frozen_manifest_sha256': digest(output / 'frozen_manifest.json'),
        'api_calls': 0, 'answer_generation': False, 'answer_f1_computed': False,
        'inferential_statistics': False, 'new_algorithm_claim': False, 'automatic_expansion': False}
    write(output / 'evidence_readout.json', result)
    return {k: result[k] for k in ('status', 'aggregate', 'per_case_contrasts', 'primary_mean_delta', 'secondary_mean_delta')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'score'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    require(args.action != 'freeze' or args.plan is not None, 'freeze requires the execution plan')
    print(json.dumps(freeze(args.plan, args.output) if args.action == 'freeze' else score(args.output), ensure_ascii=False))

if __name__ == '__main__':
    main()
