"""Offline three-arm analysis of a separately completed, exactly bound score cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import socket
import sys

# These modules define pure helpers/constants; neither runs a scorer on import.
from granularity_lift_control import prepare_parent_layout, lift_parent_max_scores
import probe_candidate_granularity as probe

ROOT, PHASE, INPUT = probe.ROOT, probe.PHASE, probe.INPUT
INPUT_SHA256 = 'aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3'
PAIR_COUNT = 117
SCORE_SCHEMA = 'slac-granularity-direct-scores-v1'
MODEL_ID = 'BAAI/bge-reranker-v2-m3'
MODEL_REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
ARMS = ('source_units', 'parent_max', 'source_atoms')
SOURCE_NAMES = (
    'docs/research/analyze_granularity_three_arm.py',
    'tests/research/test_granularity_three_arm.py',
    'docs/research/granularity_lift_control.py',
    'tests/research/test_granularity_lift_control.py',
    'docs/research/probe_candidate_granularity.py',
    'docs/research/CANDIDATE_GRANULARITY_PROTOCOL_20261005.md',
    'docs/research/GRANULARITY_PARENT_CONTROL_20261005.md',
    'docs/research/GRANULARITY_THREE_ARM_EXECUTION_20261005.md',
    'docs/research/probe_natural_partition_coverage.py',
    'SLAC/llm/io/schemas.py', 'SLAC/llm/service/renderers.py',
)
ARM_CONTRACT = {
    'source_units': {'score': 'fresh whole-source-unit raw classification logit', 'selection': 'whole source units'},
    'parent_max': {'score': 'maximum fresh child-atom raw classification logit', 'selection': 'whole source units'},
    'source_atoms': {'score': 'fresh atom raw classification logit', 'selection': 'individual source atoms'},
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}


def validate_score_bundle(sample, input_sha256, scores_raw, summary_raw, manifest_raw):
    """Validate same-byte manifests and all task/text bindings before any references."""
    scores, summary, manifest = (json.loads(raw) for raw in (scores_raw, summary_raw, manifest_raw))
    require(all(isinstance(value, dict) for value in (scores, summary, manifest)), 'invalid score envelope')
    require(summary.get('status') == manifest.get('status') == 'completed', 'scoring is not completed')
    require(summary.get('input_sha256') == scores.get('input_sha256') == input_sha256, 'score input hash mismatch')
    require(summary.get('scores_sha256') == manifest.get('scores_sha256') == sha(scores_raw), 'score file hash mismatch')
    require(manifest.get('summary_sha256') == sha(summary_raw), 'summary file hash mismatch')
    require(scores.get('schema') == SCORE_SCHEMA, 'unsupported score schema')
    require(summary.get('provider') == 'local_bge' and summary.get('score_kind') == 'raw_classification_logit',
            'only local BGE raw classification logits are supported')
    require(type(summary.get('pair_count')) is int and summary['pair_count'] == PAIR_COUNT, 'wrong pair count')
    require(type(summary.get('query_count')) is int and summary['query_count'] == 2, 'wrong query count')
    contract = summary.get('contract')
    require(isinstance(contract, dict), 'missing scoring model contract')
    require(contract.get('model_id') == MODEL_ID and contract.get('revision') == MODEL_REVISION,
            'scoring model identity mismatch')
    require((contract.get('device'), contract.get('dtype')) in (('cuda:0', 'float16'), ('cpu', 'float32')),
            'unsupported scoring device/dtype contract')
    require(contract.get('truncation') is False, 'scoring must not truncate')
    require(summary.get('references_read') is False, 'scoring must not read references')
    require(type(summary.get('api_calls')) is int and summary['api_calls'] == 0, 'scoring must have zero API calls')
    pairs = sample['pairs']
    require(isinstance(pairs, list) and len(pairs) == PAIR_COUNT, 'wrong prepared pair count')
    expected = {}
    for pair in pairs:
        require(all(isinstance(pair.get(k), str) for k in ('task_id', 'query', 'passage')), 'invalid prepared pair')
        require(pair['task_id'] and pair['task_id'] not in expected, 'duplicate prepared task')
        expected[pair['task_id']] = pair
    rows = scores.get('pair_scores')
    require(isinstance(rows, list) and len(rows) == PAIR_COUNT, 'incomplete score rows')
    result = {}
    for row in rows:
        require(isinstance(row, dict), 'invalid score row')
        task = row.get('task_id')
        require(isinstance(task, str) and task in expected and task not in result, 'unknown or duplicate score task')
        pair = expected[task]
        require(row.get('query_sha256') == sha(pair['query'].encode('utf-8')), 'query binding mismatch')
        require(row.get('passage_sha256') == sha(pair['passage'].encode('utf-8')), 'passage binding mismatch')
        value = row.get('raw_logit')
        require(not isinstance(value, bool) and isinstance(value, Real), 'logit must be a nonbool real')
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError, TypeError):
            finite = False
        require(finite, 'logit must be finite')
        result[task] = value
    require(set(result) == set(expected), 'missing score task')
    return result, summary


def phase_child(path, *, existing):
    path = Path(path).resolve()
    require(path.parent == PHASE.resolve(), 'directory must be a direct phase child')
    require(path.is_dir() if existing else not path.exists(), 'directory existence contract violated')
    return path


def load_input():
    raw = INPUT.read_bytes()
    require(sha(raw) == INPUT_SHA256, 'frozen input changed')
    sample = json.loads(raw)
    require(sample['input_bindings']['policy'] == probe.POLICY, 'prepared policy changed')
    require(sample['input_bindings']['source_sha256'] == probe.code_hashes(), 'prepared source changed')
    require([case['ordinal'] for case in sample['cases']] == [1, 2], 'wrong fixed cases')
    require([(len(c['candidates']['source_units']), len(c['candidates']['source_atoms']))
             for c in sample['cases']] == [(16, 52), (14, 45)], 'wrong fixed candidate counts')
    return sample, raw


def budget_runtime():
    """Load the original offline tokenizer and renderer, never a model or scorer."""
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in three-arm analysis')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
        os.environ[name] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(ROOT))
    from probe_natural_partition_coverage import TOKENIZER, TOKEN_HASHES
    from SLAC.llm.io.schemas import EvidenceItem
    from SLAC.llm.service.renderers import render_evidence_block
    from transformers import AutoTokenizer
    require(all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items()),
            'tokenizer file changed')
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)

    def render(case, runs):
        return render_evidence_block([
            EvidenceItem(chunk_id=f'span-{a:05d}-{b:05d}', doc_id=case['doc_id'],
                         query_id=case['question_id'], query_text=case['query'], passage_text=case['source_text'][a:b])
            for a, b in runs], preserve_source_text=True)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

    return render, count, dict(TOKEN_HASHES)


def build_packs(sample, scores, render, count_text):
    """Use one selector, source renderer and whole-render count cache for all arms."""
    packs, max_audit, cached_counts = [], [], {}
    for case in sample['cases']:
        units, atoms = (case['candidates'][name] for name in ('source_units', 'source_atoms'))
        layout = prepare_parent_layout(units, atoms)
        lifted, lifted_scores, audit = lift_parent_max_scores(layout, scores)
        max_audit.append({'ordinal': case['ordinal'], 'parents': audit})
        domain = probe.positions([u['span'] for u in units])
        require(domain == probe.positions([a['span'] for a in atoms]), 'candidate domains differ')
        for candidates in (units, atoms):
            for candidate in candidates:
                require(candidate['span'][1] <= len(case['source_text']), 'candidate exceeds source text')

        def count(runs):
            text = render(case, runs)
            if text not in cached_counts:
                value = count_text(text)
                require(type(value) is int and value >= 0, 'invalid whole-render token count')
                cached_counts[text] = value
            return cached_counts[text]

        for arm, candidates, arm_scores in (
                ('source_units', units, scores), ('parent_max', lifted, lifted_scores), ('source_atoms', atoms, scores)):
            chosen, runs, trace = probe.select_ranked(candidates, arm_scores, count)
            require(len(runs) <= 3 and count(runs) <= 1024, 'final output budget failed')
            require(probe.positions(runs) <= domain, 'pack escapes candidate domain')
            packs.append({'ordinal': case['ordinal'], 'arm': arm, 'chosen': chosen, 'runs': runs,
                          'trace': trace, 'rendered_text': render(case, runs), 'proxy_tokens': count(runs)})
    return packs, max_audit, len(cached_counts)


def load_references(sample):
    raw = (probe.PRIOR / 'run-01/records.json').read_bytes()
    require(sha(raw) == probe.REFERENCE_RECORDS_SHA, 'reference records changed')
    selected = [r for r in json.loads(raw) if r['summary']['arm'] == 'raw_b0_source']
    refs = {(r['summary']['ordinal'], r['summary']['annotation_index']): r for r in selected}
    require(len(selected) == len(refs) == 4 and set(refs) == {(o, a) for o in (1, 2) for a in (0, 1)},
            'four original annotations required')
    for case in sample['cases']:
        for ai in (0, 1):
            ref = refs[case['ordinal'], ai]
            require((ref['doc_id'], ref['question_id']) == (case['doc_id'], case['question_id']), 'reference identity mismatch')
            require(ref['summary']['annotation_status'] == 'mapped', 'reference is not mapped')
            for item in ref['mapping_details']:
                require(item['kind'] == 'single_exact_native_unit' and len(item['matched_spans']) == 1,
                        'unsupported reference mapping')
                a, b = item['matched_spans'][0]
                require(0 <= a < b <= len(case['source_text']) and sha(case['source_text'][a:b].encode()) == item['text_sha256'],
                        'reference source text mismatch')
    return refs


def evaluate_packs(sample, packs, refs):
    availability, rows, comparisons = [], [], []
    for case in sample['cases']:
        ordinal = case['ordinal']
        domain = probe.positions([c['span'] for c in case['candidates']['source_units']])
        byarm = {p['arm']: p for p in packs if p['ordinal'] == ordinal}
        require(set(byarm) == set(ARMS), 'missing arm pack')
        for ai in (0, 1):
            intervals = refs[ordinal, ai]['reference_intervals']
            ref = probe.positions(intervals)
            availability.append({'ordinal': ordinal, 'annotation_index': ai, 'reference_units': len(intervals),
                'reference_chars': len(ref), 'pool_reference_chars': len(ref & domain),
                'pool_complete_reference_units': sum(probe.positions([s]) <= domain for s in intervals),
                'pool_full_coverage': ref <= domain})
            for arm in ARMS:
                pack = byarm[arm]
                selected = probe.positions(pack['runs'])
                require(selected <= domain, 'selected characters outside domain')
                hit = len(ref & selected)
                rows.append({'ordinal': ordinal, 'annotation_index': ai, 'arm': arm,
                    'selected_input_blocks': len(pack['chosen']), 'output_runs': len(pack['runs']),
                    'selected_source_chars': len(selected), 'reference_chars': len(ref), 'overlap_chars': hit,
                    'character_precision': hit / len(selected) if selected else 0,
                    'character_recall': hit / len(ref) if ref else 0, 'full_annotation_coverage': ref <= selected,
                    'selected_complete_reference_units': sum(probe.positions([s]) <= selected for s in intervals),
                    'proxy_tokens': pack['proxy_tokens'], 'rendered_sha256': sha(pack['rendered_text'].encode())})
        for left, right in (('source_units', 'parent_max'), ('parent_max', 'source_atoms')):
            a, b = (probe.positions(byarm[name]['runs']) for name in (left, right))
            comparisons.append({'ordinal': ordinal, 'left_arm': left, 'right_arm': right,
                'same_source_character_set': a == b, 'left_only_chars': len(a-b), 'right_only_chars': len(b-a),
                'shared_chars': len(a & b), 'same_complete_render': byarm[left]['rendered_text'] == byarm[right]['rendered_text'],
                'interpretation': 'score aggregation with whole-unit packing fixed' if left == 'source_units'
                                  else 'same child scores, aggregation and selectable granularity jointly differ'})
    return availability, rows, comparisons


def public_scoring(summary):
    # Only the validated model contract; no paths, compute dictionaries or notes.
    result = {key: summary[key] for key in ('provider', 'score_kind', 'pair_count', 'query_count', 'references_read', 'api_calls')}
    for key in ('model_id', 'revision', 'device', 'dtype', 'truncation'):
        result[key] = summary['contract'][key]
    return result


def analyze(score_dir, output_dir):
    score_dir = phase_child(score_dir, existing=True)
    output = phase_child(output_dir, existing=False)
    sample, input_raw = load_input()
    score_files = {name: (score_dir / name).read_bytes() for name in ('scores.json', 'summary.json', 'run_manifest.json')}
    scores, summary = validate_score_bundle(sample, sha(input_raw), *(score_files[n] for n in ('scores.json', 'summary.json', 'run_manifest.json')))
    public_metadata = public_scoring(summary)
    sources = code_hashes()
    output.mkdir(exist_ok=False)
    render, count, tokenizer_hashes = budget_runtime()
    packs, max_audit, string_count = build_packs(sample, scores, render, count)
    require(len(packs) == 6 and len(max_audit) == 2, 'incomplete pack freeze')
    probe.write(output / 'frozen_packs.json', packs)
    probe.write(output / 'parent_max_audit.json', max_audit)
    freeze = {'schema': 'slac-granularity-three-arm-freeze-v1', 'input_sha256': sha(input_raw),
        'score_files_sha256': {n: sha(b) for n, b in score_files.items()}, 'source_sha256': sources,
        'tokenizer_sha256': tokenizer_hashes, 'arms': ARM_CONTRACT, 'packing_policy': probe.POLICY,
        'pack_sha256': sha((output / 'frozen_packs.json').read_bytes()),
        'parent_max_audit_sha256': sha((output / 'parent_max_audit.json').read_bytes()),
        'reference_records_read_during_this_analysis_before_freeze': False}
    probe.write(output / 'pack_freeze.json', freeze)
    refs = load_references(sample)
    availability, rows, comparisons = evaluate_packs(sample, packs, refs)
    require(len(rows) == 12 and len(availability) == 4, 'incomplete evaluation denominator')
    require(code_hashes() == sources and sha(INPUT.read_bytes()) == sha(input_raw), 'source or input drift')
    require(all(sha((score_dir / n).read_bytes()) == sha(b) for n, b in score_files.items()), 'score bundle drift')
    report = {'schema': 'slac-granularity-three-arm-result-v1', 'status': 'completed',
        'arms': ARM_CONTRACT, 'packing_policy': probe.POLICY, 'availability': availability, 'rows': rows,
        'comparisons': comparisons, 'scoring': public_metadata, 'input_sha256': sha(input_raw),
        'score_files_sha256': freeze['score_files_sha256'], 'reference_records_sha256': probe.REFERENCE_RECORDS_SHA,
        'pack_freeze_sha256': sha((output / 'pack_freeze.json').read_bytes()),
        'source_sha256': sources, 'tokenizer_sha256': tokenizer_hashes,
        'counts': {'questions': 2, 'annotations': 4, 'methods': 3, 'packs': 6, 'evaluation_rows': 12,
                   'score_pairs': PAIR_COUNT, 'budget_strings_tokenized': string_count},
        'resources': {'analysis_model_calls': 0, 'api_calls': 0, 'training_updates': 0, 'answer_generations': 0, 'new_documents': 0},
        'limits': ['Previously exposed two-question, post-hoc mechanism control; not independent confirmation.',
                   'A/B changes whole-unit scores to max child scores; B/C also changes selectable granularity and greedy feasibility.',
                   'Candidate counts, maximum aggregation multiplicity and realized context lengths differ.',
                   'Character coverage is not semantic sufficiency, official EvidenceF1, AnswerF1 or novelty evidence.',
                   'All four annotations remain separate, including duplicated evidence profiles and missing candidate evidence.',
                   'Tokens count the complete evidence-block proxy, excluding other messages and generated output.']}
    probe.write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.score_dir, args.output_dir)
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'availability': report['availability'],
                      'rows': report['rows'], 'comparisons': report['comparisons']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
