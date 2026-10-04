"""Two exposed questions: fixed candidate-domain, local granularity intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/research-foundation/offline-20261004'
PRIOR = ROOT / 'artifacts/research-foundation/offline-20261005/natural-granularity-coverage-01'
PHASE = ROOT / 'artifacts/research-foundation/offline-20261005/candidate-granularity-mechanism-01'
INPUT = PHASE / 'selected_inputs.json'
PINS = {
    'candidate_inputs': (BASE / 'hard-no-cache-sample-01/bounded_inputs.json', 'f92a1a787dd7a3c5b79075165f8b2356ce0518aeb5503ca310ec44e2bb5fd9ee'),
    'full_source': (BASE / 'refiner-granularity-probe-01/bounded_inputs.json', 'e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c'),
    'exports': (BASE / 'refiner-document-source-export-01/run-01/document_exports.json', 'e28538bf86ed10b452257c887067c21d5fdd1588baef3d964ac24229de87df87'),
}
REFERENCE_RECORDS_SHA = 'd2d157b56126be28d7b95717190f862c6b5caed833dbebefa146c07ee11382f5'
ARMS = ('source_units', 'source_atoms')
POLICY = {
    'sample': 'same exposed two questions; all original 16/14 candidate units; no source expansion',
    'rank': 'descending fresh raw BGE logit; original dense parent rank; source start; source end',
    'pack': 'single ranked pass; merge touching selected source intervals; at most3 output runs and1024 complete evidence-block BGE tokens; skip violations, continue, never revisit',
    'candidate_text_deduplication': False,
    'score_exact_duplicate_query_passage_pairs_once': True,
    'references_used_for_scoring_or_selection': False,
    'prior_reference_exposure': True,
    'no_semantic_score_threshold_or_gold_tuning': True,
    'max_output_runs': 3, 'max_proxy_tokens': 1024,
    'evaluation': 'all four original annotations separately; character coverage only, no official Evidence/Answer F1',
    'no_api_or_training': True,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def identity(value):
    return tuple(value[k] for k in ('family_id', 'doc_id', 'question_id'))


def merge_spans(spans):
    runs = []
    for a, b in sorted(spans):
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError('invalid source span')
        if runs and a < runs[-1][1]:
            raise ValueError('selected source spans overlap')
        if runs and a == runs[-1][1]:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return runs


def positions(spans):
    return {i for a, b in spans for i in range(a, b)}


def select_ranked(candidates, scores, count):
    ranked = sorted(candidates, key=lambda c: (-scores[c['task_id']], c['dense_rank'], *c['span']))
    selected, trace = [], []
    for candidate in ranked:
        proposed = merge_spans([c['span'] for c in selected] + [candidate['span']])
        tokens = count(proposed) if len(proposed) <= 3 else None
        action = 'admit' if tokens is not None and tokens <= 1024 else 'skip_runs' if tokens is None else 'skip_tokens'
        trace.append({'candidate_id': candidate['candidate_id'], 'action': action, 'proposed_runs': len(proposed), 'proposed_tokens': tokens})
        if action == 'admit':
            selected.append(candidate)
    return selected, merge_spans([c['span'] for c in selected]), trace


def code_hashes():
    names = ('docs/research/probe_candidate_granularity.py',
             'docs/research/CANDIDATE_GRANULARITY_PROTOCOL_20261005.md',
             'SLAC/llm/service/renderers.py', 'SLAC/llm/io/schemas.py')
    return {name: sha((ROOT / name).read_bytes()) for name in names}


def prepare():
    blobs = {name: path.read_bytes() for name, (path, _) in PINS.items()}
    assert all(sha(blobs[name]) == expected for name, (_, expected) in PINS.items())
    data = {name: json.loads(raw) for name, raw in blobs.items()}
    bounded, full = data['candidate_inputs'], data['full_source']
    queries = {identity(q): q for q in bounded['queries']}
    native = {d['doc_id']: d for d in full['native_documents']}
    exports = {d['doc_id']: d for d in data['exports']}
    mappings = {(m['doc_id'], m['native_unit_id']): m for m in full['unit_mapping']}
    assert len(native) == len(exports) == len(full['queries']) == 2
    cases, pair_map = [], {}
    for old in sorted(full['queries'], key=lambda q: q['ordinal']):
        ordinal, doc = old['ordinal'], old['doc_id']
        query, document = queries[identity(old)], exports[doc]
        source = native[doc]['native_document_text']
        assert sha(source.encode()) == native[doc]['native_document_text_sha256']
        assert source == document['view']['source_text'] and document['document_ordinal'] == ordinal
        assert query['candidate_ids'] == old['candidate_ids']
        assert len(query['candidate_ids']) == {1: 16, 2: 14}[ordinal]
        assert set(query['candidate_ids']) == set(query['ranked_ids'])
        ranks = {uid: rank for rank, uid in enumerate(query['ranked_ids'])}
        chunks, atoms = document['exports']['raw_b0_source']['refined_chunks'], document['view']['atom_char_spans']
        candidates = {arm: [] for arm in ARMS}
        for uid in query['candidate_ids']:
            a, b = mappings[doc, uid]['raw_native_char_span']
            matches = [chunk for chunk in chunks if chunk['source_char_span'][0] <= a < b <= chunk['source_char_span'][1]]
            assert len(matches) == 1
            chunk = matches[0]
            unit_span = chunk['source_char_span']
            atom_spans = atoms[chunk['atom_start']:chunk['atom_end']]
            assert merge_spans(atom_spans) == [unit_span]
            assert source[unit_span[0]:unit_span[1]] == chunk['text']
            for arm, spans in [('source_units', [unit_span]), ('source_atoms', atom_spans)]:
                for span in spans:
                    passage = source[span[0]:span[1]]
                    key = (query['doc_id'], query['question_id'], query['query'], passage)
                    task = sha(json.dumps(key, ensure_ascii=False, separators=(',', ':')).encode())
                    pair = {'task_id': task, 'unit_id': task, 'doc_id': doc, 'question_id': query['question_id'], 'query': query['query'], 'passage': passage}
                    assert task not in pair_map or pair_map[task] == pair
                    pair_map[task] = pair
                    candidates[arm].append({'candidate_id': f'{arm}:{span[0]}:{span[1]}', 'task_id': task,
                                           'parent_native_unit_id': uid, 'dense_rank': ranks[uid], 'span': span})
        unit_domain = positions([c['span'] for c in candidates['source_units']])
        atom_domain = positions([c['span'] for c in candidates['source_atoms']])
        assert unit_domain == atom_domain
        assert len(unit_domain) == sum(c['span'][1] - c['span'][0] for c in candidates['source_units'])
        assert len(atom_domain) == sum(c['span'][1] - c['span'][0] for c in candidates['source_atoms'])
        cases.append({'ordinal': ordinal, 'doc_id': doc, 'question_id': query['question_id'], 'query': query['query'],
                      'candidates': candidates, 'candidate_domain_chars': len(unit_domain),
                      'old_selected_ids': old['selected_ids'], 'source_text': source})
    pairs = list(pair_map.values())
    assert [c['ordinal'] for c in cases] == [1, 2] and 2 <= len(pairs) <= 281
    assert len({(p['query'], p['passage']) for p in pairs}) == len(pairs)
    PHASE.mkdir(exist_ok=False)
    value = {'schema': 'slac-candidate-granularity-reranker-input-v1', 'cases': cases, 'pairs': pairs,
             'input_bindings': {'files': {name: expected for name, (_, expected) in PINS.items()},
                               'policy': POLICY, 'source_sha256': code_hashes(),
                               'references_read_by_preparation': False}}
    write(INPUT, value)
    print(json.dumps({'status': 'prepared_no_model', 'questions': 2, 'unique_pairs': len(pairs),
          'candidates': [{arm: len(c['candidates'][arm]) for arm in ARMS} for c in cases], 'api_calls': 0}))


def analyze():
    input_raw = INPUT.read_bytes()
    sample = json.loads(input_raw)
    assert sample['input_bindings']['policy'] == POLICY and sample['input_bindings']['source_sha256'] == code_hashes()
    run = PHASE / 'run-01'
    score_raw = (run / 'scores.json').read_bytes()
    scoring_summary = json.loads((run / 'summary.json').read_bytes())
    assert scoring_summary['status'] == 'completed' and scoring_summary['scores_sha256'] == sha(score_raw)
    score_data = json.loads(score_raw)
    plan_raw = (PHASE / 'plan-01/plan.json').read_bytes()
    plan = json.loads(plan_raw)
    assert score_data['plan_sha256'] == scoring_summary['plan_sha256'] == sha(plan_raw)
    assert plan['input_sha256'][str(INPUT)] == sha(input_raw)
    scores = {s['task_id']: s['raw_logit'] for s in score_data['pair_scores']}
    assert len(scores) == len(sample['pairs']) and set(scores) == {p['task_id'] for p in sample['pairs']}
    output = PHASE / 'analysis-01'
    output.mkdir(exist_ok=False)
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in two-question granularity analysis')
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
    assert all(sha((TOKENIZER / n).read_bytes()) == h for n, h in TOKEN_HASHES.items())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    cached_counts = {}
    packs = []
    for case in sample['cases']:
        source = case['source_text']
        def render(runs):
            return render_evidence_block([EvidenceItem(chunk_id=f'span-{a:05d}-{b:05d}', doc_id=case['doc_id'],
                query_id=case['question_id'], query_text=case['query'], passage_text=source[a:b]) for a, b in runs], preserve_source_text=True)
        def count(runs):
            text = render(runs)
            if text not in cached_counts:
                cached_counts[text] = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            return cached_counts[text]
        for arm in ARMS:
            chosen, runs, trace = select_ranked(case['candidates'][arm], scores, count)
            assert len(runs) <= 3 and count(runs) <= 1024
            packs.append({'ordinal': case['ordinal'], 'arm': arm, 'chosen': chosen, 'runs': runs,
                          'trace': trace, 'rendered_text': render(runs), 'proxy_tokens': count(runs)})
    write(output / 'frozen_packs.json', packs)
    freeze = {'pack_sha256': sha((output / 'frozen_packs.json').read_bytes()), 'input_sha256': sha(input_raw),
              'score_sha256': sha(score_raw), 'source_sha256': code_hashes(), 'policy': POLICY,
              'reference_records_read_during_this_analysis_before_freeze': False}
    write(output / 'pack_freeze.json', freeze)
    raw = (PRIOR / 'run-01/records.json').read_bytes()
    assert sha(raw) == REFERENCE_RECORDS_SHA
    refs = {(r['summary']['ordinal'], r['summary']['annotation_index']): r
            for r in json.loads(raw) if r['summary']['arm'] == 'raw_b0_source'}
    assert len(refs) == 4
    availability, rows, comparison = [], [], []
    for case in sample['cases']:
        ordinal = case['ordinal']
        domain = positions([c['span'] for c in case['candidates']['source_units']])
        old_selected = positions([c['span'] for c in case['candidates']['source_units'] if c['parent_native_unit_id'] in case['old_selected_ids']])
        for ai in (0, 1):
            intervals = refs[ordinal, ai]['reference_intervals']
            ref = positions(intervals)
            availability.append({'ordinal': ordinal, 'annotation_index': ai, 'reference_units': len(intervals),
                'pool_complete_reference_units': sum(positions([s]) <= domain for s in intervals),
                'old_selected_complete_reference_units': sum(positions([s]) <= old_selected for s in intervals),
                'reference_chars': len(ref), 'pool_reference_chars': len(ref & domain),
                'old_selected_reference_chars': len(ref & old_selected), 'pool_full_coverage': ref <= domain})
        case_packs = [p for p in packs if p['ordinal'] == ordinal]
        for pack in case_packs:
            selected = positions(pack['runs'])
            assert selected <= domain
            for ai in (0, 1):
                ref = positions(refs[ordinal, ai]['reference_intervals'])
                hit = len(ref & selected)
                rows.append({'ordinal': ordinal, 'annotation_index': ai, 'arm': pack['arm'],
                    'selected_input_blocks': len(pack['chosen']), 'output_runs': len(pack['runs']),
                    'selected_source_chars': len(selected), 'reference_chars': len(ref), 'overlap_chars': hit,
                    'character_precision': hit / len(selected) if selected else 0,
                    'character_recall': hit / len(ref), 'full_annotation_coverage': ref <= selected,
                    'proxy_tokens': pack['proxy_tokens'], 'rendered_sha256': sha(pack['rendered_text'].encode())})
        left, right = case_packs
        a, b = positions(left['runs']), positions(right['runs'])
        comparison.append({'ordinal': ordinal, 'same_source_character_set': a == b,
                           'source_unit_only_chars': len(a - b), 'source_atom_only_chars': len(b - a),
                           'shared_chars': len(a & b), 'same_complete_render': left['rendered_text'] == right['rendered_text']})
    report = {'schema': 'slac-candidate-granularity-result-v1', 'policy': POLICY, 'availability': availability,
        'rows': rows, 'comparisons': comparison, 'input_sha256': sha(input_raw),
        'score_sha256': sha(score_raw), 'reference_records_sha256': REFERENCE_RECORDS_SHA,
        'pack_freeze_sha256': sha((output / 'pack_freeze.json').read_bytes()),
        'scoring': {k: scoring_summary[k] for k in ('query_count', 'pair_count', 'length_audit', 'compute', 'model_inference_performed')},
        'counts': {'questions': 2, 'annotations': 4, 'methods': 2, 'packs': 4, 'evaluation_rows': 8,
                   'budget_strings_tokenized': len(cached_counts)},
        'resources': {'api_calls': 0, 'training_updates': 0, 'answer_generations': 0, 'new_documents': 0},
        'limits': ['Two exposed questions; scoring/selection do not use reference labels but researchers have seen them.',
                   'Fresh local BGE scores on each arm text, not JEV or learned Refiner scores.',
                   'Same candidate character domain and output-budget policy, unequal candidate counts and realized context lengths.',
                   'Greedy prefix feasibility is representation-sensitive; skipped candidates are not retried.',
                   'Character coverage is not semantic sufficiency, official Evidence F1 or Answer F1.',
                   'Different annotation scopes and missing candidate evidence remain separate.']}
    assert sample['input_bindings']['source_sha256'] == code_hashes() and sha(INPUT.read_bytes()) == sha(input_raw)
    assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules and 'jax' not in sys.modules
    write(output / 'report.json', report)
    print(json.dumps({'availability': availability, 'rows': rows, 'comparisons': comparison, 'counts': report['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else analyze)()
