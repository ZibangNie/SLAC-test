"""Posthoc, zero-model cache attribution on the already completed four cases."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'artifacts/research-foundation/offline-20261004/score-transfer-attribution-01'
TOKENIZER = Path('D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181')
IDENTITY = ('family_id', 'doc_id', 'question_id')
TOLERANCE = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return sha(raw)


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def cached_f1(selected_ids, by_id, annotation):
    """Exact cached membership; retain BOTH prediction and reference list lengths."""
    reference_count = annotation['effective_reference_items']
    if not selected_ids and not reference_count:
        return 1.0
    matches = annotation['candidate_exact_match_ids_to_reference_indices']
    common = len({by_id[uid].native_text for uid in selected_ids if uid in matches})
    return 2 * common / (len(selected_ids) + reference_count)


def ordered_ids(query, bge_ids, probabilities, by_id, ranking):
    dense_rank = {uid: i for i, uid in enumerate(query['ranked_ids'])}
    if ranking == 'dense':
        return list(query['ranked_ids'])
    if ranking == 'bge':
        return list(bge_ids)
    require(ranking == 'raw_yes', 'unknown fixed ranking')
    return sorted(query['candidate_ids'], key=lambda uid: (-probabilities[uid]['yes'], dense_rank[uid], by_id[uid].order))


def trace_pack(units, ranked, counter, original_chosen):
    """Record decisions alongside, and check agreement with, the unchanged packer."""
    chosen, seen, events = [], set(), []
    for position, index in enumerate(ranked):
        if len(chosen) == 3:
            events.extend({'unit_id': units[i].unit_id, 'action': 'not_scanned_k_full'} for i in ranked[position:])
            break
        if units[index].native_text in seen:
            events.append({'unit_id': units[index].unit_id, 'action': 'skip_native_duplicate'})
            continue
        tokens = counter([*chosen, index])
        if tokens > 1024:
            events.append({'unit_id': units[index].unit_id, 'action': 'skip_overflow', 'prospective_tokens': tokens})
            continue
        chosen.append(index)
        seen.add(units[index].native_text)
        events.append({'unit_id': units[index].unit_id, 'action': 'admit', 'prospective_tokens': tokens})
    require(sorted(chosen) == original_chosen, 'packing trace differs from original pack_ranked')
    return events


def behavior_checks(Unit, pack_ranked):
    units = [Unit('a', 0, 'paragraph', 0, 1, 'A', 'A'),
             Unit('duplicate', 1, 'paragraph', 1, 2, 'A', 'A'),
             Unit('b', 2, 'paragraph', 2, 3, 'B', 'B')]
    by_id = {u.unit_id: u for u in units}
    ann = {'effective_reference_items': 2,
           'candidate_exact_match_ids_to_reference_indices': {'a': [0, 1], 'duplicate': [0, 1]}}
    require(cached_f1(['a'], by_id, ann) == 2 / 3, 'reference duplicate denominator')
    ann['effective_reference_items'] = 1
    ann['candidate_exact_match_ids_to_reference_indices'] = {'a': [0], 'duplicate': [0]}
    require(cached_f1(['a', 'duplicate'], by_id, ann) == 2 / 3, 'unique intersection with prediction list denominator')
    empty = {'effective_reference_items': 0, 'candidate_exact_match_ids_to_reference_indices': {}}
    require(cached_f1([], by_id, empty) == 1 and cached_f1(['a'], by_id, empty) == 0, 'empty semantics')
    require(max(cached_f1(['a'], by_id, a) for a in (empty, ann)) == 1, 'max over references')
    query = {'candidate_ids': ['a', 'b'], 'ranked_ids': ['b', 'a']}
    values = {uid: {'yes': .5} for uid in query['candidate_ids']}
    require(ordered_ids(query, ['a', 'b'], values, by_id, 'raw_yes') == ['b', 'a'], 'raw-score dense tie')
    # Nonadditive prospective-pack budget: each unit fits, but a+b does not.
    def count(indices):
        return 1100 if 0 in indices and 2 in indices else 10 * len(indices)
    chosen = pack_ranked(units, [0, 1, 2], 1024, count, max_units=3)
    events = trace_pack(units, [0, 1, 2], count, chosen)
    require(chosen == [0] and [e['action'] for e in events] == ['admit', 'skip_native_duplicate', 'skip_overflow'], 'packing behavior')
    return {'status': 'passed', 'scope': 'synthetic metric, tie and nonadditive-packing checks only'}


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to((ROOT / 'artifacts/research-foundation').resolve()), 'private output required')
    protocol_raw = (directory / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    require(protocol['status'] == 'frozen_posthoc_protocol_before_new_arm_computation', 'fixed posthoc protocol required')
    require(protocol['fixed_scope']['arms'] == 6 and protocol['fixed_scope']['queries'] == 4, 'fixed scope changed')
    for filename in ('analysis_started.json', 'records.json', 'comparisons.json', 'summary.json'):
        require(not (directory / filename).exists(), 'single-use analysis; no overwrite or resume')
    buffers = {}
    for key, item in protocol['inputs'].items():
        buffers[key] = Path(item['path']).read_bytes()
        require(sha(buffers[key]) == item['sha256'], 'cached input changed')
    for path, expected in protocol['original_helper_sha256'].items():
        require(sha(Path(path).read_bytes()) == expected, 'original helper changed')
    code_sha = sha(Path(__file__).read_bytes())
    write(directory / 'analysis_started.json', {'status': 'started_offline_posthoc_analysis',
        'protocol_sha256': sha(protocol_raw), 'code_sha256': code_sha,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'api_calls': 0})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in cached attribution')
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack
    checks = behavior_checks(Unit, pack_ranked)
    data = {key: json.loads(raw) for key, raw in buffers.items()}
    token_hashes = data['preparation_summary']['tokenizer_metadata_sha256']
    for name, expected in token_hashes.items():
        require(sha((TOKENIZER / name).read_bytes()) == expected, 'tokenizer changed')
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    prepared, judgments = data['prepared'], data['judgments']
    require(judgments['complete'] and len(prepared['queries']) == 4, 'complete fixed sample required')
    tasks = prepared['support_tasks']
    expected_ids = {t['id'] for t in tasks}
    require(len(tasks) == len(expected_ids) == 64 and set(judgments['labels']) == set(judgments['reported_scores']) == expected_ids, 'complete64 judgment identities required')
    lookup = {(t['doc_id'], t['question_id'], t['unit_id']): t['id'] for t in tasks}
    require(len(lookup) == 64, 'duplicate query-unit task')
    baseline_map = {identity(b['identity']): b for b in data['baselines']}
    old_packs = {(identity(c['identity']), p['method']): p for c in data['frozen_packs'] for p in c['packs']}
    old_rows = {(identity(r), r['method']): r for r in data['primary_readout']['records']}
    interpretations = {identity(c['identity']): c for c in data['interpretation']['cases']}
    require(len(baseline_map) == len(interpretations) == 4 and len(old_packs) == len(old_rows) == 12, 'frozen denominator differs')
    require(data['frozen_manifest']['pack_sha256'] == sha(buffers['frozen_packs'])
            and data['frozen_manifest']['preparation_sha256'] == sha(buffers['prepared'])
            and data['primary_readout']['frozen_manifest_sha256'] == sha(buffers['frozen_manifest']), 'ancestral seal differs')
    for role, current in [('prepared', 'prepared'), ('packs', 'frozen_packs'), ('readout', 'primary_readout'), ('judgments', 'judgments')]:
        require(data['interpretation']['source_sha256'][role]['sha256'] == sha(buffers[current]), 'membership-cache ancestry differs')
    records, reproduced = [], 0
    for ordinal, query in enumerate(prepared['queries'], 1):
        key = identity(query)
        baseline, interpretation = baseline_map[key], interpretations[key]
        candidate_ids = query['candidate_ids']
        require(len(candidate_ids) == len(set(candidate_ids)) == 16, 'fixed16 pool required')
        # Parse the bounded container, but touch text only for these16 candidate IDs.
        raw_units = {u['unit_id']: u for u in prepared['documents'][query['doc_id']]}
        units = sorted((Unit(**raw_units[uid]) for uid in candidate_ids), key=lambda u: u.order)
        by_id = {u.unit_id: u for u in units}
        positions = {u.unit_id: i for i, u in enumerate(units)}
        bge_ids = baseline['bge_ranking']
        require(len(bge_ids) == len(query['ranked_ids']) == 16 and set(bge_ids) == set(query['ranked_ids']) == set(candidate_ids), 'rankings differ from common pool')
        labels = {uid: judgments['labels'][lookup[(query['doc_id'], query['question_id'], uid)]] for uid in candidate_ids}
        values = {uid: judgments['reported_scores'][lookup[(query['doc_id'], query['question_id'], uid)]] for uid in candidate_ids}
        require(set(labels.values()) <= {'yes', 'no', 'unknown'}, 'invalid frozen label')
        for scores in values.values():
            require(set(scores) == {'yes', 'no', 'unknown'} and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in scores.values()) and .985 <= sum(scores.values()) <= 1.015, 'invalid frozen reported scores')
        annotations = interpretation['annotation_availability']
        require(bool(annotations), 'reference membership annotations missing')
        for annotation in annotations:
            hashes = annotation['reference_item_sha256']
            require(len(hashes) == annotation['effective_reference_items'], 'reference list denominator changed')
            expected_matches = {uid: [i for i, h in enumerate(hashes) if sha(by_id[uid].native_text.encode()) == h] for uid in candidate_ids}
            expected_matches = {uid: indices for uid, indices in expected_matches.items() if indices}
            require(expected_matches == annotation['candidate_exact_match_ids_to_reference_indices'], 'complete16 reference membership binding differs')
        counter = PackCounter(tokenizer, units)
        for arm in protocol['arms']:
            ranking = ordered_ids(query, bge_ids, values, by_id, arm['ranking'])
            filtered = [uid for uid in ranking if arm['exclude_no'] and labels[uid] == 'no']
            eligible = [uid for uid in ranking if not arm['exclude_no'] or labels[uid] != 'no']
            ranked = [positions[uid] for uid in eligible]
            chosen = pack_ranked(units, ranked, 1024, counter, max_units=3)
            selected = [units[i].unit_id for i in chosen]
            rendered = render_pack(units, chosen)
            require(len(selected) == len({by_id[uid].native_text for uid in selected}) <= 3 and counter(chosen) <= 1024, 'pack constraint failed')
            annotation_f1 = [cached_f1(selected, by_id, annotation) for annotation in annotations]
            value = max(annotation_f1)
            events = trace_pack(units, ranked, counter, chosen)
            record = {**dict(zip(IDENTITY, key)), 'ordinal': ordinal, 'arm': arm['name'],
                'ranking': arm['ranking'], 'exclude_no': arm['exclude_no'], 'selected_ids': selected,
                'pack_sha256': sha(rendered.encode()), 'actual_evidence_tokens': counter(chosen),
                'selected_units': len(selected), 'max_units': 3, 'budget': 1024,
                'official_evidence_f1': value, 'annotation_f1': annotation_f1,
                'eligible_count': len(eligible), 'filtered_no_ids': filtered,
                'selection_events': events, 'event_counts': dict(Counter(e['action'] for e in events))}
            if arm['name'] in protocol['old_arm_binding']:
                old_method = protocol['old_arm_binding'][arm['name']]
                old_pack, old_row = old_packs[key, old_method], old_rows[key, old_method]
                for field in ('selected_ids', 'pack_sha256', 'actual_evidence_tokens'):
                    require(record[field] == old_pack[field] == old_row[field], 'original per-case pack failed reproduction')
                require(value == old_row['official_evidence_f1'], 'original per-case F1 failed reproduction')
                require(rendered == old_pack['rendered_pack'], 'original rendered bytes differ')
                reproduced += 1
            records.append(record)
    require(len(records) == 24 and reproduced == 12, 'complete attribution denominator failed')
    indexed = {(r['ordinal'], r['arm']): r for r in records}
    comparisons = []
    for left, right in protocol['fixed_contrasts']:
        paired = []
        for ordinal in range(1, 5):
            a, b = indexed[ordinal, left], indexed[ordinal, right]
            delta = a['official_evidence_f1'] - b['official_evidence_f1']
            paired.append({'ordinal': ordinal, 'f1_delta': delta,
                'selected_set_equal': set(a['selected_ids']) == set(b['selected_ids']),
                'ordered_selected_ids_equal': a['selected_ids'] == b['selected_ids'],
                'pack_hash_equal': a['pack_sha256'] == b['pack_sha256'],
                'unit_count_equal': a['selected_units'] == b['selected_units'],
                'tokens_equal': a['actual_evidence_tokens'] == b['actual_evidence_tokens'],
                'f1_equal': abs(delta) <= TOLERANCE,
                'unit_count_delta': a['selected_units'] - b['selected_units'],
                'token_delta': a['actual_evidence_tokens'] - b['actual_evidence_tokens']})
        comparisons.append({'left': left, 'right': right, 'mean_f1_delta': sum(p['f1_delta'] for p in paired) / 4,
            'wins': sum(p['f1_delta'] > TOLERANCE for p in paired), 'ties': sum(p['f1_equal'] for p in paired),
            'losses': sum(p['f1_delta'] < -TOLERANCE for p in paired),
            **{field + '_cases': sum(p[field] for p in paired) for field in ('selected_set_equal', 'ordered_selected_ids_equal', 'pack_hash_equal', 'unit_count_equal', 'tokens_equal', 'f1_equal')}, 'cases': paired})
    arms = [a['name'] for a in protocol['arms']]
    means = {arm: {field: sum(r[field] for r in records if r['arm'] == arm) / 4
                   for field in ('official_evidence_f1', 'actual_evidence_tokens', 'selected_units')} for arm in arms}
    numeric = [c for c in comparisons if c['left'] == 'raw_yes_not_no' and c['right'] in ('dense_not_no', 'bge_not_no')]
    summary = {'schema': 'slac-score-transfer-filter-attribution-v1', 'status': 'complete_posthoc_cached_analysis',
        'protocol_sha256': sha(protocol_raw), 'code_sha256': code_sha, 'input_bindings': protocol['inputs'],
        'helper_bindings': protocol['original_helper_sha256'], 'tokenizer_sha256': token_hashes,
        'behavior_checks': checks, 'cases': 4, 'candidate_pairs': 64, 'arms': 6, 'records': 24,
        'original_per_case_results_exactly_reproduced': reproduced, 'means': means, 'comparisons': comparisons,
        'numeric_ranking_increment_after_same_no_filter': numeric,
        'per_case': [{'ordinal': ordinal, 'arms': {arm: {k: indexed[ordinal, arm][k]
            for k in ('official_evidence_f1', 'selected_units', 'actual_evidence_tokens', 'eligible_count', 'event_counts')}
            for arm in arms}} for ordinal in range(1, 5)],
        'packing_effects': dict(sum((Counter(r['event_counts']) for r in records), Counter())),
        'interpretation_boundary': 'Known-outcome posthoc attribution on four fixed cases. Selection equality, F1 equality and count/token equality are distinct. Filtering can change quantity; ranking can change quantity through packing. No universal causal, independence, AnswerF1, calibration or novelty claim; no new primary readout.',
        'api_calls': 0, 'model_inference': False, 'key_reads': 0, 'original_reference_body_reads': 0,
        'unselected_document_semantic_review': False, 'new_primary_result': False, 'automatic_expansion': False}
    for key, item in protocol['inputs'].items():
        require(sha(Path(item['path']).read_bytes()) == item['sha256'], 'input drift during analysis')
    require(sha(Path(__file__).read_bytes()) == code_sha and (directory / 'protocol.json').read_bytes() == protocol_raw, 'analysis source drift')
    for path, expected in protocol['original_helper_sha256'].items():
        require(sha(Path(path).read_bytes()) == expected, 'helper drift during analysis')
    summary['output_sha256'] = {'records.json': write(directory / 'records.json', records),
                                'comparisons.json': write(directory / 'comparisons.json', comparisons)}
    result_sha = write(directory / 'summary.json', summary)
    return {'status': summary['status'], 'means': means, 'numeric_ranking_increment_after_same_no_filter': numeric,
            'summary_sha256': result_sha, 'code_sha256': code_sha}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().directory), ensure_ascii=False))
