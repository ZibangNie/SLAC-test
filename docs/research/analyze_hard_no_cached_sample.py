"""Fixed six-query, zero-model hard-no cache contrast on exposed development data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'artifacts/research-foundation/offline-20261004/hard-no-cache-sample-01'
PINS = {
    'protocol.json': '17df6d62f1626406ac551cf7ad6ba298e36730dfa9ffbac1c194d4947b3331f8',
    'selection.json': 'd60d1c681659318cd6e5f2c3c3fb92b4ff158c8bf084f2a5b46c398f3151e0fb',
    'bounded_inputs.json': 'f92a1a787dd7a3c5b79075165f8b2356ce0518aeb5503ca310ec44e2bb5fd9ee',
}
IDENTITY = ('family_id', 'doc_id', 'question_id')
ARMS = ('raw_yes_not_no', 'raw_yes_all')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return sha(raw)


def pinned_bytes(item):
    raw = Path(item['path']).read_bytes()
    require(sha(raw) == item['sha256'], 'pinned input hash differs')
    return raw


def extract_judgments(label_raw, score_raw, task_ids):
    """Parse sealed cache containers; validate only the selected94 values."""
    source_labels, source_scores = json.loads(label_raw)['jev'], json.loads(score_raw)
    labels, scores = {}, {}
    for task_id in task_ids:
        labels[task_id], scores[task_id] = source_labels[task_id], source_scores[task_id]
        require(labels[task_id] in {'yes', 'no', 'unknown'}, 'invalid selected label')
        value = scores[task_id]
        require(set(value) == {'yes', 'no', 'unknown'} and all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value.values()
        ) and .985 <= sum(value.values()) <= 1.015, 'invalid selected reported scores')
    return labels, scores


def matched_rows(item, selected, method=None):
    """Hash all bytes, but JSON-decode only selected question/method lines after hash verification."""
    patterns = [re.compile(rb'"question_id"\s*:\s*' + re.escape(json.dumps(q['question_id']).encode())) for q in selected]
    method_pattern = None if method is None else re.compile(rb'"method"\s*:\s*' + re.escape(json.dumps(method).encode()))
    digest, retained = hashlib.sha256(), []
    with Path(item['path']).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if any(p.search(line) for p in patterns) and (method_pattern is None or method_pattern.search(line)):
                retained.append(line)
    require(digest.hexdigest() == item['sha256'], 'selected JSONL source changed')
    require(len(retained) == len(selected), 'matched JSONL row count differs')
    rows = [json.loads(line) for line in retained]
    require(len({identity(r) for r in rows}) == len(selected) and
            {identity(r) for r in rows} == {identity(q) for q in selected}, 'matched identities differ')
    if method is not None:
        require(all(r['method'] == method for r in rows), 'matched method differs')
    return rows, b''.join(retained)


def trace_pack(units, ranking, counter, expected):
    chosen, seen, events = [], set(), []
    for position, index in enumerate(ranking):
        if len(chosen) == 3:
            events.extend({'unit_id': units[i].unit_id, 'action': 'not_scanned_k_full'} for i in ranking[position:])
            break
        if units[index].native_text in seen:
            events.append({'unit_id': units[index].unit_id, 'action': 'skip_native_duplicate'})
            continue
        tokens = counter([*chosen, index])
        action = 'admit' if tokens <= 1024 else 'skip_overflow'
        events.append({'unit_id': units[index].unit_id, 'action': action, 'prospective_tokens': tokens})
        if action == 'admit':
            chosen.append(index)
            seen.add(units[index].native_text)
    require(sorted(chosen) == expected, 'packing trace differs from frozen helper')
    return events


def behavior_checks(Unit, pack_ranked, paragraph_f1_score):
    units = [Unit('a', 0, 'p', 0, 1, 'A', 'A'), Unit('dup', 1, 'p', 1, 2, 'A', 'A'), Unit('b', 2, 'p', 2, 3, 'B', 'B')]
    count = lambda selected: 1100 if 0 in selected and 2 in selected else 10 * len(selected)
    chosen = pack_ranked(units, [0, 1, 2], 1024, count, max_units=3)
    require(chosen == [0], 'nonadditive budget/dedup contract')
    require([e['action'] for e in trace_pack(units, [0, 1, 2], count, chosen)] == ['admit', 'skip_native_duplicate', 'skip_overflow'], 'trace contract')
    require(paragraph_f1_score(['A'], ['A', 'A']) == 2 / 3 and paragraph_f1_score([], []) == 1, 'official metric edge contract')
    label_raw = b'{"general":{"one":"yes"},"jev":{"one":"no","other":"yes"}}'
    score_raw = b'{"one":{"yes":0.2,"no":0.7,"unknown":0.1},"other":{"invalid":true}}'
    labels, values = extract_judgments(label_raw, score_raw, ['one'])
    require(labels == {'one': 'no'} and values['one']['yes'] == .2, 'bounded value projection')
    return {'status': 'passed', 'scope': 'synthetic bounded projection, duplicate-list metric and nonadditive packing'}


def run(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to((ROOT / 'artifacts/research-foundation').resolve()), 'private artifact output required')
    buffers = {name: (directory / name).read_bytes() for name in PINS}
    require(all(sha(buffers[n]) == expected for n, expected in PINS.items()), 'frozen preparation differs')
    protocol, selection, bounded = (json.loads(buffers[n]) for n in PINS)
    require(selection['protocol_sha256'] == bounded['protocol_sha256'] == PINS['protocol.json'] and
            bounded['selection_sha256'] == PINS['selection.json'], 'preparation ancestry differs')
    require(protocol['arms'] == list(ARMS) and protocol['pack']['budget'] == 1024 and protocol['pack']['max_units'] == 3, 'fixed arms or caps differ')
    selected = selection['selected']
    queries = {identity(q): q for q in bounded['queries']}
    require(len(selected) == len(queries) == 6 and len({q['family_id'] for q in selected}) == 6 and
            set(queries) == {identity(q) for q in selected}, 'fixed six identities differ')
    require([q['ordinal'] for q in selected] == list(range(1, 7)), 'case order differs')
    source_hashes = dict(protocol['helpers_sha256'])
    # The pack helper imports this source-only alignment module transitively.
    source_hashes['docs/research/qasper_alignment_v2.py'] = sha((ROOT / 'docs/research/qasper_alignment_v2.py').read_bytes())
    source_hashes[str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = sha(Path(__file__).read_bytes())
    require(all(sha((ROOT / name).read_bytes()) == expected for name, expected in source_hashes.items()), 'helper source changed')
    outputs = ['analysis_started.json', 'selected_judgments.json', 'frozen_packs.json', 'pack_freeze.json', 'matched_references.jsonl', 'matched_old_records.jsonl', 'records.json', 'summary.json']
    require(not any((directory / name).exists() for name in outputs), 'single-use analysis; no resume or overwrite')
    write(directory / 'analysis_started.json', {'status': 'started_offline_posthoc_analysis', 'created_utc': datetime.now(timezone.utc).isoformat(), 'preparation_sha256': PINS, 'source_sha256': source_hashes, 'api_calls': 0})
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in cached sample analysis')
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack
    from qasper_metrics import evidence_metrics, paragraph_f1_score, references_from_annotations
    checks = behavior_checks(Unit, pack_ranked, paragraph_f1_score)
    tokenizer_info = protocol['tokenizer']
    token_path = Path(tokenizer_info['path'])
    for name, expected in tokenizer_info['verified_file_sha256'].items():
        require(sha((token_path / name).read_bytes()) == expected, 'tokenizer file changed')
    tokenizer = AutoTokenizer.from_pretrained(str(token_path), local_files_only=True, trust_remote_code=False)
    tasks = bounded['support_tasks']
    lookup = {(t['doc_id'], t['question_id'], t['unit_id']): t['id'] for t in tasks}
    require(len(tasks) == len(lookup) == len({t['id'] for t in tasks}) == 94, 'fixed94 task binding differs')
    labels, scores = extract_judgments(pinned_bytes(protocol['inputs']['labels']), pinned_bytes(protocol['inputs']['scores']), [t['id'] for t in tasks])
    judgment_sha = write(directory / 'selected_judgments.json', {'labels': labels, 'reported_scores': scores, 'sources': {k: protocol['inputs'][k] for k in ('labels', 'scores')}})
    packs, contexts = [], {}
    for meta in selected:
        q = queries[identity(meta)]
        units = sorted((Unit(**u) for u in bounded['documents'][q['doc_id']]), key=lambda u: u.order)
        pos = {u.unit_id: i for i, u in enumerate(units)}
        ids = q['candidate_ids']
        require(len(pos) == len(units) == len(ids) == meta['candidate_count'] and set(pos) == set(ids) == set(q['ranked_ids']) and len(set(q['ranked_ids'])) == len(ids), 'candidate pool/rankings differ')
        require(len({u.order for u in units}) == len(units), 'duplicate source order')
        rank = {uid: i for i, uid in enumerate(q['ranked_ids'])}
        task = {uid: lookup[(q['doc_id'], q['question_id'], uid)] for uid in ids}
        ordering = sorted(ids, key=lambda uid: (-scores[task[uid]]['yes'], rank[uid], units[pos[uid]].order))
        counter = PackCounter(tokenizer, units)
        contexts[identity(meta)] = (units, pos, task)
        for arm in ARMS:
            eligible = [uid for uid in ordering if arm == 'raw_yes_all' or labels[task[uid]] != 'no']
            ranking = [pos[uid] for uid in eligible]
            chosen = pack_ranked(units, ranking, 1024, counter, max_units=3)
            packs.append({**{k: meta[k] for k in (*IDENTITY, 'ordinal')}, 'method': arm,
                          'selected_ids': [units[i].unit_id for i in chosen], 'pack_sha256': sha(render_pack(units, chosen).encode()),
                          'actual_evidence_tokens': counter(chosen), 'selected_count': len(chosen),
                          'eligible_count': len(eligible), 'pool_count': len(ids), 'packing_trace': trace_pack(units, ranking, counter, chosen)})
    require(len(packs) == 12, 'all12 predictions required before references')
    pack_sha = write(directory / 'frozen_packs.json', packs)
    freeze_sha = write(directory / 'pack_freeze.json', {'status': 'all_12_packs_frozen_before_reference_and_historical_metric_decoding', 'frozen_packs_sha256': pack_sha,
        'selected_judgments_sha256': judgment_sha, 'preparation_sha256': PINS, 'source_sha256': source_hashes,
        'inputs': protocol['inputs'], 'tokenizer': tokenizer_info, 'references_decoded': 0, 'old_metric_rows_decoded': 0})
    old, old_raw = matched_rows(protocol['inputs']['old_records'], selected, 'p_yes_only_k3')
    refs, ref_raw = matched_rows(protocol['inputs']['references'], selected)
    for name, raw in [('matched_old_records.jsonl', old_raw), ('matched_references.jsonl', ref_raw)]:
        with (directory / name).open('xb') as stream:
            stream.write(raw)
    old = {identity(r): r for r in old}
    refs = {identity(r): r for r in refs}
    records, availability, comparisons = [], [], []
    for meta in selected:
        key = identity(meta)
        units, pos, task = contexts[key]
        annotations = refs[key]['answer_annotations']
        effective_refs = references_from_annotations(annotations, text_evidence_only=False)
        annotation_rows = []
        for ai, (original, ref) in enumerate(zip(annotations, effective_refs, strict=True)):
            native_answer = original.get('native_answer', original.get('answer', original))
            matches = {uid: [i for i, text in enumerate(ref['evidence']) if units[pos[uid]].native_text == text] for uid in pos}
            annotation_rows.append({'annotation_index': ai, 'unanswerable': native_answer['unanswerable'],
                'original_evidence_list_length': len(native_answer['evidence']), 'effective_reference_items': len(ref['evidence']),
                'reference_item_sha256': [sha(t.encode()) for t in ref['evidence']], 'candidate_exact_match_ids_to_reference_indices': matches})
        availability.append({**meta, 'annotations': annotation_rows})
        case_records = []
        for pack in [p for p in packs if identity(p) == key]:
            predicted = [units[pos[uid]].native_text for uid in pack['selected_ids']]
            metrics = evidence_metrics(predicted, annotations, text_evidence_only=False)
            record = {**pack, 'official_evidence_f1': metrics['evidence_f1'],
                'best_f1_reference_index': metrics['best_f1_reference_index'],
                'per_annotation': [{'annotation_index': ai, 'f1': paragraph_f1_score(predicted, ref['evidence']),
                    'matched_selected_ids': [uid for uid in pack['selected_ids'] if units[pos[uid]].native_text in ref['evidence']]} for ai, ref in enumerate(effective_refs)]}
            if pack['method'] == ARMS[0]:
                baseline = old[key]
                require(all(pack[k] == baseline[k] for k in ('selected_ids', 'pack_sha256', 'actual_evidence_tokens')), 'original filtered pack failed exact reproduction')
                require(abs(metrics['evidence_f1'] - baseline['official_evidence_f1']) <= 1e-12, 'original filtered metric failed reproduction')
            records.append(record)
            case_records.append(record)
        filtered, all_ = case_records
        gained = [uid for uid in all_['selected_ids'] if uid not in filtered['selected_ids']]
        removed = [uid for uid in filtered['selected_ids'] if uid not in all_['selected_ids']]
        comparisons.append({**meta, 'delta_all_minus_not_no': all_['official_evidence_f1'] - filtered['official_evidence_f1'],
            'same_selected_set': set(filtered['selected_ids']) == set(all_['selected_ids']),
            'same_count': filtered['selected_count'] == all_['selected_count'], 'same_tokens': filtered['actual_evidence_tokens'] == all_['actual_evidence_tokens'],
            'same_f1': abs(all_['official_evidence_f1'] - filtered['official_evidence_f1']) <= 1e-12,
            'count_delta': all_['selected_count'] - filtered['selected_count'], 'token_delta': all_['actual_evidence_tokens'] - filtered['actual_evidence_tokens'],
            'newly_selected': [{'unit_id': uid, 'choice': labels[task[uid]], 'raw_yes': scores[task[uid]]['yes']} for uid in gained],
            'displaced_non_no': [{'unit_id': uid, 'choice': labels[task[uid]], 'raw_yes': scores[task[uid]]['yes']} for uid in removed],
            'changed_candidate_pool': False, 'changed_eligible_pool': filtered['eligible_count'] != all_['eligible_count']})
    records_sha = write(directory / 'records.json', {'records': records, 'comparisons': comparisons, 'annotation_availability': availability})
    wins = sum(c['delta_all_minus_not_no'] > 1e-12 for c in comparisons)
    losses = sum(c['delta_all_minus_not_no'] < -1e-12 for c in comparisons)
    decision = ('mixed_signs' if wins and losses else 'unfiltered_nondominated_with_gain' if wins else
                'filtered_nondominated_with_gain' if losses else 'all_tied_same_packs' if all(c['same_selected_set'] for c in comparisons) else 'all_tied_changed_packs')
    require(all(sha((ROOT / n).read_bytes()) == h for n, h in source_hashes.items()) and
            all(sha((directory / n).read_bytes()) == h for n, h in PINS.items()) and sha((directory / 'frozen_packs.json').read_bytes()) == pack_sha, 'frozen source or prediction drift')
    summary = {'schema': 'slac-hard-no-cached-sample-result-v1', 'status': 'completed', 'analysis_type': protocol['analysis_type'],
        'scope': protocol['scope'], 'synthetic_checks': checks, 'original_filtered_exact_pack_and_metric_reproduced': 6,
        'reference_rows_decoded': 6, 'historical_metric_rows_decoded': 6, 'selected_judgments_validated': 94,
        'pack_freeze_sha256': freeze_sha, 'frozen_packs_sha256': pack_sha, 'records_sha256': records_sha,
        'matched_references_sha256': sha(ref_raw), 'matched_old_records_sha256': sha(old_raw), 'preparation_sha256': PINS, 'source_sha256': source_hashes,
        'means': {arm: {'official_evidence_f1': sum(r['official_evidence_f1'] for r in records if r['method'] == arm) / 6,
                       'selected_count': sum(r['selected_count'] for r in records if r['method'] == arm) / 6,
                       'actual_evidence_tokens': sum(r['actual_evidence_tokens'] for r in records if r['method'] == arm) / 6} for arm in ARMS},
        'contrast': {'all_minus_not_no_mean': sum(c['delta_all_minus_not_no'] for c in comparisons) / 6,
                     'wins': wins, 'ties': 6 - wins - losses, 'losses': losses, 'per_case': comparisons},
        'records': [{k: r[k] for k in ('ordinal', 'method', 'selected_ids', 'pack_sha256', 'selected_count', 'actual_evidence_tokens', 'eligible_count', 'official_evidence_f1')} for r in records],
        'packing_events': dict(Counter(e['action'] for p in packs for e in p['packing_trace'])),
        'decision_rule': decision, 'interpretation': protocol['decision_rules'][decision], 'limits': protocol['limits']}
    write(directory / 'summary.json', summary)
    return {'status': summary['status'], 'means': summary['means'], 'contrast': {k: summary['contrast'][k] for k in ('all_minus_not_no_mean', 'wins', 'ties', 'losses')}, 'decision_rule': decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.directory), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
