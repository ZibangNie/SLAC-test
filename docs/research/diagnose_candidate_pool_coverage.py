"""Existing-cache bottlenecks for the same two questions; no model or tokenizer."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from probe_candidate_granularity import ROOT, PHASE, PRIOR, BASE, sha, write, positions


PINS = {
    'sample': (PHASE / 'selected_inputs.json', 'aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3'),
    'old_inputs': (BASE / 'hard-no-cache-sample-01/bounded_inputs.json', 'f92a1a787dd7a3c5b79075165f8b2356ce0518aeb5503ca310ec44e2bb5fd9ee'),
    'old_judgments': (BASE / 'hard-no-cache-sample-01/selected_judgments.json', 'ddc114fb52574c141171447d16c82670a018ed64af94d66f34d2b4fcd3ade0de'),
    'old_records': (BASE / 'hard-no-cache-sample-01/records.json', '8b11594fbad4604d720d958b71d8434bd5a85b883d2b9f90049ecbf1af1699e7'),
    'reference_records': (PRIOR / 'run-01/records.json', 'd2d157b56126be28d7b95717190f862c6b5caed833dbebefa146c07ee11382f5'),
}


def main():
    raw = {name: path.read_bytes() for name, (path, _) in PINS.items()}
    assert all(sha(raw[n]) == h for n, (_, h) in PINS.items())
    data = {name: json.loads(blob) for name, blob in raw.items()}
    sample, old = data['sample'], data['old_inputs']
    assert [c['ordinal'] for c in sample['cases']] == [1, 2]
    cases, reference_rows = [], []
    for case in sample['cases']:
        ordinal, doc, qid = case['ordinal'], case['doc_id'], case['question_id']
        query = next(q for q in old['queries'] if (q['doc_id'], q['question_id']) == (doc, qid))
        assert query['query'] == case['query']
        candidates = {c['parent_native_unit_id']: c for c in case['candidates']['source_units']}
        assert set(candidates) == set(query['candidate_ids']) and len(candidates) == {1: 16, 2: 14}[ordinal]
        tasks = {t['unit_id']: t['id'] for t in old['support_tasks'] if (t['doc_id'], t['question_id']) == (doc, qid)}
        assert set(tasks) == set(candidates)
        dense = {uid: rank + 1 for rank, uid in enumerate(query['ranked_ids'])}
        labels = {uid: data['old_judgments']['labels'][tasks[uid]] for uid in candidates}
        scores = {uid: data['old_judgments']['reported_scores'][tasks[uid]]['yes'] for uid in candidates}
        ranked = sorted(candidates, key=lambda uid: (-scores[uid], dense[uid], candidates[uid]['span'][0]))
        eligible = [uid for uid in ranked if labels[uid] != 'no']
        ranks = {uid: rank + 1 for rank, uid in enumerate(ranked)}
        eligible_ranks = {uid: rank + 1 for rank, uid in enumerate(eligible)}
        chosen = set(case['old_selected_ids'])
        saved_pack = next(r for r in data['old_records']['records'] if r['ordinal'] == ordinal and r['method'] == 'raw_yes_not_no')
        assert chosen == set(saved_pack['selected_ids']) == set(eligible[:3])
        assert (saved_pack['doc_id'], saved_pack['question_id']) == (doc, qid)
        assert not any(e['action'].startswith('skip') for e in saved_pack['packing_trace'])
        domain = positions([c['span'] for c in candidates.values()])
        selected = positions([candidates[uid]['span'] for uid in chosen])
        cases.append({'ordinal': ordinal, 'candidate_units': len(candidates), 'candidate_chars': len(domain),
                      'selected_units': len(chosen), 'selected_source_chars': len(selected),
                      'eligible_units': len(eligible), 'old_packing_skip_count': 0,
                      'old_official_evidence_f1_reused_not_recomputed': saved_pack['official_evidence_f1']})
        for ai in (0, 1):
            reference = next(r for r in data['reference_records'] if
                r['summary']['ordinal'] == ordinal and r['summary']['annotation_index'] == ai and r['summary']['arm'] == 'raw_b0_source')
            assert reference['doc_id'] == doc and reference['question_id'] == qid
            assert reference['summary']['annotation_status'] == 'mapped'
            ref = positions(reference['reference_intervals'])
            items, selected_ref_ids = [], set()
            for item in reference['mapping_details']:
                assert item['kind'] == 'single_exact_native_unit' and len(item['matched_unit_ids']) == 1
                uid, span = item['matched_unit_ids'][0], item['matched_spans'][0]
                in_pool, in_pack = uid in candidates, uid in chosen
                assert (positions([span]) <= domain) == in_pool
                assert (positions([span]) <= selected) == in_pack
                if in_pack:
                    selected_ref_ids.add(uid)
                status = ('selected' if in_pack else 'missing_from_candidate_pool' if not in_pool else
                          'filtered_no' if labels[uid] == 'no' else 'eligible_below_top3')
                items.append({'reference_index': item['item_index'], 'reference_chars': span[1] - span[0],
                    'status': status, 'dense_rank_1based': dense.get(uid), 'raw_yes_rank_1based': ranks.get(uid),
                    'eligible_rank_1based': eligible_ranks.get(uid), 'cached_choice': labels.get(uid),
                    'cached_reported_yes': scores.get(uid)})
            historical = next(a for a in saved_pack['per_annotation'] if a['annotation_index'] == ai)
            assert selected_ref_ids == set(historical['matched_selected_ids'])
            reference_rows.append({'ordinal': ordinal, 'annotation_index': ai, 'reference_units': len(items),
                'status_counts': dict(Counter(i['status'] for i in items)), 'reference_chars': len(ref),
                'candidate_reference_chars': len(ref & domain), 'selected_reference_chars': len(ref & selected),
                'candidate_complete_reference_units': sum(i['status'] != 'missing_from_candidate_pool' for i in items),
                'selected_complete_reference_units': sum(i['status'] == 'selected' for i in items),
                'candidate_full_coverage': ref <= domain, 'selected_full_coverage': ref <= selected,
                'references': items})
    output = PHASE / 'pool-diagnostic-01'
    output.mkdir(exist_ok=False)
    result = {'schema': 'slac-two-question-candidate-bottleneck-v1', 'cases': cases, 'annotations': reference_rows,
              'input_sha256': {n: h for n, (_, h) in PINS.items()},
              'source_sha256': {str(Path(__file__).relative_to(ROOT)).replace('\\', '/'): sha(Path(__file__).read_bytes()),
                                'docs/research/probe_candidate_granularity.py': sha((ROOT / 'docs/research/probe_candidate_granularity.py').read_bytes())},
              'resources': {'api_calls': 0, 'model_calls': 0, 'tokenizer_calls': 0, 'training_updates': 0,
                            'questions': 2, 'cached_candidate_judgments_used': 30, 'annotations': 4, 'reference_items': 11},
              'limits': ['Post-hoc diagnosis of two exposed questions, not a new selection method or quality gain.',
                         'All four original annotations retained, including the duplicate first-question evidence profile.',
                         'Old cached scores apply only to their original unit texts, not new source/atom scoring inputs.',
                         'Complete annotated evidence is not proven necessary or sufficient for an answer.',
                         'Candidate missingness and scope differences remain; no reference relabeling.']}
    assert all(sha(path.read_bytes()) == h for path, h in PINS.values())
    write(output / 'report.json', result)
    print(json.dumps({'cases': cases, 'annotations': reference_rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
