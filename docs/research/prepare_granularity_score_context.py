"""Freeze a metadata-only sample before exporting at most four source excerpts."""
from __future__ import annotations

from collections import Counter
import json

import analyze_granularity_three_arm as shared
import evaluate_granularity_jev_microdiagnostic as jev
from granularity_lift_control import prepare_parent_layout

ROOT = shared.ROOT
PHASE = ROOT / 'artifacts/research-foundation/offline-20261005/granularity-score-context-01'
OUTPUT = PHASE / 'prepare-01'
JUDGMENTS = jev.LIVE_ROOT / 'analysis-01/verified_judgments.json'
JUDGMENTS_SHA = '7e5111594b94dec71afec21fdeacf68520e170d60ccccc0603c69245a93a3659'
CATEGORIES = ('both_yes', 'whole_only_yes', 'child_only_yes', 'neither_yes')
TARGETS = ('whole_only_yes', 'child_only_yes')
SOURCE_NAMES = (
    'docs/research/prepare_granularity_score_context.py',
    'docs/research/GRANULARITY_SCORE_CONTEXT_PROTOCOL_20261005.md',
    'docs/research/analyze_granularity_three_arm.py',
    'docs/research/evaluate_granularity_jev_microdiagnostic.py',
    'docs/research/granularity_lift_control.py',
    'docs/research/probe_candidate_granularity.py',
    'docs/research/CANDIDATE_GRANULARITY_PROTOCOL_20261005.md',
    'SLAC/llm/service/renderers.py', 'SLAC/llm/io/schemas.py',
)
sha, require, write = shared.sha, shared.require, shared.probe.write


def source_hashes():
    return {n: sha((ROOT / n).read_bytes()) for n in SOURCE_NAMES}


def category(whole_yes, any_child_yes):
    if whole_yes:
        return 'both_yes' if any_child_yes else 'whole_only_yes'
    return 'child_only_yes' if any_child_yes else 'neither_yes'


def prepare():
    require(not OUTPUT.exists(), 'refusing to overwrite an earlier sample')
    sample, input_raw = shared.load_input()
    judgment_raw = JUDGMENTS.read_bytes()
    require(sha(judgment_raw) == JUDGMENTS_SHA, 'frozen judgments changed')
    judgments = json.loads(judgment_raw)
    jev.validate_judgments(sample, judgments)
    labels, scores = judgments['labels'], judgments['reported_scores']
    sources = source_hashes()
    rows, selected, strata, selected_layouts = [], [], [], []
    for case in sample['cases']:
        layout = prepare_parent_layout(case['candidates']['source_units'], case['candidates']['source_atoms'])
        case_rows = []
        by_id = {}
        for parent in layout['parents']:
            unit, atoms = parent['unit'], parent['atoms']
            whole_choice = labels[unit['task_id']]
            child_labels = [labels[c['task_id']] for c in atoms]
            row = {'ordinal': case['ordinal'], 'dense_rank': unit['dense_rank'], 'span': unit['span'],
                'child_count': len(atoms), 'whole_choice': whole_choice, 'whole_scores': scores[unit['task_id']],
                'child_choice_counts': {label: child_labels.count(label) for label in ('yes', 'no', 'unknown')},
                'child_max_yes_score': max(scores[c['task_id']]['yes'] for c in atoms),
                'category': category(whole_choice == 'yes', 'yes' in child_labels)}
            rows.append(row)
            case_rows.append((row, parent))
            require(unit['candidate_id'] not in by_id, 'duplicate parent')
            by_id[unit['candidate_id']] = parent
        strata.append({'ordinal': case['ordinal'], 'parent_count': len(case_rows),
            'category_counts': {name: sum(r['category'] == name for r, _ in case_rows) for name in CATEGORIES},
            'whole_unknown_count': sum(r['whole_choice'] == 'unknown' for r, _ in case_rows),
            'child_unknown_occurrences': sum(r['child_choice_counts']['unknown'] for r, _ in case_rows)})
        for target in TARGETS:
            pool = sorted([(r, p) for r, p in case_rows if r['category'] == target],
                          key=lambda rp: (rp[0]['dense_rank'], *rp[0]['span']))
            if not pool:
                continue
            row, parent = pool[0]
            case_id = f'S{len(selected)+1:02d}'
            selected.append({'case_id': case_id, **row, 'unit': parent['unit'], 'atoms': parent['atoms']})
            selected_layouts.append((case_id, case, parent))
    require(len(rows) == 30 and len(selected) <= 4, 'fixed sample bounds changed')
    selection = {'schema': 'slac-granularity-score-context-selection-v1',
        'input_sha256': sha(input_raw), 'judgments_sha256': sha(judgment_raw),
        'rule': 'per question and target category: first dense rank, source start, source end; no substitution',
        'all_parent_metadata': rows, 'strata': strata, 'selected': selected,
        'source_sha256': sources, 'source_excerpts_exported': False, 'gold_references_read': False,
        'api_calls': 0, 'scoring_model_calls': 0, 'tokenizer_calls': 0}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / 'selection.json', selection)
    selection_sha = sha((OUTPUT / 'selection.json').read_bytes())
    write(OUTPUT / 'selection_freeze.json', {'selection_sha256': selection_sha, 'source_sha256': sources,
        'selected_count': len(selected), 'references_read': False, 'excerpts_exported_before_freeze': False})

    # Text is exported only after the metadata sample and its freeze have been written.
    packet = {'schema': 'slac-granularity-source-review-packet-v1', 'selection_sha256': selection_sha,
        'offsets': 'absolute half-open Unicode code-point intervals in the original source', 'cases': []}
    pairs = {p['task_id']: p for p in sample['pairs']}
    for case_id, case, parent in selected_layouts:
        def excerpt(candidate):
            a, b = candidate['span']
            text = case['source_text'][a:b]
            pair = pairs[candidate['task_id']]
            require((pair['query'], pair['passage']) == (case['query'], text), 'judged source text differs')
            return {'span': [a, b], 'text': text, 'text_sha256': sha(text.encode())}
        packet['cases'].append({'case_id': case_id, 'ordinal': case['ordinal'], 'query': case['query'],
            'parent': excerpt(parent['unit']),
            'children': [{'child_index': i, **excerpt(c)} for i, c in enumerate(parent['atoms'])]})
    write(OUTPUT / 'review_packet.json', packet)
    report = {'schema': 'slac-granularity-score-context-preparation-v1', 'status': 'prepared',
        'counts': {'questions': 2, 'parents': 30, 'atoms': 97, 'unique_judgments': 117,
                   'selected_parents': len(selected), 'selected_children': sum(r['child_count'] for r in selected)},
        'strata': strata, 'selected_metadata': [{k: v for k, v in r.items() if k not in ('unit', 'atoms')} for r in selected],
        'input_sha256': sha(input_raw), 'judgments_sha256': sha(judgment_raw),
        'selection_sha256': selection_sha,
        'selection_freeze_sha256': sha((OUTPUT / 'selection_freeze.json').read_bytes()),
        'review_packet_sha256': sha((OUTPUT / 'review_packet.json').read_bytes()), 'source_sha256': sources,
        'resources': {'api_calls': 0, 'scoring_model_calls': 0, 'tokenizer_calls': 0, 'gold_reference_reads': 0},
        'limits': ['Judgment discordance is not automatically an error or proof of contextual dependence.',
                   'The packet hides model fields; its source pool was already exposed, so this is not independent confirmation.',
                   'No model accuracy, semantic ground truth, retrieval quality, answer generation or novelty is established.']}
    require(source_hashes() == sources and sha(shared.INPUT.read_bytes()) == sha(input_raw)
            and sha(JUDGMENTS.read_bytes()) == JUDGMENTS_SHA, 'source or input drift')
    write(OUTPUT / 'report.json', report)
    return report


if __name__ == '__main__':
    result = prepare()
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'strata', 'selected_metadata', 'review_packet_sha256')},
                     ensure_ascii=False))
