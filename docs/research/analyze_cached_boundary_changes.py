"""Post-hoc numeric decomposition of the frozen eight-document dev diagnostic.

No dataset, model, tokenizer or network dependency. Never changes predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics

EXPECTED = {
    'predictions.jsonl': '9bee6d99fe178b0dd4adbc5019024ea1ae9cf004e8b0298649b0e9de729003a3',
    'controls.jsonl': '60acd4aa0d6078e8d053a60b7844c504a9d266d096e2b77ca700a17c25667571',
}
SEEDS = (13, 29, 47)
ARMS = ('edit', 'direct_seed')


def load_bound_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED[path.name]:
        raise ValueError(f'Frozen input mismatch: {path.name}')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def confusion(prediction: list[int], gold: list[int]) -> dict:
    tp = sum(p == y == 1 for p, y in zip(prediction, gold))
    fp = sum(p == 1 and y == 0 for p, y in zip(prediction, gold))
    fn = sum(p == 0 and y == 1 for p, y in zip(prediction, gold))
    denominator = 2 * tp + fp + fn
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'f1': 2 * tp / denominator if denominator else 0.0}


def describe(b: list[int], y: list[int], p: list[int]) -> dict:
    assert len(b) == len(y) == len(p)
    assert all(type(v) is int and v in (0, 1) for row in (b, y, p) for v in row)
    count = Counter(''.join(map(str, values)) for values in zip(b, y, p))
    cells = {''.join(map(str, values)): count[''.join(map(str, values))]
             for values in itertools.product((0, 1), repeat=3)}
    kept_correct = cells['000'] + cells['111']
    kept_wrong = cells['010'] + cells['101']
    fixed_fn, fixed_fp = cells['011'], cells['100']
    added_fp, added_fn = cells['001'], cells['110']
    fixed, added = fixed_fn + fixed_fp, added_fp + added_fn
    old, new = confusion(b, y), confusion(p, y)
    assert sum(cells.values()) == len(b) == kept_correct + kept_wrong + fixed + added
    assert old['fp'] + old['fn'] == kept_wrong + fixed
    assert new['fp'] + new['fn'] == kept_wrong + added
    assert sum(a != c for a, c in zip(b, p)) == fixed + added
    delta = new['f1'] - old['f1']
    return {'gaps': len(b), 'b_y_p_cells': cells, 'kept_correct': kept_correct,
            'kept_wrong': kept_wrong, 'fixed_false_negative': fixed_fn,
            'fixed_false_positive': fixed_fp, 'introduced_false_positive': added_fp,
            'introduced_false_negative': added_fn, 'fixed': fixed, 'introduced': added,
            'changed_gaps': fixed + added, 'exactly_keeps_b0': b == p,
            'b0': old, 'prediction': new, 'f1_delta': delta,
            'f1_direction': 'improved' if delta > 1e-12 else 'declined' if delta < -1e-12 else 'tied'}


def run(input_dir: Path) -> dict:
    controls = load_bound_rows(input_dir / 'controls.jsonl')
    predictions = load_bound_rows(input_dir / 'predictions.jsonl')
    assert len(controls) == 24 and len(predictions) == 288
    dev = [row for row in controls if row['split'] == 'dev']
    assert len(dev) == 8 and {row['ordinal'] for row in dev} == set(range(8))
    lookup = {row['ordinal']: row for row in dev}
    final = [row for row in predictions if row['split'] == 'dev' and row['phase'] == 'final']
    assert len(final) == 48
    assert {(r['seed'], r['arm'], r['ordinal']) for r in final} == set(itertools.product(SEEDS, ARMS, range(8)))
    rows = []
    for row in sorted(final, key=lambda r: (r['seed'], r['arm'], r['ordinal'])):
        control = lookup[row['ordinal']]
        assert row['gold'] == control['gold']
        assert row['raw'] == row['projected'] and control['raw'] == control['projected']
        detail = describe(control['raw'], control['gold'], row['raw'])
        rows.append({'seed': row['seed'], 'arm': row['arm'],
                     'local_document_ordinal': row['ordinal'], **detail})
    runs = []
    total_keys = ('gaps', 'kept_correct', 'kept_wrong', 'fixed_false_negative',
                  'fixed_false_positive', 'introduced_false_positive',
                  'introduced_false_negative', 'fixed', 'introduced', 'changed_gaps')
    for seed, arm in itertools.product(SEEDS, ARMS):
        group = [r for r in rows if r['seed'] == seed and r['arm'] == arm]
        totals = {key: sum(r[key] for r in group) for key in total_keys}
        directions = Counter(r['f1_direction'] for r in group)
        runs.append({'seed': seed, 'arm': arm, 'documents': len(group), **totals,
                     'exact_b0_documents': sum(r['exactly_keeps_b0'] for r in group),
                     'document_f1_directions': {k: directions[k] for k in ('improved', 'tied', 'declined')},
                     'b0_macro_f1': statistics.mean(r['b0']['f1'] for r in group),
                     'prediction_macro_f1': statistics.mean(r['prediction']['f1'] for r in group),
                     'macro_f1_delta': statistics.mean(r['f1_delta'] for r in group)})
    return {
        'schema': 'slac-cached-boundary-change-decomposition-v1',
        'analysis_type': 'posthoc_descriptive_same_eight_exposed_legacy_dev_documents',
        'input_sha256': EXPECTED,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'additional_api_calls': 0, 'additional_training_updates': 0, 'model_loads': 0,
        'unique_documents': 8, 'document_run_records': 48,
        'cell_order': 'b0, weak reference, final prediction',
        'zero_denominator_f1': 0.0, 'raw_equals_projected_for_all_used_vectors': True,
        'runs': runs, 'document_records': rows,
        'limits': ['Counts refer to agreement with frozen weak references, not reference or semantic correctness.',
                   'Gap error counts do not determine document-macro F1.',
                   'Repeated seeds do not add independent documents.',
                   'Post-hoc changes cannot identify causes of optimization differences or prove JEV/RAG benefits.'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.input_dir)
    with args.output.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write('\n')
    print(json.dumps({'status': 'completed', 'runs': result['runs']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
