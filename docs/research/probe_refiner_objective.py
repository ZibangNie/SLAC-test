"""Two constructed exact-arithmetic examples; no neural/runtime/data inputs.

This defines a NEW distribution conditioned on legal action decompositions.
It does not reproduce the probability distribution of the production decoder.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SLAC/refiner'))
from slac_refiner.label_contract import derive_canonical_labels, replay_labels


def distribution(b0, choices, insert_probabilities, K):
    """Enumerate at most 36 authored configurations; validate via real contract."""
    G = len(b0)
    seeds = [g for g, value in enumerate(b0) if value]
    assert G <= 2 and len(seeds) <= 2 and len(choices) == len(seeds)
    assert len(insert_probabilities) == G
    assert all(sum(row.values()) == 1 for row in choices)
    assert all(0 <= p <= 1 for row in choices for p in row.values())
    assert all(0 <= p <= 1 for p in insert_probabilities)
    legal, rejected = [], []
    for actions in product(*(row.keys() for row in choices)):
        edit_mass = F(1)
        for row, action in zip(choices, actions):
            edit_mass *= row[action]
        for bits in product((0, 1), repeat=G):
            mass = edit_mass
            for p, bit in zip(insert_probabilities, bits):
                mass *= p if bit else 1 - p
            labels = {'edit': [{'g': g, 'y': a} for g, a in zip(seeds, actions)],
                      'insert': list(bits)}
            item = {'labels': labels, 'raw_mass': mass}
            try:
                item['final'] = replay_labels(b0, labels, K=K)
            except ValueError as exc:
                item['reason'] = str(exc)
                rejected.append(item)
            else:
                legal.append(item)
    assert len(legal) + len(rejected) <= 36
    assert sum(x['raw_mass'] for x in legal + rejected) == 1
    Z = sum(x['raw_mass'] for x in legal)
    assert Z > 0
    final_mass = defaultdict(F)
    for item in legal:
        final_mass[tuple(item['final'])] += item['raw_mass']
    assert sum(final_mass.values()) == Z
    return {
        'b0': b0, 'K': K, 'edit_factors': choices, 'insert_factors': insert_probabilities,
        'enumerated_configurations': len(legal) + len(rejected),
        'positive_legal_configurations': sum(x['raw_mass'] > 0 for x in legal),
        'positive_rejected_configurations': sum(x['raw_mass'] > 0 for x in rejected),
        'legal_mass_Z': Z, 'rejected_mass': 1 - Z,
        'legal_decompositions': legal, 'rejected_decompositions': rejected,
        'final_probabilities': [
            {'final': list(k), 'raw_mass': v, 'conditional_probability': v / Z}
            for k, v in sorted(final_mass.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Output must be inside the research checkout')
    target.parent.mkdir(parents=True, exist_ok=True)
    one = distribution([1], [{'KEEP': F(2, 5), 'DEL': F(3, 5)}], [F(2, 5)], K=0)
    assert one['legal_mass_Z'] == F(21, 25)
    probs = {tuple(x['final']): x['conditional_probability'] for x in one['final_probabilities']}
    assert probs == {(0,): F(3, 7), (1,): F(4, 7)}
    best_path = max(one['legal_decompositions'], key=lambda x: x['raw_mass'])
    assert best_path['final'] == [0] and best_path['raw_mass'] == F(9, 25)
    canonical = derive_canonical_labels([1], [1], K=0)
    assert canonical == {'edit': [{'g': 0, 'y': 'KEEP'}], 'insert': [0]}
    one['canonical_for_final_one'] = canonical
    one['maximum_legal_path_final'] = best_path['final']
    one['maximum_final_class'] = [1]
    one['independent_OR_probability'] = 1 - (1 - F(2, 5)) * (1 - F(2, 5))
    assert one['independent_OR_probability'] == F(16, 25)
    # These are source-derived decisions under explicit settings, not runtime calls.
    one['source_derived_zero_cost_edit_MAP_then_half_threshold_final'] = [0]
    one['source_derived_default_delete_penalty_one_final'] = [1]

    two = distribution([1, 1], [
        {'DEL': F(1, 3), 'KEEP': F(1, 3), 'SHIFT:1': F(1, 3)},
        {'DEL': F(1, 3), 'SHIFT:-1': F(1, 3), 'KEEP': F(1, 3)},
    ], [F(0), F(0)], K=1)
    assert two['legal_mass_Z'] == F(2, 3)
    assert two['positive_legal_configurations'] == 6
    assert two['positive_rejected_configurations'] == 3
    marginals = [sum(x['conditional_probability'] for x in two['final_probabilities'] if x['final'][g]) for g in range(2)]
    assert marginals == [F(1, 2), F(1, 2)]
    two['conditional_gap_marginals'] = marginals
    two['independent_OR_gap_marginals'] = [F(5, 9), F(5, 9)]
    crossing = {'edit': [{'g': 0, 'y': 'SHIFT:1'}, {'g': 1, 'y': 'SHIFT:-1'}], 'insert': [0, 0]}
    assert any(x['labels'] == crossing and x['raw_mass'] == F(1, 9) for x in two['rejected_decompositions'])
    two['crossing_control_rejected'] = True
    sources = [
        'SLAC/refiner/slac_refiner/label_contract.py',
        'SLAC/refiner/slac_refiner/models/heads.py',
        'SLAC/refiner/slac_refiner/models/losses.py',
        'SLAC/refiner/slac_refiner/decoding/dp_edit_decode.py',
        'SLAC/refiner/scripts/run_foundation_probe.py',
        'docs/research/probe_refiner_objective.py',
    ]
    report = {
        'schema': 'slac-refiner-objective-toy-v1', 'date': '2026-10-04',
        'purpose': 'Post-hoc constructed exact arithmetic; no natural quality measurement.',
        'definition': 'Product of per-seed edit and per-gap Bernoulli insertion factors, conditioned on all replay-valid (not just canonical) decompositions. No projector or edit-cost penalties in this distribution.',
        'cases': [one, two],
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'api_calls': 0, 'model_or_tensor_runtime_imported': 'torch' in sys.modules,
        'model_calls': 0, 'training_runs': 0, 'natural_samples_read': 0,
        'production_decoder_executed': False, 'production_label_contract_executed': True,
        'claim': 'Path MAP, final-class MAP and an unconstrained OR can differ. No quality improvement or novelty claim.',
    }
    assert report['model_or_tensor_runtime_imported'] is False
    def encode(value):
        if isinstance(value, F):
            return str(value)
        raise TypeError(type(value).__name__)
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, default=encode)
        handle.write('\n')
    print(json.dumps({'cases': 2, 'configurations': 40, 'exact_assertions_passed': True,
                      'result_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
