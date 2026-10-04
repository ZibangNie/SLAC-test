"""CPU-only synthetic gradient/parameter accounting; no optimizer or data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SLAC/refiner'))
from slac_refiner.models.direct_boundary import CachedSeedBoundaryClassifier, DirectBoundaryHead
from slac_refiner.models.heads import RefinerHeads


def gradients(module):
    rows = []
    for name, param in module.named_parameters():
        grad = param.grad
        rows.append({
            'name': name, 'elements': param.numel(), 'has_gradient': grad is not None,
            'all_finite': None if grad is None else bool(torch.isfinite(grad).all()),
            'nonzero_elements': 0 if grad is None else int(torch.count_nonzero(grad)),
        })
    return {
        'parameters': sum(r['elements'] for r in rows),
        'parameter_elements_in_tensors_with_gradient': sum(r['elements'] for r in rows if r['has_gradient']),
        'parameter_elements_in_tensors_without_gradient': sum(r['elements'] for r in rows if not r['has_gradient']),
        'tensors': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    target = Path(parser.parse_args().output).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Output must be inside this research checkout')
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(13)
    h = torch.randn(2, 9, 128)
    mask = torch.ones(2, 9, dtype=torch.bool)
    seeds = torch.tensor([[0, 3, 6], [1, 3, 5]])
    b0 = torch.zeros(2, 8, dtype=torch.long).scatter_(1, seeds, 1)
    gold = torch.zeros(2, 8)  # All DEL + no INSERT is the same authored final target.
    refiner = RefinerHeads(128, K=6, dropout=0.0)
    old_direct = deepcopy(refiner)
    direct = DirectBoundaryHead(128, K=6, mlp_hidden_size=37, dropout=0.0)

    ref_output = refiner(h, seeds, mask)
    ref_loss = F.binary_cross_entropy_with_logits(ref_output.insert_logits, gold)
    ref_loss = ref_loss + F.cross_entropy(ref_output.edit_choice_logits.flatten(0, 1), torch.zeros(6, dtype=torch.long))
    ref_loss.backward()
    old_output = old_direct(h, seeds, mask)
    F.binary_cross_entropy_with_logits(old_output.insert_logits, gold).backward()
    direct_output = direct(h, b0, mask)
    F.binary_cross_entropy_with_logits(direct_output.logits[direct_output.gap_mask], gold[direct_output.gap_mask]).backward()
    account = {name: gradients(module) for name, module in (
        ('edit_head', refiner), ('old_direct_using_insert_only', old_direct), ('seed_conditioned_direct', direct),
    )}
    assert account['edit_head']['parameters'] == 262914
    assert account['old_direct_using_insert_only']['parameter_elements_in_tensors_without_gradient'] == 197121
    assert account['seed_conditioned_direct']['parameters'] == 265772
    for name in ('edit_head', 'seed_conditioned_direct'):
        assert all(r['has_gradient'] and r['all_finite'] and r['nonzero_elements'] > 0 for r in account[name]['tensors'])

    wrapper = CachedSeedBoundaryClassifier(
        atom_dim=1024, hidden_size=128, doc_layers=1, doc_heads=4,
        window_size=8, K=6, mlp_hidden_size=37, dropout=0.0,
    )
    context_count = sum(p.numel() for p in wrapper.doc.parameters())
    assert context_count == 329984
    assert sum(p.numel() for p in wrapper.parameters()) == 595756
    sources = [
        'SLAC/refiner/slac_refiner/models/direct_boundary.py',
        'SLAC/refiner/slac_refiner/models/heads.py',
        'SLAC/refiner/slac_refiner/models/doc_encoder.py',
        'docs/research/probe_direct_boundary_baseline.py',
    ]
    result = {
        'schema': 'slac-direct-boundary-structural-probe-v1', 'date': '2026-10-04',
        'device': 'cpu', 'torch_version': torch.__version__, 'seed': 13, 'threads': 1,
        'inputs': {'h_shape': [2, 9, 128], 'b0_seeds': seeds.tolist(), 'gold': 'all final gaps zero'},
        'head_features': {'H': 128, 'K': 6, 'input_dim': direct.input_dim, 'mlp_width': 37},
        'loss': 'Unit-weight mean BCE and mean all-DEL CE for edit branch; mean final-gap BCE for direct branches. Synthetic dependency check, not the default training recipe.',
        'gradient_accounting': account,
        'common_context_parameters': context_count,
        'comparison_totals': {'old_saved': 592898, 'old_direct_loss_dependency_static': 395777,
                              'new_direct': 595756, 'new_minus_edit': 2858},
        'api_calls': 0, 'pretrained_weights_loaded': 0, 'natural_samples_read': 0,
        'optimizer_steps': 0, 'synthetic_head_forward_backward_pairs': 3,
        'wrapper_counted_but_not_forwarded_in_this_probe': True,
        'limits': ['Counts of tensors with gradients are not counts of nonzero gradient elements.',
                   'No training, quality comparison, same effective capacity proof or novelty claim.',
                   'Local reachable-seed features do not match softmax neighbor or global-DP dependencies.'],
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    }
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'passed': True, 'head_parameters': 265772, 'old_unused_parameters': 197121,
                      'source_and_result_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
