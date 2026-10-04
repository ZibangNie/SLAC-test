"""Fixed 16/8-cache, two-arm, three-seed optimization diagnostic; no API.

Run only under the companion 180-second supervisor. Outputs are private.
This does not authenticate historical feature identity or establish quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

START = time.monotonic()
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SLAC/refiner'))
from slac_refiner.datasets.collate import refiner_collate_fn
from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.decoding.projector import ProjectorConfig, rebuild_chunks_from_boundary_vector
from slac_refiner.eval.metrics import boundary_prf
from slac_refiner.models.direct_boundary import CachedSeedBoundaryClassifier
from slac_refiner.models.doc_encoder import DocEncoder
from slac_refiner.models.heads import RefinerHeads
from slac_refiner.models.losses import RefinerLoss


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def state_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def deadline():
    if time.monotonic() - START > 170:
        raise TimeoutError('170-second worker deadline; outer hard limit is 180 seconds')


def fixed_order(seed):
    rng = random.Random(seed)
    order = list(range(16))
    result = []
    for _ in range(8):
        rng.shuffle(order)
        result.extend(order)
    assert len(result) == 128 and all(result.count(i) == 8 for i in range(16))
    return result


class EditModel(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.doc = deepcopy(context)
        self.heads = RefinerHeads(128, K=6, dropout=0.)

    def forward(self, batch):
        h = self.doc(batch['embeddings'], batch['atom_mask']).h
        return self.heads(h, batch['g0_positions'], batch['atom_mask'])


def make_pair(seed):
    torch.manual_seed(seed)
    common = DocEncoder(1024, hidden_size=128, num_layers=1, num_heads=4, dropout=0., window_size=8)
    torch.manual_seed(seed + 1000)
    edit = EditModel(common)
    torch.manual_seed(seed + 2000)
    direct = CachedSeedBoundaryClassifier(1024, 128, 1, 4, 8, 6, 37, dropout=0., max_doc_atoms=128)
    direct.doc.load_state_dict(common.state_dict(), strict=True)
    context_hash = state_digest(common)
    assert state_digest(edit.doc) == state_digest(direct.doc) == context_hash
    for left, right in zip(edit.doc.parameters(), direct.doc.parameters(), strict=True):
        assert left.data_ptr() != right.data_ptr() and torch.equal(left, right)
    return {'edit': edit, 'direct_seed': direct}, context_hash


def forward(arm, model, batch):
    if arm == 'edit':
        return model(batch)
    return model(batch['embeddings'], batch['b0'], batch['atom_mask'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    if not protocol_path.is_relative_to(ROOT) or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Protocol must be in checkout and results must be private artifacts')
    assert digest(protocol_path) == args.protocol_sha256
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['seeds'] == [13, 29, 47] and protocol['steps_per_arm'] == 128
    assert protocol['arms'] == ['edit', 'direct_seed'] and protocol['device'] == 'cpu'
    for group in ('input_sha256', 'source_sha256', 'tokenizer_sha256'):
        for name, expected in protocol[group].items():
            assert digest(ROOT / name) == expected, f'Changed {group}: {name}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = {'status': 'running', 'protocol_sha256': args.protocol_sha256,
              'device': 'cpu', 'torch_version': torch.__version__, 'threads': 4,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors')},
              'api_calls': 0, 'encoder_calls': 0, 'runs': [], 'controls': {},
              'pretrained_weights_loaded': 0, 'optimizer_updates_completed': 0,
              'prediction_rows': 0, 'independent_evaluation': False, 'semantic_gold_verified': False}

    def save():
        report['elapsed_seconds'] = time.monotonic() - START
        write_json(output / 'result.json', report)

    try:
        cache_dir = ROOT / 'artifacts/research-foundation/probe-02-final'
        historical = json.loads((cache_dir / 'results.json').read_bytes())
        features = load_file(str(cache_dir / 'features.safetensors'), device='cpu')
        expected_keys = {f'{split}_{i}' for split, n in [('train', 16), ('dev', 8)] for i in range(n)}
        assert set(features) == expected_keys
        datasets, batches, golds, atoms, identities = {}, {}, {}, {}, {}
        for split, n in [('train', 16), ('dev', 8)]:
            selected_path = cache_dir / f'selected_{split}.jsonl'
            assert digest(selected_path) == historical['selected'][split]['sha256']
            datasets[split] = RefinerDenoiseDataset(str(selected_path), expected_k=6)
            assert len(datasets[split]) == n
            batches[split], golds[split], atoms[split], identities[split] = [], [], [], []
            for i in range(n):
                deadline()
                sample = datasets[split][i]
                embedding = features[f'{split}_{i}']  # Never use lexicographic key order.
                assert embedding.dtype == torch.float32 and embedding.shape == (sample['num_atoms'], 1024)
                assert 1 <= sample['num_atoms'] <= 128 and torch.isfinite(embedding).all()
                assert datasets[split].samples[i].get('orig_split') == 'train'
                batch = refiner_collate_fn([sample])
                batch['embeddings'] = embedding.detach().unsqueeze(0)
                batch['atom_mask'] = torch.ones((1, sample['num_atoms']), dtype=torch.bool)
                batch['sample_weight'] = torch.ones(1)
                batches[split].append(batch)
                golds[split].append(sample['b_gold'].tolist())
                atoms[split].append(sample['atoms_text'])
                identities[split].append({'ordinal': i, 'doc_id': sample['doc_id'], 'sample_id': sample['sample_id'],
                                          'atoms': sample['num_atoms'], 'cache_key': f'{split}_{i}'})
            assert sum(len(a) for a in atoms[split]) == historical['selected'][split]['atoms']
        all_ids = [x['doc_id'] for v in identities.values() for x in v]
        assert len(set(all_ids)) == 24
        report['cache_checks'] = {'explicit_keys': True, 'rows': {'train': 16, 'dev': 8},
                                  'atom_counts': {'train': 549, 'dev': 495}, 'finite': True,
                                  'historical_payload_identity_authenticated': False}
        write_json(output / 'identities.json', identities)
        tokenizer = AutoTokenizer.from_pretrained(protocol['tokenizer_path'], local_files_only=True, trust_remote_code=False)

        @lru_cache(maxsize=100000)
        def count_tokens(text):
            deadline()
            return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

        projector = ProjectorConfig(max_chunk_atoms=64, min_chunk_atoms=1, max_chunk_chars=100000,
                                    min_chunk_chars=1, max_chunk_tokens=512, min_chunk_tokens=1)

        def project(split, i, raw):
            return rebuild_chunks_from_boundary_vector(atoms[split][i], raw, projector,
                                                       token_counter=count_tokens, strict=True)['projected_b']

        def summarize(rows):
            return {f'{view}_macro_f1': sum(boundary_prf(x[view], x['gold'])['f1'] for x in rows) / len(rows)
                    for view in ('raw', 'projected')}

        with (output / 'controls.jsonl').open('x', encoding='utf-8', newline='\n') as control_file:
            for split in ('train', 'dev'):
                rows = []
                for i, batch in enumerate(batches[split]):
                    raw = batch['b0'][0].tolist()
                    row = {'split': split, 'ordinal': i, 'raw': raw, 'projected': project(split, i, raw),
                           'gold': golds[split][i]}
                    rows.append(row)
                    control_file.write(json.dumps(row) + '\n')
                report['controls'][split] = summarize(rows)
                assert abs(report['controls'][split]['raw_macro_f1'] - historical['selected'][split]['b0_raw_macro_f1']) < 1e-12

        with (output / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as prediction_file:
            @torch.no_grad()
            def evaluate(seed, arm, model, phase):
                model.eval()
                summaries = {}
                for split in ('train', 'dev'):
                    rows = []
                    for i, batch in enumerate(batches[split]):
                        deadline()
                        out = forward(arm, model, batch)
                        if arm == 'edit':
                            raw = batch_decode(batch['b0'], batch['g0_positions'], out.edit_choice_logits,
                                               out.insert_logits, K=6, num_gaps=batch['num_gaps'],
                                               insert_threshold=.5, min_sep=0, lambda_del=0., lambda_ins=0., lambda_shift=0.).pred_b[0]
                        else:
                            assert out.gap_mask.all()
                            raw = (out.logits[0].sigmoid() >= .5).int().tolist()
                        assert len(raw) == len(golds[split][i])
                        row = {'seed': seed, 'arm': arm, 'phase': phase, 'split': split, 'ordinal': i,
                               'raw': raw, 'projected': project(split, i, raw), 'gold': golds[split][i]}
                        rows.append(row)
                        prediction_file.write(json.dumps(row) + '\n')
                        prediction_file.flush()
                        report['prediction_rows'] += 1
                    summaries[split] = summarize(rows)
                    summaries[split]['projection_changed_documents'] = sum(x['raw'] != x['projected'] for x in rows)
                return summaries

            criterion = RefinerLoss(insert_pos_weight=1., alpha_insert=1., alpha_edit=1., beta_cost=0.)
            for seed in protocol['seeds']:
                deadline()
                pair, context_hash = make_pair(seed)
                order = fixed_order(seed)
                for arm in protocol['arms']:
                    model = pair[arm]
                    entry = {'seed': seed, 'arm': arm, 'status': 'running', 'steps': 0,
                             'common_context_initial_sha256': context_hash, 'context_initial_sha256': state_digest(model.doc),
                             'model_initial_sha256': state_digest(model), 'head_initial_seed': seed + (1000 if arm == 'edit' else 2000),
                             'parameter_count': sum(p.numel() for p in model.parameters()), 'order': order, 'loss_history': []}
                    report['runs'].append(entry)
                    save()
                    entry['initial'] = evaluate(seed, arm, model, 'initial')
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
                    model.train()
                    for step, index in enumerate(order):
                        deadline()
                        batch = batches['train'][index]
                        optimizer.zero_grad(set_to_none=True)
                        out = forward(arm, model, batch)
                        if arm == 'edit':
                            loss = criterion(out, batch).loss
                        else:
                            mask = out.gap_mask
                            assert mask.shape == batch['b_gold_mask'].shape and torch.equal(mask, batch['b_gold_mask'])
                            loss = F.binary_cross_entropy_with_logits(out.logits[mask], batch['b_gold'].float()[mask], reduction='sum') / mask.sum().clamp(min=1)
                        assert torch.isfinite(loss)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        assert torch.isfinite(grad_norm)
                        optimizer.step()
                        entry['loss_history'].append(float(loss.detach()))
                        entry['steps'] = step + 1
                        report['optimizer_updates_completed'] += 1
                        if (step + 1) % 16 == 0:
                            save()
                    entry['final'] = evaluate(seed, arm, model, 'final')
                    entry['context_final_sha256'] = state_digest(model.doc)
                    entry['first16_mean_loss'] = sum(entry['loss_history'][:16]) / 16
                    entry['last16_mean_loss'] = sum(entry['loss_history'][-16:]) / 16
                    checkpoint = output / 'models' / f'seed_{seed}_{arm}.safetensors'
                    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(checkpoint))
                    entry['checkpoint_sha256'] = digest(checkpoint)
                    entry['status'] = 'completed'
                    save()
                    print(json.dumps({'seed': seed, 'arm': arm, 'steps': 128, 'dev': entry['final']['dev'],
                                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
                    del optimizer
                del pair
        assert len(report['runs']) == 6 and report['optimizer_updates_completed'] == 768 and report['prediction_rows'] == 288
        for group in ('input_sha256', 'source_sha256', 'tokenizer_sha256'):
            for name, expected in protocol[group].items():
                assert digest(ROOT / name) == expected, f'Changed after run: {name}'
        report['token_counter_cache'] = count_tokens.cache_info()._asdict()
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
