"""One-shot, offline budget gate for the six frozen final-geometry additions.

No selection, scoring, reference loading, or cumulative additions are performed.
Full renders stay in the private output; the report contains anonymous metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/research-foundation/offline-20261005'
OUTPUT = BASE / 'granularity-revisit-control-01/additions-01'
INPUTS = {
    'selected_inputs': (BASE / 'candidate-granularity-mechanism-01/selected_inputs.json',
                        'aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3'),
    'original_packs': (BASE / 'jev-granularity-live-01/analysis-01/frozen_packs.json',
                       '44b3d04204cca2473211d77a18134029c7889b964ae2857408d3decca6c5d580'),
    'final_geometry': (BASE / 'granularity-run-cap-control-01/final_geometry.json',
                       'cf8aabdc64f6c14f3875f809f33381aa486562970ee06f755b809a191933df2c'),
}
SOURCE_PINS = {
    'docs/research/GRANULARITY_REVISIT_PROTOCOL_20261005.md': '93661986994191a49ace7d0a851fa088e7e0745f6746c784aeb927ce123811f9',
    'docs/research/analyze_granularity_three_arm.py': 'cdd5ba74a971faa1f62f423b932bd3899c12f0b75485c82cb3cbfa10ef21f62e',
    'docs/research/granularity_lift_control.py': '61beecf171be46125b62d8db9775d102172df62eee15739b9b6d7a4c8427ac43',
    'docs/research/probe_candidate_granularity.py': '229b495fe99fc8f892f8d1e62f8ebc01e45cc6c4bd86c13b5654eea8926e01d0',
    'docs/research/probe_natural_partition_coverage.py': '93c7d382c33ef06d6f7e48b9ce4a8bfc2f3a96e7926ce9169560eca2ebab762b',
    'SLAC/llm/io/schemas.py': 'c3229a47ac786859f2bcb07de5dfc8ddc00efe8088d34625131f462e06f52c81',
    'SLAC/llm/service/renderers.py': 'bad96dad28bf7b0fb17b7c84d32f2408e98cc36ac0ccf587ea8b3d6103335e86',
}
TOKENIZER = ROOT.parent / 'SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
TOKEN_PINS = {
    'config.json': 'aef03cacaae68933fe96fc8b9a673601d30a8b20158f0684a0e51295920714a0',
    'sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'special_tokens_map.json': '66715ae6a0dd4aff4fe228bcaaccac6e52c83fd0ba80992c2be7d0e43b362307',
    'tokenizer.json': '4829dfefc91c9f9839cf1b554a99243c8911c43439668e2f974176b1925cd138',
    'tokenizer_config.json': '3e5e7d91646b2277098e245f3e73ba565de0e3c28b9e7ee7a9fcc3cd68ae4121',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw)
    return sha(raw)


def merge(spans):
    """Union only overlapping or touching half-open source intervals."""
    result = []
    for start, end in sorted(spans):
        require(type(start) is int and type(end) is int and 0 <= start < end, 'invalid interval')
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def offline():
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
        os.environ[name] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def execute(source_sha256):
    offline()
    bindings = {str(path): expected for path, expected in INPUTS.values()}
    bindings.update({str(ROOT / name): expected for name, expected in SOURCE_PINS.items()})
    bindings.update({str(TOKENIZER / name): expected for name, expected in TOKEN_PINS.items()})
    bindings[str(Path(__file__).resolve())] = source_sha256
    for name, expected in bindings.items():
        require(sha(Path(name).read_bytes()) == expected, 'frozen input or source changed')
    data = {name: json.loads(path.read_bytes()) for name, (path, _) in INPUTS.items()}
    sample, old, geometry = (data[name] for name in ('selected_inputs', 'original_packs', 'final_geometry'))
    cases = {case['ordinal']: case for case in sample['cases']}
    packs = {pack['ordinal']: pack for pack in old if pack['arm'] == 'source_atoms'}
    require(set(cases) == set(packs) == {1, 2}, 'fixed question scope changed')
    require(len(geometry['rows']) == 70, 'geometry denominator changed')
    additions = [row for row in geometry['rows'] if row['proposed_run_count'] <= 3]
    require(len(additions) == 6 and [r['ordinal'] for r in additions] == [1, 2, 2, 2, 2, 2], 'six-addition scope changed')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)  # One attempt, including failures; never overwrite.
    try:
        import analyze_granularity_three_arm as shared
        render, count, token_hashes = shared.budget_runtime()
        require(token_hashes == TOKEN_PINS, 'tokenizer contract changed')
        require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'flax')), 'model backend imported')
        for ordinal, pack in packs.items():
            require(merge([x['span'] for x in pack['chosen']]) == pack['runs'], 'base geometry changed')
            require(len(pack['runs']) == 3, 'base run cap changed')
            require(render(cases[ordinal], pack['runs']) == pack['rendered_text'], 'base renderer changed')
        hypothetical = []
        for number, row in enumerate(additions, 1):
            ordinal = row['ordinal']
            case, pack = cases[ordinal], packs[ordinal]
            candidate_matches = [x for x in case['candidates']['source_atoms'] if x['span'] == row['span']]
            require(len(candidate_matches) == 1, 'candidate span not unique')
            candidate = candidate_matches[0]
            require(candidate['dense_rank'] == row['dense_rank'], 'candidate rank changed')
            trace = pack['trace'][row['original_trace_index']]
            require(trace['candidate_id'] == candidate['candidate_id'] and trace['action'] == 'skip_runs', 'original skip binding changed')
            require(all(x['candidate_id'] != candidate['candidate_id'] for x in pack['chosen']), 'candidate already selected')
            start, end = candidate['span']
            require(all(max(start, a) >= min(end, b) for a, b in pack['runs']), 'candidate overlaps selected source')
            runs = merge(pack['runs'] + [candidate['span']])  # Each starts from the same original final pack.
            require(len(runs) == row['proposed_run_count'] <= 3, 'final run geometry changed')
            hypothetical.append({**row, 'addition_ordinal': number, 'runs': runs,
                                 'rendered_text': render(case, runs)})
        hypothetical_sha = write(OUTPUT / 'hypothetical_packs.json', hypothetical)
        freeze_sha = write(OUTPUT / 'freeze.json', {
            'schema': 'slac-granularity-final-additions-freeze-v1',
            'status': 'all_six_renders_frozen_before_budget_counts',
            'file_sha256': bindings, 'hypothetical_packs_sha256': hypothetical_sha,
            'hypothetical_count': 6, 'independent_additions': True,
            'references_read': False, 'api_calls': 0, 'scoring_model_calls': 0,
            'complete_evidence_block_proxy_budget': 1024, 'run_cap': 3,
        })
        base_tokens = {ordinal: count(pack['rendered_text']) for ordinal, pack in packs.items()}
        require(base_tokens == {1: 651, 2: 640}, 'original complete-render token count changed')
        rows = []
        for item in hypothetical:
            ordinal = item['ordinal']
            tokens = count(item['rendered_text'])
            old_chars = sum(b - a for a, b in packs[ordinal]['runs'])
            new_chars = sum(b - a for a, b in item['runs'])
            require(new_chars - old_chars == item['span'][1] - item['span'][0], 'added source length changed')
            rows.append({key: item[key] for key in ('addition_ordinal', 'ordinal', 'span', 'dense_rank', 'original_trace_index', 'runs')})
            rows[-1].update({'original_tokens': base_tokens[ordinal], 'proposed_tokens': tokens,
                             'token_delta': tokens - base_tokens[ordinal], 'proposed_run_count': len(item['runs']),
                             'original_source_chars': old_chars, 'proposed_source_chars': new_chars,
                             'added_source_chars': new_chars - old_chars, 'within_1024': tokens <= 1024,
                             'rendered_sha256': sha(item['rendered_text'].encode('utf-8'))})
        for name, expected in bindings.items():
            require(sha(Path(name).read_bytes()) == expected, 'bound artifact drift')
        require(sha((OUTPUT / 'hypothetical_packs.json').read_bytes()) == hypothetical_sha, 'hypothetical render drift')
        report = {
            'schema': 'slac-granularity-final-additions-result-v1', 'status': 'completed',
            'freeze_sha256': freeze_sha, 'hypothetical_packs_sha256': hypothetical_sha,
            'source_sha256': source_sha256, 'input_sha256': {name: h for name, (_, h) in INPUTS.items()},
            'tokenizer_sha256': TOKEN_PINS, 'counts': {'questions': 2, 'original_packs': 2,
                'independent_additions': 6, 'within_run_and_token_caps': sum(row['within_1024'] for row in rows),
                'reference_reads': 0, 'api_calls': 0, 'scoring_model_calls': 0, 'selector_runs': 0},
            'base_tokens': [{'ordinal': n, 'tokens': base_tokens[n]} for n in (1, 2)], 'rows': rows,
            'all_bound_files_unchanged': True,
            'limits': ['Six independent single additions, never accumulated; no replacement pack or revisit selector was produced.',
                       'Whole evidence-block BGE proxy counts with special tokens and no truncation; other messages and generated output excluded.',
                       'Admission tests source geometry and budget only; no semantic, reference-coverage, answer-quality or novelty claim.'],
        }
        report_sha = write(OUTPUT / 'report.json', report)
        return {'status': 'completed', 'report_sha256': report_sha, 'freeze_sha256': freeze_sha,
                'hypothetical_packs_sha256': hypothetical_sha, 'counts': report['counts'], 'rows': rows}
    except Exception as error:
        write(OUTPUT / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                      'rerun_permitted': False, 'api_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    try:
        result = execute(args.source_sha256)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
