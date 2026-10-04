"""Fixed six-case, zero-API renderer diagnosis; local BGE proxy tokens only.

Uses frozen retrieval text, not Refiner provenance. No candidate selection,
normalization, model/answer inference, new judgment, or full dataset traversal.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/research-foundation/offline-20261004'
STAGE = BASE / 'natural-generator-budget-01'
TOKENIZER = Path('D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181')
TOKEN_HASHES = {
    'tokenizer.json': '4829dfefc91c9f9839cf1b554a99243c8911c43439668e2f974176b1925cd138',
    'tokenizer_config.json': '3e5e7d91646b2277098e245f3e73ba565de0e3c28b9e7ee7a9fcc3cd68ae4121',
    'special_tokens_map.json': '66715ae6a0dd4aff4fe228bcaaccac6e52c83fd0ba80992c2be7d0e43b362307',
    'sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'config.json': 'aef03cacaae68933fe96fc8b9a673601d30a8b20158f0684a0e51295920714a0',
}
INPUTS = {
    'packet': (BASE / 'natural-exchange-sample-01/review_packet.json', 'bd4e903a2de589d0f823288a9274428e434b5f57dab86840db00f72bec7984f1'),
    'proposals': (BASE / 'natural-exchange-sample-01/proposals.json', '99e81c58e399be1ff2a66e09129fec7baedec7a5c41004a9443e81b9fa7932a6'),
    'sample_plan': (BASE / 'natural-exchange-sample-01/plan.json', 'ca453244fb94d6dd91f119a92731425adb4684f983b6e11f57b7e845dfd16bba'),
    'jev_result': (ROOT / 'docs/research/results/natural_exchange_jev_20261004.json', '77f996edd3885c13bb8eeb8f037687d1f085d24e16336805512113981f9460d9'),
    'identity_gate': (STAGE / 'identity_gate.json', 'a2bff8bbb85a0eb3cc5d323361c073eb42b141c782c2566325516ddd230b17a0'),
}
SOURCES = (
    'docs/research/probe_natural_generator_budget.py',
    'docs/research/NATURAL_GENERATOR_BUDGET_PROTOCOL_20261004.md',
    'SLAC/llm/service/renderers.py', 'SLAC/llm/service/request_compiler.py',
    'SLAC/llm/io/schemas.py', 'SLAC/llm/io/validators.py',
    'SLAC/llm/memory/merge.py', 'SLAC/retrieval/decision/conditional.py',
)
CAP = 1024
COUNTER_VERSION = 'local-bge-m3-specials-no-truncation-v1'
POLICY = {
    'cases': [1, 2, 3, 4, 5, 6], 'packs_per_case': ['original', 'proposed'],
    'pack_items': 3, 'cap': CAP, 'unit': 'BGE proxy tokens, not generator/provider tokens',
    'projection': ['unit_id->chunk_id', 'doc_id', 'text->passage_text', 'question_id->query_id', 'query->query_text'],
    'unset': ['path', 'rank', 'score', 'role', 'token_est', 'views', 'expansion_depth'],
    'surfaces': ['old_id', 'exchange_core', 'llm_evidence'],
    'invalid_start': 'Record infeasible S without repairing or claiming keep-S is feasible.',
    'proposed_over_budget': 'If S fits, retain S; do not substitute another candidate.',
    'scope': 'Frozen retrieval text renderer diagnosis; not Refiner/FinalIntegrator end-to-end.',
    'api_calls': 0, 'new_judgments': 0, 'model_inference': False,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open('xb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def gate_action(status, s_fits, t_fits):
    require(status in ('accepted', 'rejected', 'abstained'), 'unknown frozen gate status')
    if not s_fits:
        return 'invalid_start'
    if not t_fits:
        return 'keep_S_budget'
    return 'choose_T' if status == 'accepted' else 'keep_S_' + status


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(),
            'a fresh run child of the fixed stage is required')
    blobs = {name: path.read_bytes() for name, (path, expected) in INPUTS.items()}
    for name, (_, expected) in INPUTS.items():
        require(sha(blobs[name]) == expected, 'fixed input changed: ' + name)
    for name, expected in TOKEN_HASHES.items():
        require(sha((TOKENIZER / name).read_bytes()) == expected, 'tokenizer changed: ' + name)
    sources = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    runtime = {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')}
    runtime['python'] = sys.version
    output.mkdir(parents=True)
    write(output / 'started.json', {'policy': POLICY, 'source_sha256': sources,
        'input_sha256': {name: sha(raw) for name, raw in blobs.items()},
        'tokenizer_sha256': TOKEN_HASHES, 'runtime': runtime})

    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in frozen renderer diagnosis')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
        os.environ[name] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from SLAC.llm.io.schemas import EvidenceItem, LLMRequest, ChatMessage, GenerationConfig
    from SLAC.llm.io.validators import validate_llm_request
    from SLAC.llm.service.renderers import SOURCE_RENDER_POLICY, SOURCE_RENDERER_VERSION, render_evidence_block
    from SLAC.llm.service.request_compiler import compile_provider_payload
    from SLAC.retrieval.decision.conditional import Unit, render_evidence

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast, 'expected frozen fast tokenizer')
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    data = {name: json.loads(raw) for name, raw in blobs.items()}
    packet, proposals, observed = data['packet']['cases'], data['proposals'], data['jev_result']['rows']
    require([r['ordinal'] for r in packet] == [r['ordinal'] for r in proposals]
            == [r['ordinal'] for r in observed] == POLICY['cases'], 'six-case order changed')
    require(len(data['sample_plan']['tokenizer_sha256']) == 4, 'old tokenizer inventory changed')
    require(all(TOKEN_HASHES[Path(p).name] == h for p, h in data['sample_plan']['tokenizer_sha256'].items()),
            'old tokenizer bindings differ')
    records, summaries = [], []
    for case, proposal, old in zip(packet, proposals, observed, strict=True):
        require(all(case[k] == proposal[k] for k in case), 'packet/proposal identity changed')
        packs = {}
        for kind in POLICY['packs_per_case']:
            units = case[kind + '_pack']
            require(len(units) == 3 and [u['unit_id'] for u in units] == proposal[kind + '_ids'], 'fixed pack changed')
            require([u['source_order'] for u in units] == sorted({u['source_order'] for u in units}), 'source order changed')
            require(all(sha(u['text'].encode()) == u['retrieval_text_sha256'] for u in units), 'frozen text hash changed')
            evidence = [EvidenceItem(chunk_id=u['unit_id'], doc_id=case['doc_id'], passage_text=u['text'],
                            query_id=case['question_id'], query_text=case['query']) for u in units]
            strings = {
                'old_id': '\n\n'.join(f"[{u['unit_id']}]\n{u['text']}" for u in units),
                'exchange_core': render_evidence(tuple(Unit(u['unit_id'], u['text'], u['source_order'], case['doc_id']) for u in units)),
                'llm_evidence': render_evidence_block(evidence, preserve_source_text=True),
            }
            counts = {name: count(value) for name, value in strings.items()}
            require(sha(strings['old_id'].encode()) == proposal[kind + '_render_sha256']
                    and counts['old_id'] == proposal[kind + '_tokens'], 'old count/hash differs')
            require(counts['exchange_core'] == old['core_' + kind + '_tokens'], 'exchange core count differs')
            require(all(ev.passage_text == u['text'] and u['text'] in strings['llm_evidence']
                        for ev, u in zip(evidence, units, strict=True))
                    and strings['llm_evidence'].endswith(units[-1]['text']), 'passage preservation failed')
            receipt = {'schema': 'slac-source-evidence-budget-v1', 'renderer_version': SOURCE_RENDERER_VERSION,
                'rendered_sha256': sha(strings['llm_evidence'].encode()), 'counter_version': COUNTER_VERSION,
                'count': counts['llm_evidence'], 'max_tokens': CAP}
            request = LLMRequest(schema_version='slac_llm_request_v1', record_type='answer_request',
                request_id=f"frozen-render-{case['ordinal']}-{kind}", session_id=None,
                query_id=case['question_id'], query_text=case['query'], provider='openai_compatible',
                model_name='offline-no-model', api_base='https://example.invalid', api_key_env='UNREAD_OFFLINE_KEY',
                generation_config=GenerationConfig(), messages=[ChatMessage('user', case['query'])],
                evidence=evidence, options={'evidence_render_policy': SOURCE_RENDER_POLICY},
                meta={'source_evidence_budget': receipt})
            validate_llm_request(request)
            fits = counts['llm_evidence'] <= CAP
            payload, error = None, None
            try:
                payload = compile_provider_payload(request)
            except ValueError as exc:
                require(not fits and str(exc) == 'source_evidence_budget count exceeds max_tokens', 'unexpected compiler failure')
                error = str(exc)
            else:
                require(fits and payload['messages'][-1]['content'] == strings['llm_evidence'], 'compiled surface differs')
                require(compile_provider_payload(LLMRequest.from_dict(deepcopy(request.to_dict()))) == payload, 'roundtrip differs')
            packs[kind] = {'strings': strings, 'tokens': counts,
                'string_sha256': {n: sha(t.encode()) for n, t in strings.items()},
                'llm_fits': fits, 'llm_request': request.to_dict(), 'compiled_payload': payload,
                'compiler_error': error, 'roundtrip_equal': True if fits else None,
                'selected_evidence': [asdict(ev) for ev in evidence]}
        s_fits, t_fits = packs['original']['llm_fits'], packs['proposed']['llm_fits']
        historical = {'jev_full': old['full_gate_status'], 'jev_gain_only': old['gain_only_status'],
                      'review_a': old['review_a_gate'], 'review_b': old['review_b_gate']}
        summary = {'ordinal': case['ordinal'], 'original_tokens': packs['original']['tokens'],
            'proposed_tokens': packs['proposed']['tokens'], 's_fits': s_fits, 't_fits': t_fits,
            'frozen_labels': {n: old[n] for n in ('labels', 'review_a', 'review_b')},
            'frozen_gate_status': historical,
            'old_actions': {n: gate_action(v, True, True) for n, v in historical.items()},
            'projected_budget_actions': {n: gate_action(v, s_fits, t_fits) for n, v in historical.items()}}
        summaries.append(summary)
        records.append({'ordinal': case['ordinal'], 'frozen_case': case, 'packs': packs, 'summary': summary})
    require('torch' not in sys.modules and 'tensorflow' not in sys.modules and 'jax' not in sys.modules,
            'neural model runtime unexpectedly imported')
    require(sources == {name: sha((ROOT / name).read_bytes()) for name in SOURCES}, 'source drift')
    require(all(sha(path.read_bytes()) == expected for path, expected in INPUTS.values()), 'input drift')
    require(all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items()), 'tokenizer drift')
    write(output / 'rendered_packs.json', {'cases': records})
    result = {'schema': 'slac-natural-generator-budget-v1', 'status': 'passed', 'rows': summaries,
        'policy': POLICY, 'source_sha256': sources, 'input_sha256': {n: sha(b) for n, b in blobs.items()},
        'tokenizer_sha256': TOKEN_HASHES, 'runtime': runtime,
        'rendered_packs_sha256': sha((output / 'rendered_packs.json').read_bytes()),
        'whole_string_counts': 36, 'api_calls': 0, 'new_model_judgments': 0,
        'limits': ['BGE proxy token counts; not answer-model tokens, full-context admission, usage or cost.',
                   'Frozen retrieval text with declared minimal metadata; not Refiner or FinalIntegrator end-to-end.',
                   'Historical labels only; no new model predictions, relabeling, or answer-quality measurement.']}
    write(output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STAGE / 'run-01')
    result = run(parser.parse_args().output)
    print(json.dumps({'status': result['status'], 'rows': result['rows'], 'api_calls': 0}, sort_keys=True))
