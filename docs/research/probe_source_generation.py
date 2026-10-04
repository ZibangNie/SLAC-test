"""Authored source-generation compilation; toy bytes, no provider or natural data.

The fixture and compile-only adapter are shared with the integration tests to
avoid a second implementation of this input construction. pytest is a development
dependency; this script imports the helper definitions, without running tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.abc
import json
from pathlib import Path
import runpy
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / 'artifacts/research-foundation/offline-20261004/source-generation-01'
SOURCES = (
    'docs/research/probe_source_generation.py', 'tests/research/test_final_integrator_source.py',
    'tests/research/test_integration_source_mode.py', 'tests/research/test_integration_render_budget.py',
    'tests/research/test_source_evidence_renderer.py',
    'SLAC/integration/evidence/source_mode.py', 'SLAC/integration/evidence/normalizers.py',
    'SLAC/integration/evidence/budgeter.py', 'SLAC/integration/evidence/selectors.py',
    'SLAC/integration/orchestrator/final_integrator.py', 'SLAC/integration/prompt/builders.py',
    'SLAC/integration/prompt/templates.py', 'SLAC/integration/io/schemas.py', 'SLAC/integration/io/validators.py',
    'SLAC/llm/service/renderers.py', 'SLAC/llm/service/request_compiler.py',
    'SLAC/llm/io/validators.py', 'SLAC/llm/io/schemas.py', 'SLAC/llm/memory/merge.py',
    'SLAC/retrieval/dataio/source_records.py', 'SLAC/retrieval/decision/refiner_bridge.py',
    'SLAC/retrieval/decision/conditional.py', 'SLAC/retrieval/pack/evidence_packer.py',
    'SLAC/retrieval/pack/token_counter.py', 'SLAC/retrieval/pack/dedup.py', 'SLAC/retrieval/pack/mmr.py',
    'SLAC/retrieval/preprocess/anchor_fields.py', 'SLAC/retrieval/utils/text_utils.py',
    'SLAC/retrieval/schemas/records.py', 'SLAC/refiner/pipeline/assemble/source_document_view.py',
    'SLAC/refiner/pipeline/assemble/source_coverage.py', 'SLAC/refiner/pipeline/assemble/source_atomizer.py',
    'SLAC/refiner/pipeline/assemble/build_refiner_input.py',
)
CASES = (('legacy_512', False, 512, 2), ('source_4096', True, 4096, 2), ('source_512', True, 512, 1))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('xb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.')
               for p in ('torch', 'transformers', 'tokenizers', 'sentence_transformers', 'faiss')):
            raise RuntimeError('model/tokenizer runtime forbidden in this synthetic probe')
        return None


def run(output):
    output = Path(output).resolve()
    if not output.is_relative_to(STAGE.resolve()) or output == STAGE.resolve() or output.exists():
        raise ValueError('a fresh subdirectory of the fixed stage is required')
    output.mkdir(parents=True)
    bindings = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    write(output / 'started.json', {'source_sha256': bindings, 'cases': CASES,
        'scope': 'Fixed authored inputs from the bound test helper; no natural data/model/API/tokenizer.'})
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in source generation probe')
    socket.socket.connect = denied
    socket.create_connection = denied
    sys.meta_path.insert(0, NoModels())
    sys.path.insert(0, str(ROOT))
    helpers = runpy.run_path(str(ROOT / 'tests/research/test_final_integrator_source.py'))
    fixture, adapter_type = helpers['fixture'], helpers['CompileOnlyAdapter']
    FinalIntegrator, LLMRequest = helpers['FinalIntegrator'], helpers['LLMRequest']
    compile_payload = helpers['compile_provider_payload']
    rows, full_request = [], None
    source_records = None
    for name, source_mode, cap, expected_count in CASES:
        req, mode, raw = fixture(cap=cap)
        adapter = adapter_type()
        response, artifacts = FinalIntegrator(llm_adapter=adapter,
            source_mode=mode if source_mode else None).run_with_artifacts(req)
        if response.status != 'ok' or len(adapter.payloads) != 1 or len(artifacts.selected_evidence) != expected_count:
            raise ValueError('fixed synthetic outcome differs; preserve this run without reselection')
        payload = adapter.payloads[0]
        block = payload['messages'][-1]['content']
        source_records = {key: record.meta['refiner_source'] for key, record in mode.chunk_lookup.items()}
        restored_count = sum(ev.passage_text == source_records[ev.chunk_id]['text'] for ev in artifacts.selected_evidence)
        receipt = artifacts.llm_request.meta.get('source_evidence_budget')
        if source_mode:
            if (block != artifacts.prompt_bundle.evidence_context_block
                    or restored_count != expected_count or not block.endswith(raw[expected_count - 1])
                    or receipt['count'] != len(block.encode()) or receipt['count'] > cap):
                raise ValueError('source content/render budget differs')
        elif receipt is not None or restored_count != 0:
            raise ValueError('legacy fixture unexpectedly changed semantics')
        serialized = artifacts.llm_request.to_dict()
        if compile_payload(LLMRequest.from_dict(serialized)) != payload:
            raise ValueError('serialized request compilation differs')
        if name == 'source_4096':
            full_request = serialized
        rows.append({'case': name, 'source_mode': source_mode, 'configured_limit': cap,
            'measurement_unit': 'toy UTF-8 bytes; old token_est fields are artificial values',
            'input_records': req.context.retrieval_artifacts['packed_evidence'],
            'selected_evidence': [asdict(ev) for ev in artifacts.selected_evidence],
            'selected_count': expected_count, 'exact_source_passages': restored_count,
            'sum_unchanged_token_est': sum(ev.token_est for ev in artifacts.selected_evidence),
            'actual_evidence_block_bytes': len(block.encode()), 'preview': artifacts.prompt_bundle.evidence_context_block,
            'llm_request': serialized, 'compiled_payload': payload, 'trace': asdict(response.trace),
            'serialization_roundtrip_equal': True, 'stub_invocations': len(adapter.requests), 'provider_calls': 0})
    drift = []
    for name in ('passage', 'token_est', 'missing_policy', 'legacy_policy'):
        # from_dict may retain nested dictionaries. Isolate each negative case
        # from the successful request stored above and from other mutations.
        req = LLMRequest.from_dict(deepcopy(full_request))
        if name == 'passage':
            req.evidence[-1].passage_text += 'changed'
        elif name == 'token_est':
            req.evidence[0].token_est += 1
        elif name == 'missing_policy':
            req.options.pop('evidence_render_policy')
        else:
            req.options['evidence_render_policy'] = 'append_as_context_block'
        try:
            compile_payload(req)
        except ValueError as error:
            drift.append({'mutation': name, 'rejected': True, 'reason': str(error)})
        else:
            raise ValueError('compiled request drift bypassed source budget binding')
    for row in rows:
        if compile_payload(LLMRequest.from_dict(deepcopy(row['llm_request']))) != row['compiled_payload']:
            raise ValueError('saved successful request was changed by a later probe')
    if bindings != {name: sha((ROOT / name).read_bytes()) for name in SOURCES}:
        raise ValueError('bound source drift')
    write(output / 'compiled_cases.json', {'source_records': source_records, 'cases': rows, 'drift': drift})
    result = {'schema': 'slac-source-generation-probe-v1', 'status': 'passed',
        'source_sha256': bindings, 'compiled_cases_sha256': sha((output / 'compiled_cases.json').read_bytes()),
        'cases': [{k: row[k] for k in ('case', 'source_mode', 'configured_limit', 'selected_count',
            'exact_source_passages', 'sum_unchanged_token_est', 'actual_evidence_block_bytes', 'provider_calls')}
            for row in rows], 'drift': drift,
        'scope': {'api_calls': 0, 'natural_data_read': False, 'model_inference': False, 'tokenizer_read': False,
                  'training': False, 'key_reads': 0, 'toy_counter': 'UTF-8 bytes', 'synthetic_cases': 3},
        'limits': ['Actual orchestration with a compile-only stub; no model output or quality measurement.',
                   'Complete actual evidence-block budget only; system/query/memory/total context are excluded.',
                   'Budget binding assumes a truthful injected deterministic counter; it is not source authentication.',
                   'This demo fixes retrieval_packed_evidence; tests separately cover synthetic reranker_pack_bridge.',
                   'Existing two-stage greedy order is retained; no optimality or novelty claim.']}
    write(output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STAGE / 'run-01')
    result = run(parser.parse_args().output)
    print(json.dumps({'status': result['status'], 'cases': result['cases'], 'provider_calls': 0}))
