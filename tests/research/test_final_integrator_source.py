"""Actual orchestration and payload compilation with authored text and a stub adapter."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import socket

import pytest

from SLAC.integration.evidence.source_mode import SourceModeConfig, restore_source_candidates
from SLAC.integration.io.schemas import (
    IntegrationContext, IntegrationRequest, LLMConfig, PipelineConfig,
)
from SLAC.integration.orchestrator.final_integrator import FinalIntegrator
from SLAC.integration.prompt.builders import build_prompt_bundle, to_llm_evidence
from SLAC.llm.io.schemas import LLMRequest
from SLAC.llm.io.validators import validate_llm_request
from SLAC.llm.service.renderers import SOURCE_RENDER_POLICY, render_evidence_block
from SLAC.llm.service.request_compiler import compile_provider_payload
from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.dataio.source_records import capture_or_restore_source_meta
from SLAC.retrieval.pack.evidence_packer import pack_evidence
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record
from SLAC.retrieval.schemas.records import ChunkRecord, RetrievalCandidate


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('network forbidden in synthetic integration tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def toy_bytes(text):
    return len(text.encode('utf-8'))


class CompileOnlyAdapter:
    """Exercise the real compiler, then return a labeled fake result, never a provider."""
    def __init__(self, *, tamper=False):
        self.requests, self.payloads, self.tamper = [], [], tamper

    def invoke(self, request):
        self.requests.append(request)
        if self.tamper:
            request.evidence[-1].passage_text += 'changed after budget'
        validate_llm_request(request)
        self.payloads.append(compile_provider_payload(request))
        return {'status': 'ok', 'answer_text': 'Synthetic stub only.', 'response_id': 'stub'}


def fixture(*, policy='reconstruct', cap=4096, route='retrieval_packed_evidence'):
    bodies = ('Ａmber console has four ports.', 'Blue cabinet remains locked.')
    raw = (' \t' + bodies[0] + '\r\n', '  ' + bodies[1] + '\r\n\t')
    lookup, indexes = {}, {}
    for i, body in enumerate(bodies):
        doc_id, chunk_id = f'authored-doc-{i}', f'authored-chunk-{i}'
        view = DocumentSourceView(doc_id, 'synthetic-python-chars', raw[i], (body,),
            ((0, len(raw[i])),), (tuple((2 + j, 3 + j) for j in range(len(body))),))
        indexes[doc_id] = build_native_coverage_index(view, [
            {'native_unit_id': f'authored-native-{i}', 'source_span': [2, 2 + len(body)], 'source_text': body}])
        row = {'chunk_id': chunk_id, 'doc_id': doc_id, 'chunk_index': 0, 'atom_start': 0, 'atom_end': 1,
               'num_atoms': 1, 'text': raw[i], 'source': 'refiner_source_view', 'text_mode': 'source_document',
               'source_coordinate_system': view.coordinate_system, 'source_char_span': [0, len(raw[i])]}
        saved = capture_or_restore_source_meta(row, kind='chunk', source_indexes=indexes)
        chunk = ChunkRecord(doc_id, chunk_id, 0, 0, 1, raw[i], 1, ['Panel'], 1,
                            token_est=1, meta={'refiner_source': saved})
        enrich_chunk_record(chunk)
        lookup[chunk_id] = chunk
    candidates = [RetrievalCandidate(c.chunk_id, c.doc_id, c.text, c.path, c.depth,
                    token_est=1, retrieve_rank_fused=i + 1, best_chunk_score=2 - i,
                    source_views=['chunk']) for i, c in enumerate(lookup.values())]
    packed, _ = pack_evidence(candidates, {'pack': {'evidence_budget_tokens': 2, 'max_packed_items': 2}})
    assert len(packed) == 2
    records = [asdict(item) for item in packed]
    context = IntegrationContext(retrieval_artifacts={'packed_evidence': deepcopy(records)})
    if route == 'reranker_pack_bridge':
        # A second synthetic entry verifies source selection priority cannot bypass the same contract.
        context.reranker_artifacts = {'pack_bridge': deepcopy(records)}
    config = SourceModeConfig(lookup, indexes, policy, toy_bytes, 'toy-utf8-bytes-v1')
    req = IntegrationRequest('slac_integration_request_v1', 'integration_request',
        'authored-request', None, 'authored-query', 'Describe the two devices.', context=context,
        pipeline_config=PipelineConfig(use_retrieval=True, use_reranker=route == 'reranker_pack_bridge',
            max_evidence_items=2, max_evidence_tokens=cap,
            llm=LLMConfig('openai_compatible', 'synthetic-no-model',
                          'https://example.invalid', 'SYNTHETIC_UNREAD_KEY')))
    return req, config, raw


@pytest.mark.parametrize('route', ['retrieval_packed_evidence', 'reranker_pack_bridge'])
def test_source_request_actual_preview_budget_compiler_and_roundtrip_agree(route):
    req, mode, raw = fixture(route=route)
    before = deepcopy(req.context)
    adapter = CompileOnlyAdapter()
    response, artifacts = FinalIntegrator(llm_adapter=adapter, source_mode=mode).run_with_artifacts(req)
    assert response.status == 'ok' and response.trace.candidate_source == route
    assert len(adapter.requests) == 1
    assert req.context == before
    assert [e.passage_text for e in artifacts.selected_evidence] == list(raw)
    assert [e.token_est for e in artifacts.selected_evidence] == [1, 1]
    block = adapter.payloads[0]['messages'][-1]['content']
    assert block == artifacts.prompt_bundle.evidence_context_block
    assert block.endswith(raw[-1])  # No final strip of CRLF/tab.
    for text in raw:
        assert text in block
    receipt = artifacts.llm_request.meta['source_evidence_budget']
    assert artifacts.llm_request.options['evidence_render_policy'] == SOURCE_RENDER_POLICY
    assert receipt['count'] == toy_bytes(block) <= req.pipeline_config.max_evidence_tokens
    assert receipt['rendered_sha256'] == hashlib.sha256(block.encode()).hexdigest()
    assert response.trace.meta['source_evidence_budget'] == receipt
    assert response.trace.meta['source_text_changed_ids'] == ['authored-chunk-0', 'authored-chunk-1']
    reloaded = LLMRequest.from_dict(artifacts.llm_request.to_dict())
    validate_llm_request(reloaded)
    assert compile_provider_payload(reloaded) == adapter.payloads[0]


def test_exact_whole_render_limit_changes_admission_without_changing_rank_policy():
    req, mode, raw = fixture()
    prepared = restore_source_candidates(req.context.retrieval_artifacts['packed_evidence'],
        source_mode=mode, query_id=req.query_id, query_text=req.query_text,
        source_name='retrieval_packed_evidence')
    first = render_evidence_block(to_llm_evidence(prepared[:1]), preserve_source_text=True)
    req.pipeline_config.max_evidence_tokens = toy_bytes(first)
    adapter = CompileOnlyAdapter()
    _, artifacts = FinalIntegrator(llm_adapter=adapter, source_mode=mode).run_with_artifacts(req)
    assert [e.chunk_id for e in artifacts.selected_evidence] == ['authored-chunk-0']
    assert artifacts.llm_request.meta['source_evidence_budget']['count'] == toy_bytes(first)
    assert adapter.payloads[0]['messages'][-1]['content'] == first
    assert artifacts.selected_evidence[0].passage_text == raw[0]
    legacy = CompileOnlyAdapter()
    _, old = FinalIntegrator(llm_adapter=legacy).run_with_artifacts(req)
    assert len(old.selected_evidence) == 2  # Original estimate-only route is preserved.
    assert 'source_evidence_budget' not in old.llm_request.meta
    assert old.llm_request.options['evidence_render_policy'] == 'append_as_context_block'


@pytest.mark.parametrize('failure', ['exact_on_normalized', 'summary', 'alias_conflict'])
def test_invalid_source_input_fails_before_adapter(failure):
    req, mode, _ = fixture(policy='exact' if failure == 'exact_on_normalized' else 'reconstruct')
    row = req.context.retrieval_artifacts['packed_evidence'][0]
    if failure == 'summary':
        row['text'] = 'A shortened summary.'
    elif failure == 'alias_conflict':
        row['passage_text'] = row['text'] + 'different'
    adapter = CompileOnlyAdapter()
    with pytest.raises(ValueError):
        FinalIntegrator(llm_adapter=adapter, source_mode=mode).run_with_artifacts(req)
    assert adapter.requests == []


def test_empty_selected_source_has_bound_empty_render_and_no_extra_message():
    req, mode, _ = fixture(cap=1)
    adapter = CompileOnlyAdapter()
    _, artifacts = FinalIntegrator(llm_adapter=adapter, source_mode=mode).run_with_artifacts(req)
    assert artifacts.selected_evidence == []
    assert artifacts.prompt_bundle.evidence_context_block == ''
    assert artifacts.llm_request.meta['source_evidence_budget']['count'] == 0
    assert adapter.payloads[0]['messages'][-1]['content'] == req.query_text


def test_compiler_blocks_text_drift_after_orchestrator_budget():
    req, mode, _ = fixture()
    adapter = CompileOnlyAdapter(tamper=True)
    response, _ = FinalIntegrator(llm_adapter=adapter, source_mode=mode).run_with_artifacts(req)
    assert response.status == 'error'
    assert 'rendered_sha256' in response.trace.errors[-1]
    assert adapter.payloads == []


def test_direct_request_builder_revalidates_source_and_preview():
    req, mode, _ = fixture()
    prepared = restore_source_candidates(req.context.retrieval_artifacts['packed_evidence'], source_mode=mode)
    integrator = FinalIntegrator(llm_adapter=CompileOnlyAdapter(), source_mode=mode)
    preview = build_prompt_bundle(req, prepared, preserve_source_text=True)
    integrator.build_llm_request(req, prepared, preview)
    prepared[0].meta['refiner_source']['text'] = 'tampered source metadata'
    with pytest.raises(ValueError, match='verified source snapshot'):
        integrator.build_llm_request(req, prepared, preview)
