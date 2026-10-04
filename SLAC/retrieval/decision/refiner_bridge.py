"""Opt-in Refiner source snapshots; coordinate coverage is not semantic support.

Capture validates each supplied row against an immutable coverage index, not
against an alleged whole-export identity. No file, model, score or network I/O
occurs. Existing conditional request keys bind the visible decision inputs;
the separate receipt records the current source provenance without relabeling
or transferring a cached decision based on source-interval coverage.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json

from SLAC.refiner.pipeline.assemble.source_coverage import NativeCoverageIndex, SourceCoverage
from .conditional import (
    COUNTER_VERSION, PROMPT_VERSION, QUESTION_VERSION, RENDERER_VERSION,
    SCHEMA_VERSION, ConditionalState, FrozenRequest, Unit, build_request,
)

STATE_VERSION = 'slac-refiner-source-bridge-state-v1'
ARMS = ('standalone', 'plain_conditional')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _pair(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 2
             and all(type(v) is int for v in value), f'{name} must be two integer offsets')
    return tuple(value)


def _validate_snapshot(snapshot):
    _require(isinstance(snapshot, RefinerSourceSnapshot), 'expected RefinerSourceSnapshot')
    index = snapshot._coverage_index
    _require(isinstance(index, NativeCoverageIndex), 'expected immutable NativeCoverageIndex')
    unit = snapshot.unit
    _require(isinstance(unit, Unit), 'snapshot unit must be a Unit')
    _require(isinstance(unit.id, str) and bool(unit.id.strip()), 'snapshot unit ID must be nonblank')
    _require(type(snapshot.atom_span) is tuple, 'snapshot atom span must be immutable')
    start, end = _pair(snapshot.atom_span, 'atom span')
    coverage = index.cover_atoms(start, end)
    _require(isinstance(unit.doc_id, str) and unit.doc_id == index.view.doc_id,
             'snapshot document differs from source view')
    _require(type(unit.order) is int and unit.order == start, 'snapshot order must equal atom_start')
    _require(isinstance(unit.text, str) and unit.text == index.view.render(start, end),
             'snapshot text differs from complete source render')
    _require(isinstance(snapshot.coverage, SourceCoverage) and snapshot.coverage == coverage,
             'snapshot coverage differs from source geometry or text hash')
    _require(isinstance(snapshot.source_document_sha256, str)
             and snapshot.source_document_sha256 == _digest(index.view.source_text),
             'snapshot document hash differs from source view')


@dataclass(frozen=True)
class RefinerSourceSnapshot:
    unit: Unit
    coverage: SourceCoverage
    source_document_sha256: str
    atom_span: tuple[int, int]
    _coverage_index: NativeCoverageIndex = field(repr=False)

    def __post_init__(self):
        _validate_snapshot(self)


def capture_refiner_source_chunk(chunk: Mapping, coverage_index: NativeCoverageIndex) -> RefinerSourceSnapshot:
    """Copy one exact source-mode row; ignore rank, model scores and other metadata."""
    _require(isinstance(chunk, Mapping), 'chunk must be a mapping')
    _require(isinstance(coverage_index, NativeCoverageIndex), 'expected immutable NativeCoverageIndex')
    view = coverage_index.view
    _require(isinstance(chunk.get('doc_id'), str) and chunk['doc_id'] == view.doc_id,
             'chunk document differs from source view')
    _require(chunk.get('text_mode') == 'source_document' and chunk.get('source') == 'refiner_source_view',
             'chunk must explicitly use refiner source-document mode')
    _require(chunk.get('source_coordinate_system') == view.coordinate_system, 'chunk coordinate contract differs')
    start, end = _pair((chunk.get('atom_start'), chunk.get('atom_end')), 'atom span')
    coverage = coverage_index.cover_atoms(start, end)
    _require(type(chunk.get('num_atoms')) is int and chunk['num_atoms'] == end - start, 'chunk atom count differs')
    _require(_pair(chunk.get('source_char_span'), 'source character span') == coverage.source_char_span,
             'chunk source character geometry differs')
    _require(isinstance(chunk.get('text'), str) and chunk['text'] == view.render(start, end),
             'chunk text differs from complete source render')
    _require(type(chunk.get('chunk_index')) is int and chunk['chunk_index'] >= 0,
             'chunk_index must be a nonnegative integer')
    _require(isinstance(chunk.get('chunk_id'), str) and bool(chunk['chunk_id'].strip()), 'chunk ID must be nonblank')
    source_hash = _digest(view.source_text)
    if 'source_document_sha256' in chunk:
        _require(chunk['source_document_sha256'] == source_hash, 'supplied document hash differs')
    unit = Unit(id=chunk['chunk_id'], text=chunk['text'], order=start, doc_id=view.doc_id)
    return RefinerSourceSnapshot(unit, coverage, source_hash, (start, end), coverage_index)


def _consistent_documents(snapshots):
    contracts = {}
    for snapshot in snapshots:
        _validate_snapshot(snapshot)
        view = snapshot._coverage_index.view
        contract = (snapshot.source_document_sha256, view.coordinate_system,
                    view.model_atoms, view.atom_char_spans)
        previous = contracts.setdefault(snapshot.unit.doc_id, contract)
        _require(previous == contract,
                 'same-document snapshots have different source versions, coordinate contracts or base atoms')


@dataclass(frozen=True)
class RefinerSourceRequest:
    request: FrozenRequest
    _visible_snapshots: tuple[RefinerSourceSnapshot, ...] = field(repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        _require(isinstance(self.request, FrozenRequest), 'expected frozen core request')
        payload = self.request.payload()  # Existing core verifies canonical bytes and its cache binding.
        binding = json.loads(self.request.binding_bytes)
        _require(binding['arm'] in ARMS and binding['state_version'] == STATE_VERSION, 'wrong source bridge request contract')
        _require(type(self._visible_snapshots) is tuple, 'visible snapshots must be an immutable tuple')
        _consistent_documents(self._visible_snapshots)
        visible = payload['state'].get('current_pack', []) + [payload['state']['candidate']]
        expected = [Unit(**row) for row in visible]
        _require([snapshot.unit for snapshot in self._visible_snapshots] == expected,
                 'source receipt snapshots differ from exact model-visible units')
        return payload, binding

    def provenance_receipt(self) -> dict:
        """Fresh local receipt; none of these provenance fields become model input."""
        payload, binding = self._validate()
        candidate_id = payload['state']['candidate']['id']
        rows = []
        for snapshot in self._visible_snapshots:
            coverage = snapshot.coverage
            rows.append({
                'role': 'candidate' if snapshot.unit.id == candidate_id else 'current_pack',
                'unit_id': snapshot.unit.id, 'doc_id': snapshot.unit.doc_id, 'order': snapshot.unit.order,
                'source_document_sha256': snapshot.source_document_sha256,
                'source_coordinate_system': snapshot._coverage_index.view.coordinate_system,
                'atom_span': list(snapshot.atom_span), 'source_char_span': list(coverage.source_char_span),
                'text_sha256': coverage.text_sha256,
                'native_coverage': {
                    'full_native_unit_ids': list(coverage.full_native_unit_ids),
                    'partial_native_unit_ids': list(coverage.partial_native_unit_ids),
                    'exact_native_unit_id': coverage.exact_native_unit_id, 'gap_chars': coverage.gap_chars,
                    'hits': [{'native_unit_id': hit.native_unit_id, 'kind': hit.kind,
                              'native_source_span': list(hit.native_source_span),
                              'overlap_source_span': list(hit.overlap_source_span)} for hit in coverage.hits],
                },
            })
        return {'schema': 'slac-refiner-source-provenance-v1', 'request_key': self.request.cache_key,
                'arm': binding['arm'], 'visible_snapshots': rows,
                'scope': 'Exact source-coordinate attribution only; not semantic support or a rule for reusing decisions.'}


def build_refiner_source_request(
    query: str,
    current_pack: Sequence[RefinerSourceSnapshot],
    candidate: RefinerSourceSnapshot,
    *,
    arm: str,
    endpoint_id: str,
    model_id: str,
    expected_response_model: str,
    token_counter: Callable[[str], int],
    max_tokens: int = 1024,
    max_payload_bytes: int = 65536,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
    question_version: str = QUESTION_VERSION,
    renderer_version: str = RENDERER_VERSION,
    counter_version: str = COUNTER_VERSION,
) -> RefinerSourceRequest:
    """Validate whole source snapshots then delegate all request/cache work to core.

    Same-document snapshots must share source version, coordinate system and
    base atom text/partition so atom_start orders are comparable. Different
    chunk spans and overlapping spans on that basis remain allowed.
    Core still rejects duplicate IDs, source positions and complete texts. The
    source version stays in the receipt, so the same complete visible request
    can keep its existing key when unselected source text changes. No inferred
    relations, labels or interval-based decision reuse are provided here.
    """
    _require(arm in ARMS, 'source bridge supports standalone and plain_conditional only; no automatic relations')
    _require(isinstance(current_pack, Sequence) and not isinstance(current_pack, (str, bytes, bytearray)),
             'current_pack must be a sequence of source snapshots')
    current = tuple(current_pack)
    _consistent_documents((*current, candidate))
    state = ConditionalState(query, tuple(s.unit for s in current), candidate.unit, version=STATE_VERSION)
    request = build_request(state, arm=arm, endpoint_id=endpoint_id, model_id=model_id,
        expected_response_model=expected_response_model, token_counter=token_counter,
        max_tokens=max_tokens, max_payload_bytes=max_payload_bytes,
        prompt_version=prompt_version, schema_version=schema_version, question_version=question_version,
        renderer_version=renderer_version, counter_version=counter_version)
    ordered = tuple(sorted(current, key=lambda s: (s.unit.doc_id, s.unit.order, s.unit.id)))
    visible = (candidate,) if arm == 'standalone' else (*ordered, candidate)
    return RefinerSourceRequest(request, visible)
