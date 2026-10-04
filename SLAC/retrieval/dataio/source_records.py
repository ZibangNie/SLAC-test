"""Persist and revalidate exact source snapshots beside normalized retrieval text.

Saved metadata is untrusted until checked against the explicitly supplied source
index. Only this module's JSON loader opens a file; no model or network is used.
Legacy records without source-mode fields keep their existing reader behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import NativeCoverageIndex, build_native_coverage_index
from SLAC.retrieval.decision.refiner_bridge import RefinerSourceSnapshot, capture_refiner_source_chunk
from SLAC.retrieval.schemas.records import ChunkRecord
from SLAC.retrieval.utils.text_utils import normalize_text_basic

INDEX_SCHEMA = 'slac-refiner-source-indexes-v1'
RECORD_SCHEMA = 'slac-refiner-source-record-v1'
SOURCE_FIELDS = ('text_mode', 'source_coordinate_system', 'source_char_span', 'source_document_sha256')
_MISSING = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON field in source registry')
        result[key] = value
    return result


def load_source_indexes(path: str | Path) -> dict[str, NativeCoverageIndex]:
    """Load just the named JSON registry and validate immutable source contracts."""
    data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    _require(isinstance(data, dict) and set(data) == {'schema', 'documents'}
             and data['schema'] == INDEX_SCHEMA, 'invalid source index registry schema')
    _require(isinstance(data['documents'], list), 'source registry documents must be a list')
    result = {}
    for document in data['documents']:
        _require(isinstance(document, dict) and set(document) == {'view', 'native_units'}
                 and isinstance(document['view'], dict), 'invalid source registry document')
        try:
            view = DocumentSourceView(**document['view'])
            index = build_native_coverage_index(view, document['native_units'])
        except TypeError as error:
            raise ValueError('invalid source registry constructor fields') from error
        _require(view.doc_id not in result, 'duplicate source document ID')
        result[view.doc_id] = index
    return result


def _field(obj, meta, name, default=_MISSING):
    values = [scope[name] for scope in (obj, meta) if name in scope]
    if not values:
        _require(default is not _MISSING, f'missing source record field: {name}')
        return default
    _require(all(type(value) is type(values[0]) and value == values[0] for value in values),
             f'conflicting top-level/meta field: {name}')
    return values[0]


def _integer(value, name):
    _require(type(value) is int and value >= 0, f'{name} must be a nonnegative integer')
    return value


def _pair(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 2
             and all(type(x) is int for x in value), f'{name} must contain two integer offsets')
    return list(value)


def _nonblank(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f'{name} must be a nonblank string')
    return value


def _index(doc_id, indexes):
    _require(isinstance(indexes, Mapping), 'source-mode records require explicit source_indexes')
    try:
        value = indexes[doc_id]
    except KeyError as error:
        raise ValueError('source record document missing from source_indexes') from error
    _require(isinstance(value, NativeCoverageIndex) and value.view.doc_id == doc_id,
             'source index lookup/document binding differs')
    return value


def _leaf_span(obj, meta):
    has_index = 'atom_index' in obj or 'atom_index' in meta
    has_start = 'atom_start' in obj or 'atom_start' in meta
    has_end = 'atom_end' in obj or 'atom_end' in meta
    _require(has_index or (has_start and has_end), 'leaf requires atom_index or complete atom span')
    start = _integer(_field(obj, meta, 'atom_index' if has_index else 'atom_start'), 'leaf atom index')
    if has_start:
        _require(_integer(_field(obj, meta, 'atom_start'), 'atom_start') == start, 'leaf start differs from atom_index')
    if has_end:
        _require(_integer(_field(obj, meta, 'atom_end'), 'atom_end') == start + 1, 'leaf must cover exactly one atom')
    return [start, start + 1]


def _raw_row(obj, meta, kind):
    fields = ('doc_id', 'text', 'source', 'text_mode', 'source_coordinate_system', 'source_char_span')
    raw = {name: _field(obj, meta, name) for name in fields}
    if kind == 'chunk':
        for name in ('chunk_id', 'chunk_index', 'atom_start', 'atom_end', 'num_atoms'):
            raw[name] = _field(obj, meta, name)
    else:
        start, end = _leaf_span(obj, meta)
        raw.update(atom_index=start, atom_start=start, atom_end=end)
        for name in ('leaf_id', 'leaf_index', 'owner_chunk_id'):
            raw[name] = _field(obj, meta, name)
        raw['chunk_index'] = _field(obj, meta, 'chunk_index', None)
    if 'source_document_sha256' in obj or 'source_document_sha256' in meta:
        raw['source_document_sha256'] = _field(obj, meta, 'source_document_sha256')
    return raw


def _capture(raw, kind, indexes):
    doc_id = _nonblank(raw['doc_id'], 'document ID')
    index = _index(doc_id, indexes)
    view = index.view
    if kind == 'chunk':
        captured = capture_refiner_source_chunk(raw, index)
        start, end = captured.atom_span
        object_id, source_hash = captured.unit.id, captured.source_document_sha256
        extra = {'chunk_index': raw['chunk_index'], 'num_atoms': end - start}
    else:
        _require(raw['source'] == 'refiner_source_view' and raw['text_mode'] == 'source_document',
                 'leaf must explicitly use source-document mode')
        _require(raw['source_coordinate_system'] == view.coordinate_system, 'leaf coordinate contract differs')
        start = _integer(raw['atom_index'], 'leaf atom index')
        end = start + 1
        _require(_pair(raw['source_char_span'], 'source character span') == list(view.char_span(start, end)),
                 'leaf source character geometry differs')
        _require(isinstance(raw['text'], str) and raw['text'] == view.render(start, end), 'leaf source text differs')
        object_id = _nonblank(raw['leaf_id'], 'leaf ID')
        owner = _nonblank(raw['owner_chunk_id'], 'leaf owner chunk ID')
        leaf_index = _integer(raw['leaf_index'], 'leaf_index')
        chunk_index = raw.get('chunk_index')
        if chunk_index is not None:
            _integer(chunk_index, 'leaf owner chunk_index')
        source_hash = _digest(view.source_text)
        if 'source_document_sha256' in raw:
            _require(raw['source_document_sha256'] == source_hash, 'leaf document hash differs')
        extra = {'leaf_index': leaf_index, 'owner_chunk_id': owner, 'chunk_index': chunk_index}
    return {'schema': RECORD_SCHEMA, 'kind': kind, 'doc_id': doc_id, 'object_id': object_id,
        'atom_span': [start, end], 'source_char_span': list(view.char_span(start, end)),
        'source_coordinate_system': view.coordinate_system, 'source_document_sha256': source_hash,
        'text': raw['text'], 'text_sha256': _digest(raw['text']), **extra}


def _snapshot_raw(snapshot):
    start, end = _pair(snapshot['atom_span'], 'saved atom span')
    raw = {name: snapshot[name] for name in ('doc_id', 'text', 'source_coordinate_system',
                                             'source_char_span', 'source_document_sha256')}
    raw.update(source='refiner_source_view', text_mode='source_document', atom_start=start, atom_end=end)
    if snapshot['kind'] == 'chunk':
        raw.update(chunk_id=snapshot['object_id'], chunk_index=snapshot['chunk_index'], num_atoms=snapshot['num_atoms'])
    else:
        _require(end == start + 1, 'saved leaf must cover exactly one atom')
        raw.update(leaf_id=snapshot['object_id'], leaf_index=snapshot['leaf_index'], atom_index=start,
                   owner_chunk_id=snapshot['owner_chunk_id'], chunk_index=snapshot['chunk_index'])
    return raw


def _check_current(obj, meta, snapshot):
    kind = snapshot['kind']
    expected = {'doc_id': snapshot['doc_id'], 'source': 'refiner_source_view', 'text_mode': 'source_document',
        'source_coordinate_system': snapshot['source_coordinate_system'],
        'source_document_sha256': snapshot['source_document_sha256'],
        'source_char_span': snapshot['source_char_span'],
        'atom_start': snapshot['atom_span'][0], 'atom_end': snapshot['atom_span'][1]}
    if kind == 'chunk':
        expected.update(chunk_id=snapshot['object_id'], chunk_index=snapshot['chunk_index'], num_atoms=snapshot['num_atoms'])
        required = ('doc_id', 'chunk_id', 'chunk_index', 'atom_start', 'atom_end', 'num_atoms')
    else:
        expected.update(leaf_id=snapshot['object_id'], leaf_index=snapshot['leaf_index'],
                        owner_chunk_id=snapshot['owner_chunk_id'], chunk_index=snapshot['chunk_index'],
                        atom_index=snapshot['atom_span'][0])
        required = ('doc_id', 'leaf_id', 'leaf_index', 'owner_chunk_id')
        _require(_leaf_span(obj, meta) == snapshot['atom_span'], 'current leaf atom span differs from snapshot')
    for name in required:
        _field(obj, meta, name)
    for name, value in expected.items():
        if name in obj or name in meta:
            current = _field(obj, meta, name)
            if type(value) is int:
                _integer(current, name)
            if name == 'source_char_span':
                current = _pair(current, name)
            _require(current == value, f'current source record differs from saved {name}')
    current_text = _field(obj, meta, 'text')
    _require(isinstance(current_text, str) and current_text in (
        snapshot['text'], normalize_text_basic(snapshot['text'], keep_newlines=True)),
        'current text must be exact source or the exact retrieval normalization of that source')


def capture_or_restore_source_meta(obj: Mapping, *, kind: str, source_indexes: Mapping | None) -> dict | None:
    """Capture raw source once, or revalidate its saved snapshot after enrichment.

    The result is a fresh JSON-safe value for meta['refiner_source']. A saved
    dictionary never certifies itself: its hashes, geometry, text and current
    row identities are checked against the current explicit source index.
    """
    _require(isinstance(obj, Mapping) and kind in ('chunk', 'leaf'), 'invalid source record or kind')
    supplied_meta = obj.get('meta', {})
    meta = supplied_meta if isinstance(supplied_meta, Mapping) else {}
    source_mode = ('refiner_source' in meta or any(
        scope.get('source') == 'refiner_source_view' or any(name in scope for name in SOURCE_FIELDS)
        for scope in (obj, meta)))
    if not source_mode:
        return None
    _require(isinstance(supplied_meta, Mapping), 'source metadata must be a mapping')
    _require(isinstance(source_indexes, Mapping), 'source-mode records require explicit source_indexes')
    if 'refiner_source' in meta:
        saved = meta['refiner_source']
        common = {'schema', 'kind', 'doc_id', 'object_id', 'atom_span', 'source_char_span',
                  'source_coordinate_system', 'source_document_sha256', 'text', 'text_sha256', 'chunk_index'}
        fields = common | ({'num_atoms'} if kind == 'chunk' else {'leaf_index', 'owner_chunk_id'})
        _require(isinstance(saved, Mapping) and set(saved) == fields
                 and saved['schema'] == RECORD_SCHEMA and saved['kind'] == kind,
                 'invalid saved source snapshot schema or kind')
        restored = _capture(_snapshot_raw(saved), kind, source_indexes)
        _require(dict(saved) == restored, 'saved source snapshot differs from current source index')
    else:
        restored = _capture(_raw_row(obj, meta, kind), kind, source_indexes)
    _check_current(obj, meta, restored)
    return deepcopy(restored)


def source_snapshot_from_chunk_record(record: ChunkRecord, source_indexes: Mapping) -> RefinerSourceSnapshot:
    """Restore exact JEV source text from a possibly enriched retrieval record."""
    _require(isinstance(record, ChunkRecord), 'expected ChunkRecord')
    _require(isinstance(record.meta, Mapping) and 'refiner_source' in record.meta,
             'chunk record has no saved source snapshot')
    snapshot = capture_or_restore_source_meta(record.to_dict(), kind='chunk', source_indexes=source_indexes)
    return capture_refiner_source_chunk(_snapshot_raw(snapshot), _index(record.doc_id, source_indexes))
