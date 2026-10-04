"""Source snapshot persistence with invented text and temporary JSON only."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.dataio.source_records import (
    INDEX_SCHEMA, capture_or_restore_source_meta, load_source_indexes,
    source_snapshot_from_chunk_record,
)
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record, enrich_leaf_record
from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord
from SLAC.retrieval.utils.text_utils import normalize_text_basic


def fixture(last='B'):
    source = ' \tＡ\r\n' + last + '  '
    view = DocumentSourceView('doc', 'synthetic-python-chars', source,
        ('Ａ', last), ((0, 5), (5, 8)), (((2, 3),), ((5, 6),)))
    native = [{'native_unit_id': 'a', 'source_span': [2, 3], 'source_text': 'Ａ'},
              {'native_unit_id': 'b', 'source_span': [5, 6], 'source_text': last}]
    index = build_native_coverage_index(view, native)
    registry = {'schema': INDEX_SCHEMA, 'documents': [{'view': asdict(view), 'native_units': native}]}
    return index, registry


def raw(kind='chunk'):
    index, _ = fixture()
    result = {'doc_id': 'doc', 'source': 'refiner_source_view', 'text_mode': 'source_document',
              'source_coordinate_system': index.view.coordinate_system,
              'source_char_span': [0, 8] if kind == 'chunk' else [0, 5],
              'text': index.view.source_text if kind == 'chunk' else index.view.render(0, 1)}
    if kind == 'chunk':
        result.update(chunk_id='chunk', chunk_index=3, atom_start=0, atom_end=2, num_atoms=2)
    else:
        result.update(leaf_id='leaf-a', owner_chunk_id='chunk', chunk_index=3, leaf_index=0, atom_index=0)
    return result


def captured(kind='chunk'):
    index, _ = fixture()
    obj = raw(kind)
    snapshot = capture_or_restore_source_meta(obj, kind=kind, source_indexes={'doc': index})
    return obj, snapshot, {'doc': index}


def record(kind='chunk'):
    obj, snapshot, indexes = captured(kind)
    meta = {name: deepcopy(obj[name]) for name in ('source', 'text_mode', 'source_coordinate_system', 'source_char_span')}
    meta['refiner_source'] = snapshot
    if kind == 'chunk':
        rec = ChunkRecord('doc', 'chunk', 3, 0, 2, obj['text'], 2, [], 0, meta=meta)
    else:
        meta.update(atom_index=0, chunk_index=3)
        rec = LeafRecord('doc', 'leaf-a', 'chunk', 0, 0, 1, obj['text'], [], 0, meta=meta)
    return rec, indexes


@pytest.mark.parametrize('kind', ['chunk', 'leaf'])
def test_raw_capture_is_json_safe_exact_and_does_not_alias_input(kind):
    obj, snapshot, indexes = captured(kind)
    assert snapshot['text'] == obj['text']
    assert snapshot['text_sha256'] == hashlib.sha256(obj['text'].encode()).hexdigest()
    assert snapshot['source_document_sha256'] == hashlib.sha256(indexes['doc'].view.source_text.encode()).hexdigest()
    assert snapshot['atom_span'] == ([0, 2] if kind == 'chunk' else [0, 1])
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert not {'view', 'native_units', 'index'} & snapshot.keys()
    obj['source_char_span'][0] = 99
    assert snapshot['source_char_span'][0] == 0


@pytest.mark.parametrize('kind', ['chunk', 'leaf'])
def test_actual_enrichment_json_reload_preserves_raw_snapshot(kind):
    rec, indexes = record(kind)
    original = deepcopy(rec.meta['refiner_source'])
    (enrich_chunk_record if kind == 'chunk' else enrich_leaf_record)(rec)
    assert rec.text == normalize_text_basic(original['text'], keep_newlines=True)
    assert rec.text != original['text']  # Fullwidth, CRLF and outside whitespace changed.
    reloaded = json.loads(json.dumps(rec.to_dict(), ensure_ascii=False))
    restored = capture_or_restore_source_meta(reloaded, kind=kind, source_indexes=indexes)
    assert restored == original
    restored['source_char_span'][0] = 99
    assert rec.meta['refiner_source']['source_char_span'][0] == 0
    if kind == 'chunk':
        snapshot = source_snapshot_from_chunk_record(rec, indexes)
        assert snapshot.unit.text == original['text']
        assert snapshot.unit.order == 0 and snapshot.atom_span == (0, 2)


@pytest.mark.parametrize('change', ['text', 'doc_id', 'chunk_id', 'chunk_index', 'atom_end', 'num_atoms', 'source_flag'])
def test_changed_current_chunk_cannot_restore_source(change):
    rec, indexes = record()
    enrich_chunk_record(rec)
    if change == 'source_flag':
        rec.meta['source'] = 'refiner_epoch8'
    else:
        setattr(rec, change, {'text': 'A B', 'doc_id': 'different', 'chunk_id': 'other',
                             'chunk_index': 4, 'atom_end': 1, 'num_atoms': True}[change])
    with pytest.raises(ValueError):
        source_snapshot_from_chunk_record(rec, indexes)


@pytest.mark.parametrize('change', ['raw_text', 'text_hash', 'doc_hash', 'coordinates', 'kind', 'extra_field'])
def test_saved_dictionary_is_revalidated_not_trusted(change):
    rec, indexes = record()
    saved = rec.meta['refiner_source']
    field, value = {'raw_text': ('text', 'changed'), 'text_hash': ('text_sha256', '0' * 64),
        'doc_hash': ('source_document_sha256', '0' * 64), 'coordinates': ('source_char_span', [0, 7]),
        'kind': ('kind', 'leaf'), 'extra_field': ('full_view', {})}[change]
    saved[field] = value
    with pytest.raises(ValueError):
        source_snapshot_from_chunk_record(rec, indexes)


def test_unselected_document_change_invalidates_saved_document_version():
    index, _ = fixture()
    obj = raw()
    obj.update(atom_end=1, num_atoms=1, source_char_span=[0, 5], text=index.view.render(0, 1))
    snapshot = capture_or_restore_source_meta(obj, kind='chunk', source_indexes={'doc': index})
    lookup = {**obj, 'meta': {'refiner_source': snapshot}}
    changed, _ = fixture(last='C')
    assert changed.view.render(0, 1) == snapshot['text']
    with pytest.raises(ValueError, match='hash'):
        capture_or_restore_source_meta(lookup, kind='chunk', source_indexes={'doc': changed})


@pytest.mark.parametrize('field,value', [('atom_end', 2), ('owner_chunk_id', 'wrong-owner'),
                                       ('leaf_index', True), ('atom_index', 1)])
def test_leaf_reload_validates_geometry_before_reader_can_rebuild_end(field, value):
    rec, indexes = record('leaf')
    obj = rec.to_dict()
    if field == 'atom_index':
        obj['meta'][field] = value
    else:
        obj[field] = value
    with pytest.raises(ValueError):
        capture_or_restore_source_meta(obj, kind='leaf', source_indexes=indexes)


@pytest.mark.parametrize('field,value', [('source', 'refiner_epoch8'), ('source_char_span', [0, 7]),
                                       ('doc_id', 'wrong'), ('chunk_index', True)])
def test_conflicting_top_and_meta_fields_fail_closed(field, value):
    obj, snapshot, indexes = captured()
    obj['meta'] = {'refiner_source': snapshot, field: value}
    with pytest.raises(ValueError):
        capture_or_restore_source_meta(obj, kind='chunk', source_indexes=indexes)


@pytest.mark.parametrize('obj', [{}, {'source': 'refiner_epoch8'}, {'meta': {'source': None}},
                               {'text': 'legacy text', 'meta': {'unrelated': [1, 2]}}])
def test_legacy_has_no_new_requirements_and_is_untouched(obj):
    before = deepcopy(obj)
    assert capture_or_restore_source_meta(obj, kind='chunk', source_indexes=None) is None
    assert obj == before


@pytest.mark.parametrize('obj', [{'text_mode': None}, {'source_coordinate_system': ''},
                               {'meta': {'refiner_source': None}}, {'meta': {'source': 'refiner_source_view'}}])
def test_partial_or_blank_source_metadata_is_never_legacy(obj):
    index, _ = fixture()
    with pytest.raises(ValueError):
        capture_or_restore_source_meta(obj, kind='chunk', source_indexes={'doc': index})


@pytest.mark.parametrize('kind', ['chunk', 'leaf'])
def test_source_requires_explicit_registry(kind):
    with pytest.raises(ValueError, match='source_indexes'):
        capture_or_restore_source_meta(raw(kind), kind=kind, source_indexes=None)


def test_registry_json_roundtrip_uses_real_validating_constructors(tmp_path):
    index, registry = fixture()
    path = tmp_path / 'indexes.json'
    path.write_text(json.dumps(registry, ensure_ascii=False), encoding='utf-8')
    loaded = load_source_indexes(path)
    assert set(loaded) == {'doc'} and loaded['doc'] == index
    assert isinstance(loaded['doc'].view.atom_char_spans, tuple)
    assert isinstance(loaded['doc'].native_units, tuple)


@pytest.mark.parametrize('failure', ['schema', 'duplicate_doc', 'view_partition', 'native_text', 'extra_view_field'])
def test_bad_registry_cannot_supply_source_proof(tmp_path, failure):
    _, registry = fixture()
    if failure == 'schema':
        registry['schema'] = 'wrong'
    elif failure == 'duplicate_doc':
        registry['documents'].append(deepcopy(registry['documents'][0]))
    elif failure == 'view_partition':
        registry['documents'][0]['view']['atom_char_spans'] = [[0, 4], [5, 8]]
    elif failure == 'native_text':
        registry['documents'][0]['native_units'][0]['source_text'] = 'invented'
    else:
        registry['documents'][0]['view']['extra'] = 'unexpected'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(registry), encoding='utf-8')
    with pytest.raises(ValueError):
        load_source_indexes(path)


def test_registry_duplicate_json_field_is_rejected(tmp_path):
    path = tmp_path / 'duplicates.json'
    path.write_text('{"schema":"bad","schema":"slac-refiner-source-indexes-v1","documents":[]}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate JSON field'):
        load_source_indexes(path)
