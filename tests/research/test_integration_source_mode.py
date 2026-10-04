"""Synthetic source-mode input restoration; no tokenizer, model or network."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from SLAC.integration.evidence.normalizers import normalize_candidate_to_selected_evidence
from SLAC.integration.evidence.source_mode import SourceModeConfig, restore_source_candidates
from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.dataio.source_records import capture_or_restore_source_meta
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record
from SLAC.retrieval.schemas.records import ChunkRecord


def forbidden_counter(_):
    raise AssertionError('restoration must not count, tokenize or render')


def fixture(*, policy='reconstruct', doc_id='doc', chunk_id='chunk'):
    source = ' \tＡ\r\nB  '
    view = DocumentSourceView(doc_id, 'synthetic-python-chars', source,
        ('Ａ', 'B'), ((0, 5), (5, 8)), (((2, 3),), ((5, 6),)))
    index = build_native_coverage_index(view, [
        {'native_unit_id': 'native-a', 'source_span': (2, 3), 'source_text': 'Ａ'},
        {'native_unit_id': 'native-b', 'source_span': (5, 6), 'source_text': 'B'},
    ])
    raw = {'doc_id': doc_id, 'chunk_id': chunk_id, 'chunk_index': 7,
        'atom_start': 0, 'atom_end': 2, 'num_atoms': 2, 'text': source,
        'source': 'refiner_source_view', 'text_mode': 'source_document',
        'source_coordinate_system': view.coordinate_system, 'source_char_span': [0, 8]}
    indexes = {doc_id: index}
    saved = capture_or_restore_source_meta(raw, kind='chunk', source_indexes=indexes)
    chunk = ChunkRecord(doc_id, chunk_id, 7, 0, 2, source, 2, [], 0, meta={'refiner_source': saved})
    enrich_chunk_record(chunk)
    config = SourceModeConfig({chunk_id: chunk}, indexes, policy, forbidden_counter, 'synthetic-not-used-v1')
    record = {'chunk_id': chunk_id, 'doc_id': doc_id, 'text': chunk.text,
        'path': ['Section', 'Subsection'], 'number_signature': '42', 'rerank_rank': 2,
        'rerank_score': 0.75, 'retrieve_rank_fused': 9, 'token_est': 123,
        'role': 'direct', 'hit_type': 'leaf_direct', 'source_views': ['leaf', 'chunk'],
        'meta': {'sentinel': 'preserve', 'refiner_source': {'untrusted': True}}}
    return record, chunk, config


@pytest.mark.parametrize('field,value', [
    ('chunk_lookup', []), ('source_indexes', None), ('text_policy', 'implicit'),
    ('token_counter', 7), ('counter_version', ''), ('counter_version', None),
])
def test_config_requires_explicit_valid_contract(field, value):
    _, _, config = fixture()
    with pytest.raises((TypeError, ValueError)):
        replace(config, **{field: value})


def test_restore_preserves_complete_source_and_existing_rank_metadata():
    record, chunk, config = fixture()
    before = deepcopy(record)
    evidence = restore_source_candidates([record], source_mode=config,
        query_id='query', query_text='Synthetic question', source_name='retrieval_packed_evidence')[0]
    original = normalize_candidate_to_selected_evidence(record, query_id='query',
        query_text='Synthetic question', source_name='retrieval_packed_evidence', ordinal=0)
    assert evidence.passage_text == ' \tＡ\r\nB  '
    assert original.passage_text == '42\nSection > Subsection\nA\nB'
    left, right = asdict(evidence), asdict(original)
    for field in ('passage_text', 'meta'):
        left.pop(field)
        right.pop(field)
    assert left == right
    assert evidence.meta['sentinel'] == 'preserve'
    assert evidence.meta['_candidate_source'] == 'retrieval_packed_evidence'
    assert evidence.meta['_source_ordinal'] == 0
    assert evidence.meta['source_text_policy'] == 'reconstruct' and evidence.meta['source_text_changed'] is True
    assert evidence.meta['refiner_source'] == chunk.meta['refiner_source']
    evidence.meta['refiner_source']['source_char_span'][0] = 99
    assert chunk.meta['refiner_source']['source_char_span'] == [0, 8]
    assert record == before


@pytest.mark.parametrize('policy', ['exact', 'reconstruct'])
def test_raw_aliases_and_whitespace_in_ids_are_preserved(policy):
    record, chunk, config = fixture(policy=policy, doc_id=' doc ', chunk_id=' chunk ')
    record['text'] = record['passage_text'] = chunk.meta['refiner_source']['text']
    evidence = restore_source_candidates([record], source_mode=config)[0]
    assert evidence.chunk_id == ' chunk ' and evidence.doc_id == ' doc '
    assert evidence.passage_text == record['text']
    assert evidence.meta['source_text_changed'] is False


@pytest.mark.parametrize('failure', ['conflict', 'null_text', 'null_passage', 'summary', 'missing'])
def test_aliases_are_checked_before_normalizer_can_mask_bad_input(failure):
    record, _, config = fixture()
    if failure == 'conflict':
        record['passage_text'] = 'Path-prefixed stale text'
    elif failure == 'null_text':
        record['passage_text'], record['text'] = record['text'], None
    elif failure == 'null_passage':
        record['passage_text'] = None
    elif failure == 'summary':
        record['text'] = 'An invented summary'
    else:
        del record['text']
    with pytest.raises(ValueError):
        restore_source_candidates([record], source_mode=config)


def test_passage_only_can_reconstruct_but_exact_rejects_normalized_text():
    record, _, config = fixture()
    record['passage_text'] = record.pop('text')
    assert restore_source_candidates([record], source_mode=config)[0].passage_text == ' \tＡ\r\nB  '
    with pytest.raises(ValueError, match='exact policy'):
        restore_source_candidates([record], source_mode=replace(config, text_policy='exact'))


class PointOnly(Mapping):
    def __init__(self, values):
        self.values, self.calls = values, []

    def __getitem__(self, key):
        self.calls.append(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError('source mode must not iterate the lookup')

    def __len__(self):
        raise AssertionError('source mode must not inspect whole-lookup size')


def test_point_only_restoration_keeps_duplicate_candidates_for_existing_selector():
    record, chunk, config = fixture()
    lookup = PointOnly({'chunk': chunk, 'unselected-invalid': object()})
    config = replace(config, chunk_lookup=lookup, source_indexes=PointOnly(config.source_indexes))
    result = restore_source_candidates([record, record], source_mode=config, source_name='another-source')
    assert lookup.calls == ['chunk', 'chunk']
    assert len(result) == 2 and [item.meta['_source_ordinal'] for item in result] == [0, 1]
    assert all(item.meta['_candidate_source'] == 'another-source' for item in result)


@pytest.mark.parametrize('failure', ['identity', 'snapshot_hash'])
def test_source_resolver_still_validates_selected_lookup_and_snapshot(failure):
    record, chunk, config = fixture()
    if failure == 'identity':
        record['doc_id'] = 'wrong-doc'
    else:
        chunk.meta['refiner_source']['text_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        restore_source_candidates([record], source_mode=config)


def test_normalizer_override_is_opt_in_and_does_not_strip_or_add_path():
    record, _, _ = fixture()
    legacy = normalize_candidate_to_selected_evidence(record)
    explicit_default = normalize_candidate_to_selected_evidence(record, passage_text_override=None)
    assert legacy == explicit_default
    assert legacy.passage_text == '42\nSection > Subsection\nA\nB'
    source = normalize_candidate_to_selected_evidence(record, passage_text_override=' \tＡ\r\nB  ')
    assert source.passage_text == ' \tＡ\r\nB  '
    assert source.path_text == legacy.path_text and source.rerank_rank == legacy.rerank_rank
    with pytest.raises(TypeError, match='passage_text_override'):
        normalize_candidate_to_selected_evidence(record, passage_text_override=42)
