"""Selected retrieval-to-source restoration using invented text only."""
from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.dataio.source_records import (
    capture_or_restore_source_meta, resolve_refiner_source_selection,
    source_snapshot_from_chunk_record,
)
from SLAC.retrieval.decision.conditional import decode_cached, freeze_response
from SLAC.retrieval.decision.refiner_bridge import build_refiner_source_request
from SLAC.retrieval.preprocess.anchor_fields import enrich_chunk_record
from SLAC.retrieval.schemas.records import ChunkRecord, PackedEvidenceItem, RetrievalCandidate


def fixture(last='B'):
    source = ' \tＡ\r\n' + last + '  '
    view = DocumentSourceView('doc', 'synthetic-python-chars', source,
        ('Ａ', last), ((0, 5), (5, 8)), (((2, 3),), ((5, 6),)))
    index = build_native_coverage_index(view, [
        {'native_unit_id': 'a', 'source_span': (2, 3), 'source_text': 'Ａ'},
        {'native_unit_id': 'b', 'source_span': (5, 6), 'source_text': last},
    ])
    indexes, lookup = {'doc': index}, {}
    for start, uid in enumerate(('z-first', 'a-second')):
        raw = {'doc_id': 'doc', 'chunk_id': uid, 'chunk_index': 99 - start,
            'atom_start': start, 'atom_end': start + 1, 'num_atoms': 1,
            'text': view.render(start, start + 1), 'source': 'refiner_source_view',
            'text_mode': 'source_document', 'source_coordinate_system': view.coordinate_system,
            'source_char_span': list(view.char_span(start, start + 1))}
        saved = capture_or_restore_source_meta(raw, kind='chunk', source_indexes=indexes)
        record = ChunkRecord('doc', uid, raw['chunk_index'], start, start + 1,
            raw['text'], 1, [], 0, meta={'refiner_source': saved})
        enrich_chunk_record(record)
        lookup[uid] = record
    first, second = lookup.values()
    pack = [PackedEvidenceItem(900, first.chunk_id, 'doc', 'direct', 'leaf_direct', [], 0, first.text)]
    candidate = RetrievalCandidate(second.chunk_id, 'doc', second.text, [], 0)
    return pack, candidate, lookup, indexes


def request(current, candidate, **changes):
    options = dict(arm='plain_conditional', endpoint_id='offline-only', model_id='synthetic',
        expected_response_model='synthetic-response', token_counter=len, max_tokens=1000)
    options.update(changes)
    return build_refiner_source_request('Which facts are present?', current, candidate, **options)


class PointLookup(Mapping):
    def __init__(self, values):
        self.values, self.accesses = values, []

    def __getitem__(self, key):
        self.accesses.append(key)
        return self.values[key]

    def __iter__(self):
        raise AssertionError('lookup iteration is forbidden')

    def __len__(self):
        raise AssertionError('whole-lookup size inspection is forbidden')


def test_reconstruct_point_lookups_preserve_raw_snapshots_without_aliasing():
    pack, candidate, lookup, indexes = fixture()
    lookup['unselected-invalid'] = object()
    selected, registry = PointLookup(lookup), PointLookup(indexes)
    current, restored, changed = resolve_refiner_source_selection(
        pack, candidate, selected, registry, text_policy='reconstruct')
    assert selected.accesses == ['z-first', 'a-second']
    assert set(registry.accesses) == {'doc'}
    assert type(current) is tuple and changed == ('a-second', 'z-first')
    assert [current[0].unit.text, restored.unit.text] == [' \tＡ\r\n', 'B  ']
    assert (current[0].unit.order, restored.unit.order) == (0, 1)
    lookup['z-first'].meta['refiner_source']['text'] = 'mutated'
    pack[0].text = 'mutated'
    assert current[0].unit.text == ' \tＡ\r\n'
    with pytest.raises(FrozenInstanceError):
        restored.unit.text = 'mutated'


@pytest.mark.parametrize('policy', ['exact', 'reconstruct'])
def test_already_restored_raw_pack_is_valid_with_normalized_lookup(policy):
    pack, candidate, lookup, indexes = fixture()
    pack[0].text = lookup[pack[0].chunk_id].meta['refiner_source']['text']
    candidate.text = lookup[candidate.chunk_id].meta['refiner_source']['text']
    current, restored, changed = resolve_refiner_source_selection(
        pack, candidate, lookup, indexes, text_policy=policy)
    assert changed == ()
    assert current[0].unit.text == pack[0].text != lookup[pack[0].chunk_id].text
    assert restored.unit.text == candidate.text != lookup[candidate.chunk_id].text


def test_policy_is_required_and_exact_rejects_normalized_display():
    inputs = fixture()
    with pytest.raises(TypeError, match='text_policy'):
        resolve_refiner_source_selection(*inputs)
    with pytest.raises(ValueError, match='text_policy'):
        resolve_refiner_source_selection(*inputs, text_policy='implicit')
    with pytest.raises(ValueError, match='exact policy'):
        resolve_refiner_source_selection(*inputs, text_policy='exact')


@pytest.mark.parametrize('policy', ['exact', 'reconstruct'])
def test_third_display_form_cannot_be_silently_reconstructed(policy):
    pack, candidate, lookup, indexes = fixture()
    pack[0].text = lookup[pack[0].chunk_id].meta['refiner_source']['text']
    candidate.text = 'An invented summary'
    with pytest.raises(ValueError, match='selected text'):
        resolve_refiner_source_selection(pack, candidate, lookup, indexes, text_policy=policy)


@pytest.mark.parametrize('failure', ['doc', 'key', 'missing', 'duplicate', 'type', 'source_hash'])
def test_selected_identity_and_saved_source_are_revalidated(failure):
    pack, candidate, lookup, indexes = fixture()
    if failure == 'doc':
        candidate.doc_id = 'wrong-doc'
    elif failure == 'key':
        lookup[candidate.chunk_id] = lookup[pack[0].chunk_id]
    elif failure == 'missing':
        del lookup[candidate.chunk_id]
    elif failure == 'duplicate':
        candidate.chunk_id = pack[0].chunk_id
    elif failure == 'type':
        lookup[candidate.chunk_id] = object()
    else:
        lookup[candidate.chunk_id].meta['refiner_source']['source_document_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        resolve_refiner_source_selection(pack, candidate, lookup, indexes, text_policy='reconstruct')


def test_existing_request_matches_manual_source_and_ignores_retrieval_hints():
    pack, candidate, lookup, indexes = fixture()
    current, restored, _ = resolve_refiner_source_selection(
        pack, candidate, lookup, indexes, text_policy='reconstruct')
    actual = request(current, restored)
    manual = request((source_snapshot_from_chunk_record(lookup['z-first'], indexes),),
                     source_snapshot_from_chunk_record(lookup['a-second'], indexes))
    assert actual.request == manual.request
    assert actual.provenance_receipt() == manual.provenance_receipt()
    pack[0].order, pack[0].token_est, pack[0].role = -999, -999, 'changed-role'
    candidate.retrieve_score_raw, candidate.token_est = {'ignored': 1e99}, -999
    lookup['z-first'].token_est = -999
    again = resolve_refiner_source_selection(pack, candidate, lookup, indexes, text_policy='reconstruct')
    assert request(*again[:2]).request == actual.request


def test_whole_restored_render_enforces_budget_and_changed_source_rejects_old_cache():
    pack, candidate, lookup, indexes = fixture()
    current, restored, _ = resolve_refiner_source_selection(
        pack, candidate, lookup, indexes, text_policy='reconstruct')
    expected = '[doc/z-first]\n \tＡ\r\n\n\n[doc/a-second]\nB  '
    seen = []
    def counter(text):
        seen.append(text)
        return len(text)
    built = request(current, restored, token_counter=counter, max_tokens=len(expected))
    assert seen == [expected] and built.request.rendered_evidence == expected
    with pytest.raises(ValueError, match='complete evidence'):
        request(current, restored, max_tokens=len(expected) - 1)
    envelope = {'request_key': built.request.cache_key, 'response': {
        'model': 'synthetic-response', 'answers': {
            name: {'type': 'choice', 'choice': 'unknown'} for name in built.request.expected_ids}}}
    cached = freeze_response(built.request, envelope)
    changed_inputs = fixture(last='C')
    changed = resolve_refiner_source_selection(*changed_inputs, text_policy='reconstruct')
    other = request(*changed[:2])
    assert other.request.cache_key != built.request.cache_key
    assert decode_cached(built.request, cached).envelope_valid
    assert not decode_cached(other.request, cached).envelope_valid
