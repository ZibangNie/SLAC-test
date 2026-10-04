"""Pure synthetic source-to-decision bridge checks; no model, data or network."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.decision.conditional import (
    decode_cached, execute_offline, freeze_response, render_evidence,
)
from SLAC.retrieval.decision.refiner_bridge import (
    RefinerSourceRequest, capture_refiner_source_chunk, build_refiner_source_request,
)


def index(parts=('Alpha. ', 'Beta. ', 'Tail.'), *, coordinate='synthetic_python_chars', merged=False):
    source, cursor, spans, origins = ''.join(parts), 0, [], []
    models = []
    for part in parts:
        model = part.rstrip()
        models.append(model)
        spans.append((cursor, cursor + len(part)))
        origins.append(tuple((cursor + i, cursor + i + 1) for i in range(len(model))))
        cursor += len(part)
    if merged:
        # Same raw source, different valid base atom partition. Not interchangeable
        # for atom_start source ordering, even if a selected suffix stays identical.
        models = [parts[0] + parts[1].rstrip(), models[2]]
        cut = len(parts[0]) + len(parts[1])
        spans = [(0, cut), spans[2]]
        origins = [tuple((i, i + 1) for i in range(len(models[0]))), origins[2]]
    view = DocumentSourceView('doc', coordinate, source, tuple(models), tuple(spans), tuple(origins))
    native_end = len(parts[0]) + len(parts[1].rstrip())
    tail_start = len(parts[0]) + len(parts[1])
    return build_native_coverage_index(view, [
        {'native_unit_id': 'native-ab', 'source_span': (0, native_end), 'source_text': source[:native_end]},
        {'native_unit_id': 'native-tail', 'source_span': (tail_start, len(source)), 'source_text': source[tail_start:]},
    ])


def row(idx, start=0, end=1, *, uid='chunk-a', chunk_index=99):
    return {'doc_id': idx.view.doc_id, 'chunk_id': uid, 'chunk_index': chunk_index,
        'text_mode': 'source_document', 'source': 'refiner_source_view',
        'source_coordinate_system': idx.view.coordinate_system,
        'atom_start': start, 'atom_end': end, 'num_atoms': end - start,
        'source_char_span': list(idx.view.char_span(start, end)), 'text': idx.view.render(start, end)}


def snapshot(idx=None, start=0, end=1, uid='chunk-a'):
    idx = index() if idx is None else idx
    return capture_refiner_source_chunk(row(idx, start, end, uid=uid), idx)


def options(**changes):
    result = {'arm': 'plain_conditional', 'endpoint_id': 'offline:decisions',
        'model_id': 'synthetic-choice', 'expected_response_model': 'synthetic-response',
        'token_counter': len, 'max_tokens': 10000}
    result.update(changes)
    return result


def request(pack, candidate, **changes):
    return build_refiner_source_request('What information is supplied?', pack, candidate, **options(**changes))


def test_capture_exact_text_order_and_coverage_without_mutable_input_aliases():
    idx = index()
    original = row(idx, 2, 3, uid='suffix', chunk_index=999)
    snap = capture_refiner_source_chunk(original, idx)
    assert snap.unit.order == 2 and snap.unit.id == 'suffix'
    assert snap.unit.text == 'Tail.' and snap.atom_span == (2, 3)
    assert snap.coverage.exact_native_unit_id == 'native-tail'
    assert snap.source_document_sha256 == hashlib.sha256(idx.view.source_text.encode()).hexdigest()
    original['text'] = 'mutated display'
    original['source_char_span'][0] = 0
    assert snap.unit.text == 'Tail.' and snap.coverage.source_char_span == (13, 18)
    with pytest.raises(FrozenInstanceError):
        snap.unit = replace(snap.unit, text='changed')


@pytest.mark.parametrize('field,value', [
    ('doc_id', 'wrong'), ('source', 'refiner_epoch8'), ('text_mode', 'model_atoms'),
    ('source_coordinate_system', 'different'), ('atom_start', True), ('atom_end', 9),
    ('num_atoms', 2), ('num_atoms', True), ('source_char_span', [0, 6]),
    ('source_char_span', [False, 7]), ('text', 'Alpha.'), ('chunk_index', -1),
    ('chunk_index', True), ('chunk_id', '  '), ('source_document_sha256', '0' * 64),
])
def test_capture_rejects_stale_or_non_source_row(field, value):
    idx = index()
    invalid = row(idx)
    invalid[field] = value
    with pytest.raises(ValueError):
        capture_refiner_source_chunk(invalid, idx)


@pytest.mark.parametrize('change', ['text', 'order', 'hash', 'geometry', 'coverage'])
def test_direct_snapshot_construction_cannot_bypass_validation(change):
    snap = snapshot()
    replacements = {
        'text': {'unit': replace(snap.unit, text='other')},
        'order': {'unit': replace(snap.unit, order=99)},
        'hash': {'source_document_sha256': '0' * 64},
        'geometry': {'atom_span': (1, 2)},
        'coverage': {'coverage': snap._coverage_index.cover_atoms(1, 2)},
    }
    with pytest.raises(ValueError):
        replace(snap, **replacements[change])


def test_build_revalidates_even_a_snapshot_forged_after_construction():
    idx = index()
    current, candidate = snapshot(idx), snapshot(idx, 2, 3, 'suffix')
    object.__setattr__(candidate, 'source_document_sha256', 'f' * 64)
    with pytest.raises(ValueError, match='document hash'):
        request([current], candidate)


def test_conditional_request_is_exact_and_receipt_is_fresh_and_local():
    idx = index()
    current, candidate = snapshot(idx, 0, 2), snapshot(idx, 2, 3, 'suffix')
    built = request([current], candidate)
    payload = built.request.payload()
    assert payload['state']['current_pack'][0]['text'] == idx.view.render(0, 2)
    assert set(payload['state']) == {'query', 'current_pack', 'candidate'}
    assert set(payload['state']['candidate']) == {'id', 'doc_id', 'order', 'text'}
    assert 'source_document_sha256' not in json.dumps(payload)
    receipt = built.provenance_receipt()
    assert receipt['request_key'] == built.request.cache_key
    assert [s['role'] for s in receipt['visible_snapshots']] == ['current_pack', 'candidate']
    covered = receipt['visible_snapshots'][0]['native_coverage']
    assert covered['full_native_unit_ids'] == ['native-ab']
    assert covered['gap_chars'] == 1 and covered['exact_native_unit_id'] is None
    receipt['visible_snapshots'][0]['native_coverage']['full_native_unit_ids'].clear()
    assert built.provenance_receipt()['visible_snapshots'][0]['native_coverage']['full_native_unit_ids'] == ['native-ab']
    with pytest.raises(ValueError, match='visible units'):
        RefinerSourceRequest(built.request, (candidate, current))


def test_standalone_receipt_and_key_project_to_candidate_only():
    idx = index()
    first, alternative = snapshot(idx), snapshot(idx, 1, 2, 'middle')
    candidate = snapshot(idx, 2, 3, 'suffix')
    left = request([first], candidate, arm='standalone')
    right = request([alternative], candidate, arm='standalone')
    assert left.request.cache_key == right.request.cache_key
    assert left.provenance_receipt() == right.provenance_receipt()
    assert len(left.provenance_receipt()['visible_snapshots']) == 1
    assert left.provenance_receipt()['visible_snapshots'][0]['role'] == 'candidate'


def test_same_visible_request_reuses_key_but_rebinds_unselected_source_provenance():
    first = index()
    changed = index(('Alpha. ', 'Beta. ', 'Completely different unselected tail.'))
    left = request([], snapshot(first), arm='standalone')
    right = request([], snapshot(changed), arm='standalone')
    assert left.request.payload_bytes == right.request.payload_bytes
    assert left.request.cache_key == right.request.cache_key
    a = left.provenance_receipt()['visible_snapshots'][0]
    b = right.provenance_receipt()['visible_snapshots'][0]
    assert a['text_sha256'] == b['text_sha256']
    assert a['source_document_sha256'] != b['source_document_sha256']


@pytest.mark.parametrize('different', ['source', 'coordinate', 'atoms'])
def test_same_document_mixed_source_or_atom_contracts_are_rejected(different):
    original = index()
    alternate = (index(('Alpha. ', 'Beta. ', 'Changed.')) if different == 'source'
                 else index(coordinate='different-coordinates') if different == 'coordinate'
                 else index(merged=True))
    candidate_start = 1 if different == 'atoms' else 2
    candidate = snapshot(alternate, candidate_start, candidate_start + 1, 'suffix')
    with pytest.raises(ValueError, match='same-document'):
        request([snapshot(original)], candidate, arm='standalone')


def test_deliberate_overlap_on_one_atom_basis_is_not_arbitrarily_banned():
    idx = index()
    built = request([snapshot(idx, 0, 2)], snapshot(idx, 1, 3, 'overlapping'))
    rows = built.provenance_receipt()['visible_snapshots']
    assert rows[0]['source_char_span'] == [0, 13]
    assert rows[1]['source_char_span'] == [7, 18]
    assert built.request.expected_ids == ('conditional_added_information', 'conflict')


def test_core_duplicate_source_position_rule_is_preserved():
    idx = index()
    with pytest.raises(ValueError, match='source orders'):
        request([snapshot(idx, 0, 2)], snapshot(idx, 0, 1, 'same-start'))


def test_complete_existing_renderer_is_budgeted_at_exact_cap():
    idx = index()
    current, candidate = snapshot(idx, 0, 2), snapshot(idx, 2, 3, 'suffix')
    expected = render_evidence((current.unit, candidate.unit))
    seen = []
    def counter(text):
        seen.append(text)
        return len(text)
    built = request([current], candidate, token_counter=counter, max_tokens=len(expected))
    assert seen == [expected]
    assert built.request.rendered_evidence == expected
    assert built.request.evidence_tokens == len(expected)
    with pytest.raises(ValueError, match='complete evidence'):
        request([current], candidate, token_counter=len, max_tokens=len(expected) - 1)


def test_changed_visible_text_changes_core_key_and_rejects_old_fake_cache():
    old = request([], snapshot(index()), arm='standalone')
    changed = request([], snapshot(index(('Omega. ', 'Beta. ', 'Tail.'))), arm='standalone')
    assert old.request.cache_key != changed.request.cache_key
    envelope = {'request_key': old.request.cache_key, 'response': {
        'model': 'synthetic-response', 'answers': {
            question: {'type': 'choice', 'choice': 'yes'} for question in old.request.expected_ids}}}
    result = execute_offline(old.request, transport=lambda wire: envelope)
    assert result.envelope_valid  # Fixed mechanical fake; not a model prediction.
    cached = freeze_response(old.request, envelope)
    assert decode_cached(old.request, cached).envelope_valid
    assert not decode_cached(changed.request, cached).envelope_valid


def test_relation_arm_requires_an_explicit_future_contract():
    with pytest.raises(ValueError, match='no automatic relations'):
        request([], snapshot(), arm='relation_conditioned')
