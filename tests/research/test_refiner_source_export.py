"""Synthetic source-view export and unchanged-default contracts; no model/data calls."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from SLAC.refiner.pipeline.assemble.build_refiner_input import RefinerInputBuildConfig
from SLAC.refiner.pipeline.assemble.source_atomizer import trace_atomize_unit_text
from SLAC.refiner.pipeline.assemble.source_document_view import compose_document_source_view
from SLAC.refiner.pipeline.assemble.export_refined_chunks import (
    RefinedChunkExportConfig, build_leaf_records, build_refined_chunks,
    export_refined_chunks_from_candidate, export_refined_chunks_from_candidates,
)
from SLAC.refiner.slac_refiner.decoding.projector import ProjectorConfig, project_boundary_vector


def fixture_records():
    record = {'doc_id': 'synthetic-doc', 'atoms': ['(one).', 'same.', 'same.'],
        'b0': [1, 0], 'domain': 'synthetic',
        'meta': {'source_path': 'synthetic.txt', 'source_type': 'text'},
        'chunk0_units': [
            {'unit_id': 10, 'type': 'paragraph', 'text': '（one）.', 'path': ['Sec A'], 'depth': 1, 'parent_id': 2},
            {'unit_id': 20, 'type': 'paragraph', 'text': 'same. same.', 'path': ['Sec B'], 'depth': 2, 'parent_id': 10}],
        'unit2atom_span': [{'unit_id': 10, 'start_atom': 0, 'end_atom': 1},
                           {'unit_id': 20, 'start_atom': 1, 'end_atom': 3}]}
    candidate = {'candidate_id': 'manual-boundaries', 'candidate_type': 'greedy',
        'teacher_ckpt': None, 'decode_cfg': {'synthetic': True}, 'teacher_stats': {},
        'input_stats': {'num_atoms': 3}, 'prediction': {'b_pred_sparse': [0]}}
    return record, candidate


def make_view():
    cfg = RefinerInputBuildConfig(atom_min_chars=0, atom_min_tokens=0)
    first, second = '（one）.', 'same. same.'
    prefix, separator, suffix = ' \t', '\r\n\r\n', '\t '
    source = prefix + first + separator + second + suffix
    first_start = len(prefix)
    second_start = first_start + len(first) + len(separator)
    units = [
        {'native_unit_id': 'unit-a', 'source_span': (first_start, first_start + len(first)),
         'trace': asdict(trace_atomize_unit_text(first, cfg))},
        {'native_unit_id': 'unit-b', 'source_span': (second_start, second_start + len(second)),
         'trace': asdict(trace_atomize_unit_text(second, cfg))},
    ]
    return compose_document_source_view(doc_id='synthetic-doc', source_text=source,
        coordinate_system='synthetic_document_python_chars', unit_traces=units,
        model_atoms=fixture_records()[0]['atoms'])


# Whole-output hashes were captured from the unchanged exporter before this edit
# (source SHA e46f43a7607eb695062ac2130dd2f5720d58dd45a4f77c36d9d65b3c341e3337).
@pytest.mark.parametrize('leaf,catalog,expected', [
    (True, True, '63c0d1507dad24df6478f92cbe20a6233edf355b3b4773c3e6b8654982274da8'),
    (True, False, 'e7a7b05600f22fb5b79ed573e3e472e14a04a1e2dd62ceb5906413ebc221ae87'),
    (False, True, '31d0500bac55bba87fc12aabb4c6857859a004dbc68b9f68028fc94551ac5a10'),
    (False, False, 'f872ceb0b821e7eb8b265f6058e8cee3ce74266cc5cc03ff49de5b286ad70f58'),
])
def test_complete_default_output_is_unchanged(leaf, catalog, expected):
    record, candidate = fixture_records()
    cfg = RefinedChunkExportConfig(export_leaf_records=leaf, export_doc_catalog=catalog)
    output = export_refined_chunks_from_candidate(record, candidate, cfg)
    assert output == export_refined_chunks_from_candidate(record, candidate, cfg, source_view=None)
    raw = json.dumps(output, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    assert hashlib.sha256(raw).hexdigest() == expected
    assert all(chunk['source'] == 'refiner_epoch8' for chunk in output['refined_chunks'])


def assert_source_record(row, view, start, end):
    char_start, char_end = view.char_span(start, end)
    assert row['text'].encode('utf-8') == view.source_text[char_start:char_end].encode('utf-8')
    assert row['source_char_span'] == [char_start, char_end]
    assert row['source_coordinate_system'] == view.coordinate_system
    assert row['text_mode'] == 'source_document'
    assert row['source'] == 'refiner_source_view'


def test_source_export_keeps_original_bytes_and_model_atoms_separate():
    record, candidate = fixture_records()
    before = deepcopy((record, candidate))
    view = make_view()
    output = export_refined_chunks_from_candidate(record, candidate, source_view=view)
    assert (record, candidate) == before
    assert record['atoms'] == ['(one).', 'same.', 'same.']
    assert ''.join(c['text'] for c in output['refined_chunks']) == view.source_text
    assert ''.join(leaf['text'] for leaf in output['leaf_records']) == view.source_text
    for chunk in output['refined_chunks']:
        assert_source_record(chunk, view, chunk['atom_start'], chunk['atom_end'])
        assert chunk['boundary_meta']['candidate_id'] == 'manual-boundaries'
        assert chunk['boundary_meta']['teacher_ckpt'] is None
    for leaf in output['leaf_records']:
        assert_source_record(leaf, view, leaf['atom_index'], leaf['atom_index'] + 1)
        assert leaf['boundary_meta']['candidate_id'] == 'manual-boundaries'
    assert output['text_mode'] == 'source_document'
    assert output['source_coordinate_system'] == view.coordinate_system
    assert output['source_char_span'] == [0, len(view.source_text)]
    assert output['source_document_sha256'] == hashlib.sha256(view.source_text.encode('utf-8')).hexdigest()
    assert all('source_document_sha256' not in row for row in output['refined_chunks'] + output['leaf_records'])
    assert output['refined_chunks'][0]['text'] != record['atoms'][0]


def test_direct_builders_receive_the_same_opt_in_view():
    record, candidate = fixture_records()
    view = make_view()
    chunks = build_refined_chunks(record, candidate, source_view=view)
    leaves = build_leaf_records(chunks, record['atoms'], source_view=view)
    exported = export_refined_chunks_from_candidate(record, candidate, source_view=view)
    assert chunks == exported['refined_chunks']
    assert leaves == exported['leaf_records']


@pytest.mark.parametrize('policy,expected_id,expected_count', [
    ('greedy_first', 'manual-boundaries', 2), ('best_teacher_stats', 'alternative', 1),
])
def test_multiple_candidate_selection_passes_source_view(policy, expected_id, expected_count):
    record, greedy = fixture_records()
    alternative = {**greedy, 'candidate_type': 'sample', 'candidate_id': 'alternative',
                   'teacher_stats': {'mean_edit_prob': .9}, 'prediction': {'b_pred_sparse': []}}
    view = make_view()
    output = export_refined_chunks_from_candidates(record, [alternative, greedy],
        RefinedChunkExportConfig(candidate_select_policy=policy), source_view=view)
    assert output['candidate_selected']['candidate_id'] == expected_id
    assert len(output['refined_chunks']) == expected_count
    assert ''.join(chunk['text'] for chunk in output['refined_chunks']) == view.source_text
    for chunk in output['refined_chunks']:
        assert_source_record(chunk, view, chunk['atom_start'], chunk['atom_end'])


@pytest.mark.parametrize('entry', ['chunks', 'leaves', 'candidate', 'candidates'])
@pytest.mark.parametrize('mismatch', ['doc', 'atoms'])
def test_all_source_entrypoints_reject_bad_binding_even_without_strict_validation(entry, mismatch):
    record, candidate = fixture_records()
    view = make_view()
    if mismatch == 'doc':
        record['doc_id'] = 'different-document'
    else:
        record['atoms'][0] += ' changed'
    cfg = RefinedChunkExportConfig(strict_validate=False, export_leaf_records=False)
    with pytest.raises(ValueError) as caught:
        if entry == 'chunks':
            build_refined_chunks(record, candidate, source_view=view)
        elif entry == 'leaves':
            chunks = build_refined_chunks(record, candidate)
            build_leaf_records(chunks, record['atoms'], source_view=view)
        elif entry == 'candidate':
            export_refined_chunks_from_candidate(record, candidate, cfg, source_view=view)
        else:
            export_refined_chunks_from_candidates(record, [candidate], cfg, source_view=view)
    assert caught.value.code == ('doc_id_mismatch' if mismatch == 'doc' else 'model_atoms_mismatch')


def test_empty_chunk_list_does_not_bypass_leaf_atom_binding():
    view = make_view()
    with pytest.raises(ValueError) as caught:
        build_leaf_records([], ['different model atoms'], source_view=view)
    assert caught.value.code == 'model_atoms_mismatch'


def test_actual_projector_budgets_the_exact_exported_raw_bytes():
    source = 'a.     b.'
    trace = trace_atomize_unit_text(source, RefinerInputBuildConfig(atom_min_chars=0, atom_min_tokens=0))
    view = compose_document_source_view(doc_id='budget-fixture', source_text=source,
        coordinate_system='synthetic_document_python_chars',
        unit_traces=[{'native_unit_id': 'unit-one', 'source_span': (0, len(source)), 'trace': asdict(trace)}],
        model_atoms=trace.model_atoms)
    cfg = ProjectorConfig(max_chunk_atoms=3, min_chunk_atoms=0,
        max_chunk_chars=8, min_chunk_chars=0, max_chunk_tokens=8, min_chunk_tokens=0)
    normalized = project_boundary_vector(list(view.model_atoms), [0], cfg, token_counter=len, strict=True)
    projected = project_boundary_vector(list(view.render_atoms), [0], cfg, token_counter=len,
        source_text=view.source_text, atom_char_spans=view.atom_char_spans, strict=True)
    assert normalized['projected_b'] == [0]
    assert projected['projected_b'] == [1]  # The five original spaces force a split.
    record = {'doc_id': view.doc_id, 'atoms': list(view.model_atoms), 'b0': [0]}
    candidate = {'candidate_id': 'synthetic-projector-only', 'candidate_type': 'synthetic',
        'prediction': {'b_pred_sparse': [i for i, bit in enumerate(projected['projected_b']) if bit]}}
    output = export_refined_chunks_from_candidate(record, candidate, source_view=view)
    expected_bytes = [unit['text'].encode('utf-8') for unit in projected['projected_units']]
    assert [chunk['text'].encode('utf-8') for chunk in output['refined_chunks']] == expected_bytes
    assert [leaf['text'].encode('utf-8') for leaf in output['leaf_records']] == expected_bytes
    assert b''.join(expected_bytes) == source.encode('utf-8')
    for chunk, unit in zip(output['refined_chunks'], projected['projected_units'], strict=True):
        assert len(chunk['text']) <= 8
        assert chunk['source_char_span'] == [unit['start_char'], unit['end_char']]
        assert chunk['boundary_meta']['teacher_ckpt'] is None
