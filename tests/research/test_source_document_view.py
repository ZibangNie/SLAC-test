"""Bounded synthetic composition checks; no natural inputs or model access."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

import pytest

from SLAC.refiner.pipeline.assemble.build_refiner_input import RefinerInputBuildConfig
from SLAC.refiner.pipeline.assemble.source_atomizer import trace_atomize_unit_text
from SLAC.refiner.pipeline.assemble.source_document_view import (
    DocumentSourceView, SourceDocumentViewError, compose_document_source_view,
)


def make_trace(text, *, serialized=False):
    trace = trace_atomize_unit_text(text, RefinerInputBuildConfig(atom_min_chars=0, atom_min_tokens=0))
    return asdict(trace) if serialized else trace


def example():
    first, second = "（A）. same.", "same.\r\nb."
    source = " \t" + first + "\r\n\t" + second + "  "
    start2 = 2 + len(first) + 3
    traces = [
        {"native_unit_id": "u0", "source_span": (2, 2+len(first)), "trace": make_trace(first)},
        {"native_unit_id": "u1", "source_span": (start2, start2+len(second)), "trace": make_trace(second, serialized=True)},
    ]
    models = traces[0]["trace"].model_atoms + traces[1]["trace"]["model_atoms"]
    return source, traces, models


def compose(source, traces, models):
    return compose_document_source_view(doc_id="synthetic", coordinate_system="raw-python-chars",
        source_text=source, unit_traces=traces, model_atoms=models)


def test_composes_local_origins_and_assigns_document_gaps_without_changing_models():
    source, traces, models = example()
    view = compose(source, traces, models)
    assert view.model_atoms == tuple(models) == ("(A).", "same.", "same.", "b.")
    assert "".join(view.render_atoms) == source == view.render(0, 4)
    assert view.render_atoms[0].startswith(" \t") and view.render_atoms[-1].endswith("  ")
    assert view.render_atoms[1].endswith("\r\n\t")
    assert view.model_char_origins[0][0] == (2, 3)
    assert view.model_char_origins[2][0] == (traces[1]["source_span"][0], traces[1]["source_span"][0]+1)
    assert view.char_span(1, 3) == (view.atom_char_spans[1][0], view.atom_char_spans[2][1])
    assert view.render(1, 3) == "".join(view.render_atoms[1:3])
    assert view.validate_against("synthetic", models) is None


def test_view_detaches_all_mutable_inputs_and_is_frozen():
    source, traces, models = example()
    view = compose(source, traces, models)
    saved = view.render_atoms, view.model_atoms, view.model_char_origins
    traces[1]["trace"]["atoms"][0]["model_char_origins"][0] = None
    models[0] = "changed"
    assert (view.render_atoms, view.model_atoms, view.model_char_origins) == saved
    assert isinstance(view.atom_char_spans, tuple) and all(isinstance(row, tuple) for row in view.model_char_origins)
    with pytest.raises(FrozenInstanceError):
        view.doc_id = "changed"


def test_whitespace_only_units_are_gaps_and_synthetic_join_origins_survive():
    source = " \ta.\r\nb. \n"
    traced = trace_atomize_unit_text("a.\r\nb.")
    entries = [
        {"native_unit_id": "empty", "source_span": (0, 2), "trace": make_trace(" \t")},
        {"native_unit_id": "body", "source_span": (2, 8), "trace": traced},
        {"native_unit_id": "tail", "source_span": (8, len(source)), "trace": make_trace(source[8:])},
    ]
    view = compose(source, entries, ["a. b."])
    assert view.render_atoms == (source,)
    assert view.model_char_origins == (((2, 3), (3, 4), None, (6, 7), (7, 8)),)


@pytest.mark.parametrize("mutation,code", [
    ("nonwhite_gap", "nonwhitespace_gap"), ("reverse", "nonwhitespace_gap"),
    ("overlap", "source_order_or_overlap"), ("source_mismatch", "trace_source_mismatch"),
    ("atom_source_mismatch", "trace_source_mismatch"), ("model_mismatch", "trace_model_mismatch"),
    ("cached_atoms", "model_atoms_mismatch"), ("duplicate_id", "invalid_unit_id"),
    ("bad_origin", "unsupported_origin"), ("lost_nonwhite", "unsupported_origin"),
])
def test_composition_rejects_invalid_saved_metadata(mutation, code):
    source, traces, models = example()
    traces[0]["trace"] = asdict(traces[0]["trace"])
    if mutation == "nonwhite_gap": source = "X" + source[1:]
    elif mutation == "reverse": traces.reverse()
    elif mutation == "overlap": traces[1]["source_span"] = traces[0]["source_span"]
    elif mutation == "source_mismatch": traces[0]["trace"]["source_text"] += "x"
    elif mutation == "atom_source_mismatch": traces[0]["trace"]["atoms"][0]["source_text"] += "x"
    elif mutation == "model_mismatch": traces[0]["trace"]["atoms"][0]["model_text"] = "wrong"
    elif mutation == "cached_atoms": models.reverse()
    elif mutation == "duplicate_id": traces[1]["native_unit_id"] = "u0"
    elif mutation == "bad_origin": traces[0]["trace"]["atoms"][0]["model_char_origins"][0] = (-1, 1)
    else: traces[0]["trace"]["atoms"][0]["model_char_origins"][0] = None
    with pytest.raises(SourceDocumentViewError) as caught:
        compose(source, traces, models)
    assert caught.value.code == code


def test_wholly_trivia_document_is_not_a_zero_atom_render_view():
    with pytest.raises(SourceDocumentViewError, match="empty_document_atoms"):
        compose(" \n", [{"native_unit_id": "blank", "source_span": (0, 2), "trace": make_trace(" \n")}], [])


def test_public_constructor_validates_and_copies_nested_lists():
    spans, origins = [[0, 1]], [[[0, 1]]]
    view = DocumentSourceView("d", "raw", "a", ["a"], spans, origins)
    spans[0][0] = 99
    origins[0][0][0] = 99
    assert view.atom_char_spans == ((0, 1),) and view.model_char_origins == (((0, 1),),)
    with pytest.raises(SourceDocumentViewError):
        DocumentSourceView("d", "raw", "ab", ["a"], [(0, 1)], [[(0, 1)]])
    with pytest.raises(SourceDocumentViewError, match="unrepresented_nonwhitespace"):
        DocumentSourceView("d", "raw", "ab", ["a"], [(0, 2)], [[(0, 1)]])
    with pytest.raises(SourceDocumentViewError, match="unsupported_origin"):
        DocumentSourceView("d", "raw", "a", ["z"], [(0, 1)], [[(0, 1)]])


@pytest.mark.parametrize("span", [(-1, 1), (0, 0), (1, 0), (0, 99), (False, 1)])
def test_render_rejects_invalid_atom_span(span):
    view = compose(*example())
    with pytest.raises(SourceDocumentViewError, match="invalid_atom_span"):
        view.render(*span)


def test_validate_against_rejects_document_and_cached_atom_mismatch():
    view = compose(*example())
    with pytest.raises(SourceDocumentViewError, match="doc_id_mismatch"):
        view.validate_against("another", view.model_atoms)
    with pytest.raises(SourceDocumentViewError, match="model_atoms_mismatch"):
        view.validate_against(view.doc_id, list(reversed(view.model_atoms)))
