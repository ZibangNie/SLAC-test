"""Synthetic coordinate coverage tests; no data files, models or API access."""
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from SLAC.refiner.pipeline.assemble.source_coverage import (
    NativeCoverageHit, SourceCoverageError, build_native_coverage_index,
)
from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView, SourceDocumentViewError


def view_for(source, cuts):
    # Synthetic model characters keep their identity origin; whitespace may be trivia.
    spans = list(zip(cuts, cuts[1:]))
    models, origins = [], []
    for left, right in spans:
        positions = [i for i in range(left, right) if not source[i].isspace()]
        models.append("".join(source[i] for i in positions))
        origins.append([(i, i + 1) for i in positions])
    return DocumentSourceView("synthetic", "python-characters", source, models, spans, origins)


def entry(uid, source, start, end):
    return {"native_unit_id": uid, "source_span": [start, end], "source_text": source[start:end]}


def fixture():
    source = " \tAB CD\r\nEFGH\nIJ  "
    view = view_for(source, [0, 4, 9, 11, 14, len(source)])
    units = [entry("u0", source, 2, 7), entry("u1", source, 9, 13), entry("u2", source, 14, 16)]
    return view, units


def test_split_atom_is_partial_and_never_a_whole_native_match():
    view, units = fixture()
    coverage = build_native_coverage_index(view, units).cover_atoms(0, 1)
    assert coverage.source_char_span == (0, 4)
    assert coverage.partial_native_unit_ids == ("u0",)
    assert coverage.full_native_unit_ids == ()
    assert coverage.hits[0].overlap_source_span == (2, 4)
    assert coverage.gap_chars == 2 and coverage.exact_native_unit_id is None
    assert coverage.text_sha256 == sha256(b" \tAB").hexdigest()


def test_merged_span_preserves_mixed_full_and_partial_hits_and_internal_whitespace():
    view, units = fixture()
    coverage = build_native_coverage_index(view, units).cover_atoms(0, 3)
    assert coverage.source_char_span == (0, 11)
    assert coverage.full_native_unit_ids == ("u0",)
    assert coverage.partial_native_unit_ids == ("u1",)
    assert [h.kind for h in coverage.hits] == ["full", "partial"]
    assert coverage.gap_chars == 4  # leading two + CRLF; u0's internal space is not a gap.
    assert coverage.exact_native_unit_id is None


def test_full_document_merged_coverage_counts_only_outer_and_interunit_gaps():
    view, units = fixture()
    coverage = build_native_coverage_index(view, units).cover_atoms(0, 5)
    assert coverage.full_native_unit_ids == ("u0", "u1", "u2")
    assert coverage.partial_native_unit_ids == ()
    assert coverage.gap_chars == 7 and coverage.exact_native_unit_id is None
    assert coverage.text_sha256 == sha256(view.source_text.encode("utf-8")).hexdigest()


def test_identical_text_units_stay_distinct_by_coordinates_and_id():
    source = "same.same."
    view = view_for(source, [0, 5, 10])
    index = build_native_coverage_index(view, [entry("first", source, 0, 5), entry("second", source, 5, 10)])
    first, second = index.cover_atoms(0, 1), index.cover_atoms(1, 2)
    assert first.text_sha256 == second.text_sha256
    assert first.exact_native_unit_id == "first" and second.exact_native_unit_id == "second"
    assert index.cover_atoms(0, 2).full_native_unit_ids == ("first", "second")
    assert index.cover_atoms(0, 2).exact_native_unit_id is None


def test_whole_unit_with_extra_separator_is_full_but_not_exact():
    source = "A\nB"
    index = build_native_coverage_index(view_for(source, [0, 2, 3]),
                                       [entry("a", source, 0, 1), entry("b", source, 2, 3)])
    assert index.cover_atoms(0, 1).full_native_unit_ids == ("a",)
    assert index.cover_atoms(0, 1).gap_chars == 1
    assert index.cover_atoms(0, 1).exact_native_unit_id is None
    assert index.cover_atoms(1, 2).exact_native_unit_id == "b"


def test_unicode_hash_and_offsets_are_raw_python_characters():
    source = "🙂汉"
    index = build_native_coverage_index(view_for(source, [0, 1, 2]), [entry("u", source, 0, 2)])
    part = index.cover_atoms(0, 1)
    assert part.source_char_span == (0, 1) and part.partial_native_unit_ids == ("u",)
    assert part.text_sha256 == sha256("🙂".encode("utf-8")).hexdigest()


@pytest.mark.parametrize("mutation,code", [
    ("duplicate", "invalid_unit_id"), ("blank_id", "invalid_unit_id"),
    ("bool_offset", "invalid_source_span"), ("negative", "invalid_source_span"),
    ("reversed_span", "invalid_source_span"), ("outside", "source_order_or_overlap"),
    ("overlap", "source_order_or_overlap"), ("reverse_order", "nonwhitespace_gap"),
    ("wrong_text", "source_text_mismatch"), ("missing_unit", "nonwhitespace_gap"),
    ("blank_unit", "blank_native_unit"), ("empty_unit", "blank_native_unit"),
])
def test_invalid_native_metadata_is_rejected(mutation, code):
    view, units = fixture()
    if mutation == "duplicate": units[1]["native_unit_id"] = "u0"
    elif mutation == "blank_id": units[0]["native_unit_id"] = " "
    elif mutation == "bool_offset": units[0]["source_span"][0] = True
    elif mutation == "negative": units[0]["source_span"][0] = -1
    elif mutation == "reversed_span": units[0]["source_span"] = [7, 2]
    elif mutation == "outside": units[-1]["source_span"][1] = 999
    elif mutation == "overlap": units[1] = entry("u1", view.source_text, 6, 13)
    elif mutation == "reverse_order": units.reverse()
    elif mutation == "wrong_text": units[0]["source_text"] = "wrong"
    elif mutation == "missing_unit": units.pop(1)
    elif mutation == "blank_unit": units.insert(0, entry("blank", view.source_text, 0, 2))
    else: units.insert(0, entry("empty", view.source_text, 0, 0))
    with pytest.raises(SourceCoverageError) as caught:
        build_native_coverage_index(view, units)
    assert caught.value.code == code


@pytest.mark.parametrize("span", [(-1, 1), (1, 1), (2, 1), (0, 6), (True, 2)])
def test_invalid_atom_span_uses_view_contract(span):
    view, units = fixture()
    with pytest.raises(SourceDocumentViewError, match="invalid_atom_span"):
        build_native_coverage_index(view, units).cover_atoms(*span)


def test_inputs_index_and_coverage_are_deeply_immutable():
    view, units = fixture()
    index = build_native_coverage_index(view, units)
    before = index.cover_atoms(0, 5)
    units[0]["source_span"][0] = 99
    units[0]["native_unit_id"] = "changed"
    units.clear()
    assert index.cover_atoms(0, 5) == before
    assert isinstance(index.native_units, tuple) and isinstance(before.hits, tuple)
    with pytest.raises(FrozenInstanceError): index.view = None
    with pytest.raises(FrozenInstanceError): index.native_units[0].source_text = "changed"
    with pytest.raises(FrozenInstanceError): before.gap_chars = 0
    with pytest.raises(FrozenInstanceError): before.hits[0].kind = "partial"


def test_hit_cannot_be_mislabeled_full_or_keep_mutable_spans():
    native, overlap = [0, 4], [0, 2]
    hit = NativeCoverageHit("u", native, overlap, "partial")
    native[0] = overlap[0] = 9
    assert hit.native_source_span == (0, 4) and hit.overlap_source_span == (0, 2)
    with pytest.raises(SourceCoverageError, match="invalid_coverage_kind"):
        NativeCoverageHit("u", (0, 4), (0, 2), "full")
