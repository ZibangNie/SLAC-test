"""Synthetic exporter seed identity regressions; no models or dataset access."""
from copy import deepcopy

import pytest

from SLAC.refiner.pipeline.assemble.build_refiner_input import (
    RefinerInputBuildConfig,
    build_refiner_input_from_chunk0,
)
from SLAC.refiner.pipeline.assemble.export_refined_chunks import (
    RefinedChunkExportConfig,
    _find_covering_seed_unit,
    export_refined_chunks_from_candidate,
)


def unit(uid, text, path, depth, parent):
    return {"unit_id": uid, "text": text, "path": [path],
            "depth": depth, "parent_id": parent}


def export_identity(record):
    candidate = {"candidate_type": "greedy", "prediction": {
        "b_pred_sparse": [i for i, value in enumerate(record["b0"]) if value]}}
    return export_refined_chunks_from_candidate(record, candidate)


@pytest.mark.parametrize("empty_text", ["", " \t\n"])
def test_default_builder_skipped_empty_preserves_chunk_and_leaf_seed(empty_text):
    units = [unit(0, "Alpha belongs to the first section.", "A", 1, 10),
             unit(1, empty_text, "EMPTY", 9, 99),
             unit(2, "Beta belongs to the second section.", "B", 2, 20)]
    original = deepcopy(units)
    record = build_refiner_input_from_chunk0("synthetic", units)
    assert RefinerInputBuildConfig().strict_validate
    assert RefinedChunkExportConfig().strict_validate
    assert record["unit2atom_span"] == [
        {"unit_id": 0, "start_atom": 0, "end_atom": 1},
        {"unit_id": 2, "start_atom": 1, "end_atom": 2},
    ]
    assert record["b0"] == [1]
    exported = export_identity(record)
    for rows in (exported["refined_chunks"], exported["leaf_records"]):
        assert len(rows) == 2
        for actual, expected in zip(rows, (units[0], units[2]), strict=True):
            for field in ("text", "path", "depth", "parent_id"):
                assert actual[field] == expected[field]
    assert [(c["atom_start"], c["atom_end"]) for c in exported["refined_chunks"]] == [(0, 1), (1, 2)]
    assert units == original


def test_dense_ids_keep_original_seed_metadata_and_text():
    units = [unit(0, "Alpha belongs to the first section.", "A", 1, 10),
             unit(1, "Beta belongs to the second section.", "B", 2, 20)]
    exported = export_identity(build_refiner_input_from_chunk0("synthetic", units))
    for rows in (exported["refined_chunks"], exported["leaf_records"]):
        assert [(r["text"], r["path"], r["depth"], r["parent_id"]) for r in rows] == [
            (u["text"], u["path"], u["depth"], u["parent_id"]) for u in units]


@pytest.mark.parametrize("chunk_start,expected_id", [(0, 7), (1, 11)])
def test_maximum_overlap_and_first_span_tie_do_not_follow_unit_list_order(chunk_start, expected_id):
    units = [unit(11, "B", "B", 2, 20), unit(7, "A", "A", 1, 10)]
    spans = [{"unit_id": 7, "start_atom": 0, "end_atom": 2},
             {"unit_id": 11, "start_atom": 2, "end_atom": 4}]
    assert _find_covering_seed_unit(units, spans, chunk_start, 4)["unit_id"] == expected_id


@pytest.mark.parametrize("metadata", [
    {},
    {"chunk0_units": [unit(0, "A", "A", 1, 10)]},
    {"unit2atom_span": [{"unit_id": 0, "start_atom": 0, "end_atom": 1}]},
])
def test_optional_seed_metadata_remains_optional(metadata):
    record = {"doc_id": "synthetic", "atoms": ["A"], "b0": [], **metadata}
    exported = export_identity(record)
    for row in (exported["refined_chunks"][0], exported["leaf_records"][0]):
        assert row["text"] == "A"
        assert row["path"] == []
        assert row["depth"] is row["parent_id"] is None


@pytest.mark.parametrize("mutation", [
    "missing_unit_id", "duplicate_unit_id", "missing_span_id",
    "duplicate_span_id", "unknown_span_id", "invalid_span_id",
])
def test_supplied_seed_mapping_rejects_missing_or_ambiguous_ids(mutation):
    units = [unit(0, "Alpha belongs to the first section.", "A", 1, 10),
             unit(1, "Beta belongs to the second section.", "B", 2, 20)]
    record = build_refiner_input_from_chunk0("synthetic", units)
    if mutation == "missing_unit_id":
        del record["chunk0_units"][1]["unit_id"]
    elif mutation == "duplicate_unit_id":
        record["chunk0_units"][1]["unit_id"] = 0
    elif mutation == "missing_span_id":
        del record["unit2atom_span"][1]["unit_id"]
    elif mutation == "duplicate_span_id":
        record["unit2atom_span"][1]["unit_id"] = 0
    elif mutation == "unknown_span_id":
        record["unit2atom_span"][1]["unit_id"] = 99
    else:
        record["unit2atom_span"][1]["unit_id"] = None
    with pytest.raises(ValueError, match="unit_id"):
        export_identity(record)
