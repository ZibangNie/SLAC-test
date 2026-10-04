"""Synthetic writer/reader/build roundtrips; model imports and network are blocked."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from SLAC.refiner.pipeline.assemble.export_refined_chunks import export_refined_chunks_from_candidate
from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.retrieval.dataio.readers import load_chunk_records, load_leaf_records
from SLAC.retrieval.dataio.source_records import load_source_indexes, source_snapshot_from_chunk_record
from SLAC.retrieval.dataio.writers import write_json, write_jsonl
from SLAC.retrieval.index.build_lookup_tables import (
    enrich_all_records, serialize_chunk_lookup_rows, serialize_leaf_lookup_rows,
)
from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord
from SLAC.retrieval.utils.text_utils import normalize_text_basic


ROOT = Path(__file__).resolve().parents[2]


def fixture_files(tmp_path):
    raw = " \tＡ.\r\nB.  "
    view = DocumentSourceView("toy", "python-characters", raw, ["Ａ.", "B."],
                              [(0, 6), (6, 10)], [[(2, 3), (3, 4)], [(6, 7), (7, 8)]])
    native = [{"native_unit_id": "a", "source_span": [2, 4], "source_text": "Ａ."},
              {"native_unit_id": "b", "source_span": [6, 8], "source_text": "B."}]
    record = {"doc_id": "toy", "atoms": ["Ａ.", "B."], "b0": [1],
              "meta": {"source_path": "synthetic.txt", "source_type": "text"},
              "chunk0_units": [
                  {"unit_id": 0, "text": "Ａ.", "path": ["A"], "depth": 1, "parent_id": 0},
                  {"unit_id": 1, "text": "B.", "path": ["B"], "depth": 1, "parent_id": 0}],
              "unit2atom_span": [{"unit_id": 0, "start_atom": 0, "end_atom": 1},
                                 {"unit_id": 1, "start_atom": 1, "end_atom": 2}]}
    exported = export_refined_chunks_from_candidate(record,
        {"candidate_id": "synthetic-boundaries", "prediction": {"b_pred_sparse": [0]}}, source_view=view)
    paths = {name: tmp_path / filename for name, filename in (
        ("chunks", "chunks.jsonl"), ("leaves", "leaves.jsonl"),
        ("docs", "docs.jsonl"), ("registry", "registry.json"), ("config", "config.yaml"))}
    write_jsonl(paths["chunks"], exported["refined_chunks"])
    write_jsonl(paths["leaves"], exported["leaf_records"])
    write_jsonl(paths["docs"], [exported["doc_catalog"]])
    write_json(paths["registry"], {"schema": "slac-refiner-source-indexes-v1",
        "documents": [{"view": asdict(view), "native_units": native}]})
    paths["config"].write_text("common:\n  encoder_name: MUST_NOT_LOAD\n  tokenizer_name: MUST_NOT_LOAD\n", encoding="utf-8")
    return view, paths, exported


def test_legacy_default_record_output_remains_unchanged(tmp_path):
    chunk = {"doc_id": "legacy", "chunk_id": "c", "chunk_index": 0,
             "atom_start": 0, "atom_end": 1, "text": " unchanged  text ", "source": "refiner_epoch8"}
    leaf = {"doc_id": "legacy", "leaf_id": "l", "owner_chunk_id": "c", "leaf_index": 0,
            "atom_index": 0, "text": " unchanged  text ", "source": "refiner_epoch8",
            "boundary_meta": {"legacy_top_level_field": "ignored"}}
    write_jsonl(tmp_path / "c.jsonl", [chunk])
    write_jsonl(tmp_path / "l.jsonl", [leaf])
    expected_chunk = ChunkRecord("legacy", "c", 0, 0, 1, chunk["text"], 1, [], 0,
                                 meta={"source": "refiner_epoch8", "boundary_meta": None})
    expected_leaf = LeafRecord("legacy", "l", "c", 0, 0, 1, leaf["text"], [], 0,
                               meta={"source": "refiner_epoch8", "chunk_index": None, "parent_id": None, "atom_index": 0})
    assert load_chunk_records(tmp_path / "c.jsonl") == [expected_chunk]
    assert load_leaf_records(tmp_path / "l.jsonl") == [expected_leaf]
    assert load_chunk_records(tmp_path / "c.jsonl", source_indexes=None) == [expected_chunk]


def test_actual_export_writer_enrich_serialize_reload_keeps_raw_chunk_and_leaf_snapshots(tmp_path):
    view, paths, exported = fixture_files(tmp_path)
    indexes = load_source_indexes(paths["registry"])
    chunks = load_chunk_records(paths["chunks"], source_indexes=indexes)
    leaves = load_leaf_records(paths["leaves"], source_indexes=indexes)
    snapshots = deepcopy([r.meta["refiner_source"] for r in chunks + leaves])
    assert [r.text for r in chunks] == [r["text"] for r in exported["refined_chunks"]]
    chunks, leaves = enrich_all_records(chunks, leaves)
    assert chunks[0].text != exported["refined_chunks"][0]["text"]
    assert chunks[0].text == normalize_text_basic(exported["refined_chunks"][0]["text"], keep_newlines=True)
    assert [r.meta["refiner_source"] for r in chunks + leaves] == snapshots
    write_jsonl(tmp_path / "chunk_lookup.jsonl", serialize_chunk_lookup_rows(chunks))
    write_jsonl(tmp_path / "leaf_lookup.jsonl", serialize_leaf_lookup_rows(leaves))
    restored_chunks = load_chunk_records(tmp_path / "chunk_lookup.jsonl", source_indexes=indexes)
    restored_leaves = load_leaf_records(tmp_path / "leaf_lookup.jsonl", source_indexes=indexes)
    assert restored_chunks == chunks and restored_leaves == leaves
    for row, expected in zip(restored_chunks + restored_leaves, snapshots, strict=True):
        assert row.meta["source"] == "refiner_source_view"
        assert row.meta["text_mode"] == "source_document"
        assert row.meta["source_coordinate_system"] == view.coordinate_system
        assert row.meta["refiner_source"] == expected
    snapshot = source_snapshot_from_chunk_record(restored_chunks[0], indexes)
    assert snapshot.unit.text == exported["refined_chunks"][0]["text"]
    assert snapshot.unit.text != restored_chunks[0].text


def test_legacy_lookup_nested_meta_retains_original_top_level_overwrite_behavior(tmp_path):
    chunk = ChunkRecord("legacy", "c", 0, 0, 1, "legacy", 1, [], 0,
        meta={"source": "refiner_epoch8", "boundary_meta": {"old": True}, "other": "keep"})
    leaf = LeafRecord("legacy", "l", "c", 0, 0, 1, "legacy", [], 0,
        meta={"source": "refiner_epoch8", "chunk_index": 9, "parent_id": "parent",
              "atom_index": 7, "other": "keep"})
    write_jsonl(tmp_path / "legacy-chunk.jsonl", [chunk.to_dict()])
    write_jsonl(tmp_path / "legacy-leaf.jsonl", [leaf.to_dict()])
    restored_chunk = load_chunk_records(tmp_path / "legacy-chunk.jsonl")[0]
    restored_leaf = load_leaf_records(tmp_path / "legacy-leaf.jsonl")[0]
    assert restored_chunk.meta == {"source": None, "boundary_meta": None, "other": "keep"}
    assert restored_leaf.meta == {"source": None, "chunk_index": None, "parent_id": None,
                                  "atom_index": None, "other": "keep"}


@pytest.mark.parametrize("kind,loader", [("chunks", load_chunk_records), ("leaves", load_leaf_records)])
def test_source_mode_requires_registry_before_enrichment(tmp_path, kind, loader):
    _, paths, _ = fixture_files(tmp_path)
    with pytest.raises(ValueError):
        loader(paths[kind])


@pytest.mark.parametrize("kind,loader,key", [("chunks", load_chunk_records, "refined_chunks"),
                                             ("leaves", load_leaf_records, "leaf_records")])
def test_reader_rejects_conflicting_top_level_and_meta_source_contract(tmp_path, kind, loader, key):
    _, paths, exported = fixture_files(tmp_path)
    row = deepcopy(exported[key][0])
    row["meta"] = {"source_coordinate_system": "wrong-system"}
    write_jsonl(paths[kind], [row])
    with pytest.raises(ValueError):
        loader(paths[kind], source_indexes=load_source_indexes(paths["registry"]))


def guarded_cli(tmp_path, paths, output, *, registry=True):
    guard = tmp_path / "import_guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text('''import importlib.abc
import socket
import sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        denied = ('faiss', 'torch', 'transformers', 'sentence_transformers',
                  'SLAC.retrieval.index.embedder', 'SLAC.retrieval.index.build_chunk_dense',
                  'SLAC.retrieval.index.build_leaf_dense', 'SLAC.retrieval.index.build_anchor_lexical')
        if any(fullname == name or fullname.startswith(name + '.') for name in denied):
            raise AssertionError('forbidden model/index import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
def no_network(*args, **kwargs):
    raise AssertionError('network must not be invoked')
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
''', encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(guard), str(ROOT)]), PYTHONDONTWRITEBYTECODE="1")
    args = [sys.executable, "-m", "SLAC.retrieval.run.run_build_index", "--metadata_only",
            "--config", str(paths["config"]), "--refined_chunks_jsonl", str(paths["chunks"]),
            "--leaf_records_jsonl", str(paths["leaves"]), "--doc_catalog_jsonl", str(paths["docs"]),
            "--output_dir", str(output)]
    if registry:
        args += ["--source_indexes_json", str(paths["registry"])]
    return subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)


def test_metadata_only_actual_cli_runs_common_metadata_path_without_model_or_network(tmp_path):
    _, paths, _ = fixture_files(tmp_path)
    output = tmp_path / "metadata-build"
    result = guarded_cli(tmp_path, paths, output)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summaries/run_build_index_summary.json").read_text(encoding="utf-8"))
    assert summary["stage"] == "metadata_only" and summary["indexes_built"] is False
    assert (summary["num_docs"], summary["num_chunks"], summary["num_leaves"]) == (1, 2, 2)
    assert not list((output / "index").iterdir())
    for filename in ("chunk_lookup.jsonl", "leaf_lookup.jsonl", "anchor_lookup.jsonl", "tree_adjacency.jsonl", "quality_gates.json"):
        assert (output / "meta" / filename).is_file()
    saved_registry = output / "meta/refiner_source_indexes.json"
    assert saved_registry.read_bytes() == paths["registry"].read_bytes()
    for key, filename in (("chunks", "refined_chunks.jsonl"), ("leaves", "leaf_records.jsonl"), ("docs", "doc_catalog.jsonl")):
        assert (output / "data" / filename).read_bytes() == paths[key].read_bytes()
    indexes = load_source_indexes(saved_registry)
    chunks = load_chunk_records(output / "meta/chunk_lookup.jsonl", source_indexes=indexes)
    leaves = load_leaf_records(output / "meta/leaf_lookup.jsonl", source_indexes=indexes)
    assert all(row.meta["source"] == "refiner_source_view" for row in chunks + leaves)
    assert source_snapshot_from_chunk_record(chunks[0], indexes).unit.text != chunks[0].text


def test_metadata_only_rejects_nonempty_output_without_deleting_old_index(tmp_path):
    _, paths, _ = fixture_files(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    sentinel = output / "old-index"
    sentinel.write_bytes(b"preserve")
    result = guarded_cli(tmp_path, paths, output)
    assert result.returncode != 0 and "absent or empty" in result.stderr
    assert sentinel.read_bytes() == b"preserve" and list(output.iterdir()) == [sentinel]


def test_metadata_only_source_input_without_registry_fails_before_any_embedding(tmp_path):
    _, paths, _ = fixture_files(tmp_path)
    result = guarded_cli(tmp_path, paths, tmp_path / "missing-registry", registry=False)
    assert result.returncode != 0 and "forbidden model/index import" not in result.stderr
