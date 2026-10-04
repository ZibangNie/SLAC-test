"""Synthetic entrypoint wiring checks; heavy imports and model calls never run.

Execute the actual entry function AST with explicit dependencies, so FAISS and
embedding imports are unnecessary. This checks main() wiring, not module import
side effects or the retrieval algorithm after the embedder boundary.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ENTRYPOINTS = ("run_retrieve.py", "run_retrieval_pipeline.py")
RAW = "Alpha.\n\n"


class EmbedderBoundary(RuntimeError):
    pass


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def entry_namespace(filename, tmp_path):
    source = ROOT / "SLAC/retrieval/run" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                or isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    args = SimpleNamespace(
        config=None, retrieval_build_dir=str(tmp_path / "build"),
        queries_jsonl=str(tmp_path / "queries-must-not-be-opened.jsonl"),
        output_dir=str(tmp_path / "out"), planner_cache_dir=None,
        bilingual_terms_path=None,
    )
    namespace["parse_args"] = lambda: args
    namespace["load_config"] = lambda _: {
        "common": {"encoder_name": "never-load", "batch_size": 1, "normalized": True},
    }
    for name in ("build_embedder", "load_source_indexes", "load_chunk_records",
                 "load_leaf_records", "read_jsonl", "QueryPlanner", "QueryInput",
                 "LeafDenseRetriever", "ChunkDenseRetriever", "AnchorRetriever"):
        namespace[name] = Mock(side_effect=AssertionError(f"unexpected {name}"))
    return namespace, args


def prepare_source_lookup(tmp_path):
    from SLAC.retrieval.dataio import readers, source_records
    from SLAC.retrieval.index.build_lookup_tables import (
        enrich_all_records, serialize_chunk_lookup_rows, serialize_leaf_lookup_rows,
    )

    build = tmp_path / "build"
    registry_path = build / "meta/refiner_source_indexes.json"
    view = {
        "doc_id": "synthetic-doc", "coordinate_system": "synthetic-character-offsets",
        "source_text": RAW, "model_atoms": ["Alpha."],
        "atom_char_spans": [[0, len(RAW)]],
        "model_char_origins": [[[i, i + 1] for i in range(6)]],
    }
    write_json(registry_path, {
        "schema": "slac-refiner-source-indexes-v1",
        "documents": [{"view": view, "native_units": [{
            "native_unit_id": "u0", "source_span": [0, 6], "source_text": "Alpha.",
        }]}],
    })
    registry = source_records.load_source_indexes(registry_path)
    common = {
        "doc_id": "synthetic-doc", "text": RAW, "path": [], "depth": 0,
        "parent_id": None, "source": "refiner_source_view", "text_mode": "source_document",
        "source_coordinate_system": view["coordinate_system"],
        "source_char_span": [0, len(RAW)],
        "source_document_sha256": hashlib.sha256(RAW.encode()).hexdigest(),
        "chunk_index": 0,
    }
    chunk = {**common, "chunk_id": "synthetic-doc::c0", "atom_start": 0,
             "atom_end": 1, "num_atoms": 1}
    leaf = {**common, "leaf_id": "synthetic-doc::l0", "leaf_index": 0,
            "atom_index": 0, "owner_chunk_id": chunk["chunk_id"]}
    chunk_raw, leaf_raw = tmp_path / "raw-chunks.jsonl", tmp_path / "raw-leaves.jsonl"
    write_rows(chunk_raw, [chunk])
    write_rows(leaf_raw, [leaf])
    chunks = readers.load_chunk_records(chunk_raw, source_indexes=registry)
    leaves = readers.load_leaf_records(leaf_raw, source_indexes=registry)
    chunks, leaves = enrich_all_records(chunks, leaves)
    assert chunks[0].text == leaves[0].text == "Alpha."
    assert chunks[0].meta["refiner_source"]["text"] == RAW
    assert leaves[0].meta["refiner_source"]["text"] == RAW
    write_rows(build / "meta/chunk_lookup.jsonl", serialize_chunk_lookup_rows(chunks))
    write_rows(build / "meta/leaf_lookup.jsonl", serialize_leaf_lookup_rows(leaves))
    write_rows(build / "meta/tree_adjacency.jsonl", [])
    write_json(build / "meta/quality_gates.json", {})
    write_json(build / "summaries/run_build_index_summary.json", {"status": "ok", "indexes_built": True})
    return registry_path


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_metadata_only_rejected_before_output_or_model(filename, tmp_path):
    ns, args = entry_namespace(filename, tmp_path)
    write_json(Path(args.retrieval_build_dir) / "summaries/run_build_index_summary.json",
               {"status": "ok", "indexes_built": False})
    with pytest.raises(ValueError, match="metadata only"):
        ns["main"]()
    assert not Path(args.output_dir).exists()
    for name in ("build_embedder", "load_source_indexes", "load_chunk_records",
                 "load_leaf_records", "read_jsonl", "QueryPlanner", "LeafDenseRetriever",
                 "ChunkDenseRetriever", "AnchorRetriever"):
        ns[name].assert_not_called()


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_registry_reaches_both_real_readers_before_embedder(filename, tmp_path):
    from SLAC.retrieval.dataio import readers, source_records

    registry_path = prepare_source_lookup(tmp_path)
    ns, args = entry_namespace(filename, tmp_path)
    observed = {}

    def load_registry(path):
        assert Path(path) == registry_path
        observed["registry"] = source_records.load_source_indexes(path)
        return observed["registry"]

    def checked_reader(kind, loader):
        def load(path, *, source_indexes=None):
            assert source_indexes is observed["registry"]
            records = loader(path, source_indexes=source_indexes)
            assert len(records) == 1
            row = records[0]
            assert row.text == "Alpha." and row.meta["refiner_source"]["text"] == RAW
            assert row.meta["refiner_source"]["kind"] == kind
            if kind == "chunk":
                snapshot = source_records.source_snapshot_from_chunk_record(row, source_indexes)
                assert snapshot.unit.text == RAW
            observed[kind] = records
            return records
        return load

    def read_metadata(path):
        assert Path(path) != Path(args.queries_jsonl), "query input reached before embedder boundary"
        return readers.read_jsonl(path)

    def stop_before_model(**kwargs):
        assert set(observed) == {"registry", "chunk", "leaf"}
        raise EmbedderBoundary("validated source records reached embedder boundary")

    ns.update(load_source_indexes=load_registry,
              load_chunk_records=checked_reader("chunk", readers.load_chunk_records),
              load_leaf_records=checked_reader("leaf", readers.load_leaf_records),
              read_jsonl=read_metadata, decide_tree_mode=lambda *_: "flat",
              TreeAccessor=SimpleNamespace(from_records=lambda *_: object()),
              build_embedder=stop_before_model)
    with pytest.raises(EmbedderBoundary, match="validated source records"):
        ns["main"]()
    for name in ("QueryPlanner", "LeafDenseRetriever", "ChunkDenseRetriever", "AnchorRetriever"):
        ns[name].assert_not_called()


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_source_lookup_without_registry_fails_before_model(filename, tmp_path):
    from SLAC.retrieval.dataio import readers

    registry_path = prepare_source_lookup(tmp_path)
    registry_path.unlink()  # This test's temporary registry, never repository data.
    ns, _ = entry_namespace(filename, tmp_path)
    ns.update(load_chunk_records=readers.load_chunk_records, load_leaf_records=readers.load_leaf_records)
    with pytest.raises(ValueError, match="(?i)(registry|source.index)"):
        ns["main"]()
    ns["load_source_indexes"].assert_not_called()
    ns["build_embedder"].assert_not_called()
    ns["read_jsonl"].assert_not_called()
