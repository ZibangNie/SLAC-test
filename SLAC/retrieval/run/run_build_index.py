from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from SLAC.retrieval.configs.loader import load_config
from SLAC.retrieval.dataio.readers import load_chunk_records, load_doc_catalog, load_leaf_records
from SLAC.retrieval.dataio.writers import write_json, write_jsonl
from SLAC.retrieval.index.build_anchor_lexical import build_anchor_lexical_index
from SLAC.retrieval.index.build_chunk_dense import build_chunk_dense_index
from SLAC.retrieval.index.build_leaf_dense import build_leaf_dense_index
from SLAC.retrieval.index.build_lookup_tables import (
    build_anchor_lookup,
    build_chunk_lookup,
    build_leaf_lookup,
    build_quality_gates,
    build_tree_adjacency,
    enrich_all_records,
    serialize_anchor_lookup_rows,
    serialize_chunk_lookup_rows,
    serialize_leaf_lookup_rows,
    serialize_tree_adjacency_rows,
)
from SLAC.retrieval.index.embedder import build_embedder
from SLAC.retrieval.schemas.records import IndexMeta
from SLAC.retrieval.schemas.validation import (
    validate_chunk_doc_consistency,
    validate_chunk_records,
    validate_doc_catalog_records,
    validate_leaf_doc_consistency,
    validate_leaf_records,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--refined_chunks_jsonl", type=str, required=True)
    p.add_argument("--leaf_records_jsonl", type=str, required=True)
    p.add_argument("--doc_catalog_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    index_dir = output_dir / "index"
    meta_dir = output_dir / "meta"
    eval_dir = output_dir / "eval"
    log_dir = output_dir / "logs"
    summary_dir = output_dir / "summaries"

    for d in [data_dir, index_dir, meta_dir, eval_dir, log_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(args.refined_chunks_jsonl)
    leaves = load_leaf_records(args.leaf_records_jsonl)
    docs = load_doc_catalog(args.doc_catalog_jsonl)

    validate_chunk_records(chunks)
    validate_doc_catalog_records(docs)
    validate_chunk_doc_consistency(chunks, docs)

    chunk_ids = [c.chunk_id for c in chunks]
    validate_leaf_records(leaves, chunk_ids)
    validate_leaf_doc_consistency(leaves, docs)

    chunks, leaves = enrich_all_records(chunks, leaves, path_joiner=" > ")

    chunk_lookup = build_chunk_lookup(chunks)
    leaf_lookup = build_leaf_lookup(leaves)
    adjacency_rows = build_tree_adjacency(chunks)
    anchor_rows = build_anchor_lookup(chunks, leaves)
    quality_gates = build_quality_gates(chunks)

    write_jsonl(meta_dir / "chunk_lookup.jsonl", serialize_chunk_lookup_rows(chunks))
    write_jsonl(meta_dir / "leaf_lookup.jsonl", serialize_leaf_lookup_rows(leaves))
    write_jsonl(meta_dir / "tree_adjacency.jsonl", serialize_tree_adjacency_rows(adjacency_rows))
    write_jsonl(meta_dir / "anchor_lookup.jsonl", serialize_anchor_lookup_rows(anchor_rows))
    write_json(meta_dir / "quality_gates.json", quality_gates)

    embedder = build_embedder(
        model_name=cfg["common"]["encoder_name"],
        batch_size=int(cfg["common"]["batch_size"]),
        normalized=bool(cfg["common"]["normalized"]),
    )

    leaf_info = build_leaf_dense_index(
        leaves=leaves,
        chunk_lookup=chunk_lookup,
        embedder=embedder,
        output_dir=index_dir / "leaf_dense",
    )
    chunk_info = build_chunk_dense_index(
        chunks=chunks,
        embedder=embedder,
        output_dir=index_dir / "chunk_dense",
    )
    anchor_info = build_anchor_lexical_index(
        anchors=anchor_rows,
        output_dir=index_dir / "anchor_bm25",
    )

    # 复制输入快照
    shutil.copy2(args.refined_chunks_jsonl, data_dir / "refined_chunks.jsonl")
    shutil.copy2(args.leaf_records_jsonl, data_dir / "leaf_records.jsonl")
    shutil.copy2(args.doc_catalog_jsonl, data_dir / "doc_catalog.jsonl")

    meta = IndexMeta(
        encoder_name=cfg["common"]["encoder_name"],
        tokenizer_name=cfg["common"]["tokenizer_name"],
        normalized=bool(cfg["common"]["normalized"]),
        similarity_metric=cfg["common"]["similarity_metric"],
        query_planner_version=cfg["common"]["query_planner_version"],
        corpus_version=f"retrieval_build_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        build_time=datetime.utcnow().isoformat(),
        refined_chunks_path=str(args.refined_chunks_jsonl),
        leaf_records_path=str(args.leaf_records_jsonl),
        doc_catalog_path=str(args.doc_catalog_jsonl),
        extra={
            "leaf_dense": leaf_info,
            "chunk_dense": chunk_info,
            "anchor_bm25": anchor_info,
            "num_chunks": len(chunks),
            "num_leaves": len(leaves),
            "num_docs": len(docs),
        },
    )
    write_json(meta_dir / "index_meta.json", meta.to_dict())

    run_summary = {
        "status": "ok",
        "num_chunks": len(chunks),
        "num_leaves": len(leaves),
        "num_docs": len(docs),
        "leaf_dense": leaf_info,
        "chunk_dense": chunk_info,
        "anchor_bm25": anchor_info,
        "quality_gates": quality_gates,
    }
    write_json(summary_dir / "run_build_index_summary.json", run_summary)


if __name__ == "__main__":
    main()