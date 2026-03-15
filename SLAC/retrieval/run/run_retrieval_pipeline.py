from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from SLAC.retrieval.configs.loader import load_config
from SLAC.retrieval.dataio.readers import load_chunk_records, load_leaf_records, read_jsonl
from SLAC.retrieval.dataio.writers import write_json
from SLAC.retrieval.index.embedder import build_embedder
from SLAC.retrieval.pack.evidence_packer import pack_evidence
from SLAC.retrieval.preprocess.query_planner import QueryPlanner
from SLAC.retrieval.schemas.query_schema import QueryInput
from SLAC.retrieval.retrieve.anchor_retriever import AnchorRetriever
from SLAC.retrieval.retrieve.chunk_aggregator import aggregate_hits_to_chunk_candidates
from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseRetriever
from SLAC.retrieval.retrieve.leaf_dense_retriever import LeafDenseRetriever
from SLAC.retrieval.retrieve.score_fusion import fuse_candidate_scores_rrf
from SLAC.retrieval.tree.downgrade_policy import decide_tree_mode
from SLAC.retrieval.tree.expansion_rules import generate_tree_expansions
from SLAC.retrieval.tree.local_subtree_reretrieve import reretrieve_local_branch_support
from SLAC.retrieval.tree.tree_accessor import TreeAccessor


def _load_quality_gates(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_tree_adjacency(path: Path):
    rows = []
    from SLAC.retrieval.schemas.records import TreeAdjacencyRecord

    for obj in read_jsonl(path):
        rows.append(
            TreeAdjacencyRecord(
                chunk_id=obj["chunk_id"],
                doc_id=obj["doc_id"],
                depth=int(obj["depth"]),
                path=list(obj.get("path", [])),
                parent_id=obj.get("parent_id"),
                children_ids=list(obj.get("children_ids", [])),
                prev_chunk_id=obj.get("prev_chunk_id"),
                next_chunk_id=obj.get("next_chunk_id"),
            )
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--retrieval_build_dir", type=str, required=True)
    p.add_argument("--queries_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--planner_cache_dir", type=str, default=None)
    p.add_argument("--bilingual_terms_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    build_dir = Path(args.retrieval_build_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunk_records(build_dir / "meta" / "chunk_lookup.jsonl")
    leaves = load_leaf_records(build_dir / "meta" / "leaf_lookup.jsonl")
    adjacency_rows = _load_tree_adjacency(build_dir / "meta" / "tree_adjacency.jsonl")
    quality_gates = _load_quality_gates(build_dir / "meta" / "quality_gates.json")

    tree_mode = decide_tree_mode(quality_gates, cfg)

    chunk_lookup = {c.chunk_id: c for c in chunks}
    leaf_lookup = {l.leaf_id: l for l in leaves}
    leaf_to_chunk = {l.leaf_id: l.owner_chunk_id for l in leaves}

    tree_accessor = TreeAccessor.from_records(chunks, adjacency_rows)

    embedder = build_embedder(
        model_name=cfg["common"]["encoder_name"],
        batch_size=int(cfg["common"]["batch_size"]),
        normalized=bool(cfg["common"]["normalized"]),
    )

    leaf_retriever = LeafDenseRetriever(build_dir / "index" / "leaf_dense", embedder)
    chunk_retriever = ChunkDenseRetriever(build_dir / "index" / "chunk_dense", embedder)
    anchor_retriever = AnchorRetriever(build_dir / "index" / "anchor_bm25")

    planner = QueryPlanner(
        cache_dir=args.planner_cache_dir,
        bilingual_terms_path=args.bilingual_terms_path,
    )

    from SLAC.retrieval.export.export_candidates import export_candidates_jsonl
    from SLAC.retrieval.export.export_packed_evidence import export_packed_evidence_jsonl
    from SLAC.retrieval.export.export_reranker_input import export_reranker_input_jsonl

    summaries = []

    for qobj in read_jsonl(args.queries_jsonl):
        q = QueryInput(
            query_id=qobj["query_id"],
            query=qobj["query"],
            lang_hint=qobj.get("lang_hint"),
            domain_hint=qobj.get("domain_hint"),
            meta=qobj.get("meta", {}),
        )
        plan = planner.plan(q)

        query_variants = []
        if plan.query_main_zh:
            query_variants.append(plan.query_main_zh)
        if plan.query_main_en:
            query_variants.append(plan.query_main_en)
        query_variants.extend(plan.subqueries)
        query_variants.extend(plan.keywords)
        query_variants.extend(plan.anchor_terms)

        seen = set()
        dedup_query_variants = []
        for x in query_variants:
            y = str(x).strip()
            if y and y not in seen:
                seen.add(y)
                dedup_query_variants.append(y)

        leaf_hits = leaf_retriever.search_many_variants(
            query_variants=dedup_query_variants,
            topk_per_variant=int(cfg["retrieve"]["leaf_topk_per_variant"]),
            merged_max=int(cfg["retrieve"]["leaf_topk_merged_max"]),
            score_floor=float(cfg["gates"]["leaf_dense_floor"]),
        )
        chunk_hits = chunk_retriever.search_many_variants(
            query_variants=dedup_query_variants,
            topk=int(cfg["retrieve"]["chunk_topk"]),
            score_floor=float(cfg["gates"]["chunk_dense_floor"]),
        )
        anchor_hits = anchor_retriever.search(
            query_variants=plan.anchor_terms + dedup_query_variants,
            topk=int(cfg["retrieve"]["anchor_topk"]),
        )

        candidates_map = aggregate_hits_to_chunk_candidates(
            leaf_hits=leaf_hits,
            chunk_hits=chunk_hits,
            anchor_hits=anchor_hits,
            chunk_lookup=chunk_lookup,
            leaf_lookup=leaf_lookup,
        )

        fused_direct = fuse_candidate_scores_rrf(
            candidates=candidates_map,
            leaf_hits=leaf_hits,
            chunk_hits=chunk_hits,
            anchor_hits=anchor_hits,
            leaf_to_chunk=leaf_to_chunk,
            fused_topn=int(cfg["retrieve"]["fused_candidate_topn"]),
        )

        for cand in fused_direct:
            cand.query_id = q.query_id
            cand.query_type = plan.intent
            cand.tree_mode = tree_mode

        tree_expanded = generate_tree_expansions(
            direct_candidates=fused_direct,
            tree_accessor=tree_accessor,
            query_type=plan.intent,
            tree_mode=tree_mode,
            config=cfg,
        )
        local_branch = reretrieve_local_branch_support(
            seed_candidates=fused_direct,
            query_variants=dedup_query_variants,
            chunk_retriever=chunk_retriever,
            tree_accessor=tree_accessor,
            config=cfg,
            max_subtree_depth=1,
        )

        all_candidates = fused_direct + tree_expanded + local_branch

        packed_items, pack_summary = pack_evidence(all_candidates, cfg)

        export_candidates_jsonl(
            query_id=q.query_id,
            query=q.query,
            query_type=plan.intent,
            tree_mode=tree_mode,
            candidates=all_candidates,
            output_path=output_dir / f"{q.query_id}.candidates.jsonl",
        )
        export_reranker_input_jsonl(
            query_id=q.query_id,
            query=q.query,
            candidates=all_candidates,
            output_path=output_dir / f"{q.query_id}.reranker_input.jsonl",
        )
        export_packed_evidence_jsonl(
            query_id=q.query_id,
            query=q.query,
            query_type=plan.intent,
            tree_mode=tree_mode,
            packed_items=packed_items,
            pack_summary=pack_summary,
            output_path=output_dir / f"{q.query_id}.packed_evidence.jsonl",
        )

        summaries.append(
            {
                "query_id": q.query_id,
                "query": q.query,
                "intent": plan.intent,
                "tree_mode": tree_mode,
                "num_leaf_hits": len(leaf_hits),
                "num_chunk_hits": len(chunk_hits),
                "num_anchor_hits": len(anchor_hits),
                "num_direct_candidates": len(fused_direct),
                "num_tree_expanded": len(tree_expanded),
                "num_local_branch": len(local_branch),
                "num_all_candidates": len(all_candidates),
                "pack_summary": pack_summary,
            }
        )

    write_json(output_dir / "run_retrieval_pipeline_summary.json", {"queries": summaries})


if __name__ == "__main__":
    main()