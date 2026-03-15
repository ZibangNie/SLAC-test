from __future__ import annotations

from typing import Dict, List, Optional

from SLAC.retrieval.retrieve.anchor_retriever import AnchorHit
from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseHit
from SLAC.retrieval.retrieve.leaf_dense_retriever import DenseHit
from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord, RetrievalCandidate


def aggregate_hits_to_chunk_candidates(
    leaf_hits: List[DenseHit],
    chunk_hits: List[ChunkDenseHit],
    anchor_hits: List[AnchorHit],
    chunk_lookup: Dict[str, ChunkRecord],
    leaf_lookup: Dict[str, LeafRecord],
) -> Dict[str, RetrievalCandidate]:
    candidates: Dict[str, RetrievalCandidate] = {}

    def ensure_candidate(chunk_id: str) -> RetrievalCandidate:
        if chunk_id in candidates:
            return candidates[chunk_id]

        chunk = chunk_lookup[chunk_id]
        cand = RetrievalCandidate(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            path=chunk.path,
            depth=chunk.depth,
            token_est=chunk.token_est,
            path_text=chunk.path_text,
            number_signature=chunk.number_signature,
            anchor_text=chunk.anchor_text,
            is_title_like=chunk.is_title_like,
        )
        candidates[chunk_id] = cand
        return cand

    # 1) leaf hits -> owner chunk
    for hit in leaf_hits:
        leaf = leaf_lookup.get(hit.object_id)
        if leaf is None:
            continue
        chunk_id = leaf.owner_chunk_id
        if chunk_id not in chunk_lookup:
            continue

        cand = ensure_candidate(chunk_id)
        if "leaf_dense" not in cand.source_views:
            cand.source_views.append("leaf_dense")

        cand.support_leaf_ids.append(leaf.leaf_id)
        cand.hit_count = len(set(cand.support_leaf_ids))
        prev_best = cand.best_leaf_score if cand.best_leaf_score is not None else -1e9
        cand.best_leaf_score = max(prev_best, hit.score)
        cand.retrieve_score_raw["leaf_dense"] = max(cand.retrieve_score_raw.get("leaf_dense", -1e9), hit.score)

    # 2) chunk dense hits -> direct chunk
    for hit in chunk_hits:
        chunk_id = hit.object_id
        if chunk_id not in chunk_lookup:
            continue

        cand = ensure_candidate(chunk_id)
        if "chunk_dense" not in cand.source_views:
            cand.source_views.append("chunk_dense")

        prev_best = cand.best_chunk_score if cand.best_chunk_score is not None else -1e9
        cand.best_chunk_score = max(prev_best, hit.score)
        cand.retrieve_score_raw["chunk_dense"] = max(cand.retrieve_score_raw.get("chunk_dense", -1e9), hit.score)

    # 3) anchor hits -> chunk or leaf->owner chunk
    for hit in anchor_hits:
        chunk_id: Optional[str] = None

        if hit.object_type == "chunk":
            chunk_id = hit.object_id
        elif hit.object_type == "leaf":
            leaf = leaf_lookup.get(hit.object_id)
            if leaf is not None:
                chunk_id = leaf.owner_chunk_id

        if not chunk_id or chunk_id not in chunk_lookup:
            continue

        cand = ensure_candidate(chunk_id)
        if "anchor_bm25" not in cand.source_views:
            cand.source_views.append("anchor_bm25")

        prev_rank = cand.best_anchor_rank if cand.best_anchor_rank is not None else 10**9
        cand.best_anchor_rank = min(prev_rank, hit.rank)
        cand.retrieve_score_raw["anchor_bm25"] = max(cand.retrieve_score_raw.get("anchor_bm25", -1e9), hit.score)

        if hit.exact_match:
            cand.meta["anchor_exact_match"] = True

    # 4) finalize hit_type
    for cand in candidates.values():
        if "leaf_dense" in cand.source_views and len(cand.source_views) > 1:
            cand.hit_type = "hybrid_direct"
        elif "leaf_dense" in cand.source_views:
            cand.hit_type = "leaf_direct"
        elif "chunk_dense" in cand.source_views and "anchor_bm25" in cand.source_views:
            cand.hit_type = "hybrid_direct"
        elif "chunk_dense" in cand.source_views:
            cand.hit_type = "chunk_direct"
        elif "anchor_bm25" in cand.source_views:
            cand.hit_type = "anchor_direct"
        else:
            cand.hit_type = "leaf_direct"

        cand.support_leaf_ids = sorted(set(cand.support_leaf_ids))

    return candidates