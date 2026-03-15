from __future__ import annotations

from typing import Dict, List, Tuple

from SLAC.retrieval.retrieve.anchor_retriever import AnchorHit
from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseHit
from SLAC.retrieval.retrieve.leaf_dense_retriever import DenseHit
from SLAC.retrieval.schemas.records import RetrievalCandidate


def reciprocal_rank_fusion(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse_candidate_scores_rrf(
    candidates: Dict[str, RetrievalCandidate],
    leaf_hits: List[DenseHit],
    chunk_hits: List[ChunkDenseHit],
    anchor_hits: List[AnchorHit],
    leaf_to_chunk: Dict[str, str],
    fused_topn: int = 40,
) -> List[RetrievalCandidate]:
    chunk_rrf: Dict[str, float] = {}

    # leaf -> owner chunk
    for hit in leaf_hits:
        chunk_id = leaf_to_chunk.get(hit.object_id)
        if not chunk_id or chunk_id not in candidates:
            continue
        chunk_rrf[chunk_id] = chunk_rrf.get(chunk_id, 0.0) + reciprocal_rank_fusion(hit.rank)

    # chunk dense
    for hit in chunk_hits:
        chunk_id = hit.object_id
        if chunk_id not in candidates:
            continue
        chunk_rrf[chunk_id] = chunk_rrf.get(chunk_id, 0.0) + reciprocal_rank_fusion(hit.rank)

    # anchor
    for hit in anchor_hits:
        if hit.object_type == "chunk":
            chunk_id = hit.object_id
        else:
            chunk_id = leaf_to_chunk.get(hit.object_id)

        if not chunk_id or chunk_id not in candidates:
            continue

        boost = 0.15 if hit.exact_match else 0.0
        chunk_rrf[chunk_id] = chunk_rrf.get(chunk_id, 0.0) + reciprocal_rank_fusion(hit.rank) + boost

    ranked = sorted(
        candidates.values(),
        key=lambda c: (
            -(chunk_rrf.get(c.chunk_id, 0.0)),
            -(c.best_leaf_score or -1e9),
            -(c.best_chunk_score or -1e9),
            c.best_anchor_rank if c.best_anchor_rank is not None else 10**9,
            c.chunk_id,
        ),
    )

    out: List[RetrievalCandidate] = []
    for rank, cand in enumerate(ranked[:fused_topn], start=1):
        cand.retrieve_rank_fused = rank
        cand.meta["rrf_score"] = chunk_rrf.get(cand.chunk_id, 0.0)
        out.append(cand)
    return out