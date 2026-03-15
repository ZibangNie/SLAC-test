from __future__ import annotations

from typing import Dict, List

from SLAC.retrieval.retrieve.anchor_retriever import AnchorHit
from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseHit
from SLAC.retrieval.retrieve.leaf_dense_retriever import DenseHit
from SLAC.retrieval.schemas.records import RetrievalCandidate


def reciprocal_rank_fusion(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _anchor_weight(query_intent: str) -> float:
    if query_intent == "anchor":
        return 1.0
    if query_intent in {"definition", "procedure"}:
        return 0.75
    return 0.60


def _anchor_exact_boost(query_intent: str, exact_match: bool) -> float:
    if query_intent == "anchor" and exact_match:
        return 0.15
    return 0.0


def fuse_candidate_scores_rrf(
    candidates: Dict[str, RetrievalCandidate],
    leaf_hits: List[DenseHit],
    chunk_hits: List[ChunkDenseHit],
    anchor_hits: List[AnchorHit],
    leaf_to_chunk: Dict[str, str],
    query_intent: str,
    fused_topn: int = 40,
) -> List[RetrievalCandidate]:
    chunk_rrf: Dict[str, float] = {}
    anchor_weight = _anchor_weight(query_intent)

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

        boost = _anchor_exact_boost(query_intent, hit.exact_match)
        chunk_rrf[chunk_id] = (
            chunk_rrf.get(chunk_id, 0.0)
            + anchor_weight * reciprocal_rank_fusion(hit.rank)
            + boost
        )

    ranked_rows = []
    for cand in candidates.values():
        rrf = chunk_rrf.get(cand.chunk_id, 0.0)

        # 非 anchor query 时，纯 anchor-only 候选轻微降权，但不删除
        if query_intent != "anchor" and set(cand.source_views) == {"anchor_bm25"}:
            rrf -= 0.08

        ranked_rows.append((cand, rrf))

    ranked_rows.sort(
        key=lambda x: (
            -x[1],
            -(x[0].best_leaf_score or -1e9),
            -(x[0].best_chunk_score or -1e9),
            x[0].best_anchor_rank if x[0].best_anchor_rank is not None else 10**9,
            x[0].chunk_id,
        )
    )

    out: List[RetrievalCandidate] = []
    for rank, (cand, rrf) in enumerate(ranked_rows[:fused_topn], start=1):
        cand.retrieve_rank_fused = rank
        cand.meta["rrf_score"] = rrf
        out.append(cand)
    return out