from __future__ import annotations

from pathlib import Path
from typing import List

from SLAC.retrieval.dataio.writers import write_jsonl
from SLAC.retrieval.schemas.records import RetrievalCandidate


def export_reranker_input_jsonl(
    query_id: str,
    query: str,
    candidates: List[RetrievalCandidate],
    output_path: str | Path,
) -> None:
    rows = []
    for cand in candidates:
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "chunk_id": cand.chunk_id,
                "doc_id": cand.doc_id,
                "text": cand.text,
                "path": cand.path,
                "depth": cand.depth,
                "source_views": cand.source_views,
                "retrieve_rank_fused": cand.retrieve_rank_fused,
                "retrieve_score_raw": cand.retrieve_score_raw,
                "best_leaf_score": cand.best_leaf_score,
                "best_chunk_score": cand.best_chunk_score,
                "best_anchor_rank": cand.best_anchor_rank,
                "hit_type": cand.hit_type,
                "expansion_from": cand.expansion_from,
                "expansion_depth": cand.expansion_depth,
                "token_est": cand.token_est,
                "meta": cand.meta,
            }
        )
    write_jsonl(output_path, rows)