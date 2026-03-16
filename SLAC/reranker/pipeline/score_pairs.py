from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

from SLAC.reranker.pipeline.build_pairs import build_query_pair_batch
from SLAC.reranker.pipeline.rank_candidates import stable_sigmoid


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default
    return default


def validate_pairs(pairs: Sequence[Tuple[str, str]]) -> None:
    if not isinstance(pairs, (list, tuple)):
        raise TypeError("pairs must be a list or tuple of (query_text, passage_text)")

    for idx, item in enumerate(pairs):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"pair at index {idx} must be a 2-tuple")
        q, p = item
        if _safe_str(q) == "":
            raise ValueError(f"empty query_text at pair index {idx}")
        if _safe_str(p) == "":
            raise ValueError(f"empty passage_text at pair index {idx}")


def score_pairs_with_reranker(
    *,
    pairs: Sequence[Tuple[str, str]],
    reranker: Any,
) -> Dict[str, Any]:
    validate_pairs(pairs)

    t0 = time.time()
    raw_scores = reranker.score_pairs(pairs)
    latency_sec = time.time() - t0

    if len(raw_scores) != len(pairs):
        raise ValueError(
            f"reranker returned mismatched score length: {len(raw_scores)} vs {len(pairs)}"
        )

    raw_scores = [float(x) for x in raw_scores]
    norm_scores = [float(stable_sigmoid(x)) for x in raw_scores]

    return {
        "pairs": list(pairs),
        "raw_scores": raw_scores,
        "norm_scores": norm_scores,
        "runtime_info": dict(reranker.runtime_info),
        "latency_sec": round(latency_sec, 4),
        "num_pairs": len(pairs),
    }


def build_scored_pair_records(
    *,
    selected_records: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[str, str]],
    raw_scores: Sequence[float],
    norm_scores: Sequence[float],
) -> List[Dict[str, Any]]:
    if len(selected_records) != len(pairs):
        raise ValueError(
            f"selected_records and pairs size mismatch: {len(selected_records)} vs {len(pairs)}"
        )
    if len(raw_scores) != len(pairs):
        raise ValueError(
            f"raw_scores and pairs size mismatch: {len(raw_scores)} vs {len(pairs)}"
        )
    if len(norm_scores) != len(pairs):
        raise ValueError(
            f"norm_scores and pairs size mismatch: {len(norm_scores)} vs {len(pairs)}"
        )

    out: List[Dict[str, Any]] = []

    for idx, (record, pair, raw_score, norm_score) in enumerate(
        zip(selected_records, pairs, raw_scores, norm_scores),
        start=1,
    ):
        query_text, passage_text = pair

        item = {
            "record_type": "query_chunk_pair_scored",
            "pair_index": idx,
            "query_id": record.get("query_id"),
            "query_text": query_text,
            "chunk_id": record.get("chunk_id"),
            "doc_id": record.get("doc_id"),
            "role": record.get("role"),
            "hit_type": record.get("hit_type"),
            "retrieve_rank_fused": _safe_int(record.get("retrieve_rank_fused"), 10**9),
            "passage_text": passage_text,
            "rerank_score_raw": float(raw_score),
            "rerank_score_norm": float(norm_score),
        }

        # 保留关键可解释字段，便于 debug
        for field in [
            "path_text",
            "number_signature",
            "anchor_text",
            "text",
            "source_views",
            "token_est",
            "expansion_from",
            "expansion_type",
            "expansion_depth",
            "score_leaf_dense",
            "score_chunk_dense",
            "score_anchor_bm25",
            "score_rrf",
            "tree_mode",
            "intent",
            "doc_title",
            "chunk_index",
        ]:
            if field in record:
                item[field] = record[field]

        out.append(item)

    return out


def score_query_records(
    *,
    query_records: Sequence[Dict[str, Any]],
    reranker: Any,
    candidate_pool_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    batch = build_query_pair_batch(
        query_records=query_records,
        candidate_pool_cfg=candidate_pool_cfg,
    )

    score_bundle = score_pairs_with_reranker(
        pairs=batch["pairs"],
        reranker=reranker,
    )

    scored_pair_records = build_scored_pair_records(
        selected_records=batch["selected_records"],
        pairs=batch["pairs"],
        raw_scores=score_bundle["raw_scores"],
        norm_scores=score_bundle["norm_scores"],
    )

    top_score_norm = max(score_bundle["norm_scores"]) if score_bundle["norm_scores"] else None
    min_score_norm = min(score_bundle["norm_scores"]) if score_bundle["norm_scores"] else None

    return {
        "query_id": batch["query_id"],
        "query_text": batch["query_text"],
        "input_records": batch["input_records"],
        "selected_records": batch["selected_records"],
        "pairs": batch["pairs"],
        "raw_scores": score_bundle["raw_scores"],
        "norm_scores": score_bundle["norm_scores"],
        "runtime_info": score_bundle["runtime_info"],
        "latency_sec": score_bundle["latency_sec"],
        "num_pairs": score_bundle["num_pairs"],
        "selection_stats": batch["selection_stats"],
        "scored_pair_records": scored_pair_records,
        "score_stats": {
            "top_score_norm": top_score_norm,
            "min_score_norm": min_score_norm,
        },
    }