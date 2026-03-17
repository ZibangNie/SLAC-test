from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from SLAC.reranker.io.validators import RERANKED_CANDIDATE_SCHEMA_VERSION
from SLAC.reranker.pipeline.build_pairs import build_query_pair_batch


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rerank_sort_key(record: Dict[str, Any]) -> Tuple[float, int, str]:
    return (
        -_safe_float(record.get("rerank_score_norm"), 0.0),
        _safe_int(record.get("retrieve_rank_fused"), 10**9),
        _safe_str(record.get("chunk_id")),
    )


def clean_record_for_output(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if key in {"_line_no", "_source_file"}:
            continue
        out[key] = value
    return out


def build_scored_records(
    selected_records: Sequence[Dict[str, Any]],
    raw_scores: Sequence[float],
    runtime_info: Dict[str, Any],
    *,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    if len(selected_records) != len(raw_scores):
        raise ValueError(
            f"selected_records and raw_scores size mismatch: "
            f"{len(selected_records)} vs {len(raw_scores)}"
        )

    scored_records: List[Dict[str, Any]] = []

    for record, raw_score in zip(selected_records, raw_scores):
        base = clean_record_for_output(record)
        score_raw = float(raw_score)
        score_norm = float(stable_sigmoid(score_raw))

        base["schema_version"] = RERANKED_CANDIDATE_SCHEMA_VERSION
        base["record_type"] = "query_chunk_scored"
        base["rerank_score_raw"] = score_raw
        base["rerank_score_norm"] = score_norm
        base["rerank_score"] = score_norm if normalize else score_raw
        base["rerank_score_field"] = "rerank_score_norm" if normalize else "rerank_score_raw"

        base["rerank_backend"] = runtime_info.get("backend")
        base["rerank_model_ref"] = runtime_info.get("model_ref")
        base["rerank_device"] = runtime_info.get("device")
        base["rerank_batch_size"] = runtime_info.get("batch_size")
        base["rerank_max_length"] = runtime_info.get("max_length")
        base["rerank_torch_dtype"] = runtime_info.get("torch_dtype")

        scored_records.append(base)

    scored_records = sorted(scored_records, key=rerank_sort_key)

    for rerank_rank, record in enumerate(scored_records, start=1):
        retrieve_rank_fused = _safe_int(record.get("retrieve_rank_fused"), 10**9)
        record["rerank_rank"] = rerank_rank
        record["retrieve_rank_delta"] = retrieve_rank_fused - rerank_rank

    return scored_records


def truncate_scored_records(
    scored_records: Sequence[Dict[str, Any]],
    *,
    output_top_n: int | None = None,
) -> List[Dict[str, Any]]:
    if output_top_n is None:
        trimmed = list(scored_records)
    elif output_top_n <= 0:
        trimmed = []
    else:
        trimmed = list(scored_records[:output_top_n])

    for rerank_rank, record in enumerate(trimmed, start=1):
        retrieve_rank_fused = _safe_int(record.get("retrieve_rank_fused"), 10**9)
        record["rerank_rank"] = rerank_rank
        record["retrieve_rank_delta"] = retrieve_rank_fused - rerank_rank

    return trimmed


def build_query_rerank_result(
    *,
    query_records: Sequence[Dict[str, Any]],
    raw_scores: Sequence[float],
    runtime_info: Dict[str, Any],
    candidate_pool_cfg: Dict[str, Any] | None = None,
    normalize: bool = True,
    output_top_n: int | None = 50,
) -> Dict[str, Any]:
    batch = build_query_pair_batch(
        query_records=query_records,
        candidate_pool_cfg=candidate_pool_cfg,
    )
    selected_records = batch["selected_records"]

    scored_records = build_scored_records(
        selected_records=selected_records,
        raw_scores=raw_scores,
        runtime_info=runtime_info,
        normalize=normalize,
    )
    scored_records = truncate_scored_records(
        scored_records=scored_records,
        output_top_n=output_top_n,
    )

    top1 = scored_records[0] if scored_records else None
    summary = {
        "query_id": batch["query_id"],
        "query_text": batch["query_text"],
        "num_input_candidates": batch["selection_stats"]["num_input_candidates"],
        "num_selected_for_rerank": batch["selection_stats"]["num_selected_total"],
        "num_output_candidates": len(scored_records),
        "top1_chunk_id": top1.get("chunk_id") if top1 else None,
        "top1_doc_id": top1.get("doc_id") if top1 else None,
        "top1_rerank_score_norm": top1.get("rerank_score_norm") if top1 else None,
        "selection_stats": batch["selection_stats"],
    }

    return {
        "query_id": batch["query_id"],
        "query_text": batch["query_text"],
        "pairs": batch["pairs"],
        "selected_records": selected_records,
        "scored_records": scored_records,
        "summary": summary,
    }


def rerank_query_records(
    *,
    query_records: Sequence[Dict[str, Any]],
    reranker: Any,
    candidate_pool_cfg: Dict[str, Any] | None = None,
    normalize: bool = True,
    output_top_n: int | None = 50,
) -> Dict[str, Any]:
    batch = build_query_pair_batch(
        query_records=query_records,
        candidate_pool_cfg=candidate_pool_cfg,
    )

    raw_scores = reranker.score_pairs(batch["pairs"])
    return build_query_rerank_result(
        query_records=query_records,
        raw_scores=raw_scores,
        runtime_info=reranker.runtime_info,
        candidate_pool_cfg=candidate_pool_cfg,
        normalize=normalize,
        output_top_n=output_top_n,
    )