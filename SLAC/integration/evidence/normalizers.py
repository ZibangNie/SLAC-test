from __future__ import annotations

from typing import Any, Dict, List, Optional

from SLAC.integration.io.schemas import SelectedEvidence


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_path_text(record: Dict[str, Any]) -> Optional[str]:
    if record.get("path_text"):
        return str(record["path_text"]).strip()
    path = record.get("path")
    if isinstance(path, list):
        parts = [str(x).strip() for x in path if str(x).strip()]
        return " > ".join(parts) if parts else None
    return None


def _coerce_source_views(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    return [str(value).strip()]


def _infer_role(record: Dict[str, Any]) -> Optional[str]:
    if record.get("role"):
        return str(record["role"]).strip()
    hit_type = str(record.get("hit_type", "")).lower()
    if "expand" in hit_type:
        return "expanded"
    if hit_type:
        return "direct"
    return None


def _build_passage_text(record: Dict[str, Any]) -> str:
    if record.get("passage_text"):
        return str(record["passage_text"]).strip()

    parts: List[str] = []

    number_signature = str(record.get("number_signature", "")).strip()
    if number_signature:
        parts.append(number_signature)

    path_text = _build_path_text(record)
    if path_text:
        parts.append(path_text)

    text = str(record.get("text", "")).strip()
    if text:
        parts.append(text)

    passage = "\n".join(parts).strip()
    return passage


def _estimate_token_count(text: str) -> int:
    # 当前仅作预算 fallback；若上游已有 token_est，则优先使用上游值。
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_candidate_to_selected_evidence(
    record: Dict[str, Any],
    *,
    query_id: Optional[str] = None,
    query_text: Optional[str] = None,
    source_name: Optional[str] = None,
    ordinal: Optional[int] = None,
) -> SelectedEvidence:
    chunk_id = str(record.get("chunk_id", "")).strip()
    doc_id = str(record.get("doc_id", "")).strip()
    passage_text = _build_passage_text(record)

    if not chunk_id:
        raise ValueError("candidate missing required field: chunk_id")
    if not doc_id:
        raise ValueError("candidate missing required field: doc_id")
    if not passage_text:
        raise ValueError(f"candidate {chunk_id} missing required text/passage_text")

    meta = dict(record.get("meta", {}) or {})
    if source_name:
        meta["_candidate_source"] = source_name
    if ordinal is not None:
        meta["_source_ordinal"] = ordinal

    return SelectedEvidence(
        chunk_id=chunk_id,
        doc_id=doc_id,
        passage_text=passage_text,
        query_id=(str(record.get("query_id")).strip() if record.get("query_id") is not None else query_id),
        query_text=(str(record.get("query_text")).strip() if record.get("query_text") is not None else query_text),
        rerank_rank=_coerce_int(record.get("rerank_rank")),
        rerank_score=_coerce_float(
            record.get("rerank_score", record.get("rerank_score_norm", record.get("rerank_score_raw")))
        ),
        retrieve_rank_fused=_coerce_int(record.get("retrieve_rank_fused")),
        role=_infer_role(record),
        hit_type=(str(record.get("hit_type")).strip() if record.get("hit_type") is not None else None),
        source_views=_coerce_source_views(record.get("source_views")),
        path_text=_build_path_text(record),
        token_est=_coerce_int(record.get("token_est")) or _estimate_token_count(passage_text),
        expansion_depth=_coerce_int(record.get("expansion_depth")),
        meta=meta,
    )


def normalize_candidate_list(
    records: List[Dict[str, Any]],
    *,
    query_id: Optional[str] = None,
    query_text: Optional[str] = None,
    source_name: Optional[str] = None,
) -> List[SelectedEvidence]:
    normalized: List[SelectedEvidence] = []
    for idx, record in enumerate(records):
        normalized.append(
            normalize_candidate_to_selected_evidence(
                record,
                query_id=query_id,
                query_text=query_text,
                source_name=source_name,
                ordinal=idx,
            )
        )
    return normalized