from __future__ import annotations

from typing import Any, Dict, List, Optional


RERANKER_INPUT_SCHEMA_VERSION = "slac_reranker_input_v1"
RERANKED_CANDIDATE_SCHEMA_VERSION = "slac_reranked_candidate_v1"


class RerankerInputValidationError(ValueError):
    """Raised when a reranker input record is invalid."""


def _is_non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _coerce_required_str(value: Any, field_name: str) -> str:
    if value is None:
        raise RerankerInputValidationError(f"missing required field: {field_name}")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise RerankerInputValidationError(f"empty required string field: {field_name}")
        return value
    value = str(value).strip()
    if value == "":
        raise RerankerInputValidationError(f"empty required string field after coercion: {field_name}")
    return value


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    value = str(value).strip()
    return value or None


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise RerankerInputValidationError(f"missing required integer field: {field_name}")
    if isinstance(value, bool):
        raise RerankerInputValidationError(f"boolean is not valid for integer field: {field_name}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise RerankerInputValidationError(f"empty integer field: {field_name}")
        try:
            return int(value)
        except ValueError as exc:
            raise RerankerInputValidationError(
                f"cannot parse integer field {field_name}: {value}"
            ) from exc
    raise RerankerInputValidationError(
        f"unsupported type for integer field {field_name}: {type(value).__name__}"
    )


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def normalize_source_views(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            return [p.strip() for p in value.split(",") if p.strip()]
        return [value]
    return [str(value).strip()] if str(value).strip() else []


def normalize_path_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        if not parts:
            return None
        return " > ".join(parts)
    text = str(value).strip()
    return text or None


def build_passage_text(
    *,
    number_signature: Optional[str],
    path_text: Optional[str],
    text: Optional[str],
) -> Optional[str]:
    parts: List[str] = []
    for item in (number_signature, path_text, text):
        s = _coerce_optional_str(item)
        if s:
            parts.append(s)
    if not parts:
        return None
    return "\n".join(parts)


def infer_role(role: Any, hit_type: Any) -> str:
    role_s = _coerce_optional_str(role)
    if role_s in {"direct", "expanded"}:
        return role_s

    hit_type_s = (_coerce_optional_str(hit_type) or "").lower()
    if "expand" in hit_type_s:
        return "expanded"
    return "direct"


def normalize_reranker_input_record(
    record: Dict[str, Any],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Normalize one line from *.reranker_input.jsonl into the canonical v1 schema.

    This function is intentionally tolerant for current/legacy retrieval outputs:
    - query_text may fall back from query / query_main / query_main_zh / query_main_en
    - passage_text may fall back from text or be rebuilt from
      number_signature + path_text + text
    - schema_version / record_type may be missing when strict=False
    """
    if not isinstance(record, dict):
        raise RerankerInputValidationError("record must be a dict")

    raw = dict(record)

    schema_version = _coerce_optional_str(raw.get("schema_version"))
    if strict:
        if schema_version != RERANKER_INPUT_SCHEMA_VERSION:
            raise RerankerInputValidationError(
                f"schema_version must be {RERANKER_INPUT_SCHEMA_VERSION}, got {schema_version!r}"
            )
    else:
        schema_version = schema_version or RERANKER_INPUT_SCHEMA_VERSION

    record_type = _coerce_optional_str(raw.get("record_type"))
    if strict:
        if record_type != "query_chunk_pair":
            raise RerankerInputValidationError(
                f"record_type must be 'query_chunk_pair', got {record_type!r}"
            )
    else:
        record_type = record_type or "query_chunk_pair"

    query_id = _coerce_required_str(raw.get("query_id"), "query_id")

    query_text = (
        raw.get("query_text")
        if _is_non_empty_str(raw.get("query_text"))
        else raw.get("query")
        if _is_non_empty_str(raw.get("query"))
        else raw.get("query_main")
        if _is_non_empty_str(raw.get("query_main"))
        else raw.get("query_main_zh")
        if _is_non_empty_str(raw.get("query_main_zh"))
        else raw.get("query_main_en")
    )
    query_text = _coerce_required_str(query_text, "query_text")

    chunk_id = _coerce_required_str(raw.get("chunk_id"), "chunk_id")
    doc_id = _coerce_required_str(raw.get("doc_id"), "doc_id")

    path_text = normalize_path_text(raw.get("path_text") if "path_text" in raw else raw.get("path"))
    number_signature = _coerce_optional_str(raw.get("number_signature"))
    anchor_text = _coerce_optional_str(raw.get("anchor_text"))
    text = _coerce_optional_str(raw.get("text"))

    passage_text = _coerce_optional_str(raw.get("passage_text"))
    if not passage_text:
        passage_text = build_passage_text(
            number_signature=number_signature,
            path_text=path_text,
            text=text,
        )
    passage_text = _coerce_required_str(passage_text, "passage_text")

    retrieve_rank_fused = _coerce_int(raw.get("retrieve_rank_fused"), "retrieve_rank_fused")
    role = infer_role(raw.get("role"), raw.get("hit_type"))
    hit_type = _coerce_required_str(raw.get("hit_type"), "hit_type")
    source_views = normalize_source_views(raw.get("source_views"))

    normalized: Dict[str, Any] = {
        "schema_version": schema_version,
        "record_type": record_type,
        "query_id": query_id,
        "query_text": query_text,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "passage_text": passage_text,
        "retrieve_rank_fused": retrieve_rank_fused,
        "role": role,
        "hit_type": hit_type,
        "source_views": source_views,
    }

    optional_fields = [
        "path_text",
        "number_signature",
        "anchor_text",
        "expansion_from",
        "expansion_type",
        "expansion_depth",
        "token_est",
        "query_lang",
        "doc_title",
        "chunk_index",
        "tree_mode",
        "intent",
        "score_leaf_dense",
        "score_chunk_dense",
        "score_anchor_bm25",
        "score_rrf",
        "owner_chunk_anchor",
        "text",
    ]

    if path_text is not None:
        normalized["path_text"] = path_text
    if number_signature is not None:
        normalized["number_signature"] = number_signature
    if anchor_text is not None:
        normalized["anchor_text"] = anchor_text
    if text is not None:
        normalized["text"] = text

    for field in optional_fields:
        if field in normalized:
            continue
        if field not in raw:
            continue
        value = raw[field]
        if field in {"expansion_depth", "token_est", "chunk_index"}:
            value = _coerce_optional_int(value)
        elif field == "path_text":
            value = normalize_path_text(value)
        elif field == "source_views":
            value = normalize_source_views(value)
        elif isinstance(value, str):
            value = value.strip() or None
        normalized[field] = value

    # Preserve any extra fields for downstream debugging / traceability.
    reserved = set(normalized.keys())
    extras = {}
    for key, value in raw.items():
        if key in reserved:
            continue
        extras[key] = value
    if extras:
        normalized["_extra"] = extras

    return normalized