from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_CANDIDATE_POOL_CONFIG: Dict[str, Any] = {
    "direct_top_n": 30,
    "expanded_top_n": 20,
    "max_pairs_per_query": 50,
    "dedupe_key": "chunk_id",
}


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


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_role(record: Dict[str, Any]) -> str:
    role = _safe_str(record.get("role"))
    if role in {"direct", "expanded"}:
        return role

    hit_type = _safe_str(record.get("hit_type")).lower()
    if "expand" in hit_type:
        return "expanded"
    return "direct"


def retrieval_sort_key(record: Dict[str, Any]) -> Tuple[int, str]:
    return (
        _safe_int(record.get("retrieve_rank_fused"), 10**9),
        _safe_str(record.get("chunk_id")),
    )


def group_records_by_query_id(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        query_id = _safe_str(record.get("query_id"))
        if query_id == "":
            raise ValueError("missing query_id in reranker input record")
        groups[query_id].append(record)
    return dict(groups)


def sort_records_by_retrieval(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=retrieval_sort_key)


def dedupe_records(
    records: Sequence[Dict[str, Any]],
    *,
    dedupe_key: str = "chunk_id",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for record in sort_records_by_retrieval(records):
        key_value = record.get(dedupe_key)
        if key_value is None or _safe_str(key_value) == "":
            key_value = record.get("chunk_id")
        key = _safe_str(key_value)

        if key == "":
            continue
        if key in seen:
            continue

        seen.add(key)
        out.append(record)

    return out


def split_records_by_role(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    direct: List[Dict[str, Any]] = []
    expanded: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []

    for record in sort_records_by_retrieval(records):
        role = normalize_role(record)
        if role == "direct":
            direct.append(record)
        elif role == "expanded":
            expanded.append(record)
        else:
            other.append(record)

    return direct, expanded, other


def clip_records(records: Sequence[Dict[str, Any]], top_n: int | None) -> List[Dict[str, Any]]:
    if top_n is None:
        return list(records)
    if top_n <= 0:
        return []
    return list(records[:top_n])


def select_candidate_pool(
    records: Sequence[Dict[str, Any]],
    candidate_pool_cfg: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Policy:
    1) retrieval fused rank 升序
    2) direct / expanded 分别截断
    3) 若一侧为空，则允许 other 回填
    4) union 后按 chunk_id 去重
    5) 最后按 retrieval fused rank 全局裁剪到 max_pairs_per_query
    """
    cfg = dict(DEFAULT_CANDIDATE_POOL_CONFIG)
    if candidate_pool_cfg:
        cfg.update(candidate_pool_cfg)

    direct_top_n = _safe_int(cfg.get("direct_top_n"), 30)
    expanded_top_n = _safe_int(cfg.get("expanded_top_n"), 20)
    max_pairs_per_query = _safe_int(cfg.get("max_pairs_per_query"), 50)
    dedupe_key = _safe_str(cfg.get("dedupe_key"), "chunk_id") or "chunk_id"

    sorted_records = sort_records_by_retrieval(records)
    direct_records, expanded_records, other_records = split_records_by_role(sorted_records)

    direct_selected = clip_records(direct_records, direct_top_n)
    expanded_selected = clip_records(expanded_records, expanded_top_n)

    selected_raw: List[Dict[str, Any]] = []
    selected_raw.extend(direct_selected)
    selected_raw.extend(expanded_selected)

    # 若 direct 或 expanded 缺失，用 other 回填，避免候选池过窄
    if len(direct_selected) == 0 or len(expanded_selected) == 0:
        selected_raw.extend(other_records)

    deduped_selected = dedupe_records(selected_raw, dedupe_key=dedupe_key)

    if max_pairs_per_query > 0:
        deduped_selected = deduped_selected[:max_pairs_per_query]

    stats = {
        "num_input_candidates": len(sorted_records),
        "num_direct_candidates": len(direct_records),
        "num_expanded_candidates": len(expanded_records),
        "num_other_candidates": len(other_records),
        "num_selected_direct": len([r for r in deduped_selected if normalize_role(r) == "direct"]),
        "num_selected_expanded": len([r for r in deduped_selected if normalize_role(r) == "expanded"]),
        "num_selected_total": len(deduped_selected),
        "candidate_pool_cfg": {
            "direct_top_n": direct_top_n,
            "expanded_top_n": expanded_top_n,
            "max_pairs_per_query": max_pairs_per_query,
            "dedupe_key": dedupe_key,
        },
    }
    return deduped_selected, stats


def build_rerank_pairs(selected_records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    for record in selected_records:
        query_text = _safe_str(record.get("query_text"))
        passage_text = _safe_str(record.get("passage_text"))

        if query_text == "":
            raise ValueError(f"missing query_text for query_id={record.get('query_id')}")
        if passage_text == "":
            raise ValueError(
                f"missing passage_text for query_id={record.get('query_id')} chunk_id={record.get('chunk_id')}"
            )

        pairs.append((query_text, passage_text))

    return pairs


def build_query_pair_batch(
    query_records: Sequence[Dict[str, Any]],
    candidate_pool_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not query_records:
        raise ValueError("query_records is empty")

    selected_records, select_stats = select_candidate_pool(
        query_records,
        candidate_pool_cfg=candidate_pool_cfg,
    )
    pairs = build_rerank_pairs(selected_records)

    first = query_records[0]
    query_id = _safe_str(first.get("query_id"))
    query_text = _safe_str(first.get("query_text"))

    return {
        "query_id": query_id,
        "query_text": query_text,
        "input_records": list(query_records),
        "selected_records": selected_records,
        "pairs": pairs,
        "selection_stats": select_stats,
    }