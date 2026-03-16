from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


PACK_BRIDGE_SCHEMA_VERSION = "slac_pack_bridge_candidate_v1"


DEFAULT_PACK_BRIDGE_CONFIG: Dict[str, Any] = {
    "pack_top_n": 12,
    "reserve_direct_top_n": 0,
    "reserve_expanded_top_n": 0,
    "max_token_budget": None,
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


def rerank_order_key(record: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        _safe_int(record.get("rerank_rank"), 10**9),
        _safe_int(record.get("retrieve_rank_fused"), 10**9),
        _safe_str(record.get("chunk_id")),
    )


def normalize_role(record: Dict[str, Any]) -> str:
    role = _safe_str(record.get("role"))
    if role in {"direct", "expanded"}:
        return role

    hit_type = _safe_str(record.get("hit_type")).lower()
    if "expand" in hit_type:
        return "expanded"
    return "direct"


def estimate_record_tokens(record: Dict[str, Any]) -> int:
    token_est = record.get("token_est")
    if token_est is not None:
        v = _safe_int(token_est, -1)
        if v > 0:
            return v

    passage_text = _safe_str(record.get("passage_text"))
    if passage_text == "":
        text = _safe_str(record.get("text"))
        path_text = _safe_str(record.get("path_text"))
        number_signature = _safe_str(record.get("number_signature"))
        parts = [x for x in [number_signature, path_text, text] if x]
        passage_text = "\n".join(parts)

    if passage_text == "":
        return 1

    # 中文英文混合时，用字符长度给一个稳妥近似
    # 经验上按 4 chars ≈ 1 token 取上界偏保守
    return max(1, (len(passage_text) + 3) // 4)


def dedupe_by_chunk_id(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for record in sorted(records, key=rerank_order_key):
        chunk_id = _safe_str(record.get("chunk_id"))
        if chunk_id == "":
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(record)

    return out


def select_for_packing(
    reranked_records: Sequence[Dict[str, Any]],
    pack_bridge_cfg: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = dict(DEFAULT_PACK_BRIDGE_CONFIG)
    if pack_bridge_cfg:
        cfg.update(pack_bridge_cfg)

    pack_top_n = _safe_int(cfg.get("pack_top_n"), 12)
    reserve_direct_top_n = _safe_int(cfg.get("reserve_direct_top_n"), 0)
    reserve_expanded_top_n = _safe_int(cfg.get("reserve_expanded_top_n"), 0)
    max_token_budget = cfg.get("max_token_budget")
    if max_token_budget is not None:
        max_token_budget = _safe_int(max_token_budget, 0)
        if max_token_budget <= 0:
            max_token_budget = None

    ordered = sorted(reranked_records, key=rerank_order_key)
    deduped = dedupe_by_chunk_id(ordered)

    selected: List[Dict[str, Any]] = []
    seen = set()

    def add_record(record: Dict[str, Any], reason: str) -> None:
        chunk_id = _safe_str(record.get("chunk_id"))
        if chunk_id == "" or chunk_id in seen:
            return
        new_record = dict(record)
        new_record["packing_selection_reason"] = reason
        selected.append(new_record)
        seen.add(chunk_id)

    # 1) 基础 top-n
    if pack_top_n > 0:
        for record in deduped[:pack_top_n]:
            add_record(record, "rerank_top_n")

    # 2) 角色保留
    if reserve_direct_top_n > 0:
        direct_records = [r for r in deduped if normalize_role(r) == "direct"]
        current_direct = len([r for r in selected if normalize_role(r) == "direct"])
        need = max(0, reserve_direct_top_n - current_direct)
        for record in direct_records:
            if need <= 0:
                break
            chunk_id = _safe_str(record.get("chunk_id"))
            if chunk_id in seen:
                continue
            add_record(record, "reserve_direct")
            need -= 1

    if reserve_expanded_top_n > 0:
        expanded_records = [r for r in deduped if normalize_role(r) == "expanded"]
        current_expanded = len([r for r in selected if normalize_role(r) == "expanded"])
        need = max(0, reserve_expanded_top_n - current_expanded)
        for record in expanded_records:
            if need <= 0:
                break
            chunk_id = _safe_str(record.get("chunk_id"))
            if chunk_id in seen:
                continue
            add_record(record, "reserve_expanded")
            need -= 1

    selected = sorted(selected, key=rerank_order_key)

    # 3) token budget 裁剪
    if max_token_budget is not None:
        budgeted: List[Dict[str, Any]] = []
        token_sum = 0

        for record in selected:
            est_tokens = estimate_record_tokens(record)
            if token_sum + est_tokens > max_token_budget:
                continue
            new_record = dict(record)
            new_record["packing_token_est"] = est_tokens
            budgeted.append(new_record)
            token_sum += est_tokens

        selected = budgeted
    else:
        token_sum = 0
        with_token_records: List[Dict[str, Any]] = []
        for record in selected:
            est_tokens = estimate_record_tokens(record)
            new_record = dict(record)
            new_record["packing_token_est"] = est_tokens
            with_token_records.append(new_record)
            token_sum += est_tokens
        selected = with_token_records

    for packing_order, record in enumerate(selected, start=1):
        record["packing_order"] = packing_order
        record["selected_for_packing"] = True

    summary = {
        "num_reranked_candidates": len(reranked_records),
        "num_selected_for_packing": len(selected),
        "selected_total_token_est": token_sum,
        "pack_bridge_cfg": {
            "pack_top_n": pack_top_n,
            "reserve_direct_top_n": reserve_direct_top_n,
            "reserve_expanded_top_n": reserve_expanded_top_n,
            "max_token_budget": max_token_budget,
        },
        "selected_direct_count": len([r for r in selected if normalize_role(r) == "direct"]),
        "selected_expanded_count": len([r for r in selected if normalize_role(r) == "expanded"]),
    }

    return selected, summary


def build_pack_bridge_records(
    reranked_records: Sequence[Dict[str, Any]],
    pack_bridge_cfg: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_records, summary = select_for_packing(
        reranked_records=reranked_records,
        pack_bridge_cfg=pack_bridge_cfg,
    )

    bridge_records: List[Dict[str, Any]] = []
    for record in selected_records:
        out = dict(record)
        out["schema_version"] = PACK_BRIDGE_SCHEMA_VERSION
        out["record_type"] = "query_chunk_for_packing"
        out["pack_source"] = "reranker"
        bridge_records.append(out)

    query_id = _safe_str(selected_records[0].get("query_id")) if selected_records else None
    query_text = _safe_str(selected_records[0].get("query_text")) if selected_records else None

    bundle = {
        "query_id": query_id,
        "query_text": query_text,
        "bridge_records": bridge_records,
        "summary": summary,
    }
    return bridge_records, bundle