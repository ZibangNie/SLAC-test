from __future__ import annotations

from typing import Dict, List, Tuple

from SLAC.retrieval.pack.dedup import dedup_candidates
from SLAC.retrieval.pack.mmr import mmr_select
from SLAC.retrieval.pack.token_counter import estimate_candidate_tokens
from SLAC.retrieval.schemas.records import PackedEvidenceItem, RetrievalCandidate


def _role_from_hit_type(hit_type: str) -> str:
    if hit_type in {"leaf_direct", "chunk_direct", "anchor_direct", "hybrid_direct"}:
        return "direct"
    if hit_type in {"parent_expand", "ancestor_expand"}:
        return "parent_context"
    if hit_type == "sibling_expand":
        return "sibling_support"
    if hit_type == "neighbor_expand":
        return "neighbor_support"
    if hit_type == "local_branch_expand":
        return "local_branch_support"
    return "direct"


def _sort_for_pack(candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
    direct_priority = {
        "hybrid_direct": 0,
        "leaf_direct": 1,
        "chunk_direct": 2,
        "anchor_direct": 3,
    }

    def key(c: RetrievalCandidate):
        role = _role_from_hit_type(c.hit_type)
        role_group = 0 if role == "direct" else 1

        direct_rank = direct_priority.get(c.hit_type, 9)
        parent_rank = 0 if c.hit_type in {"parent_expand", "ancestor_expand"} else 1

        rrf = float(c.meta.get("rrf_score", 0.0))
        base_score = max(float(c.best_leaf_score or 0.0), float(c.best_chunk_score or 0.0))
        anchor_exact = 0 if c.meta.get("anchor_exact_match", False) else 1
        expand_priority = int(c.meta.get("expansion_priority", 999))

        return (
            role_group,
            direct_rank,
            parent_rank,
            anchor_exact,
            -rrf,
            -base_score,
            expand_priority,
            c.retrieve_rank_fused or 10**9,
            c.chunk_id,
        )

    return sorted(candidates, key=key)


def pack_evidence(
    candidates: List[RetrievalCandidate],
    config: Dict,
) -> Tuple[List[PackedEvidenceItem], Dict]:
    pack_cfg = config.get("pack", {})
    evidence_budget = int(pack_cfg.get("evidence_budget_tokens", 2500))
    max_items = int(pack_cfg.get("max_packed_items", 10))
    dedup_jaccard = float(pack_cfg.get("dedup_jaccard_threshold", 0.85))

    # 先排序，再去重
    ordered = _sort_for_pack(candidates)
    deduped = dedup_candidates(ordered, jaccard_threshold=dedup_jaccard)

    # 先做一次 MMR，减少同类重复
    mmr_pool = mmr_select(deduped, max_items=max(max_items * 2, max_items), lambda_mult=0.7)

    packed: List[PackedEvidenceItem] = []
    used_tokens = 0

    for cand in mmr_pool:
        if len(packed) >= max_items:
            break

        tok = estimate_candidate_tokens(cand)
        if tok <= 0:
            continue

        # 超长 parent 不默认塞
        long_parent_tokens = int(config.get("gates", {}).get("long_parent_tokens", 280))
        if cand.hit_type in {"parent_expand", "ancestor_expand"} and tok > long_parent_tokens:
            continue

        if used_tokens + tok > evidence_budget:
            continue

        item = PackedEvidenceItem(
            order=len(packed),
            chunk_id=cand.chunk_id,
            doc_id=cand.doc_id,
            role=_role_from_hit_type(cand.hit_type),
            hit_type=cand.hit_type,
            path=cand.path,
            token_est=tok,
            text=cand.text,
            expansion_from=cand.expansion_from,
            retrieve_rank_fused=cand.retrieve_rank_fused,
            source_views=list(cand.source_views),
        )
        packed.append(item)
        used_tokens += tok

    summary = {
        "num_input_candidates": len(candidates),
        "num_after_dedup": len(deduped),
        "num_after_mmr": len(mmr_pool),
        "num_packed": len(packed),
        "used_tokens": used_tokens,
        "budget_tokens": evidence_budget,
    }
    return packed, summary