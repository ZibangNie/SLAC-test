from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from SLAC.retrieval.schemas.records import RetrievalCandidate


@dataclass
class ExpansionPlan:
    seed_chunk_id: str
    seed_rank: int
    query_type: str
    tree_mode: str

    enable_parent: bool = False
    parent_hops: int = 0

    enable_sibling: bool = False
    sibling_left: int = 0
    sibling_right: int = 0

    enable_neighbor: bool = False
    neighbor_left: int = 0
    neighbor_right: int = 0

    enable_local_branch: bool = False
    local_branch_topk: int = 0

    def as_dict(self) -> Dict:
        return {
            "seed_chunk_id": self.seed_chunk_id,
            "seed_rank": self.seed_rank,
            "query_type": self.query_type,
            "tree_mode": self.tree_mode,
            "enable_parent": self.enable_parent,
            "parent_hops": self.parent_hops,
            "enable_sibling": self.enable_sibling,
            "sibling_left": self.sibling_left,
            "sibling_right": self.sibling_right,
            "enable_neighbor": self.enable_neighbor,
            "neighbor_left": self.neighbor_left,
            "neighbor_right": self.neighbor_right,
            "enable_local_branch": self.enable_local_branch,
            "local_branch_topk": self.local_branch_topk,
        }


def _candidate_strength(candidate: RetrievalCandidate) -> float:
    vals = []
    if candidate.best_leaf_score is not None:
        vals.append(candidate.best_leaf_score)
    if candidate.best_chunk_score is not None:
        vals.append(candidate.best_chunk_score)
    if "anchor_bm25" in candidate.retrieve_score_raw:
        vals.append(candidate.retrieve_score_raw["anchor_bm25"] / 10.0)  # 轻量映射到近似尺度
    return max(vals) if vals else 0.0


def is_strong_direct(candidate: RetrievalCandidate, config: Dict) -> bool:
    gates = config.get("gates", {})
    leaf_strong = float(gates.get("leaf_dense_strong", 0.52))
    chunk_strong = float(gates.get("chunk_dense_strong", 0.48))

    anchor_exact = bool(candidate.meta.get("anchor_exact_match", False))
    best_leaf = candidate.best_leaf_score or -1e9
    best_chunk = candidate.best_chunk_score or -1e9

    if anchor_exact:
        return True
    if best_leaf >= leaf_strong:
        return True
    if best_chunk >= chunk_strong:
        return True
    if candidate.hit_count >= 2 and best_leaf >= 0.45:
        return True
    return False


def decide_expansion_plan(
    candidate: RetrievalCandidate,
    query_type: str,
    tree_mode: str,
    config: Dict,
) -> ExpansionPlan:
    gates = config.get("gates", {})
    retrieve_cfg = config.get("retrieve", {})

    rank = candidate.retrieve_rank_fused or 10**9
    strength = _candidate_strength(candidate)

    parent_topr = int(gates.get("parent_expand_topr", 20))
    parent_min_score = float(gates.get("parent_expand_min_score", 0.46))
    second_ancestor_min = float(gates.get("second_ancestor_min_score", 0.55))
    sibling_min_score = float(gates.get("sibling_expand_min_score", 0.50))
    neighbor_min_score = float(gates.get("neighbor_expand_min_score", 0.48))
    local_branch_trigger = float(gates.get("local_branch_trigger_score", 0.54))
    short_chunk_tokens = int(gates.get("short_chunk_tokens", 120))

    plan = ExpansionPlan(
        seed_chunk_id=candidate.chunk_id,
        seed_rank=rank,
        query_type=query_type,
        tree_mode=tree_mode,
    )

    if tree_mode == "leaf_only":
        return plan

    short_chunk = (candidate.token_est or 10**9) <= short_chunk_tokens
    title_like = bool(candidate.is_title_like)
    strong_direct = is_strong_direct(candidate, config)

    allow_parent = (
        rank <= parent_topr
        and (
            strength >= parent_min_score
            or short_chunk
            or title_like
            or query_type in {"definition", "enumerate", "summary", "procedure", "anchor"}
        )
    )

    if allow_parent:
        plan.enable_parent = True
        plan.parent_hops = 1

        if (
            tree_mode == "full_tree"
            and query_type in {"summary", "enumerate", "procedure"}
            and strength >= second_ancestor_min
        ):
            plan.parent_hops = 2

    allow_sibling = (
        tree_mode == "full_tree"
        and strength >= sibling_min_score
        and (
            query_type in {"enumerate", "compare"}
            or short_chunk
            or title_like
            or candidate.hit_type in {"anchor_direct", "hybrid_direct"}
        )
    )
    if allow_sibling:
        plan.enable_sibling = True
        if query_type in {"enumerate", "compare"}:
            plan.sibling_left = 2
            plan.sibling_right = 2
        else:
            plan.sibling_left = 1
            plan.sibling_right = 1

    allow_neighbor = (
        tree_mode in {"full_tree", "weak_tree"}
        and strength >= neighbor_min_score
        and (
            query_type in {"procedure", "summary"}
            or short_chunk
            or title_like
        )
    )
    if allow_neighbor:
        plan.enable_neighbor = True
        plan.neighbor_left = 1
        plan.neighbor_right = 1

    allow_local_branch = (
        tree_mode == "full_tree"
        and query_type in {"enumerate", "summary"}
        and strength >= local_branch_trigger
        and strong_direct
    )
    if allow_local_branch:
        plan.enable_local_branch = True
        plan.local_branch_topk = 3

    if tree_mode == "weak_tree":
        # weak tree 模式严格限制横向与 fan-out
        plan.enable_sibling = False
        plan.sibling_left = 0
        plan.sibling_right = 0
        plan.enable_local_branch = False
        plan.local_branch_topk = 0

    return plan