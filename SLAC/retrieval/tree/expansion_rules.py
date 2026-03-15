from __future__ import annotations

from typing import Dict, List, Optional

from SLAC.retrieval.schemas.records import ChunkRecord, RetrievalCandidate
from SLAC.retrieval.tree.expansion_gate import decide_expansion_plan
from SLAC.retrieval.tree.tree_accessor import TreeAccessor


_EXPANSION_PRIORITY = {
    "parent_expand": 1,
    "ancestor_expand": 2,
    "sibling_expand": 3,
    "neighbor_expand": 4,
    "local_branch_expand": 5,
}


def _make_expanded_candidate(
    chunk: ChunkRecord,
    seed: RetrievalCandidate,
    hit_type: str,
    expansion_depth: int,
) -> RetrievalCandidate:
    cand = RetrievalCandidate(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        path=chunk.path,
        depth=chunk.depth,
        retrieve_score_raw=dict(seed.retrieve_score_raw),
        retrieve_rank_fused=None,
        source_views=list(sorted(set(seed.source_views + ["tree_expand"]))),
        best_leaf_score=seed.best_leaf_score,
        best_chunk_score=seed.best_chunk_score,
        best_anchor_rank=seed.best_anchor_rank,
        support_leaf_ids=[],
        hit_count=0,
        hit_type=hit_type,
        expansion_from=seed.chunk_id,
        expansion_type=hit_type,
        expansion_depth=expansion_depth,
        token_est=chunk.token_est,
        path_text=chunk.path_text,
        number_signature=chunk.number_signature,
        anchor_text=chunk.anchor_text,
        is_title_like=chunk.is_title_like,
        query_id=seed.query_id,
        query_type=seed.query_type,
        tree_mode=seed.tree_mode,
        meta={
            "seed_chunk_id": seed.chunk_id,
            "seed_rank": seed.retrieve_rank_fused,
            "seed_rrf_score": seed.meta.get("rrf_score"),
            "expansion_priority": _EXPANSION_PRIORITY.get(hit_type, 999),
        },
    )
    return cand


def merge_expansion_candidates(
    current: Dict[str, RetrievalCandidate],
    new_candidate: RetrievalCandidate,
) -> None:
    prev = current.get(new_candidate.chunk_id)
    if prev is None:
        current[new_candidate.chunk_id] = new_candidate
        return

    prev_p = int(prev.meta.get("expansion_priority", 999))
    new_p = int(new_candidate.meta.get("expansion_priority", 999))

    # 优先级更高（数值更小）或同优先级但来自更高 direct rank 的 seed
    prev_seed_rank = prev.meta.get("seed_rank", 10**9)
    new_seed_rank = new_candidate.meta.get("seed_rank", 10**9)

    if new_p < prev_p or (new_p == prev_p and new_seed_rank < prev_seed_rank):
        current[new_candidate.chunk_id] = new_candidate


def generate_tree_expansions(
    direct_candidates: List[RetrievalCandidate],
    tree_accessor: TreeAccessor,
    query_type: str,
    tree_mode: str,
    config: Dict,
) -> List[RetrievalCandidate]:
    expanded: Dict[str, RetrievalCandidate] = {}

    for seed in direct_candidates:
        if not tree_accessor.has_chunk(seed.chunk_id):
            continue

        plan = decide_expansion_plan(
            candidate=seed,
            query_type=query_type,
            tree_mode=tree_mode,
            config=config,
        )

        # parent / ancestor
        if plan.enable_parent and plan.parent_hops > 0:
            ancestor_ids = tree_accessor.get_ancestor_ids(seed.chunk_id, max_hops=plan.parent_hops)
            for i, anc_id in enumerate(ancestor_ids, start=1):
                anc = tree_accessor.get_chunk(anc_id)
                if anc is None or anc.chunk_id == seed.chunk_id:
                    continue
                hit_type = "parent_expand" if i == 1 else "ancestor_expand"
                merge_expansion_candidates(
                    expanded,
                    _make_expanded_candidate(
                        anc,
                        seed=seed,
                        hit_type=hit_type,
                        expansion_depth=i,
                    ),
                )

        # siblings
        if plan.enable_sibling:
            siblings = tree_accessor.get_siblings(
                seed.chunk_id,
                left=plan.sibling_left,
                right=plan.sibling_right,
                include_self=False,
            )
            for sib in siblings:
                if sib.chunk_id == seed.chunk_id:
                    continue
                merge_expansion_candidates(
                    expanded,
                    _make_expanded_candidate(
                        sib,
                        seed=seed,
                        hit_type="sibling_expand",
                        expansion_depth=1,
                    ),
                )

        # neighbors
        if plan.enable_neighbor:
            neighbors = tree_accessor.get_neighbors(
                seed.chunk_id,
                left=plan.neighbor_left,
                right=plan.neighbor_right,
            )
            for nb in neighbors:
                if nb.chunk_id == seed.chunk_id:
                    continue
                merge_expansion_candidates(
                    expanded,
                    _make_expanded_candidate(
                        nb,
                        seed=seed,
                        hit_type="neighbor_expand",
                        expansion_depth=1,
                    ),
                )

    return list(expanded.values())