from __future__ import annotations

from typing import Dict, List, Sequence, Set

from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseRetriever
from SLAC.retrieval.schemas.records import ChunkRecord, RetrievalCandidate
from SLAC.retrieval.tree.expansion_gate import decide_expansion_plan
from SLAC.retrieval.tree.tree_accessor import TreeAccessor


def _make_local_branch_candidate(
    chunk: ChunkRecord,
    seed: RetrievalCandidate,
    score: float,
    rank: int,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        path=chunk.path,
        depth=chunk.depth,
        retrieve_score_raw={
            "chunk_dense": score,
        },
        retrieve_rank_fused=None,
        source_views=list(sorted(set(seed.source_views + ["local_branch_reretrieve"]))),
        best_leaf_score=None,
        best_chunk_score=score,
        best_anchor_rank=None,
        support_leaf_ids=[],
        hit_count=0,
        hit_type="local_branch_expand",
        expansion_from=seed.chunk_id,
        expansion_type="local_branch_expand",
        expansion_depth=1,
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
            "expansion_priority": 5,
            "local_branch_score": score,
            "local_branch_rank": rank,
        },
    )


def reretrieve_local_branch_support(
    seed_candidates: List[RetrievalCandidate],
    query_variants: Sequence[str],
    chunk_retriever: ChunkDenseRetriever,
    tree_accessor: TreeAccessor,
    config: Dict,
    max_subtree_depth: int = 1,
) -> List[RetrievalCandidate]:
    """
    受控 branch reretrieve：
    - 只对符合 plan.enable_local_branch 的 seed 开放
    - 只在 seed 的 parent 局部范围内 reretrieve
    - 不做全局跨枝干扩散
    """
    out: Dict[str, RetrievalCandidate] = {}

    # 先一次全局搜，再对每个 seed 过滤到本地候选集
    # 工程上更简单；逻辑上相当于 “全局搜 + 本地域约束”
    global_hits = chunk_retriever.search_many_variants(
        query_variants=query_variants,
        topk=max(50, int(config.get("retrieve", {}).get("chunk_topk", 20)) * 3),
        score_floor=0.0,
    )

    # 方便查
    hit_map: Dict[str, tuple[float, int]] = {}
    for h in global_hits:
        prev = hit_map.get(h.object_id)
        if prev is None or h.score > prev[0] or h.rank < prev[1]:
            hit_map[h.object_id] = (h.score, h.rank)

    for seed in seed_candidates:
        plan = decide_expansion_plan(
            candidate=seed,
            query_type=seed.query_type or "unknown",
            tree_mode=seed.tree_mode or "full_tree",
            config=config,
        )
        if not plan.enable_local_branch or plan.local_branch_topk <= 0:
            continue

        allowed_ids: Set[str] = set(
            tree_accessor.get_local_branch_candidate_ids(
                seed.chunk_id,
                include_siblings=True,
                include_children_of_parent=True,
                max_subtree_depth=max_subtree_depth,
            )
        )
        if not allowed_ids:
            continue

        rows = []
        for cid in allowed_ids:
            if cid == seed.chunk_id:
                continue
            if cid in hit_map:
                score, rank = hit_map[cid]
                rows.append((cid, score, rank))

        rows.sort(key=lambda x: (-x[1], x[2], x[0]))
        rows = rows[: plan.local_branch_topk]

        for cid, score, rank in rows:
            chunk = tree_accessor.get_chunk(cid)
            if chunk is None:
                continue

            cand = _make_local_branch_candidate(
                chunk=chunk,
                seed=seed,
                score=score,
                rank=rank,
            )

            prev = out.get(cid)
            if prev is None:
                out[cid] = cand
            else:
                prev_seed_rank = prev.meta.get("seed_rank", 10**9)
                new_seed_rank = cand.meta.get("seed_rank", 10**9)
                prev_score = prev.meta.get("local_branch_score", -1e9)
                new_score = cand.meta.get("local_branch_score", -1e9)

                if new_score > prev_score or (new_score == prev_score and new_seed_rank < prev_seed_rank):
                    out[cid] = cand

    return list(out.values())