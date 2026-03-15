from __future__ import annotations

from typing import Dict, List

from SLAC.retrieval.schemas.records import RetrievalCandidate
from SLAC.retrieval.utils.text_utils import jaccard_similarity_tokens, simple_word_tokenize


def _candidate_base_score(c: RetrievalCandidate) -> float:
    rrf = float(c.meta.get("rrf_score", 0.0))
    leaf = float(c.best_leaf_score or 0.0)
    chunk = float(c.best_chunk_score or 0.0)
    anchor_bonus = 0.15 if c.meta.get("anchor_exact_match", False) else 0.0

    direct_bonus = 0.0
    if c.hit_type in {"leaf_direct", "chunk_direct", "anchor_direct", "hybrid_direct"}:
        direct_bonus = 0.25
    elif c.hit_type == "parent_expand":
        direct_bonus = 0.10

    return rrf + max(leaf, chunk) + anchor_bonus + direct_bonus


def mmr_select(
    candidates: List[RetrievalCandidate],
    max_items: int = 10,
    lambda_mult: float = 0.7,
) -> List[RetrievalCandidate]:
    if len(candidates) <= 1:
        return candidates[:max_items]

    token_cache: Dict[str, List[str]] = {
        c.chunk_id: simple_word_tokenize((c.path_text or "") + " " + (c.text or ""))
        for c in candidates
    }

    remaining = list(candidates)
    selected: List[RetrievalCandidate] = []

    while remaining and len(selected) < max_items:
        best_idx = 0
        best_score = float("-inf")

        for i, cand in enumerate(remaining):
            rel = _candidate_base_score(cand)

            if not selected:
                score = rel
            else:
                sim_max = 0.0
                cand_toks = token_cache[cand.chunk_id]
                for s in selected:
                    sim = jaccard_similarity_tokens(cand_toks, token_cache[s.chunk_id])
                    sim_max = max(sim_max, sim)
                score = lambda_mult * rel - (1.0 - lambda_mult) * sim_max

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected