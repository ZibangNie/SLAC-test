from __future__ import annotations

from typing import List

from SLAC.retrieval.schemas.records import RetrievalCandidate
from SLAC.retrieval.utils.text_utils import jaccard_similarity_tokens, simple_word_tokenize


def dedup_candidates(
    candidates: List[RetrievalCandidate],
    jaccard_threshold: float = 0.85,
) -> List[RetrievalCandidate]:
    kept: List[RetrievalCandidate] = []
    token_cache = {}

    for cand in candidates:
        if not cand.text:
            kept.append(cand)
            continue

        toks = token_cache.setdefault(cand.chunk_id, simple_word_tokenize(cand.text))
        drop = False

        for prev in kept:
            # 相同 chunk 直接去重
            if cand.chunk_id == prev.chunk_id:
                drop = True
                break

            # 路径完全相同 + 文本高度重叠
            if (cand.path_text or "") == (prev.path_text or ""):
                prev_toks = token_cache.setdefault(prev.chunk_id, simple_word_tokenize(prev.text))
                if jaccard_similarity_tokens(toks, prev_toks) >= jaccard_threshold:
                    drop = True
                    break

        if not drop:
            kept.append(cand)

    return kept