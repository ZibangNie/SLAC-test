from __future__ import annotations

from SLAC.retrieval.schemas.records import RetrievalCandidate
from SLAC.retrieval.utils.text_utils import estimate_token_count


def estimate_candidate_tokens(candidate: RetrievalCandidate) -> int:
    if candidate.token_est is not None and candidate.token_est > 0:
        return int(candidate.token_est)

    # 结构头也要占 token
    header_parts = [
        candidate.doc_id or "",
        candidate.path_text or "",
        candidate.hit_type or "",
    ]
    header = " | ".join([x for x in header_parts if x])
    text = (header + "\n" + (candidate.text or "")).strip()
    return estimate_token_count(text)