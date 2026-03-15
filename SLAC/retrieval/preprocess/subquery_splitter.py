from __future__ import annotations

import re
from typing import List

from SLAC.retrieval.utils.text_utils import normalize_text_basic


_SPLIT_PATTERNS = [
    r"[；;]",
    r"[，,]",
    r"\b以及\b",
    r"\b并且\b",
    r"\band\b",
    r"\bwith\b",
]


def split_subqueries_heuristic(query: str, max_parts: int = 3) -> List[str]:
    x = normalize_text_basic(query, keep_newlines=False)
    if not x:
        return []

    text = x
    for pat in _SPLIT_PATTERNS:
        text = re.sub(pat, "\n", text, flags=re.IGNORECASE)

    parts = [p.strip() for p in text.split("\n") if p.strip()]

    # 避免把极短碎片也当子问题
    filtered: List[str] = []
    seen = set()
    for p in parts:
        if len(p) < 2:
            continue
        if p not in seen:
            seen.add(p)
            filtered.append(p)
        if len(filtered) >= max_parts:
            break

    if not filtered and x:
        return [x]
    return filtered[:max_parts]