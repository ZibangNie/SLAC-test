from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from SLAC.retrieval.utils.text_utils import normalize_text_basic


def load_bilingual_terms(path: str | Path) -> Dict[str, List[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): [str(x) for x in v] for k, v in (data.get("terms") or {}).items()}


def expand_bilingual_terms(query: str, term_map: Dict[str, List[str]]) -> List[str]:
    x = normalize_text_basic(query, keep_newlines=False)
    out: List[str] = []

    for src, tgts in term_map.items():
        if src in x:
            for t in tgts:
                if t not in out:
                    out.append(t)

    # 反向匹配：如果 query 本身包含英文扩展词，则补中文源词
    x_low = x.lower()
    for src, tgts in term_map.items():
        for t in tgts:
            if t.lower() in x_low and src not in out:
                out.append(src)

    return out


def merge_query_with_bilingual_expansion(query: str, expansions: List[str]) -> str:
    parts = [query.strip()] + [x.strip() for x in expansions if str(x).strip()]
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return " ; ".join(out)