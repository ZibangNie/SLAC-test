from __future__ import annotations

import re
import unicodedata
from typing import List

from SLAC.retrieval.schemas.query_schema import QueryInput


_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_TRAILING_PUNCT = re.compile(r"[？?。.!！]+$")


def normalize_query_text(text: str) -> str:
    if text is None:
        return ""
    x = unicodedata.normalize("NFKC", text)
    x = x.replace("\r\n", "\n").replace("\r", "\n").strip()
    x = _RE_MULTI_SPACE.sub(" ", x)
    x = _RE_TRAILING_PUNCT.sub("", x).strip()
    return x


def remove_polite_noise(text: str) -> str:
    polite_prefixes = [
        "请问",
        "帮我",
        "帮我查一下",
        "请帮我",
        "想问一下",
        "我想知道",
        "can you",
        "could you",
        "please",
    ]

    x = text.strip()
    x_low = x.lower()

    for p in polite_prefixes:
        if x.startswith(p):
            return x[len(p) :].strip()
        if x_low.startswith(p.lower()):
            return x[len(p) :].strip()

    return x


def normalize_query_input(q: QueryInput) -> str:
    x = normalize_query_text(q.query)
    x = remove_polite_noise(x)
    x = _RE_MULTI_SPACE.sub(" ", x).strip()
    return x


def split_long_query_heuristic(query: str, max_parts: int = 3) -> List[str]:
    """
    这只是 planner 前的启发式切分；真正的子问题以 LLM planner 输出为准。
    """
    if not query:
        return []

    seps = ["；", ";", "，", ",", "以及", "并且", "and", "with"]
    text = query
    for sep in seps:
        text = text.replace(sep, "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]

    # 去重并限制数量
    out = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max_parts:
            break
    return out