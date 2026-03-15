from __future__ import annotations

import re
from typing import List

from SLAC.retrieval.utils.text_utils import extract_number_signature, normalize_text_basic


_RE_CN_STRUCT = re.compile(r"(第\s*[0-9一二三四五六七八九十百千]+\s*[章节条款])")
_RE_APPENDIX = re.compile(r"(附录\s*[A-Za-z0-9]+)", flags=re.IGNORECASE)
_RE_TITLE_HINT = re.compile(
    r"\b(scope|definition|definitions|general requirements?|terminology|terms?)\b",
    flags=re.IGNORECASE,
)


def extract_anchor_terms(query: str) -> List[str]:
    x = normalize_text_basic(query, keep_newlines=False)
    out: List[str] = []

    for m in _RE_CN_STRUCT.findall(x):
        y = m.strip()
        if y and y not in out:
            out.append(y)

    for m in _RE_APPENDIX.findall(x):
        y = m.strip()
        if y and y not in out:
            out.append(y)

    for m in extract_number_signature(x):
        if m not in out:
            out.append(m)

    for m in _RE_TITLE_HINT.findall(x):
        y = m.strip()
        if y and y not in out:
            out.append(y)

    # 常见中文结构关键词
    for kw in ["适用范围", "定义", "术语", "一般规定", "要求", "条件", "步骤", "范围"]:
        if kw in x and kw not in out:
            out.append(kw)

    return out