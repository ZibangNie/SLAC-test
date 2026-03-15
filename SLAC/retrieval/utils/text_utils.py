from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, List, Sequence


_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_NUMBER_SIGNATURE = re.compile(
    r"(?:(?:第\s*[0-9一二三四五六七八九十百千]+\s*[章节条款])|(?:[A-Za-z]?\d+(?:\.\d+){1,6})|(?:附录\s*[A-Za-z0-9]+))"
)


def normalize_text_basic(text: str, keep_newlines: bool = True) -> str:
    if text is None:
        return ""
    x = unicodedata.normalize("NFKC", text)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        lines = [_RE_MULTI_SPACE.sub(" ", ln).strip() for ln in x.split("\n")]
        x = "\n".join(lines)
        x = _RE_MULTI_NL.sub("\n\n", x)
    else:
        x = x.replace("\n", " ")
        x = _RE_MULTI_SPACE.sub(" ", x).strip()
    return x.strip()


def normalize_for_anchor_match(text: str) -> str:
    x = normalize_text_basic(text, keep_newlines=False).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def join_path(path: Sequence[str], joiner: str = " > ") -> str:
    return joiner.join([str(p).strip() for p in path if str(p).strip()])


def estimate_token_count(text: str) -> int:
    """
    统一近似 token 估计：
    - 英文按词与标点切分
    - CJK 近似按字符密度折算
    目标是全系统一致，而不是绝对精确。
    """
    if not text:
        return 0

    x = normalize_text_basic(text, keep_newlines=False)
    ascii_words = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", x)
    cjk_chars = re.findall(r"[\u3400-\u9fff]", x)

    # 粗略避免双重统计：英文部分按词，CJK 按字符 / 1.7
    eng_est = len(ascii_words)
    cjk_est = math.ceil(len(cjk_chars) / 1.7)
    return max(1, max(eng_est, cjk_est, int(len(x) / 4)))


def extract_number_signature(text: str) -> List[str]:
    if not text:
        return []
    hits = _RE_NUMBER_SIGNATURE.findall(text)
    seen = set()
    out: List[str] = []
    for h in hits:
        h = normalize_for_anchor_match(h)
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def simple_word_tokenize(text: str) -> List[str]:
    x = normalize_for_anchor_match(text)
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", x)


def jaccard_similarity_tokens(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)