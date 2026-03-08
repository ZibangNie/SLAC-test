from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]|[^\w\s]",
    re.UNICODE,
)

_WEAK_SEP_RE = re.compile(r"([,，、:：])")
_BLANKLINE_RE = re.compile(r"\n\s*\n+")

# Common English abbreviations; deliberately conservative
_COMMON_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "sr.", "jr.", "st.", "no.", "fig.", "eq.", "al.", "inc.", "ltd.",
    "u.s.", "u.k.", "a.m.", "p.m.",
}


@dataclass
class SplitterConfig:
    line_first_min_lines: int = 3
    atom_max_tokens: int = 120
    atom_max_chars: int = 480
    merge_short_tokens: int = 8
    merge_short_chars: int = 20
    dot_in_cjk_as_terminator: bool = True
    blankline_split: bool = True


DEFAULT_CONFIG = SplitterConfig()


def split_text_to_atoms(text: str, config: SplitterConfig | None = None) -> List[str]:
    cfg = config or DEFAULT_CONFIG
    text = (text or "").strip()
    if not text:
        return []

    segments = _build_segments(text, cfg)

    atoms: List[str] = []
    for seg in segments:
        local_atoms = _sentence_split(seg, cfg.dot_in_cjk_as_terminator)
        local_atoms = [a.strip() for a in local_atoms if a and a.strip()]
        local_atoms = _enforce_max_len(local_atoms, cfg)
        local_atoms = _merge_short(local_atoms, cfg)   # 只在 segment 内合并
        local_atoms = [a.strip() for a in local_atoms if a and a.strip()]
        atoms.extend(local_atoms)

    return atoms


def _build_segments(text: str, cfg: SplitterConfig) -> List[str]:
    lines = [ln.strip() for ln in text.split("\n")]
    nonempty_lines = [ln for ln in lines if ln]

    if len(nonempty_lines) >= cfg.line_first_min_lines:
        return nonempty_lines

    if cfg.blankline_split:
        parts = [p.strip() for p in _BLANKLINE_RE.split(text) if p.strip()]
        if parts:
            return parts

    return [text.strip()]


def _sentence_split(text: str, dot_in_cjk_as_terminator: bool = True) -> List[str]:
    """
    Rule-based bilingual sentence split.
    Direct terminators: 。！？； ? ! ;
    Dot '.' is handled separately:
      - no split for decimals like 3.14
      - no split for abbreviations
      - split if adjacent to CJK when enabled
      - otherwise split on likely sentence-final dot
    """
    if not text.strip():
        return []

    out: List[str] = []
    buf: List[str] = []
    n = len(text)

    for i, ch in enumerate(text):
        buf.append(ch)

        if ch in "。！？；?!;":
            _flush_buf(out, buf)
            continue

        if ch == ".":
            if _should_split_on_dot(text, i, dot_in_cjk_as_terminator):
                _flush_buf(out, buf)
                continue

    _flush_buf(out, buf)
    return out


def _should_split_on_dot(text: str, idx: int, dot_in_cjk_as_terminator: bool) -> bool:
    prev_char = _prev_nonspace_char(text, idx)
    next_char = _next_nonspace_char(text, idx)

    if prev_char.isdigit() and next_char.isdigit():
        return False

    if dot_in_cjk_as_terminator and (_is_cjk(prev_char) or _is_cjk(next_char)):
        return True

    token = _dot_context_token(text, idx).lower()

    # a.m. / p.m. 的最后一个点，若后面像新句起点，则允许切
    if token in {"a.m.", "p.m."}:
        last_dot_pos = _last_dot_index_of_context_token(text, idx)
        if idx == last_dot_pos and _looks_like_sentence_start_after_dot(text, idx):
            return True
        return False

    if token in _COMMON_ABBREVIATIONS:
        return False

    if re.fullmatch(r"(?:[A-Za-z]\.){1,6}", token):
        return False

    next_visible_idx = _next_nonspace_index(text, idx)
    if next_visible_idx == -1:
        return True

    next_visible = text[next_visible_idx]

    if next_visible in "\"'”’)]}":
        after_close_idx = _next_nonspace_index(text, next_visible_idx)
        if after_close_idx == -1:
            return True
        next_visible = text[after_close_idx]

    if _is_cjk(next_visible):
        return True
    if next_visible.isupper():
        return True
    if next_visible.isdigit():
        return True

    return False

def _last_dot_index_of_context_token(text: str, idx: int) -> int:
    left = idx
    while left - 1 >= 0 and re.match(r"[A-Za-z.]", text[left - 1]):
        left -= 1

    right = idx
    while right + 1 < len(text) and re.match(r"[A-Za-z.]", text[right + 1]):
        right += 1

    token_text = text[left:right + 1]
    return left + token_text.rfind(".")


def _looks_like_sentence_start_after_dot(text: str, idx: int) -> bool:
    j = _next_nonspace_index(text, idx)
    if j == -1:
        return True

    ch = text[j]
    if ch in "\"'”’)]}":
        j = _next_nonspace_index(text, j)
        if j == -1:
            return True
        ch = text[j]

    return ch.isupper() or _is_cjk(ch)


def _enforce_max_len(atoms: Sequence[str], cfg: SplitterConfig) -> List[str]:
    out: List[str] = []
    for atom in atoms:
        out.extend(_split_long_atom_recursive(atom.strip(), cfg))
    return [x.strip() for x in out if x and x.strip()]


def _split_long_atom_recursive(text: str, cfg: SplitterConfig) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if not _is_overlong(text, cfg):
        return [text]

    # 1) split by newline first
    if "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(parts) > 1:
            return _recursively_process_parts(parts, cfg)

    # 2) split by weak separators
    weak_parts = _split_by_weak_separators(text)
    if len(weak_parts) > 1:
        return _recursively_process_parts(weak_parts, cfg)

    # 3) hard cut
    return _hard_cut(text, cfg)


def _recursively_process_parts(parts: Sequence[str], cfg: SplitterConfig) -> List[str]:
    out: List[str] = []
    for p in parts:
        out.extend(_split_long_atom_recursive(p, cfg))
    return out


def _split_by_weak_separators(text: str) -> List[str]:
    """
    Split while keeping delimiters attached to the left part.
    Example: "a, b, c" -> ["a,", "b,", "c"]
    """
    pieces = _WEAK_SEP_RE.split(text)
    if len(pieces) <= 1:
        return [text.strip()] if text.strip() else []

    out: List[str] = []
    cur = ""
    for piece in pieces:
        if piece == "":
            continue
        if _WEAK_SEP_RE.fullmatch(piece):
            cur += piece
            if cur.strip():
                out.append(cur.strip())
            cur = ""
        else:
            if cur:
                if cur.strip():
                    out.append(cur.strip())
                cur = piece
            else:
                cur = piece

    if cur.strip():
        out.append(cur.strip())

    return out if len(out) > 1 else [text.strip()]


def _hard_cut(text: str, cfg: SplitterConfig) -> List[str]:
    """
    Final fallback. Prefer cutting by char window roughly aligned with token budget.
    """
    max_chars = max(1, cfg.atom_max_chars)
    if len(text) <= max_chars and _token_len(text) <= cfg.atom_max_tokens:
        return [text]

    out: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + max_chars)

        # Try to retreat to a whitespace boundary for cleaner cuts
        if end < n:
            retreat = text.rfind(" ", start, end)
            if retreat != -1 and retreat > start + max_chars // 2:
                end = retreat

        piece = text[start:end].strip()
        if piece:
            # If still too long under token proxy, cut by token-like chunks more aggressively
            if _token_len(piece) > cfg.atom_max_tokens and len(piece) > 1:
                mid = max(1, len(piece) // 2)
                # prefer local whitespace
                local_retreat = piece.rfind(" ", 0, mid)
                if local_retreat != -1 and local_retreat > mid // 2:
                    split_at = local_retreat
                else:
                    split_at = mid
                left = piece[:split_at].strip()
                right = piece[split_at:].strip()
                if left:
                    out.append(left)
                if right:
                    # push back right via adjusting start
                    start = start + split_at
                    continue
            else:
                out.append(piece)

        start = end
        while start < n and text[start].isspace():
            start += 1

    return out


def _merge_short(atoms: Sequence[str], cfg: SplitterConfig) -> List[str]:
    if not atoms:
        return []

    merged: List[str] = []

    def looks_complete_sentence(x: str) -> bool:
        x = x.strip()
        return bool(x) and x[-1] in "。！？；?!;."

    for atom in atoms:
        atom = atom.strip()
        if not atom:
            continue

        if not merged:
            merged.append(atom)
            continue

        short_flag = _is_too_short(atom, cfg)

        # 关键：完整终止句不参与短碎片合并
        if short_flag and not looks_complete_sentence(atom):
            candidate = f"{merged[-1]} {atom}".strip()
            if not _is_overlong(candidate, cfg):
                merged[-1] = candidate
            else:
                merged.append(atom)
        else:
            merged.append(atom)

    # 第一项修复也保持同样原则：只有不完整碎片才往右并
    if len(merged) >= 2:
        first = merged[0].strip()
        if _is_too_short(first, cfg) and not looks_complete_sentence(first):
            candidate = f"{merged[0]} {merged[1]}".strip()
            if not _is_overlong(candidate, cfg):
                merged = [candidate] + merged[2:]

    return merged


def _is_overlong(text: str, cfg: SplitterConfig) -> bool:
    return len(text) > cfg.atom_max_chars or _token_len(text) > cfg.atom_max_tokens


def _is_too_short(text: str, cfg: SplitterConfig) -> bool:
    """
    Conservative short-fragment rule for bootstrap stage.

    Why conservative:
    - Current token length is only a regex proxy, not AtomEncoder tokenizer.
    - Using raw OR (tok < 8 OR chars < 20) will over-merge normal sentences.

    We therefore use:
    1) extremely short => always short
    2) otherwise require BOTH token-short and char-short
    """
    tok = _token_len(text)
    ch = len(text)

    # extremely tiny fragments
    if ch <= 6 or tok <= 2:
        return True

    # bootstrap-safe conservative rule
    return (ch < cfg.merge_short_chars) and (tok < cfg.merge_short_tokens)


def _token_len(text: str) -> int:
    """
    Temporary token proxy.
    Spec says atom_max_tokens should be computed with AtomEncoder tokenizer.
    For now we use a stable regex-based approximation; later swap to bge-m3 tokenizer.
    """
    return len(_TOKEN_RE.findall(text))


def _is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RE.search(ch))


def _prev_nonspace_char(text: str, idx: int) -> str:
    i = idx - 1
    while i >= 0:
        if not text[i].isspace():
            return text[i]
        i -= 1
    return ""


def _next_nonspace_char(text: str, idx: int) -> str:
    i = idx + 1
    while i < len(text):
        if not text[i].isspace():
            return text[i]
        i += 1
    return ""


def _next_nonspace_index(text: str, idx: int) -> int:
    i = idx + 1
    while i < len(text):
        if not text[i].isspace():
            return i
        i += 1
    return -1


def _dot_context_token(text: str, idx: int) -> str:
    """
    Extract a compact token around the current dot, e.g.:
      "Dr." -> "Dr."
      "e.g." -> "e.g."
      "U.S." -> "U.S."
    """
    left = idx
    while left - 1 >= 0 and re.match(r"[A-Za-z.]", text[left - 1]):
        left -= 1

    right = idx
    while right + 1 < len(text) and re.match(r"[A-Za-z.]", text[right + 1]):
        right += 1

    return text[left:right + 1].strip()


def _flush_buf(out: List[str], buf: List[str]) -> None:
    s = "".join(buf).strip()
    if s:
        out.append(s)
    buf.clear()