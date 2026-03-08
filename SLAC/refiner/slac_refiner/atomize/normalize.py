from __future__ import annotations

import re
import unicodedata


_MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
_HYPHEN_LINEBREAK_RE = re.compile(r"(?<=[A-Za-z])-\n(?=[A-Za-z])")


def normalize_text(text: str) -> str:
    """
    Normalize text for atomization.

    Rules aligned with the design spec:
    1) Normalize unicode.
    2) Convert CRLF/CR to LF.
    3) Repair common English OCR hyphenation: 'exam-\nple' -> 'example'
       only when both sides are ASCII letters.
    4) Collapse spaces/tabs but keep newlines.
    5) Remove most invisible control chars while preserving '\n' and '\t' logic.
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove invisible control chars except newline/tab-like whitespace
    cleaned_chars = []
    for ch in text:
        if ch == "\n":
            cleaned_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\t",):
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # OCR hyphenated line-break repair for English words
    text = _HYPHEN_LINEBREAK_RE.sub("", text)

    # Collapse spaces per line, but do not merge lines
    lines = text.split("\n")
    lines = [_MULTI_SPACE_RE.sub(" ", ln).strip() for ln in lines]
    text = "\n".join(lines)

    # Reduce excessive blank lines: 3+ -> 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()