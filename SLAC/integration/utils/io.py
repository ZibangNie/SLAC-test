from __future__ import annotations

from pathlib import Path


def safe_stem(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text