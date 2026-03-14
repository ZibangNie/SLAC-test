"""
Text normalization utilities.

Main responsibility:
- normalize unicode/newlines/spaces
- remove invisible characters
- unify text before rule segmentation
"""
from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Invisible/control characters that frequently pollute OCR / copied PDF text.
# We keep \n and \t logic separate; this list targets troublesome zero-width and BOM-like chars.
_INVISIBLE_CHARS = {
    "\ufeff",  # BOM
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
}

# Map common typography variants to stable plain-text forms.
_CHAR_REPLACEMENTS = {
    "\u00a0": " ",   # nbsp
    "\u3000": " ",   # ideographic space
    "\t": "    ",    # preserve rough indentation rather than erasing tabs
    "—": "-",
    "–": "-",
    "―": "-",
    "−": "-",
    "•": "*",
    "·": ".",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


@dataclass
class NormalizeConfig:
    apply_nfkc: bool = True
    normalize_newlines: bool = True
    remove_invisible_chars: bool = True
    replace_common_typography: bool = True
    strip_trailing_spaces: bool = True
    collapse_inner_whitespace: bool = False
    keep_blank_lines: bool = True
    max_consecutive_blank_lines: int = 2
    strip_text_edges: bool = True


def _remove_invisible_chars(text: str) -> str:
    return "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)


def _replace_common_chars(text: str) -> str:
    for src, tgt in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, tgt)
    return text


def _normalize_line(line: str, cfg: NormalizeConfig) -> str:
    if cfg.strip_trailing_spaces:
        line = line.rstrip()

    if cfg.collapse_inner_whitespace:
        # Collapse repeated spaces but keep a single space.
        # This should be used cautiously for structure-sensitive text.
        line = re.sub(r"[ \f\v]+", " ", line)

    return line


def normalize_text(text: str, cfg: Optional[NormalizeConfig] = None) -> str:
    """
    Conservative text normalization.

    Design goals:
    - make text stable across PDF/docx/txt ingestion
    - do not over-destroy structure
    - leave aggressive heuristics to clean_text.py
    """
    if text is None:
        text = ""
    cfg = cfg or NormalizeConfig()

    out = text

    if cfg.normalize_newlines:
        out = out.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.apply_nfkc:
        out = unicodedata.normalize("NFKC", out)

    if cfg.remove_invisible_chars:
        out = _remove_invisible_chars(out)

    if cfg.replace_common_typography:
        out = _replace_common_chars(out)

    lines = out.split("\n")
    normalized_lines: List[str] = []
    blank_run = 0

    for line in lines:
        line = _normalize_line(line, cfg)

        if not line.strip():
            blank_run += 1
            if not cfg.keep_blank_lines:
                continue
            if blank_run <= max(cfg.max_consecutive_blank_lines, 0):
                normalized_lines.append("")
            continue

        blank_run = 0
        normalized_lines.append(line)

    out = "\n".join(normalized_lines)

    if cfg.strip_text_edges:
        out = out.strip("\n")

    return out


def normalize_document_record(
    record: Dict[str, Any],
    cfg: Optional[NormalizeConfig] = None,
    *,
    src_field: str = "raw_text",
    dst_field: str = "normalized_text",
) -> Dict[str, Any]:
    """
    Normalize a document record and return a copied record.
    """
    if src_field not in record:
        raise KeyError(f"Document record missing field: {src_field}")

    cfg = cfg or NormalizeConfig()
    new_record = copy.deepcopy(record)
    new_record[dst_field] = normalize_text(str(record[src_field]), cfg=cfg)

    meta = dict(new_record.get("meta", {}))
    meta["normalize"] = {
        "apply_nfkc": cfg.apply_nfkc,
        "normalize_newlines": cfg.normalize_newlines,
        "remove_invisible_chars": cfg.remove_invisible_chars,
        "replace_common_typography": cfg.replace_common_typography,
        "strip_trailing_spaces": cfg.strip_trailing_spaces,
        "collapse_inner_whitespace": cfg.collapse_inner_whitespace,
        "keep_blank_lines": cfg.keep_blank_lines,
        "max_consecutive_blank_lines": cfg.max_consecutive_blank_lines,
        "strip_text_edges": cfg.strip_text_edges,
        "src_field": src_field,
        "dst_field": dst_field,
        "num_chars_before": len(str(record[src_field])),
        "num_chars_after": len(new_record[dst_field]),
    }
    new_record["meta"] = meta
    return new_record