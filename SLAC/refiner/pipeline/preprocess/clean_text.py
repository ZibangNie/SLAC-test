"""
Text cleaning utilities.

Main responsibility:
- remove repeated short lines, page artifacts, decorative lines
- clean OCR noise heuristically
- preserve content needed for rule-based chunking
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CleanTextConfig:
    drop_decorative_lines: bool = True
    drop_page_number_lines: bool = True
    drop_repeated_short_lines: bool = True
    repeated_short_line_min_repeats: int = 3
    repeated_short_line_max_len: int = 80
    repeated_short_line_min_alpha_ratio: float = 0.0
    drop_isolated_noise_lines: bool = True
    noise_line_max_len: int = 3
    keep_blank_lines: bool = True
    max_consecutive_blank_lines: int = 2
    strip_text_edges: bool = True


_DECORATIVE_LINE_RE = re.compile(r"^[\-\_=*~#·•.]{3,}$")
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:第?\s*)?\d+\s*(?:页)?\s*$", re.IGNORECASE)
_PAGE_X_OF_Y_RE = re.compile(
    r"^\s*(?:page|p\.)\s*\d+\s*(?:/|of)\s*\d+\s*$",
    re.IGNORECASE,
)
_CN_PAGE_X_OF_Y_RE = re.compile(
    r"^\s*第?\s*\d+\s*页\s*(?:共\s*\d+\s*页)?\s*$",
    re.IGNORECASE,
)


def _strip_line(line: str) -> str:
    return line.strip()


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_decorative_line(line: str) -> bool:
    s = _strip_line(line)
    if not s:
        return False
    if _DECORATIVE_LINE_RE.fullmatch(s):
        return True

    # long runs of the same punctuation-like symbol
    compact = s.replace(" ", "")
    if len(compact) >= 5 and len(set(compact)) == 1 and not compact[0].isalnum():
        return True

    return False


def _is_page_number_line(line: str) -> bool:
    s = _strip_line(line)
    if not s:
        return False
    return bool(
        _PAGE_NUMBER_ONLY_RE.fullmatch(s)
        or _PAGE_X_OF_Y_RE.fullmatch(s)
        or _CN_PAGE_X_OF_Y_RE.fullmatch(s)
    )


def _normalize_repeated_line_key(line: str) -> str:
    """
    Normalize a line for repeated-header/footer detection.
    """
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha_like = sum(1 for ch in text if ch.isalpha() or "\u4e00" <= ch <= "\u9fff")
    return alpha_like / max(len(text), 1)


def _detect_repeated_short_lines(lines: List[str], cfg: CleanTextConfig) -> Set[str]:
    """
    Detect repeated short lines that are likely page headers/footers.

    Heuristic:
    - short
    - repeated >= min_repeats
    - non-empty
    """
    counter: Counter[str] = Counter()
    for line in lines:
        s = _normalize_repeated_line_key(line)
        if not s:
            continue
        if len(s) > cfg.repeated_short_line_max_len:
            continue
        if _is_page_number_line(s):
            continue
        if _is_decorative_line(s):
            continue
        if _alpha_ratio(s) < cfg.repeated_short_line_min_alpha_ratio:
            continue
        counter[s] += 1

    bad: Set[str] = {
        key
        for key, cnt in counter.items()
        if cnt >= cfg.repeated_short_line_min_repeats
    }
    return bad


def _looks_like_isolated_noise_line(line: str, cfg: CleanTextConfig) -> bool:
    """
    Conservative detection for tiny junk lines often produced by OCR/PDF extraction.

    We intentionally keep this strict to avoid deleting real short headings.
    """
    s = _strip_line(line)
    if not s:
        return False
    if len(s) > cfg.noise_line_max_len:
        return False

    # Pure punctuation / symbols
    if all((not ch.isalnum()) and not ("\u4e00" <= ch <= "\u9fff") for ch in s):
        return True

    # Single-char Latin fragments like stray OCR artifacts
    if len(s) == 1 and s.isalpha():
        return True

    return False


def clean_text_with_stats(
    text: str,
    cfg: Optional[CleanTextConfig] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Heuristic text cleaning after conservative normalization.

    This step is allowed to be more opinionated than normalize_text(),
    but it still aims to preserve structural signals for rule segmentation.
    """
    if text is None:
        text = ""
    cfg = cfg or CleanTextConfig()

    lines = text.split("\n")
    repeated_bad_lines: Set[str] = set()
    if cfg.drop_repeated_short_lines:
        repeated_bad_lines = _detect_repeated_short_lines(lines, cfg)

    cleaned_lines: List[str] = []
    stats: Dict[str, Any] = {
        "num_lines_before": len(lines),
        "num_lines_after": 0,
        "dropped_decorative_lines": 0,
        "dropped_page_number_lines": 0,
        "dropped_repeated_short_lines": 0,
        "dropped_isolated_noise_lines": 0,
        "repeated_short_line_candidates": sorted(repeated_bad_lines),
    }

    blank_run = 0

    for line in lines:
        stripped = _strip_line(line)
        repeated_key = _normalize_repeated_line_key(line)

        if cfg.drop_decorative_lines and _is_decorative_line(line):
            stats["dropped_decorative_lines"] += 1
            continue

        if cfg.drop_page_number_lines and _is_page_number_line(line):
            stats["dropped_page_number_lines"] += 1
            continue

        if cfg.drop_repeated_short_lines and repeated_key and repeated_key in repeated_bad_lines:
            stats["dropped_repeated_short_lines"] += 1
            continue

        if cfg.drop_isolated_noise_lines and _looks_like_isolated_noise_line(line, cfg):
            stats["dropped_isolated_noise_lines"] += 1
            continue

        if not stripped:
            blank_run += 1
            if not cfg.keep_blank_lines:
                continue
            if blank_run <= max(cfg.max_consecutive_blank_lines, 0):
                cleaned_lines.append("")
            continue

        blank_run = 0
        cleaned_lines.append(line.rstrip())

    cleaned_text = "\n".join(cleaned_lines)
    if cfg.strip_text_edges:
        cleaned_text = cleaned_text.strip("\n")

    stats["num_lines_after"] = len(cleaned_text.split("\n")) if cleaned_text else 0
    stats["num_chars_before"] = len(text)
    stats["num_chars_after"] = len(cleaned_text)

    return cleaned_text, stats


def clean_text(
    text: str,
    cfg: Optional[CleanTextConfig] = None,
) -> str:
    cleaned_text, _ = clean_text_with_stats(text, cfg=cfg)
    return cleaned_text


def clean_document_record(
    record: Dict[str, Any],
    cfg: Optional[CleanTextConfig] = None,
    *,
    src_field: str = "normalized_text",
    dst_field: str = "cleaned_text",
) -> Dict[str, Any]:
    """
    Clean a document record and return a copied record.
    """
    if src_field not in record:
        raise KeyError(f"Document record missing field: {src_field}")

    cfg = cfg or CleanTextConfig()
    new_record = copy.deepcopy(record)
    cleaned_text, clean_stats = clean_text_with_stats(str(record[src_field]), cfg=cfg)
    new_record[dst_field] = cleaned_text

    meta = dict(new_record.get("meta", {}))
    meta["clean"] = {
        "src_field": src_field,
        "dst_field": dst_field,
        "config": {
            "drop_decorative_lines": cfg.drop_decorative_lines,
            "drop_page_number_lines": cfg.drop_page_number_lines,
            "drop_repeated_short_lines": cfg.drop_repeated_short_lines,
            "repeated_short_line_min_repeats": cfg.repeated_short_line_min_repeats,
            "repeated_short_line_max_len": cfg.repeated_short_line_max_len,
            "repeated_short_line_min_alpha_ratio": cfg.repeated_short_line_min_alpha_ratio,
            "drop_isolated_noise_lines": cfg.drop_isolated_noise_lines,
            "noise_line_max_len": cfg.noise_line_max_len,
            "keep_blank_lines": cfg.keep_blank_lines,
            "max_consecutive_blank_lines": cfg.max_consecutive_blank_lines,
            "strip_text_edges": cfg.strip_text_edges,
        },
        "stats": clean_stats,
    }
    new_record["meta"] = meta
    return new_record