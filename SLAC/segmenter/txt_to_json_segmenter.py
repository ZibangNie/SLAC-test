# -*- coding: utf-8 -*-
"""txt_to_json_segmenter.py

SLAC hard-coded coarse segmentation (TXT -> structure-tree JSON).

This module consumes *pre-extracted* UTF-8 text files (one per PDF) produced by
pdf_to_text_preprocessor.py (or any equivalent pipeline). It then:
  - cleans lines (repeat header/footer, page numbers, decorative lines)
  - detects TOC pages and downgrades them to type=other
  - detects markers (CN/EN numbering, bullets) with robustness to fullwidth
    punctuation and spacing around dots
  - reconstructs paragraphs (line-merge + marker-fragment stitching)
  - segments into units (high recall)
  - builds a strict tree satisfying validator constraints:
      root unit0, parent_id < unit_id, level = parent.level + 1
  - emits per-doc logs + diagnostics + optional sample markdown.

Dependencies: standard library only.

Author: SLAC project assistant
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Schema enums
# -----------------------------

ALLOWED_LANGUAGE = {"zh", "en", "other"}
ALLOWED_TYPES = {"title", "heading", "paragraph", "list-item", "other"}
ALLOWED_MARKER_TYPES = {"decimal", "roman", "chinese_num", "article", "alpha", "bullet", "unknown"}


# -----------------------------
# Config
# -----------------------------

TOC_KEYWORDS = [
    "contents",
    "table of contents",
    "目录",
    "目 录",
    "目次",
    "目 次",
    "索引",
    "图目录",
    "表目录",
]


@dataclass
class Config:
    # Input
    page_break_char: str = "\f"  # page separator in extracted txt

    # Repeat line removal
    repeat_line_page_ratio: float = 0.35
    repeat_line_min_len: int = 5
    repeat_line_max_len: int = 180

    # TOC detection
    toc_max_pages: int = 30

    # Line merge
    max_merge_line_len: int = 220

    # Units
    min_unit_text_chars: int = 3
    min_strong_marker_ratio: float = 0.02
    max_weak_marker_ratio: float = 0.45

    # Sampling
    sample_n: int = 16
    sample_seed: int = 13


# -----------------------------
# Utils
# -----------------------------


def md5_text(s: str) -> str:
    return "md5:" + hashlib.md5(s.encode("utf-8")).hexdigest()


def normalize_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def normalize_text_light(s: str) -> str:
    """Light normalization for marker stability."""
    if not s:
        return s
    s = s.replace("\u00a0", " ")
    # fullwidth digits
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    # common dot variants
    s = s.replace("．", ".").replace("。", ".").replace("·", ".").replace("‧", ".").replace("｡", ".")
    # brackets
    s = s.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    # dashes
    s = s.replace("—", "-").replace("–", "-").replace("－", "-")
    return s


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def detect_language(text: str) -> str:
    if not text:
        return "other"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if cjk >= max(10, lat // 2):
        return "zh"
    if lat >= max(10, cjk // 2):
        return "en"
    return "other"


def is_page_number_line(s: str) -> bool:
    x = normalize_spaces(s)
    if not x:
        return True
    if re.fullmatch(r"\d{1,4}", x):
        return True
    if re.fullmatch(r"[-–—]{1,3}\s*\d{1,4}\s*[-–—]{1,3}", x):
        return True
    return False


def is_decorative_line(s: str) -> bool:
    x = normalize_spaces(s)
    if not x:
        return True
    if re.fullmatch(r"[-=–—_]{5,}", x):
        return True
    return False


def sentence_end_punct(s: str) -> bool:
    s = s.rstrip()
    return bool(re.search(r"[.!?;:。！？；：]$", s))


def likely_toc_page(lines: List[str]) -> bool:
    if not lines:
        return False
    joined = " ".join(lines[:100]).lower()
    if any(k in joined for k in TOC_KEYWORDS):
        return True
    dot_leader = 0
    trailing_page = 0
    short_lines = 0
    total = 0
    for ln in lines:
        x = normalize_spaces(ln)
        if not x:
            continue
        total += 1
        if len(x) <= 40:
            short_lines += 1
        if re.search(r"\.{3,}\s*\d+\s*$", x):
            dot_leader += 1
        if re.search(r"\s\d+\s*$", x) and len(x) <= 80:
            trailing_page += 1
    if total == 0:
        return False
    if dot_leader / total >= 0.22 and trailing_page / total >= 0.22:
        return True
    if short_lines / total >= 0.72 and trailing_page / total >= 0.35:
        return True
    return False


def compute_repeated_lines(pages: List[List[str]], cfg: Config) -> set:
    page_count = len(pages)
    line_pages: Dict[str, int] = {}
    for pg in pages:
        seen = set()
        for ln in pg:
            x = normalize_spaces(ln)
            if not x:
                continue
            if len(x) < cfg.repeat_line_min_len or len(x) > cfg.repeat_line_max_len:
                continue
            seen.add(x)
        for x in seen:
            line_pages[x] = line_pages.get(x, 0) + 1
    repeated = set()
    for x, c in line_pages.items():
        if c / max(1, page_count) >= cfg.repeat_line_page_ratio:
            repeated.add(x)
    return repeated


# -----------------------------
# Marker detection
# -----------------------------


_ROMAN_RE = re.compile(r"^(?=[IVXLCDM]+\b)[IVXLCDM]+", re.I)

_RE_BULLET = re.compile(r"^\s*([•\-\*\u2022\u25CF\u25CB])\s+(.+)?$")
_RE_PAREN_NUM = re.compile(r"^\s*[\(（]\s*([0-9]{1,4})\s*[\)）]\s+(.+)?$")
_RE_PAREN_ALPHA = re.compile(r"^\s*[\(（]\s*([a-zA-Z])\s*[\)）]\s+(.+)?$")
_RE_ALPHA_DOT = re.compile(r"^\s*([a-zA-Z])[\).、]\s+(.+)?$")

# Chinese numerals and common list forms
_RE_CN_LIST = re.compile(r"^\s*([一二三四五六七八九十百千万零〇两]+)[、\)]\s+(.+)?$")
_RE_CN_PAREN = re.compile(r"^\s*[\(（]([一二三四五六七八九十百千万零〇两]+)[\)）]\s+(.+)?$")

_RE_ARTICLE = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇两0-9]+)(章|节|条|款|项)\b(.*)$")
_RE_PART = re.compile(r"^\s*(Part|Section|Annex|Appendix|Chapter)\s+([A-Z0-9IVX]+)\b(.*)$", re.I)

# decimal with flexible dot spacing, supports 3 . 1 . 2
_RE_DECIMAL_FLEX = re.compile(
    r"^\s*(\d{1,4}(?:\s*\.\s*\d{1,4}){0,6})([\.)]?)\s+(.+)?$"
)

# single number heading: "1 Scope" or "10 General" (more conservative)
_RE_DECIMAL_SINGLE = re.compile(r"^\s*(\d{1,3})(\)|\.|、)?\s+(.+)?$")


_MEASURE_UNITS = {
    "kn",
    "n",
    "mn",
    "mm",
    "cm",
    "m",
    "km",
    "mpa",
    "gpa",
    "pa",
    "hz",
    "khz",
    "mhz",
    "ghz",
    "kg",
    "g",
    "t",
    "s",
    "ms",
    "min",
    "h",
    "°c",
    "℃",
    "%",
    "‰",
    "db",
    "deg",
}


def _looks_like_measurement_tail(tail: str) -> bool:
    t = tail.strip().lower()
    if not t:
        return True
    # direct unit starts
    for u in _MEASURE_UNITS:
        if t.startswith(u):
            return True
        if re.match(r"^\d+\s*" + re.escape(u) + r"\b", t):
            return True
    # patterns like "100 kN" or "13 mm"
    if re.match(r"^[\d\.\-]+\s*[a-zA-Z%℃]+\b", t):
        return True
    return False


def _normalize_decimal_prefix(prefix: str) -> str:
    # Remove spaces around dots: "3 . 1 . 2" -> "3.1.2"
    p = prefix
    p = re.sub(r"\s*\.\s*", ".", p)
    return p


def cn_to_int(s: str) -> Optional[int]:
    CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000, "万": 10000}
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        s = "一" + s
    total = 0
    num = 0
    for ch in s:
        if ch not in CN_NUM:
            return None
        val = CN_NUM[ch]
        if val >= 10:
            if num == 0:
                num = 1
            total += num * val
            num = 0
        else:
            num = val
    total += num
    return total if total > 0 else None


@dataclass
class MarkerInfo:
    num_prefix: str
    marker_type: str
    strength: str  # strong|weak
    depth_hat: Optional[int] = None
    article_kind: Optional[str] = None


def detect_marker(line: str, *, allow: bool = True) -> Optional[MarkerInfo]:
    if not allow:
        return None
    s0 = normalize_text_light(line.rstrip("\n"))
    s = normalize_spaces(s0)
    if not s:
        return None

    m = _RE_BULLET.match(s0)
    if m:
        return MarkerInfo(num_prefix=m.group(1), marker_type="bullet", strength="weak")

    m = _RE_PAREN_NUM.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"({m.group(1)})", marker_type="alpha", strength="weak")

    m = _RE_PAREN_ALPHA.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"({m.group(1)})", marker_type="alpha", strength="weak")

    m = _RE_ALPHA_DOT.match(s0)
    if m:
        sym = s.strip()[1:2]  # delimiter char
        return MarkerInfo(num_prefix=f"{m.group(1)}{sym}", marker_type="alpha", strength="weak")

    m = _RE_CN_PAREN.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"({m.group(1)})", marker_type="chinese_num", strength="weak")

    m = _RE_CN_LIST.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"{m.group(1)}、", marker_type="chinese_num", strength="weak")

    m = _RE_ARTICLE.match(s0)
    if m:
        num_raw, kind, rest = m.group(1), m.group(2), m.group(3)
        _ = cn_to_int(num_raw)
        kind_depth = {"章": 1, "节": 2, "条": 3, "款": 4, "项": 5}.get(kind, 3)
        return MarkerInfo(num_prefix=f"第{num_raw}{kind}", marker_type="article", strength="strong", depth_hat=kind_depth, article_kind=kind)

    m = _RE_PART.match(s0)
    if m:
        head, key = m.group(1), m.group(2)
        head_low = head.lower()
        d = 1 if head_low in {"part", "chapter", "annex", "appendix"} else 2
        mt = "roman" if _ROMAN_RE.match(key) else "unknown"
        return MarkerInfo(num_prefix=f"{head} {key}", marker_type=mt, strength="strong", depth_hat=d)

    # Decimal flexible (requires dot segments or punctuation)
    m = _RE_DECIMAL_FLEX.match(s0)
    if m:
        raw = _normalize_decimal_prefix(m.group(1))
        tail = (m.group(3) or "").strip()
        # disallow measurement-only lines when no dot segments (handled below)
        parts = raw.split(".")
        d = len(parts)
        return MarkerInfo(num_prefix=raw + (m.group(2) or ""), marker_type="decimal", strength="strong", depth_hat=d)

    # single number heading: be conservative to avoid tables/measurements
    m = _RE_DECIMAL_SINGLE.match(s0)
    if m:
        num = m.group(1)
        punct = m.group(2) or ""
        tail = (m.group(3) or "").strip()
        # Avoid numeric/measurement tails
        if _looks_like_measurement_tail(tail):
            return None
        # Avoid very large numbers; headings rarely start with >= 200
        try:
            if int(num) >= 200:
                return None
        except Exception:
            pass
        # Tail must contain letter or CJK
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", tail):
            return None
        return MarkerInfo(num_prefix=num + punct, marker_type="decimal", strength="strong", depth_hat=1)

    # Roman numeral like "IV. ..."
    s_norm = normalize_spaces(s0)
    rm = _ROMAN_RE.match(s_norm)
    if rm and re.match(r"^[IVXLCDM]+[\.)]\s+", s_norm, re.I):
        key = rm.group(0)
        return MarkerInfo(num_prefix=key, marker_type="roman", strength="strong", depth_hat=1)

    return None


# -----------------------------
# Text -> cleaned lines
# -----------------------------


def read_txt_pages(txt_path: Path, cfg: Config) -> List[List[str]]:
    text = txt_path.read_text("utf-8", errors="ignore")
    # page separator: by \f char (preferred)
    if cfg.page_break_char and cfg.page_break_char in text:
        pages_raw = text.split(cfg.page_break_char)
    else:
        # fallback: treat entire text as one page
        pages_raw = [text]
    pages: List[List[str]] = []
    for p in pages_raw:
        # normalize newlines
        p = p.replace("\r\n", "\n").replace("\r", "\n")
        pages.append(p.split("\n"))
    return pages


def clean_pages(pages: List[List[str]], cfg: Config) -> Tuple[List[str], List[bool]]:
    repeated = compute_repeated_lines(pages, cfg)
    cleaned_lines: List[str] = []
    toc_flags: List[bool] = []

    for pi, pg in enumerate(pages):
        pg_norm = [normalize_spaces(normalize_text_light(x)) for x in pg]
        # basic filtering
        pg_norm = [x for x in pg_norm if x and not is_page_number_line(x) and not is_decorative_line(x)]
        is_toc = (pi < cfg.toc_max_pages) and likely_toc_page(pg_norm)
        for x in pg_norm:
            if x in repeated:
                continue
            cleaned_lines.append(x)
            toc_flags.append(is_toc)
        # page separator blank
        cleaned_lines.append("")
        toc_flags.append(is_toc)
    return cleaned_lines, toc_flags


# -----------------------------
# Paragraph reconstruction
# -----------------------------


def should_merge_lines(curr: str, nxt: str, lang_hint: str, *, nxt_has_marker: bool) -> bool:
    if not curr or not nxt:
        return False
    if nxt_has_marker:
        return False
    if len(curr) > 220:
        return False
    # hyphenation in English
    if lang_hint == "en" and re.search(r"[A-Za-z]-$", curr.strip()):
        return True
    # if curr ends with sentence-ending punctuation, avoid merging
    if sentence_end_punct(curr):
        return False
    # avoid merging if nxt looks like a heading (all-caps / short)
    if lang_hint == "en":
        if len(nxt) <= 60 and re.fullmatch(r"[A-Z0-9][A-Z0-9 \-:;,\.\(\)]+", nxt.strip()):
            return False
    return True


def join_with_space(a: str, b: str, lang: str) -> str:
    if lang == "zh" or has_cjk(a) or has_cjk(b):
        return a.rstrip() + b.lstrip()
    # handle hyphenation
    if lang == "en" and re.search(r"[A-Za-z]-$", a.rstrip()):
        return a.rstrip()[:-1] + b.lstrip()
    return a.rstrip() + " " + b.lstrip()


def try_stitch_marker_fragment(prev: str, curr: str) -> Optional[str]:
    """Fix pattern like:
      prev = '3.'
      curr = '1.2 Title'
    -> '3.1.2 Title'
    """
    p = normalize_spaces(normalize_text_light(prev))
    c = normalize_spaces(normalize_text_light(curr))
    if re.fullmatch(r"\d{1,4}\.", p) and re.match(r"^\d{1,4}(?:\s*\.\s*\d{1,4})+\b", c):
        pnum = p[:-1]
        # merge without extra space
        return pnum + "." + c
    return None


def merge_continuation_lines(lines: List[str], toc_flags: List[bool], lang: str) -> Tuple[List[str], List[bool], Dict[str, Any]]:
    out_lines: List[str] = []
    out_toc: List[bool] = []
    stitched = 0
    merged = 0

    i = 0
    while i < len(lines):
        curr = lines[i]
        curr_toc = toc_flags[i]
        if not curr:
            out_lines.append(curr)
            out_toc.append(curr_toc)
            i += 1
            continue

        # stitch marker fragments: if previous emitted line is fragment like '3.'
        if out_lines and out_lines[-1] and not out_toc[-1] and curr_toc == out_toc[-1]:
            stitched_line = try_stitch_marker_fragment(out_lines[-1], curr)
            if stitched_line is not None:
                out_lines[-1] = stitched_line
                stitched += 1
                i += 1
                continue

        mk_curr = detect_marker(curr)
        curr_has_marker = mk_curr is not None

        acc = curr
        j = i
        while True:
            if j + 1 >= len(lines):
                break
            nxt = lines[j + 1]
            nxt_toc = toc_flags[j + 1]
            if not nxt:
                break
            if nxt_toc != curr_toc:
                break
            mk_nxt = detect_marker(nxt)
            nxt_has_marker = mk_nxt is not None
            # do not merge across any marker (strong/weak)
            if curr_has_marker or nxt_has_marker:
                break
            if not should_merge_lines(acc, nxt, lang, nxt_has_marker=nxt_has_marker):
                break
            acc = join_with_space(acc, nxt, lang)
            merged += 1
            j += 1
        out_lines.append(acc)
        out_toc.append(curr_toc)
        i = j + 1

    meta = {"stitched_marker_fragments": stitched, "merged_lines": merged}
    return out_lines, out_toc, meta


# -----------------------------
# Units
# -----------------------------


@dataclass
class UnitDraft:
    lines: List[str]
    marker: Optional[MarkerInfo]
    from_toc: bool


def guess_unit_type(first_line: str, marker: Optional[MarkerInfo], from_toc: bool, lang: str) -> str:
    if from_toc:
        return "other"
    if marker:
        if marker.marker_type in {"bullet", "alpha", "chinese_num"}:
            return "list-item"
        if marker.marker_type == "article":
            if marker.article_kind in {"章", "节"}:
                return "heading"
            return "paragraph"
        if marker.marker_type in {"roman"}:
            return "heading"
        if marker.marker_type == "decimal":
            # short tail without punctuation -> heading
            tail = re.sub(r"^\s*\d+(?:\s*\.\s*\d+){0,6}[\.)、]?\s*", "", normalize_text_light(first_line))
            tail = tail.strip()
            if 0 < len(tail) <= (60 if lang == "en" else 30) and not sentence_end_punct(tail):
                return "heading"
            return "paragraph"
        return "paragraph"
    return "paragraph"


def segment_units(lines: List[str], toc_flags: List[bool], lang: str, cfg: Config) -> List[UnitDraft]:
    units: List[UnitDraft] = []
    cur_lines: List[str] = []
    cur_marker: Optional[MarkerInfo] = None
    cur_toc: bool = False
    cur_strong: bool = False

    def flush() -> None:
        nonlocal cur_lines, cur_marker, cur_toc, cur_strong
        text = "\n".join([x for x in cur_lines if x.strip()])
        if len(text.strip()) >= cfg.min_unit_text_chars:
            units.append(UnitDraft(lines=cur_lines[:], marker=cur_marker, from_toc=cur_toc))
        cur_lines = []
        cur_marker = None
        cur_toc = False
        cur_strong = False

    for ln, is_toc in zip(lines, toc_flags):
        if not ln.strip():
            if cur_lines and not cur_strong:
                flush()
            else:
                if cur_lines and cur_strong:
                    cur_lines.append("")
            continue

        mk = detect_marker(ln)
        mk_is_strong = mk is not None and mk.strength == "strong"
        mk_is_weak = mk is not None and mk.strength == "weak"

        if mk_is_strong or mk_is_weak:
            if cur_lines:
                flush()
            cur_lines = [ln]
            cur_marker = mk
            cur_toc = is_toc
            cur_strong = mk_is_strong
        else:
            if not cur_lines:
                cur_lines = [ln]
                cur_marker = None
                cur_toc = is_toc
                cur_strong = False
            else:
                if is_toc != cur_toc:
                    flush()
                    cur_lines = [ln]
                    cur_marker = None
                    cur_toc = is_toc
                    cur_strong = False
                else:
                    cur_lines.append(ln)

    if cur_lines:
        flush()

    # soften fragmentation when too many weak markers
    weak_cnt = sum(1 for u in units if u.marker and u.marker.marker_type in {"bullet", "alpha", "chinese_num"})
    if units:
        weak_ratio = weak_cnt / len(units)
        if weak_ratio > cfg.max_weak_marker_ratio:
            softened: List[UnitDraft] = []
            for u in units:
                if u.marker and u.marker.marker_type in {"bullet", "alpha", "chinese_num"}:
                    if softened and softened[-1].marker is None and not softened[-1].from_toc:
                        softened[-1].lines.append(u.lines[0])
                        continue
                softened.append(u)
            units = softened

    return units


def choose_title_unit(units: List[Dict[str, Any]], lang: str) -> Optional[int]:
    candidates: List[Tuple[float, int]] = []
    for u in units:
        if u["unit_id"] == 0:
            continue
        if u["type"] == "list-item":
            continue
        txt = (u.get("text") or "").strip()
        if not txt:
            continue
        if u.get("num_prefix"):
            continue
        if len(txt) > 140:
            continue
        score = 0.0
        # keyword boost
        if re.search(r"\b(standard|specification|guideline|manual)\b", txt, re.I):
            score += 0.8
        if re.search(r"(规范|规程|标准|办法|条例|指南)", txt):
            score += 0.8
        # length
        if len(txt) <= 60:
            score += 2.0
        elif len(txt) <= 100:
            score += 1.0
        # case
        if lang == "en":
            letters = re.findall(r"[A-Za-z]", txt)
            if letters:
                upper = sum(1 for c in letters if c.isupper())
                if upper / len(letters) > 0.5:
                    score += 0.8
        if sentence_end_punct(txt):
            score -= 1.0
        # penalize TOC-like dots
        if re.search(r"\.{3,}\s*\d+\s*$", txt):
            score -= 2.0
        candidates.append((score, u["unit_id"]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score, best_id = candidates[0]
    if best_score >= 2.0:
        return best_id
    return None


# -----------------------------
# Tree building + validation
# -----------------------------


def compute_depth_hat_from_unit(u: Dict[str, Any]) -> Optional[int]:
    mk = u.get("marker_type")
    np = u.get("num_prefix")
    if not mk or not np:
        return None
    if mk == "decimal":
        m = re.match(r"^\s*(\d+(?:\.\d+)*)", normalize_text_light(np))
        if not m:
            return None
        parts = m.group(1).split(".")
        return max(1, len([p for p in parts if p != ""]))
    if mk == "article":
        m = re.match(r"^第.+?(章|节|条|款|项)", np)
        kind = m.group(1) if m else None
        return {"章": 1, "节": 2, "条": 3, "款": 4, "项": 5}.get(kind, 3)
    if mk == "roman":
        return 1
    return None


def attach_list_item(units: List[Dict[str, Any]], idx: int) -> int:
    for j in range(idx - 1, -1, -1):
        if units[j]["type"] == "paragraph":
            return units[j]["unit_id"]
    for j in range(idx - 1, -1, -1):
        if units[j]["type"] in {"heading", "title"}:
            return units[j]["unit_id"]
    return 0


def build_strict_tree(units: List[Dict[str, Any]]) -> None:
    assert units and units[0]["unit_id"] == 0
    units[0]["parent_id"] = None
    units[0]["level"] = 0

    stack: List[int] = [0]  # stack[level] = last unit_id at this level

    for i in range(1, len(units)):
        u = units[i]
        uid = u["unit_id"]

        if u.get("type") == "list-item":
            pid = attach_list_item(units, i)
            if not isinstance(pid, int) or pid < 0 or pid >= uid:
                pid = 0
            u["parent_id"] = pid
            u["level"] = units[pid]["level"] + 1
            continue

        desired = compute_depth_hat_from_unit(u)
        prev = units[i - 1]
        if desired is None:
            if prev.get("type") in {"heading", "title"}:
                desired = prev["level"] + 1
            elif prev.get("type") == "paragraph":
                desired = prev["level"]
            else:
                desired = 1

        try:
            desired = int(desired)
        except Exception:
            desired = 1
        if desired < 1:
            desired = 1
        # IMPORTANT: desired max is len(stack)
        if desired > len(stack):
            desired = len(stack)

        while len(stack) > desired:
            stack.pop()

        parent_id = stack[desired - 1]
        u["parent_id"] = parent_id
        u["level"] = units[parent_id]["level"] + 1

        lvl = u["level"]
        if lvl < 1:
            lvl = 1
        if lvl > len(stack):
            lvl = len(stack)
            u["level"] = units[parent_id]["level"] + 1

        stack = stack[:lvl]
        stack.append(uid)


def validate_doc(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(doc.get("doc_id"), str) or not doc["doc_id"]:
        errs.append("doc_id missing/invalid")
    if not isinstance(doc.get("doc_name"), str) or not doc["doc_name"]:
        errs.append("doc_name missing/invalid")
    if doc.get("language") not in ALLOWED_LANGUAGE:
        errs.append(f"language invalid: {doc.get('language')}")
    units = doc.get("units")
    if not isinstance(units, list) or not units:
        errs.append("units missing/empty")
        return False, errs

    seen = set()
    for i, u in enumerate(units):
        if u.get("unit_id") != i:
            errs.append(f"unit_id not sequential at index {i}: {u.get('unit_id')}")
        if u.get("unit_id") in seen:
            errs.append(f"duplicate unit_id: {u.get('unit_id')}")
        seen.add(u.get("unit_id"))
        if u.get("type") not in ALLOWED_TYPES:
            errs.append(f"invalid type at {i}: {u.get('type')}")
        if not isinstance(u.get("level"), int) or u["level"] < 0:
            errs.append(f"invalid level at {i}: {u.get('level')}")
        pid = u.get("parent_id")
        if pid is not None and (not isinstance(pid, int) or pid < 0):
            errs.append(f"invalid parent_id at {i}: {pid}")
        if u.get("num_prefix") is not None:
            t = u.get("text", "")
            np = u.get("num_prefix")
            if np and not re.match(r"^\s*" + re.escape(np), t):
                errs.append(f"num_prefix not at start for unit {i}")
        if u.get("marker_type") is not None and u["marker_type"] not in ALLOWED_MARKER_TYPES:
            errs.append(f"invalid marker_type for unit {i}: {u['marker_type']}")

    roots = [u for u in units if u.get("parent_id") is None]
    if len(roots) != 1:
        errs.append(f"root count != 1: {len(roots)}")
    else:
        if roots[0].get("unit_id") != 0:
            errs.append("root must be unit_id=0")
        if roots[0].get("level") != 0:
            errs.append("root level must be 0")

    for u in units[1:]:
        pid = u["parent_id"]
        if pid is None:
            errs.append(f"non-root has parent_id=null: unit {u['unit_id']}")
            continue
        if pid >= u["unit_id"]:
            errs.append(f"parent_id >= unit_id: unit {u['unit_id']} parent {pid}")
        if u["level"] != units[pid]["level"] + 1:
            errs.append(f"level != parent.level+1: unit {u['unit_id']} level {u['level']} parent {pid} level {units[pid]['level']}")

    reachable = {0}
    changed = True
    while changed:
        changed = False
        for u in units[1:]:
            if u["parent_id"] in reachable and u["unit_id"] not in reachable:
                reachable.add(u["unit_id"])
                changed = True
    if len(reachable) != len(units):
        errs.append(f"tree not connected: reachable {len(reachable)}/{len(units)}")

    return len(errs) == 0, errs


def _align_or_drop_prefix_fields(units: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, int]:
    fixed = 0
    dropped = 0
    for u in units:
        np = u.get("num_prefix")
        mt = u.get("marker_type")
        if not np or not mt:
            continue
        t = u.get("text") or ""
        if re.match(r"^\s*" + re.escape(np), t):
            continue
        # try normalize both sides and update np to match start token
        t_norm = normalize_text_light(t)
        np_norm = normalize_text_light(np)
        if re.match(r"^\s*" + re.escape(np_norm), t_norm):
            u["num_prefix"] = np_norm
            fixed += 1
            continue
        # last attempt: recompute marker from first line
        first = (t.splitlines()[0] if t else "")
        mk = detect_marker(first)
        if mk and mk.num_prefix and re.match(r"^\s*" + re.escape(mk.num_prefix), normalize_text_light(first)):
            u["num_prefix"] = mk.num_prefix
            u["marker_type"] = mk.marker_type
            fixed += 1
            continue
        # drop
        u.pop("num_prefix", None)
        u.pop("marker_type", None)
        dropped += 1
    if dropped:
        logger.warning("Dropped prefix fields for %d units (un-alignable)", dropped)
    return {"prefix_fixed": fixed, "prefix_dropped": dropped}


# -----------------------------
# End-to-end
# -----------------------------


def setup_logger(log_file: Path, level: str = "INFO") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_file.stem + "_" + str(os.getpid()))
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def build_doc_from_txt(txt_path: Path, out_json: Path, cfg: Config, log_dir: Path, sample_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logger = setup_logger(log_dir / (txt_path.stem + ".log"))
    logger.info("Start: %s", txt_path)

    pages = read_txt_pages(txt_path, cfg)
    flat_preview = " ".join([" ".join(p[:40]) for p in pages[:3]])
    lang = detect_language(flat_preview)

    lines, toc_flags = clean_pages(pages, cfg)
    lines2, toc2, merge_meta = merge_continuation_lines(lines, toc_flags, lang)
    unit_drafts = segment_units(lines2, toc2, lang, cfg)

    # Root text: first non-empty non-strong-marker line
    root_text = ""
    for x in lines2:
        if not x.strip():
            continue
        mk = detect_marker(x)
        if mk is None or mk.strength != "strong":
            root_text = x
            break
    if not root_text:
        for x in lines2:
            if x.strip():
                root_text = x
                break

    units: List[Dict[str, Any]] = [
        {
            "unit_id": 0,
            "text": root_text,
            "type": "other",
            "level": 0,
            "parent_id": None,
            "unit_hash": md5_text(root_text),
        }
    ]

    strong_marker_starts = 0
    weak_marker_starts = 0

    for d in unit_drafts:
        text = "\n".join([x for x in d.lines if x is not None]).strip()
        if len(text) < cfg.min_unit_text_chars:
            continue
        marker = d.marker
        if marker:
            if marker.strength == "strong":
                strong_marker_starts += 1
            else:
                weak_marker_starts += 1
        utype = guess_unit_type(d.lines[0], marker, d.from_toc, lang)
        unit: Dict[str, Any] = {
            "unit_id": len(units),
            "text": text,
            "type": utype,
            "level": 0,
            "parent_id": 0,
            "unit_hash": md5_text(text),
        }
        if marker:
            unit["num_prefix"] = marker.num_prefix
            unit["marker_type"] = marker.marker_type
        units.append(unit)

    strong_ratio = strong_marker_starts / max(1, len(units) - 1)
    weak_ratio = weak_marker_starts / max(1, len(units) - 1)

    # title
    if not any(u["type"] == "title" for u in units):
        tid = choose_title_unit(units, lang)
        if tid is None:
            # fallback: pick earliest plausible line
            for u in units[1:31]:
                t = (u.get("text") or "").strip()
                if 5 <= len(t) <= 80 and not u.get("num_prefix"):
                    tid = u["unit_id"]
                    break
        if tid is not None:
            units[tid]["type"] = "title"

    title_unit = next((u for u in units if u["type"] == "title"), None)
    doc_name = title_unit["text"].strip() if title_unit else "unknown_title"

    diagnostics: Dict[str, Any] = {
        "txt_path": str(txt_path),
        "pages": len(pages),
        "units": len(units),
        "strong_marker_starts": strong_marker_starts,
        "weak_marker_starts": weak_marker_starts,
        "strong_marker_ratio": round(strong_ratio, 6),
        "weak_marker_ratio": round(weak_ratio, 6),
        "language": lang,
        **merge_meta,
    }

    # tree
    if strong_ratio < cfg.min_strong_marker_ratio:
        diagnostics["mode"] = "weak_fallback"
        units[0]["parent_id"] = None
        units[0]["level"] = 0
        for i in range(1, len(units)):
            if units[i]["type"] == "list-item":
                pid = attach_list_item(units, i)
                units[i]["parent_id"] = pid
                units[i]["level"] = units[pid]["level"] + 1
            else:
                units[i]["parent_id"] = 0
                units[i]["level"] = 1
    else:
        diagnostics["mode"] = "marker_stack_tree"
        try:
            build_strict_tree(units)
        except Exception:
            logger.exception("build_strict_tree failed")
            raise

    # prefix alignment / dropping if needed
    prefix_stats = _align_or_drop_prefix_fields(units, logger)
    diagnostics.update(prefix_stats)

    doc = {
        "doc_id": txt_path.stem,
        "doc_name": doc_name if doc_name else "unknown_title",
        "language": lang if lang in ALLOWED_LANGUAGE else "other",
        "units": units,
    }

    ok, errs = validate_doc(doc)
    diagnostics["valid"] = ok
    diagnostics["errors"] = " | ".join(errs[:10])

    if not ok:
        # safe projection
        logger.warning("Validator failed, applying safe projection")
        units[0]["parent_id"] = None
        units[0]["level"] = 0
        for i in range(1, len(units)):
            if units[i]["type"] == "list-item":
                pid = attach_list_item(units, i)
                units[i]["parent_id"] = pid
                units[i]["level"] = units[pid]["level"] + 1
            else:
                units[i]["parent_id"] = 0
                units[i]["level"] = 1
        # drop all prefixes as last resort if still failing
        doc["units"] = units
        ok2, errs2 = validate_doc(doc)
        if not ok2:
            for u in units:
                u.pop("num_prefix", None)
                u.pop("marker_type", None)
            ok3, errs3 = validate_doc(doc)
            diagnostics["valid_after_drop_all_prefix"] = ok3
            diagnostics["errors_after_drop_all_prefix"] = " | ".join(errs3[:10])
        else:
            diagnostics["valid_after_fix"] = ok2
            diagnostics["errors_after_fix"] = " | ".join(errs2[:10])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), "utf-8")

    # sampling markdown
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        rnd = random.Random(cfg.sample_seed)
        cand = list(range(0, len(units)))
        picks = cand[:1] + rnd.sample(cand[1:], k=min(cfg.sample_n, max(0, len(cand) - 1))) if len(cand) > 1 else cand
        md_lines = [f"# Sample: {txt_path.stem}", ""]
        for uid in picks:
            u = units[uid]
            md_lines.append(
                f"## unit{u['unit_id']}  type={u['type']}  level={u['level']}  parent={u['parent_id']}  prefix={u.get('num_prefix')}"
            )
            md_lines.append("```")
            md_lines.append((u.get("text") or "").strip()[:1200])
            md_lines.append("```")
            md_lines.append("")
        (sample_dir / f"{txt_path.stem}.md").write_text("\n".join(md_lines), "utf-8")

    logger.info("Done: units=%d valid=%s", len(units), diagnostics.get("valid"))
    return doc, diagnostics


def run_batch(
    input_txt_dir: Path,
    output_json_dir: Path,
    log_dir: Path,
    cfg: Config,
    sample_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> None:
    txts = sorted(input_txt_dir.rglob("*.txt"))
    output_json_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)

    diag_rows: List[Dict[str, Any]] = []
    for idx, txt in enumerate(txts, start=1):
        out_json = output_json_dir / (txt.stem + ".json")
        if (not force) and out_json.exists() and out_json.stat().st_mtime >= txt.stat().st_mtime:
            diag_rows.append({"txt_path": str(txt), "skipped": True})
            continue
        try:
            _, diag = build_doc_from_txt(txt, out_json, cfg, log_dir, sample_dir)
            diag_rows.append(diag)
        except Exception as e:
            diag_rows.append({"txt_path": str(txt), "error": f"{type(e).__name__}: {e}", "valid": False})

    # diagnostics csv
    diag_csv = output_json_dir / "diagnostics_segment.csv"
    keys = set()
    for r in diag_rows:
        keys |= set(r.keys())
    keys = sorted(keys)
    with open(diag_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in diag_rows:
            w.writerow(r)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt_dir", required=True)
    ap.add_argument("--output_json_dir", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--sample_dir", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    run_batch(
        Path(args.input_txt_dir),
        Path(args.output_json_dir),
        Path(args.log_dir),
        cfg,
        sample_dir=Path(args.sample_dir) if args.sample_dir else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
