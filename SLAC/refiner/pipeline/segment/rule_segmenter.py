"""
Rule-based segmenter for chunk0 generation.

Main responsibility:
- segment normalized text into units/tree/chunk0
- reuse and refactor logic from SLAC/segmenter/txt_to_json_segmenter.py
- fix false heading detection and stabilize chunk0 output
"""
from __future__ import annotations

import copy
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..utils.ids import stable_doc_id


PathLike = Union[str, Path]


ALLOWED_LANGUAGE = {"zh", "en", "other"}
ALLOWED_TYPES = {"title", "heading", "paragraph", "list-item", "other"}
ALLOWED_MARKER_TYPES = {
    "decimal",
    "roman",
    "chinese_num",
    "article",
    "alpha",
    "bullet",
    "unknown",
}

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
class RuleSegmenterConfig:
    # Text / line
    page_break_char: str = "\f"
    max_merge_line_len: int = 220
    min_unit_text_chars: int = 3

    # Marker / tree
    min_strong_marker_ratio: float = 0.02
    max_weak_marker_ratio: float = 0.45
    allow_single_number_heading: bool = True
    max_single_heading_number: int = 199

    # Title selection
    title_scan_limit: int = 30

    # TOC / line heuristics
    enable_toc_detection: bool = True
    toc_early_line_window: int = 200
    toc_dot_leader_ratio: float = 0.22
    toc_trailing_page_ratio: float = 0.22

    # Guards
    table_like_min_separators: int = 2
    numeric_heavy_token_ratio: float = 0.60
    strict_validate: bool = True


@dataclass
class MarkerInfo:
    num_prefix: str
    marker_type: str
    strength: str  # strong | weak
    depth_hat: Optional[int] = None
    article_kind: Optional[str] = None


@dataclass
class UnitDraft:
    lines: List[str]
    marker: Optional[MarkerInfo]
    from_toc: bool


# -----------------------------
# Basic text utils
# -----------------------------


def md5_text(s: str) -> str:
    return "md5:" + hashlib.md5((s or "").encode("utf-8")).hexdigest()


def normalize_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def normalize_text_light(s: str) -> str:
    """
    Light normalization for marker stability.
    Keep this intentionally less aggressive than preprocess.normalize_text.
    """
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\u3000", " ")
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = s.replace("．", ".").replace("。", ".").replace("·", ".").replace("‧", ".").replace("｡", ".")
    s = s.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    s = s.replace("—", "-").replace("–", "-").replace("－", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


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


def sentence_end_punct(s: str) -> bool:
    s = (s or "").rstrip()
    return bool(re.search(r"[.!?;:。！？；：]$", s))


def split_text_to_lines(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if "\f" in text:
        pages = text.split("\f")
        lines: List[str] = []
        for i, pg in enumerate(pages):
            lines.extend(pg.split("\n"))
            if i != len(pages) - 1:
                lines.append("")
        return lines
    return text.split("\n")


def _strip_line(line: str) -> str:
    return (line or "").strip()


def _is_blank(line: str) -> bool:
    return not _strip_line(line)


def is_page_number_line(s: str) -> bool:
    x = normalize_spaces(s)
    if not x:
        return True
    if re.fullmatch(r"\d{1,4}", x):
        return True
    if re.fullmatch(r"[-–—]{1,3}\s*\d{1,4}\s*[-–—]{1,3}", x):
        return True
    if re.fullmatch(r"(page|p\.)\s*\d+\s*(/|of)\s*\d+", x, re.I):
        return True
    if re.fullmatch(r"第?\s*\d+\s*页(\s*/\s*共?\s*\d+\s*页)?", x):
        return True
    return False


def is_decorative_line(s: str) -> bool:
    x = normalize_spaces(s)
    if not x:
        return True
    if re.fullmatch(r"[-=–—_#~*·•\.]{5,}", x):
        return True
    compact = x.replace(" ", "")
    if len(compact) >= 5 and len(set(compact)) == 1 and not compact[0].isalnum():
        return True
    return False


def likely_toc_lines(lines: List[str], cfg: RuleSegmenterConfig) -> List[bool]:
    """
    Line-level TOC estimation for cleaned text input.
    We no longer rely on page boundaries here.
    """
    n = len(lines)
    flags = [False] * n
    if not cfg.enable_toc_detection or n == 0:
        return flags

    window = min(cfg.toc_early_line_window, n)
    early = [normalize_spaces(x) for x in lines[:window] if normalize_spaces(x)]
    if not early:
        return flags

    joined = " ".join(early[:100]).lower()
    has_keyword = any(k in joined for k in TOC_KEYWORDS)

    total = 0
    dot_leader = 0
    trailing_page = 0
    short_lines = 0

    for ln in early:
        total += 1
        if len(ln) <= 40:
            short_lines += 1
        if re.search(r"\.{3,}\s*\d+\s*$", ln):
            dot_leader += 1
        if re.search(r"\s\d+\s*$", ln) and len(ln) <= 80:
            trailing_page += 1

    toc_like = False
    if total > 0:
        if has_keyword:
            toc_like = True
        elif (dot_leader / total >= cfg.toc_dot_leader_ratio) and (trailing_page / total >= cfg.toc_trailing_page_ratio):
            toc_like = True
        elif (short_lines / total >= 0.72) and (trailing_page / total >= 0.35):
            toc_like = True

    if toc_like:
        for i in range(window):
            x = normalize_spaces(lines[i])
            if not x:
                flags[i] = True
            elif (
                any(k in x.lower() for k in TOC_KEYWORDS)
                or re.search(r"\.{3,}\s*\d+\s*$", x)
                or (re.search(r"\s\d+\s*$", x) and len(x) <= 100)
            ):
                flags[i] = True
    return flags


# -----------------------------
# Marker detection
# -----------------------------


_ROMAN_RE = re.compile(r"^(?=[IVXLCDM]+\b)[IVXLCDM]+", re.I)

_RE_BULLET = re.compile(r"^\s*([•\-\*\u2022\u25CF\u25CB])\s+(.+)?$")
_RE_PAREN_NUM = re.compile(r"^\s*[\(（]\s*([0-9]{1,4})\s*[\)）]\s+(.+)?$")
_RE_PAREN_ALPHA = re.compile(r"^\s*[\(（]\s*([a-zA-Z])\s*[\)）]\s+(.+)?$")
_RE_ALPHA_DOT = re.compile(r"^\s*([a-zA-Z])[\).、]\s+(.+)?$")

_RE_CN_LIST = re.compile(r"^\s*([一二三四五六七八九十百千万零〇两]+)[、\)]\s+(.+)?$")
_RE_CN_PAREN = re.compile(r"^\s*[\(（]([一二三四五六七八九十百千万零〇两]+)[\)）]\s+(.+)?$")

_RE_ARTICLE = re.compile(r"^\s*第([一二三四五六七八九十百千万零〇两0-9]+)(章|节|条|款|项)\b(.*)$")
_RE_PART = re.compile(r"^\s*(Part|Section|Annex|Appendix|Chapter)\s+([A-Z0-9IVX]+)\b(.*)$", re.I)

# FIX:
# old script allowed {0,6}, which over-matched single-number lines.
# here we require at least one dotted child segment: {1,6}
_RE_DECIMAL_DOTTED = re.compile(
    r"^\s*(\d{1,4}(?:\s*\.\s*\d{1,4}){1,6})([\.)、]?)\s+(.+)?$"
)

_RE_DECIMAL_SINGLE = re.compile(r"^\s*(\d{1,3})(\)|\.|、)?\s+(.+)?$")


_MEASURE_UNITS = {
    "kn", "n", "mn", "mm", "cm", "m", "km",
    "mpa", "gpa", "pa", "hz", "khz", "mhz", "ghz",
    "kg", "g", "t", "s", "ms", "min", "h", "°c", "℃", "%", "‰", "db", "deg",
}


def _normalize_decimal_prefix(prefix: str) -> str:
    p = prefix
    p = re.sub(r"\s*\.\s*", ".", p)
    return p


def cn_to_int(s: str) -> Optional[int]:
    cn_num = {
        "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
        "千": 1000, "万": 10000,
    }
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
        if ch not in cn_num:
            return None
        val = cn_num[ch]
        if val >= 10:
            if num == 0:
                num = 1
            total += num * val
            num = 0
        else:
            num = val
    total += num
    return total if total > 0 else None


def _looks_like_measurement_tail(tail: str) -> bool:
    t = (tail or "").strip().lower()
    if not t:
        return True
    for u in _MEASURE_UNITS:
        if t.startswith(u):
            return True
        if re.match(r"^\d+\s*" + re.escape(u) + r"\b", t):
            return True
    if re.match(r"^[\d\.\-]+\s*[a-zA-Z%℃]+\b", t):
        return True
    return False


def _looks_table_like(line: str, cfg: RuleSegmenterConfig) -> bool:
    s = normalize_spaces(line)
    if not s:
        return False

    if "|" in s:
        return True
    if "\t" in line:
        return True

    sep_count = s.count("  ")
    if sep_count >= cfg.table_like_min_separators:
        return True

    tokens = re.split(r"\s+", s)
    if len(tokens) >= 4:
        numeric_like = 0
        for tok in tokens:
            tok2 = tok.strip(",;:()[]")
            if re.fullmatch(r"[\d\.\-/%]+", tok2):
                numeric_like += 1
            elif re.fullmatch(r"[\d\.\-/%]+[A-Za-z%℃]+", tok2):
                numeric_like += 1
        if numeric_like / max(len(tokens), 1) >= cfg.numeric_heavy_token_ratio:
            return True

    return False


def detect_marker(line: str, cfg: Optional[RuleSegmenterConfig] = None, *, allow: bool = True) -> Optional[MarkerInfo]:
    cfg = cfg or RuleSegmenterConfig()
    if not allow:
        return None

    s0 = normalize_text_light((line or "").rstrip("\n"))
    s = normalize_spaces(s0)
    if not s:
        return None

    if _looks_table_like(s, cfg):
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
        sym = s.strip()[1:2]
        return MarkerInfo(num_prefix=f"{m.group(1)}{sym}", marker_type="alpha", strength="weak")

    m = _RE_CN_PAREN.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"({m.group(1)})", marker_type="chinese_num", strength="weak")

    m = _RE_CN_LIST.match(s0)
    if m:
        return MarkerInfo(num_prefix=f"{m.group(1)}、", marker_type="chinese_num", strength="weak")

    m = _RE_ARTICLE.match(s0)
    if m:
        num_raw, kind = m.group(1), m.group(2)
        _ = cn_to_int(num_raw)
        kind_depth = {"章": 1, "节": 2, "条": 3, "款": 4, "项": 5}.get(kind, 3)
        return MarkerInfo(
            num_prefix=f"第{num_raw}{kind}",
            marker_type="article",
            strength="strong",
            depth_hat=kind_depth,
            article_kind=kind,
        )

    m = _RE_PART.match(s0)
    if m:
        head, key = m.group(1), m.group(2)
        head_low = head.lower()
        depth = 1 if head_low in {"part", "chapter", "annex", "appendix"} else 2
        mt = "roman" if _ROMAN_RE.match(key) else "unknown"
        return MarkerInfo(
            num_prefix=f"{head} {key}",
            marker_type=mt,
            strength="strong",
            depth_hat=depth,
        )

    m = _RE_DECIMAL_DOTTED.match(s0)
    if m:
        raw = _normalize_decimal_prefix(m.group(1))
        punct = m.group(2) or ""
        tail = (m.group(3) or "").strip()

        # extra guard for things like "3.5 mm ..." or numeric-heavy rows
        if _looks_like_measurement_tail(tail):
            return None

        parts = [p for p in raw.split(".") if p != ""]
        depth = max(1, len(parts))
        return MarkerInfo(
            num_prefix=raw + punct,
            marker_type="decimal",
            strength="strong",
            depth_hat=depth,
        )

    if cfg.allow_single_number_heading:
        m = _RE_DECIMAL_SINGLE.match(s0)
        if m:
            num = m.group(1)
            punct = m.group(2) or ""
            tail = (m.group(3) or "").strip()

            if _looks_like_measurement_tail(tail):
                return None

            try:
                if int(num) > cfg.max_single_heading_number:
                    return None
            except Exception:
                return None

            if not re.search(r"[A-Za-z\u4e00-\u9fff]", tail):
                return None

            if _looks_table_like(tail, cfg):
                return None

            return MarkerInfo(
                num_prefix=num + punct,
                marker_type="decimal",
                strength="strong",
                depth_hat=1,
            )

    s_norm = normalize_spaces(s0)
    rm = _ROMAN_RE.match(s_norm)
    if rm and re.match(r"^[IVXLCDM]+[\.)]\s+", s_norm, re.I):
        key = rm.group(0)
        return MarkerInfo(num_prefix=key, marker_type="roman", strength="strong", depth_hat=1)

    return None


# -----------------------------
# Continuation merge
# -----------------------------


def should_merge_lines(curr: str, nxt: str, lang_hint: str, *, nxt_has_marker: bool, cfg: RuleSegmenterConfig) -> bool:
    if not curr or not nxt:
        return False
    if nxt_has_marker:
        return False
    if len(curr) > cfg.max_merge_line_len:
        return False
    if sentence_end_punct(curr):
        return False

    # hyphenation in English
    if lang_hint == "en" and re.search(r"[A-Za-z]-$", curr.strip()):
        return True

    # avoid merging into all-caps / short candidate heading
    if lang_hint == "en":
        if len(nxt) <= 60 and re.fullmatch(r"[A-Z0-9][A-Z0-9 \-:;,\.\(\)]+", nxt.strip()):
            return False

    return True


def join_with_space(a: str, b: str, lang: str) -> str:
    if lang == "zh" or has_cjk(a) or has_cjk(b):
        return a.rstrip() + b.lstrip()
    if lang == "en" and re.search(r"[A-Za-z]-$", a.rstrip()):
        return a.rstrip()[:-1] + b.lstrip()
    return a.rstrip() + " " + b.lstrip()


def try_stitch_marker_fragment(prev: str, curr: str) -> Optional[str]:
    """
    Fix:
      prev = '3.'
      curr = '1.2 Title'
    -> '3.1.2 Title'
    """
    p = normalize_spaces(normalize_text_light(prev))
    c = normalize_spaces(normalize_text_light(curr))
    if re.fullmatch(r"\d{1,4}\.", p) and re.match(r"^\d{1,4}(?:\s*\.\s*\d{1,4})+\b", c):
        return p[:-1] + "." + c
    return None


def merge_continuation_lines(
    lines: List[str],
    toc_flags: List[bool],
    lang: str,
    cfg: RuleSegmenterConfig,
) -> Tuple[List[str], List[bool], Dict[str, Any]]:
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

        if out_lines and out_lines[-1] and not out_toc[-1] and curr_toc == out_toc[-1]:
            stitched_line = try_stitch_marker_fragment(out_lines[-1], curr)
            if stitched_line is not None:
                out_lines[-1] = stitched_line
                stitched += 1
                i += 1
                continue

        mk_curr = detect_marker(curr, cfg)
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

            mk_nxt = detect_marker(nxt, cfg)
            nxt_has_marker = mk_nxt is not None

            if curr_has_marker or nxt_has_marker:
                break
            if not should_merge_lines(acc, nxt, lang, nxt_has_marker=nxt_has_marker, cfg=cfg):
                break

            acc = join_with_space(acc, nxt, lang)
            merged += 1
            j += 1

        out_lines.append(acc)
        out_toc.append(curr_toc)
        i = j + 1

    meta = {
        "stitched_marker_fragments": stitched,
        "merged_lines": merged,
    }
    return out_lines, out_toc, meta


# -----------------------------
# Units + typing
# -----------------------------


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
            tail = re.sub(r"^\s*\d+(?:\s*\.\s*\d+){0,6}[\.)、]?\s*", "", normalize_text_light(first_line)).strip()
            if 0 < len(tail) <= (60 if lang == "en" else 30) and not sentence_end_punct(tail):
                return "heading"
            return "paragraph"

        return "paragraph"

    return "paragraph"


def segment_units(lines: List[str], toc_flags: List[bool], lang: str, cfg: RuleSegmenterConfig) -> List[UnitDraft]:
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

        mk = detect_marker(ln, cfg)
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
    weak_cnt = sum(
        1
        for u in units
        if u.marker and u.marker.marker_type in {"bullet", "alpha", "chinese_num"}
    )
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

        if re.search(r"\b(standard|specification|guideline|manual|code|regulation)\b", txt, re.I):
            score += 0.8
        if re.search(r"(规范|规程|标准|办法|条例|指南|手册|细则)", txt):
            score += 0.8

        if len(txt) <= 60:
            score += 2.0
        elif len(txt) <= 100:
            score += 1.0

        if lang == "en":
            letters = re.findall(r"[A-Za-z]", txt)
            if letters:
                upper = sum(1 for c in letters if c.isupper())
                if upper / len(letters) > 0.5:
                    score += 0.8

        if sentence_end_punct(txt):
            score -= 1.0

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
    stack: List[int] = [0]

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
            errs.append(
                f"level != parent.level+1: unit {u['unit_id']} level {u['level']} "
                f"parent {pid} level {units[pid]['level']}"
            )

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


def _align_or_drop_prefix_fields(units: List[Dict[str, Any]], logger: Optional[logging.Logger] = None) -> Dict[str, int]:
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

        t_norm = normalize_text_light(t)
        np_norm = normalize_text_light(np)
        if re.match(r"^\s*" + re.escape(np_norm), t_norm):
            u["num_prefix"] = np_norm
            fixed += 1
            continue

        first = t.splitlines()[0] if t else ""
        mk = detect_marker(first)
        if mk and mk.num_prefix and re.match(r"^\s*" + re.escape(mk.num_prefix), normalize_text_light(first)):
            u["num_prefix"] = mk.num_prefix
            u["marker_type"] = mk.marker_type
            fixed += 1
            continue

        u.pop("num_prefix", None)
        u.pop("marker_type", None)
        dropped += 1

    if dropped and logger is not None:
        logger.warning("Dropped prefix fields for %d units (un-alignable)", dropped)

    return {"prefix_fixed": fixed, "prefix_dropped": dropped}


# -----------------------------
# End-to-end
# -----------------------------


def _infer_root_text(lines: List[str], cfg: RuleSegmenterConfig) -> str:
    for x in lines:
        if not x.strip():
            continue
        mk = detect_marker(x, cfg)
        if mk is None or mk.strength != "strong":
            return x
    for x in lines:
        if x.strip():
            return x
    return "unknown_root"


def _build_units_from_drafts(
    unit_drafts: List[UnitDraft],
    lang: str,
    cfg: RuleSegmenterConfig,
) -> Tuple[List[Dict[str, Any]], int, int]:
    units: List[Dict[str, Any]] = []
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

        first_line = d.lines[0] if d.lines else text
        utype = guess_unit_type(first_line, marker, d.from_toc, lang)

        unit: Dict[str, Any] = {
            "unit_id": -1,
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

    return units, strong_marker_starts, weak_marker_starts


def segment_text_to_doc(
    text: str,
    *,
    doc_id: Optional[str] = None,
    source_path: Optional[PathLike] = None,
    source_type: str = "txt",
    lang: Optional[str] = None,
    cfg: Optional[RuleSegmenterConfig] = None,
    logger: Optional[logging.Logger] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main API:
    cleaned text -> structure doc JSON + diagnostics

    Returns:
        doc, diagnostics
    """
    cfg = cfg or RuleSegmenterConfig()
    raw_lines = split_text_to_lines(text)

    # keep line-level normalization local and light
    lines = [normalize_spaces(normalize_text_light(x)) for x in raw_lines]
    # preserve blanks; later logic uses them for flushing
    lines = [x if x else "" for x in lines]

    toc_flags = likely_toc_lines(lines, cfg) if cfg.enable_toc_detection else [False] * len(lines)

    flat_preview = " ".join([x for x in lines[:120] if x.strip()])
    final_lang = lang or detect_language(flat_preview)

    lines2, toc2, merge_meta = merge_continuation_lines(lines, toc_flags, final_lang, cfg)
    unit_drafts = segment_units(lines2, toc2, final_lang, cfg)

    root_text = _infer_root_text(lines2, cfg)

    content_units, strong_marker_starts, weak_marker_starts = _build_units_from_drafts(
        unit_drafts=unit_drafts,
        lang=final_lang,
        cfg=cfg,
    )

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

    for u in content_units:
        u2 = copy.deepcopy(u)
        u2["unit_id"] = len(units)
        units.append(u2)

    strong_ratio = strong_marker_starts / max(1, len(units) - 1)
    weak_ratio = weak_marker_starts / max(1, len(units) - 1)

    if not any(u["type"] == "title" for u in units):
        tid = choose_title_unit(units, final_lang)
        if tid is None:
            for u in units[1 : 1 + cfg.title_scan_limit]:
                t = (u.get("text") or "").strip()
                if 5 <= len(t) <= 80 and not u.get("num_prefix"):
                    tid = u["unit_id"]
                    break
        if tid is not None:
            units[tid]["type"] = "title"

    title_unit = next((u for u in units if u["type"] == "title"), None)
    doc_name = title_unit["text"].strip() if title_unit else "unknown_title"

    diagnostics: Dict[str, Any] = {
        "num_input_lines": len(lines),
        "num_lines_after_merge": len(lines2),
        "num_unit_drafts": len(unit_drafts),
        "units": len(units),
        "strong_marker_starts": strong_marker_starts,
        "weak_marker_starts": weak_marker_starts,
        "strong_marker_ratio": round(strong_ratio, 6),
        "weak_marker_ratio": round(weak_ratio, 6),
        "language": final_lang,
        **merge_meta,
    }

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
        build_strict_tree(units)

    prefix_stats = _align_or_drop_prefix_fields(units, logger=logger)
    diagnostics.update(prefix_stats)

    final_doc_id = stable_doc_id(
        explicit_doc_id=doc_id,
        source_path=source_path,
        title_hint=doc_name if doc_name and doc_name != "unknown_title" else root_text,
    )

    doc: Dict[str, Any] = {
        "doc_id": final_doc_id,
        "doc_name": doc_name if doc_name else "unknown_title",
        "language": final_lang if final_lang in ALLOWED_LANGUAGE else "other",
        "units": units,
        "meta": {
            "source_path": str(source_path) if source_path is not None else None,
            "source_type": source_type,
            "diagnostics": diagnostics,
        },
    }

    if extra_meta:
        doc["meta"].update(extra_meta)

    ok, errs = validate_doc(doc)
    diagnostics["valid"] = ok
    diagnostics["errors"] = " | ".join(errs[:10])

    if not ok:
        if logger is not None:
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

        ok2, errs2 = validate_doc(doc)
        diagnostics["valid_after_safe_projection"] = ok2
        diagnostics["errors_after_safe_projection"] = " | ".join(errs2[:10])

        if not ok2:
            for u in units:
                u.pop("num_prefix", None)
                u.pop("marker_type", None)

            ok3, errs3 = validate_doc(doc)
            diagnostics["valid_after_drop_all_prefix"] = ok3
            diagnostics["errors_after_drop_all_prefix"] = " | ".join(errs3[:10])

            if cfg.strict_validate and not ok3:
                raise ValueError(f"rule_segmenter validation failed: {errs3[:10]}")

    return doc, diagnostics


def segment_document_record(
    record: Dict[str, Any],
    *,
    src_field: str = "cleaned_text",
    cfg: Optional[RuleSegmenterConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Pipeline-friendly wrapper:
    document record -> structure doc record
    """
    if src_field not in record:
        raise KeyError(f"Document record missing field: {src_field}")

    text = record[src_field]
    doc, diagnostics = segment_text_to_doc(
        text=text,
        doc_id=record.get("doc_id"),
        source_path=record.get("source_path"),
        source_type=record.get("source_type", "txt"),
        lang=record.get("language"),
        cfg=cfg,
        logger=logger,
        extra_meta={"upstream_record_meta": record.get("meta", {})},
    )

    return {
        "doc_id": doc["doc_id"],
        "source_path": record.get("source_path"),
        "source_type": record.get("source_type", "txt"),
        "structure_doc": doc,
        "diagnostics": diagnostics,
    }