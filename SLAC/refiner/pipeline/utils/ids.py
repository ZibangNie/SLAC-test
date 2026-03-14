"""
ID utilities.

Main responsibility:
- stable doc_id / chunk_id / unit_id generation
- avoid filename-stem-only collisions
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


_INVALID_ID_CHARS_RE = re.compile(r"[^a-zA-Z0-9._:-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def normalize_text_for_id(text: str) -> str:
    """
    Normalize arbitrary text to a stable ASCII-ish id fragment.

    Steps:
    - NFKC normalization
    - trim
    - spaces/slashes to underscore
    - remove unsupported chars
    - collapse repeated underscores
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    s = s.replace("\\", "/")
    s = s.replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = _INVALID_ID_CHARS_RE.sub("_", s)
    s = _MULTI_UNDERSCORE_RE.sub("_", s)
    s = s.strip("._:-_")
    return s


def short_hash(text: str, length: int = 10) -> str:
    if length <= 0:
        raise ValueError("length must be > 0")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_doc_id_from_path(
    source_path: PathLike,
    *,
    root: Optional[PathLike] = None,
    prefix: Optional[str] = None,
    hash_len: int = 10,
) -> str:
    """
    Build a stable doc_id from file path.

    If root is provided and source_path is under root, use relative path;
    otherwise use resolved absolute path.

    Format:
        {prefix_}normalized_stem__{hash(relative_or_abs_posix)}
    """
    p = Path(source_path).expanduser().resolve()

    if root is not None:
        try:
            base = p.relative_to(Path(root).expanduser().resolve()).as_posix()
        except Exception:
            base = p.as_posix()
    else:
        base = p.as_posix()

    stem = normalize_text_for_id(p.stem) or "doc"
    suffix = short_hash(base.lower(), length=hash_len)

    if prefix:
        pref = normalize_text_for_id(prefix)
        if pref:
            return f"{pref}_{stem}__{suffix}"

    return f"{stem}__{suffix}"


def stable_doc_id(
    *,
    explicit_doc_id: Optional[str] = None,
    source_path: Optional[PathLike] = None,
    title_hint: Optional[str] = None,
    prefix: Optional[str] = None,
    hash_len: int = 10,
) -> str:
    """
    Priority:
    1. explicit_doc_id
    2. source_path
    3. title_hint
    """
    if explicit_doc_id:
        x = normalize_text_for_id(explicit_doc_id)
        if x:
            return x

    if source_path is not None:
        return stable_doc_id_from_path(
            source_path,
            root=None,
            prefix=prefix,
            hash_len=hash_len,
        )

    if title_hint:
        t = normalize_text_for_id(title_hint) or "doc"
        h = short_hash(title_hint, length=hash_len)
        if prefix:
            pref = normalize_text_for_id(prefix)
            if pref:
                return f"{pref}_{t}__{h}"
        return f"{t}__{h}"

    raise ValueError("Cannot build doc_id: provide explicit_doc_id, source_path, or title_hint")


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")
    return f"{doc_id}::chunk_{chunk_index:05d}"


def make_leaf_id(doc_id: str, leaf_index: int) -> str:
    if leaf_index < 0:
        raise ValueError("leaf_index must be >= 0")
    return f"{doc_id}::leaf_{leaf_index:05d}"


def make_unit_id(doc_id: str, unit_index: int) -> str:
    if unit_index < 0:
        raise ValueError("unit_index must be >= 0")
    return f"{doc_id}::unit_{unit_index:05d}"


def make_atom_id(doc_id: str, atom_index: int) -> str:
    if atom_index < 0:
        raise ValueError("atom_index must be >= 0")
    return f"{doc_id}::atom_{atom_index:05d}"


def make_span_chunk_id(doc_id: str, atom_start: int, atom_end: int) -> str:
    """
    Optional span-based id for debugging or deterministic re-derivation.
    """
    if atom_start < 0 or atom_end < 0 or atom_end < atom_start:
        raise ValueError(f"Invalid atom span: [{atom_start}, {atom_end})")
    return f"{doc_id}::a{atom_start:05d}_a{atom_end:05d}"


def make_query_id(query: str, *, prefix: str = "q", hash_len: int = 10) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("query must not be empty")
    return f"{normalize_text_for_id(prefix) or 'q'}_{short_hash(q, length=hash_len)}"


def make_candidate_id(query_id: str, rank: int) -> str:
    if rank < 0:
        raise ValueError("rank must be >= 0")
    return f"{query_id}::cand_{rank:04d}"


def sanitize_path_title_list(parts: list[str]) -> list[str]:
    """
    Normalize path/title list while preserving readability.
    """
    out: list[str] = []
    for p in parts:
        x = unicodedata.normalize("NFKC", str(p)).strip()
        if not x:
            continue
        x = re.sub(r"\s+", " ", x)
        out.append(x)
    return out


def infer_chunk_id(
    *,
    doc_id: str,
    chunk_index: Optional[int] = None,
    atom_start: Optional[int] = None,
    atom_end: Optional[int] = None,
    prefer_span: bool = False,
) -> str:
    """
    Default:
    - use index-based stable id for production
    - span-based id only when explicitly preferred or index missing
    """
    if prefer_span:
        if atom_start is None or atom_end is None:
            raise ValueError("prefer_span=True requires atom_start and atom_end")
        return make_span_chunk_id(doc_id, atom_start, atom_end)

    if chunk_index is not None:
        return make_chunk_id(doc_id, chunk_index)

    if atom_start is not None and atom_end is not None:
        return make_span_chunk_id(doc_id, atom_start, atom_end)

    raise ValueError("Cannot infer chunk_id: provide chunk_index or atom span")


def is_valid_doc_id(doc_id: str) -> bool:
    if not isinstance(doc_id, str) or not doc_id.strip():
        return False
    return normalize_text_for_id(doc_id) == doc_id


def parse_chunk_index(chunk_id: str) -> Optional[int]:
    m = re.fullmatch(r".+::chunk_(\d{5})", chunk_id or "")
    if not m:
        return None
    return int(m.group(1))