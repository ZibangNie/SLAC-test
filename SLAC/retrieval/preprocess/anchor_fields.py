from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from SLAC.retrieval.schemas.records import ChunkRecord, LeafRecord, AnchorRecord
from SLAC.retrieval.utils.text_utils import (
    estimate_token_count,
    extract_number_signature,
    join_path,
    normalize_for_anchor_match,
    normalize_text_basic,
)


_TITLE_HINT_RE = re.compile(
    r"^(?:"
    r"(?:第\s*[0-9一二三四五六七八九十百千]+\s*[章节条款])|"
    r"(?:附录\s*[A-Za-z0-9]+)|"
    r"(?:[A-Za-z]?\d+(?:\.\d+){0,6})|"
    r"(?:scope|definitions?|general requirements?|terminology|terms?)"
    r")",
    flags=re.IGNORECASE,
)

_SHORT_TITLE_RE = re.compile(r"^[^\n]{1,80}$")


def build_path_text(path: List[str], joiner: str = " > ") -> str:
    return join_path(path, joiner=joiner)


def detect_title_like(text: str) -> bool:
    x = normalize_text_basic(text, keep_newlines=True)
    first_line = x.split("\n", 1)[0].strip() if x else ""
    if not first_line:
        return False

    if _TITLE_HINT_RE.search(first_line):
        return True

    # 单行短句且没有强终止符，更像标题/条目头
    if _SHORT_TITLE_RE.match(first_line) and not re.search(r"[。！？.!?]$", first_line):
        return True

    return False


def choose_primary_number_signature(text: str, path: List[str]) -> Optional[str]:
    hits = extract_number_signature(text)
    if hits:
        return hits[0]
    for p in reversed(path):
        p_hits = extract_number_signature(str(p))
        if p_hits:
            return p_hits[0]
    return None


def build_anchor_text(
    text: str,
    path: List[str],
    number_signature: Optional[str] = None,
    joiner: str = " > ",
) -> str:
    path_text = build_path_text(path, joiner=joiner)
    first_line = normalize_text_basic(text, keep_newlines=True).split("\n", 1)[0].strip()
    pieces = []

    if number_signature:
        pieces.append(number_signature)
    if first_line:
        pieces.append(first_line)
    if path_text:
        pieces.append(path_text)

    return " | ".join(pieces).strip()


def enrich_chunk_record(
    chunk: ChunkRecord,
    joiner: str = " > ",
) -> ChunkRecord:
    chunk.text = normalize_text_basic(chunk.text, keep_newlines=True)
    chunk.path_text = build_path_text(chunk.path, joiner=joiner)
    chunk.number_signature = choose_primary_number_signature(chunk.text, chunk.path)
    chunk.is_title_like = detect_title_like(chunk.text)
    chunk.anchor_text = build_anchor_text(
        chunk.text,
        chunk.path,
        number_signature=chunk.number_signature,
        joiner=joiner,
    )
    if chunk.token_est is None:
        chunk.token_est = estimate_token_count(chunk.text)
    return chunk


def enrich_leaf_record(
    leaf: LeafRecord,
    owner_chunk: Optional[ChunkRecord] = None,
    joiner: str = " > ",
) -> LeafRecord:
    leaf.text = normalize_text_basic(leaf.text, keep_newlines=True)
    leaf.path_text = build_path_text(leaf.path, joiner=joiner)

    owner_anchor = None
    if owner_chunk is not None:
        owner_anchor = owner_chunk.anchor_text or build_anchor_text(
            owner_chunk.text,
            owner_chunk.path,
            number_signature=owner_chunk.number_signature,
            joiner=joiner,
        )

    leaf.owner_chunk_anchor = owner_anchor
    leaf.number_signature = choose_primary_number_signature(leaf.text, leaf.path)
    leaf.is_title_like = detect_title_like(leaf.text)
    leaf.anchor_text = build_anchor_text(
        leaf.text,
        leaf.path,
        number_signature=leaf.number_signature,
        joiner=joiner,
    )

    if leaf.token_est is None:
        leaf.token_est = estimate_token_count(leaf.text)
    return leaf


def make_anchor_record_from_chunk(chunk: ChunkRecord) -> AnchorRecord:
    return AnchorRecord(
        object_id=chunk.chunk_id,
        object_type="chunk",
        doc_id=chunk.doc_id,
        path=chunk.path,
        path_text=chunk.path_text or "",
        number_signature=chunk.number_signature,
        anchor_text=chunk.anchor_text or "",
        is_title_like=bool(chunk.is_title_like),
        indent_level=chunk.indent_level,
        text_norm=normalize_for_anchor_match(chunk.text),
        meta={"chunk_index": chunk.chunk_index},
    )


def make_anchor_record_from_leaf(leaf: LeafRecord) -> AnchorRecord:
    return AnchorRecord(
        object_id=leaf.leaf_id,
        object_type="leaf",
        doc_id=leaf.doc_id,
        path=leaf.path,
        path_text=leaf.path_text or "",
        number_signature=leaf.number_signature,
        anchor_text=leaf.anchor_text or "",
        is_title_like=bool(leaf.is_title_like),
        indent_level=leaf.indent_level,
        text_norm=normalize_for_anchor_match(leaf.text),
        meta={"owner_chunk_id": leaf.owner_chunk_id, "leaf_index": leaf.leaf_index},
    )