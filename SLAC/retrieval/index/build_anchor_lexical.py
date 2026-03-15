from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from SLAC.retrieval.dataio.writers import write_json
from SLAC.retrieval.schemas.records import AnchorRecord
from SLAC.retrieval.utils.text_utils import normalize_for_anchor_match, simple_word_tokenize


def compose_anchor_doc_text(anchor: AnchorRecord) -> str:
    parts = [
        anchor.number_signature or "",
        anchor.path_text or "",
        anchor.anchor_text or "",
        anchor.text_norm or "",
    ]
    return " | ".join([p.strip() for p in parts if str(p).strip()])


def tokenize_anchor_doc(anchor: AnchorRecord) -> List[str]:
    text = compose_anchor_doc_text(anchor)
    tokens = simple_word_tokenize(text)

    # 对标题型和编号型内容给予轻微重复增益
    if anchor.number_signature:
        tokens.extend(simple_word_tokenize(anchor.number_signature))
    if anchor.is_title_like:
        tokens.extend(simple_word_tokenize(anchor.path_text))
    return tokens


def build_anchor_postings(
    anchors: List[AnchorRecord],
) -> Tuple[Dict[str, List[dict]], Dict[str, int], Dict[str, int], float]:
    postings: Dict[str, List[dict]] = defaultdict(list)
    doc_freq: Dict[str, int] = defaultdict(int)
    doc_lengths: Dict[str, int] = {}
    total_len = 0

    for anchor in anchors:
        object_id = anchor.object_id
        tokens = tokenize_anchor_doc(anchor)
        counts = Counter(tokens)
        doc_len = sum(counts.values())

        doc_lengths[object_id] = doc_len
        total_len += doc_len

        for token, tf in counts.items():
            postings[token].append(
                {
                    "object_id": object_id,
                    "object_type": anchor.object_type,
                    "doc_id": anchor.doc_id,
                    "tf": int(tf),
                }
            )
            doc_freq[token] += 1

    avgdl = (total_len / len(anchors)) if anchors else 0.0
    return dict(postings), dict(doc_freq), doc_lengths, avgdl


def build_anchor_lexical_index(
    anchors: List[AnchorRecord],
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    postings, doc_freq, doc_lengths, avgdl = build_anchor_postings(anchors)

    anchor_meta = {
        "num_docs": len(anchors),
        "avgdl": avgdl,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "doc_lengths": doc_lengths,
        "doc_freq": doc_freq,
        "object_meta": {
            a.object_id: {
                "object_type": a.object_type,
                "doc_id": a.doc_id,
                "path_text": a.path_text,
                "number_signature": a.number_signature,
                "anchor_text": a.anchor_text,
                "is_title_like": a.is_title_like,
            }
            for a in anchors
        },
    }

    write_json(output_dir / "postings.json", postings)
    write_json(output_dir / "meta.json", anchor_meta)

    return {
        "num_items": len(anchors),
        "num_terms": len(postings),
        "avgdl": avgdl,
        "output_dir": str(output_dir),
    }