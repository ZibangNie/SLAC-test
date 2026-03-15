from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from SLAC.retrieval.utils.text_utils import normalize_for_anchor_match, simple_word_tokenize


@dataclass
class AnchorHit:
    object_id: str
    object_type: str
    doc_id: str
    score: float
    rank: int
    source_view: str
    query_variant: str
    exact_match: bool = False


_RE_CN_ANCHOR = re.compile(r"第\s*[0-9一二三四五六七八九十百千]+\s*[章节条款]")
_RE_APPENDIX = re.compile(r"(附录\s*[A-Za-z0-9]+|appendix\s+[A-Za-z0-9]+)", flags=re.IGNORECASE)
_RE_DOTTED = re.compile(r"\b\d+(?:\.\d+){1,3}\b")
_SHORT_STRUCT_TITLES = {
    "范围",
    "适用范围",
    "术语",
    "定义",
    "术语和定义",
    "application of this document",
    "purpose and introduction",
}


def is_anchor_like_query(text: str) -> bool:
    q = normalize_for_anchor_match(text)
    if not q:
        return False

    if _RE_CN_ANCHOR.search(q):
        return True
    if _RE_APPENDIX.search(q):
        return True
    if _RE_DOTTED.search(q):
        return True

    # 很短的结构标题词，也允许视为锚点 query
    if len(q) <= 32:
        q_low = q.lower()
        for t in _SHORT_STRUCT_TITLES:
            if t in q_low or t in q:
                return True

    return False


class AnchorRetriever:
    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        with (index_dir / "postings.json").open("r", encoding="utf-8") as f:
            self.postings = json.load(f)
        with (index_dir / "meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.doc_lengths = self.meta["doc_lengths"]
        self.doc_freq = self.meta["doc_freq"]
        self.object_meta = self.meta["object_meta"]
        self.avgdl = float(self.meta["avgdl"])
        self.k1 = float(self.meta.get("bm25_k1", 1.2))
        self.b = float(self.meta.get("bm25_b", 0.75))
        self.num_docs = int(self.meta["num_docs"])

    def search(
        self,
        query_variants: Sequence[str],
        topk: int = 30,
    ) -> List[AnchorHit]:
        merged: Dict[str, AnchorHit] = {}

        for qv in [q for q in query_variants if str(q).strip()]:
            hits = self._search_one(qv, topk=topk)
            for h in hits:
                prev = merged.get(h.object_id)
                if prev is None:
                    merged[h.object_id] = h
                else:
                    # exact 优先，其次高分
                    if (h.exact_match and not prev.exact_match) or (h.score > prev.score):
                        merged[h.object_id] = h

        out = sorted(
            merged.values(),
            key=lambda x: (not x.exact_match, -x.score, x.rank),
        )

        reranked = []
        for i, h in enumerate(out[:topk], start=1):
            reranked.append(
                AnchorHit(
                    object_id=h.object_id,
                    object_type=h.object_type,
                    doc_id=h.doc_id,
                    score=h.score,
                    rank=i,
                    source_view=h.source_view,
                    query_variant=h.query_variant,
                    exact_match=h.exact_match,
                )
            )
        return reranked

    def _search_one(self, query: str, topk: int = 30) -> List[AnchorHit]:
        q_norm = normalize_for_anchor_match(query)
        q_tokens = simple_word_tokenize(q_norm)

        scores: Dict[str, float] = {}
        exact_boost_ids = set()

        allow_exact_boost = is_anchor_like_query(query) and bool(q_norm)

        # exact / contains boost：只对 anchor-like query 开启
        if allow_exact_boost:
            for object_id, meta in self.object_meta.items():
                number_sig = normalize_for_anchor_match(meta.get("number_signature") or "")
                path_text = normalize_for_anchor_match(meta.get("path_text") or "")
                anchor_text = normalize_for_anchor_match(meta.get("anchor_text") or "")

                # 先看编号型
                if number_sig and (q_norm == number_sig or number_sig in q_norm):
                    exact_boost_ids.add(object_id)
                    continue

                # 短结构标题 query 才允许 path / anchor contains boost
                if len(q_norm) <= 48:
                    if path_text and q_norm in path_text:
                        exact_boost_ids.add(object_id)
                        continue
                    if anchor_text and q_norm in anchor_text:
                        exact_boost_ids.add(object_id)
                        continue

        # BM25 主体：所有 query 都可以参与
        for token in q_tokens:
            postings = self.postings.get(token, [])
            if not postings:
                continue

            df = max(1, int(self.doc_freq.get(token, 1)))
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

            for p in postings:
                object_id = p["object_id"]
                tf = float(p["tf"])
                dl = float(self.doc_lengths.get(object_id, 1))
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                score = idf * (tf * (self.k1 + 1)) / max(denom, 1e-6)
                scores[object_id] = scores.get(object_id, 0.0) + score

        # exact boost 仍然保留，但只给明确 anchor-like query
        for object_id in exact_boost_ids:
            scores[object_id] = scores.get(object_id, 0.0) + 2.5

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        out: List[AnchorHit] = []

        for rank, (object_id, score) in enumerate(ranked, start=1):
            meta = self.object_meta[object_id]
            out.append(
                AnchorHit(
                    object_id=object_id,
                    object_type=meta["object_type"],
                    doc_id=meta["doc_id"],
                    score=float(score),
                    rank=rank,
                    source_view="anchor_bm25",
                    query_variant=query,
                    exact_match=(object_id in exact_boost_ids),
                )
            )
        return out