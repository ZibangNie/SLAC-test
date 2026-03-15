from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from SLAC.retrieval.preprocess.anchor_extractor import extract_anchor_terms
from SLAC.retrieval.preprocess.bilingual_expander import (
    expand_bilingual_terms,
    load_bilingual_terms,
)
from SLAC.retrieval.preprocess.query_normalizer import normalize_query_input
from SLAC.retrieval.preprocess.subquery_splitter import split_subqueries_heuristic
from SLAC.retrieval.schemas.query_schema import QueryInput, QueryPlan


PLANNER_SYSTEM_PROMPT = """You are a retrieval query planner for a structure-aware long-document RAG system.
Return ONLY a JSON object with the following fields:
{
  "intent": "anchor|definition|fact|enumerate|summary|procedure|compare|unknown",
  "query_main_zh": "...",
  "query_main_en": "...",
  "keywords": ["..."],
  "anchor_terms": ["..."],
  "subqueries": ["..."],
  "must_keep_terms": ["..."],
  "avoid_terms": ["..."]
}

Rules:
- Focus on retrieval planning, not answering the question.
- Preserve critical entities, numbers, section anchors, and domain terms.
- If the query is Chinese, still produce an English retrieval version when possible.
- If the query is English, still produce a Chinese retrieval version when possible.
- Use concise keyword-style rewrites for retrieval.
- Do not invent document content.
- Prefer at most 3 subqueries.
"""


class QueryPlanner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
        cache_dir: Optional[str | Path] = None,
        bilingual_terms_path: Optional[str | Path] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.term_map = load_bilingual_terms(bilingual_terms_path) if bilingual_terms_path else {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def plan(self, q: QueryInput) -> QueryPlan:
        query_normalized = normalize_query_input(q)
        cache_key = self._make_cache_key(query_normalized)

        if self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # 先构造启发式特征，供 LLM 与 fallback 共用
        anchor_terms = extract_anchor_terms(query_normalized)
        bilingual_expansions = expand_bilingual_terms(query_normalized, self.term_map) if self.term_map else []
        subqueries = split_subqueries_heuristic(query_normalized)

        plan = None
        if self.api_key:
            try:
                plan = self._plan_with_llm(
                    q=q,
                    query_normalized=query_normalized,
                    anchor_terms=anchor_terms,
                    bilingual_expansions=bilingual_expansions,
                    subqueries=subqueries,
                    cache_key=cache_key,
                )
            except Exception:
                plan = None

        if plan is None:
            plan = self._fallback_plan(
                q=q,
                query_normalized=query_normalized,
                anchor_terms=anchor_terms,
                bilingual_expansions=bilingual_expansions,
                subqueries=subqueries,
                cache_key=cache_key,
            )

        if self.cache_dir:
            self._save_cache(cache_key, plan)

        return plan

    def _plan_with_llm(
        self,
        q: QueryInput,
        query_normalized: str,
        anchor_terms: List[str],
        bilingual_expansions: List[str],
        subqueries: List[str],
        cache_key: str,
    ) -> QueryPlan:
        user_payload = {
            "query_raw": q.query,
            "query_normalized": query_normalized,
            "lang_hint": q.lang_hint,
            "domain_hint": q.domain_hint,
            "anchor_terms_seed": anchor_terms,
            "bilingual_expansions_seed": bilingual_expansions,
            "subqueries_seed": subqueries,
        }

        body = {
            "model": self.model_name,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        latency_ms = int((time.time() - start) * 1000)

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        plan = QueryPlan(
            query_id=q.query_id,
            query_raw=q.query,
            query_normalized=query_normalized,
            intent=parsed.get("intent", "unknown"),
            query_main_zh=parsed.get("query_main_zh"),
            query_main_en=parsed.get("query_main_en"),
            keywords=parsed.get("keywords", []),
            anchor_terms=_merge_unique(anchor_terms, parsed.get("anchor_terms", [])),
            subqueries=_merge_unique(subqueries, parsed.get("subqueries", [])),
            must_keep_terms=parsed.get("must_keep_terms", []),
            avoid_terms=parsed.get("avoid_terms", []),
            planner_model=self.model_name,
            planner_cache_key=cache_key,
            planner_latency_ms=latency_ms,
            raw_response=parsed,
            meta={
                "bilingual_expansions_seed": bilingual_expansions,
            },
        ).normalize()

        # 如果 LLM 没生成双语检索串，就用启发式补一层
        if not plan.query_main_en and bilingual_expansions:
            plan.query_main_en = " ; ".join(bilingual_expansions)
        if not plan.query_main_zh:
            plan.query_main_zh = query_normalized

        return plan

    def _fallback_plan(
        self,
        q: QueryInput,
        query_normalized: str,
        anchor_terms: List[str],
        bilingual_expansions: List[str],
        subqueries: List[str],
        cache_key: str,
    ) -> QueryPlan:
        intent = infer_intent_heuristic(query_normalized, anchor_terms)
        keywords = build_keywords_heuristic(query_normalized, anchor_terms)

        plan = QueryPlan(
            query_id=q.query_id,
            query_raw=q.query,
            query_normalized=query_normalized,
            intent=intent,
            query_main_zh=query_normalized,
            query_main_en=" ; ".join(bilingual_expansions) if bilingual_expansions else None,
            keywords=keywords,
            anchor_terms=anchor_terms,
            subqueries=subqueries,
            must_keep_terms=anchor_terms[:],
            avoid_terms=[],
            planner_model="heuristic_fallback",
            planner_cache_key=cache_key,
            planner_latency_ms=None,
            raw_response=None,
            meta={"bilingual_expansions_seed": bilingual_expansions},
        ).normalize()
        return plan

    def _make_cache_key(self, normalized_query: str) -> str:
        return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Optional[QueryPlan]:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return QueryPlan(**obj).normalize()

    def _save_cache(self, cache_key: str, plan: QueryPlan) -> None:
        path = self._cache_path(cache_key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)


def infer_intent_heuristic(query: str, anchor_terms: List[str]) -> str:
    q = query.lower()

    if anchor_terms:
        return "anchor"
    if any(x in q for x in ["定义", "术语", "definition", "definitions", "what is"]):
        return "definition"
    if any(x in q for x in ["有哪些", "包括哪些", "which", "what are", "list"]):
        return "enumerate"
    if any(x in q for x in ["步骤", "流程", "how to", "procedure"]):
        return "procedure"
    if any(x in q for x in ["总结", "概述", "summary", "overview"]):
        return "summary"
    if any(x in q for x in ["区别", "compare", "difference"]):
        return "compare"
    return "fact"


def build_keywords_heuristic(query: str, anchor_terms: List[str]) -> List[str]:
    toks = []
    for part in [query] + anchor_terms:
        toks.extend([x.strip() for x in part.replace("；", " ").replace("，", " ").split() if x.strip()])

    out = []
    seen = set()
    for t in toks:
        if len(t) < 2:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _merge_unique(a: List[str], b: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in list(a) + list(b):
        if x is None:
            continue
        y = str(x).strip()
        if not y:
            continue
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out