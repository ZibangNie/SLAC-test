from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


VALID_INTENTS = {
    "anchor",
    "definition",
    "fact",
    "enumerate",
    "summary",
    "procedure",
    "compare",
    "unknown",
}


@dataclass
class QueryInput:
    query_id: str
    query: str
    lang_hint: Optional[str] = None
    domain_hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    query_id: str
    query_raw: str
    query_normalized: str

    intent: str = "unknown"

    query_main_zh: Optional[str] = None
    query_main_en: Optional[str] = None

    keywords: List[str] = field(default_factory=list)
    anchor_terms: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)

    must_keep_terms: List[str] = field(default_factory=list)
    avoid_terms: List[str] = field(default_factory=list)

    planner_model: Optional[str] = None
    planner_cache_key: Optional[str] = None
    planner_latency_ms: Optional[int] = None

    raw_response: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> "QueryPlan":
        if self.intent not in VALID_INTENTS:
            self.intent = "unknown"

        self.keywords = _uniq_clean(self.keywords)
        self.anchor_terms = _uniq_clean(self.anchor_terms)
        self.subqueries = _uniq_clean(self.subqueries)
        self.must_keep_terms = _uniq_clean(self.must_keep_terms)
        self.avoid_terms = _uniq_clean(self.avoid_terms)

        if self.query_main_zh:
            self.query_main_zh = self.query_main_zh.strip()
        if self.query_main_en:
            self.query_main_en = self.query_main_en.strip()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _uniq_clean(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x is None:
            continue
        y = str(x).strip()
        if not y:
            continue
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out