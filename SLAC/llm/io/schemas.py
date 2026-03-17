from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Literal, Optional


REQUEST_SCHEMA_VERSION = "slac_llm_request_v1"
ANSWER_SCHEMA_VERSION = "slac_llm_answer_v1"
REQUEST_RECORD_TYPE = "answer_request"
ANSWER_RECORD_TYPE = "answer_result"

ChatRole = Literal["system", "user", "assistant"]


def _drop_none(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


@dataclass
class ChatMessage:
    role: ChatRole
    content: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data["role"]).strip(),
            content=str(data["content"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class EvidenceItem:
    chunk_id: str
    doc_id: str
    passage_text: str
    query_id: Optional[str] = None
    query_text: Optional[str] = None
    path_text: Optional[str] = None
    rerank_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    retrieve_rank_fused: Optional[int] = None
    role: Optional[str] = None
    hit_type: Optional[str] = None
    source_views: List[str] = field(default_factory=list)
    token_est: Optional[int] = None
    expansion_depth: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        source_views = data.get("source_views") or []
        if not isinstance(source_views, list):
            source_views = [str(source_views)]
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"raw_meta": meta}
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            passage_text=str(data["passage_text"]),
            query_id=_opt_str(data.get("query_id")),
            query_text=_opt_str(data.get("query_text")),
            path_text=_opt_str(data.get("path_text")),
            rerank_rank=_opt_int(data.get("rerank_rank")),
            rerank_score=_opt_float(data.get("rerank_score")),
            retrieve_rank_fused=_opt_int(data.get("retrieve_rank_fused")),
            role=_opt_str(data.get("role")),
            hit_type=_opt_str(data.get("hit_type")),
            source_views=[str(v) for v in source_views],
            token_est=_opt_int(data.get("token_est")),
            expansion_depth=_opt_int(data.get("expansion_depth")),
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class ConversationMemory:
    source: str = "openwebui_current_session"
    messages: List[ChatMessage] = field(default_factory=list)
    summary_text: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        messages = [ChatMessage.from_dict(item) for item in (data.get("messages") or [])]
        return cls(
            source=_opt_str(data.get("source")) or "openwebui_current_session",
            messages=messages,
            summary_text=_opt_str(data.get("summary_text")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout_s: int = 120

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GenerationConfig":
        data = data or {}
        return cls(
            temperature=float(data.get("temperature", 0.2)),
            top_p=float(data.get("top_p", 0.95)),
            max_tokens=int(data.get("max_tokens", 1024)),
            timeout_s=int(data.get("timeout_s", 120)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class LLMRequest:
    schema_version: str
    record_type: str
    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]
    query_text: str

    provider: str
    model_name: str
    api_base: str
    api_key_env: str

    generation_config: GenerationConfig
    system_prompt: Optional[str] = None
    messages: List[ChatMessage] = field(default_factory=list)
    prompt: Optional[str] = None

    memory: Optional[ConversationMemory] = None
    evidence: List[EvidenceItem] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMRequest":
        messages = [ChatMessage.from_dict(item) for item in (data.get("messages") or [])]
        evidence = [EvidenceItem.from_dict(item) for item in (data.get("evidence") or [])]
        memory_raw = data.get("memory")
        memory = ConversationMemory.from_dict(memory_raw) if memory_raw else None
        options = data.get("options") or {}
        meta = data.get("meta") or {}
        if not isinstance(options, dict):
            raise TypeError("options must be an object")
        if not isinstance(meta, dict):
            raise TypeError("meta must be an object")
        return cls(
            schema_version=str(data.get("schema_version", REQUEST_SCHEMA_VERSION)),
            record_type=str(data.get("record_type", REQUEST_RECORD_TYPE)),
            request_id=str(data["request_id"]),
            session_id=_opt_str(data.get("session_id")),
            query_id=_opt_str(data.get("query_id")),
            query_text=str(data.get("query_text", "")),
            provider=str(data["provider"]),
            model_name=str(data["model_name"]),
            api_base=str(data["api_base"]),
            api_key_env=str(data["api_key_env"]),
            generation_config=GenerationConfig.from_dict(data.get("generation_config")),
            system_prompt=_opt_str(data.get("system_prompt")),
            messages=messages,
            prompt=_opt_str(data.get("prompt")),
            memory=memory,
            evidence=evidence,
            options=options,
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class UsageInfo:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UsageInfo":
        data = data or {}
        return cls(
            prompt_tokens=_opt_int(data.get("prompt_tokens")),
            completion_tokens=_opt_int(data.get("completion_tokens")),
            total_tokens=_opt_int(data.get("total_tokens")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass
class LLMAnswerResult:
    schema_version: str
    record_type: str
    status: str
    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]
    provider: str
    model_name: str
    response_id: Optional[str]
    answer_text: str
    finish_reason: Optional[str]
    usage: UsageInfo
    memory_used: bool
    memory_message_count: int
    evidence_count: int
    evidence_refs: List[Dict[str, Any]]
    raw_response: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMAnswerResult":
        return cls(
            schema_version=str(data.get("schema_version", ANSWER_SCHEMA_VERSION)),
            record_type=str(data.get("record_type", ANSWER_RECORD_TYPE)),
            status=str(data["status"]),
            request_id=str(data["request_id"]),
            session_id=_opt_str(data.get("session_id")),
            query_id=_opt_str(data.get("query_id")),
            provider=str(data["provider"]),
            model_name=str(data["model_name"]),
            response_id=_opt_str(data.get("response_id")),
            answer_text=str(data.get("answer_text", "")),
            finish_reason=_opt_str(data.get("finish_reason")),
            usage=UsageInfo.from_dict(data.get("usage")),
            memory_used=bool(data.get("memory_used", False)),
            memory_message_count=int(data.get("memory_message_count", 0)),
            evidence_count=int(data.get("evidence_count", 0)),
            evidence_refs=list(data.get("evidence_refs") or []),
            raw_response=data.get("raw_response"),
            meta=dict(data.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(asdict(self))


def build_error_result(
    request: LLMRequest,
    error_message: str,
    *,
    response_id: Optional[str] = None,
    raw_response: Optional[Dict[str, Any]] = None,
    error_type: Optional[str] = None,
) -> LLMAnswerResult:
    return LLMAnswerResult(
        schema_version=ANSWER_SCHEMA_VERSION,
        record_type=ANSWER_RECORD_TYPE,
        status="error",
        request_id=request.request_id,
        session_id=request.session_id,
        query_id=request.query_id,
        provider=request.provider,
        model_name=request.model_name,
        response_id=response_id,
        answer_text="",
        finish_reason=None,
        usage=UsageInfo(),
        memory_used=bool(request.memory and (request.memory.messages or request.memory.summary_text)),
        memory_message_count=len(request.memory.messages) if request.memory else 0,
        evidence_count=len(request.evidence),
        evidence_refs=[
            {
                "chunk_id": ev.chunk_id,
                "doc_id": ev.doc_id,
                "rerank_rank": ev.rerank_rank,
            }
            for ev in request.evidence
        ],
        raw_response=raw_response if request.options.get("return_raw_response") else None,
        meta={
            "error_message": error_message,
            "error_type": error_type,
        },
    )


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    return value if value != "" else None


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)
