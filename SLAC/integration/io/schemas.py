from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


ChatRole = Literal["system", "user", "assistant"]


def dataclass_to_dict(obj: Any) -> Any:
    return asdict(obj)


@dataclass
class ChatMessage:
    role: ChatRole
    content: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data.get("role", "")).strip(),  # type: ignore[arg-type]
            content=str(data.get("content", "")).strip(),
        )


@dataclass
class ConversationMemory:
    source: str = "openwebui_current_session"
    messages: List[ChatMessage] = field(default_factory=list)
    summary_text: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ConversationMemory"]:
        if not data:
            return None
        return cls(
            source=str(data.get("source", "openwebui_current_session")).strip(),
            messages=[ChatMessage.from_dict(x) for x in data.get("messages", []) or []],
            summary_text=(
                str(data["summary_text"]).strip()
                if data.get("summary_text") is not None
                else None
            ),
        )


@dataclass
class IntegrationContext:
    retrieval_run_dir: Optional[str] = None
    reranker_run_dir: Optional[str] = None
    working_dir: Optional[str] = None
    prefer_pack_bridge: bool = True

    # 允许上层直接塞预构造 artifact；便于离线回放/联调
    retrieval_artifacts: Dict[str, Any] = field(default_factory=dict)
    reranker_artifacts: Dict[str, Any] = field(default_factory=dict)

    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IntegrationContext":
        if not data:
            return cls()
        return cls(
            retrieval_run_dir=data.get("retrieval_run_dir"),
            reranker_run_dir=data.get("reranker_run_dir"),
            working_dir=data.get("working_dir"),
            prefer_pack_bridge=bool(data.get("prefer_pack_bridge", True)),
            retrieval_artifacts=dict(data.get("retrieval_artifacts", {}) or {}),
            reranker_artifacts=dict(data.get("reranker_artifacts", {}) or {}),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class LLMConfig:
    provider: str
    model_name: str
    api_base: str
    api_key_env: str

    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout_s: int = 120

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=str(data.get("provider", "")).strip(),
            model_name=str(data.get("model_name", "")).strip(),
            api_base=str(data.get("api_base", "")).strip(),
            api_key_env=str(data.get("api_key_env", "")).strip(),
            temperature=float(data.get("temperature", 0.2)),
            top_p=float(data.get("top_p", 0.95)),
            max_tokens=int(data.get("max_tokens", 1024)),
            timeout_s=int(data.get("timeout_s", 120)),
        )


@dataclass
class PipelineConfig:
    use_retrieval: bool = True
    use_reranker: bool = True
    allow_retrieval_fallback: bool = True

    retrieval_top_k: int = 40
    reranker_top_k: int = 15

    max_evidence_items: int = 6
    max_evidence_tokens: int = 1800

    prefer_direct_first: bool = True
    min_direct_evidence: int = 1

    llm: Optional[LLMConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            use_retrieval=bool(data.get("use_retrieval", True)),
            use_reranker=bool(data.get("use_reranker", True)),
            allow_retrieval_fallback=bool(data.get("allow_retrieval_fallback", True)),
            retrieval_top_k=int(data.get("retrieval_top_k", 40)),
            reranker_top_k=int(data.get("reranker_top_k", 15)),
            max_evidence_items=int(data.get("max_evidence_items", 6)),
            max_evidence_tokens=int(data.get("max_evidence_tokens", 1800)),
            prefer_direct_first=bool(data.get("prefer_direct_first", True)),
            min_direct_evidence=int(data.get("min_direct_evidence", 1)),
            llm=LLMConfig.from_dict(data.get("llm", {}) or {}),
        )


@dataclass
class PromptHints:
    answer_language: str = "zh"
    require_grounding: bool = True
    answer_style: str = "concise"
    insufficient_evidence_policy: str = "state_insufficiency"
    system_prompt_override: Optional[str] = None
    user_prompt_prefix: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PromptHints":
        if not data:
            return cls()
        return cls(
            answer_language=str(data.get("answer_language", "zh")).strip(),
            require_grounding=bool(data.get("require_grounding", True)),
            answer_style=str(data.get("answer_style", "concise")).strip(),
            insufficient_evidence_policy=str(
                data.get("insufficient_evidence_policy", "state_insufficiency")
            ).strip(),
            system_prompt_override=(
                str(data["system_prompt_override"]).strip()
                if data.get("system_prompt_override") is not None
                else None
            ),
            user_prompt_prefix=(
                str(data["user_prompt_prefix"]).strip()
                if data.get("user_prompt_prefix") is not None
                else None
            ),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class IntegrationRequest:
    schema_version: str
    record_type: str

    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]
    query_text: str

    messages: List[ChatMessage] = field(default_factory=list)
    memory: Optional[ConversationMemory] = None

    context: IntegrationContext = field(default_factory=IntegrationContext)
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    prompt_hints: PromptHints = field(default_factory=PromptHints)

    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationRequest":
        return cls(
            schema_version=str(data.get("schema_version", "")).strip(),
            record_type=str(data.get("record_type", "")).strip(),
            request_id=str(data.get("request_id", "")).strip(),
            session_id=(
                str(data["session_id"]).strip() if data.get("session_id") is not None else None
            ),
            query_id=(
                str(data["query_id"]).strip() if data.get("query_id") is not None else None
            ),
            query_text=str(data.get("query_text", "")).strip(),
            messages=[ChatMessage.from_dict(x) for x in data.get("messages", []) or []],
            memory=ConversationMemory.from_dict(data.get("memory")),
            context=IntegrationContext.from_dict(data.get("context")),
            pipeline_config=PipelineConfig.from_dict(data.get("pipeline_config", {}) or {}),
            prompt_hints=PromptHints.from_dict(data.get("prompt_hints")),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class SelectedEvidence:
    chunk_id: str
    doc_id: str
    passage_text: str

    query_id: Optional[str] = None
    query_text: Optional[str] = None

    rerank_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    retrieve_rank_fused: Optional[int] = None

    role: Optional[str] = None
    hit_type: Optional[str] = None
    source_views: List[str] = field(default_factory=list)

    path_text: Optional[str] = None
    token_est: Optional[int] = None
    expansion_depth: Optional[int] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle:
    system_prompt: str
    current_messages: List[ChatMessage]
    evidence_context_block: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalArtifacts:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reranker_input: List[Dict[str, Any]] = field(default_factory=list)
    packed_evidence: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerArtifacts:
    pack_bridge: List[Dict[str, Any]] = field(default_factory=list)
    reranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTrace:
    retrieval_used: bool = False
    reranker_used: bool = False
    degraded_to_retrieval: bool = False

    num_candidates_read: int = 0
    num_evidence_selected: int = 0
    evidence_budget_tokens: int = 0

    llm_request_id: Optional[str] = None
    llm_response_id: Optional[str] = None

    candidate_source: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResponse:
    schema_version: str
    record_type: str
    status: str

    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]
    query_text: Optional[str]

    answer_text: str
    answer_result: Optional[Dict[str, Any]] = None
    evidence: List[SelectedEvidence] = field(default_factory=list)
    trace: IntegrationTrace = field(default_factory=IntegrationTrace)

    meta: Dict[str, Any] = field(default_factory=dict)