from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


ChatRole = Literal["system", "user", "assistant"]


def dataclass_to_dict(obj: Any) -> Any:
    return asdict(obj)


@dataclass
class OpenWebUIMessage:
    role: ChatRole
    content: str
    name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenWebUIMessage":
        return cls(
            role=str(data.get("role", "")).strip(),  # type: ignore[arg-type]
            content=str(data.get("content", "")).strip(),
            name=(str(data["name"]).strip() if data.get("name") is not None else None),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class OpenWebUIMemoryOverride:
    source: str = "openwebui_current_session"
    messages: List[OpenWebUIMessage] = field(default_factory=list)
    summary_text: Optional[str] = None

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
    ) -> Optional["OpenWebUIMemoryOverride"]:
        if not data:
            return None
        return cls(
            source=str(data.get("source", "openwebui_current_session")).strip(),
            messages=[OpenWebUIMessage.from_dict(x) for x in data.get("messages", []) or []],
            summary_text=(
                str(data["summary_text"]).strip()
                if data.get("summary_text") is not None
                else None
            ),
        )


@dataclass
class FileRef:
    file_id: Optional[str] = None
    doc_id: Optional[str] = None
    file_name: Optional[str] = None
    source_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileRef":
        return cls(
            file_id=(str(data["file_id"]).strip() if data.get("file_id") is not None else None),
            doc_id=(str(data["doc_id"]).strip() if data.get("doc_id") is not None else None),
            file_name=(
                str(data["file_name"]).strip() if data.get("file_name") is not None else None
            ),
            source_type=(
                str(data["source_type"]).strip()
                if data.get("source_type") is not None
                else None
            ),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class OpenWebUIContext:
    doc_scope: List[str] = field(default_factory=list)
    expected_doc_hint: Optional[str] = None
    domain: Optional[str] = None
    source_type: Optional[str] = None
    files: List[FileRef] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OpenWebUIContext":
        if not data:
            return cls()
        return cls(
            doc_scope=[str(x).strip() for x in data.get("doc_scope", []) or [] if str(x).strip()],
            expected_doc_hint=(
                str(data["expected_doc_hint"]).strip()
                if data.get("expected_doc_hint") is not None
                else None
            ),
            domain=(str(data["domain"]).strip() if data.get("domain") is not None else None),
            source_type=(
                str(data["source_type"]).strip()
                if data.get("source_type") is not None
                else None
            ),
            files=[FileRef.from_dict(x) for x in data.get("files", []) or []],
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class OpenWebUIRuntimeConfig:
    retrieval_run_dir: Optional[str] = None
    reranker_run_dir: Optional[str] = None
    working_dir: Optional[str] = None

    pipeline_overrides: Dict[str, Any] = field(default_factory=dict)
    prompt_hints_overrides: Dict[str, Any] = field(default_factory=dict)
    response_options: Dict[str, Any] = field(default_factory=dict)

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OpenWebUIRuntimeConfig":
        if not data:
            return cls()

        known_keys = {
            "retrieval_run_dir",
            "reranker_run_dir",
            "working_dir",
            "pipeline_overrides",
            "prompt_hints_overrides",
            "response_options",
        }
        extra = {k: v for k, v in dict(data).items() if k not in known_keys}

        return cls(
            retrieval_run_dir=(
                str(data["retrieval_run_dir"]).strip()
                if data.get("retrieval_run_dir") is not None
                else None
            ),
            reranker_run_dir=(
                str(data["reranker_run_dir"]).strip()
                if data.get("reranker_run_dir") is not None
                else None
            ),
            working_dir=(
                str(data["working_dir"]).strip()
                if data.get("working_dir") is not None
                else None
            ),
            pipeline_overrides=dict(data.get("pipeline_overrides", {}) or {}),
            prompt_hints_overrides=dict(data.get("prompt_hints_overrides", {}) or {}),
            response_options=dict(data.get("response_options", {}) or {}),
            extra=extra,
        )


@dataclass
class OpenWebUIRequest:
    schema_version: str
    record_type: str

    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]
    query_text: str

    raw_messages: List[OpenWebUIMessage] = field(default_factory=list)
    memory_override: Optional[OpenWebUIMemoryOverride] = None

    context: OpenWebUIContext = field(default_factory=OpenWebUIContext)
    runtime_config: OpenWebUIRuntimeConfig = field(default_factory=OpenWebUIRuntimeConfig)

    openwebui_meta: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenWebUIRequest":
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
            raw_messages=[OpenWebUIMessage.from_dict(x) for x in data.get("raw_messages", []) or []],
            memory_override=OpenWebUIMemoryOverride.from_dict(data.get("memory_override")),
            context=OpenWebUIContext.from_dict(data.get("context")),
            runtime_config=OpenWebUIRuntimeConfig.from_dict(data.get("runtime_config")),
            openwebui_meta=dict(data.get("openwebui_meta", {}) or {}),
            meta=dict(data.get("meta", {}) or {}),
        )


@dataclass
class ExtractedConversation:
    query_text: str
    current_messages: List[OpenWebUIMessage] = field(default_factory=list)
    memory: Optional[OpenWebUIMemoryOverride] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenWebUIResponse:
    schema_version: str
    record_type: str
    status: str

    request_id: str
    session_id: Optional[str]
    query_id: Optional[str]

    answer_text: str
    display_messages: List[OpenWebUIMessage] = field(default_factory=list)

    integration_response: Optional[Dict[str, Any]] = None
    trace_summary: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    meta: Dict[str, Any] = field(default_factory=dict)