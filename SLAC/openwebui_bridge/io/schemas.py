from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BridgeMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class MemoryObject(BaseModel):
    source: str = "openwebui_current_session"
    messages: List[BridgeMessage] = Field(default_factory=list)
    summary_text: Optional[str] = None


class OpenWebUIContext(BaseModel):
    doc_scope: List[str] = Field(default_factory=list)
    expected_doc_hint: Optional[str] = None
    domain: Optional[str] = None
    source_type: Optional[str] = None
    files: List[Dict[str, Any]] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    retrieval_run_dir: Optional[str] = None
    reranker_run_dir: Optional[str] = None
    working_dir: Optional[str] = None
    pipeline_overrides: Dict[str, Any] = Field(default_factory=dict)
    prompt_hints_overrides: Dict[str, Any] = Field(default_factory=dict)
    response_options: Dict[str, Any] = Field(default_factory=dict)


class OpenWebUIRequest(BaseModel):
    schema_version: str = "slac_openwebui_request_v1"
    record_type: str = "openwebui_request"
    request_id: str
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    query_text: str
    raw_messages: List[BridgeMessage] = Field(default_factory=list)
    memory_override: Optional[MemoryObject] = None
    context: OpenWebUIContext = Field(default_factory=OpenWebUIContext)
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    openwebui_meta: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class IntegrationRequest(BaseModel):
    schema_version: str = "slac_integration_request_v1"
    record_type: str = "integration_request"
    request_id: str
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    query_text: str
    messages: List[BridgeMessage] = Field(default_factory=list)
    memory: Optional[MemoryObject] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    pipeline_config: Dict[str, Any] = Field(default_factory=dict)
    prompt_hints: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class OpenWebUIResponse(BaseModel):
    schema_version: str = "slac_openwebui_response_v1"
    record_type: str = "openwebui_response"
    status: str
    request_id: str
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    answer_text: str
    display_messages: List[BridgeMessage] = Field(default_factory=list)
    integration_response: Optional[Dict[str, Any]] = None
    trace_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
