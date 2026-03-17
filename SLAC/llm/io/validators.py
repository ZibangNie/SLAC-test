from __future__ import annotations

from typing import Iterable

from .schemas import (
    REQUEST_RECORD_TYPE,
    REQUEST_SCHEMA_VERSION,
    ChatMessage,
    EvidenceItem,
    LLMRequest,
)

_ALLOWED_ROLES = {"system", "user", "assistant"}
_ALLOWED_MEMORY_MERGE_POLICIES = {"prepend"}
_ALLOWED_EVIDENCE_RENDER_POLICIES = {"append_as_context_block"}
_ALLOWED_PROVIDERS = {"openai_compatible"}


class ValidationError(ValueError):
    """Structured validation error for LLM request validation."""


def validate_llm_request(req: LLMRequest) -> None:
    if req.schema_version != REQUEST_SCHEMA_VERSION:
        raise ValidationError(
            f"schema_version must be {REQUEST_SCHEMA_VERSION}, got {req.schema_version!r}"
        )
    if req.record_type != REQUEST_RECORD_TYPE:
        raise ValidationError(
            f"record_type must be {REQUEST_RECORD_TYPE}, got {req.record_type!r}"
        )
    if not req.request_id:
        raise ValidationError("request_id must be non-empty")
    if not req.provider:
        raise ValidationError("provider must be non-empty")
    if req.provider not in _ALLOWED_PROVIDERS:
        raise ValidationError(
            f"unsupported provider {req.provider!r}; current supported providers: {sorted(_ALLOWED_PROVIDERS)}"
        )
    if not req.model_name:
        raise ValidationError("model_name must be non-empty")
    if not req.api_base:
        raise ValidationError("api_base must be non-empty")
    if not req.api_key_env:
        raise ValidationError("api_key_env must be non-empty")

    if not req.messages and not req.prompt:
        raise ValidationError("either messages or prompt must be provided")

    _validate_messages(req.messages, field_name="messages")
    if req.memory:
        _validate_messages(req.memory.messages, field_name="memory.messages")

    _validate_evidence(req.evidence)

    gc = req.generation_config
    if gc.temperature < 0:
        raise ValidationError("generation_config.temperature must be >= 0")
    if not 0 < gc.top_p <= 1:
        raise ValidationError("generation_config.top_p must be in (0, 1]")
    if gc.max_tokens <= 0:
        raise ValidationError("generation_config.max_tokens must be > 0")
    if gc.timeout_s <= 0:
        raise ValidationError("generation_config.timeout_s must be > 0")

    options = req.options or {}
    memory_merge_policy = options.get("memory_merge_policy", "prepend")
    if memory_merge_policy not in _ALLOWED_MEMORY_MERGE_POLICIES:
        raise ValidationError(
            f"options.memory_merge_policy must be one of {sorted(_ALLOWED_MEMORY_MERGE_POLICIES)}"
        )
    evidence_render_policy = options.get("evidence_render_policy", "append_as_context_block")
    if evidence_render_policy not in _ALLOWED_EVIDENCE_RENDER_POLICIES:
        raise ValidationError(
            f"options.evidence_render_policy must be one of {sorted(_ALLOWED_EVIDENCE_RENDER_POLICIES)}"
        )
    if "return_raw_response" in options and not isinstance(options["return_raw_response"], bool):
        raise ValidationError("options.return_raw_response must be bool when provided")


def _validate_messages(messages: Iterable[ChatMessage], *, field_name: str) -> None:
    for idx, msg in enumerate(messages):
        if msg.role not in _ALLOWED_ROLES:
            raise ValidationError(
                f"{field_name}[{idx}].role must be one of {sorted(_ALLOWED_ROLES)}, got {msg.role!r}"
            )
        if not isinstance(msg.content, str) or msg.content == "":
            raise ValidationError(f"{field_name}[{idx}].content must be non-empty string")


def _validate_evidence(evidence: list[EvidenceItem]) -> None:
    rerank_ranks: list[int] = []
    for idx, ev in enumerate(evidence):
        if not ev.chunk_id:
            raise ValidationError(f"evidence[{idx}].chunk_id must be non-empty")
        if not ev.doc_id:
            raise ValidationError(f"evidence[{idx}].doc_id must be non-empty")
        if not ev.passage_text:
            raise ValidationError(f"evidence[{idx}].passage_text must be non-empty")
        if not isinstance(ev.source_views, list):
            raise ValidationError(f"evidence[{idx}].source_views must be a list")
        if ev.rerank_rank is not None:
            rerank_ranks.append(ev.rerank_rank)

    if rerank_ranks and len(rerank_ranks) != len(evidence):
        raise ValidationError(
            "if any evidence item provides rerank_rank, all evidence items must provide rerank_rank"
        )

    if rerank_ranks:
        if rerank_ranks != sorted(rerank_ranks):
            raise ValidationError(
                "evidence already contains rerank_rank, but its sequence is not ascending; "
                "module 2 is not allowed to reorder evidence internally"
            )
        if len(set(rerank_ranks)) != len(rerank_ranks):
            raise ValidationError("evidence.rerank_rank values must be unique within one request")
