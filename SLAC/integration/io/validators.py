from __future__ import annotations

from typing import Any, Dict, Iterable, List

from SLAC.integration.io.schemas import (
    ChatMessage,
    ConversationMemory,
    IntegrationRequest,
)


_ALLOWED_ROLES = {"system", "user", "assistant"}
_ALLOWED_LANGUAGES = {"zh", "en", "auto"}
_ALLOWED_ANSWER_STYLES = {"concise", "balanced", "detailed"}
_ALLOWED_INSUFF_POLICIES = {
    "state_insufficiency",
    "ask_for_more_context",
    "return_uncertain_answer",
}


def _require_nonempty_str(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _validate_messages(messages: Iterable[ChatMessage], field_name: str) -> None:
    for idx, msg in enumerate(messages):
        if msg.role not in _ALLOWED_ROLES:
            raise ValueError(
                f"{field_name}[{idx}].role must be one of {_ALLOWED_ROLES}, got {msg.role!r}."
            )
        if not str(msg.content or "").strip():
            raise ValueError(f"{field_name}[{idx}].content must be non-empty.")


def _validate_memory(memory: ConversationMemory | None) -> None:
    if memory is None:
        return
    _require_nonempty_str(memory.source, "memory.source")
    _validate_messages(memory.messages, "memory.messages")


def _validate_positive_int(value: int, field_name: str, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}.")
    else:
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0, got {value}.")


def validate_integration_request(
    data: Dict[str, Any] | IntegrationRequest,
) -> IntegrationRequest:
    req = data if isinstance(data, IntegrationRequest) else IntegrationRequest.from_dict(data)

    if req.schema_version != "slac_integration_request_v1":
        raise ValueError(
            "schema_version must be 'slac_integration_request_v1', "
            f"got {req.schema_version!r}."
        )
    if req.record_type != "integration_request":
        raise ValueError(
            "record_type must be 'integration_request', "
            f"got {req.record_type!r}."
        )

    _require_nonempty_str(req.request_id, "request_id")
    _require_nonempty_str(req.query_text, "query_text")

    if req.session_id is not None:
        _require_nonempty_str(req.session_id, "session_id")
    if req.query_id is not None:
        _require_nonempty_str(req.query_id, "query_id")

    _validate_messages(req.messages, "messages")
    _validate_memory(req.memory)

    pc = req.pipeline_config

    if pc.llm is None:
        raise ValueError("pipeline_config.llm is required.")

    _require_nonempty_str(pc.llm.provider, "pipeline_config.llm.provider")
    _require_nonempty_str(pc.llm.model_name, "pipeline_config.llm.model_name")
    _require_nonempty_str(pc.llm.api_base, "pipeline_config.llm.api_base")
    _require_nonempty_str(pc.llm.api_key_env, "pipeline_config.llm.api_key_env")

    if pc.llm.provider != "openai_compatible":
        raise ValueError(
            "Current implementation only supports provider='openai_compatible'."
        )

    _validate_positive_int(pc.retrieval_top_k, "pipeline_config.retrieval_top_k")
    _validate_positive_int(pc.reranker_top_k, "pipeline_config.reranker_top_k")
    _validate_positive_int(pc.max_evidence_items, "pipeline_config.max_evidence_items")
    _validate_positive_int(pc.max_evidence_tokens, "pipeline_config.max_evidence_tokens")
    _validate_positive_int(pc.min_direct_evidence, "pipeline_config.min_direct_evidence", allow_zero=True)
    _validate_positive_int(pc.llm.max_tokens, "pipeline_config.llm.max_tokens")
    _validate_positive_int(pc.llm.timeout_s, "pipeline_config.llm.timeout_s")

    if not (0.0 <= pc.llm.temperature <= 2.0):
        raise ValueError("pipeline_config.llm.temperature must be in [0.0, 2.0].")
    if not (0.0 < pc.llm.top_p <= 1.0):
        raise ValueError("pipeline_config.llm.top_p must be in (0.0, 1.0].")

    if pc.use_reranker and pc.reranker_top_k > pc.retrieval_top_k:
        raise ValueError(
            "pipeline_config.reranker_top_k should not exceed retrieval_top_k."
        )

    ph = req.prompt_hints
    if ph.answer_language not in _ALLOWED_LANGUAGES:
        raise ValueError(
            f"prompt_hints.answer_language must be one of {_ALLOWED_LANGUAGES}, "
            f"got {ph.answer_language!r}."
        )
    if ph.answer_style not in _ALLOWED_ANSWER_STYLES:
        raise ValueError(
            f"prompt_hints.answer_style must be one of {_ALLOWED_ANSWER_STYLES}, "
            f"got {ph.answer_style!r}."
        )
    if ph.insufficient_evidence_policy not in _ALLOWED_INSUFF_POLICIES:
        raise ValueError(
            "prompt_hints.insufficient_evidence_policy must be one of "
            f"{_ALLOWED_INSUFF_POLICIES}, got {ph.insufficient_evidence_policy!r}."
        )

    return req