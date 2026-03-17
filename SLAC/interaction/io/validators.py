from __future__ import annotations

from typing import Any, Dict, Iterable

from SLAC.interaction.io.schemas import (
    OpenWebUIMessage,
    OpenWebUIRequest,
    OpenWebUIMemoryOverride,
)


_ALLOWED_ROLES = {"system", "user", "assistant"}
_ALLOWED_STATUS = {"ok", "degraded", "error"}


def _require_nonempty_str(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _validate_messages(messages: Iterable[OpenWebUIMessage], field_name: str) -> None:
    for idx, msg in enumerate(messages):
        if msg.role not in _ALLOWED_ROLES:
            raise ValueError(
                f"{field_name}[{idx}].role must be one of {_ALLOWED_ROLES}, got {msg.role!r}."
            )
        if not str(msg.content or "").strip():
            raise ValueError(f"{field_name}[{idx}].content must be non-empty.")


def _validate_memory(memory: OpenWebUIMemoryOverride | None) -> None:
    if memory is None:
        return
    _require_nonempty_str(memory.source, "memory_override.source")
    _validate_messages(memory.messages, "memory_override.messages")


def validate_openwebui_request(
    data: Dict[str, Any] | OpenWebUIRequest,
) -> OpenWebUIRequest:
    req = data if isinstance(data, OpenWebUIRequest) else OpenWebUIRequest.from_dict(data)

    if req.schema_version != "slac_openwebui_request_v1":
        raise ValueError(
            "schema_version must be 'slac_openwebui_request_v1', "
            f"got {req.schema_version!r}."
        )
    if req.record_type != "openwebui_request":
        raise ValueError(
            "record_type must be 'openwebui_request', "
            f"got {req.record_type!r}."
        )

    _require_nonempty_str(req.request_id, "request_id")
    _require_nonempty_str(req.query_text, "query_text")

    if req.session_id is not None:
        _require_nonempty_str(req.session_id, "session_id")
    if req.query_id is not None:
        _require_nonempty_str(req.query_id, "query_id")

    _validate_messages(req.raw_messages, "raw_messages")
    _validate_memory(req.memory_override)

    rc = req.runtime_config
    if rc.retrieval_run_dir is not None:
        _require_nonempty_str(rc.retrieval_run_dir, "runtime_config.retrieval_run_dir")
    if rc.reranker_run_dir is not None:
        _require_nonempty_str(rc.reranker_run_dir, "runtime_config.reranker_run_dir")
    if rc.working_dir is not None:
        _require_nonempty_str(rc.working_dir, "runtime_config.working_dir")

    return req