from __future__ import annotations

from typing import Any, Dict, List, Optional

from SLAC.interaction.io.schemas import (
    OpenWebUIContext,
    OpenWebUIMessage,
    OpenWebUIRequest,
    OpenWebUIRuntimeConfig,
)
from SLAC.interaction.utils.ids import make_request_id, make_query_id


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    text = str(item["text"]).strip()
                    if text:
                        parts.append(text)
                    continue
                if item.get("text") is not None:
                    text = str(item["text"]).strip()
                    if text:
                        parts.append(text)
                    continue
        return "\n".join(parts).strip()

    return str(content).strip()


def _normalize_message(msg: Dict[str, Any]) -> Optional[OpenWebUIMessage]:
    role = str(msg.get("role", "")).strip()
    content = _flatten_content(msg.get("content"))

    if not role or not content:
        return None

    name = str(msg.get("name")).strip() if msg.get("name") is not None else None

    meta = {}
    for key in ("id", "message_id", "timestamp", "created_at", "updated_at"):
        if key in msg:
            meta[key] = msg[key]
    if "meta" in msg and isinstance(msg["meta"], dict):
        meta.update(dict(msg["meta"]))

    return OpenWebUIMessage(
        role=role,  # type: ignore[arg-type]
        content=content,
        name=name,
        meta=meta,
    )


def _extract_messages(payload: Dict[str, Any]) -> List[OpenWebUIMessage]:
    body = payload.get("body", {}) if isinstance(payload.get("body"), dict) else {}

    raw_messages = (
        payload.get("raw_messages")
        or payload.get("messages")
        or body.get("messages")
        or []
    )

    normalized: List[OpenWebUIMessage] = []
    for item in raw_messages:
        if isinstance(item, dict):
            msg = _normalize_message(item)
            if msg is not None:
                normalized.append(msg)
    return normalized


def _last_user_query(messages: List[OpenWebUIMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


class PipeRequestParser:
    def parse(self, raw_payload: Dict[str, Any]) -> OpenWebUIRequest:
        if (
            raw_payload.get("schema_version") == "slac_openwebui_request_v1"
            and raw_payload.get("record_type") == "openwebui_request"
        ):
            return OpenWebUIRequest.from_dict(raw_payload)

        body = raw_payload.get("body", {}) if isinstance(raw_payload.get("body"), dict) else {}
        messages = _extract_messages(raw_payload)

        query_text = (
            str(raw_payload.get("query_text", "")).strip()
            or str(body.get("query_text", "")).strip()
            or str(raw_payload.get("query", "")).strip()
            or str(body.get("query", "")).strip()
            or str(raw_payload.get("prompt", "")).strip()
            or str(body.get("prompt", "")).strip()
            or _last_user_query(messages)
        )

        request_id = (
            str(raw_payload.get("request_id", "")).strip()
            or str(body.get("request_id", "")).strip()
            or str(raw_payload.get("id", "")).strip()
            or str(body.get("id", "")).strip()
            or make_request_id()
        )

        session_id = (
            str(raw_payload.get("session_id", "")).strip()
            or str(body.get("session_id", "")).strip()
            or str(raw_payload.get("chat_id", "")).strip()
            or str(body.get("chat_id", "")).strip()
            or str(raw_payload.get("conversation_id", "")).strip()
            or None
        )
        session_id = session_id or None

        query_id = (
            str(raw_payload.get("query_id", "")).strip()
            or str(body.get("query_id", "")).strip()
            or make_query_id(request_id)
        )

        context_obj = (
            raw_payload.get("context")
            if isinstance(raw_payload.get("context"), dict)
            else body.get("context")
        )
        runtime_obj = (
            raw_payload.get("runtime_config")
            if isinstance(raw_payload.get("runtime_config"), dict)
            else body.get("runtime_config")
        )

        openwebui_meta = {}
        selected_model = (
            raw_payload.get("selected_model")
            or body.get("selected_model")
            or raw_payload.get("model")
            or body.get("model")
        )
        if selected_model is not None:
            openwebui_meta["selected_model"] = selected_model

        for key in ("chat_id", "message_id", "stream", "user_id", "conversation_id"):
            if key in raw_payload:
                openwebui_meta[key] = raw_payload[key]
            elif key in body:
                openwebui_meta[key] = body[key]

        memory_override_obj = (
            raw_payload.get("memory_override")
            if isinstance(raw_payload.get("memory_override"), dict)
            else body.get("memory_override")
        )

        return OpenWebUIRequest(
            schema_version="slac_openwebui_request_v1",
            record_type="openwebui_request",
            request_id=request_id,
            session_id=session_id,
            query_id=query_id,
            query_text=query_text,
            raw_messages=messages,
            memory_override=(
                None
                if not isinstance(memory_override_obj, dict)
                else OpenWebUIRequest.from_dict(
                    {
                        "schema_version": "slac_openwebui_request_v1",
                        "record_type": "openwebui_request",
                        "request_id": request_id,
                        "session_id": session_id,
                        "query_id": query_id,
                        "query_text": query_text,
                        "raw_messages": [],
                        "memory_override": memory_override_obj,
                    }
                ).memory_override
            ),
            context=OpenWebUIContext.from_dict(context_obj),
            runtime_config=OpenWebUIRuntimeConfig.from_dict(runtime_obj),
            openwebui_meta=openwebui_meta,
            meta=dict(raw_payload.get("meta", {}) or {}),
        )