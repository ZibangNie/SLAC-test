from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ..io.schemas import BridgeMessage, OpenWebUIContext, OpenWebUIRequest, RuntimeConfig


_TEXT_ROLES = {"system", "user", "assistant"}



def _gen_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"



def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif "content" in item:
                parts.append(str(item.get("content", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if "content" in content:
            return _coerce_content_to_text(content.get("content"))
        if "text" in content:
            return str(content.get("text", ""))
        return str(content)
    return str(content)



def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[BridgeMessage]:
    normalized: List[BridgeMessage] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip().lower() or "user"
        if role not in _TEXT_ROLES:
            role = "user"
        content = _coerce_content_to_text(msg.get("content"))
        if not content.strip() and role != "system":
            continue
        normalized.append(
            BridgeMessage(
                role=role,
                content=content,
                name=msg.get("name"),
                meta={k: v for k, v in msg.items() if k not in {"role", "content", "name"}},
            )
        )
    return normalized



def _extract_latest_user_query(messages: List[BridgeMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return None



def parse_pipe_payload(raw_payload: Dict[str, Any]) -> OpenWebUIRequest:
    body = dict(raw_payload or {})
    metadata = body.get("__metadata__") or body.get("metadata") or {}

    raw_messages = _normalize_messages(body.get("messages") or [])
    query_text = (
        body.get("query_text")
        or body.get("question")
        or body.get("prompt")
        or body.get("body")
        or _extract_latest_user_query(raw_messages)
        or ""
    )
    query_text = str(query_text).strip()

    request_id = str(body.get("request_id") or metadata.get("request_id") or _gen_id("ow_req"))
    session_id = (
        body.get("session_id")
        or metadata.get("chat_id")
        or metadata.get("session_id")
        or body.get("chat_id")
    )
    query_id = str(body.get("query_id") or metadata.get("message_id") or _gen_id("qt"))

    context = OpenWebUIContext(
        doc_scope=list((body.get("context") or {}).get("doc_scope") or []),
        expected_doc_hint=(body.get("context") or {}).get("expected_doc_hint"),
        domain=(body.get("context") or {}).get("domain"),
        source_type=(body.get("context") or {}).get("source_type"),
        files=list((body.get("context") or {}).get("files") or body.get("files") or []),
    )

    runtime = body.get("runtime_config") or {}
    runtime_config = RuntimeConfig(
        retrieval_run_dir=runtime.get("retrieval_run_dir"),
        reranker_run_dir=runtime.get("reranker_run_dir"),
        working_dir=runtime.get("working_dir"),
        pipeline_overrides=dict(runtime.get("pipeline_overrides") or {}),
        prompt_hints_overrides=dict(runtime.get("prompt_hints_overrides") or {}),
        response_options=dict(runtime.get("response_options") or {}),
    )

    openwebui_meta = dict(body.get("openwebui_meta") or {})
    if metadata:
        openwebui_meta.setdefault("metadata", metadata)
    if body.get("model") is not None:
        openwebui_meta.setdefault("selected_model", body.get("model"))

    memory_override = body.get("memory_override")

    return OpenWebUIRequest(
        request_id=request_id,
        session_id=str(session_id) if session_id is not None else None,
        query_id=query_id,
        query_text=query_text,
        raw_messages=raw_messages,
        memory_override=memory_override,
        context=context,
        runtime_config=runtime_config,
        openwebui_meta=openwebui_meta,
        meta=dict(body.get("meta") or {}),
    )
