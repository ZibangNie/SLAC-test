from __future__ import annotations

from typing import List, Tuple

from ..io.schemas import BridgeMessage, MemoryObject, OpenWebUIRequest



def extract_current_messages_and_memory(req: OpenWebUIRequest) -> Tuple[List[BridgeMessage], MemoryObject]:
    if req.memory_override is not None:
        memory = req.memory_override
        current_messages = [BridgeMessage(role="user", content=req.query_text)]
        return current_messages, memory

    messages = list(req.raw_messages or [])
    if not messages:
        return [BridgeMessage(role="user", content=req.query_text)], MemoryObject(messages=[])

    latest_user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user" and messages[idx].content.strip():
            latest_user_idx = idx
            break

    if latest_user_idx is None:
        return [BridgeMessage(role="user", content=req.query_text)], MemoryObject(messages=messages)

    current_messages = [messages[latest_user_idx]]
    history = messages[:latest_user_idx]
    memory = MemoryObject(messages=history)
    return current_messages, memory
