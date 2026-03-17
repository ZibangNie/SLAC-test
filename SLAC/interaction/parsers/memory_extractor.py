from __future__ import annotations

from typing import List

from SLAC.interaction.io.schemas import (
    ExtractedConversation,
    OpenWebUIMessage,
    OpenWebUIRequest,
    OpenWebUIMemoryOverride,
)


class MemoryExtractor:
    """
    推荐规则：
    - 当前轮问题单独进入 current_messages
    - 先前轮次进入 memory.messages
    - 若上层已给 memory_override，则优先透传该 memory
    """

    def extract(self, req: OpenWebUIRequest) -> ExtractedConversation:
        if req.raw_messages:
            anchor_idx = self._find_anchor_index(req.raw_messages)
            current_messages = self._build_current_messages(req, anchor_idx)
            memory = self._build_memory(req, anchor_idx)
            query_text = self._resolve_query_text(req, current_messages)
            return ExtractedConversation(
                query_text=query_text,
                current_messages=current_messages,
                memory=memory,
                meta={
                    "anchor_index": anchor_idx,
                    "memory_from_override": req.memory_override is not None,
                    "raw_message_count": len(req.raw_messages),
                },
            )

        current_messages = [OpenWebUIMessage(role="user", content=req.query_text)]
        memory = req.memory_override
        return ExtractedConversation(
            query_text=req.query_text,
            current_messages=current_messages,
            memory=memory,
            meta={
                "anchor_index": None,
                "memory_from_override": req.memory_override is not None,
                "raw_message_count": 0,
            },
        )

    def _find_anchor_index(self, messages: List[OpenWebUIMessage]) -> int:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.role == "user" and msg.content.strip():
                return idx
        return max(0, len(messages) - 1)

    def _build_current_messages(
        self,
        req: OpenWebUIRequest,
        anchor_idx: int,
    ) -> List[OpenWebUIMessage]:
        if not req.raw_messages:
            return [OpenWebUIMessage(role="user", content=req.query_text)]

        tail = req.raw_messages[anchor_idx:]
        if tail:
            return [
                OpenWebUIMessage(
                    role=msg.role,
                    content=msg.content,
                    name=msg.name,
                    meta=dict(msg.meta),
                )
                for msg in tail
                if msg.content.strip()
            ]

        return [OpenWebUIMessage(role="user", content=req.query_text)]

    def _build_memory(
        self,
        req: OpenWebUIRequest,
        anchor_idx: int,
    ) -> OpenWebUIMemoryOverride | None:
        if req.memory_override is not None:
            return req.memory_override

        if not req.raw_messages:
            return None

        history = req.raw_messages[:anchor_idx]
        if not history:
            return None

        return OpenWebUIMemoryOverride(
            source="openwebui_current_session",
            messages=[
                OpenWebUIMessage(
                    role=msg.role,
                    content=msg.content,
                    name=msg.name,
                    meta=dict(msg.meta),
                )
                for msg in history
                if msg.content.strip()
            ],
            summary_text=None,
        )

    def _resolve_query_text(
        self,
        req: OpenWebUIRequest,
        current_messages: List[OpenWebUIMessage],
    ) -> str:
        if req.query_text.strip():
            return req.query_text.strip()

        for msg in current_messages:
            if msg.role == "user" and msg.content.strip():
                return msg.content.strip()

        return req.query_text.strip()