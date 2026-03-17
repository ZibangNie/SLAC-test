from __future__ import annotations

from typing import Any


class LLMAdapter:
    def __init__(self, llm_service: Any | None = None) -> None:
        if llm_service is None:
            from SLAC.llm.service.llm_service import LLMService

            llm_service = LLMService()
        self.llm_service = llm_service

    def invoke(self, llm_request: Any) -> Any:
        return self.llm_service.invoke(llm_request)