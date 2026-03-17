from __future__ import annotations

import os
from typing import Any, Dict

from .base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, api_base: str, api_key_env: str, timeout_s: int = 120):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"environment variable {api_key_env!r} is not set; cannot call OpenAI-compatible provider"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import-time dependency check
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleClient. "
                "Please install it before running the LLM module."
            ) from exc

        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout_s)

    def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.chat.completions.create(**payload)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return dict(response)
