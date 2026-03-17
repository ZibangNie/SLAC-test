from __future__ import annotations

from SLAC.llm.clients.base import BaseLLMClient
from SLAC.llm.clients.openai_compatible import OpenAICompatibleClient


def get_client(provider: str, *, api_base: str, api_key_env: str, timeout_s: int) -> BaseLLMClient:
    if provider == "openai_compatible":
        return OpenAICompatibleClient(
            api_base=api_base,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unsupported provider: {provider!r}")
