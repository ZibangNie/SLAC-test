from __future__ import annotations

from typing import Dict, List

from SLAC.llm.io.schemas import ChatMessage, LLMRequest
from SLAC.llm.memory.merge import merge_memory_into_messages
from SLAC.llm.service.renderers import render_evidence_block


def compile_provider_payload(req: LLMRequest) -> Dict:
    current_messages = req.messages[:]
    if not current_messages and req.prompt:
        current_messages = [ChatMessage(role="user", content=req.prompt)]

    merged = merge_memory_into_messages(
        system_prompt=req.system_prompt,
        memory=req.memory,
        current_messages=current_messages,
    )

    evidence_policy = req.options.get("evidence_render_policy", "append_as_context_block")
    if evidence_policy != "append_as_context_block":
        raise ValueError(f"unsupported evidence_render_policy: {evidence_policy!r}")

    evidence_block = render_evidence_block(req.evidence)
    if evidence_block:
        merged.append({"role": "user", "content": evidence_block})

    return {
        "model": req.model_name,
        "messages": merged,
        "temperature": req.generation_config.temperature,
        "top_p": req.generation_config.top_p,
        "max_tokens": req.generation_config.max_tokens,
    }
