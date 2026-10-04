from __future__ import annotations

from hashlib import sha256
from typing import Dict, List

from SLAC.llm.io.schemas import ChatMessage, LLMRequest
from SLAC.llm.memory.merge import merge_memory_into_messages
from SLAC.llm.service.renderers import (
    SOURCE_RENDER_POLICY, SOURCE_RENDERER_VERSION, render_evidence_block,
)


def _validate_source_evidence_budget(req: LLMRequest, evidence_block: str) -> None:
    """Bind a trusted caller's count to these exact bytes; do not recount it.

    This receipt certifies neither counter honesty nor source provenance and
    covers the evidence block only, not the complete provider context.
    """
    receipt = req.meta.get("source_evidence_budget") if isinstance(req.meta, dict) else None
    fields = {"schema", "renderer_version", "rendered_sha256", "counter_version", "count", "max_tokens"}
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError("source_evidence_budget must contain exactly the six budget fields")
    if receipt["schema"] != "slac-source-evidence-budget-v1":
        raise ValueError("source_evidence_budget schema differs")
    if receipt["renderer_version"] != SOURCE_RENDERER_VERSION:
        raise ValueError("source_evidence_budget renderer_version differs")
    if receipt["rendered_sha256"] != sha256(evidence_block.encode("utf-8")).hexdigest():
        raise ValueError("source_evidence_budget rendered_sha256 differs from actual evidence block")
    version = receipt["counter_version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("source_evidence_budget counter_version must be a nonblank string")
    for field in ("count", "max_tokens"):
        value = receipt[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"source_evidence_budget {field} must be a nonnegative integer")
    if receipt["count"] > receipt["max_tokens"]:
        raise ValueError("source_evidence_budget count exceeds max_tokens")


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
    if (isinstance(req.meta, dict) and "source_evidence_budget" in req.meta
            and evidence_policy != SOURCE_RENDER_POLICY):
        raise ValueError("source_evidence_budget requires explicit source evidence_render_policy")
    if evidence_policy not in ("append_as_context_block", SOURCE_RENDER_POLICY):
        raise ValueError(f"unsupported evidence_render_policy: {evidence_policy!r}")

    source_mode = evidence_policy == SOURCE_RENDER_POLICY
    evidence_block = render_evidence_block(req.evidence, preserve_source_text=source_mode)
    if source_mode:
        _validate_source_evidence_budget(req, evidence_block)
    if evidence_block:
        merged.append({"role": "user", "content": evidence_block})

    return {
        "model": req.model_name,
        "messages": merged,
        "temperature": req.generation_config.temperature,
        "top_p": req.generation_config.top_p,
        "max_tokens": req.generation_config.max_tokens,
    }
