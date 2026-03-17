from __future__ import annotations

from SLAC.integration.utils.io import safe_stem


def resolve_query_key(query_id: str | None, request_id: str) -> str:
    return safe_stem(query_id or request_id)