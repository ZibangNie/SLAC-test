from __future__ import annotations

import re
import uuid


def safe_stem(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unknown"

    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-\.]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def make_request_id(prefix: str = "ow_req") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_query_id(request_id: str, prefix: str = "q") -> str:
    base = safe_stem(request_id)
    return f"{prefix}_{base}"