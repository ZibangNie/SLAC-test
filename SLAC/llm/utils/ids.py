from __future__ import annotations

import uuid


def make_request_id(prefix: str = "req") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
