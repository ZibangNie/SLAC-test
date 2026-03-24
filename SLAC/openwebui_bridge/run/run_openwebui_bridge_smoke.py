from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..service.openwebui_bridge_service import OpenWebUIBridgeService


class MockService(OpenWebUIBridgeService):
    async def handle(self, raw_payload: dict) -> dict:
        return {
            "schema_version": "slac_openwebui_response_v1",
            "record_type": "openwebui_response",
            "status": "ok",
            "request_id": raw_payload.get("request_id", "smoke_req_001"),
            "session_id": (raw_payload.get("__metadata__") or {}).get("chat_id") or raw_payload.get("session_id"),
            "query_id": raw_payload.get("query_id", "smoke_qt_001"),
            "answer_text": "这是一条 smoke test 返回，用于验证 OpenWebUI bridge 的请求解析与响应适配流程。",
            "display_messages": [
                {
                    "role": "assistant",
                    "content": "这是一条 smoke test 返回，用于验证 OpenWebUI bridge 的请求解析与响应适配流程。",
                }
            ],
            "trace_summary": {"num_evidence_selected": 0, "llm_status": "mock_ok"},
            "meta": {"mode": "mock"},
        }


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=False, help="Path to raw OpenWebUI payload JSON")
    args = parser.parse_args()

    if args.payload:
        raw = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    else:
        raw = {
            "request_id": "ow_req_smoke_001",
            "session_id": "chat_smoke_001",
            "query_id": "qt_smoke_001",
            "messages": [
                {"role": "user", "content": "请解释桥梁上部结构设计要求。"},
            ],
            "context": {"domain": "rail", "source_type": "mixed"},
        }

    service = MockService(integration_runner="/tmp/unused.py")
    resp = await service.handle(raw)
    print(json.dumps(resp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
