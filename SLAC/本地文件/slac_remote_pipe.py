"""
title: SLAC Remote Pipe
version: 1.0.0
author: OpenAI
required_open_webui_version: 0.6.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        PIPE_ID: str = Field(default="slac_remote")
        PIPE_NAME: str = Field(default="SLAC Remote")
        SLAC_API_BASE: str = Field(default="http://YOUR_SERVER_IP:18080")
        API_TOKEN: str = Field(default="")
        REQUEST_TIMEOUT_S: int = Field(default=600)
        RETURN_FULL_UI_RESPONSE: bool = Field(default=False)
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": self.valves.PIPE_ID, "name": self.valves.PIPE_NAME}]

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        __request__=None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ):
        if __event_emitter__ is not None:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "SLAC 远程后端处理中...", "done": False},
                    }
                )
            except Exception:
                pass

        payload = dict(body or {})
        payload["__user__"] = __user__
        payload["__metadata__"] = __metadata__ or {}

        payload.setdefault("openwebui_meta", {})
        if __metadata__:
            payload["openwebui_meta"].update(__metadata__)

        if __user__:
            payload.setdefault("meta", {})
            payload["meta"]["openwebui_user"] = __user__

        headers = {
            "Content-Type": "application/json",
        }
        if self.valves.API_TOKEN.strip():
            headers["Authorization"] = f"Bearer {self.valves.API_TOKEN.strip()}"

        url = self.valves.SLAC_API_BASE.rstrip("/") + "/chat"

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.valves.REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()

            if __event_emitter__ is not None:
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"SLAC 远程后端处理完成: {data.get('status', 'ok')}",
                                "done": True,
                            },
                        }
                    )
                except Exception:
                    pass

            if self.valves.RETURN_FULL_UI_RESPONSE:
                return data

            answer_text = (
                data.get("answer_text")
                or data.get("text")
                or data.get("content")
                or ""
            )
            if answer_text:
                return answer_text

            return "[SLAC] 远端服务已返回响应，但未找到 answer_text。"

        except requests.HTTPError as e:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            return f"[SLAC Remote Pipe HTTPError] {e} | detail={detail}"

        except Exception as e:
            return f"[SLAC Remote Pipe Error] {e}"