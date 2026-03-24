"""
title: SLAC Remote Pipe
version: 1.2.0
author: OpenAI
required_open_webui_version: 0.6.0
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        PIPE_ID: str = Field(default="slac_remote")
        PIPE_NAME: str = Field(default="SLAC Remote")
        SLAC_API_BASE: str = Field(default="https://uu180896-a9d0-fdd5aeb5.westb.seetacloud.com:8443")
        API_TOKEN: str = Field(default="")
        REQUEST_TIMEOUT_S: int = Field(default=1800)

        # OpenWebUI 容器内部基地址
        OPENWEBUI_INTERNAL_BASE: str = Field(default="http://127.0.0.1:8080")

        # OpenWebUI 自身 API Key（到 Settings > Account 里复制）
        OPENWEBUI_API_KEY: str = Field(default="")

        RETURN_FULL_UI_RESPONSE: bool = Field(default=False)
        DEBUG: bool = Field(default=True)

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": self.valves.PIPE_ID, "name": self.valves.PIPE_NAME}]

    def _slac_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.valves.API_TOKEN.strip():
            headers["Authorization"] = f"Bearer {self.valves.API_TOKEN.strip()}"
        return headers

    def _openwebui_headers(self) -> Dict[str, str]:
        headers = {}
        if self.valves.OPENWEBUI_API_KEY.strip():
            headers["Authorization"] = f"Bearer {self.valves.OPENWEBUI_API_KEY.strip()}"
        return headers

    def _derive_session_id(self, body: Dict[str, Any], __metadata__: Optional[dict]) -> str:
        return (
            body.get("chat_id")
            or body.get("id")
            or (__metadata__ or {}).get("chat_id")
            or "sess_openwebui_default"
        )

    def _resolve_relative_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return urljoin(self.valves.OPENWEBUI_INTERNAL_BASE.rstrip("/") + "/", url.lstrip("/"))

    def _try_read_local_path(self, path_str: str) -> Optional[bytes]:
        p = Path(path_str)
        if p.exists() and p.is_file():
            return p.read_bytes()
        return None

    def _extract_file_id(self, item: Dict[str, Any]) -> Optional[str]:
        file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}
        for key in ("file_id", "id"):
            if item.get(key):
                return str(item[key])
        for key in ("file_id", "id"):
            if file_obj.get(key):
                return str(file_obj[key])
        return None

    def _download_from_openwebui_file_api(self, file_id: str) -> Optional[bytes]:
        # 社区常用路径：/api/v1/files/{file_id}/content
        candidate_paths = [
            f"/api/v1/files/{file_id}/content",
            f"/api/v1/files/{file_id}",
        ]
        for p in candidate_paths:
            url = self._resolve_relative_url(p)
            try:
                resp = requests.get(
                    url,
                    headers=self._openwebui_headers(),
                    timeout=120,
                )
                if resp.status_code == 200 and resp.content:
                    return resp.content
            except Exception:
                pass
        return None

    def _normalize_upload_candidates(self, files_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for item in files_meta or []:
            if not isinstance(item, dict):
                continue

            file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}
            name = (
                item.get("name")
                or file_obj.get("name")
                or item.get("filename")
                or file_obj.get("filename")
                or "uploaded_file"
            )
            content_type = item.get("content_type") or item.get("type") or file_obj.get("content_type")

            # 0) 优先尝试 file_id / id -> 从 OpenWebUI API 下载
            file_id = self._extract_file_id(item)
            if file_id:
                raw = self._download_from_openwebui_file_api(file_id)
                if raw is not None:
                    normalized.append(
                        {
                            "name": name,
                            "content": raw,
                            "content_type": content_type,
                            "source": "openwebui_file_api",
                            "file_id": file_id,
                        }
                    )
                    continue

            # 1) 直接本地路径
            hit = False
            for k in ("path", "local_path"):
                if item.get(k):
                    raw = self._try_read_local_path(str(item[k]))
                    if raw is not None:
                        normalized.append(
                            {
                                "name": name,
                                "content": raw,
                                "content_type": content_type,
                                "source": "local_path",
                            }
                        )
                        hit = True
                        break
            if hit:
                continue

            # 2) 嵌套 file.path
            if file_obj.get("path"):
                raw = self._try_read_local_path(str(file_obj["path"]))
                if raw is not None:
                    normalized.append(
                        {
                            "name": name,
                            "content": raw,
                            "content_type": content_type,
                            "source": "nested_local_path",
                        }
                    )
                    continue

            # 3) base64
            b64 = item.get("content_base64") or file_obj.get("content_base64")
            if b64:
                try:
                    raw = base64.b64decode(b64)
                    normalized.append(
                        {
                            "name": name,
                            "content": raw,
                            "content_type": content_type,
                            "source": "base64",
                        }
                    )
                    continue
                except Exception:
                    pass

            # 4) URL 下载
            url = item.get("url") or item.get("file_url") or file_obj.get("url")
            if url:
                resolved = self._resolve_relative_url(str(url))
                try:
                    resp = requests.get(
                        resolved,
                        headers=self._openwebui_headers(),
                        timeout=120,
                    )
                    resp.raise_for_status()
                    normalized.append(
                        {
                            "name": name,
                            "content": resp.content,
                            "content_type": content_type,
                            "source": "url",
                        }
                    )
                    continue
                except Exception:
                    pass

        return normalized

    def _upload_files_if_needed(
        self,
        *,
        session_id: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        files_meta = body.get("files") or []
        if not files_meta:
            return {"uploaded": False, "saved_file_refs": []}

        uploadables = self._normalize_upload_candidates(files_meta)
        if not uploadables:
            return {
                "uploaded": False,
                "saved_file_refs": [],
                "warning": "body.files 存在，但 Pipe 未解析出可上传的文件字节。",
                "files_meta_preview": files_meta[:3],
            }

        multipart = []
        for item in uploadables:
            multipart.append(
                (
                    "files",
                    (item["name"], item["content"], item.get("content_type") or "application/octet-stream"),
                )
            )

        url = self.valves.SLAC_API_BASE.rstrip("/") + f"/sessions/{session_id}/files"
        resp = requests.post(
            url,
            files=multipart,
            timeout=self.valves.REQUEST_TIMEOUT_S,
            headers={k: v for k, v in self._slac_headers().items() if k != "Content-Type"},
        )
        resp.raise_for_status()
        result = resp.json()
        result["uploaded"] = True
        result["num_uploadables"] = len(uploadables)
        result["upload_sources"] = [x.get("source") for x in uploadables]
        return result

    async def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[dict] = None,
        __request__=None,
        __metadata__: Optional[dict] = None,
        __event_emitter__=None,
        **kwargs,
    ):
        session_id = self._derive_session_id(body or {}, __metadata__)

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

        upload_result = {}
        try:
            upload_result = self._upload_files_if_needed(session_id=session_id, body=body or {})
        except Exception as e:
            return f"[SLAC Remote Pipe Upload Error] {e}"

        payload = dict(body or {})
        payload["session_id"] = session_id
        payload["__user__"] = __user__
        payload["__metadata__"] = __metadata__ or {}

        payload.setdefault("openwebui_meta", {})
        if __metadata__:
            payload["openwebui_meta"].update(__metadata__)

        payload.setdefault("meta", {})
        if __user__:
            payload["meta"]["openwebui_user"] = __user__
        if upload_result:
            payload["meta"]["upload_result"] = upload_result

        if self.valves.DEBUG:
            payload["meta"]["pipe_debug"] = {
                "files_meta_preview": (body or {}).get("files", [])[:3],
                "has_files": bool((body or {}).get("files")),
                "session_id": session_id,
            }

        url = self.valves.SLAC_API_BASE.rstrip("/") + "/chat"

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._slac_headers(),
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

            answer_text = data.get("answer_text") or data.get("text") or data.get("content") or ""
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