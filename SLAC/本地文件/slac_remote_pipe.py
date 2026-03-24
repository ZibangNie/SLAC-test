"""
title: SLAC Remote Pipe
version: 1.5.0
author: OpenAI
required_open_webui_version: 0.6.0
"""

from __future__ import annotations

import base64
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        PIPE_ID: str = Field(default="slac_remote")
        PIPE_NAME: str = Field(default="SLAC Remote")

        SLAC_API_BASE: str = Field(
            default="https://uu180896-a9d0-fdd5aeb5.westb.seetacloud.com:8443"
        )
        API_TOKEN: str = Field(default="")
        REQUEST_TIMEOUT_S: int = Field(default=1800)

        OPENWEBUI_INTERNAL_BASE: str = Field(default="http://127.0.0.1:8080")
        OPENWEBUI_API_KEY: str = Field(default="sk-569365f1716c4e83add630c5e8d1a3b0")

        RETURN_FULL_UI_RESPONSE: bool = Field(default=False)
        DEBUG: bool = Field(default=True)

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": self.valves.PIPE_ID, "name": self.valves.PIPE_NAME}]

    def _log(self, *parts):
        if self.valves.DEBUG:
            try:
                print("[SLAC_REMOTE_PIPE]", *parts, flush=True)
            except Exception:
                pass

    def _slac_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.valves.API_TOKEN.strip():
            headers["Authorization"] = f"Bearer {self.valves.API_TOKEN.strip()}"
        return headers

    def _openwebui_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.valves.OPENWEBUI_API_KEY.strip():
            headers["Authorization"] = f"Bearer {self.valves.OPENWEBUI_API_KEY.strip()}"
        return headers

    def _derive_session_id(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[dict],
    ) -> str:
        return (
            body.get("chat_id")
            or body.get("id")
            or (__metadata__ or {}).get("chat_id")
            or "sess_openwebui_default"
        )

    def _resolve_relative_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return urljoin(
            self.valves.OPENWEBUI_INTERNAL_BASE.rstrip("/") + "/",
            url.lstrip("/"),
        )

    def _try_read_local_path(self, path_str: str) -> Optional[bytes]:
        try:
            p = Path(path_str)
            if p.exists() and p.is_file():
                self._log("read local path ok", str(p))
                return p.read_bytes()
            else:
                self._log("local path missing", str(p))
        except Exception as e:
            self._log("local path exception", path_str, repr(e))
        return None

    def _safe_preview(self, obj: Any, max_len: int = 3000) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
        if len(s) > max_len:
            s = s[:max_len] + "...<truncated>"
        return s

    def _collect_files_meta(
        self,
        body: Dict[str, Any],
        __metadata__: Optional[dict],
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []

        top_files = body.get("files")
        if isinstance(top_files, list):
            for x in top_files:
                if isinstance(x, dict):
                    collected.append(x)

        messages = body.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_files = msg.get("files")
            if isinstance(msg_files, list):
                for x in msg_files:
                    if isinstance(x, dict):
                        collected.append(x)

            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "file":
                        collected.append(part)

        if isinstance(__metadata__, dict):
            meta_files = __metadata__.get("files")
            if isinstance(meta_files, list):
                for x in meta_files:
                    if isinstance(x, dict):
                        collected.append(x)

            attachments = __metadata__.get("attachments")
            if isinstance(attachments, list):
                for x in attachments:
                    if isinstance(x, dict):
                        collected.append(x)

        return collected

    def _dedupe_files_meta(
        self, files_meta: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()

        for item in files_meta:
            if not isinstance(item, dict):
                continue

            file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}

            fid = (
                item.get("file_id")
                or item.get("id")
                or file_obj.get("file_id")
                or file_obj.get("id")
                or item.get("name")
                or file_obj.get("name")
            )

            key = str(fid)
            if key in seen:
                continue

            seen.add(key)
            out.append(item)

        return out

    def _extract_file_id(self, item: Dict[str, Any]) -> Optional[str]:
        file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}

        for key in ("file_id", "id"):
            if item.get(key):
                return str(item[key])

        for key in ("file_id", "id"):
            if file_obj.get(key):
                return str(file_obj[key])

        return None

    def _extract_name_and_type(self, item: Dict[str, Any]) -> Dict[str, Optional[str]]:
        file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}

        name = (
            item.get("name")
            or file_obj.get("name")
            or item.get("filename")
            or file_obj.get("filename")
            or "uploaded_file"
        )
        content_type = (
            item.get("content_type")
            or item.get("type")
            or file_obj.get("content_type")
            or file_obj.get("type")
        )
        return {
            "name": str(name),
            "content_type": str(content_type) if content_type else None,
        }

    def _download_using_metadata(self, meta: Dict[str, Any]) -> Optional[bytes]:
        if not isinstance(meta, dict):
            return None

        candidates: List[str] = []

        for key in ("path", "local_path", "url", "file_url"):
            if meta.get(key):
                candidates.append(str(meta[key]))

        file_obj = meta.get("file") if isinstance(meta.get("file"), dict) else {}
        for key in ("path", "local_path", "url", "file_url"):
            if file_obj.get(key):
                candidates.append(str(file_obj[key]))

        for c in candidates:
            raw = self._try_read_local_path(c)
            if raw is not None:
                return raw

        for c in candidates:
            if c.startswith("http://") or c.startswith("https://") or c.startswith("/"):
                resolved = self._resolve_relative_url(c)
                try:
                    self._log("try metadata url", resolved)
                    resp = requests.get(
                        resolved,
                        headers=self._openwebui_headers(),
                        timeout=10,
                    )
                    resp.raise_for_status()
                    return resp.content
                except Exception as e:
                    self._log("metadata url exception", resolved, repr(e))

        return None

    def _download_from_openwebui_file_api(self, file_id: str) -> Optional[bytes]:
        headers = self._openwebui_headers()

        candidate_urls = [
            self._resolve_relative_url(f"/api/v1/files/{file_id}/content"),
            self._resolve_relative_url(f"/api/v1/files/{file_id}"),
        ]

        for url in candidate_urls:
            try:
                self._log("try file api", url)
                resp = requests.get(url, headers=headers, timeout=10)

                if resp.status_code != 200:
                    self._log("file api non-200", resp.status_code, url)
                    continue

                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        meta = resp.json()
                        self._log(
                            "file api returned json meta", self._safe_preview(meta)
                        )
                        raw = self._download_using_metadata(meta)
                        if raw is not None:
                            return raw
                    except Exception as e:
                        self._log("file api json parse exception", repr(e))
                else:
                    if resp.content:
                        return resp.content
            except Exception as e:
                self._log("file api exception", file_id, repr(e))

        return None

    def _normalize_upload_candidates(
        self,
        files_meta: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for item in files_meta:
            if not isinstance(item, dict):
                continue

            info = self._extract_name_and_type(item)
            name = info["name"]
            content_type = info["content_type"]

            file_obj = item.get("file") if isinstance(item.get("file"), dict) else {}

            # 1) 优先直接读 path / local_path
            for key in ("path", "local_path"):
                if item.get(key):
                    raw = self._try_read_local_path(str(item[key]))
                    if raw is not None:
                        normalized.append(
                            {
                                "name": name,
                                "content": raw,
                                "content_type": content_type,
                                "source": "local_path",
                            }
                        )
                        break
            else:
                # 2) nested file.path / file.local_path
                nested_hit = False
                for key in ("path", "local_path"):
                    if file_obj.get(key):
                        raw = self._try_read_local_path(str(file_obj[key]))
                        if raw is not None:
                            normalized.append(
                                {
                                    "name": name,
                                    "content": raw,
                                    "content_type": content_type,
                                    "source": "nested_local_path",
                                }
                            )
                            nested_hit = True
                            break
                if nested_hit:
                    continue

                # 3) file_id -> OpenWebUI API
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

                # 4) base64
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
                    except Exception as e:
                        self._log("base64 decode exception", repr(e))

                # 5) URL
                url = item.get("url") or item.get("file_url") or file_obj.get("url")
                if url:
                    resolved = self._resolve_relative_url(str(url))
                    try:
                        self._log("try url", resolved)
                        resp = requests.get(
                            resolved,
                            headers=self._openwebui_headers(),
                            timeout=10,
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
                    except Exception as e:
                        self._log("url exception", resolved, repr(e))

        return normalized

    def _upload_files_if_needed(
        self,
        *,
        session_id: str,
        body: Dict[str, Any],
        __metadata__: Optional[dict],
    ) -> Dict[str, Any]:
        raw_files_meta = self._collect_files_meta(body, __metadata__)
        files_meta = self._dedupe_files_meta(raw_files_meta)

        self._log("body keys", list((body or {}).keys()))
        self._log(
            "__metadata__ keys",
            (
                list((__metadata__ or {}).keys())
                if isinstance(__metadata__, dict)
                else None
            ),
        )
        self._log("collected files_meta count", len(files_meta))
        self._log("files_meta preview", self._safe_preview(files_meta[:3]))

        if not files_meta:
            return {
                "uploaded": False,
                "saved_file_refs": [],
                "has_files_meta": False,
                "files_meta_preview": [],
            }

        uploadables = self._normalize_upload_candidates(files_meta)
        self._log("uploadables count", len(uploadables))
        self._log(
            "uploadables preview",
            self._safe_preview(
                [
                    {
                        "name": x.get("name"),
                        "source": x.get("source"),
                        "file_id": x.get("file_id"),
                        "content_len": len(x.get("content", b"")),
                    }
                    for x in uploadables[:3]
                ]
            ),
        )

        if not uploadables:
            return {
                "uploaded": False,
                "saved_file_refs": [],
                "has_files_meta": True,
                "warning": "发现文件元信息，但未解析出可上传字节。",
                "files_meta_preview": files_meta[:3],
            }

        multipart = []
        for item in uploadables:
            multipart.append(
                (
                    "files",
                    (
                        item["name"],
                        item["content"],
                        item.get("content_type") or "application/octet-stream",
                    ),
                )
            )

        url = self.valves.SLAC_API_BASE.rstrip("/") + f"/sessions/{session_id}/files"
        self._log("POST upload to", url, "num_files", len(multipart))

        resp = requests.post(
            url,
            files=multipart,
            timeout=self.valves.REQUEST_TIMEOUT_S,
            headers={
                k: v for k, v in self._slac_headers().items() if k != "Content-Type"
            },
        )
        resp.raise_for_status()

        result = resp.json()
        result["uploaded"] = True
        result["has_files_meta"] = True
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
        __task__=None,
        **kwargs,
    ):
        session_id = self._derive_session_id(body or {}, __metadata__)

        if __event_emitter__ is not None:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "SLAC 远程后端处理中...",
                            "done": False,
                        },
                    }
                )
            except Exception:
                pass

        upload_result: Dict[str, Any] = {}
        try:
            upload_result = self._upload_files_if_needed(
                session_id=session_id,
                body=body or {},
                __metadata__=__metadata__,
            )
        except Exception as e:
            self._log("upload exception", repr(e), traceback.format_exc())
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

        if __task__ is not None:
            payload["meta"]["openwebui_task"] = str(__task__)

        payload["meta"]["upload_result"] = upload_result

        if self.valves.DEBUG:
            payload["meta"]["pipe_debug"] = {
                "session_id": session_id,
                "task": str(__task__) if __task__ is not None else None,
                "files_meta_preview": self._dedupe_files_meta(
                    self._collect_files_meta(body or {}, __metadata__)
                )[:3],
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

            answer_text = (
                data.get("answer_text") or data.get("text") or data.get("content") or ""
            )
            if answer_text:
                return answer_text

            return "[SLAC] 远端服务已返回响应，但未找到 answer_text。"

        except requests.HTTPError as e:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            self._log("chat http error", repr(e), self._safe_preview(detail))
            return f"[SLAC Remote Pipe HTTPError] {e} | detail={detail}"
        except Exception as e:
            self._log("chat exception", repr(e), traceback.format_exc())
            return f"[SLAC Remote Pipe Error] {e}"
