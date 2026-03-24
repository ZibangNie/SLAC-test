from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _detect_lang_hint(query_text: str) -> str:
    for ch in query_text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


class UploadIngestService:
    """
    会话级文件落盘 + Refiner/Build/Query 运行编排器
    """

    def __init__(
        self,
        *,
        project_root: str,
        work_root: str,
        python_bin: str,
        refiner_config: str,
        retrieval_config: str,
        reranker_config: str,
        bilingual_terms_path: str,
        default_domain: str = "rail",
        debug: bool = False,
    ):
        self.project_root = Path(project_root).resolve()
        self.work_root = Path(work_root).resolve()
        self.python_bin = python_bin
        self.refiner_config = refiner_config
        self.retrieval_config = retrieval_config
        self.reranker_config = reranker_config
        self.bilingual_terms_path = bilingual_terms_path
        self.default_domain = default_domain
        self.debug = debug

        _ensure_dir(self.work_root)

    # ----------------------------
    # session state
    # ----------------------------
    def _session_dir(self, session_id: str) -> Path:
        return self.work_root / "sessions" / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session_state.json"

    def _load_state(self, session_id: str) -> Dict[str, Any]:
        return _read_json(
            self._state_path(session_id),
            {
                "session_id": session_id,
                "files": [],
                "needs_rebuild": False,
                "active_asset_version": None,
                "latest_query_runs": {},
                "meta": {},
            },
        )

    def _save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        _write_json(self._state_path(session_id), state)

    # ----------------------------
    # upload
    # ----------------------------
    def save_uploaded_files(
        self,
        session_id: str,
        files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        files: [{"name": str, "content": bytes, "content_type": str|None}, ...]
        """
        session_dir = self._session_dir(session_id)
        upload_dir = _ensure_dir(session_dir / "uploads" / "raw")
        state = self._load_state(session_id)

        existing_by_sha = {item["sha256"]: item for item in state.get("files", [])}
        saved_refs: List[Dict[str, Any]] = []
        changed = False

        for item in files:
            name = item["name"]
            content = item["content"]
            content_type = item.get("content_type")

            sha = _sha256_bytes(content)
            ext = Path(name).suffix
            stored_name = f"{sha}{ext}" if ext else sha
            stored_path = upload_dir / stored_name

            if sha not in existing_by_sha:
                with stored_path.open("wb") as f:
                    f.write(content)

                record = {
                    "file_id": f"rf_{sha[:12]}",
                    "original_name": name,
                    "stored_name": stored_name,
                    "stored_path": str(stored_path),
                    "sha256": sha,
                    "size_bytes": len(content),
                    "content_type": content_type,
                    "uploaded_at": _now_ts(),
                }
                state["files"].append(record)
                existing_by_sha[sha] = record
                changed = True

            saved_refs.append(
                {
                    "file_id": existing_by_sha[sha]["file_id"],
                    "sha256": sha,
                    "name": name,
                    "stored_path": existing_by_sha[sha]["stored_path"],
                }
            )

        if changed:
            state["needs_rebuild"] = True
            state["active_asset_version"] = None

        self._save_state(session_id, state)

        return {
            "session_id": session_id,
            "saved_file_refs": saved_refs,
            "needs_rebuild": state["needs_rebuild"],
            "num_files_total": len(state.get("files", [])),
        }

    # ----------------------------
    # runners
    # ----------------------------
    def _run_cmd(self, cmd: List[str], *, cwd: Optional[str] = None, stage: str = "unknown") -> Dict[str, Any]:
        proc = subprocess.run(
            cmd,
            cwd=cwd or str(self.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"[{stage}] command failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    def _collect_doc_scope_from_stage2(self, stage2_dir: Path) -> List[str]:
        doc_catalog = stage2_dir / "doc_catalog.jsonl"
        rows = _read_jsonl(doc_catalog)
        doc_ids: List[str] = []
        for row in rows:
            doc_id = row.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                doc_ids.append(doc_id)
        return doc_ids

    def _ensure_stage2_exists(self, stage2_dir: Path) -> None:
        required = [
            stage2_dir / "refined_chunks.jsonl",
            stage2_dir / "leaf_records.jsonl",
            stage2_dir / "doc_catalog.jsonl",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Refiner stage2 output missing: {missing}")

    def _ensure_build_exists(self, build_dir: Path) -> None:
        required = [
            build_dir / "meta" / "chunk_lookup.jsonl",
            build_dir / "meta" / "leaf_lookup.jsonl",
            build_dir / "summaries" / "run_build_index_summary.json",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Retrieval build output missing: {missing}")

    def _rebuild_session_assets(self, session_id: str, state: Dict[str, Any], domain: str) -> Dict[str, Any]:
        session_dir = self._session_dir(session_id)
        asset_version = f"assets_{_now_ts()}"
        asset_root = _ensure_dir(session_dir / "asset_versions" / asset_version)

        raw_docs_dir = _ensure_dir(asset_root / "raw_docs")
        for item in state.get("files", []):
            src = Path(item["stored_path"])
            dst = raw_docs_dir / item["original_name"]
            if not dst.exists():
                shutil.copy2(src, dst)

        refiner_out = asset_root / "refiner_run"
        build_out = asset_root / "retrieval_build"

        # Refiner full pipeline
        refiner_cmd = [
            self.python_bin,
            "-m",
            "SLAC.refiner.pipeline.run.run_refiner_pipeline",
            "--config",
            self.refiner_config,
            "--input_paths",
            str(raw_docs_dir),
            "--output_dir",
            str(refiner_out),
            "--recursive",
            "--dump_structure_doc",
            "--dump_chunk0_json",
            "--dump_intermediate_records",
        ]
        self._run_cmd(refiner_cmd, stage="refiner_full")

        stage2_dir = refiner_out / "stage2_refiner_infer"
        self._ensure_stage2_exists(stage2_dir)

        # Retrieval build
        build_cmd = [
            self.python_bin,
            "-m",
            "SLAC.retrieval.run.run_build_index",
            "--config",
            self.retrieval_config,
            "--refined_chunks_jsonl",
            str(stage2_dir / "refined_chunks.jsonl"),
            "--leaf_records_jsonl",
            str(stage2_dir / "leaf_records.jsonl"),
            "--doc_catalog_jsonl",
            str(stage2_dir / "doc_catalog.jsonl"),
            "--output_dir",
            str(build_out),
        ]
        self._run_cmd(build_cmd, stage="retrieval_build")
        self._ensure_build_exists(build_out)

        doc_scope = self._collect_doc_scope_from_stage2(stage2_dir)

        state["needs_rebuild"] = False
        state["active_asset_version"] = asset_version
        state["active_assets"] = {
            "asset_version": asset_version,
            "raw_docs_dir": str(raw_docs_dir),
            "refiner_out_dir": str(refiner_out),
            "stage2_dir": str(stage2_dir),
            "retrieval_build_dir": str(build_out),
            "doc_scope": doc_scope,
            "domain": domain,
            "built_at": _now_ts(),
        }
        self._save_state(session_id, state)
        return state["active_assets"]

    def ensure_session_assets(self, session_id: str, domain: Optional[str] = None) -> Dict[str, Any]:
        state = self._load_state(session_id)
        domain = domain or self.default_domain

        if not state.get("files"):
            raise RuntimeError(f"session={session_id} has no uploaded files")

        if state.get("needs_rebuild") or not state.get("active_assets"):
            return self._rebuild_session_assets(session_id, state, domain)

        return state["active_assets"]

    def prepare_query_runs(
        self,
        *,
        session_id: str,
        query_id: str,
        query_text: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        assets = self.ensure_session_assets(session_id, domain=domain)
        session_dir = self._session_dir(session_id)
        query_root = _ensure_dir(session_dir / "query_runs" / query_id)

        retrieval_run_dir = query_root / "retrieval_run"
        reranker_run_dir = query_root / "reranker_run"
        planner_cache_dir = _ensure_dir(query_root / "planner_cache")
        query_file = query_root / "queries.jsonl"

        query_obj = {
            "query_id": query_id,
            "query": query_text,
            "lang_hint": _detect_lang_hint(query_text),
            "domain_hint": domain or assets.get("domain") or self.default_domain,
        }
        with query_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(query_obj, ensure_ascii=False) + "\n")

        # Retrieval online
        retrieval_cmd = [
            self.python_bin,
            "-m",
            "SLAC.retrieval.run.run_retrieval_pipeline",
            "--config",
            self.retrieval_config,
            "--retrieval_build_dir",
            assets["retrieval_build_dir"],
            "--queries_jsonl",
            str(query_file),
            "--output_dir",
            str(retrieval_run_dir),
            "--planner_cache_dir",
            str(planner_cache_dir),
            "--bilingual_terms_path",
            self.bilingual_terms_path,
        ]
        self._run_cmd(retrieval_cmd, stage="retrieval_online")

        # Reranker pipeline
        reranker_cmd = [
            self.python_bin,
            "SLAC/reranker/run/run_reranker_pipeline.py",
            "--config",
            self.reranker_config,
            "--input_path",
            str(retrieval_run_dir),
            "--output_dir",
            str(reranker_run_dir),
        ]
        self._run_cmd(reranker_cmd, cwd=str(self.project_root), stage="reranker_pipeline")

        state = self._load_state(session_id)
        state.setdefault("latest_query_runs", {})
        state["latest_query_runs"][query_id] = {
            "query_text": query_text,
            "retrieval_run_dir": str(retrieval_run_dir),
            "reranker_run_dir": str(reranker_run_dir),
            "planner_cache_dir": str(planner_cache_dir),
            "prepared_at": _now_ts(),
        }
        self._save_state(session_id, state)

        return {
            "session_id": session_id,
            "query_id": query_id,
            "query_text": query_text,
            "retrieval_run_dir": str(retrieval_run_dir),
            "reranker_run_dir": str(reranker_run_dir),
            "doc_scope": assets.get("doc_scope", []),
            "asset_version": assets.get("asset_version"),
        }