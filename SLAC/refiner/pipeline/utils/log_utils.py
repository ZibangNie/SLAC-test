"""
Logging utilities for refiner pipeline.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, Path]


class JsonLineFormatter(logging.Formatter):
    """
    JSONL formatter for structured run logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "lineno": record.lineno,
        }

        if hasattr(record, "doc_id"):
            payload["doc_id"] = getattr(record, "doc_id")
        if hasattr(record, "source_path"):
            payload["source_path"] = getattr(record, "source_path")
        if hasattr(record, "event"):
            payload["event"] = getattr(record, "event")
        if hasattr(record, "extra_json") and isinstance(getattr(record, "extra_json"), dict):
            payload.update(getattr(record, "extra_json"))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def ensure_parent(path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(
    name: str,
    *,
    log_file: Optional[PathLike] = None,
    level: int = logging.INFO,
    console: bool = True,
    jsonl_file: bool = False,
) -> logging.Logger:
    """
    Create or reset a logger with optional console/file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs across repeated runs.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(sh)

    if log_file is not None:
        fp = ensure_parent(log_file)
        fh = logging.FileHandler(fp, mode="w", encoding="utf-8")
        fh.setLevel(level)
        if jsonl_file:
            fh.setFormatter(JsonLineFormatter())
        else:
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
        logger.addHandler(fh)

    return logger


def log_doc_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    doc_id: Optional[str] = None,
    source_path: Optional[str] = None,
    event: Optional[str] = None,
    extra_json: Optional[Dict[str, Any]] = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "doc_id": doc_id,
            "source_path": source_path,
            "event": event,
            "extra_json": extra_json or {},
        },
    )