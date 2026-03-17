from __future__ import annotations

from pathlib import Path
from typing import Iterable

from SLAC.llm.io.schemas import LLMAnswerResult, LLMRequest
from SLAC.llm.utils.io import append_jsonl, write_json, write_jsonl


def write_llm_requests_jsonl(path: str | Path, requests: Iterable[LLMRequest]) -> None:
    write_jsonl(path, [req.to_dict() for req in requests])


def write_llm_answers_jsonl(path: str | Path, answers: Iterable[LLMAnswerResult]) -> None:
    write_jsonl(path, [ans.to_dict() for ans in answers])


def append_llm_answer_jsonl(path: str | Path, answer: LLMAnswerResult) -> None:
    append_jsonl(path, answer.to_dict())


def write_summary_json(path: str | Path, payload: dict) -> None:
    write_json(path, payload)
