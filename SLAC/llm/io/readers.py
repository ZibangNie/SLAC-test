from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from SLAC.llm.io.schemas import LLMRequest
from SLAC.llm.utils.io import read_jsonl, read_json


def read_llm_requests_jsonl(path: str | Path) -> list[LLMRequest]:
    records = read_jsonl(path)
    return [LLMRequest.from_dict(item) for item in records]


def iter_llm_requests_jsonl(path: str | Path) -> Iterator[LLMRequest]:
    for item in read_jsonl(path):
        yield LLMRequest.from_dict(item)


def read_single_llm_request_json(path: str | Path) -> LLMRequest:
    payload = read_json(path)
    if isinstance(payload, list):
        raise TypeError("single request json file must contain an object, not a list")
    return LLMRequest.from_dict(payload)
