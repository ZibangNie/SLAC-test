from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DISABLE_TQDM", "true")


@dataclass
class BGERerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path: str | None = None
    device: str = "cuda"
    torch_dtype: str = "float16"
    batch_size: int = 8
    max_length: int = 1024
    trust_remote_code: bool = False


class BGERerankerV2M3:
    """
    Preferred backend: FlagEmbedding.FlagReranker
    Fallback backend: transformers AutoModelForSequenceClassification
    """

    def __init__(self, config: BGERerankerConfig):
        self.config = config
        self.backend: str | None = None
        self.model_ref: str = self._resolve_model_ref(config.model_path, config.model_name)

        self._flag_reranker = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

        self._init_backend()

    @staticmethod
    def _resolve_model_ref(model_path: str | None, model_name: str) -> str:
        if model_path:
            p = Path(model_path)
            if p.exists():
                return str(p)
        return model_name

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _init_backend(self) -> None:
        flag_error: Exception | None = None

        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            use_fp16 = (
                self.config.device.startswith("cuda")
                and self.config.torch_dtype.lower() in {"float16", "fp16", "half"}
            )
            self._flag_reranker = FlagReranker(self.model_ref, use_fp16=use_fp16)
            self.backend = "flagembedding"
            return
        except Exception as exc:  # pragma: no cover
            flag_error = exc

        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._torch = torch

            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                self._device = self.config.device
            else:
                self._device = "cpu"

            dtype = None
            dtype_name = self.config.torch_dtype.lower()
            if self._device.startswith("cuda"):
                if dtype_name in {"float16", "fp16", "half"}:
                    dtype = torch.float16
                elif dtype_name in {"bfloat16", "bf16"}:
                    dtype = torch.bfloat16

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_ref,
                trust_remote_code=self.config.trust_remote_code,
            )

            model_kwargs = {}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_ref,
                trust_remote_code=self.config.trust_remote_code,
                **model_kwargs,
            )
            self._model.eval()
            self._model.to(self._device)

            self.backend = "transformers"
            return
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "failed to initialize reranker backend. "
                f"FlagEmbedding error: {repr(flag_error)} ; "
                f"Transformers error: {repr(exc)}"
            ) from exc

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []

        if self.backend == "flagembedding":
            return self._score_pairs_flagembedding(pairs)

        if self.backend == "transformers":
            return self._score_pairs_transformers(pairs)

        raise RuntimeError("reranker backend is not initialized")

    def score_pairs_norm(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [self._sigmoid(x) for x in self.score_pairs(pairs)]

    def _score_pairs_flagembedding(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        assert self._flag_reranker is not None

        results: List[float] = []
        bs = max(1, int(self.config.batch_size))
        for start in range(0, len(pairs), bs):
            batch = pairs[start : start + bs]
            payload = [[q, p] for q, p in batch]
            scores = self._flag_reranker.compute_score(
                payload,
                max_length=int(self.config.max_length),
            )

            if isinstance(scores, (int, float)):
                results.append(float(scores))
            else:
                results.extend(float(x) for x in scores)

        return results

    def _score_pairs_transformers(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        torch = self._torch
        results: List[float] = []
        bs = max(1, int(self.config.batch_size))

        with torch.inference_mode():
            for start in range(0, len(pairs), bs):
                batch = pairs[start : start + bs]
                encoded = self._tokenizer(
                    [[q, p] for q, p in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=int(self.config.max_length),
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
                results.extend(logits.detach().cpu().tolist())

        return [float(x) for x in results]

    @property
    def runtime_info(self) -> dict:
        return {
            "backend": self.backend,
            "model_ref": self.model_ref,
            "device": self._device if self._device is not None else self.config.device,
            "batch_size": int(self.config.batch_size),
            "max_length": int(self.config.max_length),
            "torch_dtype": self.config.torch_dtype,
        }