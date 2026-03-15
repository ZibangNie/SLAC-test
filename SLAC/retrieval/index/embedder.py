from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    device: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32
    normalized: bool = True


class HFTextEmbedder:
    """
    使用 HuggingFace tokenizer + AutoModel 的通用向量器。
    对 BGE-M3 / XLM-R 类 encoder 采用 attention-mask mean pooling。
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading embedder model=%s on device=%s", cfg.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        all_vecs: List[np.ndarray] = []
        bs = self.cfg.batch_size

        for start in range(0, len(texts), bs):
            batch = list(texts[start : start + bs])

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            attention_mask = encoded["attention_mask"]

            vecs = mean_pool(last_hidden, attention_mask)
            if self.cfg.normalized:
                vecs = F.normalize(vecs, dim=-1)

            all_vecs.append(vecs.detach().cpu().float().numpy())

        return np.concatenate(all_vecs, axis=0).astype(np.float32)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def build_embedder(
    model_name: str = "BAAI/bge-m3",
    device: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 32,
    normalized: bool = True,
) -> HFTextEmbedder:
    cfg = EmbedderConfig(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        normalized=normalized,
    )
    return HFTextEmbedder(cfg)