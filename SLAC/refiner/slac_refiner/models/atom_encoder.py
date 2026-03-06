from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class AtomEncoderOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    atom_embeddings: torch.Tensor   # [B, D]
    hidden_size: int


class AtomEncoder(nn.Module):
    """
    Minimal AtomEncoder for Boundary Refiner.

    Input:
        list[str] atoms_text
    Output:
        dense embedding per atom, shape [B, D]

    Engineering choice for MVP:
        - HF AutoTokenizer + AutoModel
        - masked mean pooling over last_hidden_state
        - freeze by default
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_length: int = 128,
        freeze: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        self.hidden_size = int(self.backbone.config.hidden_size)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(
        self,
        atoms_text: Sequence[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(atoms_text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        return batch

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        last_hidden_state: [B, T, D]
        attention_mask:    [B, T]
        """
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                   # [B, D]
        denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
        return summed / denom

    @torch.no_grad()
    def encode(
        self,
        atoms_text: Sequence[str],
        normalize: bool = False,
    ) -> AtomEncoderOutput:
        if len(atoms_text) == 0:
            raise ValueError("atoms_text must be non-empty")

        batch = self.tokenize(atoms_text)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.backbone(**batch)
        last_hidden = outputs.last_hidden_state
        atom_embeddings = self.mean_pool(last_hidden, batch["attention_mask"])

        if normalize:
            atom_embeddings = torch.nn.functional.normalize(atom_embeddings, p=2, dim=-1)

        return AtomEncoderOutput(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            atom_embeddings=atom_embeddings,
            hidden_size=self.hidden_size,
        )

    def forward(
        self,
        atoms_text: Sequence[str],
        normalize: bool = False,
    ) -> AtomEncoderOutput:
        # 保持和 encode 一致；如果将来要解冻训练，再去掉 no_grad 逻辑
        batch = self.tokenize(atoms_text)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.backbone(**batch)
        last_hidden = outputs.last_hidden_state
        atom_embeddings = self.mean_pool(last_hidden, batch["attention_mask"])

        if normalize:
            atom_embeddings = torch.nn.functional.normalize(atom_embeddings, p=2, dim=-1)

        return AtomEncoderOutput(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            atom_embeddings=atom_embeddings,
            hidden_size=self.hidden_size,
        )