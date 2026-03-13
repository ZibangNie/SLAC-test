from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any

import os
os.environ["HF_HUB_OFFLINE"] = "1"

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

    Engineering choice:
        - HF AutoTokenizer + AutoModel
        - masked mean pooling over last_hidden_state
        - freeze by default
        - support mini-batch encoding to avoid OOM on long docs
    """

    def __init__(
        self,
        model_name: str = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        max_length: int = 128,
        freeze: bool = True,
        device: str | None = None,
        local_files_only: bool = True,
        encode_batch_size: int = 16,   # 新增：分批编码 atom
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.encode_batch_size = int(encode_batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        self.hidden_size = int(self.backbone.config.hidden_size)

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device
        self.to(device)
        print(f"[AtomEncoder] loading model from: {model_name}")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(
        self,
        atoms_text: Sequence[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        这里改成 padding='max_length'，这样不同 mini-batch 的宽度一致，
        后面可以安全 torch.cat。
        """
        batch = self.tokenizer(
            list(atoms_text),
            padding="max_length",
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

    def _encode_impl(
        self,
        atoms_text: Sequence[str],
        normalize: bool = False,
        use_grad: bool = False,
    ) -> AtomEncoderOutput:
        if len(atoms_text) == 0:
            raise ValueError("atoms_text must be non-empty")

        all_input_ids: List[torch.Tensor] = []
        all_attention_mask: List[torch.Tensor] = []
        all_atom_embeddings: List[torch.Tensor] = []

        batch_size = max(1, int(self.encode_batch_size))

        for start in range(0, len(atoms_text), batch_size):
            sub_atoms = atoms_text[start:start + batch_size]

            batch = self.tokenize(sub_atoms)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if use_grad:
                outputs = self.backbone(**batch)
                last_hidden = outputs.last_hidden_state
                atom_embeddings = self.mean_pool(last_hidden, batch["attention_mask"])
                if normalize:
                    atom_embeddings = torch.nn.functional.normalize(atom_embeddings, p=2, dim=-1)

                # 保持 tensor 在 device 上，供上层继续用
                all_input_ids.append(batch["input_ids"])
                all_attention_mask.append(batch["attention_mask"])
                all_atom_embeddings.append(atom_embeddings)
            else:
                with torch.inference_mode():
                    outputs = self.backbone(**batch)
                    last_hidden = outputs.last_hidden_state
                    atom_embeddings = self.mean_pool(last_hidden, batch["attention_mask"])
                    if normalize:
                        atom_embeddings = torch.nn.functional.normalize(atom_embeddings, p=2, dim=-1)

                    all_input_ids.append(batch["input_ids"])
                    all_attention_mask.append(batch["attention_mask"])
                    all_atom_embeddings.append(atom_embeddings)

            # 尽早释放中间变量，缓解显存峰值
            del outputs
            del last_hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_mask, dim=0)
        atom_embeddings = torch.cat(all_atom_embeddings, dim=0)

        return AtomEncoderOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            atom_embeddings=atom_embeddings,
            hidden_size=self.hidden_size,
        )

    @torch.no_grad()
    def encode(
        self,
        atoms_text: Sequence[str],
        normalize: bool = False,
    ) -> AtomEncoderOutput:
        """
        推理路径：默认无梯度 + 分批 atom 编码
        """
        return self._encode_impl(
            atoms_text=atoms_text,
            normalize=normalize,
            use_grad=False,
        )

    def forward(
        self,
        atoms_text: Sequence[str],
        normalize: bool = False,
    ) -> AtomEncoderOutput:
        """
        训练/通用路径：如果 backbone freeze，则仍然走无梯度；
        如果以后要解冻训练，则自动保留梯度。
        """
        use_grad = not self.freeze
        return self._encode_impl(
            atoms_text=atoms_text,
            normalize=normalize,
            use_grad=use_grad,
        )