from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from slac_refiner.models.atom_encoder import AtomEncoder
from slac_refiner.models.doc_encoder import DocEncoder
from slac_refiner.models.heads import RefinerHeads, HeadOutput


@dataclass
class RefinerForwardOutput:
    atom_embeddings: torch.Tensor   # [B, T, D_atom]
    atom_mask: torch.Tensor         # [B, T]
    doc_hidden: torch.Tensor        # [B, T, H]
    insert_logits: torch.Tensor     # [B, G]
    edit_logits: torch.Tensor       # [B, B0, 3]
    offset_pred: torch.Tensor       # [B, B0]


class BoundaryRefinerModel(nn.Module):
    def __init__(
        self,
        atom_model_name: str = r"D:\code\Github\SLAC-test\SLAC\refiner\slac_refiner\models\bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
        atom_max_length: int = 128,
        atom_freeze: bool = True,
        doc_hidden_size: int = 768,
        doc_layers: int = 4,
        doc_heads: int = 12,
        doc_dropout: float = 0.1,
        window_size: int = 512,
        device: str | None = None,
    ):
        super().__init__()

        self.atom_encoder = AtomEncoder(
            model_name=atom_model_name,
            max_length=atom_max_length,
            freeze=atom_freeze,
            device=device,
        )

        self.doc_encoder = DocEncoder(
            atom_dim=self.atom_encoder.hidden_size,
            hidden_size=doc_hidden_size,
            num_layers=doc_layers,
            num_heads=doc_heads,
            dropout=doc_dropout,
            window_size=window_size,
        )

        self.heads = RefinerHeads(
            hidden_size=doc_hidden_size,
            dropout=doc_dropout,
        )

        self.to(self.atom_encoder.device)

    @property
    def device(self) -> torch.device:
        return self.atom_encoder.device

    def encode_atom_batch(self, atoms_text_batch: List[List[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode variable-length atom lists one sample at a time, then pad.
        Returns:
          atom_embeddings: [B, T, D]
          atom_mask:       [B, T]  True means valid
        """
        per_sample = []
        lengths = []

        for atoms in atoms_text_batch:
            out = self.atom_encoder.encode(atoms, normalize=False)
            emb = out.atom_embeddings   # [T_i, D]
            per_sample.append(emb)
            lengths.append(emb.shape[0])

        B = len(per_sample)
        T = max(lengths)
        D = per_sample[0].shape[1]

        padded = torch.zeros(B, T, D, dtype=per_sample[0].dtype, device=self.device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)

        for i, emb in enumerate(per_sample):
            n = emb.shape[0]
            padded[i, :n] = emb
            mask[i, :n] = True

        return padded, mask

    def forward(self, batch: Dict) -> RefinerForwardOutput:
        atoms_text_batch: List[List[str]] = batch["atoms_text"]
        g0_positions: torch.Tensor = batch["g0_positions"].to(self.device)

        atom_embeddings, atom_mask = self.encode_atom_batch(atoms_text_batch)
        doc_out = self.doc_encoder(
            atom_embeddings=atom_embeddings,
            padding_mask=atom_mask,
        )

        head_out: HeadOutput = self.heads(
            h=doc_out.h,
            g0_positions=g0_positions,
        )

        return RefinerForwardOutput(
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            doc_hidden=doc_out.h,
            insert_logits=head_out.insert_logits,
            edit_logits=head_out.edit_logits,
            offset_pred=head_out.offset_pred,
        )