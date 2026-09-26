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
    insert_logits: torch.Tensor
    edit_choice_logits: torch.Tensor


class BoundaryRefinerModel(nn.Module):
    def __init__(
        self,
        atom_model_name: str = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        atom_max_length: int = 128,
        atom_freeze: bool = True,
        doc_hidden_size: int = 768,
        doc_layers: int = 4,
        doc_heads: int = 12,
        doc_dropout: float = 0.1,
        window_size: int = 512,
        device: str | None = None,
        atom_overflow_policy: str = "error",
        max_doc_atoms: int = 1024,
        k_shift: int = 6,
    ):
        super().__init__()
        if max_doc_atoms < 1:
            raise ValueError("max_doc_atoms must be positive")
        self.max_doc_atoms = max_doc_atoms

        self.atom_encoder = AtomEncoder(
            model_name=atom_model_name,
            max_length=atom_max_length,
            freeze=atom_freeze,
            device=device,
            overflow_policy=atom_overflow_policy,
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
            K=k_shift,
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
        if not atoms_text_batch:
            raise ValueError("batch must contain at least one document")
        per_sample = []
        lengths = []

        for atoms in atoms_text_batch:
            if not 1 <= len(atoms) <= self.max_doc_atoms:
                raise ValueError(f"document must contain 1..{self.max_doc_atoms} atoms; explicit windowing is required")
            out = self.atom_encoder(atoms, normalize=False)
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
        g0_positions: torch.Tensor = batch["g0_positions"].to(self.device)
        if "atom_embeddings" in batch:
            if not self.atom_encoder.freeze:
                raise ValueError("cached embeddings require a frozen atom encoder")
            atom_embeddings = batch["atom_embeddings"].detach().to(self.device)
            atom_mask = batch["atom_mask"].to(self.device, dtype=torch.bool)
            if atom_embeddings.ndim != 3 or atom_mask.shape != atom_embeddings.shape[:2] or atom_embeddings.shape[0] == 0:
                raise ValueError("invalid cached embeddings or atom mask")
            if atom_embeddings.shape[1] > self.max_doc_atoms or not atom_mask.any(dim=1).all():
                raise ValueError("cached document exceeds length contract or is empty")
            if atom_embeddings.shape[2] != self.doc_encoder.atom_dim or not torch.isfinite(atom_embeddings).all():
                raise ValueError("cached embeddings have an invalid dimension or nonfinite values")
            counts = atom_mask.sum(dim=1)
            expected_mask = torch.arange(atom_mask.shape[1], device=self.device)[None, :] < counts[:, None]
            if not torch.equal(atom_mask, expected_mask):
                raise ValueError("cached atom masks must use right padding")
            for key, expected in (("num_atoms", counts), ("num_gaps", counts - 1)):
                if key in batch and not torch.equal(torch.as_tensor(batch[key], device=self.device), expected):
                    raise ValueError(f"cached embeddings disagree with {key}")
            if "atoms_text" in batch and [len(atoms) for atoms in batch["atoms_text"]] != counts.tolist():
                raise ValueError("cached embeddings disagree with source atom counts")
        else:
            atom_embeddings, atom_mask = self.encode_atom_batch(batch["atoms_text"])
        atom_embeddings = atom_embeddings.to(dtype=self.doc_encoder.in_proj.weight.dtype)
        doc_out = self.doc_encoder(
            atom_embeddings=atom_embeddings,
            padding_mask=atom_mask,
        )

        head_out: HeadOutput = self.heads(
            h=doc_out.h,
            g0_positions=g0_positions,
            atom_mask=atom_mask,
        )

        return RefinerForwardOutput(
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            doc_hidden=doc_out.h,
            insert_logits=head_out.insert_logits,
            edit_choice_logits=head_out.edit_choice_logits,
        )
