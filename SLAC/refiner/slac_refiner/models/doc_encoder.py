from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DocEncoderOutput:
    h: torch.Tensor          # [B, T, H]
    padding_mask: torch.Tensor   # [B, T], True means valid token


class DocEncoder(nn.Module):
    """
    MVP DocEncoder:
    - project atom embeddings to hidden size
    - local-attention TransformerEncoder via attention mask

    Note:
    This is a semantically correct local-window approximation,
    but not the final memory-optimal implementation for T~3k.
    """
    def __init__(
        self,
        atom_dim: int,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1,
        window_size: int = 512,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.in_proj = nn.Linear(atom_dim, hidden_size)
        self.in_norm = nn.LayerNorm(hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_size)

    def build_local_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns boolean mask [T, T]:
        True  => blocked
        False => allowed
        """
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        blocked = dist > self.window_size
        return blocked

    def forward(
        self,
        atom_embeddings: torch.Tensor,   # [B, T, D]
        padding_mask: torch.Tensor,      # [B, T], True means valid
    ) -> DocEncoderOutput:
        if atom_embeddings.ndim != 3:
            raise ValueError(f"atom_embeddings must be [B,T,D], got {tuple(atom_embeddings.shape)}")
        if padding_mask.ndim != 2:
            raise ValueError(f"padding_mask must be [B,T], got {tuple(padding_mask.shape)}")

        x = self.in_proj(atom_embeddings)
        x = self.in_norm(x)

        B, T, _ = x.shape
        attn_mask = self.build_local_attn_mask(T, x.device)  # [T, T], True=blocked

        # src_key_padding_mask: True means PAD / ignore
        src_key_padding_mask = ~padding_mask

        h = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        h = self.out_norm(h)

        return DocEncoderOutput(
            h=h,
            padding_mask=padding_mask,
        )