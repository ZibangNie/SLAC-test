from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    gap_repr: torch.Tensor        # [B, G, 4H]
    insert_logits: torch.Tensor   # [B, G]
    edit_logits: torch.Tensor     # [B, B0, 3]  0=KEEP,1=DEL,2=SHIFT
    offset_pred: torch.Tensor     # [B, B0]


class RefinerHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        gap_dim = hidden_size * 4
        self.gap_dim = gap_dim

        self.insert_head = nn.Sequential(
            nn.Linear(gap_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.edit_head = nn.Sequential(
            nn.Linear(gap_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),   # KEEP / DEL / SHIFT
        )

        self.offset_head = nn.Sequential(
            nn.Linear(gap_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def build_gap_repr(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, T, H]
        return gap_repr: [B, T-1, 4H]
        r_i = [h_i; h_{i+1}; h_{i+1}-h_i; h_i*h_{i+1}]
        """
        h_l = h[:, :-1, :]
        h_r = h[:, 1:, :]
        gap_repr = torch.cat(
            [h_l, h_r, h_r - h_l, h_l * h_r],
            dim=-1,
        )
        return gap_repr

    def gather_boundary_repr(
        self,
        gap_repr: torch.Tensor,      # [B, G, 4H]
        g0_positions: torch.Tensor,  # [B, B0], padded with -1
    ) -> torch.Tensor:
        B, G, D = gap_repr.shape
        _, N = g0_positions.shape

        pos = g0_positions.clamp(min=0)
        idx = pos.unsqueeze(-1).expand(B, N, D)
        gathered = torch.gather(gap_repr, dim=1, index=idx)

        valid = (g0_positions >= 0).unsqueeze(-1)
        gathered = gathered * valid
        return gathered

    def forward(
        self,
        h: torch.Tensor,              # [B, T, H]
        g0_positions: torch.Tensor,   # [B, B0]
    ) -> HeadOutput:
        gap_repr = self.build_gap_repr(h)                    # [B, G, 4H]
        insert_logits = self.insert_head(gap_repr).squeeze(-1)  # [B, G]

        boundary_repr = self.gather_boundary_repr(gap_repr, g0_positions)   # [B, B0, 4H]
        edit_logits = self.edit_head(boundary_repr)          # [B, B0, 3]
        offset_pred = self.offset_head(boundary_repr).squeeze(-1)   # [B, B0]

        return HeadOutput(
            gap_repr=gap_repr,
            insert_logits=insert_logits,
            edit_logits=edit_logits,
            offset_pred=offset_pred,
        )