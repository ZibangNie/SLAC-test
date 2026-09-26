from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    gap_repr: torch.Tensor             # [B, G, 4H]
    insert_logits: torch.Tensor        # [B, G]
    edit_choice_logits: torch.Tensor   # [B, B0, 2K+2]


class RefinerHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        K: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        gap_dim = hidden_size * 4
        self.gap_dim = gap_dim
        self.ptr_dim = hidden_size

        self.insert_head = nn.Sequential(
            nn.Linear(gap_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # pointer-style local scoring
        self.query_proj = nn.Linear(gap_dim, self.ptr_dim)
        self.key_proj = nn.Linear(gap_dim, self.ptr_dim)

        self.delete_head = nn.Sequential(
            nn.Linear(gap_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def build_gap_repr(self, h: torch.Tensor) -> torch.Tensor:
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
        if G == 0:
            return gap_repr.new_zeros((B, N, D))
        pos = g0_positions.clamp(min=0, max=G - 1)
        idx = pos.unsqueeze(-1).expand(B, N, D)
        gathered = torch.gather(gap_repr, dim=1, index=idx)

        valid = ((g0_positions >= 0) & (g0_positions < G)).unsqueeze(-1)
        gathered = gathered * valid
        return gathered

    def build_local_candidate_repr(
        self,
        gap_repr: torch.Tensor,       # [B, G, 4H]
        g0_positions: torch.Tensor,   # [B, B0]
        gap_mask: torch.Tensor | None = None,  # [B, G]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return:
          local_repr: [B, B0, 2K+1, 4H]
          local_mask: [B, B0, 2K+1]  True = valid
        """
        B, G, D = gap_repr.shape
        _, N = g0_positions.shape
        W = 2 * self.K + 1
        if G == 0:
            return (
                gap_repr.new_zeros((B, N, W, D)),
                torch.zeros((B, N, W), device=gap_repr.device, dtype=torch.bool),
            )

        offsets = torch.arange(-self.K, self.K + 1, device=gap_repr.device)  # [W]
        base = g0_positions.unsqueeze(-1)                                     # [B, B0, 1]
        cand_pos = base + offsets.view(1, 1, W)                               # [B, B0, W]

        valid = ((g0_positions.unsqueeze(-1) >= 0)
                 & (g0_positions.unsqueeze(-1) < G) & (cand_pos >= 0) & (cand_pos < G))

        cand_pos_clamped = cand_pos.clamp(min=0, max=max(G - 1, 0))
        if gap_mask is not None:
            candidate_mask = torch.gather(
                gap_mask.unsqueeze(1).expand(B, N, G), 2, cand_pos_clamped
            )
            source_mask = torch.gather(gap_mask, 1, g0_positions.clamp(min=0, max=G - 1))
            valid = valid & candidate_mask & source_mask.unsqueeze(-1)
        idx = cand_pos_clamped.unsqueeze(-1).expand(B, N, W, D)

        gap_repr_exp = gap_repr.unsqueeze(1).expand(B, N, G, D)
        local_repr = torch.gather(gap_repr_exp, dim=2, index=idx)
        local_repr = local_repr * valid.unsqueeze(-1)

        return local_repr, valid

    def forward(
        self,
        h: torch.Tensor,              # [B, T, H]
        g0_positions: torch.Tensor,   # [B, B0]
        atom_mask: torch.Tensor | None = None,  # [B, T], True = real atom
    ) -> HeadOutput:
        gap_repr = self.build_gap_repr(h)                          # [B, G, 4H]
        if atom_mask is None:
            gap_mask = torch.ones(gap_repr.shape[:2], device=h.device, dtype=torch.bool)
        else:
            if atom_mask.shape != h.shape[:2]:
                raise ValueError("atom_mask must match the batch and atom dimensions of h")
            atom_mask = atom_mask.to(device=h.device, dtype=torch.bool)
            gap_mask = atom_mask[:, :-1] & atom_mask[:, 1:]
        gap_repr = gap_repr.masked_fill(~gap_mask.unsqueeze(-1), 0)
        insert_logits = self.insert_head(gap_repr).squeeze(-1)    # [B, G]
        insert_logits = insert_logits.masked_fill(~gap_mask, torch.finfo(insert_logits.dtype).min)

        boundary_repr = self.gather_boundary_repr(gap_repr, g0_positions)   # [B, B0, 4H]
        local_repr, local_mask = self.build_local_candidate_repr(gap_repr, g0_positions, gap_mask)  # [B,B0,W,4H]

        q = self.query_proj(boundary_repr)                        # [B, B0, D]
        k = self.key_proj(local_repr)                             # [B, B0, W, D]

        ptr_scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.ptr_dim)   # [B,B0,W]

        # invalid candidates -> very negative
        ptr_scores = ptr_scores.masked_fill(~local_mask, torch.finfo(ptr_scores.dtype).min)

        delete_logits = self.delete_head(boundary_repr)           # [B,B0,1]

        # final class order:
        # 0 -> DEL
        # 1..(2K+1) -> k in [-K..K]
        edit_choice_logits = torch.cat([delete_logits, ptr_scores], dim=-1)   # [B,B0,2K+2]

        return HeadOutput(
            gap_repr=gap_repr,
            insert_logits=insert_logits,
            edit_choice_logits=edit_choice_logits,
        )
