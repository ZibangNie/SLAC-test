from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefinerLossOutput:
    loss: torch.Tensor
    loss_insert: torch.Tensor
    loss_edit: torch.Tensor
    loss_cost_reg: torch.Tensor
    stats: Dict[str, float]


class RefinerLoss(nn.Module):
    """
    MVP loss for Boundary Refiner.

    Components:
      - L_ins: BCEWithLogits (+ pos_weight)
      - L_edit: CrossEntropy over {KEEP, DEL, SHIFT}
      - L_offset: SmoothL1 on shift offsets, only where class == SHIFT
      - optional L_cost_reg: expected edit-cost regularizer (lightweight approximation)

    Notes:
      - This matches the current simplified head design.
      - Later, when you switch to pointer-style local scoring, L_edit/L_offset
        can be replaced by a single local-candidate CE + delete class.
    """
    def __init__(
        self,
        insert_pos_weight: float = 16.0,
        alpha_insert: float = 3.0,
        alpha_edit: float = 1.0,
        alpha_offset: float = 0.5,
        beta_cost: float = 0.0,
        lambda_del: float = 1.0,
        lambda_ins: float = 1.0,
        lambda_shift: float = 0.25,
    ):
        super().__init__()
        self.insert_pos_weight = insert_pos_weight
        self.alpha_insert = alpha_insert
        self.alpha_edit = alpha_edit
        self.alpha_offset = alpha_offset
        self.beta_cost = beta_cost
        self.lambda_del = lambda_del
        self.lambda_ins = lambda_ins
        self.lambda_shift = lambda_shift

    def compute_insert_loss(
        self,
        insert_logits: torch.Tensor,   # [B, G]
        insert_labels: torch.Tensor,   # [B, G], float
        insert_mask: torch.Tensor,     # [B, G], bool
    ) -> torch.Tensor:
        pos_weight = torch.tensor(
            self.insert_pos_weight,
            dtype=insert_logits.dtype,
            device=insert_logits.device,
        )

        raw = F.binary_cross_entropy_with_logits(
            insert_logits,
            insert_labels,
            reduction="none",
            pos_weight=pos_weight,
        )

        valid = insert_mask.to(raw.dtype)
        denom = valid.sum().clamp(min=1.0)
        return (raw * valid).sum() / denom

    def compute_edit_loss(
            self,
            edit_choice_logits: torch.Tensor,  # [B, B0, 2K+2]
            edit_choice: torch.Tensor,  # [B, B0], long, ignore=-100
    ) -> torch.Tensor:
        B, N, C = edit_choice_logits.shape
        raw = F.cross_entropy(
            edit_choice_logits.reshape(B * N, C),
            edit_choice.reshape(B * N),
            ignore_index=-100,
            reduction="mean",
        )
        return raw

    def compute_cost_regularizer(
            self,
            insert_logits: torch.Tensor,  # [B, G]
            edit_choice_logits: torch.Tensor,  # [B, B0, 2K+2]
            edit_choice_mask: torch.Tensor,  # [B, B0]
            K: int = 6,
    ) -> torch.Tensor:
        p_ins = torch.sigmoid(insert_logits)
        ins_cost = self.lambda_ins * p_ins.mean()

        p = torch.softmax(edit_choice_logits, dim=-1)  # [B,B0,2K+2]
        p_del = p[..., 0]

        shift_cost = 0.0
        valid = edit_choice_mask.to(edit_choice_logits.dtype)
        denom = valid.sum().clamp(min=1.0)

        # classes 1..2K+1 correspond to k in [-K..K]
        for cls_idx in range(1, 2 * K + 2):
            k = cls_idx - 1 - K
            shift_cost = shift_cost + self.lambda_shift * abs(k) * p[..., cls_idx]

        shift_cost = (shift_cost * valid).sum() / denom
        del_cost = self.lambda_del * (p_del * valid).sum() / denom

        return ins_cost + del_cost + shift_cost

    def forward(self, outputs, batch) -> RefinerLossOutput:
        loss_insert = self.compute_insert_loss(
            insert_logits=outputs.insert_logits,
            insert_labels=batch["insert_labels"].to(outputs.insert_logits.device),
            insert_mask=batch["insert_mask"].to(outputs.insert_logits.device),
        )

        loss_edit = self.compute_edit_loss(
            edit_choice_logits=outputs.edit_choice_logits,
            edit_choice=batch["edit_choice"].to(outputs.edit_choice_logits.device),
        )

        if self.beta_cost > 0:
            loss_cost_reg = self.compute_cost_regularizer(
                insert_logits=outputs.insert_logits,
                edit_choice_logits=outputs.edit_choice_logits,
                edit_choice_mask=batch["edit_choice_mask"].to(outputs.edit_choice_logits.device),
                K=6,
            )
        else:
            loss_cost_reg = outputs.insert_logits.new_zeros(())

        total = (
                self.alpha_insert * loss_insert
                + self.alpha_edit * loss_edit
                + self.beta_cost * loss_cost_reg
        )

        with torch.no_grad():
            stats = {
                "loss_total": float(total.item()),
                "loss_insert": float(loss_insert.item()),
                "loss_edit": float(loss_edit.item()),
                "loss_cost_reg": float(loss_cost_reg.item()),
            }

        return RefinerLossOutput(
            loss=total,
            loss_insert=loss_insert,
            loss_edit=loss_edit,
            loss_cost_reg=loss_cost_reg,
            stats=stats,
        )