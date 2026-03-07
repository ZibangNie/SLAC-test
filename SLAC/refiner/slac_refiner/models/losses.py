from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


SHIFT_CLASS_ID = 2


@dataclass
class RefinerLossOutput:
    loss: torch.Tensor
    loss_insert: torch.Tensor
    loss_edit: torch.Tensor
    loss_offset: torch.Tensor
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
        insert_pos_weight: float = 4.0,
        alpha_insert: float = 1.0,
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
        edit_logits: torch.Tensor,     # [B, B0, 3]
        edit_cls: torch.Tensor,        # [B, B0], long, ignore=-100
    ) -> torch.Tensor:
        B, N, C = edit_logits.shape
        raw = F.cross_entropy(
            edit_logits.reshape(B * N, C),
            edit_cls.reshape(B * N),
            ignore_index=-100,
            reduction="mean",
        )
        return raw

    def compute_offset_loss(
        self,
        offset_pred: torch.Tensor,     # [B, B0]
        edit_offset: torch.Tensor,     # [B, B0]
        edit_cls: torch.Tensor,        # [B, B0]
        edit_cls_mask: torch.Tensor,   # [B, B0]
    ) -> torch.Tensor:
        shift_mask = (edit_cls == SHIFT_CLASS_ID) & edit_cls_mask
        if shift_mask.sum() == 0:
            return offset_pred.new_zeros(())

        pred = offset_pred[shift_mask]
        gold = edit_offset[shift_mask].to(offset_pred.dtype)
        return F.smooth_l1_loss(pred, gold, reduction="mean")

    def compute_cost_regularizer(
        self,
        insert_logits: torch.Tensor,   # [B, G]
        edit_logits: torch.Tensor,     # [B, B0, 3]
        edit_cls_mask: torch.Tensor,   # [B, B0]
    ) -> torch.Tensor:
        """
        Lightweight expectation-style regularizer:
          E[cost] ≈
            lambda_ins * sum(sigmoid(insert_logits))
          + lambda_del * P(DEL)
          + lambda_shift * P(SHIFT)

        This is only a soft prior, not the final decode-time cost.
        """
        # insert expected cost
        p_ins = torch.sigmoid(insert_logits)
        ins_cost = self.lambda_ins * p_ins.mean()

        # edit expected cost
        p_edit = torch.softmax(edit_logits, dim=-1)  # [B, B0, 3]
        p_del = p_edit[..., 1]
        p_shift = p_edit[..., 2]

        valid = edit_cls_mask.to(edit_logits.dtype)
        denom = valid.sum().clamp(min=1.0)

        del_cost = self.lambda_del * (p_del * valid).sum() / denom
        shift_cost = self.lambda_shift * (p_shift * valid).sum() / denom

        return ins_cost + del_cost + shift_cost

    def forward(self, outputs, batch) -> RefinerLossOutput:
        loss_insert = self.compute_insert_loss(
            insert_logits=outputs.insert_logits,
            insert_labels=batch["insert_labels"].to(outputs.insert_logits.device),
            insert_mask=batch["insert_mask"].to(outputs.insert_logits.device),
        )

        loss_edit = self.compute_edit_loss(
            edit_logits=outputs.edit_logits,
            edit_cls=batch["edit_cls"].to(outputs.edit_logits.device),
        )

        loss_offset = self.compute_offset_loss(
            offset_pred=outputs.offset_pred,
            edit_offset=batch["edit_offset"].to(outputs.offset_pred.device),
            edit_cls=batch["edit_cls"].to(outputs.offset_pred.device),
            edit_cls_mask=batch["edit_cls_mask"].to(outputs.offset_pred.device),
        )

        if self.beta_cost > 0:
            loss_cost_reg = self.compute_cost_regularizer(
                insert_logits=outputs.insert_logits,
                edit_logits=outputs.edit_logits,
                edit_cls_mask=batch["edit_cls_mask"].to(outputs.edit_logits.device),
            )
        else:
            loss_cost_reg = outputs.insert_logits.new_zeros(())

        total = (
            self.alpha_insert * loss_insert
            + self.alpha_edit * loss_edit
            + self.alpha_offset * loss_offset
            + self.beta_cost * loss_cost_reg
        )

        with torch.no_grad():
            stats = {
                "loss_total": float(total.item()),
                "loss_insert": float(loss_insert.item()),
                "loss_edit": float(loss_edit.item()),
                "loss_offset": float(loss_offset.item()),
                "loss_cost_reg": float(loss_cost_reg.item()),
            }

        return RefinerLossOutput(
            loss=total,
            loss_insert=loss_insert,
            loss_edit=loss_edit,
            loss_offset=loss_offset,
            loss_cost_reg=loss_cost_reg,
            stats=stats,
        )