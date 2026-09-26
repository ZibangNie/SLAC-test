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
    Weighted MVP loss for Boundary Refiner.

    Components:
      - L_ins: BCEWithLogits (+ pos_weight)
      - L_edit: CrossEntropy over {KEEP, DEL, SHIFT(k)}
      - optional L_cost_reg: expected edit-cost regularizer
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
        sample_weight_reduction: str = "absolute",
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
        if sample_weight_reduction not in {"absolute", "normalized"}:
            raise ValueError("sample_weight_reduction must be absolute or normalized")
        self.sample_weight_reduction = sample_weight_reduction

    def _weighted_mean(self, per_sample: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
        if sample_weight is None:
            return per_sample.mean()
        sw = sample_weight.to(per_sample.device, dtype=per_sample.dtype)
        if sw.shape != per_sample.shape or not torch.isfinite(sw).all() or (sw < 0).any():
            raise ValueError("sample weights must be finite, non-negative and have shape [B]")
        weighted = (per_sample * sw).sum()
        if self.sample_weight_reduction == "normalized":
            return weighted / sw.sum().clamp(min=1e-8)
        # Confidence remains effective at batch size one. Weights of 1 retain the
        # unweighted per-document mean; legacy normalized behavior is opt-in.
        return weighted / max(per_sample.numel(), 1)

    def compute_insert_loss(
        self,
        insert_logits: torch.Tensor,   # [B, G]
        insert_labels: torch.Tensor,   # [B, G]
        insert_mask: torch.Tensor,     # [B, G]
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_weight = torch.tensor(
            self.insert_pos_weight,
            dtype=insert_logits.dtype,
            device=insert_logits.device,
        )

        valid_mask = insert_mask.to(device=insert_logits.device, dtype=torch.bool)
        raw = F.binary_cross_entropy_with_logits(
            torch.where(valid_mask, insert_logits, 0.0),
            torch.where(valid_mask, insert_labels, 0.0),
            reduction="none",
            pos_weight=pos_weight,
        )

        valid = insert_mask.to(raw.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        per_sample = (raw * valid).sum(dim=1) / denom
        return self._weighted_mean(per_sample, sample_weight)

    def compute_edit_loss(
        self,
        edit_choice_logits: torch.Tensor,  # [B, B0, C]
        edit_choice: torch.Tensor,         # [B, B0]
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = edit_choice_logits.shape
        safe_logits = torch.where((edit_choice != -100).unsqueeze(-1), edit_choice_logits, 0.0)
        raw = F.cross_entropy(
            safe_logits.reshape(B * N, C),
            edit_choice.reshape(B * N),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, N)

        valid = (edit_choice != -100).to(raw.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        per_sample = (raw * valid).sum(dim=1) / denom
        return self._weighted_mean(per_sample, sample_weight)

    def compute_cost_regularizer(
        self,
        insert_logits: torch.Tensor,      # [B, G]
        edit_choice_logits: torch.Tensor, # [B, B0, 2K+2]
        edit_choice_mask: torch.Tensor,   # [B, B0]
        sample_weight: torch.Tensor | None = None,
        K: int = 6,
        insert_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p_ins = torch.sigmoid(insert_logits)
        if insert_mask is None:
            insert_mask = torch.ones_like(p_ins, dtype=torch.bool)
        ins_valid = insert_mask.to(p_ins.device, dtype=p_ins.dtype)
        ins_cost = self.lambda_ins * (p_ins * ins_valid).sum(dim=1) / ins_valid.sum(dim=1).clamp(min=1)

        safe_edit_logits = torch.where(edit_choice_mask.unsqueeze(-1).bool(), edit_choice_logits, 0.0)
        p = torch.softmax(safe_edit_logits, dim=-1)   # [B,B0,2K+2]
        p_del = p[..., 0]

        valid = edit_choice_mask.to(edit_choice_logits.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)

        shift_cost = 0.0
        for cls_idx in range(1, 2 * K + 2):
            k = cls_idx - 1 - K
            shift_cost = shift_cost + self.lambda_shift * abs(k) * p[..., cls_idx]

        shift_cost = (shift_cost * valid).sum(dim=1) / denom
        del_cost = self.lambda_del * (p_del * valid).sum(dim=1) / denom

        per_sample = ins_cost + del_cost + shift_cost
        return self._weighted_mean(per_sample, sample_weight)

    def forward(self, outputs, batch) -> RefinerLossOutput:
        sample_weight = batch.get("sample_weight", None)
        if sample_weight is not None:
            sample_weight = sample_weight.to(outputs.insert_logits.device)

        loss_insert = self.compute_insert_loss(
            insert_logits=outputs.insert_logits,
            insert_labels=batch["insert_labels"].to(outputs.insert_logits.device),
            insert_mask=batch["insert_mask"].to(outputs.insert_logits.device),
            sample_weight=sample_weight,
        )

        loss_edit = self.compute_edit_loss(
            edit_choice_logits=outputs.edit_choice_logits,
            edit_choice=batch["edit_choice"].to(outputs.edit_choice_logits.device),
            sample_weight=sample_weight,
        )

        if self.beta_cost > 0:
            loss_cost_reg = self.compute_cost_regularizer(
                insert_logits=outputs.insert_logits,
                edit_choice_logits=outputs.edit_choice_logits,
                edit_choice_mask=batch["edit_choice_mask"].to(outputs.edit_choice_logits.device),
                sample_weight=sample_weight,
                K=(outputs.edit_choice_logits.shape[-1] - 2) // 2,
                insert_mask=batch["insert_mask"].to(outputs.insert_logits.device),
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
