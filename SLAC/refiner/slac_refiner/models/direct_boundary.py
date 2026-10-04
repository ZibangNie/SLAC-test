"""Isolated b0-conditioned final-gap baseline for bounded comparisons.

No pretrained encoder, checkpoint, file access, decoder, or optimizer is created.
The head matches local reachable-seed features, not the edit softmax's wider
candidate dependencies or the decoder's global monotonic constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from slac_refiner.models.doc_encoder import DocEncoder


@dataclass
class DirectBoundaryOutput:
    logits: torch.Tensor       # [B, T-1]; invalid gaps have finfo.min
    gap_mask: torch.Tensor     # [B, T-1]; consumers must mask loss/predictions


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_and_clean(
    values: torch.Tensor, b0: torch.Tensor, atom_mask: torch.Tensor, dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("atom values must be a floating [B,T,D] tensor")
    B, T, D = values.shape
    if B < 1 or T < 1 or D != dimension:
        raise ValueError("atom values have empty batch/time or wrong feature dimension")
    if atom_mask.shape != (B, T) or atom_mask.dtype != torch.bool:
        raise ValueError("atom_mask must be bool with shape [B,T]")
    if atom_mask.device != values.device or b0.device != values.device:
        raise ValueError("atom values, atom_mask and b0 must share a device")
    counts = atom_mask.sum(dim=1)
    prefix = torch.arange(T, device=values.device)[None, :] < counts[:, None]
    if (counts < 1).any() or not torch.equal(atom_mask, prefix):
        raise ValueError("each atom_mask must be a nonempty True prefix")
    integer_dtypes = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if b0.shape != (B, T - 1) or b0.dtype not in integer_dtypes:
        raise ValueError("b0 must be a bool/integer [B,T-1] tensor")
    if ((b0 != 0) & (b0 != 1)).any():
        raise ValueError("b0 must contain only zero and one")
    gap_mask = atom_mask[:, :-1] & atom_mask[:, 1:]
    if ((b0 != 0) & ~gap_mask).any():
        raise ValueError("b0 must be zero at every invalid gap")
    # Sanitize before differences, products or Linear; NaN * 0 is still NaN.
    clean = values.masked_fill(~atom_mask.unsqueeze(-1), 0)
    if not torch.isfinite(clean).all():
        raise ValueError("real atom values must be finite")
    return clean, gap_mask


class DirectBoundaryHead(nn.Module):
    """Direct final-gap logits with explicit ordered local seed conditioning.

    Feature order: target gap's [left,right,right-left,left*right], then the
    same representation for seed gaps at offsets -K..K (zero if absent), then
    their presence bits. These are contextual h features, not raw text spans.
    mlp_hidden_size is explicit; no data-dependent capacity selection occurs.
    """

    def __init__(self, hidden_size: int, K: int, mlp_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        _positive_int(hidden_size, "hidden_size")
        _positive_int(mlp_hidden_size, "mlp_hidden_size")
        if type(K) is not int or K < 0:
            raise ValueError("K must be a nonnegative integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.hidden_size = hidden_size
        self.K = K
        self.mlp_hidden_size = mlp_hidden_size
        self.input_dim = 4 * hidden_size * (2 * K + 2) + (2 * K + 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, 1),
        )

    def build_features(
        self, h: torch.Tensor, b0: torch.Tensor, atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean, gap_mask = _validate_and_clean(h, b0, atom_mask, self.hidden_size)
        B, T, _ = clean.shape
        G = T - 1
        if G == 0:
            return clean.new_empty((B, 0, self.input_dim)), gap_mask
        left, right = clean[:, :-1], clean[:, 1:]
        gap = torch.cat((left, right, right - left, left * right), dim=-1)
        gap = gap.masked_fill(~gap_mask.unsqueeze(-1), 0)
        offsets = torch.arange(-self.K, self.K + 1, device=h.device)
        positions = torch.arange(G, device=h.device)[:, None] + offsets[None, :]
        inside = (positions >= 0) & (positions < G)
        clamped = positions.clamp(0, G - 1)
        present = inside.unsqueeze(0) & (b0[:, clamped] != 0)
        present = present & gap_mask.unsqueeze(-1)
        neighbors = gap[:, clamped, :].masked_fill(~present.unsqueeze(-1), 0)
        features = torch.cat((gap, neighbors.flatten(start_dim=2), present.to(h.dtype)), dim=-1)
        return features, gap_mask

    def forward(self, h: torch.Tensor, b0: torch.Tensor, atom_mask: torch.Tensor) -> DirectBoundaryOutput:
        features, gap_mask = self.build_features(h, b0, atom_mask)
        logits = self.classifier(features).squeeze(-1)
        logits = logits.masked_fill(~gap_mask, torch.finfo(logits.dtype).min)
        return DirectBoundaryOutput(logits=logits, gap_mask=gap_mask)


class CachedSeedBoundaryClassifier(nn.Module):
    """Existing DocEncoder plus direct head; accepts frozen cached embeddings.

    This is an experimental comparator, not an epoch8 checkpoint loader. It
    deliberately has no edit heads. Any paired experiment must separately bind
    common embeddings, context initialization, training and projection policy.
    """

    def __init__(
        self, atom_dim: int, hidden_size: int, doc_layers: int, doc_heads: int,
        window_size: int, K: int, mlp_hidden_size: int, dropout: float = 0.0,
        max_doc_atoms: int = 1024,
    ):
        super().__init__()
        for name, value in (("atom_dim", atom_dim), ("doc_layers", doc_layers),
                            ("doc_heads", doc_heads), ("max_doc_atoms", max_doc_atoms)):
            _positive_int(value, name)
        if type(window_size) is not int or window_size < 0:
            raise ValueError("window_size must be a nonnegative integer")
        self.atom_dim = atom_dim
        self.max_doc_atoms = max_doc_atoms
        self.head = DirectBoundaryHead(hidden_size, K, mlp_hidden_size, dropout)
        self.doc = DocEncoder(
            atom_dim, hidden_size=hidden_size, num_layers=doc_layers,
            num_heads=doc_heads, dropout=dropout, window_size=window_size,
        )

    def forward(
        self, atom_embeddings: torch.Tensor, b0: torch.Tensor, atom_mask: torch.Tensor,
    ) -> DirectBoundaryOutput:
        if atom_embeddings.ndim == 3 and atom_embeddings.shape[1] > self.max_doc_atoms:
            raise ValueError("document exceeds max_doc_atoms; no silent truncation")
        clean, _ = _validate_and_clean(atom_embeddings.detach(), b0, atom_mask, self.atom_dim)
        contextual = self.doc(clean, atom_mask).h
        return self.head(contextual, b0, atom_mask)
