from __future__ import annotations

from typing import Dict, List

import torch


def _pad_1d_long(xs: List[torch.Tensor], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.numel() for x in xs) if xs else 0
    out = torch.full((len(xs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.numel()
        out[i, :n] = x
        mask[i, :n] = True
    return out, mask


def _pad_1d_float(xs: List[torch.Tensor], pad_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.numel() for x in xs) if xs else 0
    out = torch.full((len(xs), max_len), pad_value, dtype=torch.float)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.numel()
        out[i, :n] = x
        mask[i, :n] = True
    return out, mask


def refiner_collate_fn(batch: List[Dict]) -> Dict:
    atoms_text = [x["atoms_text"] for x in batch]
    sample_ids = [x["sample_id"] for x in batch]
    doc_ids = [x["doc_id"] for x in batch]
    num_atoms = torch.tensor([x["num_atoms"] for x in batch], dtype=torch.long)
    num_gaps = torch.tensor([x["num_gaps"] for x in batch], dtype=torch.long)

    b0, b0_mask = _pad_1d_long([x["b0"] for x in batch], pad_value=0)
    b_gold, b_gold_mask = _pad_1d_long([x["b_gold"] for x in batch], pad_value=0)

    insert_labels, insert_mask = _pad_1d_float(
        [x["insert_labels"] for x in batch],
        pad_value=0.0,
    )

    g0_positions, g0_mask = _pad_1d_long(
        [x["g0_positions"] for x in batch],
        pad_value=-1,
    )

    edit_choice, edit_choice_mask = _pad_1d_long(
        [x["edit_choice"] for x in batch],
        pad_value=-100,  # CE ignore_index
    )

    return {
        "sample_ids": sample_ids,
        "doc_ids": doc_ids,
        "atoms_text": atoms_text,      # 后续 AtomEncoder/tokenizer 再接
        "num_atoms": num_atoms,
        "num_gaps": num_gaps,
        "b0": b0,
        "b0_mask": b0_mask,
        "b_gold": b_gold,
        "b_gold_mask": b_gold_mask,
        "insert_labels": insert_labels,
        "insert_mask": insert_mask,
        "g0_positions": g0_positions,
        "g0_mask": g0_mask,
        "edit_choice": edit_choice,
        "edit_choice_mask": edit_choice_mask,
    }