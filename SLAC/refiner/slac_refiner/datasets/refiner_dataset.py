from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


EDIT_LABEL2ID = {
    "KEEP": 0,
    "DEL": 1,
    "SHIFT": 2,
}

K_DEFAULT = 6


def _choice_from_label(y: str, K: int) -> int:
    if y == "DEL":
        return 0
    if y == "KEEP":
        return 1 + K   # k=0
    if y.startswith("SHIFT:"):
        k = int(y.split(":", 1)[1])
        if not (-K <= k <= K):
            raise ValueError(f"SHIFT offset out of range: {k}, K={K}")
        return 1 + (k + K)
    raise ValueError(f"Unknown edit label: {y}")


def _vector_to_gaps(b: Sequence[int]) -> List[int]:
    return [i for i, x in enumerate(b) if int(x) == 1]


def _parse_shift(y: str) -> int:
    if not y.startswith("SHIFT:"):
        raise ValueError(f"Invalid SHIFT label: {y}")
    return int(y.split(":", 1)[1])


class RefinerDenoiseDataset(Dataset):
    """
    One training sample contains:
    - atoms text list
    - b0 over all gaps
    - insert_labels over all gaps
    - g0_positions over initial boundaries only
    - edit_cls / edit_offset aligned to g0_positions
    """
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self.samples: List[Dict] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Failed parsing {self.path} line {line_no}: {e}") from e

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        atoms_raw = s["atoms"]
        atoms_text = []
        for a in atoms_raw:
            if isinstance(a, dict):
                atoms_text.append(str(a["text"]))
            else:
                atoms_text.append(str(a))

        b0 = [int(x) for x in s["b0"]]
        insert_labels = [int(x) for x in s["labels"]["insert"]]

        g0_positions = _vector_to_gaps(b0)
        edit_items = s["labels"]["edit"]
        edit_map = {int(x["g"]): str(x["y"]) for x in edit_items}

        if set(edit_map.keys()) != set(g0_positions):
            raise ValueError(
                f"Sample {s.get('sample_id')} has mismatched G0/edit keys: "
                f"G0={g0_positions}, edit_keys={sorted(edit_map.keys())}"
            )

        K = int(s.get("meta", {}).get("K", K_DEFAULT))
        edit_choice: List[int] = []

        for g in g0_positions:
            y = edit_map[g]
            edit_choice.append(_choice_from_label(y, K))

        return {
            "sample_id": s["sample_id"],
            "doc_id": s["doc_id"],
            "atoms_text": atoms_text,
            "num_atoms": len(atoms_text),
            "num_gaps": len(b0),
            "b0": torch.tensor(b0, dtype=torch.long),
            "insert_labels": torch.tensor(insert_labels, dtype=torch.float),
            "g0_positions": torch.tensor(g0_positions, dtype=torch.long),
            "edit_choice": torch.tensor(edit_choice, dtype=torch.long),
            # debug / eval convenience
            "b_gold": torch.tensor([int(x) for x in s["b_gold"]], dtype=torch.long),
        }