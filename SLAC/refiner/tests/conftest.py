"""Portable offline fixtures; never load user corpora or pretrained weights."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def legal_records():
    return [
        {"sample_id": "mixed-long", "doc_id": "synthetic-long",
         "atoms": [{"text": x} for x in ["甲", "乙", "丙", "丁", "戊"]],
         "b0": [1, 0, 1, 0], "b_gold": [0, 1, 1, 1],
         "labels": {"edit": [{"g": 0, "y": "SHIFT:1"}, {"g": 2, "y": "KEEP"}],
                    "insert": [0, 0, 0, 1]}, "meta": {"K": 2, "confidence": 0.4}},
        {"sample_id": "mixed-short", "doc_id": "synthetic-short", "atoms": ["a", "b"],
         "b0": [0], "b_gold": [1], "labels": {"edit": [], "insert": [1]}, "meta": {"K": 2}},
        {"sample_id": "single-atom", "doc_id": "synthetic-single", "atoms": ["single"],
         "b0": [], "b_gold": [], "labels": {"edit": [], "insert": []}, "meta": {"K": 2}},
    ]

@pytest.fixture
def fixture_jsonl(tmp_path, legal_records):
    path = tmp_path / "synthetic_refiner.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in legal_records), encoding="utf-8")
    return path

@pytest.fixture
def refiner_batch(fixture_jsonl):
    from slac_refiner.datasets.collate import refiner_collate_fn
    from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
    dataset = RefinerDenoiseDataset(str(fixture_jsonl), sample_weight_field="meta.confidence", expected_k=2)
    return refiner_collate_fn([dataset[i] for i in range(len(dataset))])

class TinyTokenizer:
    def encode(self, text, *, add_special_tokens=True, truncation=False):
        ids = [2 + ord(char) % 61 for char in text]
        return [1, *ids, 1] if add_special_tokens else ids

    def __call__(self, texts, **kwargs):
        rows = [self.encode(text) for text in texts]
        if kwargs.get("return_length"):
            return {"length": [len(row) for row in rows]}
        width = kwargs["max_length"]
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            n = min(width, len(row))
            ids[i, :n] = torch.tensor(row[:n])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(64, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))

@pytest.fixture
def offline_pretrained(monkeypatch):
    from slac_refiner.models import atom_encoder
    monkeypatch.setattr(atom_encoder.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())
    monkeypatch.setattr(atom_encoder.AutoModel, "from_pretrained", lambda *a, **k: TinyBackbone())

@pytest.fixture
def tiny_refiner(offline_pretrained):
    from slac_refiner.models.refiner import BoundaryRefinerModel
    torch.manual_seed(55)
    return BoundaryRefinerModel(
        atom_model_name="offline-test-fixture", atom_max_length=16, atom_freeze=True,
        doc_hidden_size=8, doc_layers=1, doc_heads=2, doc_dropout=0,
        window_size=2, device="cpu", max_doc_atoms=16, k_shift=2,
    )
