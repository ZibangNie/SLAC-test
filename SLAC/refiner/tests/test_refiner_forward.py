import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn
from slac_refiner.models.refiner import BoundaryRefinerModel


def main():
    data_path = r"D:\code\Github\SLAC-test\SLAC\refiner\data\interim\atoms_b0\refiner_train_demo.jsonl"

    ds = RefinerDenoiseDataset(data_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=refiner_collate_fn)
    batch = next(iter(loader))

    model = BoundaryRefinerModel(
        atom_model_name=r"D:\code\Github\SLAC-test\SLAC\refiner\slac_refiner\models\bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
        atom_max_length=64,
        atom_freeze=True,
        doc_hidden_size=768,
        doc_layers=2,
        doc_heads=12,
        doc_dropout=0.1,
        window_size=16,
    )

    with torch.no_grad():
        out = model(batch)

    print("atom_embeddings shape =", tuple(out.atom_embeddings.shape))
    print("atom_mask shape =", tuple(out.atom_mask.shape))
    print("doc_hidden shape =", tuple(out.doc_hidden.shape))
    print("insert_logits shape =", tuple(out.insert_logits.shape))
    print("edit_logits shape =", tuple(out.edit_logits.shape))
    print("offset_pred shape =", tuple(out.offset_pred.shape))

    B = 1
    T = batch["num_atoms"][0].item()
    G = T - 1
    B0 = batch["g0_positions"].shape[1]

    assert out.atom_embeddings.shape[0] == B
    assert out.atom_embeddings.shape[1] == T

    assert out.doc_hidden.shape == (B, T, 768)
    assert out.insert_logits.shape == (B, G)
    assert out.edit_logits.shape == (B, B0, 3)
    assert out.offset_pred.shape == (B, B0)

    assert torch.isfinite(out.insert_logits).all()
    assert torch.isfinite(out.edit_logits).all()
    assert torch.isfinite(out.offset_pred).all()

    print("Refiner forward test passed.")


if __name__ == "__main__":
    main()