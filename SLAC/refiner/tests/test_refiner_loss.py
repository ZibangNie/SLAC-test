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
from slac_refiner.models.losses import RefinerLoss


LOCAL_BGE_M3_DIR = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"


def main():
    data_path = r"D:\code\Github\SLAC-test\SLAC\refiner\data\interim\atoms_b0\refiner_train_demo.jsonl"

    ds = RefinerDenoiseDataset(data_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=refiner_collate_fn)
    batch = next(iter(loader))

    model = BoundaryRefinerModel(
        atom_model_name=LOCAL_BGE_M3_DIR,
        atom_max_length=64,
        atom_freeze=True,
        doc_hidden_size=768,
        doc_layers=2,
        doc_heads=12,
        doc_dropout=0.1,
        window_size=16,
    )

    criterion = RefinerLoss(
        insert_pos_weight=16.0,
        alpha_insert=3.0,
        alpha_edit=1.0,
        beta_cost=0.05,
        lambda_del=1.0,
        lambda_ins=1.0,
        lambda_shift=0.25,
    )

    outputs = model(batch)
    loss_out = criterion(outputs, batch)

    print("loss_total =", float(loss_out.loss.item()))
    print("loss_insert =", float(loss_out.loss_insert.item()))
    print("loss_edit =", float(loss_out.loss_edit.item()))
    print("loss_cost_reg =", float(loss_out.loss_cost_reg.item()))
    print("stats =", loss_out.stats)

    assert torch.isfinite(loss_out.loss)
    assert torch.isfinite(loss_out.loss_insert)
    assert torch.isfinite(loss_out.loss_edit)
    assert torch.isfinite(loss_out.loss_cost_reg)

    print("Refiner loss test passed.")


if __name__ == "__main__":
    main()