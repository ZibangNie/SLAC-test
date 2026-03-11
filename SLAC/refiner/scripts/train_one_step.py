from __future__ import annotations

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run one training step for Boundary Refiner.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--doc_layers", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=16)
    return parser.parse_args()


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def main():
    args = parse_args()

    ds = RefinerDenoiseDataset(args.data)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=refiner_collate_fn,
    )
    batch = next(iter(loader))

    model = BoundaryRefinerModel(
        atom_model_name=LOCAL_BGE_M3_DIR,
        atom_max_length=64,
        atom_freeze=True,
        doc_hidden_size=768,
        doc_layers=args.doc_layers,
        doc_heads=12,
        doc_dropout=0.1,
        window_size=args.window_size,
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(batch)
    loss_out = criterion(outputs, batch)
    loss_out.loss.backward()

    gnorm = grad_norm(trainable_params)
    optimizer.step()

    print("one-step train done")
    print("loss_total =", float(loss_out.loss.item()))
    print("loss_insert =", float(loss_out.loss_insert.item()))
    print("loss_edit =", float(loss_out.loss_edit.item()))
    print("loss_cost_reg =", float(loss_out.loss_cost_reg.item()))
    print("grad_norm =", gnorm)

    atom_trainable = any(p.requires_grad for p in model.atom_encoder.backbone.parameters())
    print("atom_encoder_trainable =", atom_trainable)


if __name__ == "__main__":
    main()