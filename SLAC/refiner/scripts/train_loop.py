from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn
from slac_refiner.models.refiner import BoundaryRefinerModel
from slac_refiner.models.losses import RefinerLoss
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.eval.metrics import (
    boundary_prf,
    edit_action_accuracy,
    insert_accuracy,
    aggregate_metric_dicts,
)


LOCAL_BGE_M3_DIR = r"D:\code\Github\SLAC-test\SLAC\refiner\slac_refiner\models\bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal train loop for Boundary Refiner")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--doc_layers", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
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
    return model


def build_criterion():
    return RefinerLoss(
        insert_pos_weight=4.0,
        alpha_insert=1.0,
        alpha_edit=1.0,
        alpha_offset=0.5,
        beta_cost=0.05,
        lambda_del=1.0,
        lambda_ins=1.0,
        lambda_shift=0.25,
    )


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, model.device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss_out = criterion(outputs, batch)
        loss_out.loss.backward()
        optimizer.step()

        total_loss += float(loss_out.loss.item())
        total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    boundary_metrics = []
    edit_metrics = []
    insert_metrics = []

    for batch in loader:
        batch = move_batch_to_device(batch, model.device)
        outputs = model(batch)

        dec = batch_decode(
            b0=batch["b0"],
            g0_positions=batch["g0_positions"],
            edit_logits=outputs.edit_logits,
            offset_pred=outputs.offset_pred,
            insert_logits=outputs.insert_logits,
            K=6,
            insert_threshold=0.5,
            min_sep=1,
            lambda_del=1.0,
            lambda_ins=1.0,
            lambda_shift=0.25,
        )

        B = batch["b0"].shape[0]

        for i in range(B):
            pred_b = dec.pred_b[i]
            gold_b = batch["b_gold"][i].tolist()

            pred_edit = dec.pred_edit_labels[i]
            pred_insert = dec.pred_insert_labels[i]

            # 从 dataset 原始样本中无法直接拿 gold_edit，这里用当前 batch 还原
            g0_positions = [int(x) for x in batch["g0_positions"][i].tolist() if int(x) >= 0]
            edit_cls = batch["edit_cls"][i].tolist()
            edit_offset = batch["edit_offset"][i].tolist()

            gold_edit = []
            for g, c, off in zip(g0_positions, edit_cls, edit_offset):
                if c == 0:
                    y = "KEEP"
                elif c == 1:
                    y = "DEL"
                elif c == 2:
                    y = f"SHIFT:{int(off)}"
                else:
                    continue
                gold_edit.append({"g": g, "y": y})

            gold_insert = batch["insert_labels"][i].tolist()

            boundary_metrics.append(boundary_prf(pred_b, gold_b))
            edit_metrics.append(edit_action_accuracy(pred_edit, gold_edit))
            insert_metrics.append(insert_accuracy(pred_insert, gold_insert))

    return {
        "boundary": aggregate_metric_dicts(boundary_metrics),
        "edit": aggregate_metric_dicts(edit_metrics),
        "insert": aggregate_metric_dicts(insert_metrics),
    }


def maybe_save(model, save_dir: str | None, epoch: int, dev_metrics: Dict):
    if save_dir is None:
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "dev_metrics": dev_metrics,
    }
    torch.save(ckpt, save_path / f"epoch_{epoch}.pt")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds = RefinerDenoiseDataset(args.train)
    dev_ds = RefinerDenoiseDataset(args.dev)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=refiner_collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=refiner_collate_fn,
    )

    model = build_model(args)
    criterion = build_criterion()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer)
        dev_stats = evaluate(model, dev_loader)

        boundary_f1 = dev_stats["boundary"].get("f1", 0.0)
        if boundary_f1 > best_f1:
            best_f1 = boundary_f1
            maybe_save(model, args.save_dir, epoch, dev_stats)

        print(f"epoch={epoch}")
        print("  train loss =", train_stats["loss"])
        print("  dev boundary =", dev_stats["boundary"])
        print("  dev edit =", dev_stats["edit"])
        print("  dev insert =", dev_stats["insert"])
        print("  best boundary f1 =", best_f1)


if __name__ == "__main__":
    main()