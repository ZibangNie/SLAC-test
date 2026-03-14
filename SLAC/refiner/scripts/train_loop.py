from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

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
from slac_refiner.decoding.projector import rebuild_chunks_from_boundary_vector, ProjectorConfig
from slac_refiner.eval.metrics import (
    boundary_prf,
    edit_action_accuracy,
    insert_accuracy,
    insert_prf,
    shift_mae,
    chunk_length_stats_from_spans,
    aggregate_metric_dicts,
)

LOCAL_BGE_M3_DIR = r"/root/autodl-tmp/models/bge-m3/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

DEFAULT_PROJECTOR_CFG = ProjectorConfig(
    max_chunk_atoms=64,
    min_chunk_atoms=2,
    max_chunk_chars=1600,
    min_chunk_chars=20,
    max_chunk_tokens=384,
    min_chunk_tokens=48,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train loop for Boundary Refiner (supports continuation finetune).")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2, help="Additional epochs to run from init_ckpt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--doc_layers", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    parser.add_argument("--insert_pos_weight", type=float, default=16.0)
    parser.add_argument("--alpha_insert", type=float, default=3.0)
    parser.add_argument("--alpha_edit", type=float, default=1.0)
    parser.add_argument("--beta_cost", type=float, default=0.05)

    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--sample_weight_field", type=str, default=None)

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


def build_criterion(args):
    return RefinerLoss(
        insert_pos_weight=args.insert_pos_weight,
        alpha_insert=args.alpha_insert,
        alpha_edit=args.alpha_edit,
        alpha_offset=0.5,
        beta_cost=args.beta_cost,
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


def load_init_ckpt(model, ckpt_path: str | None) -> int:
    if ckpt_path is None:
        return 0

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing[:20]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected[:20]}")

    ckpt_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    print(f"[init_ckpt] loaded model from {ckpt_path}")
    print(f"[init_ckpt] checkpoint epoch = {ckpt_epoch}")
    return ckpt_epoch


def train_one_epoch(model, loader, criterion, optimizer, epoch: int):
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, model.device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss_out = criterion(outputs, batch)
        loss_out.loss.backward()
        optimizer.step()

        total_loss += float(loss_out.loss.item())
        total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "ins": f"{float(loss_out.loss_insert.item()):.4f}",
            "edit": f"{float(loss_out.loss_edit.item()):.4f}",
        })

    return {
        "loss": total_loss / max(total_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader, projector_cfg):
    model.eval()

    boundary_metrics = []
    edit_metrics = []
    insert_acc_metrics = []
    insert_prf_metrics = []
    shift_metrics = []
    length_metrics = []

    K = 6
    pbar = tqdm(loader, desc="dev", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, model.device)
        outputs = model(batch)

        dec = batch_decode(
            b0=batch["b0"],
            g0_positions=batch["g0_positions"],
            edit_choice_logits=outputs.edit_choice_logits,
            insert_logits=outputs.insert_logits,
            K=K,
            insert_threshold=0.5,
            min_sep=1,
            lambda_del=1.0,
            lambda_ins=1.0,
            lambda_shift=0.25,
        )

        B = batch["b0"].shape[0]

        for i in range(B):
            raw_pred_b = dec.pred_b[i]
            pred_edit = dec.pred_edit_labels[i]
            pred_insert = dec.pred_insert_labels[i]

            gold_b = batch["b_gold"][i].tolist()
            gold_insert = batch["insert_labels"][i].tolist()

            atoms_text = batch["atoms_text"][i]
            gap_scores = outputs.insert_logits[i].detach().cpu().tolist()

            projected = rebuild_chunks_from_boundary_vector(
                atoms_text=atoms_text,
                b=raw_pred_b,
                cfg=projector_cfg,
                gap_scores=gap_scores,
            )
            pred_b = projected["projected_b"]
            spans_eval = projected["spans_after_merge"]

            g0_positions_i = [int(x) for x in batch["g0_positions"][i].tolist() if int(x) >= 0]
            edit_choice_i = batch["edit_choice"][i].tolist()

            gold_edit = []
            for g, choice in zip(g0_positions_i, edit_choice_i):
                choice = int(choice)
                if choice < 0:
                    continue
                if choice == 0:
                    y = "DEL"
                else:
                    k = (choice - 1) - K
                    y = "KEEP" if k == 0 else f"SHIFT:{k}"
                gold_edit.append({"g": g, "y": y})

            boundary_metrics.append(boundary_prf(pred_b, gold_b))
            edit_metrics.append(edit_action_accuracy(pred_edit, gold_edit))
            insert_acc_metrics.append(insert_accuracy(pred_insert, gold_insert))
            insert_prf_metrics.append(insert_prf(pred_insert, gold_insert))
            shift_metrics.append(shift_mae(pred_edit, gold_edit))
            length_metrics.append(chunk_length_stats_from_spans(spans_eval))

    return {
        "boundary": aggregate_metric_dicts(boundary_metrics),
        "edit": aggregate_metric_dicts(edit_metrics),
        "insert_acc": aggregate_metric_dicts(insert_acc_metrics),
        "insert_prf": aggregate_metric_dicts(insert_prf_metrics),
        "shift": aggregate_metric_dicts(shift_metrics),
        "length": aggregate_metric_dicts(length_metrics),
    }


def maybe_save(model, optimizer, save_dir: str | None, epoch: int, dev_metrics: Dict, args):
    if save_dir is None:
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dev_metrics": dev_metrics,
        "train_args": vars(args),
    }
    torch.save(ckpt, save_path / f"epoch_{epoch}.pt")


def append_epoch_log(log_dir: str | None, row: Dict):
    if log_dir is None:
        return
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "train_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_ds = RefinerDenoiseDataset(args.train, sample_weight_field=args.sample_weight_field)
    dev_ds = RefinerDenoiseDataset(args.dev, sample_weight_field=args.sample_weight_field)

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
    start_epoch_base = load_init_ckpt(model, args.init_ckpt)

    criterion = build_criterion(args)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    start_epoch = start_epoch_base + 1
    end_epoch = start_epoch_base + args.epochs

    for epoch in range(start_epoch, end_epoch + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        dev_stats = evaluate(model, dev_loader, projector_cfg=DEFAULT_PROJECTOR_CFG)

        boundary_f1 = dev_stats["boundary"].get("f1", 0.0)
        if boundary_f1 > best_f1:
            best_f1 = boundary_f1
            maybe_save(model, optimizer, args.save_dir, epoch, dev_stats, args)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "dev_boundary": dev_stats["boundary"],
            "dev_edit": dev_stats["edit"],
            "dev_insert_acc": dev_stats["insert_acc"],
            "dev_insert_prf": dev_stats["insert_prf"],
            "dev_shift": dev_stats["shift"],
            "dev_length": dev_stats["length"],
            "best_boundary_f1": best_f1,
        }
        append_epoch_log(args.log_dir, row)

        print(f"epoch={epoch}")
        print("  train loss =", train_stats["loss"])
        print("  dev boundary =", dev_stats["boundary"])
        print("  dev edit =", dev_stats["edit"])
        print("  dev insert_acc =", dev_stats["insert_acc"])
        print("  dev insert_prf =", dev_stats["insert_prf"])
        print("  dev shift =", dev_stats["shift"])
        print("  dev length =", dev_stats["length"])
        print("  best boundary f1 =", best_f1)


if __name__ == "__main__":
    main()