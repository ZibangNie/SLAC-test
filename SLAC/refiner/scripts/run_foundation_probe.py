"""Bounded local optimization diagnostic; never a paper-quality evaluation.

Uses explicitly named repaired legacy train/dev files, filters by an explicit
encoder visibility contract, freezes BGE features, and compares two small heads.
No test split, network inference, or existing output is used or overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slac_refiner.datasets.refiner_dataset import RefinerDenoiseDataset
from slac_refiner.datasets.collate import refiner_collate_fn
from slac_refiner.label_contract import CONTRACT_VERSION, replay_labels
from slac_refiner.models.atom_encoder import AtomEncoder
from slac_refiner.models.doc_encoder import DocEncoder
from slac_refiner.models.heads import RefinerHeads
from slac_refiner.models.losses import RefinerLoss
from slac_refiner.decoding.dp_edit_decode import batch_decode
from slac_refiner.decoding.projector import ProjectorConfig, rebuild_chunks_from_boundary_vector
from slac_refiner.eval.metrics import boundary_prf


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_rows(path, tokenizer, n, max_atoms, max_length, excluded):
    selected, reasons, seen = [], Counter(), set(excluded)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reasons["examined"] += 1
            if row.get("orig_split") != "train":
                reasons["not_ancestral_train"] += 1
                continue
            if int(row.get("meta", {}).get("K", 6)) != 6:
                reasons["incompatible_shift_radius"] += 1
                continue
            if row["doc_id"] in seen:
                reasons["duplicate_doc_id"] += 1
                continue
            atoms = [a["text"] if isinstance(a, dict) else a for a in row["atoms"]]
            if not 4 <= len(atoms) <= max_atoms:
                reasons["atom_count_outside_contract"] += 1
                continue
            lengths = tokenizer(atoms, add_special_tokens=True, truncation=False, return_length=True)["length"]
            if max(lengths) > max_length:
                reasons["would_truncate_atom"] += 1
                continue
            if replay_labels(row["b0"], row["labels"]) != row["b_gold"]:
                raise ValueError("input violates repaired label contract")
            seen.add(row["doc_id"])
            selected.append(row)
            if len(selected) == n:
                break
    if len(selected) != n:
        raise ValueError(f"only {len(selected)} eligible diagnostic rows, requested {n}")
    return selected, dict(reasons), seen


class ProbeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.doc = DocEncoder(1024, hidden_size=128, num_layers=1, num_heads=4, dropout=0., window_size=8)
        self.heads = RefinerHeads(128, K=6, dropout=0.)

    def forward(self, batch):
        h = self.doc(batch["embeddings"], batch["atom_mask"]).h
        return self.heads(h, batch["g0_positions"], atom_mask=batch["atom_mask"])


def gpu_batch(dataset, index, embeddings, device):
    batch = refiner_collate_fn([dataset[index]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["embeddings"] = embeddings[index].to(device).unsqueeze(0)
    batch["atom_mask"] = torch.ones(1, batch["embeddings"].shape[1], dtype=torch.bool, device=device)
    # Both diagnostics use uniform document weights; this is not a confidence ablation.
    batch["sample_weight"] = torch.ones_like(batch["sample_weight"])
    return batch


def loss_for(mode, output, batch, criterion):
    if mode == "refiner":
        return criterion(output, batch).loss
    mask = batch["b_gold_mask"]
    logits = torch.where(mask, output.insert_logits, 0.)
    raw = F.binary_cross_entropy_with_logits(logits, batch["b_gold"].float(), reduction="none")
    return (raw * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, mode, dataset, embeddings, device, tokenizer, deadline):
    model.eval()
    sums = Counter()
    cfg = ProjectorConfig(max_chunk_atoms=64, min_chunk_atoms=1, max_chunk_chars=100000,
                          min_chunk_chars=1, max_chunk_tokens=512, min_chunk_tokens=1)
    for index in range(len(dataset)):
        if time.monotonic() > deadline:
            raise TimeoutError("diagnostic wall-time cap reached")
        batch = gpu_batch(dataset, index, embeddings, device)
        output = model(batch)
        if mode == "refiner":
            pred = batch_decode(batch["b0"], batch["g0_positions"], output.edit_choice_logits,
                                output.insert_logits, num_gaps=batch["num_gaps"], min_sep=0,
                                lambda_del=0., lambda_ins=0., lambda_shift=0.).pred_b[0]
        else:
            pred = (output.insert_logits[0].sigmoid() >= .5).int().tolist()
        gold = batch["b_gold"][0].tolist()
        projected = rebuild_chunks_from_boundary_vector(
            batch["atoms_text"][0], pred, cfg,
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False)),
            strict=True,
        )
        sums["raw_macro_f1"] += boundary_prf(pred, gold)["f1"]
        sums["projected_macro_f1"] += boundary_prf(projected["projected_b"], gold)["f1"]
        sums["projection_changed"] += int(pred != projected["projected_b"])
    return {k: value / len(dataset) if k.endswith("f1") else value for k, value in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "dev", "model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--train_rows", type=int, default=16)
    parser.add_argument("--dev_rows", type=int, default=8)
    parser.add_argument("--max_atoms", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_seconds", type=int, default=900)
    args = parser.parse_args()
    if not (1 <= args.steps <= 256 and 1 <= args.train_rows <= 32 and 1 <= args.dev_rows <= 32
            and 4 <= args.max_atoms <= 256 and 4 <= args.max_length <= 256 and 1 <= args.max_seconds <= 1800):
        raise ValueError("diagnostic hard caps exceeded")
    for name in ("train", "dev"):
        path = Path(getattr(args, name))
        if "test" in path.stem.lower() or name not in path.stem.lower():
            raise ValueError("only explicit train/dev JSONL files are accepted")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + args.max_seconds
    report = {"status": "running", "purpose": "legacy optimization diagnostic only",
              "independent_evaluation": False, "semantic_gold_verified": False,
              "contract": CONTRACT_VERSION, "configuration": vars(args), "models": {},
              "test_payload_read": False, "paid_api_calls": 0}
    source_files = [Path(__file__).resolve(), *sorted((ROOT / "slac_refiner").rglob("*.py"))]
    report["source_sha256"] = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        destination = output_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        report["source_sha256"][relative.as_posix()] = digest(path)
    def save():
        (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        torch.set_num_threads(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        report["versions"] = {"torch": torch.__version__}
        report["network"] = {"atom_dim": 1024, "hidden_size": 128, "layers": 1,
                             "heads": 4, "dropout": 0., "window_size": 8, "K": 6,
                             "seed": 13, "optimizer": "AdamW", "lr": .002,
                             "weight_decay": .01, "batch_size": 1,
                             "sample_weight": "uniform", "atom_pooling": "legacy_masked_mean"}
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        train, train_filter, used = select_rows(args.train, tokenizer, args.train_rows, args.max_atoms, args.max_length, set())
        dev, dev_filter, _ = select_rows(args.dev, tokenizer, args.dev_rows, args.max_atoms, args.max_length, used)
        report["selection"] = {"train": train_filter, "dev": dev_filter, "rule": "first eligible unique doc; ancestral train only; no semantic inspection"}
        report["input_sha256"] = {name: digest(getattr(args, name)) for name in ("train", "dev")}
        report["model_config_sha256"] = digest(Path(args.model) / "config.json")
        report["tokenizer_sha256"] = digest(Path(args.model) / "tokenizer.json")
        report["model_weights_sha256"] = digest(Path(args.model) / "model.safetensors")
        report["runner_sha256"] = digest(__file__)
        datasets = {}
        for split, rows in (("train", train), ("dev", dev)):
            path = output_dir / f"selected_{split}.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            datasets[split] = RefinerDenoiseDataset(str(path))
        report["selected"] = {split: {"documents": len(ds),
            "atoms": sum(ds[i]["num_atoms"] for i in range(len(ds))),
            "sha256": digest(output_dir / f"selected_{split}.jsonl"),
            "b0_raw_macro_f1": sum(boundary_prf(ds[i]["b0"].tolist(), ds[i]["b_gold"].tolist())["f1"]
                                   for i in range(len(ds))) / len(ds)} for split, ds in datasets.items()}
        save()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        encoder = AtomEncoder(args.model, max_length=args.max_length, freeze=True, device=device,
                              local_files_only=True, encode_batch_size=8, overflow_policy="error")
        if device == "cuda":
            encoder.half()
        embeddings = {}
        for split, dataset in datasets.items():
            embeddings[split] = []
            for index in range(len(dataset)):
                if time.monotonic() > deadline:
                    raise TimeoutError("diagnostic wall-time cap reached")
                emb = encoder.encode(dataset[index]["atoms_text"]).atom_embeddings.detach().float().cpu()
                embeddings[split].append(emb)
        report["truncated_atoms"] = encoder.truncated_atoms
        report["encoder_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        report["encoder_memory_scope"] = "PyTorch allocated including model loading and FP16 conversion; excludes driver and other applications"
        del encoder
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        from safetensors.torch import save_file
        save_file({f"{split}_{i}": tensor for split, values in embeddings.items() for i, tensor in enumerate(values)}, str(output_dir / "features.safetensors"))
        for mode in ("boundary", "refiner"):
            random.seed(13)
            torch.manual_seed(13)
            if device == "cuda":
                torch.cuda.manual_seed_all(13)
                torch.cuda.reset_peak_memory_stats()
            model = ProbeNetwork().to(device)
            criterion = RefinerLoss(insert_pos_weight=1., alpha_insert=1., alpha_edit=1., beta_cost=0.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
            initial = evaluate(model, mode, datasets["train"], embeddings["train"], device, tokenizer, deadline)
            history = []
            order = list(range(len(datasets["train"])))
            model.train()
            for step in range(args.steps):
                if time.monotonic() > deadline:
                    raise TimeoutError("diagnostic wall-time cap reached")
                if step % len(order) == 0:
                    random.shuffle(order)
                index = order[step % len(order)]
                batch = gpu_batch(datasets["train"], index, embeddings["train"], device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(mode, model(batch), batch, criterion)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite diagnostic loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                history.append(float(loss.detach()))
            metrics = {"initial_train": initial, "train": evaluate(model, mode, datasets["train"], embeddings["train"], device, tokenizer, deadline),
                       "legacy_dev": evaluate(model, mode, datasets["dev"], embeddings["dev"], device, tokenizer, deadline),
                       "first_16_mean_loss": sum(history[:16]) / len(history[:16]), "last_16_mean_loss": sum(history[-16:]) / len(history[-16:]),
                       "steps": len(history), "loss_history": history,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None}
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(output_dir / f"{mode}.safetensors"))
            report["models"][mode] = metrics
            save()
            print(json.dumps({"mode": mode, "train": metrics["train"], "legacy_dev": metrics["legacy_dev"]}), flush=True)
            del model, optimizer
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        save()


if __name__ == "__main__":
    main()
