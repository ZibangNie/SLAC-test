# src/training/trainer.py
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import json
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW           # 用 torch 自带的 AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm              # 进度条

from src.models.struct_reconstructor import StructRecConfig, StructureReconstructor
from src.data.structure_dataset import StructureDataset, build_tokenizer, make_collate_fn
from src.utils.logging_utils import init_logger

# 日志根目录（绝对路径）
LOG_ROOT = Path(r"D:\code\Github\SLAC-test\log\structure_reconstructor")


def train_one_run(
    index_path: str,
    run_dir: str,
    config: StructRecConfig,
    num_epochs: int = 3,
    batch_size: int = 1,
    max_unit_len: int = 256,
    log_interval: int = 50,
):
    """
    单次训练入口：给定 index + run_dir + 配置，训练结构重建器。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ===== 初始化日志 =====
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = LOG_ROOT / f"{run_dir.name}_train.log"
    logger = init_logger(log_file)
    logger.info("Start training structure reconstructor")

    # 保存 config
    with (run_dir / "config_structrec.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    # ===== 1) 数据集 / DataLoader =====
    train_ds = StructureDataset(
        index_path=index_path,
        split="train",
        source_types=["A", "B"],
        domains=None,
        max_level=config.max_level,
    )
    dev_ds = StructureDataset(
        index_path=index_path,
        split="dev",
        source_types=["A", "B"],
        domains=None,
        max_level=config.max_level,
    )

    tokenizer = build_tokenizer(config)
    collate_fn = make_collate_fn(tokenizer, max_unit_len=max_unit_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ===== 2) 模型 / 优化器 / AMP scaler =====
    model = StructureReconstructor(config).to(device)

    # 只优化需要梯度的参数（冻结 BERT 后不会包含 BERT 参数）
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=2e-4, weight_decay=0.01)

    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== 3) 训练循环 =====
    best_dev_loss = float("inf")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            unit="batch",
            leave=True,
        )

        for batch in pbar:
            global_step += 1

            # ==== 新增：获取当前 batch 的 doc_id 并打日志 ====
            doc_id = batch.get("doc_id", None)
            # StructureDataset/ collate_fn 默认就是单文档 batch，doc_id 是 str
            # 如果将来改成 list，也可以取第一个：
            if isinstance(doc_id, (list, tuple)) and len(doc_id) == 1:
                doc_id = doc_id[0]

            # 在每个 batch 开始时打印一次当前文档 ID
            msg_start = f"[epoch {epoch} step {global_step}] START doc_id={doc_id}"
            print(msg_start)
            logger.info(msg_start)
            # =================================================

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            type_labels = batch["type_labels"].to(device)
            level_labels = batch["level_labels"].to(device)
            parent_indices = batch["parent_indices"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    type_labels=type_labels,
                    level_labels=level_labels,
                    parent_indices=parent_indices,
                )
                loss = outputs["loss"]

            optimizer.zero_grad()
            # AMP 下用 scaler 反向和 step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if global_step % log_interval == 0:
                msg = (
                    f"[epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.4f} "
                    f"lt={outputs['loss_type']:.4f} "
                    f"ll={outputs['loss_level']:.4f} "
                    f"lp={outputs['loss_parent']:.4f}"
                )
                logger.info(msg)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lt=f"{outputs['loss_type'].item():.4f}",
                    ll=f"{outputs['loss_level'].item():.4f}",
                    lp=f"{outputs['loss_parent'].item():.4f}",
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch} finished. "
            f"train_loss={avg_train_loss:.4f}, "
            f"time={time.time()-t0:.1f}s"
        )
        print(
            f"Epoch {epoch} finished. "
            f"train_loss={avg_train_loss:.4f}, "
            f"time={time.time()-t0:.1f}s"
        )

        # ===== dev 评估 =====
        dev_loss = evaluate_dev(model, dev_loader, device, epoch=epoch, use_amp=use_amp)
        logger.info(f"Epoch {epoch} dev_loss={dev_loss:.4f}")
        print(f"Epoch {epoch} dev_loss={dev_loss:.4f}")

        ckpt_path = run_dir / f"epoch_{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), run_dir / "best.pt")
            logger.info(f"[BEST] Updated best model at epoch {epoch}")
            print(f"[BEST] Updated best model at epoch {epoch}")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_dev_loss": best_dev_loss}, f, indent=2)

    logger.info(f"Training finished. best_dev_loss={best_dev_loss:.4f}")


@torch.no_grad()
def evaluate_dev(model, dev_loader, device, epoch: int | None = None, use_amp: bool = False):
    """
    dev 评估，同时用 tqdm 显示进度条。
    """
    model.eval()
    total_loss = 0.0
    n = 0

    desc = f"Eval epoch {epoch}" if epoch is not None else "Eval dev"
    pbar = tqdm(dev_loader, desc=desc, unit="batch", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        type_labels = batch["type_labels"].to(device)
        level_labels = batch["level_labels"].to(device)
        parent_indices = batch["parent_indices"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                type_labels=type_labels,
                level_labels=level_labels,
                parent_indices=parent_indices,
            )
            loss_val = outputs["loss"].item()

        total_loss += loss_val
        n += 1

        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / max(n, 1)
