# src/data/structure_dataset.py
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.models.struct_reconstructor import TYPE2ID, StructRecConfig


class StructureDataset(Dataset):
    """
    结构重建器训练用数据集：
      - 读取 index jsonl
      - 根据 split/source_type/domain 过滤
      - 每个样本 = 一篇文档
    """

    def __init__(
        self,
        index_path: str,
        split: str = "train",
        source_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        max_level: int = 8,
        max_docs: Optional[int] = None,
    ):
        super().__init__()
        self.index_path = Path(index_path)
        self.split = split
        self.source_types = set(source_types) if source_types else None
        self.domains = set(domains) if domains else None
        self.max_level = max_level

        self.entries: List[Dict] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                if self.source_types and rec.get("source_type") not in self.source_types:
                    continue
                if self.domains and rec.get("domain") not in self.domains:
                    continue
                self.entries.append(rec)
                if max_docs and len(self.entries) >= max_docs:
                    break

        if not self.entries:
            raise ValueError(f"No entries found for split={split} in {index_path}")

        print(f"[StructureDataset] Loaded {len(self.entries)} docs for split={split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        abs_path = Path(entry["abs_path"])

        with abs_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        units = doc["units"]  # 假设你的结构树 JSON 有一个 units 列表

        unit_ids = []
        unit_texts = []
        type_ids = []
        level_ids = []
        parent_ids = []

        for u in units:
            unit_ids.append(u["unit_id"])
            unit_texts.append(u["text"])

            t = u.get("type", "paragraph")
            type_ids.append(TYPE2ID.get(t, TYPE2ID["paragraph"]))

            level = int(u.get("level", 0))
            if level < 0:
                level = 0
            if level > self.max_level:
                level = self.max_level
            level_ids.append(level)

            parent_ids.append(u.get("parent_id", None))

        # parent_id → 序号索引：0 表示 root，>0 表示前面某个 unit
        id2pos = {uid: i for i, uid in enumerate(unit_ids)}
        parent_indices = []
        for i, p in enumerate(parent_ids):
            if p is None:
                parent_indices.append(0)
            else:
                pos = id2pos.get(p, None)
                if pos is None:
                    parent_indices.append(0)
                else:
                    # +1 是因为 0 预留给 root；同时保证父节点在前面
                    parent_indices.append(min(pos + 1, i + 1))

        sample = {
            "doc_id": doc.get("doc_id", entry.get("doc_id")),
            "unit_ids": unit_ids,
            "unit_texts": unit_texts,
            "type_labels": torch.tensor(type_ids, dtype=torch.long),
            "level_labels": torch.tensor(level_ids, dtype=torch.long),
            "parent_indices": torch.tensor(parent_indices, dtype=torch.long),
        }
        return sample


def build_tokenizer(config: StructRecConfig):
    return AutoTokenizer.from_pretrained(config.lm_name)


def make_collate_fn(tokenizer, max_unit_len: int = 256):
    """
    DataLoader 的 collate_fn，假设 batch_size=1：
      - 对这一篇文档的 unit 文本做 tokenize
      - 返回模型 forward 所需的张量
    """

    def collate_fn(batch: List[Dict]) -> Dict:
        assert len(batch) == 1, "当前版本假设 batch_size=1（一篇文档一个 batch）"
        sample = batch[0]

        unit_texts = sample["unit_texts"]
        enc = tokenizer(
            unit_texts,
            padding=True,
            truncation=True,
            max_length=max_unit_len,
            return_tensors="pt",
        )

        out = {
            "doc_id": sample["doc_id"],
            "unit_ids": sample["unit_ids"],
            "unit_texts": unit_texts,
            "input_ids": enc["input_ids"],             # (num_units, seq_len)
            "attention_mask": enc["attention_mask"],   # (num_units, seq_len)
            "type_labels": sample["type_labels"],
            "level_labels": sample["level_labels"],
            "parent_indices": sample["parent_indices"],
        }
        return out

    return collate_fn
