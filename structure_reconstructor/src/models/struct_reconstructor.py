# src/models/struct_reconstructor.py
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
from contextlib import nullcontext

# ============ 基本 label 定义 ============

TYPE_LABELS = ["title", "heading", "paragraph", "list-item", "other"]
TYPE2ID = {t: i for i, t in enumerate(TYPE_LABELS)}


@dataclass
class StructRecConfig:
    """
    结构重建器模型的关键超参数配置。
    """
    lm_name: str = "bert-base-multilingual-cased"
    lm_hidden_size: int = 768          # 预训练 LM 的隐藏维度

    doc_hidden_size: int = 512         # 文档级 Transformer 的隐藏维度
    num_doc_layers: int = 4            # 文档级 Transformer 层数
    num_heads: int = 8                 # 文档级 Transformer 注意力头数
    dropout: float = 0.1

    num_type_labels: int = len(TYPE_LABELS)
    max_level: int = 8                 # 支持的最大 level 值（0..max_level）
    pointer_hidden_size: int = 256     # 指针网络内部维度

    loss_weight_type: float = 1.0
    loss_weight_level: float = 1.0
    loss_weight_parent: float = 2.0    # 父指针任务更难，权重大一些

    # 显存优化相关
    use_lm_grad_checkpoint: bool = True
    use_doc_grad_checkpoint: bool = True

    # 是否冻结预训练 LM，只训练上层结构模块
    freeze_lm: bool = True

    lm_unit_batch_size: int = 16


class StructureReconstructor(nn.Module):
    """
    结构重建器模型：
    - 输入：一篇文档的 unit 序列（每个 unit 对应一段文本的 token ids）
    - 输出：每个 unit 的 type / level / parent 预测，以及训练 loss。

    注意：
      为了简化父指针建模，这个版本假设 DataLoader 的 batch_size=1，
      即一个 batch 里只放一篇文档。
    """

    def __init__(self, config: StructRecConfig):
        super().__init__()
        self.config = config

        # 1) 预训练 LM：对每个 unit 独立编码
        self.lm = AutoModel.from_pretrained(config.lm_name)

        # === 新增：根据配置决定是否冻结 LM 参数 ===
        if config.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            # 冻结后设为 eval，可以关掉部分 dropout / 正则
            self.lm.eval()
        # =======================================

        # gradient checkpointing 可以显著降低 activations 显存占用，代价是反向时多一次前向
        if config.use_lm_grad_checkpoint and hasattr(self.lm, "gradient_checkpointing_enable"):
            self.lm.gradient_checkpointing_enable()
            if hasattr(self.lm.config, "use_cache"):
                self.lm.config.use_cache = False

        # 2) 从 LM hidden → 文档级 hidden
        self.unit_proj = nn.Linear(config.lm_hidden_size, config.doc_hidden_size)

        # 3) 文档级 Transformer，在 unit 序列上建模全局结构依赖
        #    用 ModuleList 方便逐层 checkpoint
        self.doc_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.doc_hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.doc_hidden_size * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=False,  # [seq_len, batch, dim]
                    norm_first=True,
                )
                for _ in range(config.num_doc_layers)
            ]
        )
        self.doc_norm = nn.LayerNorm(config.doc_hidden_size)

        # 4) 多任务输出头：type、level、parent pointer
        self.type_head = nn.Linear(config.doc_hidden_size, config.num_type_labels)
        self.level_head = nn.Linear(config.doc_hidden_size, config.max_level + 1)

        # Pointer network: q, k 映射 + 一个 root 节点向量
        self.pointer_q = nn.Linear(config.doc_hidden_size, config.pointer_hidden_size)
        self.pointer_k = nn.Linear(config.doc_hidden_size, config.pointer_hidden_size)
        self.root_pointer_embedding = nn.Parameter(
            torch.randn(config.pointer_hidden_size)
        )

        self.dropout = nn.Dropout(config.dropout)

        self._init_parameters()

    def _init_parameters(self):
        # 简单的 xavier 初始化，LM 保持预训练权重
        nn.init.xavier_uniform_(self.unit_proj.weight)
        nn.init.zeros_(self.unit_proj.bias)

        for head in [self.type_head, self.level_head, self.pointer_q, self.pointer_k]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        nn.init.normal_(self.root_pointer_embedding, mean=0.0, std=0.02)

    # ---------- 前向计算 ----------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        type_labels: Optional[torch.Tensor] = None,
        level_labels: Optional[torch.Tensor] = None,
        parent_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        参数：
          input_ids:      (num_units, seq_len)
          attention_mask: (num_units, seq_len)
          type_labels:    (num_units,)
          level_labels:   (num_units,)
          parent_indices: (num_units,) 每个位置 ∈ [0 .. i]，0 表示 root，其余表示前面某个 unit

        返回：
          dict: {
            "loss": 总 loss（训练时有）,
            "loss_type", "loss_level", "loss_parent",
            "logits_type", "logits_level", "pointer_logits"
          }
        """
        device = input_ids.device
        num_units, seq_len = input_ids.size()

        # ===== 分块调用 BERT，避免一次性吃下超长文档 =====
        # 如果 freeze_lm=True，则在 no_grad() 下调用 LM，进一步节省显存
        lm_ctx = torch.no_grad() if self.config.freeze_lm else nullcontext()

        cls_list = []
        with lm_ctx:
            # 按 unit 维度分块：每次处理 lm_unit_batch_size 个 unit
            for start in range(0, num_units, self.config.lm_unit_batch_size):
                end = min(start + self.config.lm_unit_batch_size, num_units)

                chunk_input_ids = input_ids[start:end]  # (chunk_units, seq_len)
                chunk_attention_mask = attention_mask[start:end]

                lm_out = self.lm(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                )
                # 取 CLS 作为当前 chunk 的 unit 表示
                cls_chunk = lm_out.last_hidden_state[:, 0, :]  # (chunk_units, lm_hidden)
                cls_list.append(cls_chunk)

        # 拼回整篇文档的 unit 表示：(num_units, lm_hidden)
        unit_repr = torch.cat(cls_list, dim=0)
        # ===== 后面继续用原来的逻辑 =====

        # 2) 项映射 + dropout
        unit_repr = self.dropout(self.unit_proj(unit_repr))  # (num_units, doc_hidden)

        # 3) 文档级 Transformer（seq_len = num_units, batch = 1）
        x = unit_repr.unsqueeze(1)  # (num_units, 1, hidden)
        for layer in self.doc_layers:
            if self.config.use_doc_grad_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.doc_norm(x)
        doc_output = x.squeeze(1)  # (num_units, hidden)

        # 4) 多任务头
        logits_type = self.type_head(self.dropout(doc_output))        # (num_units, num_type)
        logits_level = self.level_head(self.dropout(doc_output))      # (num_units, max_level+1)

        # pointer q/k： (num_units, pointer_hidden)
        q = self.pointer_q(doc_output)
        k = self.pointer_k(doc_output)

        # root k: (1, pointer_hidden)
        root_k = self.root_pointer_embedding.unsqueeze(0)

        # 拼接 root： k_all[0] = root, k_all[1:] = 各 unit
        k_all = torch.cat([root_k, k], dim=0)   # (num_units+1, pointer_hidden)

        # pointer logits: scores[i, j] = q_i · k_all_j
        pointer_logits = torch.matmul(q, k_all.t())  # (num_units, num_units+1)

        # 遮罩不合法的父节点（只允许指向 root 或前序单元）
        idx_i = torch.arange(num_units, device=device).unsqueeze(1)           # (num_units, 1)
        idx_j = torch.arange(num_units + 1, device=device).unsqueeze(0)       # (1, num_units+1)
        valid_mask = idx_j <= idx_i        # (num_units, num_units+1)
        mask_value = torch.finfo(pointer_logits.dtype).min  # 对 float16/float32 自适应
        pointer_logits = pointer_logits.masked_fill(~valid_mask, mask_value)

        output: Dict[str, torch.Tensor] = {
            "logits_type": logits_type,
            "logits_level": logits_level,
            "pointer_logits": pointer_logits,
        }

        # 5) 计算多任务 loss（训练时）
        if type_labels is not None and level_labels is not None and parent_indices is not None:
            loss_type = F.cross_entropy(logits_type, type_labels)
            loss_level = F.cross_entropy(logits_level, level_labels)
            loss_parent = F.cross_entropy(pointer_logits, parent_indices)

            loss = (
                self.config.loss_weight_type * loss_type
                + self.config.loss_weight_level * loss_level
                + self.config.loss_weight_parent * loss_parent
            )

            output.update(
                {
                    "loss": loss,
                    "loss_type": loss_type.detach(),
                    "loss_level": loss_level.detach(),
                    "loss_parent": loss_parent.detach(),
                }
            )

        return output

    # ---------- 推理时的结构树解码 ----------

    @torch.no_grad()
    def predict_structure(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_texts: List[str],
        unit_ids: List[int],
    ):
        """
        给定一篇文档（已经按 unit 分好、完成 tokenization），
        输出预测的结构树列表 [{unit_id, text, type, level, parent_id}, ...]。
        """
        self.eval()
        device = next(self.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.forward(input_ids, attention_mask)
        logits_type = outputs["logits_type"]          # (num_units, num_type)
        logits_level = outputs["logits_level"]        # (num_units, max_level+1)
        pointer_logits = outputs["pointer_logits"]    # (num_units, num_units+1)

        type_ids = logits_type.argmax(dim=-1).cpu().tolist()
        level_ids = logits_level.argmax(dim=-1).cpu().tolist()
        parent_idx = pointer_logits.argmax(dim=-1).cpu().tolist()  # 0=root, 1..i

        results = []
        num_units = len(unit_ids)
        for i in range(num_units):
            t_id = type_ids[i]
            l_id = level_ids[i]
            p_idx = parent_idx[i]
            if p_idx == 0:
                parent_id = None
            else:
                parent_id = unit_ids[p_idx - 1]   # p_idx-1 是父单元在序列中的 index

            results.append(
                {
                    "unit_id": unit_ids[i],
                    "text": unit_texts[i],
                    "type": TYPE_LABELS[t_id],
                    "level": int(l_id),
                    "parent_id": parent_id,
                }
            )

        return results
