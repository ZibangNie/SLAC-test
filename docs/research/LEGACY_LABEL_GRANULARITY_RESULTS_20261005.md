# 弱标签粒度与生产输入差异核查

2026-10-05。继续解释[全部切开也能获得较高 F1 的现象](CACHED_BOUNDARY_CHANGE_DIAGNOSIS_20261005.md)，只取已有 legacy-dev 文件的前两行，不按结果换样本。**两例的全部 `b_gold` 都恰好等于来源 unit 末端；旧纯规则 atomizer 重现了全部 150 个 atom 文本和 110 个 unit span，文本完全相同、span 字段和值一致。** 这些目标衡量的是来源单元分区恢复，不能直接说明改变分区有利于 JEV 判断、证据选择或回答。

本轮 API、模型推理、训练、tokenizer 调用均为 0。只解析这两个已曝光文档及其两个明确指向的 flat 源文件；没有解析上游大 JSONL、扩充样本或修改数据。

## 两例的实际重建

| 项目 | Dev ordinal 0 | Dev ordinal 1 |
|---|---:|---:|
| 来源 units / gold chunks | 52 | 58 |
| Atoms / 内部 gaps | 68 / 67 | 82 / 81 |
| Gold 正边界 | 51 | 57 |
| Gold 正例比例 | 0.761194 | 0.703704 |
| 只含一个 atom 的来源 unit | 40 | 38 |
| 每 unit 的 atom 数分布（1 / 2 / 3 / 4） | 40 / 9 / 2 / 1 | 38 / 17 / 2 / 1 |
| b0 相对 gold 不同的 gaps | 16 | 22 |
| b0 boundary F1 | 0.851852 | 0.830769 |
| 全部切开 boundary F1 | 0.864407 | 0.826087 |
| 按来源 unit 末端重建参考的 F1 | 1.000000 | 1.000000 |

两个命名 flat 源文件共 110 个 unit，其文本、类型、层级和 parent 值与所选记录的 `chunk0_units` 按序一致。原始 unit ID 为整数、保存后为字符串，110 项身份均在明确的 `str()` 转换后匹配，并非原 JSON 类型不变。用记录保存的 atomizer 配置调用归档生成器的纯函数，得到相同的 atom 文本与连续区间；未执行完整数据构建器或历史随机噪声生成器。

再分别用区间末端和相邻 atom 的 `chunk0_unit_id` 变化重建边界，两种计算都逐位等于保存的 `b_gold`。因此对于这两例，`U` 个非空来源 unit、`N` 个 atom 意味着 `U−1` 个正边界，正例比例为 `(U−1)/(N−1)`。多数 unit 只生成一个 atom，就会形成密集正边界。

**重建参考得到 1.0 是标签定义的机械复现，不是模型性能或可部署改进。** unit span／owner 携带了产生弱参考的干净结构；当前模型的 Dataset 使用 atom 文本、b0 等训练输入，不能把另行取用这份结构的完美重建冒充公平学习结果。密度高也不等于标签错误，来源 unit 是否具备良好语义仍未认证。

## 生成器实际定义了什么

[归档生成器](../../SLAC/refiner/data_backup_20260310_185454/build_refiner_from_real_dataset.py)的可核查路径为：

- `atomize_text`（288–322 行）在每个来源 unit 内规范化、句子切分、超长拆分和短片段合并；使用 `estimate_tokens`（122–128 行）的启发式估算。保存的 `tokenizer="bge-m3"` 字段不代表这里调用了 BGE tokenizer。
- `boundaries_from_spans`（330–341 行）把每个非末尾有效 unit 的 `e−1` 设为 1。682 行将该结果命名为 `b_gold`，不按 unit 的 type、level 或 parent 再决定是否合并。
- 693 行才对这个目标执行 DELETE／INSERT／SHIFT 噪声，741–742 行分别保存 noisy `b0` 与原 `b_gold`。本轮未复算历史随机过程，当前 b0 与目标不同只能作为观察，不能据此认证历史随机种子。

元数据的 `min_chunk_atoms=2`、`min_chunk_tokens=48`、`max_chunk_atoms=64` 和 `max_chunk_tokens=384` 在该文件中仅进入参数和记录，没有约束 gold 分区。这两例各有 40/38 个 gold chunk 只含一个 atom，足以说明保存的 minimum 并非目标上的硬保证。本轮没有把它们合并、重标或修改成满足 minimum 的新目标，也没有计算真实 BGE token 长度。

[Dataset](../../SLAC/refiner/slac_refiner/datasets/refiner_dataset.py)直接读取 `b_gold`（96、134 行）；[canonical 修复契约](../../SLAC/refiner/LABEL_CONTRACT.md)只重建编辑动作并验证回放，保留 atoms、b0 与 b_gold。机械修复不构成语义参考验收。

## 与生产输入的区别

[生产 builder](../../SLAC/refiner/pipeline/assemble/build_refiner_input.py)在 744–749 行从 `chunk0_units` 构建 atoms/spans，再通过 `build_b0_from_unit_spans`（613–644 行）把相邻非空 unit 末端作为 **b0**。这条路径没有训练数据中的 `apply_noise_to_gold`。

因此，同一种“来源 unit 末端投影”在旧训练路径中定义干净目标，在生产路径中定义初始输入。旧任务可以用于检验或研究合成去噪，但不能单凭其 F1 宣称模型纠正了实际 chunk0 的语义错误。此处说的是代码中的角色差异，不是已经证明生产输入本来完美。

两套 atomizer 也不是同一实现：默认约束数值相近，但 token 估算、规范化、长段切分和短段合并方向不同。生产实现优先向后合并短段，归档实现尝试并入前段；不能由共同 unit-end 原则推出 atom 序列、训练／生产边界向量或输入分布相同。本阶段未执行生产 builder 的自然样本对照。

## 范围、证据与研究决定

源文件目前的 unit 列表可以解释这两例目标，但没有单独的 `b_gold/gold_boundaries/boundaries` 顶层字段。这不排除 units 本身由人工或 LLM 设计；`llm_structured`、`llm_gold` 名称也不能认证语义审核。两例 selected 的 `orig_split=train`，而 flat 源的 `split=unknown`，保留旧谱系未认证标记，不推导新的独立性结论。

当前源码重现和当前源文件字节绑定不等于认证历史代码执行。独立审查核对了两条来源映射、150 个 atom、110 个 span 及全部数值；atomizer 部分复用了同一个归档纯函数，只证明机械可复现。完整结果、当前 SHA 与范围见[公开聚合 JSON](results/legacy_label_granularity_20261005.json)，[执行脚本](probe_legacy_label_granularity.py)不含 API 或模型依赖。原文、真实文档身份和逐 gap 向量留在本地私有产物中。

继续保留旧小样本作为去噪和实现诊断，不再把其边界 F1 当作 SLAC×JEV 框架改进的准入证据。下一步转到已有自然问答中的粒度与证据预算：先用少量已曝光文档的原文映射，检查不同切分能保留多少实际参考证据、占用多少完整输入预算；仅做离线可行性分析，明确区分利用参考的上界与可部署方法。若仍没有机制增量，不据此扩大训练或付费调用。
