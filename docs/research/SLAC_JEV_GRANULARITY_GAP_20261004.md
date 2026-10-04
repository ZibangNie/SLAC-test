# 当前 JEV 证据与 SLAC 接入粒度的缺口

2026-10-04，代码与既有结果的只读核对。本次没有调用 API、读取权重、训练或扫描数据集正文。

**当前正向结果支持原生单元上的 JEV 证据选择，尚未验证真正 Refiner 输出粒度上的完整 SLAC×JEV 框架。** 需要先把判断单位和来源映射接清楚，才能提出可比较的框架实验。接线本身不是创新贡献。

## 当前实验实际使用的机制

| 环节 | 当前 Qasper/JEV 研究入口 | 生产 SLAC 接口及差异 |
|---|---|---|
| 文本单元 | `run_qasper_evidence_baselines.build_units` 将标题、摘要、章节名和原生段落映射为 `Unit`；`text` 用于模型输入，`native_text` 用于原生精确评价 | Refiner 在 atom 边界上产生新的 chunk；`export_refined_chunks.build_leaf_records` 为每个 atom 建立 leaf |
| 候选与判断 | 固定 Dense/BGE 缓存候选，逐项判断原问题与单个 `Unit.text` | 生产流程还含 QueryPlanner、leaf/chunk/anchor 通道、owner 聚合和树扩展，当前单元 support 对照没有执行它们 |
| 排序与包 | `probability_ranking` 排序，研究版 `pack_ranked` 按最多三单元／1,024 tokens 打包 | 生产 `run_retrieval_pipeline` 使用融合、扩展和 `pack_evidence`；不能将其存在等同于已被当前实验验证 |
| 既有结构对照 | 历史规则 chunk 桥接复用了 `aggregate_hits_to_chunk_candidates` 和 `fuse_candidate_scores_rrf` | 这些 chunk 是固定规则构造，尚不是学习到的 Boundary Refiner 输出 |

代码入口：[原生 Unit](run_qasper_evidence_baselines.py)、[当前 JEV 迁移评价](evaluate_score_transfer_microdiagnostic.py)、[生产检索流程](../../SLAC/retrieval/run/run_retrieval_pipeline.py)、[Refiner 导出器](../../SLAC/refiner/pipeline/assemble/export_refined_chunks.py)。

## 一个值得先检查的问题

**原生段落上测到的 JEV 选择信号，能否迁移到实际 Refiner 的 atom/chunk 粒度？**

当前导出器的 `build_refined_chunks` 从 `b_pred_sparse` 恢复 atom 区间，再通过 `_join_atoms` 规范化和拼接文字；`build_leaf_records` 则按 atom 输出 leaf。原生段落与这些新单元可能在边界、文本字节、包含的信息、身份和 token 数上发生变化。边界解码器已有准确 span 时的无损拼接能力，也不等于生产导出器已经沿用同一文本契约，必须核对实际路径。

[support 契约](openrouter_decision_client.py)明确接收 `query` 与含 `id/text` 的单个 `unit`。旧段落判断不等价于新 chunk 判断。不能把多个段落分数求和、取最大值，或将段落标签贴到 atom/chunk 上，然后声称测量了 JEV 对新粒度的效果。

下一步只做固定小样本的来源／粒度机械检查：沿真实 adapter/exporter 路径核对原生段落到 atom 再到 chunk 的覆盖、空白转换、身份、实际整包预算和精确缓存可复用范围；区分已保存的真实边界输出与用于接口检查的人工边界。首先判定现有产物是否足以构造公平对照，不先启动训练或模型调用。

缓存可以回答“旧文本是否完整保留”“哪些输入字节真正相同”“哪些旧判断确实不可复用”；不能独自回答新 chunk 的 support 质量或新框架的 Answer F1。若没有真实 Refiner 输出，必须将检查称为接口可行性，不能把人工边界或规则 chunk 伪装成 Refiner 的质量结果。

原生 Evidence F1 比较完整参考段落字符串，任意新 chunk 不能直接套用同一读数作公平质量比较。来源映射必须先明确哪些内容实际进入模型证据包，部分片段不能自动回填成整个原生段落命中；预算也应计算实际输入的 chunk。这个评价对应关系与标签复用边界，都需要在后续质量实验前固定。

## 必须带入的既有负结果

[Refiner 基础微诊断](FOUNDATION_PROGRESS_20260926.md)的 legacy-dev 边界 F1 为 0.8358，原 `b0` 为 0.8427，普通分类器为 0.8786。这是八篇 legacy-dev 的小诊断，不能据此决定最终架构优劣；标签谱系及跨 split 问题仍未获得正式训练准入，后续编辑机制对照也缺少公平匹配的 `b0`-conditioned classifier。当前不扩大训练。

[规则双索引对照](NATIVE_DUAL_INDEX_RESULTS_20260927.md)增加了候选召回，但指定文档和跨库的按题最终 F1 点差分别为 −0.004638、−0.011216。它不证明真实 Refiner 必然失败，也不能被省略以制造结构机制已有收益的印象。

因此当前只保留可检验的框架外推问题，不提出新的结构模块、不复活已关闭的标题/定义补齐/普通条件选择/廉价路由分支，也不把这份接口检查当作论文创新成立。
