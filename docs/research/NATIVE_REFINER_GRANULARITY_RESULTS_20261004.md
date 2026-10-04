# 原生单元到 Refiner 的两文档机械检查

2026-10-04。新增 API、模型推理、训练、权重读取和金标读取均为 0。完整精度与来源 hash 见[聚合结果](results/native_refiner_granularity_20261004.json)，执行入口见[探针源码](probe_native_refiner_granularity.py)。

**真实 builder/exporter 的恒等边界路径可以处理这两篇固定文档，但保持边界不保证保持输入字节。** 这解决了部分接口可行性问题，尚未验证学习到的 Refiner 边界、JEV 对新粒度的判断或完整框架的质量收益。

## 范围与真实执行路径

按[上一阶段固定六题](HARD_NO_CACHED_SAMPLE_20261004.md)的既定顺序，取前两篇文档的全部 60/23 个 native Unit，而不是只取它们的 16/14 个候选。原生文档、规范文档、block、Unit、leaf 的文本、来源坐标和 hash 在这 83 个单元内核对；没有扩展到其他文档。它们是已曝光开发数据，不是独立确认样本。

将 native Unit 直接投影为 `chunk0_units`，数值 `unit_id=Unit.order`，保留独立原生 ID 映射与已有 leaf 的 path/depth。调用生产 `build_refiner_input_from_chunk0` 的默认配置，再用明确人工构造的 **`b0_identity_fixture`** 设置 `b_pred_sparse=b0_sparse`，最后调用生产 `export_refined_chunks_from_candidate`。没有执行 raw segmenter、`chunk0_adapter`、边界预测模型或完整检索流程；固定 native 边界正是本检查的输入条件。

检查前已冻结输入、协议、执行代码、生产代码与本地 tokenizer 的 SHA256；首次运行完成，没有按结果调参或换样本。发布前仅将 exporter 的混合换行统一为 LF，源码 AST 不变；重跑的完整 fixture 字节及报告指标一致（仅来源绑定变化），两次运行不作为独立观察。BGE tokenizer 仅本地加载，计数包含 special tokens、禁用截断。`AtomEncoder` 的默认 `max_length=128`、`overflow_policy="error"` 来自源码，未实例化编码器。

## 实测结果

| 指标 | 文档 1 | 文档 2 | 合计或最大值 |
|---|---:|---:|---:|
| 原生单元 / 恒等 chunk | 60 | 23 | 83 |
| atom / leaf | 177 | 74 | 251 |
| atom 最大实际 BGE tokens | 82 | 84 | 84 |
| 超过 128 tokens 的 atom | 0 | 0 | 0 |
| 原生文本与导出文本字节相同 | 54 | 19 | 73 |
| 仅空白变化 | 6 | 0 | 6 |
| 匹配 builder 轻量规范化的其余变化 | 0 | 4 | 4 |
| 其他变化 | 0 | 0 | 0 |
| 原候选文本字节相同 | 16/16 | 12/14 | 28/30 |

83 个单元的 atom 区间连续、完整且无重叠；恒等 chunk 保持各自区间，251 个 leaf 的文本和 owner 与对应 atom/chunk 一致。原生文本与研究 Unit 文本在这 83 项中全部相同；atom 拼接文本与最终导出文本也全部相同。因此观察到的十项变化已经出现在 builder 的 atom 化路径中。这些分类是字节／规范化事实，没有判定语义等价或语义损失。

独立核验复算了固定 83 单元的映射／文本类别、251 个 atom 长度和两包的三种完整渲染，全部通过。

所有 atom 均通过此次默认长度条件，不能据此推断模型质量、内存可行性或全库长度分布。

对同两个旧三单元证据包保持成员及原文顺序，使用研究版 `[id]\ntext` renderer 计算整个包：

| 渲染条件 | 包 1 tokens | 包 2 tokens | 与各自旧包的字节关系 |
|---|---:|---:|---|
| 原生 ID + 原文本 | 324 | 430 | 两个旧 hash、计数精确复现 |
| 原生 ID + 恒等导出文本 | 324 | 430 | 包 1 相同；包 2 hash 改变 |
| 新 chunk ID + 恒等导出文本 | 357 | 466 | 两包均改变 |

两包均低于原 1,024-token 上限。这只是相同成员的渲染对照，不是重新选包、生产 `pack_evidence` 对照或质量评价。ID 格式本身增加了 33/36 tokens；相同 token 数也不能证明相同文本。

**28/30 文本相同不等于 28 个 API 缓存命中。** 完整缓存身份还涉及 endpoint、prompt version、任务和单元 ID、payload、模型及参数。本轮没有读取或复算 JEV 标签、分数、答案和 reference。边界变化后的 atom/chunk 更不能直接继承原生段落的判断或 Evidence F1。

## 同时修复的导出器错误

合成复现发现：中间空 seed 被 builder 跳过后，`unit2atom_span` 的列表下标不再等于 `chunk0_units` 下标。旧 exporter 仍以获胜 span 的下标取 seed，导致第二个非空 chunk 的 path/depth/parent 错取自中间空单元；原有两个严格校验器仍通过。chunk 文本和 atom 边界本身未改变。

[修复](../../SLAC/refiner/pipeline/assemble/export_refined_chunks.py)按 span 的 `unit_id` 查找实际 seed，并拒绝缺失、重复或未知 ID，保留最大 overlap、平局选首个 span 和可选 metadata 缺失行为。[14 项定向回归测试](../../tests/research/test_refiner_seed_mapping.py)通过，覆盖空白单元、稀疏／重排 ID、平局及无效映射。此次两文档的 83 项均非空，不声称该错误污染了它们或旧研究指标；正常 chunk0 adapter 会预先过滤空文本，复现针对直接 builder 入口允许的输入。

## 已解决与仍未解决

当前只证明 native Unit 可沿这条真实机械路径建立 unit→atom→chunk→leaf 映射，并量化文本和渲染变化。生产 builder 没有 atom 对应的精确原文字符坐标；本轮只保留已验证的 native/canonical block 坐标和 unit→atom 索引区间，没有虚构 atom 原文 span。部分 atom 与原生段落有来源重叠，不能自动算作整段参考证据被覆盖。

所检查的产物元数据尚未找到并验证这两篇文档的真实 Refiner 预测。原始 exporter 对 chunk/leaf 硬编码 `source="refiner_epoch8"`，即使输入是人工恒等边界也如此；私有输出已在外层显式标记 fixture、`model_inference=false`、`teacher_ckpt=null`。这个字符串不能作为模型来源证据。

下一步继续零 API：先检查现有无损边界解码组件能否为这条 builder/exporter 路径提供可验证的 atom 原文对应关系，用固定小样本及合成反例明确哪些转换可精确恢复、哪些必须保留双文本视图。不会因本次接口通过就复用旧标签、扩大训练或启动付费评价。接入和修复本身不算论文创新；研究主张仍需真实粒度下的公平对照和质量结果。
