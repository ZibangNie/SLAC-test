# 固定两题：粒度与完整参考证据的预算可行性

2026-10-05。零 API、零模型、零训练。固定旧 hard-no 抽样中的题序 1、2，身份三元组已核对为既有 Refiner 来源文档 1、2。只使用这两篇已曝光文档的 83 个 native Unit、251 个 model atom 和完整原文映射；参考从已有六行子集中按身份匹配，不能按文件行序取前两行。保留每题所有原 annotation，不换题、不补参考、不扫原 77 题或上游语料。

## 比较对象与判断目标

固定三个完整原文分区：保存的原始 b0 来源块、保存的规则投影 b0 来源块、每个 model atom 对应一个来源块。都保留原文字符，不重新 atomize、投影或运行 Refiner。各分区必须从字符 0 到全文末端连续、不重叠、无缺口，且每块文本等于对应原文切片。

分别对每份原 annotation 的 evidence 字符串精确匹配完整 native Unit，沿已有映射转为原文半开区间。重复、重叠参考区间取并集，但保留原参考条数；不同 annotation 不合并。字符串若未匹配或匹配多个位置，不归一化、模糊匹配或任取一个，整份 annotation 标记为映射未决。不可回答、空参考、非法参考单独保留，不算完整覆盖成功。

本阶段考察的是**若已知参考，当前分区中是否存在满足数量与证据块预算的完整覆盖方案**。允许使用固定文档的全部块，并非旧检索候选池。这是利用参考的表示能力上界检查，不是实际选包、Evidence F1、语义充分性、Answer F1、JEV 准确率或新算法收益。

## 必需集合与状态

对于完整不重叠分区，所有与参考区间并集有正长度交集的块组成唯一必需集合 M。任一完整覆盖方案必须包含 M；M 自身覆盖所有参考字符。仅接触端点不算相交，不把部分段落自动计为整段命中。

最多三块，证据块上限为 1,024 个本地 BGE proxy tokens。按如下固定规则报告：

- `empty_reference` / `unanswerable_reference` / `invalid_reference` / `mapping_unresolved`：分别保留，不能计作成功。
- `infeasible_chunk_count`：M 大于三块；任何覆盖 superset 都违反数量限制。无需为这个集合继续计 tokens。
- `feasible_required_set`：M 不超过三块，且完整渲染后不超过 token 上限，构成可行见证。
- `infeasible_budget_no_superset`：M 超 token 预算，且已用满三块或包含该分区全部块，没有合法真 superset。
- `unresolved_budget_nonmonotone`：M 超 token 预算，但还存在允许加入其他块的余地。不能假定 tokenizer 对追加内容单调，因此本轮不声称所有覆盖包都不可行，也不扩大搜索。

记录全部 annotation × 三分区的状态、必需块数、参考原文字符数、必需块的原文字符数及适用时的整个渲染字符串 hash／token 数。按状态汇总分母，不挑最有利 annotation 代替完整表，不生成显著性或总体效果推断。

## 渲染与计数范围

调用当前 `SLAC.llm.service.renderers.render_evidence_block(..., preserve_source_text=True)`；只赋值真实 doc_id、query_id、query_text、完整 passage_text，以及由原文区间产生的固定格式 `span-{start:05d}-{end:05d}` chunk ID。按原文顺序排列。path、score、rank、role、views、token_est、expansion_depth 均留空，不虚构检索 metadata；三分区使用同一字段政策。

本地固定 BGE tokenizer 包含 special tokens、禁用截断；每次对完整 evidence block 编码，不相加片段计数。BGE 是预算代理，不是 JEV／生成器收费 token。该预算不含 system、query message、memory、provider framing 或输出，不是完整模型请求的上下文准入。本轮仅运行 renderer，不发送或编译生成请求。

源文、区间、实际包和参考保留在私有产物中；公开只保存匿名数值、完整精度、hash、方法与限制。执行前绑定输入、源码、tokenizer 和固定策略；数值 helper 先用合成区间的独立逐字符枚举验证。自然样本运行一次，缺失或失败保留，不调整阈值或扩大样本。
