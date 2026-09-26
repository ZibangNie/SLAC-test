# Native chunk / dual-index 离线桥接协议

日期：2026-09-27。本协议在新增chunk编码和该对照评分之前固定。它不改变正在运行的77题JEV/Qwen证据及答案实验，也不追加API调用。

## 要回答的问题

检验现有SLAC的leaf→owner聚合及leaf/chunk RRF能否在可核验的原文、固定表示与证据预算下运行，并量化添加chunk检索这一项的影响。此轮采用规则chunk；不把它称为JEV分块、Boundary Refiner训练、完整线上SLAC或独立泛化验证。

同分区主对照为leaf-only→owner聚合与leaf+chunk→owner聚合。两者共用完全相同的chunk分区、leaf表示、原始query、RRF实现、返回单元及打包规则。直接leaf dense作为背景基线保留，不能将与它的差异全部归因于双索引。

## 数据、表示与分区

- 固定此前77题/24family开发分母；语料为原32篇validation论文，旧pilot文档可作干扰文档。给定文档与32文档全库检索分别完整报告，不混合平均。
- 主leaf轴严格复用原1,850个native unit及其完整文本，按原embedding index逐条核对(doc_id, unit_id)和内容。原始Qasper字段、构造native文档视图的字符坐标、canonical坐标分别保存，不声称它们是PDF坐标。
- 规则chunk在同section内合并相邻完整native unit；最大512个BGE tokens（含special tokens）。单个超限unit完整保留、标记oversize。不得截断、删掉长段或按gold选择边界。分区只构建一次，不按结果改变上限。
- chunk编码复用固定BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`，FP16 backbone、CLS、FP32 L2归一化；query和leaf用已验证缓存。chunk输入为适配器明确保存的完整文本，不隐式添加query、gold、anchor或标题。
- 所有chunk先审计完整token长度；最大8192（含special tokens）。若超限或OOM，停止此阶段、保存失败，不截断或反复放宽条件。microbatch至多4，编码时限600秒。记录设备、依赖、实际耗时和峰值显存。

## 固定检索与打包规则

1. 原始query单一变体，不调用QueryPlanner、LLM adapter、reranker、anchor或tree expansion。
2. leaf分支取得top8完整unit；dual分支额外取得top8规则chunk。FP32内积穷举作为精确参考，按已冻结index顺序解决同分。若使用FAISS作实现核对，单独记录其tie差异，不替换参考顺序。
3. 调用现有 `aggregate_hits_to_chunk_candidates` 和 `fuse_candidate_scores_rrf`，固定RRF k=60、generic intent、保留至多8个owner。leaf-only与dual使用相同代码路径；仅dual传入chunk hits。
4. 按fused owner顺序投影回所属完整native units；owner内部按同一query的leaf内积降序排列，以原全局index解同分。按(doc_id, unit_id)去重，前16个组成候选列表。不得利用金标调整投影。
5. 最终包沿此固定候选顺序贪心选择至多3个完整unit、至多1024个实际BGE evidence tokens；超长单元跳过，不截断。给定文档及全库采用同一明确renderer，若跨文档引用需要加来源标记，必须另报与原renderer的预算差异，不能静默改变某一组。
6. 现有RRF为每个leaf hit给owner一票，同owner多个leaf会累加。记录命中leaf数和owner长度；同分区对照保持此因素一致，不把长chunk多票解释为语义关系证据。

## 全量报告与限制

报告全部77题的完整native证据召回、来源限定Evidence F1/recall、官方纯字符串兼容分数、candidate recall、实际tokens、空包、候选数，以及配对选集改变、获益/受损/相同题数。跨文档同名heading或重复文本在错误来源不算正确命中；去重政策和碰撞数必须披露。bootstrap沿既定family分组，仅作开发描述，不选最好设置。

费用为零API；新增本地chunk编码/索引/检索耗时与缓存复用分别报告，不伪装为冷启动端到端延迟。dual多使用一个top8检索通道，其计算与候选覆盖增加是对照的一部分。这个桥接实验尚不能归因于JEV关系或Refiner。

新候选不在原1,214 support标注集合时，不沿用缺失标签或把未知当no。需要JEV的跨库比较必须另行冻结完整计划和预算；本轮不执行。

适配器与runner来源hash、输入文件hash、局部映射与逐题输出保存在ignored artifacts；仅发布聚合与不含真实ID的代码。正式执行前测试原文roundtrip、缓存映射、跨文档邻域/身份、RRF来源控制及实际token预算。
