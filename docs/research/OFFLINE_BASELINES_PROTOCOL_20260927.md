# Qasper 离线基线协议：2026-09-27

本说明对应已冻结的 `qasper-reranker-02`、`qasper-reranker-analysis-plan-01` 和 `qasper-native-dual-index-plan-01`。撰写时已分别调用其 `load_plan` / `load_specification` 验证配置、封印及绑定来源。三者状态分别为 `prepared_no_model_inference`、`analysis_specification_frozen_before_inference`、`prepared_not_executed`。**这两项新增实验尚未执行 GPU 推理，尚无本轮 reranker 或 dual-index 质量结果。** 长度审计、缓存一致性检查、合成测试和 CPU 准备成功不代表检索质量提升。

此前 A 的 CPU 跨文档桥接已经有结果，见 [A 结果及限制](CORPUS_BRIDGE_RESULTS_20260927.md)。本轮协议在新增模型输出产生前固定，但使用的是已经暴露的开发数据，不能称为未见数据上的预注册确认实验。

## 分母、来源与冻结身份

两项都保留完整 **77 题、24 个 family**，不因空参考、不可回答、图表证据、候选覆盖不足或失败倾向筛题。问题来自此前排除全部 8 个 pilot family 后的 validation 开发集合；没有读取 official test QA。参考标注只用于最终评分，不参与 chunk 分区、排序、阈值或问题子集选择。

以下目录都位于本地 Git 忽略的 `artifacts/research-foundation/` 下。原始文本、实际文档/问题标识、逐题选择、向量和来源路径留在本地；公开内容仅保留代码、协议和不含这些信息的聚合。

| 已冻结配置 | SHA256 |
|---|---|
| `qasper-reranker-02/experiment_config.json` | `84dab462e62ab8de4cb48673efbf6a8cc836b99b4c33ae262361971197fa82ad` |
| `qasper-reranker-analysis-plan-01/analysis_config.json` | `622cb72d4d99fd3e74509df96b9661d016e1203af986bd1c544b7bd08c381709` |
| `qasper-native-dual-index-plan-01/plan.json` | `7c1859d5bb891d9651a3eb1048325b1c36ae2db3db0fbeb8c4866c272dc33f1d` |

后续执行使用上述已有计划，不重新覆盖准备目录。源码、模型、tokenizer、数据或参数发生变化时，原封印应拒绝执行；需要另建计划并记录变更，不能将旧计划的身份附到新运行上。

## Reranker：固定候选池内的排序基线

代码为 [runner](run_qasper_reranker_baseline.py) 和 [配对分析器](analyze_qasper_reranker_baseline.py)。检验的问题是：在相同的给定文档候选池与证据预算下，通用 cross-encoder 能否改善 dense 排序。它没有新增候选检索，也没有复用 JEV 的标签进行 eligibility 筛选。

- 使用 `BAAI/bge-reranker-v2-m3`，revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，567,755,777 个参数。模型文件按固定文件大小及官方身份核验，并绑定本地 SHA256。
- 所有 **1,214 个 query–unit pair** 来自既有 77 题 prepared 候选池；其生成规则为 dense top8 加同文档相邻单元、最多 16 个候选。dense 与 reranker 使用相同候选、相同完整文本和原始问题字符串，不添加 query/passage instruction。
- 输入是 reranker tokenizer 的有序 `(query, canonical unit text)` 对。FP16 backbone、SDPA、CUDA:0、microbatch 4；输出唯一 classification logit，转换为 FP32 后由高到低排序。没有 sigmoid、分数归一化或学习阈值；同分按原 dense 名次、再按原生单元顺序处理。logit 不解释为校准后的 support 概率。
- 输入上限固定为 **1,024 pair tokens**，包括问题、段落和 pair special tokens；不截断。准备审计为 156,384 tokens，p50=110、p95=312、p99=521、最大 804；14 对超过 512、0 对超过 1,024，全部保留。
- 最终按各自排名贪心选完整原生单元，最多 k 个、整个渲染包最多 **1,024 个 BGE evidence tokens**。精确原生文本去重，保留首先被选中且可放入预算的来源位置；过长完整单元跳过，不裁切。pair 输入长度与 evidence 包长度是两种统计，不能互换。

配对分析固定报告六个方法：`dense_k1`、`bge_reranker_v2_m3_k1`、`dense_k2`、`bge_reranker_v2_m3_k2`、`dense_k3`、`bge_reranker_v2_m3_k3`。比较方向统一为同 k 下 **reranker − dense**；k=3 为主比较，k=1/2 为敏感性分析。一次完整 reranker 运行生成 231 条方法记录，分析器另重放 231 条 dense 记录；不按分数选最好 k，不只报告正向结果。

全部三项配对均报告官方字符串 Evidence F1、原有参考证据 recall 诊断、排除图表标记的 text-only Evidence F1，以及实际 evidence tokens。每项同时报告 question-weighted 与 family-balanced 均值和差值、整 family bootstrap 95% 区间、逐题胜/平/负或 token 增/平/减。recall 是已有诊断指标，不能泛称所有指标都是官方指标。

## Native dual-index：固定原文分区的检索桥接

代码为 [原文适配器](build_qasper_native_chunks.py) 和 [prepare/run/audit runner](run_qasper_native_dual_index.py)，分区与检索设计见 [native chunk 协议](NATIVE_CHUNK_BRIDGE_PROTOCOL_20260927.md)。同分区核心对照是 `dual_owner − leaf_owner`；直接 leaf dense 作为背景方法完整保留。

原文适配器使用既有 **32 篇 validation 论文、1,850 个 native units**。主 leaf 的文档身份、单元身份、顺序和完整编码文本逐项匹配旧缓存，复用 1,850×1,024 leaf 向量及已有 104×1,024 query 向量中的本轮 77 行。额外 8 篇 pilot 论文仅作为检索库中的干扰文档，不扩充问题分母。

规则分区在同 section 内按相邻完整单元贪心合并，固定上限 **512 个实际 BGE tokens**，标题与摘要独立成组；超限整段单独保留。适配器产出 671 个 chunk，10 个超限 singleton；保存 1,892 个原文字段，其中 42 个空白字段仅留 sidecar，没有增加主 leaf 分母。原生字段表示与缓存编码表示分别保存坐标/hash；这批数据的 1,850 个非空单元恰好相同，但接口不假设两种表示永远相等。构造的 native 文档视图坐标不是 PDF 坐标。

新增 chunk 仅编码适配器保存的完整 `ChunkRecord.text`，不隐式加入标题、路径或 anchor。模型固定为 BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`，FP16 backbone、CLS、FP32 L2 归一化、microbatch 4。准备阶段冻结全部 token IDs；671 个 chunk 共 187,607 tokens，p50=276、p95=505、最大 787，全部低于固定 8,192 上限。没有裁切或删去长段。

每个范围分别报告以下三种方法，共 **六组、462 条逐题方法记录**：

| 方法 | 固定规则 |
|---|---|
| `leaf_direct` | dense top8 leaf 加同文档直接相邻单元，候选最多 16；按 dense 顺序打包。 |
| `leaf_owner` | top8 leaf 映射到 owner，调用生产 aggregator/RRF，保留最多 8 个 owner 后投影回原生 leaf。 |
| `dual_owner` | 与上一组相同，额外加入 top8 chunk hits；其他聚合、投影和打包规则不变。 |

两个范围为 `given_document` 与 `corpus_32`，分别完整报告，不混合平均。后者仅用原问题字符串检索 32 篇论文，是移除来源论文上下文的 query-only 压力诊断，不是标准 Qasper 全库成绩；不据此判定完整 SLAC 成败。

检索参考实现是 CPU FP32 穷举内积，同分按冻结 index 顺序。生产 `aggregate_hits_to_chunk_candidates` 和 `fuse_candidate_scores_rrf` 使用 generic intent、RRF k=60；owner 排序采用该生产实现的固定 tie policy。每个 leaf hit 各投一票，同 owner 的多个 leaf 会累加，逐题保留命中 leaf 数、owner 长度及来源通道；不能将大 owner 的多票效应解释为关系建模收益。没有调用 FAISS、planner、anchor、tree expansion 或 reranker。

按 fused owner 顺序投影完整 leaf；owner 内部按同一 query 的 leaf 内积分数排序，再以全局缓存行解同分。投影按 `(doc_id, unit_id)` 去重，前 16 个形成候选。六组都按其候选顺序贪心打包最多 3 个完整单元、最多 1,024 个实际 BGE evidence tokens；打包按 `(doc_id, native_text)` 去重，保留不同来源的同文字符串。渲染沿原 `[unit_id]` 格式，最终按全局来源顺序排列，两个范围均不加新来源前缀。

每个范围固定报告 `leaf_owner − leaf_direct`、`dual_owner − leaf_direct`、`dual_owner − leaf_owner` 三项配对，保留全部正、负、零差值。指标覆盖完整 native 证据可用 recall、选集的来源限定 Evidence F1/recall、官方字符串兼容分数及 text-only F1、候选 recall、实际 tokens、单元/文档数、空包、重复 header 和跨文档同文字符串诊断；另报选集改变题数。来源限定指标对完整 `(文档, 原生字符串)` 精确匹配，错误文档的同文字符串仍计入预测分母。

原渲染 ID 可能跨文档重复，局部映射虽可保证离线评分来源明确，这些 pack 仍不能直接作为无歧义的答案生成输入。新检索候选不保证在原 1,214 个 support pair 中；本轮不借用缺失的 JEV 标签，也不把未知候选当作 no。

## 与 A 的区别及解释边界

| 项目 | 已完成的 A | 固定池 reranker | Native dual-index |
|---|---|---|---|
| 要改变的因素 | dense 检索是否限制来源文档 | 同候选池的排序方式 | 同分区 owner 聚合是否加入 chunk 检索通道 |
| 新增模型执行 | 无，复用旧缓存 | 对 1,214 个 query–unit pair 编码 | 对 671 个 chunk 编码，leaf/query 复用缓存 |
| 候选 | top8 加邻域、最多 16 | 原 prepared 候选不增加 | owner 投影最多 16；另保留 leaf-direct 背景 |
| 打包去重 | 包内精确原生文本 | 给定文档内精确原生文本 | `(来源文档, 原生文本)` |
| 主要比较范围 | 指定文档与 query-only 跨库 | 给定文档、同 k 配对 | 各范围内同分区配对；两个范围分开报告 |

因此 native `leaf_direct` 不能冒充 A 数值的同配置重跑；它使用了本轮统一的来源限定去重。与 `leaf_direct` 的差异也不能全部归因于“双索引”，因为 owner 聚合与候选投影同时介入。更接近单因素解释的是同分区 `dual_owner − leaf_owner`，但额外 top8 通道带来的计算量和候选覆盖增加仍是这个处理的一部分。

两项均不训练 Boundary Refiner，不执行 JEV 关系推理、答案生成或完整线上 SLAC。它们用于判断候选排序与检索结构是否值得进一步研究，尚不能支持上述模块的质量、创新性或发表结论。

## 预定统计、成本与运行门禁

两个分析均固定 PCG64 seed `20260927`、10,000 次整 family bootstrap：每次抽取 24 个 family 并保留各自全部问题。question-weighted 以抽中问题总数加权；family-balanced 等权平均各 family 内的问题均值。使用双侧 percentile 95% 区间、线性分位插值，并在各自分析的所有比较和指标间共享重采样。逐题 tie 容差为 `1e-12`。没有 p-value、多个比较校正、最好设置选择或问题子集筛选；区间是开发描述，不构成独立确认，也不覆盖重新运行模型所带来的设备/精度变异。

所有新增 API 调用及付费 API 费用为 0。模型下载、磁盘读取、CPU/GPU 占用与电力是真实资源成本，不能因无 API 账单而称为零计算成本，也不能把 reranker 的 pair tokens 与 chunk 编码 tokens 当作同一种工作量直接比较。相同 evidence 上限也不保证相同实际长度，必须保留每种方法的实际 tokens 和配对长度差。

| 项目 | Reranker 冻结门禁与记录 | Native dual-index 冻结门禁与记录 |
|---|---|---|
| 用户工作负载 | `FIFA18.exe` 存在即推迟，不终止或修改用户进程 | 检出 FIFA 进程或 GPU compute 进程即推迟，不终止或修改用户进程 |
| 空闲采样 | 连续 3 次，间隔 1 秒；利用率 ≤20%，free ≥4,096 MiB | 连续 3 次，间隔 5 秒；利用率 ≤10%，free ≥4,096 MiB，设备容量满足 7 GiB 上限 |
| 时间边界 | run 协作检查 600 秒 | 编码与后续检索/评估各协作检查 600 秒 |
| 显存 | 报告 peak allocated/reserved | allocated ≤6 GiB、reserved ≤7 GiB，并报告实测峰值 |
| 输入和计算量 | 真实未 padding pair tokens、实际 padded tokens、最大 batch tokens、microbatch | chunk tokens、编码耗时、缓存复用行数、共享 leaf/chunk 打分耗时及各组后处理时间 |
| 失败策略 | 停止并写失败记录，不自动重试、不 CPU 回退、不截断 | 停止并写失败记录，不自动改 batch/预算、不 CPU 回退、不截断 |

内部时间限制在阶段和批次之间检查，不能打断单次阻塞的模型载入/forward。本地单次队列另外为 reranker run 设置 720 秒、native run 设置 1,380 秒硬 watchdog；超时仅终止自己的子进程，保存失败状态且不自动重试。运行中每 5 秒检查游戏进程是否重新出现，出现则停止自身推理，不操作游戏。

只有主任务确认 GPU 已释放后才执行下方 run 命令，不能单凭瞬时低利用率抢占仍在运行的游戏。`prepare` / 配对分析规格冻结不加载模型到 GPU；`audit` / `analyze` 只在 CPU 上回放保存的输出。不要同时运行两个 GPU 基线。

Reranker 的 `elapsed_seconds` 包含 run 中的校验、门禁、加载、推理与后处理；其 compute 字段记录 padding 工作量和显存，不能从该字段推导未测量的独立推理延迟。Native 的 `encoding_wall_seconds` 包含模型加载及 chunk 编码；`execution.total_run_wall_seconds` 从前置重载和 idle gate 之后开始，记录到聚合生成前，最终 `summary.elapsed_seconds` 还包含聚合及写出。共享 score arrays 和缓存复用时间不得伪装为每组独立的冷启动端到端延迟。审计只核验历史计时的范围、绑定及汇总一致性，不重新测量过去的耗时。

## 执行、审计与分析命令

以下 PowerShell 命令在仓库根目录运行。目录名是本轮预留的新输出位置；执行前应保持不存在，脚本拒绝覆盖。已有 prepare/analysis plan 不需要再次生成。撰写本说明时仅验证计划，**没有执行以下 GPU run 命令**。

```powershell
$py = 'C:/Environment/python/venvs/slac-research/Scripts/python.exe'

# 仅在主任务确认游戏结束、GPU 已释放后运行。
& $py -X utf8 docs/research/run_qasper_reranker_baseline.py run `
  --plan artifacts/research-foundation/qasper-reranker-02 `
  --output artifacts/research-foundation/qasper-reranker-run-01

# reranker 完整完成后，独立 CPU 审计全部 1,214 scores、排序及 231 条方法记录。
& $py -X utf8 docs/research/audit_qasper_reranker_metadata.py `
  --plan artifacts/research-foundation/qasper-reranker-02 `
  --run artifacts/research-foundation/qasper-reranker-run-01 `
  --output artifacts/research-foundation/qasper-reranker-audit-01

# 分析器也会先独立审计，再重放 dense、执行全部预定配对。
& $py -X utf8 docs/research/analyze_qasper_reranker_baseline.py analyze `
  --analysis-plan artifacts/research-foundation/qasper-reranker-analysis-plan-01 `
  --run artifacts/research-foundation/qasper-reranker-run-01 `
  --output artifacts/research-foundation/qasper-reranker-analysis-01

# 与上一个 GPU 运行分开；仍须主任务确认空闲，并通过三次门禁。
& $py -X utf8 docs/research/run_qasper_native_dual_index.py run `
  --plan artifacts/research-foundation/qasper-native-dual-index-plan-01 `
  --output artifacts/research-foundation/qasper-native-dual-index-run-01 `
  --confirm-idle

# 此 audit 只打印验证报告，不在被封印的 run 目录中新增文件。
& $py -X utf8 docs/research/run_qasper_native_dual_index.py audit `
  --plan artifacts/research-foundation/qasper-native-dual-index-plan-01 `
  --run artifacts/research-foundation/qasper-native-dual-index-run-01
```

Reranker 的独立 `analyze` 将聚合写入新目录的 `public_aggregate.json`；仅在完整 77 题输出审计通过后运行。新增 [元数据审计器](audit_qasper_reranker_metadata.py) 先调用原科学记录审计，再补核模型名称/revision、费用与训练声明、实际 padding 工作量、输入数量、包长度、设备及门禁记录。它不修改原冻结 runner 或分析方案。原门禁样本没有单独保存历史 FIFA 进程快照，所以这里只能从已绑定代码的控制流推断该检查执行成功，不能事后重新证明当时的进程状态；实际 GPU 峰值与计时也只做记录一致性核验。

Native runner 已把固定 bootstrap 与所有配对分析并入 `run`，完成后写入其 `public_aggregate.json`，**没有另一个 native `analyze` 子命令**；独立 `audit` 在 CPU 上重算 462 条记录、选集、trace、全部比较及聚合元数据。它核验保存 chunk 向量的文件 hash、索引、维度和 CLS/FP32 L2 单位范数，不声称重新执行 backbone 证明每个向量的模型来源。改聚合字段再重算本地封印仍不能替代记录重放。

完整结果产生并通过审计之前，不填入估计的质量表、不从部分题目外推整体结果、不把准备态标成 completed，也不因失败换用有利配置后继续沿用本协议身份。
