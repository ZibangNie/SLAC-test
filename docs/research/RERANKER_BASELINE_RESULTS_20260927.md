# Qasper 固定候选池 cross-encoder 基线：2026-09-27

固定 **77 个开发问题、24 个 family、1,214 个 query–passage 对** 的完整本地 BGE reranker 运行、科学回放、补充 metadata 审计和预先封印的配对分析均已通过。预先固定的主比较 **k=3** 中，Evidence F1 的 question-weighted 均值从 dense 的 **0.202453** 升至 **0.238374**，差值 **0.035920**；family-balanced 差值为 **0.027127**，其 95% 描述性区间 **[-0.001029, 0.054872] 跨过 0**。平均实际 evidence 长度同时从 **420.844** 增至 **510.247 tokens**。

结果支持继续保留这一普通 cross-encoder 作为开发基线，但无法把证据得分变化与包长度变化完全分离，也没有验证 JEV、关系共享、完整 SLAC 或答案质量的收益。本报告只比较同一批 77 题上的 reranker 与 dense，不把旧 15 题 pilot 视为同一主分母。

## 数据、评分与固定执行规则

这 77 题是排除全部 8 个旧 pilot family 后的完整剩余开发问题，来自 24 篇 validation 论文，已暴露于既有检索实验。没有按答案可用性、金标可达性、问题措辞或观察到的成绩过滤；图表参考、空参考与不可回答问题均保留。研究阶段没有读取官方 test QA。

- 同一给定论文、同一冻结候选池与 query；全部 1,214 个候选对都送入模型。Dense 直接使用原 `ranked_ids`，reranker 只改变同池排序。参考证据仅用于后续评分，没有进入模型输入、排序或阈值选择。
- 模型为 `BAAI/bge-reranker-v2-m3`，固定 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。运行采用 CUDA 0、FP16、SDPA、microbatch 4；单个分类 logit 转为 FP32 数值保存，按降序排序，不归一化为概率。平分依次遵循原 dense rank 与原生来源顺序。
- Query 与完整 canonical unit text 组成有序 tokenizer pair，不增加指令、不截断。两种排序都使用相同 renderer、完整单元和精确原生文本去重规则，按顺序保留能放入包的单元；整个证据包受 **1,024 个真实 BGE tokens** 上限和相同 k 上限约束。没有通过局部字符串 overlap 把碎片算成整段证据。
- 主比较固定 k=3，k=1/2 是敏感性分析；没有因 k=2 的 F1 均值更高而替换主比较。六个方法、四项指标和全部结果方向都保留。

Evidence F1 沿用现有 Qasper 官方字符串匹配语义；Evidence recall 是同参考上的诊断指标，不是新增官方任务指标。Text-only F1 另报去除图表类证据后的结果，本批六组均值恰好与完整 Evidence F1 相同；这不意味着已解决图表证据问题。保留既有多参考、重复项与空参考评分规则，不能把证据得分解释为答案正确率。

## 全 77 题结果

F1 与 recall 均在 [0,1] 上报告。每个方法都有完整 77 个包；每包恰好选中 k 个单元，空包数为 0，全部包在 token 上限内。主比较的最大实际包长度分别为 dense 976、reranker 1,010 tokens。

### Question-weighted 均值

| k | 方法 | Evidence F1 | Evidence recall | Text-only F1 | 实际 tokens |
|---|---|---:|---:|---:|---:|
| 3（主比较） | Dense | 0.202453 | 0.339260 | 0.202453 | 420.844 |
| 3（主比较） | BGE reranker | 0.238374 | 0.429004 | 0.238374 | 510.247 |
| 1（敏感性） | Dense | 0.132323 | 0.115043 | 0.132323 | 118.909 |
| 1（敏感性） | BGE reranker | 0.205195 | 0.197403 | 0.205195 | 178.636 |
| 2（敏感性） | Dense | 0.185684 | 0.237080 | 0.185684 | 275.221 |
| 2（敏感性） | BGE reranker | 0.249165 | 0.344589 | 0.249165 | 363.740 |

### Family-balanced 均值

每个 family 先对其全部问题取均值，再对 24 个 family 等权平均。

| k | 方法 | Evidence F1 | Evidence recall | Text-only F1 | 实际 tokens |
|---|---|---:|---:|---:|---:|
| 3（主比较） | Dense | 0.209361 | 0.352910 | 0.209361 | 430.114 |
| 3（主比较） | BGE reranker | 0.236488 | 0.421528 | 0.236488 | 507.618 |
| 1（敏感性） | Dense | 0.111397 | 0.094792 | 0.111397 | 125.955 |
| 1（敏感性） | BGE reranker | 0.204630 | 0.193056 | 0.204630 | 180.194 |
| 2（敏感性） | Dense | 0.188271 | 0.241948 | 0.188271 | 287.393 |
| 2（敏感性） | BGE reranker | 0.255704 | 0.351736 | 0.255704 | 362.468 |

主 k=3 的 F1 和 recall 均为 **13 题增加、55 题相同、9 题下降**。实际 tokens 是 **54 题增加、5 题相同、18 题减少**。相同 k 和相同 token 上限并不等于实际长度相同；当前结果不能单独识别重排语义与更长证据包各自的贡献。

## 完整 family 配对分析

方向统一为 **reranker − dense**。分析规范在本次模型推理前封印，但这批开发数据已有其他实验结果暴露，因此并非独立确认。对全部三个 k、四项指标共同采用 PCG64 seed `20260927`、10,000 次整 family 重采样：每次抽 24 个 family 并保留每个抽中 family 的所有问题；question-weighted 用该次抽到的问题数作分母，family-balanced 用抽中 family 的问题均值等权平均。区间为双侧 percentile 95%，采用线性分位插值。

| k | 指标 | Question-weighted Δ [95% 区间] | Family-balanced Δ [95% 区间] | 逐题增 / 同 / 减 |
|---|---|---:|---:|---:|
| 3 | Evidence F1 | 0.035920 [0.002893, 0.064668] | 0.027127 [-0.001029, 0.054872] | 13 / 55 / 9 |
| 3 | Evidence recall | 0.089744 [0.027631, 0.145834] | 0.068618 [0.010417, 0.126391] | 13 / 55 / 9 |
| 3 | Text-only Evidence F1 | 0.035920 [0.002893, 0.064668] | 0.027127 [-0.001029, 0.054872] | 13 / 55 / 9 |
| 3 | 实际 evidence tokens | 89.403 [46.916, 139.247] | 77.504 [39.729, 118.922] | 54 / 5 / 18 |
| 1 | Evidence F1 | 0.072872 [0.002809, 0.148325] | 0.093233 [0.022491, 0.169712] | 14 / 53 / 10 |
| 1 | Evidence recall | 0.082359 [0.012910, 0.155357] | 0.098264 [0.031241, 0.171884] | 14 / 53 / 10 |
| 1 | Text-only Evidence F1 | 0.072872 [0.002809, 0.148325] | 0.093233 [0.022491, 0.169712] | 14 / 53 / 10 |
| 1 | 实际 evidence tokens | 59.727 [38.027, 81.373] | 54.240 [31.019, 80.043] | 39 / 18 / 20 |
| 2 | Evidence F1 | 0.063481 [0.016555, 0.109766] | 0.067433 [0.009215, 0.121464] | 14 / 54 / 9 |
| 2 | Evidence recall | 0.107509 [0.035808, 0.178042] | 0.109788 [0.021296, 0.189603] | 14 / 54 / 9 |
| 2 | Text-only Evidence F1 | 0.063481 [0.016555, 0.109766] | 0.067433 [0.009215, 0.121464] | 14 / 54 / 9 |
| 2 | 实际 evidence tokens | 88.519 [51.861, 130.618] | 75.075 [42.078, 110.689] | 52 / 10 / 15 |

“增 / 同 / 减”描述指标数值变化；tokens 增加代表证据更长，不代表质量更高。没有只选正向结果或剔除负值问题，没有计算 p-value 或进行多重比较校正，不将任何区间表述为独立显著性或未来性能保证。尤其主 k=3 的 family-balanced F1 区间跨过 0，不能只引用其 question-weighted 区间。重采样基于单次成功的模型运行，不覆盖设备、精度或重复推理的变动。

## 完整输入、执行修正与成本

完整模型输入共 **156,384 pair tokens**，含 query 与 pair special tokens；长度最小 11，nearest-rank p50/p95/p99 为 110/312/521，最大 804。超过 512 tokens 的对有 14 个，超过 1,024 的对为 0，截断数为 0。实际 padded 输入共 **157,434 tokens**，最大批 3,216 tokens。模型 pair tokens 与最终证据包的 BGE tokens 来自各自 tokenizer，不能混用。

首次执行在模型加载前因 CUDA allocator 尚未初始化而停在 `reset_peak_memory_stats(0)`，耗时 **13.187 秒**，没有产生模型成绩。失败现场保留后，另行封印并明确授权的 launcher 先执行空闲 GPU/FIFA 准入检查、显式 `torch.cuda.init()`，再调用未改动的旧 runner；旧 runner 自己的第二次准入检查仍然执行。纠正后的成功尝试只调用 runner 一次，没有自动重试或 CPU fallback。[初始化诊断与官方来源](RERANKER_INITIALIZATION_RECOVERY_20260927.md)

| 保存的时间记录 | 秒 | 范围 |
|---|---:|---|
| 原失败尝试 | 13.187 | 模型加载前失败，独立保留 |
| 成功 launcher 函数内总计 | 47.375 | 包含初始化、外层准入、验证与原 runner |
| 其中原 runner 调用区间 | 26.641 | 被 launcher 包围测得，包含原 runner 写盘返回 |
| 其中原 runner 自报 | 26.360 | 原 runner 汇总字段，非纯 forward 延迟 |
| 其余 launcher 开销 | 20.734 | 总计减调用区间，包含来源 hash 验证等 |
| 其中显式 CUDA 初始化 | 0.125 | 包含在 launcher 开销中 |
| 其中外层准入检查 | 2.906 | 包含在 launcher 开销中 |
| 后续科学与 metadata 审计 | 21.516 | CPU 回放与 metadata 校验，另计 |
| 后续配对分析 | 35.109 | 包含再次回放、dense 重算与 bootstrap，另计 |

这些是嵌套或分开的历史函数内计时，不含解释器启动和模块导入，不能把表内各行直接求和，也不能把总执行时间叫作纯模型推理延迟。原 runner 的 600 秒限制不含新增 launcher 开销；launcher 总耗时另记。GPU allocator 的 peak allocated/reserved 分别为 1,225,415,168 / 1,268,776,960 bytes，不等于整卡显存占用。

实际环境为 RTX 5070 Laptop GPU、PyTorch `2.9.1+cu130`、Transformers `4.57.3`。外层和原 runner 各保存 3 次准入采样，均为 7,879 MiB free、0% utilization；阈值为至少 4,096 MiB free、最多 20% utilization。原代码执行 FIFA 进程否决检查，但没有保存历史进程快照，审计不声称独立证明过去的进程不存在，也没有通过重新测量伪证历史资源记录。

本轮模型、审计、分析新增 API 调用均为 **0**，记录的 paid API charge 为 **$0**；不把本地计算、电力或下载成本宣称为零。没有训练或答案生成。公开聚合 JSON 的 `model_inference_performed=false` 属于分析阶段，不能解读成此前 reranker 阶段没有执行 GPU 模型。

## 校验、产物与研究边界

完整科学审计重建全部 1,214 对的身份与输入编码、排序和 231 条 reranker 评分记录；metadata 审计核验固定模型、计数、预算、费用声明、运行环境与资源测量字段。分析在相同 77 题上另重算 dense 的 231 条记录。发布前独立核验了 **59 个来源绑定**、所有六方法均值、三个 k × 四指标 × 两种加权的 **24 个区间及全部逐题方向计数**，并核对初始化 receipt 与成功 run manifest 指向同一产物；没有再次运行模型。

公共结果为[聚合 JSON](results/qasper_reranker_baseline_20260927.json)，与已校验的原 `public_aggregate.json` 逐字节相同，SHA-256 为 `6558c0c519877a6823cbcf74637c5f0e8f220ba20ca6da3aa5820b531b700361`。逐题记录、文本、原始标识、模型输入、receipt 中的本地路径及绑定清单留在 Git 忽略目录。代码入口为[冻结 runner](run_qasper_reranker_baseline.py)、[独立初始化 launcher](run_qasper_reranker_initialized.py)、[补充 metadata 审计](audit_qasper_reranker_metadata.py)和[配对分析](analyze_qasper_reranker_baseline.py)；四个相关测试模块共 99 项通过。

[共享 dense 基线一致性记录](results/qasper_local_dense_parity_20260927.json)还确认本轮 dense_k3 与 native given-document leaf_direct 在全部 77 题的选集、pack hash、tokens 和共同评分字段相同。这证明共同基线可复现，不构成独立质量确认。

当前证据回答的是“在这个已暴露的 given-document 候选池上，固定 cross-encoder 排序如何改变证据选择”。跨文档召回、答案质量、关系结构机制、JEV 增量以及最终论文创新性仍需各自的公平实验；不能因本轮基线提升就宣称完整 SLAC 已有收益。后续机制方案需要保留这个可复现基线，并在明确同候选、同生成器及长度影响的协议下检验增量价值。
