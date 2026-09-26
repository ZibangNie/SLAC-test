# 固定候选、同预算的精确 evidence oracle 结果

给定文档时，direct / leaf owner / dual owner 的问题等权 actual Evidence F1 为 **0.202 / 0.144 / 0.139**，同候选、同预算 oracle 为 **0.801 / 0.817 / 0.833**。两项上界增量（leaf owner−direct、dual owner−leaf owner）的双权重区间均跨 0。这说明当前候选池存在选择空间，尚不能认定 owner 候选扩展带来稳定的额外可达收益。

跨库 dual owner−leaf owner 为 2 增、74 同、1 减；其双权重区间下界恰为 0，不据此宣称显著或稳定正收益。

本轮完整覆盖既有 77 个开发问题、24 个 family、六组冻结候选。它用参考答案寻找同候选、同预算的可达上界；gold 参与选集，不能当成可部署检索表现、Answer F1、独立确认或 JEV/完整 SLAC 的收益。旧 104 题 subset-oracle 不进入比较。

本报告仅在完整运行、root 完整 CPU 重放审计与发布来源校验通过后生成。源码、测试与协议先于新 oracle 分数冻结；逐题参考、选集与原始身份只留在 ignored artifacts。

## 六组完整结果

下表每格依次为问题等权 / family 等权。Candidate 是全部冻结候选的 recall；actual 是原部署式选集；oracle 是 gold-guided 最优合法子集。三者分别解释。

| Scope | 方法 | Candidate recall | Actual F1 | Actual recall | Oracle F1 | Oracle recall | Oracle−actual F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| given_document | Direct | 0.679046 / 0.678018 | 0.202453 / 0.209361 | 0.339260 / 0.352910 | 0.801105 / 0.817647 | 0.776263 / 0.794039 | 0.598651 / 0.608285 |
| given_document | Leaf owner | 0.716378 / 0.712211 | 0.143845 / 0.153821 | 0.206926 / 0.221875 | 0.816724 / 0.836071 | 0.792352 / 0.810127 | 0.672878 / 0.682250 |
| given_document | Dual owner | 0.730880 / 0.729051 | 0.139207 / 0.154317 | 0.210173 / 0.230903 | 0.833298 / 0.853498 | 0.806854 / 0.826100 | 0.694091 / 0.699181 |
| corpus_32 | Direct | 0.277561 / 0.253588 | 0.082127 / 0.074644 | 0.123846 / 0.110648 | 0.419317 / 0.424591 | 0.404185 / 0.409491 | 0.337191 / 0.349947 |
| corpus_32 | Leaf owner | 0.290404 / 0.267593 | 0.104359 / 0.107095 | 0.144805 / 0.142014 | 0.415147 / 0.422910 | 0.404040 / 0.411343 | 0.310788 / 0.315815 |
| corpus_32 | Dual owner | 0.303391 / 0.275926 | 0.093143 / 0.089892 | 0.128571 / 0.114236 | 0.429989 / 0.432235 | 0.420274 / 0.421412 | 0.336846 / 0.342343 |

| Scope | 方法 | Actual tokens QW / FB | Oracle tokens QW / FB | Oracle 单元 QW / FB | Oracle empty / 77 | 有 empty reference / 77 |
|---|---|---:|---:|---:|---:|---:|
| given_document | Direct | 420.844 / 430.114 | 152.779 / 151.006 | 0.961 / 0.958 | 22 | 11 |
| given_document | Leaf owner | 331.078 / 335.815 | 157.364 / 158.855 | 1.026 / 1.017 | 21 | 11 |
| given_document | Dual owner | 343.039 / 349.410 | 159.403 / 161.351 | 1.013 / 1.017 | 19 | 11 |
| corpus_32 | Direct | 238.727 / 234.391 | 64.247 / 56.610 | 0.455 / 0.440 | 52 | 11 |
| corpus_32 | Leaf owner | 294.273 / 290.872 | 60.636 / 54.667 | 0.481 / 0.470 | 53 | 11 |
| corpus_32 | Dual owner | 324.688 / 330.967 | 63.974 / 56.689 | 0.468 / 0.465 | 52 | 11 |

所有 462 行 actual 选集均先验证属于原候选、同源同文不重复、完整渲染一致、最多三项且实际 BGE tokens ≤1,024，再作为 oracle 可行域见证；oracle F1 每行均不低于 actual。相同上限不代表相同实际长度，上表不做长度匹配声明。

任一 empty reference 可让 empty evidence 的官方 F1 和 recall 同时为 1；当所有预算合法子集的 F1 均为 0 时，最少 tokens 的 tie 规则也会选择 empty。因此 oracle empty 数不等于不可回答问题数，也不代表可部署弃答识别能力。重复 reference 项保留原 list-length 分母，FLOAT evidence 不删除；candidate recall 不能在这些语义下直接充当 oracle recall 的单调上界。

## 四项预固定配对

仅比较 oracle source-qualified Evidence F1。24 个 family 整簇 PCG64 seed 20260927、10,000 次共享抽样、双侧线性 percentile 95% CI；全部方向与双权重保留，不做多重比较校正。

| Scope | Plus − minus | 问题等权 Δ [95% CI] | Family 等权 Δ [95% CI] | 增 / 同 / 减 |
|---|---|---:|---:|---:|
| given_document | Leaf owner − Direct | +0.015619 [-0.018148, +0.054520] | +0.018424 [-0.020272, +0.064034] | 6 / 68 / 3 |
| given_document | Dual owner − Leaf owner | +0.016574 [-0.023868, +0.060190] | +0.017427 [-0.011905, +0.053241] | 5 / 69 / 3 |
| corpus_32 | Leaf owner − Direct | -0.004171 [-0.049744, +0.040631] | -0.001682 [-0.041708, +0.034558] | 5 / 69 / 3 |
| corpus_32 | Dual owner − Leaf owner | +0.014842 [+0.000000, +0.041667] | +0.009325 [+0.000000, +0.026984] | 2 / 74 / 1 |

更大 oracle 上界只意味着该冻结候选内存在更好的 gold-guided 选择；不能据此推断实际 selector 或答案质量必然提升。Oracle−actual gap 是每组选择损失的描述，不是实际可达收益预测。

## 完整搜索与资源

| 组合核对项 | 数量 |
|---|---:|
| 全部枚举，包含各行 empty | 311,028 |
| 同源同文重复而不合法 | 0 |
| 超过实际 1,024-token 预算 | 1,150 |
| 合法可评分子集 | 309,878 |
| 实际 token cache 条目 | 150,461 |
| 全局不同 index 子集上限 | 150,461 |

三个合法性类别之和等于 311,028；每行至多 697 个子集。穷举所有大小 0–3 的候选组合，保留同文不同位置作为互斥备选，不只搜索 reference 匹配项。选优顺序为精确 Fraction F1、独立 max-reference recall、最少实际 tokens、global-index 字典序。只共享整包 token 缓存，不跨问题共享 gold 分数。

run 函数记录总耗时 **64.642 秒**，包含来源校验、原 native CPU 审计、新 oracle 枚举与聚合；不是纯枚举内核耗时，也不含解释器启动。期限为 1,200 秒，按检查点失败退出，并非外部强制杀进程保证；本结果通过完成与时间门槛。独立重放审计的额外时间未合并进该数。

枚举串行、PyTorch CPU threads=1；NumPy/BLAS 线程数未单独强制。新增 API 调用与费用均为 0，无 GPU 或模型推理。此诊断不改变夜间费用账本，也没有解决原来的未知收费请求。

## 解释边界

这是已经暴露的开发集上的事后机制诊断。Qasper 原任务给定论文；corpus_32 query-only 是额外压力测试，不能仅据下降推断标准 Qasper 或完整 SLAC 架构失败。跨文档 header 歧义仍在，任何 oracle pack 均不得送入答案生成、选新部署方法或回写 retriever。

完整浮点精度、四项比较与来源 hash 保留在 [公开聚合 JSON](results/qasper_candidate_oracle_20260927.json)。
