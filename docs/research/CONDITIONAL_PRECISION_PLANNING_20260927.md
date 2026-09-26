# 条件精度敏感性：family-balanced 规划附录（2026-09-27）

本附录把已有完整 development 结果的 family 间差异，代入一个**有条件的精度公式**，展示独立 family 数及假设标准差变化时的尺度。它是 conditional family-balanced precision sensitivity，**不是 power、实际置信区间、样本足量证明、数据准入决定或共享关系 effect 估计**。77 题、24 个已暴露 family 不能据此被宣布具有足够功效。

四个对照全部保留：`p_yes_only_k3` 分别减 `dense_k3`、`reranker_k3`、`I_jev_k3`、`I_general_k3`。这些对照和标准差来自已完成答案实验，而不是未来确认集。本附录不新增对照、数据、阈值或样本选择，不把已观察的均值差当成 target effect。完整 [JSON](results/qasper_conditional_precision_scenarios_20260927.json)保留 **4 × 3 × 6 = 72 个半宽情景**及 **4 × 3 × 4 = 48 个整数 ceiling 情景**，没有选取乐观对照或倍率。

## 固定公式与数据来源

对每个现有 family，先求其全部问题的已保存 official Answer F1 配对差均值 `d_f`；以全部 24 个 family 的 `d_f` 计算样本标准差 `s`，分母为 `24 − 1`（ddof=1）。不同大小的 family 在这一步等权。以下 `F` 是假设的、相互独立且与现有分布及题目组成可比的未来 family 数，不是题数或文档数。

```text
s² = sum((d_f − mean(d_f))²) / (24 − 1)
z = Φ⁻¹(0.975) = 1.9599639845400536
h(F, m) = z × s × m / sqrt(F)
F_formula(h, m) = ceil((z × s × m / h)²)
```

`h` 是示意半宽；标准差倍率 `m` 固定为 1、1.5、2。`F` 固定为 24、48、72、96、144、192；半宽目标固定为 0.10、0.075、0.05、0.025。所有数值使用原情景的全精度；正文表格仅作显示舍入。原公式采用标准正态分位数，**没有把估计标准差的不确定性纳入区间**，也不是小样本 t 区间。

| 已完成对照 | Family 均值差的样本 SD（ddof=1） | 已观察 FB 均值差（仅来源背景） |
|---|---:|---:|
| p_yes_only_k3 − dense_k3 | 0.164120757 | +0.097241493 |
| p_yes_only_k3 − reranker_k3 | 0.191298068 | +0.086381732 |
| p_yes_only_k3 − I_jev_k3 | 0.151210660 | +0.032200733 |
| p_yes_only_k3 − I_general_k3 | 0.149340046 | +0.026604804 |

右列只说明这些 SD 来自什么已观察对照，**不进入半宽或 ceiling 公式，也不是功效计算中的目标效应**。两种 JEV 原始分数规则在已有结果上逐题相同，本附录按原情景只使用固定的 `p_yes_only_k3` 名称，没有将它们视为两份独立样本。正式质量比较另见[主答案与 reranker 的事后结果](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md)。

## 固定全部 F：标准差倍率 m=1 的示意半宽

| 对照 | F=24 | F=48 | F=72 | F=96 | F=144 | F=192 |
|---|---:|---:|---:|---:|---:|---:|
| p_yes_only_k3 − dense_k3 | 0.065661 | 0.046429 | 0.037909 | 0.032830 | 0.026806 | 0.023215 |
| p_yes_only_k3 − reranker_k3 | 0.076534 | 0.054118 | 0.044187 | 0.038267 | 0.031245 | 0.027059 |
| p_yes_only_k3 − I_jev_k3 | 0.060496 | 0.042777 | 0.034927 | 0.030248 | 0.024697 | 0.021388 |
| p_yes_only_k3 − I_general_k3 | 0.059747 | 0.042248 | 0.034495 | 0.029874 | 0.024392 | 0.021124 |

这张紧凑表固定展示全部四对、全部六个 F 的 m=1 情景；m=1.5 和 m=2 的全部 48 个额外半宽在 JSON 中完整保留，对每个单元分别正比放大为 1.5 倍、2 倍。倍率只是敏感性设定，不是 SD 的置信界，也不保证足以覆盖未来分布变化。F 增大四倍使公式半宽减半，这一代数关系不保证新数据能满足独立性或分布条件。

## 全部半宽目标与全部倍率的 ceiling 输出

表中是满足该**假设公式**的最小整数 F，不是建议招募数或已验证的最低充分样本量。倍率对未取整的 F 平方放大；必须先按公式计算再向上取整，不能简单将 m=1 的已取整结果乘 2.25 或 4。

| 对照 | SD 倍率 m | h=0.10 | h=0.075 | h=0.05 | h=0.025 |
|---|---:|---:|---:|---:|---:|
| p_yes_only_k3 − dense_k3 | 1 | 11 | 19 | 42 | 166 |
| p_yes_only_k3 − dense_k3 | 1.5 | 24 | 42 | 94 | 373 |
| p_yes_only_k3 − dense_k3 | 2 | 42 | 74 | 166 | 663 |
| p_yes_only_k3 − reranker_k3 | 1 | 15 | 25 | 57 | 225 |
| p_yes_only_k3 − reranker_k3 | 1.5 | 32 | 57 | 127 | 507 |
| p_yes_only_k3 − reranker_k3 | 2 | 57 | 100 | 225 | 900 |
| p_yes_only_k3 − I_jev_k3 | 1 | 9 | 16 | 36 | 141 |
| p_yes_only_k3 − I_jev_k3 | 1.5 | 20 | 36 | 80 | 317 |
| p_yes_only_k3 − I_jev_k3 | 2 | 36 | 63 | 141 | 563 |
| p_yes_only_k3 − I_general_k3 | 1 | 9 | 16 | 35 | 138 |
| p_yes_only_k3 − I_general_k3 | 1.5 | 20 | 35 | 78 | 309 |
| p_yes_only_k3 − I_general_k3 | 2 | 35 | 61 | 138 | 549 |

某些宽目标得到小于 24 的整数，是公式的代数输出，不能覆盖仅用 24 个暴露 family 估计 SD 的不稳定性，也不能支撑“小样本已经足够”的说法。所有目标、倍率和对照均保留；没有用观察到的正差大小倒推一个容易达到的 target effect。

## 必须保留的条件与限制

- 这 24 个 family 已被反复用于 development；SD 可能不稳定或乐观。未来方法锁定、样本准入和确认性分析必须另行决定，不能由本情景自动完成。
- family 是假设的独立单位。版本、近重复或共享来源可能让不同标签的 family 仍有关联；存在依赖时，`s/sqrt(F)` 的尺度可能不成立。剩余文档数不能直接当作独立且已准入的 family 数。
- 未来文档分布、问题难度以及每个 family 的题数和题目组成可能变化；这里的 SD 不必可迁移。倍率 1.5/2 只展示不确定性的尺度，不是对分布转移的保证。
- 目标量是各 family 等权的平均差（FB）。它不是按题加权的比率估计量（QW），不能直接用于后者的精度规划。
- 标准正态近似加上估计 SD 没有小样本覆盖保证；这是条件尺度示意，不是已经获得的 95% 区间。四个对照和全部情景没有多重比较或同时覆盖控制，不作显著性声明。
- 不估计 power，也不把已观察的均值差作为 target effect。不能从四对中挑最小 SD 或最有利差值，宣称未来确认样本已经足够。
- 这些是已完成 ranking 对照。没有测量新的 shared-relation 机制，因此无法为未测的关系效应提供功效或 effect-size 证明；也不增加模型新颖性的证据。

## 数值核验与公开边界

原情景只读取已完成答案分数及来源审计，未读取新 QA、gold、原文或 official test。独立核验使用 NumPy `std(ddof=1)` 重算全部四个 SD/FB 均值差、72 个半宽及 48 个 ceiling；正态分位数通过 `erf` 二分另行求解，没有调用原情景或正式统计函数。最大绝对误差 **1.11e−16**，容差 1e−12；原七项来源绑定全部一致。数学复核不是独立数据复现。

本次公开文件直接导出已经核验的情景；没有重新评分、选样本或设阈值。新增 API、模型调用、数据准入均为 0。聚合不含真实 family/document/question 标识、问题、答案或本机路径，数值和公式足以独立复算全部情景。

| 来源角色 | SHA-256 |
|---|---|
| primary_answer_summary | `a2b5fdce093c0d6f0d95c075b5923fa61294b7b742c6efaa9a80794ae6a2b790` |
| primary_answer_per_question | `50127968a409186e20ecf26cb26749e1ca51eb3e465d806dccfcc4905ee30337` |
| local_answer_summary | `2db10e49db82f37b3b4999a594a38f7cc57b4856e1248e1ccc716d73cae6cc9a` |
| local_answer_per_question | `ef588a613bda688cba381a2c3d76e99120812890bcd334b7229bc0641ce600a1` |
| primary_answer_independent_verification | `9903d0d9b33bf219106b697d062469515c4e03dec5e3d959895d2deb25d7a2f6` |
| reranker_comparison_independent_verification | `6dec99715738709e1f8b7d3e19e6e50db312712193e84b96d9bc1e4381e8cac5` |
| precision_arithmetic_source | `75e395c70b589efdd5f9b54c6de3dfc7f51a82f2f37ac9fe2ec194d8f04dc1b8` |
| scenarios | `f6969cd6f7332f3bfe36d71fecf8fcde0633818d12ca31b8ffa9602e07d192a4` |
| planning_binding | `d401fc0c5cbd3e71ab8e578a488589fa3225c2701290d5b49e75348e3d69d01d` |
| verification | `82bf5ba310cf629fac8e2fa62613c74b025bd7b3788c84ee079e220997e04853` |
| review_binding | `19a4ae762458ac038ca918b455aff06c2eb1cc69eacfdf31888f6f3e82821571` |
