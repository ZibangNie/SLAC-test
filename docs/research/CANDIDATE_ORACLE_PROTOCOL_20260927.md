# 固定候选、同预算的精确 evidence oracle

本协议先于新 oracle 分数冻结。对象为已审计 native run02 的全部 77 个开发问题、24 个 family、六组：`given_document` 和 `corpus_32` 各自的 `leaf_direct`、`leaf_owner`、`dual_owner`。它使用既有同 77 题的参考，是 gold-guided 可达上界诊断；不是部署检索器、答案实验、独立确认或 JEV 成绩。旧 104 题 subset-oracle 的结果不进入比较分母。

## 冻结可行域与搜索

每行保留原 `candidate_global_indices`，最多 16 个。穷举全部大小 0、1、2、3 的 index 子集，包含 empty；不只枚举 reference 匹配项，不按答案、figure、gold 是否可达或是否不可回答筛题。不新增候选、检索、embedding、训练或模型调用。

| scope | direct | leaf owner | dual owner |
|---|---:|---:|---:|
| given_document | 51,596 | 48,181 | 52,416 |
| corpus_32 | 53,669 | 52,310 | 52,856 |

以上为扣除去重与预算不合法项之前的精确组合数，总计 **311,028**，每行最多 697。跨全部行共有 **150,461** 个不同 global-index 子集，可共享实际 token 计数；任何指标仍按各问题的参考分别评分，不跨题缓存 gold 分数。

合法选择最多包含一个完全相同 `(document, native_text)` 身份，保留完整原生单元，使用原 `render_pack` 与 `PackCounter`，整个包不超过 1,024 个实际 BGE tokens，不截断。不预先合并同文的不同位置：它们可能拥有不同 header、token 数和末级 index tie。对组合直接检查合法性，不交给贪心 packer 改成另一个集合。

在使用 gold 选择 oracle 前，逐行核对原 actual 选集：属于候选、最多三项、无同源同文重复、原渲染 SHA 和实际 tokens 一致且预算合法。原选集因此是可行域见证；每题 oracle F1 必须不低于 actual F1。

## 目标与参考语义

排序键固定为：最大 source-qualified Evidence F1；并列时最大 source-qualified recall；再并列时最少实际 tokens；最后为排序后的 global indices 字典序。F1 与 recall 各自跨原 annotation 取最大，可能来自不同参考，不强制同一参考。

采用同一 exact-string 交集和原 list-length 分母的 `Fraction` 比较，避免浮点尾差改变 tie；最终见证输出再与原 `source_qualified_metrics` 浮点接口逐项核对。保存每个 reference 的重复项分母，不合并 annotation，不删除 `FLOAT SELECTED`。来源身份要求 `(document, exact native string)` 同时匹配，别的文档同名标题不算命中。

沿用 [qasper_metrics.py](qasper_metrics.py) 的已冻结官方语义：unanswerable reference 的 evidence 为 empty；empty prediction 对任一 empty reference 的 F1 和 recall 都是 1；对非空 reference 的 empty prediction 为 0。公开每组包含 empty reference 的问题数与 oracle empty 选集数。空参考使 candidate recall 与 oracle recall 不满足简单单调上界关系，不能把 oracle 弃答解释为可部署 answerability 能力。

## 预固定报告与比较

全部六组各报告 candidate source-qualified recall、actual source-qualified F1/recall、oracle source-qualified F1/recall、oracle−actual F1 gap、actual/oracle 实际 tokens、oracle 单元数及 empty 数。均值同时保留问题等权与 family 等权；gap 到 actual 是每组描述，不据此选择部署策略。

四项主配对固定为两个 scope 各自的 `leaf_owner − leaf_direct` 与 `dual_owner − leaf_owner`，主指标为 oracle source-qualified Evidence F1。复用 24 个 family 的整簇 PCG64 seed 20260927、10,000 次共享重采样，报告全部方向、增/同/减和双侧线性 percentile 95% 区间，两种权重均保留，不做多重比较校正。

候选 recall、实际表现和同预算可达上界分别解释。更大 oracle 上界只说明对应固定候选存在更好的 gold-guided 选择，不证明实际 selector、共享关系或完整 SLAC 改善。实际长度仍可能不同。跨库 scope 是既有 query-only 压力诊断，不是标准给定论文 Qasper；其 header 歧义未修，任何 oracle 选集都不得送答案生成或回写部署检索器。

## 执行、完整性与公开边界

先封印新源码、测试、本协议、原 native plan/run/audit、所有既有数据与 tokenizer 绑定，再运行。prepare 只核来源和候选组合规模，不计算新 oracle 分数。run 首先重放已有 native CPU 审计，再检查全部 actual 见证和穷举新上界。穷举串行执行，PyTorch CPU threads 固定为 1；NumPy/BLAS 的线程数未单独强制，因此不声称整个进程严格单线程。无 API、无 GPU、无模型推理；使用现有 tokenizer 与缓存数据。

计划固定新 run 目录，拒绝覆盖或改名重跑；含来源核验与计算的时限为 1,200 秒。失败或超时留下失败记录，不声明完整、不公开部分质量结果。完整审计从同一封印来源重算全部候选、Fraction tie、见证、统计和四项比较，检查组合数与合法性分区对账。原问题、参考、逐题 oracle 和 global indices 留在 ignored artifacts，公开只聚合、成本/计时说明和来源 hash。实际 CPU 时间只在完整运行后报告；既有组合数量是资源估计，不是已测 oracle 性能。
