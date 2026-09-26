# 主答案与 BGE reranker 的事后配对分析协议

本协议固定后才计算新增的跨 baseline 差值，但两轮父实验的完整方法均值已经可见。因此这是 **posthoc development analysis**，不是预注册验证或独立确认。协议自身不启动分析；完整父审计、独立核验完成和 Root 对本计划的单独 release 均为运行条件。无新增 API、模型执行、问题、答案标注或 official test 读取。

## 唯一新增问题与完整比较

在相同的 given-document 候选池和答案生成设置下，JEV/通用判断器的选择是否优于已有的 `BAAI/bge-reranker-v2-m3`？固定全部 77 题、24 个 family，k=3、完整原生段落包、BGE 1024-token 上限。候选来自同一份 frozen prepared：dense top8 加原定相邻段，最多 16，合计 1214 个 query–unit 对。不同方法保留其原有的候选评分和选择规则，不新增排序变体或改动已保存结果。

仅新增四个方向固定的比较：`I_jev_k3 − reranker_k3`、`I_general_k3 − reranker_k3`、`ordinal_then_p_yes_k3 − reranker_k3`、`p_yes_only_k3 − reranker_k3`。全部报告，不挑最高分方法。每对报告已通过完整父审计的 `official_answer_f1` 与 `actual_evidence_tokens`，不重新读取 gold 或修改官方评分。F1 是父实验的逐题官方 max-reference F1；长度正差只表示更长，不表示质量更好。

两个原始 JEV score 排序方法在已有主结果中产生相同选集和共享答案。本分析再次逐题核对并报告一致数量；这两行不是独立复现，也不能把原始分数解释为已校准概率。

## 可比性门槛

读取本地完整答案和恢复后的完整主答案两个父实验的 plan/run/audit。检查各自 462 行、77 题、24 family、完成状态、Root 完整审计回执、plan/输出文件 hash 和精确实际生成模型；任何缺项或变动拒绝。保存来源路径只放本地 ignored binding，公开聚合不含真实标识、路径、问题、答案或正文。

两个阶段必须继承同一 prepared 内容 hash；reranker 与主 support 的计划必须绑定该候选池及原 1024/max3 设置。完整 payload 检查相同 system prompt、`slac-qasper-answer-v1`、`qwen/qwen3.6-plus`、Alibaba 单 provider、temperature=0、max output=512、reasoning disabled 和 JSON answer 合同。沿用 exact-payload cache，绝不重新请求答案。

计算差值前，完整匹配 `(family_id, doc_id, question_id)`。对共同的 dense 与 empty，各 77 题逐项验证完整 payload、选中 IDs、pack hash、实际 tokens、原始预测字符串和官方 F1 完全一致。empty 是同 prompt 的空 evidence abstention 对照。任一不符则停止，不通过删除题目、宽松文本匹配或重新生成来修复。

## 固定统计及解释边界

复用原纯统计函数，按排序后的全部 family 使用 NumPy PCG64、seed=20260927、10,000 次 multinomial 全 family 重采样。同一 draws 用于四对、两个指标。每次保留被抽中 family 的所有题。

- Question-weighted：抽样后的全部逐题差值和除以题数。
- Family-balanced：抽样后的 family 内逐题平均差，再对 family 等权平均。

报告全部 16 个双侧 95% percentile 区间（线性分位数）、五方法双权重均值、逐题胜/平/负（绝对容差 1e-12），以及长度增/平/减。无 p 值、无多重比较控制、无最高 k 选择。选集或 payload 相同所共享的响应不贡献独立生成重复；区间不覆盖重复采样 LLM 的变异。相同长度上限不是实际长度匹配，此处只报告长度差，不声称已识别长度之外的因果效果。

这补充的是强 reranking baseline 的直接配对证据。它不证明 corpus 检索改进、共享关系机制、概率校准、生产速度/成本优势，或未暴露数据上的收益。正负结果与跨零区间全部保留。

## 两阶段执行

`analyze_qasper_primary_vs_reranker_answers.py prepare --output <new-plan-directory>` 只检查完成元数据、来源及共同合同，hash 绑定完整父产物、新协议/源码/tests/统计来源；不打开逐题质量记录、不计算跨 baseline 差值。计划目录只包含 `plan.json`、`plan_manifest.json`，禁止覆盖。

Root 在两个完整独立核验通过后另存 JSON release：`root_release=compute_posthoc_primary_vs_reranker_answers`，绑定 `plan_sha256`、`input_binding_sha256`，并写 `complete_parent_independent_reviews=true`。该回执本身的 SHA-256 必须显式传入分析命令。

`analyze_qasper_primary_vs_reranker_answers.py analyze --plan <plan-directory> --released-complete --release-receipt <receipt.json> --release-sha256 <sha256> --output <new-analysis-directory>` 执行全部桥接校验后一次生成聚合 `analysis.json` 和 ignored 逐题/来源文件。Root 的单独放行前不执行该命令。来源重校验失败不发布结果，不改任何父计划、源码、run 或 audit。
