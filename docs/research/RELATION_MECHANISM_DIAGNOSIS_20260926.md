# 局部关系机制诊断与扩展开发设计

日期：2026-09-26。承接 [真实接口 pilot](RELATION_PILOT_EXECUTION_20260926.md)。本阶段只使用已保存的真实判断，**新增 API 调用为 0**，不改原计划、提示词、源数据或主结果。下列实验均为已看过 pilot 结果后的探索诊断，不能当作新的独立确认，也不能选其中最高分替换原成绩。

## 支持判断与关系判断分开换源

固定原候选、排序、tokenizer、1,024-token 最终预算和最多 3 单元。将 support 后端与 static-relation 后端交叉成 2×2；每个组合分别回放原 384-token 合并门槛，以及取消该门槛的诊断版本。另保留两个 support 对应的 I 基准，共 10 方法、150 条逐题结果。

所有 8 个 S 组合相对同 support 的 I，均为 **Evidence F1/recall：0 胜、15 平、0 负**。JEV support 的问题宏平均 F1 均为 0.235873，Qwen support 均为 0.286984。换关系后端或单独解除合并长度限制都未得到质量收益。

原策略的关系使用过程如下。边数按 query 出现次数统计，跨题复用会重复计数；去重后的 document-edge 数另存于聚合结果。

| 同源配置 | dependent 边 | 接受合并 | 两端 support 可选 | 实际使用 bonus | 改变已接受选择的优先级 | 最终选集变化题数 |
|---|---:|---:|---:|---:|---:|---:|
| JEV/JEV | 45 | 35 | 8 | 4 | 3 | 2/15 |
| Qwen/Qwen | 31 | 25 | 2 | 2 | 1 | 1/15 |

取消合并长度限制后，JEV 的前三项变为 45→45→9，后续仍为 4→3、2 题选集变化；Qwen 的 accepted 边增至 31，后续不变。因此，本批数据的关系作用大多在“关系存在”到“两个单元都有资格被选”之间消失；仅放宽 chunk 大小不足以解决这个问题。

## 关系方向检查

原静态问题定义为“B 是否依赖 A”，原 S 则给两端对称加分。原协议已经明确记录这种对称策略，本轮保留它，并将方向作为新的诊断因素：

- symmetric：原无向加分，必须与旧 S 的每一步选择 trace、选集、tokens 和指标一致。
- prerequisite：只有已选 B 才给 A 加分。
- reverse_placebo：只有已选 A 才给 B 加分。

三种方向固定同一组原 S 已接受的边，其他选择规则不变，共 90 条逐题结果。两个后端的 prerequisite 在全部 15 题上，最终选集、pack 与 tokens 都与 I 相同。JEV 仅一次 bonus 改变了选择顺序，Qwen 没有触发；Qwen reverse_placebo 与原 S 输出全等。六组配置的 F1/recall 均逐题不变。

方向值得在方法定义中明确，但本批没有证据表明仅调整方向便能提高质量。两后端各出现 3 个 query-unit 在方向上可被触发、却因 `no` 被排除；这只是规则机会计数，不能证明补入它们有用。

## 用参考证据定位上界损失

本项明确使用 gold，不能部署。对每题最多 16 个冻结候选，穷举所有最多 3 个不同 native 文本的单元组合，包括空集和不同重复文本位置，使用真实 BGE 渲染 token 数筛选。没有假设删除文本后 token 数必然下降。参考指标仍逐 annotation 计算再取官方最大值，不合并参考。

| 可行域 / 实际结果 | JEV F1 上界或实测 | Qwen F1 上界或实测 |
|---|---:|---:|
| 完整文档可得 native 证据，无数量/token 限制 | 0.911111 | 0.911111 |
| 当前候选可得证据，无数量/token 限制 | 0.644444 | 0.644444 |
| 当前候选，最多 3 单元、无 token 限制 | 0.644444 | 0.644444 |
| 当前候选，最多 3 单元、1,024 tokens | 0.644444 | 0.644444 |
| 再限定为原 support 非 `no` | 0.644444 | 0.600000 |
| 再要求与该题实际 I **相同的选集数量** | 0.314921 | 0.332698 |
| 原 I 实测 | 0.235873 | 0.286984 |

这些是嵌套可行域的描述分解，不是因果效应估计。当前候选遗漏造成平均 0.266667 的上界差距，涉及 4 题。1,024-token 与最多 3 单元没有进一步压低候选 oracle；JEV 的 `no` 排除也没有降低最优可达 F1，Qwen 则在 1 题下降，平均差 0.044444。

固定为实际所选数量后，oracle 可行域明显缩小：JEV 平均差 0.329524，Qwen 0.267302。但这是知道参考后的数量选择机会，不是已经证明可部署的 early-stop 收益。即使数量不变，仍分别有 0.079048、0.045714 的构成/排序空间。

其中 3 题至少含一个空参考，空集可在官方逐参考取最大值的规则下得到 1 分；这解释了部分数量上界差距。其余 12 题的所有参考均非空：JEV eligible oracle / 固定实际数量 oracle / 原 I 分别为 0.555556 / 0.351984 / 0.253175，Qwen 为 0.500000 / 0.332540 / 0.275397。因此，该差距不全由空参考造成。这只是按参考结构分层，不等同于 answerability 标注。

还检查了单跳补充证据上界：允许原 `no` 的 A 进入，但必须同时选入原本 eligible 的相邻 B；B 不能由另一个 `no` 递归救援。分别使用所有真实相邻边和模型标为 dependent 的边。**两种救援上界均未超过原 eligible oracle**，即 JEV 仍为 0.644444、Qwen 仍为 0.600000。当前 15 题不足以支持为这种相邻救援机制扩大付费查询。

未命中 Qasper 参考的文本仍可能有助于回答；这些证据指标不能直接测量语义必要性或最终答案质量。

## 统一检查选集数量

在上述 gold 上界诊断之后，增加不让 gold 参与选择的统一敏感性检查：所有方法都测试最多选 1、2、3 个单元，完整报告三档；固定候选、检索顺序、支持标签与最终 1,024-token 上限。dense top-k 使用相同候选、数量上限和渲染规则。共 15 方法、225 条记录，k=3 与原实验逐项一致。

| 最多单元数 | dense F1 / 平均 tokens | JEV I F1 / 平均 tokens | Qwen I F1 / 平均 tokens |
|---|---:|---:|---:|
| 1 | 0.133333 / 86.9 | 0.244444 / 198.3 | 0.244444 / 177.9 |
| 2 | 0.122222 / 233.9 | 0.266667 / 397.3 | 0.300000 / 357.7 |
| 3 | 0.205714 / 437.1 | 0.235873 / 526.9 | 0.286984 / 439.9 |

三档下，两后端 S−I 的每题 F1/recall 仍全部为零。k=2 的平均分较高只是本批探索现象，不据此宣布最优超参数；同一 k 的实际 tokens 也不同。Evidence F1 的精度项会惩罚更多未标注单元，需要答案生成评价来判断更短的包是否保留回答所需的信息。

## 使用 JEV 已返回的原始分数排序

保存的 238 条 support 判断都含完整 `yes/no/unknown` 分数，全部有限且位于 [0,1]；236 条总和为 1，2 条为 0.99，其中 1 条原标签可选。默认 `strict-distribution` 契约要求总和在 1e-8 内等于 1，因此已保存拒绝结果，没有产生排序成绩。

在看到总和偏差、但尚未比较排序质量时，显式增加 `reported-scores` 契约，允许原始总和位于 [0.985,1.015]。这个范围对应三个两位小数值可能产生的舍入量级，**并未证实偏差确由舍入造成**。所有 238 条都保留原值，不归一化、不丢样本、不使用 `confidence`，也不声称这些值是校准概率。原 `no` 排除规则不变。

统一检查两条固定规则：`ordinal_then_p_yes` 先按原 yes > unknown，再按返回的 yes 分数降序；`p_yes_only` 在原可选集合内直接按 yes 分数降序。分数相同均按原检索顺序、文档顺序决定。两条规则都报告 k=1/2/3，共 6 个新配置；加上同档 I JEV 和 dense，共 180 条记录。选择不读取参考，不扫描阈值，也不加关系 bonus。

| 最多单元数 | 原 I JEV F1 | 两种原始分数排序 F1 | 相对 I 胜/平/负 | 原 I / 新排序平均 tokens |
|---|---:|---:|---:|---:|
| 1 | 0.244444 | 0.293333 | 3/10/2 | 198.3 / 248.3 |
| 2 | 0.266667 | 0.300000 | 2/12/1 | 397.3 / 449.1 |
| 3 | 0.235873 | 0.261587 | 2/12/1 | 526.9 / 573.1 |

两条规则在本批的选集和指标相同。k=3 时 recall 从 0.366667 到 0.416667；相对同档 dense，F1 高 0.055873，逐题为 3 胜、11 平、1 负。原始分数保留了粗标签之外的排序信息，是值得扩展验证的信号；但样本只有 15 题，且证据变长、存在退步题，没有显著性或答案质量证据。不能将其解释为关系模块增益或纯效率收益。

## 下一轮的分母与验证方向

已按 family 排除本次 pilot 的全部 8 family，冻结原开发池剩余 **24 family、77 个问题**的完整身份清单与来源 hash。选择没有使用 gold、模型分歧或 oracle 可恢复性；没有因为无答案/图表问题而剔除样本。原 8 family 里未进入本次 15 题的其他问题也一并排除，避免 family 混用。

清单 SHA256 为 `0204b0b3cc3daac49cb787e75c6d77e96602be50406d457d1431dd084d8b96f9`。这些 family 已暴露过基础检索结果，所以只称**扩展开发验证**。这只是评价分母，不是已执行实验，也不是允许无限增加调用的 API 计划。

下一轮优先回答两件事：

1. 在相同数量/预算与统一答案生成器下，JEV 粗标签和原始分数是否持续改善证据选择，较低费用是否对应可接受的答案质量？全部数量档和同档 dense 对照都需保留，不能拿不同 k 的最佳分相比较；原始分数契约与覆盖统计也需提前固定。候选召回差距单独记录，避免把扩大候选的收益归给关系。
2. 关系内容能否超过简单邻接或固定随机关系的作用？只有出现可检验机会后才新增 query-conditioned 判断。随机对照应按文档固定、保留标签数量，并报告合并门槛后实际有效边数；不能按 oracle 可恢复题筛选主评价分母。

现有 75 例分歧富集盲审材料仍用于错误分类，人工标注数仍为 0，不替代独立标注。原冻结 I/C/S 并未随 chunk 变化重建索引/召回，整套 SLAC 集成、Answer F1、冷/热缓存实验和最终独立确认仍未完成。

## 本地复现与产物

五个离线分析器都先验证真实 pilot 的完整来源与响应；不接受残缺运行，也不覆盖已有输出目录。使用同一参数形式：

```powershell
$py = 'C:/Environment/python/venvs/slac-research/Scripts/python.exe'
& $py -X utf8 docs/research/analyze_qasper_relation_bottlenecks.py `
  --plan artifacts/research-foundation/qasper-relation-plan-03 `
  --run artifacts/research-foundation/qasper-relation-run-03 `
  --prepared artifacts/research-foundation/qasper-relation-prepared-01 `
  --output artifacts/research-foundation/qasper-relation-bottlenecks-new
```

同样参数可用于 `analyze_qasper_relation_counterfactuals.py`、`analyze_qasper_relation_direction.py` 和 `analyze_qasper_relation_cardinality.py`，每次指定不同的新目录。

`analyze_qasper_jev_probability.py` 默认执行严格契约检查；复现原始分数实验必须额外显式传入 `--score-contract reported-scores`，并使用另一新目录。两种契约的结果均保留。

本轮原始产物位于 `artifacts/research-foundation/` 下的 `qasper-relation-counterfactual-01`、`qasper-relation-direction-01`、`qasper-relation-bottlenecks-01`、`qasper-relation-cardinality-01`、`qasper-jev-probability-coverage-01`、`qasper-jev-score-ranking-01`，以及 `qasper-extended-development-manifest-01`。逐题 ID、文本、oracle 选集和全部模型响应只留本地忽略目录。

可公开的完整聚合指标、配对变化、机制计数和来源 hash 见 [聚合 JSON](results/qasper_relation_mechanisms_20260926.json)。参考结构分层为对本地 bottleneck 逐题结果的补充分组统计，单独绑定其产物 hash。

回归检查：`python -X utf8 -m pytest SLAC/refiner/tests tests/research -q`，**394 passed，10 warnings**。警告来自既有 PyTorch nested-tensor 设置及 SentencePiece/SWIG 类型弃用提示。发布检查另验证了聚合结果不含真实逐题标识、待提交文件不含本地密钥、原始产物持续被 Git 忽略。此次没有训练或新增 API 调用。
