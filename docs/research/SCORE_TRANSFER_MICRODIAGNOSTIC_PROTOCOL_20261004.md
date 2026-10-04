# 四题 JEV 分数选择迁移：执行与读数协议

2026-10-04；在新增模型判断与 reference 读取之前固定。沿用[已推送的离线准备](SCORE_TRANSFER_PREPARATION_20261004.md)，不重新选样或扩候选池。目标是检验旧 support 分数选择在这四题上的接口行为和证据选择差异，不是提出新算法或确认历史 Answer F1 优势。

## 固定执行

- 四个操作性组件各一题，保留原问题、完整的 16 候选池与候选原序；总计 64 项 support 判断。
- 恰为已封印的八批，每批八项，同批不跨问题。沿用 `slac-local-decision-v1`，请求正文不修改。一般模型与答案生成请求均为零。
- 2026-10-04 10:05:17 UTC 的[官方端点元数据](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints)确认请求模型 `typesafe/jev-1.13`、TypeSafe 路由、版本名 `typesafe/jev-1.13-20260917` 及输入 $0.042／百万 tokens、输出 $0。请求仍发至[官方 Decisions API](https://openrouter.ai/docs/guides/community/jev)的 `/api/alpha/decisions`。
- 单独建立本阶段 **$0.10** 客户端预算；八批保守预留总计 **$0.0562514400**。最多八次物理尝试、64 项判断，不重试、不切模型或供应方，不继承旧大实验额度。预留不等于实际支出或服务端硬扣费上限。
- 新计划记录当前时间窗口；单请求看门狗最多 65 秒，整个工作进程最多 180 秒。执行前复核来源、请求 hash、预算、时限及未消费标记，成功占用一次性标记后才读取凭据。结束或首次失败后本次入口不可重启。
- 首个失败、超时、未知费用、超预留用量、身份或分数契约变化即停。保留响应和账本已知费用，未返回费用的尝试保持未知，不能当作免费或退款。公开仅聚合统计与 hash，不含凭据、原文或请求／响应全文。

## 固定选择与评测

三臂是 `dense_k3`、`reranker_k3`、`p_yes_only_k3`。前两臂直接使用精确核验过的缓存包；第三臂须完整验证全部 64 项 choice 与三项原始分数。各分数有限且在 [0,1]，和在 [.985,1.015]，不重新归一化、不以 confidence 调阈值。标签为 `no` 的候选排除，剩余 `yes/unknown` 按原始 yes 分数降序、Dense 顺序、原生顺序排序。

打包沿用原逐项贪心、已接受 `native_text` 精确去重、最多三单元、实际完整渲染不超过 1,024 tokens。渲染与 JEV 输入使用 `Unit.text`；评测映射回 `Unit.native_text`。所有四个 JEV 包和八个缓存包先保存并封印，随后才能读取评价 reference。

评价只解码原 `references.jsonl` 第 **1、5、11、18** 行，校验五项身份字段与来源 hash，跳过其余行且在第 18 行后停止；不读取原始数据归档或扩展审查。调用现有 `qasper_metrics.evidence_metrics(..., text_evidence_only=False)`，保持空选集、不可回答、图表标记、重复证据与多标注取最大 F1 的原语义。

完整产出四题 × 三臂共 12 项 Evidence F1，保留每题差异、包长与入选数量。主描述性比较为 **score − BGE** 的四题均值；score − Dense 为次要描述。不得用 Dense 改善掩盖 BGE 比较，也不能只报告正例。未完成全体判断时不读 gold 或计算成功子集主均值。

四题非随机、各组件一题，因此不计算显著性、总体区间或两套独立权重证据。结果为正、零、负均结束此小诊断，不自动扩样、改阈值、追加答案生成或重新呼叫同一问题。下一步只解释本次固定读数及原文中的具体失败；不将这四题用于调参后称独立确认。

## 核验边界

新入口复用已有受限客户端的版本、账本与分数检查，但不调用旧确认实验的准备／执行入口。定向模拟测试核验完整成功、失败停止、时限、预算、身份绑定、消费标记及 reference 读取次序；模拟通过不是模型效果。

离线准备、执行计划、原始响应、一次性消费标记、包封印和只读评价产物分别保存。旧 900 题及大关系实验仍暂停。仅当前固定八批属于这一协议。

实际计划窗口、当前供应方快照与来源承诺见[机器协议](results/slac_score_transfer_microdiagnostic_protocol_20261004.json)。执行代码为 [run_score_transfer_microdiagnostic.py](run_score_transfer_microdiagnostic.py)，分两步封印与评测的代码为 [evaluate_score_transfer_microdiagnostic.py](evaluate_score_transfer_microdiagnostic.py)。
