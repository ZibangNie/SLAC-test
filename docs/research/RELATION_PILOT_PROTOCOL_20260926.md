# Qasper 局部关系开发 pilot：锁定协议

日期：2026-09-26。此协议承接 [开发基线](QASPER_DEVELOPMENT_BASELINES_20260926.md)；原始 [实验草案](EXPERIMENT_PROTOCOL_DRAFT.md) 保留其当时状态。本轮是其固定候选归因模式的一个小规模实现，**不是整套 SLAC 检索链路或独立确认评价**。实际执行状态与验证结果见 [本阶段进展](RELATION_PILOT_PROGRESS_20260926.md)。

执行修订：以下保留最初冻结配置；正确凭据已找到并验证。GPT-4.1 Mini 请求被拒绝后，通用后端改为 Qwen3.6 Plus，并按提供方文档适配 JSON Object 输出。失败记录、修订时间顺序、预算继承和实际结果统一记录在 [真实执行报告](RELATION_PILOT_EXECUTION_20260926.md)。

## 问题与锁定范围

在相同候选文本、检索排序、后端判断和最终预算下，让已判断的局部依赖关系同时参与候选分组与证据选择，是否改变选集及官方 Evidence F1？同时比较 JEV 与通用 LLM 的判断后端。

本轮只执行每个后端的一套真实判断，并回放 I/C/S 三种策略。I 与 C 在相同判断结果下应完全一致，这是正确性检查；没有分别实跑六套冷/热系统，不能用回放计算实测缓存节省。没有生成答案，不报告 Answer F1，也不填写原草案的“QA 不退化”门槛为通过。索引和召回不随 chunk 改变，结果只能检验局部关系选择机制，不能证明完整的跨阶段 RAG 收益。

## 数据与候选

- 使用既有 32 篇 validation 开发池；此前已经看过 BM25、dense 和 oracle 结果。此次选择和参数属于公开记录的开发调整。
- 按 `SHA256("SLAC-JEV-DEVELOPMENT-v1|" + family_id)` 选前 8 个 family；每篇按相同前缀对 question_id 排序，最多取 2 题。实际为 **8 篇、8 family、15 题**，不更换 seed 或补抽来填缺额。
- 每题从完整 BGE 排序取前 8 个 seed；按 seed 排名依次加入原文左邻居、右邻居，最多 16 个 native units。实际 14 题有 16 个候选，1 题有 14 个；共 238 个 query-unit 判断。
- 静态关系只枚举两端都在该题候选中、且在完整原文中真正相邻的 pair；同一论文内跨 query 去重后为 **139 个关系判断**。
- 模型字段只含 unit ID/正文及 support 任务的 query。准备阶段从 QA sidecar 只使用问题、身份与来源字段；参考答案与证据仅用于单独评分与明确命名的 gold subset oracle，不参与候选选择或模型输入。
- 输入、旧 dense 排序、准备产物、模型配置、脚本和 tokenizer 的 hash 在执行前绑定。旧 dense 排序本轮通过完整 ID 排列及来源检查并重新冻结，不伪称再次从 embedding 独立重算。

原始正文、问题、模型请求/响应、向量和 key 不发布到 GitHub。Qasper 官方 test QA payload 不读取。文档查找时一次过宽的文件搜索扫描了旧 `structure_dataset/.../test/*.tree.json`，已保守登记为 legacy test 暴露；也覆盖已知 train 探索文件。它们未参与此次选题或提示词设计，后续读取限定具体代码文件；不能据此宣称旧 test 仍独立未暴露。本地 `qasper-relation-public-metadata-01/execution_scope.json` 保存了范围记录。

## 后端与可见上下文

| 后端 | 请求 ID / 路由 | 本次公开元数据 |
|---|---|---|
| JEV | `typesafe/jev-1.13`；`/api/alpha/decisions`；只允许 typesafe | 端点快照 `typesafe/jev-1.13-20260917`，32,000 context；输入 $0.042/M，输出 $0/M |
| General | `openai/gpt-4.1-mini`；`/api/v1/chat/completions`；只允许 openai | canonical slug `openai/gpt-4.1-mini-2025-04-14`；输入 $0.40/M，输出 $1.60/M；支持 JSON Schema |

以上来自当日 [JEV 端点](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints)、[GPT-4.1 Mini 端点](https://openrouter.ai/api/v1/models/openai/gpt-4.1-mini/endpoints) 和 [OpenAPI](https://openrouter.ai/openapi.json)，原始公开响应本地留存。请求 alias、公开 canonical slug 和真正返回的 model 字段分别记录；alias 不等于已证明底层权重版本。首个模型身份必须在冻结 allowlist，后续变化即停机。provider 若未在响应报告，会明确标记缺省，不补造供应商证据。

每篇静态任务、每题 support 任务分别合批，最多 8 题；若任一后端 payload 超过 24,000 UTF-8 bytes，两个后端共同缩小批次。超长完整单元无法装入则拒绝，绝不截断。**实际可见上下文是整批 state**，而非物理隔离的单 pair；指令指定本题 item 并要求忽略其他 item。JEV 使用原生 choice，通用模型以相同 state/questions 和固定 JSON Schema 返回标签，temperature=0、最多 1,024 output tokens。两种接口的包装不同，保留完整 payload。

静态标签为 dependent / independent / unknown；只有明确的未完句/列表、局部定义、引用或标题作用域才算 dependent，主题相近不算。support 标签为 yes / no / unknown。两后端采用同一套离散标签权重：yes/dependent=1，unknown=0.5，no/independent=0。原始概率和 confidence 如返回则保存，但主策略不把不同后端的分布当作同一校准尺度；缺少可选分布不会补造。

## I/C/S 策略与基线

I/C 保留 native unit 边界，按 support 权重、dense 排名、原文顺序依次选择。no 不可选，unknown 可选；最多 3 个不同 native 文本，使用完整证据包的 BGE tokenizer 计数，最终上限 1,024 tokens。

S 按原文顺序处理 dependent 边。两组完整渲染后不超过 384-token chunk 目标才合并；超目标的原始 singleton 保留并显式记录，最终证据上限仍是硬约束。只有在分组阶段接受的边可参与选择：待选节点若与已选节点有这种边，获得固定 +1 bonus，每步重新排序。多个邻居不累加，no 仍被排除。它是局部贪心启发式，既不声称全局最优，也不要求整个 chunk 都进入证据包。

所有模式共用同一最多 3 单元与 1,024-token 上限；真实长度另报。top-3 来源于先前开发基线观察，不伪装事前独立验证。trace 区分接受了合并、给予了 bonus、改变了排序和最终选集变化；仅产生关系或使用 bonus 不是收益证据。

保留相同题目的完整原文 dense top-3、受限候选 dense top-3、空证据与受限候选 gold subset oracle。此次 oracle 同样最多选 3 个单元，与上一阶段无此单元数限制的 oracle 不直接同列。oracle 利用参考证据，只是可达性诊断。候选截断带来的变化与后端/策略变化分开呈现。

## 执行与评价

客户端总费用准入上限 **$2**、最多 **160 次请求 / 1,600 个判断 / 8,000,000 个保守输入 token allowance**，运行限时 30 分钟。provider max_price 随每次请求发送；输入以 UTF-8 bytes 加余量估算，JEV 再按问题数保守重复计入，附加 50% 费用余量和每次至少 $0.005 预留。请求前写盘预留，失败不退回预留。

在任何真实调用之前，完整请求预检得到 106 次请求、约 $1.034 总预留和 6,503,815 个保守输入 allowance。因此将初拟 4M allowance 调整为 8M；没有放宽 $2 费用或请求/问题数量上限。这些 allowance 不是实际模型 token 或已产生的费用，最终精确清单以冻结配置为准。

这是客户端保守准入账本，**不是服务端账户硬限额或准确 tokenizer 证明**。实际 usage/cost 缺失、超过预留、模型漂移、非法 ID/标签、transport 不确定或供应商错误均立即停机；不自动重试、不换模型、不把失败填成零分或成功。完全相同的 payload 再次付费提交会被拒绝；I/C/S 从保存的判断回放。认证仅从用户指定文件在执行时加载，请求头与 key 不写盘，响应和账本统一脱敏。

主报告官方完整 Evidence F1 的 question macro 和 document/family macro，辅助报告原参考 evidence recall（这是列表 recall，**不是**原草案尚未实现的 token-span proxy）、实际证据长度、每题选集变化和失败数量。配对差异以 family 为单位；8 family 仅用于开发，不宣称统计确认。只有取得真实输出后才报告后端/策略分数。若未执行、执行不全或 H1/H2 无收益，照实保留该状态。
