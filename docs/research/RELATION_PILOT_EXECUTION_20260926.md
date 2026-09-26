# 局部关系 pilot：真实接口执行与修订记录

日期：2026-09-26。承接 [原始冻结协议](RELATION_PILOT_PROTOCOL_20260926.md) 与 [离线准备报告](RELATION_PILOT_PROGRESS_20260926.md)。下列运行保留不同目录，不覆盖失败记录；数据、候选、业务判断指令与 I/C/S 策略保持原配置，通用模型输出包装的修订明确记录。

**结论：真实双后端 pilot 已完成。JEV 的调用费用较低，两后端的支持判断均改善了本批开发题的平均 Evidence F1；共享关系 S 尚未带来额外 F1 收益。当前证据支持继续研究低成本判断与误差来源，不支持宣布共享框架创新或论文主结论已经成立。**

后续已经执行零 API 的换源、合并门槛、方向、oracle 可行域与选集数量诊断，并冻结剩余 24 family / 77 题的扩展开发清单。结果见 [机制诊断报告](RELATION_MECHANISM_DIAGNOSIS_20260926.md)；本页原 pilot 成绩保持不变。

## 凭据与 JEV 接口

用户补充指定的本地 RTF 文件可提取 OpenRouter 通用 key。key 仅在进程中用于 Bearer 认证，未写入源代码、请求 JSON 或 Git。此前 `apikey.txt` 的格式判断仅针对另一个文件，不能据此断定 OpenRouter 账户不可用。

已重新核对 [OpenRouter Decisions 文档](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)：JEV 请求走 `POST /api/alpha/decisions`，model 为 `typesafe/jev-1.13`，使用 state 与 choice questions。2026-09-26 的实际首响应是 `typesafe/jev-1.13-20260917` / TypeSafe，7 个任务全部通过身份、schema、usage 和费用校验；输入 4,860 tokens、输出 671 tokens，reported cost 为 **$0.00020412**。

## 原计划被阻断的原因

`qasper-relation-run-01` 的第二个请求（GPT-4.1 Mini）收到 HTTP 403，客户端按约定停止，没有生成六格结果。一轮独立人工诊断再次得到明确的服务条款拒绝，原文为：`The request is prohibited due to a violation of provider Terms Of Service.` 没有收到更具体的原因，不能断言一定来自地区、余额或某一种账户状态；之后停止该模型的请求。

旧运行共两次尝试，加上一次诊断合计三次尝试，保守预留 **$0.0398937630**。只有上述 JEV 成功调用返回了费用，两次 403 没有返回 usage，记为费用未报告，不能补写零。失败目录与诊断目录均保留。

## 通用模型的显式修订

改用 **Qwen3.6 Plus** 作为通用判断对照，选择发生在没有通用模型成功输出、没有六格结果时。它是另一模型的独立配置，不是把失败的 GPT 请求改名为成功结果，也不代表对 GPT 的质量比较。

| 字段 | 修订值 |
|---|---|
| profile | `qwen36plus` |
| 请求模型 | `qwen/qwen3.6-plus` |
| 公开 canonical slug | `qwen/qwen3.6-plus-04-02` |
| provider | 仅 `alibaba`，禁止 fallback |
| 输出 | 固定 JSON Schema；temperature=0；最多 1,024 tokens |
| reasoning | 显式 `enabled=false`，用于有限标签分类 |
| 本次公开价 | 输入 $0.325/M，输出 $1.95/M；每次请求发送 max_price |

来源为 [OpenRouter Qwen3.6 Plus 模型页](https://openrouter.ai/qwen/qwen3.6-plus) 与 [端点元数据](https://openrouter.ai/api/v1/models/qwen/qwen3.6-plus/endpoints)。别名、公开 canonical slug 和实际响应身份分别保存；不能把 alias 当作独立证明的不可变底层权重。

两个后端仍使用同一批次内的 state/questions。对全部 53 个旧 batch 独立重算确认：切换 general profile 后，**JEV payload 的每个字节哈希保持不变**；Qwen 请求最大的 payload 为 22,547 bytes，满足 24,000-byte 上限。相同的实际上下文和选择规则不等于两个模型的原生 tokenizer 或最终实际证据长度相同。

## 失败恢复与费用边界

新计划只复用状态为 completed 且与新请求的 endpoint、payload、prompt version、cache key、模型身份、usage 与 task IDs 精确匹配的旧响应；重新解析标签并绑定所有来源文件 hash。缓存命中记为旧付费结果复用，不伪称新 API 调用，也不拿它计算实测缓存加速。

旧失败和诊断预留计入同一个 $2 总准入预算；新运行只获得扣除历史尝试后的剩余额度，请求/判断/token allowance 上限也相应扣减。缺失费用与已报告费用分列。任何来源不匹配、版本变化、响应不完整或预算异常仍停机，不自动重试。

HTTP 错误处理另补了安全诊断：最多读取 2 MiB，只保存脱敏且限长的结构化错误与有效数值 usage；非法 JSON、达到尺寸上限或读取失败不保存原始内容，也不覆盖原 HTTP 状态。

## Qwen 输出格式兼容性修订

`plan-02` / `run-02` 使用上述严格 JSON Schema 配置，复用首个 JEV 响应后，Qwen 首次请求返回 HTTP 400。一次同 payload 的受限诊断也返回 400，仅确认提供方为 Alibaba，消息为 `Provider returned error`，没有得到可用的更具体结构化原因。两次调用各预留 $0.013701675，均未报告费用；原目录和请求保留，不记为成功判断。

随后核对 [Alibaba 官方结构化输出文档](https://help.aliyun.com/en/model-studio/qwen-structured-output)：Qwen3.6 Plus 在非思考模式支持 JSON Object，但不在其 JSON Schema 支持列表中。这与 OpenRouter 端点的概括性 `structured_outputs` 标记不一致；格式不兼容是有依据的解释，仍不是服务端确认的唯一 400 原因。

因此显式新增 `qwen36plus-json` profile：使用 `response_format={"type":"json_object"}`，将同一输出 schema 作为格式说明附在 system message，保留原 state/questions、非思考模式、temperature、输出上限与本地严格解析。拒绝额外/遗漏/重复 ID、非法标签、截断输出，不进行 JSON 修补或补造判断。原 `qwen36plus` profile 留存用于重建失败请求。该修订发生在没有任何通用模型有效标签或六格分数时。

新计划计入原成功请求、两次 GPT 403 和两次 Qwen 400，历史共五次尝试、四次未知费用，保守预留 $0.0672971130，已报告费用仍为 $0.00020412。多个旧运行及诊断目录均绑定哈希；只有原 JEV 成功响应可提供复用标签，其他记录仅计账。

## 最终冻结计划

`qasper-relation-plan-03` 使用 `qwen36plus-json`，配置 SHA256 为 `4f84c59a6d9de6facfeef96dc4b20b06f6d75b8c56c370bfbac5699c2cff4b1f`。它绑定 36 份输入/代码/历史文件与 4 份计划产物，其中历史文件 11 份。再次从 prepared 独立重建得到相同 53 个批次，全部 JEV payload SHA 与 plan-01 相同；通用模型的 state/questions 相同，JSON Object 包装最大 22,650 bytes。

计划含 106 个逻辑请求（两个后端各 53）、754 个判断；复用 1 次旧 JEV 响应，新增 105 次请求。新增保守预留 $0.9599546755，连同历史累计 110 次尝试、782 个判断、input allowance 6,599,698、保守预留 **$1.0272517885**。这属于准入预留，不是实际账单。

冻结前完整测试 **318 passed，8 warnings**；warnings 均为已知 PyTorch nested-tensor 提示。独立只读审查核对了模型包装、全部批次、旧响应复用、历史计账与 hash 绑定，无额外 API 调用。

## 真实结果

`run-03` 完整执行耗时 528.19 秒，新增 105 次调用、复用 1 次，得到每个后端 377 个有效判断，生成 8 family、15 题 × 6 配置 = **90 条完整记录**。响应模型为 `typesafe/jev-1.13-20260917` / TypeSafe 和 `qwen/qwen3.6-plus` / Alibaba。没有模型身份漂移、残缺标签或超预算证据包。

以下均为官方 exact-native-evidence / max-reference 指标，最终证据不超过 1,024 BGE tokens、最多 3 个 native units；两后端独立/缓存模式 I/C 的选集、pack hash 与指标逐题全等，因此合并展示。

| 方法 | Evidence F1 问题宏平均 | 文档宏平均 | 平均实际证据 tokens |
|---|---:|---:|---:|
| dense top-3（相同候选） | 0.2057 | 0.1929 | 437.1 |
| JEV I/C | 0.2359 | 0.2211 | 526.9 |
| JEV S | 0.2359 | 0.2211 | 516.3 |
| Qwen I/C | 0.2870 | 0.3315 | 439.9 |
| Qwen S | 0.2870 | 0.3315 | 430.1 |
| 空证据 | 0.2000 | 0.2500 | 0.0 |
| 受限 gold subset oracle（不可部署） | 0.6444 | 0.6667 | 99.3 |

与 dense 基线逐题配对，JEV 为 **2 胜 / 13 平 / 0 负**，问题宏平均增加 0.03016；Qwen 为 **4 胜 / 9 平 / 2 负**，增加 0.08127。JEV I/C 的证据比 dense 平均多 89.7 tokens，不能把它的分数变化描述为等长度提升。Qwen 本轮 F1 高于 JEV；未计算显著性或声称可泛化优势。

共享机制确实执行，但效果有限：

- JEV S 在 13 题发生合并、3 题使用关系 bonus，最终只改变 **2/15 题**的选集，平均少 10.53 tokens。
- Qwen S 在 12 题发生合并、2 题使用关系 bonus，最终只改变 **1/15 题**的选集，平均少 9.87 tokens。
- 两后端 **所有 15 题的 S−C Evidence F1 与 recall 都为 0**，不是正负样本抵消后的平均为零。边界改变、选择顺序改变和质量提升必须分开报告。

静态关系的后端一致率为 111/139 = 79.86%；支持判断一致率为 211/238 = 88.66%。JEV 判断了 43 个 dependent、75 个 yes，Qwen 为 29 个 dependent、58 个 yes。JEV 在本批数据更倾向保留依赖和支持，但没有独立人工标签，不能据此判断哪一个更准确。

按参考证据分层（事后诊断，不是可回答性分类）：3 题有至少一份空参考，12 题的所有参考均非空。在后 12 题上，dense / JEV / Qwen 的 F1 分别为 0.2155 / 0.2532 / 0.2754，S 仍没有额外收益。空参考会影响整体排序与差距，不能用全部平均分代替分层分析。

## 实际费用与验证

| 成功后端调用 | 唯一请求 | 判断数 | 服务端 input tokens | 服务端 output tokens | 已报告费用 |
|---|---:|---:|---:|---:|---:|
| JEV（含一次旧响应） | 53 | 377 | 192,895 | 35,514 | $0.008101590 |
| Qwen3.6 Plus | 53 | 377 | 231,227 | 24,921 | $0.123744725 |

本轮新增费用 $0.131642195；连同先前成功 JEV，累计**已报告费用 $0.131846315，另有 4 次失败/诊断请求未报告费用**。不得将此已知费用当作已确认的账户总账单。两个后端的 tokenizer 与接口包装不同，以上是同任务量的实际服务端记录，不是统一 token 单价实验，也不包括索引、答案生成或完整 SLAC 的成本。

新增 `analyze_qasper_relation_pilot.py` 只读重解析所有保存的响应和复用来源，核对完整任务覆盖，并用同一 tokenizer 从输入重放 90 条结果、trace、pack hash 和指标。输出仅有聚合统计与匿名来源哈希，标记为事后探索分析。最终完整测试 **335 passed，8 warnings**。计划与原始输入哈希在执行及分析后均保持一致。

另一份独立审计另行实现 F1/recall 公式与分组/贪心选择步骤，核对真实相邻边、384-token 合并门槛、1,024-token/3 单元限制、逐题与配对统计，全部通过。S 改变的选集均是在未命中任何参考标注的单元之间替换，解释了本次指标不变；未命中参考不等于已经证明这些文本对回答没有用。公开的 [机器可读聚合结果](results/qasper_relation_pilot_20260926.json) 不含题文、答案、逐题 ID 或凭据。

## 已推进的下一步

已在本地生成 **75 个盲审样本**：全部 28 个关系分歧、27 个支持分歧，加上各 10 个由固定 hash 选择的一致对照，整体按 hash 打乱。盲审文件不含后端名称、模型标签、分歧组别、参考答案；模型标签另存。当前人工标注数为 **0**，不能将这份材料当作已有 gold，也不能把分歧富集样本直接用于估计总体准确率。

后续顺序：

1. 先固定盲审意见和理由，再揭示模型标签；重点判断多出的 yes/dependent 是有用的上下文支持，还是主题相关性误判。独立评审应使用盲审文件，避免先看模型输出。
2. 结合本次仅 2/1 题选集变化，检查“文段理解依赖”是否真正对应“回答问题所需证据依赖”，以及 384-token 合并门槛、最多 3 单元和 no 排除对关系利用的限制。在当前开发题上诊断，保留本轮结果，不回改原阈值来包装提升。
3. 锁定下一版假设后，再扩展开发规模并加入答案生成、长度控制和真实冷/热执行；最终在未参与这些开发决策的独立划分上确认。当前固定候选回放没有检验 chunk 改变后的索引/召回，也没有验证 Answer F1、缓存节省或整套 SLAC 收益。

本阶段没有继续训练或扩大付费样本：首先需要说明关系共享为何应改善证据，而不是靠更多训练掩盖这一未验证环节。

## 本地产物与复现

原始内容、模型输出与盲审文本均留在 Git 忽略目录：

- `artifacts/research-foundation/qasper-relation-plan-03`：最终冻结配置、批次与基线。
- `artifacts/research-foundation/qasper-relation-run-03`：完整账本、请求/响应、标签、逐题结果与 trace。
- `artifacts/research-foundation/qasper-relation-analysis-01`：事后聚合诊断及成功调用费用拆分。
- `artifacts/research-foundation/qasper-relation-independent-audit-01`：独立重实现审计与来源 hash。
- `artifacts/research-foundation/qasper-relation-blind-review-01`：盲审文本、单独存放的模型标签、规则及来源 hash。
- `artifacts/research-foundation/phase4-verification-release`：最终测试记录。

从现有冻结结果复算，无 API 调用、无需 key：

```powershell
& 'C:/Environment/python/venvs/slac-research/Scripts/python.exe' -X utf8 `
  docs/research/analyze_qasper_relation_pilot.py `
  --plan artifacts/research-foundation/qasper-relation-plan-03 `
  --run artifacts/research-foundation/qasper-relation-run-03 `
  --prepared artifacts/research-foundation/qasper-relation-prepared-01 `
  --output artifacts/research-foundation/qasper-relation-analysis-new
```

输出目录必须不存在；已冻结源代码或输入改变时会拒绝运行，不应修改旧 seal。迁移机器应保留原产物并在明确记录路径变化后重新建立分析绑定。
