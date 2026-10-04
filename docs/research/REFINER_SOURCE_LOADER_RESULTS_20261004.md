# Refiner 原文快照贯通检索准备与重读

2026-10-04。实际读取、文本预处理、元数据保存和重读入口已经保留可验证的原文快照。固定两篇文档的 139 个 chunk 中，135 个检索文本被规范化改写；从保存的快照恢复原文后，**139 份离线 JEV 请求的完整 payload、binding、缓存键、证据渲染和来源 receipt 均与上一轮精确一致**。

本轮 API、凭据读取、本地模型推理、权重读取、训练和新增分词均为 0。没有读取真实问题、支持标签、判断分数或扩大语料。59 项定向合成测试通过。这是实际接口与持久化验证，没有新增 JEV 判断、Evidence F1、Answer F1 或论文创新性证据。

## 改动及用途

新增 [`source_records.py`](../../SLAC/retrieval/dataio/source_records.py)，从显式 registry 构造已验证的文档来源视图及 native 区间索引。chunk/leaf reader 在预处理前将原文、对象身份、atom/字符范围、坐标系统和哈希保存到 `meta.refiner_source`。检索仍使用现有规范化文本，快照随实际 lookup JSONL 序列化保存。

重读快照时，再次对照当前 registry 检查原文、哈希、坐标、对象身份以及当前记录字段；当前文本必须恰为原文或其规定的检索规范化结果。顶层、meta 与快照有冲突时拒绝。`source_snapshot_from_chunk_record` 由此恢复现有 JEV 适配器所需的不可变来源快照。保存的 dict 本身不是可信证明；校验只说明它与调用者提供的来源版本一致，不认证外部文档发布者。

实际 [`run_build_index`](../../SLAC/retrieval/run/run_build_index.py) 新增 `--source_indexes_json` 和 `--metadata_only`。后者要求空或不存在的输出目录，沿用真实校验、预处理、lookup/tree/anchor/quality 元数据构建和输入保存流程，在导入 embedding 与索引构建模块前返回；它明确记录 `indexes_built: false`，不生成可用检索索引。

两个检索入口读取构建目录中保存的 registry，并传入 chunk/leaf reader；遇到 metadata-only 阶段会在调用 embedder 之前拒绝。完整检索入口模块仍有原来的运行时导入，不能据此声称它们导入时不加载模型库。这里没有执行完整 dense retrieval，也没有把 JEV 自动接入默认选择器。

旧格式记录继续使用原来的 reader 行为。独立审阅发现初版把 nested-meta 回退也用于旧格式；发布前已将回退限定到来源快照分支，并补上 chunk/leaf 的旧格式重载回归测试。

## 固定两文档结果

仍沿用此前封印的两篇文档、83 个 native Unit 和 251 个 model atom；直接使用已保存的原始 b0 与规则投影导出，不重跑 Refiner、投影器或导出器。每种模式各执行一次实际 metadata-only 构建，再从落盘 lookup 和来源 registry 重新读取。

| 保存模式 | chunk 数 | 检索文本与原文不同的 chunk | leaf 行数 | 检索文本与原文不同的 leaf | 恢复后完整请求一致 |
|---|---:|---:|---:|---:|---:|
| 原始 b0 | 83 | 81 | 251 | 249 | 83/83 |
| 规则投影 | 56 | 54 | 251 | 249 | 56/56 |
| 两模式合计 | 139 | 135 | 502 | 498 | 139/139 |

502 是相同 251 个 atom 在两种分块模式下的持久化行数，不是 502 个独立样本。表中比较的是当前检索规范化文本与原文，不是上一轮 source/legacy exporter 的配对比较；两种转换的计数不可混用。

原文快照文本全部保持一致，leaf 与其 owner chunk 的文档及 atom 包含关系均通过检查。保存的 registry 和三类输入文件均与对应输入字节一致，索引目录为空，构建摘要明确为 metadata-only。

JEV 请求只使用上一轮相同的人工身份检查问题及 offline endpoint/model 标识。token 计数器只接受与上一轮已独立验证的完整证据渲染字符串完全相同的输入，并返回对应旧计数；任何未见字符串立即失败。本轮没有重新读取 tokenizer，也没有产生新的预算测量，更没有 API 执行或历史响应缓存命中统计。

## 验证与边界

59 项定向测试在本轮合并运行中全部通过（0.80 秒）：来源保存/篡改与版本验证 43 项，真实 reader/CLI 往返 10 项，两个检索入口接线 6 项。实际 metadata-only CLI 合成测试和两文档检查都阻断模型/索引模块导入与 socket 连接。入口接线测试通过 AST 提取实际 `main` 并注入依赖，在 embedder sentinel 处停止；它覆盖入口逻辑，不覆盖模型导入副作用或检索算法。

固定运行在源文件、测试和输入哈希封印后执行，全部分母保留。独立复核只检查保存的输入/输出、来源几何、文本、哈希与请求，不重跑新流程或分词。具体绑定、输出摘要与复核范围见[机器结果](results/refiner_source_loader_20261004.json)。公开文件不含自然文档/单元 ID、正文、真实问题或凭据。

下一步继续零 API：检查候选选择与证据打包入口如何从已验证 lookup 恢复来源快照，并对最终 JEV／生成器使用的文本执行同一预算。当前 `chunk_aggregator` 复制规范化 `text` 和 `token_est` 到候选；`pack_evidence` 使用候选文本，而其计数函数优先采用已有 `token_est`。因此原文恢复必须通过 chunk lookup，旧 token 估计也不能直接视作最终来源请求的完整渲染预算。当前已解决“规范化后原文丢失”这一接入问题；最终选择器、生成器文本政策以及端到端效果仍须分别验证。
