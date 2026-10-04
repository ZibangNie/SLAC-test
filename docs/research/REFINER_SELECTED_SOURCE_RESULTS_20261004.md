# 选中证据的原文恢复与最终渲染预算检查

2026-10-04。已将实际 `PackedEvidenceItem` 与 `RetrievalCandidate` 接到此前验证过的 Refiner 来源快照和离线 JEV 请求接口。新增解析函数只按选中 ID 查询 lookup，校验身份与文本后恢复原文；不扫描全量记录，不读取分数，不改变排序或选择算法。

75 项相关合成测试通过，其中 16 项为本阶段新增。三个固定人工反例还确认：**旧逐段估计、JEV 的完整证据渲染、最终回答模型实际收到的证据块不能共用一个预算读数**。这是文本/预算契约的接入进展；新增 API、模型推理、权重、训练、tokenizer 和自然数据读取均为 0，没有新增准确率、回答质量或创新性结果。

## 已接通的选择接口

[`source_records.resolve_refiner_source_selection`](../../SLAC/retrieval/dataio/source_records.py) 接受当前 pack、candidate、chunk lookup、来源索引及显式文本政策，返回当前 pack 的不可变来源快照、候选快照和排序后的变化 ID。

- `exact`：选中对象必须已经具有完整原文。即使 lookup 文本已规范化，也接受此前恢复过的原文 pack。
- `reconstruct`：还允许当前已验证的 lookup 文本，恢复完整原文并记录变化 ID。
- 两者都拒绝第三种文本，例如摘要、截断或不再匹配的展示文本；也拒绝缺项、重复选中 ID、lookup key/对象/文档错配及损坏来源快照。

每条来源快照复用上一阶段的 registry、字符/atom 范围、原文及哈希验证。解析函数本身不排序、不裁剪、不选证据、不执行预算。返回的前两项直接交给现有 [`build_refiner_source_request`](../../SLAC/retrieval/decision/refiner_bridge.py)，继续使用其共同来源契约检查、完整渲染、预算、请求键和来源 receipt。没有新建另一套选择器、请求格式或缓存。

实际 reader → enrich → `pack_evidence` → resolver → JEV builder 的合成测试覆盖 standalone 与 plain_conditional。旧 pack 的规范化文本和估计保持原行为，JEV 使用恢复的完整原文；低估计不会绕过 JEV builder 对最终证据渲染的限制。所用计数器明确为 toy UTF-8 bytes，不是假定的服务商 token 计数。

## 三个不同的渲染对象

源码审查追踪到了实际发送路径：

1. [`conditional.render_evidence`](../../SLAC/retrieval/decision/conditional.py) 是 JEV 的证据预算文本，含来源头和分隔符，不含 query、问题契约或完整请求开销。
2. [`integration.prompt.builders`](../../SLAC/integration/prompt/builders.py) 生成 integration 的证据预览。这个 block 没有作为最终证据文本传入 LLM compiler。
3. [`llm.service.request_compiler`](../../SLAC/llm/service/request_compiler.py) 调用 [`render_evidence_block`](../../SLAC/llm/service/renderers.py)，把实际证据块追加到 provider messages。该模板包含说明、编号、ID、排名、路径等字段，与前两种不同。

三个反例使用固定人工输入，执行原有 packer、纯 renderer、请求 builder 和 provider payload compiler，没有调用 provider。直接断言 compiler 最后一条追加消息等于实际 LLM block，并且不等于 preview。以下都是 **toy UTF-8 字节数，不是真实模型 tokens**；旧字段 `token_est` 的数值也是人工设置的。

| 人工例子 | 提供的估计和 | 正文字节和 | JEV 完整证据 | Integration 预览 | 实际 LLM 证据块 | toy 限额 |
|---|---:|---:|---:|---:|---:|---:|
| A：人为低估正文 | 1 | 96 | 102 | 276 | 399 | 8 |
| B：正文计数准确、缺少头与分隔 | 11 | 11 | 25 | 273 | 436 | 11 |
| C：JEV 通过而生成模板更长 | 3 | 3 | 9 | 183 | 306 | 24 |

原 packer 在三例中均选入全部候选，摘要未改写。现有 JEV builder 注入完整字符串字节计数器后，拒绝 A/B、接受 C。B 的逐单元 JEV 渲染共 23 字节，合并渲染为 25 字节，另有 2 字节分隔符；即使单段计数准确，也不等于整包计数。C 说明通过 JEV 证据预算不能认证生成器预算。

这些人为反例用于反驳“旧估计可以认证完整渲染上限”，不能估计自然样本溢出率、真实 token 数或费用。实际 LLM evidence block 的读数也不包含 system、query、memory 等内容，仍不是完整模型上下文预算。这里采用显式人工字段映射来比较 renderer，没有运行完整 retrieval/reranker/generator 链路。

## 尚需落实的实际生成器接入

本轮没有修改默认 packer 或最终回答流程。已有代码仍存在三个明确接入点：

- [`integration.evidence.normalizers`](../../SLAC/integration/evidence/normalizers.py) 会给候选添加路径/编号、执行 `.strip()`，并优先保留旧 `token_est`；source 模式需要保留独立原文。
- [`integration.evidence.budgeter`](../../SLAC/integration/evidence/budgeter.py) 会再次按估计和进行选择。最终预算应计数选定顺序下的完整实际 evidence block；不能只修 retrieval 侧。
- 实际 LLM renderer 最后整体 `.strip()`，会裁去最后一条 passage 的尾部空白。其 metadata 还会渲染 `token_est`；精确整包计数应另存，不能回填此字段后继续沿用旧的渲染计数。

下一步继续零 API，复用实际 LLM renderer 和现有选择流程，先用人工输入验证原文保存与完整渲染预算，再检查编译前后的文本一致性。第一个机械接入固定 `retrieval_packed_evidence` 路线，因为 final integrator 也可能优先采用 reranker 产物；其他路线必须随后分别验证。JEV 和生成器应共享经过验证的来源内容，但各自的实际渲染与预算须明确区分。

## 验证记录

合并运行：`python -m pytest -q tests/research/test_refiner_source_selection.py tests/research/test_refiner_source_pack_contract.py tests/research/test_source_records.py tests/research/test_source_reader_pipeline.py tests/research/test_retrieval_source_entry.py`，**75 passed in 0.93s**。新增 14 项解析器测试及 2 项真实 pack 接线测试；另 59 项复核此前的来源保存、重读及入口行为。

独立复核只读新代码、测试与保存的人工诊断，检查源码/输入哈希、三类字符串的字节数及 compiler 渲染关系，没有重跑流程或访问模型。机器结果保存人工输入、实际读数、测试范围和证据边界：[结果 JSON](results/refiner_selected_source_20261004.json)。
