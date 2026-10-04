# 原文与完整证据预算接入实际生成请求

2026-10-04。已把验证过的 Refiner 来源接入 `FinalIntegrator`，在来源模式下恢复原文、按实际生成器 evidence block 选择证据，并在编译请求时校验文本与预算绑定。默认流程保持原行为，来源模式需显式提供 lookup、来源索引、文本政策和计数器。

**99 项相关合成测试通过，三个固定人工示例完成真实编排与请求编译。** 本轮 API、自然数据读取、模型推理、tokenizer、权重、密钥读取和训练均为 0。示例使用 compile-only adapter，返回明确标记的假回答；没有测量生成质量或证明创新性。

## 完成的接入

上一轮[实际渲染诊断](REFINER_SELECTED_SOURCE_RESULTS_20261004.md)发现，integration 预览与最终 LLM 模板不同，逐段 `token_est` 又不能认证带 metadata 的完整证据块。本轮修改真实发送路径：

- [`source_mode`](../../SLAC/integration/evidence/source_mode.py) 按当前候选 ID 查找来源，复用已有原文恢复与校验。`exact` 只接受完整原文；`reconstruct` 还接受当前 lookup 的对应规范化文本，恢复原文并记录变化。摘要、截断、过期文本、错配 ID 及冲突文本别名均拒绝。此处逐条处理当前输入候选，不遍历全库。
- [`FinalIntegrator`](../../SLAC/integration/orchestrator/final_integrator.py) 复用原选择器，通过 `pack_cost` 对最终顺序的整包实际渲染计数。既有 direct-first 配额、remaining 阶段、去重和稳定顺序保持不变；第一阶段拒绝的项仍可能在第二阶段被重试。未引入新搜索算法或最优性承诺。
- [`renderer`](../../SLAC/llm/service/renderers.py) 的来源模式保留 passage 中完整的空白、CRLF、tab 和 Unicode。共享字段映射使来源模式预览等于实际编译的 evidence block。旧逐段估计仍作为 metadata 保留，不回填整包读数，避免计数后又改变渲染内容。
- [`compiler`](../../SLAC/llm/service/request_compiler.py) 要求来源政策和六字段预算 receipt 配对，校验最终字符串 SHA-256、版本及限额。改写正文、改写已渲染的 metadata、删除政策或降级政策都会被拒绝。独立代码审阅发现并修复了政策丢失时可能退回旧 renderer 的缺口。

选中来源在生成请求构造前再次校验。空集使用空字符串 receipt，不追加空证据消息。普通 `FinalIntegrator` 在编译之后仍会调用其 adapter；来源模式本身不是离线开关，本轮显式注入了 compile-only adapter。

## 固定人工示例

公开脚本 [`probe_source_generation.py`](probe_source_generation.py) 在执行前保存 33 项代码/测试文件的 SHA-256 和三个固定模式，结束时重核源码。输入复用测试中的两条人工原文，包括全角字符、前后空白与 CRLF；先执行真实 enrich 和 retrieval packer，再提供给实际 integration。没有查询自然数据或加载检索模型。

脚本固定使用 `retrieval_packed_evidence`，逐例保存完整请求和编译后的 payload，并检查请求序列化后重编译一致。**下表单位为 toy UTF-8 字节，不是模型 token；旧 `token_est` 人为设为每项 1。**

| 模式 | toy 限额 | 选中项数 | 旧估计和 | 实际 LLM 证据块字节 | 完整原文项数 |
|---|---:|---:|---:|---:|---:|
| 原默认流程 | 512 | 2 | 2 | 674 | 0 |
| 来源模式，宽预算 | 4096 | 2 | 2 | 673 | 2 |
| 来源模式，窄预算 | 512 | 1 | 1 | 428 | 1 |

默认流程在这个人为低估输入上选入两项，实际证据块大于 toy 限额。来源模式在宽预算下保留两条完整原文，在窄预算下保留第一项、满足整块限额。这证明固定示例中计数对象与最终发送对象一致；少选一项并不等于答案更好，也不能据此估计自然样本的溢出率或费用。

完整来源请求另做四次固定改写：正文、`token_est`、删除 source policy、降为 legacy policy；编译器全部拒绝。三个模式的序列化重编译检查均通过。脚本没有改变问题或选择示例来追求质量结果。

## 验证与边界

合并执行四个定向 suite：`test_integration_source_mode.py`、`test_integration_render_budget.py`、`test_source_evidence_renderer.py`、`test_final_integrator_source.py`，结果 **99 passed in 0.35s**。覆盖来源恢复、别名冲突、非加性/非单调整包成本、两阶段稳定顺序、空集/限额/重复项、渲染和预算绑定，以及实际编排的序列化与文本漂移。

编排测试还用人工 artifact 验证了 `reranker_pack_bridge` 优先路径，未运行真实 reranker。三个公开示例固定走 retrieval pack 路线，不能称为完整自然检索—重排—回答实验。

独立复核只读取保存的人工输出与源码，复算源码和结果哈希、原文身份、最终块字节数及 receipt，核对预览与编译字符串。序列化一致和拒绝行为以保存的运行记录及源码为证据，复核者没有重新运行流程、compiler、renderer 或测试。[机器结果](results/source_generation_20261004.json)保留人工输入、编译结果、固定模式、源码绑定与复核范围。

首次 `run-01` 的计数一致，但复核发现探测脚本的负例测试共享了成功请求中的 `options` 字典，把已保存的 `source_4096` policy 改成 legacy。该次记录被标记为不一致，保留原输出且不计入有效示例。仅修正脚本的深拷贝隔离，并加入负例之后对全部保存请求的重编译检查；生产代码、输入、预算和预期均未改。最终采用同样三个固定模式的 `run-02`。这次修正没有增加数据或 API 规模。

预算 receipt 绑定的是受信任调用者给出的计数和确切字符串。它不能证明计数器正确、认证外部调用者或代替来源索引校验。完整 evidence block 也不包含 system、query、memory、provider framing 或输出，不等于模型总上下文预算。生产计数需显式注入与生成器匹配的真实计数器；本轮没有验证任何服务商 tokenizer。

JEV 与最终生成器现在可以使用同一份已验证原文，但各自 renderer 与预算仍独立。此次接入修复了可靠比较方法前的文本和预算条件，本身不构成新的选择机制或 RAG 收益。后续继续零 API，优先复用固定少量已曝光案例，检查原文恢复与真实模板开销是否改变此前交换的可行性；文本改变时不可搬用旧 JEV 判断，也不直接扩样或恢复旧的大规模付费计划。

此前[六例自然交换](NATURAL_EXCHANGE_JEV_RESULTS_20261004.md)中唯一共同正例仍被 JEV 的 loss 判断拒绝，预算接线并未修复这一语义问题。下一步限于这些既有例子的 12 个冻结 S/T 包：先核所需身份及文本层级，不能假定它们与 Refiner 两文档样本相同。没有现成的精确来源映射，就只做冻结文本的渲染诊断。保持原问题、成员和文本，先核对原预算所用计数器及单位，再比较实际模板开销；代理 token 计数不能冒充生成器真实 tokens。若可行性均未变化，应结束预算解释分支，回到 loss 判断假设，避免继续增加无助于研究判断的接线工作。

配置与调用方式见 [integration 文档](../../SLAC/integration/README.md)。
