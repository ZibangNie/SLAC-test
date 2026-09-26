# 本次超时的恢复边界

2026-09-27，扩展support的第165次尝试在JEV `/api/alpha/decisions` 路由发生60秒传输超时。请求文件和预留已保存；没有响应文件、generation ID或费用回执。164次成功、1次未知、143次未发出的审计见[中断聚合](results/qasper_extended_interruption_20260927.json)。其中accounting沿用旧字段`questions`，含义是support判断项数，**不是Qasper问题数**；主实验分母仍是77题。

## 已核对的官方能力

- [Generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)可按generation ID查询费用和状态；[generation content](https://openrouter.ai/docs/api/api-reference/generations/list-generation-content)可按ID查询已存内容。二者都需要ID。本次没有这个ID；文档也不足以证明alpha decisions响应能够完整由content接口恢复。
- [Activity API](https://openrouter.ai/docs/api/api-reference/analytics/get-user-activity)需要management key，并按已完成UTC日期和endpoint聚合。这个接口不是逐请求响应恢复入口，不能仅按时间/金额把某条记录断言为本次请求。当前普通推理key不转换用途，不寻找其他凭证。
- [Response caching](https://openrouter.ai/docs/guides/features/response-caching)需要事前启用，且明确列出的支持入口没有alpha decisions。并发cache miss还可能各自收费。因此不能把现在重发视为免费、幂等或读取旧结果。此次请求没有启用这项缓存。

上述为公开文档核对，无新增推理调用。另一次直接读取公开OpenAPI返回HTTP 403，未绕过；这里没有基于新schema声称存在未记录的恢复入口。

## 当前处理及后续改进

本次保持不自动重试，不替换模型/账号，不减少分母，不把未知收费记为零。成功结果、失败尝试及全部预留原样保留；完整证据比较和依赖它的答案阶段仍未完成。可以先完成独立的本地reranker和检索实验。

后续执行器可在收到响应头时立即保存白名单回执字段（generation/request ID、HTTP状态、时间），再读取正文；不能保存Authorization、全部headers或任意exception文本。这只能改善“已收到headers后正文读取失败”的可追踪性，不能保证headers前超时可恢复。服务商关于该alpha入口的幂等或回执查询能力需要明确核实。

若以后决定重新执行不确定请求，应使用新的、明确记载该决定的恢复协议，保留原尝试及未知费用，重新计入全部剩余预留并绑定原payload与模型；不能修改旧ledger使其看似从未发生，也不能只保留较好的响应。本夜冻结协议没有自动执行这种恢复。
