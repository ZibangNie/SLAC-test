# Refiner 权重与固定输入的兼容性核查

2026-10-04。**原论文模型暂未满足推理条件：已查范围未找到 `epoch_8.pt`，且文档记录的 64-token 输入上限与此前两文档存在冲突。** 本轮没有启动模型、训练、tokenizer 或 API；只检查文件名、已有聚合结果、源码与两个小模型的 safetensors 头部。没有扫描数据正文。

## 权重身份

| 对象 | 当前证据 | 能支持的判断 |
|---|---|---|
| 论文 Week2 Refiner | [使用文档](../../SLAC/refiner/refiner模型使用文档.txt) 和 pipeline 配置指向历史 Linux `week2_mixed_llmgold_x4_e3/checkpoints/epoch_8.pt` | 原仓库及迁移包的限定文件名检查未发现它；不代表其他磁盘或电脑上不存在 |
| 九月微诊断 `refiner.safetensors` | hidden128、1层、4头、window8、atom_dim1024、K6；固定128训练步 | `ProbeNetwork`，不是论文 hidden768、12头模型 |
| 同期 `boundary.safetensors` | 相同保存结构，训练目标是普通边界分类 | 不能代替 Refiner，参数总数相同也不代表有效训练容量或输入信息相同 |

两个微诊断文件均为 2,374,408 字节，头部 2,808 字节、30 个 F32 tensor、592,898 个参数。这里只验证了头部、shape、offset 和文件长度算术；没有读取 tensor payload、验证当前数值、加载模型或认证训练后权重身份。两者相同的 header SHA 说明结构相同，不能说明权重相同。

绑定的五份历史源码与当前源码逐字节一致，足以解释微诊断的结构。历史 `results.json` 的 `model_weights_sha256` 来自 BGE encoder 文件，不能当作训练后 Refiner 的 SHA。本次检查的微诊断 results、verification 和保存代码没有提供这两个输出 checkpoint 的历史 SHA；现在补算摘要也不能倒推历史身份。

## 固定两文档的长度门

直接复用[已发布聚合结果](results/native_refiner_granularity_20261004.json)，没有重读或重新分词 atom：

| 固定文档 | atom 数 | 最大实际 BGE tokens | 当前128上限 | 历史文档64上限 |
|---|---:|---:|---|---|
| 1 | 177 | 82 | 长度条件通过 | 至少一个 atom 超限 |
| 2 | 74 | 84 | 长度条件通过 | 至少一个 atom 超限 |

长度包含 special tokens。64 来自模型使用文档，尚未由真实 checkpoint 的 `train_args` 或训练日志确认；128 是当前默认值。因此结论是**按文档配置，两篇均不通过无截断输入条件**，不是已经复现了旧模型报错。没有计算超64的总数，也没有通过截断、重切分或放宽上限改变本轮固定输入。177 atoms 还超过微诊断训练样本的128上限；可变长度网络能接受张量，不代表训练分布或质量已获验证。

## 当前加载器的证据边界

[`load_model_from_ckpt`](../../SLAC/refiner/scripts/infer_bestofn.py) 用 `weights_only=True` 读取 checkpoint，但按命令行参数构造模型，不用 checkpoint 内的 `train_args` 恢复配置。虽然调用 `strict=False`，任一 missing/unexpected key 都会报错，形状冲突也不会被接受。生产 state 包含冻结的 atom backbone；微诊断仅保存 `doc.*` 和 `heads.*`，不能混载。

K、window 和 atom 长度等配置不能靠参数 shape 认证；成功加载也不足以证明推理等价。默认 pipeline 还有两轮 refinement、identity/greedy/sample 多候选以及逐轮规则投影，不能称为“一次固定前向”。当前入口先读完 JSONL 再应用 `max_docs`，若未来执行两文档检查，必须先构造固定小输入文件。

## 本轮决定

本次兼容门结束，论文模型的两文档推理保持**未执行**。路径已向用户询问；找回后仍须核对训练配置、encoder/tokenizer 身份、精度和解码设置，并先解决64/128长度差异。微诊断权重可以服务于明确另立范围的小实验，但本轮不将它替代为论文模型。

没有新的边界预测、RAG 收益或创新性结果。此前微诊断 legacy-dev 的 Refiner F1 0.8358、b0 0.8427、普通分类器 0.8786 仍是有标签谱系限制的开发性记录，不因本次结构核验升级为独立确认。真正尚未回答的问题仍是：在相同输入、有效容量、优化及选模预算下，编辑式 Refiner 是否优于以 b0 为条件的直接边界预测。

公开[机器记录](results/refiner_checkpoint_gate_20261004.json)仅包含配置、聚合值及来源绑定；权重和自然文本保持私有。此处关闭的是本次兼容性核查，不是停止全部研究，也不恢复旧900题付费计划。
