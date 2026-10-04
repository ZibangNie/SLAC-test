# 局部边界编辑会改变哪些模型输入

2026-10-05。当前路径不能把“固定 atom 文本”视为“固定检索与模型输入”。按[事前固定的四 atom 协议](LOCAL_EDIT_VISIBILITY_PROTOCOL_20261005.md)，只执行一次 split 和一次 merge 的合成见证：**leaf 正文与 ID 全部不变，但完整检索编码字符串分别有 3/4、4/4 改变。** 本轮 API、模型、tokenizer、训练和自然数据记录读取均为零，没有测向量、排序或答案。

## 合成见证的全部读数

baseline 为 atom 区间 [0,3)、[3,4)；split 为 [0,1)、[1,3)、[3,4)；merge 为 [0,4)。三者使用同样四个 atom、同样三个来源 unit 和原始 b0。执行真实 exporter、JSONL reader、enrichment 与 compose 函数，固定输入和源码先封印。

| 相对 baseline 的变化 | split | merge |
|---|---:|---:|
| leaf ID 改变 | 0/4 | 0/4 |
| leaf 正文改变 | 0/4 | 0/4 |
| leaf owner ID 改变 | 3/4 | 1/4 |
| leaf 路径改变 | 1/4 | 1/4 |
| leaf owner anchor 改变 | 3/4 | 4/4 |
| leaf 完整 encoder 字符串改变 | 3/4 | 4/4 |
| 按顺序串联的规范化全文相同 | 是 | 是 |
| 强制全选 chunks 后的完整证据字符串相同 | 否 | 否 |

这组全选证据是为检查 renderer 人为构造的输入，没有经过实际检索、重排或预算选择。它不能证明真实系统会返回这些包，也没有验证 token 预算。

更小的见证是 [3,4) 尾块：baseline 与 split 中正文、路径和编码字符串相同，但 chunk ID 从 `chunk_00001` 变成 `chunk_00002`；只将这个尾块送入 renderer，完整证据仍不同。完整 renderer 会暴露 chunk ID，不能用正文相等代替模型可见输入相等。本轮只比较 evidence block，不认证完整 HTTP 请求。

这些是特意构造的依赖关系见证，不代表自然样本中的发生率，更不证明改变后的向量、排序或回答一定不同。编码器截断等处理可能让不同字符串具有相同的有效输入。

## 从边界到索引：哪些可复用

[exporter](../../SLAC/refiner/pipeline/assemble/export_refined_chunks.py) 177–251、312–427 行按最大重叠来源 unit 选路径／父信息，按新输出序号生成 chunk ID，leaf 从其新 owner 继承这些字段。[enrichment](../../SLAC/retrieval/preprocess/anchor_fields.py) 78–127 行再计算路径与 owner anchor。

[leaf dense compose](../../SLAC/retrieval/index/build_leaf_dense.py) 12–30 行真正编码的是路径、owner anchor 与 leaf 正文；[chunk compose](../../SLAC/retrieval/index/build_chunk_dense.py) 12–25 行也包含路径、编号和 anchor。因而只有完整编码输入及 encoder／tokenizer、截断、pooling、归一化等合同匹配时，才可考虑复用向量。**Refiner 自身的 atom embedding 和这里的 retrieval embedding 不是同一输入合同。** 本轮没有实现增量向量缓存或测加速。

即使某个向量可复用，仍须重绑 ID map、owner 和邻接。[candidate aggregation](../../SLAC/retrieval/retrieve/chunk_aggregator.py) 20–57 行按当前 owner 组成候选，固定 leaf hits 也不会固定父块正文、支持 leaf 集合或分数聚合。词法检索还使用整个 chunk＋leaf 语料的统计；只改一处边界，也不能直接沿用旧 BM25 排名。

## 从候选到最终提示词：哪些实验会失去干预

[检索入口](../../SLAC/retrieval/run/run_retrieval_pipeline.py) 195–220 行将候选分别导出到 retrieval pack 和 reranker input。它不是“先选定 retrieval pack，再只重排这些项”。默认生产选包保留候选记录；此前研究脚本按相邻原文坐标合并为 run，是另一个显式的输出变换，不能把其性质套到生产路径。

[RerankerAdapter](../../SLAC/integration/adapters/reranker_adapter.py) 63–84 行优先读取传入或已存在的重排产物；[FinalIntegrator](../../SLAC/integration/orchestrator/final_integrator.py) 129–149 行又按配置优先选用 pack_bridge／reranked_candidates。若只修改上游边界，却继续指向旧产物，可能没有向最终模型施加预期的变化。这是缓存复用路径的静态事实，本轮没有证据说明过去某项实验已经因此污染。

新实验必须绑定当前问题、完整候选输入、评分配置和实际选用的产物；改变候选文本后不能继承旧 JEV 标签，正文未变也必须检查真实评分请求是否相同。最终输出比较应覆盖全部模型可见字段、顺序、渲染与预算。上下文相关判断更不能只按单块文本复用。

## 本轮实现、核验和研究决定

为了直接运行现有 compose 函数，将两个 dense builder 的后端导入延后到实际建索引时；compose 函数正文与此前提交一致。三个针对导入隔离和 builder 编码／保存调用顺序的测试通过，耗时 0.18 秒。builder 测试使用明确的假 encoder／FAISS 函数，不等于验证真实数值索引。合成脚本拒绝重模型与数值后端导入、网络连接及子进程。

索引与提示词路径分别完成源码审计；合成输出另由独立字符串／JSON 实现复核。来源、冻结哈希、完整合成字符串、全部计数和边界见[结果记录](results/local_edit_visibility_20261005.json)。没有新增自然问题或恢复此前两题的选包变体试验。

后续必须区分两种研究问题：检索前静态分块会连同编码、索引与候选集合一起改变；检索后固定来源域中的局部编辑可以单独检验查询条件的证据组织，但它是新的作用位置，不能冒充当前 Refiner 已有能力。下一步先为后一种写清最小可验证干预：固定候选曝光范围，使用稳定来源身份，标记仅元数据变化／正文变化／输出不变，并要求独立效用证据。若干预不能区别于已知粒度选择或仅改命名，不据此扩大评分或训练。此次仅消除了不公平复用和错误归因的前提，没有建立新方法效果或创新结论。
