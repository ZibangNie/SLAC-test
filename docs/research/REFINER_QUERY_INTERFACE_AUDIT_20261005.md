# Refiner 信息接口：当前分块模型不接收用户问题

2026-10-05。沿当前训练入口、文档推理入口与上传文档聊天链路逐段核对源码：**当前 Refiner 在文档侧产生边界，再由下游检索使用用户问题；不能直接把逐问题的 JEV 监督称为查询感知的 Refiner。** 本轮只读源码与配置，未读取自然数据记录、key 或 checkpoint，未实例化模型、tokenizer，也没有 API 调用或训练。审计针对下述明确路径，不声称穷尽所有历史脚本或外部集成；允许自定义的 `--infer_script` 和实际部署使用的 checkpoint／配置也未认证。

## 训练中实际可见的信息

| 路径 | 已核对的数据流 |
|---|---|
| [Dataset](../../SLAC/refiner/slac_refiner/datasets/refiner_dataset.py)，80–136 行 | 从原记录投影 atom 文本、b0、初始边界位置、编辑／插入标签、b_gold 和数值 sample_weight。返回字段没有问题。原 JSON 虽存于 Dataset 内存，任意额外字段不会因此进入 batch。 |
| [collate](../../SLAC/refiner/slac_refiner/datasets/collate.py)，30–73 行 | 对上述张量进行 padding，保留文档身份和 atom 文本，没有问题编码或问题字段透传。 |
| [训练循环](../../SLAC/refiner/scripts/train_loop.py)，149–158、299–312 行 | 用该 Dataset 和 collate 构建 loader，然后直接 model(batch)、criterion(outputs, batch)、反向传播。 |
| [BoundaryRefinerModel.forward](../../SLAC/refiner/slac_refiner/models/refiner.py)，107–148 行 | 文本路径以 atom 文本编码文档，heads 接收文档隐藏态、g0_positions 和 atom mask；没有问题通道。缓存 embedding 分支也没有独立问题字段，但只检查形状、数值及长度，不能认证外部 embedding 的语义来源。 |
| [heads](../../SLAC/refiner/slac_refiner/models/heads.py)，116–150 行 | `query_proj` 的输入是初始边界的 `boundary_repr`；这是指针计算的 query，不是用户自然语言问题。 |
| [loss](../../SLAC/refiner/slac_refiner/models/losses.py)，68–112、147–181 行 | 插入标签使用 BCE，已有边界动作使用交叉熵，可加编辑成本正则和样本权重。该损失未读取 JEV 偏好、答案或检索效用。b_gold 在 Dataset 验证标签回放，并在训练脚本的评估中作为边界参考。 |

`sample_weight_field` 可从原记录选择一个数值权重；它会改变优化贡献，不会为部署模型增加问题输入。训练 labels 也可能未来由问题相关教师产生，但这不同于学生能看到当前问题。

## 推理、建库与问题的先后关系

[infer_bestofn.py](../../SLAC/refiner/scripts/infer_bestofn.py) 的 `build_single_doc_batch`（322–333 行）仅用 atoms 与当前边界构造 `atoms_text`、`g0_positions`；`forward_one_doc`（337–350 行）将其送入模型。不同采样候选或随机状态可造成不同结果，本审计没有执行逐位相同的运行测试；关键事实是问题没有进入这条计算路径。

上传文档聊天也遵循这个分工：

1. [server_api.py](../../SLAC/openwebui_bridge/server_api.py) 223–227 行把问题传给 `prepare_query_runs`。
2. [upload_ingest_service.py](../../SLAC/openwebui_bridge/upload_ingest/upload_ingest_service.py) 330 行先确保文档资产存在；254–268 行的 Refiner 子进程参数只有文档、配置和输出选项，没有问题。317–320 行在无需重建时复用现有资产。
3. 同文件 339–358 行才把问题写入查询文件并交给检索。这里“聊天触发首次建库”不等于“问题条件化的分块”。
4. [run_build_index.py](../../SLAC/retrieval/run/run_build_index.py) 64–83 行读取已导出的 chunks/leaves；[run_retrieval_pipeline.py](../../SLAC/retrieval/run/run_retrieval_pipeline.py) 113–121 行读取问题，197–220 行打包并导出候选、重排输入与证据。

另有确实接收 query 的 [refiner_bridge.py](../../SLAC/retrieval/decision/refiner_bridge.py)：164–204 行针对**已有来源片段**构造条件判断请求，未调用边界模型。它提供查询相关的下游 JEV 请求接口，不改变上游 Refiner 的输入事实。

## 对 JEV 监督方案的实质限制

在固定权重、预处理、atom／初始边界输入和推理配置时，当前模型没有依据未提供的问题作出不同决策。若同一可见输入对应两个问题，而教师希望相反的边界动作，单纯换标签不能让学生按问题区分它们。这是接口的信息限制；本轮没有检查自然样本是否实际出现这种冲突，也没有量化冲突率。人为将问题拼入 atom、修改初始边界或提供问题相关 embedding 会改变前提，均不是当前已追踪路径的行为。

后续方案必须先明确选择哪一种目标：

| 目标 | 需要补足的定义 | 当前尚未成立的部分 |
|---|---|---|
| 文档侧静态分块 | 在明确的查询分布上聚合局部编辑的独立下游效用，保留相反偏好和无效编辑。 | 没有可用的编辑效用标签、查询分布或聚合验证；不能称为逐问题适应。 |
| 检索后的问题条件局部编辑 | 明确输入问题、固定候选来源域和操作时机，追踪编辑如何改变最终完整证据。 | 当前模型接口和训练目标都未实现这一机制；增加 query 输入本身也不证明新颖性或收益。 |

[相关工作准入](UTILITY_TRAINED_CHUNKING_GATE_20261005.md)仍然适用。已有 117 个单块相关性判断不能直接当作 split／merge 效用标签；对相同最终证据文本也不能因为内部边界不同而制造效用差异。此次结果关闭的是“原接口换 JEV 标签就成为 query-aware Refiner”的假设，没有否定其他形式的监督，也没有证明新框架有效。

下一步只沿局部 split／merge 到最终证据的路径定义可辨识干预：先区分是否改变候选可见文本、是否改变最终输出，以及哪里需要新的独立效用反馈。先完成零模型可行性检查，再决定是否值得小样本验证，不启动批量评分或训练。源码绑定与两个分工审计的摘要见[接口核查记录](results/refiner_query_interface_20261005.json)。
