# 固定缓存上的边界模型小对照：执行前协议

2026-10-04。目的仅是检查[新b0条件基线](DIRECT_BOUNDARY_BASELINE_20261004.md)与编辑式微模型在相同小缓存和优化预算下的表现。**固定原16个训练、8个legacy-dev样本；2个arm、3个seed、每次128步。** 不扩样、不重新编码、不使用API，不加载旧训练权重。该批样本与结果已用于开发，不是独立质量确认。

## 缓存准入与限制

仅使用 `probe-02-final/selected_train.jsonl`、`selected_dev.jsonl` 和4,278,064字节的 `features.safetensors`。两个selected文件的当前SHA与历史记录一致。缓存头部为24个F32 `[N,1024]` tensor，train/dev合计549/495 atoms；必须按显式 `train_i/dev_i` 数字索引取值，不能按词典序把 `train_10` 接到第2行。

历史保存代码按同一dataset index写入和读取缓存，但本次检查的旧记录没有features内容摘要或逐tensor行ID。执行前保存当前完整cache SHA，并检查每行长度、dtype、有限性、标签replay和来源字段；这只能冻结本轮使用的当前缓存，不能倒推认证历史BGE输出或排除同shape内容曾被置换。此限制保留，不临时重编码或另选样本。

只解析这两个已选小文件，不访问上游全量数据、test split、密钥或论文checkpoint。两份selected文件的记录都来自祖先train；legacy-dev沿用旧命名。语义标签与family独立性仍未获认证。

## 固定执行

| 项目 | 设置 |
|---|---|
| Arms | `edit`、`direct_seed`；按此顺序执行，全部保留 |
| Seeds | 13、29、47；每个seed先生成共同DocEncoder初值，两arm拷贝独立state并核对摘要与不共享storage |
| Head初始化 | edit为seed+1000，direct为seed+2000；不同形状不宣称逐参数相同 |
| Context | atom_dim1024、hidden128、1层、4头、window8、dropout0；双方独立可训练，训练后h可不同 |
| Direct head | K6、MLP width37；595,756整体参数，对比edit592,898，仅容量近似 |
| 优化 | CPU、4threads、确定性算法；AdamW lr0.002、weight_decay0.01、batch1；每arm128updates、全部参数clip norm1 |
| 样本顺序 | 每seed从0..15开始，每16步原地shuffle同一列表，共8轮；提前生成128索引，两arm复用 |
| 样本权重 | 全部1，不比较confidence权重 |
| Edit损失 | residual INSERT BCE + edit CE，两项系数1、pos_weight1、cost regularizer0；分别按有效gap/seed均值，不额外平均两个分量 |
| Direct损失 | 最终b_gold的有效gap平均BCE；不能与edit绝对loss跨任务比较 |
| 时间边界 | 外层180秒硬终止，worker170秒软截止用于留存；失败/超时不自动重启、续跑或扩大预算 |

两arm共享当前冻结embedding和context初始数值，不共享可变context实例。基线仍未复现编辑softmax邻居依赖与全局DP；共同输入、步数及近似参数量不消除这些架构差异。

## 评价与留存

初始化和第128步结束时，分别保存全部train/dev的预测：`3 seeds × 2 arms × 2 phases × 24 docs = 288`行。没有中途按dev选checkpoint、早停或试参数。共768次自然缓存上的更新；没有额外自然样本训练。

原始与投影后预测分别报告。Edit沿用微诊断解码：K6、插入阈值0.5、min_sep0、三项编辑成本0；direct以sigmoid≥0.5预测。共同projector：max atoms64、max chars100000、max BGE tokens512，三个minimum均1、strict=True。用原atom文本视图，完整字符串计数、包含special tokens、禁用截断。仅加载本地tokenizer，不加载BGE模型。另保存24行不经模型的b0原始/同规则投影控制。

沿用既有逐文档boundary F1定义，包括预测和目标同时为空时记0，避免事后换指标。主读数为每个seed的**legacy-dev projected macro F1及direct−edit差值**，并报告三seed均值；完整保留raw、train、initial、投影改变数、每个文档预测和所有loss history。三个seed、八个开发样本不支持总体显著性或RAG收益推断，不增加bootstrap显著性包装。

若direct相当或更好，不据本轮主张编辑架构优势；若edit更好，也只作为这批弱参考缓存的有限优化信号。raw/projected或seed方向不一致时如实报告。结束即停止本次六个运行，不在同批结果上调参寻找胜出。

执行前绑定[机器协议](results/cached_boundary_protocol_20261004.json)、[runner](run_cached_boundary_comparison.py)、依赖源码、当前缓存及tokenizer文件。外层一次性执行器和完整原始输出留在私有阶段目录；公开只含协议、匿名聚合、hash与边界。若失败，保留失败和缺失结果，不能将已有部分当成完整六次比较。
