# SLAC × JEV 研究记录

从[当前研究判断](RESEARCH_STATUS_20260927.md)开始阅读。它区分已完成的开发结果、失败或未完成的实验，以及仍待验证的创新主张。[夜间执行记录](OVERNIGHT_RESEARCH_20260927.md)保留准备、失败和后续完成的历史；不要根据旧段落重复启动实验。

夜间阶段之后，已完成[确认实验元数据清单](CONFIRMATION_METADATA_RESULTS_20260927.md)：281 篇 validation 全保留去向记录，33 篇隔离，248 篇形成操作性候选，独立算法核验通过。它们尚不是获认证的独立测试集。新的[五组分析协议](CONFIRMATION_ANALYSIS_PROTOCOL_20260927.md)及[数值退化修订 V2](CONFIRMATION_ANALYSIS_AMENDMENT_20260927.md)在新结果出现前固定；后续以 [V2 机器协议](results/qasper_confirmation_protocol_v2_20260927.json)为当前分析入口。本次元数据阶段未打开 QA 或发送 API 请求；历史解析记录见清单报告。

当前所有质量结果都是开发性证据。原77题JEV扩展经历一次未知超时，随后按事先公开的操作修订完成全部support，并通过完整审计和独立240区间复算；主答案六组462个预测也已完整审计和独立60区间复算。旧失败及未知费用仍保留。原始问答、文档、逐题输出、API请求响应、凭据及权重均不在公开结果目录中。

## 已完成结果的入口

| 研究问题 | 执行前协议或定义 | 完整结果及边界 |
|---|---|---|
| Qasper原生评价与初始基线是否正确 | [初版可证伪协议](EXPERIMENT_PROTOCOL_DRAFT.md) | [104题开发基线](QASPER_DEVELOPMENT_BASELINES_20260926.md)，不与后续77题混合 |
| 关系共享和JEV后端在小pilot的作用 | [15题pilot锁定协议](RELATION_PILOT_PROTOCOL_20260926.md) | [执行记录](RELATION_PILOT_EXECUTION_20260926.md)、[共享机制诊断](RELATION_MECHANISM_DIAGNOSIS_20260926.md) |
| 同库的给定论文与跨库背景 | [原生chunk桥接](NATIVE_CHUNK_BRIDGE_PROTOCOL_20260927.md) | [跨库桥接](CORPUS_BRIDGE_RESULTS_20260927.md)，query-only跨库仅作压力诊断 |
| 词面检索、常规重排和层级候选 | [离线基线协议](OFFLINE_BASELINES_PROTOCOL_20260927.md) | [BM25](LEXICAL_BASELINE_RESULTS_20260927.md)、[BGE reranker](RERANKER_BASELINE_RESULTS_20260927.md)、[native双索引](NATIVE_DUAL_INDEX_RESULTS_20260927.md) |
| 固定候选下改变入选优先级 | [CPU顺序诊断与固定定义](NATIVE_OWNER_ORDER_RESULTS_20260927.md) | 同文档含全部范围、负结果和128个描述性区间 |
| 证据指标是否转化为回答 | [六组答案协议](LOCAL_BASELINE_ANSWER_PROTOCOL_20260927.md) | [完整六组答案](LOCAL_BASELINE_ANSWER_RESULTS_20260927.md)、[真实请求资源](LOCAL_ANSWER_RESOURCES_20260927.md) |
| 两个已有顺序控制的答案表现 | [小范围继承响应协议](OWNER_ORDER_ANSWER_PROTOCOL_20260927.md) | [完整五组对照](OWNER_ORDER_ANSWER_RESULTS_20260927.md)，恢复接近dense而非超过dense |
| JEV粗标签、原始分数与一般模型判断 | [操作恢复修订](PRIMARY_SUPPORT_RECOVERY_AMENDMENT_20260927.md) | [完整77题support](PRIMARY_SUPPORT_RESULTS_20260927.md)，原主k=3及全部k=1/2、240区间；[后续答案冻结计划](RECOVERED_PRIMARY_ANSWER_PROTOCOL_20260927.md) |
| 主证据选择能否改善最终回答 | [完整主答案冻结计划](RECOVERED_PRIMARY_ANSWER_PROTOCOL_20260927.md) | [全部六组主答案](PRIMARY_ANSWER_RESULTS_20260927.md)，相对dense有正向开发信号，score相对粗标签的答案区间仍跨零 |
| 主方法相对强BGE基线的答案表现 | [明确事后的四组比较协议](POSTHOC_RERANKER_ANSWER_COMPARISON_20260927.md) | [完整事后答案比较](POSTHOC_RERANKER_ANSWER_RESULTS_20260927.md)，score双权重区间为正，粗标签双区间跨零；全部16区间保留 |
| 关系规则能否实际改变证据包 | [固定501种掩码的CPU协议](RELATION_OPPORTUNITY_PROTOCOL_20260927.md) | [全部501种组合结果](RELATION_OPPORTUNITY_RESULTS_20260927.md)：17/77题存在包变化机会，独立实现全量核验通过；不代表质量改善 |
| 能否保持全部选包结果并省去无影响判断 | [按需编译固定协议](RELATION_DEMAND_COMPILATION_PROTOCOL_20260927.md)、[对照与经典方法边界](RELATION_DEMAND_DESIGN_REVIEW_20260927.md) | [完整按需编译结果](RELATION_DEMAND_COMPILATION_RESULTS_20260927.md)：101条eligible唯一边中19条必要，全部501赋值及23,915缓存路径精确等价；未显示空缓存自适应或跨题必要边复用收益 |
| 关系规则全部可达包的回答是否不同 | [完整可达包回答协议](RELATION_PACK_ANSWER_PROTOCOL_20260927.md) | [全部可达包答案结果](RELATION_PACK_ANSWER_RESULTS_20260927.md)：96包完整覆盖，14新请求全成功；全邻接相对I的答案双区间跨零，完整16区间及观察极值保留 |
| 固定候选在同预算内能达到多高 | [精确候选oracle协议](CANDIDATE_ORACLE_PROTOCOL_20260927.md) | [完整六组上界](CANDIDATE_ORACLE_RESULTS_20260927.md)与[图](results/qasper_candidate_oracle_20260927.svg)，gold参与选择，不能部署或当答案收益 |
| 下一批数据是否存在文档重叠 | [249篇固定文档筛查定义及报告](REMAINING_VALIDATION_SCREEN_20260927.md) | 8批完整，1个标记经[文本复核](VALIDATION_OVERLAP_REVIEW_20260927.md)及[官方来源核对](VALIDATION_SOURCE_RELATION_REVIEW_20260927.md)发现明确材料沿用关系；历史筛查未排除样本，后续[新元数据政策](CONFIRMATION_METADATA_RESULTS_20260927.md)已形成248篇操作性候选，尚未准入 |

各结果报告链接对应完整精度的公开聚合JSON及图。JSON保留固定方法、全部预定比较、双权重区间、费用和来源hash；没有为了只展示正结果删去失败或下降项。

## 创新判断与继续工作的依据

[相关工作缺口](RELATED_WORK_GAPS_20260927.md)、[机制来源核对](MECHANISM_RESEARCH_DIRECTION_20260927.md)及[近期ETS与预算控制文献](RECENT_RELATED_WORK_20260927.md)说明了层级展平、上下文过滤、普通重排和依赖选择已有的研究。JEV发布时间本身不是方法创新。[论文工作提纲](PAPER_WORKING_OUTLINE_20260927.md)整理可检验贡献与尚缺证据；[英文论文工作稿](MANUSCRIPT_DEVELOPMENT_DRAFT_20260927.md)将完整开发结果整理为方法、实验及讨论，尚不是可投稿稿件。

[语义算子优化补充检索](SEMANTIC_OPERATOR_RELATED_WORK_20260927.md)核对LOTUS、Palimpzest、Larch与ScaleDoc；廉价判断、缓存或自适应调用本身已有先例，有限真值表编译尚非可扩展系统贡献。

[条件精度规划](CONDITIONAL_PRECISION_PLANNING_20260927.md)保留四组对照的全部72个半宽、48个整数情景，说明独立family数及假设标准差的影响。它不计算功效，不决定样本准入，也不证明现有或未来样本足量。

当前优先任务是冻结分数选择方法并完成独立family准入，再对强排序基线确认开发信号。额外候选与关系共享仍是待检验问题；当前邻接依赖规则未得到语义内容增益支持。Refiner标签的来源与跨split问题尚未获得正式训练准入。任何未见family确认都应先完成曝光/版本/近重复审核和方法锁定；不能继续在已看到结果的77题上调参后称独立确认。

## 重放与运行记录

本地完整审计依赖忽略目录`artifacts/research-foundation/`中的封印输入、响应、账本和模型文件。公开聚合hash是来源核对点；仅有Git代码和聚合JSON不等于已能重建不公开的响应或全部数据。

研究脚本和测试以LF字节保存；公开JSON/SVG使用`.gitattributes`的`-text`保留各自封印字节，避免Windows/Git换行转换改变hash。早期部分生成文件是CRLF，不能只因数值相同就当成同一字节来源；[公开字节核对](PUBLIC_RESULT_BYTES_20260927.md)记录了此次修正。已有生产模块的换行及旧封印没有改写；迁移旧artifact时仍应保留绑定的原始字节与路径映射。

优先使用各阶段的只读`audit`或分析入口核对保存结果。`plan`与`run`记录的是特定冻结实验；已存在的目录、注册和失败账本不可覆盖或复用为新尝试。复现实验需要重新明确数据来源、方法、模型实际版本、预算和时间窗口，再建立独立计划，不能简单重启历史命令。

当前自动推进与费用状态在本地`artifacts/research-foundation/overnight-20260927/state.json`；科学source、plan、run和已完成receipt保持不变。费用的已知小计、未知请求和保守预留是三个不同字段，不能互换。

[完整原域置换结果](RELATION_PLACEBO_DEMAND_RESULTS_20260927.md)已独立验证：76题两函数相同，仅1题可能不同，4/501赋值改变输出；共同需求20条；后续实际执行已完整完成，见下方报告。

[直接上下界选择结果](RELATION_LAZY_BOUNDS_RESULTS_20260927.md)也已完整独验：运行策略不读取真值表，501空cache与23915全known-subset路径精确输出相同；共读取326/9682次，保留22/70次非必要读取。两题随赋值读取1–2条，但无实际跨题共享调度、API费用或质量收益结论。

[20条实际关系执行协议](RELATION_CONTENT_EXECUTION_PROTOCOL_20260927.md)和[完整冻结计划](results/qasper_relation_content_protocol_20260927.json)已通过33项合成测试及独立源码/计划审查。全部七组方法共539条记录、24区间预先固定；20单例总预留$0.100，所有答案精确继承，不新增生成。源码和计划先发布后单次执行，20条均成功；[完整结果](RELATION_CONTENT_EXECUTION_RESULTS_20260927.md)及539条记录、24区间已独立复核。内容组与placebo全部77题证据包相同，当前规则不支持语义内容增益。

[Activation-only 对照协议](RELATION_ACTIVATION_CONTROL_PROTOCOL_20260927.md)及[冻结计划](results/qasper_relation_activation_protocol_20260927.json)已完成44项合成测试和独立审查。它在同一501/23915完整路径上分开检查激活过滤与区间认证的省读；当前最多2读来自邻接结构与k=3。[完整结果](RELATION_ACTIVATION_CONTROL_RESULTS_20260927.md)现已完成全量回放和独立验证：空cache的读取为2166→352→326，全known-subset为76390→9760→9682；区间证书额外减少26/78次，大部分省读来自激活过滤。

[晨间交接与下一步决定](MORNING_RESEARCH_HANDOFF_20260927.md)汇总正负证据、最后的归因控制、完整费用和独立确认仍需满足的条件。
