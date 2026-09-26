# 下一轮关系内容实验：固定候选、先检验机制机会

**状态：草稿，未冻结、未执行、未准入付费。** 2026-09-27。本文件只提出下一项开发实验；不修改旧协议或允许增加本夜预算。已读取完整 support 审计及既有 native/order/oracle 产物的必要字段，没有读取运行中的主答案、QA sidecar 或新增 QA，没有产生关系标签或回放本提案的 selector。

建议顺序是：**先完成无 gold 的 501 个关系 mask 机会检查，再决定未来是否值得支付静态判断和答案费用。** 当前预算不足以准入下面完整的静态判断计划。这里的“预算不足”指本夜协议的保守预留门禁，不是 OpenRouter 账户余额不足。

## 1. 只检验一个窄问题

对已固定的 query-unit support 和候选池，如果已选中单元 B，那么优先补入“解释 B 所必需的前一单元 A”，能否比独立 support 排序、全部邻接和匹配的随机标签改善最终答案？

本轮关系是 **query-independent 的局部解释依赖**，记作 `B depends on A`；选择时的触发方向是 **已选 B → 提升待选 A**。只考虑原文紧邻的 `A, B`。沿用旧静态定义：A 提供 B 所需的未完句/列表上下文、局部定义、显式引用或标题作用域；仅主题相近不成立。`independent` 表示 B 可独立解释；可见文本不足则 `unknown`。判断只接收完整 A/B 文本与其原 ID，不接收 query、support、rank、gold、答案或 oracle 选择。原 prompt 与标签定义见 [client](openrouter_decision_client.py)。

这既不是“二者联合回答该问题才有用”的 query-dependent evidence complementarity，也不是逻辑蕴含、同主题、两跳事实链或任意跨段依赖。不要把它们混成一个标签。本轮暂不引入 query-conditioned 判断：先用已有任务定义排除最简单机制无效的情况。如果随后研究 query-dependent 关系，必须另立 prompt、query-included cache identity 和成本计划，不能继续称为静态文档关系。

固定候选不会改变 chunk、index 或 retrieval；本轮只能检验**关系参与选择**，不能称为跨阶段 sharing。相比 [ETS 等既有依赖选择工作](RECENT_RELATED_WORK_20260927.md)，即使结果为正，仍未证明方法新颖性或同一关系跨阶段复用的增量。

## 2. 数据与任务覆盖：为何暂不换 owner 池

全程保留原 **77 questions / 24 families**，全部为已暴露开发数据；不因 gold、oracle gap、空参考、figure、unanswerable、support 标签或关系机会而删题。候选为旧准备产物的 top-8 dense seeds 加 native 邻居、最多 16 单元，与 native `given_document / leaf_direct` 的候选身份逐题相同。不要把另外两个 owner 池的较高候选覆盖直接搬到此合同中。[原机制诊断](RELATION_MECHANISM_DIAGNOSIS_20260926.md)、[顺序答案结果](OWNER_ORDER_ANSWER_RESULTS_20260927.md)、[完整候选上界](CANDIDATE_ORACLE_RESULTS_20260927.md)。

| 已核对项目 | 完整计数与含义 |
|---|---|
| 原池 support | 1,214 个 query-unit 判断 / 后端；完整恢复结果有 15 methods × 77 = 1,155 条记录 |
| 已准备 static | 562 个 document-A-B 唯一邻接 pair，覆盖原池全部 844 个 edge-query 出现；每题 8–14 条边，207 条唯一边在多题出现 |
| static 执行状态 | 原 308 个逻辑任务全部为 support；恢复新增 144 项也全部为 support。已完成 labels 中无 static ID。本阶段 562 static 尚未调用；旧 15 题 pilot 属于不相交论文，不能移用其关系标签 |
| JEV 支持规则下潜在有效边 | 两端均非 `no` 且 native 文本不同：107 个 edge-query 出现，分布于 46 题；其余 31 题没有这种边，仍保留在分母 |
| 原 I 包的一个宽松机会描述 | 24 题共 27 次“B 已入选而 A 未入选”；不保证 B 足够早选入、A 放得下或关系能改变优先级，不能当作实际触发数 |

| 给定论文候选池 | 候选出现 | 超出该题旧 support 的出现 | 有未判 support 的题数 | 邻接出现 | 不在原 562 static 中的边出现 |
|---|---:|---:|---:|---:|---:|
| leaf_direct / 原池 | 1,214 | 0 | 0 | 844 | 0 |
| leaf_owner | 1,179 | 258 | 72 | 871 | 156 |
| dual_owner | 1,220 | 313 | 77 | 904 | 177 |

上表按 query 出现计数，不是去重后的新任务数。只对 owner 池中有旧标签的单元运行，会改变评价对象；需要新完整判断计划才能比较。因此本草稿只用原池，不新增 owner/corpus 变体。旧 oracle 是带 gold 的可达性诊断；不读取其 witness 来挑关系、题目或参数，也不把 oracle pack 送生成器。

## 3. 建议冻结的唯一 selector

| 参数 | 本草稿固定值 |
|---|---|
| support 来源 | 完整审计后的 JEV coarse labels；无论最终答案质量如何均选这套作为本窄实验的固定输入，不事后挑后端 |
| base score / eligibility | yes = 1，unknown = 0.5；no 永远不可选，关系不能救援 no |
| 关系激活 | 先限制为下述 `E*_q`（两端 eligible、不同 native 文本）；其中仅 dependent 为 active。independent/unknown 均无 bonus，不归一化或阈值扫描 |
| bonus | 待选 A 有已选 B 且 `B depends on A` 时 +1；只取布尔值，不随边数累加；无递归闭包或强制成对入选 |
| 排序 | 每步按 `-(base + bonus)`、原 frozen dense rank、原 source order；无分数学习 |
| 容量 | 最多 3 个完整 native 单元、完整渲染后实际 ≤1,024 BGE tokens；不截断 |
| 去重与渲染 | `(doc_id, exact native_text)` 去重，原 `[unit_id]` header 与 source-order 渲染；本轮仅 given-document |
| 不拟合的参数 | 不使用旧 384-token merge gate，不改 chunk，不引入 raw-score bonus、方向变体、k 扫描或新早停阈值 |

可执行规则：pending 初始为全部 eligible 候选，selected 为空。每步对 pending 重算上述 key，取第一项并从 pending 移除；若同源同文重复则跳过，否则实际 tokenizer 计算 `render_pack(sorted(selected + candidate))`，放得下才接受。每次跳过仍需记录；不可用“单段 token 和”代替整包计数。达到 3 项或 pending 为空才停止。无 active edge 时每步选择/拒绝 trace、最终选集、pack bytes 和 token 数必须还原原 `I_jev_k3`，而非只要求指标相等。原实现入口为 [replay](qasper_relation_replay.py) 的 `_select` 与 [packer](run_qasper_evidence_baselines.py) 的 `PackCounter/render_pack`；用新独立纯函数适配方向，不修改冻结实现。

## 4. 先行 CPU gate：501 个 mask，不计算质量

令 `E_q` 是两端在原候选中的真实邻接边；令 `E*_q` 再要求两端 eligible、native 文本不同。其余边在此规则下不可能触发有效 bonus，可以从 mask 维度消去，但必须验证消去的充要条件。令 `m_q = |E*_q|`。

已核 `m_q` 直方图为 `0:31, 1:19, 2:8, 3:10, 4:6, 5:1, 6:1, 7:1`；故 **Σq 2^m_q = 501**。对每题全部二值 mask 回放上节的唯一 selector，包括零 mask 和全 mask。逐题 mask 不要求满足跨 query 的同一静态标签赋值，所以这只是宽松的机制机会包络；它不产生可部署关系策略或质量上界。不要挑“最好 mask”，也不为有可变 pack 的题另建主分母。

**完整输入合同：** 原 prepared/manifest；完整 recovery plan/run/audit 及 execution receipt；JEV labels 和其精确 support task 映射；完整 native units、原 query candidate/rank；原 `I_jev_k3` selection 和 pack hash；冻结 tokenizer 文件/配置；新纯 CPU selector、测试、gate 协议及固定 plan。所有直接消费文件预先 hash 绑定并前后验证，完整审计由既有 receipt 与其绑定证明继承。不要调用会重读 QA、重算科学指标的旧 `verify_completed_run` 或 prepared loader；gate 用新的只读 metadata/selection adapter 消费已完整审计的封印产物。query 原文不是 gate 算法输入；gold/answer annotations、reference scorer、oracle witness 和任何生成答案都不进入 gate 数据对象。旧 per-question 文件含质量字段，但 adapter 只投影 identity/method/selected/pack/token 白名单，不向核心传递 F1/recall。

**必须满足：**

1. 77/24 完整，边由 source order 推导且不存在跨 doc/假邻接；每题 masks 恰为 `2^m_q`，总数恰 501，无漏重。
2. 零 mask 逐步还原 independent selector，并与冻结 I 的 selected IDs/pack hash/actual tokens 相同。
3. 全 mask 与单独的全部邻接策略逐步一致；不以此替代其余 mask 枚举，因为全邻接无变化不推出子图无变化。
4. 每包合法，去重和真实 budget 均通过。建议整次 gate 限时300秒，CPU 串行、tokenizer `local_files_only`，无 encoder/GPU；分别记加载校验和枚举时间。固定新目录、超时/失败无 complete summary，不改源或继续部分结果。

**输出合同：** public aggregate 只含 77/24、边数及 m 直方图、501 完整计数、零/全 mask parity 通过数、有至少一个不同最终 pack 的题数、各题不同 pack 数的匿名直方图、相对零 mask 的 added/removed 数分布、实际 token 差及 min/max 范围、CPU 时间/环境/来源哈希。不输出 F1、recall、Answer F1、题文、ID、关系文本、推荐 mask 或性能“收益”。每 query/mask 的完整 identities、trace、pack/hash 和 tokens 仅进 ignored local。token 范围来自全部 mask，不是优化目标。

如果 **77 题所有 mask 的最终 pack 都等于零 mask**，此固定机制在此池中无行为区分，停止静态付费；不据此否定其他关系表示或其他任务。如果存在变化，只证明规则有机会影响输入，不证明真实标签会激活该机会或改变质量。

## 5. 未来若准入，四个固定主臂

| 方法 | active relation set | 要排除的解释 |
|---|---|---|
| R0 independent | 空集 | support 本身的排序作用 |
| Rcontent | 562 个完整 JEV 静态判断中的 dependent，按每题候选限制 | 所研究的局部关系内容 |
| Radjacent | 每题全部真实邻接；方向、eligibility、bonus 与 Rcontent 相同 | 仅邻接扩展/补上下文就能解释变化 |
| Rplacebo | 预设匹配分层中的标签置换，见下文 | positive 边数量及可触发机会本身 |

JEV-only 是控制范围的选择，不是后端胜出结论；此处不新增 general 静态判断。原 dense、BGE reranker、JEV raw-score 结果作为已审计外部参照完整保留，不能选最好 `k` 来比较，也不把 Rcontent−R0 解释为胜过强排序器。如果需投稿级“优于独立 ranking”结论，最终同生成器全77比较必须覆盖这些既定强基线，不能只胜过 coarse ordinal baseline。

**placebo 的可执行建议：** 对每题 `E*_q` 按 `(A.kind, B.kind, support[A], support[B], baseline_trigger_class)` 分层。`baseline_trigger_class` 从零 mask acquisition trace 决定：B 在第1/第2次成功选择、且当时 A 未选时，分别记1/2，其余记0。每层原 edges 以 `(A.order,B.order)` 排序；以 `SHA256(canonical_json(["SLAC-local-dependency-placebo-v1", 20260927, family_id, question_id, stratum, edge_identity]))` 确定目标顺序，同 hash 时以原 edge order 决胜。精确定义 `placebo[target_order[i]] = original_label[source_order[i]]`，其中 original_label 是 dependent 的布尔值。不重复抽样、不尝试另一个 seed、不以质量选择排列。

这样逐题保留 eligible positive 数、endpoint kinds、两端 base score 和零 mask 下早期触发机会计数。它**不保证**动态图上的度数、改序后的触发次数或最终长度相同；全部记录这些差异，不能单靠一条正 CI 声称纯内容因果效应。placebo 是依赖 query eligibility 的诊断 null，不是 query-independent 可缓存文档标注；本阶段也不检验其跨 query 一致性。层内同标签、单边层或 identity permutation 不强行重抽：报告可移动边数、实际变标签边数和改变 pack 的题数。若 content/placebo 在全部最终 packs 上相同，则该控制对比退化，停止语义内容收益主张及新的对应生成调用；如要更宽 null，应另写未来协议，不事后松分层取得想要的差异。

## 6. 质量、长度与停止规则

仅在全部静态标签完整有效后回放全部四臂；若有拒绝、漂移、不确定响应或缺失，不用子集作为主结果。首先完整保存不含 gold 的 packs，再统一生成原 evidence-only prompt 的答案，固定 Qwen3.6 Plus / Alibaba、temperature 0、512 max output、no reasoning，实际返回 model 必须与父响应的 `qwen/qwen3.6-plus` 字符串一致。与已审计历史答案只按 endpoint、prompt version、完整 canonical payload 字节继承。四臂每组77，保持 empty/no-op cases，不按能触发关系的46题计算主均值。若某臂全部 pack 与已审计臂相同，可逐 payload 继承，不把实验复用说成线上缓存收益。

主比较预固定为 `Rcontent−R0`、`Rcontent−Radjacent`、`Rcontent−Rplacebo`。全部报告官方 Answer F1、Evidence F1、reference evidence recall 和 actual evidence tokens；24-family clustered bootstrap、PCG64 seed 20260927、10,000 同组 draws、question-weighted 与 family-balanced 双权重，共 `3×4×2=24` 个探索 CI，保留胜/平/负、不做选择性显著性叙述。不能用先前15题为本轮主分母；本轮也不是 blinded evaluation。

相同 caps 不等于等实际长度。每题同时输出 selected count、token 差、added/removed IDs 的匿名汇总与关系→bonus→priority→pack 的事件计数。四臂比较可排除部分结构/数量解释，但不能完全隔离文本内容与实际长度。如果优势只伴随更长包，结论仍是该策略的总效果；不得写“与长度无关的语义贡献”。严格等长度的因果断言需未来另冻对照，不在看到结果后补调 token 区间。

停止/收窄条件：预算或截止不通过则不调用；gate 全部 pack 不可变则不购买标签；静态/生成不完整则只报告状态与账目；content 无选集变化则不为相同 payload 重复生成；content 未优于结构/placebo 则不主张关系内容收益。宽区间表示不确定，零下界、跨零与全零分别如实描述。即使开发结果为正，也只能支持下一次独立冻结验证，不能直接声称 sharing、新架构成功或投稿级泛化。

## 7. 请求与预留：已算一部分，尚未准入

令每文档 unique edges 数为 `e_d`。保持原顺序、每批≤8、完整 payload≤24,000 UTF-8 bytes、两后端共同 admissible 的旧批规则，批数 `B=Σd batch_count(E_d)`，有 `B≥Σd ceil(e_d/8)`。本次离线用原函数完整构造 static payload，得到 **B=82**。估计未发出任何请求，不改变原 frozen client。

对请求 payload p，旧规则为 `input_allowance=(len(canonical_bytes(p))+2048)×n`；JEV 的 n 为该批判断数，general 的 n=1。output allowance=1,024；每次预留 `max(0.005, 1.5×(input_allowance×prompt_price + output_allowance×completion_price)/1e6)`。价格仅沿用原冻结快照，不是对未来价格可用性的承诺。

| 原 prompt / common batches 的离线估计 | 请求 / 判断 | input allowance | output allowance | 保守预留 USD |
|---|---:|---:|---:|---:|
| 建议的 JEV static-only | 82 / 562 | 9,434,931 | 83,968 | 0.6510020850 |
| 非本窄实验的 general static，供范围比较 | 82 / 562 | 1,465,442 | 83,968 | 0.9600093750 |

JEV allowance 超过旧单 segment 的8M，因此未来即使准入也需连续分片并共用一个 stage/night 账本，不能每片重置预算。保留完整562判断，不通过改批量、删题、删任务或阈值凑本夜空间。batch state 可见全部 items；即使指令指定单 pair，也不能声称物理隔离。若未来改成 singleton 隔离，须新 prompt/batching 和完整重估，不能沿用本表准入。

当前完整 support 的整夜预留是4.1353945925；已冻结、正在执行的主答案另承诺0.4699782750，合计 **4.6053728675**，在5美元协议上限下仅余 **0.3946271325**。JEV static 单项就超出 **0.2563749525**，尚未计入关系新答案。因此本夜本方案 **不准入**。当前实际已报告费用较低也不能退回保守预留；历史 unknown1 继续保留。这是本夜客户端协议，不是供应商账户余额或限额证明。

未来静态标签完整后，令 `U` 为四臂308逻辑答案的唯一完整 payload 集，`H` 为有效完成的父响应集合；新生成数 `G=|U\H|`，新增预留 `Σp∈U\H reserve_answer(p)`。目前 **G 未知且未估**。必须全构造、精确字节匹配并预算整轮后才能 freeze/release；不按已有缓存子样本或表现好的臂先跑。未来需要新的明确阶段预算/时限合同，保留全部历史尝试与未知，不沿用已结束的09:00窗口。

## 8. 下一步最小实现与证据锚点

下一步只需新纯 CPU 模块：严格 whitelist adapter、上述 directed selector、有限 mask 枚举器、不可覆盖 prepare/run/audit。测试覆盖 direction、no 禁止、unknown、非累积、source-duplicate、整包非加性 token、真实 tokenizer parity、零/全 mask 逐步 parity、枚举无漏重、预算/超时 fail-closed、输入变更拒绝。synthetic 用显式小型反例独立推导预期，不按真实 F1 调规则。冻结源/测试/协议/plan 后才单次执行机会 gate；**本文没有启动 gate，也没有实现或授权付费执行器。**

只读可行性记录在 ignored `qasper-next-relation-feasibility-01`，包含独立检查脚本、`receipt.json` 和绝对路径→SHA 的 `source_binding.json`。公开文稿只保留汇总和以下来源锚点，不发布私有 ID/题文/请求。

| 来源 | SHA256 |
|---|---|
| prepared.json | `b3bc8a3272d2898246aa7295aba7b366881368188c3a008b0f456e4a838cc885` |
| 完整 recovery summary | `3382bd53d43980d71bb030a99f46573fbd0beb47e7464b9282521f34ca564ad3` |
| 完整 recovery audit | `4a353f8259defe7f65230f2e1da6908e4ea60534a2df036df183832f92abdc80` |
| native run02 per-question | `3dfef8ecff11d3347d3bc8edcbad4ce2a83cdb7a5ef85311aae9a36effbd59cd` |
| 本次 feasibility receipt | `3f672a6e09f8b1e48dea1578746162743adad858edd160176b9bc5f584737819` |

本次检查仅核绑定文件、覆盖和费用公式，消费既有完整审计，不重新运行封印科学实验，也不冒充新的完整 support 审计。后续正式 gate 必须实现上述来源封印与完整性验证，而不是仅相信这里的手录 hash；该验证不需要重开 QA 或重算旧质量指标。
