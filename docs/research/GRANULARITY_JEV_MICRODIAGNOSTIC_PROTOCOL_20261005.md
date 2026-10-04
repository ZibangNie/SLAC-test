# 固定两题的 JEV 评分粒度小探测

2026-10-05，新增模型输出前固定。沿用[离线请求检查](results/granularity_jev_wire_feasibility_20261005.json)的完整两题、117 个唯一 query/passage 配对和 15 批请求，不扩样、不重选问题、不改写证据。研究者已经看过这两题及全部参考，因此本探测是开发性机制诊断。

本机 GPU 及内存暂不满足本地 BGE 准入，本次直接检查 JEV 下的三个已固定粒度方案，不宣称完成原 BGE 实验。它使用独立目录、协议、分数语义和评价入口；旧失败运行及大规模付费计划保持原状态。

## 唯一一次受限执行

- 输入文件 SHA256 固定为 `aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3`。两题共有 30 个完整 source Unit、97 个 atom 候选；完全相同的 query/passage 只判断一次，共 117 个配对。不同原文位置仍保留不同候选身份，不将它们当作同一来源去重。
- 沿用 `slac-local-decision-v1` support 判断和原始完整文本。每批最多八项、同批不跨题；保留离线序列化的顺序、全部十五个 payload hash 和预留，不拆批重试，不引入其他模型或答案生成。
- UTC **2026-10-04 18:00:49.772176** 的[官方端点元数据](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints)确认 `typesafe/jev-1.13`、TypeSafe 路由、版本 `typesafe/jev-1.13-20260917`，输入 $0.042／百万 tokens、输出 $0。[官方文档](https://openrouter.ai/docs/guides/community/jev)仍指定 `POST /api/alpha/decisions`。只允许该版本和供应方，不使用 latest alias 或 fallback。
- 单独的客户端预算为 **$0.10**，十五批保守预留共 **$0.0903957440**；输入 allowance 上限 1,395,313，输出 allowance 上限 15,360。这些 allowance 是客户端的保守账本量，不是实际 tokenizer 结果。预留亦不是实际花费或服务端强制扣费上限。
- 最多十五次物理尝试、117 项判断，无重试。单请求看门狗 65 秒，整个工作进程硬时限 180 秒；执行窗口最多一小时。完整来源、请求、当前供应方快照、预算和时限校验通过后，先占用一次性消费标记，再读取凭据。结束后该入口不能自动重启。
- 首次失败、超时、费用未知、超预留用量、身份或分数契约变化立即停止。保留已返回费用和未知尝试；没有费用字段不等于免费或退款。只有全部十五批和 117 项完成并经执行收据核验，才冻结选包、读取参考和评价。

## 三臂共用的选择政策

| 臂 | 排序值 | 可选及输出范围 |
|---|---|---|
| A `source_units` | JEV 对完整 source Unit 的原始 yes 报分 | 完整 source Unit |
| B `parent_max` | 同一父单元内全部新 atom yes 报分的最大值 | 完整 source Unit |
| C `source_atoms` | JEV 对 source atom 的原始 yes 报分 | source atom |

全部候选均参加排序，**不依据 choice=no 排除，也不为 unknown 设置回退或阈值**。这区别于旧 `p_yes_only` 主策略中的 no 过滤；即使所有标签均为 no/unknown，排序与预算仍可能选出证据。choice 和三项原始分数独立保留，不要求 choice 必须等于分数 argmax，不自行改标签。

每项 `yes/no/unknown` 报分必须为非布尔、有限的 [0,1] 数值，三项之和在 [0.985,1.015]；缺项或失败不能代入零或 unknown。不得归一化、采用 confidence 调阈值或称为校准概率。B 的最大值仅作排序启发式，不是父单元的 JEV 标签、相关概率或充分性判断；父项拥有更多子项时更容易取得高最大值。

三臂继续复用原 `select_ranked`：分数降序，同分按原 Dense parent rank、原文起止位置；单次扫描、超限跳过后继续、不回访拒绝项。仅合并接触的已选原文区间，不跨空隙、不裁剪，最多三个输出区间，完整 source evidence block 不超过 1,024 BGE proxy tokens。该上限不包含系统、独立 query message、供应商 framing 或输出，不能作账单 token。

## 冻结与读数

六个完整预测包、B 的全部最大值贡献者、完整新判断、执行收据及代码绑定先保存并封印，之后才读取此前保存的四份参考区间。原问题、所有 annotation 和候选域覆盖上限全部保留；不合成参考、不挑容易覆盖的一份，也不读取全数据集。

报告全部十二条 annotation×方法的参考字符交集、precision/recall、完整覆盖、来源字符数和整块 tokens，以及 A/B、B/C 的来源集合、渲染和 token 差别。第一题两份证据相同，不能当作独立重复；第二题参考范围差异和池外缺失继续单列。公开文件只含序号、数值及 hash，正文、真实身份和请求／响应全文留在私有产物，凭据不写入产物。

A/B 同时改变评分文本上下文与分数聚合，B/C 同时改变聚合、可选粒度和贪心可行性；跨文本与批次分数也没有校准。本探测只能定位固定两题上的选择及字符覆盖变化，不能解释成纯粒度因果效应、语义充分性、原段落 Evidence F1、Answer F1、learned Refiner 收益或论文创新。

结果为正、零或负均结束本次模型调用。若 C 相比已知 B 对照没有观察到覆盖与资源优势，就不以该机制作为扩调用依据；正结果也只能留下待验证假设，不自动扩样、改策略或重新呼叫。随后只分析已存结果和真实来源，继续寻找有证据支持的机制增量。

执行入口：[run_granularity_jev_microdiagnostic.py](run_granularity_jev_microdiagnostic.py)。独立 JEV 选包与评价入口：[evaluate_granularity_jev_microdiagnostic.py](evaluate_granularity_jev_microdiagnostic.py)。实际计划窗口和匿名来源承诺在准备完成后另行记录；这份协议本身不是执行成功的证明。
