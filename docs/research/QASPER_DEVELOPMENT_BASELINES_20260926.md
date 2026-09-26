# Qasper 开发评价底座与基线进展

日期：2026-09-26。第一阶段已发布至 `codex/research-foundation-20260926`，提交 `ab2bfcbc837f53cd9e0841857c268b2e7a176e17`。本报告记录随后实际完成的评价准备和本地实验，未进行 JEV/通用 LLM 调用或答案生成。

## 范围与完成情况

冻结的候选仍为 **32 篇 Qasper validation 论文、104 个问题、185 份原始回答标注**。这批数据现在已用于方法开发和结果分析，不能作为独立确认集。官方 test 没有打开，原始数据、QA 和权重不随代码推送。本文 `artifacts/` 链接仅在保留本地产物的工作树内有效。

### 原生证据对齐

此前 11 项未匹配文本实际是 **5 篇论文的完整章节标题**；旧适配器只索引段落。新版先完整索引原生 title、abstract、section_name 和 paragraph，再定位证据，使用源指针与 NFC/空白等价检查，不使用模糊匹配或语义猜测。

- 32 个 title、32 个 abstract、423 个章节标题、1,363 个非空段落均通过 native→canonical 校验。
- 276 项段落证据、11 项标题证据定位成功，未匹配文本为零。
- 19 项图表证据可追溯到原生 caption，仍保持图表类型；caption 没有被冒充图表内容或正文证据。
- 全部 104 行原有 question/answer 字段保留；287 个文本证据 span 经写盘后二次验证。

完整可定位文本的回答为 150/185；至少一份回答的文本证据可完整定位的问题为 88/104。标题可定位不等于语义充分。完整官方指标保留图表参考；纯文本模式另列，未静默改变分母。

[对齐代码](qasper_alignment_v2.py)、[审计](../../artifacts/research-foundation/qasper-alignment-v2/alignment_audit_v2.json)、[写盘复核](../../artifacts/research-foundation/qasper-alignment-v2/serialized_verification.json)。

### 词面近重复 screening

采用 NFC/casefold 的 Unicode word 5-shingle 集合，对 32 篇候选与 1,169 篇 canonical Qasper train/validation、9,499 条旧 Refiner train/dev 做流式精确比较；排除自身和重复计数后共 **340,848 对**，约 33.45 秒完成，零跳过。声明的 family/arXiv 身份冲突与达到阈值的近重复候选均为零，输入哈希未变。

阈值、全部比较范围与最近邻记录见[screening 报告](../../artifacts/research-foundation/qasper-overlap-screen/overlap_report.json)。它不能排除译本、改写、未知家族和未检查语料；`near_duplicate_clearance=false` 保留，不将词面零命中写成独立评价资格。

### 官方评价语义

固定 AllenAI evaluator commit `afd0fb96bf78ce8cd8157639c6f6a6995e4f9089`，按完整原文字符串计算 Evidence F1，对每问题的多个参考分别计算并取最大，再按问题宏平均。重复证据的列表分母、空参考、不可回答和 FLOAT 处理均按上游保留。[定义、源码归属与许可证](QASPER_METRICS.md)。

适配器与固定官方源码直接比较了 1,600 对证据列表、81 对答案字符串、8 个合成问题的两种聚合模式，结果一致。只在生成真实答案后才使用 Answer F1；本轮不报告答案质量。

## 实际基线设置

任务明确为 **给定正确论文后的证据选择**，不声称跨库检索。所有方法使用相同的原生 title/heading/abstract/paragraph 候选，经过 canonical source span 验证；人工添加的 canonical “Abstract” 标题不作为原生候选。完整单元才可选入，精确重复文本只选一次。

512/1,024/2,048 的上限按本地 BGE-M3 tokenizer 对**完整渲染证据包**计数，包含单元 ID、连接符和 special tokens，零截断。空包计零。query/指令不在本次证据预算内；这是编码器计量的开发诊断，不能直接替代未选定生成器的上下文预算。

部署可用基线只读取 query 与正文白名单，不读取 gold。BM25 为 Unicode word/casefold、k1=1.5、b=0.75、源顺序打破并列。报告装满预算及 top-1/3/5 的全部变体；top-k 表示最多 k 个可装入的完整不同单元，过长单元跳过。

另设 `gold_subset_oracle`：只在各份原始参考可精确匹配的候选内穷举所有子集，按实际渲染预算选最高官方 Evidence F1。本池最多 10 项原始 evidence/annotation；代码对超过 16 个可达单元的情况明确拒绝，未用近似结果冒充穷举。重复文本使用第一次源位置。它是**利用 gold 的可实现参考结果**，不是可部署模型，也不是任意文本跨度的全局最优证明。

### BM25 与可达性结果

以下均为官方 Evidence F1 的 **question macro**，保留完整参考（含图表）。document macro、recall 诊断、实际 tokens 与全部逐题结果另存。

| 方法 | 512 tokens | 1,024 tokens | 2,048 tokens |
|---|---:|---:|---:|
| 按原文顺序装入 | 0.0128 | 0.0233 | 0.0414 |
| BM25 装满预算 | 0.0993 | 0.1172 | 0.0998 |
| BM25 top-1 | 0.0951 | 0.1047 | 0.1047 |
| BM25 top-3 | 0.1305 | 0.1553 | 0.1553 |
| BM25 top-5 | 0.1121 | 0.1407 | 0.1471 |
| 始终空证据 | 0.1346 | 0.1346 | 0.1346 |
| Gold subset oracle | 0.9058 | 0.9580 | 0.9628 |

空证据能得分，是因为 14/104 题至少有一份按官方规则转换后为空的参考证据列表，包含不可回答情况。不能把该分数误当有效检索。

最终 BM25 运行约 **25.19 秒**，共 2,184 个 question×method×budget 记录，身份唯一、所有实际证据包均未超预算，所有绑定输入哈希前后一致。[最终结果](../../artifacts/research-foundation/qasper-baselines-03-final/summary.json)、[写盘复核](../../artifacts/research-foundation/qasper-baselines-03-final/serialized_verification.json)。

已保留开发过程：`qasper-baselines-01` 在计分前因 canonical 合成 Abstract 标题与 native 文本不匹配而拒绝；修正为原生字段枚举后，`-02` 跑完原先四种策略；观察到装满预算加入无关证据后，`-03-final` 增加 top-k 并补齐冻结哈希/重复身份检查。新增变体是开发期调整，不伪装预注册验证，也没有覆盖旧结果。

### BGE-M3 dense 结果

随后使用本地固定 revision `5617a9f61b028005a4858fdac845db406aefb181`，按照模型的 [CLS pooling 配置](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json) 和 [Normalize 模块](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/modules.json) 提取向量：FP16 backbone、CLS 向量转 FP32 后 L2 归一化、cosine 排序。query 无附加指令。这与旧 Refiner 训练诊断的 mean pooling 分开记录，不能混作同一配置。

全部 1,850 个候选和 104 个 query 均完整编码；候选最长 787 tokens，query 最长 30 tokens。模型以本地 safetensors 加载，不训练，batch size 4，推理与证据包均零截断。候选、冻结输入、预算计算和评价器与前述 BM25 一致。

| 方法 | 512 tokens | 1,024 tokens | 2,048 tokens |
|---|---:|---:|---:|
| BGE-M3 dense 装满预算 | 0.1218 | 0.1252 | 0.1131 |
| BGE-M3 dense top-1 | 0.1268 | 0.1364 | 0.1364 |
| BGE-M3 dense top-3 | 0.1807 | 0.2063 | 0.2112 |
| BGE-M3 dense top-5 | 0.1410 | 0.1785 | 0.1887 |

在 1,024-token 上限下，dense top-3 的官方 Evidence F1 为 **0.2063**，BM25 top-3 为 **0.1553**；平均实际证据长度分别为 **441.8** 和 **498.8** tokens。因此这里是同上限的开发集比较，实际长度并不相等，也没有做显著性或独立确认检验。Gold subset oracle 为 0.9580，是已知参考证据条件下的可达性诊断，不能据此推断模型能实现相同提升。Dense 装满预算增加了参考证据 recall，却可能降低 F1，表明冗余选择也需要控制。

`qasper-dense-01` 约 **52.77 秒**完成，RTX 5070 Laptop 上 peak allocated 显存 **1.13 GiB**。1,248 条 question×method×budget 结果、104 条排序、1,850×1,024 候选向量和 104×1,024 query 向量均写盘复核；向量全为有限值且 L2 范数约 1，summary 可从逐题结果重算，所有证据包满足预算，输入、脚本与权重哈希未变。[Dense 代码](run_qasper_dense_baseline.py)、[本地结果](../../artifacts/research-foundation/qasper-dense-01/summary.json)。

本轮尚未比较 sparse/multi-vector、hybrid 或 reranker；dense 优于本池 BM25 不代表已经穷尽强基线。

### 实现验证

完整运行 `python -X utf8 -m pytest SLAC/refiner/tests tests/research -q`：**171 passed**。8 条 warning 来自 PyTorch 对 `norm_first` 的 nested-tensor 优化提示，不是失败或数值错误。覆盖标签/解码、实际 token 预算、原生对齐、官方指标一致性、数据谱系与哈希绑定、重复身份拒绝、时间上限和 dense 输入/排序契约。原始记录保存在本地 `artifacts/research-foundation/phase2-verification/`。

## 结论边界与接续实验

这些结果表明：在当前给定文档候选池内，已有原生标注证据通常能够在较小预算中装入；单纯扩大输入或装满预算并不保证提高 Evidence F1。它支持继续比较证据判别与选择策略。

它尚不能支持共享关系优于独立处理、JEV 优于通用 LLM、Refiner 更优或真实答案质量提升。Gold oracle 的实际输入长度明显短于装满预算的检索基线，后续必须同时报告实际 tokens 和等预算上限，避免只借长度差归因。正式核心对照继续采用 independent/cache-only/shared × 通用 LLM/JEV；先固定候选、模型可见字段、模型版本、调用/费用上限与生成器，再运行。

Qasper QA 可支持原生证据选择任务；条件/定义/例外的必要或替代证据集合仍需要独立审核。当前开发池、原生定位、词面筛查和官方指标适配都不能替代该审核。

## 复现入口

在研究工作树根目录，使用 `C:/Environment/python/venvs/slac-research/Scripts/python.exe`：

```powershell
$py = 'C:/Environment/python/venvs/slac-research/Scripts/python.exe'
& $py -X utf8 -m pytest SLAC/refiner/tests tests/research -q
& $py -X utf8 docs/research/qasper_alignment_v2.py --help
& $py -X utf8 docs/research/screen_qasper_overlap.py --help
& $py -X utf8 docs/research/run_qasper_evidence_baselines.py `
  --pool artifacts/research-foundation/qasper-pool `
  --sidecar artifacts/research-foundation/qasper-alignment-v2/native_qa_sidecar_v2.jsonl `
  --tokenizer 'D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181' `
  --output artifacts/research-foundation/qasper-baselines-new --max_seconds 600
& $py -X utf8 docs/research/run_qasper_dense_baseline.py `
  --pool artifacts/research-foundation/qasper-pool `
  --sidecar artifacts/research-foundation/qasper-alignment-v2/native_qa_sidecar_v2.jsonl `
  --model 'D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181' `
  --output artifacts/research-foundation/qasper-dense-new --max-seconds 300
```

输出目录必须不存在。文档中的本机路径是已执行配置；迁移时需要提供同 hash 的输入、模型和 tokenizer。未随代码发布的原始产物按相同脚本重建。
