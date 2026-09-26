# SLAC research foundation: 2026-09-26

本轮完成研究计划第一阶段的实现修复、旧数据机械修复和有限训练诊断。尚未运行 JEV 架构对照、独立下游评价或正式规模训练。所有改动在隔离工作树 `D:/code/Github/SLAC-research-foundation-20260926`，分支 `codex/research-foundation-20260926`，基于 `7df78733c02ed34a285b869dafceced1bce1a4b4`。原工作树已有改动、源训练数据和迁移包未改写；本报告初版记录时尚未提交或推送，付费 API 调用为零。

本文指向 `artifacts/research-foundation/` 的链接依赖本地研究产物；原始数据、QA、缓存和权重不随代码发布到 GitHub。

## 1. 已修复的实验基础

- **标签与解码一致**：定义确定性的最小成本单调对齐，`KEEP/DEL/SHIFT/INSERT` 可严格回放最终边界。DP 在 DEL 后保留最近的真实输出位置；默认允许相邻 INSERT；逐文档真实 gap 长度控制 padding。
- **监督对应最终产物**：best-of-N 推理以原始 `b0` 到最终投影结果重建 canonical labels，逐轮原始 trace 单独保留。采样采用扰动后最优解方法，并非精确路径分布采样。
- **硬预算真正生效**：soft-min 合并不能突破 atom/字符/token 最大值；完整 span 交给实际 tokenizer 计数，包含 special tokens；不可拆分的超长 atom 在严格模式下报错。提供准确原文 span 时支持无损拼接；旧数据默认仍为显式标记的换行拼接。
- **训练信号与输入完整性**：冻结 BGE 始终保持 eval；batch size 1 下置信权重有效；屏蔽 padding 后计算 loss，避免非有限数；默认拒绝 atom 截断和超出文档长度约定的输入。SHIFT 半径必须与模型一致。
- **缓存和记录**：缓存特征校验 dtype、维度、右侧 padding、atom/gap 数和有限性；每轮训练记录 raw/projected 指标、投影修改数量和截断次数。checkpoint 加载使用 `weights_only=True`。

长文档显式窗口与跨窗口连接尚未实现。`max_doc_atoms` 是输入保护，不是内存复杂度改进。旧训练循环仍为较大 DocEncoder；本轮实际训练的是下面明确列出的微型诊断网络。

## 2. 旧标签修复与使用边界

| 项目 | Train | Dev |
|---|---:|---:|
| 原始行数 / 修复后严格回放通过 | 8,443 | 1,056 |
| 原标签跨序行数 | 5,784 | 746 |
| canonical 动作分配改变 | 8,045 | 1,006 |
| `orig_split=test` 历史来源行数 | 857 | 91 |

atoms、`b0`、`b_gold` 与历史元数据保留，输出到新目录；第二次逐行比对验证所有非标签字段与来源哈希。存在 1 个跨 split 共享 `doc_id`，近重复和语义参考仍未审核。因此完整导出明确标记为 **legacy diagnostic only / not cleared for formal training / not independent evaluation**。修复标签并未解决旧数据谱系问题。

记录：

- [标签契约](../../SLAC/refiner/LABEL_CONTRACT.md)
- [导出清单](../../artifacts/research-foundation/labels/manifest.json)
- [独立序列化复核](../../artifacts/research-foundation/labels/serialized_verification.json)

## 3. 实际 GPU 训练诊断

固定选择首批合格的 16 篇 train、8 篇 legacy-dev，均要求 `orig_split=train`，文档 ID 不重叠，每篇 4–128 atoms，每 atom 含 special tokens 后不超过 128 tokens。仅机械过滤，未按模型成绩或语义挑样。总计 1,044 atoms，实际截断 **0**。这仍然是旧来源开发诊断，不能宣称独立泛化评价。

两组使用相同冻结 BGE-M3、旧实现的 masked-mean pooling、缓存特征、随机种子 13、文档顺序和统一权重。DocEncoder hidden 128、1 layer、4 heads、window 8、dropout 0；AdamW lr 0.002、batch 1、各 128 steps。普通分类器预测最终边界；Refiner 预测 canonical 编辑和 residual INSERT。最终证据投影预算在此诊断固定为 512 BGE tokens。

| 模型 | Raw train macro F1 | Raw legacy-dev macro F1 | 训练损失：首 16 → 末 16 步均值 |
|---|---:|---:|---:|
| 原始 `b0`，无需训练 | 0.8461 | 0.8427 | — |
| 普通边界分类器 | 0.9882 | 0.8786 | 0.6397 → 0.0522 |
| 修正版 Refiner | 0.9571 | 0.8358 | 2.2740 → 0.6468 |

普通分类器有 1 篇训练文档被预算投影修改，projected train F1 为 0.9869；其余这批结果 raw/projected 相同。两种 loss 的含义不同，数值仅作各自优化诊断，不横向比较绝对大小。

**结论**：前向、反向、保存及解码可运行，模型能拟合小样本；当前结果没有显示 Refiner 优势。不能以 8 篇开发文档决定架构胜负，也不能把高训练 F1 作为论文结果。Refiner 可读取 `b0` 先验，普通分类器没有这一输入；正式编辑机制对照还必须加入 `b0`-conditioned classifier，并匹配容量、训练选择与调参预算。

最终 `probe-02-final` 在 RTX 5070 Laptop 上约 **13.47 秒**完成，PyTorch allocated 峰值约 **2,654 MiB**，包括 BGE 加载与 FP16 转换，不含驱动、其它程序或全部系统显存占用。只验证了这个短文小网络配置，不能外推全量长文训练耗时。

- [最终诊断结果与来源哈希](../../artifacts/research-foundation/probe-02-final/results.json)
- 同目录保存两组 safetensors 权重、缓存特征、所选样本与执行时源码快照。
- 权重对应脚本中的 hidden-128 `ProbeNetwork`，并非默认 hidden-768 生产模型；加载时需要相同架构。权重有限性、样本谱系、doc ID 隔离、样本和源码快照哈希另经[产物复核](../../artifacts/research-foundation/probe-02-final/artifact_verification.json)。
- `probe-01` 保留初跑；补齐 SHIFT 检查、加载峰值计量和源码快照后进行一次复跑，指标相同。

## 4. 环境与复现

新环境：`C:/Environment/python/venvs/slac-research`，Python 3.12.10、PyTorch 2.9.1+cu130、transformers 4.57.3、tokenizers 0.22.1。`pip check` 通过。旧失效 `.venv` 未修改；本环境版本不等同于迁移包 tokenizers 0.23.2 环境。

[直接依赖](../../SLAC/refiner/requirements-research.txt)；[本机完整 freeze](../../artifacts/research-foundation/environment-freeze.txt)。CUDA wheel 来自 [PyTorch 官方历史安装页面](https://pytorch.org/get-started/previous-versions/)。

在隔离工作树执行：

```powershell
$py = 'C:/Environment/python/venvs/slac-research/Scripts/python.exe'
& $py -X utf8 -m pytest SLAC/refiner/tests -q
& $py -u -X utf8 SLAC/refiner/scripts/run_foundation_probe.py `
  --train artifacts/research-foundation/labels/refiner_train_diagnostic.jsonl `
  --dev artifacts/research-foundation/labels/refiner_dev_diagnostic.jsonl `
  --model 'D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181' `
  --output artifacts/research-foundation/probe-new `
  --steps 128 --train_rows 16 --dev_rows 8 --max_seconds 900
```

输出目录必须不存在；脚本拒绝覆盖。实验数据和 checkpoint 目录已加入 `.gitignore`。旧测试的硬编码绝对路径和只打印不执行断言的形式已改为可执行离线 fixture。

完整测试 **82 passed**。覆盖 5,120 个小状态独立 DP 最优性对照、200 个随机 projector 分区，以及真实训练循环、K=2 的混长评价、raw/projected 分离和安全 checkpoint 逐张量恢复。8 条 PyTorch 提示仅说明 `norm_first` 关闭 nested tensor 优化，不影响测试通过。[最终验证记录](../../artifacts/research-foundation/verification.json)。

## 5. Qasper 非 test 开发候选池

已按固定 family hash 冻结 **32 篇官方 validation 文档**，来源候选共 281 篇。机械检查 1,169 条 canonical/index、既有开发保留清单和三篇 Qasper 暴露 ledger，再对旧 train/dev 的 9,499 行做规范化全文/正文精确哈希比较。所选池无已知暴露或精确重叠；近重复、论文版本和更完整家族关系仍待核验，不能据此宣称独立确认评价。

canonical 有正文与定位信息，但刻意没有导出 QA。已从本地 `qasper-train-dev-v0.3.tgz` 的 **`qasper-dev-v0.3.json`** 成员补出单独的原生 QA 侧文件，无需下载或 API。保留所有原始多回答及各自证据集合，省略与评价无关的 worker 标识；原始包与 canonical 不改写。

| 本轮机械结果 | 数量 |
|---|---:|
| 文档 / 问题 / 原始回答标注 | 32 / 104 / 185 |
| 原生非空段落回映 canonical 通过 | 1,363 |
| 唯一定位至文本段落的证据项 | 276 |
| 图表证据项，保留单独状态 | 19 |
| 尚无法定位的文本证据项，保留原文与失败状态 | 11 |
| 至少一份回答的全部证据可唯一定位的问题 | 86 / 104 |
| 原生不可回答标注 | 11 |

这是机械对齐报告，不是准确率、完整充分证据验收或下游效果。没有根据模型成绩筛题；图表、不可回答及未匹配项均未丢弃。侧文件的 `cleared_for_evaluation` 仍为 `false`：需要调查 11 项未匹配、定义原生/文本子集评价分母、验收 evaluator，并完成近重复隔离。

- [候选冻结清单](../../artifacts/research-foundation/qasper-pool/pool_manifest.json)
- [原生 QA 与证据对齐统计](../../artifacts/research-foundation/qasper-pool/native_qa_alignment.json)
- [候选冻结脚本](freeze_qasper_candidate_pool.py) / [原生 QA 导出脚本](export_qasper_native_sidecar.py)

早期 `pool_manifest.json` 的 missing-QA 状态描述 canonical 本身；后续原生侧文件供给情况由 `native_qa_alignment.json` 记录，保留前一阶段的历史记录。

## 6. 下一阶段

[实验协议草案](EXPERIMENT_PROTOCOL_DRAFT.md) 将后端与系统机制分开：independent / cache-only / shared 三种执行方式乘以通用 LLM / JEV 两个后端。统一原文、候选、检索器、生成器与真实最终 token 预算。Qasper 原生 QA/证据用于下游评价；条件、定义、例外的“充分证据”仍需另外审核，不能从 Qasper 自动推定。

当前实际代码修复和训练诊断不包含这个六格架构实验。下一步完成上述候选池的剩余验收，运行预算匹配的规则基线和证据选择 oracle，验证相同候选池是否存在可利用的证据缺口，再连接 JEV 并比较机制。保留简单基线，按效果证据决定是否扩大 Refiner 训练。
