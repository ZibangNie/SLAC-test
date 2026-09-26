# 剩余249篇validation的文档重复筛查

2026-09-27。全部249篇完成8批文档筛查及聚合回放审计：**1个中等词面重叠标记，涉及1篇剩余validation文档与1篇canonical train文档**。没有高重叠或family/arXiv base-ID标记。标记需要复核，不等于泄漏、同一论文或应当剔除；其他零标记文档也没有获得独立评测准入。

本次没有选择holdout、排除任何文档、读取新QA/答案或官方test文件，没有模型、GPU或API调用。派生manifest明确标注 `screen-only-not-evaluation-pool`；原32篇开发池、旧manifest与筛查器均未修改。[完整公开聚合](results/qasper_remaining_validation_screen_20260927.json)是本地已审计聚合的逐字节副本，不含真实文档ID、原文、QA或本地源路径。

## 范围与固定方法

从冻结index与eligibility记录验证全部281篇validation，减去已有32篇开发池，得到完整249篇；按doc_id排序分为 **32×7 + 25**，不依据标签或结果筛选。每批复用旧 `screen_qasper_overlap.py`，对同一组 **888篇canonical train、281篇validation、8,443行legacy train、1,056行legacy dev** 逐一比较。

正文准入依据是旧pool的完整审计记录：1,169篇canonical文档的 `qa_field_present=0`，全部记录了未导出QA；本次先校验绑定index仅含train/validation，再核对相同canonical shard哈希，才进行机器正文处理。没有打开QA sidecar、原始压缩档案或新增来源。正文未输出到工具日志或公开报告。

算法和阈值沿用旧实现：NFC、casefold、Unicode词的精确唯一5-gram集合，排除声明为heading的块，不使用近似哈希或抽样。少于5词的非空文本以完整词序列作为单个集合元素。高标记要求shared≥100且（Jaccard≥0.8或任一方向containment≥0.9）；中标记要求shared≥25且（Jaccard≥0.1或任一containment≥0.3）。另以声明family或身份/路径字段内的arXiv基础ID提示版本关系，忽略版本后缀；legacy的语料级source_family不作论文身份。

## 完整配对分母与结果

| 249篇对照范围 | 原始比较次数 | 去重配对数 | 原始标记数 | 去重标记数 |
|---|---:|---:|---:|---:|
| Canonical train | 221,112 | 221,112 | 1 | 1 |
| 当前32篇开发池 | 7,968 | 7,968 | 0 | 0 |
| 其余249篇彼此，排除自身 | 57,980 | 30,876 | 0 | 0 |
| Legacy train | 2,102,307 | 2,102,307 | 0 | 0 |
| Legacy dev | 262,944 | 262,944 | 0 | 0 |
| 合计 | **2,652,311** | **2,625,207** | **1** | **1** |

同批剩余文档对只出现一次，跨批对会双向出现；聚合按无向canonical文档对去重，并验证双向交集、集合大小与标记一致。legacy按文件split、行号和文档身份计数，避免把重复ID的不同源行错误合并。逐批flag与nearest记录、原始方向及去重记录都留在本地忽略目录；未将低于阈值的重叠宣称为不存在。

历史legacy train/dev内分别有 **857 / 91** 行的 `orig_split=test`，合计948个源行；8次扫描累计为7,584次行出现。它们属于既已绑定的命名train/dev文件，仅按历史来源计数，没有打开真正test文件，也不构成test评测。全部8批参考分母、历史来源计数和输入哈希均保持一致，0跳过、0自动排除。

## 审计与复现

新增包装器19项合成测试及旧筛查器10项测试，共 **29项通过**。测试覆盖249篇完整分批、拒绝mixed-test index且不打开正文、QA来源证明缺失、输入绑定、跨批去重/缺失反向标记、失败不得发布全体结果，以及修改聚合后重签输出也无法通过回放。运行后审计从8批保存的计数/配对重建公开聚合；它验证包装与来源一致性，没有第二次执行完整词面扫描。

扫描时间逐批为32.516–39.859秒，合计 **284.610秒**；包含各批末尾哈希检查为 **291.190秒**；wrapper运行总计 **292.094秒**，不含解释器启动。每批扫描上限300秒；旧算法规定的最终哈希核验在上限之外单独计时。没有把这些共享进程时间作为不同检索方法的速度比较。

以下为已执行的prepare/run记录；原目录不可覆盖、不可重复run。audit可只读复核。

```powershell
python docs/research/screen_qasper_remaining_validation.py prepare --pool-manifest artifacts/research-foundation/qasper-pool/pool_manifest.json --prior-screen artifacts/research-foundation/qasper-overlap-screen/overlap_report.json --metadata-review artifacts/research-foundation/overnight-20260927/remaining_validation_metadata_review.json --output-dir artifacts/research-foundation/qasper-remaining-validation-screen-01
python docs/research/screen_qasper_remaining_validation.py run --directory artifacts/research-foundation/qasper-remaining-validation-screen-01
python docs/research/screen_qasper_remaining_validation.py audit --directory artifacts/research-foundation/qasper-remaining-validation-screen-01
```

| 绑定 | SHA-256 |
|---|---|
| Plan | `4900de642c74d08454f36216e06b829590705ef76efb49e2c4b94544b8c04197` |
| Prepare seal | `afb719e6aa9e69e6f0249057e48ca73382b5640bd2e4c02490d476a66c750e14` |
| 新wrapper | `5dead1cf00d7887e66fc2e565d37731b9d6b517aab6d17c63ea92dad89caf26c` |
| 新tests | `0d244005d3cd9f942935352342eccae105ff0d7a763fe7c03c3ef200b7e19d07` |
| 公开聚合及本地原件 | `7c2b5be39c161dc7e139849daf756af48398782bde5ac555314a11188f742e5b` |
| Audit receipt | `f160683e787c68cf68eb25b2328b86375f1d588f8bf87408dab776a850409e26` |

## 仍未完成的准入事项

需复核唯一标记的论文版本、共同来源或方法段落关系，并保留判定依据；不能自动删除它。index没有显式版本字段，基础ID未命中不能完成版本/family归并；词面方法也无法排除改写、翻译或OCR差异。暴露登记仍须覆盖开发、训练及prompt实例；最终方法和评测协议要在任何后续QA导出/确认评测前冻结。未来QA来源、多参考与原生证据对齐须独立验证，本次文档筛查没有替代这些步骤。
