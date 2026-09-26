# Qasper 指标适配与验收

本模块将 AllenAI 官方评价定义适配到本项目的 `native_qa_sidecar.jsonl`。当前实验只做给定文档的证据选择，因此使用 Evidence F1；Answer F1 接口保留供后续真正产生答案时使用。证据覆盖、token 预算和 recall 诊断均不等价于答案质量。

## 固定的官方来源

- 仓库与文件：[allenai/qasper-led-baseline/scripts/evaluator.py](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py)。
- Git commit：`afd0fb96bf78ce8cd8157639c6f6a6995e4f9089`。
- evaluator 文件最后一次上游修改：`e996b6c7b1b5f95d9308a74e3586416c6e780df1`，2021-05-20。
- 官方源码字节 SHA-256：`781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe`。
- 验收日期：2026-09-26。

数据版本为 Qasper v0.3，代码版本以上述独立的 commit 为准。本次未下载或打开 `test-and-evaluator` 归档，未验证该归档内脚本与仓库脚本的字节一致性。固定源码仅来自 AllenAI 官方仓库。

## Evidence F1 的精确定义

对一个 prediction 列表 `P` 和一份 annotation 的 evidence 列表 `G`：

1. **按完整字符串精确比较**，不做大小写、空白、Unicode、标点或词语归一化。
2. 交集大小 `c = len(set(P) & set(G))`，但 precision 和 recall 的分母分别为原始 `len(P)` 和 `len(G)`。不能去重后再计算分母。例如 `[A,A]` 对 `[A,A]` 的 F1 是 `0.5`，不是 `1`。
3. 两个列表都为空时 F1 为 `1`；其余无交集情况为 `0`。
4. 否则 precision=`c/len(P)`，recall=`c/len(G)`，F1 为二者调和平均。
5. 同一问题有多份 annotation 时，对每份分别计算，取最高 F1。不能将多份证据先合并成一个参考集合。
6. 全集结果按**问题**宏平均，每个问题权重相同；不按论文或 annotation 平均。

`unanswerable=true` 的官方参考强制转换为 `answer="Unanswerable"`、`evidence=[]`，即使原字段意外包含证据也不使用。

官方 `text_evidence_only` 模式只从**参考** evidence 删除包含大小写精确子串 `FLOAT SELECTED` 的项；不修改 prediction。它不表示“可定位到 paragraph 的参考”，也不剔除 unmatched、heading 或空白差异。完整参考默认保留 FLOAT 项，不能静默减少分母。

本项目 canonical 文本可用于展示和 tokenizer 计量；用于官方 F1 的 prediction 必须返回已通过源定位与内容校验的 native 原文字符串。不能把字符跨度交并比、部分 paragraph 覆盖、空白归一化匹配率称作官方 Evidence F1。

## 接口

```python
references_from_annotations(answer_annotations, *, text_evidence_only=False)
# -> [{"answer": str, "evidence": list[str],
#      "type": "none" | "extractive" | "abstractive" | "boolean"}, ...]

paragraph_f1_score(prediction: list[str], ground_truth: list[str]) -> float

evidence_metrics(predicted_evidence, answer_annotations, *, text_evidence_only=False)
# -> {
#   "evidence_f1": float,
#   "evidence_recall": float,             # 非官方诊断
#   "best_f1_reference_index": int,      # 零起始；并列时第一份
#   "best_recall_reference_index": int,
#   "reference_count": int,
# }

evaluate_qa(gold_annotations, predictions, *, text_evidence_only=False)
# gold_annotations: {question_id: answer_annotations}
# predictions: {question_id: {
#   "predicted_answer": str, "predicted_evidence": list[str]
# }}
# -> 官方字段 "Answer F1", "Answer F1 by type",
#    "Evidence F1", "Missing predictions"
```

annotation 可为 sidecar 的 `{"native_answer": ...}`、官方 `{"answer": ...}` 或原生 answer 字典。转换保留顺序、重复 evidence 及全部多参考，不修改输入。无 annotation 或不合法 schema 明确报错；对符合原生 schema 的输入保持官方数值行为。模块没有网络请求、文件读取或语料加载。

`evidence_recall` 单独对各参考取最大值，最大 recall 与最大 F1 可能来自不同 annotation。它采用相同交集与原始参考列表分母；空参考且空 prediction 记为 `1`，空参考但非空 prediction 记为 `0`。这是本项目显式选择的诊断约定，**不是官方附加指标**。

## Answer F1 的精确定义

参考答案类型按官方顺序决定：unanswerable 优先，其次非空 extractive spans（以 `", "` 连接）、非空 free-form answer、`yes_no=true`、`yes_no=false`。依次对应 `none`、`extractive`、`abstractive`、`boolean`、`boolean`。

答案采用官方 SQuAD v1.1 归一化：小写、删除 ASCII `string.punctuation`、删除英文冠词 `a/an/the`、合并空白。词语交集保留词频，得到 token F1。两个归一化后均为空的答案在该上游实现中得分 `0`。

每问题的最高 Answer F1 与最高 Evidence F1 **独立**选取参考。Answer F1 by type 将问题归到取得最高答案分数的参考类型，并列采用原 annotation 顺序中的第一份。缺失 prediction 的问题在两个总体均值中记 `0`，但上游不将其加入按类型的均值；额外 prediction ID 被忽略。这些行为均保留，不能据此将按类型指标误当固定类型子集表现。

## 验证结果

`tests/research/test_qasper_metrics.py` 的 **37 项测试通过**，覆盖：

- unanswerable、Yes、No、extractive、free-form 及类型优先级；
- 多份参考分别取最大、答案与证据独立取最大；
- 缺失/额外 prediction、宏平均和类型并列；
- 重复证据的列表分母、顺序、大小写、空白和 Unicode 差异；
- FLOAT 过滤、空列表约定、输入不变性与 schema 拒绝。

另一次独立验收从固定官方 URL 取得源码，在临时目录加载并运行；逐项比较 **1,600 对 evidence 列表、81 对答案字符串、8 个合成问题的完整/纯文本两种聚合模式**，全部与本地接口数值一致。该验收不加载任何真实数据集 payload。临时官方源码随后由临时目录清理；长期回归测试仅含人工合成字符串与期望数值。

运行离线回归测试：

```text
C:/Environment/python/venvs/slac-research/Scripts/python.exe -X utf8 -m pytest tests/research/test_qasper_metrics.py -q
```

## 适用范围与归属

评价器验收只说明实现匹配上述固定官方定义，不自动解除候选集的近重复、泄漏或独立评估资格限制。只评价证据选择的实验不报告 Answer F1；restricted relevant-subset oracle 也只能说明指定候选与预算条件下的参考证据可达性。

`qasper_metrics.py` 改编自 Allen Institute for AI 的上述官方 evaluator，保留其 SQuAD v1.1 答案指标来源说明。修改包含 sidecar 适配、类型校验、证据单独接口和非官方 recall 诊断。上游使用 Apache License 2.0，完整副本见 [licenses/QASPER_APACHE_2_0.txt](licenses/QASPER_APACHE_2_0.txt)。固定仓库树没有独立 NOTICE 文件。数据集的许可证与代码许可证独立。
