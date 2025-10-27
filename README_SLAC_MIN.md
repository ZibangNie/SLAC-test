# SLAC-Min（法规 PDF 的自学习式分块原型）

一个极简但可用的**闭环前奏**：从 PDF 提取文本 → 伪标签分块边界 → 训练线性分类器 → 段落化输出。
仅依赖 Python + numpy（可选：scikit-learn；pdfminer.six/PyPDF2）。

## 安装依赖
```bash
pip install numpy
# 可选但推荐：
pip install scikit-learn pdfminer.six
# 或者用 PyPDF2 做 PDF 备选
pip install PyPDF2
```

## 1) 从 PDF 批量抽取为 .txt
```bash
python slac_min_pipeline.py prepare --pdf_dir ./pdfs --out_txt_dir ./txts
```

## 2) 训练（使用伪标签）
```bash
python slac_min_pipeline.py train --txt_glob "./txts/*.txt" --model_out ./slac_min_model.npz
```
输出会打印与伪标签对齐的 P/R/F1（仅作代理指标）。

## 3) 评估（代理）
```bash
python slac_min_pipeline.py eval --txt_glob "./txts/*.txt" --model ./slac_min_model.npz
```

## 4) 分块推理
```bash
python slac_min_pipeline.py segment --txt_glob "./txts/*.txt"     --model ./slac_min_model.npz --out_dir ./chunks     --threshold 0.55 --min_lines 2 --max_lines 15
```
输出的每个 `*.chunks.txt` 用 `-----` 分隔区块。

## 5) 半监督自训练
```bash
python slac_min_pipeline.py selftrain --txt_glob "./txts/*.txt"     --model_out ./slac_min_model_refined.npz --iters 3 --hi 0.9 --lo 0.1
```

## 结构先验与可定制化
- 正则在代码里：`NUM_PAT/CHAPTER_PAT/APPENDIX_PAT/SECTION_PAT/BULLET_PAT`，按你的法规风格增添花样编号。
- 伪标签规则：`pseudo_label()`；特征：`feature_for_line()`、`boundary_features()`。
- 粒度旋钮：`--threshold/--min_lines/--max_lines`。

## 推荐训练流程（法规 PDF）
1. **准备数据**：按法规种类/年份分文件夹，`./pdfs/train`、`./pdfs/val`。
2. **文本抽取**：`prepare` 生成 `./txts/train/*.txt`、`./txts/val/*.txt`。
3. **首轮训练**：在 `train` 上训练，保存 `model.npz`。
4. **代理评估**：在 `val` 上 `eval`，观察与伪标签的一致度（不代表真实上限）。
5. **自训练**：`selftrain` 做 1–3 轮，把模型的高置信预测回灌到标签。
6. **分块导出**：`segment` 生成 `*.chunks.txt` 供你的 RAG 管线快速接入。

## 接下来的升级（可选）
- **DP 最优切分**：替换贪心，用长度惩罚 + 边界负对数似然做动态规划最优解。
- **版面特征**：加入 PDF 坐标/字体大小/粗细信息（pdfminer.six 可拿到）以增强 heading 识别。
- **弱监督校准**：从目录（TOC）或跨页标题生成更干净的银标。
- **下游回传**：用你的检索覆盖率/Faithfulness/最短证据链长度做外部 reward，对 `threshold/min/max` 做贝叶斯优化或网格搜索。

> 这是 SLAC 的“可落地骨架”。先让它稳定分块，再把 Evaluator 四件套与回传接上，就能形成闭环。
