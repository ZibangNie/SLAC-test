# 以 b0 为条件的直接边界基线

2026-10-04。新增隔离的 [`DirectBoundaryHead` 与 `CachedSeedBoundaryClassifier`](../../SLAC/refiner/slac_refiner/models/direct_boundary.py)，直接预测最终 gap 边界，显式读取初始边界 `b0`。**29 项 CPU 合成测试及独立代码/张量复核通过；本轮没有训练、自然数据读取、预训练权重加载或 API 调用。** 这补足了[上轮发现](REFINER_OBJECTIVE_GATE_20261004.md)的基线组件缺口，还不是完整公平比较或质量结果。

旧 `heads.py`、`DocEncoder`、原模型、微诊断 runner 和权重未改动。新模块尚未接入默认 pipeline，也不加载论文 `epoch_8.pt`。

## 输入和预测范围

直接 head 接收共同文档编码器的 `h[B,T,H]`、二值 `b0[B,T-1]` 和 `atom_mask[B,T]`，返回最终边界 logits 与明确的 `gap_mask`。每个目标 gap 的输入按固定顺序包含：

1. 当前 gap 的左右表示、差和乘积，共4H维；
2. 偏移 `[-K,K]` 内各 seed gap 的同类表示；非 seed、越界和 padding 项置零；
3. 对应的 `b0` presence bits，保留偏移位置。

因此它读取当前文本以及能够移动到该位置的初始边界表示，区别于旧分类器只读取当前 gap 文本。它不输出编辑动作，也不运行 DP。这里的 `h` 已含 DocEncoder 上下文，不能把 head 的K窗口解释为原文只可见K个atom。

该接口匹配原始输入入口和局部可达 seed 表示。编辑 softmax 会涉及其他候选，单调 DP 又耦合全局动作；新 head 不复现这些依赖，因此不称为所有信息依赖完全等价。后续比较须明确这些架构差异及共同的可见文本、编码器与投影政策。

缓存 wrapper 只使用调用方显式传入的 embedding tensor，经现有 DocEncoder 后调用新 head。输入缓存会 detach，DocEncoder/head 保持可训练；它不实例化 BGE、未使用的编辑 head、optimizer 或文件读取器。配对实验仍须显式复制/绑定共同的 context 初始状态，**给两个构造器相同随机种子本身不保证相同 context 初值**。

## 输入保护与验证

真实 atom 必须有限，mask 必须为非空连续True前缀；b0只接受bool/整数0或1，所有无效gap必须为0。padding允许NaN/Inf，但在差、乘积和Linear之前清零。空batch、零atom或全padding文档拒绝；单atom允许返回空gap。文档超出显式 `max_doc_atoms` 时拒绝，不静默截断。无效logit使用有限的最小值；损失与预测仍必须使用返回mask。

[`test_direct_boundary_baseline.py`](../../tests/research/test_direct_boundary_baseline.py) 的29项检查覆盖精确特征布局、固定h下改变b0、padding巨大值/NaN/Inf的前向与梯度隔离、混合长度/单独/置换batch、单atom、非法输入、局部依赖范围、缓存detach及参数梯度路径。独立审查另用逐槽oracle和两个小型CPU检查核对窗口与wrapper，没有复用测试作为唯一依据。

测试用的h、embedding和目标均为人工张量，不读取自然文档或旧标签，不运行优化器步骤。三条PyTorch提示仅说明现有 `norm_first` 设置禁用了nested-tensor优化；没有测试失败。有限输入检查也不是对任意巨大有限值的算术溢出保证。

## 参数与梯度核算

固定 `H=128,K=6` 时，输入为7,181维。单MLP的参数数为 `width*(input_dim+2)+1`；在观察质量结果之前选定width37，它是接近编辑head参数预算的整数宽度，不通过数据调参。

[`probe_direct_boundary_baseline.py`](probe_direct_boundary_baseline.py) 用同一批两个9-atom的随机CPU表示和人工全零最终边界，分别作一次前向/反向。编辑目标为全部DEL、无INSERT；两个直接分支使用最终边界BCE。使用单位权重的结构检查损失，不声称复现默认训练recipe。

| 分支 | head保存参数 | 本次有梯度张量所包含的参数数 |
|---|---:|---:|
| 编辑head | 262,914 | 262,914 |
| 旧直接分类，仅使用insert logits | 262,914 | 65,793 |
| 新b0条件直接head | 265,772 | 265,772 |

旧分支有197,121个保存参数没有该损失的梯度；新head所有参数张量均有有限且至少一个非零梯度。表中计数不等于每个元素都取得非零梯度，完整逐张量记录见[机器结果](results/direct_boundary_baseline_20261004.json)。这三次检查没有optimizer step。

同配置DocEncoder计数329,984，因此新完整组件595,756参数，相对旧编辑模型592,898多2,858：head约多1.09%，包括共同context约多0.48%。这是容量近似，不是相同有效表达能力或计算成本。H128 wrapper在计数探针中仅构造；其前向/反向行为由另行的小尺寸合成测试检查，未运行自然长度的H128模型。

## 接下来可做什么

本轮提供了一个可检查、显式使用b0的直接预测组件，没有证明Refiner或新基线更好。此前已保存的负结果、checkpoint缺失和64/128长度差异继续有效。

下一步可建立很小的缓存特征对照：固定共同context初值、训练/投影预算、原始与投影后指标及全部预定种子，并单独审查现有小样本的标签与来源边界。沿用旧微诊断样本只允许回答局部优化问题，不能冒充独立质量确认。无需因此启动BGE重编码、大数据扫描或900题付费实验。
