# 局部编辑可见性：四个 atom 的零模型见证

2026-10-05。只检验当前接口的依赖关系，不检验自然任务效果。固定一个人工文档与两项编辑，不读取自然数据，不追加样本或变体。所有结果保留。

固定 atoms 为 `Amber hardware is available.`、`Blue services are available.`、`Cobalt services are available.`、`Delta hardware is available.`。三个来源 unit 的 atom 半开区间分别为 [0,1)、[1,3)、[3,4)，路径分别为 Alpha、Beta、Gamma。baseline 的稀疏 gap 为 [2]；split 为 [0,2]，仅增加 gap 0；merge 为 []，仅移除 gap 2。原输入 b0 固定为 [0,0,1]，全程不改变 atom 文本。

执行真实的默认规范化 exporter、JSONL reader、enrich_all_records 和 leaf/chunk 检索文本 compose 函数。先保存 fixture 与全部导入仓库源码哈希，再执行这三种分区。向量后端依赖只在实际建索引时导入，使 compose 可直接调用；其函数正文保持不变，不使用 AST 执行替身。进程拒绝模型／数值后端导入、socket 连接与子进程启动。

记录每个 chunk 的 span、ID、路径和完整 encoder 输入；逐 leaf 记录原文、ID、owner 与完整 encoder 输入。分别比较 baseline→split、baseline→merge 的相等／变化，不将正文或 ID 相同自动解释为向量可复用。

另对三个分区各自的全部 chunks 强制构造 EvidenceItem，调用现有完整 renderer。这里没有检索、重排、选包、预算计数或答案生成，不能把这组人为全选输出描述成真实检索结果。比较规范化全文串联与完整模型可见字符串是否相等；再单独比较 baseline/split 中同一个 [3,4) 尾块，检查正文相同但 ID 变化的影响。该见证仅覆盖默认规范化导出和明确给定的 renderer 字段，不认证可选 source_document 输出模式或完整 HTTP 请求相等。

API、训练、模型、tokenizer、自然参考和真实数据读取均为零。不同输入不保证向量、排序或答案一定不同；相同输入只有在 encoder 及预处理配置相同时才满足按内容复用的必要前提。索引身份、owner 映射、邻接与语料统计仍需另行核对。下一步由源码审计和本见证共同决定公平干预的位置，不据此扩展训练。
