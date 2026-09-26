# 六组本地基线的独立答案评测

本协议在完整本地证据实验之后、答案调用之前固定。它使用全部77个开发问题、24个family，检验证据选择变化是否对应最终回答变化。它是事后开发比较，**不是独立确认，也不是尚未完成的JEV主实验**。原JEV/Qwen support中断现场不变，不读取官方test QA。

## 方法与共同生成器

保留以下六组全部问题，不按证据分数、答案类型或可恢复性筛选：

| 方法 | 证据来源 |
|---|---|
| `dense_k3` | 已审计native实验的given-document `leaf_direct`，与旧固定dense结果一致 |
| `reranker_k3` | 同一原候选池的BGE cross-encoder固定k=3；不替换为较高分的k=2 |
| `bm25_k3` | 固定BM25 given-document结果，使用其自己的已审计候选池 |
| `leaf_owner_k3` | 原单通道owner聚合及投影结果 |
| `dual_owner_k3` | 原双通道、相同规则分区的owner聚合结果 |
| `empty` | 空证据，仍实际调用同一生成器 |

不把随后顺序诊断的改进版本替换进本轮方法。五组非空证据均来自完整原生单元，最多3个单元、1,024个实际BGE evidence tokens；各组真实长度不同，不能声称长度匹配。BM25和owner候选不必属于原support池，但必须属于各自已审计候选；适配器逐一校验来源文档、global/native身份、完整渲染、pack hash及真实tokens。

固定使用现有`slac-qasper-answer-v1`提示词和[原答案客户端](run_qasper_answer_evaluation.py)：`qwen/qwen3.6-plus`、Alibaba only、禁止fallback、temperature 0、reasoning关闭、JSON object，最多512输出tokens。输入只有相同问题和相应可见证据，不额外加入标题、摘要、金标、selector分数或方法名。实际返回模型必须属于冻结允许版本且整批一致。

提示词要求仅依据证据，在信息不足时输出`Unanswerable`。因此empty是无证据回答/弃答对照，**不能解释为不受限制的闭卷知识能力**。输出必须正常结束且严格符合单一字符串`answer`字段；不修补、截取或重写模型答案，截断、拒答或契约失败均停下保存。

按endpoint、prompt版本及完整payload精确去重，462个逻辑预测对应367次请求、95次复用。每个方法仍有77个预测；相同payload共用同一响应，避免把生成随机变化混入相同证据比较。请求按payload hash排序，不依据方法成绩或参考；这种实验去重不等于实测生产缓存或线上速度收益。

## 固定统计与限制

Answer F1沿用已验证的官方Qasper多参考语义，对每个参考计算token F1后取最大值。保留全部答案类型、空参考、图表证据及不可回答问题，不使用模型裁判。六组均报告问题等权、family等权、实际证据tokens和`Unanswerable`预测数；官方其他指标作为描述性补充。

固定六个配对：reranker−dense、BM25−dense、leaf_owner−dense、dual_owner−dense、dense−empty、dual_owner−leaf_owner。全部报告差值、增/同/减计数、问题等权和family等权区间。复用PCG64 seed 20260927的10,000次整family重采样；双侧线性percentile 95%，无多重比较校正。不能只报告正向方法或单一权重。

同一批已暴露的开发数据、单个生成器及每个独特payload单次响应，不能推断跨数据泛化、生成器间稳定性或运行间方差。Evidence F1与Answer F1可以不同；对本地方法的结论不代表JEV能力、共享关系或完整SLAC收益。

## 费用、单次执行与审计

| 账目 | 美元 |
|---|---:|
| 旧support实际165次尝试的保守预留，含1笔未知 | 1.4692631425 |
| 本轮367次答案请求的新增计划预留 | 1.3374991500 |
| 完成本轮时预计累计预留 | 2.8067622925 |
| 本夜总预留上限 | 5.0000000000 |

新调用总input allowance为1,616,164，output allowance为187,904；价格及1.5倍预留规则沿用原客户端。以上是保守额度，不是实际收费。旧support已知费用$0.212802982另有1笔未知，不能当成全部结清；新增实际费用和未知项待真实响应记录。旧165次尝试的来源与审计账本直接绑定，不依赖可变夜间状态，也不因换目录清零。

计划固定run目录并一次性注册；每次请求发送前保存预留和attempt，另保留只追加的ledger快照及hash链。09:00 Asia/Shanghai之后不开始新付费请求。失败或不确定结果不自动重试、不退预留、不换provider；全量未完成时只允许费用与来源前缀审计，不输出部分质量成绩。

计划、运行及审计由[独立适配器](run_qasper_local_answer_evaluation.py)执行，不修改原support/答案链。完成后从原响应独立重建所有答案、完整462条逻辑预测、官方评分和配对统计。原始问题、文本、请求响应、逐题答案和key留本地忽略目录；公开仅完整聚合、代码及限制。运行状态和结果以[夜间记录](OVERNIGHT_RESEARCH_20260927.md)及本地账本为准，协议本身不表示调用已经完成。

## 执行前封印

有效计划为`qasper-local-answer-plan-02`，配置SHA-256为`ccc215ebe4ac9aceadb312d7f66a850791d3b6da4ca02d08e6e131996960998d`，绑定442个来源文件，输出固定为`qasper-local-answer-run-01`。29项新增离线测试及独立审查通过；另一实现直接从三份已审计来源的原生单元重建所有payload和mapping，与计划完全一致。一次性调度器的11项mock检查另验证重复启动、预算、截止、异常子进程清理与状态处理。

准备期间的plan01没有执行API；在发现人工中断可能留下终末`in_flight`事件后，先补齐审计及禁止续发测试，再另生成plan02。旧plan01保留为未执行的准备记录，源码绑定已经过期，不能启动它。两计划的jobs/mapping字节完全相同，科学方法、分母和费用未变。[公开封印摘要](results/qasper_local_answer_protocol_20260927.json)
