# 局部关系 pilot 实现与执行进展

日期：2026-09-26。上一阶段 `7e013cc933807a109dd7026604df0fe13b8da2e9` 已推送并重新核对远端。本阶段继续在隔离分支 `codex/research-foundation-20260926` 实现 [锁定协议](RELATION_PILOT_PROTOCOL_20260926.md)。原工作目录、迁移数据及凭据文件未改写。

## 当前执行状态

真实模型推理 **尚未开始，付费调用 0 次**。用户指定的 `apikey.txt` 未检测到 OpenRouter 格式的 key，当前进程也没有 `OPENROUTER_API_KEY`；已请求提供正确的本地路径或更新该文件。没有将未知格式的凭据发送到服务商验证，也没有复制或发布凭据。拿到可用文件后可以执行同一冻结计划，无须重新选题。

已经完成的内容：

- `prepare_qasper_relation_pilot.py`：按固定 hash 从现有开发池冻结 8 family、15 个问题，生成 139 个静态关系和 238 个 query-unit 判断；输入白名单、邻接关系、排序与来源 hash 均校验。
- `openrouter_decision_client.py`：JEV choice 与 GPT-4.1 Mini JSON Schema 使用相同 state/questions；限制供应商和价格；请求前费用预留；响应版本/ID/usage 校验；无自动重试；重复请求拒绝；所有落盘渠道脱敏。
- `qasper_relation_replay.py`：独立/缓存策略等价回放，共享模式以同一组通过分组预算的依赖边影响分组与选择，记录两阶段 trace 和真实选集差异。
- `run_qasper_relation_pilot.py`：离线冻结请求、代码与输入清单；实际运行只允许绑定计划；失败保留账本与输出，标签不完整时不生成伪造的六格分数。

完整 public model endpoint 元数据与请求预检均已本地保存。53 个共用批次对应两个后端的 106 次预定请求、754 个判断；不是已经发出的调用，也不是六套独立冷/热实跑。I/C/S 将复用每后端的唯一判断结果。

## 已执行的离线检查与基线

全套 `SLAC/refiner/tests` 与 `tests/research`：**249 passed，8 warnings**，测试时间约 3.50 秒。warning 是此前已有的 PyTorch `norm_first` / nested-tensor 优化提示。新增测试包括使用假 transport 的完整编排、第二次调用失败即停机、无残缺分数，以及运行中同时改写配置和 seal 仍被初始 hash 捕获；这些测试不是模型实测。

真实开发数据的 `plan` 已完成，绑定 25 项输入/脚本和 4 份计划产物；冻结配置 SHA256 为 `3239e393b73e8a55cf5159e940e2e9ea4623bb9119cc0c7c366f34f1c444d1ea`。全部 106 个请求通过 payload 白名单、字节上限和预算准入；保守总预留 **$1.0337814010**，实际付费仍为零。

以下均针对这 **15 题**，最终上限 1,024 BGE tokens、最多 3 个原生单元。列出的 oracle 使用参考证据，不能部署：

| 方法 | Evidence F1：问题宏平均 | 文档宏平均 | 平均实际 tokens |
|---|---:|---:|---:|
| 原文完整候选 dense top-3 | 0.2057 | 0.1929 | 437.1 |
| 本次受限候选 dense top-3 | 0.2057 | 0.1929 | 437.1 |
| 空证据 | 0.2000 | 0.2500 | 0 |
| 本次候选 gold subset oracle，最多 3 单元 | 0.6444 | 0.6667 | 99.3 |

候选缩小在这批题上没有改变 dense top-3 的成绩；已标注证据仍存在可选择空间。空参考与不可回答题影响宏平均，不能忽略空证据基线。Oracle 的实际长度显著更短，而且样本/单元上限与上一阶段不同；这不是 JEV 提升或必要充分证据的证明。所有后端/共享策略成绩仍为 **pending**。

写盘后另行复核 60 条 baseline 记录：身份唯一、summary 可完整重算；从保存的 selected IDs 重新渲染计数，全部满足 1,024-token 与最多 3 单元上限；冻结输入及代码 hash 再次通过。

本地产物（原始内容不随 Git 发布）：

- [冻结数据 manifest](../../artifacts/research-foundation/qasper-relation-prepared-01/manifest.json)
- [冻结运行配置](../../artifacts/research-foundation/qasper-relation-plan-01/experiment_config.json)
- [离线基线](../../artifacts/research-foundation/qasper-relation-plan-01/baseline_summary.json)
- [测试记录](../../artifacts/research-foundation/phase3-verification/pytest.txt)

## 本地复现

在研究工作树根目录使用：

```powershell
$py = 'C:/Environment/python/venvs/slac-research/Scripts/python.exe'
$tok = 'D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
& $py -X utf8 -m pytest SLAC/refiner/tests tests/research -q
& $py -X utf8 docs/research/prepare_qasper_relation_pilot.py `
  --pool artifacts/research-foundation/qasper-pool `
  --sidecar artifacts/research-foundation/qasper-alignment-v2/native_qa_sidecar_v2.jsonl `
  --dense artifacts/research-foundation/qasper-dense-01 `
  --output artifacts/research-foundation/qasper-relation-prepared-new
& $py -X utf8 docs/research/run_qasper_relation_pilot.py plan `
  --prepared artifacts/research-foundation/qasper-relation-prepared-new `
  --sidecar artifacts/research-foundation/qasper-alignment-v2/native_qa_sidecar_v2.jsonl `
  --tokenizer $tok --output artifacts/research-foundation/qasper-relation-plan-new
```

所有输出目录必须不存在。模型调用入口（以下为使用者提供本地路径后的命令模板，未执行）：

```powershell
& $py -X utf8 docs/research/run_qasper_relation_pilot.py run `
  --plan artifacts/research-foundation/qasper-relation-plan-new `
  --key-file '<LOCAL_OPENROUTER_KEY_FILE>' `
  --proxy http://127.0.0.1:7897 `
  --output artifacts/research-foundation/qasper-relation-run-new
```

`--proxy` 是本机当前系统代理的可选配置，迁移机器需要按实际网络调整。输入或代码改变会使旧计划拒绝执行，应保留旧计划并生成新计划；不要修改已有 plan 的 seal 来绕过检查。原始模型输出和实际账单返回前，不报告 JEV 分数、架构提升或论文贡献已成立。
