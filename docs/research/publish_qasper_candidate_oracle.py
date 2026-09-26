"""Publish complete root-released candidate-oracle aggregates, without scoring.

Never runs the oracle/audit, loads QA, reads per-question content, or calls models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import run_qasper_candidate_oracle as experiment

SCHEMA = "slac-qasper-candidate-oracle-publication-v1"
RELEASE = "complete_oracle_reviewed_for_publication"
RELEASE_SCHEMA = "slac-qasper-candidate-oracle-publication-release-v1"
WEIGHTS = ("question_weighted", "family_balanced")
COUNTS = ("oracle_empty_questions", "any_empty_reference_questions", "actual_witness_dominated_or_tied_questions")
PAIR_COUNTS = ("question_positive", "question_ties", "question_negative", "question_wins", "question_losses")
AUDIT = {"status": "verified", "records": 462, "all_subsets_enumerated": True,
    "exact_fraction_ties_replayed": True, "all_actual_witnesses_validated": True,
    "all_four_comparisons_recomputed": True, "oracle_only": True, "api_calls": 0,
    "gpu_used": False, "model_inference_performed": False, "answer_generation_performed": False}
FIXED = {"schema": experiment.SCHEMA, "status": "completed", "config": experiment.CONFIG,
    "limits": experiment.LIMITS, "question_count": 77, "family_count": 24, "record_count": 462,
    "api_calls": 0, "gpu_used": False, "model_inference_performed": False,
    "answer_generation_performed": False, "test_payload_read": False,
    "raw_text_or_question_ids_in_public_output": False, "all_actual_witnesses_validated": True,
    "all_subsets_enumerated": True}
LABELS = {"leaf_direct": "Direct", "leaf_owner": "Leaf owner", "dual_owner": "Dual owner"}


def exact(value, expected):
    if type(value) is not type(expected) or value != expected:
        raise ValueError("complete publication prerequisite differs")


def number(value, lower=0, upper=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("invalid finite aggregate value")
    return value


def integer(value, upper=77, lower=0):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("invalid aggregate count")
    return value


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a digest, not private source text")
    return value


def project(public, audit, elapsed):
    """Validate aggregate arithmetic and copy only the fixed public schema."""
    for key, value in AUDIT.items(): exact(audit.get(key), value)
    for key, value in FIXED.items(): exact(public.get(key), value)
    output = dict(FIXED)
    output["input_binding_sha256"] = sha(public["input_binding_sha256"])
    output["shared_bootstrap_draws_sha256"] = sha(public["shared_bootstrap_draws_sha256"])
    stats = public["subset_accounting"]
    output["subset_accounting"] = {k: integer(stats[k], 311028) for k in (
        "enumerated_subsets", "duplicate_rejected_subsets", "overbudget_subsets", "feasible_subsets")}
    output["subset_accounting"]["token_cache_entries"] = integer(stats["token_cache_entries"], 150461, 1)
    exact(stats["enumerated_subsets"], 311028)
    if sum(stats[k] for k in ("duplicate_rejected_subsets", "overbudget_subsets", "feasible_subsets")) != 311028 or stats["feasible_subsets"] < 462:
        raise ValueError("complete subset feasibility partition differs")

    rows = public["methods"]
    if [(r.get("scope"), r.get("method")) for r in rows] != [(s,m) for s in experiment.SCOPES for m in experiment.METHODS]:
        raise ValueError("all six methods in frozen order are required")
    methods = []
    for row in rows:
        exact(row.get("questions"), 77); exact(row.get("families"), 24)
        result = {"scope":row["scope"], "method":row["method"], "questions":77, "families":24, "metrics":{}}
        for metric in experiment.METRICS:
            upper = 1024 if metric.endswith("_tokens") else 3 if metric == "oracle_selected_units" else 1
            result["metrics"][metric] = {w:number(row["metrics"][metric][w], upper=upper) for w in WEIGHTS}
        for key in COUNTS: result[key] = integer(row[key])
        exact(result["actual_witness_dominated_or_tied_questions"],77)
        if result["oracle_empty_questions"] < result["any_empty_reference_questions"]:
            raise ValueError("every empty-reference question must admit optimal empty evidence")
        for weight in WEIGHTS:
            metrics = result["metrics"]
            actual = metrics["actual_source_qualified_evidence_f1"][weight]
            oracle = metrics["oracle_source_qualified_evidence_f1"][weight]
            gap = metrics["oracle_minus_actual_f1"][weight]
            if oracle < actual - 1e-12 or abs(gap-(oracle-actual)) > 1e-12:
                raise ValueError("oracle actual-witness dominance/gap does not reconcile")
        methods.append(result)
    if len({row["any_empty_reference_questions"] for row in methods}) != 1:
        raise ValueError("all six groups must retain the same annotation denominator")
    output["methods"] = methods

    by_method = {(row["scope"],row["method"]):row for row in methods}
    rows = public["comparisons"]
    if [(r.get("scope"),r.get("plus"),r.get("minus")) for r in rows] != [(s,p,m) for s in experiment.SCOPES for p,m in experiment.PAIRS]:
        raise ValueError("all four fixed signed comparisons are required")
    pairs = []
    for row in rows:
        exact(row.get("questions"),77); exact(row.get("families"),24)
        exact(row.get("metric"),experiment.CONFIG["primary_metric"])
        result = {k:row[k] for k in ("scope","plus","minus","metric","questions","families")}
        for key in PAIR_COUNTS: result[key] = integer(row[key])
        if (sum(result[k] for k in ("question_positive","question_ties","question_negative")) != 77
                or result["question_wins"] != result["question_positive"] or result["question_losses"] != result["question_negative"]):
            raise ValueError("paired win/tie/loss counts do not reconcile")
        for weight in WEIGHTS:
            delta = number(row[weight]["delta"],-1,1)
            ci = row[weight]["bootstrap_percentile_95"]
            if not isinstance(ci,list) or len(ci) != 2: raise ValueError("two CI endpoints required")
            ci = [number(value,-1,1) for value in ci]
            if ci[0] > ci[1]: raise ValueError("reversed CI")
            expected = (by_method[row["scope"],row["plus"]]["metrics"][row["metric"]][weight]
                - by_method[row["scope"],row["minus"]]["metrics"][row["metric"]][weight])
            if abs(delta-expected) > 1e-12: raise ValueError("paired delta differs from six-group means")
            result[weight] = {"delta":delta,"bootstrap_percentile_95":ci}
        pairs.append(result)
    output["comparisons"] = pairs
    output["publication"] = {"schema":SCHEMA,"status":"complete_audited_development_results",
        "run_elapsed_seconds":number(elapsed,0,1200),
        "timing_scope":"run function including source verification, original native audit, oracle enumeration and aggregation; not pure oracle kernel time or an external hard-kill guarantee",
        "new_api_calls":0,"new_api_cost_usd":"0","night_cost_accounting_unchanged":True,
        "prior_unknown_charge_resolved_by_this_diagnostic":False,
        "publication_performed_scoring":False,"publication_read_per_question_content":False}
    return output


def interval_text(values):
    low, high = values["bootstrap_percentile_95"]
    return f"{values['delta']:+.6f} [{low:+.6f}, {high:+.6f}]"


def markdown(public):
    given=[row for row in public["methods"] if row["scope"]=="given_document"]
    actual=" / ".join(f"{r['metrics']['actual_source_qualified_evidence_f1']['question_weighted']:.3f}" for r in given)
    optimal=" / ".join(f"{r['metrics']['oracle_source_qualified_evidence_f1']['question_weighted']:.3f}" for r in given)
    given_pairs=[row for row in public["comparisons"] if row["scope"]=="given_document"]
    all_cross=all(row[w]["bootstrap_percentile_95"][0]<0<row[w]["bootstrap_percentile_95"][1] for row in given_pairs for w in WEIGHTS)
    interval_note="两项上界增量（leaf owner−direct、dual owner−leaf owner）的双权重区间均跨 0。" if all_cross else "两项上界增量的完整双权重区间列于下表。"
    corpus_dual=next(row for row in public["comparisons"] if row["scope"]=="corpus_32" and row["plus"]=="dual_owner")
    lower_zero=all(corpus_dual[w]["bootstrap_percentile_95"][0]==0<corpus_dual[w]["bootstrap_percentile_95"][1] for w in WEIGHTS)
    corpus_note="其双权重区间下界恰为 0，" if lower_zero else "其完整双权重区间保留如下，"
    lines = ["# 固定候选、同预算的精确 evidence oracle 结果", "",
        f"给定文档时，direct / leaf owner / dual owner 的问题等权 actual Evidence F1 为 **{actual}**，同候选、同预算 oracle 为 **{optimal}**。{interval_note}这说明当前候选池存在选择空间，尚不能认定 owner 候选扩展带来稳定的额外可达收益。", "",
        f"跨库 dual owner−leaf owner 为 {corpus_dual['question_positive']} 增、{corpus_dual['question_ties']} 同、{corpus_dual['question_negative']} 减；{corpus_note}不据此宣称显著或稳定正收益。", "",
        "本轮完整覆盖既有 77 个开发问题、24 个 family、六组冻结候选。它用参考答案寻找同候选、同预算的可达上界；gold 参与选集，不能当成可部署检索表现、Answer F1、独立确认或 JEV/完整 SLAC 的收益。旧 104 题 subset-oracle 不进入比较。", "",
        "本报告仅在完整运行、root 完整 CPU 重放审计与发布来源校验通过后生成。源码、测试与协议先于新 oracle 分数冻结；逐题参考、选集与原始身份只留在 ignored artifacts。", "",
        "## 六组完整结果", "",
        "下表每格依次为问题等权 / family 等权。Candidate 是全部冻结候选的 recall；actual 是原部署式选集；oracle 是 gold-guided 最优合法子集。三者分别解释。", "",
        "| Scope | 方法 | Candidate recall | Actual F1 | Actual recall | Oracle F1 | Oracle recall | Oracle−actual F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    quality = experiment.METRICS[:6]
    for row in public["methods"]:
        cells = [f"{row['metrics'][m]['question_weighted']:.6f} / {row['metrics'][m]['family_balanced']:.6f}" for m in quality]
        lines.append(f"| {row['scope']} | {LABELS[row['method']]} | " + " | ".join(cells) + " |")
    lines += ["", "| Scope | 方法 | Actual tokens QW / FB | Oracle tokens QW / FB | Oracle 单元 QW / FB | Oracle empty / 77 | 有 empty reference / 77 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in public["methods"]:
        cells = [f"{row['metrics'][m]['question_weighted']:.3f} / {row['metrics'][m]['family_balanced']:.3f}" for m in experiment.METRICS[6:]]
        lines.append(f"| {row['scope']} | {LABELS[row['method']]} | " + " | ".join(cells)
            + f" | {row['oracle_empty_questions']} | {row['any_empty_reference_questions']} |")
    lines += ["", "所有 462 行 actual 选集均先验证属于原候选、同源同文不重复、完整渲染一致、最多三项且实际 BGE tokens ≤1,024，再作为 oracle 可行域见证；oracle F1 每行均不低于 actual。相同上限不代表相同实际长度，上表不做长度匹配声明。", "",
        "任一 empty reference 可让 empty evidence 的官方 F1 和 recall 同时为 1；当所有预算合法子集的 F1 均为 0 时，最少 tokens 的 tie 规则也会选择 empty。因此 oracle empty 数不等于不可回答问题数，也不代表可部署弃答识别能力。重复 reference 项保留原 list-length 分母，FLOAT evidence 不删除；candidate recall 不能在这些语义下直接充当 oracle recall 的单调上界。", "",
        "## 四项预固定配对", "",
        "仅比较 oracle source-qualified Evidence F1。24 个 family 整簇 PCG64 seed 20260927、10,000 次共享抽样、双侧线性 percentile 95% CI；全部方向与双权重保留，不做多重比较校正。", "",
        "| Scope | Plus − minus | 问题等权 Δ [95% CI] | Family 等权 Δ [95% CI] | 增 / 同 / 减 |", "|---|---|---:|---:|---:|"]
    for row in public["comparisons"]:
        lines.append(f"| {row['scope']} | {LABELS[row['plus']]} − {LABELS[row['minus']]} | {interval_text(row['question_weighted'])} | {interval_text(row['family_balanced'])} | {row['question_positive']} / {row['question_ties']} / {row['question_negative']} |")
    stats = public["subset_accounting"]
    lines += ["", "更大 oracle 上界只意味着该冻结候选内存在更好的 gold-guided 选择；不能据此推断实际 selector 或答案质量必然提升。Oracle−actual gap 是每组选择损失的描述，不是实际可达收益预测。", "",
        "## 完整搜索与资源", "",
        "| 组合核对项 | 数量 |", "|---|---:|",
        f"| 全部枚举，包含各行 empty | {stats['enumerated_subsets']:,} |",
        f"| 同源同文重复而不合法 | {stats['duplicate_rejected_subsets']:,} |",
        f"| 超过实际 1,024-token 预算 | {stats['overbudget_subsets']:,} |",
        f"| 合法可评分子集 | {stats['feasible_subsets']:,} |",
        f"| 实际 token cache 条目 | {stats['token_cache_entries']:,} |",
        "| 全局不同 index 子集上限 | 150,461 |", "",
        "三个合法性类别之和等于 311,028；每行至多 697 个子集。穷举所有大小 0–3 的候选组合，保留同文不同位置作为互斥备选，不只搜索 reference 匹配项。选优顺序为精确 Fraction F1、独立 max-reference recall、最少实际 tokens、global-index 字典序。只共享整包 token 缓存，不跨问题共享 gold 分数。", "",
        f"run 函数记录总耗时 **{public['publication']['run_elapsed_seconds']:.3f} 秒**，包含来源校验、原 native CPU 审计、新 oracle 枚举与聚合；不是纯枚举内核耗时，也不含解释器启动。期限为 1,200 秒，按检查点失败退出，并非外部强制杀进程保证；本结果通过完成与时间门槛。独立重放审计的额外时间未合并进该数。", "",
        "枚举串行、PyTorch CPU threads=1；NumPy/BLAS 线程数未单独强制。新增 API 调用与费用均为 0，无 GPU 或模型推理。此诊断不改变夜间费用账本，也没有解决原来的未知收费请求。", "",
        "## 解释边界", "",
        "这是已经暴露的开发集上的事后机制诊断。Qasper 原任务给定论文；corpus_32 query-only 是额外压力测试，不能仅据下降推断标准 Qasper 或完整 SLAC 架构失败。跨文档 header 歧义仍在，任何 oracle pack 均不得送入答案生成、选新部署方法或回写 retriever。", "",
        "完整浮点精度、四项比较与来源 hash 保留在 [公开聚合 JSON](results/qasper_candidate_oracle_20260927.json)。", ""]
    return "\n".join(lines)


def publish(args):
    release_path=Path(args.receipt).resolve()
    release_bytes=release_path.read_bytes(); release=json.loads(release_bytes)
    exact(release.get("schema"),RELEASE_SCHEMA)
    exact(release.get("root_release"),RELEASE)
    bindings=release["input_sha256"]
    paths={name:Path(getattr(args,name)).resolve() for name in ("plan","run","audit")}
    required=[Path(__file__).resolve(),Path(__file__).resolve().parents[2]/"tests/research/test_publish_qasper_candidate_oracle.py",
        Path(experiment.__file__).resolve(),Path(experiment.__file__).resolve().with_name("CANDIDATE_ORACLE_PROTOCOL_20260927.md"),
        paths["audit"],*(paths["plan"]/name for name in ("plan.json","plan_seal.json")),
        *(paths["run"]/name for name in experiment.RUN_FILES)]
    if not set(map(str,required)) <= set(bindings): raise ValueError("root release must bind all publication prerequisites")
    experiment.pilot.verify_hashes(bindings)
    plan,plan_hashes=experiment.load_plan(paths["plan"])
    if Path(plan["run_output"]).resolve()!=paths["run"]: raise ValueError("publication run differs from frozen output")
    if set(p.name for p in paths["run"].iterdir())!=set(experiment.RUN_FILES): raise ValueError("only a complete oracle output inventory can publish")
    buffers={name:(paths["run"]/name).read_bytes() for name in ("summary.json","public_aggregate.json")}
    summary=json.loads(buffers["summary.json"]); raw=json.loads(buffers["public_aggregate.json"])
    if (summary["schema"]!=experiment.SCHEMA or summary["status"]!="completed" or summary["plan_sha256"]!=plan_hashes
            or summary["input_sha256"]!=plan["input_sha256"] or summary["public"]!=raw
            or summary["output_sha256"]!={name:bindings[str(paths["run"]/name)] for name in experiment.RUN_FILES[:-1]}
            or raw["input_binding_sha256"]!=plan["input_binding_sha256"]):
        raise ValueError("published aggregate differs from audited plan/run seals")
    result=project(raw,json.loads(paths["audit"].read_bytes()),summary["elapsed_seconds"])
    result["publication"]["source_release_sha256"]=hashlib.sha256(release_bytes).hexdigest()
    result["publication"]["source_manifest_sha256"]=experiment.pilot.stable_hash(bindings)
    result["publication"]["plan_sha256"]=bindings[str(paths["plan"]/"plan.json")]
    result["publication"]["audit_sha256"]=bindings[str(paths["audit"])]
    result["publication"]["summary_sha256"]=bindings[str(paths["run"]/"summary.json")]
    report=markdown(result)
    experiment.pilot.verify_hashes(bindings);experiment.pilot.verify_hashes({str(release_path):hashlib.sha256(release_bytes).hexdigest()})
    json_path,report_path=Path(args.output_json).resolve(),Path(args.output_report).resolve()
    if json_path==report_path or json_path.exists() or report_path.exists(): raise FileExistsError("publication output must be new")
    for path in (json_path,report_path): path.parent.mkdir(parents=True,exist_ok=True)
    experiment.pilot.write_json(json_path,result)
    with report_path.open("x",encoding="utf-8") as stream: stream.write(report)
    return {"status":"published_complete_aggregate","question_count":77,"methods":6,"comparisons":4,
        "output_sha256":{"public_json":experiment.native.digest(json_path),"report":experiment.native.digest(report_path)}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","run","audit","receipt","output-json","output-report"): parser.add_argument("--"+name,required=True)
    print(json.dumps(publish(parser.parse_args()),indent=2))
