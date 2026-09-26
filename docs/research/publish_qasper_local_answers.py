"""Publish only complete, independently released local-baseline answer aggregates.

No provider calls, credential reads, model inference, partial-result scoring or
per-question reads. The root task must first save its complete audit and release.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import run_qasper_local_answer_evaluation as experiment


SCHEMA = "slac-qasper-local-answer-publication-v1"
ROOT_RELEASE = "complete_run_reviewed_for_publication"
MEAN_FIELDS = ("official_answer_f1_question_weighted", "answer_f1_family_balanced",
    "actual_evidence_tokens_question_weighted", "actual_evidence_tokens_family_balanced")
PAIR_FIELDS = ("question_positive", "question_ties", "question_negative", "question_wins", "question_losses")
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")
LABELS = {"dense_k3": "Dense", "reranker_k3": "BGE reranker", "bm25_k3": "BM25",
    "leaf_owner_k3": "Leaf owner", "dual_owner_k3": "Dual owner", "empty": "Empty evidence"}
LIMITS = [
    "All 77 questions and 24 families are exposed development data; this is a posthoc comparison, not independent confirmation.",
    "All six methods use one frozen generator and one response per unique complete payload; this does not estimate run-to-run or cross-generator variation.",
    "Evidence has the same maximum unit and token caps but differs in actual length; the comparisons are not length matched.",
    "Empty evidence is an evidence-only abstention control, not unrestricted closed-book question answering.",
    "All six fixed contrasts and both family-bootstrap weightings are retained; intervals are descriptive and have no multiple-comparison adjustment.",
    "Official answer-type fields assign each prediction to its best-scoring reference; type denominators may differ across methods and are not fixed-subset comparisons.",
    "These local-baseline answers do not complete the stopped JEV main experiment and establish no JEV or full-SLAC gain.",
    "Exact payload sharing is an experimental control, not measured production cache savings.",
    "The prior support timeout remains one unknown charge; known reported cost is a subtotal, not the final night bill.",
]


def exact(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"publication prerequisite differs: {name}")


def number(value, name, lower=0, upper=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"invalid aggregate number: {name}")
    return value


def integer(value, name, lower=0, upper=77):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"invalid aggregate count: {name}")
    return value


def sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a SHA256 digest, not raw source identity")
    return value


def project_summary(summary, audit):
    """Strict public whitelist; this never reads or recomputes question scores."""
    exact(audit.get("status"), "verified_complete", "complete independent audit")
    for key, expected in {"all_bound_inputs_outputs_unchanged": True, "api_calls_by_audit": 0,
            "key_read": False, "partial_quality_metrics_computed": False}.items():
        exact(audit.get(key), expected, key)
    for key, expected in {"schema": experiment.SCHEMA, "status": "completed", "main_results_available": True,
            "question_count": 77, "family_count": 24, "record_count": 462, "new_api_calls": 367,
            "completed_requests": 367, "unknown_generation_cost_attempts": 0,
            "automatic_retries": 0, "independent_confirmation": False, "not_a_jev_primary_result": True,
            "test_payload_read": False, "specification": experiment.SPEC}.items():
        exact(summary.get(key), expected, key)
    if not audit.get("accounting") or any(summary.get(k) != v for k, v in audit["accounting"].items()):
        raise ValueError("complete audit accounting differs from summary")
    prior = summary["prior_night_accounting"]
    for key, value in experiment.PRIOR.items():
        exact(prior.get(key), value, "prior " + key)
    reserve = Decimal(experiment.RESERVATION)
    total = Decimal(experiment.PRIOR["reservation_usd"]) + reserve
    for key, value in {"generation_attempted_reservation_usd": reserve,
            "generation_planned_reservation_usd": reserve, "night_attempted_reservation_usd": total}.items():
        if experiment.legacy.pilot.nonnegative_decimal(summary[key]) != value:
            raise ValueError("complete attempted budget differs")
    known_generation = experiment.legacy.pilot.nonnegative_decimal(summary["known_generation_cost_usd"])
    if known_generation > reserve:
        raise ValueError("reported generation cost exceeds the audited reservation")

    metrics = summary["metrics"]
    if [m.get("method") for m in metrics] != list(experiment.METHODS):
        raise ValueError("all six methods in fixed order are required")
    public_metrics = []
    for row in metrics:
        exact(row.get("questions"), 77, "method question count")
        exact(row.get("families"), 24, "method family count")
        public = {"method": row["method"], "questions": 77, "families": 24}
        for name in MEAN_FIELDS:
            public[name] = number(row[name], name, upper=1024 if name.startswith("actual_") else 1)
        public["unanswerable_predictions"] = integer(row["unanswerable_predictions"], "Unanswerable predictions")
        official = row["official_metrics"]
        exact(official["Missing predictions"], 0, "missing predictions")
        if set(official["Answer F1 by type"]) != set(ANSWER_TYPES):
            raise ValueError("official answer-type aggregate fields differ")
        public["official_metrics"] = {"Answer F1": number(official["Answer F1"], "official F1"),
            "Answer F1 by type": {kind: number(official["Answer F1 by type"][kind], kind) for kind in ANSWER_TYPES},
            "Evidence F1": number(official["Evidence F1"], "official evidence F1"), "Missing predictions": 0}
        if abs(public["official_metrics"]["Answer F1"] - public["official_answer_f1_question_weighted"]) > 1e-12:
            raise ValueError("official aggregate Answer F1 disagrees with question-weighted mean")
        if row["method"] == "empty" and any(public[k] != 0 for k in MEAN_FIELDS if k.startswith("actual_")):
            raise ValueError("empty evidence must have zero evidence tokens")
        public_metrics.append(public)
    by_method = {row["method"]: row for row in public_metrics}
    comparisons = summary["paired_comparisons"]
    if [(r.get("plus"), r.get("minus")) for r in comparisons] != list(experiment.PAIRS):
        raise ValueError("all six fixed signed comparisons are required")
    pairs = []
    for row in comparisons:
        exact(row.get("questions"), 77, "paired question count")
        exact(row.get("families"), 24, "paired family count")
        pair = {"plus": row["plus"], "minus": row["minus"], "questions": 77, "families": 24}
        for name in PAIR_FIELDS:
            pair[name] = integer(row[name], name)
        if (sum(pair[k] for k in ("question_positive", "question_ties", "question_negative")) != 77
                or pair["question_wins"] != pair["question_positive"] or pair["question_losses"] != pair["question_negative"]):
            raise ValueError("paired win/tie/loss denominator differs")
        for weighting, mean_field in (("question_weighted", MEAN_FIELDS[0]), ("family_balanced", MEAN_FIELDS[1])):
            values = row[weighting]
            delta = number(values["delta"], "paired delta", -1, 1)
            interval = values["bootstrap_percentile_95"]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("paired interval must have two endpoints")
            interval = [number(x, "interval endpoint", -1, 1) for x in interval]
            if interval[0] > interval[1]:
                raise ValueError("paired interval endpoints reversed")
            expected_delta = by_method[row["plus"]][mean_field] - by_method[row["minus"]][mean_field]
            if abs(delta - expected_delta) > 1e-12:
                raise ValueError("paired mean difference disagrees with method means")
            pair[weighting] = {"delta": delta, "bootstrap_percentile_95": interval}
        pairs.append(pair)
    return {"schema": SCHEMA, "status": "complete_audited_development_results",
        "question_count": 77, "family_count": 24, "logical_predictions": 462, "unique_requests": 367,
        "exact_payload_reuses": 95, "specification": experiment.SPEC,
        "metrics": public_metrics, "paired_comparisons": pairs,
        "shared_resamples_sha256": sha(summary["shared_resamples_sha256"]),
        "experiment_plan_sha256": sha(summary["plan_sha256"]),
        "audited_input_binding_sha256": sha(summary["input_binding_sha256"]),
        "cost_accounting": {"prior_attempts": 165, "generation_attempts": 367, "night_attempts": 532,
            "prior_known_reported_cost_usd": experiment.PRIOR["known_cost_usd"],
            "prior_unknown_cost_attempts": 1, "generation_known_reported_cost_usd": str(known_generation),
            "generation_unknown_cost_attempts": 0, "night_unknown_cost_attempts": 1,
            "night_known_reported_cost_subtotal_usd": str(Decimal(experiment.PRIOR["known_cost_usd"]) + known_generation),
            "night_actual_total_known": False, "prior_attempted_reservation_usd": experiment.PRIOR["reservation_usd"],
            "generation_attempted_reservation_usd": str(reserve), "night_attempted_reservation_usd": str(total),
            "night_reservation_cap_usd": str(experiment.legacy.NIGHT_CAP)},
        "raw_text_or_question_ids_in_public_output": False, "limits": LIMITS}


def markdown(public):
    lines = ["# 六组本地基线的完整答案评测", "",
        "全部77个开发问题、24个family、六组方法均完成并通过独立审计。462个逻辑预测按完整payload精确去重为367次生成请求；本报告没有部分样本成绩。",
        "", "这是同一批已暴露开发数据上的事后比较，不是独立确认，也没有完成已停止的JEV主实验。原始问题、证据、答案及逐题结果均留在本地忽略目录。",
        "", "固定生成器为Qwen3.6 Plus，Alibaba路由、temperature 0、关闭reasoning、最大512输出tokens，使用同一证据约束提示词。主比较固定k=3、最多1,024个实际BGE证据tokens；完整原生证据的实际长度不同。Empty仍实际调用相同提示词，是无证据弃答对照，不能解释为不受限制的闭卷问答。",
        "", "![六组答案均值及全部固定配对区间](results/qasper_local_answer_evaluation_20260927.svg)",
        "", "## 全部方法", "",
        "Answer F1与下表区间均用0–1尺度。问题等权保留每题相同权重；family等权先在各family内取均值。",
        "", "| 方法 | Answer F1 问题等权 | Answer F1 family等权 | 实际tokens 问题等权 | 实际tokens family等权 | Unanswerable / 77 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in public["metrics"]:
        lines.append(f"| {LABELS[row['method']]} | {row[MEAN_FIELDS[0]]:.6f} | {row[MEAN_FIELDS[1]]:.6f} | {row[MEAN_FIELDS[2]]:.3f} | {row[MEAN_FIELDS[3]]:.3f} | {row['unanswerable_predictions']} |")
    lines += ["", "## 固定配对比较", "",
        "全部六项比较和两种权重均保留，包括负值和跨零区间。采用PCG64 seed 20260927、10,000次整family重采样、双侧线性percentile 95%区间；不做多重比较校正。这些区间是描述性结果，不代表独立确认或运行间方差。",
        "", "| 比较（前者−后者） | 问题等权 Δ [95%区间] | family等权 Δ [95%区间] | 增 / 同 / 减 |",
        "|---|---:|---:|---:|"]
    for row in public["paired_comparisons"]:
        displays = []
        for weight in ("question_weighted", "family_balanced"):
            v = row[weight]; lo, hi = v["bootstrap_percentile_95"]
            displays.append(f"{v['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
        lines.append(f"| {LABELS[row['plus']]} − {LABELS[row['minus']]} | {displays[0]} | {displays[1]} | {row['question_positive']} / {row['question_ties']} / {row['question_negative']} |")
    cost = public["cost_accounting"]
    lines += ["", "## 费用与未知项", "",
        "费用表区分保守预留与已知实际收费。旧support的1笔超时收费仍未知；新生成全部已知不代表本夜总账已结清。",
        "", "| 阶段 | 尝试数 | 已知收费（美元） | 未知收费尝试 | 保守预留（美元） |", "|---|---:|---:|---:|---:|",
        f"| 旧support | 165 | {cost['prior_known_reported_cost_usd']} | 1 | {cost['prior_attempted_reservation_usd']} |",
        f"| 本轮生成 | 367 | {cost['generation_known_reported_cost_usd']} | 0 | {cost['generation_attempted_reservation_usd']} |",
        f"| 本夜合计 | 532 | 已知小计 {cost['night_known_reported_cost_subtotal_usd']} | 1 | {cost['night_attempted_reservation_usd']} |",
        "", "本夜保守预留上限为$5，未把未知费用视为零，也未因新建目录重置旧账。",
        "", "## 解释边界", "",
        "Answer F1按官方Qasper的逐参考token F1取最大值，不使用模型裁判。全部答案类型保留。公开JSON中的官方按type指标根据每个预测的最佳参考归类，各方法的类型分母可能变化，只是官方描述字段，不能当作固定同一类型子集的比较。证据分数与答案分数可能不同；单个生成器、每个payload单次响应和不同实际证据长度限制了因果归因。相同payload共享响应是实验控制，不是实测线上缓存收益。",
        "", "本结果只能检验这六组本地证据选择在给定论文开发设定下的答案表现，不能据此宣称JEV、共享关系或完整SLAC的收益。未按成绩挑方法、挑问题、选择权重或替换原owner顺序。",
        "", "完整精度、官方描述指标、全部区间及来源hash见配套公开聚合JSON。公开产物不包含原始文本、问题标识或逐题输出。", ""]
    return "\n".join(lines)


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=experiment.legacy.client.unique_object)


def validate_driver(driver):
    for key, value in {"status": "completed_and_audited_pending_report", "paid_run_exit_code": 0,
            "audit_exit_code": 0, "child_pid": None}.items():
        if key not in driver:
            raise ValueError("completed driver field is missing")
        exact(driver.get(key), value, "driver " + key)
    stages = driver["stages"]
    if [row.get("name") for row in stages] != ["local_answer_run", "local_answer_audit"]:
        raise ValueError("driver must contain the fixed two completed stages")
    for row in stages:
        exact(row.get("exit_code"), 0, "driver stage exit")
    sha(driver["readiness_sha256"])


def publish(args):
    summary_path, driver_path, plan = (Path(getattr(args, name)).resolve() for name in ("summary", "driver_state", "plan"))
    audit_dir = Path(args.audit_directory).resolve()
    audit_path, binding_path = audit_dir / "audit.json", audit_dir / "source_binding.json"
    report, aggregate = Path(args.report).resolve(), Path(args.aggregate).resolve()
    if report == aggregate or report.exists() or aggregate.exists():
        raise FileExistsError("publication paths must be distinct unused files")
    inputs = experiment.hashes([summary_path, audit_path, binding_path, driver_path,
        *(plan / name for name in experiment.PLAN_FILES), Path(experiment.__file__), Path(__file__)])
    release = read(binding_path)
    exact(release.get("root_release"), ROOT_RELEASE, "root complete-run publication release")
    bound = release["input_sha256"]
    experiment.legacy.pilot.verify_hashes(bound)
    required = {p: value for p, value in inputs.items() if p not in (str(binding_path), str(Path(__file__).resolve()))}
    if any(bound.get(path) != value for path, value in required.items()):
        raise ValueError("root release must bind adapter, full plan, summary, audit and completed driver state")
    driver = read(driver_path)
    validate_driver(driver)
    public = project_summary(read(summary_path), read(audit_path))
    if public["experiment_plan_sha256"] != inputs[str(plan / "experiment_config.json")]:
        raise ValueError("summary references another experiment plan")
    public["publication_provenance"] = {"summary_sha256": inputs[str(summary_path)],
        "audit_sha256": inputs[str(audit_path)], "root_release_binding_sha256": inputs[str(binding_path)],
        "completed_driver_state_sha256": inputs[str(driver_path)], "driver_readiness_sha256": driver["readiness_sha256"],
        "adapter_sha256": inputs[str(Path(experiment.__file__).resolve())],
        "publisher_sha256": inputs[str(Path(__file__).resolve())]}
    experiment.legacy.pilot.verify_hashes(experiment.merge(inputs, bound))
    for path in (report, aggregate):
        path.parent.mkdir(parents=True, exist_ok=True)
    experiment.write(aggregate, public)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(markdown(public))
    return {"status": "published_complete_aggregates", "report_sha256": experiment.legacy.digest(report),
        "aggregate_sha256": experiment.legacy.digest(aggregate), "api_calls": 0, "private_outputs_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "audit-directory", "driver-state", "plan", "report", "aggregate"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(publish(parser.parse_args())))
