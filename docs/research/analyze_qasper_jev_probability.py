"""Offline JEV probability coverage and post-hoc raw-score ranking diagnostics.

Saved choice probabilities are not assumed calibrated. Missing or invalid
probabilities refuse every ranking result, with no fallback or renormalization.
An explicit reported-scores contract permits bounded near-unit-sum raw values,
without calling them a mathematical probability distribution. Confidence is
never read. Original support=no exclusions remain in force.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

import analyze_qasper_relation_pilot as audit
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, pack_ranked, score_selection


SIZES = (1, 2, 3)
RULES = ("ordinal_then_p_yes", "p_yes_only")
SUM_TOLERANCE = 1e-8
CHOICES = {"yes", "no", "unknown"}
CONTRACTS = ("strict-distribution", "reported-scores")
REPORTED_SCORE_SUM_RANGE = (0.985, 1.015)


def validate_probabilities(value, contract="strict-distribution"):
    if contract not in CONTRACTS:
        raise ValueError("unknown score contract")
    if not isinstance(value, dict) or set(value) != CHOICES:
        return "missing_or_invalid_probability_keys"
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in value.values()):
        return "probability_not_finite_in_unit_interval"
    total = sum(value.values())
    if contract == "reported-scores":
        return None if REPORTED_SCORE_SUM_RANGE[0] <= total <= REPORTED_SCORE_SUM_RANGE[1] else "reported_score_sum_outside_fixed_range"
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=SUM_TOLERANCE):
        return "probabilities_do_not_sum_to_one"
    return None


def collect_probabilities(config, run_dir, prepared, labels, checked, contract="strict-distribution"):
    """Read only already verified scheduled JEV support responses, including reuse."""
    ledger_path = run_dir / "provider_calls" / "ledger.json"
    ledger = pilot.read_json(checked(ledger_path)) if ledger_path.exists() else {"attempts": []}
    attempts = {row["cache_key"]: row for row in ledger["attempts"]}
    expected = {task["id"] for task in prepared["support_tasks"]}
    probabilities, local, issues, strict_issues, sums = {}, [], Counter(), Counter(), Counter()
    strict_invalid_eligible = 0
    for item in config["schedule"]:
        if item["backend"] != "jev" or item["kind"] != "support":
            continue
        reused = config["prior_execution"]["reused_responses"].get(item["cache_key"])
        path = Path(reused["response_path"]) if reused else (
            run_dir / "provider_calls" / f"response_{attempts[item['cache_key']]['attempt']:03d}.json")
        response = pilot.read_json(checked(path))
        if set(response["answers"]) != set(item["task_ids"]):
            raise ValueError("response probability coverage differs from frozen schedule")
        for task_id, answer in response["answers"].items():
            if task_id in probabilities or task_id not in expected or answer["choice"] != labels["jev"][task_id]:
                raise ValueError("duplicate, unexpected or changed support choice")
            value = answer.get("probabilities")
            issue = validate_probabilities(value, contract)
            strict_issue = validate_probabilities(value)
            probabilities[task_id] = value
            if issue:
                issues[issue] += 1
            if strict_issue:
                strict_issues[strict_issue] += 1
                strict_invalid_eligible += answer["choice"] != "no"
            probability_sum = (sum(value.values()) if isinstance(value, dict) and set(value) == CHOICES
                and all(type(p) in (int, float) and math.isfinite(p) for p in value.values()) else None)
            if probability_sum is not None:
                sums[str(round(probability_sum, 12))] += 1
            local.append({"task_id": task_id, "choice": answer["choice"], "probabilities": value,
                "probability_sum": probability_sum, "unit_sum_deviation": probability_sum - 1 if probability_sum is not None else None,
                "issue": issue, "strict_distribution_issue": strict_issue,
                "source_response_sha256": digest(path), "source_reused": bool(reused)})
    if set(probabilities) != expected:
        raise ValueError("probabilities do not cover all prepared support tasks")
    return probabilities, local, {
        "support_task_count": len(expected), "observed_task_count": len(probabilities),
        "valid_task_count": len(probabilities) - sum(issues.values()), "invalid_task_count": sum(issues.values()),
        "score_contract": contract, "strict_distribution_invalid_task_count": sum(strict_issues.values()),
        "strict_distribution_issue_counts": dict(strict_issues),
        "strict_distribution_invalid_but_support_eligible_count": strict_invalid_eligible,
        "issue_counts": dict(issues), "probability_sum_distribution": dict(sums),
        "sum_absolute_tolerance": SUM_TOLERANCE,
        "reported_score_sum_range": list(REPORTED_SCORE_SUM_RANGE),
        "renormalized": False, "confidence_used": False, "calibration_claimed": False,
        "complete_coverage_satisfying_contract": not issues}


def probability_ranking(units, candidate_ids, relevance, probabilities, retrieval_ranking, rule, contract="strict-distribution"):
    if rule not in RULES or set(relevance) != set(candidate_ids) or set(probabilities) != set(candidate_ids):
        raise ValueError("invalid probability ranking inputs")
    if any(label not in CHOICES for label in relevance.values()):
        raise ValueError("invalid support choice")
    if any(validate_probabilities(value, contract) for value in probabilities.values()):
        raise ValueError("ranking requires all values to satisfy the explicit score contract")
    by_id = {unit.unit_id: index for index, unit in enumerate(units)}
    if (len(set(candidate_ids)) != len(candidate_ids) or not set(candidate_ids) <= set(by_id)
            or len(retrieval_ranking) != len(candidate_ids) or set(retrieval_ranking) != set(candidate_ids)):
        raise ValueError("ranking differs from the frozen candidate pool")
    ranks = {uid: i for i, uid in enumerate(retrieval_ranking)}
    eligible = [uid for uid in candidate_ids if relevance[uid] != "no"]

    def key(uid):
        suffix = (-probabilities[uid]["yes"], ranks[uid], units[by_id[uid]].order)
        return (-(relevance[uid] == "yes"), *suffix) if rule == "ordinal_then_p_yes" else suffix

    return [by_id[uid] for uid in sorted(eligible, key=key)]


def replay(prepared, documents, annotations, labels, probabilities, tokenizer, contract="strict-distribution"):
    records, traces = [], []
    support = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
               for task in prepared["support_tasks"]}
    for query in prepared["queries"]:
        doc, qid = query["doc_id"], query["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        by_id = {unit.unit_id: i for i, unit in enumerate(units)}
        relevance = {uid: labels["jev"][support[(doc, qid, uid)]] for uid in query["candidate_ids"]}
        values = {uid: probabilities[support[(doc, qid, uid)]] for uid in query["candidate_ids"]}
        rankings = {rule: probability_ranking(units, query["candidate_ids"], relevance, values,
                                              query["ranked_ids"], rule, contract) for rule in RULES}
        count = PackCounter(tokenizer, units)
        identity = {name: query[name] for name in audit.IDENTITY}

        def record(method, chosen):
            return {**identity, "method": method, "budget": 1024,
                    **score_selection(units, chosen, gold, count, 1024)}

        for k in SIZES:
            dense = pack_ranked(units, [by_id[uid] for uid in query["ranked_ids"]], 1024, count, max_units=k)
            records.append(record(f"dense_k{k}", dense))
            original = pilot.replay_policy(units, query["candidate_ids"], relevance, [], query["ranked_ids"],
                mode="I", tokenizer=tokenizer, budget=1024, chunk_budget=384, max_units=k)
            records.append(record(f"I_jev_k{k}", original["selected_indices"]))
            for rule, ranking in rankings.items():
                chosen = pack_ranked(units, ranking, 1024, count, max_units=k)
                method = f"{rule}_k{k}"
                records.append(record(method, chosen))
                traces.append({**identity, "method": method, "max_units": k, "budget": 1024,
                    "candidate_information_sha256": original["candidate_information_sha256"],
                    "retrieval_ranking": query["ranked_ids"],
                    "probability_ranking": [units[i].unit_id for i in ranking],
                    "selected_ids": [units[i].unit_id for i in chosen],
                    "original_I_selected_ids": original["selected_ids"],
                    "excluded_no_ids": [uid for uid in query["candidate_ids"] if relevance[uid] == "no"]})
    return records, traces


def summarize(prepared, records):
    questions = [tuple(query[name] for name in audit.IDENTITY) for query in prepared["queries"]]
    methods = [f"{base}_k{k}" for k in SIZES for base in ("dense", "I_jev", *RULES)]
    indexed = audit.index_rows(records, methods, questions)
    tables = {method: {key: indexed[(method, *key)] for key in questions} for method in methods}
    comparisons = []
    for k in SIZES:
        for rule in RULES:
            for baseline in (f"I_jev_k{k}", f"dense_k{k}"):
                method = f"{rule}_k{k}"
                comparisons.append({"plus": method, "minus": baseline,
                    "changed_selected_sets": sum(set(tables[method][key]["selected_ids"]) !=
                                                  set(tables[baseline][key]["selected_ids"]) for key in questions),
                    **audit.compare(tables[method], tables[baseline], questions)})
    return {"question_count": len(questions), "record_count": len(records), "new_method_count": len(RULES) * len(SIZES),
            "metrics": aggregate(records), "paired_comparisons": comparisons}


def analyze(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script_hash = digest(__file__)
    verified = audit.analyze(argparse.Namespace(plan=args.plan, run=args.run, prepared=args.prepared,
                                               output=str(output / "verified_inputs")))
    bindings = {row["path_sha256"]: row["content_sha256"] for row in verified["input_sha256"]}
    used = {}

    def checked(path):
        path = Path(path).resolve()
        expected = bindings[hashlib.sha256(str(path).encode()).hexdigest()]
        if digest(path) != expected:
            raise ValueError("input changed after completed-run verification")
        used[str(path)] = expected
        return path

    config, _ = pilot.load_plan(args.plan)
    checked(Path(args.plan) / "experiment_config.json")
    checked(Path(args.prepared) / "prepared.json")
    prepared, _, documents = pilot.load_prepared(args.prepared)
    labels = pilot.read_json(checked(Path(args.run) / "labels.json"))
    contract = getattr(args, "score_contract", "strict-distribution")
    probabilities, local, coverage = collect_probabilities(config, Path(args.run).resolve(), prepared, labels, checked, contract)
    result = {"schema": "slac-qasper-jev-probability-v1", "api_calls": 0,
        "analysis_type": "post-hoc API probabilities-field raw-score ranking diagnosis; not independent confirmation",
        "score_contract": contract,
        "probability_coverage": coverage, "input_binding_sha256": verified["input_binding_sha256"],
        "script_sha256": script_hash, "gold_used_for_selection": False, "test_payload_read": False,
        "sizes": list(SIZES), "rules": list(RULES), "threshold_search_performed": False,
        "calibration_claimed": False, "confidence_used": False,
        "limitations": ["Returned probabilities-field values are used as raw ranking scores, not assumed calibrated distributions.",
                        "No probability renormalization, missing-value fallback, or confidence score is used.",
                        "Reported-scores bounds were explicitly revised after inspecting unit-sum deviations, before comparing ranking quality.",
                        "The [0.985, 1.015] range is a three-value two-decimal rounding-scale allowance, not evidence that rounding caused the deviations.",
                        "The original support=no exclusions and frozen candidate pool remain unchanged.",
                        "All six fixed ranking/k combinations are reported; no winning k is promoted.",
                        "No answer generation or statistical significance claim."]}
    records = traces = None
    if coverage["complete_coverage_satisfying_contract"]:
        annotations = pilot.selected_gold(checked(config["sidecar"]), prepared)
        tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
        records, traces = replay(prepared, documents, annotations, labels, probabilities, tokenizer, contract)
        result.update(summarize(prepared, records), status="completed")
        old = audit.read_rows(checked(Path(args.run) / "per_question.jsonl"))
        old_index = {(row["doc_id"], row["question_id"]): row for row in old if row["method"] == "I_jev"}
        for row in records:
            if row["method"] == "I_jev_k3" and {**row, "method": "I_jev"} != old_index[(row["doc_id"], row["question_id"])]:
                raise ValueError("original I_jev_k3 no longer matches frozen output")
    else:
        result.update(status="refused_score_contract", ranking_results_available=False,
                      refusal_reason="All support values must satisfy the selected score contract before any ranking results are emitted.")
    pilot.load_plan(args.plan)
    pilot.verify_hashes(used)
    if digest(__file__) != script_hash:
        raise ValueError("analysis code changed during execution")
    pilot.write_rows(output / "probability_sources.jsonl", local)
    if records is not None:
        pilot.write_rows(output / "per_question.jsonl", records)
        pilot.write_rows(output / "traces.jsonl", traces)
    pilot.write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--score-contract", choices=CONTRACTS, default="strict-distribution")
    report = analyze(parser.parse_args())
    print({"status": report["status"], "support_tasks": report["probability_coverage"]["support_task_count"],
           "invalid_tasks": report["probability_coverage"]["invalid_task_count"], "api_calls": 0})
