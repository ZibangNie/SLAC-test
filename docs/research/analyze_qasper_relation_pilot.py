"""Post-hoc descriptive pilot diagnostics; local files only, aggregate output.

This never creates a provider client, reads credentials, generates answers or
estimates significance. Frozen inputs and the completed execution must verify.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import run_qasper_relation_pilot as pilot
from qasper_metrics import references_from_annotations
from run_qasper_evidence_baselines import aggregate, digest


BACKENDS = ("jev", "general")
METHODS = tuple(f"{mode}_{backend}" for backend in BACKENDS for mode in ("I", "C", "S"))
METRICS = ("official_evidence_f1", "reference_evidence_recall", "official_text_only_evidence_f1",
           "actual_evidence_tokens")
IDENTITY = ("family_id", "doc_id", "question_id")
BASELINES = {"dense_top3_capped", "dense_top3_full_document", "gold_subset_oracle_capped_max3", "empty"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def index_rows(rows, methods, questions):
    expected = {(method, *question) for method in methods for question in questions}
    indexed = {}
    for row in rows:
        key = (row["method"], *(row[name] for name in IDENTITY))
        if key in indexed:
            raise ValueError("duplicate result identity")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("result coverage differs from prepared questions")
    return indexed


def label_diagnostics(prepared, labels):
    if set(labels) != set(BACKENDS):
        raise ValueError("label backends differ from frozen comparison")
    expected = [task["id"] for kind in ("static", "support") for task in prepared[kind + "_tasks"]]
    if len(expected) != len(set(expected)) or any(set(labels[b]) != set(expected) for b in BACKENDS):
        raise ValueError("label coverage differs from prepared tasks")
    result = {}
    for kind in ("static", "support"):
        task_ids = [task["id"] for task in prepared[kind + "_tasks"]]
        choices = list(pilot.client.CRITERIA[kind])
        if any(labels[b][task_id] not in choices for b in BACKENDS for task_id in task_ids):
            raise ValueError("unsupported task label")
        counts = {b: dict.fromkeys(choices, 0) for b in BACKENDS}
        confusion = {left: dict.fromkeys(choices, 0) for left in choices}
        for task_id in task_ids:
            left, right = labels["jev"][task_id], labels["general"][task_id]
            counts["jev"][left] += 1
            counts["general"][right] += 1
            confusion[left][right] += 1
        agreement = sum(confusion[choice][choice] for choice in choices)
        result[kind] = {"task_count": len(task_ids), "label_counts": counts,
                        "confusion_rows_jev_columns_general": confusion,
                        "agreement_count": agreement,
                        "agreement_rate": agreement / len(task_ids) if task_ids else None}
    return result


def delta_stats(plus, minus, questions, metric):
    values, families, documents = [], defaultdict(list), defaultdict(list)
    for key in questions:
        value = plus[key][metric] - minus[key][metric]
        values.append(value)
        families[key[0]].append(value)
        documents[key[1]].append(value)
    mean = lambda vals: sum(vals) / len(vals) if vals else None
    return {"questions": len(values), "sum": sum(values), "question_macro_delta": mean(values),
            "family_macro_delta": mean([mean(v) for v in families.values()]),
            "document_macro_delta": mean([mean(v) for v in documents.values()]),
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "positive": sum(v > 0 for v in values), "equal": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values)}


def compare(plus, minus, questions):
    return {metric: delta_stats(plus, minus, questions, metric) for metric in METRICS}


def summarize(prepared, labels, records, traces, baseline_records, annotations):
    questions = [tuple(row[name] for name in IDENTITY) for row in prepared["queries"]]
    if not questions or len(questions) != len(set(questions)):
        raise ValueError("invalid prepared question coverage")
    rows = index_rows(records, METHODS, questions)
    trace_rows = index_rows(traces, METHODS, questions)
    baseline_rows = index_rows(baseline_records, BASELINES, questions)
    by_method = {method: {key: rows[(method, *key)] for key in questions} for method in METHODS}
    dense = {key: baseline_rows[("dense_top3_capped", *key)] for key in questions}
    if set(annotations) != {(key[1], key[2]) for key in questions}:
        raise ValueError("annotation coverage differs from prepared questions")
    strata = {"any_empty_reference": [], "all_nonempty_references": []}
    for key in questions:
        refs = references_from_annotations(annotations[(key[1], key[2])])
        if not refs:
            raise ValueError("question has no official references")
        name = "any_empty_reference" if any(not ref["evidence"] for ref in refs) else "all_nonempty_references"
        strata[name].append(key)
    same, shared = {}, {}
    for backend in BACKENDS:
        independent, cached, sharing = (by_method[f"{mode}_{backend}"] for mode in ("I", "C", "S"))
        fields = ("selected_ids", "pack_sha256", *METRICS, "selected_units", "budget")
        matching = {field: sum(independent[key][field] == cached[key][field] for key in questions) for field in fields}
        same[backend] = {"question_count": len(questions), "matching_counts": matching,
                         "all_equal": all(count == len(questions) for count in matching.values())}
        if not same[backend]["all_equal"]:
            raise ValueError("independent and cached replay outputs differ")
        totals = Counter()
        merge_edges, bonus_edges = set(), set()
        for key in questions:
            trace = trace_rows[(f"S_{backend}", *key)]
            merges, bonuses, steps = trace["accepted_merge_edge_ids"], trace["selection_bonus_used_edge_ids"], trace["selection_trace"]
            totals["questions_with_merge"] += bool(merges)
            totals["merge_edge_query_occurrences"] += len(merges)
            totals["questions_with_bonus_used"] += bool(bonuses)
            totals["bonus_edge_query_occurrences"] += len(bonuses)
            totals["bonus_accepted_selection_steps"] += sum(step["accepted"] and step["relation_bonus"] > 0 for step in steps)
            totals["priority_changed_steps"] += trace["selection_priority_changed_steps"]
            totals["questions_with_priority_change"] += trace["selection_priority_changed_steps"] > 0
            totals["priority_changed_accepted_steps"] += sum(step["accepted"] and step["relation_changed_priority"] for step in steps)
            merge_edges.update((key[1], edge) for edge in merges)
            bonus_edges.update((key[1], edge) for edge in bonuses)
        totals.update(distinct_document_merge_edges=len(merge_edges), distinct_document_bonus_edges=len(bonus_edges))
        shared[backend] = {
            "changed_selected_sets": sum(set(sharing[key]["selected_ids"]) != set(cached[key]["selected_ids"]) for key in questions),
            "changed_pack_hashes": sum(sharing[key]["pack_sha256"] != cached[key]["pack_sha256"] for key in questions),
            "selected_set_symmetric_difference_total": sum(len(set(sharing[key]["selected_ids"]) ^ set(cached[key]["selected_ids"])) for key in questions),
            "paired_shared_minus_cached": compare(sharing, cached, questions), "trace_usage": dict(totals)}
    result = {"question_count": len(questions), "family_count": len({key[0] for key in questions}),
              "label_diagnostics": label_diagnostics(prepared, labels), "I_C_equivalence": same,
              "shared_vs_cached": shared,
              "paired_method_minus_dense_top3_capped": {m: compare(by_method[m], dense, questions) for m in METHODS},
              "reference_strata": {}}
    for name, subset in strata.items():
        result["reference_strata"][name] = {
            "questions": len(subset), "families": len({key[0] for key in subset}),
            "method_means": {method: {metric: sum(table[key][metric] for key in subset) / len(subset) if subset else None
                                      for metric in METRICS}
                             for method, table in {**by_method, "dense_top3_capped": dense}.items()},
            "shared_minus_cached": {b: compare(by_method[f"S_{b}"], by_method[f"C_{b}"], subset) for b in BACKENDS},
            "method_minus_dense_top3_capped": {m: compare(by_method[m], dense, subset) for m in METHODS}}
    return result


def validate_execution(config, batches, run_dir, labels, summary, bind):
    """Reparse saved successful responses and reconcile each schedule entry."""
    schedule = config["schedule"]
    reused = config["prior_execution"]["reused_responses"]
    by_batch = {batch["id"]: batch for batch in batches}
    ledger_path = run_dir / "provider_calls" / "ledger.json"
    ledger = pilot.read_json(bind(ledger_path)) if ledger_path.exists() else {"attempts": [], "halt_reason": None}
    if ledger.get("halt_reason") is not None or any(row["status"] != "completed" for row in ledger["attempts"]):
        raise ValueError("execution ledger is incomplete or halted")
    expected_new = [item for item in schedule if item["cache_key"] not in reused]
    if len(ledger["attempts"]) != len(expected_new):
        raise ValueError("execution request coverage differs from frozen schedule")
    decisions, new_index, reuse_index = {b: {} for b in BACKENDS}, 0, 0
    expected_call_files = {"ledger.json"} if ledger_path.exists() else set()
    for index, item in enumerate(schedule):
        provenance = reused.get(item["cache_key"])
        batch = by_batch[item["batch_id"]]
        if provenance:
            reuse_index += 1
            event = pilot.read_json(bind(run_dir / f"reuse_{reuse_index:03d}.json"))
            if (event["schedule_index"] != index or event["cache_key"] != item["cache_key"]
                    or event["source"] != provenance or event["new_api_call"] is not False):
                raise ValueError("reuse event differs from frozen provenance")
            record = pilot.read_json(bind(Path(provenance["ledger_path"])))["attempts"][provenance["attempt"] - 1]
            payload = pilot.client.make_payload(batch["tasks"], batch["kind"], item["backend"])
            response = pilot.read_json(bind(Path(provenance["response_path"])))
        else:
            new_index += 1
            record = ledger["attempts"][new_index - 1]
            request_path = run_dir / "provider_calls" / f"request_{new_index:03d}.json"
            response_path = run_dir / "provider_calls" / f"response_{new_index:03d}.json"
            expected_call_files.update((request_path.name, response_path.name))
            payload = pilot.prior_request(bind(request_path), record, config["prompt_version"])
            response = pilot.read_json(bind(response_path))
            if record["attempt"] != new_index:
                raise ValueError("execution request ordinal differs")
        if (record["cache_key"] != item["cache_key"] or record["backend"] != item["backend"]
                or record["kind"] != item["kind"] or set(record["task_ids"]) != set(item["task_ids"])
                or pilot.client.object_hash(payload) != item["payload_sha256"]):
            raise ValueError("saved response is not from the frozen request")
        parsed = pilot.validate_reused_response(response, payload, record)
        if set(parsed) & set(decisions[item["backend"]]):
            raise ValueError("duplicate executed task")
        decisions[item["backend"]].update(parsed)
    if decisions != labels:
        raise ValueError("saved labels differ from completed responses")
    actual_call_files = {path.name for path in ledger_path.parent.glob("*.json")}
    if actual_call_files != expected_call_files or len(list(run_dir.glob("reuse_*.json"))) != reuse_index:
        raise ValueError("unaccounted execution artifact")
    if (summary["new_api_calls"] != new_index or summary["actual_requests"] != new_index
            or summary["reused_count"] != reuse_index):
        raise ValueError("summary request coverage differs from execution")


def analyze(args):
    plan_dir, run_dir, prepared_dir = (Path(getattr(args, name)).resolve() for name in ("plan", "run", "prepared"))
    hashes = {}

    def bind(path):
        path = Path(path).resolve()
        actual = digest(path)
        if str(path) in hashes and hashes[str(path)] != actual:
            raise ValueError("analysis input changed during read")
        hashes[str(path)] = actual
        return path

    bind(Path(__file__))
    for name in ("experiment_config.json", "plan_manifest.json"):
        bind(plan_dir / name)
    config, batches = pilot.load_plan(plan_dir)
    if prepared_dir != Path(config["prepared_dir"]).resolve():
        raise ValueError("prepared directory differs from frozen plan")
    for path, expected in config["input_sha256"].items():
        hashes[str(Path(path).resolve())] = expected
    for name, expected in config["plan_files_sha256"].items():
        hashes[str((plan_dir / name).resolve())] = expected
    prepared, _, documents = pilot.load_prepared(prepared_dir)
    summary = pilot.read_json(bind(run_dir / "summary.json"))
    if (summary.get("status") != "completed" or summary.get("all_results_available") is not True
            or summary.get("input_hashes_unchanged") is not True or summary.get("test_payload_read") is not False
            or summary.get("answer_generation_performed") is not False or (run_dir / "failure.json").exists()):
        raise ValueError("analysis requires a completed non-test evidence-selection run")
    if (summary["plan_sha256"] != hashes[str(plan_dir / "experiment_config.json")]
            or summary["general_profile"] != config["general_profile"]):
        raise ValueError("completed run belongs to a different frozen plan")
    labels = pilot.read_json(bind(run_dir / "labels.json"))
    records = read_rows(bind(run_dir / "per_question.jsonl"))
    traces = read_rows(bind(run_dir / "traces.jsonl"))
    baseline = read_rows(plan_dir / "baseline_per_question.jsonl")
    validate_execution(config, batches, run_dir, labels, summary, bind)
    annotations = pilot.selected_gold(config["sidecar"], prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    expected_records, expected_traces = pilot.replay_records(prepared, documents, annotations, labels, tokenizer)
    if records != expected_records or traces != expected_traces:
        raise ValueError("saved policy outputs differ from deterministic local replay")
    if (summary["record_count"] != len(records) or summary["question_count"] != len(prepared["queries"])
            or summary["family_count"] != len(documents) or summary["metrics"] != aggregate(records)):
        raise ValueError("completed summary differs from verified per-question outputs")
    result = summarize(prepared, labels, records, traces, baseline, annotations)
    pilot.verify_hashes(hashes)
    result.update(schema="slac-qasper-relation-analysis-v1", status="completed",
                  analysis_type="post-hoc exploratory descriptive analysis; not independent confirmation",
                  significance_claimed=False, api_calls=0, test_payload_read=False,
                  interpretation_limits=[
                      "Agreement is between model judgments, not agreement with independently audited gold labels.",
                      "I and C share offline decisions; equality does not measure cache speed or cost savings.",
                      "Any-empty-reference follows official per-annotation evidence handling and is not an answerability label.",
                      "Trace edge occurrences may repeat across queries; distinct document-edge counts are also reported.",
                      "This is fixed-pool given-document evidence selection, not answer generation or end-to-end RAG quality."],
                  input_hashes_unchanged=True,
                  input_binding_sha256=pilot.client.object_hash(hashes),
                  input_sha256=[{"path_sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(), "content_sha256": value}
                                for path, value in sorted(hashes.items())])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "analysis.json", result)
    if pilot.read_json(output / "analysis.json") != result:
        raise ValueError("aggregate analysis round-trip mismatch")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        report = analyze(parse_args())
    except Exception as exc:
        print(json.dumps({"status": "refused", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"status": report["status"], "question_count": report["question_count"],
                      "input_binding_sha256": report["input_binding_sha256"]}))
