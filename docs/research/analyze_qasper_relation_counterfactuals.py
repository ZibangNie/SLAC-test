"""Post-hoc local relation/support swaps on a verified completed pilot.

No API calls, credentials, threshold search, new labels, or test payloads. The
uncapped-chunk variant is a diagnostic intervention, not the frozen S policy.
Gold enters only the existing metric after all selection decisions are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import analyze_qasper_relation_pilot as analysis
import run_qasper_relation_pilot as pilot
from qasper_relation_replay import AdjacentRelation, replay_policy
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, score_selection


BACKENDS = analysis.BACKENDS
VARIANTS = ("original", "uncapped_chunk_diagnostic")
STAGES = ("dependent", "accepted_merge", "both_support_eligible", "bonus_used", "priority_changed")


def method_name(support, relation, variant):
    return f"S_support_{support}_relation_{relation}_{variant}"


def uncapped_chunk_budget(units, candidate_ids, tokenizer):
    """Bound every possible contiguous candidate group, without monotonicity assumptions."""
    indices = [index for index, unit in enumerate(units) if unit.unit_id in set(candidate_ids)]
    count = PackCounter(tokenizer, units)
    return max(1, *(count(indices[start:end]) for start in range(len(indices))
                    for end in range(start + 1, len(indices) + 1)))


def funnel(trace, relations, relevance):
    dependent = {edge.edge_id for edge in relations if edge.label == "dependent"}
    accepted = set(trace["accepted_merge_edge_ids"])
    eligible = {edge.edge_id for edge in relations if edge.edge_id in accepted
                and relevance[edge.left_id] != "no" and relevance[edge.right_id] != "no"}
    used = set(trace["selection_bonus_used_edge_ids"])
    changed = {edge for step in trace["selection_trace"]
               if step["accepted"] and step["relation_changed_priority"]
               for edge in step["relation_edge_ids"]}
    stages = dict(zip(STAGES, (dependent, accepted, eligible, used, changed)))
    if not changed <= used <= eligible <= accepted <= dependent:
        raise ValueError("mechanism funnel stage is not a subset of its predecessor")
    return {"edge_ids": {name: sorted(edges) for name, edges in stages.items()},
            "counts": {name: len(edges) for name, edges in stages.items()},
            "priority_changed_accepted_steps": sum(step["accepted"] and step["relation_changed_priority"]
                                                   for step in trace["selection_trace"]),
            "selected_set_changed": bool(trace["selection_symmetric_difference_count"]),
            "selected_set_symmetric_difference_count": trace["selection_symmetric_difference_count"]}


def replay_counterfactuals(prepared, documents, annotations, labels, tokenizer):
    analysis.label_diagnostics(prepared, labels)
    support_lookup = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
                      for task in prepared["support_tasks"]}
    records, diagnostics = [], []
    for query in prepared["queries"]:
        doc, qid = query["doc_id"], query["question_id"]
        units, candidates = documents[doc], set(query["candidate_ids"])
        identity = {name: query[name] for name in analysis.IDENTITY}
        count = PackCounter(tokenizer, units)
        uncapped = uncapped_chunk_budget(units, candidates, tokenizer)
        relations_by_backend = {backend: [
            AdjacentRelation(task["left_id"], task["right_id"], labels[backend][task["id"]])
            for task in prepared["static_tasks"] if task["doc_id"] == doc
            and task["left_id"] in candidates and task["right_id"] in candidates]
            for backend in BACKENDS}
        for support in BACKENDS:
            relevance = {uid: labels[support][support_lookup[(doc, qid, uid)]] for uid in candidates}
            independent = replay_policy(units, query["candidate_ids"], relevance, [], query["ranked_ids"],
                mode="I", tokenizer=tokenizer, **{k: pilot.SELECTOR[k] for k in ("budget", "chunk_budget", "max_units")})
            records.append({**identity, "method": f"I_{support}", "budget": pilot.SELECTOR["budget"],
                **score_selection(units, independent["selected_indices"], annotations[(doc, qid)], count, pilot.SELECTOR["budget"])})
            for relation in BACKENDS:
                relations = relations_by_backend[relation]
                for variant in VARIANTS:
                    chunk_budget = pilot.SELECTOR["chunk_budget"] if variant == "original" else uncapped
                    trace = replay_policy(units, query["candidate_ids"], relevance, relations, query["ranked_ids"],
                        mode="S", tokenizer=tokenizer, budget=pilot.SELECTOR["budget"],
                        chunk_budget=chunk_budget, max_units=pilot.SELECTOR["max_units"])
                    if trace["independent_selected_ids"] != independent["selected_ids"]:
                        raise ValueError("counterfactual independent baseline changed")
                    if variant != "original" and any(step["action"] == "keep_boundary_chunk_budget"
                                                     for step in trace["boundary_trace"]):
                        raise ValueError("uncapped diagnostic still rejected a dependent merge")
                    method = method_name(support, relation, variant)
                    records.append({**identity, "method": method, "budget": pilot.SELECTOR["budget"],
                        **score_selection(units, trace["selected_indices"], annotations[(doc, qid)], count, pilot.SELECTOR["budget"])})
                    diagnostics.append({**identity, "method": method, "support_backend": support,
                        "relation_backend": relation, "variant": variant, "chunk_budget": chunk_budget,
                        "candidate_information_sha256": trace["candidate_information_sha256"],
                        "selected_ids": trace["selected_ids"], "independent_selected_ids": independent["selected_ids"],
                        "funnel": funnel(trace, relations, relevance),
                        "boundary_trace": trace["boundary_trace"], "selection_trace": trace["selection_trace"]})
    return records, diagnostics


def summarize(prepared, records, diagnostics):
    questions = [tuple(query[name] for name in analysis.IDENTITY) for query in prepared["queries"]]
    if not questions or len(set(questions)) != len(questions):
        raise ValueError("invalid query coverage")
    methods = [f"I_{backend}" for backend in BACKENDS] + [method_name(s, r, v)
               for s in BACKENDS for r in BACKENDS for v in VARIANTS]
    indexed = analysis.index_rows(records, methods, questions)
    diagnostic_index = analysis.index_rows(diagnostics, methods[2:], questions)
    by_method = {method: {key: indexed[(method, *key)] for key in questions} for method in methods}
    comparisons, funnels = {}, {}
    for support in BACKENDS:
        baseline = by_method[f"I_{support}"]
        for relation in BACKENDS:
            for variant in VARIANTS:
                method = method_name(support, relation, variant)
                rows = by_method[method]
                comparisons[method] = {
                    "minus_same_support_I": analysis.compare(rows, baseline, questions),
                    "changed_selected_sets": sum(set(rows[q]["selected_ids"]) != set(baseline[q]["selected_ids"]) for q in questions),
                    "changed_pack_hashes": sum(rows[q]["pack_sha256"] != baseline[q]["pack_sha256"] for q in questions)}
                distinct = {stage: set() for stage in STAGES}
                totals, has_stage = Counter(), Counter()
                for key in questions:
                    item = diagnostic_index[(method, *key)]["funnel"]
                    for stage in STAGES:
                        edges = item["edge_ids"][stage]
                        totals[stage] += len(edges)
                        has_stage[stage] += bool(edges)
                        distinct[stage].update((key[1], edge) for edge in edges)
                    totals["priority_changed_accepted_steps"] += item["priority_changed_accepted_steps"]
                funnels[method] = {
                    "edge_query_occurrences": {stage: totals[stage] for stage in STAGES},
                    "distinct_document_edges": {stage: len(distinct[stage]) for stage in STAGES},
                    "questions_reaching_stage": {stage: has_stage[stage] for stage in STAGES},
                    "priority_changed_accepted_steps": totals["priority_changed_accepted_steps"],
                    "questions_with_selected_set_change": comparisons[method]["changed_selected_sets"]}
    return {"question_count": len(questions), "family_count": len({q[0] for q in questions}),
            "method_count": len(methods), "metrics": aggregate(records), "comparisons_to_I": comparisons,
            "mechanism_funnels": funnels,
            "general_relation_minus_jev_relation": {
                f"support_{s}_{v}": analysis.compare(by_method[method_name(s, "general", v)],
                                                    by_method[method_name(s, "jev", v)], questions)
                for s in BACKENDS for v in VARIANTS},
            "uncapped_minus_original": {
                f"support_{s}_relation_{r}": analysis.compare(by_method[method_name(s, r, VARIANTS[1])],
                                                           by_method[method_name(s, r, VARIANTS[0])], questions)
                for s in BACKENDS for r in BACKENDS}}


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
    bind(Path(analysis.__file__))
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
    original_records = analysis.read_rows(bind(run_dir / "per_question.jsonl"))
    original_traces = analysis.read_rows(bind(run_dir / "traces.jsonl"))
    analysis.validate_execution(config, batches, run_dir, labels, summary, bind)
    annotations = pilot.selected_gold(config["sidecar"], prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    expected_records, expected_traces = pilot.replay_records(prepared, documents, annotations, labels, tokenizer)
    if original_records != expected_records or original_traces != expected_traces:
        raise ValueError("saved policy outputs differ from deterministic local replay")
    if (summary["record_count"] != len(original_records) or summary["question_count"] != len(prepared["queries"])
            or summary["family_count"] != len(documents) or summary["metrics"] != aggregate(original_records)):
        raise ValueError("completed summary differs from verified per-question outputs")
    records, diagnostics = replay_counterfactuals(prepared, documents, annotations, labels, tokenizer)
    result = summarize(prepared, records, diagnostics)
    # Diagonal original variants and I must exactly reproduce the frozen run.
    original = {(row["method"], *(row[name] for name in analysis.IDENTITY)): row for row in original_records}
    for row in records:
        equivalent = row["method"] if row["method"].startswith("I_") else next(
            (f"S_{b}" for b in BACKENDS if row["method"] == method_name(b, b, "original")), None)
        if equivalent:
            expected = original[(equivalent, *(row[name] for name in analysis.IDENTITY))]
            if {**row, "method": equivalent} != expected:
                raise ValueError("counterfactual diagonal does not reproduce the original run")
    pilot.verify_hashes(hashes)
    result.update(schema="slac-qasper-relation-counterfactual-v1", status="completed",
        analysis_type="post-hoc exploratory descriptive counterfactuals; not independent confirmation",
        api_calls=0, test_payload_read=False, new_gold_labels_created=False, significance_claimed=False,
        thresholds_selected_using_gold=False, input_hashes_unchanged=True,
        interpretation_limits=[
            "Support and relation decisions are reused from the same exposed development pilot.",
            "Uncapped chunk merging is a diagnostic intervention, not the frozen deployable S strategy.",
            "All methods retain the fixed candidate pool, ranking, 1024 evidence-token limit and maximum of three native units.",
            "Gold is used only for scoring selected evidence; no new gold labels or threshold search.",
            "Funnel edge counts can repeat across queries; distinct document-edge counts are separate.",
            "Selected-set change is a query-level outcome, not a causal effect attributed to individual edges.",
            "Priority-changed edges count only accepted selection steps; rejected attempts are preserved in local traces.",
            "No answer generation, cache-efficiency measurement or statistical significance claim."],
        input_binding_sha256=pilot.client.object_hash(hashes),
        input_sha256=[{"path_sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(), "content_sha256": value}
                      for path, value in sorted(hashes.items())])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_rows(output / "per_question.jsonl", records)
    pilot.write_rows(output / "diagnostics.jsonl", diagnostics)
    result["output_files_sha256"] = {name: digest(output / name) for name in ("per_question.jsonl", "diagnostics.jsonl")}
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
    import json
    try:
        report = analyze(parse_args())
    except Exception as exc:
        print(json.dumps({"status": "refused", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"status": report["status"], "question_count": report["question_count"],
                      "method_count": report["method_count"], "input_binding_sha256": report["input_binding_sha256"]}))
