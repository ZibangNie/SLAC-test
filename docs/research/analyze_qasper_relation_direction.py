"""Post-hoc direction diagnosis on verified saved judgments; no provider calls.

The original S partition and accepted edges stay fixed. Only the direction of
its one-hop selection bonus changes. No query-conditioned labels are invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import analyze_qasper_relation_pilot as source
import qasper_relation_replay as replay
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, render_pack, score_selection


DIRECTIONS = ("symmetric", "prerequisite", "reverse_placebo")
VERSION = "fixed-accepted-edge-direction-v1"


def replay_direction(units, candidate_ids, relevance_labels, relations, retrieval_ranking,
                     accepted_merge_edge_ids, *, direction, tokenizer, budget=1024,
                     chunk_budget=384, max_units=3, deadline=math.inf):
    """Select gold-free; prerequisite B->A follows the judged 'B depends on A'."""
    if direction not in DIRECTIONS:
        raise ValueError("unsupported direction")
    units, candidate_ids, relations = tuple(units), tuple(candidate_ids), tuple(relations)
    by_id = replay._validate(units, candidate_ids, relevance_labels, relations,
                             retrieval_ranking, "S", budget, chunk_budget, max_units)
    accepted_ids = tuple(accepted_merge_edge_ids)
    by_edge = {edge.edge_id: edge for edge in relations}
    if (len(set(accepted_ids)) != len(accepted_ids) or not set(accepted_ids) <= set(by_edge)
            or any(by_edge[key].label != "dependent" for key in accepted_ids)):
        raise ValueError("accepted edges must be unique known dependent edges")
    indices = sorted(by_id[key] for key in candidate_ids)
    count = PackCounter(tokenizer, units, deadline=deadline)
    ranks = {key: index for index, key in enumerate(retrieval_ranking)}
    scores = {index: replay.RELEVANCE_WEIGHTS[relevance_labels[units[index].unit_id]] for index in indices}
    neighbours = {index: [] for index in indices}
    for edge_id in accepted_ids:
        edge = by_edge[edge_id]
        left, right = by_id[edge.left_id], by_id[edge.right_id]
        if direction in {"symmetric", "prerequisite"}:
            neighbours[left].append((right, edge_id))
        if direction in {"symmetric", "reverse_placebo"}:
            neighbours[right].append((left, edge_id))
    pending = {index for index in indices if scores[index] > 0}
    eligible_ids = [units[index].unit_id for index in indices if scores[index] > 0]
    selected, selected_text, trace, opportunities = [], set(), [], []

    def independent_key(index):
        return -scores[index], ranks[units[index].unit_id], units[index].order

    while pending and len(selected) < max_units:
        linked = {index: sorted(edge_id for other, edge_id in neighbours[index] if other in selected)
                  for index in indices}
        opportunities.append({"step": len(trace),
            "eligible_bonus_candidate_ids": [units[i].unit_id for i in sorted(pending) if linked[i]],
            "excluded_no_linked_ids": [units[i].unit_id for i in indices if scores[i] == 0 and linked[i]]})
        independent_first = min(pending, key=independent_key)
        index = min(pending, key=lambda i: (-(scores[i] + bool(linked[i])),
                                           ranks[units[i].unit_id], units[i].order))
        pending.remove(index)
        proposed = sorted([*selected, index])
        item = {"step": len(trace), "unit_id": units[index].unit_id,
                "selected_before": [units[i].unit_id for i in sorted(selected)],
                "base_score": scores[index], "relation_bonus": float(bool(linked[index])),
                "effective_score": scores[index] + bool(linked[index]),
                "relation_edge_ids": linked[index],
                "independent_first_remaining": units[independent_first].unit_id,
                "relation_changed_priority": index != independent_first,
                "proposed_tokens": None, "accepted": False}
        if units[index].native_text in selected_text:
            item["action"] = "skip_exact_native_duplicate"
        else:
            item["proposed_tokens"] = count(proposed)
            if item["proposed_tokens"] <= budget:
                selected.append(index)
                selected_text.add(units[index].native_text)
                item.update(action="select", accepted=True)
            else:
                item["action"] = "skip_evidence_budget"
        trace.append(item)
    selected.sort()
    return {"direction_version": VERSION, "direction": direction,
            "accepted_merge_edge_ids": list(accepted_ids),
            "eligible_ids": eligible_ids,
            "excluded_no_ids": [units[i].unit_id for i in indices if scores[i] == 0],
            "selected_indices": selected, "selected_ids": [units[i].unit_id for i in selected],
            "actual_evidence_tokens": count(selected),
            "pack_sha256": hashlib.sha256(render_pack(units, selected).encode("utf-8")).hexdigest(),
            "selection_trace": trace, "bonus_opportunities": opportunities,
            "selection_bonus_used_edge_ids": sorted({edge for step in trace if step["accepted"]
                                                       for edge in step["relation_edge_ids"]}),
            "selection_priority_changed_steps": sum(step["relation_changed_priority"] for step in trace),
            "gold_consumed": False, "api_calls": 0}


def require_symmetric_parity(actual, original):
    fields = ("accepted_merge_edge_ids", "excluded_no_ids", "selected_indices", "selected_ids",
              "actual_evidence_tokens", "pack_sha256", "selection_trace",
              "selection_bonus_used_edge_ids", "selection_priority_changed_steps")
    if actual["direction"] != "symmetric" or any(actual[key] != original[key] for key in fields):
        raise ValueError("symmetric replay differs from original S")


def direction_records(prepared, documents, labels, annotations, original_records, original_traces, tokenizer):
    questions = [tuple(row[name] for name in source.IDENTITY) for row in prepared["queries"]]
    old_rows = source.index_rows(original_records, source.METHODS, questions)
    old_traces = source.index_rows(original_traces, source.METHODS, questions)
    support = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
               for task in prepared["support_tasks"]}
    rows, traces = [], []
    for query, key in zip(prepared["queries"], questions):
        doc, qid = query["doc_id"], query["question_id"]
        units, candidates = documents[doc], set(query["candidate_ids"])
        for backend in source.BACKENDS:
            original = old_traces[(f"S_{backend}", *key)]
            relevance = {uid: labels[backend][support[(doc, qid, uid)]] for uid in candidates}
            relations = [replay.AdjacentRelation(task["left_id"], task["right_id"], labels[backend][task["id"]])
                         for task in prepared["static_tasks"] if task["doc_id"] == doc
                         and task["left_id"] in candidates and task["right_id"] in candidates]
            for direction in DIRECTIONS:
                value = replay_direction(units, query["candidate_ids"], relevance, relations,
                    query["ranked_ids"], original["accepted_merge_edge_ids"], direction=direction,
                    tokenizer=tokenizer, **{name: pilot.SELECTOR[name]
                                           for name in ("budget", "chunk_budget", "max_units")})
                if direction == "symmetric":
                    require_symmetric_parity(value, original)
                if value["eligible_ids"] != [uid for uid in original["candidate_ids"]
                                              if relevance[uid] != "no"]:
                    raise ValueError("direction changed support eligibility")
                method = f"{direction}_{backend}"
                identity = {**dict(zip(source.IDENTITY, key)), "method": method}
                traces.append({**identity, **value})
                row = {**identity, "budget": pilot.SELECTOR["budget"],
                       **score_selection(units, value["selected_indices"], annotations[(doc, qid)],
                                         PackCounter(tokenizer, units), pilot.SELECTOR["budget"])}
                if direction == "symmetric":
                    expected = {**old_rows[(f"S_{backend}", *key)], "method": method}
                    if row != expected:
                        raise ValueError("symmetric metrics differ from original S")
                rows.append(row)
    return rows, traces


def summarize(prepared, records, traces, original_records):
    questions = [tuple(row[name] for name in source.IDENTITY) for row in prepared["queries"]]
    methods = tuple(f"{direction}_{backend}" for backend in source.BACKENDS for direction in DIRECTIONS)
    rows = source.index_rows(records, methods, questions)
    trace_rows = source.index_rows(traces, methods, questions)
    original = source.index_rows(original_records, source.METHODS, questions)
    comparisons, usage = {}, {}
    for backend in source.BACKENDS:
        for direction in DIRECTIONS:
            method = f"{direction}_{backend}"
            table = {key: rows[(method, *key)] for key in questions}
            comparisons[method] = {}
            totals = Counter()
            for baseline in ("I", "S"):
                prior = {key: original[(f"{baseline}_{backend}", *key)] for key in questions}
                comparisons[method][f"minus_original_{baseline}"] = {
                    "paired_metrics": source.compare(table, prior, questions),
                    "changed_selected_sets": sum(set(table[k]["selected_ids"]) != set(prior[k]["selected_ids"])
                                                 for k in questions),
                    "changed_pack_hashes": sum(table[k]["pack_sha256"] != prior[k]["pack_sha256"] for k in questions)}
            for key in questions:
                trace = trace_rows[(method, *key)]
                steps, opportunities = trace["selection_trace"], trace["bonus_opportunities"]
                totals["eligible_query_unit_occurrences"] += len(trace["eligible_ids"])
                totals["excluded_no_query_unit_occurrences"] += len(trace["excluded_no_ids"])
                totals["fixed_accepted_edge_query_occurrences"] += len(trace["accepted_merge_edge_ids"])
                totals["questions_with_bonus_opportunity"] += any(x["eligible_bonus_candidate_ids"] for x in opportunities)
                totals["bonus_opportunity_steps"] += sum(bool(x["eligible_bonus_candidate_ids"]) for x in opportunities)
                totals["questions_with_excluded_no_link"] += any(x["excluded_no_linked_ids"] for x in opportunities)
                totals["excluded_no_linked_query_units"] += len({uid for x in opportunities for uid in x["excluded_no_linked_ids"]})
                totals["questions_with_bonus_selected"] += bool(trace["selection_bonus_used_edge_ids"])
                totals["bonus_selected_steps"] += sum(x["accepted"] and x["relation_bonus"] > 0 for x in steps)
                totals["priority_changed_steps"] += trace["selection_priority_changed_steps"]
                totals["questions_with_priority_change"] += trace["selection_priority_changed_steps"] > 0
                totals["priority_changed_accepted_steps"] += sum(x["accepted"] and x["relation_changed_priority"] for x in steps)
            usage[method] = dict(totals)
    return {"question_count": len(questions), "family_count": len({key[0] for key in questions}),
            "record_count": len(records), "metrics": aggregate(records),
            "comparisons": comparisons, "trace_usage": usage,
            "symmetric_step_and_metric_parity": True, "support_eligibility_unchanged": True,
            "original_partition_and_accepted_edges_fixed": True}


def analyze(args):
    plan_dir, run_dir, prepared_dir = (Path(getattr(args, name)).resolve() for name in ("plan", "run", "prepared"))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    hashes = {}

    def bind(path):
        path = Path(path).resolve()
        value = digest(path)
        if str(path) in hashes and hashes[str(path)] != value:
            raise ValueError("direction input changed while reading")
        hashes[str(path)] = value
        return path

    for path in (Path(__file__), Path(source.__file__), plan_dir / "experiment_config.json", plan_dir / "plan_manifest.json"):
        bind(path)
    config, _ = pilot.load_plan(plan_dir)
    hashes.update({str(Path(path).resolve()): value for path, value in config["input_sha256"].items()})
    for name, expected in config["plan_files_sha256"].items():
        hashes[str((plan_dir / name).resolve())] = expected
    for name in ("summary.json", "labels.json", "per_question.jsonl", "traces.jsonl"):
        bind(run_dir / name)
    for path in sorted((run_dir / "provider_calls").glob("*.json")):
        bind(path)
    for path in sorted(run_dir.glob("reuse_*.json")):
        bind(path)
    verified = source.analyze(SimpleNamespace(plan=plan_dir, run=run_dir, prepared=prepared_dir,
                                              output=output / "source_verification"))
    bind(output / "source_verification" / "analysis.json")
    pilot.verify_hashes(hashes)
    prepared, _, documents = pilot.load_prepared(prepared_dir)
    annotations = pilot.selected_gold(config["sidecar"], prepared)
    labels = pilot.read_json(run_dir / "labels.json")
    original_records = source.read_rows(run_dir / "per_question.jsonl")
    original_traces = source.read_rows(run_dir / "traces.jsonl")
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    records, traces = direction_records(prepared, documents, labels, annotations, original_records, original_traces, tokenizer)
    report = summarize(prepared, records, traces, original_records)
    pilot.verify_hashes(hashes)
    report.update(schema="slac-qasper-relation-direction-v1", status="completed", version=VERSION,
        analysis_type="post-hoc exploratory direction diagnostic; not independent confirmation",
        api_calls=0, key_read=False, test_payload_read=False, significance_claimed=False,
        source_verification_binding_sha256=verified["input_binding_sha256"],
        input_hashes_unchanged=True, input_binding_sha256=pilot.client.object_hash(hashes),
        input_sha256=[{"path_sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(), "content_sha256": value}
                      for path, value in sorted(hashes.items())],
        interpretation_limits=["Static labels specify B depends on A; this changes only bonus direction, not labels.",
            "All directions retain original S accepted merge edges and the hard no exclusion.",
            "Recorded blocked-no opportunities are eligibility observations, not evidence of usefulness.",
            "Equal maximum budgets do not imply equal actual evidence lengths.",
            "No query-conditioned judgments, answer generation, training, cache timing or independent evaluation occurred."])
    pilot.write_rows(output / "per_question.jsonl", records)
    pilot.write_rows(output / "traces.jsonl", traces)
    pilot.write_json(output / "analysis.json", report)
    if (source.read_rows(output / "per_question.jsonl") != records or source.read_rows(output / "traces.jsonl") != traces
            or pilot.read_json(output / "analysis.json") != report):
        raise ValueError("direction output round-trip mismatch")
    return report


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
    print(json.dumps({key: report[key] for key in ("status", "question_count", "record_count", "input_binding_sha256")}))
