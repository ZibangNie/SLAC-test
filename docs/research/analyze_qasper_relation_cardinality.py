"""Uniform k=1,2,3 post-hoc sensitivity analysis; no best-k selection or API.

Selection uses frozen support/relationship labels only. Gold is read solely for
scoring. Dense top-k shares candidates, rendering, token budget and unit cap.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import analyze_qasper_relation_pilot as audit
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, pack_ranked, score_selection


SIZES = (1, 2, 3)


def replay(prepared, documents, annotations, labels, tokenizer):
    records, traces = [], []
    support = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
               for task in prepared["support_tasks"]}
    for query in prepared["queries"]:
        doc, qid = query["doc_id"], query["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        by_id = {unit.unit_id: i for i, unit in enumerate(units)}
        candidates = set(query["candidate_ids"])
        count = PackCounter(tokenizer, units)
        identity = {name: query[name] for name in audit.IDENTITY}

        def record(method, chosen):
            return {**identity, "method": method, "budget": 1024,
                    **score_selection(units, chosen, gold, count, 1024)}

        for k in SIZES:
            dense = pack_ranked(units, [by_id[uid] for uid in query["ranked_ids"]], 1024, count, max_units=k)
            records.append(record(f"dense_k{k}", dense))
            for backend in audit.BACKENDS:
                relevance = {uid: labels[backend][support[(doc, qid, uid)]] for uid in candidates}
                relations = [pilot.AdjacentRelation(task["left_id"], task["right_id"], labels[backend][task["id"]])
                             for task in prepared["static_tasks"] if task["doc_id"] == doc
                             and task["left_id"] in candidates and task["right_id"] in candidates]
                for mode in ("I", "S"):
                    trace = pilot.replay_policy(units, query["candidate_ids"], relevance, relations,
                                               query["ranked_ids"], mode=mode, tokenizer=tokenizer,
                                               budget=1024, chunk_budget=384, max_units=k)
                    method = f"{mode}_{backend}_k{k}"
                    records.append(record(method, trace["selected_indices"]))
                    traces.append({**identity, "method": method, **trace})
    return records, traces


def analyze(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = digest(__file__)
    verified = audit.analyze(argparse.Namespace(plan=args.plan, run=args.run, prepared=args.prepared,
                                               output=str(output / "verified_inputs")))
    bindings = {row["path_sha256"]: row["content_sha256"] for row in verified["input_sha256"]}
    used = {}

    def checked(path):
        path = Path(path).resolve()
        expected = bindings[hashlib.sha256(str(path).encode()).hexdigest()]
        if digest(path) != expected:
            raise ValueError("input changed after verification")
        used[str(path)] = expected
        return path

    config, _ = pilot.load_plan(args.plan)
    checked(Path(args.plan) / "experiment_config.json")
    checked(Path(args.prepared) / "prepared.json")
    prepared, _, documents = pilot.load_prepared(args.prepared)
    labels = pilot.read_json(checked(Path(args.run) / "labels.json"))
    old_records = audit.read_rows(checked(Path(args.run) / "per_question.jsonl"))
    old_baseline = audit.read_rows(checked(Path(args.plan) / "baseline_per_question.jsonl"))
    annotations = pilot.selected_gold(checked(config["sidecar"]), prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    records, traces = replay(prepared, documents, annotations, labels, tokenizer)
    questions = [tuple(query[name] for name in audit.IDENTITY) for query in prepared["queries"]]
    methods = [f"{base}_k{k}" for k in SIZES for base in ("dense", "I_jev", "S_jev", "I_general", "S_general")]
    indexed = audit.index_rows(records, methods, questions)
    old_indexed = {(row["method"], *(row[name] for name in audit.IDENTITY)): row for row in old_records + old_baseline}
    for method in methods:
        if not method.endswith("_k3"):
            continue
        old_method = "dense_top3_capped" if method == "dense_k3" else method[:-3]
        for key in questions:
            row = {**indexed[(method, *key)], "method": old_method}
            if row != old_indexed[(old_method, *key)]:
                raise ValueError("k3 no longer matches the original frozen experiment")
    tables = {method: {key: indexed[(method, *key)] for key in questions} for method in methods}
    comparisons = []
    for k in SIZES:
        for backend in audit.BACKENDS:
            for plus, minus in ((f"I_{backend}_k{k}", f"dense_k{k}"),
                                (f"S_{backend}_k{k}", f"I_{backend}_k{k}")):
                comparisons.append({"plus": plus, "minus": minus, **audit.compare(tables[plus], tables[minus], questions)})
    pilot.load_plan(args.plan)
    pilot.verify_hashes(used)
    if digest(__file__) != source_hash:
        raise ValueError("analysis code changed")
    summary = {"schema": "slac-qasper-cardinality-v1", "status": "completed", "api_calls": 0,
               "analysis_type": "post-hoc uniform sensitivity analysis; no winning k promoted to main result",
               "sizes": list(SIZES), "record_count": len(records), "question_count": len(questions),
               "metrics": aggregate(records), "paired_comparisons": comparisons,
               "input_binding_sha256": verified["input_binding_sha256"], "script_sha256": source_hash,
               "k3_matches_original": True, "gold_used_for_selection": False,
               "limitations": ["Changing k changes available evidence length and may change answer utility.",
                               "Evidence F1 rewards precision; no answer generation or Answer F1 is measured.",
                               "All k values were evaluated on the same exposed development questions.",
                               "The k sweep was motivated by the preceding gold-assisted diagnostic."]}
    pilot.write_rows(output / "per_question.jsonl", records)
    pilot.write_rows(output / "traces.jsonl", traces)
    pilot.write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    result = analyze(parser.parse_args())
    print({"status": result["status"], "record_count": result["record_count"], "api_calls": 0})
