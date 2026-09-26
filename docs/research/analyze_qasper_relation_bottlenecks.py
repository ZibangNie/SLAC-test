"""Gold-assisted, post-hoc bottleneck diagnosis; never a deployable selector.

Enumerate every distinct-native-text subset of <=3 units in the frozen <=16
candidate pool. This includes alternative duplicate locations and nongold units
and therefore does not assume that removing a unit monotonically reduces BGE
token counts. References never enter the original model judgments or replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from pathlib import Path

import analyze_qasper_relation_pilot as audit
import run_qasper_relation_pilot as pilot
from qasper_metrics import evidence_metrics, references_from_annotations
from run_qasper_evidence_baselines import PackCounter, digest


def exhaustive_options(units, pool, annotations, count, max_units=3):
    pool = sorted(pool)
    if (not pool or len(pool) > 16 or len(pool) != len(set(pool))
            or any(type(i) is not int or not 0 <= i < len(units) for i in pool)
            or not 1 <= max_units <= 3):
        raise ValueError("requires unique frozen candidate indices, at most 16, and <=3 units")
    options = []
    for size in range(min(max_units, len(pool)) + 1):
        for selected in combinations(pool, size):
            native = [units[i].native_text for i in selected]
            if len(native) != len(set(native)):
                continue
            metrics = evidence_metrics(native, annotations)
            options.append({"indices": selected, "tokens": count(selected),
                            "f1": metrics["evidence_f1"], "recall": metrics["evidence_recall"]})
    return options


def best_option(options, eligible, *, budget=None, exact_size=None):
    eligible = set(eligible)
    allowed = [row for row in options if set(row["indices"]) <= eligible
               and (budget is None or row["tokens"] <= budget)
               and (exact_size is None or len(row["indices"]) == exact_size)]
    if not allowed:
        raise ValueError("no feasible oracle subset for the specified constraints")
    # Stable source order resolves residual ties, without model-label tuning.
    return max(allowed, key=lambda row: (row["f1"], row["recall"], -row["tokens"]))


def availability_ceiling(units, pool, annotations):
    """Exact F1 ceiling with no cardinality/token limit, only native availability.

    For each official reference, selecting every available distinct matching
    native text maximizes its F1; then maximize over references as Qasper does.
    The empty set remains a candidate. Reference duplicates are not removed.
    """
    available = {units[i].native_text for i in pool}
    choices = [[]] + [sorted(set(ref["evidence"]) & available)
                     for ref in references_from_annotations(annotations)]
    return max(evidence_metrics(choice, annotations)["evidence_f1"] for choice in choices)


def rescue_option(options, eligible, edges, *, budget=1024):
    """One-hop prerequisite rescue, with its eligible B retained in the pack.

    A previously excluded A is allowed only if (A, B) is an allowed edge and B
    is both eligible and selected. No unanchored or recursive no-to-no rescue.
    """
    eligible, edges = set(eligible), set(edges)
    allowed = []
    for row in options:
        selected = set(row["indices"])
        if row["tokens"] <= budget and all(
                any((a, b) in edges for b in selected & eligible) for a in selected - eligible):
            allowed.append(row)
    if not allowed:
        raise ValueError("no feasible anchored rescue subset")
    return max(allowed, key=lambda row: (row["f1"], row["recall"], -row["tokens"]))


def means(values, keys):
    families = defaultdict(list)
    documents = defaultdict(list)
    for value, key in zip(values, keys, strict=True):
        families[key[0]].append(value)
        documents[key[1]].append(value)
    mean = lambda xs: sum(xs) / len(xs)
    return {"questions": len(values), "question_macro": mean(values),
            "family_macro": mean([mean(v) for v in families.values()]),
            "document_macro": mean([mean(v) for v in documents.values()])}


def summarize(rows):
    keys = [tuple(row[name] for name in audit.IDENTITY) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("nonempty unique question coverage required")
    levels = list(rows[0]["f1_levels"])
    if any(set(row["f1_levels"]) != set(levels) for row in rows):
        raise ValueError("inconsistent bottleneck levels")
    summary = {"question_count": len(rows), "family_count": len({key[0] for key in keys}),
               "f1_levels": {level: means([row["f1_levels"][level] for row in rows], keys)
                             for level in levels}, "decomposition": {}, "support_reference_overlap": {}}
    for backend in audit.BACKENDS:
        chain = ["full_native_unconstrained", "capped_unconstrained", "capped_max3_no_token_cap",
                 "capped_max3_budget", f"eligible_{backend}_max3_budget",
                 f"eligible_{backend}_observed_count_budget", f"observed_I_{backend}"]
        names = ["candidate_omission", "unit_limit", "token_limit", "support_no_exclusion",
                 "observed_cardinality_restriction", "composition_at_observed_cardinality"]
        summary["decomposition"][backend] = {}
        for high, low, name in zip(chain, chain[1:], names):
            deltas = [row["f1_levels"][high] - row["f1_levels"][low] for row in rows]
            if any(value < -1e-12 for value in deltas):
                raise ValueError("oracle hierarchy is not monotone")
            summary["decomposition"][backend][name] = {
                **means(deltas, keys), "questions_with_positive_gap": sum(value > 1e-12 for value in deltas)}
        counts = Counter()
        for row in rows:
            counts.update(row["support_reference_overlap"][backend])
        summary["support_reference_overlap"][backend] = dict(counts)
    return summary


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
            raise ValueError("input changed after completed-run validation")
        used[str(path)] = expected
        return path

    config, _ = pilot.load_plan(args.plan)
    checked(Path(args.plan) / "experiment_config.json")
    checked(Path(args.prepared) / "prepared.json")
    prepared, _, documents = pilot.load_prepared(args.prepared)
    labels = pilot.read_json(checked(Path(args.run) / "labels.json"))
    records = audit.read_rows(checked(Path(args.run) / "per_question.jsonl"))
    annotations = pilot.selected_gold(checked(config["sidecar"]), prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    observed = {(row["method"], row["doc_id"], row["question_id"]): row for row in records}
    support = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
               for task in prepared["support_tasks"]}
    rows = []
    for query in prepared["queries"]:
        doc, qid = query["doc_id"], query["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        by_id = {unit.unit_id: index for index, unit in enumerate(units)}
        pool = [by_id[uid] for uid in query["candidate_ids"]]
        count = PackCounter(tokenizer, units)
        options = exhaustive_options(units, pool, gold, count)
        selected = {"capped_max3_no_token_cap": best_option(options, pool),
                    "capped_max3_budget": best_option(options, pool, budget=1024)}
        levels = {"full_native_unconstrained": availability_ceiling(units, range(len(units)), gold),
                  "capped_unconstrained": availability_ceiling(units, pool, gold)}
        reference_texts = {text for ref in references_from_annotations(gold) for text in ref["evidence"]}
        overlap = {}
        for backend in audit.BACKENDS:
            unit_labels = {i: labels[backend][support[(doc, qid, units[i].unit_id)]] for i in pool}
            eligible = [i for i in pool if unit_labels[i] != "no"]
            positive = [i for i in pool if unit_labels[i] == "yes"]
            actual = observed[(f"I_{backend}", doc, qid)]
            actual_indices = {by_id[uid] for uid in actual["selected_ids"]}
            if not actual_indices <= set(eligible) or len(actual_indices) != actual["selected_units"]:
                raise ValueError("observed selection violates support eligibility")
            selected[f"eligible_{backend}_max3_budget"] = best_option(options, eligible, budget=1024)
            selected[f"eligible_{backend}_observed_count_budget"] = best_option(
                options, eligible, budget=1024, exact_size=actual["selected_units"])
            selected[f"yes_only_{backend}_max3_budget"] = best_option(options, positive, budget=1024)
            # Explicitly gold-assisted opportunity checks, never model inputs.
            adjacent = {(a, a + 1) for a in pool if a + 1 in set(pool)}
            dependent = {(by_id[task["left_id"]], by_id[task["right_id"]])
                         for task in prepared["static_tasks"] if task["doc_id"] == doc
                         and task["left_id"] in query["candidate_ids"]
                         and task["right_id"] in query["candidate_ids"]
                         and labels[backend][task["id"]] == "dependent"}
            selected[f"adjacent_anchored_rescue_{backend}"] = rescue_option(options, eligible, adjacent)
            selected[f"static_anchored_rescue_{backend}"] = rescue_option(options, eligible, dependent)
            levels[f"observed_I_{backend}"] = actual["official_evidence_f1"]
            counts = Counter()
            for i, label in unit_labels.items():
                group = "reference_matched" if units[i].native_text in reference_texts else "not_reference_matched"
                counts[f"{group}_{label}"] += 1
            overlap[backend] = dict(counts)
        levels.update({name: value["f1"] for name, value in selected.items()})
        rows.append({**{name: query[name] for name in audit.IDENTITY}, "f1_levels": levels,
                     "support_reference_overlap": overlap,
                     "oracle_selections": {name: {"selected_ids": [units[i].unit_id for i in row["indices"]],
                                                   "tokens": row["tokens"], "f1": row["f1"]}
                                           for name, row in selected.items()},
                     "enumerated_distinct_text_subsets": len(options)})
    result = summarize(rows)
    pilot.load_plan(args.plan)
    pilot.verify_hashes(used)
    if digest(__file__) != script_hash:
        raise ValueError("diagnostic script changed during execution")
    result.update(schema="slac-qasper-bottleneck-v1", status="completed", api_calls=0,
                  analysis_type="post-hoc gold-assisted diagnostic; not deployable or independent confirmation",
                  input_binding_sha256=verified["input_binding_sha256"], script_sha256=script_hash,
                  plan_sha256=digest(Path(args.plan) / "experiment_config.json"),
                  raw_records_publishable=False,
                  limitations=["Oracle references are unavailable to deployed selectors.",
                               "Decomposition is a nested feasible-set diagnostic, not causal effect estimation.",
                               "Reference-nonmatching units may still help answer generation.",
                               "Rescue oracles require the eligible B anchor to remain selected and never recurse through no labels.",
                               "Full/capped availability ceilings ignore token and unit-count limits.",
                               "No model judgments, candidates, thresholds or original scores were changed."])
    pilot.write_rows(output / "per_question.jsonl", rows)
    pilot.write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    result = analyze(parser.parse_args())
    print({"status": result["status"], "question_count": result["question_count"], "api_calls": 0})
