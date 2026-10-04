"""Score frozen development selections using existing references; zero inference.

Only aggregate summary.json is public-safe. This module never reads answer
caches, provider responses, keys, or the new confirmation reference dataset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import socket

from qasper_metrics import evidence_metrics

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/research-foundation"
PHASE = "offline-20261004"
METHODS = ("score_greedy", "score_exact", "density_greedy", "matched_resource_exact")
IDENTITY = ("family_id", "doc_id", "question_id")
PREPARED = "qasper-extended-development-prepared-01/prepared.json"
OLD_RECORDS = "qasper-primary-support-recovery-run-01/per_question.jsonl"
REFERENCE = "qasper-alignment-v2/native_qa_sidecar_v2.jsonl"
REFERENCE_SHA = "929f5cbdaf05e0c86d6e51a5c90265e729886c59b528824ddd014ca94d06a97f"
PINS = {
    "plan": (f"{PHASE}/budget-run-01/plan.json", "dca4932ad2462d6ab2263a682a8faff13ab45511a2e62757e8f96f025f938c8a"),
    "selections": (f"{PHASE}/budget-run-01/selections.json", "57043ecf73ad3d025ce6fa46062d2c74256805a34a09b4a6f912629ad22a9df1"),
    "summary": (f"{PHASE}/budget-run-01/summary.json", "753522d6defa19b37c5aba7e81896739ee22b5653bffe891382c56756324a467"),
    "contract": (f"{PHASE}/cache_contract.json", "dd1fe7cab7c176be82a27965e4bffc75bcc1f79d384df72f52efb81990d3e20c"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def artifact_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ARTIFACTS.resolve()) and path != ARTIFACTS.resolve(),
            "path must stay within the research artifact directory")
    return path


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def render(units, selected):
    return "\n\n".join(f"[{units[i]['unit_id']}]\n{units[i]['text']}"
                       for i in sorted(selected, key=lambda i: units[i]["order"]))


def index_selections(documents, queries, selections):
    """Validate complete frozen selections without opening any references."""
    query_map = {identity(q): q for q in queries}
    require(len(query_map) == len(queries) and bool(queries), "duplicate or empty question scope")
    result = {}
    for row in selections:
        key = (row["method"], identity(row))
        require(key not in result and key[0] in METHODS and key[1] in query_map,
                "duplicate or unexpected selection")
        units = documents[row["doc_id"]]
        chosen = row["selected_indexes"]
        require(isinstance(chosen, list) and all(type(i) is int and 0 <= i < len(units) for i in chosen),
                "invalid selected index")
        require(chosen == sorted(set(chosen)) and len(chosen) <= 3, "invalid selection cardinality or order")
        require({units[i]["unit_id"] for i in chosen} <= set(query_map[key[1]]["candidate_ids"]),
                "selection outside frozen candidates")
        require(len({units[i]["native_text"] for i in chosen}) == len(chosen), "duplicate native evidence")
        require(type(row["tokens"]) is int and 0 <= row["tokens"] <= 1024 and row["unit_count"] == len(chosen),
                "invalid saved resource count")
        require(row["pack_sha256"] == sha(render(units, chosen).encode("utf-8")), "frozen pack content differs")
        result[key] = row
    require(set(result) == {(method, key) for method in METHODS for key in query_map},
            "complete four-method coverage required")
    return query_map, result


def project_baseline(records, query_map, selections, documents):
    baseline = {}
    for row in records:
        if row["method"] != "p_yes_only_k3":
            continue
        key = identity(row)
        require(key in query_map and key not in baseline, "unexpected or duplicate historical baseline")
        frozen = selections["score_greedy", key]
        units = documents[row["doc_id"]]
        ids = [units[i]["unit_id"] for i in frozen["selected_indexes"]]
        require(row["selected_ids"] == ids and row["pack_sha256"] == frozen["pack_sha256"]
                and row["actual_evidence_tokens"] == frozen["tokens"], "historical baseline selection differs")
        value = row["official_evidence_f1"]
        require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                "invalid historical evidence metric")
        baseline[key] = value
    require(set(baseline) == set(query_map), "complete historical baseline required")
    return baseline


def project_references(raw, queries):
    expected = {(q["doc_id"], q["question_id"]): identity(q) for q in queries}
    require(len(expected) == len(queries), "duplicate reference identity in question scope")
    result = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = parse(line)
        key = (row["doc_id"], row["question_id"])
        if key not in expected:
            continue
        full_key = expected[key]
        require(identity(row) == full_key and full_key not in result, "duplicate or mismatched reference identity")
        result[full_key] = row["answer_annotations"]
    require(set(result) == set(expected.values()), "complete reference coverage required")
    return result


def score_selections(documents, queries, selections, references, historical_baseline):
    """Pure full-denominator score computation, with no file or network access."""
    query_map, indexed = index_selections(documents, queries, selections)
    require(set(references) == set(query_map) == set(historical_baseline), "complete scoring inputs required")
    records = []
    for method in METHODS:
        for key in sorted(query_map):
            selected = indexed[method, key]
            units = documents[key[1]]
            predicted = [units[i]["native_text"] for i in selected["selected_indexes"]]
            value = evidence_metrics(predicted, references[key], text_evidence_only=False)["evidence_f1"]
            require(math.isfinite(value) and 0 <= value <= 1, "invalid computed Evidence F1")
            if method == "score_greedy":
                require(value == historical_baseline[key], "historical Evidence F1 reproduction failed")
            records.append(dict(zip(IDENTITY, key)) | {"method": method, "official_evidence_f1": value})
    return records


def mean(values):
    return math.fsum(values) / len(values)


def weighted(values):
    families = defaultdict(list)
    for key, value in values.items():
        families[key[0]].append(value)
    family_values = [mean(v) for v in families.values()]
    return {"question_weighted": mean(list(values.values())), "family_balanced": mean(family_values)}, family_values


def win_tie_loss(values):
    return {"wins": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values), "losses": sum(v < 0 for v in values)}


def summarize(records, queries):
    keys = {identity(q) for q in queries}
    tables = {method: {} for method in METHODS}
    for row in records:
        key = identity(row)
        require(row["method"] in METHODS and key in keys and key not in tables[row["method"]],
                "unexpected or duplicate score")
        tables[row["method"]][key] = row["official_evidence_f1"]
    require(bool(keys) and all(set(table) == keys for table in tables.values()), "complete score coverage required")
    metrics = {method: weighted(tables[method])[0] for method in METHODS}
    pairs = []
    for method in METHODS[1:]:
        deltas = {key: tables[method][key] - tables["score_greedy"][key] for key in sorted(keys)}
        average, family_deltas = weighted(deltas)
        pairs.append({"plus": method, "minus": "score_greedy", "mean_delta": average,
                      "question_win_tie_loss": win_tie_loss(list(deltas.values())),
                      "family_win_tie_loss": win_tie_loss(family_deltas)})
    return {"question_count": len(keys), "family_count": len({key[0] for key in keys}),
            "record_count": len(records), "official_evidence_f1": metrics, "paired_comparisons": pairs}


def evaluate(output):
    output = artifact_path(output)
    output.mkdir(parents=True, exist_ok=False)
    consumed = {}

    def read_bound(path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        value = sha(raw)
        require(expected is None or value == expected, "frozen input binding mismatch")
        require(str(path) not in consumed or consumed[str(path)] == value, "input changed during evaluation")
        consumed[str(path)] = value
        return raw

    source = {name: parse(read_bound(artifact_path(ARTIFACTS / rel), pin)) for name, (rel, pin) in PINS.items()}
    plan, summary, contract = source["plan"], source["summary"], source["contract"]
    require(plan["schema"] == "slac-cached-budget-plan-v1" and plan["question_count"] == 77
            and plan["family_count"] == 24 and tuple(plan["methods"]) == METHODS and plan["reference_input"] is None,
            "frozen opportunity plan scope differs")
    require(summary["schema"] == "slac-cached-budget-opportunity-v1" and summary["status"] == "completed"
            and summary["baseline_reproduced_questions"] == summary["question_count"] == 77
            and summary["family_count"] == 24 and summary["quality_metrics_computed"] is False
            and summary["api_calls"] == summary["answer_generation_calls"] == 0
            and summary["methods"]["score_exact"]["utility_win_tie_loss"][0] > 0,
            "completed reference-free opportunity gate required")
    require(summary["input_content_sha256"] == sorted(plan["input_sha256"].values())
            and summary["code_content_sha256"] == sorted(plan["code_sha256"].values()), "opportunity commitment mismatch")
    for name in (PREPARED, OLD_RECORDS):
        path = artifact_path(ARTIFACTS / name)
        require(plan["input_sha256"].get(str(path)) == contract["verified_consumed_sha256"][name],
                "opportunity input lineage differs")
    prepared = parse(read_bound(ARTIFACTS / PREPARED, contract["verified_consumed_sha256"][PREPARED]))
    old_records = [parse(line) for line in read_bound(ARTIFACTS / OLD_RECORDS,
                   contract["verified_consumed_sha256"][OLD_RECORDS]).splitlines() if line.strip()]
    documents, queries = prepared["documents"], prepared["queries"]
    require(len(queries) == 77 and len(documents) == 24 and len({q["family_id"] for q in queries}) == 24,
            "development question scope differs")
    query_map, indexed = index_selections(documents, queries, source["selections"])
    historical = project_baseline(old_records, query_map, indexed, documents)
    for path, expected in plan["code_sha256"].items():
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "source commitment outside repository")
        read_bound(path, expected)
    scorer = ROOT / "docs/research/qasper_metrics.py"
    read_bound(scorer, contract["verified_source_code_sha256"]["docs/research/qasper_metrics.py"])
    for path in (Path(__file__).resolve(), ROOT / "docs/research/BUDGET_SELECTION_EVALUATION_PROTOCOL_20261004.md",
                 ROOT / "tests/research/test_evaluate_cached_budget_selection.py"):
        read_bound(path)
    reference = artifact_path(ARTIFACTS / REFERENCE)
    require(Path(contract["deferred_reference"]["path"]).resolve() == reference
            and contract["deferred_reference"]["expected_sha256"] == REFERENCE_SHA,
            "reference lineage differs")
    evaluation_plan = {"schema": "slac-cached-budget-evaluation-plan-v1", "status": "frozen_before_reference_read",
                       "input_sha256": dict(consumed), "reference": {"path": str(reference), "sha256": REFERENCE_SHA},
                       "methods": METHODS, "questions": 77, "families": 24, "quality_computed": False,
                       "api_calls": 0, "answer_cache_read": False, "new_confirmation_references_read": False}
    write_json(output / "plan.json", evaluation_plan)
    evaluation_plan_sha = sha((output / "plan.json").read_bytes())

    references = project_references(read_bound(reference, REFERENCE_SHA), queries)
    records = score_selections(documents, queries, source["selections"], references, historical)
    public = {"schema": "slac-cached-budget-evidence-evaluation-v1", "status": "completed",
              **summarize(records, queries), "baseline_reproduced_questions": 77,
              "api_calls": 0, "key_read": False, "answer_cache_read": False,
              "answer_metrics_computed": False, "new_confirmation_references_read": False,
              "independent_confirmation": False, "question_exclusions": 0, "confidence_intervals_computed": False,
              "selection_modified_after_reference_read": False, "evaluation_plan_sha256": evaluation_plan_sha,
              "input_content_sha256": sorted(consumed.values()),
              "interpretation": "descriptive exposed-development Evidence F1; no significance, novelty or Answer F1 claim"}
    for path, expected in consumed.items():
        require(sha(Path(path).read_bytes()) == expected, "input changed during evaluation")
    require(sha((output / "plan.json").read_bytes()) == evaluation_plan_sha, "evaluation plan changed")
    write_json(output / "per_question.json", records)
    public["per_question_sha256"] = sha((output / "per_question.json").read_bytes())
    write_json(output / "summary.json", public)
    return public


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled for this offline evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ARTIFACTS / PHASE / "budget-evaluation-01"))
    args = parser.parse_args()
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    try:
        print(json.dumps(evaluate(args.output), ensure_ascii=True, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "api_calls": 0}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
