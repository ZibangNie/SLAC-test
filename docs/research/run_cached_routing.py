"""Family-isolated routing between two complete audited answer caches.

No reference file, provider response, tokenizer or inference client is loaded.
Feature preparation is a separate, target-free stage. Only aggregates are public.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SLAC.retrieval.routing.offline_router import RoutingInput, assign_family_folds, oof_routes

ARTIFACTS = ROOT / "artifacts/research-foundation"
PHASE = ARTIFACTS / "offline-20261004"
PREPARED_PLAN_SHA = "862f79b8987c88f4cafbd176d3829e33a9bec26f6b2bc181cf35990b9bf2264a"
IDENTITY = ("family_id", "doc_id", "question_id")
METHODS = ("always_bge", "always_jev", "random_expectation", "low_gap", "gap_ridge", "four_feature_ridge")
PACK_FIELDS = ("selected_ids", "pack_sha256", "actual_evidence_tokens", "cache_key")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ARTIFACTS.resolve()) and path != ARTIFACTS.resolve(),
            "path must remain within research artifacts")
    return path


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def project_arm(raw_records, raw_mapping, method, expected):
    """Use only the named audited arm, requiring its exact original pack mapping."""
    mappings = {}
    for line in raw_mapping.splitlines():
        row = parse(line)
        if row["method"] != method:
            continue
        key = identity(row)
        require(key in expected and key not in mappings, "unexpected or duplicate arm mapping")
        mappings[key] = {k: row[k] for k in PACK_FIELDS}
    require(set(mappings) == set(expected), "incomplete original arm mapping")
    result = {}
    for line in raw_records.splitlines():
        row = parse(line)
        if row["method"] != method:
            continue
        key = identity(row)
        require(key in expected and key not in result, "unexpected or duplicate answer identity")
        pack = {k: row[k] for k in PACK_FIELDS}
        require(pack == mappings[key], "answer does not match its complete original pack")
        value, tokens = row["official_answer_f1"], row["actual_evidence_tokens"]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                "invalid cached Answer F1")
        require(type(tokens) is int and 0 <= tokens <= 1024, "invalid cached token count")
        require(isinstance(pack["selected_ids"], list) and len(pack["selected_ids"]) <= 3
                and all(isinstance(x, str) and x for x in pack["selected_ids"])
                and len(set(pack["selected_ids"])) == len(pack["selected_ids"]), "invalid cached pack identity")
        result[key] = {**pack, "answer_f1": value}
    require(set(result) == set(expected), "complete two-arm answer coverage required")
    return result


def verify_target_isolation(rows, targets, folds, original):
    """Perturb every held-out value; check only that fold's learned decisions."""
    reports = []
    for fold, before in zip(folds, original):
        changed = dict(targets)
        for key in fold.test_keys:
            changed[key] = -targets[key] if targets[key] != 0 else 1.0
        count = sum(changed[key] != targets[key] for key in fold.test_keys)
        require(count == len(fold.test_keys), "isolation probe must change every held-out target")
        after = oof_routes(rows, changed, folds)[fold.fold_index]
        require(canonical(asdict(before)) == canonical(asdict(after)), "held-out targets affected their own fold")
        reports.append({"fold_index": fold.fold_index, "changed_targets": count,
                        "prediction_and_routes_unchanged": True})
    return reports


def make_decisions(results):
    decisions = []
    for result in results:
        fold = result.fold
        sets = {"low_gap": set(result.low_gap_selected_keys),
                "gap_ridge": set(result.gap_ridge_selected_keys),
                "four_feature_ridge": set(result.four_feature_selected_keys)}
        require(all(len(value) == fold.quota and value <= set(fold.test_keys) for value in sets.values()),
                "routing quota differs")
        for i, key in enumerate(fold.test_keys):
            probabilities = {"always_bge": 0.0, "always_jev": 1.0,
                             "random_expectation": fold.quota / len(fold.test_keys),
                             **{method: float(key in selected) for method, selected in sets.items()}}
            decisions.append({**dict(zip(IDENTITY, key)), "fold_index": fold.fold_index,
                "gap_prediction": result.gap_predictions[i], "four_feature_prediction": result.four_feature_predictions[i],
                "jev_probabilities": probabilities})
    require(len({identity(row) for row in decisions}) == len(decisions), "duplicate out-of-fold decision")
    return sorted(decisions, key=identity)


def score_decisions(decisions, feature_rows, arms):
    features = {identity(row): row for row in feature_rows}
    require(set(features) == {identity(row) for row in decisions} == set(arms["bge"]) == set(arms["jev"]),
            "all-question scoring identities differ")
    records = []
    for row in decisions:
        key = identity(row)
        count = features[key]["support_task_count"]
        require(type(count) is int and 1 <= count <= 16, "invalid judgment count")
        for method in METHODS:
            p = row["jev_probabilities"][method]
            require(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1, "invalid routing probability")
            require(method == "random_expectation" or p in (0, 1), "deterministic strategy must select one arm")
            bge, jev = arms["bge"][key], arms["jev"][key]
            if p in (0, 1):
                selected = jev if p == 1 else bge
                answer_f1, tokens = selected["answer_f1"], selected["actual_evidence_tokens"]
            else:
                # Preserve exact ties when both saved arms have the same value.
                answer_f1 = bge["answer_f1"] + p * (jev["answer_f1"] - bge["answer_f1"])
                tokens = bge["actual_evidence_tokens"] + p * (jev["actual_evidence_tokens"] - bge["actual_evidence_tokens"])
            records.append({**dict(zip(IDENTITY, key)), "fold_index": row["fold_index"], "method": method,
                "answer_f1": answer_f1, "evidence_tokens": tokens,
                "jev_probability": p, "selected_support_judgments": p * count,
                "skipped_support_judgments": (1 - p) * count,
                "expectation_only": method == "random_expectation"})
    return records


def mean(values):
    return math.fsum(values) / len(values)


def weighted(values):
    families = defaultdict(list)
    for key, value in values.items():
        families[key[0]].append(value)
    family_means = [mean(v) for v in families.values()]
    return {"question_weighted": mean(list(values.values())), "family_balanced": mean(family_means)}, family_means


def wtl(values):
    return {"wins": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values), "losses": sum(v < 0 for v in values)}


def summarize(records):
    tables = {method: {} for method in METHODS}
    for row in records:
        key = identity(row)
        require(row["method"] in METHODS and key not in tables[row["method"]], "unexpected or duplicate score")
        tables[row["method"]][key] = row
    keys = set(tables["always_bge"])
    require(keys and all(set(table) == keys for table in tables.values()), "full strategy coverage required")
    metrics = {}
    for method, table in tables.items():
        p = math.fsum(r["jev_probability"] for r in table.values())
        per_fold = {}
        for fold in sorted({r["fold_index"] for r in table.values()}):
            subset = {key: r for key, r in table.items() if r["fold_index"] == fold}
            per_fold[str(fold)] = {"questions": len(subset),
                "answer_f1": weighted({key: r["answer_f1"] for key, r in subset.items()})[0],
                "jev_queries": math.fsum(r["jev_probability"] for r in subset.values())}
        metrics[method] = {"answer_f1": weighted({key: r["answer_f1"] for key, r in table.items()})[0],
            "evidence_tokens": weighted({key: r["evidence_tokens"] for key, r in table.items()})[0],
            "jev_queries": p, "skipped_jev_queries": len(keys) - p,
            "selected_support_judgments": math.fsum(r["selected_support_judgments"] for r in table.values()),
            "skipped_support_judgments": math.fsum(r["skipped_support_judgments"] for r in table.values()),
            "expectation_only": method == "random_expectation", "folds": per_fold}
    pairs = []
    for minus, plus in combinations(METHODS, 2):
        delta = {key: tables[plus][key]["answer_f1"] - tables[minus][key]["answer_f1"] for key in sorted(keys)}
        average, family_deltas = weighted(delta)
        pairs.append({"plus": plus, "minus": minus, "answer_f1_delta": average,
                      "question_win_tie_loss": wtl(list(delta.values())), "family_win_tie_loss": wtl(family_deltas)})
    contrasts = {r["minus"]: r["answer_f1_delta"] for r in pairs if r["plus"] == "four_feature_ridge"}
    primary, random = contrasts["gap_ridge"], contrasts["random_expectation"]
    gate = (primary["family_balanced"] > 0 and random["family_balanced"] > 0
            and primary["question_weighted"] >= 0 and random["question_weighted"] >= 0)
    return {"question_count": len(keys), "family_count": len({k[0] for k in keys}), "record_count": len(records),
        "methods": metrics, "paired_comparisons": pairs,
        "primary_comparison": {"plus": "four_feature_ridge", "minus": "gap_ridge", "answer_f1_delta": primary},
        "descriptive_structural_feature_gate_passed": gate}


def run(prepared_dir, output):
    prepared_dir, output = artifact(prepared_dir), artifact(output)
    require(not output.is_relative_to(prepared_dir) and not prepared_dir.is_relative_to(output),
            "output must not overlap frozen feature preparation")
    output.mkdir(parents=True, exist_ok=False)
    consumed = {}

    def bound(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "source must remain in the repository")
        raw = path.read_bytes()
        value = sha(raw)
        require(expected is None or value == expected, "frozen source hash mismatch")
        require(str(path) not in consumed or consumed[str(path)] == value, "source changed during run")
        consumed[str(path)] = value
        return raw

    prep = parse(bound(prepared_dir / "plan.json", PREPARED_PLAN_SHA))
    require(prep["status"] == "prepared_without_quality_targets" and prep["quality_targets_read"] is False
            and prep["question_count"] == 77 and prep["family_count"] == 24, "feature-only preparation required")
    for path, expected in {**prep["input_sha256"], **prep["code_sha256"]}.items():
        bound(path, expected)
    contract = parse(bound(artifact(prep["contract_path"]), prep["contract_sha256"]))
    require(contract["status"] == "direct_metadata_bindings_verified"
            and contract["common_generator_contract"]["actual_returned_model"] == "qwen/qwen3.6-plus"
            and contract["inherited_complete_bridge"]["common_setting_bridge"] == {"dense_exact_matches": 77, "empty_exact_matches": 77},
            "common audited generator bridge required")
    feature_rows = parse(bound(prepared_dir / "features.json", prep["artifact_sha256"]["features.json"]))
    saved_folds = parse(bound(prepared_dir / "folds.json", prep["artifact_sha256"]["folds.json"]))
    rows = tuple(RoutingInput(identity(row), tuple(row["features"])) for row in feature_rows)
    require(all(len(row.features) == 4 and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
                for x in row.features) for row in rows), "invalid frozen pre-JEV features")
    folds = assign_family_folds(tuple(row.key for row in rows))
    require(canonical([asdict(fold) for fold in folds]) == canonical(saved_folds), "feature-stage fold partition changed")
    keys = {row.key for row in rows}
    require(len(rows) == len(keys) == 77 and len({key[0] for key in keys}) == 24, "full frozen cohort required")
    for source in (Path(__file__), ROOT / "tests/research/test_cached_routing.py",
                   ROOT / "docs/research/CACHED_ROUTING_PROTOCOL_20261004.md"):
        bound(source)
    # Metadata-only verification inherits completed parent audits; no tree scan.
    bridge = contract["inherited_complete_bridge"]
    bound(ARTIFACTS / bridge["analysis"], bridge["analysis_sha256"])
    bound(ARTIFACTS / bridge["independent_review"], bridge["independent_review_sha256"])
    for arm in contract["arms"].values():
        for name in ("audit", "summary", "config", "model_ledger"):
            path_key = "model_ledger_path" if name == "model_ledger" else name
            bound(ARTIFACTS / arm[path_key], arm[f"{name}_sha256"])
    run_plan = {"schema": "slac-cached-routing-run-plan-v1", "input_sha256": dict(consumed),
        "deferred_quality_sources": contract["arms"], "api_calls": 0, "quality_targets_read": False,
        "methods": METHODS, "question_count": 77, "family_count": 24, "reference_files_read": False}
    write(output / "plan.json", run_plan)
    consumed[str(output / "plan.json")] = sha((output / "plan.json").read_bytes())

    arms = {}
    require(set(contract["arms"]) == {"bge", "jev"}, "exactly two original arms required")
    for name, info in contract["arms"].items():
        mapping = info["mapping"]
        arms[name] = project_arm(bound(artifact(ARTIFACTS / info["quality_records_path"]), info["quality_records_sha256"]),
            bound(artifact(ARTIFACTS / mapping["path"]), mapping["sha256"]), info["method"], keys)
    for row in feature_rows:
        require({k: arms["bge"][identity(row)][k] for k in PACK_FIELDS} == row["bge_pack"], "BGE arm differs from feature pack")
    targets = {key: arms["jev"][key]["answer_f1"] - arms["bge"][key]["answer_f1"] for key in keys}
    results = oof_routes(rows, targets, folds)
    decisions = make_decisions(results)
    write(output / "models_and_predictions.json", [asdict(r) for r in results])
    write(output / "decisions.json", decisions)
    isolation = verify_target_isolation(rows, targets, folds, results)
    write(output / "target_isolation.json", isolation)
    # Held-out metric aggregation happens only after immutable decisions exist.
    records = score_decisions(decisions, feature_rows, arms)
    result = {"schema": "slac-cached-routing-results-v1", "status": "completed", **summarize(records),
        "api_calls": 0, "key_read": False, "new_answers_generated": 0, "reference_files_read": False,
        "new_confirmation_data_read": False, "raw_provider_responses_read": False,
        "independent_confirmation": False, "confidence_intervals_computed": False,
        "primary_small_ridge_fits": 10, "isolation_probe_small_ridge_fits": 50, "dollar_savings_measured": False,
        "target_isolation_checks": isolation, "qa_status": "pending_fixed_sample_review",
        "input_content_sha256": sorted(consumed.values()),
        "limitations": ["Exposed development data; cached single-response realizations, not independent confirmation.",
                        "Uniform random row contains exact expectations, not a realized routing sample.",
                        "Logical skipped judgments are not physical API request or dollar savings.",
                        "Fixed folds, four features, ridge penalty and quotas were not tuned."]}
    for path, expected in consumed.items():
        require(sha(Path(path).read_bytes()) == expected, "frozen source changed during run")
    write(output / "per_question.json", records)
    result["output_content_sha256"] = {name: sha((output / name).read_bytes()) for name in
        ("plan.json", "models_and_predictions.json", "decisions.json", "target_isolation.json", "per_question.json")}
    write(output / "summary.json", result)
    return result


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled for cached routing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", default=str(PHASE / "routing-prepared-01"))
    parser.add_argument("--output", default=str(PHASE / "routing-run-01"))
    args = parser.parse_args()
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    try:
        print(json.dumps(run(args.prepared, args.output), ensure_ascii=True, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "api_calls": 0}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
