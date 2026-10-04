"""Join only existing numeric endpoints for the frozen single-addition diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/research"))
from cached_addition_analysis import AdditionEdge, CachedEndpoint, analyze_additions

ARTIFACTS = ROOT / "artifacts/research-foundation"
PHASE = ARTIFACTS / "offline-20261004"
IDENTITY = ("family_id", "doc_id", "question_id")
PACK_FIELDS = ("selected_ids", "pack_sha256", "cache_key", "actual_evidence_tokens")
PREPARED_PLAN_SHA = "777686edc433967883d3251e538181b3d00435439681775db534df46903d96a7"
LABEL_SOURCE = "qasper-primary-support-recovery-run-01/labels.json"
LABEL_SHA = "c00c0eab01381b738bef9d674f95bb6f94e04cd906bcd289565ce8928a734092"
SCORE_SOURCE = "qasper-primary-support-recovery-run-01/raw_scores.json"
SCORE_SHA = "06494c70f4bbbd4eb03627aa5ad40bcea4ae41039bb71b6532fb7deff68b00cc"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ARTIFACTS.resolve()) and path != ARTIFACTS.resolve(),
            "path must remain inside research artifacts")
    return path


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def identity(row):
    return tuple(row[field] for field in IDENTITY)


def project_score_rows(raw, expected):
    """Project required cache identities only; no quality-dependent membership.

    Unused fields (including saved answer text) are parsed in their old JSON
    container but are never returned, inspected, or used for a decision.
    """
    scores, counts = {}, {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = parse(line)
        cache_id = row["cache_key"]
        if cache_id not in expected:
            continue
        frozen = expected[cache_id]
        require(identity(row) == identity(frozen), "cached endpoint question differs")
        require({field: row[field] for field in PACK_FIELDS}
                == {field: frozen[field] for field in PACK_FIELDS}, "cached endpoint pack differs")
        value = row["official_answer_f1"]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                "invalid saved official Answer F1")
        if cache_id in scores:
            require(scores[cache_id] == value, "same cache identity has conflicting scores")
        scores[cache_id] = value
        counts[cache_id] = counts.get(cache_id, 0) + 1
    return scores, counts


def merge_scores(parts, expected):
    result = {}
    for part in parts:
        require(set(part) <= set(expected), "unexpected scored endpoint")
        for key, value in part.items():
            if key in result:
                require(result[key] == value, "same cache identity has conflicting scores across sources")
            result[key] = value
    require(set(result) == set(expected), "missing required endpoint; cannot shrink manifest")
    return result


def project_judgments(labels, scores, task_ids):
    result = {}
    jev = labels["jev"]
    require(task_ids <= set(jev) and task_ids <= set(scores), "missing added-unit judgment")
    for task_id in sorted(task_ids):
        label, value = jev[task_id], scores[task_id]["yes"]
        require(label in {"yes", "no", "unknown"}, "invalid cached JEV support label")
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                "invalid raw JEV yes score")
        result[task_id] = {"support_label": label, "raw_yes_score": value}
    return result


def scored_edges(manifest, endpoint_scores, judgments):
    result = []
    for row in manifest:
        key = identity(row)
        endpoints = []
        for name in ("base", "larger"):
            endpoint = row[name]
            require(endpoint["cache_key"] in endpoint_scores, "missing edge endpoint")
            endpoints.append(CachedEndpoint(key=key, unit_ids=tuple(endpoint["selected_ids"]),
                cache_id=endpoint["cache_key"], answer_f1=endpoint_scores[endpoint["cache_key"]],
                evidence_tokens=endpoint["actual_evidence_tokens"]))
        task_id = row["added_support_task_id"]
        require(task_id in judgments, "missing added-unit judgment")
        result.append(AdditionEdge(key=key, subset=endpoints[0], superset=endpoints[1],
            added_unit_id=row["added_unit_id"], support_task_id=task_id, **judgments[task_id]))
    return result


def run(prepared_dir, output):
    prepared_dir, output = artifact(prepared_dir), artifact(output)
    require(not output.is_relative_to(prepared_dir) and not prepared_dir.is_relative_to(output),
            "output must not overlap frozen preparation")
    output.mkdir(parents=True, exist_ok=False)
    consumed = {}

    def bound(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "bound input must remain in the repository")
        raw = path.read_bytes()
        digest = sha(raw)
        require(expected is None or digest == expected, "frozen source hash mismatch")
        require(str(path) not in consumed or consumed[str(path)] == digest, "input changed during diagnostic")
        consumed[str(path)] = digest
        return raw

    prep = parse(bound(prepared_dir / "plan.json", PREPARED_PLAN_SHA))
    require(prep["status"] == "prepared_without_quality_or_support_labels"
            and prep["quality_targets_read"] is False and prep["support_labels_or_scores_read"] is False,
            "metadata-only frozen preparation required")
    for path, expected_sha in (prep["input_sha256"] | prep["code_sha256"]).items():
        bound(path, expected_sha)
    manifest = parse(bound(prepared_dir / "edges.json", prep["artifact_sha256"]["edges.json"]))
    controls = parse(bound(prepared_dir / "controls.json", prep["artifact_sha256"]["controls.json"]))
    require(controls == prep["controls"] and controls["all"]["paired_edges"] == len(manifest) == 32
            and controls["all"]["questions"] == 22 and controls["all"]["families"] == 15,
            "complete fixed addition coverage required")
    pool_keys = [tuple(key) for key in prep["pool_keys"]]
    require(len(pool_keys) == len(set(pool_keys)) == 77 and len({k[0] for k in pool_keys}) == 24,
            "complete original denominator required")
    sample_plan = parse(bound(PHASE / "routing-prepared-01/plan.json",
        "862f79b8987c88f4cafbd176d3829e33a9bec26f6b2bc181cf35990b9bf2264a"))
    require(prep["sample_keys"] == sample_plan["sample_keys"], "fixed question sample changed")
    expected, expected_by_origin = {}, {}
    for row in manifest:
        require(identity(row) in pool_keys, "manifest question outside original pool")
        for name in ("base", "larger"):
            endpoint = row[name]
            projected = {**dict(zip(IDENTITY, identity(row))),
                         **{field: endpoint[field] for field in PACK_FIELDS}}
            cache_id = endpoint["cache_key"]
            require(cache_id not in expected or expected[cache_id] == projected, "manifest cache conflict")
            expected[cache_id] = projected
            for source in endpoint["sources"]:
                expected_by_origin.setdefault(source["origin"], {})[cache_id] = projected
    require(len(expected) == 54, "all 54 cached endpoints required")
    quality_files = []
    for source in prep["deferred_quality_sources"]:
        summary_path = artifact(ARTIFACTS / source["summary_path"])
        old_summary = parse(bound(summary_path, source["summary_sha256"]))
        require(old_summary["status"] == "completed" and old_summary["main_results_available"] is True,
                "completed audited parent required")
        for name in source["record_names"]:
            require(name in {"per_question.jsonl", "pack_records.jsonl"}, "unsupported numeric record file")
            quality_files.append({"origin": source["origin"], "path": str(summary_path.parent / name),
                                  "sha256": old_summary["output_sha256"][name]})
    require(len(quality_files) == 5 and set(expected_by_origin) == {"local", "owner", "primary", "relation"},
            "all four completed sources required")
    for path in (Path(__file__), ROOT / "docs/research/cached_addition_analysis.py",
                 ROOT / "tests/research/test_cached_addition_runner.py",
                 ROOT / "tests/research/test_cached_addition_analysis.py",
                 ROOT / "docs/research/CACHED_ADDITION_PROTOCOL_20261004.md"):
        bound(path)
    plan = {"schema": "slac-cached-addition-run-plan-v1", "status": "frozen_before_endpoint_scores",
            "preparation_sha256": PREPARED_PLAN_SHA, "input_sha256": dict(consumed),
            "quality_files": quality_files, "label_source_sha256": LABEL_SHA, "score_source_sha256": SCORE_SHA,
            "fixed_cells": 12, "expected_edges": 32, "expected_endpoints": 54,
            "qa_sample_keys": prep["sample_keys"], "qa_sample_edge_ids": prep["sample_edge_ids"],
            "api_calls": 0, "key_read": False, "analysis_may_not_shrink_manifest": True}
    write(output / "plan.json", plan)

    parts, source_coverage = [], {}
    for source in quality_files:
        projected, row_counts = project_score_rows(bound(source["path"], source["sha256"]), expected)
        require(set(projected) <= set(expected_by_origin[source["origin"]]), "unexpected source-cache binding")
        parts.append(projected)
        coverage = source_coverage.setdefault(source["origin"], {"seen": set(), "record_occurrences": 0})
        coverage["seen"].update(projected)
        coverage["record_occurrences"] += sum(row_counts.values())
    for origin, coverage in source_coverage.items():
        require(coverage["seen"] == set(expected_by_origin[origin]), "missing original source endpoint")
    scores = merge_scores(parts, expected)
    task_ids = {row["added_support_task_id"] for row in manifest}
    judgments = project_judgments(parse(bound(ARTIFACTS / LABEL_SOURCE, LABEL_SHA)),
                                 parse(bound(ARTIFACTS / SCORE_SOURCE, SCORE_SHA)), task_ids)
    edges = scored_edges(manifest, scores, judgments)
    aggregates = analyze_additions(edges, pool_keys=pool_keys, expected_edge_count=32)
    require(len(aggregates["strata"]) == 12, "all predeclared crossed strata required")
    write(output / "projected_edges.json", [asdict(edge) for edge in edges])
    for path, expected_sha in consumed.items():
        require(sha(Path(path).read_bytes()) == expected_sha, "source changed during diagnostic")
    summary = {"schema": "slac-cached-addition-result-v1", "status": "completed", "qa_status": "pending_fixed_sample",
        "preparation_sha256": PREPARED_PLAN_SHA, "aggregates": aggregates,
        "endpoint_count": len(expected), "unique_added_support_tasks": len(task_ids),
        "source_coverage": {origin: {"unique_required_endpoints": len(value["seen"]),
                                    "record_occurrences": value["record_occurrences"]}
                            for origin, value in source_coverage.items()},
        "scope": {"api_calls": 0, "key_read": False, "new_answers": 0, "references_read": False,
                  "raw_provider_responses_read": False, "answer_strings_used_or_retained": False,
                  "old_score_containers_parsed_then_projected": True, "independent_confirmation": False,
                  "new_selector_tested": False, "complete_inclusion_squares": 0},
        "output_sha256": {name: sha((output / name).read_bytes()) for name in ("plan.json", "projected_edges.json")},
        "consumed_content_sha256": sorted(set(consumed.values())),
        "limitations": ["Conditional observed additions among exposed development caches; uncovered questions are not imputed.",
                        "One cached response per payload; content, position and generation variation are not separated.",
                        "Support usefulness is not a promise of marginal Answer F1 improvement.",
                        "Repeated edges and crossed cells are dependent; no significance, causal or second-order interaction claim."]}
    write(output / "summary.json", summary)
    return {"status": "completed", "edges": len(edges), "endpoints": len(expected), "api_calls": 0,
            "summary_sha256": sha((output / "summary.json").read_bytes()), "qa_status": summary["qa_status"]}


def blocked_socket(*args, **kwargs):
    raise RuntimeError("network disabled for cached addition diagnostic")


if __name__ == "__main__":
    socket.socket = blocked_socket
    socket.create_connection = blocked_socket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PHASE / "addition-prepared-01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prepared, args.output)
    print(json.dumps(result, sort_keys=True))
