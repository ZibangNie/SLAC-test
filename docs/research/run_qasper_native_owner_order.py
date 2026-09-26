"""Post-hoc CPU diagnostic: reorder each frozen native-owner candidate pool.

Prepare binds the complete audited source without evaluating the new control.
Run/audit first replay the upstream experiment, then compare only original owner
order versus cached leaf-score order in the identical per-question candidate set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import analyze_qasper_extended_results as statistics
import run_qasper_native_dual_index_v2 as upstream

pilot, bridge = upstream.pilot, upstream.bridge
SCHEMA = "slac-qasper-native-owner-order-diagnostic-v1"
OWNERS = ("leaf_owner", "dual_owner")
ORDERS = ("original_owner", "leaf_score")
SCOPES, METRICS = upstream.SCOPES, upstream.METRICS
IDENTITY = ("family_id", "doc_id", "question_id")
TIMES = ("retrieval_wall_seconds", "packing_wall_seconds")
SOURCE_FILES = ("summary.json", "public_aggregate.json", "per_question.jsonl",
                "chunk_index.json", "chunk_embeddings.safetensors")
RUN_FILES = ("per_question.jsonl", "public_aggregate.json", "summary.json")
CONFIG = {"questions": 77, "families": 24, "leaves": 1850,
    "scopes": list(SCOPES), "owner_methods": list(OWNERS), "orders": list(ORDERS),
    "records": 616, "comparisons": 4, "candidate_cap": 16,
    "candidate_source": "exact saved candidate_global_indices of each native-v2 owner arm",
    "control": "descending cached CPU FP32 leaf inner product, then global cache row",
    "evidence_budget_bge_tokens": 1024, "max_selected_units": 3,
    "packing": "unchanged whole-native (document,text) deduplication and global-source rendering",
    "bootstrap_seed": statistics.BOOTSTRAP_SEED,
    "bootstrap_replicates": statistics.BOOTSTRAP_REPLICATES,
    "resampling": "shared family-cluster PCG64 multinomial draws; linear percentile 95%",
    "paired_orientation": "leaf_score minus original_owner within each owner method and scope",
    "multiple_comparison_adjustment": "none", "posthoc": True,
    "gold_used_for_ranking_selection": False, "cpu_threads": 1, "max_seconds": 1200}
LIMITS = [
    "Post-hoc diagnosis after observing native-v2 results on exposed development questions; not preregistered or independent confirmation.",
    "Only packing priority changes; no new candidates, embeddings, model inference, support labels, API calls, or answer generation.",
    "Original and control candidates are identical within each pair; candidate recall must be exactly unchanged.",
    "This tests the owner-priority bottleneck, not the causal effect of RRF repeated votes, JEV relations, or Boundary Refiner.",
    "All four comparisons, all 16 metrics, both weightings, and all positive and negative outcomes are reported; no best setting is selected.",
    "Equal maximum budgets do not imply equal realized tokens; whole native units can differ in length.",
    "CPU timing includes upstream verification and cached-vector reuse, not independent arm speed or cold-start latency.",
    "Source-qualified scoring and duplicate unit headers follow the native experiment; these packs are not an answer-generation format.",
]


def snapshot(directory, names):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != set(names):
        raise ValueError("artifact file inventory differs")
    buffers = {name: (directory / name).read_bytes() for name in names}
    return buffers, {str(directory / name): hashlib.sha256(blob).hexdigest()
                     for name, blob in buffers.items()}


def strip_times(row):
    return {key: value for key, value in row.items() if key not in TIMES}


def row_key(row):
    return (row["scope"], row["method"], *(row[name] for name in IDENTITY))


def validate_source_rows(rows, plan):
    questions = plan["identity"]["queries"]
    identities = [tuple(q[k] for k in IDENTITY) for q in questions]
    leaves = plan["identity"]["leaves"]
    expected = {(scope, method, *key) for scope in SCOPES for method in upstream.METHODS for key in identities}
    indexed = {row_key(row): row for row in rows}
    if (len(identities) != CONFIG["questions"] or len(set(identities)) != len(identities)
            or len({key[0] for key in identities}) != CONFIG["families"]
            or len(leaves) != CONFIG["leaves"] or set(indexed) != expected or len(rows) != len(expected)):
        raise ValueError("source requires complete unique native-v2 denominator")
    for row in rows:
        candidates, selected = row["candidate_global_indices"], row["selected_global_indices"]
        if (not isinstance(candidates, list) or len(candidates) > CONFIG["candidate_cap"]
                or len(candidates) != len(set(candidates))
                or any(type(i) is not int or not 0 <= i < len(leaves) for i in candidates)
                or len(selected) != len(set(selected)) or not set(selected) <= set(candidates)
                or len(selected) > 3 or not 0 <= row["actual_evidence_tokens"] <= 1024
                or any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in METRICS)):
            raise ValueError("invalid saved candidate set/pack/metrics")
        if row["scope"] == "given_document" and any(leaves[i]["doc_id"] != row["doc_id"] for i in candidates):
            raise ValueError("given-document candidates cross source documents")
    owners = [row for row in rows if row["method"] in OWNERS]
    candidate_identity = [{**{key: row[key] for key in (*IDENTITY, "scope", "method")},
                           "candidate_global_indices": row["candidate_global_indices"]} for row in owners]
    return owners, candidate_identity


def source_snapshot(native_plan, native_run, native_audit):
    plan, _, plan_hashes = upstream.load_plan(native_plan)
    buffers, hashes = snapshot(native_run, SOURCE_FILES)
    report = json.loads(buffers["summary.json"])
    public = json.loads(buffers["public_aggregate.json"])
    if (report.get("status") != "completed" or report.get("schema") != upstream.SCHEMA
            or report.get("config") != upstream.CONFIG or report.get("plan_sha256") != plan_hashes
            or report.get("input_sha256") != plan["input_sha256"] or report.get("public") != public
            or report.get("output_sha256") != {name: hashes[str(Path(native_run).resolve() / name)]
                                             for name in SOURCE_FILES if name != "summary.json"}):
        raise ValueError("source run metadata/plan/output seals differ")
    audit_path = Path(native_audit).resolve()
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    expected_audit = {"status": "verified", "records": CONFIG["questions"] * 6, "configurations": 6,
        "saved_tensor_hash_index_norm_verified": True, "selection_metrics_and_comparisons_recomputed": True,
        "gpu_used": False, "model_loaded": False, "api_calls": 0}
    if any(audit.get(k) != v for k, v in expected_audit.items()):
        raise ValueError("a completed upstream CPU audit receipt is required")
    rows = [json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    owners, identity = validate_source_rows(rows, plan)
    bindings = upstream.merge_hashes(plan["input_sha256"], plan_hashes, hashes,
        {str(audit_path): hashlib.sha256(audit_bytes).hexdigest()},
        {str(Path(__file__).resolve()): upstream.digest(__file__),
         str(Path(statistics.__file__).resolve()): upstream.digest(statistics.__file__)})
    pilot.verify_hashes(bindings)
    return plan, owners, identity, bindings


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("diagnostic plan output exists")
    _, _, identity, bindings = source_snapshot(args.native_plan, args.native_run, args.native_audit)
    plan = {"schema": SCHEMA, "status": "prepared_not_scored", "config": CONFIG, "limits": LIMITS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "environment": upstream.environment(),
        "native_plan": str(Path(args.native_plan).resolve()), "native_run": str(Path(args.native_run).resolve()),
        "native_audit": str(Path(args.native_audit).resolve()), "input_sha256": bindings,
        "input_binding_sha256": pilot.stable_hash(bindings), "candidate_identity_sha256": pilot.stable_hash(identity),
        "original_owner_records": len(identity), "planned_records": 2 * len(identity),
        "api_calls": 0, "gpu_used": False, "model_inference_performed": False,
        "new_control_scored": False,
        "source_verification": "prepare checks recorded audit and sealed metadata; run/audit replay the upstream experiment before new scoring"}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "plan.json", plan)
    pilot.write_json(output / "plan_seal.json", {"plan.json": upstream.digest(output / "plan.json")})
    return {"status": plan["status"], "source_records": len(identity), "planned_records": 2 * len(identity),
            "plan_sha256": upstream.digest(output / "plan.json"), "new_control_scored": False, "gpu_used": False}


def load_plan(directory):
    buffers, hashes = snapshot(directory, ("plan.json", "plan_seal.json"))
    if json.loads(buffers["plan_seal.json"]) != {"plan.json": hashlib.sha256(buffers["plan.json"]).hexdigest()}:
        raise ValueError("diagnostic plan seal differs")
    plan = json.loads(buffers["plan.json"])
    fixed = {"schema": SCHEMA, "status": "prepared_not_scored", "config": CONFIG, "limits": LIMITS,
        "environment": upstream.environment(), "original_owner_records": CONFIG["records"] // 2,
        "planned_records": CONFIG["records"], "api_calls": 0, "gpu_used": False,
        "model_inference_performed": False, "new_control_scored": False,
        "source_verification": "prepare checks recorded audit and sealed metadata; run/audit replay the upstream experiment before new scoring"}
    if (set(plan) != set(fixed) | {"created_at_utc", "native_plan", "native_run", "native_audit", "input_sha256",
            "input_binding_sha256", "candidate_identity_sha256"} or any(plan.get(k) != v for k, v in fixed.items())
            or plan["input_binding_sha256"] != pilot.stable_hash(plan["input_sha256"])):
        raise ValueError("diagnostic specification/binding differs")
    pilot.verify_hashes(plan["input_sha256"])
    return plan, hashes


def verified_source(plan):
    original, owners, identity, bindings = source_snapshot(plan["native_plan"], plan["native_run"], plan["native_audit"])
    if bindings != plan["input_sha256"] or pilot.stable_hash(identity) != plan["candidate_identity_sha256"]:
        raise ValueError("diagnostic source identity changed")
    # The old audit proves every original selection/metric/trace and all 462 rows.
    # It uses saved CPU tensors, never instantiates the backbone or accesses CUDA.
    result = upstream.audit(SimpleNamespace(plan=plan["native_plan"], run=plan["native_run"]))
    if result["status"] != "verified" or result["records"] != CONFIG["questions"] * 6:
        raise ValueError("upstream complete replay failed")
    source = upstream.load_inputs(original["prepared"], original["dense"], original["chunks"])
    if source["input_sha256"] != original["input_sha256"]:
        raise ValueError("reloaded cached-vector source differs")
    pilot.verify_hashes(plan["input_sha256"])
    return source, owners


def order_candidates(candidates, scores):
    if (not isinstance(candidates, list) or len(set(candidates)) != len(candidates)
            or len(candidates) > CONFIG["candidate_cap"]
            or any(type(i) is not int or not 0 <= i < len(scores) for i in candidates)):
        raise ValueError("invalid frozen candidate identities")
    return bridge.rank_scores(scores, candidates)


def score_record(base, source, selected, counter):
    units, keys = source["units"], source["keys"]
    candidates = base["candidate_global_indices"]
    gold = source["qa_by_key"][(base["doc_id"], base["question_id"])]["answer_annotations"]
    pairs = lambda indices: [(keys[i][0], units[i].native_text) for i in indices]
    qualified = bridge.source_qualified_metrics(pairs(selected), base["doc_id"], gold)
    candidate = bridge.source_qualified_metrics(pairs(candidates), base["doc_id"], gold)
    full = bridge.source_qualified_metrics(pairs(source["positions"][base["doc_id"]]), base["doc_id"], gold)
    return {**{k: base[k] for k in (*IDENTITY, "scope", "method")},
        **upstream.score_selection(units, selected, gold, counter, 1024),
        "source_qualified_evidence_f1": qualified["evidence_f1"], "source_qualified_evidence_recall": qualified["evidence_recall"],
        "candidate_source_qualified_evidence_recall": candidate["evidence_recall"],
        "candidate_official_string_evidence_recall": upstream.evidence_metrics([units[i].native_text for i in candidates], gold)["evidence_recall"],
        "full_native_source_qualified_evidence_recall": full["evidence_recall"],
        "candidate_count": len(candidates), "candidate_document_count": len({keys[i][0] for i in candidates}),
        "empty_pack": int(not selected), "duplicate_rendered_headers_in_pack": len(selected) - len({units[i].unit_id for i in selected}),
        **bridge.duplicate_diagnostics(units, keys, candidates, selected, base["doc_id"]),
        "candidate_global_indices": candidates, "selected_global_indices": selected, "trace": base["trace"]}


def evaluate(source, owners, deadline=math.inf):
    by_key = {row_key(r): r for r in owners}
    records = []
    for q in source["prepared"]["queries"]:
        upstream.check_time(deadline)
        vector = source["query_vectors"][source["query_positions"][(q["doc_id"], q["question_id"])]]
        scores = (source["candidate_vectors"] @ vector).tolist()
        counter = upstream.PackCounter(source["tokenizer"], source["units"], deadline)
        selections = []
        for scope in SCOPES:
            for method in OWNERS:
                base = by_key[(scope, method, *(q[k] for k in IDENTITY))]
                candidates = base["candidate_global_indices"]
                reranked = order_candidates(candidates, scores)
                if set(reranked) != set(candidates):
                    raise ValueError("control changed candidate membership")
                for ordering, ordered in zip(ORDERS, (candidates, reranked), strict=True):
                    selected = bridge.pack_source_qualified(source["units"], source["keys"], ordered, 1024, counter, max_units=3)
                    selections.append((base, ordering, ordered, selected))
        # Every selection for this question is fixed before consulting annotations.
        for base, ordering, ordered, selected in selections:
            scored = score_record(base, source, selected, counter)
            if ordering == "original_owner" and scored != strip_times(base):
                raise ValueError("original owner order failed full record replay")
            if any(scored[k] != base[k] for k in ("candidate_source_qualified_evidence_recall",
                    "candidate_official_string_evidence_recall", "candidate_count", "candidate_document_count")):
                raise ValueError("candidate metrics changed under order-only control")
            records.append({**scored, "owner_method": base["method"], "ordering": ordering,
                "method": base["method"] + "__" + ordering,
                "packing_order_global_indices": ordered,
                "leaf_scores_in_candidate_order": [scores[i] for i in base["candidate_global_indices"]]})
    return records


def summarize(records, queries):
    identities = statistics.questions_from({"queries": queries})
    groups, draws = statistics.family_resamples(identities)
    expected = {(scope, owner + "__" + order, *key) for scope in SCOPES for owner in OWNERS for order in ORDERS for key in identities}
    indexed = {row_key(r): r for r in records}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("requires all eight arms on the same complete denominator")
    result = []
    for scope in SCOPES:
        for owner in OWNERS:
            tables, means = {}, []
            for order in ORDERS:
                method = owner + "__" + order
                rows = [indexed[(scope, method, *key)] for key in identities]
                table = {field: np.asarray([r[field] for r in rows], dtype=float) for field in METRICS}
                if any(not np.isfinite(values).all() for values in table.values()) or any(
                        not 0 <= r["actual_evidence_tokens"] <= 1024 or not 0 <= r["selected_units"] <= 3 for r in rows):
                    raise ValueError("invalid metric/pack in diagnostic")
                tables[order] = table
                means.append({"ordering": order, "metrics": {field: {"question_weighted": float(values.mean()),
                    "family_balanced": float(np.mean([values[g].mean() for g in groups]))} for field, values in table.items()}})
            paired = {}
            for field in METRICS:
                entry = statistics.clustered_delta(tables["leaf_score"][field] - tables["original_owner"][field], groups, draws, field)
                # Count and length increases are descriptive changes, not quality wins.
                if field not in METRICS[:8]:
                    entry.pop("question_wins", None); entry.pop("question_losses", None)
                    entry["interpretation"] = "positive means more/longer, not better quality"
                paired[field] = entry
            original = [indexed[(scope, owner + "__original_owner", *key)] for key in identities]
            control = [indexed[(scope, owner + "__leaf_score", *key)] for key in identities]
            if any(a["candidate_global_indices"] != b["candidate_global_indices"] or any(a[k] != b[k] for k in (
                    "candidate_source_qualified_evidence_recall", "candidate_official_string_evidence_recall", "candidate_count"))
                    for a, b in zip(original, control, strict=True)):
                raise ValueError("paired candidate set or candidate recall changed")
            result.append({"scope": scope, "owner_method": owner, "methods": means,
                "comparison": {"plus": "leaf_score", "minus": "original_owner", "metrics": paired,
                    "selected_set_changed_questions": sum(set(a["selected_global_indices"]) != set(b["selected_global_indices"])
                        for a, b in zip(original, control, strict=True)), "candidate_membership_and_recall_unchanged": True}})
    return {"comparisons": result, "shared_bootstrap_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}


def public_result(plan, source, records, execution):
    return {"schema": SCHEMA, "status": "completed", "config": CONFIG, "limits": LIMITS,
        "question_count": len(source["prepared"]["queries"]), "record_count": len(records),
        "environment": upstream.environment(), "input_binding_sha256": plan["input_binding_sha256"],
        "api_calls": 0, "gpu_used": False, "model_inference_performed": False, "test_payload_read": False,
        "posthoc": True, "independent_confirmation": False, "new_labels_used": False,
        "all_original_records_replayed": True, "all_candidate_sets_and_recall_unchanged": True,
        "execution": execution, **summarize(records, source["prepared"]["queries"])}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("diagnostic run output exists")
    wall = time.perf_counter(); deadline = time.monotonic() + CONFIG["max_seconds"]
    torch.set_num_threads(1)
    plan, plan_hashes = load_plan(args.plan)
    source, owners = verified_source(plan)
    pilot.verify_hashes(plan_hashes)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "started", "plan_sha256": plan_hashes, "input_sha256": plan["input_sha256"]}
    try:
        records = evaluate(source, owners, deadline)
        execution = {"device": "cpu", "threads": 1, "seconds_before_aggregate": time.perf_counter() - wall,
                     "timing_scope": "upstream audit, input loading, cached leaf scores and both packing orders; no per-arm speed claim"}
        public = public_result(plan, source, records, execution)
        pilot.verify_hashes(plan["input_sha256"]); pilot.verify_hashes(plan_hashes); upstream.check_time(deadline)
        with (output / "per_question.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        pilot.write_json(output / "public_aggregate.json", public)
        report.update(status="completed", public=public, output_sha256={name: upstream.digest(output/name) for name in RUN_FILES[:-1]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, retry_or_fallback=False)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - wall
        pilot.write_json(output / "summary.json", report)
    return {"status": "completed", "records": len(records), "posthoc": True, "gpu_used": False, "api_calls": 0}


def audit(args):
    deadline = time.monotonic() + CONFIG["max_seconds"]
    buffers, output_hashes = snapshot(args.run, RUN_FILES)
    plan, plan_hashes = load_plan(args.plan)
    source, owners = verified_source(plan)
    report, public = json.loads(buffers["summary.json"]), json.loads(buffers["public_aggregate.json"])
    if (set(report) != {"schema", "status", "plan_sha256", "input_sha256", "public", "output_sha256", "elapsed_seconds"}
            or report["schema"] != SCHEMA or report["status"] != "completed" or report["plan_sha256"] != plan_hashes
            or report["input_sha256"] != plan["input_sha256"] or report["public"] != public
            or report["output_sha256"] != {name: output_hashes[str(Path(args.run).resolve()/name)] for name in RUN_FILES[:-1]}):
        raise ValueError("diagnostic summary/output seals differ")
    execution = public["execution"]
    if (set(execution) != {"device", "threads", "seconds_before_aggregate", "timing_scope"}
            or execution["device"] != "cpu" or execution["threads"] != 1
            or execution["timing_scope"] != "upstream audit, input loading, cached leaf scores and both packing orders; no per-arm speed claim"
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (execution["seconds_before_aggregate"], report["elapsed_seconds"]))
            or not 0 <= execution["seconds_before_aggregate"] <= report["elapsed_seconds"] <= CONFIG["max_seconds"]):
        raise ValueError("diagnostic timing metadata invalid")
    saved = [json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    torch.set_num_threads(1)
    if saved != evaluate(source, owners, deadline):
        raise ValueError("complete original/control records failed replay")
    if public != public_result(plan, source, saved, execution):
        raise ValueError("full aggregate failed independent recomputation")
    pilot.verify_hashes(plan["input_sha256"]); pilot.verify_hashes(plan_hashes); pilot.verify_hashes(output_hashes)
    upstream.check_time(deadline)
    return {"status": "verified", "records": len(saved), "original_and_control_replayed": True,
        "candidate_membership_and_recall_unchanged": True, "all_four_comparisons_recomputed": True,
        "posthoc": True, "gpu_used": False, "model_inference_performed": False, "api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("native-plan", "native-run", "native-audit", "output"):
        p.add_argument("--" + name, required=True)
    for command, destination in (("run", "output"), ("audit", "run")):
        p = sub.add_parser(command); p.add_argument("--plan", required=True); p.add_argument("--" + destination, required=True)
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run, "audit": audit}[args.command](args), indent=2))


if __name__ == "__main__":
    main()
