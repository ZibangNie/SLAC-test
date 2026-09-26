"""Document-only screen of ALL remaining validation metadata; no evaluation pool.

Reuses the frozen exact-shingle screen unchanged. All IDs, source paths and pair
records stay local; only complete eight-batch aggregates may be published.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_qasper_overlap as old

SCHEMA = "slac-qasper-remaining-validation-screen-v1"
PURPOSE = "screen-only-not-evaluation-pool"
BATCH_SIZE = 32
MAX_SECONDS = 300
EXPECTED = {"qasper_train": 888, "qasper_validation": 281,
            "legacy_train": 8443, "legacy_dev": 1056, "current": 32}
SCOPES = ("qasper_train", "current_development", "remaining_validation", "legacy_train", "legacy_dev")
COMPLETED = ["canonical_qasper_train_validation_and_candidate_pairs", "legacy_train", "legacy_dev"]


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def guard_flags():
    return dict(purpose=PURPOSE, api_calls=0, qa_payload_read=False,
                official_test_payload_read=False, holdout_selected=False,
                documents_automatically_excluded=0, cleared_for_evaluation=False,
                near_duplicate_clearance=False, independent_confirmation_test=False)


def sources():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(old.__file__),
             root / "tests/research/test_qasper_remaining_validation.py"]
    return {str(p.resolve()): old.digest(p) for p in paths}


def batches_for(ids):
    ids = sorted(ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("empty or duplicate remaining identities")
    return [ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]


def verify_hashes(bindings, index_path):
    # Reject a changed/mixed index before hashing any canonical payload bytes.
    ordered = [str(index_path)] + sorted(set(bindings) - {str(index_path)})
    for raw in ordered:
        if old.digest(raw) != bindings[raw]:
            raise ValueError("bound source hash changed")


def metadata_inputs(pool_manifest, prior_screen, metadata_review):
    pool_path, prior_path, review_path = [Path(p).resolve(strict=True)
                                         for p in (pool_manifest, prior_screen, metadata_review)]
    before = {str(p): old.digest(p) for p in (pool_path, prior_path, review_path)}
    pool, prior, review = map(read_json, (pool_path, prior_path, review_path))
    if prior.get("status") != "complete_lexical_screen_with_limits" or prior.get("all_input_hashes_unchanged") is not True or prior.get("skipped_documents") != [] or prior.get("completed_scopes") != COMPLETED:
        raise ValueError("prior document-only screen is incomplete")
    if prior.get("source_sha256") != old.digest(old.__file__):
        raise ValueError("frozen screen source differs")
    if prior.get("qa_payload_read") is not False or prior.get("test_payload_read") is not False:
        raise ValueError("prior scope does not establish document-only inputs")
    counts = pool.get("counts", {})
    if (pool.get("all_input_hashes_unchanged") is not True or
            counts.get("qa_field_present") != 0 or
            counts.get("qa_not_exported_metadata") != EXPECTED["qasper_train"] + EXPECTED["qasper_validation"] or
            counts.get("canonical_train") != EXPECTED["qasper_train"] or
            counts.get("canonical_validation") != EXPECTED["qasper_validation"]):
        raise ValueError("missing complete no-QA canonical provenance")
    allowed, index, shards, legacy, candidates = old.input_paths(pool_path, pool)
    allowed = {str(p): h for p, h in allowed.items()}
    if any(prior["input_sha256"].get(p) != h for p, h in allowed.items()):
        raise ValueError("input is not bound by the prior screen")
    if prior["input_sha256"].get(str(pool_path)) != before[str(pool_path)]:
        raise ValueError("pool manifest differs from prior screen")
    # Only the metadata index is read before validating all declared splits.
    if old.digest(index) != allowed[str(index)]:
        raise ValueError("canonical index hash changed")
    rows = [r for _, r in old.jsonl(index)]
    known = {r["doc_id"]: r for r in rows}
    if len(known) != len(rows) or any(r.get("source") != "qasper" or r.get("original_split") not in {"train", "validation"} for r in rows):
        raise ValueError("unsupported index identities/splits; payload not opened")
    if Counter(r["original_split"] for r in rows) != {"train": EXPECTED["qasper_train"], "validation": EXPECTED["qasper_validation"]}:
        raise ValueError("canonical metadata counts differ")
    if {r["shard"] for r in rows} != {p.name for p in shards}:
        raise ValueError("metadata shard set differs")
    for scope in ("qasper_train", "qasper_validation", "legacy_train", "legacy_dev"):
        if prior["counts"].get(scope + "_documents") != EXPECTED[scope]:
            raise ValueError("prior reference denominator differs")
    eligibility_path = pool_path.parent / "eligibility_audit.json"
    # The earlier metadata-only receipt binds eligibility, pool, candidates/index.
    for path in (pool_path, index, candidates, eligibility_path):
        expected = review["source_sha256"].get(str(path))
        if expected is None or old.digest(path) != expected:
            raise ValueError("metadata receipt binding differs")
        before[str(path)] = expected
    if old.digest(candidates) != pool["candidate_manifest_sha256"]:
        raise ValueError("current candidate manifest changed")
    current_rows = [r for _, r in old.jsonl(candidates)]
    current = {r["doc_id"] for r in current_rows}
    validation = {r["doc_id"] for r in rows if r["original_split"] == "validation"}
    eligible_list = read_json(eligibility_path)["eligible_ids"]
    if len(eligible_list) != len(set(eligible_list)) or set(eligible_list) != validation:
        raise ValueError("eligibility must cover all validation identities exactly")
    if len(current_rows) != EXPECTED["current"] or len(current) != len(current_rows) or not current <= validation or pool["selected_documents"] != len(current):
        raise ValueError("current development identities differ")
    for row in current_rows:
        if row.get("official_split") != "validation" or any(row.get(k) != known[row["doc_id"]].get(k) for k in ("family_id", "source_id", "normalized_body_sha256")):
            raise ValueError("current candidate metadata differs from index")
    before.update(allowed)
    verify_hashes(before, index)
    return dict(pool_path=str(pool_path), prior_path=str(prior_path), review_path=str(review_path),
                index_path=str(index), known=known, current=sorted(current),
                remaining=sorted(validation - current), allowed=allowed, bindings=before,
                prior_legacy_lineage={k: v for k, v in prior["counts"].items() if "_orig_split_" in k})


def prepare(pool_manifest, prior_screen, metadata_review, output_dir):
    data = metadata_inputs(pool_manifest, prior_screen, metadata_review)
    batch_ids = batches_for(data["remaining"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for number, ids in enumerate(batch_ids, 1):
        prefix = f"batch-{number:02d}"
        directory = output / prefix
        directory.mkdir()
        candidate_rows = [{**{k: data["known"][doc_id][k] for k in ("doc_id", "source_id", "family_id", "normalized_body_sha256")},
                           "official_split": "validation", "purpose": PURPOSE} for doc_id in ids]
        write_rows(directory / "candidates.jsonl", candidate_rows)
        write_json(directory / "pool_manifest.json", {
            "schema": SCHEMA + "-batch", **guard_flags(), "selected_documents": len(ids),
            "selection_policy": "All validation minus existing development32; sorted doc_id; fixed consecutive batches, no quality-based selection.",
            "source_split": "validation", "input_sha256": data["allowed"],
            "parent_pool_manifest_sha256": data["bindings"][data["pool_path"]],
            "candidate_manifest_sha256": old.digest(directory / "candidates.jsonl")})
        files.extend([f"{prefix}/candidates.jsonl", f"{prefix}/pool_manifest.json"])
    plan = {"schema": SCHEMA, **guard_flags(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_size": BATCH_SIZE, "max_seconds_per_batch": MAX_SECONDS, "expected": EXPECTED,
            "batch_doc_ids": batch_ids, "current_doc_ids": data["current"],
            "input_sha256": data["bindings"], "source_sha256": sources(),
            "inputs": {k: data[k] for k in ("pool_path", "prior_path", "review_path", "index_path")},
            "document_only_proof": "Unchanged canonical hashes from prior complete pool audit: qa_field_present=0 and qa_exported=False metadata for all 1169 canonical rows. Index permits train/validation only before payload hash/read.",
            "prior_legacy_lineage": data["prior_legacy_lineage"]}
    write_json(output / "plan.json", plan)
    files.append("plan.json")
    write_json(output / "prepare_seal.json", {name: old.digest(output / name) for name in sorted(files)})
    return {"status": "prepared", "remaining_documents": len(data["remaining"]), "batch_sizes": list(map(len, batch_ids)), **guard_flags()}


def load_plan(directory):
    directory = Path(directory).resolve(strict=True)
    plan = read_json(directory / "plan.json")
    if plan.get("schema") != SCHEMA or any(plan.get(k) != v for k, v in guard_flags().items()) or plan.get("expected") != EXPECTED or plan.get("source_sha256") != sources() or plan.get("batch_size") != BATCH_SIZE or plan.get("max_seconds_per_batch") != MAX_SECONDS:
        raise ValueError("plan/source contract differs")
    names = {"plan.json"} | {f"batch-{i:02d}/{name}" for i in range(1, len(plan["batch_doc_ids"]) + 1) for name in ("candidates.jsonl", "pool_manifest.json")}
    seal = read_json(directory / "prepare_seal.json")
    if set(seal) != names or any(old.digest(directory / name) != value for name, value in seal.items()):
        raise ValueError("prepare seal differs")
    inputs = plan["inputs"]
    data = metadata_inputs(inputs["pool_path"], inputs["prior_path"], inputs["review_path"])
    if plan["input_sha256"] != data["bindings"] or plan["batch_doc_ids"] != batches_for(data["remaining"]) or plan["current_doc_ids"] != data["current"] or plan["prior_legacy_lineage"] != data["prior_legacy_lineage"]:
        raise ValueError("plan does not cover remaining validation exactly")
    for number, ids in enumerate(plan["batch_doc_ids"], 1):
        folder = directory / f"batch-{number:02d}"
        manifest = read_json(folder / "pool_manifest.json")
        rows = [r for _, r in old.jsonl(folder / "candidates.jsonl")]
        if [r["doc_id"] for r in rows] != ids or manifest.get("purpose") != PURPOSE or manifest["input_sha256"] != data["allowed"]:
            raise ValueError("derived batch differs")
    return plan, data


def pair_identity(pair, remaining, current, known):
    query, reference = pair["query_doc_id"], pair["reference_doc_id"]
    if query not in remaining:
        raise ValueError("flag query outside remaining identities")
    scope = pair["reference_scope"]
    if scope.startswith("qasper_"):
        row = known.get(reference)
        if row is None or scope != "qasper_" + row["original_split"] or reference == query:
            raise ValueError("invalid canonical reference")
        role = "qasper_train" if scope == "qasper_train" else ("current_development" if reference in current else "remaining_validation")
        endpoints = sorted((query, reference))
        key = (role, *endpoints)
        reversed_pair = query != endpoints[0]
    elif scope in {"legacy_train", "legacy_dev"}:
        if type(pair["reference_line"]) is not int or not 1 <= pair["reference_line"] <= EXPECTED[scope]:
            raise ValueError("invalid legacy source row")
        role, key, reversed_pair = scope, (scope, query, reference, pair["reference_line"]), False
    else:
        raise ValueError("unsupported reference scope")
    metrics = pair["metrics"]
    shared, qn, rn = (metrics[k] for k in ("shared_shingles", "query_shingles", "reference_shingles"))
    if any(type(v) is not int or v < 0 for v in (shared, qn, rn)) or shared > min(qn, rn) or metrics != old.overlap(shared, qn, rn):
        raise ValueError("invalid exact overlap metrics")
    reasons = pair["review_reasons"]
    permitted = {"high_lexical_overlap_review", "moderate_lexical_overlap_review", "declared_family_identity_review", "arxiv_base_identity_review"}
    if not reasons or len(reasons) != len(set(reasons)) or not set(reasons) <= permitted or set(old.lexical_flags(metrics)) != set(reasons) & {"high_lexical_overlap_review", "moderate_lexical_overlap_review"}:
        raise ValueError("invalid review reasons")
    signature = (shared, rn if reversed_pair else qn, qn if reversed_pair else rn, tuple(sorted(reasons)))
    return role, key, signature


def aggregate(directory, plan, data):
    directory = Path(directory)
    remaining, current = set(data["remaining"]), set(data["current"])
    raw, unique, seen_keys, receipts, flagged_queries = [], {}, set(), [], set()
    raw_counts, unique_counts, reasons_raw, reasons_unique = Counter(), Counter(), Counter(), Counter()
    scan_times, total_times, lineage = [], [], []
    for number, ids in enumerate(plan["batch_doc_ids"], 1):
        folder = directory / f"batch-{number:02d}"
        result = folder / "screen"
        report = read_json(result / "overlap_report.json")
        if report.get("status") != "complete_lexical_screen_with_limits" or report.get("all_input_hashes_unchanged") is not True or report.get("skipped_documents") != [] or report.get("completed_scopes") != COMPLETED or report.get("source_sha256") != old.digest(old.__file__):
            raise ValueError("not all batches completed unchanged")
        if any(report.get(k) is not v for k, v in {"qa_payload_read": False, "test_payload_read": False, "cleared_for_evaluation": False, "near_duplicate_clearance": False}.items()):
            raise ValueError("batch scope/clearance differs")
        expected_inputs = {**data["allowed"], **{str(folder / name): old.digest(folder / name) for name in ("pool_manifest.json", "candidates.jsonl")}}
        if report["input_sha256"] != expected_inputs:
            raise ValueError("batch input inventory differs")
        if set(report["output_sha256"]) != {"flagged_pairs.jsonl", "nearest_pairs.jsonl"} or any(old.digest(result / name) != h for name, h in report["output_sha256"].items()):
            raise ValueError("batch output hash differs")
        n, counts = len(ids), report["counts"]
        expected_counts = {"query_documents": n}
        for scope in ("qasper_train", "qasper_validation", "legacy_train", "legacy_dev"):
            expected_counts[scope + "_documents"] = EXPECTED[scope]
            expected_counts[scope + "_pairs"] = n * EXPECTED[scope] if scope != "qasper_validation" else n * (EXPECTED[scope] - n) + n * (n - 1) // 2
        if any(counts.get(k) != v for k, v in expected_counts.items()):
            raise ValueError("incomplete batch denominator")
        batch_lineage = {k: v for k, v in counts.items() if "_orig_split_" in k}
        if batch_lineage != plan["prior_legacy_lineage"]:
            raise ValueError("legacy source lineage changed")
        lineage.append(batch_lineage)
        pairs = [r for _, r in old.jsonl(result / "flagged_pairs.jsonl")]
        if len(pairs) != report["flagged_pairs"] or dict(Counter(reason for p in pairs for reason in p["review_reasons"])) != report["flagged_reason_counts"]:
            raise ValueError("flag count differs")
        for pair in pairs:
            if pair["query_doc_id"] not in ids:
                raise ValueError("query in wrong batch")
            role, key, signature = pair_identity(pair, remaining, current, data["known"])
            batch_key = (number, key)
            if batch_key in seen_keys:
                raise ValueError("duplicate pair inside one batch")
            seen_keys.add(batch_key)
            flagged_queries.add(pair["query_doc_id"])
            if role == "remaining_validation":
                flagged_queries.add(pair["reference_doc_id"])
            raw.append({"batch": number, "comparison_scope": role, **pair})
            raw_counts[role] += 1
            reasons_raw.update(pair["review_reasons"])
            if key in unique:
                if unique[key]["signature"] != signature:
                    raise ValueError("reverse cross-batch pair disagrees")
                unique[key]["raw_occurrences"] += 1
            else:
                unique[key] = {"pair_key": list(key), "comparison_scope": role,
                               "signature": signature, "raw_occurrences": 1,
                               "representative": pair}
                unique_counts[role] += 1
                reasons_unique.update(pair["review_reasons"])
        scan_times.append(report["scan_elapsed_seconds"])
        total_times.append(report["elapsed_seconds"])
        receipts.append({"batch": number, "documents": n,
                         "report_sha256": old.digest(result / "overlap_report.json"),
                         "flagged_pairs_sha256": report["output_sha256"]["flagged_pairs.jsonl"]})
    batch_of = {doc_id: number for number, ids in enumerate(plan["batch_doc_ids"]) for doc_id in ids}
    for key, pair in unique.items():
        expected_occurrences = 2 if key[0] == "remaining_validation" and batch_of[key[1]] != batch_of[key[2]] else 1
        if pair["raw_occurrences"] != expected_occurrences:
            raise ValueError("missing reverse cross-batch flag occurrence")
    verify_hashes(plan["input_sha256"], data["index_path"])
    if plan["source_sha256"] != sources():
        raise ValueError("screen sources changed during execution")
    count = len(remaining)
    within_batch_pairs = sum(len(b) * (len(b) - 1) // 2 for b in plan["batch_doc_ids"])
    unique_denominators = {"qasper_train": count * EXPECTED["qasper_train"],
                           "current_development": count * len(current),
                           "remaining_validation": count * (count - 1) // 2,
                           "legacy_train": count * EXPECTED["legacy_train"], "legacy_dev": count * EXPECTED["legacy_dev"]}
    raw_denominators = {**unique_denominators, "remaining_validation": count * (count - 1) - within_batch_pairs}
    public = {"schema": SCHEMA + "-aggregate", "status": "complete_document_screen_with_limits", **guard_flags(),
              "validation_documents": EXPECTED["qasper_validation"], "current_development_documents": len(current),
              "remaining_documents": count, "completed_batches": len(receipts), "batch_sizes": list(map(len, plan["batch_doc_ids"])),
              "reference_documents_unique": {k: EXPECTED[k] for k in ("qasper_train", "qasper_validation", "legacy_train", "legacy_dev")},
              "comparison_scopes": {scope: {"raw_pairs_compared": raw_denominators[scope], "unique_pairs_compared": unique_denominators[scope],
                                            "raw_flagged_pairs": raw_counts[scope], "unique_flagged_pairs": unique_counts[scope]} for scope in SCOPES},
              "flagged_pairs_raw": len(raw), "flagged_pairs_unique": len(unique),
              "flagged_remaining_documents": len(flagged_queries),
              "flagged_reason_counts_raw": dict(reasons_raw), "flagged_reason_counts_unique": dict(reasons_unique),
              "legacy_orig_split_counts_unique_source_rows": lineage[0],
              "legacy_orig_split_counts_across_batch_scans": dict(sum((Counter(x) for x in lineage), Counter())),
              "scan_seconds_per_batch": scan_times, "total_seconds_per_batch_including_hash_checks": total_times,
              "max_seconds_per_batch": MAX_SECONDS, "all_input_hashes_unchanged": True,
              "algorithm": "Frozen screen_qasper_overlap.py; exact unique Unicode word 5-grams; NFC/casefold; declared headings excluded; no sketches or sampling.",
              "thresholds": read_json(directory / "batch-01/screen/overlap_report.json")["thresholds"],
              "batch_receipts": receipts,
              "plan_sha256": old.digest(directory / "plan.json"), "prepare_seal_sha256": old.digest(directory / "prepare_seal.json"),
              "source_hashes": {Path(p).name: h for p, h in plan["source_sha256"].items()},
              "limitations": ["Review flags do not establish leakage, same-paper identity or independence; zero flags do not establish clearance.",
                              "No new QA, answers or official test payload; only prior-bound document train/validation and named legacy train/dev are opened.",
                              "Legacy orig_split=test is historical lineage inside named train/dev inputs, not an opened test file or test evaluation.",
                              "Cross-batch remaining pairs occur twice; within-batch pairs once. Canonical pairs deduplicate as unordered document pairs; legacy row identity includes file split and line.",
                              "Index lacks explicit version metadata; arXiv base-ID matching is a review signal, not completed version/family adjudication.",
                              "No paraphrase/translation/OCR guarantee, no automatic exclusion or holdout selection; exposure, manual family review and protocol freeze remain gates.",
                              "Each 300-second cap covers scanning; mandatory final hash verification is outside the cap, as in the frozen screen."]}
    return public, raw, list(unique.values())


def run(directory):
    directory = Path(directory).resolve(strict=True)
    plan, data = load_plan(directory)
    write_json(directory / "execution_started.json", {"plan_sha256": old.digest(directory / "plan.json"), **guard_flags()})
    completed = 0
    started = time.monotonic()
    try:
        for number, _ in enumerate(plan["batch_doc_ids"], 1):
            folder = directory / f"batch-{number:02d}"
            report = old.screen(folder / "pool_manifest.json", folder / "screen", MAX_SECONDS)
            if report["status"] != "complete_lexical_screen_with_limits" or report["all_input_hashes_unchanged"] is not True:
                raise ValueError("batch screen incomplete; no full aggregate")
            completed += 1
            print(json.dumps({"completed_batches": completed, "total_batches": len(plan["batch_doc_ids"])}), flush=True)
        public, raw, unique = aggregate(directory, plan, data)
        write_rows(directory / "flagged_pairs_raw.jsonl", raw)
        write_rows(directory / "flagged_pairs_unique.jsonl", unique)
        write_json(directory / "public_aggregate.json", public)
        write_json(directory / "completion.json", {"status": "completed", "elapsed_seconds": time.monotonic() - started,
                   "output_sha256": {name: old.digest(directory / name) for name in ("public_aggregate.json", "flagged_pairs_raw.jsonl", "flagged_pairs_unique.jsonl")}})
        return public
    except BaseException as error:
        write_json(directory / "failure.json", {"status": "incomplete_no_full_population_result", "completed_batches": completed,
                   "error_type": type(error).__name__, **guard_flags()})
        raise


def audit(directory):
    directory = Path(directory).resolve(strict=True)
    plan, data = load_plan(directory)
    if (directory / "failure.json").exists():
        raise ValueError("failed run has no complete audit")
    completion = read_json(directory / "completion.json")
    names = {"public_aggregate.json", "flagged_pairs_raw.jsonl", "flagged_pairs_unique.jsonl"}
    if completion.get("status") != "completed" or set(completion["output_sha256"]) != names or any(old.digest(directory / name) != h for name, h in completion["output_sha256"].items()):
        raise ValueError("completion output seals differ")
    expected, raw, unique = aggregate(directory, plan, data)
    if read_json(directory / "public_aggregate.json") != expected or [r for _, r in old.jsonl(directory / "flagged_pairs_raw.jsonl")] != raw or [r for _, r in old.jsonl(directory / "flagged_pairs_unique.jsonl")] != json.loads(json.dumps(unique)):
        raise ValueError("complete aggregate replay differs")
    return {"status": "verified_complete_document_screen", "remaining_documents": len(data["remaining"]),
            "public_aggregate_sha256": old.digest(directory / "public_aggregate.json"), **guard_flags()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for arg in ("pool-manifest", "prior-screen", "metadata-review", "output-dir"):
        p.add_argument("--" + arg, required=True)
    for name in ("run", "audit"):
        sub.add_parser(name).add_argument("--directory", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.pool_manifest, args.prior_screen, args.metadata_review, args.output_dir)
    elif args.command == "run":
        result = run(args.directory)
    else:
        result = audit(args.directory)
    print(json.dumps(result, ensure_ascii=False))
