"""Prepare exactly the previously frozen 24-family / 77-question development set.

No model, API, credentials or official test QA. All selected families' questions
are retained, including empty-reference and figure/table questions. Original
pilot candidate expansion and visible support/static task contracts are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import prepare_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import Unit, digest, load_frozen_pool
from run_qasper_dense_baseline import load_documents


SCHEMA = "slac-qasper-extended-development-prepared-v1"
FAMILIES, QUESTIONS = 24, 77
IDENTITY = ("family_id", "doc_id", "question_id")
CONFIG = {**pilot.CONFIG, "selection_salt": "SLAC-EXTENDED-DEVELOPMENT-v1",
          "max_families": FAMILIES, "max_questions_per_family": None,
          "selection": "all frozen inventory questions; all nonpilot families; no outcome-based exclusions"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_source(hashes, name):
    matches = [Path(path).resolve() for path in hashes if Path(path).name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen source named {name}")
    return matches[0]


def validate_inventory(inventory, candidates, qa_rows, original_prepared):
    if (inventory.get("schema") != "slac-extended-development-inventory-v1"
            or inventory.get("status") != "denominator_frozen_not_an_executable_api_plan"
            or inventory.get("families") != FAMILIES or inventory.get("questions") != QUESTIONS
            or inventory.get("independent_confirmation") is not False
            or inventory.get("gold_or_model_error_used_for_selection") is not False):
        raise ValueError("unsupported extended development inventory")
    by_doc = {row["doc_id"]: row for row in candidates}
    by_family = {row["family_id"]: row for row in candidates}
    if len(by_doc) != len(candidates) or len(by_family) != len(candidates):
        raise ValueError("development candidates must have unique documents and families")
    excluded = {row["family_id"] for row in original_prepared["queries"]}
    if (len(excluded) != 8 or len(inventory["excluded_pilot_families"]) != 8
            or set(inventory["excluded_pilot_families"]) != excluded):
        raise ValueError("inventory must exclude all original pilot families")
    wanted = {doc for doc, row in by_doc.items() if row["family_id"] not in excluded}
    selected_docs = inventory["documents"]
    if (len(selected_docs) != FAMILIES or len({row["doc_id"] for row in selected_docs}) != FAMILIES
            or {row["doc_id"] for row in selected_docs} != wanted):
        raise ValueError("inventory document coverage differs from all nonpilot families")
    for row in selected_docs:
        if (row.get("official_split") != "validation" or row["family_id"] != by_doc[row["doc_id"]]["family_id"]
                or by_doc[row["doc_id"]].get("official_split") != "validation"):
            raise ValueError("inventory family identity or validation split differs")
    query_lookup = {}
    for row in qa_rows:
        key = tuple(row[name] for name in IDENTITY)
        if key in query_lookup or row.get("official_split") != "validation":
            raise ValueError("duplicate or non-validation source question")
        if row["doc_id"] not in by_doc or row["family_id"] != by_doc[row["doc_id"]]["family_id"]:
            raise ValueError("question source lineage differs")
        query_lookup[key] = row
    expected = {key for key in query_lookup if key[1] in wanted}
    actual = [tuple(row[name] for name in IDENTITY) for row in inventory["query_inventory"]]
    if len(actual) != QUESTIONS or len(set(actual)) != QUESTIONS or set(actual) != expected:
        raise ValueError("inventory must retain every question of all selected families")
    chosen = []
    for key in actual:
        row = query_lookup[key]
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("selected question has no usable query; refusing rather than dropping it")
        chosen.append({**{name: row[name] for name in IDENTITY}, "query": row["question"]})
    return chosen


def build_prepared(chosen, documents, rankings, inventory_sha256):
    queries, static, support, selected_documents = [], {}, [], {}
    for row in chosen:
        doc, qid = row["doc_id"], row["question_id"]
        units = documents[doc]
        seeds, candidate_ids = pilot.expand_candidates(units, rankings[(doc, qid)])
        candidate_set = set(candidate_ids)
        selected_documents.setdefault(doc, [asdict(unit) for unit in units])
        queries.append({**row, "candidate_ids": candidate_ids, "seed_ids": seeds,
                        "ranked_ids": [uid for uid in rankings[(doc, qid)] if uid in candidate_set]})
        for unit in units:
            if unit.unit_id not in candidate_set:
                continue
            item = {"query": row["query"], "unit": {"id": unit.unit_id, "text": unit.text}}
            task_id = "support:" + pilot.stable_hash({"kind": "support", "doc_id": doc,
                                                       "question_id": qid, "item": item})
            support.append({"id": task_id, "doc_id": doc, "question_id": qid,
                            "unit_id": unit.unit_id, "item": item})
        for left, right in zip(units, units[1:]):
            if left.unit_id not in candidate_set or right.unit_id not in candidate_set:
                continue
            item = {"unit_a": {"id": left.unit_id, "text": left.text},
                    "unit_b": {"id": right.unit_id, "text": right.text}}
            static[(doc, left.unit_id, right.unit_id)] = {
                "id": "static:" + pilot.stable_hash({"kind": "static", "doc_id": doc, "item": item}),
                "doc_id": doc, "left_id": left.unit_id, "right_id": right.unit_id, "item": item}
    tasks = list(static.values()) + support
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate prepared task identity")
    return {"schema": SCHEMA, "config": dict(CONFIG), "inventory_sha256": inventory_sha256,
            "documents": selected_documents, "queries": queries,
            "static_tasks": list(static.values()), "support_tasks": support}


def load_sources(inventory_path, inventory_sha256, pilot_prepared):
    inventory_path, pilot_prepared = Path(inventory_path).resolve(), Path(pilot_prepared).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", inventory_sha256) or digest(inventory_path) != inventory_sha256:
        raise ValueError("frozen extended inventory digest mismatch")
    inventory = read_json(inventory_path)
    original_manifest_path = pilot_prepared / "manifest.json"
    original_path = pilot_prepared / "prepared.json"
    original_manifest = read_json(original_manifest_path)
    if (original_manifest.get("status") != "prepared" or original_manifest.get("test_payload_read") is not False
            or digest(original_path) != original_manifest["prepared_sha256"]
            or digest(original_path) != inventory["source_sha256"]["pilot_prepared"]):
        raise ValueError("original pilot preparation differs from frozen inventory")
    hashes = {str(Path(path).resolve()): value for path, value in original_manifest["input_sha256"].items()}
    pilot.verify_hashes(hashes)
    paths = {name: unique_source(hashes, filename) for name, filename in {
        "candidates": "candidates.jsonl", "pool_manifest": "pool_manifest.json",
        "sidecar": "native_qa_sidecar_v2.jsonl", "dense_summary": "summary.json",
        "rankings": "rankings.jsonl"}.items()}
    for name, field in (("candidates", "development_candidates"), ("sidecar", "qa_sidecar")):
        if digest(paths[name]) != inventory["source_sha256"][field]:
            raise ValueError("development source differs from inventory hash")
    dense = read_json(paths["dense_summary"])
    if dense.get("status") != "completed" or dense.get("test_payload_read") is not False:
        raise ValueError("requires completed validation dense source")
    if any(hashes.get(str(Path(path).resolve())) != value for path, value in dense["input_sha256"].items()):
        raise ValueError("dense source lineage differs from original preparation")
    frozen, candidates, qa_rows = load_frozen_pool(paths["pool_manifest"].parent, paths["sidecar"])
    original = read_json(original_path)
    chosen = validate_inventory(inventory, candidates, qa_rows, original)
    documents, shard, archive = load_documents(paths["pool_manifest"].parent, frozen, candidates)
    for path in (shard, archive):
        if str(path.resolve()) not in hashes:
            raise ValueError("native document source missing from frozen lineage")
    rankings = pilot.validate_rankings(read_rows(paths["rankings"]), qa_rows, documents)
    if dense["question_count"] != len(qa_rows) or dense["document_count"] != len(candidates):
        raise ValueError("dense source counts differ from original development pool")
    for path in (inventory_path, original_manifest_path, original_path, Path(__file__).resolve()):
        hashes[str(path)] = digest(path)
    return chosen, documents, rankings, hashes, paths


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("output directory must not exist")
    chosen, documents, rankings, hashes, sources = load_sources(args.inventory, args.inventory_sha256, args.pilot_prepared)
    prepared = build_prepared(chosen, documents, rankings, args.inventory_sha256)
    pilot.verify_hashes(hashes)
    counts = {"documents": len(prepared["documents"]), "families": len({q["family_id"] for q in prepared["queries"]}),
              "queries": len(prepared["queries"]), "static_tasks": len(prepared["static_tasks"]),
              "support_tasks": len(prepared["support_tasks"]),
              "candidates_per_query": dict(Counter(len(q["candidate_ids"]) for q in prepared["queries"]))}
    if counts["documents"] != FAMILIES or counts["families"] != FAMILIES or counts["queries"] != QUESTIONS:
        raise ValueError("extended prepared counts differ from the frozen denominator")
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "prepared.json", prepared)
    manifest = {"schema": "slac-qasper-extended-development-manifest-v1", "status": "prepared",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": dict(CONFIG), "counts": counts,
        "inventory_path": str(Path(args.inventory).resolve()), "inventory_sha256": args.inventory_sha256,
        "pilot_prepared_dir": str(Path(args.pilot_prepared).resolve()),
        "source_paths": {name: str(path) for name, path in sources.items()}, "input_sha256": hashes,
        "prepared_sha256": digest(output / "prepared.json"), "prepared_bytes": (output / "prepared.json").stat().st_size,
        "dense_rankings_provenance": "Original development rankings, frozen by the pilot preparation; no embedding recomputation.",
        "source_hashes_unchanged": True, "api_calls": 0, "model_loaded": False, "test_payload_read": False,
        "answer_generation_performed": False, "independent_evaluation": False, "gold_in_model_payload": False,
        "gold_or_model_error_used_for_selection": False, "all_inventory_queries_retained": True,
        "exposure_status": "Extended development: prior baseline exposure; excludes all original pilot families.",
        "payloads_local_only": True}
    pilot.write_json(output / "manifest.json", manifest)
    if read_json(output / "prepared.json") != prepared:
        raise ValueError("prepared output round-trip mismatch")
    return manifest


def load_prepared(directory):
    """Extended-loader interface matching the old runner, with independently locked scope."""
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("schema") != "slac-qasper-extended-development-manifest-v1"
            or manifest.get("status") != "prepared" or manifest.get("test_payload_read") is not False
            or digest(directory / "prepared.json") != manifest["prepared_sha256"]):
        raise ValueError("invalid extended prepared manifest or digest")
    pilot.verify_hashes(manifest["input_sha256"])
    chosen, source_documents, rankings, hashes, _ = load_sources(
        manifest["inventory_path"], manifest["inventory_sha256"], manifest["pilot_prepared_dir"])
    if hashes != manifest["input_sha256"]:
        raise ValueError("extended prepared source bindings changed")
    prepared = read_json(directory / "prepared.json")
    if prepared != build_prepared(chosen, source_documents, rankings, manifest["inventory_sha256"]):
        raise ValueError("extended prepared payload differs from frozen query and native sources")
    documents = {doc: [Unit(**unit) for unit in values] for doc, values in prepared["documents"].items()}
    return prepared, manifest, documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "inventory-sha256", "pilot-prepared", "output"):
        parser.add_argument("--" + name, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "counts", "prepared_sha256", "prepared_bytes")}, indent=2))
