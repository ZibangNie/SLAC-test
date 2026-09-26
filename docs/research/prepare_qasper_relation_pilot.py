"""Freeze a small, gold-free Qasper relation pilot from existing dense rankings.

This module makes no API calls and loads no model. Full native text, queries and
prepared tasks are local artifacts, not material intended for publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_qasper_dense_baseline import load_documents
from run_qasper_evidence_baselines import digest, load_frozen_pool


SALT = "SLAC-JEV-DEVELOPMENT-v1"
CONFIG = {
    "selection_salt": SALT,
    "max_families": 8,
    "max_questions_per_family": 2,
    "dense_seeds": 8,
    "max_candidates_per_question": 16,
    "evidence_budget_bge_tokens": 1024,
    "neighbor_policy": "seed rank order; immediate left then right; retain all seeds first",
    "static_edge_policy": "original native-unit neighbors both present in at least one query pool; deduplicate by document",
    "candidate_storage_order": "original native-unit source order",
    "scope": "given-document selection-only development; no answer generation",
}


def stable_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selection_hash(identity):
    return hashlib.sha256(f"{SALT}|{identity}".encode("utf-8")).hexdigest()


def select_queries(candidates, qa_rows):
    """Selection uses identifiers only, never answers, evidence or model scores."""
    by_family = {}
    by_doc = {}
    for row in candidates:
        family, doc = row["family_id"], row["doc_id"]
        if family in by_family or doc in by_doc:
            raise ValueError("pilot requires unique family and document identifiers")
        by_family[family] = row
        by_doc[doc] = row
    selected = []
    for family in sorted(by_family, key=lambda item: (selection_hash(item), item))[:8]:
        doc = by_family[family]["doc_id"]
        rows = [row for row in qa_rows if row["doc_id"] == doc]
        if not rows:
            raise ValueError("selected family has no questions")
        for row in sorted(rows, key=lambda item: (selection_hash(item["question_id"]), item["question_id"]))[:2]:
            if row["family_id"] != family or not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError("invalid question identity or text")
            selected.append({"doc_id": doc, "family_id": family,
                             "question_id": row["question_id"], "query": row["question"]})
    return selected


def validate_rankings(rows, qa_rows, documents):
    wanted = {(row["doc_id"], row["question_id"]) for row in qa_rows}
    rankings = {}
    for row in rows:
        key = (row["doc_id"], row["question_id"])
        if key not in wanted or key in rankings:
            raise ValueError("dense ranking has unexpected or duplicate question identity")
        expected = [unit.unit_id for unit in documents[key[0]]]
        actual = row["ranked_ids"]
        if not isinstance(actual, list) or len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError("dense ranking must be a complete unique permutation of native units")
        rankings[key] = actual
    if set(rankings) != wanted:
        raise ValueError("dense ranking is missing frozen questions")
    return rankings


def expand_candidates(units, ranked_ids):
    by_id = {unit.unit_id: index for index, unit in enumerate(units)}
    if len(by_id) != len(units) or [unit.order for unit in units] != list(range(len(units))):
        raise ValueError("native units must have unique IDs and contiguous original order")
    if len(ranked_ids) != len(units) or set(ranked_ids) != set(by_id):
        raise ValueError("candidate expansion requires a full unique dense ranking")
    seeds = list(ranked_ids[:8])
    selected = set(seeds)
    for seed in seeds:
        index = by_id[seed]
        for neighbor in (index - 1, index + 1):
            if len(selected) >= 16:
                break
            if 0 <= neighbor < len(units):
                selected.add(units[neighbor].unit_id)
    return seeds, [unit.unit_id for unit in units if unit.unit_id in selected]


def build_prepared(candidates, qa_rows, documents, rankings):
    queries, static, support = [], {}, []
    chosen = select_queries(candidates, qa_rows)
    selected_documents = {}
    for row in chosen:
        doc, qid = row["doc_id"], row["question_id"]
        units = documents[doc]
        seeds, candidate_ids = expand_candidates(units, rankings[(doc, qid)])
        candidate_set = set(candidate_ids)
        selected_documents.setdefault(doc, [asdict(unit) for unit in units])
        queries.append({**row, "candidate_ids": candidate_ids, "seed_ids": seeds,
                        "ranked_ids": [uid for uid in rankings[(doc, qid)] if uid in candidate_set]})
        for unit in units:
            if unit.unit_id not in candidate_set:
                continue
            item = {"query": row["query"], "unit": {"id": unit.unit_id, "text": unit.text}}
            task_id = "support:" + stable_hash({"kind": "support", "doc_id": doc,
                                                  "question_id": qid, "item": item})
            support.append({"id": task_id, "doc_id": doc, "question_id": qid,
                            "unit_id": unit.unit_id, "item": item})
        for left, right in zip(units, units[1:]):
            if left.unit_id not in candidate_set or right.unit_id not in candidate_set:
                continue
            key = (doc, left.unit_id, right.unit_id)
            item = {"unit_a": {"id": left.unit_id, "text": left.text},
                    "unit_b": {"id": right.unit_id, "text": right.text}}
            static[key] = {"id": "static:" + stable_hash({"kind": "static", "doc_id": doc, "item": item}),
                           "doc_id": doc, "left_id": left.unit_id,
                           "right_id": right.unit_id, "item": item}
    payload = {"schema": "slac-qasper-relation-pilot-prepared-v1", "config": dict(CONFIG),
               "documents": selected_documents, "queries": queries,
               "static_tasks": list(static.values()), "support_tasks": support}
    ids = [task["id"] for kind in ("static_tasks", "support_tasks") for task in payload[kind]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prepared task identity")
    return payload


def verify_hashes(expected):
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError(f"source hash mismatch: {Path(path).name}")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(args):
    pool, sidecar, dense, output = (Path(getattr(args, name)).resolve()
                                   for name in ("pool", "sidecar", "dense", "output"))
    if output.exists():
        raise FileExistsError("output directory must not exist")
    summary_path, ranking_path = dense / "summary.json", dense / "rankings.jsonl"
    local_inputs = [pool / "pool_manifest.json", pool / "candidates.jsonl",
                    pool / "native_qa_alignment.json", sidecar,
                    sidecar.parent / "alignment_audit_v2.json", summary_path, ranking_path]
    script_names = ("prepare_qasper_relation_pilot.py", "run_qasper_dense_baseline.py",
                    "run_qasper_evidence_baselines.py", "qasper_alignment_v2.py")
    local_inputs.extend(Path(__file__).resolve().parent / name for name in script_names)
    initial_hashes = {str(path): digest(path) for path in local_inputs}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("test_payload_read") is not False:
        raise ValueError("dense source must be a completed non-test run")
    dense_inputs = summary["input_sha256"]
    if not isinstance(dense_inputs, dict) or not dense_inputs:
        raise ValueError("dense source has no frozen input hashes")
    normalized_dense_inputs = {str(Path(path).resolve()): value for path, value in dense_inputs.items()}
    for path in local_inputs[:5]:
        if normalized_dense_inputs.get(str(path)) != initial_hashes[str(path)]:
            raise ValueError("dense source is not bound to the current frozen pool and sidecar")
    verify_hashes(dense_inputs)
    frozen, candidates, qa_rows = load_frozen_pool(pool, sidecar)
    documents, shard, archive = load_documents(pool, frozen, candidates)
    for path in (shard, archive):
        if str(path.resolve()) not in normalized_dense_inputs:
            raise ValueError("dense source lacks document/archive lineage")
    rankings_rows = [json.loads(line) for line in ranking_path.read_text(encoding="utf-8").splitlines()]
    rankings = validate_rankings(rankings_rows, qa_rows, documents)
    if summary.get("question_count") != len(qa_rows) or summary.get("document_count") != len(candidates):
        raise ValueError("dense source counts disagree with frozen pool")
    payload = build_prepared(candidates, qa_rows, documents, rankings)
    verify_hashes(initial_hashes)
    verify_hashes(dense_inputs)
    counts = {"documents": len(payload["documents"]), "families": len({q["family_id"] for q in payload["queries"]}),
              "queries": len(payload["queries"]), "static_tasks": len(payload["static_tasks"]),
              "support_tasks": len(payload["support_tasks"]),
              "candidates_per_query": dict(Counter(len(q["candidate_ids"]) for q in payload["queries"]))}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prepared.json", payload)
    manifest = {"schema": "slac-qasper-relation-pilot-manifest-v1", "status": "prepared",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": dict(CONFIG),
                "counts": counts, "input_sha256": {**normalized_dense_inputs, **initial_hashes},
                "prepared_sha256": digest(output / "prepared.json"),
                "dense_rankings_provenance": "Existing development ranking artifact; no prior ranking digest in dense summary; current bytes bound at pilot preparation. Not independently recomputed from embeddings.",
                "source_hashes_unchanged": True, "api_calls": 0, "answer_generation_performed": False,
                "model_loaded": False, "test_payload_read": False, "independent_evaluation": False,
                "exposure_status": "post-baseline exposed development pool; family/question hash selection is not independent confirmation",
                "gold_in_model_payload": False, "payloads_local_only": True}
    write_json(output / "manifest.json", manifest)
    if json.loads((output / "prepared.json").read_text(encoding="utf-8")) != payload:
        raise ValueError("prepared output round-trip mismatch")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pool", "sidecar", "dense", "output"):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "prepared_sha256": result["prepared_sha256"]}, indent=2))
