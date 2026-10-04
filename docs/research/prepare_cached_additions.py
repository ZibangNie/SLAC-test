"""Freeze every cached single-unit addition using metadata only, with zero API."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/research-foundation"
PHASE = "offline-20261004"
INVENTORY = f"{PHASE}/cached-interaction-opportunity.json"
INVENTORY_SHA = "37ca0b2147c53be7a8f1680b9ec1fb912137193078dced558941debe88c59fb2"
CONTRACT = f"{PHASE}/cache_contract.json"
CONTRACT_SHA = "dd1fe7cab7c176be82a27965e4bffc75bcc1f79d384df72f52efb81990d3e20c"
PREPARED = "qasper-extended-development-prepared-01/prepared.json"
SAMPLE_PLAN = f"{PHASE}/routing-prepared-01/plan.json"
SAMPLE_PLAN_SHA = "862f79b8987c88f4cafbd176d3829e33a9bec26f6b2bc181cf35990b9bf2264a"
IDENTITY = ("family_id", "doc_id", "question_id")
PACK_FIELDS = ("selected_ids", "pack_sha256", "cache_key", "actual_evidence_tokens")
SAMPLE_SALT = "slac-budget-selection-20261004-sample-v1"
SOURCE_MAPPINGS = {
    "local": ("qasper-local-answer-plan-02/mapping.jsonl",),
    "owner": ("qasper-owner-order-answer-plan-01/mapping.jsonl",),
    "primary": ("qasper-recovered-primary-answer-plan-01/mapping.jsonl",),
    "relation": ("qasper-relation-pack-answer-plan-01/packs.jsonl", "qasper-relation-pack-answer-plan-01/baselines.jsonl"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def artifact_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ARTIFACTS.resolve()) and path != ARTIFACTS.resolve(), "path outside research artifacts")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def fixed_sample(keys):
    families = sorted({k[0] for k in keys}, key=lambda f: (sha(f"{SAMPLE_SALT}|{f}".encode()), f))[:8]
    selected = []
    for family in families:
        members = [k for k in keys if k[0] == family]
        members.sort(key=lambda k: (sha("|".join((SAMPLE_SALT, *k)).encode()), k))
        selected.extend(members[:2])
    return selected


def build_manifest(prepared, sources):
    """Join only identities and cached pack metadata; never labels or outcomes.

    Exact rendering/token provenance is inherited from the pinned inventory over
    identical source bytes. This function does not repeat its text/token audit.
    """
    queries = {identity(q): q for q in prepared["queries"]}
    require(bool(queries) and len(queries) == len(prepared["queries"]), "duplicate or empty question scope")
    documents = prepared["documents"]
    unit_maps = {doc: {u["unit_id"]: u for u in units} for doc, units in documents.items()}
    for doc, units in documents.items():
        require(len(unit_maps[doc]) == len(units) and [u["order"] for u in units] == list(range(len(units))), "source unit identity differs")
    expected_tasks = set()
    for key, query in queries.items():
        ids = query["candidate_ids"]
        require(query["doc_id"] in documents and len(ids) == len(set(ids)) and 1 <= len(ids) <= 16
                and set(ids) <= set(unit_maps[key[1]]), "candidate scope differs")
        expected_tasks.update((key[1], key[2], uid) for uid in ids)
    tasks = {(t["doc_id"], t["question_id"], t["unit_id"]): t["id"] for t in prepared["support_tasks"]}
    require(len(tasks) == len(prepared["support_tasks"]) == len(set(tasks.values())) and set(tasks) == expected_tasks,
            "complete original support-task identity map required")
    packs = {key: {} for key in queries}
    cache_keys = {}
    excluded = 0
    for source in sources:
        origin, path = source["origin"], source["path"]
        for ordinal, row in enumerate(source["rows"]):
            key = identity(row)
            require(key in queries, "pack question outside original scope")
            ids = row["selected_ids"]
            require(isinstance(ids, list) and len(ids) == len(set(ids)) and len(ids) <= 3, "invalid pack cardinality")
            if not set(ids) <= set(queries[key]["candidate_ids"]):
                excluded += 1
                continue
            require(ids == sorted(ids, key=lambda uid: unit_maps[key[1]][uid]["order"]), "pack source order differs")
            require(type(row["actual_evidence_tokens"]) is int and 0 <= row["actual_evidence_tokens"] <= 1024
                    and bool(ids) == bool(row["actual_evidence_tokens"]), "invalid cached token count")
            for field in ("pack_sha256", "cache_key"):
                require(isinstance(row[field], str) and len(row[field]) == 64
                        and all(c in "0123456789abcdef" for c in row[field]), "invalid cached content identity")
            frozen = {field: row[field] for field in PACK_FIELDS}
            selected = frozenset(ids)
            old = packs[key].get(selected)
            require(old is None or {field: old[field] for field in PACK_FIELDS} == frozen,
                    "same question pack has conflicting cache identity")
            cache_identity = (key, tuple(ids), row["pack_sha256"], row["actual_evidence_tokens"])
            require(row["cache_key"] not in cache_keys or cache_keys[row["cache_key"]] == cache_identity,
                    "cached payload identity crosses questions or packs")
            cache_keys[row["cache_key"]] = cache_identity
            provenance = {"origin": origin, "mapping_path": path, "row_index": ordinal}
            if "method" in row:
                provenance["method"] = row["method"]
            if old is None:
                packs[key][selected] = frozen | {"sources": [provenance]}
            else:
                old["sources"].append(provenance)
    edges = []
    squares = 0
    for key, nodes in sorted(packs.items()):
        for base in sorted(nodes, key=lambda ids: tuple(nodes[ids]["selected_ids"])):
            additions = {next(iter(sup - base)): sup for sup in nodes if len(sup) == len(base) + 1 and base < sup}
            for added, sup in sorted(additions.items()):
                edge = dict(zip(IDENTITY, key)) | {"base": nodes[base], "larger": nodes[sup],
                        "added_unit_id": added, "added_support_task_id": tasks[key[1], key[2], added],
                        "base_is_empty": not base}
                key_fields = {k: edge[k] for k in IDENTITY} | {"base_cache_key": edge["base"]["cache_key"],
                             "larger_cache_key": edge["larger"]["cache_key"], "added_unit_id": added}
                edge["edge_id"] = sha(canonical(key_fields))
                edges.append(edge)
            squares += sum(base | {a, b} in nodes for a, b in itertools.combinations(additions, 2))
    require(len({e["edge_id"] for e in edges}) == len(edges), "duplicate addition edge")

    def coverage(group):
        covered = {identity(e) for e in group}
        endpoints = {(identity(e), endpoint["cache_key"]) for e in group for endpoint in (e["base"], e["larger"])}
        return {"paired_edges": len(group), "questions": len(covered), "families": len({k[0] for k in covered}),
                "unique_query_packs_in_edges": len(endpoints), "question_denominator": len(queries),
                "family_denominator": len({key[0] for key in queries})}

    controls = {"question_count": len(queries), "family_count": len({key[0] for key in queries}),
                "support_task_count": len(tasks), "unique_query_packs": sum(len(nodes) for nodes in packs.values()),
                "unique_payload_cache_keys": len(cache_keys), "excluded_outside_common_candidate_pool_rows": excluded,
                "query_pack_size_histogram": {str(n): count for n, count in sorted(Counter(len(p) for ns in packs.values() for p in ns).items())},
                "all": coverage(edges), "empty_base": coverage([e for e in edges if e["base_is_empty"]]),
                "nonempty_base": coverage([e for e in edges if not e["base_is_empty"]]), "complete_inclusion_squares": squares}
    sample = fixed_sample(list(queries))
    return edges, controls, sample


def verify_controls(controls, inventory):
    require(controls["question_count"] == inventory["question_denominator"] == 77
            and controls["family_count"] == inventory["family_denominator"] == 24
            and controls["support_task_count"] == 1214
            and controls["unique_query_packs"] == inventory["pack_scope"]["unique_query_packs"] == 454
            and controls["unique_payload_cache_keys"] == inventory["pack_scope"]["unique_payload_cache_keys"]
            and controls["excluded_outside_common_candidate_pool_rows"] == inventory["excluded_mapping_row_counts"]["outside_common_candidate_pool_rows"] == 90
            and controls["query_pack_size_histogram"] == inventory["pack_scope"]["query_pack_size_histogram"],
            "complete cached pack inventory differs")
    for stratum in ("all", "empty_base", "nonempty_base"):
        require(controls[stratum] == inventory["single_unit_addition"][stratum], "complete addition coverage differs")
    require(controls["all"]["paired_edges"] == 32 and controls["empty_base"]["paired_edges"] == 9
            and controls["nonempty_base"]["paired_edges"] == 23
            and controls["complete_inclusion_squares"] == inventory["two_by_two_inclusion"]["all"]["squares"] == 0,
            "frozen edge or square count differs")


def prepare(output):
    output = artifact_path(output)
    output.mkdir(parents=True, exist_ok=False)
    consumed = {}

    def read_bound(path, expected):
        path = Path(path).resolve()
        raw = path.read_bytes()
        require(sha(raw) == expected, "frozen metadata source mismatch")
        consumed[str(path)] = expected
        return raw

    inventory = parse(read_bound(ARTIFACTS / INVENTORY, INVENTORY_SHA))
    contract = parse(read_bound(ARTIFACTS / CONTRACT, CONTRACT_SHA))
    require(inventory["status"] == "metadata_inventory_complete"
            and set(inventory["sources_included"]) == set(SOURCE_MAPPINGS)
            and not inventory["sources_not_included"], "complete four-source inventory required")
    prepared = parse(read_bound(ARTIFACTS / PREPARED, inventory["source_sha256"][PREPARED]))
    require(contract["verified_consumed_sha256"][PREPARED] == inventory["source_sha256"][PREPARED], "prepared lineage differs")
    sources = []
    for origin, paths in SOURCE_MAPPINGS.items():
        for path in paths:
            rows = [parse(line) for line in read_bound(ARTIFACTS / path, inventory["source_sha256"][path]).splitlines() if line.strip()]
            sources.append({"origin": origin, "path": path, "rows": rows})
    code_paths = (Path(__file__).resolve(), ROOT / "tests/research/test_prepare_cached_additions.py",
                  ROOT / "docs/research/NEXT_CACHED_ADDITION_DIAGNOSIS_20261004.md",
                  ROOT / "docs/research/CACHED_ADDITION_PROTOCOL_20261004.md")
    code_hashes = {str(path): sha(path.read_bytes()) for path in code_paths}
    edges, controls, sample = build_manifest(prepared, sources)
    verify_controls(controls, inventory)
    sample_plan = parse(read_bound(ARTIFACTS / SAMPLE_PLAN, SAMPLE_PLAN_SHA))
    require([list(key) for key in sample] == sample_plan["sample_keys"], "previous fixed QA sample differs")
    quality_sources = []
    for source in contract["answer_cache_sources"]:
        quality_sources.append({"origin": source["origin"], "summary_path": source["summary"],
                                "summary_sha256": source["summary_sha256"],
                                "record_names": ["per_question.jsonl", "pack_records.jsonl"] if source["origin"] == "relation" else ["per_question.jsonl"],
                                "instruction": "After manifest freeze, bind each record file through this pinned summary output_sha256 before reading scores."})
    write_json(output / "edges.json", edges)
    write_json(output / "controls.json", controls)
    outputs = {name: sha((output / name).read_bytes()) for name in ("edges.json", "controls.json")}
    plan = {"schema": "slac-cached-addition-prepared-v1", "status": "prepared_without_quality_or_support_labels",
            "input_sha256": consumed, "code_sha256": code_hashes, "artifact_sha256": outputs,
            "inventory_sha256": INVENTORY_SHA, "contract_sha256": CONTRACT_SHA,
            "parent_audit_and_generator_commitments_inherited": inventory["source_sha256"],
            "generator_contract": inventory["generator_contract"], "controls": controls,
            "pool_keys": sorted(identity(q) for q in prepared["queries"]),
            "sample_keys": sample, "sample_edge_ids": [e["edge_id"] for e in edges if identity(e) in sample],
            "deferred_quality_sources": quality_sources,
            "deferred_support_sources": {name: {"path": f"qasper-primary-support-recovery-run-01/{name}",
                "sha256": contract["verified_consumed_sha256"][f"qasper-primary-support-recovery-run-01/{name}"]} for name in ("labels.json", "raw_scores.json")},
            "api_calls": 0, "quality_targets_read": False, "support_labels_or_scores_read": False,
            "answer_text_read": False, "references_read": False, "raw_provider_responses_read": False, "key_read": False,
            "rendering_and_native_dedup_validation": "inherited exact inventory verification over unchanged bound mapping and prepared bytes"}
    for path, expected in (consumed | code_hashes).items():
        require(sha(Path(path).read_bytes()) == expected, "source changed during manifest preparation")
    write_json(output / "plan.json", plan)
    summary = {"schema": "slac-cached-addition-prepared-public-v1", "status": plan["status"], "controls": controls,
               "api_calls": 0, "quality_targets_read": False, "support_labels_or_scores_read": False, "key_read": False,
               "question_sample_count": len(sample), "sample_edge_count": len(plan["sample_edge_ids"]),
               "sample_covered_question_count": len({identity(e) for e in edges if identity(e) in sample}),
               "plan_sha256": sha((output / "plan.json").read_bytes()), "artifact_sha256": outputs,
               "source_content_sha256": sorted(consumed.values()), "code_content_sha256": sorted(code_hashes.values()),
               "interpretation": "metadata feasibility only; no second-order interaction identification or quality effect measured"}
    write_json(output / "summary.json", summary)
    return summary


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled for metadata preparation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ARTIFACTS / PHASE / "addition-prepared-01"))
    args = parser.parse_args()
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    try:
        print(json.dumps(prepare(args.output), ensure_ascii=True, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "api_calls": 0}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
