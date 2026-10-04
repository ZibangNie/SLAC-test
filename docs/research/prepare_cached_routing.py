"""Freeze four pre-JEV features and family folds without reading quality targets."""
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
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts/research-foundation"
CONTRACT = "offline-20261004/routing-cache-contract.json"
CONTRACT_SHA = "1c286a62dcfde05b7f1d4d38e884a929da5145e64bb9c1c01751c9ff05310f37"
SOURCES = {
    "prepared": "qasper-extended-development-prepared-01/prepared.json",
    "mapping": "qasper-local-answer-plan-02/mapping.jsonl",
    "scores": "qasper-reranker-run-02/pair_scores.jsonl",
    "rankings": "qasper-reranker-run-02/rankings.jsonl",
}
IDENTITY = ("family_id", "doc_id", "question_id")
FEATURE_NAMES = ("bge_gap", "dense_bge_disagreement", "bge_source_span", "bge_heading_share")
SALT = "slac-budget-selection-20261004-sample-v1"
PACK_FIELDS = ("selected_ids", "pack_sha256", "actual_evidence_tokens", "cache_key")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def parse_rows(raw):
    return [parse(line) for line in raw.splitlines() if line.strip()]


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def artifact_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ARTIFACTS.resolve()) and path != ARTIFACTS.resolve(),
            "path must stay within research artifacts")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def render(units, selected_ids):
    wanted = set(selected_ids)
    return "\n\n".join(f"[{unit['unit_id']}]\n{unit['text']}" for unit in units if unit["unit_id"] in wanted)


def sample_keys(keys):
    families = sorted({key[0] for key in keys}, key=lambda family: (sha(f"{SALT}|{family}".encode()), family))[:8]
    result = []
    for family in families:
        members = [key for key in keys if key[0] == family]
        members.sort(key=lambda key: (sha("|".join((SALT, *key)).encode()), key))
        result.extend(members[:2])
    return result


def build_features(prepared, mapping, pair_scores, rankings):
    """Use only source structure, local BGE logits, and original Dense/BGE packs."""
    documents, queries, support_tasks = prepared["documents"], prepared["queries"], prepared["support_tasks"]
    query_map = {identity(q): q for q in queries}
    query_pairs = {(q["doc_id"], q["question_id"]): identity(q) for q in queries}
    require(bool(queries) and len(query_map) == len(query_pairs) == len(queries), "duplicate or empty query scope")
    unit_maps = {}
    for doc, units in documents.items():
        require(bool(units) and [u["order"] for u in units] == list(range(len(units))), "invalid source unit order")
        unit_maps[doc] = {u["unit_id"]: u for u in units}
        require(len(unit_maps[doc]) == len(units), "duplicate native unit ID")
    expected_pairs = set()
    for q in queries:
        ids, ranking = q["candidate_ids"], q["ranked_ids"]
        require(q["doc_id"] in documents and 1 <= len(ids) <= 16 and len(ids) == len(set(ids))
                and len(ranking) == len(ids) and set(ranking) == set(ids)
                and set(ids) <= set(unit_maps[q["doc_id"]]), "frozen candidate identity mismatch")
        expected_pairs.update((q["doc_id"], q["question_id"], uid) for uid in ids)
    tasks = {(t["doc_id"], t["question_id"], t["unit_id"]): t["id"] for t in support_tasks}
    require(len(tasks) == len(support_tasks) == len(set(tasks.values())) and set(tasks) == expected_pairs,
            "complete pre-judgment task map required")
    logits = {}
    for row in pair_scores:
        key = (row["doc_id"], row["question_id"], row["unit_id"])
        require(key in tasks and key not in logits and row["task_id"] == tasks[key], "scored candidate identity mismatch")
        value = row["raw_logit"]
        require(type(value) in (int, float) and math.isfinite(value), "nonfinite or invalid BGE logit")
        logits[key] = value
    require(set(logits) == expected_pairs, "complete original candidate logits required")
    saved_rankings = {}
    for row in rankings:
        key = (row["doc_id"], row["question_id"])
        require(key in query_pairs and key not in saved_rankings, "ranking identity mismatch")
        saved_rankings[key] = row["ranked_ids"]
    require(set(saved_rankings) == set(query_pairs), "complete BGE ranking coverage required")
    packs = {}
    for row in mapping:
        if row["method"] not in {"dense_k3", "reranker_k3"}:
            continue
        key = (row["method"], identity(row))
        require(key[1] in query_map and key not in packs, "pack mapping identity mismatch")
        q, doc = query_map[key[1]], row["doc_id"]
        ids = row["selected_ids"]
        require(isinstance(ids, list) and len(ids) == len(set(ids)) and len(ids) <= 3
                and set(ids) <= set(q["candidate_ids"]), "invalid original pack selection")
        require(ids == sorted(ids, key=lambda uid: unit_maps[doc][uid]["order"]), "original pack source order differs")
        require(len({unit_maps[doc][uid]["native_text"] for uid in ids}) == len(ids), "duplicate original native evidence")
        require(type(row["actual_evidence_tokens"]) is int and 0 <= row["actual_evidence_tokens"] <= 1024
                and bool(ids) == (row["actual_evidence_tokens"] > 0), "invalid original token count")
        require(isinstance(row["cache_key"], str) and bool(row["cache_key"]), "missing original payload cache key")
        require(row["pack_sha256"] == sha(render(documents[doc], ids).encode()), "original pack rendering mismatch")
        packs[key] = {field: row[field] for field in PACK_FIELDS}
    require(set(packs) == {(method, key) for method in ("dense_k3", "reranker_k3") for key in query_map},
            "complete Dense and BGE pack coverage required")
    rows = []
    for key, q in sorted(query_map.items()):
        doc, qid = q["doc_id"], q["question_id"]
        positions = {uid: i for i, uid in enumerate(q["ranked_ids"])}
        ordered = sorted(q["candidate_ids"], key=lambda uid: (-logits[doc, qid, uid], positions[uid], unit_maps[doc][uid]["order"]))
        require(saved_rankings[doc, qid] == ordered, "saved BGE ranking does not match logits")
        values = [logits[doc, qid, uid] for uid in ordered]
        difference = values[0] - values[-1]
        gap = (values[2] - values[3]) / difference if len(values) >= 4 and difference else 0.0
        bge, dense = packs["reranker_k3", key], packs["dense_k3", key]
        left, right = set(bge["selected_ids"]), set(dense["selected_ids"])
        union = left | right
        disagreement = 1 - len(left & right) / len(union) if union else 0.0
        selected = [unit_maps[doc][uid] for uid in bge["selected_ids"]]
        span = ((max(u["order"] for u in selected) - min(u["order"] for u in selected)) / (len(documents[doc]) - 1)
                if len(selected) >= 2 and len(documents[doc]) > 1 else 0.0)
        heading = sum(u["kind"] in {"title", "heading"} for u in selected) / len(selected) if selected else 0.0
        features = [gap, disagreement, span, heading]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in features), "invalid pre-JEV feature")
        rows.append(dict(zip(IDENTITY, key)) | {"features": features, "support_task_count": len(q["candidate_ids"]),
                    "bge_pack": bge, "dense_pack": dense})
    return rows


def prepare(output):
    output = artifact_path(output)
    output.mkdir(parents=True, exist_ok=False)
    consumed = {}

    def read_bound(path, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        value = sha(raw)
        require(expected is None or value == expected, "frozen source binding mismatch")
        consumed[str(path)] = value
        return raw

    contract_path = artifact_path(ARTIFACTS / CONTRACT)
    contract = parse(read_bound(contract_path, CONTRACT_SHA))
    feature_sources = contract["feature_sources"]
    specifications = {"prepared": feature_sources["prepared"], "mapping": feature_sources["dense_and_bge_packs"],
                      "scores": feature_sources["pair_scores.jsonl"], "rankings": feature_sources["rankings.jsonl"]}
    source_data = {}
    for role, rel in SOURCES.items():
        specification = specifications[role]
        require(specification["path"] == rel, "feature source is outside the fixed whitelist")
        path = artifact_path(ARTIFACTS / rel)
        raw = read_bound(path, specification["sha256"])
        source_data[role] = parse(raw) if role == "prepared" else parse_rows(raw)
    source_files = [Path(__file__).resolve(), ROOT / "SLAC/retrieval/routing/offline_router.py",
                    ROOT / "docs/research/CACHED_ROUTING_PROTOCOL_20261004.md",
                    ROOT / "tests/research/test_prepare_cached_routing.py",
                    ROOT / "tests/research/test_offline_router.py"]
    code_hashes = {str(path): sha(path.read_bytes()) for path in source_files}
    features = build_features(source_data["prepared"], source_data["mapping"], source_data["scores"], source_data["rankings"])
    keys = tuple(identity(row) for row in features)
    require(len(keys) == 77 and len({key[0] for key in keys}) == 24
            and sum(row["support_task_count"] for row in features) == 1214, "full development scope required")
    from SLAC.retrieval.routing.offline_router import assign_family_folds
    folds = assign_family_folds(keys)
    serialized_folds = [asdict(fold) for fold in folds]
    write_json(output / "features.json", features)
    write_json(output / "folds.json", serialized_folds)
    artifacts = {name: sha((output / name).read_bytes()) for name in ("features.json", "folds.json")}
    plan = {"schema": "slac-cached-routing-prepared-v1", "status": "prepared_without_quality_targets",
            "contract_path": str(contract_path), "contract_sha256": CONTRACT_SHA,
            "input_sha256": consumed, "code_sha256": code_hashes, "artifact_sha256": artifacts,
            "quality_targets_read": False, "question_count": 77, "family_count": 24,
            "feature_names": FEATURE_NAMES, "sample_keys": sample_keys(keys),
            "deferred_quality_sources": contract["arms"], "api_calls": 0, "key_read": False}
    for path, expected in (consumed | code_hashes).items():
        require(sha(Path(path).read_bytes()) == expected, "source changed during feature preparation")
    write_json(output / "plan.json", plan)
    columns = list(zip(*(row["features"] for row in features)))
    summary = {"schema": "slac-cached-routing-prepared-public-v1", "status": "prepared_without_quality_targets",
               "question_count": 77, "family_count": 24, "support_task_count": 1214, "fold_count": len(folds),
               "feature_statistics": {name: {"min": min(column), "max": max(column), "mean": math.fsum(column) / len(column),
                                             "constant": min(column) == max(column)} for name, column in zip(FEATURE_NAMES, columns)},
               "constant_columns": [name for name, column in zip(FEATURE_NAMES, columns) if min(column) == max(column)],
               "sample_question_count": len(sample_keys(keys)), "quality_targets_read": False,
               "jev_judgments_or_packs_read": False, "api_calls": 0, "key_read": False,
               "plan_sha256": sha((output / "plan.json").read_bytes()), "artifact_sha256": artifacts,
               "source_content_sha256": sorted(consumed.values()), "code_content_sha256": sorted(code_hashes.values())}
    # The core's dataclass exposes fixed training/test keys; no identities are public.
    summary["fold_sizes"] = [len(fold.test_keys) for fold in folds]
    write_json(output / "summary.json", summary)
    return summary


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled for feature preparation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ARTIFACTS / "offline-20261004/routing-prepared-01"))
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
