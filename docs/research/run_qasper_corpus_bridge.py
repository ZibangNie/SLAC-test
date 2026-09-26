"""Offline corpus bridge on the already exposed 77-question development set.

Only the document filter changes. Cached CLS/FP32-normalized vectors are used
without loading a model, invoking a provider, or reading official test QA.
Global identities are (doc_id, unit_id); rendering retains the original unit ID
so that within-document packs remain byte-identical to the frozen experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import prepare_qasper_extended_development as preparation
import prepare_qasper_relation_pilot as pilot
from qasper_metrics import (
    evidence_metrics, paragraph_f1_score, paragraph_recall_diagnostic,
    references_from_annotations,
)
from run_qasper_dense_baseline import MODEL_REVISION, load_documents
from run_qasper_evidence_baselines import (
    PackCounter, check_time, digest, load_frozen_pool, pack_ranked, render_pack,
    score_selection,
)


SCHEMA = "slac-qasper-corpus-bridge-v1"
METHODS = ("given_document", "corpus_32")
CONFIG = {
    "query_count": 77, "query_families": 24, "corpus_documents": 32,
    "corpus_units": 1850, "cached_queries": 104, "dimension": 1024,
    "seed_units": 8, "candidate_cap": 16, "budget_bge_tokens": 1024,
    "max_selected_units": 3, "cpu_threads": 1,
    "ranking": "CPU FP32 exhaustive inner product; ties use frozen global source order",
    "global_order": "sorted document ID, then original native-unit order",
    "neighbors": "all seeds first; seed rank order; same-document immediate left then right",
    "deduplication": "exact native text across the entire pack, matching the old packer",
    "rendering": "original [unit_id] header and source text; global source order; unchanged in both arms",
    "source_doc_hit_at_k": "any of the first K dense-ranked units comes from the source document; not K distinct documents",
    "source_qualified_scoring": "exact JSON [doc_id,native_text] identity; all references belong to query source document",
    "empty_reference_recall": "existing diagnostic semantics: empty-empty=1; nonempty prediction with empty gold=0",
}
LIMITS = [
    "All 77 questions are previously exposed validation development data, not independent confirmation.",
    "The 32-paper corpus is a small fixed development corpus, not the full Qasper collection.",
    "Given-document retrieval uses source-document knowledge and is only a diagnostic reference.",
    "CPU exhaustive inner product is not a run of the production SLAC/FAISS pipeline.",
    "No JEV labels, refiner, new chunks, tree expansion, reranker or answer generation are evaluated.",
    "Official-string Evidence F1 is retained for compatibility; source-qualified metrics are diagnostic extensions.",
    "Evidence matching requires a full native string, not partial overlap or semantic sufficiency.",
    "BGE pack tokens exclude the query/instructions and are not generator context accounting.",
    "Original unit IDs may repeat across documents; packs are not ready for answer generation without a separately frozen unambiguous renderer.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def globalize_documents(documents):
    """Keep source strings/IDs unchanged; only order becomes corpus-wide."""
    units, keys, positions = [], [], {}
    for doc in sorted(documents):
        native = documents[doc]
        if ([unit.order for unit in native] != list(range(len(native)))
                or len({unit.unit_id for unit in native}) != len(native)):
            raise ValueError("native units need contiguous order and unique within-document IDs")
        positions[doc] = []
        for unit in native:
            index = len(units)
            units.append(replace(unit, order=index))
            keys.append((doc, unit.unit_id))
            positions[doc].append(index)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate corpus identity")
    return units, keys, positions


def validate_vectors(candidate_vectors, query_vectors, metadata, *, candidate_count,
                     query_count, dimension=1024):
    if metadata != {"pooling": "CLS_then_FP32_L2", "model_revision": MODEL_REVISION}:
        raise ValueError("cached representation metadata differs")
    for vectors, count in ((candidate_vectors, candidate_count), (query_vectors, query_count)):
        if vectors.device.type != "cpu" or vectors.dtype != torch.float32 or vectors.shape != (count, dimension):
            raise ValueError("cached representation shape, device or dtype differs")
        if not torch.isfinite(vectors).all() or not torch.allclose(
                vectors.norm(dim=1), torch.ones(count), atol=2e-6, rtol=0):
            raise ValueError("cached representations must be finite unit L2 vectors")


def validate_index(index, keys, qa_rows):
    expected = {"candidates": [{"doc_id": doc, "unit_id": uid} for doc, uid in keys],
                "queries": [{"doc_id": row["doc_id"], "question_id": row["question_id"]} for row in qa_rows]}
    if index != expected:
        raise ValueError("embedding index differs from reconstructed frozen input order")


def rank_scores(scores, allowed=None):
    if not isinstance(scores, list) or not all(math.isfinite(value) for value in scores):
        raise ValueError("finite dense scores required")
    indices = list(range(len(scores))) if allowed is None else list(allowed)
    if len(set(indices)) != len(indices) or any(not 0 <= index < len(scores) for index in indices):
        raise ValueError("invalid ranking scope")
    return sorted(indices, key=lambda index: (-scores[index], index))


def expand_corpus_candidates(ranking, keys, positions, *, seed_count=8, cap=16):
    """The old adjacency rule, with an explicit same-document boundary."""
    if not 1 <= seed_count <= cap or len(set(ranking)) != len(ranking):
        raise ValueError("invalid candidate expansion configuration")
    local = {index: (doc, offset) for doc, indices in positions.items()
             for offset, index in enumerate(indices)}
    if set(local) != set(range(len(keys))) or any(keys[index][0] != local[index][0] for index in local):
        raise ValueError("invalid corpus-to-native mapping")
    if any(index not in local for index in ranking):
        raise ValueError("ranking index outside corpus")
    seeds = list(ranking[:seed_count])
    selected = set(seeds)
    for seed in seeds:
        doc, offset = local[seed]
        for neighbor in (offset - 1, offset + 1):
            if len(selected) >= cap:
                break
            if 0 <= neighbor < len(positions[doc]):
                selected.add(positions[doc][neighbor])
    return seeds, sorted(selected)


def source_qualified_metrics(predicted, source_doc, annotations):
    """Do not grant a hit to another paper's identical heading/paragraph.

    Predicted pairs remain in the denominator even when their document is wrong.
    Reference duplicates, empty references and independent maxima are preserved.
    """
    encode = lambda pair: json.dumps(list(pair), ensure_ascii=False, separators=(",", ":"))
    predicted = [encode(pair) for pair in predicted]
    references = references_from_annotations(annotations)
    values = [[encode((source_doc, text)) for text in ref["evidence"]] for ref in references]
    return {
        "evidence_f1": max(paragraph_f1_score(predicted, ref) for ref in values),
        "evidence_recall": max(paragraph_recall_diagnostic(predicted, ref) for ref in values),
    }


def pack_source_qualified(units, keys, ranking, budget, count, max_units=3):
    """Optional later-stage packer: deduplicate (document, native string).

    Not used by this bridge's main arms, which retain the old text-only policy.
    Rendering/counting is supplied by the same PackCounter as the caller.
    """
    if len(units) != len(keys) or budget < 0 or max_units < 1:
        raise ValueError("invalid source-qualified packing contract")
    ranking = list(ranking)
    if (len(set(ranking)) != len(ranking)
            or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(units)
                   for index in ranking)):
        raise ValueError("source-qualified ranking requires unique in-range integer indices")
    chosen, seen = [], set()
    for index in ranking:
        identity = (keys[index][0], units[index].native_text)
        if identity in seen:
            continue
        if count([*chosen, index]) <= budget:
            chosen.append(index)
            seen.add(identity)
            if len(chosen) >= max_units:
                break
    return sorted(chosen)


def validate_within_document_replay(prepared, documents, units, keys, positions,
                                    scores_by_query, frozen_rankings, tokenizer, deadline=math.inf):
    """Validate rankings, candidates, rendered bytes, tokens and selected IDs."""
    replay = {}
    for q in prepared["queries"]:
        check_time(deadline)
        identity = (q["doc_id"], q["question_id"])
        doc = q["doc_id"]
        ranking = rank_scores(scores_by_query[identity], positions[doc])
        ids = [keys[index][1] for index in ranking]
        if ids != frozen_rankings[identity]:
            raise ValueError("within-document ranking replay differs from frozen dense ranking")
        seeds, candidates = expand_corpus_candidates(ranking, keys, positions)
        candidate_set = set(candidates)
        ranked_candidates = [index for index in ranking if index in candidate_set]
        if ([keys[index][1] for index in seeds] != q["seed_ids"]
                or [keys[index][1] for index in candidates] != q["candidate_ids"]
                or [keys[index][1] for index in ranked_candidates] != q["ranked_ids"]):
            raise ValueError("within-document candidate replay differs from prepared experiment")
        native = documents[doc]
        local_by_id = {unit.unit_id: index for index, unit in enumerate(native)}
        local_ranking = [local_by_id[uid] for uid in q["ranked_ids"]]
        local_count = PackCounter(tokenizer, native, deadline)
        global_count = PackCounter(tokenizer, units, deadline)
        local_selected = pack_ranked(native, local_ranking, 1024, local_count, max_units=3)
        selected = pack_ranked(units, ranked_candidates, 1024, global_count, max_units=3)
        if (render_pack(native, local_selected) != render_pack(units, selected)
                or local_count(local_selected) != global_count(selected)
                or [native[index].unit_id for index in local_selected] != [keys[index][1] for index in selected]):
            raise ValueError("within-document rendered pack replay differs")
        replay[identity] = {"ranking": ranking, "seeds": seeds, "candidates": candidates,
                            "selected": selected, "local_selected": local_selected}
    return replay


def load_verified_inputs(prepared_dir, dense_dir, *, deadline=math.inf):
    """Reusable read-only loader for later offline chunk/corpus diagnostics.

    Returns validated documents, QA only for scoring, globally mapped vectors,
    tokenizer, full rankings and a complete immutable-input hash inventory.
    It does not read support run/plan/registry files.
    """
    prepared_dir, dense_dir = Path(prepared_dir).resolve(), Path(dense_dir).resolve()
    prepared, manifest, selected_documents = preparation.load_prepared(prepared_dir)
    paths = {name: Path(path).resolve() for name, path in manifest["source_paths"].items()}
    if paths["dense_summary"] != dense_dir / "summary.json" or paths["rankings"] != dense_dir / "rankings.jsonl":
        raise ValueError("dense directory differs from frozen preparation")
    summary = read_json(paths["dense_summary"])
    required = {"status": "completed", "test_payload_read": False, "api_calls": 0,
                "question_count": 104, "document_count": 32, "unit_count": 1850,
                "truncated_inputs": 0, "training_performed": False,
                "model_revision": MODEL_REVISION, "backbone_precision": "float16",
                "pooling": "last_hidden_state[:,0] then FP32 L2 normalization"}
    if any(summary.get(key) != value for key, value in required.items()):
        raise ValueError("dense source does not satisfy frozen CLS development contract")
    hashes = dict(manifest["input_sha256"])
    for path, value in summary["input_sha256"].items():
        if hashes.get(str(Path(path).resolve())) != value:
            raise ValueError("dense inputs differ from prepared lineage")
    root = Path(__file__).resolve().parent
    for name, value in summary["source_sha256"].items():
        if digest(root / name) != value:
            raise ValueError("dense source implementation changed")
    frozen, candidates, qa_rows = load_frozen_pool(paths["pool_manifest"].parent, paths["sidecar"])
    documents, _, _ = load_documents(paths["pool_manifest"].parent, frozen, candidates)
    if any(documents[doc] != native for doc, native in selected_documents.items()):
        raise ValueError("prepared document content differs from corpus")
    units, keys, positions = globalize_documents(documents)
    if (len(documents) != 32 or len(units) != 1850 or len(qa_rows) != 104
            or len(prepared["queries"]) != 77
            or len({q["family_id"] for q in prepared["queries"]}) != 24):
        raise ValueError("frozen development denominator differs")
    index = read_json(dense_dir / "embedding_index.json")
    validate_index(index, keys, qa_rows)
    vector_path = dense_dir / "embeddings.safetensors"
    if digest(vector_path) != summary["embedding_sha256"]:
        raise ValueError("embedding cache hash mismatch")
    with safe_open(vector_path, framework="pt", device="cpu") as cache:
        if set(cache.keys()) != {"candidate_embeddings", "query_embeddings"}:
            raise ValueError("unexpected cached tensor inventory")
        metadata = cache.metadata()
        candidate_vectors, query_vectors = cache.get_tensor("candidate_embeddings"), cache.get_tensor("query_embeddings")
    validate_vectors(candidate_vectors, query_vectors, metadata, candidate_count=1850, query_count=104)
    tokenizer_path = Path(summary["model"]).resolve()
    if str(tokenizer_path / "tokenizer.json") not in hashes:
        raise ValueError("tokenizer not bound to dense source")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    rankings = pilot.validate_rankings(read_rows(paths["rankings"]), qa_rows, documents)
    qa_by_key = {(row["doc_id"], row["question_id"]): row for row in qa_rows}
    query_positions = {(row["doc_id"], row["question_id"]): index for index, row in enumerate(qa_rows)}
    if any(q["query"] != qa_by_key[(q["doc_id"], q["question_id"])]["question"] for q in prepared["queries"]):
        raise ValueError("prepared query text differs from cached query provenance")
    new_paths = [prepared_dir / "manifest.json", prepared_dir / "prepared.json",
                 dense_dir / "summary.json", dense_dir / "embedding_index.json", vector_path]
    new_paths += [root / name for name in (
        Path(__file__).name, "prepare_qasper_extended_development.py", "prepare_qasper_relation_pilot.py",
        "run_qasper_dense_baseline.py", "run_qasper_evidence_baselines.py", "qasper_metrics.py", "qasper_alignment_v2.py")]
    for path in new_paths:
        hashes[str(path.resolve())] = digest(path)
    pilot.verify_hashes(hashes)
    check_time(deadline)
    return {"prepared": prepared, "documents": documents, "units": units, "keys": keys,
            "positions": positions, "qa_by_key": qa_by_key, "query_positions": query_positions,
            "candidate_vectors": candidate_vectors, "query_vectors": query_vectors,
            "tokenizer": tokenizer, "rankings": rankings, "input_sha256": hashes}


def duplicate_diagnostics(units, keys, ranking, selected, source_doc):
    by_text = {}
    for index in ranking:
        by_text.setdefault(units[index].native_text, []).append(index)
    groups = [indices for indices in by_text.values() if len({keys[index][0] for index in indices}) > 1]
    wrong_selected = {units[index].native_text for index in selected if keys[index][0] != source_doc}
    return {"cross_document_duplicate_text_groups": len(groups),
            "correct_source_candidates_same_text_as_selected_wrong_source": sum(
                keys[index][0] == source_doc and units[index].native_text in wrong_selected for index in ranking)}


def evaluate_query(q, source, scores, replay, *, deadline=math.inf):
    units, keys = source["units"], source["keys"]
    gold = source["qa_by_key"][(q["doc_id"], q["question_id"])]["answer_annotations"]
    rows = []
    for method in METHODS:
        started = time.perf_counter()
        ranking = rank_scores(scores, source["positions"][q["doc_id"]]) if method == "given_document" else rank_scores(scores)
        seeds, candidates = expand_corpus_candidates(ranking, keys, source["positions"])
        candidate_set = set(candidates)
        ranked_candidates = [index for index in ranking if index in candidate_set]
        retrieval_seconds = time.perf_counter() - started
        packing_started = time.perf_counter()
        count = PackCounter(source["tokenizer"], units, deadline)
        selected = pack_ranked(units, ranked_candidates, 1024, count, max_units=3)
        packing_seconds = time.perf_counter() - packing_started
        if method == "given_document" and selected != replay["selected"]:
            raise ValueError("within-document selection changed after parity check")
        pairs = lambda indices: [(keys[index][0], units[index].native_text) for index in indices]
        qualified = source_qualified_metrics(pairs(selected), q["doc_id"], gold)
        candidate_metric = source_qualified_metrics(pairs(candidates), q["doc_id"], gold)
        score = score_selection(units, selected, gold, count, 1024)
        if method == "given_document":
            old = score_selection(source["documents"][q["doc_id"]], replay["local_selected"], gold,
                                  PackCounter(source["tokenizer"], source["documents"][q["doc_id"]], deadline), 1024)
            if old != score:
                raise ValueError("original within-document score replay differs")
        # These are pack-equivalence diagnostics, not additional tuned methods.
        full_selected = pack_ranked(units, ranking, 1024, count, max_units=3)
        seed_selected = pack_ranked(units, seeds, 1024, count, max_units=3)
        rows.append({**{name: q[name] for name in ("family_id", "doc_id", "question_id")},
            "method": method, "budget": 1024, **score,
            "source_qualified_evidence_f1": qualified["evidence_f1"],
            "source_qualified_evidence_recall": qualified["evidence_recall"],
            "candidate_source_qualified_evidence_recall": candidate_metric["evidence_recall"],
            "candidate_official_string_evidence_recall": evidence_metrics(
                [units[index].native_text for index in candidates], gold)["evidence_recall"],
            **{f"source_doc_hit_at_{k}_units": int(any(keys[index][0] == q["doc_id"] for index in ranking[:k]))
               for k in (1, 5, 8)},
            "candidate_count": len(candidates), "candidate_document_count": len({keys[index][0] for index in candidates}),
            "empty_pack": int(not selected),
            "duplicate_rendered_headers_in_pack": len(selected) - len({units[index].unit_id for index in selected}),
            "same_pack_as_dense_full_ranking": int(selected == full_selected),
            "same_pack_as_seed_only": int(selected == seed_selected),
            **duplicate_diagnostics(units, keys, ranked_candidates, selected, q["doc_id"]),
            "retrieval_wall_seconds": retrieval_seconds, "packing_wall_seconds": packing_seconds,
            "seed_global_indices": seeds, "candidate_global_indices": candidates,
            "selected_global_indices": selected})
    return rows


METRIC_FIELDS = (
    "official_evidence_f1", "reference_evidence_recall", "official_text_only_evidence_f1",
    "source_qualified_evidence_f1", "source_qualified_evidence_recall",
    "candidate_source_qualified_evidence_recall", "candidate_official_string_evidence_recall",
    "source_doc_hit_at_1_units", "source_doc_hit_at_5_units", "source_doc_hit_at_8_units",
    "actual_evidence_tokens", "selected_units", "candidate_count", "candidate_document_count",
    "empty_pack", "duplicate_rendered_headers_in_pack", "same_pack_as_dense_full_ranking",
    "same_pack_as_seed_only", "cross_document_duplicate_text_groups",
    "correct_source_candidates_same_text_as_selected_wrong_source",
)


def summarize(records, prepared):
    expected = {(q["doc_id"], q["question_id"]) for q in prepared["queries"]}
    tables, metrics = {}, []
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        indexed = {(row["doc_id"], row["question_id"]): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != expected:
            raise ValueError("main metric denominator must include every frozen query exactly once")
        tables[method] = indexed
        by_doc = {}
        for row in rows:
            by_doc.setdefault(row["doc_id"], []).append(row)
        metric = {"method": method, "questions": len(rows), "families": len(by_doc), "budget": 1024}
        for field in METRIC_FIELDS:
            metric[field + "_question_macro"] = statistics.mean(row[field] for row in rows)
            metric[field + "_document_macro"] = statistics.mean(
                statistics.mean(row[field] for row in group) for group in by_doc.values())
        for field in ("retrieval_wall_seconds", "packing_wall_seconds"):
            ordered = sorted(row[field] for row in rows)
            metric[field] = {"total": sum(ordered), "median": statistics.median(ordered),
                             "p95": ordered[math.ceil(.95 * len(ordered)) - 1]}
        metrics.append(metric)
    if len(records) != len(expected) * len(METHODS):
        raise ValueError("unexpected method records")
    paired = {}
    for field in METRIC_FIELDS:
        differences = [tables["corpus_32"][key][field] - tables["given_document"][key][field] for key in sorted(expected)]
        paired[field] = {"corpus_minus_given_question_macro": statistics.mean(differences),
                         "positive": sum(value > 0 for value in differences),
                         "negative": sum(value < 0 for value in differences),
                         "equal": sum(value == 0 for value in differences)}
    return {"metrics": metrics, "paired_comparisons": paired}


def run(args):
    started, cpu_started = time.monotonic(), time.process_time()
    if not 1 <= args.max_seconds <= 600:
        raise ValueError("max_seconds must be 1-600")
    deadline = started + args.max_seconds
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("output directory must not exist")
    torch.set_num_threads(1)
    source = load_verified_inputs(args.prepared, args.dense, deadline=deadline)
    scores_by_query, scoring_times = {}, []
    with torch.inference_mode():
        for q in source["prepared"]["queries"]:
            check_time(deadline)
            key = (q["doc_id"], q["question_id"])
            tick = time.perf_counter()
            vector = source["query_vectors"][source["query_positions"][key]]
            scores_by_query[key] = (source["candidate_vectors"] @ vector).tolist()
            scoring_times.append(time.perf_counter() - tick)
    replay = validate_within_document_replay(source["prepared"], source["documents"], source["units"],
        source["keys"], source["positions"], scores_by_query, source["rankings"], source["tokenizer"], deadline)
    print(json.dumps({"stage": "source_and_within_document_replay_verified", "questions": len(replay), "api_calls": 0}), flush=True)
    records = []
    for q in source["prepared"]["queries"]:
        key = (q["doc_id"], q["question_id"])
        records.extend(evaluate_query(q, source, scores_by_query[key], replay[key], deadline=deadline))
    tables = summarize(records, source["prepared"])
    pilot.verify_hashes(source["input_sha256"])
    check_time(deadline)
    empty_metrics = [evidence_metrics([], source["qa_by_key"][(q["doc_id"], q["question_id"])]["answer_annotations"])
                     for q in source["prepared"]["queries"]]
    header_counts = Counter(unit.unit_id for unit in source["units"])
    public = {"schema": SCHEMA, "status": "completed", "config": CONFIG,
        "question_count": 77, "family_count": 24, "corpus_document_count": 32, "corpus_unit_count": 1850,
        "api_calls": 0, "test_payload_read": False, "model_loaded": False, "training_performed": False,
        "answer_generation_performed": False, "independent_confirmation": False,
        "gold_used_for_ranking_or_selection": False, "jev_labels_reused": False,
        "within_document_replay": {"questions": len(replay), "ranking_candidate_render_token_selection_score_parity": True},
        "all_packs_within_budget": all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= 3 for row in records),
        "global_rendered_id_collisions": sum(value - 1 for value in header_counts.values()),
        "global_identity": "separate (doc_id,unit_id) mapping; original rendered unit IDs retained",
        "empty_pack_reference": {"questions": 77,
            "official_evidence_f1_question_macro": statistics.mean(row["evidence_f1"] for row in empty_metrics),
            "reference_evidence_recall_question_macro": statistics.mean(row["evidence_recall"] for row in empty_metrics),
            "questions_with_any_empty_reference": sum(row["evidence_f1"] == 1 for row in empty_metrics)},
        "input_binding_sha256": stable_hash(source["input_sha256"]), "script_sha256": digest(__file__),
        "execution": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float32", "threads": 1,
            "dense_scoring_wall_seconds": sum(scoring_times), "total_wall_seconds": time.monotonic() - started,
            "total_process_cpu_seconds": time.process_time() - cpu_started,
            "timing_scope": "Full-corpus vector scores computed once and shared; arm timings measure scope filtering/ranking/expansion and packing, not standalone deployment latency."},
        **tables, "limits": LIMITS}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "experiment_config.json", {"schema": SCHEMA, "config": CONFIG,
        "prepared_dir": str(Path(args.prepared).resolve()), "dense_dir": str(Path(args.dense).resolve()),
        "max_seconds": args.max_seconds, "input_sha256": source["input_sha256"]})
    pilot.write_json(output / "corpus_unit_index.json", [{"global_index": index, "doc_id": doc, "unit_id": uid}
        for index, (doc, uid) in enumerate(source["keys"])])
    with (output / "per_question.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pilot.write_json(output / "public_aggregate.json", public)
    pilot.write_json(output / "summary.json", {**public, "input_sha256": source["input_sha256"],
        "output_sha256": {path.name: digest(path) for path in sorted(output.iterdir())}})
    print(json.dumps({"status": public["status"], "output": str(output), "question_count": 77,
                      "all_packs_within_budget": public["all_packs_within_budget"], "execution": public["execution"]}), flush=True)
    return public


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    run(parser.parse_args())
