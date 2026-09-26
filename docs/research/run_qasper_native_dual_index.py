"""Frozen native-rule chunk bridge: prepare on CPU, explicitly run when GPU idle.

No provider/API, new labels, training, or official test QA. The same partition,
native leaf representation and whole-pack budget are used by both owner arms.
All six configurations and all within-scope paired comparisons are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file, load as load_tensors
from transformers import AutoModel

import build_qasper_native_chunks as native
import prepare_qasper_relation_pilot as pilot
import run_qasper_corpus_bridge as bridge
from run_qasper_dense_baseline import MODEL_REVISION, audit_token_lengths, dense_cls
from run_qasper_evidence_baselines import PackCounter, check_time, digest, score_selection
from qasper_metrics import evidence_metrics
from SLAC.retrieval.retrieve.chunk_aggregator import aggregate_hits_to_chunk_candidates
from SLAC.retrieval.retrieve.chunk_dense_retriever import ChunkDenseHit
from SLAC.retrieval.retrieve.leaf_dense_retriever import DenseHit
from SLAC.retrieval.retrieve.score_fusion import fuse_candidate_scores_rrf


SCHEMA = "slac-qasper-native-dual-index-v1"
SCOPES = ("given_document", "corpus_32")
METHODS = ("leaf_direct", "leaf_owner", "dual_owner")
METRICS = ("official_evidence_f1", "reference_evidence_recall", "official_text_only_evidence_f1",
    "source_qualified_evidence_f1", "source_qualified_evidence_recall",
    "candidate_source_qualified_evidence_recall", "candidate_official_string_evidence_recall",
    "full_native_source_qualified_evidence_recall", "actual_evidence_tokens", "selected_units",
    "candidate_count", "candidate_document_count", "empty_pack", "duplicate_rendered_headers_in_pack",
    "cross_document_duplicate_text_groups", "correct_source_candidates_same_text_as_selected_wrong_source")
CONFIG = {
    "queries": 77, "families": 24, "documents": 32, "leaves": 1850,
    "scopes": list(SCOPES), "methods": list(METHODS), "leaf_topk": 8, "chunk_topk": 8,
    "owner_topk": 8, "candidate_cap": 16, "rrf_k": 60, "query_intent": "generic",
    "evidence_budget_bge_tokens": 1024, "max_selected_units": 3,
    "ranking": "CPU FP32 exhaustive inner product; ties use frozen index order",
    "owner_fusion": "production aggregate_hits_to_chunk_candidates and fuse_candidate_scores_rrf",
    "owner_ties": "production RRF tie policy, ending with stable chunk_id",
    "projection": "fused owner order, then descending leaf score and frozen global row; unique (doc_id,unit_id)",
    "leaf_direct": "dense top8 plus immediate same-document neighbors capped16; dense order",
    "packing_deduplication": "(doc_id,native_text); repeated same-source strings selected once",
    "rendering": "unchanged [unit_id] and native text; final global source order; no source prefix in either scope",
    "chunk_model_input": "exact adapter ChunkRecord.text only, no path/anchor/instruction",
    "model_revision": MODEL_REVISION, "pooling": "CLS_then_FP32_L2", "backbone_precision": "float16",
    "batch_size": 4, "max_length": 8192, "encoding_max_seconds": 600,
    "evaluation_max_seconds": 600, "gpu_allocated_limit_bytes": 6 * 1024**3,
    "gpu_reserved_limit_bytes": 7 * 1024**3, "cpu_threads": 1, "seed": 13,
    "active_user_gpu_workload_requires_idle": True, "idle_samples": 3,
    "idle_sample_interval_seconds": 5, "idle_max_utilization_percent": 10,
    "idle_min_memory_free_mib": 4096,
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "bootstrap": "family-cluster multinomial resampling, shared PCG64 draws; linear percentile 95%",
    "paired_orientation": "later minus earlier in fixed method list; all three pairs within each scope",
    "multiple_comparison_adjustment": "none", "gold_used_for_partition_ranking_selection": False,
}
LIMITS = [
    "Previously exposed validation development data; exploratory cluster intervals are not independent confirmation or multiplicity-controlled inference.",
    "Rule chunks are not JEV relations or Boundary Refiner output; no complete online SLAC or answer-quality claim.",
    "Dual retrieval adds a top8 chunk channel and its compute; this is part of the comparison.",
    "Production RRF gives each leaf hit a vote, so multiple leaves in a large owner accumulate votes.",
    "Leaf-direct is a background baseline; source-qualified deduplication differs from the earlier text-only corpus bridge.",
    "The exact CPU ranking reference is used; no FAISS latency or production ANN claim.",
    "Source-qualified exact-string evidence metrics require the correct paper; official-string scores are also retained.",
    "Original rendered unit IDs may repeat across documents; these packs are not an answer-generation format.",
    "BGE evidence tokens include headers/separators/specials but exclude generator query/instructions.",
    "Cached query/leaf reuse and one shared chunk encoding are separately timed, not cold-start end-to-end latency.",
    "No support labels are reused for newly retrieved candidates and no API calls are made.",
]


def read_json(path):
    return json.loads(Path(path).read_bytes())


def environment():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "tokenizers", "safetensors", "numpy", "faiss-cpu")}


def source_hashes():
    """Explicit dependency closure; unrelated caller imports cannot alter it."""
    research = (Path(__file__).name, "NATIVE_CHUNK_BRIDGE_PROTOCOL_20260927.md", "build_qasper_native_chunks.py",
        "prepare_qasper_relation_pilot.py", "prepare_qasper_extended_development.py", "run_qasper_corpus_bridge.py",
        "run_qasper_dense_baseline.py", "run_qasper_evidence_baselines.py", "qasper_metrics.py", "qasper_alignment_v2.py")
    production = ("schemas/records.py", "schemas/validation.py", "retrieve/chunk_aggregator.py",
        "retrieve/score_fusion.py", "retrieve/leaf_dense_retriever.py", "retrieve/chunk_dense_retriever.py",
        "retrieve/anchor_retriever.py", "index/embedder.py", "index/faiss_utils.py", "utils/text_utils.py")
    paths = {Path(__file__).with_name(name).resolve() for name in research}
    paths.update((native.REPO_ROOT / "SLAC/retrieval" / name).resolve() for name in production)
    for package in ("SLAC", "SLAC/retrieval", "SLAC/retrieval/schemas", "SLAC/retrieval/retrieve",
                    "SLAC/retrieval/index", "SLAC/retrieval/utils"):
        path = native.REPO_ROOT / package / "__init__.py"
        if path.exists():
            paths.add(path.resolve())
    return {str(path): digest(path) for path in sorted(paths)}


def merge_hashes(*inventories):
    result = {}
    for inventory in inventories:
        for name, value in inventory.items():
            path = str(Path(name).resolve())
            if path in result and result[path] != value:
                raise ValueError("conflicting input/source hash")
            result[path] = value
    return result


def load_inputs(prepared, dense, chunks):
    source = bridge.load_verified_inputs(prepared, dense)
    manifest, documents, leaves, owners = native.load_artifacts(chunks)
    if len(documents) != 32 or len(leaves) != 1850:
        raise ValueError("native adapter denominator differs")
    if [(leaf.doc_id, leaf.meta["native_unit_id"], leaf.text, leaf.meta["cache_embedding_row"])
            for leaf in leaves] != [(doc, uid, unit.text, i)
            for i, ((doc, uid), unit) in enumerate(zip(source["keys"], source["units"], strict=True))]:
        raise ValueError("adapter text/identity/order differs from cached vectors")
    if manifest["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("adapter tokenizer revision differs")
    paths = {str((Path(chunks) / name).resolve()): digest(Path(chunks) / name)
             for name in ("manifest.json", *manifest["output_files_sha256"])}
    source.update(leaves=leaves, chunks=owners, native_manifest=manifest)
    source["input_sha256"] = merge_hashes(source["input_sha256"], manifest["input_sha256"], paths, source_hashes())
    return source


def input_identity(source):
    return {"queries": [{name: q[name] for name in ("family_id", "doc_id", "question_id", "query")}
                        for q in source["prepared"]["queries"]],
        "leaves": [{"doc_id": l.doc_id, "unit_id": l.meta["native_unit_id"],
                    "text_sha256": native.text_hash(l.text), "cache_row": l.meta["cache_embedding_row"]}
                   for l in source["leaves"]],
        "chunks": [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text_sha256": native.text_hash(c.text),
                    "native_unit_ids": c.meta["native_unit_ids"]} for c in source["chunks"]]}


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("prepared output already exists")
    torch.set_num_threads(1)
    source = load_inputs(args.prepared, args.dense, args.chunks)
    token_ids, audit = audit_token_lengths(source["tokenizer"], [c.text for c in source["chunks"]], 8192)
    if audit["above_max_length"]:
        raise ValueError("complete chunk exceeds8192; preparation stops without truncation")
    if [len(ids) for ids in token_ids] != [c.token_est for c in source["chunks"]]:
        raise ValueError("actual tokenizer lengths differ from native adapter")
    model_path = Path(source["native_manifest"]["tokenizer"]).resolve()
    if str(model_path / "model.safetensors") not in source["input_sha256"]:
        raise ValueError("local model weights must be bound before any GPU execution")
    pilot.verify_hashes(source["input_sha256"])
    plan = {"schema": SCHEMA, "status": "prepared_not_executed", "config": CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared": str(Path(args.prepared).resolve()), "dense": str(Path(args.dense).resolve()),
        "chunks": str(Path(args.chunks).resolve()), "model": str(model_path),
        "input_sha256": source["input_sha256"], "input_binding_sha256": pilot.stable_hash(source["input_sha256"]),
        "identity": input_identity(source), "token_ids_sha256": pilot.stable_hash(token_ids),
        "chunk_length_audit": audit, "environment": environment(),
        "api_calls": 0, "model_loaded": False, "gpu_execution_started": False, "limits": LIMITS}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "chunk_token_ids.json", token_ids)
    pilot.write_json(output / "plan.json", plan)
    pilot.write_json(output / "plan_seal.json", {name: digest(output / name) for name in ("plan.json", "chunk_token_ids.json")})
    return {"status": plan["status"], "chunks": len(token_ids), "length_audit": audit,
            "plan_sha256": digest(output / "plan.json"), "gpu_execution_started": False}


def load_plan(directory):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != {"plan.json", "chunk_token_ids.json", "plan_seal.json"}:
        raise ValueError("prepared file inventory differs")
    seal_bytes = (directory / "plan_seal.json").read_bytes()
    seal = json.loads(seal_bytes)
    if set(seal) != {"plan.json", "chunk_token_ids.json"}:
        raise ValueError("invalid plan seal inventory")
    buffers = {name: (directory / name).read_bytes() for name in seal}
    if any(hashlib.sha256(value).hexdigest() != seal[name] for name, value in buffers.items()):
        raise ValueError("prepared bytes differ from seal")
    plan, ids = json.loads(buffers["plan.json"]), json.loads(buffers["chunk_token_ids.json"])
    fixed = {"schema": SCHEMA, "config": CONFIG, "status": "prepared_not_executed", "api_calls": 0,
             "model_loaded": False, "gpu_execution_started": False, "limits": LIMITS}
    expected_fields = set(fixed) | {"created_at_utc", "prepared", "dense", "chunks", "model", "input_sha256",
        "input_binding_sha256", "identity", "token_ids_sha256", "chunk_length_audit", "environment"}
    if (plan.get("schema") != SCHEMA or plan.get("config") != CONFIG
            or plan.get("status") != "prepared_not_executed" or plan["environment"] != environment()
            or plan["input_binding_sha256"] != pilot.stable_hash(plan["input_sha256"])
            or set(plan) != expected_fields or any(plan.get(key) != value for key, value in fixed.items())):
        raise ValueError("prepared specification/environment/binding differs")
    if (pilot.stable_hash(ids) != plan["token_ids_sha256"] or len(ids) != len(plan["identity"]["chunks"])
            or any(not row or len(row) > 8192 or any(type(v) is not int or v < 0 for v in row) for row in ids)):
        raise ValueError("invalid untruncated token IDs")
    pilot.verify_hashes(plan["input_sha256"])
    return plan, ids, {**{str(directory / name): hashlib.sha256(value).hexdigest() for name, value in buffers.items()},
                       str(directory / "plan_seal.json"): hashlib.sha256(seal_bytes).hexdigest()}


def validate_reloaded(plan, ids, source):
    if input_identity(source) != plan["identity"] or source["input_sha256"] != plan["input_sha256"]:
        raise ValueError("reloaded input identity or complete source inventory differs")
    model_path = Path(source["native_manifest"]["tokenizer"]).resolve()
    weights = str(model_path / "model.safetensors")
    if (Path(plan["model"]).resolve() != model_path or weights not in plan["input_sha256"]
            or plan["input_sha256"][weights] != source["input_sha256"].get(weights)
            or digest(weights) != plan["input_sha256"][weights]):
        raise ValueError("model path/weights are not the frozen native BGE representation")
    actual, audit = audit_token_lengths(source["tokenizer"], [c.text for c in source["chunks"]], 8192)
    if actual != ids or audit != plan["chunk_length_audit"]:
        raise ValueError("frozen token IDs differ from exact chunk text")


def gpu_sample():
    """Read-only process/device checks; never kill or alter another workload."""
    def command(args):
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=3).stdout.strip()
    names = command(["powershell", "-NoProfile", "-Command",
        "Get-Process | Where-Object { $_.ProcessName -match 'fifa' } | Select-Object -ExpandProperty ProcessName"])
    values = command(["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader,nounits"])
    lines = values.splitlines()
    if len(lines) != 1:
        raise RuntimeError("one explicit GPU0 status row is required")
    utilization, used, total = [int(value.strip()) for value in lines[0].split(",")]
    compute = command(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    return {"fifa_process_present": bool(names), "compute_process_present": bool(compute),
            "gpu_utilization_percent": utilization, "memory_used_mib": used, "memory_total_mib": total,
            "memory_free_mib": total - used}


def assert_idle(sample):
    if (sample["fifa_process_present"] or sample["compute_process_present"]
            or sample["gpu_utilization_percent"] > CONFIG["idle_max_utilization_percent"]
            or sample["memory_free_mib"] < CONFIG["idle_min_memory_free_mib"]):
        raise RuntimeError("user GPU workload is active; retain prepared state and do not execute")
    if sample["memory_total_mib"] * 1024**2 < CONFIG["gpu_reserved_limit_bytes"]:
        raise RuntimeError("device capacity below frozen memory limit")
    if sample["memory_used_mib"] + sample["memory_free_mib"] != sample["memory_total_mib"]:
        raise ValueError("inconsistent device memory report")


def confirm_idle(sample=gpu_sample, sleep=time.sleep):
    observed = []
    for index in range(CONFIG["idle_samples"]):
        row = sample()
        assert_idle(row)
        observed.append(row)
        if index + 1 < CONFIG["idle_samples"]:
            sleep(CONFIG["idle_sample_interval_seconds"])
    return observed


def encode_chunks(model_path, tokenizer, token_ids):
    """The only model-loading path; called only after explicit run and idle gate."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.set_device(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., CONFIG["gpu_reserved_limit_bytes"] / total_memory), 0)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    deadline = started + CONFIG["encoding_max_seconds"]
    model = None
    try:
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
        model.requires_grad_(False).eval()
        if 8192 > model.config.max_position_embeddings - model.config.pad_token_id - 1:
            raise ValueError("frozen max length exceeds model capacity")
        result = torch.empty((len(token_ids), 1024), dtype=torch.float32)
        order = sorted(range(len(token_ids)), key=lambda i: (len(token_ids[i]), i))
        with torch.inference_mode():
            for offset in range(0, len(order), CONFIG["batch_size"]):
                check_time(deadline)
                positions = order[offset:offset + CONFIG["batch_size"]]
                batch = tokenizer.pad({"input_ids": [token_ids[i] for i in positions]}, padding=True, return_tensors="pt")
                if [int(v) for v in batch["attention_mask"].sum(-1)] != [len(token_ids[i]) for i in positions]:
                    raise ValueError("padding changed frozen token lengths")
                batch = {key: value.to("cuda:0") for key, value in batch.items()}
                vectors = dense_cls(model(**batch).last_hidden_state).cpu()
                if vectors.shape != (len(positions), 1024):
                    raise ValueError("encoder dimension differs")
                result[positions] = vectors
                if (torch.cuda.max_memory_allocated(0) > CONFIG["gpu_allocated_limit_bytes"]
                        or torch.cuda.max_memory_reserved(0) > CONFIG["gpu_reserved_limit_bytes"]):
                    raise RuntimeError("frozen GPU memory limit exceeded")
                check_time(deadline)
        execution = {"encoding_wall_seconds": time.monotonic() - started,
            "device": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        return result, execution
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def retrieve_candidates(q, source, leaf_scores, chunk_scores, scope, method):
    if scope not in SCOPES or method not in METHODS:
        raise ValueError("unsupported fixed retrieval arm")
    leaves, chunks = source["leaves"], source["chunks"]
    if len(leaf_scores) != len(leaves) or len(chunk_scores) != len(chunks):
        raise ValueError("dense score dimensions differ")
    allowed = source["positions"][q["doc_id"]] if scope == "given_document" else None
    leaf_rank = bridge.rank_scores(leaf_scores, allowed)
    chunk_allowed = [i for i, c in enumerate(chunks) if c.doc_id == q["doc_id"]] if scope == "given_document" else None
    chunk_rank = bridge.rank_scores(chunk_scores, chunk_allowed)
    if method == "leaf_direct":
        seeds, candidates = bridge.expand_corpus_candidates(leaf_rank, source["keys"], source["positions"])
        return [i for i in leaf_rank if i in set(candidates)], {"leaf_hits": seeds, "chunk_hits": [], "owners": []}
    leaf_hits = [DenseHit(leaves[i].leaf_id, leaf_scores[i], rank, "leaf_dense", q["query"])
                 for rank, i in enumerate(leaf_rank[:8], 1)]
    chunk_indices = chunk_rank[:8] if method == "dual_owner" else []
    chunk_hits = [ChunkDenseHit(chunks[i].chunk_id, chunk_scores[i], rank, "chunk_dense", q["query"])
                  for rank, i in enumerate(chunk_indices, 1)]
    leaf_lookup, chunk_lookup = {l.leaf_id: l for l in leaves}, {c.chunk_id: c for c in chunks}
    owners = aggregate_hits_to_chunk_candidates(leaf_hits, chunk_hits, [], chunk_lookup, leaf_lookup)
    fused = fuse_candidate_scores_rrf(owners, leaf_hits, chunk_hits, [],
        {l.leaf_id: l.owner_chunk_id for l in leaves}, "generic", fused_topn=8)
    by_owner = {}
    for i, leaf in enumerate(leaves):
        by_owner.setdefault(leaf.owner_chunk_id, []).append(i)
    candidates, seen, trace = [], set(), []
    for owner in fused:
        members = sorted(by_owner[owner.chunk_id], key=lambda i: (-leaf_scores[i], i))
        if scope == "given_document" and owner.doc_id != q["doc_id"]:
            raise ValueError("given-document owner escaped scope")
        trace.append({"chunk_id": owner.chunk_id, "rrf_score": owner.meta["rrf_score"],
            "hit_count": owner.hit_count, "owner_native_units": len(members), "owner_bge_tokens": owner.token_est,
            "source_views": owner.source_views, "projected_global_indices": members})
        for index in members:
            key = source["keys"][index]
            if key not in seen and len(candidates) < 16:
                candidates.append(index)
                seen.add(key)
    return candidates, {"leaf_hits": leaf_rank[:8], "chunk_hits": chunk_indices, "owners": trace}


def evaluate_query(q, source, leaf_scores, chunk_scores, *, deadline=math.inf):
    units, keys = source["units"], source["keys"]
    # Retrieval and selection are complete before reading this query's annotations.
    selections = []
    counter = PackCounter(source["tokenizer"], units, deadline)
    for scope in SCOPES:
        for method in METHODS:
            check_time(deadline)
            started = time.perf_counter()
            candidates, trace = retrieve_candidates(q, source, leaf_scores, chunk_scores, scope, method)
            retrieval_seconds = time.perf_counter() - started
            started = time.perf_counter()
            selected = bridge.pack_source_qualified(units, keys, candidates, 1024, counter, max_units=3)
            selections.append((scope, method, candidates, selected, trace, retrieval_seconds, time.perf_counter() - started))
    gold = source["qa_by_key"][(q["doc_id"], q["question_id"])]["answer_annotations"]
    pairs = lambda indices: [(keys[i][0], units[i].native_text) for i in indices]
    full = bridge.source_qualified_metrics(pairs(source["positions"][q["doc_id"]]), q["doc_id"], gold)
    records = []
    for scope, method, candidates, selected, trace, retrieval_seconds, packing_seconds in selections:
        qualified = bridge.source_qualified_metrics(pairs(selected), q["doc_id"], gold)
        candidate_metrics = bridge.source_qualified_metrics(pairs(candidates), q["doc_id"], gold)
        records.append({**{name: q[name] for name in ("family_id", "doc_id", "question_id")},
            "scope": scope, "method": method, **score_selection(units, selected, gold, counter, 1024),
            "source_qualified_evidence_f1": qualified["evidence_f1"],
            "source_qualified_evidence_recall": qualified["evidence_recall"],
            "candidate_source_qualified_evidence_recall": candidate_metrics["evidence_recall"],
            "candidate_official_string_evidence_recall": evidence_metrics([units[i].native_text for i in candidates], gold)["evidence_recall"],
            "full_native_source_qualified_evidence_recall": full["evidence_recall"],
            "candidate_count": len(candidates), "candidate_document_count": len({keys[i][0] for i in candidates}),
            "empty_pack": int(not selected), "duplicate_rendered_headers_in_pack": len(selected) - len({units[i].unit_id for i in selected}),
            **bridge.duplicate_diagnostics(units, keys, candidates, selected, q["doc_id"]),
            "retrieval_wall_seconds": retrieval_seconds, "packing_wall_seconds": packing_seconds,
            "candidate_global_indices": candidates, "selected_global_indices": selected, "trace": trace})
    return records


def summarize(records, queries):
    identities = sorted((q["family_id"], q["doc_id"], q["question_id"]) for q in queries)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("unique nonempty question denominator required")
    expected = {(scope, method, *key) for scope in SCOPES for method in METHODS for key in identities}
    indexed = {(r["scope"], r["method"], r["family_id"], r["doc_id"], r["question_id"]): r for r in records}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("every method/scope must cover the entire fixed denominator exactly once")
    families = sorted({key[0] for key in identities})
    groups = [np.array([i for i, key in enumerate(identities) if key[0] == family]) for family in families]
    draws = np.random.Generator(np.random.PCG64(CONFIG["bootstrap_seed"])).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=CONFIG["bootstrap_replicates"])
    sizes = np.array([len(g) for g in groups])
    def estimates(values):
        means = np.array([values[g].mean() for g in groups])
        return {"question_weighted": float(values.mean()), "family_balanced": float(means.mean())}
    def delta(values):
        sums = np.array([values[g].sum() for g in groups])
        weighted = (draws @ sums) / (draws @ sizes)
        balanced = (draws @ (sums / sizes)) / len(groups)
        return {**estimates(values), "question_weighted_percentile95": np.quantile(weighted, [.025, .975]).tolist(),
            "family_balanced_percentile95": np.quantile(balanced, [.025, .975]).tolist(),
            "question_positive": int((values > 1e-12).sum()), "question_negative": int((values < -1e-12).sum()),
            "question_ties": int((np.abs(values) <= 1e-12).sum())}
    result = []
    for scope in SCOPES:
        values, metrics = {}, []
        for method in METHODS:
            rows = [indexed[(scope, method, *key)] for key in identities]
            table = {metric: np.array([row[metric] for row in rows], dtype=float) for metric in METRICS}
            if any(not np.isfinite(value).all() for value in table.values()):
                raise ValueError("nonfinite report metric")
            if any(not 0 <= r["actual_evidence_tokens"] <= 1024 or not 0 <= r["selected_units"] <= 3 for r in rows):
                raise ValueError("pack limit violated")
            values[method] = table
            metrics.append({"method": method, "questions": len(rows), "families": len(groups),
                "metrics": {name: estimates(value) for name, value in table.items()},
                "timing": {name: {"total_seconds": sum(r[name] for r in rows),
                    "median_seconds": statistics.median(r[name] for r in rows)}
                    for name in ("retrieval_wall_seconds", "packing_wall_seconds")}})
        comparisons = []
        for minus, plus in combinations(METHODS, 2):
            changed = sum(set(indexed[(scope, plus, *key)]["selected_global_indices"])
                != set(indexed[(scope, minus, *key)]["selected_global_indices"]) for key in identities)
            comparisons.append({"plus": plus, "minus": minus, "selected_set_changed_questions": changed,
                "metrics": {metric: delta(values[plus][metric] - values[minus][metric]) for metric in METRICS},
                "sign_interpretation": "positive quality deltas are wins; positive token/count deltas mean larger, not better"})
        result.append({"scope": scope, "methods": metrics, "paired_comparisons": comparisons})
    return result


def evaluate_all(source, vectors, *, deadline):
    records, leaf_times, chunk_times = [], [], []
    for q in source["prepared"]["queries"]:
        check_time(deadline)
        key = (q["doc_id"], q["question_id"])
        vector = source["query_vectors"][source["query_positions"][key]]
        tick = time.perf_counter()
        leaf_scores = (source["candidate_vectors"] @ vector).tolist()
        leaf_times.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        chunk_scores = (vectors @ vector).tolist()
        chunk_times.append(time.perf_counter() - tick)
        bridge.validate_within_document_replay({"queries": [q]}, source["documents"], source["units"], source["keys"],
            source["positions"], {key: leaf_scores}, source["rankings"], source["tokenizer"], deadline)
        records.extend(evaluate_query(q, source, leaf_scores, chunk_scores, deadline=deadline))
    return records, leaf_times, chunk_times


def public_result(source, plan, records, execution):
    header_counts = Counter(unit.unit_id for unit in source["units"])
    return {"schema": SCHEMA, "status": "completed", "config": CONFIG, "question_count": 77,
        "family_count": 24, "leaf_count": len(source["leaves"]), "chunk_count": len(source["chunks"]),
        "api_calls": 0, "test_payload_read": False, "gold_used_for_ranking_selection": False,
        "source_hashes_unchanged": True, "input_binding_sha256": plan["input_binding_sha256"],
        "global_rendered_id_collisions": sum(v - 1 for v in header_counts.values()),
        "all_packs_within_budget": True, "scopes": summarize(records, source["prepared"]["queries"]),
        "environment": environment(), "execution": execution, "limits": LIMITS}


def run(args):
    if not args.confirm_idle:
        raise ValueError("run requires explicit --confirm-idle after the root/user releases the GPU")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("run output already exists")
    plan, token_ids, plan_hashes = load_plan(args.plan)
    source = load_inputs(plan["prepared"], plan["dense"], plan["chunks"])
    validate_reloaded(plan, token_ids, source)
    pilot.verify_hashes(plan_hashes)
    idle = confirm_idle()  # failure leaves the experiment prepared, without a model or run directory
    output.mkdir(parents=True, exist_ok=False)
    started, stage = time.monotonic(), "chunk_encoding"
    report = {"schema": SCHEMA, "status": "started", "config": CONFIG, "api_calls": 0,
        "test_payload_read": False, "training_performed": False, "answer_generation_performed": False,
        "independent_confirmation": False, "jev_labels_reused": False, "idle_samples": idle,
        "input_sha256": plan["input_sha256"], "plan_sha256": plan_hashes, "environment": environment(), "limits": LIMITS}
    try:
        torch.set_num_threads(1)
        vectors, encoding = encode_chunks(plan["model"], source["tokenizer"], token_ids)
        bridge.validate_vectors(vectors, source["query_vectors"],
            {"pooling": "CLS_then_FP32_L2", "model_revision": MODEL_REVISION},
            candidate_count=len(source["chunks"]), query_count=len(source["query_vectors"]))
        save_file({"chunk_embeddings": vectors.contiguous()}, str(output / "chunk_embeddings.safetensors"),
            metadata={"pooling": "CLS_then_FP32_L2", "model_revision": MODEL_REVISION})
        pilot.write_json(output / "chunk_index.json", plan["identity"]["chunks"])
        stage = "retrieval_and_scoring"
        deadline = time.monotonic() + CONFIG["evaluation_max_seconds"]
        records, leaf_times, chunk_times = evaluate_all(source, vectors, deadline=deadline)
        stage = "verification"
        pilot.verify_hashes(plan["input_sha256"])
        pilot.verify_hashes(plan_hashes)
        check_time(deadline)
        execution = {**encoding, "leaf_score_wall_seconds": sum(leaf_times), "chunk_score_wall_seconds": sum(chunk_times),
                "total_run_wall_seconds": time.monotonic() - started, "cached_leaf_rows": 1850, "cached_query_rows": 104,
                "timing_scope": "one shared chunk encoding and score arrays; owner projection/packing timings per arm; not cold-start latency"}
        public = public_result(source, plan, records, execution)
        check_time(deadline)
        with (output / "per_question.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        pilot.write_json(output / "public_aggregate.json", public)
        report.update(status="completed", public=public,
            output_sha256={p.name: digest(p) for p in sorted(output.iterdir())})
    except Exception as error:
        report.update(status="failed", failure_stage=stage, error_type=type(error).__name__, retry_or_fallback=False)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        pilot.write_json(output / "summary.json", report)
    return {"status": "completed", "question_count": 77, "configurations": 6, "output": str(output)}


def validate_execution(execution, records):
    expected = {"encoding_wall_seconds", "device", "cuda_runtime", "peak_allocated_bytes", "peak_reserved_bytes",
        "leaf_score_wall_seconds", "chunk_score_wall_seconds", "total_run_wall_seconds", "cached_leaf_rows",
        "cached_query_rows", "timing_scope"}
    if set(execution) != expected or execution["cached_leaf_rows"] != 1850 or execution["cached_query_rows"] != 104:
        raise ValueError("execution metadata inventory/cache counts differ")
    if not all(isinstance(execution[key], str) and execution[key] for key in ("device", "cuda_runtime", "timing_scope")):
        raise ValueError("missing device/timing metadata")
    if execution["timing_scope"] != "one shared chunk encoding and score arrays; owner projection/packing timings per arm; not cold-start latency":
        raise ValueError("execution timing scope differs")
    def bounded(value, maximum):
        return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= maximum
    for name, value in execution.items():
        if name.endswith("seconds") and not bounded(value, 1200 if name == "total_run_wall_seconds" else 600):
            raise ValueError("execution time outside frozen bounds")
    if (execution["encoding_wall_seconds"] > execution["total_run_wall_seconds"]
            or not bounded(execution["peak_allocated_bytes"], CONFIG["gpu_allocated_limit_bytes"])
            or not bounded(execution["peak_reserved_bytes"], CONFIG["gpu_reserved_limit_bytes"])
            or execution["peak_allocated_bytes"] > execution["peak_reserved_bytes"]):
        raise ValueError("execution memory/time metadata inconsistent")
    for row in records:
        if any(not bounded(row[name], 600) for name in ("retrieval_wall_seconds", "packing_wall_seconds")):
            raise ValueError("per-question timing outside frozen bounds")


def audit(args):
    """CPU-only replay from sealed cached tensors; never loads a model or uses GPU.

    A tensor hash and CLS-unit-norm check bind the saved representations, not a
    second execution of the backbone. Quality/selection/summary are recomputed.
    """
    directory = Path(args.run).resolve()
    expected = {"summary.json", "per_question.jsonl", "public_aggregate.json", "chunk_index.json", "chunk_embeddings.safetensors"}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError("completed run file inventory differs")
    # Capture every output before parsing/replaying; verify the same bytes again at the end.
    buffers = {name: (directory / name).read_bytes() for name in expected}
    snapshots = {str(directory / name): hashlib.sha256(value).hexdigest() for name, value in buffers.items()}
    plan, ids, plan_hashes = load_plan(args.plan)
    source = load_inputs(plan["prepared"], plan["dense"], plan["chunks"])
    validate_reloaded(plan, ids, source)
    report, public = json.loads(buffers["summary.json"]), json.loads(buffers["public_aggregate.json"])
    fixed = {"schema": SCHEMA, "status": "completed", "config": CONFIG, "api_calls": 0,
        "test_payload_read": False, "training_performed": False, "answer_generation_performed": False,
        "independent_confirmation": False, "jev_labels_reused": False, "input_sha256": plan["input_sha256"],
        "plan_sha256": plan_hashes, "environment": environment(), "limits": LIMITS}
    if any(report.get(key) != value for key, value in fixed.items()):
        raise ValueError("run metadata differs from sealed plan")
    if set(report) != set(fixed) | {"idle_samples", "public", "output_sha256", "elapsed_seconds"}:
        raise ValueError("run summary metadata inventory differs")
    if len(report["idle_samples"]) != CONFIG["idle_samples"]:
        raise ValueError("completed run lacks all three idle samples")
    for sample in report["idle_samples"]:
        assert_idle(sample)
    if report["output_sha256"] != {name: snapshots[str(directory / name)] for name in expected - {"summary.json"}}:
        raise ValueError("completed output seals differ")
    if report["public"] != public or json.loads(buffers["chunk_index.json"]) != plan["identity"]["chunks"]:
        raise ValueError("public summary or chunk embedding index differs")
    blob = buffers["chunk_embeddings.safetensors"]
    header_size = struct.unpack("<Q", blob[:8])[0]
    if header_size > 1024 * 1024:
        raise ValueError("unexpected chunk tensor header")
    metadata = json.loads(blob[8:8 + header_size]).get("__metadata__")
    tensors = load_tensors(blob)
    if set(tensors) != {"chunk_embeddings"}:
        raise ValueError("unexpected chunk tensor inventory")
    vectors = tensors["chunk_embeddings"]
    bridge.validate_vectors(vectors, source["query_vectors"], metadata,
        candidate_count=len(source["chunks"]), query_count=len(source["query_vectors"]))
    saved = [json.loads(line) for line in buffers["per_question.jsonl"].splitlines() if line.strip()]
    validate_execution(public["execution"], saved)
    if (type(report["elapsed_seconds"]) not in (int, float) or not math.isfinite(report["elapsed_seconds"])
            or not public["execution"]["total_run_wall_seconds"] <= report["elapsed_seconds"] <= 1260):
        raise ValueError("total elapsed time inconsistent")
    torch.set_num_threads(1)
    recomputed, _, _ = evaluate_all(source, vectors, deadline=time.monotonic() + 600)
    def without_times(row):
        return {key: value for key, value in row.items() if key not in ("retrieval_wall_seconds", "packing_wall_seconds")}
    if [without_times(row) for row in saved] != [without_times(row) for row in recomputed]:
        raise ValueError("replayed selections/traces/metrics differ from saved records")
    # Reuse observed times only after deterministic records match; recompute their aggregate too.
    if public_result(source, plan, saved, public["execution"]) != public:
        raise ValueError("recomputed full aggregate metadata/comparisons differ")
    pilot.verify_hashes(plan["input_sha256"])
    pilot.verify_hashes(plan_hashes)
    pilot.verify_hashes(snapshots)
    return {"status": "verified", "records": len(saved), "configurations": 6,
        "saved_tensor_hash_index_norm_verified": True, "selection_metrics_and_comparisons_recomputed": True,
        "gpu_used": False, "model_loaded": False, "api_calls": 0,
        "limitation": "Tensor provenance is hash/index/norm validation; the backbone is not executed again."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    for flag in ("prepared", "dense", "chunks", "output"):
        prepare_parser.add_argument("--" + flag, required=True)
    run_parser = subparsers.add_parser("run")
    for flag in ("plan", "output"):
        run_parser.add_argument("--" + flag, required=True)
    run_parser.add_argument("--confirm-idle", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--plan", required=True)
    audit_parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run, "audit": audit}[args.command](args), indent=2))


if __name__ == "__main__":
    main()
