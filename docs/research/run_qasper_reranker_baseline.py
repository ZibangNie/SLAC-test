"""Pinned local cross-encoder baseline on the frozen 77-query candidate pool.

prepare loads tokenizers and hashes files, but never loads model weights onto a
device. run requires an explicit frozen preparation and an idle CUDA device.
No API, training, answers, truncation, CPU fallback, or automatic OOM retry.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import prepare_qasper_extended_development as preparation
from prepare_qasper_relation_pilot import verify_hashes
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, pack_ranked, score_selection
from run_qasper_relation_pilot import selected_gold


MODEL_ID = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PARAMETERS = 567755777
EXPECTED_QUERIES, EXPECTED_PAIRS = 77, 1214
SCHEMA = "slac-qasper-reranker-plan-v1"
CONTRACT = {
    "model_id": MODEL_ID, "revision": REVISION, "parameters": PARAMETERS,
    "architecture": "XLMRobertaForSequenceClassification", "dtype": "float16",
    "device": "cuda:0", "attention": "sdpa", "microbatch": 4, "max_seconds": 600,
    "max_pair_tokens": 1024, "truncation": False, "query_instruction": None,
    "passage_instruction": None, "input": "raw query, canonical unit text as an ordered tokenizer pair",
    "score": "single classification logit converted to float32; larger means more relevant",
    "normalize": False, "tie_break": "original dense rank, then native source order",
    "candidate_scope": "all 1214 frozen support pairs from 77 questions; no candidate additions",
    "selection_k": [1, 2, 3], "evidence_budget_bge_tokens": 1024,
    "deduplication": "exact native text, retain the first ranked fitting source location",
    "gpu_admission": {"minimum_free_mib": 4096, "maximum_utilization_percent": 20,
                      "samples": 3, "sample_interval_seconds": 1,
                      "defer_if_process_running": "FIFA18.exe"},
    "automatic_retries": 0, "cpu_fallback": False,
}
# Official HF model API ?blobs=true, pinned above. LFS files use SHA256;
# small Git blobs use the repository's blob ID, plus SHA256 in every local plan.
FILES = {
    "model.safetensors": (2271071852, "sha256", "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"),
    "tokenizer.json": (17098273, "sha256", "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15"),
    "sentencepiece.bpe.model": (5069051, "sha256", "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "config.json": (795, "git_blob_sha1", "9f62673cb00ec41dcec8947b9ed16f6f2eb23ba2"),
    "tokenizer_config.json": (1173, "git_blob_sha1", "328a00a9a560aadcf2a3064f917517359eb3cc26"),
    "special_tokens_map.json": (964, "git_blob_sha1", "b1879d702821e753ffe4245048eee415d54a9385"),
}
CODE_FILES = ("run_qasper_reranker_baseline.py", "prepare_qasper_extended_development.py",
              "prepare_qasper_relation_pilot.py", "run_qasper_evidence_baselines.py",
              "run_qasper_relation_pilot.py", "qasper_metrics.py")
OUTPUT_FILES = ("pair_scores.jsonl", "rankings.jsonl", "per_question.jsonl")
LIMITS = [
    "Exposed validation development data; no independent confirmation or official test QA.",
    "Given-document fixed-pool evidence selection; no corpus retrieval or answer generation.",
    "Cross-encoder scores are raw ranking logits, not calibrated support probabilities.",
    "All k=1/2/3 are retained; no metric-driven threshold or best-k substitution.",
    "Equal evidence caps do not imply equal actual evidence lengths.",
    "Local inference has no paid API charge; download and device costs are not monetary measurements.",
    "Reported model-input tokens include the query and pair special tokens; evidence tokens use the separate frozen BGE tokenizer.",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def object_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("bounded reranker stage time exceeded")


def failure(output, stage, exc, started):
    write(output / "failure.json", {"status": "failed", "stage": stage,
        "error_class": type(exc).__name__, "elapsed_seconds": time.monotonic() - started,
        "api_calls": 0, "automatic_retries": 0, "cpu_fallback": False,
        "experimental_scores_available": False})


def verify_model(directory):
    directory = Path(directory).resolve()
    hashes = {}
    for name, (size, kind, expected) in FILES.items():
        path = directory / name
        if path.stat().st_size != size:
            raise ValueError("pinned model file size mismatch")
        hashes[str(path)] = digest(path)
        if kind == "sha256":
            actual = hashes[str(path)]
        else:
            raw = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != expected:
            raise ValueError("pinned model file identity mismatch")
    config = read(directory / "config.json")
    if (config.get("architectures") != [CONTRACT["architecture"]]
            or config.get("max_position_embeddings") != 8194 or set(config.get("id2label", {})) != {"0"}):
        raise ValueError("pinned reranker classification architecture differs")
    if read(directory / "tokenizer_config.json").get("model_max_length") != 8192:
        raise ValueError("pinned tokenizer context differs")
    return hashes


def support_pairs(prepared, documents):
    queries = {(q["doc_id"], q["question_id"]): q for q in prepared["queries"]}
    if len(queries) != len(prepared["queries"]):
        raise ValueError("duplicate frozen query")
    wanted = {(doc, qid, uid) for (doc, qid), row in queries.items() for uid in row["candidate_ids"]}
    rows, seen, task_ids = [], set(), set()
    lookup = {doc: {unit.unit_id: unit for unit in units} for doc, units in documents.items()}
    for task in prepared["support_tasks"]:
        key = (task["doc_id"], task["question_id"], task["unit_id"])
        if key not in wanted or key in seen or task["id"] in task_ids:
            raise ValueError("support pair coverage or identity mismatch")
        query = queries[key[:2]]
        unit = lookup[key[0]][key[2]]
        expected = {"query": query["query"], "unit": {"id": unit.unit_id, "text": unit.text}}
        if task["item"] != expected:
            raise ValueError("support pair differs from frozen visible text")
        seen.add(key)
        task_ids.add(task["id"])
        rows.append({"task_id": task["id"], "doc_id": key[0], "question_id": key[1],
                     "unit_id": key[2], "query": query["query"], "passage": unit.text})
    if seen != wanted:
        raise ValueError("support pair coverage incomplete")
    return rows


def encode_pairs(pairs, tokenizer, *, deadline=math.inf):
    encoded, audit = [], []
    for row in pairs:
        deadline_check(deadline)
        item = dict(tokenizer(row["query"], row["passage"], add_special_tokens=True,
                              truncation=False, padding=False))
        ids = item.get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError("invalid complete pair tokenization")
        if len(ids) > CONTRACT["max_pair_tokens"]:
            raise ValueError("complete pair exceeds frozen context cap; no truncation")
        if any(not isinstance(values, list) or len(values) != len(ids) for values in item.values()):
            raise ValueError("encoded pair fields differ in length")
        encoded.append(item)
        audit.append({key: row[key] for key in ("task_id", "doc_id", "question_id", "unit_id")}
                     | {"pair_tokens": len(ids), "encoding_sha256": object_hash(item)})
    return encoded, audit


def length_summary(audit):
    lengths = sorted(row["pair_tokens"] for row in audit)
    if not lengths:
        raise ValueError("empty pair audit")
    percentile = lambda percent: lengths[math.ceil(len(lengths) * percent) - 1]
    return {"pairs": len(lengths), "actual_pair_tokens": sum(lengths), "minimum": min(lengths),
            "p50_nearest_rank": percentile(.5), "p95_nearest_rank": percentile(.95),
            "p99_nearest_rank": percentile(.99), "maximum": max(lengths),
            "pairs_over_512": sum(x > 512 for x in lengths), "pairs_over_1024": sum(x > 1024 for x in lengths),
            "truncated_pairs": 0, "tokenizer": "pinned reranker fast tokenizer, complete query-passage pairs"}


def prepare(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        prepared_path, model_path, bge_path = (Path(getattr(args, name)).resolve()
                                              for name in ("prepared", "model", "bge_tokenizer"))
        prepared, manifest, documents = preparation.load_prepared(prepared_path)
        pairs = support_pairs(prepared, documents)
        if len(prepared["queries"]) != EXPECTED_QUERIES or len(pairs) != EXPECTED_PAIRS:
            raise ValueError("requires all 77 frozen queries and 1214 frozen support pairs")
        hashes = dict(manifest["input_sha256"])
        hashes.update(verify_model(model_path))
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
            path = bge_path / name
            if hashes.get(str(path)) != digest(path):
                raise ValueError("BGE evidence tokenizer differs from frozen baseline")
        for path in (prepared_path / "prepared.json", prepared_path / "manifest.json",
                     *(Path(__file__).resolve().parent / name for name in CODE_FILES)):
            hashes[str(path)] = digest(path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        encoded, audit = encode_pairs(pairs, tokenizer, deadline=started + CONTRACT["max_seconds"])
        del encoded
        verify_hashes(hashes)
        write_rows(output / "pair_audit.jsonl", audit)
        summary = length_summary(audit)
        write(output / "length_audit.json", summary)
        config = {"schema": SCHEMA, "status": "prepared_no_model_inference",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "contract": dict(CONTRACT),
            "prepared_dir": str(prepared_path), "model_dir": str(model_path), "bge_tokenizer": str(bge_path),
            "input_sha256": hashes, "plan_files_sha256": {name: digest(output / name)
                for name in ("pair_audit.jsonl", "length_audit.json")},
            "question_count": len(prepared["queries"]), "pair_count": len(pairs),
            "length_audit": summary, "api_calls": 0, "model_inference_performed": False,
            "test_payload_read": False, "limits": LIMITS, "elapsed_seconds": time.monotonic() - started}
        write(output / "experiment_config.json", config)
        write(output / "plan_manifest.json", {"config_sha256": digest(output / "experiment_config.json"),
                                               "status": "prepared_no_model_inference"})
        return config
    except Exception as exc:
        failure(output, "prepare", exc, started)
        raise


def load_plan(directory):
    directory = Path(directory).resolve()
    seal = read(directory / "plan_manifest.json")
    if digest(directory / "experiment_config.json") != seal["config_sha256"]:
        raise ValueError("reranker configuration seal differs")
    config = read(directory / "experiment_config.json")
    if config.get("schema") != SCHEMA or config.get("contract") != CONTRACT:
        raise ValueError("reranker frozen contract differs")
    if config.get("status") != "prepared_no_model_inference":
        raise ValueError("requires a prepared reranker plan")
    verify_hashes(config["input_sha256"])
    verify_hashes({str(directory / name): value for name, value in config["plan_files_sha256"].items()})
    return config


def gpu_sample():
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                            check=True)
    parts = result.stdout.strip().split(",")
    if len(parts) != 3:
        raise ValueError("GPU admission output invalid")
    return {"name": parts[0].strip(), "free_mib": int(parts[1]), "utilization_percent": int(parts[2])}


def user_game_running():
    """Query only the named foreground-workload risk; never change a process."""
    image_name = CONTRACT["gpu_admission"]["defer_if_process_running"]
    result = subprocess.run(["tasklist.exe", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=True)
    return any(row and row[0].casefold() == image_name.casefold()
               for row in csv.reader(result.stdout.splitlines()))


def ensure_gpu_ready():
    limit = CONTRACT["gpu_admission"]
    samples = []
    for index in range(limit["samples"]):
        if user_game_running():
            raise RuntimeError("FIFA18 is running; defer without changing the user process")
        sample = gpu_sample()
        if sample["free_mib"] < limit["minimum_free_mib"] or sample["utilization_percent"] > limit["maximum_utilization_percent"]:
            raise RuntimeError("CUDA device is busy; defer without changing other processes")
        samples.append(sample)
        if index + 1 < limit["samples"]:
            time.sleep(limit["sample_interval_seconds"])
    return samples


def infer_pairs(encoded, tokenizer, model, *, device, deadline):
    order = sorted(range(len(encoded)), key=lambda index: (-len(encoded[index]["input_ids"]), index))
    scores = [None] * len(encoded)
    max_padded = 0
    padded_total = 0
    model.eval()
    model.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(order), CONTRACT["microbatch"]):
            deadline_check(deadline)
            positions = order[start:start + CONTRACT["microbatch"]]
            batch = tokenizer.pad([encoded[index] for index in positions], padding=True, return_tensors="pt")
            for row, index in enumerate(positions):
                mask = batch["attention_mask"][row].bool()
                actual = batch["input_ids"][row][mask].tolist()
                if actual != encoded[index]["input_ids"]:
                    raise ValueError("padded model input differs from complete audited encoding")
            padded = batch["input_ids"].numel()
            padded_total += padded
            max_padded = max(max_padded, padded)
            logits = model(**{name: tensor.to(device) for name, tensor in batch.items()}, return_dict=True).logits
            if tuple(logits.shape) != (len(positions), 1) or not torch.isfinite(logits).all().item():
                raise ValueError("reranker logits must be finite with one score per pair")
            for index, value in zip(positions, logits[:, 0].float().cpu().tolist()):
                scores[index] = value
            deadline_check(deadline)
    if any(value is None for value in scores):
        raise ValueError("incomplete reranker scores")
    return scores, {"padded_input_tokens": padded_total, "max_padded_tokens_per_batch": max_padded,
                    "max_microbatch": CONTRACT["microbatch"]}


def evaluate(prepared, documents, annotations, pairs, scores, tokenizer, *, deadline=math.inf):
    if len(pairs) != len(scores) or any(not math.isfinite(value) for value in scores):
        raise ValueError("score coverage or finiteness differs")
    by_query = defaultdict(dict)
    for pair, score in zip(pairs, scores):
        key, uid = (pair["doc_id"], pair["question_id"]), pair["unit_id"]
        if uid in by_query[key]:
            raise ValueError("duplicate scored unit")
        by_query[key][uid] = score
    records, rankings = [], []
    for query in prepared["queries"]:
        deadline_check(deadline)
        key = query["doc_id"], query["question_id"]
        units, values = documents[key[0]], by_query[key]
        if set(values) != set(query["candidate_ids"]):
            raise ValueError("reranker must score every original candidate")
        by_id = {unit.unit_id: index for index, unit in enumerate(units)}
        dense_position = {uid: index for index, uid in enumerate(query["ranked_ids"])}
        ranked = sorted(values, key=lambda uid: (-values[uid], dense_position[uid], units[by_id[uid]].order))
        rankings.append({"doc_id": key[0], "question_id": key[1], "ranked_ids": ranked})
        count = PackCounter(tokenizer, units, deadline=deadline)
        for k in CONTRACT["selection_k"]:
            chosen = pack_ranked(units, [by_id[uid] for uid in ranked],
                                 CONTRACT["evidence_budget_bge_tokens"], count, max_units=k)
            records.append({"doc_id": key[0], "family_id": query["family_id"], "question_id": key[1],
                "method": f"bge_reranker_v2_m3_k{k}", "budget": CONTRACT["evidence_budget_bge_tokens"],
                **score_selection(units, chosen, annotations[key], count, CONTRACT["evidence_budget_bge_tokens"])})
    if set(by_query) != {(q["doc_id"], q["question_id"]) for q in prepared["queries"]}:
        raise ValueError("unexpected scored query")
    return records, rankings


def run(args):
    started = time.monotonic()
    deadline = started + CONTRACT["max_seconds"]
    directory, output = Path(args.plan).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        initial = {str(directory / name): digest(directory / name) for name in
                   ("plan_manifest.json", "experiment_config.json", "pair_audit.jsonl", "length_audit.json")}
        config = load_plan(directory)
        prepared, manifest, documents = preparation.load_prepared(config["prepared_dir"])
        pairs = support_pairs(prepared, documents)
        if len(pairs) != EXPECTED_PAIRS or len(prepared["queries"]) != EXPECTED_QUERIES:
            raise ValueError("frozen execution coverage differs")
        tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        encoded, audit = encode_pairs(pairs, tokenizer, deadline=deadline)
        frozen_audit = [json.loads(line) for line in (directory / "pair_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        if audit != frozen_audit or length_summary(audit) != config["length_audit"]:
            raise ValueError("complete input encodings differ from pre-inference audit")
        verify_hashes(initial)
        samples = ensure_gpu_ready()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        deadline_check(deadline)
        torch.cuda.reset_peak_memory_stats(0)
        model = AutoModelForSequenceClassification.from_pretrained(config["model_dir"],
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0")
        scores, compute = infer_pairs(encoded, tokenizer, model, device="cuda:0", deadline=deadline)
        torch.cuda.synchronize(0)
        compute.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        del model
        annotations = selected_gold(manifest["source_paths"]["sidecar"], prepared)
        bge = AutoTokenizer.from_pretrained(config["bge_tokenizer"], local_files_only=True, trust_remote_code=False)
        records, rankings = evaluate(prepared, documents, annotations, pairs, scores, bge, deadline=deadline)
        verify_hashes(config["input_sha256"])
        verify_hashes(initial)
        deadline_check(deadline)
        write_rows(output / "pair_scores.jsonl", [{**row, "raw_logit": score} for row, score in zip(audit, scores)])
        write_rows(output / "rankings.jsonl", rankings)
        write_rows(output / "per_question.jsonl", records)
        summary = {"status": "completed", "model_id": MODEL_ID, "revision": REVISION,
            "question_count": len(prepared["queries"]), "family_count": len(documents),
            "pair_count": len(pairs), "record_count": len(records), "metrics": aggregate(records),
            "contract": CONTRACT, "length_audit": config["length_audit"], "compute": compute,
            "gpu_admission_samples": samples, "plan_sha256": initial[str(directory / "experiment_config.json")],
            "output_sha256": {name: digest(output / name) for name in OUTPUT_FILES},
            "execution": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda, "device": "cuda:0", "dtype": "float16",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0))},
            "input_hashes_unchanged": True, "api_calls": 0, "paid_api_cost_usd": "0",
            "elapsed_seconds": time.monotonic() - started, "limits": LIMITS,
            "answer_generation_performed": False, "training_performed": False, "test_payload_read": False,
            "all_packs_within_budget": all(row["actual_evidence_tokens"] <= row["budget"] for row in records),
            "raw_text_or_question_ids_in_summary": False}
        write(output / "summary.json", summary)
        write(output / "run_manifest.json", {"status": "completed", "summary_sha256": digest(output / "summary.json"),
            "output_sha256": summary["output_sha256"], "plan_sha256": summary["plan_sha256"],
            "plan_manifest_sha256": initial[str(directory / "plan_manifest.json")],
            "pair_audit_sha256": initial[str(directory / "pair_audit.jsonl")]})
        return summary
    except Exception as exc:
        failure(output, "run", exc, started)
        raise


def restored_scores(saved_rows, frozen_audit):
    if len(saved_rows) != len(frozen_audit):
        raise ValueError("saved score count differs from frozen pair audit")
    scores = []
    for saved, expected in zip(saved_rows, frozen_audit):
        if set(saved) != set(expected) | {"raw_logit"} or {key: saved[key] for key in expected} != expected:
            raise ValueError("saved score identity or encoding differs from exact pair audit")
        value = saved["raw_logit"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("saved raw logit is invalid")
        scores.append(value)
    return scores


def audit_saved_run(plan_dir, run_dir, *, deadline=math.inf):
    """Independently replay persisted scores; no model load or GPU operation."""
    plan_dir, run_dir = Path(plan_dir).resolve(), Path(run_dir).resolve()
    initial = {str(path): digest(path) for path in
        [*(plan_dir / name for name in ("plan_manifest.json", "experiment_config.json", "pair_audit.jsonl", "length_audit.json")),
         *(run_dir / name for name in (*OUTPUT_FILES, "summary.json", "run_manifest.json"))]}
    config, seal, summary = load_plan(plan_dir), read(run_dir / "run_manifest.json"), read(run_dir / "summary.json")
    if (seal.get("status") != "completed" or summary.get("status") != "completed"
            or seal["summary_sha256"] != digest(run_dir / "summary.json")
            or seal["plan_sha256"] != digest(plan_dir / "experiment_config.json")
            or seal["plan_manifest_sha256"] != digest(plan_dir / "plan_manifest.json")
            or seal["pair_audit_sha256"] != digest(plan_dir / "pair_audit.jsonl")
            or summary["plan_sha256"] != seal["plan_sha256"]):
        raise ValueError("persisted run seal or plan identity differs")
    if set(seal["output_sha256"]) != set(OUTPUT_FILES) or summary["output_sha256"] != seal["output_sha256"]:
        raise ValueError("persisted output inventory differs")
    verify_hashes({str(run_dir / name): value for name, value in seal["output_sha256"].items()})
    if summary["contract"] != CONTRACT or summary["length_audit"] != config["length_audit"]:
        raise ValueError("persisted execution contract or length audit differs")
    prepared, manifest, documents = preparation.load_prepared(config["prepared_dir"])
    pairs = support_pairs(prepared, documents)
    if len(pairs) != EXPECTED_PAIRS or len(prepared["queries"]) != EXPECTED_QUERIES:
        raise ValueError("independent audit scope differs")
    frozen = [json.loads(line) for line in (plan_dir / "pair_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    reranker_tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
    _, reproduced_audit = encode_pairs(pairs, reranker_tokenizer, deadline=deadline)
    if frozen != reproduced_audit or length_summary(frozen) != config["length_audit"]:
        raise ValueError("frozen pair audit differs from reconstructed complete inputs")
    saved_rows = [json.loads(line) for line in (run_dir / "pair_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = restored_scores(saved_rows, frozen)
    annotations = selected_gold(manifest["source_paths"]["sidecar"], prepared)
    bge = AutoTokenizer.from_pretrained(config["bge_tokenizer"], local_files_only=True, trust_remote_code=False)
    records, rankings = evaluate(prepared, documents, annotations, pairs, scores, bge, deadline=deadline)
    actual_records = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    actual_rankings = [json.loads(line) for line in (run_dir / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
    if records != actual_records or rankings != actual_rankings or aggregate(records) != summary["metrics"]:
        raise ValueError("saved scores do not reproduce every persisted ranking and metric")
    if (summary["pair_count"] != len(pairs) or summary["question_count"] != len(prepared["queries"])
            or summary["family_count"] != len(documents) or summary["record_count"] != len(records)
            or len(records) != EXPECTED_QUERIES * len(CONTRACT["selection_k"])):
        raise ValueError("saved metric denominator differs")
    verify_hashes(config["input_sha256"])
    verify_hashes(initial)
    deadline_check(deadline)
    return {"status": "verified", "pairs": len(pairs), "queries": len(prepared["queries"]),
            "records": len(records), "all_pair_identities_and_encoding_hashes_match": True,
            "all_rankings_and_metrics_reproduced": True, "all_source_plan_output_hashes_unchanged": True,
            "run_manifest_sha256": initial[str(run_dir / "run_manifest.json")],
            "model_inference_performed": False, "api_calls": 0, "raw_text_or_question_ids_in_summary": False}


def audit(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit_saved_run(args.plan, args.run, deadline=started + CONTRACT["max_seconds"])
        result["elapsed_seconds"] = time.monotonic() - started
        write(output / "verification.json", result)
        return result
    except Exception as exc:
        failure(output, "audit", exc, started)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("prepare")
    for name in ("prepared", "model", "bge-tokenizer", "output"):
        offline.add_argument("--" + name, required=True)
    execute = commands.add_parser("run")
    for name in ("plan", "output"):
        execute.add_argument("--" + name, required=True)
    verify = commands.add_parser("audit")
    for name in ("plan", "run", "output"):
        verify.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = {"prepare": prepare, "run": run, "audit": audit}[arguments.command](arguments)
    print(json.dumps({key: result.get(key) for key in
          ("status", "question_count", "pair_count", "length_audit", "elapsed_seconds")}, indent=2))
