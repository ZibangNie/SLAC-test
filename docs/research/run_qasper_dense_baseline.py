"""Bounded, offline BGE-M3 dense evidence selection on the frozen Qasper dev pool.

Dense pooling follows the pinned BAAI model's CLS pooling plus Normalize:
https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json
https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/modules.json
The matching model card says no query instruction is needed. Inference uses
FP16 backbone, FP32 CLS L2 normalization and cosine scoring; no fine-tuning.
Only query/source-unit text is encoded. Reference answers enter only scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import torch
from safetensors.torch import save_file
import transformers
from transformers import AutoModel, AutoTokenizer

from run_qasper_evidence_baselines import (
    PackCounter, aggregate, build_units, check_time, digest, load_frozen_pool,
    pack_ranked, score_selection,
)


MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_URL = f"https://huggingface.co/BAAI/bge-m3/blob/{MODEL_REVISION}"


def audit_token_lengths(tokenizer, texts, max_length, deadline=math.inf):
    """Return complete token IDs; the same untruncated IDs enter the backbone."""
    if not 2 <= max_length <= 8192:
        raise ValueError("max_length must be between 2 and 8192 including specials")
    all_ids = []
    for text in texts:
        check_time(deadline)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model inputs must be nonempty strings")
        all_ids.append(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    lengths = [len(ids) for ids in all_ids]
    ordered = sorted(lengths)
    summary = {
        "count": len(lengths), "total_tokens": sum(lengths),
        "min": min(lengths, default=0), "max": max(lengths, default=0),
        "p50": ordered[(len(ordered) - 1) // 2] if ordered else 0,
        "p95": ordered[math.ceil(len(ordered) * .95) - 1] if ordered else 0,
        "above_max_length": sum(length > max_length for length in lengths),
        "max_length_includes_special_tokens": max_length,
        "truncation": False,
    }
    return all_ids, summary


def dense_cls(hidden_states):
    """Use the raw CLS token, never pooler_output or mean pooling."""
    if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
        raise ValueError("expected [batch, sequence, hidden] model states")
    vectors = hidden_states[:, 0].float()
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if not torch.isfinite(vectors).all() or (norms <= 0).any():
        raise ValueError("nonfinite or zero-norm dense representation")
    return vectors / norms


def encode_token_ids(model, tokenizer, all_ids, *, batch_size, device, deadline,
                     max_length, progress=None):
    if not 1 <= batch_size <= 4:
        raise ValueError("microbatch size must be 1-4")
    if any(not ids or len(ids) > max_length for ids in all_ids):
        raise ValueError("audited input exceeds max_length; truncation is forbidden")
    ordering = sorted(range(len(all_ids)), key=lambda index: (len(all_ids[index]), index))
    result = None
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(ordering), batch_size):
            check_time(deadline)
            indices = ordering[offset:offset + batch_size]
            batch = tokenizer.pad({"input_ids": [all_ids[index] for index in indices]},
                                  padding=True, return_tensors="pt")
            if batch["input_ids"].shape[1] > max_length:
                raise ValueError("padding exceeds audited max_length")
            if [int(value) for value in batch["attention_mask"].sum(-1)] != [len(all_ids[index]) for index in indices]:
                raise ValueError("padding changed an audited non-padding length")
            batch = {key: value.to(device) for key, value in batch.items()}
            vectors = dense_cls(model(**batch).last_hidden_state).cpu()
            check_time(deadline)
            if result is None:
                result = torch.empty((len(all_ids), vectors.shape[-1]), dtype=torch.float32)
            result[indices] = vectors
            if progress is not None:
                progress(min(offset + batch_size, len(ordering)), len(ordering))
    if result is None:
        raise ValueError("no model inputs")
    return result


def dense_ranking(query_embedding, unit_embeddings, units):
    if unit_embeddings.ndim != 2 or query_embedding.ndim != 1 or unit_embeddings.shape != (len(units), len(query_embedding)):
        raise ValueError("dense representation shape mismatch")
    if not torch.isfinite(unit_embeddings).all() or not torch.isfinite(query_embedding).all():
        raise ValueError("nonfinite ranking input")
    scores = (unit_embeddings.float() @ query_embedding.float()).tolist()
    return sorted(range(len(units)), key=lambda index: (-scores[index], units[index].order))


def load_documents(pool, manifest, candidates):
    wanted = {row["doc_id"] for row in candidates}
    shard = Path(next(path for path in manifest["input_sha256"] if "documents-" in path))
    alignment = json.loads((pool / "native_qa_alignment.json").read_text(encoding="utf-8"))
    archive = Path(next(path for path in alignment["input_sha256"] if path.endswith("qasper-train-dev-v0.3.tgz")))
    for path, expected in ((shard, manifest["input_sha256"][str(shard)]), (archive, alignment["input_sha256"][str(archive)])):
        if digest(path) != expected:
            raise ValueError("source hash differs from frozen lineage")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("qasper-dev-v0.3.json")
        if not member.isfile() or member.size > 32 * 1024 * 1024:
            raise ValueError("unexpected native validation member")
        with tar.extractfile(member) as stream:
            raw = json.load(stream)
    documents = {}
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["doc_id"] in wanted:
                if row["original_split"] != "validation":
                    raise ValueError("canonical split mismatch")
                paper = {key: raw[row["source_id"]][key] for key in ("title", "abstract", "full_text")}
                documents[row["doc_id"]] = build_units(row, paper)
    if set(documents) != wanted:
        raise ValueError("missing canonical candidates")
    return documents, shard, archive


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def run(args):
    started = time.monotonic()
    deadline = started + args.max_seconds
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "scope": "given-document evidence-selection development diagnostic",
              "independent_evaluation": False, "answer_generation_performed": False,
              "test_payload_read": False, "api_calls": 0, "truncated_inputs": 0}
    stage = "inputs"
    try:
        pool, sidecar, model_path = Path(args.pool), Path(args.sidecar), Path(args.model)
        manifest, candidates, qa_rows = load_frozen_pool(pool, sidecar)
        documents, shard, archive = load_documents(pool, manifest, candidates)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        unit_keys = [(doc_id, index) for doc_id in sorted(documents) for index in range(len(documents[doc_id]))]
        unit_texts = [documents[doc_id][index].text for doc_id, index in unit_keys]
        query_texts = [row["question"] for row in qa_rows]
        unit_ids, unit_audit = audit_token_lengths(tokenizer, unit_texts, args.max_length, deadline)
        query_ids, query_audit = audit_token_lengths(tokenizer, query_texts, args.max_length, deadline)
        length_audit = {"candidate_units": unit_audit, "queries": query_audit,
                        "candidate_kinds": dict(Counter(unit.kind for units in documents.values() for unit in units)),
                        "gold_used_for_ranking_or_length_filtering": False}
        write_json(output / "length_audit.json", length_audit)
        print(json.dumps({"stage": "length_audit", **length_audit}), flush=True)
        if unit_audit["above_max_length"] or query_audit["above_max_length"]:
            raise ValueError("complete inputs exceed model max_length; no truncation or dropping allowed")
        tokenizer_paths = [model_path / name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "sentencepiece.bpe.model") if (model_path / name).exists()]
        weights = model_path / "model.safetensors"
        if not weights.is_file():
            raise ValueError("explicit local model.safetensors weights required")
        input_paths = [pool / "pool_manifest.json", pool / "candidates.jsonl", pool / "native_qa_alignment.json", sidecar, sidecar.parent / "alignment_audit_v2.json", shard, archive, *tokenizer_paths, weights]
        input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
        report.update({"input_sha256": input_hashes, "weights_sha256": digest(weights),
                       "model": str(model_path.resolve()), "model_revision": MODEL_REVISION,
                       "question_count": len(qa_rows), "document_count": len(documents),
                       "unit_count": len(unit_keys), "length_audit": length_audit})
        check_time(deadline)
        stage = "device_and_model"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this bounded FP16 baseline; no CPU fallback")
        torch.manual_seed(13)
        torch.cuda.reset_peak_memory_stats()
        report["device"] = {"name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "transformers": transformers.__version__}
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda")
        model.requires_grad_(False).eval()
        if args.max_length > model.config.max_position_embeddings - model.config.pad_token_id - 1:
            raise ValueError("configured max_length exceeds backbone position capacity")
        stage = "encoding"
        last_progress = [time.monotonic()]
        def progress(done, total):
            now = time.monotonic()
            if now - last_progress[0] >= 30 or done == total:
                print(json.dumps({"stage": "encoding", "encoded": done, "total": total,
                                  "elapsed_seconds": now - started}), flush=True)
                last_progress[0] = now
        representations = encode_token_ids(model, tokenizer, [*unit_ids, *query_ids],
            batch_size=args.batch_size, device="cuda", deadline=deadline,
            max_length=args.max_length, progress=progress)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        save_file({"candidate_embeddings": representations[:len(unit_keys)].contiguous(),
                   "query_embeddings": representations[len(unit_keys):].contiguous()},
                  str(output / "embeddings.safetensors"), metadata={"pooling": "CLS_then_FP32_L2", "model_revision": MODEL_REVISION})
        write_json(output / "embedding_index.json", {"candidates": [{"doc_id": doc_id, "unit_id": documents[doc_id][index].unit_id} for doc_id, index in unit_keys],
            "queries": [{"doc_id": row["doc_id"], "question_id": row["question_id"]} for row in qa_rows]})
        stage = "selection_and_scoring"
        by_doc = {}
        for offset, (doc_id, _) in enumerate(unit_keys):
            by_doc.setdefault(doc_id, []).append(offset)
        counters = {doc_id: PackCounter(tokenizer, units, deadline) for doc_id, units in documents.items()}
        records, rankings = [], []
        for query_index, qa in enumerate(qa_rows):
            check_time(deadline)
            doc_id = qa["doc_id"]
            units, counter = documents[doc_id], counters[doc_id]
            ranking = dense_ranking(representations[len(unit_keys) + query_index], representations[by_doc[doc_id]], units)
            rankings.append({"doc_id": doc_id, "question_id": qa["question_id"],
                             "ranked_ids": [units[index].unit_id for index in ranking]})
            for budget in args.budgets:
                selections = {"bge_m3_dense_fill": pack_ranked(units, ranking, budget, counter)}
                selections.update({f"bge_m3_dense_top{topk}": pack_ranked(units, ranking, budget, counter, max_units=topk) for topk in (1, 3, 5)})
                for method, selected in selections.items():
                    records.append({"doc_id": doc_id, "question_id": qa["question_id"], "method": method, "budget": budget,
                        **score_selection(units, selected, qa["answer_annotations"], counter, budget)})
        stage = "verification"
        check_time(deadline)
        if any(digest(Path(path)) != expected for path, expected in input_hashes.items()):
            raise ValueError("an input changed during inference")
        for name, rows in (("per_question.jsonl", records), ("rankings.jsonl", rankings)):
            with (output / name).open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        source_paths = [Path(__file__), Path(__file__).with_name("run_qasper_evidence_baselines.py"), Path(__file__).with_name("qasper_metrics.py"), Path(__file__).with_name("qasper_alignment_v2.py")]
        report.update({"status": "completed", "input_hashes_unchanged": True,
            "source_sha256": {path.name: digest(path) for path in source_paths},
            "budgets": args.budgets, "batch_size": args.batch_size, "max_seconds": args.max_seconds,
            "max_length": args.max_length, "max_length_includes_special_tokens": True,
            "backbone_precision": "float16", "pooling": "last_hidden_state[:,0] then FP32 L2 normalization",
            "ranking": "FP32 cosine similarity; source-order ties; no query instruction; source unit text alone",
            "attention_implementation": "sdpa", "training_performed": False,
            "candidate_policy": "All nonempty native title/section heading/abstract/paragraph units verified against canonical spans; synthetic Abstract heading excluded; full units only; exact duplicate strings selected once",
            "token_budget_definition": "BGE tokenizer, complete rendered evidence pack including unit IDs and special tokens; query/instructions excluded; empty pack zero",
            "top_k": [1, 3, 5], "top_k_policy": "up to k distinct complete units that fit; overlong units skipped; fill includes every ranked fitting distinct unit",
            "all_packs_within_budget": all(row["actual_evidence_tokens"] <= row["budget"] for row in records),
            "embedding_sha256": digest(output / "embeddings.safetensors"),
            "official_model_sources": [f"{MODEL_URL}/README.md", f"{MODEL_URL}/1_Pooling/config.json", f"{MODEL_URL}/modules.json"],
            "metrics": aggregate(records),
            "limits": ["Exposed validation development pool; no independent confirmation claim.", "Given-document evidence selection, not answer quality or full RAG.", "Dense mode only; sparse/multi-vector/hybrid/reranker baselines not evaluated here.", "FP16 inference can differ slightly from FP32.", "BGE evidence budgets are not a selected generator's context accounting."]})
        check_time(deadline)
    except torch.OutOfMemoryError:
        report.update({"status": "failed", "failure_stage": stage, "error_type": "OutOfMemoryError", "retry_or_fallback": False})
        raise
    except Exception as error:
        report.update({"status": "failed", "failure_stage": stage, "error_type": type(error).__name__, "retry_or_fallback": False})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("status", "question_count", "document_count", "elapsed_seconds", "peak_allocated_bytes", "all_packs_within_budget", "metrics")}), flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 300 or not 1 <= args.batch_size <= 4 or not 2 <= args.max_length <= 8192:
        raise ValueError("bounded inference caps exceeded")
    if not args.budgets or len(args.budgets) > 3 or len(set(args.budgets)) != len(args.budgets) or any(budget < 64 or budget > 4096 for budget in args.budgets):
        raise ValueError("unsupported budget list")
    return args


if __name__ == "__main__":
    run(parse_args())
