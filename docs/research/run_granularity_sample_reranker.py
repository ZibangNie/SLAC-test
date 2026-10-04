"""One local BGE pass over two frozen queries and at most 281 unique text pairs.

--prepare seals complete encodings without loading model weights. --run consumes
that preparation in a fresh fixed directory. The parent must impose a hard
300-second process deadline; this child checks a 180-second stage deadline.
No predecessor corpus, references, packing, evaluation, or network operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _deny_network(*_args, **_kwargs):
    raise RuntimeError("network disabled for the bounded local reranker")


# Installed before shared imports, tokenizer loading, or CUDA admission.
socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
socket.create_connection = _deny_network

import run_qasper_reranker_baseline as baseline


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "artifacts/research-foundation/offline-20261005/candidate-granularity-mechanism-01"
INPUT = PHASE / "selected_inputs.json"
PLAN = PHASE / "plan-01"
RUN = PHASE / "run-01"
PRIOR_CONFIG = ROOT / "artifacts/research-foundation/qasper-reranker-02/experiment_config.json"
MODEL = Path("C:/Environment/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots") / baseline.REVISION
INPUT_SCHEMA = "slac-candidate-granularity-reranker-input-v1"
SCHEMA = "slac-candidate-granularity-reranker-v1"
PAIR_FIELDS = {"task_id", "doc_id", "question_id", "unit_id", "query", "passage"}
CONTRACT = {
    "model_id": baseline.MODEL_ID, "revision": baseline.REVISION,
    "parameters": baseline.PARAMETERS,
    "architecture": "XLMRobertaForSequenceClassification",
    "query_count": 2, "maximum_pairs": 281,
    "max_pair_tokens": 1024, "truncation": False,
    "dtype": "float16", "device": "cuda:0", "attention": "sdpa", "microbatch": 4,
    "max_seconds": 180, "child_deadline": "cooperative stage and batch checks",
    "parent_deadline_seconds": 300, "parent_deadline": "external hard process timeout required",
    "automatic_retries": 0, "cpu_fallback": False, "downloads": False, "api_calls": 0,
    "query_instruction": None, "passage_instruction": None, "normalize": False,
    "input": "complete raw query and passage as an ordered tokenizer pair",
    "score": "finite single raw classification logit converted to float32; not a probability",
    "gpu_admission": dict(baseline.CONTRACT["gpu_admission"]),
    "packing_performed": False, "evaluation_performed": False,
}


def _unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pairs_from_sample(sample):
    """Validate only this bounded file; never resolve its provenance bindings."""
    if (not isinstance(sample, dict)
            or set(sample) != {"schema", "pairs", "cases", "input_bindings"}
            or sample["schema"] != INPUT_SCHEMA
            or not isinstance(sample["cases"], list) or len(sample["cases"]) != 2
            or any(not isinstance(case, dict) for case in sample["cases"])
            or not isinstance(sample["input_bindings"], dict)):
        raise ValueError("bounded input schema or case count differs")
    pairs = sample["pairs"]
    if not isinstance(pairs, list) or not 2 <= len(pairs) <= CONTRACT["maximum_pairs"]:
        raise ValueError("bounded pair count differs")
    tasks, identities, texts, queries = set(), set(), set(), {}
    for row in pairs:
        if not isinstance(row, dict) or set(row) != PAIR_FIELDS:
            raise ValueError("pair fields differ")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in PAIR_FIELDS):
            raise ValueError("pair identity or text invalid")
        if (len(row["query"]) > 10000 or len(row["passage"]) > 100000
                or any(len(row[key]) > 1024 for key in ("task_id", "doc_id", "question_id", "unit_id"))):
            raise ValueError("bounded pair string length exceeded")
        identity = row["doc_id"], row["question_id"], row["unit_id"]
        text_pair = row["query"], row["passage"]
        query_identity = row["doc_id"], row["query"]
        if (row["task_id"] in tasks or identity in identities or text_pair in texts
                or (row["question_id"] in queries and queries[row["question_id"]] != query_identity)):
            raise ValueError("duplicate pair or inconsistent query identity")
        tasks.add(row["task_id"])
        identities.add(identity)
        texts.add(text_pair)
        queries[row["question_id"]] = query_identity
    if len(queries) != CONTRACT["query_count"]:
        raise ValueError("requires exactly two query identities")
    return pairs


def sources():
    """Read only fixed input/config/code/model files, never provenance targets."""
    if (baseline.CONTRACT["max_pair_tokens"] != CONTRACT["max_pair_tokens"]
            or baseline.CONTRACT["microbatch"] != CONTRACT["microbatch"]
            or baseline.CONTRACT["gpu_admission"] != CONTRACT["gpu_admission"]):
        raise ValueError("shared utility contract differs")
    old = read(PRIOR_CONFIG)
    if (old["contract"] != baseline.CONTRACT
            or Path(old["model_dir"]).resolve() != MODEL.resolve()):
        raise ValueError("prior pinned model configuration differs")
    hashes = baseline.verify_model(MODEL)
    if any(old["input_sha256"].get(path) != actual for path, actual in hashes.items()):
        raise ValueError("model files differ from the prior pinned baseline")
    code = Path(__file__).resolve().parent
    for path in (INPUT, PRIOR_CONFIG, Path(__file__).resolve(),
                 code / "run_definition_sample_reranker.py",
                 *(code / name for name in baseline.CODE_FILES)):
        hashes[str(path.resolve())] = baseline.digest(path)
    return hashes


def encode(pairs, deadline):
    tokenizer = baseline.AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False, use_fast=True)
    encoded, audit = baseline.encode_pairs(pairs, tokenizer, deadline=deadline)
    if len(encoded) != len(pairs) or len(audit) != len(pairs):
        raise ValueError("complete encoding coverage differs")
    for pair, row in zip(pairs, audit):
        row.update(query_sha256=text_hash(pair["query"]),
                   passage_sha256=text_hash(pair["passage"]))
    return tokenizer, encoded, audit


def runtime():
    return {"torch": str(baseline.torch.__version__),
            "transformers": baseline.transformers.__version__,
            "cuda_runtime": baseline.torch.version.cuda}


def prepare(deadline):
    hashes = sources()
    sample = read(INPUT)
    pairs = pairs_from_sample(sample)
    _, _, audit = encode(pairs, deadline)
    if sources() != hashes:
        raise ValueError("input or dependency changed during preparation")
    baseline.deadline_check(deadline)
    plan = {"schema": SCHEMA, "status": "prepared_no_model_inference",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": CONTRACT, "input_sha256": hashes,
            "input_bindings_recorded_not_read": sample["input_bindings"],
            "case_count": 2, "query_count": 2, "pair_count": len(pairs),
            "pair_audit": audit, "length_audit": baseline.length_summary(audit),
            "runtime": runtime(), "model_inference_performed": False, "api_calls": 0}
    write(PLAN / "plan.json", plan)
    write(PLAN / "plan_manifest.json", {"schema": SCHEMA,
          "plan_sha256": baseline.digest(PLAN / "plan.json")})
    return {"status": plan["status"], "queries": 2, "pairs": len(pairs), "api_calls": 0}


def run(deadline):
    seal_hash = baseline.digest(PLAN / "plan_manifest.json")
    seal = read(PLAN / "plan_manifest.json")
    plan_hash = baseline.digest(PLAN / "plan.json")
    if seal != {"schema": SCHEMA, "plan_sha256": plan_hash}:
        raise ValueError("plan seal differs")
    plan = read(PLAN / "plan.json")
    if (plan["schema"] != SCHEMA or plan["contract"] != CONTRACT
            or plan["status"] != "prepared_no_model_inference"
            or plan["runtime"] != runtime()):
        raise ValueError("plan contract or runtime differs")
    if sources() != plan["input_sha256"]:
        raise ValueError("frozen source, input, or model hashes differ")
    sample = read(INPUT)
    pairs = pairs_from_sample(sample)
    tokenizer, encoded, audit = encode(pairs, deadline)
    if (audit != plan["pair_audit"] or len(pairs) != plan["pair_count"]
            or plan["query_count"] != 2 or plan["case_count"] != 2
            or sample["input_bindings"] != plan["input_bindings_recorded_not_read"]
            or baseline.length_summary(audit) != plan["length_audit"]):
        raise ValueError("frozen input or tokenization differs")
    if not baseline.torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    gpu_samples = baseline.ensure_gpu_ready()
    baseline.deadline_check(deadline)
    baseline.torch.cuda.init()
    baseline.torch.cuda.reset_peak_memory_stats(0)
    model = baseline.AutoModelForSequenceClassification.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=baseline.torch.float16, attn_implementation="sdpa").to("cuda:0")
    scores, compute = baseline.infer_pairs(
        encoded, tokenizer, model, device="cuda:0", deadline=deadline)
    baseline.torch.cuda.synchronize(0)
    if len(scores) != len(pairs) or any(type(x) not in (float, int) or not math.isfinite(x) for x in scores):
        raise ValueError("score coverage or finiteness differs")
    compute.update(peak_allocated_bytes=baseline.torch.cuda.max_memory_allocated(0),
                   peak_reserved_bytes=baseline.torch.cuda.max_memory_reserved(0))
    del model
    if (baseline.digest(PLAN / "plan.json") != plan_hash
            or baseline.digest(PLAN / "plan_manifest.json") != seal_hash
            or sources() != plan["input_sha256"]):
        raise ValueError("frozen files changed during inference")
    baseline.deadline_check(deadline)
    output = {"schema": SCHEMA, "plan_sha256": plan_hash,
              "pair_scores": [row | {"raw_logit": score} for row, score in zip(audit, scores)]}
    write(RUN / "scores.json", output)
    summary = {"schema": SCHEMA, "status": "completed", "contract": CONTRACT,
               "case_count": 2, "query_count": 2, "pair_count": len(pairs),
               "plan_sha256": plan_hash, "plan_manifest_sha256": seal_hash,
               "scores_sha256": baseline.digest(RUN / "scores.json"),
               "length_audit": plan["length_audit"], "compute": compute,
               "gpu_admission_samples": gpu_samples, "runtime": runtime(),
               "gpu_name": baseline.torch.cuda.get_device_name(0),
               "gpu_capability": list(baseline.torch.cuda.get_device_capability(0)),
               "model_inference_performed": True, "api_calls": 0, "training_performed": False,
               "references_read": False, "provenance_target_files_read": False,
               "packing_performed": False, "evaluation_performed": False,
               "answer_generation_performed": False, "raw_text_in_summary": False,
               "elapsed_seconds": CONTRACT["max_seconds"] - (deadline - time.monotonic())}
    write(RUN / "summary.json", summary)
    write(RUN / "run_manifest.json", {"schema": SCHEMA, "status": "completed",
          "plan_sha256": plan_hash, "plan_manifest_sha256": seal_hash,
          "scores_sha256": summary["scores_sha256"],
          "summary_sha256": baseline.digest(RUN / "summary.json")})
    return {"status": "completed", "queries": 2, "pairs": len(pairs), "api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    output = RUN if args.run else PLAN
    if output.resolve().parent != PHASE.resolve() or output.exists():
        raise SystemExit("requires a fresh fixed phase output directory")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = (run if args.run else prepare)(started + CONTRACT["max_seconds"])
        result["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(result))
    except Exception as exc:
        failure = {"schema": SCHEMA, "status": "failed", "stage": "run" if args.run else "prepare",
                   "error_class": type(exc).__name__, "elapsed_seconds": time.monotonic() - started,
                   "api_calls": 0, "automatic_retries": 0, "cpu_fallback": False,
                   "experimental_scores_complete": False}
        write(output / "failure.json", failure)
        print(json.dumps(failure))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
