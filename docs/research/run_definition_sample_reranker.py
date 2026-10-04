"""One frozen, local-only BGE pass over at most six questions and 96 pairs.

Default/--prepare hashes local model files and tokenizes; only --run loads weights.
Never calls the old runner's prepare/run, reads annotations, or adds candidates.
The parent must enforce a 300-second process deadline; this child checks 180s.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

# Set before importing the shared runner or any Hugging Face package.
for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import run_qasper_reranker_baseline as baseline


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "artifacts/research-foundation/offline-20261004"
INPUT = PHASE / "definition-pack-opportunity-01/selected_inputs.json"
INPUT_SHA256 = "ee8f41fee2d4de983a0a9de4c2f91eac4a8392b64dbcb6f06353ede7cd7495d1"
PRIOR_CONFIG = ROOT / "artifacts/research-foundation/qasper-reranker-02/experiment_config.json"
MODEL = Path("C:/Environment/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots") / baseline.REVISION
PLAN = PHASE / "definition-sample-reranker-plan-01"
RUN = PHASE / "definition-sample-reranker-run-01"
CONTRACT = {
    "model_id": baseline.MODEL_ID, "revision": baseline.REVISION,
    "maximum_queries": 6, "maximum_candidates_per_query": 16, "maximum_pairs": 96,
    "max_pair_tokens": 1024, "truncation": False, "dtype": "float16",
    "device": "cuda:0", "attention": "sdpa", "microbatch": 4,
    "max_seconds": 180, "parent_deadline_seconds": 300,
    "automatic_retries": 0, "cpu_fallback": False, "downloads": False,
    "api_calls": 0, "selection_uses_gold": False,
    "input": "raw query and selected canonical text as an ordered tokenizer pair",
    "score": "finite single raw classification logit; not calibrated support probability",
    "tie_break": "original dense rank then native source order",
    "gpu_admission": dict(baseline.CONTRACT["gpu_admission"]),
}
SCHEMA = "slac-definition-sample-reranker-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pairs_from_sample(sample):
    """Pure bounded projection: no annotations, other files or candidate expansion."""
    queries, documents = sample["queries"], sample["documents"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= 6:
        raise ValueError("sample query cap")
    if not isinstance(documents, dict) or not 1 <= len(documents) <= 6:
        raise ValueError("sample document cap")
    lookup = {}
    for doc_id, units in documents.items():
        if not isinstance(units, list) or not 1 <= len(units) <= 300:
            raise ValueError("sample unit cap")
        lookup[doc_id] = {}
        for unit in units:
            uid = unit["unit_id"]
            if (not isinstance(uid, str) or not uid or uid in lookup[doc_id]
                    or not isinstance(unit["text"], str) or not unit["text"].strip()
                    or len(unit["text"]) > 100000 or type(unit["order"]) is not int):
                raise ValueError("sample unit identity or text invalid")
            lookup[doc_id][uid] = unit
    pairs, seen = [], set()
    for query in queries:
        doc, qid = query["doc_id"], query["question_id"]
        ids, ranks, text = query["candidate_ids"], query["ranked_ids"], query["query"]
        if (not isinstance(doc, str) or not isinstance(qid, str) or not qid
                or (doc, qid) in seen or doc not in lookup
                or not isinstance(text, str) or not text.strip() or len(text) > 10000
                or not isinstance(ids, list) or not 1 <= len(ids) <= 16
                or any(not isinstance(uid, str) for uid in ids)
                or len(set(ids)) != len(ids) or not isinstance(ranks, list)
                or len(ranks) != len(ids) or set(ranks) != set(ids)):
            raise ValueError("sample query or candidate identity invalid")
        seen.add((doc, qid))
        for uid in ids:
            unit = lookup[doc][uid]
            task_id = baseline.object_hash([doc, qid, uid])
            pairs.append({"task_id": task_id, "doc_id": doc, "question_id": qid,
                          "unit_id": uid, "query": text, "passage": unit["text"]})
    if len(pairs) > 96:
        raise ValueError("sample pair cap")
    return pairs


def sources():
    """Only statically named model/source files; never follow sample provenance paths."""
    if baseline.digest(INPUT) != INPUT_SHA256:
        raise ValueError("frozen selected input identity differs")
    old = read(PRIOR_CONFIG)
    if (old["contract"] != baseline.CONTRACT
            or Path(old["model_dir"]).resolve() != MODEL.resolve()):
        raise ValueError("pinned model configuration differs")
    hashes = baseline.verify_model(MODEL)
    for path, actual in hashes.items():
        if old["input_sha256"].get(path) != actual:
            raise ValueError("model differs from prior pinned baseline")
    code = Path(__file__).resolve().parent
    for path in (INPUT, PRIOR_CONFIG, Path(__file__).resolve(),
                 *(code / name for name in baseline.CODE_FILES)):
        hashes[str(path)] = baseline.digest(path)
    return hashes


def encode(pairs, deadline):
    tokenizer = baseline.AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False, use_fast=True)
    encoded, audit = baseline.encode_pairs(pairs, tokenizer, deadline=deadline)
    for pair, row in zip(pairs, audit):
        row.update(query_sha256=text_hash(pair["query"]),
                   passage_sha256=text_hash(pair["passage"]))
    return tokenizer, encoded, audit


def prepare(deadline):
    hashes = sources()
    sample = read(INPUT)
    pairs = pairs_from_sample(sample)
    _, _, audit = encode(pairs, deadline)
    baseline.deadline_check(deadline)
    plan = {"schema": SCHEMA, "status": "prepared_no_model_inference",
            "contract": CONTRACT, "input_sha256": hashes,
            "sample_provenance_sha256": sample.get("source_sha256", {}),
            "sample_provenance_files_read": False,
            "sample_plan_sha256": sample.get("plan_sha256"),
            "query_count": len(sample["queries"]), "pair_count": len(pairs),
            "pair_audit": audit, "length_audit": baseline.length_summary(audit),
            "model_inference_performed": False, "api_calls": 0}
    write(PLAN / "plan.json", plan)
    write(PLAN / "plan_manifest.json", {"schema": SCHEMA,
          "plan_sha256": baseline.digest(PLAN / "plan.json")})
    return {"status": plan["status"], "queries": plan["query_count"],
            "pairs": len(pairs), "api_calls": 0}


def rankings(sample, pairs, scores):
    grouped = defaultdict(dict)
    for pair, score in zip(pairs, scores):
        grouped[pair["doc_id"], pair["question_id"]][pair["unit_id"]] = score
    result = []
    for query in sample["queries"]:
        doc, qid = query["doc_id"], query["question_id"]
        values = grouped[doc, qid]
        dense = {uid: index for index, uid in enumerate(query["ranked_ids"])}
        orders = {row["unit_id"]: row["order"] for row in sample["documents"][doc]}
        ranked = sorted(values, key=lambda uid: (-values[uid], dense[uid], orders[uid]))
        result.append({"doc_id": doc, "question_id": qid, "ranked_ids": ranked})
    return result


def run(deadline):
    seal = read(PLAN / "plan_manifest.json")
    plan_hash = baseline.digest(PLAN / "plan.json")
    if seal != {"schema": SCHEMA, "plan_sha256": plan_hash}:
        raise ValueError("plan seal differs")
    plan = read(PLAN / "plan.json")
    if (plan["schema"] != SCHEMA or plan["contract"] != CONTRACT
            or plan["status"] != "prepared_no_model_inference"):
        raise ValueError("plan contract differs")
    if sources() != plan["input_sha256"]:
        raise ValueError("frozen source or model hashes differ")
    sample = read(INPUT)
    pairs = pairs_from_sample(sample)
    tokenizer, encoded, audit = encode(pairs, deadline)
    if (audit != plan["pair_audit"] or len(pairs) != plan["pair_count"]
            or len(sample["queries"]) != plan["query_count"]
            or baseline.length_summary(audit) != plan["length_audit"]):
        raise ValueError("frozen pairs or tokenization differ")
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
    if baseline.digest(PLAN / "plan.json") != plan_hash:
        raise ValueError("plan changed during inference")
    # Repeat only statically scoped integrity checks, never provenance/full-pool reads.
    if sources() != plan["input_sha256"]:
        raise ValueError("input identity changed during inference")
    baseline.deadline_check(deadline)
    output = {"schema": SCHEMA, "plan_sha256": plan_hash,
              "pair_scores": [row | {"raw_logit": score} for row, score in zip(audit, scores)],
              "rankings": rankings(sample, pairs, scores)}
    write(RUN / "scores.json", output)
    summary = {"schema": SCHEMA, "status": "completed", "contract": CONTRACT,
               "query_count": len(sample["queries"]), "pair_count": len(pairs),
               "plan_sha256": plan_hash, "scores_sha256": baseline.digest(RUN / "scores.json"),
               "length_audit": plan["length_audit"], "compute": compute,
               "gpu_admission_samples": gpu_samples,
               "torch": str(baseline.torch.__version__),
               "transformers": baseline.transformers.__version__,
               "model_inference_performed": True, "api_calls": 0,
               "annotations_read": False, "answer_generation_performed": False,
               "ranking_is_not_answer_quality_measurement": True}
    write(RUN / "summary.json", summary)
    return {"status": "completed", "queries": len(sample["queries"]),
            "pairs": len(pairs), "api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    output = RUN if args.run else PLAN
    # Exclusive directory makes both successful and failed invocations non-resumable.
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = (run if args.run else prepare)(started + CONTRACT["max_seconds"])
        result["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(result))
    except Exception as exc:
        failure = {"status": "failed", "error_class": type(exc).__name__,
                   "stage": "run" if args.run else "prepare", "api_calls": 0,
                   "automatic_retries": 0, "cpu_fallback": False,
                   "elapsed_seconds": time.monotonic() - started}
        write(output / "failure.json", failure)
        print(json.dumps(failure))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
