"""Replay six cached rankings and tokenize fixed whole-pack replacements locally.

No model inference, source relation extraction, gold, answer or credential reads.
Retrieval text is displayed/counted; native text supplies deduplication and hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/research-foundation"
PHASE = BASE / "offline-20261004"
INPUT = PHASE / "definition-pack-opportunity-01/selected_inputs.json"
SCORES = PHASE / "definition-sample-reranker-run-01/scores.json"
OPPORTUNITIES = PHASE / "definition-pack-opportunity-01/opportunities.json"
PINNED = {INPUT: "ee8f41fee2d4de983a0a9de4c2f91eac4a8392b64dbcb6f06353ede7cd7495d1",
          SCORES: "634834133be80cb2ad1a18bc02e840e9fe2807c2bc453ce02cd2f981fcf1ff9b",
          OPPORTUNITIES: "e9f8164caa74012bf399837d2a84815207810c9064aeea736372bc990e8b3859"}
CONFIG = BASE / "qasper-reranker-02/experiment_config.json"
MANIFEST = BASE / "qasper-relation-prepared-01/manifest.json"
PROTOCOL = ROOT / "docs/research/NATURAL_EXCHANGE_SAMPLE_PROTOCOL_20261004.md"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")
POLICY = {"queries": 6, "documents": 3, "k": 3, "whole_pack_token_cap": 1024,
          "candidate": "first saved BGE-ranked unit outside S with native text distinct from every S unit",
          "removed": "worst saved BGE rank in S", "no_substitution_after_failure": True,
          "renderer": "[unit_id]\\nretrieval_text, double-newline joined in native source order",
          "tokenizer": "prior BGE tokenizer, add_special_tokens=True, truncation=False",
          "ranking_replayed": True, "new_model_inference": False, "api_calls": 0}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def render(units, selected):
    return "\n\n".join(f"[{uid}]\n{units[uid]['text']}" for uid in sorted(selected, key=lambda uid: units[uid]["order"]))


def propose(units, ranking, selected, count):
    """Choose c/r once by saved rank; count whole S/T without adaptive fallback."""
    require(len(ranking) == len(set(ranking)) and set(ranking) <= units.keys(), "invalid ranking identity")
    require(len(selected) == len(set(selected)) and set(selected) <= set(ranking), "invalid selected identity")
    require(len({units[u]["order"] for u in ranking}) == len(ranking), "duplicate source order")
    require(len({units[u]["native_text"] for u in selected}) == len(selected), "duplicate selected native text")
    selected = sorted(selected, key=lambda uid: units[uid]["order"])
    seen = {units[uid]["native_text"] for uid in selected}
    candidate = next((uid for uid in ranking if uid not in selected and units[uid]["native_text"] not in seen), None)
    removed = max(selected, key=ranking.index) if selected else None
    require(all(isinstance(units[uid][field], str) and units[uid][field].strip()
                for uid in selected + ([] if candidate is None else [candidate])
                for field in ("text", "native_text")), "invalid chosen source text")
    proposed = sorted([u for u in selected if u != removed] + [candidate], key=lambda uid: units[uid]["order"]) if candidate and removed else None
    original_tokens, proposed_tokens = count(selected), None if proposed is None else count(proposed)
    require(type(original_tokens) is int and original_tokens >= 0
            and (proposed_tokens is None or type(proposed_tokens) is int and proposed_tokens >= 0), "invalid whole-pack count")
    status = ("no_selected_unit" if removed is None else "no_candidate" if candidate is None else
              "feasible" if len(selected) <= 3 and len(proposed) <= 3 and max(original_tokens, proposed_tokens) <= 1024 else "infeasible")
    return {"original_ids": selected, "candidate_id": candidate, "removed_id": removed,
            "proposed_ids": proposed, "original_tokens": original_tokens,
            "proposed_tokens": proposed_tokens, "status": status}


def project_sample(inputs, scores, opportunities, tokenizer):
    """Only six saved query records and their bounded three-document input file."""
    queries, documents = inputs["queries"], inputs["documents"]
    require(len(queries) == 6 and len(documents) == 3, "fixed sample size changed")
    key = lambda row: (row["doc_id"], row["question_id"])
    keys = [key(q) for q in queries]
    rankings = {key(r): r["ranked_ids"] for r in scores["rankings"]}
    saved = {key(r): r for r in opportunities["records"]}
    require(len(set(keys)) == len(rankings) == len(scores["rankings"]) == len(saved) == len(opportunities["records"]) == 6
            and set(keys) == set(rankings) == set(saved) and {q["doc_id"] for q in queries} == set(documents), "sample identity mismatch")
    proposals, packet = [], []
    for ordinal, query in enumerate(queries, 1):
        source = documents[query["doc_id"]]
        units = {u["unit_id"]: u for u in source}
        require(len(units) == len(source), "duplicate native unit ID")
        ranking, previous = rankings[key(query)], saved[key(query)]
        baseline = previous["methods"]["local_bge_reranker"]
        pool = query["candidate_ids"]
        require(1 <= len(pool) <= 16 and len(pool) == len(set(pool)) == len(ranking)
                and set(pool) == set(ranking) == set(query["ranked_ids"])
                and ranking == baseline["ranked_ids"] and pool == previous["candidate_ids"]
                and query["query"] == previous["query"], "pool/ranking/query drift")
        for uid in pool:
            require(uid in units and type(units[uid]["order"]) is int and units[uid]["order"] >= 0, "invalid candidate source identity/order")
        selected = baseline["selected_ids"]
        count = lambda ids: len(tokenizer.encode(render(units, ids), add_special_tokens=True, truncation=False)) if ids else 0
        result = propose(units, ranking, selected, count)
        require(selected == result["original_ids"] and result["original_tokens"] == baseline["evidence_tokens"]
                and sha(render(units, selected).encode("utf-8")) == baseline["pack_sha256"], "saved baseline pack or tokens changed")
        def visible(uid):
            if uid is None:
                return None
            unit = units[uid]
            return {"unit_id": uid, "source_order": unit["order"], "text": unit["text"],
                    "retrieval_text_sha256": sha(unit["text"].encode("utf-8")),
                    "native_text_sha256": sha(unit["native_text"].encode("utf-8"))}
        review = {"ordinal": ordinal, "doc_id": query["doc_id"], "question_id": query["question_id"], "query": query["query"],
                  "original_pack": [visible(uid) for uid in result["original_ids"]],
                  "candidate": visible(result["candidate_id"]), "removed": visible(result["removed_id"]),
                  "proposed_pack": None if result["proposed_ids"] is None else [visible(uid) for uid in result["proposed_ids"]]}
        packet.append(review)
        proposals.append(review | result | {"saved_bge_ranking": ranking,
                         "original_render_sha256": sha(render(units, result["original_ids"]).encode("utf-8")),
                         "proposed_render_sha256": None if result["proposed_ids"] is None else sha(render(units, result["proposed_ids"]).encode("utf-8"))})
    return proposals, packet


def prepare(output):
    paths = [*PINNED, CONFIG, MANIFEST, PROTOCOL, Path(__file__).resolve(),
             ROOT / "tests/research/test_natural_exchange_sample.py",
             ROOT / "docs/research/analyze_definition_pack_opportunity.py"]
    blobs = {path: path.read_bytes() for path in paths}
    require(all(sha(blobs[path]) == expected for path, expected in PINNED.items()), "pinned cache changed")
    tokenizer_path = Path(json.loads(blobs[CONFIG])["bge_tokenizer"])
    manifest = json.loads(blobs[MANIFEST])["input_sha256"]
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        path = tokenizer_path / name
        digest = sha(path.read_bytes())
        require(manifest.get(str(path)) == digest, "prior tokenizer identity changed")
        tokenizer_hashes[str(path)] = digest
    source_hashes = {str(path.relative_to(ROOT)): sha(value) for path, value in blobs.items()}
    output.mkdir(parents=True, exist_ok=False)
    # Freeze policy and source commitments BEFORE parsing any source/query/score/opportunity content.
    write(output / "plan.json", {"schema": "slac-natural-exchange-sample-plan-v1", "policy": POLICY,
          "source_sha256": source_hashes, "tokenizer_sha256": tokenizer_hashes, "source_projection_performed": False})
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    proposals, packet = project_sample(json.loads(blobs[INPUT]), json.loads(blobs[SCORES]), json.loads(blobs[OPPORTUNITIES]), tokenizer)
    require(all(sha(path.read_bytes()) == sha(value) for path, value in blobs.items())
            and all(sha(Path(path).read_bytes()) == digest for path, digest in tokenizer_hashes.items()), "source changed during preparation")
    write(output / "proposals.json", proposals)
    write(output / "review_packet.json", {"schema": "slac-natural-exchange-review-inputs-v1", "cases": packet})
    summary = {"queries": 6, "documents": 3, "status_counts": dict(Counter(row["status"] for row in proposals)),
               "proposals": sum(row["proposed_ids"] is not None for row in proposals), "ranking_replayed_from_cache": True,
               "new_whole_pack_tokenization": True, "new_model_inference": False, "api_calls": 0,
               "gold_answer_or_relation_files_read": False, "quality_or_acceptance_measured": False,
               "artifact_sha256": {name: sha((output / name).read_bytes()) for name in ("plan.json", "proposals.json", "review_packet.json")}}
    write(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PHASE / "natural-exchange-sample-01")
    print(json.dumps(prepare(parser.parse_args().output_dir), sort_keys=True))
