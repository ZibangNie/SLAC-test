"""Select at most six input-only pilot queries in the fixed three documents.

The existing prepared JSON is parsed to index IDs. Only selected query/source
records are validated or retained. No QA sidecar, response or gold is opened.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/research-foundation"
OUT = BASE / "offline-20261004/definition-pack-opportunity-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    OUT.mkdir(exist_ok=False)
    source = BASE / "qasper-relation-prepared-01/prepared.json"
    manifest = BASE / "qasper-relation-prepared-01/manifest.json"
    prior = BASE / "offline-20261004/native-definition-sample-01/selected_sources.json"
    links = BASE / "offline-20261004/native-definition-sample-01/extraction.json"
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (source, manifest, prior, links)}
    if sha(source) != json.loads(manifest.read_bytes())["prepared_sha256"]:
        raise ValueError("prepared input hash changed")
    sample = json.loads(prior.read_bytes())
    prepared = json.loads(source.read_bytes())
    docs = [row["doc_id"] for row in sample["documents"]]
    selected = [q for doc in docs for q in [q for q in prepared["queries"] if q["doc_id"] == doc][:2]]
    if not 1 <= len(selected) <= 6 or any(len(q["candidate_ids"]) > 16 for q in selected):
        raise ValueError("selected query/candidate cap exceeded")
    plan = {"schema": "native-definition-pack-sample-v1", "source_sha256": source_hashes,
            "documents": docs, "query_ids": [[q["doc_id"], q["question_id"]] for q in selected],
            "selection": "first up to two input queries in original pilot prepared order per unchanged source document",
            "reason_for_pilot": "all three original source documents are absent from the 77-query extended cohort",
            "max_queries": 6, "max_original_pairs": 96, "no_substitutions": True,
            "pack_contract": {"k": 3, "BGE_whole_render_tokens": 1024,
                              "methods": ["dense", "local_bge_reranker"]},
            "opportunity": "selected mention, omitted definition; no answer relevance assumed",
            "controls": "original candidate membership, immediate native neighbors of any selected unit, other baseline pack",
            "budget": "report token-only append separately from k3 feasibility; enumerate single replacements retaining every triggering selected mention",
            "query_alignment": "exact ASCII-boundary acronym in query is a descriptive lexical indicator only",
            "no_api": True, "no_gold_or_response_reads": True, "not_independent_confirmation": True}
    write("sample_plan.json", plan)
    documents = {doc: prepared["documents"][doc] for doc in docs}
    blocks = {(b["doc_id"], b["native_unit_id"]): b for b in sample["blocks"] if b["retrievable"]}
    for doc, units in documents.items():
        if len({u["unit_id"] for u in units}) != len(units):
            raise ValueError("duplicate selected source IDs")
        for unit in units:
            block = blocks[doc, unit["unit_id"]]
            if unit["native_text"] != block["raw_native_text"]:
                raise ValueError("prepared native text differs from sampled source")
            begin, end = block["canonical_char_span"]
            document = next(d for d in sample["documents"] if d["doc_id"] == doc)
            if unit["text"] != document["canonical_text"][begin:end]:
                raise ValueError("prepared retrieval text differs from sampled source")
        if [u["order"] for u in units] != list(range(len(units))):
            raise ValueError("old native unit order is not contiguous")
    for query in selected:
        if set(query) != {"doc_id", "family_id", "question_id", "query", "candidate_ids", "seed_ids", "ranked_ids"}:
            raise ValueError("unexpected query fields; do not retain labels")
        if set(query["ranked_ids"]) != set(query["candidate_ids"]):
            raise ValueError("candidate/ranking mismatch")
    write("selected_inputs.json", {"queries": selected, "documents": documents,
                                   "source_sha256": source_hashes, "plan_sha256": sha(OUT / "sample_plan.json")})
    print(json.dumps({"documents": len(docs), "queries": len(selected),
                      "pairs": sum(len(q["candidate_ids"]) for q in selected),
                      "queries_per_document": [sum(q["doc_id"] == d for q in selected) for d in docs],
                      "input_sha256": sha(OUT / "selected_inputs.json"), "api_calls": 0}))


if __name__ == "__main__":
    main()
