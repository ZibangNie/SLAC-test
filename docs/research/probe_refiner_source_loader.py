"""Fixed two-document source persistence through the real metadata-only build."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/research-foundation/offline-20261004"
STAGE = BASE / "refiner-source-loader-01"
INPUTS = {
    "bounded_inputs": (BASE / "refiner-granularity-probe-01/bounded_inputs.json", "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"),
    "exports": (BASE / "refiner-document-source-export-01/run-01/document_exports.json", "e28538bf86ed10b452257c887067c21d5fdd1588baef3d964ac24229de87df87"),
    "prior_requests": (BASE / "refiner-jev-text-contract-01/run-01/coverage_requests.json", "087877e09d4f9f7792e158d31d05b0538807324d0790b2c3a655dc1b179ad374"),
}
SOURCES = {
    "docs/research/probe_refiner_source_loader.py",
    "SLAC/retrieval/dataio/source_records.py", "SLAC/retrieval/dataio/readers.py",
    "SLAC/retrieval/dataio/writers.py", "SLAC/retrieval/configs/loader.py",
    "SLAC/retrieval/run/run_build_index.py", "SLAC/retrieval/run/run_retrieve.py",
    "SLAC/retrieval/run/run_retrieval_pipeline.py",
    "SLAC/retrieval/index/build_lookup_tables.py", "SLAC/retrieval/preprocess/anchor_fields.py",
    "SLAC/retrieval/utils/text_utils.py", "SLAC/retrieval/schemas/records.py",
    "SLAC/retrieval/schemas/validation.py", "SLAC/retrieval/decision/refiner_bridge.py",
    "SLAC/retrieval/decision/conditional.py",
    "SLAC/refiner/pipeline/assemble/source_document_view.py",
    "SLAC/refiner/pipeline/assemble/source_coverage.py",
    "tests/research/test_source_records.py", "tests/research/test_source_reader_pipeline.py",
    "tests/research/test_retrieval_source_entry.py",
}
QUERY = "Offline text-identity fixture; no model judgment is requested."
CONFIG = {"arm": "standalone", "endpoint_id": "offline-only", "model_id": "offline-identity-model",
          "expected_response_model": "offline-identity-response",
          "counter_version": "local-bge-m3-specials-no-truncation-v1", "max_tokens": 1024}
MODES = (("raw_b0_source", "raw_b0_chunks", 83), ("rule_projected_source", "rule_projected_chunks", 56))
BLOCKED = ("torch", "transformers", "faiss", "sentence_transformers",
           "SLAC.retrieval.index.embedder", "SLAC.retrieval.index.build_chunk_dense",
           "SLAC.retrieval.index.build_leaf_dense", "SLAC.retrieval.index.build_anchor_lexical")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write(path, obj):
    with Path(path).open("xb") as handle:
        handle.write((json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in BLOCKED):
            raise RuntimeError("model/index runtime forbidden in this probe")
        return None


def run(protocol_sha, output):
    protocol_path = STAGE / "protocol.json"
    require(digest(protocol_path.read_bytes()) == protocol_sha, "protocol drift")
    protocol = read(protocol_path)
    require(protocol["schema"] == "slac-refiner-source-loader-protocol-v1"
            and set(protocol["source_sha256"]) == SOURCES and set(protocol["inputs"]) == set(INPUTS),
            "protocol inventory differs")
    require(protocol["synthetic_query"] == QUERY and protocol["request_config"] == CONFIG, "request config differs")
    bindings = {str(protocol_path): protocol_sha}
    loaded = {}
    for name, (path, expected) in INPUTS.items():
        require(Path(protocol["inputs"][name]["path"]).resolve() == path.resolve()
                and protocol["inputs"][name]["sha256"] == expected
                and digest(path.read_bytes()) == expected, "fixed input drift")
        loaded[name] = read(path)
        bindings[str(path)] = expected
    for name, expected in protocol["source_sha256"].items():
        require(digest((ROOT / name).read_bytes()) == expected, "source drift")
        bindings[str(ROOT / name)] = expected
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(), "fresh stage run required")
    output.mkdir(parents=True)
    write(output / "started.json", {"bindings": bindings})
    require(not any(n == p or n.startswith(p + ".") for n in sys.modules for p in BLOCKED), "forbidden runtime already imported")
    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in this probe")
    socket.create_connection = denied
    socket.socket.connect = denied
    sys.meta_path.insert(0, NoModels())
    sys.path.insert(0, str(ROOT))
    from SLAC.retrieval.run import run_build_index
    from SLAC.retrieval.dataio.readers import load_chunk_records, load_leaf_records
    from SLAC.retrieval.dataio.source_records import load_source_indexes, source_snapshot_from_chunk_record
    from SLAC.retrieval.dataio.writers import write_jsonl
    from SLAC.retrieval.utils.text_utils import normalize_text_basic
    from SLAC.retrieval.decision.refiner_bridge import build_refiner_source_request

    data, exports, prior = loaded["bounded_inputs"], loaded["exports"], loaded["prior_requests"]
    ids = [e["doc_id"] for e in exports]
    require(len(ids) == len(set(ids)) == 2 and [p["doc_id"] for p in prior] == ids
            and [len(data["documents"][d]) for d in ids] == [60, 23], "fixed document scope differs")
    registry = {"schema": "slac-refiner-source-indexes-v1", "documents": []}
    for e in exports:
        doc_id = e["doc_id"]
        mapping = {m["native_unit_id"]: m for m in data["unit_mapping"] if m["doc_id"] == doc_id}
        units = [{"native_unit_id": u["unit_id"], "source_span": mapping[u["unit_id"]]["raw_native_char_span"],
                  "source_text": u["native_text"]} for u in data["documents"][doc_id]]
        registry["documents"].append({"view": e["view"], "native_units": units})
    registry_path = output / "source_indexes.json"
    write(registry_path, registry)
    counts = {}
    for p in prior:
        for _, group, _ in MODES:
            for row in p["requests"][group]:
                req = row["source_request"]
                text, count = req["rendered_evidence"], req["binding"]["evidence_tokens"]
                require(text not in counts or counts[text] == count, "inherited exact counter conflict")
                counts[text] = count
    def count_known(text):
        require(text in counts, "rendered evidence unseen in frozen verified counts")
        return counts[text]

    stats, detail = [], []
    for mode, group, planned_chunks in MODES:
        inputs = output / mode
        inputs.mkdir()
        rows = {name: [r for e in exports for r in e["exports"][mode][name]]
                for name in ("refined_chunks", "leaf_records")}
        rows["doc_catalog"] = [e["exports"][mode]["doc_catalog"] for e in exports]
        for name, values in rows.items():
            write_jsonl(inputs / (name + ".jsonl"), values)
        build = inputs / "build"
        argv = sys.argv
        sys.argv = ["run_build_index", "--refined_chunks_jsonl", str(inputs / "refined_chunks.jsonl"),
                    "--leaf_records_jsonl", str(inputs / "leaf_records.jsonl"), "--doc_catalog_jsonl",
                    str(inputs / "doc_catalog.jsonl"), "--output_dir", str(build),
                    "--source_indexes_json", str(registry_path), "--metadata_only"]
        try:
            run_build_index.main()
        finally:
            sys.argv = argv
        summary = read(build / "summaries/run_build_index_summary.json")
        require(summary["indexes_built"] is False and summary["stage"] == "metadata_only"
                and summary["num_chunks"] == planned_chunks and summary["num_leaves"] == 251
                and summary["num_docs"] == 2 and not list((build / "index").iterdir()), "metadata build scope differs")
        require((build / "meta/refiner_source_indexes.json").read_bytes() == registry_path.read_bytes(), "registry copy drift")
        for name in rows:
            require((build / "data" / (name + ".jsonl")).read_bytes() == (inputs / (name + ".jsonl")).read_bytes(), "input copy drift")
        indexes = load_source_indexes(build / "meta/refiner_source_indexes.json")
        chunks = load_chunk_records(build / "meta/chunk_lookup.jsonl", source_indexes=indexes)
        leaves = load_leaf_records(build / "meta/leaf_lookup.jsonl", source_indexes=indexes)
        require(len(chunks) == planned_chunks and len(leaves) == 251, "reload count differs")
        changed = {}
        hashes = {}
        for name, records, raw in (("chunks", chunks, rows["refined_chunks"]), ("leaves", leaves, rows["leaf_records"])):
            changed[name], hashes[name] = 0, []
            for record, source in zip(records, raw, strict=True):
                require(record.text == normalize_text_basic(source["text"], keep_newlines=True)
                        and record.meta["refiner_source"]["text"] == source["text"], "source/normalization drift")
                changed[name] += record.text != source["text"]
                hashes[name].append({"source_sha256": digest(source["text"].encode()),
                                     "retrieval_sha256": digest(record.text.encode())})
        chunk_map = {c.chunk_id: c for c in chunks}
        for leaf in leaves:
            owner = chunk_map[leaf.owner_chunk_id]
            require(owner.doc_id == leaf.doc_id and owner.atom_start <= leaf.atom_start < leaf.atom_end <= owner.atom_end,
                    "leaf owner geometry differs")
        expected = [r["source_request"] for p in prior for r in p["requests"][group]]
        requests = []
        for chunk, old in zip(chunks, expected, strict=True):
            snapshot = source_snapshot_from_chunk_record(chunk, indexes)
            wrapped = build_refiner_source_request(QUERY, (), snapshot, token_counter=count_known, **CONFIG)
            req = wrapped.request
            actual = {"payload": req.payload(), "binding": json.loads(req.binding_bytes), "cache_key": req.cache_key,
                      "rendered_evidence": req.rendered_evidence, "provenance": wrapped.provenance_receipt()}
            require(actual == old, "reloaded source request differs from frozen request")
            requests.append(actual)
        stats.append({"mode": mode, "chunks": len(chunks), "leaves": len(leaves), "documents": 2,
                      "retrieval_text_changed": changed, "exact_source_requests": len(requests), "indexes_built": False})
        detail.append({"mode": mode, "hashes": hashes, "requests": requests})
    for path, expected in bindings.items():
        require(digest(Path(path).read_bytes()) == expected, "post-run input/source drift")
    write(output / "roundtrip_requests.json", detail)
    artifact_hashes = {str(p.relative_to(output)): digest(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}
    report = {"schema": "slac-refiner-source-loader-result-v1", "status": "completed", "bindings": bindings,
              "scope": protocol["scope"], "modes": stats, "artifact_sha256": artifact_hashes,
              "totals": {"unique_documents": 2, "chunk_rows": 139, "leaf_rows": 502, "exact_source_requests": 139,
                         "api_calls": 0, "fresh_tokenizer_calls": 0, "indexes_built": 0},
              "counter": "Inherited verified whole-render counts, accepted only for byte-identical saved strings; no new tokenization.",
              "limits": protocol["limits"]}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=STAGE / "run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"], "modes": result["modes"]}))
