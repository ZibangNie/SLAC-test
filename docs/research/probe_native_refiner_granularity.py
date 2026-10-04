"""Two frozen native documents through the real builder and an identity export.

This is a mechanical interface fixture: no refiner prediction, scores or quality
evaluation. Model weights and credentials are never loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SLAC.refiner.pipeline.assemble import build_refiner_input as builder
from SLAC.refiner.pipeline.assemble import export_refined_chunks as exporter

SCHEMA = "slac-native-refiner-granularity-probe-v1"
DIRECTORY = ROOT / "artifacts/research-foundation/offline-20261004/refiner-granularity-probe-01"
INPUT_SHA = "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"
UNIT_FIELDS = {"unit_id", "order", "kind", "start", "end", "text", "native_text"}
SOURCE_NAMES = {"docs/research/probe_native_refiner_granularity.py",
                "SLAC/refiner/pipeline/assemble/build_refiner_input.py",
                "SLAC/refiner/pipeline/assemble/export_refined_chunks.py",
                "SLAC/refiner/pipeline/segment/chunk0_adapter.py",
                "SLAC/refiner/slac_refiner/models/atom_encoder.py"}
TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "sentencepiece.bpe.model"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def text_difference(left, right):
    """Byte/normalization facts only; no semantic-equivalence inference."""
    whitespace = lambda value: " ".join(value.split())
    exact = left.encode("utf-8") == right.encode("utf-8")
    whitespace_equal = whitespace(left) == whitespace(right)
    light_equal = whitespace(builder.normalize_text_light(left)) == whitespace(right)
    category = ("exact" if exact else "whitespace_only" if whitespace_equal
                else "matches_builder_light_normalization" if light_equal else "other_change")
    return {"left_sha256": sha(left.encode()), "right_sha256": sha(right.encode()),
            "left_utf8_bytes": len(left.encode()), "right_utf8_bytes": len(right.encode()),
            "byte_equal": exact, "whitespace_normalized_equal": whitespace_equal,
            "builder_light_normalized_equal": light_equal, "category": category}


def probe_document(doc_id, units, token_count, native_leaves):
    require(isinstance(doc_id, str) and doc_id and units, "nonempty document required")
    require(all(set(unit) == UNIT_FIELDS for unit in units), "native unit schema differs")
    require([unit["order"] for unit in units] == list(range(len(units))), "native order changed")
    require(len({unit["unit_id"] for unit in units}) == len(units), "duplicate native unit")
    require(all(isinstance(unit["text"], str) and unit["text"].strip()
                and isinstance(unit["native_text"], str) for unit in units), "empty or invalid source text")
    leaves_by_unit = {leaf["meta"]["native_unit_id"]: leaf for leaf in native_leaves}
    require(len(leaves_by_unit) == len(native_leaves) == len(units)
            and set(leaves_by_unit) == {unit["unit_id"] for unit in units}, "native leaf identity differs")
    require(all(leaf["doc_id"] == doc_id for leaf in native_leaves), "native leaf document differs")
    # Native units are declared chunk0 seeds. Copy existing source path/depth;
    # no hierarchy, heading text, rule segmentation or predictor is invented.
    seeds = [{"unit_id": unit["order"], "source_unit_id": unit["unit_id"],
              "native_unit_id": unit["unit_id"], "text": unit["text"], "type": unit["kind"],
              "path": leaves_by_unit[unit["unit_id"]]["path"],
              "depth": leaves_by_unit[unit["unit_id"]]["depth"],
              "parent_id": None} for unit in units]
    config = builder.RefinerInputBuildConfig()
    record = builder.build_refiner_input_from_chunk0(doc_id, seeds, domain="qasper", cfg=config,
        meta={"probe_schema": SCHEMA, "seed_policy": "complete_native_units_in_frozen_order",
              "model_inference": False})
    builder.validate_refiner_input_record(record)
    atoms, spans = record["atoms"], record["unit2atom_span"]
    require(len(spans) == len(units) and [s["unit_id"] for s in spans] == list(range(len(units))),
            "builder dropped or reordered a native seed")
    previous = 0
    for span in spans:
        require(span["start_atom"] == previous < span["end_atom"], "unit atom coverage gap/overlap")
        previous = span["end_atom"]
    require(previous == len(atoms), "unit spans do not cover all atoms")
    gaps = [index for index, value in enumerate(record["b0"]) if value == 1]
    require(gaps == [s["end_atom"]-1 for s in spans[:-1]], "seed boundaries differ")
    fixture = {"candidate_id": "b0-identity-fixture", "candidate_type": "b0_identity_fixture",
               "teacher_ckpt": None, "prediction": {"b0_sparse": gaps, "b_pred_sparse": gaps}}
    exported = exporter.export_refined_chunks_from_candidate(record, fixture)
    chunks, leaves = exported["refined_chunks"], exported["leaf_records"]
    require(len(chunks) == len(units) and len(leaves) == len(atoms), "identity export count differs")
    require(exported["refined_boundary"]["b_pred_sparse"] == gaps, "identity boundaries differ")
    require([leaf["atom_index"] for leaf in leaves] == list(range(len(atoms))), "leaf atom order differs")
    require([leaf["text"] for leaf in leaves] == atoms, "leaf text differs from atoms")
    lengths = [token_count(atom) for atom in atoms]
    require(all(type(n) is int and n > 0 for n in lengths), "invalid full atom token count")
    rows = []
    for unit, span, chunk in zip(units, spans, chunks, strict=True):
        start, end = span["start_atom"], span["end_atom"]
        require((chunk["atom_start"], chunk["atom_end"]) == (start, end), "identity export merged or split a seed")
        require(all(leaf["owner_chunk_id"] == chunk["chunk_id"] for leaf in leaves[start:end]), "leaf owner differs")
        source_leaf = leaves_by_unit[unit["unit_id"]]
        require(chunk["path"] == source_leaf["path"] and chunk["depth"] == source_leaf["depth"],
                "identity fixture did not preserve source seed metadata")
        joined = " ".join(atoms[start:end])
        rows.append({"native_unit_id": unit["unit_id"], "native_order": unit["order"],
                     "canonical_char_span": [unit["start"], unit["end"]],
                     "raw_native_char_span": source_leaf["meta"]["raw_native_char_span"],
                     "native_block_id": source_leaf["meta"]["native_block_id"],
                     "atom_span": [start, end], "fixture_chunk_id": chunk["chunk_id"],
                     "atom_character_offsets_available": False,
                     "differences": {"native_to_retrieval": text_difference(unit["native_text"], unit["text"]),
                         "retrieval_to_atom_join": text_difference(unit["text"], joined),
                         "retrieval_to_export": text_difference(unit["text"], chunk["text"]),
                         "native_to_export": text_difference(unit["native_text"], chunk["text"]),
                         "atom_join_to_export": text_difference(joined, chunk["text"])},
                     "atom_token_lengths": lengths[start:end]})
    changes = {axis: dict(Counter(row["differences"][axis]["category"] for row in rows))
               for axis in rows[0]["differences"]}
    report = {"doc_id": doc_id, "source_units": len(units), "atoms": len(atoms),
              "fixture_chunks": len(chunks), "fixture_leaves": len(leaves),
              "unit_spans_contiguous_complete": True, "identity_boundaries_preserved": True,
              "builder_config": asdict(config), "builder_statistics": record["meta"]["stats"],
              "atom_encoder_default_max_length": 128, "max_atom_bge_tokens": max(lengths),
              "atoms_over_128": [{"atom_index": i, "tokens": n} for i, n in enumerate(lengths) if n > 128],
              "default_encoder_length_admissible": all(n <= 128 for n in lengths),
              "difference_counts": changes, "units": rows,
              "exporter_source_literals": sorted({item["source"] for item in chunks + leaves})}
    wrapped = {"schema": SCHEMA, "mode": "b0_identity_fixture", "model_inference": False,
               "warning": "Exporter source literals are hardcoded; they do not establish model prediction provenance.",
               "refiner_input": record, "authored_identity_candidate": fixture, "raw_export": exported}
    return report, wrapped


def render_research_pack(units):
    """Historical research renderer; this is not production pack_evidence."""
    return "\n\n".join(f"[{unit['unit_id']}]\n{unit['text']}" for unit in sorted(units, key=lambda unit: unit["order"]))


def compare_query_pack(query, units, wrapped, token_count):
    by_id = {unit["unit_id"]: unit for unit in units}
    chunks = wrapped["raw_export"]["refined_chunks"]
    mapped = {unit["unit_id"]: chunk for unit, chunk in zip(units, chunks, strict=True)}
    selected = query["selected_ids"]
    candidates = query["candidate_ids"]
    require(len(selected) == len(set(selected)) == 3 and set(selected) <= set(candidates) <= set(by_id),
            "frozen candidate/selected identity differs")
    require(len(candidates) == len(set(candidates)), "duplicate frozen candidate")
    original = [by_id[uid] for uid in selected]
    same_ids = [dict(by_id[uid], text=mapped[uid]["text"]) for uid in selected]
    chunk_ids = [dict(by_id[uid], unit_id=mapped[uid]["chunk_id"], text=mapped[uid]["text"]) for uid in selected]
    packs = {}
    for name, pack in (("native_ids_original_text", original), ("native_ids_fixture_text", same_ids),
                       ("fixture_chunk_ids_fixture_text", chunk_ids)):
        text = render_research_pack(pack)
        count = token_count(text)
        packs[name] = {"rendered_sha256": sha(text.encode()), "utf8_bytes": len(text.encode()),
                       "bge_tokens": count, "within_1024": count <= 1024,
                       "selected_ids_source_order": [u["unit_id"] for u in sorted(pack, key=lambda u: u["order"])]}
    require(packs["native_ids_original_text"]["rendered_sha256"] == query["pack_sha256"]
            and packs["native_ids_original_text"]["bge_tokens"] == query["actual_evidence_tokens"],
            "historical three-unit research pack did not reproduce")
    compatibility = [{"native_unit_id": uid,
                      "source_text_byte_compatible": by_id[uid]["text"].encode() == mapped[uid]["text"].encode()}
                     for uid in candidates]
    return {"doc_id": query["doc_id"], "question_id": query["question_id"],
            "candidate_count": len(candidates), "candidate_text_compatibility": compatibility,
            "source_text_byte_compatible_count": sum(row["source_text_byte_compatible"] for row in compatibility),
            "api_cache_hit_claimed": False, "selected_members_unchanged": True,
            "renderer": "source-ordered [id] newline text; double-newline separator; research format only",
            "packs": packs}


def verify_source_slice(data):
    """Verify only the bounded two-document copy, not the wider source corpus."""
    documents = data["documents"]
    queries = data["queries"]
    require(len(documents) == len(queries) == 2 and sorted(map(len, documents.values())) == [23, 60],
            "fixed two-document source scope changed")
    require([q["ordinal"] for q in queries] == [1, 2]
            and {q["doc_id"] for q in queries} == set(documents), "fixed query identity differs")
    require([len(q["candidate_ids"]) for q in queries] == [16, 14], "30-candidate scope changed")
    expected = {(doc, unit["unit_id"]) for doc, units in documents.items() for unit in units}
    require(len(expected) == 83, "native source identity count differs")
    source_docs = {doc["doc_id"]: doc for doc in data["native_documents"]}
    require(len(source_docs) == len(data["native_documents"]) == 2 and set(source_docs) == set(documents),
            "bounded native document inventory differs")
    blocks = {(b["doc_id"], b["native_unit_id"]): b for b in data["native_blocks"]}
    mappings = {(b["doc_id"], b["native_unit_id"]): b for b in data["unit_mapping"]}
    leaves = {(b["doc_id"], b["meta"]["native_unit_id"]): b for b in data["native_leaves"]}
    for key, index in (("native_blocks", blocks), ("unit_mapping", mappings), ("native_leaves", leaves)):
        require(len(data[key]) == len(index) == 83 and set(index) == expected, "bounded sidecar inventory differs")
    for doc_id, units in documents.items():
        source = source_docs[doc_id]
        require(sha(source["canonical_text"].encode()) == source["canonical_text_sha256"]
                and sha(source["native_document_text"].encode()) == source["native_document_text_sha256"],
                "bounded source document hash differs")
        for unit in units:
            key = (doc_id, unit["unit_id"])
            block, mapping, leaf = blocks[key], mappings[key], leaves[key]
            a, b = block["native_char_span"]
            require(source["canonical_text"][unit["start"]:unit["end"]] == unit["text"]
                    and source["native_document_text"][a:b] == unit["native_text"] == block["raw_native_text"]
                    and leaf["text"] == unit["text"], "native source slice text differs")
            require([unit["start"], unit["end"]] == block["canonical_char_span"] == mapping["canonical_char_span"]
                    == leaf["meta"]["canonical_char_span"]
                    and [a, b] == mapping["raw_native_char_span"] == leaf["meta"]["raw_native_char_span"],
                    "native source coordinate binding differs")
            require(block["native_block_id"] == mapping["native_block_id"] == leaf["meta"]["native_block_id"]
                    and mapping["leaf_id"] == leaf["leaf_id"], "native source block/leaf identity differs")
            for item in (block, mapping, leaf["meta"]):
                require(item["raw_native_text_sha256"] == sha(unit["native_text"].encode())
                        and item["retrieval_text_sha256"] == sha(unit["text"].encode()), "source unit hash differs")
    return {"documents_verified": 2, "source_units_verified": 83,
            "scope": "bounded slice only; ancestry commitments retained without rescanning upstream corpus",
            "atom_character_offsets_inferred": False}


def run(protocol_sha, output):
    protocol_path, input_path = DIRECTORY / "protocol.json", DIRECTORY / "bounded_inputs.json"
    require(len(protocol_sha) == 64 and all(c in "0123456789abcdef" for c in protocol_sha), "explicit protocol SHA required")
    require(sha(protocol_path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = read(protocol_path)
    require(protocol["bounded_inputs_sha256"] == INPUT_SHA == sha(input_path.read_bytes()), "frozen input changed")
    sources = protocol["source_sha256"]
    require(SOURCE_NAMES <= set(sources), "source closure incomplete")
    bindings = {str(protocol_path): protocol_sha, str(input_path): INPUT_SHA}
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.suffix == ".py", "unexpected source binding")
        require(sha(path.read_bytes()) == expected, "source drift")
        bindings[str(path)] = expected
    tokenizer_config = protocol["tokenizer"]
    tokenizer_path = Path(tokenizer_config["path"])
    require(set(tokenizer_config["verified_file_sha256"]) == TOKENIZER_FILES, "tokenizer closure differs")
    for name, expected in tokenizer_config["verified_file_sha256"].items():
        path = tokenizer_path / name
        require(sha(path.read_bytes()) == expected, "frozen tokenizer changed")
        bindings[str(path)] = expected
    data = read(input_path)
    source_check = verify_source_slice(data)
    output = Path(output).resolve()
    require(output.is_relative_to(DIRECTORY.resolve()) and output != DIRECTORY.resolve()
            and not output.exists(), "new output subdirectory required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    documents, fixtures, packs = [], [], []
    for query in data["queries"]:
        doc_id = query["doc_id"]
        units = data["documents"][doc_id]
        leaves = [leaf for leaf in data["native_leaves"] if leaf["doc_id"] == doc_id]
        report, fixture = probe_document(doc_id, units, count, leaves)
        documents.append(report); fixtures.append(fixture)
        packs.append(compare_query_pack(query, units, fixture, count))
    summary = {"schema": SCHEMA, "status": "mechanical_fixture_completed", "mode": "b0_identity_fixture",
        "scope": {"documents": 2, "native_units": 83, "candidate_pairs": 30, "old_three_unit_packs": 2,
                  "api_calls": 0, "key_reads": 0, "weight_reads": 0, "model_inference": False,
                  "labels_scores_gold_read": False, "semantic_or_quality_evaluation": False},
        "source_verification": source_check,
        "atom_coordinate_boundary": "Only unit-to-atom ranges are available; no atom source character offsets are inferred.",
        "source_literal_warning": "The unmodified exporter emits refiner_epoch8 even for authored identity boundaries; this is not prediction provenance.",
        "encoder_admission_boundary": "The 128-token result checks full atom length only, not model quality or complete inference feasibility.",
        "cache_boundary": "Source-text equality only; new IDs/payloads are not audited API cache hits.",
        "source_ancestry_commitments": data["input_bindings"], "bindings": bindings,
        "totals": {"source_units": sum(d["source_units"] for d in documents),
                   "atoms": sum(d["atoms"] for d in documents),
                   "fixture_chunks": sum(d["fixture_chunks"] for d in documents),
                   "fixture_leaves": sum(d["fixture_leaves"] for d in documents),
                   "atoms_over_128": sum(len(d["atoms_over_128"]) for d in documents),
                   "max_atom_bge_tokens": max(d["max_atom_bge_tokens"] for d in documents),
                   "candidate_text_compatible": sum(p["source_text_byte_compatible_count"] for p in packs)},
        "documents": documents, "query_packs": packs}
    for path, expected in bindings.items():
        require(sha(Path(path).read_bytes()) == expected, "input/source/tokenizer changed during probe")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "fixture_exports.json", fixtures)
    summary["fixture_exports_sha256"] = sha((output / "fixture_exports.json").read_bytes())
    write(output / "report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "totals": result["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
