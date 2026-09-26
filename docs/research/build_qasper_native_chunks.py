"""Lossless native sidecars and cache-compatible deterministic rule chunks.

The main leaf index contains exactly the 1850 previously embedded native units,
with byte-identical frozen Unit.text. Raw Qasper field strings are preserved in
a separate native-document coordinate system, never conflated with canonical
retrieval text. No legacy atom normalization, model, vector encoding or API.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import tarfile

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from SLAC.retrieval.schemas.records import ChunkRecord as SLACChunkRecord, LeafRecord, DocCatalogRecord
from SLAC.retrieval.schemas.validation import validate_chunk_records, validate_leaf_records, validate_doc_catalog_records
import prepare_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import Unit, digest


SCHEMA = "slac-qasper-native-rule-chunks-v1"
CONFIG = {"chunk_budget_bge_tokens": 512, "grouping": "greedy consecutive frozen native units within one section",
    "title_and_abstract": "separate groups", "oversize_policy": "retain complete singleton; never truncate or split",
    "retrieval_representation": "frozen canonical Unit.text for leaves; exact canonical document slice for chunks",
    "raw_native_representation": "all raw title/abstract/section-name/paragraph strings joined by explicit two-newline separators",
    "raw_native_separator": "\n\n", "empty_fields": "preserved only in native sidecar; no new retrieval leaves",
    "atom_coordinate_system": "existing native Unit.order; no sentence atomization",
    "token_counting": "exact retrieval text, add_special_tokens=True, truncation=False; no IDs or path/anchor prefixes",
    "budget_rationale": "preset 512 after source-only length audit p95=272/max=787; no retrieval outcomes used"}
FILES = ("native_documents.jsonl", "native_blocks.jsonl", "leaves.jsonl", "chunks.jsonl", "unit_mapping.jsonl", "doc_catalog.jsonl")


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_id(prefix, parts):
    return prefix + "-" + pilot.stable_hash(parts)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, values):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def native_fields(paper, source_id):
    prefix = "/" + source_id
    yield "title", prefix + "/title", prefix + "/title", "title", [], paper["title"]
    yield "abstract", prefix + "/abstract", prefix + "/abstract", "abstract", [], paper["abstract"]
    for si, section in enumerate(paper["full_text"]):
        pointer = f"{prefix}/full_text/{si}"
        heading = section["section_name"]
        path = [heading] if heading.strip() else []
        group = f"section:{si}"
        yield "heading", pointer + "/section_name", pointer, group, path, heading
        for pi, text in enumerate(section["paragraphs"]):
            locator = pointer + f"/paragraphs/{pi}"
            yield "paragraph", locator, locator, group, path, text


def build_document(canonical, paper, frozen_units, tokenizer, cache_rows, *, chunk_budget=512):
    if type(chunk_budget) is not int or chunk_budget < 1:
        raise ValueError("chunk budget must be a positive integer")
    doc, source = canonical["doc_id"], canonical["source_id"]
    text = canonical["canonical_text"]
    if canonical.get("original_split") != "validation" or not isinstance(text, str):
        raise ValueError("requires an unchanged canonical validation document")
    if (not frozen_units or len({u.unit_id for u in frozen_units}) != len(frozen_units)
            or [u.order for u in frozen_units] != list(range(len(frozen_units)))):
        raise ValueError("duplicate unit IDs or noncontiguous frozen native order")
    if any(left.end > right.start for left, right in zip(frozen_units, frozen_units[1:])):
        raise ValueError("frozen canonical unit spans overlap or reverse source order")
    by_id = {u.unit_id: u for u in frozen_units}
    canonical_lookup = {}
    for block in canonical["blocks"]:
        key = (block["source_locator"]["json_pointer"], block["kind"])
        canonical_lookup.setdefault(key, []).append(block)
    native, blocks, unit_info, cursor, seen = [], [], {}, 0, []
    for index, (kind, pointer, canonical_pointer, group, path, raw) in enumerate(native_fields(paper, source)):
        if not isinstance(raw, str):
            raise ValueError("native field is not a string")
        separator = CONFIG["raw_native_separator"] if index else ""
        coverage_start = cursor
        native.append(separator + raw)
        start, end = cursor + len(separator), cursor + len(separator) + len(raw)
        cursor = end
        item = {"doc_id": doc, "native_block_id": object_id("native-block", [doc, pointer]),
            "block_order": index, "kind": kind, "native_json_pointer": pointer,
            "section_group": group, "raw_native_text": raw, "raw_native_text_sha256": text_hash(raw),
            "native_char_span": [start, end], "native_coverage_char_span": [coverage_start, end],
            "native_unit_id": None, "retrievable": False}
        if raw.strip():
            matches = canonical_lookup.get((canonical_pointer, "heading" if kind == "title" else kind), [])
            if len(matches) != 1 or matches[0]["block_id"] not in by_id:
                raise ValueError("native field does not map to exactly one frozen canonical unit")
            block, unit = matches[0], by_id[matches[0]["block_id"]]
            if (unit.kind != kind or unit.native_text != raw or [unit.start, unit.end] != block["char_span"]
                    or not 0 <= unit.start < unit.end <= len(text) or text[unit.start:unit.end] != unit.text):
                raise ValueError("raw native text or frozen retrieval text/offset differs")
            seen.append(unit.unit_id)
            item.update(native_unit_id=unit.unit_id, retrievable=True,
                        canonical_char_span=[unit.start, unit.end], retrieval_text_sha256=text_hash(unit.text),
                        raw_equals_retrieval_text=(raw == unit.text))
            unit_info[unit.unit_id] = {**item, "path": path}
        blocks.append(item)
    if seen != [u.unit_id for u in frozen_units]:
        raise ValueError("native field order/coverage differs from every frozen unit")
    native_text = "".join(native)
    document = {"doc_id": doc, "source_id": source, "canonical_text": text,
        "canonical_text_sha256": text_hash(text), "native_document_text": native_text,
        "native_document_text_sha256": text_hash(native_text), "native_coordinate_system": CONFIG["raw_native_representation"],
        "native_field_count": len(blocks), "native_unit_count": len(frozen_units)}

    def tokens(value):
        return len(tokenizer.encode(value, add_special_tokens=True, truncation=False))

    partitions, current = [], []
    for index, unit in enumerate(frozen_units):
        info = unit_info[unit.unit_id]
        if current and (unit_info[frozen_units[current[0]].unit_id]["section_group"] != info["section_group"]
                        or tokens(text[frozen_units[current[0]].start:unit.end]) > chunk_budget):
            partitions.append(current)
            current = []
        current.append(index)
        if tokens(unit.text) > chunk_budget:
            if len(current) != 1:
                raise ValueError("oversize unit must be a complete singleton")
            partitions.append(current)
            current = []
    if current:
        partitions.append(current)
    chunks, leaves, mappings = [], [], []
    for chunk_index, indices in enumerate(partitions):
        first, last = frozen_units[indices[0]], frozen_units[indices[-1]]
        info, last_info = unit_info[first.unit_id], unit_info[last.unit_id]
        chunk_text = text[first.start:last.end]
        chunk_id = object_id("native-chunk", [SCHEMA, doc, indices[0], indices[-1] + 1, chunk_budget])
        span = [info["native_char_span"][0], last_info["native_char_span"][1]]
        count = tokens(chunk_text)
        metadata = {"representation": "frozen-canonical-document-slice", "native_unit_ids": [frozen_units[i].unit_id for i in indices],
            "canonical_char_span": [first.start, last.end], "canonical_text_sha256": text_hash(chunk_text),
            "raw_native_char_span": span, "raw_native_slice_sha256": text_hash(native_text[span[0]:span[1]]),
            "section_group": info["section_group"], "oversize_singleton": count > chunk_budget,
            "chunk_budget_bge_tokens": chunk_budget, "atom_coordinate_system": CONFIG["atom_coordinate_system"]}
        chunk = SLACChunkRecord(doc, chunk_id, chunk_index, indices[0], indices[-1] + 1, chunk_text,
            len(indices), info["path"], len(info["path"]), token_est=count, domain="qasper", meta=metadata)
        chunks.append(chunk)
        for index in indices:
            unit = frozen_units[index]
            source_info = unit_info[unit.unit_id]
            key = (doc, unit.unit_id)
            if key not in cache_rows:
                raise ValueError("native leaf absent from the verified cached-vector index")
            leaf_id = object_id("native-leaf", [SCHEMA, doc, unit.unit_id])
            meta = {"representation": "frozen-canonical-native-unit-text", "native_unit_id": unit.unit_id,
                "native_block_id": source_info["native_block_id"], "native_json_pointer": source_info["native_json_pointer"],
                "cache_embedding_row": cache_rows[key], "retrieval_text_sha256": text_hash(unit.text),
                "raw_native_text_sha256": text_hash(unit.native_text), "raw_equals_retrieval_text": unit.native_text == unit.text,
                "canonical_char_span": [unit.start, unit.end], "raw_native_char_span": source_info["native_char_span"],
                "atom_coordinate_system": CONFIG["atom_coordinate_system"]}
            leaf = LeafRecord(doc, leaf_id, chunk_id, index, index, index + 1, unit.text,
                source_info["path"], len(source_info["path"]), token_est=tokens(unit.text), domain="qasper", meta=meta)
            leaves.append(leaf)
            mappings.append({"doc_id": doc, "native_unit_id": unit.unit_id, "leaf_id": leaf_id,
                             "owner_chunk_id": chunk_id, **meta})
    for index, leaf in enumerate(leaves):
        leaf.prev_leaf_id = leaves[index - 1].leaf_id if index else None
        leaf.next_leaf_id = leaves[index + 1].leaf_id if index + 1 < len(leaves) else None
    for index, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[index - 1].chunk_id if index else None
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return document, blocks, leaves, chunks, mappings


def validate_roundtrip(documents, blocks, leaves, chunks, mappings):
    validate_chunk_records(chunks)
    validate_leaf_records(leaves, [chunk.chunk_id for chunk in chunks])
    by_doc = {row["doc_id"]: row for row in documents}
    if len(by_doc) != len(documents) or len({row["native_block_id"] for row in blocks}) != len(blocks):
        raise ValueError("duplicate document or native block identity")
    if any({getattr(row, "doc_id", None) for row in values} != set(by_doc) for values in (leaves, chunks)):
        raise ValueError("production record document coverage differs")
    if {row["doc_id"] for row in blocks} != set(by_doc):
        raise ValueError("native block document coverage differs")
    if len(mappings) != len(leaves) or len({(r["doc_id"], r["native_unit_id"]) for r in mappings}) != len(leaves):
        raise ValueError("native-unit to leaf mapping is not one-to-one")
    if len({leaf.meta["cache_embedding_row"] for leaf in leaves}) != len(leaves):
        raise ValueError("cached embedding row assigned to multiple leaves")
    leaf_lookup = {leaf.leaf_id: leaf for leaf in leaves}
    for mapping in mappings:
        leaf = leaf_lookup[mapping["leaf_id"]]
        if (mapping["doc_id"] != leaf.doc_id or mapping["native_unit_id"] != leaf.meta["native_unit_id"]
                or mapping["owner_chunk_id"] != leaf.owner_chunk_id
                or any(mapping.get(key) != value for key, value in leaf.meta.items())):
            raise ValueError("mapping identity differs from production leaf record")
    for doc, document in by_doc.items():
        native = document["native_document_text"]
        canonical = document["canonical_text"]
        if text_hash(native) != document["native_document_text_sha256"] or text_hash(canonical) != document["canonical_text_sha256"]:
            raise ValueError("document text digest differs")
        native_blocks = sorted([row for row in blocks if row["doc_id"] == doc], key=lambda row: row["block_order"])
        if [row["block_order"] for row in native_blocks] != list(range(len(native_blocks))):
            raise ValueError("native block coverage is not complete and unique")
        reconstructed, cursor = [], 0
        for index, row in enumerate(native_blocks):
            start, end = row["native_char_span"]
            separator = CONFIG["raw_native_separator"] if index else ""
            if (start != cursor + len(separator) or end != start + len(row["raw_native_text"])
                    or row["native_coverage_char_span"] != [cursor, end]
                    or native[start:end] != row["raw_native_text"] or text_hash(row["raw_native_text"]) != row["raw_native_text_sha256"]):
                raise ValueError("raw native field offset/hash round-trip mismatch")
            reconstructed.append(separator + row["raw_native_text"])
            cursor = end
        if "".join(reconstructed) != native:
            raise ValueError("raw native document has lost or duplicated characters")
        doc_leaves = sorted([leaf for leaf in leaves if leaf.doc_id == doc], key=lambda leaf: leaf.leaf_index)
        doc_chunks = sorted([chunk for chunk in chunks if chunk.doc_id == doc], key=lambda chunk: chunk.chunk_index)
        if ([leaf.leaf_index for leaf in doc_leaves] != list(range(len(doc_leaves)))
                or document["native_unit_count"] != len(doc_leaves)
                or document["native_field_count"] != len(native_blocks)):
            raise ValueError("native leaf/field counts differ from document")
        visible_blocks = [row for row in native_blocks if row["retrievable"]]
        if ([row["native_unit_id"] for row in visible_blocks] != [leaf.meta["native_unit_id"] for leaf in doc_leaves]
                or any(bool(row["raw_native_text"].strip()) != row["retrievable"] for row in native_blocks)):
            raise ValueError("empty sidecars changed the main native leaf denominator")
        covered = []
        for index, chunk in enumerate(doc_chunks):
            if chunk.chunk_index != index or chunk.atom_start != len(covered):
                raise ValueError("chunk partition has missing/overlapping native units")
            owned = doc_leaves[chunk.atom_start:chunk.atom_end]
            covered.extend(leaf.leaf_id for leaf in owned)
            if any(leaf.owner_chunk_id != chunk.chunk_id for leaf in owned):
                raise ValueError("leaf ownership crosses chunk boundary")
            start, end = chunk.meta["canonical_char_span"]
            raw_start, raw_end = chunk.meta["raw_native_char_span"]
            if (canonical[start:end] != chunk.text or text_hash(chunk.text) != chunk.meta["canonical_text_sha256"]
                    or text_hash(native[raw_start:raw_end]) != chunk.meta["raw_native_slice_sha256"]
                    or [start, end] != [owned[0].meta["canonical_char_span"][0], owned[-1].meta["canonical_char_span"][1]]
                    or [raw_start, raw_end] != [owned[0].meta["raw_native_char_span"][0], owned[-1].meta["raw_native_char_span"][1]]
                    or chunk.meta["native_unit_ids"] != [leaf.meta["native_unit_id"] for leaf in owned]
                    or chunk.meta["oversize_singleton"] != (chunk.token_est > chunk.meta["chunk_budget_bge_tokens"])
                    or (chunk.meta["oversize_singleton"] and len(owned) != 1)):
                raise ValueError("chunk text or original-unit membership changed")
        if covered != [leaf.leaf_id for leaf in doc_leaves]:
            raise ValueError("chunks do not cover each native leaf once")
        for leaf in doc_leaves:
            start, end = leaf.meta["canonical_char_span"]
            raw_start, raw_end = leaf.meta["raw_native_char_span"]
            if (leaf.atom_start != leaf.leaf_index or leaf.atom_end != leaf.leaf_index + 1
                    or canonical[start:end] != leaf.text or text_hash(leaf.text) != leaf.meta["retrieval_text_sha256"]
                    or text_hash(native[raw_start:raw_end]) != leaf.meta["raw_native_text_sha256"]):
                raise ValueError("leaf cache-compatible text or dual offset mapping changed")


def unique_path(hashes, name):
    found = [Path(path).resolve() for path in hashes if Path(path).name == name]
    if len(found) != 1:
        raise ValueError("ambiguous frozen input filename")
    return found[0]


def load_sources(pilot_prepared, extended_prepared):
    frozen, hashes = {}, {}
    for directory in (Path(pilot_prepared).resolve(), Path(extended_prepared).resolve()):
        manifest = read_json(directory / "manifest.json")
        if (manifest.get("status") != "prepared" or manifest.get("test_payload_read") is not False
                or digest(directory / "prepared.json") != manifest["prepared_sha256"]):
            raise ValueError("invalid frozen native-unit preparation")
        for path, value in manifest["input_sha256"].items():
            path = str(Path(path).resolve())
            if path in hashes and hashes[path] != value:
                raise ValueError("prepared source hashes conflict")
            hashes[path] = value
        for doc, values in read_json(directory / "prepared.json")["documents"].items():
            if doc in frozen:
                raise ValueError("pilot and extended preparation overlap by document")
            frozen[doc] = [Unit(**unit) for unit in values]
        for name in ("manifest.json", "prepared.json"):
            hashes[str(directory / name)] = digest(directory / name)
    pilot.verify_hashes(hashes)
    if len(frozen) != 32 or sum(map(len, frozen.values())) != 1850:
        raise ValueError("requires all 32 documents and exactly 1850 existing native units")
    candidates_path = unique_path(hashes, "candidates.jsonl")
    candidates = rows(candidates_path)
    if (len(candidates) != 32 or {r["doc_id"] for r in candidates} != set(frozen)
            or any(r["official_split"] != "validation" for r in candidates)):
        raise ValueError("frozen unit documents differ from validation candidate pool")
    shard = unique_path(hashes, "documents-00000.jsonl.gz")
    canonicals = {}
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["doc_id"] in frozen:
                if row["doc_id"] in canonicals:
                    raise ValueError("duplicate canonical document")
                canonicals[row["doc_id"]] = row
    if set(canonicals) != set(frozen):
        raise ValueError("missing canonical development document")
    archive = unique_path(hashes, "qasper-train-dev-v0.3.tgz")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("qasper-dev-v0.3.json")
        if not member.isfile() or member.size > 32 * 1024 * 1024:
            raise ValueError("unexpected native validation member")
        with tar.extractfile(member) as stream:
            native = json.load(stream)
    papers = {doc: {name: native[value["source_id"]][name] for name in ("title", "abstract", "full_text")}
              for doc, value in canonicals.items()}
    summary_path = unique_path(hashes, "summary.json")
    summary = read_json(summary_path)
    if summary.get("status") != "completed" or summary.get("test_payload_read") is not False:
        raise ValueError("cached dense run is not a complete validation run")
    index_path, vectors_path = summary_path.parent / "embedding_index.json", summary_path.parent / "embeddings.safetensors"
    index = read_json(index_path)["candidates"]
    expected = [{"doc_id": doc, "unit_id": unit.unit_id} for doc in sorted(frozen) for unit in frozen[doc]]
    if index != expected or digest(vectors_path) != summary["embedding_sha256"]:
        raise ValueError("cached-vector identities, order or tensor digest differ")
    with vectors_path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 1024 * 1024:
            raise ValueError("unexpected cached-vector header size")
        header = json.loads(stream.read(length))
    if header["candidate_embeddings"]["shape"] != [1850, 1024] or header["candidate_embeddings"]["dtype"] != "F32":
        raise ValueError("unexpected cached candidate-vector shape or dtype")
    tokenizer_path = Path(summary["model"])
    if str((tokenizer_path / "tokenizer.json").resolve()) not in hashes:
        raise ValueError("tokenizer identity is absent from frozen dense provenance")
    for path in (index_path, vectors_path, Path(__file__).resolve(), REPO_ROOT / "SLAC/retrieval/schemas/records.py",
                 REPO_ROOT / "SLAC/retrieval/schemas/validation.py"):
        hashes[str(path)] = digest(path)
    cache_rows = {(item["doc_id"], item["unit_id"]): row for row, item in enumerate(index)}
    return frozen, canonicals, papers, hashes, summary, tokenizer_path, cache_rows, vectors_path


def distribution(values):
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / len(ordered), "p50": ordered[int(.5 * (len(ordered) - 1))],
            "p95": ordered[int(.95 * (len(ordered) - 1))]}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("native adapter output already exists")
    frozen, canonicals, papers, hashes, dense, tokenizer_path, cache_rows, vectors_path = load_sources(
        args.pilot_prepared, args.extended_prepared)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    documents, blocks, leaves, chunks, mappings, catalogs = [], [], [], [], [], []
    for doc in sorted(frozen):
        document, ds, ls, cs, ms = build_document(canonicals[doc], papers[doc], frozen[doc], tokenizer, cache_rows)
        documents.append(document); blocks.extend(ds); leaves.extend(ls); chunks.extend(cs); mappings.extend(ms)
        catalogs.append(DocCatalogRecord(doc, papers[doc]["title"], "qasper-dev-v0.3.json#" + canonicals[doc]["source_id"],
            domain="qasper", num_chunks=len(cs), num_leaves=len(ls),
            meta={"canonical_document_sha256": document["canonical_text_sha256"],
                  "native_document_sha256": document["native_document_text_sha256"]}))
    validate_roundtrip(documents, blocks, leaves, chunks, mappings)
    validate_doc_catalog_records(catalogs)
    if [(leaf.doc_id, leaf.meta["native_unit_id"], leaf.text) for leaf in leaves] != [
            (doc, unit.unit_id, unit.text) for doc in sorted(frozen) for unit in frozen[doc]]:
        raise ValueError("main leaf index differs from cached inputs")
    pilot.verify_hashes(hashes)
    public = {"schema": SCHEMA, "status": "completed", "config": CONFIG, "documents": len(documents),
        "native_fields": len(blocks), "retrieval_leaves": len(leaves), "chunks": len(chunks),
        "empty_or_whitespace_fields_preserved": sum(not row["retrievable"] for row in blocks),
        "raw_and_cache_text_differ": sum(not leaf.meta["raw_equals_retrieval_text"] for leaf in leaves),
        "oversize_native_units": sum(leaf.token_est > 512 for leaf in leaves),
        "oversize_singleton_chunks": sum(chunk.meta["oversize_singleton"] for chunk in chunks),
        "leaf_bge_tokens": distribution([leaf.token_est for leaf in leaves]),
        "chunk_bge_tokens": distribution([chunk.token_est for chunk in chunks]),
        "native_units_per_chunk": distribution([chunk.num_atoms for chunk in chunks]),
        "source_length_audit": dense["length_audit"]["candidate_units"],
        "identity_roundtrip_passed": True, "all_1850_cached_vector_rows_matched": True,
        "api_calls": 0, "model_loaded": False, "embedding_vectors_loaded": False,
        "tokenizer_loaded": True, "test_payload_read": False, "qa_used_for_partition": False,
        "legacy_normalization_used": False, "dual_representation_explicit": True,
        "index_built": False, "source_hashes_unchanged": True,
        "limits": ["Raw native offsets address an explicitly materialized field-concatenation view, not PDF offsets.",
                   "Production LeafRecord.text preserves the old canonical encoder input; raw source strings live in sidecars.",
                   "Production enrich/compose helpers normalize or strip/add anchors and must not be used for exact cache reuse.",
                   "No retrieval, reranking, training, answer generation or outcome-based boundary selection performed.",
                   "The 512-token raw retrieval-text cap includes tokenizer specials, not evidence IDs or generator instructions."]}
    output.mkdir(parents=True, exist_ok=False)
    content = (documents, blocks, [v.to_dict() for v in leaves], [v.to_dict() for v in chunks], mappings,
               [v.to_dict() for v in catalogs])
    for name, values in zip(FILES, content, strict=True):
        write_rows(output / name, values)
    pilot.write_json(output / "public_summary.json", public)
    manifest = {**public, "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": hashes,
        "input_binding_sha256": pilot.stable_hash(hashes), "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_revision": dense["model_revision"], "cached_vectors": str(vectors_path.resolve()),
        "cached_candidate_tensor": "candidate_embeddings", "cache_identity_order": "sorted doc_id then frozen native Unit.order",
        "ordered_leaf_input_sha256": pilot.stable_hash([{"doc_id": l.doc_id, "unit_id": l.meta["native_unit_id"],
            "text_sha256": l.meta["retrieval_text_sha256"]} for l in leaves]),
        "output_files_sha256": {name: digest(output / name) for name in (*FILES, "public_summary.json")}}
    pilot.write_json(output / "manifest.json", manifest)
    load_artifacts(output)
    return manifest


def load_artifacts(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed" or manifest.get("config") != CONFIG:
        raise ValueError("unsupported native rule-chunk manifest")
    if set(manifest["output_files_sha256"]) != set((*FILES, "public_summary.json")):
        raise ValueError("native artifact file inventory differs")
    pilot.verify_hashes(manifest["input_sha256"])
    pilot.verify_hashes({str(directory / name): value for name, value in manifest["output_files_sha256"].items()})
    documents, blocks, leaf_rows, chunk_rows, mappings, catalog_rows = [rows(directory / name) for name in FILES]
    leaves, chunks = [LeafRecord(**r) for r in leaf_rows], [SLACChunkRecord(**r) for r in chunk_rows]
    validate_roundtrip(documents, blocks, leaves, chunks, mappings)
    validate_doc_catalog_records([DocCatalogRecord(**r) for r in catalog_rows])
    return manifest, documents, leaves, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-prepared", "extended-prepared", "output"):
        parser.add_argument("--" + name, required=True)
    result = run(parser.parse_args())
    print(json.dumps({name: result[name] for name in ("status", "documents", "native_fields", "retrieval_leaves",
        "chunks", "empty_or_whitespace_fields_preserved", "raw_and_cache_text_differ", "oversize_singleton_chunks")}, indent=2))
