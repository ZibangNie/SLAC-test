"""Fixed two-document source-view/export probe; offline tokenizer only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SLAC.refiner.pipeline.assemble.source_document_view import compose_document_source_view
from SLAC.refiner.pipeline.assemble.export_refined_chunks import export_refined_chunks_from_candidate
from SLAC.refiner.slac_refiner.decoding import projector

BASE = ROOT / "artifacts/research-foundation/offline-20261004"
STAGE = BASE / "refiner-document-source-export-01"
INPUTS = {
    "bounded_inputs": (BASE / "refiner-granularity-probe-01/bounded_inputs.json", "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"),
    "fixture_exports": (BASE / "refiner-granularity-probe-01/run-02/fixture_exports.json", "ba8bfbcc4b12587323195a9395c1c3b32048d5657d354a3e09c0d50f05636024"),
    "dual_traces": (BASE / "refiner-dual-text-prototype-01/run-01/traces.json", "e48dcaaf054eaca51a2823a523f24410267d4e4bf3ddb4572646ada17bbdd8d2"),
}
SOURCES = {
    "docs/research/probe_refiner_document_source.py",
    "SLAC/refiner/pipeline/assemble/source_document_view.py",
    "SLAC/refiner/pipeline/assemble/source_atomizer.py",
    "SLAC/refiner/pipeline/assemble/export_refined_chunks.py",
    "SLAC/refiner/slac_refiner/decoding/projector.py",
    "tests/research/test_source_document_view.py",
    "tests/research/test_refiner_source_export.py",
    "tests/research/test_refiner_seed_mapping.py",
}
TOKENIZER = Path("D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
TOKEN_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open("xb") as output:
        output.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def indexed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, "duplicate source identity")
        result[key] = row
    return result


def candidate(b0, boundary, mode):
    return {"candidate_type": mode, "teacher_ckpt": None,
            "prediction": {"b0_sparse": [i for i, value in enumerate(b0) if value],
                           "b_pred_sparse": [i for i, value in enumerate(boundary) if value]}}


def chunk_stats(chunks, count, cfg):
    rows = [{"atom_start": c["atom_start"], "atom_end": c["atom_end"],
             "chars": len(c["text"]), "tokens": count(c["text"]),
             "text_sha256": sha(c["text"].encode())} for c in chunks]
    return {"chunks": len(rows), "max_tokens": max(r["tokens"] for r in rows),
            "max_chars": max(r["chars"] for r in rows),
            "hard_limit_violations": sum(r["tokens"] > cfg.max_chunk_tokens or r["chars"] > cfg.max_chunk_chars
                or r["atom_end"] - r["atom_start"] > cfg.max_chunk_atoms for r in rows),
            "chunk_rows": rows}


def compare(projected_units, exported_chunks, count):
    require(len(projected_units) == len(exported_chunks), "projection/export chunk count differs")
    rows = []
    for unit, chunk in zip(projected_units, exported_chunks, strict=True):
        require((unit["start_atom"], unit["end_atom"]) == (chunk["atom_start"], chunk["atom_end"]),
                "projection/export atom span differs")
        rows.append({"bytes_equal": unit["text"].encode() == chunk["text"].encode(),
                     "tokens_equal": count(unit["text"]) == count(chunk["text"]),
                     "chars_equal": len(unit["text"]) == len(chunk["text"])})
    return {"chunks": len(rows), **{key: sum(r[key] for r in rows) for key in rows[0]}}


def source_export_check(view, export):
    chunks, leaves = export["refined_chunks"], export["leaf_records"]
    require("".join(c["text"] for c in chunks).encode() == view.source_text.encode(),
            "chunk concatenation does not reproduce structured source")
    require(len(leaves) == len(view.model_atoms)
            and "".join(c["text"] for c in leaves).encode() == view.source_text.encode(),
            "leaf concatenation does not reproduce structured source")
    for chunk in chunks:
        a, b = chunk["atom_start"], chunk["atom_end"]
        require(chunk["text"] == view.render(a, b)
                and tuple(chunk["source_char_span"]) == view.char_span(a, b), "chunk source view differs")
        require("".join(leaf["text"] for leaf in leaves[a:b]) == chunk["text"], "leaf/chunk mismatch")
    for i, leaf in enumerate(leaves):
        require(leaf["atom_index"] == i and leaf["text"] == view.render(i, i + 1)
                and tuple(leaf["source_char_span"]) == view.char_span(i, i + 1), "leaf source view differs")


def run(protocol_sha, output):
    protocol_path = STAGE / "protocol.json"
    require(sha(protocol_path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = read(protocol_path)
    require(protocol["schema"] == "slac-refiner-document-source-protocol-v1"
            and set(protocol["inputs"]) == set(INPUTS) and set(protocol["source_sha256"]) == SOURCES,
            "protocol inventory differs")
    bindings, loaded = {str(protocol_path): protocol_sha}, {}
    for role, (path, expected) in INPUTS.items():
        item = protocol["inputs"][role]
        require(Path(item["path"]).resolve() == path.resolve() and item["sha256"] == expected
                and sha(path.read_bytes()) == expected, "fixed input binding differs")
        loaded[role] = read(path)
        bindings[str(path)] = expected
    for name, expected in protocol["source_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path.read_bytes()) == expected, "source changed")
        bindings[str(path)] = expected
    tok = protocol["tokenizer"]
    require(Path(tok["path"]).resolve() == TOKENIZER.resolve() and set(tok["verified_file_sha256"]) == TOKEN_FILES,
            "tokenizer inventory differs")
    for name, expected in tok["verified_file_sha256"].items():
        path = TOKENIZER / name
        require(sha(path.read_bytes()) == expected, "tokenizer changed")
        bindings[str(path)] = expected
    data, fixtures = loaded["bounded_inputs"], loaded["fixture_exports"]
    ids = [q["doc_id"] for q in data["queries"]]
    require(len(ids) == len(set(ids)) == 2 and [len(data["documents"][d]) for d in ids] == [60, 23]
            and [f["refiner_input"]["doc_id"] for f in fixtures] == ids, "fixed two-document scope differs")
    docs = indexed(data["native_documents"], ("doc_id",))
    mappings = indexed(data["unit_mapping"], ("doc_id", "native_unit_id"))
    blocks = indexed(data["native_blocks"], ("doc_id", "native_unit_id"))
    traces = indexed(loaded["dual_traces"], ("doc_id", "native_unit_id"))
    require(len(docs) == 2 and len(mappings) == len(blocks) == len(traces) == 83, "fixed input denominator differs")
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(),
            "new stage output subdirectory required")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", {"status": "started", "bindings": bindings})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in offline document source probe")
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    cfg = projector.ProjectorConfig()
    require(protocol["projector_config"] == asdict(cfg), "projector configuration changed")
    document_rows, artifacts = [], []
    for ordinal, fixture in enumerate(fixtures, 1):
        record = fixture["refiner_input"]
        doc_id = record["doc_id"]
        native = docs[(doc_id,)]
        source = native["native_document_text"]
        require(sha(source.encode()) == native["native_document_text_sha256"], "native document hash differs")
        row = {"document_ordinal": ordinal, "units": len(data["documents"][doc_id]),
               "model_atoms": len(record["atoms"]), "source_chars": len(source)}
        document_rows.append(row)
        unit_traces, gaps, previous = [], [], 0
        try:
            for unit, span in zip(data["documents"][doc_id], record["unit2atom_span"], strict=True):
                key = (doc_id, unit["unit_id"])
                mapping, block, trace = mappings[key], blocks[key], traces[key]["trace"]
                a, b = mapping["raw_native_char_span"]
                require(type(a) is int and type(b) is int and previous <= a < b <= len(source),
                        "native document unit coordinates invalid")
                gap = source[previous:a]
                gaps.append({"span": [previous, a], "chars": len(gap), "nonwhitespace_chars": sum(not c.isspace() for c in gap)})
                require(not gap.strip(), "nonwhitespace document gap lacks model provenance")
                require(block["native_char_span"] == [a, b]
                        and source[a:b] == block["raw_native_text"] == unit["native_text"] == trace["source_text"]
                        and sha(source[a:b].encode()) == mapping["raw_native_text_sha256"] == block["raw_native_text_sha256"],
                        "native unit provenance differs")
                require(unit["order"] == span["unit_id"]
                        and record["atoms"][span["start_atom"]:span["end_atom"]] == trace["model_atoms"],
                        "cached model atom mapping differs")
                unit_traces.append({"native_unit_id": unit["unit_id"], "source_span": [a, b], "trace": trace})
                previous = b
            gap = source[previous:]
            gaps.append({"span": [previous, len(source)], "chars": len(gap), "nonwhitespace_chars": sum(not c.isspace() for c in gap)})
            require(not gap.strip(), "nonwhitespace trailing gap lacks model provenance")
            view = compose_document_source_view(doc_id=doc_id, source_text=source,
                coordinate_system="structured_native_document_python_character_offsets; " + native["native_coordinate_system"],
                unit_traces=unit_traces, model_atoms=record["atoms"])
            view.validate_against(doc_id, record["atoms"])
        except ValueError as exc:
            row.update(status="composition_failed", error_code=getattr(exc, "code", type(exc).__name__))
            artifacts.append({"doc_id": doc_id, "document_ordinal": ordinal, "status": row["status"], "gaps": gaps})
            continue
        b0, atoms = record["b0"], record["atoms"]
        b0_copy, atoms_copy = list(b0), list(atoms)
        raw_legacy = export_refined_chunks_from_candidate(record, fixture["authored_identity_candidate"])
        require(raw_legacy == fixture["raw_export"], "default exporter differs from frozen identity export")
        raw_source = export_refined_chunks_from_candidate(record, candidate(b0, b0, "b0_source_identity_fixture"), source_view=view)
        raw_spans = projector.boundary_vector_to_spans(len(atoms), b0)
        raw_source_units = projector.spans_to_units(view.render_atoms, raw_spans,
            source_text=view.source_text, atom_char_spans=view.atom_char_spans)
        legacy_projection = projector.project_boundary_vector(atoms, b0, cfg, token_counter=count, strict=True)
        source_projection = projector.project_boundary_vector(view.render_atoms, b0, cfg, token_counter=count,
            source_text=view.source_text, atom_char_spans=view.atom_char_spans, strict=True)
        legacy_export = export_refined_chunks_from_candidate(record,
            candidate(b0, legacy_projection["projected_b"], "rule_projected_b0_legacy_fixture"))
        source_export = export_refined_chunks_from_candidate(record,
            candidate(b0, source_projection["projected_b"], "rule_projected_b0_source_fixture"), source_view=view)
        require(record["atoms"] == atoms_copy and record["b0"] == b0_copy, "normalized model input mutated")
        source_export_check(view, raw_source)
        source_export_check(view, source_export)
        checks = {"raw_b0_source": compare(raw_source_units, raw_source["refined_chunks"], count),
                  "rule_projected_legacy": compare(legacy_projection["projected_units"], legacy_export["refined_chunks"], count),
                  "rule_projected_source": compare(source_projection["projected_units"], source_export["refined_chunks"], count)}
        require(all(checks[m]["chunks"] == checks[m][k] for m in ("raw_b0_source", "rule_projected_source")
                    for k in ("bytes_equal", "tokens_equal", "chars_equal")), "source projection/export contract differs")
        modes = {"raw_b0_legacy": raw_legacy, "raw_b0_source": raw_source,
                 "rule_projected_legacy": legacy_export, "rule_projected_source": source_export}
        stats = {name: chunk_stats(export["refined_chunks"], count, cfg) for name, export in modes.items()}
        raw_delta = Counter(count(a["text"]) - count(b["text"]) for a, b in
            zip(raw_source["refined_chunks"], raw_legacy["refined_chunks"], strict=True))
        equal_boundaries = source_projection["projected_b"] == legacy_projection["projected_b"]
        projected_delta = Counter(count(a["text"]) - count(b["text"]) for a, b in
            zip(source_export["refined_chunks"], legacy_export["refined_chunks"], strict=True)) if equal_boundaries else None
        row.update(status="certified", model_atoms_and_b0_unchanged=True, legacy_export_frozen_equality=True,
            document_gap_chars=sum(g["chars"] for g in gaps), nonempty_document_gaps=sum(g["chars"] > 0 for g in gaps),
            nonwhitespace_document_gap_chars=sum(g["nonwhitespace_chars"] for g in gaps),
            source_leaf_records=len(source_export["leaf_records"]), source_reconstruction_verified=True,
            raw_b0_source_minus_legacy_token_histogram=dict(raw_delta),
            projected_source_minus_legacy_token_histogram=dict(projected_delta) if projected_delta is not None else None,
            projection_boundaries_equal_between_modes=equal_boundaries, projection_export_comparison=checks,
            modes={mode: {k: v for k, v in values.items() if k != "chunk_rows"} for mode, values in stats.items()},
            boundary_changes={name: {"removed": sum(a == 1 and b == 0 for a, b in zip(b0, p["projected_b"], strict=True)),
                "added": sum(a == 0 and b == 1 for a, b in zip(b0, p["projected_b"], strict=True)),
                "hard_max_satisfied": p["hard_max_satisfied"], "remaining_short_spans": len(p["short_spans"])}
                for name, p in (("legacy", legacy_projection), ("source", source_projection))})
        artifacts.append({"document_ordinal": ordinal, "doc_id": doc_id, "view": asdict(view), "gaps": gaps,
            "model_inference": False, "fixture_warning": "Authored b0 and deterministic rules only; no model boundary prediction.",
            "raw_b0_source_projected_units": raw_source_units,
            "legacy_projection": legacy_projection, "source_projection": source_projection, "exports": modes, "stats": stats})
    good = [r for r in document_rows if r["status"] == "certified"]
    totals = {"documents": len(document_rows), "source_units": sum(r["units"] for r in document_rows),
              "model_atoms": sum(r["model_atoms"] for r in document_rows), "certified_documents": len(good),
              "failed_documents": len(document_rows) - len(good),
              "certified_units": sum(r["units"] for r in good), "certified_atoms": sum(r["model_atoms"] for r in good),
              "document_gap_chars": sum(r["document_gap_chars"] for r in good),
              "nonempty_document_gaps": sum(r["nonempty_document_gaps"] for r in good),
              "nonwhitespace_document_gap_chars": sum(r["nonwhitespace_document_gap_chars"] for r in good)}
    for path, expected in bindings.items():
        require(sha(Path(path).read_bytes()) == expected, "bound file drift")
    write(output / "document_exports.json", artifacts)
    report = {"schema": "slac-refiner-document-source-result-v1", "status": "completed",
              "scope": protocol["scope"], "bindings": bindings, "documents": document_rows, "totals": totals,
              "artifact_sha256": sha((output / "document_exports.json").read_bytes()), "limits": protocol["interpretation"]}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=STAGE / "run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))
