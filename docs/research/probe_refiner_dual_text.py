"""Fixed83-unit optional dual-text trace probe; no models, labels or API."""
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
from SLAC.refiner.pipeline.assemble import build_refiner_input as builder
from SLAC.refiner.pipeline.assemble.source_atomizer import SourceTraceError, trace_atomize_unit_text
from SLAC.refiner.slac_refiner.decoding import projector

BASE = ROOT / "artifacts/research-foundation/offline-20261004"
STAGE = BASE / "refiner-dual-text-prototype-01"
INPUTS = {
    "bounded_inputs": (BASE / "refiner-granularity-probe-01/bounded_inputs.json", "e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c"),
    "fixture_exports": (BASE / "refiner-granularity-probe-01/run-02/fixture_exports.json", "ba8bfbcc4b12587323195a9395c1c3b32048d5657d354a3e09c0d50f05636024"),
    "prior_alignment": (BASE / "refiner-source-budget-probe-01/source-alignment-run-01/report.json", "cb90cf3b9aea6df1b8a0ee892d7647dbf32845f912ec0ded22bab12e0e5eeb1a"),
}
SOURCES = {
    "docs/research/probe_refiner_dual_text.py",
    "SLAC/refiner/pipeline/assemble/source_atomizer.py",
    "SLAC/refiner/pipeline/assemble/build_refiner_input.py",
    "SLAC/refiner/slac_refiner/decoding/projector.py",
    "tests/research/test_source_atomizer.py",
    "tests/research/test_refiner_split_order.py",
    "tests/research/test_refiner_seed_mapping.py",
}
TOKENIZER = Path("D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
TOKEN_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}
REPLACEMENTS = dict(zip("（ ） 【 】 — – － “ ” ‘ ’".split(), "( ) [ ] - - - \" \" ' '".split()))


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
    with Path(path).open("xb") as f:
        f.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def verify_trace(source, trace):
    """Independently check emitted character claims, not the traced algorithm."""
    require(trace.source_text == source and not trace.trivia_only, "nonempty native source changed")
    require(trace.model_atoms == [a["model_text"] for a in trace.atoms] and trace.atoms,
            "model atom representation differs")
    previous_partition = previous_origin = 0
    covered = set()
    inserted = changed_atoms = 0
    for atom in trace.atoms:
        start, end = atom["source_span"]
        require(0 <= start == previous_partition < end <= len(source), "source partition gap/overlap")
        require(atom["source_text"] == source[start:end], "rendering is not its exact source slice")
        previous_partition = end
        changed_atoms += atom["model_text"] != atom["source_text"]
        require(len(atom["model_text"]) == len(atom["model_char_origins"]), "character origin count differs")
        for char, origin in zip(atom["model_text"], atom["model_char_origins"], strict=True):
            if origin is None:
                require(char == " ", "unsupported inserted model character")
                inserted += 1
                continue
            a, b = origin
            require(type(a) is int and type(b) is int and start <= a < b <= end and previous_origin <= a,
                    "origin interval is invalid or out of order")
            previous_origin = b
            raw = source[a:b]
            require(raw == char or (raw.isspace() and char in " \n")
                    or (len(raw) == 1 and REPLACEMENTS.get(raw) == char),
                    "model character has no supported source transformation")
            covered.update(range(a, b))
    require(previous_partition == len(source) and "".join(a["source_text"] for a in trace.atoms) == source,
            "render partitions do not reconstruct the full native Unit")
    require(all(c.isspace() or i in covered for i, c in enumerate(source)),
            "nonwhitespace source characters absent from model origins")
    rendered = [a["source_text"] for a in trace.atoms]
    char_spans = [a["source_span"] for a in trace.atoms]
    partitions = projector.spans_to_units(rendered, [(i, i+1) for i in range(len(rendered))],
        source_text=source, atom_char_spans=char_spans)
    whole = projector.spans_to_units(rendered, [(0, len(rendered))],
        source_text=source, atom_char_spans=char_spans)
    require("".join(p["text"] for p in partitions).encode() == source.encode()
            and len(whole) == 1 and whole[0]["text"].encode() == source.encode(),
            "real projector rendering view did not preserve source bytes")
    return {"inserted_model_spaces": inserted, "model_render_different_atoms": changed_atoms,
            "source_mode_reconstruction": True, "nonwhitespace_origin_coverage": True}


def run(protocol_sha, output):
    path = STAGE / "protocol.json"
    require(sha(path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = read(path)
    require(protocol["schema"] == "slac-refiner-dual-text-prototype-protocol-v1"
            and set(protocol["inputs"]) == set(INPUTS) and set(protocol["source_sha256"]) == SOURCES,
            "protocol inventory differs")
    bindings, loaded = {str(path): protocol_sha}, {}
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
            and [f["refiner_input"]["doc_id"] for f in fixtures] == ids, "fixed scope differs")
    prior = {(d["doc_id"], r["native_unit_id"]): r["alignment"]["status"]
             for d in loaded["prior_alignment"]["documents"] for r in d["units"]}
    require(len(prior) == 83, "prior alignment denominator differs")
    output = Path(output).resolve()
    require(output.is_relative_to(STAGE.resolve()) and output != STAGE.resolve() and not output.exists(),
            "new stage output subdirectory required")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", {"status": "started", "bindings": bindings})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in offline dual-text probe")
    socket.create_connection = denied
    socket.socket.connect = denied
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    documents, traces = [], []
    for fixture in fixtures:
        record = fixture["refiner_input"]
        units = data["documents"][record["doc_id"]]
        spans, chunks = record["unit2atom_span"], fixture["raw_export"]["refined_chunks"]
        require(len(units) == len(spans) == len(chunks), "unit range scope differs")
        rows = []
        for unit, span, chunk in zip(units, spans, chunks, strict=True):
            a, b = span["start_atom"], span["end_atom"]
            require(unit["order"] == span["unit_id"] and (a, b) == (chunk["atom_start"], chunk["atom_end"]),
                    "unit/atom identity differs")
            source, cached = unit["native_text"], record["atoms"][a:b]
            row = {"native_unit_id": unit["unit_id"], "cached_atoms": len(cached),
                   "prior_exact_alignment": prior[(record["doc_id"], unit["unit_id"])]}
            try:
                trace = trace_atomize_unit_text(source)
                require(trace.model_atoms == builder.atomize_unit_text(source), "corrected builder parity differs")
                checks = verify_trace(source, trace)
            except SourceTraceError as exc:
                row.update(status="trace_failed", error_code=exc.code, error_stage=exc.stage)
            except ValueError as exc:
                row.update(status="verification_failed", error_code="independent_verification_failed",
                           error_stage="probe", error_message=str(exc))
            else:
                row.update(status="certified", model_atoms=len(trace.model_atoms),
                    model_atoms_equal_cached=trace.model_atoms == cached, corrected_builder_parity=True,
                    source_sha256=sha(source.encode()), old_export_sha256=sha(chunk["text"].encode()),
                    source_equals_old_export=source == chunk["text"],
                    source_bge_tokens=count(source), old_export_bge_tokens=count(chunk["text"]), **checks)
                traces.append({"doc_id": record["doc_id"], "native_unit_id": unit["unit_id"], "trace": asdict(trace)})
            rows.append(row)
        documents.append({"doc_id": record["doc_id"], "units": rows})
    rows = [r for d in documents for r in d["units"]]
    good = [r for r in rows if r["status"] == "certified"]
    result = {"schema": "slac-refiner-dual-text-prototype-result-v1", "status": "completed",
        "scope": protocol["scope"], "bindings": bindings, "documents": documents,
        "totals": {"source_units": len(rows), "cached_atoms": sum(r["cached_atoms"] for r in rows),
            "certified_units": len(good), "failed_units": len(rows)-len(good),
            "certified_model_atoms": sum(r["model_atoms"] for r in good),
            "certified_units_model_atoms_equal_cached": sum(r["model_atoms_equal_cached"] for r in good),
            "prior_missing_units_certified": sum(r["prior_exact_alignment"] == "missing" for r in good),
            "inserted_model_spaces": sum(r["inserted_model_spaces"] for r in good),
            "model_render_different_atoms": sum(r["model_render_different_atoms"] for r in good),
            "source_equals_old_export_units": sum(r["source_equals_old_export"] for r in good),
            "source_minus_old_export_token_histogram": dict(Counter(r["source_bge_tokens"]-r["old_export_bge_tokens"] for r in good))},
        "limits": protocol["interpretation"]}
    for path, expected in bindings.items():
        require(sha(Path(path).read_bytes()) == expected, "bound source/input/tokenizer drift")
    write(output / "traces.json", traces)
    result["traces_sha256"] = sha((output / "traces.json").read_bytes())
    write(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=STAGE / "run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))
