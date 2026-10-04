"""Exact per-unit atom/source alignment; no normalization or model execution."""
from __future__ import annotations

from collections.abc import Sequence


def align_exact_atoms(source_text: str, atoms_text: Sequence[str]) -> dict:
    """Count ordered nonoverlapping full mappings, saturating at two.

    Individual occurrence counts include overlapping substring matches. Only a
    unique complete chain receives offsets; missing/ambiguous chains never do.
    """
    if not isinstance(source_text, str) or not isinstance(atoms_text, Sequence) or isinstance(atoms_text, (str, bytes)):
        raise ValueError("source must be text and atoms must be a sequence")
    if not atoms_text or any(not isinstance(atom, str) or not atom for atom in atoms_text):
        raise ValueError("a nonempty sequence of nonempty text atoms is required")
    occurrences = []
    for atom in atoms_text:
        matches, cursor = [], 0
        while True:
            start = source_text.find(atom, cursor)
            if start < 0:
                break
            matches.append((start, start + len(atom)))
            cursor = start + 1
        occurrences.append(matches)
    counts, parents = [], []
    for index, matches in enumerate(occurrences):
        if index == 0:
            counts.append([1] * len(matches))
            parents.append([None] * len(matches))
            continue
        current, links = [], []
        for start, _ in matches:
            total, parent = 0, None
            for previous, (_, end) in enumerate(occurrences[index - 1]):
                if end <= start and counts[index - 1][previous]:
                    ways = counts[index - 1][previous]
                    parent = previous if total == 0 and ways == 1 else None
                    total = min(2, total + ways)
                    if total == 2:
                        parent = None
                        break
            current.append(total)
            links.append(parent)
        counts.append(current)
        parents.append(links)
    total = min(2, sum(counts[-1]))
    spans = None
    if total == 1:
        position = counts[-1].index(1)
        spans = []
        for index in range(len(occurrences) - 1, -1, -1):
            spans.append(occurrences[index][position])
            position = parents[index][position]
        spans.reverse()
    status = ("missing", "unique", "ambiguous")[total]
    reason = ("at_least_one_atom_absent" if any(not matches for matches in occurrences)
              else "no_ordered_nonoverlapping_full_chain" if total == 0
              else "unique_full_chain" if total == 1 else "multiple_full_chains")
    return {"status": status, "chain_count_capped": total,
            "occurrence_counts": [len(matches) for matches in occurrences],
            "spans": spans, "reason": reason}


def summarize_gaps(source_text: str, spans: Sequence[tuple[int, int]]) -> list[dict]:
    """Describe unmatched characters without calling them semantically empty."""
    if not isinstance(source_text, str) or not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)) or not spans:
        raise ValueError("text and nonempty span sequence required")
    previous, gaps = 0, []
    for index, span in enumerate(spans):
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            raise ValueError("invalid span")
        start, end = span
        if type(start) is not int or type(end) is not int or not 0 <= previous <= start < end <= len(source_text):
            raise ValueError("spans must be ordered nonoverlapping character intervals")
        gaps.append(("leading" if index == 0 else "inter", previous, start))
        previous = end
    gaps.append(("trailing", previous, len(source_text)))
    return [{"kind": kind, "start": start, "end": end, "chars": end - start,
             "nonwhitespace_chars": sum(not char.isspace() for char in source_text[start:end]),
             "all_whitespace": all(char.isspace() for char in source_text[start:end])}
            for kind, start, end in gaps]


def run(protocol_sha, output):
    """Consume only the two frozen files; write a new private output directory."""
    import json
    import os
    from collections import Counter
    from pathlib import Path
    import probe_refiner_projection_export as common

    require = common.require
    protocol_path = common.STAGE / "protocol.json"
    require(len(protocol_sha) == 64 and all(c in "0123456789abcdef" for c in protocol_sha),
            "explicit protocol SHA required")
    require(common.sha(protocol_path.read_bytes()) == protocol_sha, "frozen protocol changed")
    protocol = common.read(protocol_path)
    require(protocol["schema"] == "slac-refiner-source-budget-protocol-v1", "protocol schema differs")
    require(set(protocol["inputs"]) == set(common.INPUTS), "unexpected input roles")
    bindings = {str(protocol_path): protocol_sha}
    loaded = {}
    for name, (path, expected) in common.INPUTS.items():
        item = protocol["inputs"][name]
        require(Path(item["path"]).resolve() == path.resolve() and item["sha256"] == expected,
                "input path or commitment differs")
        require(common.sha(path.read_bytes()) == expected, "fixed input changed")
        loaded[name] = common.read(path)
        bindings[str(path)] = expected
    require(set(protocol["source_sha256"]) == common.ALLOWED_SOURCES, "source closure differs")
    for name, expected in protocol["source_sha256"].items():
        path = (common.ROOT / name).resolve()
        require(path.is_relative_to(common.ROOT) and path.suffix == ".py", "invalid source path")
        require(common.sha(path.read_bytes()) == expected, "source changed")
        bindings[str(path)] = expected
    token_config = protocol["tokenizer"]
    require(Path(token_config["path"]).resolve() == common.TOKENIZER.resolve()
            and set(token_config["verified_file_sha256"]) == common.TOKENIZER_FILES,
            "tokenizer path or inventory differs")
    for name, expected in token_config["verified_file_sha256"].items():
        path = common.TOKENIZER / name
        require(common.sha(path.read_bytes()) == expected, "tokenizer changed")
        bindings[str(path)] = expected
    common.verify_inputs(loaded["bounded_inputs"], loaded["fixture_exports"])
    output = Path(output).resolve()
    require(output.is_relative_to(common.STAGE.resolve()) and output != common.STAGE.resolve()
            and not output.exists(), "new stage output subdirectory required")
    output.mkdir(parents=True, exist_ok=False)
    common.write(output / "started.json", {"status": "started", "bindings": bindings})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def denied(*args, **kwargs):
        raise RuntimeError("network forbidden in offline source alignment")

    common.socket.create_connection = denied
    common.socket.socket.connect = denied
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(common.TOKENIZER), local_files_only=True,
                                               trust_remote_code=False)

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

    documents = []
    for fixture in loaded["fixture_exports"]:
        record = fixture["refiner_input"]
        units = loaded["bounded_inputs"]["documents"][record["doc_id"]]
        spans = record["unit2atom_span"]
        chunks = fixture["raw_export"]["refined_chunks"]
        require(len(units) == len(spans) == len(chunks), "native-unit scope changed")
        rows = []
        for unit, atom_range, chunk in zip(units, spans, chunks, strict=True):
            start, end = atom_range["start_atom"], atom_range["end_atom"]
            require(unit["order"] == atom_range["unit_id"]
                    and (start, end) == (chunk["atom_start"], chunk["atom_end"]),
                    "unit/atom/chunk binding differs")
            atoms = record["atoms"][start:end]
            source = unit["native_text"]
            require(source == unit["text"], "previous native/retrieval equality changed")
            aligned = align_exact_atoms(source, atoms)
            row = {"native_unit_id": unit["unit_id"], "native_order": unit["order"],
                   "atom_span": [start, end], "atoms": len(atoms),
                   "source_sha256": common.sha(source.encode()), "alignment": aligned,
                   "projector_source_mode_checked": False}
            if aligned["status"] == "unique":
                char_spans = aligned["spans"]
                gaps = summarize_gaps(source, char_spans)
                require(sum(g["chars"] for g in gaps) + sum(b-a for a, b in char_spans) == len(source),
                        "source accounting differs")
                partitions = common.projector.spans_to_units(atoms,
                    [(i, i+1) for i in range(len(atoms))],
                    source_text=source, atom_char_spans=char_spans)
                whole = common.projector.spans_to_units(atoms, [(0, len(atoms))],
                    source_text=source, atom_char_spans=char_spans)
                require("".join(p["text"] for p in partitions).encode() == source.encode()
                        and len(whole) == 1 and whole[0]["text"].encode() == source.encode(),
                        "real source-mode projector did not reconstruct native Unit")
                row.update(projector_source_mode_checked=True, source_gaps=gaps,
                    source_nonwhitespace_gap_chars=sum(g["nonwhitespace_chars"] for g in gaps),
                    source_gap_chars=sum(g["chars"] for g in gaps),
                    lossless_reconstruction=True,
                    lossless_partition_char_spans=[[p["start_char"], p["end_char"]] for p in partitions],
                    source_equals_old_export_bytes=source.encode() == chunk["text"].encode(),
                    source_bge_tokens=count(whole[0]["text"]), old_export_bge_tokens=count(chunk["text"]))
            rows.append(row)
        documents.append({"doc_id": record["doc_id"], "units": rows,
                          "status_counts": dict(Counter(r["alignment"]["status"] for r in rows))})
    rows = [r for doc in documents for r in doc["units"]]
    certified = [r for r in rows if r["projector_source_mode_checked"]]
    result = {"schema": "slac-refiner-source-alignment-result-v1", "status": "completed",
        "scope": {"documents": 2, "source_units": len(rows), "atoms": sum(r["atoms"] for r in rows),
                  "api_calls": 0, "key_reads": 0, "weight_reads": 0, "model_inference": False,
                  "gold_labels_scores_read": False}, "bindings": bindings, "documents": documents,
        "totals": {"unit_status_counts": dict(Counter(r["alignment"]["status"] for r in rows)),
            "atom_status_counts": {state: sum(r["atoms"] for r in rows if r["alignment"]["status"] == state)
                                   for state in ("unique", "missing", "ambiguous")},
            "atoms_with_zero_individual_occurrences": sum(n == 0 for r in rows for n in r["alignment"]["occurrence_counts"]),
            "atoms_with_multiple_individual_occurrences": sum(n > 1 for r in rows for n in r["alignment"]["occurrence_counts"]),
            "source_mode_verified_units": len(certified),
            "certified_units_with_nonwhitespace_gaps": sum(r["source_nonwhitespace_gap_chars"] > 0 for r in certified),
            "certified_nonwhitespace_gap_chars": sum(r["source_nonwhitespace_gap_chars"] for r in certified),
            "certified_units_source_equals_old_export_bytes": sum(r["source_equals_old_export_bytes"] for r in certified),
            "certified_units_source_tokens_equal_export": sum(r["source_bge_tokens"] == r["old_export_bge_tokens"] for r in certified)},
        "limits": ["Offsets are Python string characters inside each native Unit; no full-document atom spans invented.",
            "Atom status counts group atoms by the whole-unit chain result; individual occurrence counts are distinct.",
            "Count2 means at least two, not an exact count. Missing/ambiguous units remain in the full83 denominator.",
            "Exact correspondence does not by itself prove semantic equivalence or model-input coverage of gaps.",
            "Source-mode checks are per-unit reconstruction, not document-level projection, retrieval or model inference.",
            "No API cache hits, reused support labels or quality metrics are claimed."]}
    for path, expected in bindings.items():
        require(common.sha(Path(path).read_bytes()) == expected, "input/source/tokenizer drift")
    common.write(output / "report.json", result)
    return result


def main():
    import argparse
    import json
    from pathlib import Path
    import probe_refiner_projection_export as common
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--output", type=Path, default=common.STAGE / "source-alignment-run-01")
    args = parser.parse_args()
    result = run(args.protocol_sha, args.output)
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
