"""Bind a known parent-max control to the existing two-case input, without scores."""

import hashlib
import json
from pathlib import Path

from granularity_lift_control import prepare_parent_layout


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/research-foundation/offline-20261005/candidate-granularity-mechanism-01"
INPUT_SHA = "aa5ce96858b5dae7ef50ac2c3c039d50d225bcbb4d10ebbfa6915d0bcef957e3"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    raw = (BASE / "selected_inputs.json").read_bytes()
    assert sha(raw) == INPUT_SHA, "frozen input changed"
    data = json.loads(raw)
    assert data["schema"] == "slac-candidate-granularity-reranker-input-v1"
    assert len(data["cases"]) == 2 and len(data["pairs"]) == 117
    pairs = {pair["task_id"]: pair for pair in data["pairs"]}
    assert len(pairs) == 117
    layouts, counts, used = [], [], set()
    assert {case["ordinal"] for case in data["cases"]} == {1, 2}
    for case in sorted(data["cases"], key=lambda value: value["ordinal"]):
        units = case["candidates"]["source_units"]
        atoms = case["candidates"]["source_atoms"]
        assert (len(units), len(atoms)) == {1: (16, 52), 2: (14, 45)}[case["ordinal"]]
        layout = prepare_parent_layout(units, atoms)
        for candidate in units + atoms:
            pair = pairs[candidate["task_id"]]
            assert all(pair[key] == case[key] for key in ("doc_id", "question_id", "query"))
            a, b = candidate["span"]
            assert b <= len(case["source_text"])
            assert pair["passage"] == case["source_text"][a:b]
            used.add(pair["task_id"])
        assert sum(b - a for a, b in (u["span"] for u in units)) == case["candidate_domain_chars"]
        layouts.append({"ordinal": case["ordinal"], "layout": layout})
        sizes = [len(parent["atoms"]) for parent in layout["parents"]]
        counts.append({"ordinal": case["ordinal"], "parents": len(units), "children": len(atoms),
                       "minimum_children_per_parent": min(sizes), "maximum_children_per_parent": max(sizes),
                       "parents_with_multiple_children": sum(n > 1 for n in sizes),
                       "candidate_domain_chars": case["candidate_domain_chars"]})
    assert used == set(pairs), "control must reuse exactly the frozen input universe"
    names = ("granularity_lift_control.py", "prepare_granularity_parent_control.py",
             "GRANULARITY_PARENT_CONTROL_20261005.md", "probe_candidate_granularity.py",
             "CANDIDATE_GRANULARITY_PROTOCOL_20261005.md", "run_granularity_sample_reranker.py")
    bindings = {"docs/research/" + name: sha((ROOT / "docs/research" / name).read_bytes()) for name in names}
    output = BASE / "parent-control-01"
    public_path = ROOT / "docs/research/results/granularity_parent_control_20261005.json"
    assert not output.exists() and not public_path.exists(), "do not overwrite prior records"
    output.mkdir()
    write_new(output / "layouts.json", {"input_sha256": INPUT_SHA, "code_and_protocol_sha256": bindings,
                                         "cases": layouts})
    result = {"schema": "slac-parent-max-preparation-v1", "status": "prepared_no_fresh_scores",
              "input_sha256": INPUT_SHA, "code_and_protocol_sha256": bindings,
              "private_layout_sha256": sha((output / "layouts.json").read_bytes()), "cases": counts,
              "unique_existing_model_pairs": len(used), "additional_model_pairs_required_by_control": 0,
              "api_calls_this_preparation": 0, "model_calls_this_preparation": 0,
              "tokenizer_calls_this_preparation": 0, "reference_files_read_this_preparation": 0,
              "natural_quality_results": None, "three_arm_execution_integrated": False}
    write_new(output / "report.json", result)
    write_new(public_path, result)
    print(json.dumps({"status": result["status"], "cases": counts, "unique_pairs": len(used)}))


if __name__ == "__main__":
    main()
