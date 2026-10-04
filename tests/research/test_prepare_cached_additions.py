"""Synthetic nesting, metadata lineage and all-or-nothing manifest checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import prepare_cached_additions as builder


def fixture():
    units = [{"unit_id": f"u{i}", "order": i} for i in range(3)]
    queries = [{"family_id": f"f{i}", "doc_id": "d", "question_id": f"q{i}",
                "candidate_ids": ["u0", "u1", "u2"]} for i in range(2)]
    tasks = [{"doc_id": "d", "question_id": q["question_id"], "unit_id": uid, "id": f"task-{q['question_id']}-{uid}"}
             for q in queries for uid in q["candidate_ids"]]
    prepared = {"documents": {"d": units}, "queries": queries, "support_tasks": tasks}

    def pack(query, ids):
        return {**{k: query[k] for k in builder.IDENTITY}, "selected_ids": ids,
                "pack_sha256": builder.sha(builder.canonical({"units": ids})),
                "cache_key": builder.sha(builder.canonical({"question": query["question_id"], "units": ids})),
                "actual_evidence_tokens": len(ids) * 10, "method": "synthetic"}

    rows = [pack(queries[0], ids) for ids in ([], ["u0"], ["u1"], ["u0", "u1"])]
    rows += [pack(queries[1], ids) for ids in (["u0"], ["u0", "u2"])]
    source = {"origin": "local", "path": "synthetic/mapping.jsonl", "rows": rows}
    return prepared, [source]


def test_all_edges_empty_endpoints_complete_square_and_support_join():
    prepared, sources = fixture()
    edges, controls, sample = builder.build_manifest(prepared, sources)
    assert len(edges) == controls["all"]["paired_edges"] == 5
    assert controls["empty_base"]["paired_edges"] == 2
    assert controls["nonempty_base"]["paired_edges"] == 3
    assert controls["all"]["questions"] == controls["all"]["families"] == 2
    assert controls["complete_inclusion_squares"] == 1
    assert controls["unique_query_packs"] == 6
    for edge in edges:
        assert set(edge["base"]["selected_ids"]) < set(edge["larger"]["selected_ids"])
        assert set(edge["larger"]["selected_ids"]) - set(edge["base"]["selected_ids"]) == {edge["added_unit_id"]}
        assert edge["added_support_task_id"] == f"task-{edge['question_id']}-{edge['added_unit_id']}"
        assert edge["base"]["sources"][0]["mapping_path"] == "synthetic/mapping.jsonl"
    assert len(sample) == 2


def test_duplicate_source_packs_merge_provenance_without_replicating_edges():
    prepared, sources = fixture()
    duplicate = deepcopy(sources[0])
    duplicate["origin"], duplicate["path"] = "primary", "synthetic/second.jsonl"
    edges, controls, _ = builder.build_manifest(prepared, sources + [duplicate])
    assert len(edges) == 5 and controls["unique_payload_cache_keys"] == 6
    assert all({p["origin"] for p in edge["base"]["sources"]} == {"local", "primary"} for edge in edges)


def test_missing_square_endpoint_does_not_impute_a_counterfactual():
    prepared, sources = fixture()
    sources[0]["rows"].pop(3)
    edges, controls, _ = builder.build_manifest(prepared, sources)
    assert len(edges) == 3 and controls["complete_inclusion_squares"] == 0
    assert controls["question_count"] == 2


def test_outside_candidate_pack_is_only_structural_exclusion():
    prepared, sources = fixture()
    outside = deepcopy(sources[0]["rows"][1])
    outside["selected_ids"] = ["outside-original-pool"]
    sources[0]["rows"].append(outside)
    edges, controls, _ = builder.build_manifest(prepared, sources)
    assert len(edges) == 5 and controls["excluded_outside_common_candidate_pool_rows"] == 1
    assert controls["question_count"] == 2


@pytest.mark.parametrize("change", ["same_pack_conflict", "cache_cross_question", "missing_task", "wrong_order", "unknown_question"])
def test_conflicting_or_incomplete_identity_bindings_reject_entire_manifest(change):
    prepared, sources = fixture()
    rows = sources[0]["rows"]
    if change == "same_pack_conflict":
        repeated = deepcopy(rows[0]); repeated["cache_key"] = "f" * 64; rows.append(repeated)
    elif change == "cache_cross_question": rows[4]["cache_key"] = rows[1]["cache_key"]
    elif change == "missing_task": prepared["support_tasks"].pop()
    elif change == "wrong_order": rows[3]["selected_ids"].reverse()
    else: rows[0]["question_id"] = "unknown"
    with pytest.raises(ValueError):
        builder.build_manifest(prepared, sources)


def test_full_experiment_counts_cannot_be_relaxed_to_available_subset():
    prepared, sources = fixture()
    _, controls, _ = builder.build_manifest(prepared, sources)
    with pytest.raises(ValueError, match="inventory"):
        builder.verify_controls(controls, {"question_denominator": 77, "family_denominator": 24})


def test_wrong_inventory_hash_stops_without_any_other_data_read(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ARTIFACTS", tmp_path)
    source = tmp_path / builder.INVENTORY
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    # No contract, mappings, support, references or quality files exist here.
    with pytest.raises(ValueError, match="metadata source"):
        builder.prepare(tmp_path / "out")
    assert not (tmp_path / "out/edges.json").exists()


def test_order_changes_preserve_edge_membership_and_hash_identity():
    prepared, sources = fixture()
    original, _, _ = builder.build_manifest(prepared, sources)
    prepared["queries"].reverse()
    sources[0]["rows"].reverse()
    permuted, _, _ = builder.build_manifest(prepared, sources)
    # Source row ordinals necessarily change, but edge identity must not.
    assert [e["edge_id"] for e in original] == [e["edge_id"] for e in permuted]


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate JSON"):
        builder.parse(b'{"field": 1, "field": 2}')
