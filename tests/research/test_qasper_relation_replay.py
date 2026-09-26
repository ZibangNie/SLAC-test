"""Offline policy contracts; no data files, gold annotations or model downloads."""
import inspect
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
from qasper_relation_replay import AdjacentRelation, replay_policy
from run_qasper_evidence_baselines import PackCounter, Unit, render_pack


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [0, *text, 1]


def document(texts):
    units, cursor = [], 0
    for index, text in enumerate(texts):
        units.append(Unit(f"u{index}", index, "paragraph", cursor, cursor + len(text), text, text))
        cursor += len(text) + 2
    return units


def run(units, *, mode="S", labels=None, relations=(), ranking=None, candidates=None, **kwargs):
    candidates = candidates or [unit.unit_id for unit in units]
    return replay_policy(units, candidates,
                         labels or {unit_id: "yes" for unit_id in candidates}, relations,
                         ranking or candidates, mode=mode, tokenizer=CharacterTokenizer(), **kwargs)


def test_i_and_c_identical_for_same_labels_including_trace_and_no_hidden_gold_argument():
    units = document(["seed", "near", "far", "end"])
    edges = [AdjacentRelation("u0", "u1", "dependent")]
    independent = run(units, mode="I", relations=edges, ranking=["u0", "u2", "u3", "u1"])
    cached = run(units, mode="C", relations=edges, ranking=["u0", "u2", "u3", "u1"])
    assert independent.pop("mode") == "I" and cached.pop("mode") == "C"
    assert independent == cached
    assert independent["gold_consumed"] is False and independent["api_calls"] == 0
    assert not any(word in inspect.signature(replay_policy).parameters
                   for word in ("question", "gold", "answers", "annotations", "references"))
    with pytest.raises(TypeError):
        replay_policy(units, ["u0"], {"u0": "yes"}, [], ["u0"],
                      mode="I", tokenizer=CharacterTokenizer(), gold={})


def test_same_edge_changes_boundary_and_final_evidence_selection_with_unchanged_information():
    units = document(["seed", "near", "far", "end"])
    edges = [AdjacentRelation("u0", "u1", "dependent")]
    options = dict(relations=edges, ranking=["u0", "u2", "u3", "u1"])
    shared, independent = run(units, **options), run(units, mode="I", **options)
    assert shared["candidate_information_sha256"] == independent["candidate_information_sha256"]
    assert shared["chunks"] == [["u0", "u1"], ["u2"], ["u3"]]
    assert shared["selected_ids"] == ["u0", "u1", "u2"]
    assert independent["selected_ids"] == ["u0", "u2", "u3"]
    edge_id = edges[0].edge_id
    assert shared["boundary_trace"][0]["changed_boundary"] is True
    changed = shared["selection_trace"][1]
    assert changed["relation_edge_ids"] == [edge_id] and changed["relation_changed_priority"] is True
    assert shared["selection_bonus_used_edge_ids"] == [edge_id]
    assert shared["selection_symmetric_difference_ids"] == ["u1", "u3"]
    assert shared["added_vs_independent_ids"] == ["u1"]
    assert shared["removed_vs_independent_ids"] == ["u3"]


@pytest.mark.parametrize("relation_label", [None, "independent", "unknown"])
def test_zero_active_edges_degenerate_to_independent_selection_and_partition(relation_label):
    units = document(["a", "b", "c", "d"])
    edges = [] if relation_label is None else [AdjacentRelation("u0", "u1", relation_label)]
    independent = run(units, mode="I", relations=edges)
    shared = run(units, relations=edges)
    for key in ("chunks", "selected_ids", "actual_evidence_tokens", "pack_sha256", "selection_trace"):
        assert shared[key] == independent[key]
    assert shared["selection_symmetric_difference_count"] == 0


def test_chunk_budget_rejection_cannot_leak_an_edge_into_evidence_selection():
    units = document(["aaa", "bbb", "ccc", "ddd"])
    result = run(units, relations=[AdjacentRelation("u0", "u1", "dependent")],
                 ranking=["u0", "u2", "u3", "u1"], chunk_budget=10)
    assert result["boundary_trace"][0]["action"] == "keep_boundary_chunk_budget"
    assert result["accepted_merge_edge_ids"] == []
    assert result["selected_ids"] == ["u0", "u2", "u3"]
    assert result["selection_symmetric_difference_count"] == 0


def test_rendered_whole_pack_with_ids_separators_and_specials_controls_hard_budget():
    units = document(["a", "b", "c"])
    counter = PackCounter(CharacterTokenizer(), units)
    assert counter([0]) == 8 and counter([0, 1]) == 16
    result = run(units, budget=15, chunk_budget=15,
                 relations=[AdjacentRelation("u0", "u1", "dependent")])
    assert result["actual_evidence_tokens"] == 8
    assert result["selected_ids"] == ["u0"]
    assert result["boundary_trace"][0]["proposed_tokens"] == 16
    assert [row["action"] for row in result["selection_trace"]] == ["select", "skip_evidence_budget", "skip_evidence_budget"]
    assert "[u0]\na" == render_pack(units, result["selected_indices"])


def test_overlong_native_units_remain_whole_and_are_reported_then_skipped():
    units = document(["a" * 40, "short"])
    result = run(units, budget=20, chunk_budget=10)
    assert result["oversize_singleton_ids"] == ["u0", "u1"]
    assert result["selected_ids"] == ["u1"]
    assert result["selection_trace"][0]["action"] == "skip_evidence_budget"
    assert result["actual_evidence_tokens"] == 12


def test_no_is_excluded_even_for_dependent_edge_but_unknown_can_receive_bonus():
    units = document(["seed", "unknown", "no", "other"])
    labels = {"u0": "yes", "u1": "unknown", "u2": "no", "u3": "yes"}
    result = run(units, labels=labels,
                 relations=[AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")],
                 ranking=["u0", "u3", "u2", "u1"], max_units=2)
    assert result["selected_ids"] == ["u0", "u1"]
    assert result["selection_trace"][1]["effective_score"] == 1.5
    assert result["excluded_no_ids"] == ["u2"]
    assert "u2" not in [row["unit_id"] for row in result["selection_trace"]]


def test_all_no_abstains_and_duplicate_native_strings_are_not_selected_twice():
    units = document(["same", "same", "other", "last"])
    empty = run(units, labels={u.unit_id: "no" for u in units})
    assert empty["selected_ids"] == [] and empty["actual_evidence_tokens"] == 0
    result = run(units)
    assert result["selected_ids"] == ["u0", "u2", "u3"]
    assert result["selection_trace"][1]["action"] == "skip_exact_native_duplicate"


def test_candidate_and_relation_input_order_do_not_change_source_order_or_decisions():
    units = document(["a", "b", "c", "d"])
    edges = [AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")]
    first = run(units, relations=edges, ranking=["u2", "u3", "u0", "u1"])
    second = run(units, relations=list(reversed(edges)), candidates=["u3", "u0", "u2", "u1"],
                 ranking=["u2", "u3", "u0", "u1"])
    assert first == second
    assert first["selected_ids"] == sorted(first["selected_ids"])


def test_bonus_on_accepted_edge_does_not_imply_final_selected_set_changed():
    units = document(["seed", "near"])
    result = run(units, relations=[AdjacentRelation("u0", "u1", "dependent")])
    assert result["accepted_merge_edge_ids"] and result["selection_bonus_used_edge_ids"]
    assert result["selection_symmetric_difference_count"] == 0


@pytest.mark.parametrize("edges,match", [
    ([AdjacentRelation("u0", "u2", "dependent")], "truly adjacent"),
    ([AdjacentRelation("u1", "u0", "dependent")], "truly adjacent"),
    ([AdjacentRelation("u0", "u1", "dependent")] * 2, "duplicate relation"),
    ([AdjacentRelation("u0", "u1", "almost")], "relation labels"),
    ([AdjacentRelation("u0", "missing", "dependent")], "outside"),
])
def test_relations_reject_wrong_adjacency_duplicates_unknown_endpoints_and_labels(edges, match):
    with pytest.raises(ValueError, match=match):
        run(document(["a", "b", "c"]), relations=edges)


def test_sparse_candidate_pool_cannot_make_nonadjacent_native_units_adjacent():
    units = document(["left", "omitted", "right"])
    with pytest.raises(ValueError, match="truly adjacent"):
        run(units, candidates=["u0", "u2"], relations=[AdjacentRelation("u0", "u2", "dependent")])


@pytest.mark.parametrize("options,match", [
    ({"labels": {"u0": "yes"}}, "exactly cover"),
    ({"labels": {"u0": "maybe", "u1": "yes"}}, "relevance labels"),
    ({"ranking": ["u0", "u0"]}, "permutation"),
    ({"candidates": ["u0", "u0"]}, "candidate IDs"),
    ({"budget": 0}, "budget"), ({"chunk_budget": -1}, "chunk_budget"),
    ({"max_units": True}, "max_units"), ({"mode": "shared"}, "mode"),
])
def test_invalid_frozen_policy_inputs_fail_closed(options, match):
    with pytest.raises(ValueError, match=match):
        run(document(["a", "b"]), **options)


def test_complete_document_sequence_and_deadline_are_enforced():
    units = document(["a", "b", "c"])
    with pytest.raises(ValueError, match="complete document"):
        run([units[0], units[2]])
    with pytest.raises(TimeoutError):
        run(units, deadline=time.monotonic() - 1)
