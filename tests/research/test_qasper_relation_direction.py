"""Direction semantics, source verification and saved-development-run parity."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_relation_direction as direction
from qasper_relation_replay import AdjacentRelation, replay_policy
from run_qasper_evidence_baselines import PackCounter
from test_qasper_relation_replay import CharacterTokenizer, document
from test_qasper_relation_analysis import completed


def run(units, edges, ranking, *, mode="prerequisite", labels=None, accepted=None, **kwargs):
    candidates = [unit.unit_id for unit in units]
    return direction.replay_direction(units, candidates, labels or dict.fromkeys(candidates, "yes"),
        edges, ranking, [edge.edge_id for edge in edges] if accepted is None else accepted,
        direction=mode, tokenizer=CharacterTokenizer(), **kwargs)


def test_prerequisite_follows_b_to_a_through_chain_and_reverse_is_placebo():
    units = document(["a", "b", "c", "d", "e"])
    edges = [AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")]
    rank = ["u2", "u3", "u4", "u1", "u0"]
    prerequisite = run(units, edges, rank)
    reverse = run(units, edges, rank, mode="reverse_placebo")
    assert prerequisite["selected_ids"] == ["u0", "u1", "u2"]
    assert [row["unit_id"] for row in prerequisite["selection_trace"]] == ["u2", "u1", "u0"]
    assert reverse["selected_ids"] == ["u2", "u3", "u4"]
    assert all(row["relation_bonus"] in {0, 1} for row in prerequisite["selection_trace"])
    assert prerequisite["selection_priority_changed_steps"] == 2


def test_reverse_traverses_a_to_b_and_prerequisite_does_not():
    units = document(["a", "b", "c", "d", "e"])
    edges = [AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")]
    rank = ["u0", "u3", "u4", "u1", "u2"]
    assert run(units, edges, rank)["selected_ids"] == ["u0", "u3", "u4"]
    assert run(units, edges, rank, mode="reverse_placebo")["selected_ids"] == ["u0", "u1", "u2"]


@pytest.mark.parametrize("mode", direction.DIRECTIONS)
def test_no_excluded_unknown_half_weight_and_all_no_abstains(mode):
    units = document(["no", "maybe", "seed", "other"])
    edges = [AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")]
    rank = ["u2", "u3", "u1", "u0"]
    labels = {"u0": "no", "u1": "unknown", "u2": "yes", "u3": "yes"}
    result = run(units, edges, rank, mode=mode, labels=labels)
    assert result["eligible_ids"] == ["u1", "u2", "u3"]
    assert result["excluded_no_ids"] == ["u0"]
    assert "u0" not in result["selected_ids"]
    assert next(row for row in result["selection_trace"] if row["unit_id"] == "u1")["base_score"] == .5
    if mode in {"prerequisite", "symmetric"}:
        assert result["bonus_opportunities"][-1]["excluded_no_linked_ids"] == ["u0"]
    abstain = run(units, edges, rank, mode=mode, labels=dict.fromkeys(rank, "no"))
    assert abstain["selected_ids"] == [] and abstain["actual_evidence_tokens"] == 0


@pytest.mark.parametrize("mode", direction.DIRECTIONS)
def test_full_render_token_cap_and_native_duplicate(mode):
    units = document(["same", "same", "large" * 30, "end"])
    edges = [AdjacentRelation("u0", "u1", "dependent")]
    count = PackCounter(CharacterTokenizer(), units)
    result = run(units, edges, ["u0", "u1", "u2", "u3"], mode=mode, budget=count([0, 3]))
    assert result["selected_ids"] == ["u0", "u3"]
    assert result["actual_evidence_tokens"] == count([0, 3])
    assert [row["action"] for row in result["selection_trace"]] == [
        "select", "skip_exact_native_duplicate", "skip_evidence_budget", "select"]


def test_fixed_accepted_set_cannot_be_expanded_or_use_unknown_edges():
    units = document(["a", "b", "c", "d"])
    edges = [AdjacentRelation("u0", "u1", "dependent")]
    rank = ["u0", "u2", "u3", "u1"]
    assert run(units, edges, rank, mode="symmetric", accepted=[])["selected_ids"] == ["u0", "u2", "u3"]
    for accepted in (["missing"], [edges[0].edge_id] * 2):
        with pytest.raises(ValueError, match="accepted edges"):
            run(units, edges, rank, accepted=accepted)
    with pytest.raises(ValueError, match="accepted edges"):
        run(units, [AdjacentRelation("u0", "u1", "unknown")], rank)
    with pytest.raises(ValueError, match="unsupported direction"):
        run(units, edges, rank, mode="invented")


def test_symmetric_all_steps_equal_original_and_tamper_is_refused():
    units = document(["a", "b", "c", "d"])
    edges = [AdjacentRelation("u0", "u1", "dependent"), AdjacentRelation("u1", "u2", "dependent")]
    rank = ["u0", "u2", "u3", "u1"]
    labels = dict.fromkeys(rank, "yes")
    original = replay_policy(units, rank, labels, edges, rank, mode="S", tokenizer=CharacterTokenizer())
    actual = run(units, edges, rank, mode="symmetric", accepted=original["accepted_merge_edge_ids"])
    direction.require_symmetric_parity(actual, original)
    tampered = deepcopy(actual)
    tampered["selection_trace"][1]["effective_score"] += 1
    with pytest.raises(ValueError, match="symmetric replay"):
        direction.require_symmetric_parity(tampered, original)


def test_complete_analysis_verifies_source_and_only_aggregate_is_public(completed, monkeypatch):
    monkeypatch.setattr(direction.pilot.client, "read_key", lambda *_: pytest.fail("credential access forbidden"))
    result = direction.analyze(completed)
    assert result["status"] == "completed" and result["record_count"] == 12
    assert result["symmetric_step_and_metric_parity"] is True
    assert result["api_calls"] == 0 and result["key_read"] is False
    assert result["support_eligibility_unchanged"] is True
    assert result["input_hashes_unchanged"] is True
    text = (completed.output / "analysis.json").read_text(encoding="utf-8")
    for forbidden in ('"q0"', '"u0"', "FORBIDDEN_QA_ANSWER", "What is 0?", "Evidence 0", str(completed.run)):
        assert forbidden not in text
    assert (completed.output / "source_verification" / "analysis.json").is_file()
    with pytest.raises(FileExistsError):
        direction.analyze(completed)


def test_changed_source_trace_refused_before_direction_outputs(completed):
    path = completed.run / "traces.jsonl"
    rows = direction.source.read_rows(path)
    rows[0]["selected_ids"] = []
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="saved policy outputs"):
        direction.analyze(completed)
    assert not (completed.output / "analysis.json").exists()


def test_real_saved_development_run_symmetric_parity(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "artifacts" / "research-foundation"
    plan, run_dir, prepared_dir = (root / name for name in (
        "qasper-relation-plan-03", "qasper-relation-run-03", "qasper-relation-prepared-01"))
    if not (run_dir / "summary.json").exists():
        pytest.skip("ignored local development artifacts unavailable")
    monkeypatch.setattr(direction.pilot.client, "read_key", lambda *_: pytest.fail("credential access forbidden"))
    try:
        config, _ = direction.pilot.load_plan(plan)
        prepared, _, documents = direction.pilot.load_prepared(prepared_dir)
        annotations = direction.pilot.selected_gold(config["sidecar"], prepared)
        labels = direction.pilot.read_json(run_dir / "labels.json")
        original_rows = direction.source.read_rows(run_dir / "per_question.jsonl")
        original_traces = direction.source.read_rows(run_dir / "traces.jsonl")
        tokenizer = direction.pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
        rows, traces = direction.direction_records(prepared, documents, labels, annotations,
                                                   original_rows, original_traces, tokenizer)
        result = direction.summarize(prepared, rows, traces, original_rows)
        assert result["record_count"] == 90 and result["question_count"] == 15
        assert result["symmetric_step_and_metric_parity"] is True
    finally:
        direction.pilot.client.select_general_profile("gpt41mini")
