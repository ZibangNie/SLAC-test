"""Artificial-only contract checks; none score model semantic predictions."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_conditional_contract as runner


@pytest.fixture
def fixture():
    return json.loads(runner.FIXTURE_PATH.read_bytes())


def test_complete_synthetic_contract_and_no_network(fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network is forbidden in synthetic contract")
    monkeypatch.setattr(socket, "socket", forbidden)
    summary, records = runner.run_contract(fixture)
    assert summary["pair_count"] == 8
    assert summary["case_count"] == 16
    assert summary["request_records"] == 50
    assert summary["unique_request_keys"] == 42
    assert summary["variant_counts"] == {"standalone": 16, "plain_conditional": 16,
                                         "relation_conditioned": 9, "permuted_relation_control": 9}
    assert summary["contract_checks"] == {
        "supervision_isolation_cases": 16, "arm_isolation_cases": 16,
        "exact_budget_acceptances": 50, "over_budget_rejections": 50,
        "fake_transport_calls": 42, "cache_hits": 8,
        "counterfactual_pairs": 8, "relation_controls": 9,
        "malformed_response_probes": 6, "stale_cache_rejections": 1,
        "cache_hit_transport_suppression": 1,
    }
    assert len(records) == 50
    assert summary["api_calls"] == summary["model_inference_calls"] == 0
    assert not summary["real_dataset_read"]
    assert "NOT model predictions" in summary["interpretation"]
    assert not any("accuracy" in key for key in summary)


def test_counterfactual_state_changes_only_conditional_key(fixture):
    for pair in fixture["pairs"]:
        left, right = [runner.variants(pair, case) for case in pair["states"]]
        assert left["standalone"].payload_bytes == right["standalone"].payload_bytes
        assert left["standalone"].cache_key == right["standalone"].cache_key
        assert left["plain_conditional"].cache_key != right["plain_conditional"].cache_key
        a, b = [item["plain_conditional"].payload()["state"] for item in (left, right)]
        assert a["query"] == b["query"]
        assert a["candidate"] == b["candidate"]
        assert a["current_pack"] != b["current_pack"]


def test_supervision_and_fixture_identifiers_are_not_model_inputs(fixture):
    pair = fixture["pairs"][2]
    before = runner.variants(pair, pair["states"][0])
    changed = deepcopy(pair)
    changed["pair_id"] = "FORBIDDEN_PAIR"
    changed["supervision"] = {"expected_labels_by_case": {"FORBIDDEN_CASE": "FORBIDDEN_LABEL"},
                              "fixture_group": "FORBIDDEN_GROUP"}
    changed["states"][0]["case_id"] = "FORBIDDEN_CASE"
    changed["states"][0]["expected"] = "FORBIDDEN_INLINE_LABEL"
    after = runner.variants(changed, changed["states"][0])
    for key, request in before.items():
        assert request.payload_bytes == after[key].payload_bytes
        assert request.cache_key == after[key].cache_key
        assert b"FORBIDDEN" not in after[key].transport_bytes()


def test_fixed_fake_values_are_independent_of_fixture_semantics(fixture):
    for pair in fixture["pairs"]:
        for case in pair["states"]:
            for request in runner.variants(pair, case).values():
                response = runner.fixed_fake_response(request.transport_bytes())
                assert response["response"]["model"] == "typesafe/jev-1.13-20260917"
                assert response["response"]["answers"] == {
                    name: {"type": "choice", "choice": runner.FIXED_CHOICES[name]}
                    for name in request.expected_ids
                }


def test_relation_controls_preserve_text_order_count_and_metadata_bytes(fixture):
    count = 0
    for pair in fixture["pairs"]:
        for case in pair["states"]:
            built = runner.variants(pair, case)
            if not case["relations"]:
                assert set(built) == {"standalone", "plain_conditional"}
                continue
            count += 1
            a = built["relation_conditioned"]
            b = built["permuted_relation_control"]
            runner.check_relation_control(a, b)
            assert len(a.payload_bytes) == len(b.payload_bytes)
            for request in (a, b):
                state = request.payload()["state"]
                visible = {unit["id"]: unit for unit in state["current_pack"] + [state["candidate"]]}
                for relation in state["relations"]:
                    anchor = relation["anchor"]
                    assert relation["prerequisite_id"] in visible
                    dependent = visible[relation["dependent_id"]]["text"]
                    assert dependent[anchor["start"]:anchor["end"]] == anchor["text"]
            assert set(a.payload()["state"]) == {"query", "current_pack", "candidate", "relations"}
    assert count == 9


def test_scope_fixture_maps_to_core_qualifier(fixture):
    pair = fixture["pairs"][6]
    assert runner.project_state(pair, pair["states"][0]).relations[0].kind == "qualifier"


def test_budget_counts_exact_evidence_not_full_payload(fixture):
    pair, case = fixture["pairs"][0], fixture["pairs"][0]["states"][0]
    state = runner.project_state(pair, case)
    plain = runner.request_for(state, "plain_conditional")
    alone = runner.request_for(state, "standalone")
    assert alone.evidence_tokens < plain.evidence_tokens
    assert plain.evidence_tokens == len(plain.rendered_evidence.split())
    assert plain.evidence_tokens != len(plain.payload_bytes.decode().split())
    assert runner.request_for(state, "plain_conditional", max_tokens=plain.evidence_tokens)
    with pytest.raises(ValueError, match="max_tokens"):
        runner.request_for(state, "plain_conditional", max_tokens=plain.evidence_tokens - 1)
    assert runner.request_for(state, "plain_conditional", max_payload_bytes=len(plain.payload_bytes))
    with pytest.raises(ValueError, match="max_payload_bytes"):
        runner.request_for(state, "plain_conditional", max_payload_bytes=len(plain.payload_bytes) - 1)


@pytest.mark.parametrize("probe, valid, added, conflict", [
    ("missing_added", True, "unknown", "no"),
    ("invalid_added_choice", True, "unknown", "no"),
    ("invalid_conflict_type", True, "yes", "unknown"),
    ("wrong_request_key", False, "unknown", "unknown"),
    ("wrong_returned_model", False, "unknown", "unknown"),
    ("extra_answer_id", False, "unknown", "unknown"),
])
def test_dimensions_fail_closed_independently(fixture, probe, valid, added, conflict):
    pair = fixture["pairs"][0]
    request = runner.variants(pair, pair["states"][0])["plain_conditional"]
    result = runner.decode_response(request, runner.malformed_probes(request)[probe])
    assert result.envelope_valid is valid
    choices = {decision.id: decision.choice for decision in result.decisions}
    assert choices == {"conditional_added_information": added, "conflict": conflict}
    if not valid:
        with pytest.raises(ValueError, match="unaligned"):
            runner.freeze_response(request, runner.malformed_probes(request)[probe])


def test_cache_hit_suppresses_transport_and_cross_arm_cache_is_rejected(fixture):
    pair = fixture["pairs"][0]
    built = runner.variants(pair, pair["states"][0])
    cache, calls = {}, []
    def counted(data):
        calls.append(data)
        return runner.fixed_fake_response(data)
    _, first_hit = runner.cached_roundtrip(built["standalone"], cache, counted)
    _, second_hit = runner.cached_roundtrip(built["standalone"], cache, counted)
    assert not first_hit and second_hit and len(calls) == 1
    stale = runner.decode_cached(built["plain_conditional"], cache[built["standalone"].cache_key])
    assert not stale.envelope_valid
    assert {decision.choice for decision in stale.decisions} == {"unknown"}


def test_absent_relation_control_is_not_invented(fixture):
    pair = fixture["pairs"][0]
    case = deepcopy(pair["states"][0])
    case["control_relations"] = [{"kind": "reference", "from": "u002", "to": "u003", "anchor_text": "Orion"}]
    with pytest.raises(ValueError, match="invented"):
        runner.variants(pair, case)


def test_immutable_output_hashes_and_existing_directory_rejection(tmp_path):
    output = tmp_path / "synthetic-contract"
    summary = runner.run_to_directory(output)
    plan = json.loads((output / "plan.json").read_bytes())
    assert summary["protocol_sha256"] == runner.sha(runner.PROTOCOL_PATH.read_bytes())
    for name, expected in plan["artifact_sha256"].items():
        assert runner.sha((output / name).read_bytes()) == expected
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        runner.run_to_directory(output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
