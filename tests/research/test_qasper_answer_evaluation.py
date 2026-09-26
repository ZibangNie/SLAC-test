"""Synthetic fixtures for bounded, shared-generator evaluation."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_answer_evaluation as answers


def fixture():
    unit = answers.Unit("u1", 0, "paragraph", 0, 4, "Blue", "Blue")
    prepared = {"documents": {"doc": [asdict(unit)]}, "queries": [{
        "doc_id": "doc", "family_id": "family", "question_id": "q", "query": "What color?",
        "candidate_ids": ["u1"], "ranked_ids": ["u1"]}]}
    pack = answers.render_pack([unit], [0])
    records = [{"doc_id": "doc", "family_id": "family", "question_id": "q", "method": method,
                "selected_ids": ["u1"], "budget": 1024, "actual_evidence_tokens": 6,
                "pack_sha256": hashlib.sha256(pack.encode()).hexdigest()}
               for method in answers.METHODS if method != "empty"]
    annotations = {("doc", "q"): [{"native_answer": {"unanswerable": False,
        "extractive_spans": ["Blue"], "free_form_answer": "", "yes_no": None, "evidence": ["Blue"]}}]}
    return prepared, records, annotations


def response(answer="Blue", **changes):
    return {"model": answers.MODEL, "provider": "Alibaba",
            "usage": {"cost": .0001, "prompt_tokens": 40, "completion_tokens": 8},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answer": answer})}}], **changes}


def test_exact_payload_dedup_and_empty_baseline_are_complete():
    prepared, records, _ = fixture()
    jobs, mapping = answers.build_jobs(prepared, records)
    assert len(jobs) == 2 and len(mapping) == 6
    assert len({row["cache_key"] for row in mapping if row["method"] != "empty"}) == 1
    assert {row["method"] for row in mapping} == set(answers.METHODS)
    inputs = [json.loads(job["payload"]["messages"][1]["content"]) for job in jobs]
    assert any(item["evidence"] == "" for item in inputs)
    assert all(set(item) == {"question", "evidence"} for item in inputs)
    assert all("Blue" not in job["payload"]["messages"][0]["content"] for job in jobs)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "changed_hash", "noncandidate", "too_many"])
def test_incomplete_or_tampered_evidence_is_rejected(mutation):
    prepared, records, _ = fixture()
    if mutation == "missing": records.pop()
    if mutation == "duplicate": records.append(deepcopy(records[0]))
    if mutation == "changed_hash": records[0]["pack_sha256"] = "wrong"
    if mutation == "noncandidate": records[0]["selected_ids"] = ["unknown"]
    if mutation == "too_many": records[0]["selected_ids"] = ["u1"] * 4
    with pytest.raises(ValueError): answers.build_jobs(prepared, records)


@pytest.mark.parametrize("content,finish", [('{"answer":"X","extra":1}', "stop"),
    ('{"answer":"X","answer":"Y"}', "stop"), ('{"answer":""}', "stop"),
    ('{"answer":"X"}', "length"), ('```json\n{"answer":"X"}\n```', "stop")])
def test_invalid_or_truncated_answers_are_not_repaired(content, finish):
    item = response()
    item["choices"][0] = {"finish_reason": finish, "message": {"content": content}}
    with pytest.raises(ValueError): answers.parse_answer(item)


def test_no_answer_normalization_before_official_scoring():
    assert answers.parse_answer(response(" The, Blue! ")) == " The, Blue! "
    prepared, records, gold = fixture()
    jobs, mapping = answers.build_jobs(prepared, records)
    predicted = {job["cache_key"]: " The, Blue! " for job in jobs}
    rows, summary = answers.score_answers(prepared, mapping, predicted, gold)
    assert len(rows) == 6
    assert all(row["official_answer_f1"] == 1 for row in rows)
    assert all(row["official_metrics"]["Answer F1"] == 1 for row in summary["metrics"])
    with pytest.raises(ValueError): answers.score_answers(prepared, mapping, {}, gold)


def test_night_reservation_includes_prior_stage(tmp_path):
    jobs, _ = answers.build_jobs(*fixture()[:2])
    with pytest.raises(ValueError):
        answers.AnswerClient(tmp_path / "calls", prior_reservation="4.99999", jobs=jobs, transport=lambda _: response())
    assert not (tmp_path / "calls").exists()


def test_duplicate_request_refused_and_no_auth_saved(tmp_path):
    jobs, _ = answers.build_jobs(*fixture()[:2])
    transport_calls = []
    def transport(payload):
        transport_calls.append(payload)
        return response()
    bounded = answers.AnswerClient(tmp_path / "calls", prior_reservation="3", jobs=jobs, transport=transport)
    assert bounded.submit(jobs[0]) == "Blue"
    with pytest.raises(ValueError): bounded.submit(jobs[0])
    assert len(transport_calls) == 1
    assert Decimal(bounded.ledger["reservation_total_usd"]) == Decimal(jobs[0]["reserved_usd"])
    assert all(b"test-credential" not in path.read_bytes() for path in (tmp_path / "calls").iterdir())


def test_ambiguous_failure_stays_charged_and_prevents_any_more_calls(tmp_path):
    jobs, _ = answers.build_jobs(*fixture()[:2])
    calls = []
    def transport(payload):
        calls.append(payload)
        raise TimeoutError("sensitive arbitrary exception")
    bounded = answers.AnswerClient(tmp_path / "calls", prior_reservation="3", jobs=jobs, transport=transport)
    with pytest.raises(RuntimeError, match="TimeoutError"): bounded.submit(jobs[0])
    with pytest.raises(ValueError): bounded.submit(jobs[1])
    assert len(calls) == 1
    assert bounded.ledger["attempts"][0]["status"] == "halted"
    assert "actual_cost_usd" not in bounded.ledger["attempts"][0]
    assert Decimal(bounded.ledger["reservation_total_usd"]) > 0
    assert "sensitive arbitrary" not in (tmp_path / "calls/ledger.json").read_text()


@pytest.mark.parametrize("change", [{"model": "other/model"}, {"provider": "other"},
    {"usage": {"cost": 10, "prompt_tokens": 40, "completion_tokens": 8}},
    {"usage": {"cost": .0001, "prompt_tokens": True, "completion_tokens": 8}}])
def test_response_identity_usage_and_cost_fail_closed(tmp_path, change):
    jobs, _ = answers.build_jobs(*fixture()[:2])
    bounded = answers.AnswerClient(tmp_path / "calls", prior_reservation="3", jobs=jobs,
                                   transport=lambda _: response(**change))
    with pytest.raises(RuntimeError): bounded.submit(jobs[0])
    assert bounded.ledger["halt_reason"]


def test_saved_response_redacts_credential_echo(tmp_path):
    jobs, _ = answers.build_jobs(*fixture()[:2])
    bounded = answers.AnswerClient(tmp_path / "calls", prior_reservation="3", jobs=jobs,
                                   transport=lambda _: response("test-credential"))
    assert bounded.submit(jobs[0]) == "[REDACTED]"
    assert "test-credential" not in (tmp_path / "calls/response_001.json").read_text()


def fake_plan(tmp_path, monkeypatch):
    prepared, records, gold = fixture()
    jobs, mapping = answers.build_jobs(prepared, records)
    plan_dir, support_run = tmp_path / "plan", tmp_path / "support"
    plan_dir.mkdir()
    support_run.mkdir()
    (plan_dir / "experiment_config.json").write_text('{}')
    (plan_dir / "plan_manifest.json").write_text('{}')
    inputs = tmp_path / "input.txt"
    inputs.write_text('frozen')
    total = sum((Decimal(job["reserved_usd"]) for job in jobs), Decimal("0"))
    config = {"support_plan": str(plan_dir), "support_run": str(support_run), "sidecar": str(tmp_path / "unused"),
              "prior_night_accounting": {"reservation_usd": "3"},
              "answer_reservation_usd": str(total), "night_reservation_usd": str(Decimal("3") + total),
              "input_sha256": {str(inputs): answers.digest(inputs)}, "limits": []}
    monkeypatch.setattr(answers, "load_plan", lambda _: (config, prepared, jobs, mapping))
    monkeypatch.setattr(answers.pilot, "selected_gold", lambda *_: gold)
    args = SimpleNamespace(plan=str(plan_dir), output=str(tmp_path / "run"), key_file=None, proxy=None)
    return args, config, jobs, inputs


def test_full_synthetic_run_and_independent_response_audit(tmp_path, monkeypatch):
    args, config, jobs, _ = fake_plan(tmp_path, monkeypatch)
    def factory(path, **kwargs):
        return answers.AnswerClient(path, **kwargs, transport=lambda _: response())
    result = answers.run(args, client_factory=factory)
    assert result["status"] == "completed" and result["new_api_calls"] == 2
    verified = answers.audit(SimpleNamespace(plan=args.plan, run=args.output))
    assert verified["status"] == "verified" and verified["record_count"] == 6
    ledger_path = Path(args.output) / "provider_calls/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["attempts"][0]["actual_cost_usd"] = "0"
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError): answers.verify_calls(config, jobs, args.output)


def test_failure_cannot_reset_night_accounting_via_new_output_name(tmp_path, monkeypatch):
    args, _, _, _ = fake_plan(tmp_path, monkeypatch)
    calls = []
    def transport(_):
        calls.append(1)
        raise TimeoutError()
    def factory(path, **kwargs):
        return answers.AnswerClient(path, **kwargs, transport=transport)
    with pytest.raises(RuntimeError): answers.run(args, client_factory=factory)
    args.output = str(tmp_path / "retry")
    with pytest.raises(FileExistsError): answers.run(args, client_factory=factory)
    assert len(calls) == 1 and not Path(args.output).exists()


def test_input_change_during_generation_prevents_publishing_results(tmp_path, monkeypatch):
    args, _, _, inputs = fake_plan(tmp_path, monkeypatch)
    def transport(_):
        inputs.write_text('changed')
        return response()
    def factory(path, **kwargs):
        return answers.AnswerClient(path, **kwargs, transport=transport)
    with pytest.raises(ValueError): answers.run(args, client_factory=factory)
    assert (Path(args.output) / "failure.json").exists()
    assert not (Path(args.output) / "summary.json").exists()


@pytest.mark.parametrize("target", ["extra_request", "extra_response", "metadata", "binding", "model"])
def test_audit_rejects_unaccounted_files_and_all_summary_metadata(tmp_path, monkeypatch, target):
    args, _, _, _ = fake_plan(tmp_path, monkeypatch)
    def factory(path, **kwargs):
        return answers.AnswerClient(path, **kwargs, transport=lambda _: response())
    answers.run(args, client_factory=factory)
    output = Path(args.output)
    if target.startswith("extra_"):
        (output / "provider_calls" / (target.removeprefix("extra_") + "_999.json")).write_text('{}')
    else:
        path = output / "summary.json"
        summary = json.loads(path.read_text())
        if target == "metadata": summary["new_api_calls"] = 999
        if target == "binding": summary["input_binding_sha256"] = "wrong"
        if target == "model": summary["resolved_models"] = {"generator": "different/model"}
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError): answers.audit(SimpleNamespace(plan=args.plan, run=args.output))


def test_response_change_after_parse_is_not_rebound_as_verified(tmp_path, monkeypatch):
    args, _, _, _ = fake_plan(tmp_path, monkeypatch)
    def factory(path, **kwargs):
        return answers.AnswerClient(path, **kwargs, transport=lambda _: response())
    answers.run(args, client_factory=factory)
    original = answers.parse_answer
    changed = False
    def parse_then_mutate(value):
        nonlocal changed
        result = original(value)
        if not changed:
            changed = True
            path = Path(args.output) / "provider_calls/response_001.json"
            path.write_text(json.dumps(response("Red")))
        return result
    monkeypatch.setattr(answers, "parse_answer", parse_then_mutate)
    with pytest.raises(ValueError): answers.audit(SimpleNamespace(plan=args.plan, run=args.output))


def test_plan_change_during_load_is_refused_before_registration(tmp_path, monkeypatch):
    args, _, _, _ = fake_plan(tmp_path, monkeypatch)
    original = answers.load_plan
    def load_then_mutate(path):
        result = original(path)
        (Path(path) / "experiment_config.json").write_text('{"changed":true}')
        return result
    monkeypatch.setattr(answers, "load_plan", load_then_mutate)
    with pytest.raises(ValueError): answers.run(args)
    assert not Path(args.output).exists()
    assert not (Path(args.plan) / "answer_evaluation_registration.json").exists()
