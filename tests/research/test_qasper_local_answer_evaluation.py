"""Offline science, accounting, single-use and failure tests; no live transport."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_local_answer_evaluation as local


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return list(text)


def fixture():
    unit = local.Unit("u1", 0, "paragraph", 0, 4, "Blue", "Blue")
    outside = local.Unit("u2", 1, "paragraph", 5, 8, "Red", "Red")
    prepared = {"documents": {"doc": [asdict(unit), asdict(outside)]}, "queries": [{
        "doc_id": "doc", "family_id": "family", "question_id": "q", "query": "What color?",
        "candidate_ids": ["u1"], "ranked_ids": ["u1"]}]}
    pack = local.render_pack([unit], [0])
    records = [{"doc_id": "doc", "family_id": "family", "question_id": "q", "method": method,
        "selected_ids": ["u1"], "actual_evidence_tokens": len(pack),
        "pack_sha256": hashlib.sha256(pack.encode()).hexdigest(),
        **({"selected_global_indices": [0], "candidate_global_indices": [0, 1]} if method != "reranker_k3" else {})}
        for method in local.METHODS[:-1]]
    index = [{"doc_id": "doc", "unit_id": uid} for uid in ("u1", "u2")]
    gold = {("doc", "q"): [{"native_answer": {"unanswerable": False, "extractive_spans": ["Blue"],
        "free_form_answer": "", "yes_no": None, "evidence": ["Blue"]}}]}
    return prepared, records, index, gold


def jobs_fixture():
    p, r, i, g = fixture()
    jobs, mapping = local.build_jobs(p, r, i, Tokenizer())
    return p, jobs, mapping, g


def response(answer="Blue", **changes):
    return {"model": local.legacy.MODEL, "provider": "Alibaba",
        "usage": {"cost": .0001, "prompt_tokens": 40, "completion_tokens": 8},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answer": answer})}}], **changes}


def config_for(jobs):
    return {"prior_night_accounting": local.PRIOR.copy(),
        "answer_reservation_usd": str(sum((Decimal(j["reserved_usd"]) for j in jobs), Decimal("0")))}


def client_for(tmp_path, jobs, transport):
    return local.LocalAnswerClient(tmp_path / "provider_calls", prior_reservation=local.PRIOR["reservation_usd"],
        jobs=jobs, transport=transport)


def test_six_methods_share_only_complete_payload_and_actual_empty():
    p, jobs, mapping, _ = jobs_fixture()
    assert len(jobs) == 2 and len(mapping) == 6
    assert {r["method"] for r in mapping} == set(local.METHODS)
    inputs = [json.loads(j["payload"]["messages"][1]["content"]) for j in jobs]
    assert all(set(x) == {"question", "evidence"} for x in inputs)
    assert any(x["evidence"] == "" for x in inputs)
    assert next(r for r in mapping if r["method"] == "empty")["actual_evidence_tokens"] == 0
    assert all(j["payload"] == local.legacy.make_payload(x["question"], x["evidence"]) for j, x in zip(jobs, inputs))
    assert all(j["cache_key"] == local.legacy.client.object_hash({"endpoint": local.legacy.ENDPOINT,
        "prompt_version": local.legacy.PROMPT_VERSION, "payload": j["payload"]}) for j in jobs)


def test_new_retrieval_can_use_own_candidates_outside_old_support_pool():
    p, rows, index, _ = fixture()
    row = next(r for r in rows if r["method"] == "bm25_k3")
    pack = local.render_pack([local.Unit(**u) for u in p["documents"]["doc"]], [1])
    row.update(selected_ids=["u2"], selected_global_indices=[1], actual_evidence_tokens=len(pack),
               pack_sha256=hashlib.sha256(pack.encode()).hexdigest())
    jobs, mapping = local.build_jobs(p, rows, index, Tokenizer())
    assert len(jobs) == 3 and next(r for r in mapping if r["method"] == "bm25_k3")["selected_ids"] == ["u2"]
    index[1]["doc_id"] = "different-source"
    with pytest.raises(ValueError, match="global native identity"):
        local.build_jobs(p, rows, index, Tokenizer())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "tokens", "duplicate_ids", "negative_global", "own_pool", "family"])
def test_incomplete_or_changed_native_inputs_rejected(mutation):
    p, rows, index, _ = fixture()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "hash": rows[0]["pack_sha256"] = "bad"
    elif mutation == "tokens": rows[0]["actual_evidence_tokens"] += 1
    elif mutation == "duplicate_ids": rows[0]["selected_ids"] *= 2
    elif mutation == "negative_global": rows[0]["selected_global_indices"] = [-1]
    elif mutation == "own_pool": rows[0]["candidate_global_indices"] = [1]
    elif mutation == "family": rows[0]["family_id"] = "another"
    with pytest.raises(ValueError): local.build_jobs(p, rows, index, Tokenizer())


def test_gold_used_only_for_max_reference_scoring_all_methods_and_pairs():
    p, jobs, mapping, gold = jobs_fixture()
    gold["doc", "q"].append({"native_answer": {"unanswerable": False, "extractive_spans": ["Red"],
        "free_form_answer": "", "yes_no": None, "evidence": ["Red"]}})
    rows, summary = local.score_all(p, mapping, {j["cache_key"]: " The, Red! " for j in jobs}, gold)
    assert len(rows) == 6 and all(r["official_answer_f1"] == 1 for r in rows)
    assert len(summary["metrics"]) == 6
    assert [(r["plus"], r["minus"]) for r in summary["paired_comparisons"]] == list(local.PAIRS)
    assert all(r["families"] == 1 and r["questions"] == 1 for r in summary["paired_comparisons"])
    with pytest.raises(ValueError, match="partial quality"): local.score_all(p, mapping, {}, gold)
    with pytest.raises(ValueError): local.score_all(p, mapping[:-1], {j["cache_key"]: "Red" for j in jobs}, gold)


def test_reservation_event_is_durable_before_transport_and_full_replay(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    def transport(payload):
        ledger, _ = local.verify_events(tmp_path)
        assert ledger["attempts"][-1]["status"] == "in_flight"
        assert Decimal(ledger["reservation_total_usd"]) > 0
        return response()
    bounded = client_for(tmp_path, jobs, transport)
    for job in jobs: bounded.submit(job)
    answers, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=True)
    assert complete and len(answers) == 2 and len(list((tmp_path / "attempt_ledger").iterdir())) == 5
    assert Decimal(ledger["reservation_total_usd"]) == Decimal(config_for(jobs)["answer_reservation_usd"])
    with pytest.raises(ValueError): bounded.submit(jobs[0])
    assert all(b"test-credential" not in f.read_bytes() for f in tmp_path.rglob("*.json"))


def test_unknown_timeout_retains_attempt_reservation_and_halts(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    calls = []
    def timeout(payload):
        calls.append(payload)
        raise TimeoutError("private arbitrary exception")
    bounded = client_for(tmp_path, jobs, timeout)
    with pytest.raises(RuntimeError): bounded.submit(jobs[0])
    with pytest.raises(ValueError): bounded.submit(jobs[1])
    answers, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)
    summary = local.accounting(config_for(jobs), ledger, complete)
    assert len(calls) == 1 and not answers and not complete
    assert summary["unknown_generation_cost_attempts"] == 1
    assert Decimal(summary["night_attempted_reservation_usd"]) == Decimal(local.PRIOR["reservation_usd"]) + Decimal(jobs[0]["reserved_usd"])
    assert "private arbitrary exception" not in (tmp_path / "provider_calls/ledger.json").read_text()
    with pytest.raises(ValueError, match="no main quality"):
        local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=True)


@pytest.mark.parametrize("failure", ["length", "route", "usage", "http_known", "http_unknown"])
def test_failed_saved_response_cost_is_known_or_unknown_exactly(tmp_path, failure):
    _, jobs, _, _ = jobs_fixture()
    def transport(payload):
        if failure.startswith("http"):
            body = {"error": {"code": 429, "message": "rate limited"}}
            if failure == "http_known": body["usage"] = {"cost": .0002}
            raise HTTPError("https://invalid.example", 429, "private", {}, io.BytesIO(json.dumps(body).encode()))
        item = response()
        if failure == "length": item["choices"][0]["finish_reason"] = "length"
        if failure == "route": item["model"] = "changed/model"
        if failure == "usage": item["usage"]["completion_tokens"] = 513
        return item
    bounded = client_for(tmp_path, jobs, transport)
    with pytest.raises(RuntimeError): bounded.submit(jobs[0])
    _, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)
    assert not complete
    assert ("actual_cost_usd" not in ledger["attempts"][0]) == (failure == "http_unknown")


def rewrite_event_chain(tmp_path, mutation):
    events = sorted((tmp_path / "attempt_ledger").iterdir())
    values = [local.read(p) for p in events]
    mutation(values)
    previous = None
    for path, value in zip(events, values):
        value["previous_sha256"] = previous
        path.write_text(json.dumps(value), encoding="utf-8")
        previous = local.legacy.digest(path)
    (tmp_path / "provider_calls/ledger.json").write_text(json.dumps(values[-1]["ledger"]), encoding="utf-8")


def test_partial_known_charge_cannot_be_removed_from_ledger(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    item = response(); item["choices"][0]["finish_reason"] = "length"
    bounded = client_for(tmp_path, jobs, lambda _: item)
    with pytest.raises(RuntimeError): bounded.submit(jobs[0])
    def mutation(events):
        events[-1]["ledger"]["attempts"][0].pop("actual_cost_usd")
        events[-1]["ledger"]["actual_reported_cost_usd"] = "0"
    rewrite_event_chain(tmp_path, mutation)
    with pytest.raises(ValueError, match="known cost"):
        local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)


def test_partial_cannot_hide_additional_request_outside_attempt_count(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    bounded = client_for(tmp_path, jobs, lambda _: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError): bounded.submit(jobs[0])
    extra = tmp_path / "provider_calls/request_002.json"
    extra.write_bytes(local.legacy.client.canonical_bytes(jobs[1]["payload"]))
    def mutation(events): events[-1]["new_provider_artifact_sha256"][extra.name] = local.legacy.digest(extra)
    rewrite_event_chain(tmp_path, mutation)
    with pytest.raises(ValueError, match="artifact"):
        local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)


def fake_plan(monkeypatch, tmp_path):
    prepared, jobs, mapping, gold = jobs_fixture()
    plan_dir, output = tmp_path / "plan", tmp_path / "run"
    plan_dir.mkdir()
    (plan_dir / "experiment_config.json").write_text("{}")
    own = local.hashes([plan_dir / "experiment_config.json"])
    config = {**config_for(jobs), "input_sha256": {}, "sidecar": "unused-gold-path",
        "run_output": str(output), "single_use_registration": str(tmp_path / "single_use.json")}
    monkeypatch.setattr(local, "load_plan", lambda _: (config, prepared, jobs, mapping, own))
    monkeypatch.setattr(local.legacy.pilot, "selected_gold", lambda *args: gold)
    return plan_dir, output, jobs


def test_complete_single_use_run_and_audit_all_metadata(tmp_path, monkeypatch):
    plan, output, jobs = fake_plan(monkeypatch, tmp_path)
    args = SimpleNamespace(plan=plan, key_file="never-read", proxy=None)
    def factory(path, **kwargs): return local.LocalAnswerClient(path, **kwargs, transport=lambda _: response())
    summary = local.run(args, client_factory=factory)
    audited = local.audit(SimpleNamespace(plan=plan, run=output, allow_incomplete=False))
    assert audited["status"] == "verified_complete"
    assert summary["record_count"] == 6 and summary["new_api_calls"] == 2
    with pytest.raises(FileExistsError): local.run(args, client_factory=factory)
    changed = local.read(output / "summary.json"); changed["family_count"] += 1
    (output / "summary.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        local.audit(SimpleNamespace(plan=plan, run=output, allow_incomplete=False))


def test_failed_run_prefix_audited_without_quality_and_cannot_retry(tmp_path, monkeypatch):
    plan, output, jobs = fake_plan(monkeypatch, tmp_path)
    calls = []
    def transport(payload):
        calls.append(payload)
        if len(calls) == 2: raise TimeoutError()
        return response()
    def factory(path, **kwargs): return local.LocalAnswerClient(path, **kwargs, transport=transport)
    args = SimpleNamespace(plan=plan, key_file="never-read", proxy=None)
    with pytest.raises(RuntimeError): local.run(args, client_factory=factory)
    assert not (output / "summary.json").exists()
    audited = local.audit(SimpleNamespace(plan=plan, run=output, allow_incomplete=True))
    assert audited["status"] == "verified_incomplete"
    assert audited["accounting"]["completed_requests"] == 1
    assert audited["accounting"]["unknown_generation_cost_attempts"] == 1
    with pytest.raises(FileExistsError): local.run(args, client_factory=factory)
    assert len(calls) == 2


def test_cutoff_before_next_request_preserves_complete_prefix(tmp_path, monkeypatch):
    plan, output, jobs = fake_plan(monkeypatch, tmp_path)
    class StopsAfterOne(local.LocalAnswerClient):
        def submit(self, job):
            if self.ledger["attempts"]: raise TimeoutError("night paid-work window ended")
            return super().submit(job)
    def factory(path, **kwargs): return StopsAfterOne(path, **kwargs, transport=lambda _: response())
    with pytest.raises(TimeoutError): local.run(SimpleNamespace(plan=plan, key_file="unused", proxy=None), client_factory=factory)
    audit = local.audit(SimpleNamespace(plan=plan, run=output, allow_incomplete=True))
    assert audit["accounting"]["completed_requests"] == 1 and audit["accounting"]["new_api_calls"] == 1
    assert audit["accounting"]["unknown_generation_cost_attempts"] == 0


def test_single_use_registration_refuses_even_empty_fresh_output(tmp_path, monkeypatch):
    plan, output, _ = fake_plan(monkeypatch, tmp_path)
    (tmp_path / "single_use.json").write_text("{}")
    with pytest.raises(FileExistsError): local.run(SimpleNamespace(plan=plan, key_file="unused", proxy=None),
        client_factory=lambda *a, **k: pytest.fail("must fail before reading credential"))
    assert not output.exists()


def test_budget_includes_all_prior_unknown_reservation_before_key_read(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    with pytest.raises(ValueError):
        local.LocalAnswerClient(tmp_path / "provider_calls", prior_reservation="4.999999", jobs=jobs,
                                key_file="never-read", transport=lambda _: response())
    assert not (tmp_path / "provider_calls").exists()


def test_real_dispatch_checks_fixed_cutoff_before_each_attempt(tmp_path, monkeypatch):
    _, jobs, _, _ = jobs_fixture()
    bounded = client_for(tmp_path, jobs, lambda _: response())
    bounded.submit(jobs[0])
    bounded.transport = None  # Exercise real dispatch gate, with no key read or network.
    class AfterCutoff:
        @staticmethod
        def now(tz): return local.datetime.fromisoformat(local.legacy.STOP_AT)
        @staticmethod
        def fromisoformat(value): return local.datetime.fromisoformat(value)
    monkeypatch.setattr(local.legacy, "datetime", AfterCutoff)
    monkeypatch.setattr(bounded.opener, "open", lambda *a, **k: pytest.fail("cutoff must precede network"))
    with pytest.raises(TimeoutError, match="window ended"): bounded.submit(jobs[1])
    _, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)
    assert not complete and len(ledger["attempts"]) == 1


def test_process_interruption_in_flight_is_unknown_and_not_a_score(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    bounded = client_for(tmp_path, jobs, lambda _: response())
    job = jobs[0]
    body = local.legacy.client.canonical_bytes(job["payload"])
    (bounded.output / "request_001.json").write_bytes(body)
    bounded.ledger["attempts"].append({"attempt": 1, "cache_key": job["cache_key"],
        "request_sha256": hashlib.sha256(body).hexdigest(), "reserved_usd": job["reserved_usd"],
        "input_allowance": job["input_allowance"], "output_allowance": job["output_allowance"],
        "status": "in_flight", "started_at": "2026-09-27T00:00:00+00:00"})
    bounded.ledger["reservation_total_usd"] = job["reserved_usd"]
    bounded.save()
    answers, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)
    assert answers == {} and not complete
    assert local.accounting(config_for(jobs), ledger, False)["unknown_generation_cost_attempts"] == 1


def test_keyboard_interrupt_finally_event_stays_unknown_and_cannot_continue(tmp_path):
    _, jobs, _, _ = jobs_fixture()
    def interrupt(payload): raise KeyboardInterrupt()
    bounded = client_for(tmp_path, jobs, interrupt)
    with pytest.raises(KeyboardInterrupt): bounded.submit(jobs[0])
    answers, ledger, _, complete = local.inspect_calls(config_for(jobs), jobs, tmp_path, require_complete=False)
    assert not complete and not answers and ledger["attempts"][0]["status"] == "in_flight"
    assert "elapsed_seconds" in ledger["attempts"][0]
    assert "actual_cost_usd" not in ledger["attempts"][0]
    assert Decimal(ledger["reservation_total_usd"]) == Decimal(jobs[0]["reserved_usd"])
    with pytest.raises(ValueError, match="unresolved previous attempt"): bounded.submit(jobs[1])


def test_keyboard_interrupt_run_has_auditable_failure_without_partial_scores(tmp_path, monkeypatch):
    plan, output, jobs = fake_plan(monkeypatch, tmp_path)
    calls = []
    def transport(payload):
        calls.append(payload)
        if len(calls) == 2: raise KeyboardInterrupt()
        return response()
    def factory(path, **kwargs): return local.LocalAnswerClient(path, **kwargs, transport=transport)
    args = SimpleNamespace(plan=plan, key_file="unused", proxy=None)
    with pytest.raises(KeyboardInterrupt): local.run(args, client_factory=factory)
    report = local.audit(SimpleNamespace(plan=plan, run=output, allow_incomplete=True))
    assert report["status"] == "verified_incomplete" and not report["accounting"]["main_results_available"]
    assert report["accounting"]["completed_requests"] == 1
    assert report["accounting"]["unknown_generation_cost_attempts"] == 1
    assert not (output / "summary.json").exists()
    with pytest.raises(FileExistsError): local.run(args, client_factory=factory)
