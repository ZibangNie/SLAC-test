"""Synthetic tests only: denominator, bridge, posthoc statistics and admission."""
from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import analyze_qasper_primary_vs_reranker_answers as a


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *x, **k: pytest.fail("network forbidden"))
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *x, **k: pytest.fail("GPU forbidden"))


def fixture_data():
    queries = [{"family_id": f"synthetic-family-{i % 24:02d}", "doc_id": f"synthetic-doc-{i % 24:02d}",
                "question_id": f"synthetic-question-{i:02d}"} for i in range(77)]
    payloads, jobs, result = {}, {}, {}
    for prefix, methods in (("local", a.LOCAL_METHODS), ("primary", a.PRIMARY_METHODS)):
        mapping, rows, answers = [], [], {}
        for method in methods:
            for i, query in enumerate(queries):
                evidence = "" if method == "empty" else "[synthetic-unit]\nsynthetic evidence"
                user = {"question": f"synthetic query {i}", "evidence": evidence}
                payload = {**copy.deepcopy(a.GENERATOR), "messages": [{"role": "system", "content": a.SYSTEM},
                    {"role": "user", "content": a.canonical(user).decode()}]}
                key = a.object_hash({"endpoint": a.ENDPOINT, "prompt_version": a.PROMPT, "payload": payload})
                jobs[key] = {"cache_key": key, "payload": payload}
                payloads[key] = payload
                row = {**query, "method": method, "cache_key": key,
                    "selected_ids": [] if method == "empty" else ["synthetic-unit"],
                    "pack_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                    "actual_evidence_tokens": 0 if method == "empty" else 100}
                if prefix == "primary": row["response_origin"] = "local"
                mapping.append(row)
                rows.append({**row, "predicted_answer": "Unanswerable" if method == "empty" else "synthetic answer",
                             "official_answer_f1": .5})
                answers[key] = rows[-1]["predicted_answer"]
        result[prefix] = rows
        result[prefix + "_mapping"] = mapping
        result[prefix + "_answers"] = answers
    return {**result, "payloads": payloads}, list(jobs.values())


def test_all_four_pairs_two_metrics_two_weights_and_equal_score_arms():
    data, jobs = fixture_data()
    assert a.index_payloads(jobs) == data["payloads"]
    result, rows = a.compare(**data)
    assert (result["question_count"], result["family_count"], result["record_count"], len(rows)) == (77, 24, 385, 308)
    assert result["reported_interval_count"] == 16
    assert result["score_rules_exact_payload_answer_f1_matches"] == 77
    assert result["score_rules_are_independent_replications"] is False
    assert result["specification"]["posthoc"] is True
    assert [(p["plus"], p["minus"]) for p in result["answer"]["paired_comparisons"]] == list(a.PAIRS)
    for pair in result["answer"]["paired_comparisons"]:
        for metric in a.METRICS:
            item = pair["metrics"][metric]
            assert item["question_ties"] == 77
            for weight in ("question_weighted", "family_balanced"):
                assert item[weight] == {"delta": 0., "bootstrap_percentile_95": [0., 0.]}


def test_negative_positive_and_family_weighting_are_not_filtered():
    data, _ = fixture_data()
    for row in data["primary"]:
        if row["method"] == "I_jev_k3":
            row["official_answer_f1"] = 1. if row["family_id"].endswith("00") else .4
        if row["method"] == "I_general_k3": row["official_answer_f1"] = .2
    result, _ = a.compare(**data)
    first, second = result["answer"]["paired_comparisons"][:2]
    f = first["metrics"]["official_answer_f1"]
    assert (f["question_wins"], f["question_ties"], f["question_losses"]) == (4, 0, 73)
    assert f["question_weighted"]["delta"] == pytest.approx((4*.5 - 73*.1)/77)
    assert f["family_balanced"]["delta"] == pytest.approx((.5 - 23*.1)/24)
    g = second["metrics"]["official_answer_f1"]
    assert g["question_weighted"]["bootstrap_percentile_95"] == pytest.approx([-.3, -.3])


@pytest.mark.parametrize("field", ["cache_key", "selected_ids", "pack_sha256", "actual_evidence_tokens", "predicted_answer", "official_answer_f1"])
def test_dense_bridge_rejects_each_scientific_difference(field):
    data, _ = fixture_data()
    row, mapping = data["primary"][0], data["primary_mapping"][0]
    if field == "actual_evidence_tokens": row[field] += 1; mapping[field] += 1
    elif field == "official_answer_f1": row[field] += .1
    elif field == "predicted_answer": row[field] = "changed"
    elif field == "selected_ids": row[field] = ["changed"]; mapping[field] = ["changed"]
    else: row[field] = "changed"; mapping[field] = "changed"
    with pytest.raises(ValueError): a.compare(**data)


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign_method", "family", "nan", "bool_score", "fractional_tokens", "oversize", "empty_nonempty"])
def test_incomplete_or_invalid_rows_refused(change):
    data, _ = fixture_data()
    if change == "missing": data["primary"].pop()
    elif change == "duplicate": data["primary"].append(copy.deepcopy(data["primary"][0]))
    elif change == "foreign_method": data["primary"][0]["method"] = "unplanned"
    elif change == "family": data["primary"][0]["family_id"] = "other"
    elif change == "nan": data["primary"][0]["official_answer_f1"] = float("nan")
    elif change == "bool_score": data["primary"][0]["official_answer_f1"] = True
    elif change == "fractional_tokens": data["primary"][0]["actual_evidence_tokens"] = 1.5
    elif change == "oversize": data["primary"][0]["actual_evidence_tokens"] = 1025
    else:
        data["primary"][-1]["actual_evidence_tokens"] = 1
        data["primary_mapping"][-1]["actual_evidence_tokens"] = 1
    with pytest.raises(ValueError): a.compare(**data)


@pytest.mark.parametrize("change", ["model", "temperature", "provider", "prompt", "cache", "noncanonical", "duplicate"])
def test_payload_contract_drift_rejected(change):
    _, jobs = fixture_data()
    job = jobs[0]
    if change in ("model", "temperature", "provider"): job["payload"][change] = "different"
    elif change == "prompt": job["payload"]["messages"][0]["content"] += " changed"
    elif change == "cache": job["cache_key"] = "changed"
    elif change == "noncanonical": job["payload"]["messages"][1]["content"] += " "
    else: jobs.append(copy.deepcopy(job))
    with pytest.raises(ValueError): a.index_payloads(jobs)


def test_missing_release_rejected_before_any_load(monkeypatch):
    monkeypatch.setattr(a, "load_plan", lambda *x: pytest.fail("parent loaded before release flag"))
    with pytest.raises(ValueError, match="Root release"):
        a.analyze(Namespace(released_complete=False))


def test_bad_release_rejected_before_quality(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    a.write(receipt, {"root_release": "wrong"})
    monkeypatch.setattr(a, "load_plan", lambda *x: ({"input_binding_sha256": "input"}, "plan"))
    monkeypatch.setattr(a, "compare", lambda **x: pytest.fail("quality before release"))
    with pytest.raises(ValueError, match="does not authorize"):
        a.analyze(Namespace(released_complete=True, plan="unused", release_receipt=receipt, release_sha256=a.sha(receipt)))


def test_prepare_only_binds_metadata_and_does_not_read_quality(tmp_path, monkeypatch):
    parent = tmp_path / "parent"; parent.mkdir()
    marker = parent / "metadata.json"; a.write(marker, {"complete": True})
    bindings = {str(marker.resolve()): a.sha(marker)}
    monkeypatch.setattr(a, "source_hashes", lambda: {})
    monkeypatch.setattr(a, "complete_metadata", lambda paths: (bindings, {"complete": True}))
    monkeypatch.setattr(a, "compare", lambda **x: pytest.fail("prepare computed quality"))
    output = tmp_path / "plan"
    args = Namespace(output=output, **{key: parent for key in a.DEFAULTS})
    a.prepare(args)
    config, digest = a.load_plan(output)
    assert config["cross_baseline_quality_computed"] is False
    assert digest == a.sha(output / "plan.json")
    marker.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"): a.load_plan(output)


def test_bound_buffer_and_no_overwrite_or_overlap(tmp_path):
    p = tmp_path / "parent"; p.mkdir()
    path = p / "data.json"; a.write(path, {"value": 1})
    bindings = {str(path): a.sha(path)}
    assert a.bound_json(path, bindings) == {"value": 1}
    with pytest.raises(FileExistsError): a.write(path, {})
    with pytest.raises(ValueError, match="overlaps"): a.disjoint(p / "new", [p])
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"): a.bound_json(path, bindings)


def test_complete_metadata_gate_refuses_partial_before_scores(tmp_path):
    paths = {key: str(tmp_path / key) for key in a.DEFAULTS}
    for path in paths.values(): Path(path).mkdir()
    a.write(Path(paths["local_plan"]) / "experiment_config.json", {})
    a.write(Path(paths["local_run"]) / "summary.json", {"status": "incomplete"})
    a.write(Path(paths["local_audit"]) / "audit.json", {"status": "verified_incomplete"})
    a.write(Path(paths["local_audit"]) / "source_binding.json", {})
    with pytest.raises(ValueError, match="complete audited parent"): a.complete_metadata(paths)


def metadata_fixture(tmp_path, change=None):
    paths = {key: str(tmp_path / key) for key in a.DEFAULTS}
    for path in paths.values(): Path(path).mkdir()
    prepared, rp, sp, owner = (tmp_path / p for p in ("prepared", "reranker", "support", "owner"))
    for path in (prepared, rp, sp, owner): path.mkdir()
    # The payload sentinel must never be JSON-decoded by metadata-only prepare.
    (prepared / "prepared.json").write_text("NOT JSON: metadata preparation must only hash this", encoding="utf-8")
    a.write(prepared / "manifest.json", {"prepared_sha256": a.sha(prepared / "prepared.json"),
        "config": {"dense_seeds": 8, "max_candidates_per_question": 17 if change == "candidates" else 16,
                   "evidence_budget_bge_tokens": 1024}})
    a.write(rp / "experiment_config.json", {"prepared_dir": str(prepared), "contract": {
        "candidate_scope": "all 1214 frozen support pairs from 77 questions; no candidate additions",
        "model_id": "BAAI/bge-reranker-v2-m3", "revision": "synthetic-revision", "selection_k": [1, 2, 3],
        "evidence_budget_bge_tokens": 1024}})
    a.write(sp / "experiment_config.json", {"prepared_dir": str(prepared if change != "pool" else owner),
        "selector": {"budget": 1024, "sizes": [1, 2, 3]}})
    a.write(owner / "jobs.json", [])
    inputs = {str(path): a.sha(path) for path in (prepared / "prepared.json", prepared / "manifest.json",
        rp / "experiment_config.json", sp / "experiment_config.json", owner / "jobs.json")}
    for prefix, methods in (("local", a.LOCAL_METHODS), ("primary", a.PRIMARY_METHODS)):
        plan, run, audit = (Path(paths[f"{prefix}_{kind}"]) for kind in ("plan", "run", "audit"))
        a.write(plan / "jobs.json", [])
        a.write(plan / "mapping.jsonl", [])
        source_paths = {"prepared": str(prepared), "reranker_plan": str(rp)} if prefix == "local" else {
            **{f"local_{kind}": paths[f"local_{kind}"] for kind in ("plan", "run", "audit")},
            "recovery_plan": str(sp), "owner_plan": str(owner)}
        config = {"questions": 77, "families": 24, "logical_predictions": 462,
            "specification": {"methods": list(methods), "generator": a.MODEL,
                "prompt_version": "bad" if change == "prompt" and prefix == "primary" else a.PROMPT,
                "maximum_output_tokens": 512},
            "source_paths": source_paths, "input_sha256": inputs,
            "plan_files_sha256": {name: a.sha(plan / name) for name in ("jobs.json", "mapping.jsonl")}}
        a.write(plan / "experiment_config.json", config)
        plan_sha = a.sha(plan / "experiment_config.json")
        a.write(plan / "plan_manifest.json", {"experiment_config_sha256": plan_sha})
        (run / "provider_calls").mkdir()
        a.write(run / "provider_calls/ledger.json", {"resolved_models": {"generator": "bad" if change == "actual_model" else a.MODEL}})
        (run / "per_question.jsonl").write_text("NOT JSON: no quality reads in prepare", encoding="utf-8")
        output_hashes = {str(Path(p).relative_to(run)): h for p, h in a.tree(run).items()}
        a.write(run / "summary.json", {"status": "completed", "main_results_available": True,
            "question_count": 77, "family_count": 24, "record_count": 462, "plan_sha256": plan_sha,
            "output_sha256": output_hashes})
        a.write(audit / "audit.json", {"status": "verified_complete", "all_bound_inputs_outputs_unchanged": True,
            "partial_quality_metrics_computed": False, "accounting": {"status": "completed"}})
        a.write(audit / "source_binding.json", {"root_release": "complete_run_reviewed_for_publication",
            "input_sha256": {str(p): a.sha(p) for p in (plan / "experiment_config.json", run / "summary.json", audit / "audit.json")}})
    if change == "extra_output": (Path(paths["local_run"]) / "extra.json").write_text("{}")
    return paths


def test_metadata_accepts_complete_contract_without_parsing_any_quality(tmp_path):
    paths = metadata_fixture(tmp_path)
    bindings, contract = a.complete_metadata(paths)
    assert contract["candidate_pairs"] == 1214
    assert contract["parents_complete_and_audited"] is True
    assert any(Path(path).name == "per_question.jsonl" for path in bindings)
    a.verify(bindings)


@pytest.mark.parametrize("change", ["candidates", "pool", "prompt", "actual_model", "extra_output"])
def test_complete_metadata_rejects_incompatible_pool_generator_or_inventory(tmp_path, change):
    with pytest.raises(ValueError): a.complete_metadata(metadata_fixture(tmp_path, change))
