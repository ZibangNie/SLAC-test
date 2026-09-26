"""Offline execution-contract tests; no credential reads or external requests."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import openrouter_decision_client as api
import run_qasper_relation_pilot as runner
from prepare_qasper_relation_pilot import build_prepared
from run_qasper_evidence_baselines import Unit


class TinyTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [1] + list(range(len(text.split()))) + [2]


def annotation(evidence):
    return {"native_answer": {"unanswerable": False, "extractive_spans": [],
        "free_form_answer": "FORBIDDEN_QA_ANSWER", "yes_no": None, "evidence": evidence}}


def make_prepared(tmp_path, monkeypatch):
    source, prepared_dir, tokenizer_dir = (tmp_path / name for name in ("source", "prepared", "tokenizer"))
    for directory in (source, prepared_dir, tokenizer_dir):
        directory.mkdir()
    units = [Unit(f"u{i}", i, "paragraph", i * 20, i * 20 + 10,
                  f"Evidence {i}", f"Evidence {i}") for i in range(6)]
    candidates = [{"doc_id": "doc", "family_id": "family"}]
    qas = [{"doc_id": "doc", "family_id": "family", "question_id": f"q{i}", "question": f"What is {i}?",
            "official_split": "validation", "answer_annotations": [annotation(["Evidence 0", "Evidence 1", "Evidence 2", "Evidence 3"])]}
           for i in range(2)]
    documents = {"doc": units}
    ranks = {("doc", row["question_id"]): ["u0", "u2", "u4", "u1", "u3", "u5"] for row in qas}
    prepared = build_prepared(candidates, qas, documents, ranks)
    sidecar = source / "sidecar.jsonl"
    sidecar.write_text("\n".join(json.dumps(row) for row in qas), encoding="utf-8")
    ranking_path = source / "rankings.jsonl"
    ranking_path.write_text("\n".join(json.dumps({"doc_id": key[0], "question_id": key[1], "ranked_ids": value})
                                      for key, value in ranks.items()), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer_dir / name).write_text("{}", encoding="utf-8")
    (prepared_dir / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    manifest = {"status": "prepared", "test_payload_read": False,
                "prepared_sha256": runner.digest(prepared_dir / "prepared.json"),
                "input_sha256": {str(path): runner.digest(path) for path in (sidecar, ranking_path, *tokenizer_dir.iterdir())}}
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    return SimpleNamespace(prepared=prepared_dir, sidecar=sidecar, tokenizer=tokenizer_dir, output=tmp_path / "plan")


def responses(backend, payload):
    if backend == "jev":
        questions = payload["questions"]
    else:
        questions = json.loads(payload["messages"][1]["content"])["questions"]
    labels = {key: ("dependent" if "dependent" in row["criteria"] else "yes") for key, row in questions.items()}
    result = {"id": "mock-id", "provider": api.MODELS[backend]["provider"],
              "model": sorted(api.RESPONSE_MODELS[backend])[0],
              "usage": {"cost": "0.001", "prompt_tokens": 10, "completion_tokens": 5}}
    if backend == "jev":
        result["answers"] = {key: {"type": "choice", "choice": value} for key, value in labels.items()}
    else:
        result["choices"] = [{"finish_reason": "stop", "message": {"content": json.dumps(labels)}}]
    return result


def test_plan_is_gold_free_fixed_batch_visibility_and_model_order(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    config = runner.plan(args)
    assert config["status"] == "planned" and config["api_calls"] == 0
    assert config["predicted_reservations"]["requests"] == 6
    assert config["predicted_reservations"]["questions"] == 34
    assert [row["backend"] for row in config["schedule"]] == ["jev", "general", "general", "jev", "jev", "general"]
    batches = runner.read_json(args.output / "batches.json")
    assert len(batches) == 3 and all(len(batch["tasks"]) <= 8 for batch in batches)
    assert "FORBIDDEN_QA_ANSWER" not in json.dumps(batches)
    assert {len(batch["group"]) for batch in batches if batch["kind"] == "static"} == {1}
    assert {len(batch["group"]) for batch in batches if batch["kind"] == "support"} == {2}
    for batch in batches:
        jev = api.make_payload(batch["tasks"], batch["kind"], "jev")
        general = api.make_payload(batch["tasks"], batch["kind"], "general")
        decoded = json.loads(general["messages"][1]["content"])
        assert decoded["state"] == jev["state"] and decoded["questions"] == jev["questions"]
    restored, _ = runner.load_plan(args.output)
    assert restored == config


def test_oracle_matches_three_unit_cap_and_keeps_full_baseline(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    runner.plan(args)
    rows = [json.loads(line) for line in (args.output / "baseline_per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert {row["method"] for row in rows} == {"dense_top3_capped", "dense_top3_full_document", "gold_subset_oracle_capped_max3", "empty"}
    oracle = [row for row in rows if row["method"] == "gold_subset_oracle_capped_max3"]
    assert all(row["selected_units"] == 3 for row in oracle)
    assert all(row["official_evidence_f1"] == pytest.approx(6 / 7) for row in oracle)


def test_plan_refuses_changed_source_and_preserves_failure(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    args.sidecar.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.plan(args)
    assert runner.read_json(args.output / "failure.json")["actual_model_scores_available"] is False
    assert not (args.output / "experiment_config.json").exists()


def test_plan_refuses_existing_output(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        runner.plan(args)
    assert list(args.output.iterdir()) == []


def test_prediction_over_cap_refuses_without_paid_execution(monkeypatch):
    task = {"id": "task", "item": {"query": "query", "unit": {"id": "unit", "text": "text"}}}
    batch = {"id": "b", "kind": "support", "tasks": [task]}
    with pytest.raises(ValueError, match="exceeds pilot caps"):
        runner.schedule_requests([batch] * 81)


def test_run_complete_uses_unique_real_decisions_for_six_replays(tmp_path, monkeypatch, capsys):
    args = make_prepared(tmp_path, monkeypatch)
    config = runner.plan(args)
    run_args = SimpleNamespace(plan=args.output, output=tmp_path / "run", key_file="unused", proxy=None)
    def factory(path, **kwargs):
        return api.BoundedClient(path, transport=responses, **kwargs)
    result = runner.run(run_args, client_factory=factory)
    assert result["status"] == "completed" and result["record_count"] == 12
    assert result["actual_requests"] == config["predicted_reservations"]["requests"] == 6
    assert result["cache_savings_measured"] is False and result["answer_generation_performed"] is False
    assert result["actual_reported_cost_usd"] == "0.006"
    assert len(result["paired_differences"]) == 5
    rows = [json.loads(line) for line in (run_args.output / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    for backend in api.MODELS:
        scores = {row["method"]: row["official_evidence_f1"] for row in rows if row["question_id"] == "q0"}
        assert scores[f"I_{backend}"] == scores[f"C_{backend}"]
    printed = capsys.readouterr().out
    assert "Evidence" not in printed and "What is" not in printed and "FORBIDDEN" not in printed


def test_partial_provider_failure_never_produces_experimental_scores(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    runner.plan(args)
    run_args = SimpleNamespace(plan=args.output, output=tmp_path / "failed_run", key_file="unused", proxy=None)
    calls = []
    def transport(backend, payload):
        calls.append(backend)
        if len(calls) == 2:
            raise RuntimeError("mock secret must not enter failure artifact")
        return responses(backend, payload)
    def factory(path, **kwargs):
        return api.BoundedClient(path, transport=transport, **kwargs)
    with pytest.raises(RuntimeError):
        runner.run(run_args, client_factory=factory)
    assert len(calls) == 2
    assert (run_args.output / "labels.json").exists()
    assert not (run_args.output / "summary.json").exists()
    assert not (run_args.output / "per_question.jsonl").exists()
    text = (run_args.output / "failure.json").read_text(encoding="utf-8")
    assert "mock secret" not in text and '"actual_model_scores_available": false' in text


def test_changed_plan_or_code_is_rejected_before_client_creation(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    runner.plan(args)
    (args.output / "experiment_config.json").write_text("{}", encoding="utf-8")
    run_args = SimpleNamespace(plan=args.output, output=tmp_path / "failed_run", key_file="unused", proxy=None)
    def forbidden(*args, **kwargs):
        raise AssertionError("client must not be created")
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.run(run_args, client_factory=forbidden)


def test_incomplete_labels_are_not_filled_with_unknown(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    prepared, _, documents = runner.load_prepared(args.prepared)
    annotations = runner.selected_gold(args.sidecar, prepared)
    with pytest.raises(ValueError, match="incomplete actual model labels"):
        runner.replay_records(prepared, documents, annotations, {"jev": {}, "general": {}}, TinyTokenizer())


def test_changing_config_and_its_seal_during_execution_is_rejected(tmp_path, monkeypatch):
    args = make_prepared(tmp_path, monkeypatch)
    config = runner.plan(args)
    run_args = SimpleNamespace(plan=args.output, output=tmp_path / "changed_run", key_file="unused", proxy=None)
    calls = []
    def transport(backend, payload):
        calls.append(backend)
        if len(calls) == len(config["schedule"]):
            config_path = args.output / "experiment_config.json"
            changed = runner.read_json(config_path)
            changed["unexpected_post_start_field"] = True
            config_path.write_text(json.dumps(changed), encoding="utf-8")
            (args.output / "plan_manifest.json").write_text(json.dumps({"experiment_config_sha256": runner.digest(config_path)}), encoding="utf-8")
        return responses(backend, payload)
    def factory(path, **kwargs):
        return api.BoundedClient(path, transport=transport, **kwargs)
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner.run(run_args, client_factory=factory)
    assert not (run_args.output / "summary.json").exists()
    assert not (run_args.output / "per_question.jsonl").exists()


def test_paired_differences_use_family_means_not_many_queries_as_independent():
    records = []
    for family, count, effect in (("f1", 2, 1.), ("f2", 1, 0.)):
        for question in range(count):
            for backend in ("jev", "general"):
                for mode in ("I", "C", "S"):
                    value = effect if mode == "S" else 0.
                    records.append({"family_id": family, "doc_id": family, "question_id": str(question),
                        "method": f"{mode}_{backend}", "official_evidence_f1": value,
                        "reference_evidence_recall": value, "actual_evidence_tokens": 10})
    differences = runner.paired_differences(records)
    result = differences[0]["statistics"]["official_evidence_f1"]
    assert result["family_macro_delta"] == .5 and result["document_macro_delta"] == .5
    assert result["confidence_interval"] is None
