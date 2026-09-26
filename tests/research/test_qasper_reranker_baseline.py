"""Offline contracts for the pinned local reranker; no model download or GPU."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_reranker_baseline as runner
from run_qasper_evidence_baselines import Unit


class TinyTokenizer:
    def __call__(self, query, passage, *, add_special_tokens, truncation, padding):
        assert add_special_tokens is True and truncation is False and padding is False
        ids = [0] + [10 + len(word) for word in query.split()] + [2, 2]
        ids += [20 + len(word) for word in passage.split()] + [2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [0] + [5] * len(text.split()) + [2]

    def pad(self, values, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        length = max(len(row["input_ids"]) for row in values)
        return {name: torch.tensor([row[name] + [0] * (length - len(row[name])) for row in values])
                for name in ("input_ids", "attention_mask")}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, *, return_dict):
        assert return_dict is True and not self.training and not torch.is_grad_enabled()
        self.calls.append(tuple(input_ids.shape))
        return SimpleNamespace(logits=input_ids.sum(dim=1, keepdim=True).float())


def fake_data():
    units = [Unit("u0", 0, "paragraph", 0, 1, "zero", "zero"),
             Unit("u1", 1, "paragraph", 1, 2, "one evidence", "one evidence"),
             Unit("u2", 2, "paragraph", 2, 3, "two longer evidence", "two longer evidence")]
    query = {"doc_id": "doc-secret", "family_id": "family-secret", "question_id": "question-secret",
             "query": "forbidden question text", "candidate_ids": ["u0", "u1", "u2"],
             "ranked_ids": ["u2", "u0", "u1"]}
    tasks = [{"id": f"t{i}", "doc_id": query["doc_id"], "question_id": query["question_id"],
              "unit_id": unit.unit_id, "item": {"query": query["query"],
              "unit": {"id": unit.unit_id, "text": unit.text}}} for i, unit in enumerate(units)]
    return {"queries": [query], "support_tasks": tasks}, {query["doc_id"]: units}


def annotation(text):
    return [{"native_answer": {"unanswerable": False, "extractive_spans": [],
             "free_form_answer": "gold must not reach model", "yes_no": None, "evidence": [text]}}]


def preparation_fixture(tmp_path, monkeypatch):
    prepared, documents = fake_data()
    paths = {name: tmp_path / name for name in ("prepared", "model", "bge")}
    for path in paths.values():
        path.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
        (paths["bge"] / name).write_text("{}", encoding="utf-8")
    (paths["model"] / "model.safetensors").write_text("synthetic", encoding="utf-8")
    hashes = {str(path.resolve()): runner.digest(path) for path in paths["bge"].iterdir()}
    manifest = {"input_sha256": hashes, "source_paths": {"sidecar": "unused"}}
    runner.write(paths["prepared"] / "prepared.json", prepared)
    runner.write(paths["prepared"] / "manifest.json", manifest)
    monkeypatch.setattr(runner.preparation, "load_prepared", lambda path: (prepared, manifest, documents))
    monkeypatch.setattr(runner, "verify_model", lambda path: {str((paths["model"] / "model.safetensors").resolve()):
                         runner.digest(paths["model"] / "model.safetensors")})
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(runner, "EXPECTED_QUERIES", 1)
    monkeypatch.setattr(runner, "EXPECTED_PAIRS", 3)
    args = SimpleNamespace(prepared=paths["prepared"], model=paths["model"], bge_tokenizer=paths["bge"],
                           output=tmp_path / "plan")
    return args, prepared, documents


def test_prepare_freezes_complete_gold_free_pairs_without_loading_model(tmp_path, monkeypatch):
    args, prepared, documents = preparation_fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not load model or query GPU")
    monkeypatch.setattr(runner.AutoModelForSequenceClassification, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "ensure_gpu_ready", forbidden)
    config = runner.prepare(args)
    assert config["status"] == "prepared_no_model_inference"
    assert config["pair_count"] == 3 and config["question_count"] == 1
    assert config["model_inference_performed"] is False and config["api_calls"] == 0
    assert config["length_audit"]["truncated_pairs"] == 0
    assert runner.load_plan(args.output) == config
    rows = [json.loads(line) for line in (args.output / "pair_audit.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all("query" not in row and "passage" not in row for row in rows)
    assert "gold must not reach model" not in json.dumps(runner.support_pairs(prepared, documents))


def test_prepare_refuses_incomplete_scope(tmp_path, monkeypatch):
    args, _, _ = preparation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_PAIRS", 4)
    with pytest.raises(ValueError, match="all 77"):
        runner.prepare(args)
    assert not (args.output / "experiment_config.json").exists()


def test_changed_plan_or_bound_model_refuses_before_loading(tmp_path, monkeypatch):
    args, _, _ = preparation_fixture(tmp_path, monkeypatch)
    runner.prepare(args)
    (args.model / "model.safetensors").write_text("tampered!", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner.load_plan(args.output)


def test_changed_configuration_seal_refuses(tmp_path, monkeypatch):
    args, _, _ = preparation_fixture(tmp_path, monkeypatch)
    runner.prepare(args)
    (args.output / "experiment_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="seal differs"):
        runner.load_plan(args.output)


@pytest.mark.parametrize("change", ["missing", "duplicate", "text", "query", "unexpected"])
def test_support_pairs_require_original_id_and_visible_text_coverage(change):
    prepared, documents = fake_data()
    if change == "missing":
        prepared["support_tasks"].pop()
    elif change == "duplicate":
        prepared["support_tasks"].append(prepared["support_tasks"][0])
    elif change == "text":
        prepared["support_tasks"][0]["item"]["unit"]["text"] = "changed"
    elif change == "query":
        prepared["support_tasks"][0]["item"]["query"] = "changed"
    else:
        prepared["support_tasks"][0]["unit_id"] = "unknown"
    with pytest.raises(ValueError, match="support pair"):
        runner.support_pairs(prepared, documents)


def test_no_truncation_1024_pair_limit_is_enforced_before_inference():
    prepared, documents = fake_data()
    pairs = runner.support_pairs(prepared, documents)
    pairs[0]["passage"] = "word " * 1024
    with pytest.raises(ValueError, match="no truncation"):
        runner.encode_pairs(pairs, TinyTokenizer())


def test_actual_pair_token_hash_and_length_are_reproducible():
    prepared, documents = fake_data()
    pairs = runner.support_pairs(prepared, documents)
    encoded, audit = runner.encode_pairs(pairs, TinyTokenizer())
    assert [row["encoding_sha256"] for row in audit] == [runner.object_hash(row) for row in encoded]
    assert runner.length_summary(audit)["actual_pair_tokens"] == sum(len(row["input_ids"]) for row in encoded)


def test_sorted_microbatches_restore_original_score_identity():
    prepared, documents = fake_data()
    pairs = runner.support_pairs(prepared, documents) * 3
    encoded, _ = runner.encode_pairs(pairs, TinyTokenizer())
    model = TinyModel()
    scores, compute = runner.infer_pairs(encoded, TinyTokenizer(), model, device="cpu", deadline=math.inf)
    assert scores == [float(sum(row["input_ids"])) for row in encoded]
    assert [shape[0] for shape in model.calls] == [4, 4, 1]
    assert compute["max_microbatch"] == 4


def test_padding_may_not_change_actual_model_input_ids():
    prepared, documents = fake_data()
    encoded, _ = runner.encode_pairs(runner.support_pairs(prepared, documents), TinyTokenizer())
    class WrongTokenizer(TinyTokenizer):
        def pad(self, *args, **kwargs):
            batch = super().pad(*args, **kwargs)
            batch["input_ids"][0, 1] += 1
            return batch
    with pytest.raises(ValueError, match="audited encoding"):
        runner.infer_pairs(encoded, WrongTokenizer(), TinyModel(), device="cpu", deadline=math.inf)


@pytest.mark.parametrize("failure", ["nonfinite", "shape", "oom"])
def test_bad_model_result_or_oom_halts_without_retry(failure):
    prepared, documents = fake_data()
    encoded, _ = runner.encode_pairs(runner.support_pairs(prepared, documents), TinyTokenizer())
    class BadModel(TinyModel):
        def forward(self, input_ids, **kwargs):
            self.calls.append(tuple(input_ids.shape))
            if failure == "oom":
                raise torch.cuda.OutOfMemoryError("synthetic")
            return SimpleNamespace(logits=torch.full((len(input_ids), 2 if failure == "shape" else 1),
                                                     float("nan") if failure == "nonfinite" else 1.))
    model = BadModel()
    with pytest.raises((ValueError, torch.cuda.OutOfMemoryError)):
        runner.infer_pairs(encoded, TinyTokenizer(), model, device="cpu", deadline=math.inf)
    assert len(model.calls) == 1


def test_deadline_stops_before_another_batch():
    prepared, documents = fake_data()
    encoded, _ = runner.encode_pairs(runner.support_pairs(prepared, documents), TinyTokenizer())
    model = TinyModel()
    with pytest.raises(TimeoutError):
        runner.infer_pairs(encoded, TinyTokenizer(), model, device="cpu", deadline=0)
    assert model.calls == []


def test_ranking_ties_follow_dense_order_and_all_k_are_reported():
    prepared, documents = fake_data()
    pairs = runner.support_pairs(prepared, documents)
    gold = {("doc-secret", "question-secret"): annotation("one evidence")}
    rows, rankings = runner.evaluate(prepared, documents, gold, pairs, [1., 1., 1.], TinyTokenizer())
    assert rankings[0]["ranked_ids"] == ["u2", "u0", "u1"]
    assert [row["selected_units"] for row in rows] == [1, 2, 3]
    assert {row["method"] for row in rows} == {f"bge_reranker_v2_m3_k{k}" for k in [1, 2, 3]}
    assert all(row["actual_evidence_tokens"] <= 1024 for row in rows)
    changed_gold = {("doc-secret", "question-secret"): annotation("zero")}
    _, changed_ranking = runner.evaluate(prepared, documents, changed_gold, pairs, [1., 1., 1.], TinyTokenizer())
    assert changed_ranking == rankings


def test_score_coverage_and_extra_queries_are_rejected():
    prepared, documents = fake_data()
    pairs = runner.support_pairs(prepared, documents)
    gold = {("doc-secret", "question-secret"): annotation("zero")}
    with pytest.raises(ValueError, match="coverage"):
        runner.evaluate(prepared, documents, gold, pairs, [1.], TinyTokenizer())
    with pytest.raises(ValueError, match="duplicate"):
        runner.evaluate(prepared, documents, gold, pairs + [pairs[0]], [1.] * 4, TinyTokenizer())


@pytest.mark.parametrize("sample", [{"name": "test", "free_mib": 6000, "utilization_percent": 87},
                                    {"name": "test", "free_mib": 1000, "utilization_percent": 0}])
def test_busy_or_memory_limited_gpu_never_launches_model(monkeypatch, sample):
    monkeypatch.setattr(runner, "user_game_running", lambda: False)
    monkeypatch.setattr(runner, "gpu_sample", lambda: sample)
    with pytest.raises(RuntimeError, match="defer"):
        runner.ensure_gpu_ready()


def test_idle_gpu_requires_all_three_samples(monkeypatch):
    monkeypatch.setattr(runner, "user_game_running", lambda: False)
    calls = []
    def sample():
        calls.append(1)
        return {"name": "test", "free_mib": 6000, "utilization_percent": 0}
    monkeypatch.setattr(runner, "gpu_sample", sample)
    monkeypatch.setattr(runner.time, "sleep", lambda value: None)
    assert len(runner.ensure_gpu_ready()) == len(calls) == 3


def test_busy_execution_leaves_no_scores_and_never_loads_model(tmp_path, monkeypatch):
    args, _, _ = preparation_fixture(tmp_path, monkeypatch)
    runner.prepare(args)
    def busy():
        raise RuntimeError("synthetic busy")
    def forbidden(*args, **kwargs):
        raise AssertionError("must not load model when device is busy")
    monkeypatch.setattr(runner, "ensure_gpu_ready", busy)
    monkeypatch.setattr(runner.AutoModelForSequenceClassification, "from_pretrained", forbidden)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="busy"):
        runner.run(SimpleNamespace(plan=args.output, output=output))
    assert runner.read(output / "failure.json")["experimental_scores_available"] is False
    assert not (output / "per_question.jsonl").exists() and not (output / "summary.json").exists()


def test_pinned_model_file_hash_is_checked(tmp_path, monkeypatch):
    config = {"architectures": [runner.CONTRACT["architecture"]], "max_position_embeddings": 8194,
              "id2label": {"0": "LABEL_0"}}
    runner.write(tmp_path / "config.json", config)
    runner.write(tmp_path / "tokenizer_config.json", {"model_max_length": 8192})
    (tmp_path / "model.safetensors").write_text("fixed", encoding="utf-8")
    files = {path.name: (path.stat().st_size, "sha256", runner.digest(path)) for path in tmp_path.iterdir()}
    monkeypatch.setattr(runner, "FILES", files)
    assert len(runner.verify_model(tmp_path)) == 3
    (tmp_path / "model.safetensors").write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.verify_model(tmp_path)


def test_running_game_defers_even_when_gpu_utilization_is_low(monkeypatch):
    monkeypatch.setattr(runner, "user_game_running", lambda: True)
    def forbidden():
        raise AssertionError("game process veto must precede GPU sampling")
    monkeypatch.setattr(runner, "gpu_sample", forbidden)
    with pytest.raises(RuntimeError, match="FIFA18"):
        runner.ensure_gpu_ready()


@pytest.mark.parametrize("output,expected", [
    ('"FIFA18.exe","1234","Console","1","100 K"\n', True),
    ('"fifa18.EXE","1234","Console","1","100 K"\n', True),
    ('INFO: No tasks are running which match the specified criteria.\n', False)])
def test_game_check_uses_only_exact_named_process(monkeypatch, output, expected):
    seen = []
    def fake_run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.user_game_running() is expected
    assert seen == [["tasklist.exe", "/FI", "IMAGENAME eq FIFA18.exe", "/FO", "CSV", "/NH"]]


def saved_run_fixture(tmp_path, monkeypatch):
    args, prepared, documents = preparation_fixture(tmp_path, monkeypatch)
    config = runner.prepare(args)
    pairs = runner.support_pairs(prepared, documents)
    gold = {("doc-secret", "question-secret"): annotation("one evidence")}
    monkeypatch.setattr(runner, "selected_gold", lambda *a: gold)
    _, audit = runner.encode_pairs(pairs, TinyTokenizer())
    scores = [1., 3., 2.]
    records, rankings = runner.evaluate(prepared, documents, gold, pairs, scores, TinyTokenizer())
    output = tmp_path / "saved_run"
    output.mkdir()
    runner.write_rows(output / "pair_scores.jsonl", [{**row, "raw_logit": score} for row, score in zip(audit, scores)])
    runner.write_rows(output / "per_question.jsonl", records)
    runner.write_rows(output / "rankings.jsonl", rankings)
    summary = {"status": "completed", "contract": runner.CONTRACT, "length_audit": config["length_audit"],
        "plan_sha256": runner.digest(args.output / "experiment_config.json"), "metrics": runner.aggregate(records),
        "output_sha256": {name: runner.digest(output / name) for name in runner.OUTPUT_FILES},
        "pair_count": 3, "question_count": 1, "family_count": 1, "record_count": 3}
    runner.write(output / "summary.json", summary)
    runner.write(output / "run_manifest.json", {"status": "completed", "summary_sha256": runner.digest(output / "summary.json"),
        "output_sha256": summary["output_sha256"], "plan_sha256": summary["plan_sha256"],
        "plan_manifest_sha256": runner.digest(args.output / "plan_manifest.json"),
        "pair_audit_sha256": runner.digest(args.output / "pair_audit.jsonl")})
    return args.output, output


def reseal_test_run(output):
    summary = runner.read(output / "summary.json")
    summary["output_sha256"] = {name: runner.digest(output / name) for name in runner.OUTPUT_FILES}
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    seal = runner.read(output / "run_manifest.json")
    seal["output_sha256"] = summary["output_sha256"]
    seal["summary_sha256"] = runner.digest(output / "summary.json")
    (output / "run_manifest.json").write_text(json.dumps(seal), encoding="utf-8")


def test_independent_audit_reproduces_saved_scores_without_model_or_gpu(tmp_path, monkeypatch):
    plan, output = saved_run_fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("audit must not load a model or probe GPU")
    monkeypatch.setattr(runner.AutoModelForSequenceClassification, "from_pretrained", forbidden)
    monkeypatch.setattr(runner, "ensure_gpu_ready", forbidden)
    result = runner.audit_saved_run(plan, output)
    assert result["status"] == "verified" and result["pairs"] == result["records"] == 3
    assert result["all_pair_identities_and_encoding_hashes_match"] is True
    assert result["all_source_plan_output_hashes_unchanged"] is True
    assert result["model_inference_performed"] is False
    text = json.dumps(result)
    assert "doc-secret" not in text and "question-secret" not in text and "forbidden question text" not in text


@pytest.mark.parametrize("name", list(runner.OUTPUT_FILES) + ["summary.json"])
def test_output_hash_tampering_is_rejected(tmp_path, monkeypatch, name):
    plan, output = saved_run_fixture(tmp_path, monkeypatch)
    path = output / name
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch|seal"):
        runner.audit_saved_run(plan, output)


@pytest.mark.parametrize("change", ["missing_score", "wrong_identity", "wrong_encoding", "reordered_scores",
                                     "changed_score", "missing_metric", "wrong_ranking", "changed_metric"])
def test_resealed_outputs_still_require_exact_pair_and_metric_reproduction(tmp_path, monkeypatch, change):
    plan, output = saved_run_fixture(tmp_path, monkeypatch)
    filename = "pair_scores.jsonl"
    if change == "missing_metric":
        filename = "per_question.jsonl"
    elif change == "wrong_ranking":
        filename = "rankings.jsonl"
    elif change == "changed_metric":
        summary = runner.read(output / "summary.json")
        summary["metrics"][0]["official_evidence_f1_question_macro"] = 99
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if change != "changed_metric":
        path = output / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if change in ("missing_score", "missing_metric"):
            rows.pop()
        elif change == "wrong_identity":
            rows[0]["unit_id"] = "different"
        elif change == "wrong_encoding":
            rows[0]["encoding_sha256"] = "0" * 64
        elif change == "reordered_scores":
            rows.reverse()
        elif change == "changed_score":
            rows[0]["raw_logit"] = 1000
        elif change == "wrong_ranking":
            rows[0]["ranked_ids"].reverse()
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    reseal_test_run(output)
    with pytest.raises(ValueError, match="saved score|identity|encoding|ranking|metric"):
        runner.audit_saved_run(plan, output)
