"""Synthetic-only checks: no model, API, key, or real inference outputs."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_qasper_native_dual_index as experiment
from SLAC.retrieval.schemas.records import LeafRecord, ChunkRecord
from run_qasper_evidence_baselines import Unit, PackCounter


class Tokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return [1, *text.encode(), 2]


def annotation(evidence):
    return {"native_answer": {"unanswerable": False, "extractive_spans": [],
        "free_form_answer": "answer", "yes_no": None, "evidence": evidence}}


def source():
    documents = {doc: [Unit(f"u{i}", i, "paragraph", 10*i, 10*i+5,
                           f"{doc} text{i}", f"{doc} text{i}") for i in range(5)] for doc in ("a", "b")}
    units, keys, positions = experiment.bridge.globalize_documents(documents)
    leaves, chunks = [], []
    for doc in documents:
        for k, indices in enumerate(([0, 1, 2], [3, 4])):
            text = "\n\n".join(documents[doc][i].text for i in indices)
            cid = f"{doc}c{k}"
            chunks.append(ChunkRecord(doc, cid, k, indices[0], indices[-1]+1, text, len(indices), [], 0,
                token_est=len(text)+2, meta={"native_unit_ids": [f"u{i}" for i in indices]}))
            for i in indices:
                global_index = len(leaves)
                leaves.append(LeafRecord(doc, f"{doc}l{i}", cid, i, i, i+1, documents[doc][i].text, [], 0,
                    token_est=len(documents[doc][i].text)+2,
                    meta={"native_unit_id": f"u{i}", "cache_embedding_row": global_index}))
    q = {"family_id": "f", "doc_id": "a", "question_id": "q", "query": "question",
         "seed_ids": [f"u{i}" for i in range(5)], "candidate_ids": [f"u{i}" for i in range(5)],
         "ranked_ids": [f"u{i}" for i in range(5)]}
    vectors = torch.zeros((10, 1024)); vectors[:, 0] = 1
    queries = torch.zeros((1, 1024)); queries[:, 0] = 1
    return {"prepared": {"queries": [q]}, "documents": documents, "units": units, "keys": keys,
        "positions": positions, "leaves": leaves, "chunks": chunks, "tokenizer": Tokenizer(),
        "qa_by_key": {("a", "q"): {"answer_annotations": [annotation(["a text1"])]}},
        "query_positions": {("a", "q"): 0}, "rankings": {("a", "q"): [f"u{i}" for i in range(5)]},
        "candidate_vectors": vectors, "query_vectors": queries}


def test_shared_partition_source_scope_and_exact_projection():
    s = source(); q = s["prepared"]["queries"][0]
    leaves = [.7, .9, .8, .1, .2, 1., 1., 1., 1., 1.]
    chunks = [.1, .9, 1., 1.]
    candidates, trace = experiment.retrieve_candidates(q, s, leaves, chunks, "given_document", "leaf_owner")
    assert candidates == [1, 2, 0, 4, 3]
    assert all(s["keys"][i][0] == "a" for i in candidates)
    assert trace["chunk_hits"] == []
    first = trace["owners"][0]
    assert first["hit_count"] == 3 and first["owner_native_units"] == 3
    assert first["rrf_score"] == pytest.approx(1/61 + 1/62 + 1/63)
    assert first["source_views"] == ["leaf_dense"]


def test_dual_adds_one_channel_without_changing_leaf_hits_or_owner_partition():
    s = source(); q = s["prepared"]["queries"][0]
    scores = [1. - i*.05 for i in range(10)]
    _, leaf = experiment.retrieve_candidates(q, s, scores, [.1,.9,.8,.7], "corpus_32", "leaf_owner")
    _, dual = experiment.retrieve_candidates(q, s, scores, [.1,.9,.8,.7], "corpus_32", "dual_owner")
    assert leaf["leaf_hits"] == dual["leaf_hits"] == list(range(8))
    assert dual["chunk_hits"] == [1, 2, 3, 0]
    by_owner = {row["chunk_id"]: row for row in dual["owners"]}
    for row in leaf["owners"]:
        new = by_owner[row["chunk_id"]]
        assert new["projected_global_indices"] == row["projected_global_indices"]
        assert new["rrf_score"] > row["rrf_score"]


def test_dense_ties_follow_global_cache_order_and_cross_doc_ids_stay_distinct():
    s = source(); q = s["prepared"]["queries"][0]
    candidates, trace = experiment.retrieve_candidates(q, s, [0.] * 10, [0.] * 4, "corpus_32", "dual_owner")
    assert trace["leaf_hits"] == list(range(8))
    assert trace["chunk_hits"] == list(range(4))
    assert len(candidates) == len(set(candidates)) == 10
    assert len({s["keys"][i] for i in candidates}) == 10
    assert len({s["keys"][i][1] for i in candidates}) == 5


def test_wrong_source_same_text_not_scored_as_correct_and_pack_keeps_both_sources():
    s = source(); q = s["prepared"]["queries"][0]
    s["units"][0] = replace(s["units"][0], native_text="same", text="same")
    s["units"][5] = replace(s["units"][5], native_text="same", text="same")
    count = PackCounter(s["tokenizer"], s["units"])
    assert experiment.bridge.pack_source_qualified(s["units"], s["keys"], [5, 0], 1024, count) == [0, 5]
    assert experiment.bridge.source_qualified_metrics([("b", "same")], "a", [annotation(["same"])])["evidence_f1"] == 0


def test_all_six_arms_whole_pack_budget_and_gold_independence():
    s = source(); q = s["prepared"]["queries"][0]
    s["units"][0] = replace(s["units"][0], native_text="x" * 2000, text="x" * 2000)
    first = experiment.evaluate_query(q, s, list(range(10)), list(range(4)))
    s["qa_by_key"][("a", "q")]["answer_annotations"] = [annotation(["different gold"])]
    second = experiment.evaluate_query(q, s, list(range(10)), list(range(4)))
    assert len(first) == 6
    assert all(row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= 3 for row in first)
    assert all(0 not in row["selected_global_indices"] for row in first)
    for left, right in zip(first, second):
        assert left["candidate_global_indices"] == right["candidate_global_indices"]
        assert left["selected_global_indices"] == right["selected_global_indices"]


def test_projection_cap_preserves_owner_order_then_leaf_score():
    s = source(); q = s["prepared"]["queries"][0]
    # One owner larger than the cap, with additional native units in the same doc.
    for i in range(5, 25):
        s["leaves"].append(LeafRecord("a", f"extra{i}", "ac0", i, i, i+1, "extra", [], 0,
            meta={"native_unit_id": f"extra{i}"}))
        s["keys"].append(("a", f"extra{i}")); s["positions"]["a"].append(len(s["leaves"])-1)
    scores = [0.] * len(s["leaves"])
    for i in range(10, len(scores)):
        scores[i] = float(i)
    candidates, trace = experiment.retrieve_candidates(q, s, scores, [0.] * 4, "given_document", "leaf_owner")
    assert candidates == list(reversed(range(14, 30)))
    assert len(candidates) == 16 and trace["owners"][0]["hit_count"] == 8


@pytest.mark.parametrize("field,value", [("fifa_process_present", True), ("compute_process_present", True),
    ("gpu_utilization_percent", 11), ("memory_free_mib", 4095), ("memory_total_mib", 6000)])
def test_idle_gate_refuses_foreground_game_even_low_utilization(field, value):
    sample = {"fifa_process_present": False, "compute_process_present": False,
              "gpu_utilization_percent": 0, "memory_used_mib": 100, "memory_total_mib": 8192, "memory_free_mib": 8092}
    sample[field] = value
    with pytest.raises(RuntimeError):
        experiment.assert_idle(sample)


def test_three_idle_samples_and_two_short_pauses_only():
    sleeps, calls = [], []
    def sample():
        calls.append(1)
        return {"fifa_process_present": False, "compute_process_present": False,
                "gpu_utilization_percent": 0, "memory_used_mib": 100, "memory_total_mib": 8192, "memory_free_mib": 8092}
    assert len(experiment.confirm_idle(sample, sleeps.append)) == 3
    assert len(calls) == 3 and sleeps == [5, 5]


def test_invalid_arm_nonfinite_scores_and_scope_dimensions_refused():
    s = source(); q = s["prepared"]["queries"][0]
    for scope, method, leaf in [("wrong", "leaf_owner", [0.]*10), ("corpus_32", "wrong", [0.]*10),
                               ("corpus_32", "leaf_owner", [float("nan")]*10), ("corpus_32", "leaf_owner", [0.])]:
        with pytest.raises(ValueError):
            experiment.retrieve_candidates(q, s, leaf, [0.]*4, scope, method)


def test_all_negative_and_positive_pairs_reported_with_fixed_cluster_bootstrap():
    s = source(); q = s["prepared"]["queries"][0]
    rows = experiment.evaluate_query(q, s, list(range(10)), list(range(4)))
    expanded, queries = [], []
    for i, family in enumerate(("f1", "f1", "f2")):
        queries.append({**q, "family_id": family, "question_id": f"q{i}"})
        for row in rows:
            r = {**row, "family_id": family, "question_id": f"q{i}"}
            r["source_qualified_evidence_f1"] = {"leaf_direct": .8, "leaf_owner": .6, "dual_owner": .2}[r["method"]]
            expanded.append(r)
    first = experiment.summarize(expanded, queries)
    assert first == experiment.summarize(expanded, queries)
    assert len(first) == 2 and all(len(scope["paired_comparisons"]) == 3 for scope in first)
    for scope in first:
        for pair in scope["paired_comparisons"]:
            metric = pair["metrics"]["source_qualified_evidence_f1"]
            assert metric["question_negative"] == 3 and metric["question_positive"] == 0
            assert metric["question_weighted_percentile95"][1] < 0
    with pytest.raises(ValueError, match="denominator"):
        experiment.summarize(expanded[:-1], queries)
    with pytest.raises(ValueError, match="denominator"):
        experiment.summarize(expanded + [expanded[0]], queries)


def prepared_fixture(tmp_path, monkeypatch):
    s = source(); model = tmp_path / "model"; model.mkdir()
    weights = model / "model.safetensors"; weights.write_bytes(b"not a real model")
    s["native_manifest"] = {"tokenizer": str(model)}
    s["input_sha256"] = {str(weights.resolve()): experiment.digest(weights)}
    monkeypatch.setattr(experiment, "load_inputs", lambda *a: s)
    monkeypatch.setattr(experiment, "environment", lambda: {"test": "synthetic"})
    monkeypatch.setattr(experiment.AutoModel, "from_pretrained", lambda *a, **k: pytest.fail("never load model during preparation"))
    output = tmp_path / "plan"
    result = experiment.prepare(SimpleNamespace(prepared=tmp_path / "p", dense=tmp_path / "d", chunks=tmp_path / "c", output=output))
    return s, output, result


def test_prepare_freezes_inputs_tokens_and_spec_without_gpu_model_or_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "confirm_idle", lambda: pytest.fail("preparation must not query/use GPU"))
    s, path, result = prepared_fixture(tmp_path, monkeypatch)
    assert result["gpu_execution_started"] is False
    plan, ids, hashes = experiment.load_plan(path)
    assert plan["status"] == "prepared_not_executed" and plan["model_loaded"] is False
    assert len(ids) == 4 and set(Path(p).name for p in hashes) == {"plan.json", "chunk_token_ids.json", "plan_seal.json"}
    assert "answer_annotations" not in json.dumps(plan) and "answer" not in json.dumps(ids)
    experiment.validate_reloaded(plan, ids, s)
    with pytest.raises(FileExistsError):
        experiment.prepare(SimpleNamespace(output=path))


def test_source_binding_ignores_torch_synthetic_relative_module_filenames(monkeypatch):
    baseline = experiment.source_hashes()
    monkeypatch.setitem(sys.modules, "synthetic_torch_module", SimpleNamespace(__file__="_classes.py"))
    monkeypatch.setitem(sys.modules, "unrelated_audit_caller", SimpleNamespace(__file__=str(Path(__file__).resolve())))
    hashes = experiment.source_hashes()
    assert hashes == baseline
    assert str(Path(experiment.__file__).resolve()) in hashes
    assert not any(Path(path).name == "_classes.py" for path in hashes)


def test_plan_seal_tamper_extra_file_source_change_and_token_replay_refused(tmp_path, monkeypatch):
    s, path, _ = prepared_fixture(tmp_path, monkeypatch)
    plan, ids, _ = experiment.load_plan(path)
    changed = deepcopy(ids); changed[0][0] += 1
    with pytest.raises(ValueError, match="token IDs"):
        experiment.validate_reloaded(plan, changed, s)
    extra = path / "unexpected"; extra.write_text("x")
    with pytest.raises(ValueError, match="inventory"):
        experiment.load_plan(path)
    extra.unlink()
    (path / "chunk_token_ids.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="seal"):
        experiment.load_plan(path)


def test_prepare_rejects_overlength_without_truncation(tmp_path, monkeypatch):
    s = source(); s["chunks"][0].text = "x" * 9000
    monkeypatch.setattr(experiment, "load_inputs", lambda *a: s)
    with pytest.raises(ValueError, match="without truncation"):
        experiment.prepare(SimpleNamespace(prepared="p", dense="d", chunks="c", output=tmp_path / "new"))
    assert not (tmp_path / "new").exists()


def test_explicit_run_gate_and_busy_gpu_keep_prepared_state(tmp_path, monkeypatch):
    s, path, _ = prepared_fixture(tmp_path, monkeypatch)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="explicit"):
        experiment.run(SimpleNamespace(confirm_idle=False, plan=path, output=output))
    def busy():
        raise RuntimeError("FIFA18 active")
    monkeypatch.setattr(experiment, "confirm_idle", busy)
    with pytest.raises(RuntimeError, match="FIFA"):
        experiment.run(SimpleNamespace(confirm_idle=True, plan=path, output=output))
    assert not output.exists()
    assert experiment.load_plan(path)[0]["status"] == "prepared_not_executed"


def test_synthetic_full_run_output_seals_and_failure_stop_no_retry(tmp_path, monkeypatch):
    s, path, _ = prepared_fixture(tmp_path, monkeypatch)
    idle = {"fifa_process_present": False, "compute_process_present": False, "gpu_utilization_percent": 0,
            "memory_used_mib": 100, "memory_total_mib": 8192, "memory_free_mib": 8092}
    monkeypatch.setattr(experiment, "confirm_idle", lambda: [idle] * 3)
    vectors = torch.zeros((4, 1024)); vectors[:, 0] = 1
    monkeypatch.setattr(experiment, "encode_chunks", lambda *args: (vectors, {"encoding_wall_seconds": .00001,
        "device": "synthetic", "cuda_runtime": "synthetic", "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}))
    result = experiment.run(SimpleNamespace(confirm_idle=True, plan=path, output=tmp_path / "run"))
    assert result["status"] == "completed"
    report = experiment.read_json(tmp_path / "run/summary.json")
    assert report["status"] == "completed" and len(report["public"]["scopes"]) == 2
    for name, expected in report["output_sha256"].items():
        assert experiment.digest(tmp_path / "run" / name) == expected
    assert experiment.audit(SimpleNamespace(plan=path, run=tmp_path / "run"))["status"] == "verified"
    public = experiment.read_json(tmp_path / "run/public_aggregate.json")
    public["scopes"][0]["methods"][0]["metrics"]["source_qualified_evidence_f1"]["question_weighted"] = .987
    (tmp_path / "run/public_aggregate.json").write_text(json.dumps(public), encoding="utf-8")
    report["public"] = public
    report["output_sha256"]["public_aggregate.json"] = experiment.digest(tmp_path / "run/public_aggregate.json")
    (tmp_path / "run/summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        experiment.audit(SimpleNamespace(plan=path, run=tmp_path / "run"))
    def oom(*a):
        raise torch.OutOfMemoryError("synthetic")
    monkeypatch.setattr(experiment, "encode_chunks", oom)
    with pytest.raises(torch.OutOfMemoryError):
        experiment.run(SimpleNamespace(confirm_idle=True, plan=path, output=tmp_path / "failed"))
    failure = experiment.read_json(tmp_path / "failed/summary.json")
    assert failure["status"] == "failed" and failure["retry_or_fallback"] is False
    assert not (tmp_path / "failed/public_aggregate.json").exists()


@pytest.mark.parametrize("field,value", [("api_calls", 1), ("model_loaded", True),
    ("gpu_execution_started", True), ("status", "completed"), ("limits", [])])
def test_plan_fixed_metadata_resealed_tampering_refused(tmp_path, monkeypatch, field, value):
    _, path, _ = prepared_fixture(tmp_path, monkeypatch)
    plan = experiment.read_json(path / "plan.json"); plan[field] = value
    (path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    seal = experiment.read_json(path / "plan_seal.json"); seal["plan.json"] = experiment.digest(path / "plan.json")
    (path / "plan_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="specification"):
        experiment.load_plan(path)


def test_resealed_model_path_substitution_fails_before_gpu(tmp_path, monkeypatch):
    _, path, _ = prepared_fixture(tmp_path, monkeypatch)
    plan = experiment.read_json(path / "plan.json"); plan["model"] = str(tmp_path / "other-unbound-model")
    (path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    seal = experiment.read_json(path / "plan_seal.json"); seal["plan.json"] = experiment.digest(path / "plan.json")
    (path / "plan_seal.json").write_text(json.dumps(seal), encoding="utf-8")
    monkeypatch.setattr(experiment, "confirm_idle", lambda: pytest.fail("must fail before GPU idle sampling"))
    with pytest.raises(ValueError, match="model path"):
        experiment.run(SimpleNamespace(confirm_idle=True, plan=path, output=tmp_path / "run"))
