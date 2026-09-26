"""Synthetic-only order-control tests; no real scoring, GPU, provider, or keys."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_qasper_native_owner_order as exp
from test_qasper_native_dual_index import source, annotation


def overwrite_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def no_cuda_or_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic must not use CUDA/model")
    monkeypatch.setattr(exp.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(exp.torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(exp.upstream.AutoModel, "from_pretrained", forbidden)


def case():
    s = source()
    scores = [.1, .2, .3, .9, 1., .05, .04, .03, .02, .01]
    s["candidate_vectors"][:, 0] = exp.torch.tensor(scores)
    scores = (s["candidate_vectors"] @ s["query_vectors"][0]).tolist()
    all_rows = exp.upstream.evaluate_query(s["prepared"]["queries"][0], s, scores, [.5, .4, .3, .2])
    return s, [r for r in all_rows if r["method"] in exp.OWNERS], all_rows


def test_fixed_candidate_set_only_order_changes_global_tie_policy():
    assert exp.order_candidates([3, 1, 2], [99., .5, .5, .1]) == [1, 2, 3]
    assert exp.order_candidates([3, 2, 1], [0.] * 4) == [1, 2, 3]
    assert exp.order_candidates([], [0.]) == []


@pytest.mark.parametrize("candidates,scores", [([0, 0], [1.]), ([True], [1.]), ([-1], [1.]),
    ([1], [1.]), ([0], [float("nan")]), ([0], [float("inf")]), (list(range(17)), [0.] * 17)])
def test_invalid_candidate_or_score_is_rejected(candidates, scores):
    with pytest.raises(ValueError):
        exp.order_candidates(candidates, scores)


def test_all_eight_records_exact_old_replay_and_unchanged_candidate_recall():
    s, owners, _ = case()
    rows = exp.evaluate(s, owners)
    assert len(rows) == 8
    original = {exp.row_key(r): r for r in owners}
    for row in rows:
        base = original[(row["scope"], row["owner_method"], *(row[k] for k in exp.IDENTITY))]
        assert row["candidate_global_indices"] == base["candidate_global_indices"]
        assert set(row["packing_order_global_indices"]) == set(base["candidate_global_indices"])
        for field in ("candidate_source_qualified_evidence_recall", "candidate_official_string_evidence_recall"):
            assert row[field] == base[field]
        assert row["actual_evidence_tokens"] <= 1024 and row["selected_units"] <= 3
        if row["ordering"] == "original_owner":
            restored = {k: v for k, v in row.items() if k not in (
                "owner_method", "ordering", "packing_order_global_indices", "leaf_scores_in_candidate_order")}
            restored["method"] = row["owner_method"]
            assert restored == exp.strip_times(base)
    assert any(rows[i]["selected_global_indices"] != rows[i+1]["selected_global_indices"] for i in range(0, 8, 2))


def test_labels_only_read_after_every_query_selection(monkeypatch):
    s, owners, _ = case()
    packing = exp.bridge.pack_source_qualified
    calls = []
    def tracked(*args, **kwargs):
        calls.append(1)
        return packing(*args, **kwargs)
    class GoldGuard(dict):
        def __getitem__(self, key):
            assert len(calls) == 8
            return super().__getitem__(key)
    s["qa_by_key"] = GoldGuard(s["qa_by_key"])
    monkeypatch.setattr(exp.bridge, "pack_source_qualified", tracked)
    exp.evaluate(s, owners)


def test_gold_changes_neither_control_ranking_nor_selection():
    s, owners, _ = case()
    first = exp.evaluate(s, owners)
    s["qa_by_key"][("a", "q")]["answer_annotations"] = [annotation(["different"])]
    counter = exp.upstream.PackCounter(s["tokenizer"], s["units"])
    changed = [{**exp.score_record(r, s, r["selected_global_indices"], counter),
                **{field: r[field] for field in exp.TIMES}} for r in owners]
    second = exp.evaluate(s, changed)
    for a, b in zip(first, second, strict=True):
        assert a["packing_order_global_indices"] == b["packing_order_global_indices"]
        assert a["selected_global_indices"] == b["selected_global_indices"]


def test_oversize_skipped_whole_and_same_source_dedup_preserved():
    s, owners, _ = case()
    s["units"][4] = replace(s["units"][4], text="X" * 2000, native_text="X" * 2000)
    s["units"][3] = replace(s["units"][3], text=s["units"][2].text, native_text=s["units"][2].native_text)
    counter = exp.upstream.PackCounter(s["tokenizer"], s["units"])
    refreshed = []
    for r in owners:
        selected = exp.bridge.pack_source_qualified(s["units"], s["keys"], r["candidate_global_indices"], 1024, counter)
        refreshed.append({**exp.score_record(r, s, selected, counter), **{k: r[k] for k in exp.TIMES}})
    result = exp.evaluate(s, refreshed)
    assert all(4 not in r["selected_global_indices"] for r in result)
    assert all(not {2, 3} <= set(r["selected_global_indices"]) for r in result)


def test_cross_document_identical_text_preserved_and_correct_source_scored():
    s, owners, _ = case()
    s["units"][5] = replace(s["units"][5], text=s["units"][0].text, native_text=s["units"][0].native_text)
    counter = exp.upstream.PackCounter(s["tokenizer"], s["units"])
    assert exp.bridge.pack_source_qualified(s["units"], s["keys"], [5, 0], 1024, counter) == [0, 5]
    base = next(r for r in owners if r["scope"] == "corpus_32")
    base["candidate_global_indices"] = [0, 5]
    wrong = exp.score_record(base, s, [5], counter)
    assert wrong["source_qualified_evidence_f1"] == 0


def test_original_record_corruption_refused():
    s, owners, _ = case()
    owners[0]["pack_sha256"] = "invalid"
    with pytest.raises(ValueError, match="original owner order"):
        exp.evaluate(s, owners)


def test_four_pairs_all_metrics_both_weightings_and_negative_changes_retained():
    s, owners, _ = case()
    rows = exp.evaluate(s, owners)
    result = exp.summarize(rows, s["prepared"]["queries"])
    assert len(result["comparisons"]) == 4
    assert all(len(c["comparison"]["metrics"]) == 16 for c in result["comparisons"])
    assert result == exp.summarize(rows, s["prepared"]["queries"])
    f1 = result["comparisons"][0]["comparison"]["metrics"]["source_qualified_evidence_f1"]
    assert f1["question_weighted"]["delta"] < 0
    assert f1["question_negative"] == 1
    assert f1["family_balanced"]["bootstrap_percentile_95"] == [f1["family_balanced"]["delta"]] * 2
    for comparison in result["comparisons"]:
        for metric in ("candidate_source_qualified_evidence_recall", "candidate_official_string_evidence_recall"):
            assert comparison["comparison"]["metrics"][metric]["question_weighted"]["delta"] == 0
        assert "question_wins" not in comparison["comparison"]["metrics"]["selected_units"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "candidate", "nonfinite"])
def test_incomplete_or_changed_denominator_and_candidate_metrics_fail(mutation):
    s, owners, _ = case()
    rows = exp.evaluate(s, owners)
    if mutation == "missing": rows.pop()
    if mutation == "duplicate": rows[-1] = deepcopy(rows[0])
    if mutation == "candidate": rows[1]["candidate_source_qualified_evidence_recall"] += .1
    if mutation == "nonfinite": rows[1]["official_evidence_f1"] = float("nan")
    with pytest.raises(ValueError): exp.summarize(rows, s["prepared"]["queries"])


def test_unequal_family_sizes_have_distinct_estimands():
    questions = [("f1", "a", "q1"), ("f1", "a", "q2"), ("f2", "b", "q3")]
    groups, draws = exp.statistics.family_resamples(questions)
    values = exp.statistics.clustered_delta(np.array([1., 1., -1.]), groups, draws, "official_evidence_f1")
    assert values["question_weighted"]["delta"] == pytest.approx(1/3)
    assert values["family_balanced"]["delta"] == 0
    assert draws.shape == (10000, 2)


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    s, owners, rows = case()
    monkeypatch.setattr(exp, "CONFIG", {**exp.CONFIG, "questions": 1, "families": 1, "leaves": 10, "records": 8})
    native_plan_dir = tmp_path / "native-plan"; native_plan_dir.mkdir()
    bound_path = native_plan_dir / "source.json"; bound_path.write_text("{}")
    bindings = {str(bound_path.resolve()): exp.upstream.digest(bound_path)}
    original = {"input_sha256": bindings, "identity": {"queries": s["prepared"]["queries"],
        "leaves": [{"doc_id": key[0]} for key in s["keys"]]}, "prepared": "p", "dense": "d", "chunks": "c"}
    plan_hashes = dict(bindings)
    monkeypatch.setattr(exp.upstream, "load_plan", lambda _: (original, [], plan_hashes))
    monkeypatch.setattr(exp.upstream, "environment", lambda: {"synthetic": "1"})
    native_run = tmp_path / "native-run"; native_run.mkdir()
    public = {"status": "completed"}
    exp.pilot.write_json(native_run / "public_aggregate.json", public)
    exp.pilot.write_json(native_run / "chunk_index.json", [])
    (native_run / "chunk_embeddings.safetensors").write_bytes(b"synthetic")
    (native_run / "per_question.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = {"status": "completed", "schema": exp.upstream.SCHEMA, "config": exp.upstream.CONFIG,
        "input_sha256": bindings, "plan_sha256": plan_hashes, "public": public,
        "output_sha256": {name: exp.upstream.digest(native_run/name) for name in exp.SOURCE_FILES if name != "summary.json"}}
    exp.pilot.write_json(native_run / "summary.json", report)
    receipt = tmp_path / "audit.json"
    exp.pilot.write_json(receipt, {"status": "verified", "records": 6, "configurations": 6,
        "saved_tensor_hash_index_norm_verified": True, "selection_metrics_and_comparisons_recomputed": True,
        "gpu_used": False, "model_loaded": False, "api_calls": 0})
    args = SimpleNamespace(native_plan=native_plan_dir, native_run=native_run, native_audit=receipt, output=tmp_path/"plan")
    return s, owners, args, original


def test_prepare_binds_sources_without_new_scoring_and_refuses_overwrite(plan_fixture, monkeypatch):
    s, owners, args, _ = plan_fixture
    monkeypatch.setattr(exp, "evaluate", lambda *_: (_ for _ in ()).throw(AssertionError("new scoring")))
    result = exp.prepare(args)
    assert result["new_control_scored"] is False and result["planned_records"] == 8
    plan, _ = exp.load_plan(args.output)
    assert plan["candidate_identity_sha256"]
    with pytest.raises(FileExistsError): exp.prepare(args)


@pytest.mark.parametrize("mutation", ["plan", "source", "receipt", "inventory"])
def test_prepared_tamper_and_source_changes_refused(plan_fixture, mutation):
    _, _, args, _ = plan_fixture
    exp.prepare(args)
    if mutation == "plan":
        p = exp.upstream.read_json(args.output/"plan.json"); p["config"]["max_selected_units"] = 2
        overwrite_json(args.output/"plan.json", p)
        overwrite_json(args.output/"plan_seal.json", {"plan.json": exp.upstream.digest(args.output/"plan.json")})
    if mutation == "source": (args.native_run/"per_question.jsonl").write_text("changed")
    if mutation == "receipt": Path(args.native_audit).write_text("{}")
    if mutation == "inventory": (args.output/"extra").write_text("extra")
    with pytest.raises(ValueError): exp.load_plan(args.output)


def test_full_upstream_audit_called_before_source_reuse(plan_fixture, monkeypatch):
    s, owners, args, original = plan_fixture
    exp.prepare(args); plan, _ = exp.load_plan(args.output)
    calls = []
    monkeypatch.setattr(exp.upstream, "audit", lambda a: calls.append((a.plan, a.run)) or {"status": "verified", "records": 6})
    s["input_sha256"] = original["input_sha256"]
    monkeypatch.setattr(exp.upstream, "load_inputs", lambda *a: s)
    loaded, saved = exp.verified_source(plan)
    assert loaded is s and len(saved) == 4 and len(calls) == 1
    monkeypatch.setattr(exp.upstream, "audit", lambda a: {"status": "incomplete", "records": 5})
    with pytest.raises(ValueError, match="upstream complete replay"): exp.verified_source(plan)


def reseal_run(directory):
    report = exp.upstream.read_json(directory/"summary.json")
    report["public"] = exp.upstream.read_json(directory/"public_aggregate.json")
    report["output_sha256"] = {name: exp.upstream.digest(directory/name) for name in exp.RUN_FILES[:-1]}
    overwrite_json(directory/"summary.json", report)


@pytest.mark.parametrize("tamper", [None, "aggregate", "records", "time", "extra"])
def test_complete_run_and_audit_reject_resealed_quality_changes(plan_fixture, monkeypatch, tamper):
    s, owners, args, _ = plan_fixture
    exp.prepare(args)
    monkeypatch.setattr(exp, "verified_source", lambda _: (s, owners))
    output = args.output.parent / "diagnostic-run"
    assert exp.run(SimpleNamespace(plan=args.output, output=output))["records"] == 8
    with pytest.raises(FileExistsError): exp.run(SimpleNamespace(plan=args.output, output=output))
    if tamper is None:
        assert exp.audit(SimpleNamespace(plan=args.output, run=output))["status"] == "verified"
        return
    if tamper == "aggregate":
        p = exp.upstream.read_json(output/"public_aggregate.json")
        p["comparisons"][0]["comparison"]["metrics"]["official_evidence_f1"]["question_weighted"]["delta"] = .999
        overwrite_json(output/"public_aggregate.json", p)
    if tamper == "records":
        rows = [json.loads(l) for l in (output/"per_question.jsonl").read_text().splitlines()]
        rows[1]["packing_order_global_indices"] = list(reversed(rows[1]["packing_order_global_indices"]))
        (output/"per_question.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if tamper == "time":
        p = exp.upstream.read_json(output/"public_aggregate.json"); p["execution"]["seconds_before_aggregate"] = -1
        overwrite_json(output/"public_aggregate.json", p)
    if tamper == "extra": (output/"extra").write_text("extra")
    reseal_run(output)
    with pytest.raises(ValueError): exp.audit(SimpleNamespace(plan=args.output, run=output))
