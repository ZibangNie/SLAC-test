"""Pre-inference synthetic tests for all-query, paired reranker statistics."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_reranker_baseline as analysis
from run_qasper_evidence_baselines import Unit


def small_records():
    prepared = {"queries": [{"family_id": family, "doc_id": family + "-doc", "question_id": f"q{i}"}
                for i, family in enumerate(("family-a", "family-a", "family-b"))]}
    dense, reranked = [], []
    for i, query in enumerate(prepared["queries"]):
        for k in analysis.SIZES:
            effect = 0. if k == 1 else -.4 if k == 2 else (.4 if i < 2 else -.4)
            token_effect = 0 if k == 1 else -10 if k == 2 else (10 if i < 2 else -10)
            row = {**query, "method": f"dense_k{k}", "budget": 1024,
                   **{metric: .5 for metric in analysis.METRICS if metric != "actual_evidence_tokens"},
                   "actual_evidence_tokens": 100}
            dense.append(row)
            reranked.append({**row, "method": f"bge_reranker_v2_m3_k{k}",
                **{metric: .5 + effect for metric in analysis.METRICS if metric != "actual_evidence_tokens"},
                "actual_evidence_tokens": 100 + token_effect})
    return prepared, reranked, dense


def test_primary_k3_all_signs_and_all_four_metrics_retained():
    prepared, reranked, dense = small_records()
    result, private = analysis.statistics_for(prepared, reranked, dense)
    assert result["primary_k"] == 3 and result["sensitivity_k"] == [1, 2]
    assert result["best_k_selected"] is False
    assert [row["k"] for row in result["paired_comparisons"]] == [3, 1, 2]
    assert [row["role"] for row in result["paired_comparisons"]] == ["primary", "sensitivity", "sensitivity"]
    assert len(result["method_means"]) == 6 and len(private) == 9
    assert all(set(row["metrics"]) == set(analysis.METRICS) for row in result["paired_comparisons"])
    assert result["paired_comparisons"][1]["metrics"]["official_evidence_f1"]["question_ties"] == 3
    assert result["paired_comparisons"][2]["metrics"]["official_evidence_f1"]["question_losses"] == 3


def test_family_balanced_and_question_weighted_denominators_are_distinct():
    prepared, reranked, dense = small_records()
    result, _ = analysis.statistics_for(prepared, reranked, dense)
    metric = result["paired_comparisons"][0]["metrics"]["official_evidence_f1"]
    assert metric["question_weighted"]["delta"] == pytest.approx(.4 / 3)
    assert metric["family_balanced"]["delta"] == pytest.approx(0.)
    assert metric["families"] == 2 and metric["questions"] == 3
    assert metric["question_wins"] == 2 and metric["question_losses"] == 1
    assert metric["family_balanced"]["bootstrap_percentile_95"] == pytest.approx([-.4, .4])
    tokens = result["paired_comparisons"][0]["metrics"]["actual_evidence_tokens"]
    assert tokens["question_token_increases"] == 2 and tokens["question_token_decreases"] == 1
    assert "not better quality" in tokens["interpretation"]


def test_same_draw_matrix_is_reused_for_every_k_and_metric(monkeypatch):
    prepared, reranked, dense = small_records()
    seen = []
    original = analysis.bootstrap.clustered_delta
    def observe(values, groups, draws, metric):
        seen.append((id(draws), draws.shape))
        return original(values, groups, draws, metric)
    monkeypatch.setattr(analysis.bootstrap, "clustered_delta", observe)
    first, _ = analysis.statistics_for(prepared, reranked, dense)
    assert len(seen) == 12 and len({item[0] for item in seen}) == 1
    assert {item[1] for item in seen} == {(10000, 2)}
    second, _ = analysis.statistics_for(prepared, list(reversed(reranked)), list(reversed(dense)))
    assert first == second


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_family", "unexpected_method", "nonfinite", "tokens"])
def test_statistical_denominator_and_metrics_fail_closed(mutation):
    prepared, reranked, dense = small_records()
    if mutation == "missing":
        reranked.pop()
    elif mutation == "duplicate":
        reranked.append(reranked[0])
    elif mutation == "wrong_family":
        reranked[0]["family_id"] = "wrong"
    elif mutation == "unexpected_method":
        reranked[0]["method"] = "tuned_winner"
    elif mutation == "nonfinite":
        reranked[0]["official_evidence_f1"] = float("nan")
    else:
        reranked[0]["actual_evidence_tokens"] = 1025
    with pytest.raises(ValueError):
        analysis.statistics_for(prepared, reranked, dense)


class TinyTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return [0] + [5] * len(text.split()) + [2]


def complete_fixture(tmp_path, monkeypatch):
    plan_dir = tmp_path / "reranker_plan"
    plan_dir.mkdir()
    for name in analysis.PLAN_FILES:
        (plan_dir / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "frozen_source.json"
    source.write_text("fixed", encoding="utf-8")
    queries, documents, annotations = [], {}, {}
    for family_index in range(24):
        doc, family = f"private_doc_{family_index}", f"private_family_{family_index}"
        documents[doc] = [Unit("unit", 0, "paragraph", 0, 1, "evidence", "evidence")]
        for question_index in range(4 if family_index < 5 else 3):
            qid = f"private_question_{family_index}_{question_index}"
            queries.append({"family_id": family, "doc_id": doc, "question_id": qid,
                "query": f"PRIVATE QUERY TEXT {family_index} {question_index}",
                "candidate_ids": ["unit"], "ranked_ids": ["unit"]})
            annotations[(doc, qid)] = [{"native_answer": {"unanswerable": False,
                "extractive_spans": [], "free_form_answer": "PRIVATE ANSWER", "yes_no": None,
                "evidence": ["evidence"]}}]
    prepared = {"queries": queries}
    manifest = {"source_paths": {"sidecar": "unused"}}
    plan = {"question_count": 77, "pair_count": 1214, "prepared_dir": str(tmp_path / "prepared"),
        "bge_tokenizer": str(tmp_path / "bge"), "input_sha256": {str(source): analysis.reranker.digest(source)}}
    monkeypatch.setattr(analysis.reranker, "load_plan", lambda path: plan)
    monkeypatch.setattr(analysis.reranker.preparation, "load_prepared", lambda path: (prepared, manifest, documents))
    monkeypatch.setattr(analysis.reranker, "selected_gold", lambda *args: annotations)
    monkeypatch.setattr(analysis.reranker.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: TinyTokenizer())
    spec_dir = tmp_path / "analysis_spec"
    config = analysis.freeze(SimpleNamespace(reranker_plan=plan_dir, output=spec_dir))
    run_dir = tmp_path / "reranker_run"
    run_dir.mkdir()
    dense = analysis.stage.baseline_records(prepared, documents, annotations, TinyTokenizer())
    rows = [{**row, "method": row["method"].replace("dense_", "bge_reranker_v2_m3_")} for row in dense]
    for name in analysis.RUN_FILES:
        if name == "per_question.jsonl":
            analysis.reranker.write_rows(run_dir / name, rows)
        else:
            (run_dir / name).write_text("{}", encoding="utf-8")
    audit = {"status": "verified", "queries": 77, "pairs": 1214, "records": 231,
        "all_pair_identities_and_encoding_hashes_match": True, "all_rankings_and_metrics_reproduced": True,
        "all_source_plan_output_hashes_unchanged": True, "model_inference_performed": False}
    seen = []
    def verified(plan_path, run_path, **kwargs):
        seen.append((Path(plan_path), Path(run_path)))
        return copy.deepcopy(audit)
    monkeypatch.setattr(analysis.reranker, "audit_saved_run", verified)
    args = SimpleNamespace(analysis_plan=spec_dir, run=run_dir, output=tmp_path / "analysis")
    return args, config, source, audit, seen, prepared


def test_full77_analysis_requires_audit_then_recomputes_dense_and_outputs_no_public_ids(tmp_path, monkeypatch):
    args, config, source, audit, seen, prepared = complete_fixture(tmp_path, monkeypatch)
    result = analysis.analyze(args)
    assert len(seen) == 1
    assert result["statistics"]["question_count"] == 77 and result["statistics"]["family_count"] == 24
    assert result["api_calls"] == 0 and result["paid_outputs_read"] is False
    assert len(analysis.read_rows(args.output / "dense_per_question.jsonl")) == 231
    assert len(analysis.read_rows(args.output / "paired_per_question.jsonl")) == 231
    public = json.dumps(result, ensure_ascii=False)
    for q in prepared["queries"]:
        for name in ("family_id", "doc_id", "question_id", "query"):
            assert q[name] not in public
    assert "PRIVATE ANSWER" not in public
    assert "input_sha256" not in result
    seal = analysis.reranker.read(args.output / "analysis_output_manifest.json")
    assert seal["public_aggregate_sha256"] == analysis.reranker.digest(args.output / "public_aggregate.json")
    assert all(analysis.reranker.digest(args.output / name) == value for name, value in seal["local_output_sha256"].items())


@pytest.mark.parametrize("field,value", [("status", "partial"), ("queries", 76), ("pairs", 1213),
    ("records", 230), ("all_rankings_and_metrics_reproduced", False)])
def test_partial_or_unverified_reranker_has_no_primary_result(tmp_path, monkeypatch, field, value):
    args, _, _, audit, _, _ = complete_fixture(tmp_path, monkeypatch)
    audit[field] = value
    with pytest.raises(ValueError, match="complete audited"):
        analysis.analyze(args)
    assert not args.output.exists()


def test_plan_source_tamper_refuses_before_any_output_audit(tmp_path, monkeypatch):
    args, _, source, _, seen, _ = complete_fixture(tmp_path, monkeypatch)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.analyze(args)
    assert seen == [] and not args.output.exists()


def test_source_or_run_mutation_during_audit_is_rejected(tmp_path, monkeypatch):
    args, _, source, audit, _, _ = complete_fixture(tmp_path, monkeypatch)
    def changing(*args, **kwargs):
        source.write_text("changed", encoding="utf-8")
        return audit
    monkeypatch.setattr(analysis.reranker, "audit_saved_run", changing)
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.analyze(args)
    assert not args.output.exists()


def test_missing_persisted_record_rejected_even_if_mock_audit_claims_complete(tmp_path, monkeypatch):
    args, _, _, _, _, _ = complete_fixture(tmp_path, monkeypatch)
    path = args.run / "per_question.jsonl"
    records = analysis.read_rows(path)
    path.write_text("\n".join(json.dumps(row) for row in records[:-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.analyze(args)
    assert not args.output.exists()


def test_specification_seal_and_scope_cannot_be_silently_changed(tmp_path, monkeypatch):
    args, _, _, _, _, _ = complete_fixture(tmp_path, monkeypatch)
    path = args.analysis_plan / "analysis_config.json"
    config = analysis.reranker.read(path)
    config["specification"]["primary_k"] = 2
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="seal differs"):
        analysis.load_specification(args.analysis_plan)
    seal_path = args.analysis_plan / "analysis_manifest.json"
    seal = analysis.reranker.read(seal_path)
    seal["analysis_config_sha256"] = analysis.reranker.digest(path)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="specification differs"):
        analysis.load_specification(args.analysis_plan)


def test_bootstrap_contract_change_is_rejected(monkeypatch):
    monkeypatch.setattr(analysis.bootstrap, "BOOTSTRAP_SEED", 1)
    with pytest.raises(ValueError, match="contract changed"):
        analysis.validate_implementation_contract()


def test_source_binding_conflicts_cannot_be_overwritten(tmp_path):
    path = str(tmp_path / "same")
    with pytest.raises(ValueError, match="conflicting"):
        analysis.merge_bindings({path: "a"}, {path: "b"})
