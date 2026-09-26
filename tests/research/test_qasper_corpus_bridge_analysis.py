"""All metrics/all families are retained and altered completed artifacts refuse."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_corpus_bridge as analysis


def fixtures():
    identities = [("f1", "d1", "q1"), ("f1", "d1", "q2"), ("f1", "d1", "q3"), ("f2", "d2", "q4")]
    prepared = {"queries": [dict(zip(analysis.bootstrap.IDENTITY, key)) for key in identities]}
    rows = []
    for method in analysis.bridge.METHODS:
        for i, key in enumerate(identities):
            value = .4 if method == "given_document" else (.1 if i < 3 else .8)
            row = {**dict(zip(analysis.bootstrap.IDENTITY, key)), "method": method}
            row.update({metric: 100 if metric == "actual_evidence_tokens" else value for metric in analysis.METRICS})
            rows.append(row)
    return prepared, rows


def test_all_eight_metrics_keep_negative_and_positive_families_and_shared_fixed_draws():
    prepared, rows = fixtures()
    first, local = analysis.descriptive_statistics(prepared, rows)
    second, _ = analysis.descriptive_statistics(prepared, rows)
    assert first == second and len(local) == 4
    assert set(first["paired_delta_corpus_minus_given"]) == set(analysis.METRICS)
    metric = first["paired_delta_corpus_minus_given"]["source_qualified_evidence_f1"]
    assert metric["question_weighted"]["delta"] == pytest.approx(-.125)
    assert metric["family_balanced"]["delta"] == pytest.approx(.05)
    assert metric["question_positive"] == 1 and metric["question_negative"] == 3
    assert metric["question_weighted"]["bootstrap_percentile_95"] == pytest.approx([-.3, .4])
    tokens = first["paired_delta_corpus_minus_given"]["actual_evidence_tokens"]
    assert tokens["question_weighted"]["bootstrap_percentile_95"] == [0, 0]
    assert "longer" in tokens["interpretation"]


def test_missing_query_or_metric_refuses_instead_of_changing_denominator():
    prepared, rows = fixtures()
    with pytest.raises(ValueError):
        analysis.descriptive_statistics(prepared, rows[:-1])
    rows[0]["source_qualified_evidence_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analysis.descriptive_statistics(prepared, rows)


def write_completed(directory):
    bridge = analysis.bridge
    bridge.pilot.write_json(directory / "experiment_config.json", {})
    bridge.pilot.write_json(directory / "corpus_unit_index.json", [])
    (directory / "per_question.jsonl").write_text("", encoding="utf-8")
    public = {"status": "completed"}
    bridge.pilot.write_json(directory / "public_aggregate.json", public)
    bridge.pilot.write_json(directory / "summary.json", {**public, "input_sha256": {}, "output_sha256": {
        path.name: bridge.digest(path) for path in directory.iterdir()}})


def test_saved_output_hash_and_exact_inventory_checked(tmp_path):
    write_completed(tmp_path)
    analysis.saved_artifacts(tmp_path)
    (tmp_path / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        analysis.saved_artifacts(tmp_path)
    (tmp_path / "extra.json").unlink()
    (tmp_path / "per_question.jsonl").write_text('{"fake":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        analysis.saved_artifacts(tmp_path)


def test_public_summary_mismatch_rejected_even_when_file_hash_resealed(tmp_path):
    write_completed(tmp_path)
    public = tmp_path / "public_aggregate.json"
    public.write_text(json.dumps({"status": "wrong"}), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary = analysis.bridge.read_json(summary_path)
    summary["output_sha256"][public.name] = analysis.bridge.digest(public)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="public aggregate"):
        analysis.saved_artifacts(tmp_path)
