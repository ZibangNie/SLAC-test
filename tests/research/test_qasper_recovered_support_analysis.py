"""Denominator, complete-result and source-integrity gates for amended support."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import analyze_qasper_recovered_support as analysis


def fixture():
    queries = [{"family_id": f"synthetic-family-{i % 24:02d}",
                "doc_id": f"synthetic-document-{i % 24:02d}",
                "question_id": f"synthetic-question-{i:02d}"} for i in range(77)]
    records = [{**query, "method": method, "official_evidence_f1": .25,
                "reference_evidence_recall": .5, "official_text_only_evidence_f1": .25,
                "actual_evidence_tokens": 100}
               for method in analysis.stats.SUPPORT_METHODS for query in queries]
    return {"queries": queries}, records


def test_complete_all_pairs_and_weights_including_ties():
    prepared, records = fixture()
    result, local = analysis.summarize(prepared, records)
    assert (result["record_count"], result["reported_interval_count"], len(local)) == (1155, 240, 2310)
    assert result["specification"] == analysis.stats.SPECIFICATION
    assert [(p["plus"], p["minus"]) for p in result["support"]["paired_comparisons"]] == list(analysis.stats.SUPPORT_PAIRS)
    for pair in result["support"]["paired_comparisons"]:
        assert set(pair["metrics"]) == set(analysis.stats.SUPPORT_METRICS)
        for row in pair["metrics"].values():
            assert row["question_ties"] == 77
            for weighting in ("question_weighted", "family_balanced"):
                assert row[weighting] == {"delta": 0., "bootstrap_percentile_95": [0., 0.]}


@pytest.mark.parametrize("change", ["missing_question", "wrong_family_count", "duplicate", "missing_method", "foreign_method", "invalid_tokens", "invalid_score"])
def test_scope_or_metric_change_rejected(change):
    prepared, records = fixture()
    if change == "missing_question": prepared["queries"].pop()
    elif change == "wrong_family_count": prepared["queries"][0]["family_id"] = "synthetic-extra-family"
    elif change == "duplicate": records.append(copy.deepcopy(records[0]))
    elif change == "missing_method": records = records[77:]
    elif change == "foreign_method": records[0]["method"] = "unplanned"
    elif change == "invalid_tokens": records[0]["actual_evidence_tokens"] = 1025
    else: records[0]["official_evidence_f1"] = float("nan")
    with pytest.raises(ValueError): analysis.summarize(prepared, records)


def setup_analysis(tmp_path):
    plan, run, output = (tmp_path / name for name in ("plan", "run", "analysis"))
    plan.mkdir(); run.mkdir()
    (plan / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text("{}", encoding="utf-8")
    prepared, records = fixture()
    data = ({"input_sha256": {}}, prepared, {}, records,
            {"status": "completed", "all_results_available": True})
    return Namespace(plan=plan, run=run, output=output), data


def test_incomplete_recovery_never_scores_or_writes(tmp_path, monkeypatch):
    args, data = setup_analysis(tmp_path)
    data[-1]["all_results_available"] = False
    monkeypatch.setattr(analysis, "summarize", lambda *a: pytest.fail("partial quality was computed"))
    with pytest.raises(ValueError, match="complete recovery"):
        analysis.analyze(args, verifier=lambda *a: data)
    assert not args.output.exists()


def test_changed_completed_source_is_not_published(tmp_path):
    args, data = setup_analysis(tmp_path)
    def verifier(*a):
        (args.run / "summary.json").write_text('{"changed":true}', encoding="utf-8")
        return data
    with pytest.raises(ValueError, match="source directory changed"):
        analysis.analyze(args, verifier=verifier)
    assert not args.output.exists()


def test_bound_data_mutation_fails_before_publication(tmp_path):
    args, data = setup_analysis(tmp_path)
    source = tmp_path / "bound.json"
    source.write_text("before", encoding="utf-8")
    data[0]["input_sha256"][str(source)] = analysis.digest(source)
    source.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError): analysis.analyze(args, verifier=lambda *a: data)
    assert not args.output.exists()


def test_existing_analysis_is_never_overwritten(tmp_path):
    args, data = setup_analysis(tmp_path)
    args.output.mkdir()
    with pytest.raises(FileExistsError):
        analysis.analyze(args, verifier=lambda *a: pytest.fail("existing output accepted"))


def test_complete_outputs_have_verified_input_and_output_seals(tmp_path):
    args, data = setup_analysis(tmp_path)
    report = analysis.analyze(args, verifier=lambda *a: data)
    saved = json.loads((args.output / "analysis.json").read_text(encoding="utf-8"))
    assert saved["recovery_summary_sha256"] == analysis.digest(args.run / "summary.json")
    assert report["api_calls"] == 0 and report["key_read"] is False
    for name, expected in saved["local_output_files_sha256"].items():
        assert analysis.digest(args.output / name) == expected
