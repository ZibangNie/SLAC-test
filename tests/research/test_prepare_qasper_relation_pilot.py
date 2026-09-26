"""Gold-leakage, candidate identity and frozen-source checks for the API pilot."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import prepare_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import Unit


def units(count=40):
    return [Unit(f"u{i}", i, "paragraph", i * 10, (i + 1) * 10,
                 f"Visible unit {i}", f"Native unit {i}") for i in range(count)]


def fixture_rows(families=2, questions=3, count=40):
    candidates, qas, documents, ranks = [], [], {}, []
    for index in range(families):
        doc, family = f"d{index}", f"f{index}"
        candidates.append({"doc_id": doc, "family_id": family, "gold": "FORBIDDEN_GOLD"})
        documents[doc] = units(count)
        for q in range(questions):
            qid = f"{doc}-q{q}"
            qas.append({"doc_id": doc, "family_id": family, "question_id": qid,
                        "question": f"Question {q}?", "answer_annotations": [{"answer": "FORBIDDEN_GOLD"}]})
            ranks.append({"doc_id": doc, "question_id": qid,
                          "ranked_ids": [unit.unit_id for unit in documents[doc]]})
    return candidates, qas, documents, ranks


def test_hash_selection_is_identifier_only_order_invariant_and_capped():
    candidates, qas, _, _ = fixture_rows(families=10)
    selected = pilot.select_queries(candidates, qas)
    assert len(selected) == 16
    assert len({row["family_id"] for row in selected}) == 8
    assert selected == pilot.select_queries(list(reversed(candidates)), list(reversed(qas)))
    expected = sorted((row["family_id"] for row in candidates), key=pilot.selection_hash)[:8]
    assert list(dict.fromkeys(row["family_id"] for row in selected)) == expected
    assert all(set(row) == {"doc_id", "family_id", "question_id", "query"} for row in selected)


def test_neighbors_preserve_all_seeds_then_fill_left_right_in_seed_order():
    original = units()
    seeds = ["u10", "u20", "u30", "u0", "u5", "u15", "u25", "u35"]
    ranking = seeds + [u.unit_id for u in original if u.unit_id not in seeds]
    actual_seeds, chosen = pilot.expand_candidates(original, ranking)
    assert actual_seeds == seeds and len(chosen) == 16
    assert set(chosen) == set(seeds + ["u9", "u11", "u19", "u21", "u29", "u31", "u1", "u4"])
    assert chosen == sorted(chosen, key=lambda value: int(value[1:]))


def test_short_document_retains_all_units_without_padding():
    seeds, chosen = pilot.expand_candidates(units(3), ["u2", "u0", "u1"])
    assert seeds == ["u2", "u0", "u1"] and chosen == ["u0", "u1", "u2"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign", "extra_question", "duplicate_question"])
def test_rankings_reject_identity_corruption(change):
    _, qas, documents, ranks = fixture_rows()
    if change == "duplicate":
        ranks[0]["ranked_ids"][1] = ranks[0]["ranked_ids"][0]
    elif change == "missing":
        ranks.pop()
    elif change == "foreign":
        ranks[0]["ranked_ids"][0] = "unknown"
    elif change == "extra_question":
        ranks[0]["question_id"] = "other"
    else:
        ranks.append(dict(ranks[0]))
    with pytest.raises(ValueError):
        pilot.validate_rankings(ranks, qas, documents)


def test_static_edges_are_real_neighbors_deduplicated_without_query_or_gold():
    candidates, qas, documents, ranks = fixture_rows(families=1, questions=2)
    ranking = [f"u{i}" for i in (0, 5, 10, 15, 20, 25, 30, 35)]
    ranking += [u.unit_id for u in documents["d0"] if u.unit_id not in ranking]
    for row in ranks:
        row["ranked_ids"] = ranking
    validated = pilot.validate_rankings(ranks, qas, documents)
    prepared = pilot.build_prepared(candidates, qas, documents, validated)
    assert "FORBIDDEN_GOLD" not in json.dumps(prepared)
    assert len(prepared["support_tasks"]) == 32
    edges = {(t["left_id"], t["right_id"]) for t in prepared["static_tasks"]}
    assert len(edges) == len(prepared["static_tasks"])
    assert all(int(right[1:]) - int(left[1:]) == 1 for left, right in edges)
    assert all(set(task["item"]) == {"unit_a", "unit_b"} for task in prepared["static_tasks"])
    assert all(set(task["item"]) == {"query", "unit"} for task in prepared["support_tasks"])
    assert prepared == pilot.build_prepared(candidates, qas, documents, validated)


def test_unit_order_corruption_is_rejected():
    original = units(3)
    original[1] = replace(original[1], order=9)
    with pytest.raises(ValueError, match="original order"):
        pilot.expand_candidates(original, ["u0", "u1", "u2"])


def test_families_are_not_silently_duplicated():
    candidates, qas, _, _ = fixture_rows()
    candidates[1]["family_id"] = candidates[0]["family_id"]
    with pytest.raises(ValueError, match="unique family"):
        pilot.select_queries(candidates, qas)


def mock_run_inputs(tmp_path, monkeypatch):
    pool, aligned, dense = (tmp_path / name for name in ("pool", "aligned", "dense"))
    for path in (pool, aligned, dense):
        path.mkdir()
    candidates, qas, documents, ranks = fixture_rows(families=1, questions=2, count=6)
    paths = [pool / "pool_manifest.json", pool / "candidates.jsonl", pool / "native_qa_alignment.json",
             aligned / "native_qa_sidecar_v2.jsonl", aligned / "alignment_audit_v2.json",
             tmp_path / "documents.jsonl.gz", tmp_path / "qasper-train-dev-v0.3.tgz"]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    summary = {"status": "completed", "test_payload_read": False,
               "input_sha256": {str(path): pilot.digest(path) for path in paths},
               "question_count": 2, "document_count": 1}
    (dense / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (dense / "rankings.jsonl").write_text("\n".join(json.dumps(row) for row in ranks), encoding="utf-8")
    monkeypatch.setattr(pilot, "load_frozen_pool", lambda *args: ({}, candidates, qas))
    monkeypatch.setattr(pilot, "load_documents", lambda *args: (documents, paths[-2], paths[-1]))
    args = SimpleNamespace(pool=pool, sidecar=paths[3], dense=dense, output=tmp_path / "out")
    return args


def test_prepare_freezes_bytes_round_trip_and_refuses_existing_output(tmp_path, monkeypatch):
    args = mock_run_inputs(tmp_path, monkeypatch)
    result = pilot.run(args)
    assert result["counts"] == {"documents": 1, "families": 1, "queries": 2,
                                "static_tasks": 5, "support_tasks": 12, "candidates_per_query": {6: 2}}
    assert result["prepared_sha256"] == pilot.digest(args.output / "prepared.json")
    assert result["api_calls"] == 0 and result["model_loaded"] is False
    assert result["gold_in_model_payload"] is False and result["independent_evaluation"] is False
    assert "post-baseline" in result["exposure_status"]
    with pytest.raises(FileExistsError):
        pilot.run(args)


def test_dense_summary_must_bind_current_sidecar_before_preparing(tmp_path, monkeypatch):
    args = mock_run_inputs(tmp_path, monkeypatch)
    args.sidecar.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="current frozen pool"):
        pilot.run(args)
    assert not args.output.exists()


def test_changed_source_during_preparation_is_rejected(tmp_path, monkeypatch):
    args = mock_run_inputs(tmp_path, monkeypatch)
    original = pilot.build_prepared
    def modify_source(*values):
        result = original(*values)
        (args.dense / "rankings.jsonl").write_text("changed", encoding="utf-8")
        return result
    monkeypatch.setattr(pilot, "build_prepared", modify_source)
    with pytest.raises(ValueError, match="source hash mismatch"):
        pilot.run(args)
    assert not args.output.exists()
