"""Frozen 77-question preparation, native identity, and source-tamper checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import prepare_qasper_extended_development as extended
import prepare_qasper_relation_pilot as pilot
from test_prepare_qasper_relation_pilot import fixture_rows, units


def fixture_inventory():
    candidates, qas, documents, ranks = [], [], {}, []
    for index in range(32):
        doc, family = f"d{index}", f"f{index}"
        candidates.append({"doc_id": doc, "family_id": family, "official_split": "validation"})
        documents[doc] = units(20)
        count = 2 if index < 8 else 4 if index < 13 else 3
        seeds = ["u0", "u3", "u6", "u9", "u12", "u15", "u18", "u1"]
        ranking = seeds + [u.unit_id for u in documents[doc] if u.unit_id not in seeds]
        for number in range(count):
            qid = f"{doc}-q{number}"
            qas.append({"doc_id": doc, "family_id": family, "question_id": qid, "official_split": "validation",
                        "question": f"Question {qid}", "answer_annotations": [{"hidden": "FORBIDDEN_GOLD"}]})
            ranks.append({"doc_id": doc, "question_id": qid, "ranked_ids": ranking})
    original = {"queries": [{name: row[name] for name in extended.IDENTITY} for row in qas if int(row["doc_id"][1:]) < 8]}
    inventory = {"schema": "slac-extended-development-inventory-v1",
        "status": "denominator_frozen_not_an_executable_api_plan", "families": 24, "questions": 77,
        "independent_confirmation": False, "gold_or_model_error_used_for_selection": False,
        "excluded_pilot_families": [f"f{i}" for i in range(8)], "documents": candidates[8:],
        "query_inventory": [{name: row[name] for name in extended.IDENTITY} for row in qas if int(row["doc_id"][1:]) >= 8]}
    return inventory, candidates, qas, documents, ranks, original


def test_all_selected_family_questions_retained_in_frozen_order_without_gold_selection():
    inventory, candidates, qas, _, _, original = fixture_inventory()
    chosen = extended.validate_inventory(inventory, candidates, qas, original)
    assert len(chosen) == 77 and len({q["family_id"] for q in chosen}) == 24
    assert [{name: q[name] for name in extended.IDENTITY} for q in chosen] == inventory["query_inventory"]
    assert "FORBIDDEN_GOLD" not in json.dumps(chosen)
    for q in qas:
        q.pop("answer_annotations")
    assert chosen == extended.validate_inventory(inventory, candidates, list(reversed(qas)), original)


@pytest.mark.parametrize("change", ["missing_question", "duplicate_question", "wrong_family", "pilot_family", "test", "empty_query"])
def test_inventory_changes_or_unusable_inputs_refuse_without_silent_exclusions(change):
    inventory, candidates, qas, _, _, original = fixture_inventory()
    if change == "missing_question":
        inventory["query_inventory"].pop()
    elif change == "duplicate_question":
        inventory["query_inventory"][-1] = deepcopy(inventory["query_inventory"][0])
    elif change == "wrong_family":
        inventory["documents"][0]["family_id"] = "other"
    elif change == "pilot_family":
        inventory["excluded_pilot_families"][0] = "other"
    elif change == "test":
        qas[-1]["official_split"] = "test"
    else:
        qas[-1]["question"] = "  "
    with pytest.raises(ValueError):
        extended.validate_inventory(inventory, candidates, qas, original)


def test_task_content_hashes_and_candidate_expansion_match_original_pilot():
    candidates, qas, documents, rows = fixture_rows(families=2, questions=2, count=40)
    rankings = pilot.validate_rankings(rows, qas, documents)
    original = pilot.build_prepared(candidates, qas, documents, rankings)
    chosen = [{name: q[name] for name in (*extended.IDENTITY, "query")} for q in original["queries"]]
    actual = extended.build_prepared(chosen, documents, rankings, "f" * 64)
    for key in ("queries", "documents", "static_tasks", "support_tasks"):
        assert actual[key] == original[key]
    assert "FORBIDDEN_GOLD" not in json.dumps(actual)
    assert actual["schema"] != original["schema"]


@pytest.fixture
def source_files(tmp_path, monkeypatch):
    inventory, candidates, qas, documents, ranks, original = fixture_inventory()
    pool, aligned, dense, old = [tmp_path / name for name in ("pool", "aligned", "dense", "pilot-prepared")]
    for path in (pool, aligned, dense, old):
        path.mkdir()
    paths = [pool / "pool_manifest.json", pool / "candidates.jsonl", pool / "native_qa_alignment.json",
             aligned / "native_qa_sidecar_v2.jsonl", aligned / "alignment_audit_v2.json",
             tmp_path / "documents-00000.jsonl.gz", tmp_path / "qasper-train-dev-v0.3.tgz"]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    original_inputs = {str(path): pilot.digest(path) for path in paths}
    summary = {"status": "completed", "test_payload_read": False, "input_sha256": original_inputs,
               "question_count": len(qas), "document_count": len(candidates)}
    (dense / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (dense / "rankings.jsonl").write_text("\n".join(json.dumps(row) for row in ranks), encoding="utf-8")
    for path in (dense / "summary.json", dense / "rankings.jsonl"):
        original_inputs = {**original_inputs, str(path): pilot.digest(path)}
    (old / "prepared.json").write_text(json.dumps(original), encoding="utf-8")
    (old / "manifest.json").write_text(json.dumps({"status": "prepared", "test_payload_read": False,
        "prepared_sha256": pilot.digest(old / "prepared.json"), "input_sha256": original_inputs}), encoding="utf-8")
    inventory["source_sha256"] = {"pilot_prepared": pilot.digest(old / "prepared.json"),
        "development_candidates": pilot.digest(paths[1]), "qa_sidecar": pilot.digest(paths[3])}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(extended, "load_frozen_pool", lambda *_: ({}, candidates, qas))
    monkeypatch.setattr(extended, "load_documents", lambda *_: (documents, paths[-2], paths[-1]))
    return SimpleNamespace(inventory=inventory_path, inventory_sha256=pilot.digest(inventory_path),
                           pilot_prepared=old, output=tmp_path / "prepared")


def test_preparation_round_trip_complete_denominator_and_loader(source_files):
    manifest = extended.run(source_files)
    assert manifest["counts"]["families"] == 24 and manifest["counts"]["queries"] == 77
    assert manifest["counts"]["support_tasks"] == 77 * 16
    assert manifest["all_inventory_queries_retained"] is True
    assert manifest["api_calls"] == 0 and manifest["test_payload_read"] is False
    assert manifest["gold_in_model_payload"] is False and manifest["independent_evaluation"] is False
    prepared, restored, documents = extended.load_prepared(source_files.output)
    assert manifest["prepared_sha256"] == restored["prepared_sha256"]
    assert len(documents) == 24 and len(prepared["queries"]) == 77
    assert all(set(task["item"]) == {"query", "unit"} for task in prepared["support_tasks"])
    assert all(set(task["item"]) == {"unit_a", "unit_b"} for task in prepared["static_tasks"])
    with pytest.raises(FileExistsError):
        extended.run(source_files)


@pytest.mark.parametrize("target", ["inventory_digest", "source", "pilot_prepared", "rankings"])
def test_changed_frozen_inputs_refused_before_output_creation(source_files, target):
    if target == "inventory_digest":
        source_files.inventory_sha256 = "0" * 64
    elif target == "source":
        (source_files.inventory.parent / "aligned" / "native_qa_sidecar_v2.jsonl").write_text("changed", encoding="utf-8")
    elif target == "pilot_prepared":
        (source_files.pilot_prepared / "prepared.json").write_text("{}", encoding="utf-8")
    else:
        (source_files.inventory.parent / "dense" / "rankings.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        extended.run(source_files)
    assert not source_files.output.exists()


def test_source_mutated_during_preparation_refused(source_files, monkeypatch):
    original = extended.build_prepared

    def mutate(*args):
        result = original(*args)
        (source_files.inventory.parent / "dense" / "rankings.jsonl").write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(extended, "build_prepared", mutate)
    with pytest.raises(ValueError, match="source hash mismatch"):
        extended.run(source_files)
    assert not source_files.output.exists()


def test_loader_reconstructs_tasks_even_if_modified_prepared_digest_is_updated(source_files):
    extended.run(source_files)
    path = source_files.output / "prepared.json"
    prepared = extended.read_json(path)
    prepared["support_tasks"][0]["item"]["query"] = "changed visible query"
    path.write_text(json.dumps(prepared), encoding="utf-8")
    manifest_path = source_files.output / "manifest.json"
    manifest = extended.read_json(manifest_path)
    manifest["prepared_sha256"] = pilot.digest(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="payload differs"):
        extended.load_prepared(source_files.output)
