"""Synthetic feature preparation and source-isolation tests; no real datasets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import prepare_cached_routing as builder


def units(count=5):
    return [{"unit_id": f"u{i}", "order": i, "kind": "title" if i == 0 else "heading" if i == 3 else "paragraph",
             "text": f"canonical {i}", "native_text": f"native {i}"} for i in range(count)]


def pack(query, source_units, method, selected):
    return {**{k: query[k] for k in builder.IDENTITY}, "method": method,
            "selected_ids": selected, "pack_sha256": builder.sha(builder.render(source_units, selected).encode()),
            "actual_evidence_tokens": 10 * len(selected), "cache_key": f"synthetic-{method}"}


def fixture():
    source_units = units()
    query = {"family_id": "f", "doc_id": "d", "question_id": "q",
             "candidate_ids": [f"u{i}" for i in range(5)], "ranked_ids": [f"u{i}" for i in range(5)]}
    tasks = [{"id": f"t{i}", "doc_id": "d", "question_id": "q", "unit_id": f"u{i}"} for i in range(5)]
    prepared = {"documents": {"d": source_units}, "queries": [query], "support_tasks": tasks}
    mapping = [pack(query, source_units, "dense_k3", ["u0", "u1"]),
               pack(query, source_units, "reranker_k3", ["u1", "u3", "u4"])]
    pair_scores = [{"task_id": t["id"], "doc_id": t["doc_id"], "question_id": t["question_id"],
                    "unit_id": t["unit_id"], "raw_logit": value} for t, value in zip(tasks, [8., 6., 5., 1., 0.])]
    rankings = [{"doc_id": "d", "question_id": "q", "ranked_ids": list(query["ranked_ids"])}]
    return prepared, mapping, pair_scores, rankings


def test_four_features_use_complete_prejev_candidates_and_original_packs():
    data = fixture()
    row = builder.build_features(*data)[0]
    assert row["features"] == pytest.approx([.5, .75, .75, 1 / 3])
    assert row["support_task_count"] == 5
    assert row["bge_pack"]["selected_ids"] == ["u1", "u3", "u4"]
    assert set(row) == {*builder.IDENTITY, "features", "support_task_count", "bge_pack", "dense_pack"}
    assert "raw_logit" not in json.dumps(row) and "canonical" not in json.dumps(row)


def test_constant_logits_empty_packs_and_no_optional_jev_or_query_input():
    prepared, mapping, scores, rankings = fixture()
    for row in scores:
        row["raw_logit"] = .123
    query = prepared["queries"][0]
    mapping = [pack(query, prepared["documents"]["d"], method, []) for method in ("dense_k3", "reranker_k3")]
    # There are no query strings, references, JEV labels or scores in this fixture.
    assert builder.build_features(prepared, mapping, scores, rankings)[0]["features"] == [0., 0., 0., 0.]


@pytest.mark.parametrize("mutation", ["missing_score", "duplicate_score", "wrong_task", "nan", "wrong_rank", "missing_task"])
def test_incomplete_or_changed_candidate_contract_fails(mutation):
    prepared, mapping, scores, rankings = fixture()
    if mutation == "missing_score": scores.pop()
    elif mutation == "duplicate_score": scores.append(deepcopy(scores[0]))
    elif mutation == "wrong_task": scores[0]["task_id"] = "wrong"
    elif mutation == "nan": scores[0]["raw_logit"] = float("nan")
    elif mutation == "wrong_rank": rankings[0]["ranked_ids"].reverse()
    else: prepared["support_tasks"].pop()
    with pytest.raises(ValueError):
        builder.build_features(prepared, mapping, scores, rankings)


@pytest.mark.parametrize("mutation", ["hash", "order", "token", "missing_pack"])
def test_original_pack_tampering_fails(mutation):
    prepared, mapping, scores, rankings = fixture()
    if mutation == "hash": mapping[0]["pack_sha256"] = "0" * 64
    elif mutation == "order": mapping[0]["selected_ids"].reverse()
    elif mutation == "token": mapping[0]["actual_evidence_tokens"] = 1025
    else: mapping.pop()
    with pytest.raises(ValueError):
        builder.build_features(prepared, mapping, scores, rankings)


def put(path, obj, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in obj) if lines else json.dumps(obj)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def full_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ARTIFACTS", tmp_path)
    documents = {f"d{i}": units(16) for i in range(24)}
    queries, tasks, mappings, scores, rankings = [], [], [], [], []
    for i in range(77):
        doc = f"d{i % 24}"
        ids = [f"u{j}" for j in range(15 if i < 18 else 16)]
        query = {"family_id": f"f{i % 24}", "doc_id": doc, "question_id": f"q{i}",
                 "candidate_ids": ids, "ranked_ids": ids}
        queries.append(query)
        for uid in ids:
            tid = f"task-{i}-{uid}"
            tasks.append({"id": tid, "doc_id": doc, "question_id": f"q{i}", "unit_id": uid})
            scores.append({"task_id": tid, "doc_id": doc, "question_id": f"q{i}", "unit_id": uid,
                           "raw_logit": -float(int(uid[1:]))})
        rankings.append({"doc_id": doc, "question_id": f"q{i}", "ranked_ids": ids})
        mappings += [pack(query, documents[doc], "dense_k3", ["u0", "u1", "u2"]),
                     pack(query, documents[doc], "reranker_k3", ["u1", "u2", "u3"])]
    prepared = {"documents": documents, "queries": queries, "support_tasks": tasks}
    source_data = {"prepared": prepared, "mapping": mappings, "scores": scores, "rankings": rankings}
    feature_names = {"prepared": "prepared", "mapping": "dense_and_bge_packs", "scores": "pair_scores.jsonl", "rankings": "rankings.jsonl"}
    specs = {}
    for role, data in source_data.items():
        rel = builder.SOURCES[role]
        specs[feature_names[role]] = {"path": rel, "sha256": put(tmp_path / rel, data, lines=role != "prepared")}
    # Deliberately absent files: successful preparation proves they were not read.
    arms = {"bge": {"quality_records_path": "FORBIDDEN-bge-quality.jsonl", "quality_records_sha256": "a" * 64},
            "jev": {"quality_records_path": "FORBIDDEN-jev-quality.jsonl", "quality_records_sha256": "b" * 64}}
    contract_hash = put(tmp_path / builder.CONTRACT, {"feature_sources": specs, "arms": arms})
    monkeypatch.setattr(builder, "CONTRACT_SHA", contract_hash)
    return tmp_path


def test_complete_feature_only_preparation_seals_inputs_folds_and_defers_targets(full_fixture):
    output = full_fixture / "prepared"
    summary = builder.prepare(output)
    plan = json.loads((output / "plan.json").read_bytes())
    features = json.loads((output / "features.json").read_bytes())
    folds = json.loads((output / "folds.json").read_bytes())
    assert len(features) == 77 and len(folds) == 5
    assert sum(summary["fold_sizes"]) == 77 and summary["support_task_count"] == 1214
    assert summary["quality_targets_read"] is False and summary["jev_judgments_or_packs_read"] is False
    assert summary["api_calls"] == 0
    assert plan["artifact_sha256"] == {name: builder.sha((output / name).read_bytes()) for name in ("features.json", "folds.json")}
    assert summary["plan_sha256"] == builder.sha((output / "plan.json").read_bytes())
    assert len(plan["input_sha256"]) == 5  # one metadata contract and four feature inputs
    assert "FORBIDDEN" not in json.dumps(summary)
    assert len({tuple(key) for fold in folds for key in fold["test_keys"]}) == 77
    assert all(not set(fold["test_families"]) & set(fold["train_families"]) for fold in folds)


def test_changed_source_binding_stops_before_feature_output(full_fixture):
    source = full_fixture / builder.SOURCES["scores"]
    source.write_bytes(source.read_bytes() + b" ")
    output = full_fixture / "prepared"
    with pytest.raises(ValueError, match="source binding"):
        builder.prepare(output)
    assert not (output / "features.json").exists() and not (output / "summary.json").exists()


def test_saved_sample_is_deterministic_and_does_not_use_feature_values():
    keys = [(f"f{i}", f"d{i}", f"q{j}") for i in range(10) for j in range(3)]
    assert builder.sample_keys(keys) == builder.sample_keys(list(reversed(keys)))
    assert len(builder.sample_keys(keys)) == 16
