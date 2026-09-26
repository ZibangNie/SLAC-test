"""Offline checks for corpus scope, source identity, parity and full denominators."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
import run_qasper_corpus_bridge as bridge
from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack
from qasper_metrics import evidence_metrics


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        return list(range(len(text) + 2))


def native(count, prefix="text"):
    return [Unit(f"u{i}", i, "paragraph", i * 10, i * 10 + 5, f"{prefix}{i}", f"{prefix}{i}")
            for i in range(count)]


def annotation(evidence, *, unanswerable=False):
    return {"native_answer": {"unanswerable": unanswerable, "extractive_spans": [],
        "free_form_answer": "answer" if not unanswerable else "", "yes_no": None, "evidence": evidence}}


def test_global_map_retains_local_ids_and_byte_identical_within_doc_render():
    documents = {"b": native(3), "a": native(2, "other")}
    units, keys, positions = bridge.globalize_documents(documents)
    assert keys == [("a", "u0"), ("a", "u1"), ("b", "u0"), ("b", "u1"), ("b", "u2")]
    assert len({unit.unit_id for unit in units}) < len(units)
    assert [unit.order for unit in units] == list(range(5))
    assert render_pack(units, [2, 4]) == render_pack(documents["b"], [0, 2])
    counter = PackCounter(CharacterTokenizer(), units)
    assert counter([2, 4]) == PackCounter(CharacterTokenizer(), documents["b"])([0, 2])


def test_adjacency_never_crosses_document_boundary_and_matches_pilot():
    documents = {"a": native(19), "b": native(6, "b")}
    units, keys, positions = bridge.globalize_documents(documents)
    ranking = [18, 0, 2, 4, 6, 8, 10, 12] + [i for i in range(19) if i not in {18, 0, 2, 4, 6, 8, 10, 12}]
    seeds, candidates = bridge.expand_corpus_candidates(ranking, keys, positions)
    old_seeds, old_candidates = bridge.pilot.expand_candidates(documents["a"], [keys[i][1] for i in ranking])
    assert [keys[i][1] for i in seeds] == old_seeds
    assert [keys[i][1] for i in candidates] == old_candidates
    assert len(candidates) == 16 and all(keys[i][0] == "a" for i in candidates)
    assert set(seeds) <= set(candidates)
    _, one_seed = bridge.expand_corpus_candidates([19], keys, positions, seed_count=1, cap=3)
    assert one_seed == [19, 20]  # no accidental previous-document neighbor 18


def test_rank_ties_use_global_source_order_not_target_document():
    assert bridge.rank_scores([.4, .9, .9, .1]) == [1, 2, 0, 3]
    assert bridge.rank_scores([.4, .9, .9, .1], [3, 2, 0]) == [2, 0, 3]
    with pytest.raises(ValueError, match="finite"):
        bridge.rank_scores([float("nan")])


def test_packing_counts_full_headers_separator_specials_and_exact_duplicates():
    documents = {"a": native(4)}
    documents["a"][0] = replace(documents["a"][0], text="x" * 100, native_text="x" * 100)
    documents["a"][2] = replace(documents["a"][2], text="text1", native_text="text1")
    units, _, _ = bridge.globalize_documents(documents)
    counter = PackCounter(CharacterTokenizer(), units)
    assert pack_ranked(units, [0, 1, 2, 3], 28, counter, max_units=3) == [1, 3]
    assert counter([1, 3]) == len(render_pack(units, [1, 3])) + 2
    assert counter([]) == 0
    assert pack_ranked(units, [0, 1, 2, 3], 1, counter, max_units=3) == []


def test_optional_qualified_dedup_keeps_different_sources_without_changing_main_policy():
    units, keys, _ = bridge.globalize_documents({"a": native(1), "b": native(1)})
    count = PackCounter(CharacterTokenizer(), units)
    assert pack_ranked(units, [1, 0], 1024, count, max_units=3) == [1]
    assert bridge.pack_source_qualified(units, keys, [1, 0], 1024, count) == [0, 1]
    for invalid in ([-1], [2], [0, 0], [True]):
        with pytest.raises(ValueError, match="ranking"):
            bridge.pack_source_qualified(units, keys, invalid, 1024, count)


def test_wrong_source_same_text_is_false_positive_and_keeps_precision_denominator():
    gold = [annotation(["Introduction"])]
    assert evidence_metrics(["Introduction"], gold)["evidence_f1"] == 1
    assert bridge.source_qualified_metrics([("wrong", "Introduction")], "right", gold) == {
        "evidence_f1": 0, "evidence_recall": 0}
    result = bridge.source_qualified_metrics([("right", "Introduction"), ("wrong", "other")], "right", gold)
    assert result["evidence_f1"] == pytest.approx(2 / 3)
    assert result["evidence_recall"] == 1
    collision = bridge.source_qualified_metrics([("a", 'b","c')], 'a","b', [annotation(["c"])])
    assert collision["evidence_f1"] == 0


def test_qualified_metrics_keep_empty_float_duplicate_and_multi_reference_semantics():
    gold = [annotation(["a", "a", "FLOAT SELECTED: fig 1"])]
    result = bridge.source_qualified_metrics([("doc", "a")], "doc", gold)
    assert result["evidence_f1"] == pytest.approx(.5)
    assert result["evidence_recall"] == pytest.approx(1 / 3)
    empty = [annotation([], unanswerable=True)]
    assert bridge.source_qualified_metrics([], "doc", empty) == {"evidence_f1": 1, "evidence_recall": 1}
    assert bridge.source_qualified_metrics([("wrong", "x")], "doc", empty) == {"evidence_f1": 0, "evidence_recall": 0}
    several = gold + [annotation(["a"]), annotation([], unanswerable=True)]
    assert bridge.source_qualified_metrics([("doc", "a")], "doc", several)["evidence_f1"] == 1


def test_vector_contract_rejects_dtype_nonfinite_norm_and_metadata():
    metadata = {"pooling": "CLS_then_FP32_L2", "model_revision": bridge.MODEL_REVISION}
    vectors = torch.eye(2)
    bridge.validate_vectors(vectors, vectors, metadata, candidate_count=2, query_count=2, dimension=2)
    for bad in (vectors.double(), vectors * 2, vectors * float("nan")):
        with pytest.raises(ValueError):
            bridge.validate_vectors(bad, vectors, metadata, candidate_count=2, query_count=2, dimension=2)
    with pytest.raises(ValueError, match="metadata"):
        bridge.validate_vectors(vectors, vectors, {**metadata, "pooling": "mean"}, candidate_count=2, query_count=2, dimension=2)


def test_embedding_index_is_reconstructed_not_trusted():
    keys = [("d", "u0"), ("d", "u1")]
    qa = [{"doc_id": "d", "question_id": "q"}]
    index = {"candidates": [{"doc_id": d, "unit_id": u} for d, u in keys], "queries": qa}
    bridge.validate_index(index, keys, qa)
    index["candidates"].reverse()
    with pytest.raises(ValueError, match="index"):
        bridge.validate_index(index, keys, qa)


def test_within_doc_replay_detects_candidate_changes_and_gold_cannot_change_selection():
    documents = {"a": native(10), "b": native(3, "other")}
    units, keys, positions = bridge.globalize_documents(documents)
    ids = [unit.unit_id for unit in documents["a"]]
    seeds, candidates = bridge.pilot.expand_candidates(documents["a"], ids)
    q = {"family_id": "f", "doc_id": "a", "question_id": "q", "query": "question",
         "seed_ids": seeds, "candidate_ids": candidates, "ranked_ids": [uid for uid in ids if uid in candidates]}
    prepared = {"queries": [q]}
    scores = list(reversed(range(len(units))))
    replay = bridge.validate_within_document_replay(prepared, documents, units, keys, positions,
        {("a", "q"): scores}, {("a", "q"): ids}, CharacterTokenizer())
    source = {"units": units, "keys": keys, "positions": positions, "documents": documents,
              "tokenizer": CharacterTokenizer(), "qa_by_key": {("a", "q"): {"answer_annotations": [annotation(["text1"])]}}}
    first = bridge.evaluate_query(q, source, scores, replay[("a", "q")])
    source["qa_by_key"][("a", "q")]["answer_annotations"] = [annotation(["unreachable gold"])]
    second = bridge.evaluate_query(q, source, scores, replay[("a", "q")])
    assert [row["selected_global_indices"] for row in first] == [row["selected_global_indices"] for row in second]
    assert first[0]["official_evidence_f1"] != second[0]["official_evidence_f1"]
    bad = deepcopy(prepared)
    bad["queries"][0]["candidate_ids"].pop()
    with pytest.raises(ValueError, match="candidate replay"):
        bridge.validate_within_document_replay(bad, documents, units, keys, positions,
            {("a", "q"): scores}, {("a", "q"): ids}, CharacterTokenizer())
    with pytest.raises(ValueError, match="every frozen query"):
        bridge.summarize(first[:1], prepared)
    assert bridge.summarize(first, prepared)["metrics"][0]["questions"] == 1


def test_existing_output_refused_before_reading_any_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "load_verified_inputs", lambda *_args, **_kwargs: pytest.fail("must not read inputs"))
    with pytest.raises(FileExistsError):
        bridge.run(SimpleNamespace(output=tmp_path, max_seconds=300))


REAL_ROOT = Path(__file__).resolve().parents[2]
REAL_DENSE = REAL_ROOT / "artifacts/research-foundation/qasper-dense-01"
REAL_PREPARED = REAL_ROOT / "artifacts/research-foundation/qasper-extended-development-prepared-01"


@pytest.mark.skipif(not (REAL_DENSE / "embeddings.safetensors").exists() or not REAL_PREPARED.exists(),
                    reason="ignored frozen local validation artifacts are not distributed")
def test_real_frozen_cache_all_77_within_document_rank_candidate_render_score_parity():
    torch.set_num_threads(1)
    source = bridge.load_verified_inputs(REAL_PREPARED, REAL_DENSE)
    scores = {}
    for q in source["prepared"]["queries"]:
        key = (q["doc_id"], q["question_id"])
        scores[key] = (source["candidate_vectors"] @ source["query_vectors"][source["query_positions"][key]]).tolist()
    replay = bridge.validate_within_document_replay(source["prepared"], source["documents"], source["units"],
        source["keys"], source["positions"], scores, source["rankings"], source["tokenizer"])
    assert len(replay) == 77
    records = []
    for q in source["prepared"]["queries"]:
        key = (q["doc_id"], q["question_id"])
        records.extend(bridge.evaluate_query(q, source, scores[key], replay[key]))
    result = bridge.summarize(records, source["prepared"])
    assert len(records) == 154 and all(row["actual_evidence_tokens"] <= 1024 for row in records)
    assert all(row["questions"] == 77 and row["families"] == 24 for row in result["metrics"])
