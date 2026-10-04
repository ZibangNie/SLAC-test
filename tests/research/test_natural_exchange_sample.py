"""Four synthetic preparation checks; no natural input or tokenizer file reads."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import prepare_natural_exchange_sample as builder


def units():
    return {uid: {"unit_id": uid, "order": index, "text": f"retrieval {uid}", "native_text": f"native {uid}"}
            for index, uid in enumerate("abcde")}


def test_saved_rank_alone_determines_candidate_and_worst_selected_removal():
    source = units()
    ranking = ["c", "a", "d", "b", "e"]
    selected = ["b", "c", "a"]
    result = builder.propose(source, ranking, selected, lambda ids: len(ids) * 10)
    assert result["candidate_id"] == "d" and result["removed_id"] == "b"
    assert result["original_ids"] == ["a", "b", "c"]
    assert result["proposed_ids"] == ["a", "c", "d"]
    # Reversing map and selected iteration cannot introduce a new score tie-break.
    assert builder.propose(dict(reversed(list(source.items()))), ranking, list(reversed(selected)),
                           lambda ids: len(ids) * 10) == result


def test_duplicate_native_candidate_is_skipped_and_no_candidate_retained():
    source = units()
    source["d"]["native_text"] = source["a"]["native_text"]
    result = builder.propose(source, list("abcde"), list("abc"), lambda ids: 30)
    assert result["candidate_id"] == "e"  # Retrieval strings differ; native identity controls.
    source["e"]["native_text"] = source["b"]["native_text"]
    absent = builder.propose(source, list("abcde"), list("abc"), lambda ids: 30)
    assert absent["status"] == "no_candidate"
    assert absent["candidate_id"] is None and absent["proposed_ids"] is None
    assert absent["removed_id"] == "c" and absent["proposed_tokens"] is None


def test_budget_failure_never_changes_candidate_or_removed_unit():
    calls = []
    def count(ids):
        calls.append(tuple(ids))
        return 1025 if "d" in ids else 100
    result = builder.propose(units(), list("abcde"), list("abc"), count)
    assert result["status"] == "infeasible"
    assert result["candidate_id"] == "d" and result["removed_id"] == "c"
    assert result["proposed_ids"] == ["a", "b", "d"]
    assert calls == [("a", "b", "c"), ("a", "b", "d")]
    assert "e" not in {uid for call in calls for uid in call}  # A fitting alternate is never tried.


def test_bounded_projection_uses_whole_nonadditive_render_and_checks_baseline():
    class Tokenizer:
        def __init__(self):
            self.calls = []
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens is True and truncation is False
            self.calls.append(text)
            return [0] * (len(text) + 11)  # Nonadditive per-pack overhead.
    documents, queries, rankings, records = {}, [], [], []
    for d in range(3):
        doc = f"doc{d}"
        source = units()
        documents[doc] = list(source.values())
        for q in range(2):
            query = {"doc_id": doc, "question_id": f"q{q}", "query": "Synthetic query",
                     "candidate_ids": list("abcde"), "ranked_ids": list("abcde")}
            queries.append(query)
            rankings.append({"doc_id": doc, "question_id": f"q{q}", "ranked_ids": list("abcde")})
            text = builder.render(source, list("abc"))
            records.append(query | {"methods": {"local_bge_reranker": {
                "ranked_ids": list("abcde"), "selected_ids": list("abc"),
                "evidence_tokens": len(text) + 11, "pack_sha256": builder.sha(text.encode()),
                "gaps": object()}}})  # Unread relation-like data is deliberately unusable.
    data, scores, opportunities = {"documents": documents, "queries": queries}, {"rankings": rankings}, {"records": records}
    tokenizer = Tokenizer()
    proposals, packet = builder.project_sample(data, scores, opportunities, tokenizer)
    assert len(proposals) == len(packet) == 6 and len(tokenizer.calls) == 12
    source = units()
    assert proposals[0]["original_tokens"] == len(builder.render(source, list("abc"))) + 11
    assert proposals[0]["original_tokens"] != sum(len(builder.render(source, [uid])) + 11 for uid in "abc")
    assert tokenizer.calls[0] == "[a]\nretrieval a\n\n[b]\nretrieval b\n\n[c]\nretrieval c"
    assert set(packet[0]) == {"ordinal", "doc_id", "question_id", "query", "original_pack", "candidate", "removed", "proposed_pack"}
    assert all("rank" not in key and "score" not in key and "status" not in key for row in packet for key in row)
    for mutation in ("tokens", "rank", "duplicate"):
        changed = deepcopy(opportunities)
        baseline = changed["records"][0]["methods"]["local_bge_reranker"]
        if mutation == "tokens":
            baseline["evidence_tokens"] += 1
        elif mutation == "rank":
            baseline["ranked_ids"].reverse()
        else:
            baseline["selected_ids"] = ["a", "a", "b"]
        with pytest.raises(ValueError):
            builder.project_sample(data, scores, changed, Tokenizer())
