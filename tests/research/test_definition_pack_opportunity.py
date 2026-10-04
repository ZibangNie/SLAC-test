"""Test budget and denominator mistakes that would misstate opportunities."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("definition_opportunity", Path(__file__).parents[2] / "docs/research/analyze_definition_pack_opportunity.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def units():
    return {uid: {"unit_id": uid, "text": text, "native_text": text, "order": order}
            for uid, text, order in [("d", "natural language processing (NLP)", 0),
                                      ("a", "unrelated", 1), ("m", "NLP works", 3),
                                      ("b", "elsewhere", 4)]}


def test_append_token_fit_does_not_imply_k3_fit_and_mentions_are_retained():
    u = units()
    row, = M.gaps(u, ["m", "a", "b"], [("m", "d", "NLP")], list(u), ["m"], "NLP?", lambda s: len(s)*10)
    assert row["append_fits_tokens"] and not row["append_fits_k_and_tokens"]
    assert {r["removed_id"] for r in row["single_replacements"]} == {"a", "b"}
    assert row["adjacent_to_any_selected_native_unit"]  # via a, not m
    assert row["exact_acronym_in_query"]


def test_target_grouping_keeps_all_triggering_mentions():
    u = units()
    row, = M.gaps(u, ["m", "a", "b"], [("m", "d", "NLP"), ("a", "d", "NLP"), ("b", "d", "NLP")], [], [], "NLPx", lambda s: 100)
    assert len(row["dependent_ids"]) == 3
    assert row["single_replacements"] == []
    assert not row["exact_acronym_in_query"]


def test_exact_native_text_at_another_position_counts_as_covered():
    u = units(); u["a"]["native_text"] = u["d"]["native_text"]
    assert M.gaps(u, ["m", "a"], [("m", "d", "NLP")], [], [], "", lambda s: 100) == []


def test_each_whole_replacement_is_counted_without_additive_token_assumption():
    u = units()
    def count(s):
        return 2000 if set(s) == {"m", "b", "d"} else 500
    row, = M.gaps(u, ["m", "a", "b"], [("m", "d", "NLP")], [], [], "", count)
    assert [r["removed_id"] for r in row["single_replacements"]] == ["b"]


def test_ranking_packs_complete_units_with_native_dedup_and_whole_counter():
    u = units(); u["a"]["native_text"] = u["m"]["native_text"]
    def count(s):
        return 1025 if "d" in s else len(s)*100
    assert M.choose(u, ["m", "a", "d", "b"], count) == ["m", "b"]
    assert M.render(u, ["b", "m"]) == "[m]\nNLP works\n\n[b]\nelsewhere"


def test_repeated_mentions_do_not_inflate_target_denominator():
    u = units()
    rows = M.gaps(u, ["m"], [("m", "d", "NLP")]*5, [], [], "", lambda s: 1)
    assert len(rows) == 1 and rows[0]["dependent_ids"] == ["m"]
