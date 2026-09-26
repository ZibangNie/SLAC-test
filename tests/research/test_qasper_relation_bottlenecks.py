from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "research"))
from analyze_qasper_relation_bottlenecks import availability_ceiling, best_option, exhaustive_options, rescue_option
from run_qasper_evidence_baselines import Unit


def units(*texts):
    return [Unit(str(i), i, "text", 0, len(text), text, text) for i, text in enumerate(texts)]


def gold(*refs):
    return [{"unanswerable": False, "extractive_spans": [], "free_form_answer": "answer",
             "yes_no": None, "evidence": list(ref)} for ref in refs]


def test_cardinality_restriction_is_distinct_from_ordering():
    opts = exhaustive_options(units("gold", "filler", "other"), [0, 1, 2], gold(["gold"]), lambda xs: 10 * len(xs))
    assert best_option(opts, [0, 1, 2], budget=1024)["f1"] == 1
    assert best_option(opts, [0, 1, 2], budget=1024, exact_size=3)["f1"] == .5
    assert best_option(opts, [1, 2], budget=1024)["f1"] == 0


def test_all_duplicate_locations_enumerated_without_selecting_duplicate_text():
    opts = exhaustive_options(units("gold", "gold"), [0, 1], gold(["gold"]),
                              lambda xs: 1500 if 0 in xs else 20 * len(xs))
    assert len(opts) == 3
    assert best_option(opts, [0, 1], budget=1024)["indices"] == (1,)


def test_multiple_references_do_not_turn_into_a_union():
    us = units("a", "b", "c")
    refs = gold(["a", "b"], ["c"])
    opts = exhaustive_options(us, [0, 1, 2], refs, lambda xs: len(xs))
    assert best_option(opts, [0, 1, 2])["indices"] == (2,)
    assert availability_ceiling(us, [0, 2], refs) == 1


def test_empty_reference_and_empty_eligible_pool():
    refs = [{"unanswerable": True}]
    opts = exhaustive_options(units("unneeded"), [0], refs, lambda xs: len(xs))
    assert best_option(opts, [], budget=1024)["f1"] == 1
    assert availability_ceiling(units("unneeded"), [0], refs) == 1


def test_duplicate_reference_denominators_retained():
    assert availability_ceiling(units("a"), [0], gold(["a", "a"])) == pytest.approx(2 / 3)


def test_nonmonotone_token_counter_does_not_drop_nongold_options():
    opts = exhaustive_options(units("gold", "filler"), [0, 1], gold(["gold"]),
                              lambda xs: 1025 if xs == (0,) else len(xs))
    selected = best_option(opts, [0, 1], budget=1024)
    assert selected["indices"] == (0, 1)
    assert selected["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("pool", [[], [0, 0], [-1], [1]])
def test_invalid_candidate_pools_rejected(pool):
    with pytest.raises(ValueError):
        exhaustive_options(units("a"), pool, gold(["a"]), lambda xs: len(xs))


def test_exhaustive_cap_and_impossible_cardinality_rejected():
    with pytest.raises(ValueError):
        exhaustive_options(units(*map(str, range(17))), list(range(17)), gold(["0"]), lambda xs: len(xs))
    opts = exhaustive_options(units("a"), [0], gold(["a"]), lambda xs: len(xs))
    with pytest.raises(ValueError):
        best_option(opts, [], exact_size=1)


def test_rescue_requires_selected_eligible_anchor_and_respects_direction():
    opts = exhaustive_options(units("gold", "anchor", "other"), [0, 1, 2], gold(["gold"]), lambda xs: len(xs))
    assert rescue_option(opts, [1], {(0, 1)})["indices"] == (0, 1)
    assert rescue_option(opts, [1], {(0, 1)})["f1"] == pytest.approx(2 / 3)
    assert rescue_option(opts, [1], {(1, 0)})["f1"] == 0
    assert rescue_option(opts, [2], {(0, 1), (1, 2)})["f1"] == 0
    assert rescue_option(opts, [1], {(0, 1)}, budget=1)["f1"] == 0
