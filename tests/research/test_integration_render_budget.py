"""Synthetic full-render budget checks, with no providers, tokenizers or models."""
from copy import deepcopy

import pytest

from SLAC.integration.evidence.budgeter import fit_evidence_to_budget
from SLAC.integration.evidence.selectors import select_evidence
from SLAC.integration.io.schemas import SelectedEvidence
from SLAC.llm.io.schemas import EvidenceItem
from SLAC.llm.service.renderers import render_evidence_block


def ev(name, *, rank=1, role="direct", token_est=1, doc="doc", text=None):
    return SelectedEvidence(name, doc, text or name, rerank_rank=rank, role=role, token_est=token_est)


def ids(items):
    return tuple(item.chunk_id for item in items)


def llm_toy_bytes(items):
    assert isinstance(items, tuple)
    evidence = [EvidenceItem(chunk_id=e.chunk_id, doc_id=e.doc_id, passage_text=e.passage_text,
                             rerank_rank=e.rerank_rank, role=e.role) for e in items]
    return len(render_evidence_block(evidence).encode("utf-8"))


def test_none_callback_keeps_legacy_additive_and_empty_shortcut_behavior():
    a, b = ev("a", token_est=2), ev("b", token_est=1, rank=2)
    assert fit_evidence_to_budget([a, b], max_items=2, max_tokens=2) == [a]
    assert fit_evidence_to_budget([a, b], max_items=2, max_tokens=2, pack_cost=None) == [a]
    assert select_evidence([a, b], max_items=2, max_tokens=2) == [a]
    assert select_evidence([a, b], max_items=2, max_tokens=2, pack_cost=None) == [a]
    assert select_evidence([], max_items=-1, max_tokens=-1) == []
    # Existing fit behavior accepts bool caps and preselected duplicate counts.
    assert fit_evidence_to_budget([a], max_items=True, max_tokens=2) == [a]
    assert fit_evidence_to_budget([], max_items=1, max_tokens=0, already_selected=[a, a]) == []


def test_actual_llm_render_toy_bytes_override_low_estimates_and_match_exact_boundary():
    items = [ev("a", text="A" * 20), ev("b", rank=2, text="B" * 20)]
    before = deepcopy(items)
    limit = llm_toy_bytes((items[0],))
    assert llm_toy_bytes(tuple(items)) > limit
    assert select_evidence(items, max_items=2, max_tokens=limit) == items
    selected = select_evidence(items, max_items=2, max_tokens=limit, pack_cost=llm_toy_bytes)
    assert selected == [items[0]] and llm_toy_bytes(tuple(selected)) == limit
    assert select_evidence(items, max_items=2, max_tokens=limit - 1, pack_cost=llm_toy_bytes) == []
    assert items == before


@pytest.mark.parametrize("tied", [False, True])
def test_direct_first_cost_uses_final_global_stable_order_including_exact_ties(tied):
    non_direct = ev("n", rank=1, role="expanded")
    direct = ev("d", rank=1 if tied else 2)
    seen = []
    def cost(pack):
        assert isinstance(pack, tuple)
        order = ids(pack)
        seen.append(order)
        return 100 if order == ("d", "n") else len(pack)
    result = select_evidence([non_direct, direct], max_items=2, max_tokens=2, pack_cost=cost)
    assert result == [non_direct, direct]
    assert ("d", "n") not in seen and ("n", "d") in seen
    assert seen[-1] == ids(result)


def test_seed_cost_and_item_limit_include_existing_pack_but_return_only_additions():
    a, b, c = ev("a", rank=2), ev("b", rank=1), ev("c", rank=3)
    seen = []
    def cost(pack):
        seen.append(ids(pack))
        return len(pack) * 3
    result = fit_evidence_to_budget([a, b, c], max_items=2, max_tokens=6,
        already_selected=iter([a]), pack_cost=cost)
    assert result == [b] and ("b", "a") in seen
    assert ("b", "a", "c") not in seen


@pytest.mark.parametrize("seed,cap_items,cap_cost,message", [
    ("duplicate", 2, 4, "duplicate"), ("count", 0, 4, "max_items"),
    ("cost", 1, 0, "already_selected exceeds max_tokens"),
])
def test_invalid_seed_is_rejected(seed, cap_items, cap_cost, message):
    a = ev("a")
    seeds = [a, a] if seed == "duplicate" else [a]
    with pytest.raises(ValueError, match=message):
        fit_evidence_to_budget([], max_items=cap_items, max_tokens=cap_cost,
                               already_selected=seeds, pack_cost=lambda pack: len(pack))


def test_identity_includes_document_and_accepted_duplicate_is_not_charged_twice():
    a, same, other_doc = ev("a"), ev("a"), ev("a", doc="other")
    result = fit_evidence_to_budget([same, other_doc], max_items=2, max_tokens=2,
                                    already_selected=[a], pack_cost=lambda pack: len(pack))
    assert result == [other_doc]


@pytest.mark.parametrize("function", [fit_evidence_to_budget, select_evidence])
@pytest.mark.parametrize("bad", [True, 1.5, -1])
def test_invalid_callback_cost_rejected_even_with_empty_candidates(function, bad):
    with pytest.raises((ValueError, TypeError), match="pack_cost result"):
        function([], max_items=0, max_tokens=0, pack_cost=lambda pack: bad)


@pytest.mark.parametrize("function", [fit_evidence_to_budget, select_evidence])
@pytest.mark.parametrize("field,value", [("max_items", True), ("max_tokens", 1.5), ("max_tokens", -1)])
def test_exact_limits_are_strict_even_on_empty_input(function, field, value):
    kwargs = {"max_items": 1, "max_tokens": 1, field: value}
    with pytest.raises((ValueError, TypeError), match=field):
        function([], **kwargs, pack_cost=lambda pack: 0)


def test_empty_cost_and_noncallable_are_not_silently_ignored():
    for function in (fit_evidence_to_budget, select_evidence):
        assert function([], max_items=0, max_tokens=3, pack_cost=lambda pack: 3) == []
        with pytest.raises(ValueError, match="empty evidence pack"):
            function([], max_items=0, max_tokens=2, pack_cost=lambda pack: 3)
        with pytest.raises(TypeError, match="callable"):
            function([], max_items=0, max_tokens=0, pack_cost=False)
    with pytest.raises(TypeError, match="min_direct_evidence"):
        select_evidence([], max_items=0, max_tokens=0, min_direct_evidence=True, pack_cost=lambda p: 0)


def test_nonadditive_decreasing_cost_is_accepted_without_per_item_estimate_pruning():
    a, b = ev("a", token_est=999), ev("b", rank=2, token_est=999)
    costs = {(): 0, ("a",): 8, ("a", "b"): 3}
    assert select_evidence([a, b], max_items=2, max_tokens=8, prefer_direct_first=False,
                           pack_cost=lambda pack: costs[ids(pack)]) == [a, b]


def test_greedy_does_not_revisit_rejected_candidate_after_later_admission():
    a, b = ev("a"), ev("b", rank=2)
    calls = []
    def cost(pack):
        calls.append(ids(pack))
        return {(): 0, ("a",): 11, ("b",): 1, ("a", "b"): 2}[ids(pack)]
    result = select_evidence([a, b], max_items=2, max_tokens=10, prefer_direct_first=False, pack_cost=cost)
    assert result == [b] and ("a", "b") not in calls


def test_rejected_direct_retains_existing_second_phase_retry_with_nonmonotone_cost():
    n, d = ev("n", rank=1, role="expanded"), ev("d", rank=2)
    calls = []
    def cost(pack):
        calls.append(ids(pack))
        return {(): 0, ("d",): 11, ("n",): 1, ("n", "d"): 2}[ids(pack)]
    assert select_evidence([n, d], max_items=2, max_tokens=10, pack_cost=cost) == [n, d]
    assert calls.count(("d",)) == 1 and ("n", "d") in calls


@pytest.mark.parametrize("function", [fit_evidence_to_budget, select_evidence])
def test_final_recheck_rejects_nondeterministic_overbudget_cost(function):
    nonempty_calls = 0
    def cost(pack):
        nonlocal nonempty_calls
        if not pack:
            return 0
        nonempty_calls += 1
        return 1 if nonempty_calls == 1 else 100
    with pytest.raises(ValueError, match="final evidence pack"):
        function([ev("a")], max_items=1, max_tokens=1, pack_cost=cost)


def test_callback_exception_propagates_without_legacy_fallback():
    def cost(pack):
        raise RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        select_evidence([ev("a")], max_items=1, max_tokens=1, pack_cost=cost)
