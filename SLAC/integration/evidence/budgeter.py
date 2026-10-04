from __future__ import annotations

from typing import Callable, Iterable, List, Set, Tuple

from SLAC.integration.io.schemas import SelectedEvidence


def evidence_order_key(ev: SelectedEvidence) -> Tuple[int, int, int]:
    source_ordinal = int(ev.meta.get("_source_ordinal", 10**9))

    if ev.rerank_rank is not None:
        return (0, ev.rerank_rank, source_ordinal)

    if ev.retrieve_rank_fused is not None:
        return (1, ev.retrieve_rank_fused, source_ordinal)

    return (2, source_ordinal, source_ordinal)


def stable_sort_evidence(items: Iterable[SelectedEvidence]) -> List[SelectedEvidence]:
    return sorted(items, key=evidence_order_key)


def _nonnegative_integer(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _checked_pack_cost(items, pack_cost):
    return _nonnegative_integer(pack_cost(tuple(items)), "pack_cost result")


def _fit_with_pack_cost(items, *, max_items, max_tokens, already_selected, pack_cost):
    _nonnegative_integer(max_items, "max_items")
    _nonnegative_integer(max_tokens, "max_tokens")
    if not callable(pack_cost):
        raise TypeError("pack_cost must be callable")
    seeds = list(already_selected) if already_selected is not None else []
    seen = {(ev.doc_id, ev.chunk_id) for ev in seeds}
    if len(seen) != len(seeds):
        raise ValueError("already_selected contains duplicate evidence identities")
    if len(seeds) > max_items:
        raise ValueError("already_selected exceeds max_items")

    def cost(pack):
        return _checked_pack_cost(stable_sort_evidence(pack), pack_cost)

    # These are input preconditions, not a claim that no other combination fits.
    if cost(()) > max_tokens:
        raise ValueError("empty evidence pack exceeds max_tokens")
    if seeds and cost(seeds) > max_tokens:
        raise ValueError("already_selected exceeds max_tokens")
    selected = []
    for ev in items:
        key = (ev.doc_id, ev.chunk_id)
        if key in seen:
            continue
        if len(seeds) + len(selected) >= max_items:
            break
        if cost([*seeds, *selected, ev]) > max_tokens:
            continue
        selected.append(ev)
        seen.add(key)
    if cost([*seeds, *selected]) > max_tokens:
        raise ValueError("final evidence pack exceeds max_tokens; pack_cost must be deterministic")
    return selected


def fit_evidence_to_budget(
    items: Iterable[SelectedEvidence],
    *,
    max_items: int,
    max_tokens: int,
    already_selected: Iterable[SelectedEvidence] | None = None,
    pack_cost: Callable[[tuple[SelectedEvidence, ...]], int] | None = None,
) -> List[SelectedEvidence]:
    """Return greedy additions; an optional pure callback measures the whole pack.

    Exact mode measures the stable-sorted union with already_selected and ignores
    token_est. The callback receives a tuple and must not mutate evidence items.
    Costs need not be additive or monotone. Over-budget candidates are skipped.
    """
    if pack_cost is not None:
        return _fit_with_pack_cost(items, max_items=max_items, max_tokens=max_tokens,
                                   already_selected=already_selected, pack_cost=pack_cost)
    selected: List[SelectedEvidence] = []
    seen: Set[tuple[str, str]] = set()

    used_tokens = 0
    used_items = 0

    if already_selected:
        for ev in already_selected:
            seen.add((ev.doc_id, ev.chunk_id))
            used_items += 1
            used_tokens += int(ev.token_est or 0)

    for ev in items:
        key = (ev.doc_id, ev.chunk_id)
        if key in seen:
            continue

        ev_tokens = int(ev.token_est or 0)

        if used_items + 1 > max_items:
            break
        if used_tokens + ev_tokens > max_tokens:
            continue

        selected.append(ev)
        seen.add(key)
        used_items += 1
        used_tokens += ev_tokens

    return selected
