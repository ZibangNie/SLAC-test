from __future__ import annotations

from typing import Callable, Iterable, List, Set, Tuple

from SLAC.integration.evidence.budgeter import (
    _checked_pack_cost, _nonnegative_integer, fit_evidence_to_budget, stable_sort_evidence,
)
from SLAC.integration.io.schemas import SelectedEvidence


def _dedupe_preserve_order(items: Iterable[SelectedEvidence]) -> List[SelectedEvidence]:
    deduped: List[SelectedEvidence] = []
    seen: Set[Tuple[str, str]] = set()

    for ev in items:
        key = (ev.doc_id, ev.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    return deduped


def _select_with_pack_cost(candidates, *, max_items, max_tokens, prefer_direct_first,
                           min_direct_evidence, pack_cost):
    _nonnegative_integer(max_items, "max_items")
    _nonnegative_integer(max_tokens, "max_tokens")
    _nonnegative_integer(min_direct_evidence, "min_direct_evidence")
    if not callable(pack_cost):
        raise TypeError("pack_cost must be callable")
    ordered = _dedupe_preserve_order(stable_sort_evidence(candidates))
    def final_order_cost(trial):
        identities = {(ev.doc_id, ev.chunk_id) for ev in trial}
        # Global stable ties must survive direct-first selection as well.
        final_order = tuple(ev for ev in ordered if (ev.doc_id, ev.chunk_id) in identities)
        return pack_cost(final_order)

    selected = []
    if prefer_direct_first and min_direct_evidence > 0:
        direct = [ev for ev in ordered if (ev.role or "").strip() == "direct"]
        if direct:
            selected.extend(fit_evidence_to_budget(direct,
                max_items=min(max_items, min_direct_evidence), max_tokens=max_tokens,
                pack_cost=final_order_cost))
    # Preserve the existing two phases, including retries of unselected direct items.
    selected_ids = {(ev.doc_id, ev.chunk_id) for ev in selected}
    remaining = [ev for ev in ordered if (ev.doc_id, ev.chunk_id) not in selected_ids]
    selected.extend(fit_evidence_to_budget(remaining, max_items=max_items, max_tokens=max_tokens,
                                         already_selected=selected, pack_cost=final_order_cost))
    identities = {(ev.doc_id, ev.chunk_id) for ev in selected}
    final = [ev for ev in ordered if (ev.doc_id, ev.chunk_id) in identities]
    if _checked_pack_cost(final, pack_cost) > max_tokens:
        raise ValueError("final evidence pack exceeds max_tokens; pack_cost must be deterministic")
    return final


def select_evidence(
    candidates: List[SelectedEvidence],
    *,
    max_items: int,
    max_tokens: int,
    prefer_direct_first: bool = True,
    min_direct_evidence: int = 1,
    pack_cost: Callable[[tuple[SelectedEvidence, ...]], int] | None = None,
) -> List[SelectedEvidence]:
    """Keep legacy selection by default; opt in to whole-render greedy costs.

    The pure callback receives tuples in the exact final stable order, including
    tie order. It must not mutate items. The existing two selection phases are
    retained; an unselected direct item may be evaluated again in the second.
    """
    if pack_cost is not None:
        return _select_with_pack_cost(candidates, max_items=max_items, max_tokens=max_tokens,
            prefer_direct_first=prefer_direct_first, min_direct_evidence=min_direct_evidence,
            pack_cost=pack_cost)
    if not candidates:
        return []

    ordered = _dedupe_preserve_order(stable_sort_evidence(candidates))

    if not prefer_direct_first:
        return fit_evidence_to_budget(
            ordered,
            max_items=max_items,
            max_tokens=max_tokens,
        )

    direct = [ev for ev in ordered if (ev.role or "").strip() == "direct"]
    non_direct = [ev for ev in ordered if (ev.role or "").strip() != "direct"]

    selected: List[SelectedEvidence] = []

    if min_direct_evidence > 0 and direct:
        selected.extend(
            fit_evidence_to_budget(
                direct,
                max_items=min(max_items, min_direct_evidence),
                max_tokens=max_tokens,
            )
        )

    remaining_ordered = [ev for ev in ordered if (ev.doc_id, ev.chunk_id) not in {(x.doc_id, x.chunk_id) for x in selected}]
    selected.extend(
        fit_evidence_to_budget(
            remaining_ordered,
            max_items=max_items,
            max_tokens=max_tokens,
            already_selected=selected,
        )
    )

    # 最终保持稳定顺序：仍按原冻结排序语义输出
    selected_keys = {(x.doc_id, x.chunk_id) for x in selected}
    final_selected = [ev for ev in ordered if (ev.doc_id, ev.chunk_id) in selected_keys]
    return final_selected[:max_items]
