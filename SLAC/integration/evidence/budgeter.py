from __future__ import annotations

from typing import Iterable, List, Set, Tuple

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


def fit_evidence_to_budget(
    items: Iterable[SelectedEvidence],
    *,
    max_items: int,
    max_tokens: int,
    already_selected: Iterable[SelectedEvidence] | None = None,
) -> List[SelectedEvidence]:
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