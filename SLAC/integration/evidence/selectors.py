from __future__ import annotations

from typing import Iterable, List, Set, Tuple

from SLAC.integration.evidence.budgeter import fit_evidence_to_budget, stable_sort_evidence
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


def select_evidence(
    candidates: List[SelectedEvidence],
    *,
    max_items: int,
    max_tokens: int,
    prefer_direct_first: bool = True,
    min_direct_evidence: int = 1,
) -> List[SelectedEvidence]:
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