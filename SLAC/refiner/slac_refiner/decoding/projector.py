from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def boundary_vector_to_spans(num_atoms: int, b: Sequence[int]) -> List[Tuple[int, int]]:
    if num_atoms <= 0:
        return []

    gaps = [i for i, x in enumerate(b) if int(x) == 1]
    spans: List[Tuple[int, int]] = []
    start = 0

    for g in gaps:
        end = g + 1
        spans.append((start, end))
        start = end

    spans.append((start, num_atoms))
    return spans


def spans_to_units(atoms_text: Sequence[str], spans: Sequence[Tuple[int, int]]) -> List[Dict]:
    units = []
    for uid, (s, e) in enumerate(spans):
        text = "\n".join(atoms_text[s:e]).strip()
        units.append(
            {
                "unit_id": uid,
                "text": text,
                "start_atom": s,
                "end_atom": e,
            }
        )
    return units


def rebuild_chunks_from_boundary_vector(
    atoms_text: Sequence[str],
    b: Sequence[int],
) -> Dict:
    spans = boundary_vector_to_spans(len(atoms_text), b)
    units = spans_to_units(atoms_text, spans)
    return {
        "spans": spans,
        "units": units,
    }