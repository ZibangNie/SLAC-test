"""Exact source-coordinate coverage, without semantic or token-monotonicity claims."""

from collections.abc import Sequence


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer (not bool)")


def _intervals(values: Sequence, name: str, source_length: int) -> list[tuple[int, int]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of interval pairs")
    result = []
    for pair in values:
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray))
                or len(pair) != 2):
            raise ValueError(f"{name} entries must be interval pairs")
        start, end = pair
        _integer(start, f"{name} start")
        _integer(end, f"{name} end")
        if not start < end <= source_length:
            raise ValueError(f"{name} intervals must have positive length and lie within source")
        result.append((start, end))
    return result


def minimal_cover(partition, references, source_length: int) -> dict:
    """Return the unique required chunk set for exact character coverage.

    Partition order is source order. Reference duplicates, overlaps, and adjacent
    intervals are merged. This does not establish semantic evidence sufficiency.
    """
    _integer(source_length, "source_length")
    chunks = _intervals(partition, "partition", source_length)
    cursor = 0
    for start, end in chunks:
        if start != cursor:
            raise ValueError("partition must continuously cover source in order without overlap")
        cursor = end
    if cursor != source_length:
        raise ValueError("partition must cover the complete source")

    union: list[list[int]] = []
    for start, end in sorted(_intervals(references, "references", source_length)):
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])

    required = [
        index for index, (start, end) in enumerate(chunks)
        if any(max(start, ref_start) < min(end, ref_end) for ref_start, ref_end in union)
    ]
    return {
        "required_indices": required,
        "reference_union": union,
        "reference_chars": sum(end - start for start, end in union),
        "required_source_chars": sum(chunks[index][1] - chunks[index][0] for index in required),
    }


def assess_cover(required_count: int, partition_count: int, required_tokens: int | None,
                 *, max_chunks: int = 3, max_tokens: int = 1024) -> str:
    """Assess the required set; an over-budget set need not rule out supersets.

    required_tokens is a caller-supplied complete-render measurement. No tokenizer
    is loaded here and no additive or monotone token-cost assumption is made.
    """
    _integer(required_count, "required_count")
    _integer(partition_count, "partition_count")
    _integer(max_chunks, "max_chunks", positive=True)
    _integer(max_tokens, "max_tokens", positive=True)
    if required_tokens is not None:
        _integer(required_tokens, "required_tokens")
    if required_count > partition_count:
        raise ValueError("required_count cannot exceed partition_count")

    if required_count == 0:
        return "empty_reference"
    if required_count > max_chunks:
        return "infeasible_chunk_count"
    if required_tokens is None:
        return "budget_unmeasured"
    if required_tokens <= max_tokens:
        return "feasible_required_set"
    if required_count == max_chunks or required_count == partition_count:
        return "infeasible_budget_no_superset"
    return "unresolved_budget_nonmonotone"
