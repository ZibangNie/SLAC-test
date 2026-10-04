"""Pure aggregation for complete cached, single-unit evidence additions.

This module reads no artifacts, references, provider responses, or credentials.
It analyzes already joined endpoints; the caller must first freeze the complete
eligible edge manifest and verify inherited cache/prompt/model contracts. The
optional count check cannot replace that manifest identity check.

An edge is a within-question difference between two saved answers. It is not
an independent trial, a repeated-generation estimate, or evidence of a causal
support-label effect or a second-order interaction. Support strata partition
all edges without admitting or removing an edge by its label or answer score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real


QueryKey = tuple[str, str, str]
SUPPORT_LABELS = ("yes", "no", "unknown")
STRATA = (
    "all", "empty", "nonempty", "support_yes", "support_no", "support_unknown",
    "empty_support_yes", "empty_support_no", "empty_support_unknown",
    "nonempty_support_yes", "nonempty_support_no", "nonempty_support_unknown",
)


@dataclass(frozen=True)
class CachedEndpoint:
    """A complete saved answer for one source-ordered, deduplicated query-pack."""

    key: QueryKey
    unit_ids: tuple[str, ...]
    cache_id: str
    answer_f1: float
    evidence_tokens: int


@dataclass(frozen=True)
class AdditionEdge:
    """One frozen addition; raw yes scores are retained but never thresholded."""

    key: QueryKey
    subset: CachedEndpoint
    superset: CachedEndpoint
    added_unit_id: str
    support_task_id: str
    support_label: str
    raw_yes_score: float | None = None


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity string")
    return value


def _key(value: object) -> QueryKey:
    if type(value) is not tuple or len(value) != 3:
        raise ValueError("query key must be an immutable (family, document, question) tuple")
    for part in value:
        _identity(part, "query key part")
    return value


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real score, excluding bool")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return number


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _endpoint(endpoint: CachedEndpoint) -> tuple[QueryKey, tuple[str, ...], float, int]:
    if not isinstance(endpoint, CachedEndpoint):
        raise TypeError("both complete endpoints must be CachedEndpoint records")
    key = _key(endpoint.key)
    _identity(endpoint.cache_id, "cache ID")
    if type(endpoint.unit_ids) is not tuple:
        raise ValueError("endpoint unit IDs must be an immutable canonical tuple")
    for unit_id in endpoint.unit_ids:
        _identity(unit_id, "unit ID")
    if len(set(endpoint.unit_ids)) != len(endpoint.unit_ids):
        raise ValueError("endpoint unit IDs must be unique")
    return (key, endpoint.unit_ids, _score(endpoint.answer_f1, "endpoint Answer F1"),
            _integer(endpoint.evidence_tokens, "endpoint evidence tokens"))


def _pool(pool_keys: Sequence[QueryKey]) -> tuple[QueryKey, ...]:
    if not isinstance(pool_keys, Sequence) or not pool_keys:
        raise ValueError("pool_keys must be a nonempty sequence of the original cohort")
    keys = tuple(_key(key) for key in pool_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("original pool query keys must be unique")
    return keys


def _validated_edges(
    edges: Sequence[AdditionEdge], pool_keys: tuple[QueryKey, ...], expected_edge_count: int | None,
) -> tuple[AdditionEdge, ...]:
    if not isinstance(edges, Sequence):
        raise TypeError("edges must be a sequence of the complete frozen manifest")
    result = tuple(edges)
    if expected_edge_count is not None:
        expected = _integer(expected_edge_count, "expected_edge_count")
        if len(result) != expected:
            raise ValueError("edge count does not match the complete frozen manifest")

    pool = set(pool_keys)
    edge_ids: set[tuple[QueryKey, tuple[str, ...], tuple[str, ...]]] = set()
    cache_bindings: dict[str, tuple] = {}
    pack_bindings: dict[tuple[QueryKey, tuple[str, ...]], str] = {}
    task_bindings: dict[str, tuple] = {}
    added_unit_tasks: dict[tuple[QueryKey, str], str] = {}
    for edge in result:
        if not isinstance(edge, AdditionEdge):
            raise TypeError("every edge must be an AdditionEdge record")
        key = _key(edge.key)
        if key not in pool:
            raise ValueError("edge query is absent from the original pool")
        subset = _endpoint(edge.subset)
        superset = _endpoint(edge.superset)
        if subset[0] != key or superset[0] != key:
            raise ValueError("edge endpoints must share the complete query key")
        added = _identity(edge.added_unit_id, "added unit ID")
        small, large = subset[1], superset[1]
        if (len(large) != len(small) + 1 or added in small or added not in large
                or tuple(unit for unit in large if unit != added) != small):
            raise ValueError("edge must add exactly one unit and preserve canonical source order")
        edge_id = (key, small, large)
        if edge_id in edge_ids:
            raise ValueError("duplicate addition edge")
        edge_ids.add(edge_id)

        for endpoint, signature in ((edge.subset, subset), (edge.superset, superset)):
            prior = cache_bindings.setdefault(endpoint.cache_id, signature)
            if prior != signature:
                raise ValueError("cache identity conflicts in query, pack, score, or tokens")
            pack = (key, endpoint.unit_ids)
            prior_cache = pack_bindings.setdefault(pack, endpoint.cache_id)
            if prior_cache != endpoint.cache_id:
                raise ValueError("identical query-pack maps to multiple cache identities")

        task = _identity(edge.support_task_id, "support task ID")
        if edge.support_label not in SUPPORT_LABELS:
            raise ValueError("support label must be exactly yes, no, or unknown")
        raw_yes = None if edge.raw_yes_score is None else _score(edge.raw_yes_score, "raw yes score")
        task_signature = (key, added, edge.support_label, raw_yes)
        if task_bindings.setdefault(task, task_signature) != task_signature:
            raise ValueError("support task identity conflicts in query, added unit, label, or score")
        if added_unit_tasks.setdefault((key, added), task) != task:
            raise ValueError("one query/added-unit pair maps to multiple support task identities")
    return result


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sign_counts(values: Sequence[float], names: tuple[str, str, str]) -> dict[str, int]:
    # Strict numerical comparison: no epsilon or rounding-based tie definition.
    return {
        names[0]: sum(value > 0 for value in values),
        names[1]: sum(value == 0 for value in values),
        names[2]: sum(value < 0 for value in values),
    }


def _deltas(edges: tuple[AdditionEdge, ...], *, tokens: bool) -> dict | None:
    if not edges:
        return None
    by_question: dict[QueryKey, list[float]] = defaultdict(list)
    edge_values = []
    for edge in edges:
        if tokens:
            delta = edge.superset.evidence_tokens - edge.subset.evidence_tokens
        else:
            delta = float(edge.superset.answer_f1) - float(edge.subset.answer_f1)
        edge_values.append(delta)
        by_question[edge.key].append(delta)
    question_values = {key: _mean(values) for key, values in by_question.items()}
    by_family: dict[str, list[float]] = defaultdict(list)
    for key, value in question_values.items():
        by_family[key[0]].append(value)
    family_values = [_mean(values) for values in by_family.values()]
    suffix = "increase_equal_decrease" if tokens else "win_tie_loss"
    names = ("increases", "equal", "decreases") if tokens else ("wins", "ties", "losses")
    return {
        "covered_question_mean": _mean(tuple(question_values.values())),
        "covered_family_balanced_mean": _mean(family_values),
        f"edge_{suffix}": _sign_counts(edge_values, names),
        f"question_{suffix}": _sign_counts(tuple(question_values.values()), names),
        f"family_{suffix}": _sign_counts(family_values, names),
    }


def analyze_additions(
    edges: Sequence[AdditionEdge],
    *,
    pool_keys: Sequence[QueryKey],
    expected_edge_count: int | None = None,
) -> dict:
    """Return only aggregates over twelve predeclared coverage strata.

    The original pool supplies coverage denominators, not zero-filled quality
    observations. First average eligible edges within each covered question,
    then either average covered questions, or average questions within each
    covered family and give those families equal weight. Token deltas use the
    same hierarchy. Empty strata have zero coverage and unavailable (``None``)
    contrasts, never imputed zero effects.

    The twelve strata are base all/empty/nonempty crossed with support
    all/yes/no/unknown. Their question/family coverage overlaps; it must not be
    added across strata. This function rejects all malformed or conflicting
    edges instead of reducing the denominator. Real identities, raw scores,
    and per-edge outcomes are not included in its aggregate return value.
    """
    pool = _pool(pool_keys)
    checked = _validated_edges(edges, pool, expected_edge_count)
    n_questions = len(pool)
    n_families = len({key[0] for key in pool})
    buckets: dict[str, list[AdditionEdge]] = {name: [] for name in STRATA}
    for edge in checked:
        base = "empty" if not edge.subset.unit_ids else "nonempty"
        for name in ("all", base, f"support_{edge.support_label}", f"{base}_support_{edge.support_label}"):
            buckets[name].append(edge)
    strata = {}
    for name in STRATA:
        selected = tuple(buckets[name])
        covered_questions = len({edge.key for edge in selected})
        covered_families = len({edge.key[0] for edge in selected})
        strata[name] = {
            "available": bool(selected),
            "edge_count": len(selected),
            "covered_questions": covered_questions,
            "covered_families": covered_families,
            "question_coverage": covered_questions / n_questions,
            "family_coverage": covered_families / n_families,
            "answer_f1_delta": _deltas(selected, tokens=False),
            "evidence_token_delta": _deltas(selected, tokens=True),
        }
    return {
        "pool": {"question_count": n_questions, "family_count": n_families},
        "edge_count": len(checked),
        "strata": strata,
    }
