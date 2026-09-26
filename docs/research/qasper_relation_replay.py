"""Gold-free, deterministic I/C/S policy replay on a fixed native-unit pool.

This is an attribution-mode *heuristic*, not an optimal graph decoder. I and C
have identical decisions; an offline replay cannot measure real cache savings.
S uses dependent adjacency labels first for bounded local merges, then uses
those same accepted edges as a one-hop bonus during evidence selection. Merges
do not add candidates, alter text, change retrieval scores, or force a whole
chunk into the final evidence pack. Singleton units exceeding the chunk target
are retained and reported; the final evidence budget is always a hard limit.

Inputs deliberately exclude questions, answers, annotations, metrics and gold.
Backend labels are ordinal policy weights, not calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from run_qasper_evidence_baselines import PackCounter, Unit, render_pack


HEURISTIC_VERSION = "dependent-adjacency-bonus-greedy-v1"
RELEVANCE_WEIGHTS = {"yes": 1.0, "no": 0.0, "unknown": 0.5}
RELATION_WEIGHTS = {"dependent": 1.0, "independent": 0.0, "unknown": 0.5}


@dataclass(frozen=True)
class AdjacentRelation:
    left_id: str
    right_id: str
    label: str

    @property
    def edge_id(self) -> str:
        payload = json.dumps([self.left_id, self.right_id], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        return "adj-" + hashlib.sha256(payload).hexdigest()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate(units, candidate_ids, relevance_labels, relations,
              retrieval_ranking, mode, budget, chunk_budget, max_units):
    if mode not in {"I", "C", "S"}:
        raise ValueError("mode must be I, C or S")
    for value, name in ((budget, "budget"), (chunk_budget, "chunk_budget"),
                        (max_units, "max_units")):
        _positive_integer(value, name)
    if not units or any(not isinstance(unit, Unit) for unit in units):
        raise ValueError("units must contain the complete document Unit sequence")
    by_id = {unit.unit_id: index for index, unit in enumerate(units)}
    if len(by_id) != len(units) or any(not isinstance(unit.unit_id, str) or not unit.unit_id for unit in units):
        raise ValueError("unit IDs must be nonempty and unique")
    # Full native-document order makes adjacency independently checkable rather
    # than inferring it from the potentially sparse candidate subset.
    if [unit.order for unit in units] != list(range(len(units))):
        raise ValueError("units must be the complete document in contiguous source order")
    if any(not isinstance(unit.text, str) or not unit.text.strip()
           or not isinstance(unit.native_text, str) or not unit.native_text.strip()
           for unit in units):
        raise ValueError("native units must contain nonempty text")
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids) or not set(candidate_ids) <= set(by_id):
        raise ValueError("candidate IDs must be nonempty, unique and present in units")
    if set(relevance_labels) != set(candidate_ids):
        raise ValueError("relevance labels must exactly cover the frozen candidate pool")
    if any(label not in RELEVANCE_WEIGHTS for label in relevance_labels.values()):
        raise ValueError("relevance labels must be yes/no/unknown")
    if len(retrieval_ranking) != len(candidate_ids) or set(retrieval_ranking) != set(candidate_ids):
        raise ValueError("retrieval ranking must be a permutation of all candidate IDs")
    seen = set()
    for relation in relations:
        if not isinstance(relation, AdjacentRelation):
            raise ValueError("relations must be AdjacentRelation instances")
        if relation.left_id not in candidate_ids or relation.right_id not in candidate_ids:
            raise ValueError("relation endpoint is outside the frozen candidate pool")
        if by_id[relation.right_id] != by_id[relation.left_id] + 1:
            raise ValueError("relation endpoints must be truly adjacent in source order")
        if relation.label not in RELATION_WEIGHTS:
            raise ValueError("relation labels must be dependent/independent/unknown")
        pair = (relation.left_id, relation.right_id)
        if pair in seen:
            raise ValueError("duplicate relation endpoints")
        seen.add(pair)
    return by_id


def _partition(indices, relations, by_id, count, mode, chunk_budget):
    groups = [[index] for index in indices]
    accepted, trace = [], []
    for edge in sorted(relations, key=lambda item: by_id[item.left_id]):
        item = {"edge_id": edge.edge_id, "left_id": edge.left_id,
                "right_id": edge.right_id, "label": edge.label,
                "ordinal_weight": RELATION_WEIGHTS[edge.label],
                "changed_boundary": False, "proposed_tokens": None}
        if mode != "S":
            item["action"] = "independent_unit_boundaries"
        elif edge.label != "dependent":
            item["action"] = "keep_boundary_not_dependent"
        else:
            left = next(i for i, group in enumerate(groups) if by_id[edge.left_id] in group)
            right = next(i for i, group in enumerate(groups) if by_id[edge.right_id] in group)
            if right != left + 1:
                raise ValueError("invalid local partition adjacency")
            proposed = groups[left] + groups[right]
            tokens = count(proposed)
            item["proposed_tokens"] = tokens
            if tokens <= chunk_budget:
                groups[left:right + 1] = [proposed]
                accepted.append(edge)
                item.update(action="merge", changed_boundary=True)
            else:
                item["action"] = "keep_boundary_chunk_budget"
        trace.append(item)
    return groups, accepted, trace


def _select(units, indices, relevance_labels, retrieval_ranking, accepted,
            by_id, count, budget, max_units):
    ranks = {unit_id: index for index, unit_id in enumerate(retrieval_ranking)}
    scores = {index: RELEVANCE_WEIGHTS[relevance_labels[units[index].unit_id]] for index in indices}
    pending = {index for index in indices if scores[index] > 0}
    selected, selected_text, trace = [], set(), []
    neighbours = {index: [] for index in indices}
    for edge in accepted:
        left, right = by_id[edge.left_id], by_id[edge.right_id]
        neighbours[left].append((right, edge.edge_id))
        neighbours[right].append((left, edge.edge_id))

    def independent_key(index):
        return -scores[index], ranks[units[index].unit_id], units[index].order

    while pending and len(selected) < max_units:
        linked = {index: sorted(edge_id for other, edge_id in neighbours[index] if other in selected)
                  for index in pending}
        # The maximum adjacent dependent-edge weight is exactly 1. Multiple
        # neighbours do not accumulate an arbitrarily large score advantage.
        first_independent = min(pending, key=independent_key)
        index = min(pending, key=lambda i: (-(scores[i] + bool(linked[i])),
                                           ranks[units[i].unit_id], units[i].order))
        pending.remove(index)
        proposed = sorted([*selected, index])
        item = {"step": len(trace), "unit_id": units[index].unit_id,
                "selected_before": [units[i].unit_id for i in sorted(selected)],
                "base_score": scores[index], "relation_bonus": float(bool(linked[index])),
                "effective_score": scores[index] + bool(linked[index]),
                "relation_edge_ids": linked[index],
                "independent_first_remaining": units[first_independent].unit_id,
                "relation_changed_priority": index != first_independent,
                "proposed_tokens": None, "accepted": False}
        if units[index].native_text in selected_text:
            item["action"] = "skip_exact_native_duplicate"
        else:
            item["proposed_tokens"] = count(proposed)
            if item["proposed_tokens"] <= budget:
                selected.append(index)
                selected_text.add(units[index].native_text)
                item.update(action="select", accepted=True)
            else:
                item["action"] = "skip_evidence_budget"
        trace.append(item)
    return sorted(selected), trace


def replay_policy(
    units: Sequence[Unit],
    candidate_ids: Sequence[str],
    relevance_labels: Mapping[str, str],
    relations: Sequence[AdjacentRelation],
    retrieval_ranking: Sequence[str],
    *,
    mode: str,
    budget: int = 1024,
    tokenizer,
    chunk_budget: int = 384,
    max_units: int = 3,
    deadline: float = math.inf,
) -> dict:
    """Replay a policy without loading files, invoking a model, or reading gold.

    ``units`` is the complete document, with ``order == range(len(units))``;
    ``candidate_ids`` freezes the allowed subset. All modes see identical inputs.
    ``relations`` may be sparse, but each edge must join truly adjacent units.
    I/C keep native-unit boundaries and share the same relevance-only policy.
    S merges dependent edges left-to-right when the rendered group fits the
    chunk target. Only edges accepted there can supply selection bonuses.

    All policies choose at most ``max_units`` distinct complete native strings;
    yes/unknown are eligible, no is excluded. A shared score is q + max accepted
    edge weight to an already selected neighbour. Ties use frozen retrieval rank
    then source order; outputs always render in source order. Non-fitting units
    are skipped, never truncated. There is no answer generation or metric here.

    The comparison to I uses the same candidate pool, labels, ranking, budget
    and tokenizer. ``changed_boundary`` and ``relation_changed_priority`` are
    traces of decisions, not evidence of a quality improvement or cache saving.
    """
    units, candidate_ids = tuple(units), tuple(candidate_ids)
    relations, retrieval_ranking = tuple(relations), tuple(retrieval_ranking)
    relevance_labels = dict(relevance_labels)
    by_id = _validate(units, candidate_ids, relevance_labels, relations,
                      retrieval_ranking, mode, budget, chunk_budget, max_units)
    indices = sorted(by_id[unit_id] for unit_id in candidate_ids)
    count = PackCounter(tokenizer, units, deadline=deadline)
    groups, accepted, boundary_trace = _partition(indices, relations, by_id, count, mode, chunk_budget)
    selected, selection_trace = _select(units, indices, relevance_labels, retrieval_ranking,
                                        accepted, by_id, count, budget, max_units)
    independent, _ = _select(units, indices, relevance_labels, retrieval_ranking,
                             [], by_id, count, budget, max_units)
    final_tokens = count(selected)
    if final_tokens > budget:
        raise ValueError("actual rendered evidence exceeds the hard token budget")
    difference = sorted(set(selected) ^ set(independent))
    used_edge_ids = sorted({edge_id for item in selection_trace if item["accepted"]
                            for edge_id in item["relation_edge_ids"]})
    payload = [{"unit_id": units[i].unit_id, "order": units[i].order,
                "kind": units[i].kind, "start": units[i].start, "end": units[i].end,
                "text": units[i].text, "native_text": units[i].native_text} for i in indices]
    candidate_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "heuristic_version": HEURISTIC_VERSION, "mode": mode,
        "candidate_ids": [units[i].unit_id for i in indices],
        "candidate_information_sha256": candidate_hash,
        "retrieval_ranking": list(retrieval_ranking),
        "budget": budget, "chunk_budget": chunk_budget, "max_units": max_units,
        "chunks": [[units[i].unit_id for i in group] for group in groups],
        "chunk_tokens": [count(group) for group in groups],
        "oversize_singleton_ids": [units[group[0]].unit_id for group in groups
                                    if len(group) == 1 and count(group) > chunk_budget],
        "accepted_merge_edge_ids": [edge.edge_id for edge in accepted],
        "boundary_trace": boundary_trace, "selection_trace": selection_trace,
        "selected_ids": [units[i].unit_id for i in selected],
        "selected_indices": selected, "actual_evidence_tokens": final_tokens,
        "pack_sha256": hashlib.sha256(render_pack(units, selected).encode("utf-8")).hexdigest(),
        "independent_selected_ids": [units[i].unit_id for i in independent],
        "selection_symmetric_difference_ids": [units[i].unit_id for i in difference],
        "selection_symmetric_difference_count": len(difference),
        "added_vs_independent_ids": [units[i].unit_id for i in selected if i not in independent],
        "removed_vs_independent_ids": [units[i].unit_id for i in independent if i not in selected],
        "selection_bonus_used_edge_ids": used_edge_ids,
        "selection_priority_changed_steps": sum(item["relation_changed_priority"] for item in selection_trace),
        "excluded_no_ids": [units[i].unit_id for i in indices if relevance_labels[units[i].unit_id] == "no"],
        "api_calls": 0, "gold_consumed": False, "cache_cost_measured": False,
        "optimality_claimed": False,
    }
