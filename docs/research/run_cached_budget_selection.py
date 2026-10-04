"""Zero-network, reference-free opportunity gate on cached JEV development data.

Outputs with identities remain in the ignored research artifact directory.
Only summary.json is intended for public reporting. This is not a quality test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from SLAC.retrieval.pack.budget_selection import BudgetCandidate, select_budgeted, select_greedy

ARTIFACTS = REPO / "artifacts/research-foundation"
SALT = "slac-budget-selection-20261004-sample-v1"
IDENTITY = ("family_id", "doc_id", "question_id")
METHODS = ("score_greedy", "score_exact", "density_greedy", "matched_resource_exact")
PREPARED = "qasper-extended-development-prepared-01/prepared.json"
LABELS = "qasper-primary-support-recovery-run-01/labels.json"
SCORES = "qasper-primary-support-recovery-run-01/raw_scores.json"
OLD_RECORDS = "qasper-primary-support-recovery-run-01/per_question.jsonl"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")


def under_artifacts(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()) or path == ARTIFACTS.resolve():
        raise ValueError("paths must stay within the research artifact directory")
    return path


def render(units, indexes):
    return "\n\n".join(f"[{units[i]['unit_id']}]\n{units[i]['text']}"
                       for i in sorted(indexes, key=lambda i: units[i]["order"]))


def sample_identities(queries):
    families = sorted({q["family_id"] for q in queries},
                      key=lambda family: (text_hash(f"{SALT}|{family}"), family))[:8]
    result = []
    for family in families:
        members = [q for q in queries if q["family_id"] == family]
        members.sort(key=lambda q: (text_hash("|".join((SALT, *(q[k] for k in IDENTITY)))),
                                    tuple(q[k] for k in IDENTITY)))
        result.extend(tuple(q[k] for k in IDENTITY) for q in members[:2])
    return result


def build_candidates(query, units, support, labels, scores):
    """Project cached inputs without references or answer-quality fields."""
    ids = query["candidate_ids"]
    by_id = {unit["unit_id"]: i for i, unit in enumerate(units)}
    ranking = query["ranked_ids"]
    if (not 1 <= len(ids) <= 16 or len(ids) != len(set(ids)) or
            len(by_id) != len(units) or len(ranking) != len(ids) or
            set(ids) != set(ranking) or not set(ids) <= set(by_id)):
        raise ValueError("candidate identity contract failed")
    retrieval_rank = {uid: rank for rank, uid in enumerate(ranking)}
    eligible, values = [], {}
    for uid in ids:
        task = support[(query["doc_id"], query["question_id"], uid)]
        choice, value = labels[task], scores[task]
        if choice not in {"yes", "no", "unknown"} or set(value) != {"yes", "no", "unknown"}:
            raise ValueError("invalid cached JEV judgment")
        if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1
               for x in value.values()) or not 0.985 <= sum(value.values()) <= 1.015:
            raise ValueError("cached JEV reported-score contract failed")
        values[uid] = value["yes"]
        if choice != "no":
            eligible.append(uid)
    priority = sorted(eligible, key=lambda uid: (-values[uid], retrieval_rank[uid], units[by_id[uid]]["order"]))
    ranks = {uid: rank for rank, uid in enumerate(priority)}
    eligible.sort(key=lambda uid: by_id[uid])
    indexes = tuple(by_id[uid] for uid in eligible)
    candidates = tuple(BudgetCandidate(uid, values[uid], units[by_id[uid]]["native_text"], ranks[uid])
                       for uid in eligible)
    return candidates, indexes


def describe(values):
    return {"mean": math.fsum(values) / len(values), "min": min(values), "max": max(values)}


def summarize(records, gate):
    by_method = {method: [r for r in records if r["method"] == method] for method in METHODS}
    base = {tuple(r[k] for k in IDENTITY): r for r in by_method["score_greedy"]}
    result = {}
    for method, rows in by_method.items():
        families = defaultdict(list)
        deltas, token_deltas, unit_deltas, changed = [], [], [], 0
        for row in rows:
            before = base[tuple(row[k] for k in IDENTITY)]
            deltas.append(row["utility"] - before["utility"])
            token_deltas.append(row["tokens"] - before["tokens"])
            unit_deltas.append(row["unit_count"] - before["unit_count"])
            changed += row["selected_indexes"] != before["selected_indexes"]
            families[row["family_id"]].append(deltas[-1])
        result[method] = {"questions": len(rows), "changed_packs": changed,
            "utility": describe([r["utility"] for r in rows]),
            "tokens": describe([r["tokens"] for r in rows]),
            "unit_count": describe([r["unit_count"] for r in rows]),
            "utility_delta": describe(deltas), "token_delta": describe(token_deltas),
            "unit_delta": describe(unit_deltas),
            "family_balanced_utility_delta": math.fsum(math.fsum(v) / len(v) for v in families.values()) / len(families),
            "utility_win_tie_loss": [sum(v > 0 for v in deltas), sum(v == 0 for v in deltas), sum(v < 0 for v in deltas)]}
    return {"methods": result,
        "greedy_budget_blocked_questions": sum(r["budget_blocked"] for r in gate),
        "feasible_top_priority_questions": sum(r["top_priority_feasible"] for r in gate),
        "feasible_top_priority_with_exact_improvement": sum(r["top_priority_feasible"] and r["exact_improved"] for r in gate),
        "exact_enumerated_subsets": sum(r["enumerated_subsets"] for r in gate)}


def run(args):
    started = time.monotonic()
    output, contract_path = under_artifacts(args.output), under_artifacts(args.contract)
    output.mkdir(parents=True, exist_ok=False)
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    consumed = {str(contract_path): hashlib.sha256(contract_bytes).hexdigest()}
    input_bytes = {}
    for name in (PREPARED, LABELS, SCORES, OLD_RECORDS):
        path = under_artifacts(ARTIFACTS / name)
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != contract["verified_consumed_sha256"][name]:
            raise ValueError("frozen source binding mismatch")
        consumed[str(path)] = actual
        input_bytes[name] = payload
    prepared = json.loads(input_bytes[PREPARED])
    labels = json.loads(input_bytes[LABELS])["jev"]
    scores = json.loads(input_bytes[SCORES])
    documents, queries = prepared["documents"], prepared["queries"]
    identities = [tuple(q[k] for k in IDENTITY) for q in queries]
    support = {(t["doc_id"], t["question_id"], t["unit_id"]): t["id"] for t in prepared["support_tasks"]}
    task_ids = set(support.values())
    if (len(documents) != 24 or len(queries) != 77 or len(set(identities)) != 77 or
            len({q["family_id"] for q in queries}) != 24 or len(support) != 1214 or
            len(task_ids) != 1214 or set(labels) != task_ids or set(scores) != task_ids):
        raise ValueError("frozen all-question scope mismatch")
    for units in documents.values():
        if [u["order"] for u in units] != list(range(len(units))):
            raise ValueError("source unit order changed")

    # Project only structural/token fields from historical records; discard metrics.
    baselines, token_cache = {}, {doc: {(): 0} for doc in documents}
    for line in input_bytes[OLD_RECORDS].decode("utf-8").splitlines():
            old = json.loads(line)
            doc, units = old["doc_id"], documents[old["doc_id"]]
            by_id = {u["unit_id"]: i for i, u in enumerate(units)}
            indexes = tuple(sorted(by_id[uid] for uid in old["selected_ids"]))
            if text_hash(render(units, indexes)) != old["pack_sha256"]:
                raise ValueError("saved pack content changed")
            tokens = old["actual_evidence_tokens"]
            if type(tokens) is not int or tokens < 0 or (indexes in token_cache[doc] and token_cache[doc][indexes] != tokens):
                raise ValueError("saved pack token contract failed")
            token_cache[doc][indexes] = tokens
            if old["method"] == "p_yes_only_k3":
                identity = tuple(old[k] for k in IDENTITY)
                if identity in baselines:
                    raise ValueError("duplicate baseline identity")
                baselines[identity] = (indexes, tokens)
    if set(baselines) != set(identities):
        raise ValueError("incomplete baseline coverage")

    tokenizer_path = Path(contract["tokenizer"]["path"]).resolve()
    for name, expected in contract["tokenizer"]["verified_file_sha256"].items():
        path = (tokenizer_path / name).resolve()
        if path.parent != tokenizer_path or digest(path) != expected:
            raise ValueError("local tokenizer binding mismatch")
        consumed[str(path)] = expected
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    code_paths = (Path(__file__), REPO / "SLAC/retrieval/pack/budget_selection.py",
                  REPO / "docs/research/BUDGET_SELECTION_PROTOCOL_20261004.md")
    code_hashes = {str(path): digest(path) for path in code_paths}
    write_json(output / "plan.json", {"schema": "slac-cached-budget-plan-v1", "api_calls": 0,
        "input_sha256": consumed, "code_sha256": code_hashes,
        "question_count": 77, "family_count": 24, "methods": METHODS,
        "reference_input": None, "qa_sample": sample_identities(queries)})

    records, gates, traces, new_tokenizations = [], [], [], 0
    for query in queries:
        identity = {k: query[k] for k in IDENTITY}
        identity_key = tuple(identity[k] for k in IDENTITY)
        doc, units = query["doc_id"], documents[query["doc_id"]]
        candidates, native_indexes = build_candidates(query, units, support, labels, scores)

        def count(selected):
            nonlocal new_tokenizations
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("offline experiment deadline exceeded")
            indexes = tuple(sorted(native_indexes[i] for i in selected))
            if indexes not in token_cache[doc]:
                token_cache[doc][indexes] = len(tokenizer.encode(render(units, indexes), add_special_tokens=True, truncation=False))
                new_tokenizations += 1
            return token_cache[doc][indexes]

        comparison = select_budgeted(candidates, count, budget_tokens=1024, max_units=3)
        greedy = comparison.greedy
        original_indexes = tuple(native_indexes[i] for i in greedy.selected_indexes)
        if (original_indexes, greedy.cost_tokens) != baselines[identity_key]:
            raise ValueError("greedy no longer reproduces the frozen baseline")
        singleton_costs = [count((i,)) for i in range(len(candidates))]
        if any(cost <= 0 for cost in singleton_costs):
            raise ValueError("nonempty single-unit rendering must consume tokens")
        density_order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].utility / singleton_costs[i], candidates[i].priority))
        density_ranks = {i: rank for rank, i in enumerate(density_order)}
        density_candidates = tuple(replace(c, priority=density_ranks[i]) for i, c in enumerate(candidates))
        density = select_greedy(density_candidates, count, budget_tokens=1024, max_units=3)
        matched = select_budgeted(candidates, count, budget_tokens=greedy.cost_tokens,
                                  max_units=len(greedy.selected_indexes))
        if matched.optimal.utility < greedy.utility:
            raise ValueError("matched-resource optimum cannot lose to its feasible original pack")
        # The matched cap may change its own greedy traversal for nonmonotone
        # costs; ties still preserve the original comparison pack by protocol.
        matched_selection = greedy if matched.optimal.utility == greedy.utility else matched.optimal
        selections = (greedy, comparison.optimal, density, matched_selection)
        for method, chosen in zip(METHODS, selections):
            selected = tuple(native_indexes[i] for i in chosen.selected_indexes)
            records.append({**identity, "method": method, "selected_indexes": selected,
                "pack_sha256": text_hash(render(units, selected)), "utility": chosen.utility,
                "tokens": chosen.cost_tokens, "unit_count": len(selected)})

        # Explain whether the ordinary top-ranked pack was already affordable.
        priority = sorted(range(len(candidates)), key=lambda i: (candidates[i].priority, i))
        top, seen, blocked, scan = [], set(), False, []
        for i in priority:
            if candidates[i].duplicate_key not in seen:
                top.append(i)
                seen.add(candidates[i].duplicate_key)
                if len(top) == 3:
                    break
        seen = set()
        for i in priority:
            if len(scan) == 3:
                break
            if candidates[i].duplicate_key in seen:
                continue
            if count(tuple(sorted((*scan, i)))) > 1024:
                blocked = True
            else:
                scan.append(i)
                seen.add(candidates[i].duplicate_key)
        gates.append({**identity, "budget_blocked": blocked, "top_priority_feasible": count(tuple(sorted(top))) <= 1024,
            "exact_improved": comparison.optimal.utility > greedy.utility,
            "enumerated_subsets": comparison.enumerated_subsets})
        traces.append({**identity, "eligible_native_indexes": native_indexes,
            "score_priority": priority, "enumerated_subsets": comparison.enumerated_subsets})

    for path, expected in {**consumed, **code_hashes}.items():
        if digest(path) != expected:
            raise ValueError("input or implementation changed during execution")
    summary = {"schema": "slac-cached-budget-opportunity-v1", "status": "completed",
        "question_count": 77, "family_count": 24, "api_calls": 0, "key_read": False,
        "reference_input": None, "quality_metrics_computed": False, "answer_generation_calls": 0,
        "independent_confirmation": False, "baseline_reproduced_questions": len(queries),
        "new_tokenizations": new_tokenizations, "qa_sample_count": len(sample_identities(queries)),
        "qa_status": "pending_independent_sample_check", "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": code_hashes[str(code_paths[-1])],
        "input_content_sha256": sorted(consumed.values()), "code_content_sha256": sorted(code_hashes.values()),
        **summarize(records, gates)}
    write_json(output / "selections.json", records)
    write_json(output / "gate.json", gates)
    write_json(output / "traces.json", traces)
    write_json(output / "summary.json", summary)
    return summary


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled for this offline stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 600:
        parser.error("max-seconds must be between 1 and 600")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    try:
        print(json.dumps(run(args), ensure_ascii=True, allow_nan=False))
    except Exception as error:
        # Never echo source text, query identities, or arbitrary exception details.
        print(json.dumps({"status": "failed", "error_class": type(error).__name__, "api_calls": 0}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
