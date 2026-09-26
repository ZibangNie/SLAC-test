"""Freeze and execute a bounded, selection-only OpenRouter development pilot.

`plan` is entirely offline. `run` performs only the exact frozen requests. I/C/S
reuse each backend's unique judgments; this is not six cold/warm API runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import time

from transformers import AutoTokenizer

import openrouter_decision_client as client
from prepare_qasper_relation_pilot import verify_hashes
from qasper_metrics import references_from_annotations
from qasper_relation_replay import AdjacentRelation, HEURISTIC_VERSION, replay_policy
from run_qasper_evidence_baselines import (
    PackCounter, Unit, aggregate, check_time, choose_oracle, digest,
    oracle_candidates, pack_ranked, score_selection,
)


CAPS = {"budget_usd": "2", "request_cap": 160, "question_cap": 1600,
        "input_allowance_cap": 8000000, "max_seconds": 1800}
SELECTOR = {"chunk_budget": 384, "budget": 1024, "max_units": 3,
            "heuristic_version": HEURISTIC_VERSION}
CODE_FILES = ("run_qasper_relation_pilot.py", "openrouter_decision_client.py",
              "qasper_relation_replay.py", "prepare_qasper_relation_pilot.py",
              "run_qasper_evidence_baselines.py", "run_qasper_dense_baseline.py",
              "qasper_alignment_v2.py", "qasper_metrics.py")
LIMITS = [
    "Post-baseline exposed development pool, not independent confirmation.",
    "Given-document fixed-pool evidence selection; no answer generation or Answer F1.",
    "I/C/S reuse unique decisions offline; no measured cold/warm caching or cost-saving claim.",
    "All batch items are visible to each backend; instructions request item-local decisions.",
    "BGE rendered-evidence token limits exclude query/instructions and are not generator context limits.",
    "UTF-8 byte reservations are conservative client admission accounting, not a service-side hard monetary cap.",
    "Ordinal labels are not calibrated probabilities; local merges and bonus selection are heuristic.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def failure(output, stage, error, started):
    # Arbitrary exception messages can echo model text, credentials or headers.
    write_json(output / "failure.json", {"status": "failed", "stage": stage,
        "error_class": type(error).__name__, "elapsed_seconds": time.monotonic() - started,
        "actual_model_scores_available": False, "automatic_retries": 0, "fallbacks": 0})


def load_prepared(directory):
    directory = Path(directory).resolve()
    manifest_path, prepared_path = directory / "manifest.json", directory / "prepared.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "prepared" or manifest.get("test_payload_read") is not False:
        raise ValueError("requires a prepared non-test manifest")
    if digest(prepared_path) != manifest["prepared_sha256"]:
        raise ValueError("prepared artifact digest mismatch")
    verify_hashes(manifest["input_sha256"])
    prepared = read_json(prepared_path)
    if prepared.get("schema") != "slac-qasper-relation-pilot-prepared-v1":
        raise ValueError("unsupported prepared schema")
    documents = {doc: [Unit(**unit) for unit in values]
                 for doc, values in prepared["documents"].items()}
    queries = prepared["queries"]
    if not 1 <= len(documents) <= 8 or not 1 <= len(queries) <= 16:
        raise ValueError("prepared dataset exceeds locked pilot scope")
    if len({(row["doc_id"], row["question_id"]) for row in queries}) != len(queries):
        raise ValueError("duplicate prepared query")
    per_family = defaultdict(int)
    for row in queries:
        per_family[row["family_id"]] += 1
        units = documents[row["doc_id"]]
        ids = [unit.unit_id for unit in units]
        candidate_ids = row["candidate_ids"]
        if (len(set(ids)) != len(ids) or [unit.order for unit in units] != list(range(len(units)))
                or not 1 <= len(candidate_ids) <= 16 or len(set(candidate_ids)) != len(candidate_ids)
                or not set(candidate_ids) <= set(ids)
                or len(row["ranked_ids"]) != len(candidate_ids)
                or set(row["ranked_ids"]) != set(candidate_ids)):
            raise ValueError("invalid prepared candidate identity")
    if any(count > 2 for count in per_family.values()):
        raise ValueError("prepared family exceeds query cap")
    return prepared, manifest, documents


def freeze_batches(prepared):
    """Group static edges by document and supports by query before batching."""
    batches, seen = [], set()
    for kind, key in (("static", "static_tasks"), ("support", "support_tasks")):
        groups = {}
        for task in prepared[key]:
            if task["id"] in seen:
                raise ValueError("duplicate task identity")
            seen.add(task["id"])
            group = (task["doc_id"],) if kind == "static" else (task["doc_id"], task["question_id"])
            groups.setdefault(group, []).append({"id": task["id"], "item": task["item"]})
        for group, tasks in groups.items():
            for chunk in client.task_batches(tasks, kind):
                batch = {"kind": kind, "group": list(group), "tasks": chunk}
                batch["id"] = "batch-" + client.object_hash(batch)
                batches.append(batch)
    return batches


def schedule_requests(batches):
    schedule = []
    totals = {"requests": 0, "questions": 0, "input_allowance": 0, "output_allowance": 0}
    total_cost = Decimal("0")
    for index, batch in enumerate(batches):
        order = ("jev", "general") if index % 2 == 0 else ("general", "jev")
        for backend in order:
            payload = client.make_payload(batch["tasks"], batch["kind"], backend)
            reserved, input_allowance, output_allowance = client.reservation(payload, backend)
            schedule.append({"batch_id": batch["id"], "kind": batch["kind"], "backend": backend,
                "task_ids": [task["id"] for task in batch["tasks"]],
                "payload_sha256": client.object_hash(payload), "reserved_usd": str(reserved),
                "input_allowance": input_allowance, "output_allowance": output_allowance})
            totals["requests"] += 1
            totals["questions"] += len(batch["tasks"])
            totals["input_allowance"] += input_allowance
            totals["output_allowance"] += output_allowance
            total_cost += reserved
    totals["reservation_usd"] = str(total_cost)
    if (totals["requests"] > CAPS["request_cap"] or totals["questions"] > CAPS["question_cap"]
            or totals["input_allowance"] > CAPS["input_allowance_cap"]
            or total_cost > Decimal(CAPS["budget_usd"])):
        raise ValueError("frozen request prediction exceeds pilot caps")
    return schedule, totals


def selected_gold(sidecar, prepared):
    wanted = {(row["doc_id"], row["question_id"]): row for row in prepared["queries"]}
    found = {}
    with Path(sidecar).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["doc_id"], row["question_id"])
            if key not in wanted:
                continue
            query = wanted[key]
            if key in found or row["official_split"] != "validation":
                raise ValueError("invalid or duplicate selected sidecar reference")
            if row["family_id"] != query["family_id"] or row["question"] != query["query"]:
                raise ValueError("prepared question differs from frozen sidecar")
            found[key] = row["answer_annotations"]
    if set(found) != set(wanted):
        raise ValueError("selected sidecar references are incomplete")
    return found


def full_rankings(prepared, prepared_manifest):
    paths = [Path(path) for path in prepared_manifest["input_sha256"] if Path(path).name == "rankings.jsonl"]
    if len(paths) != 1:
        raise ValueError("prepared manifest must identify one original dense ranking artifact")
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    found = {}
    for row in rows:
        key = (row["doc_id"], row["question_id"])
        if key in found:
            raise ValueError("duplicate full-pool ranking")
        found[key] = row["ranked_ids"]
    for query in prepared["queries"]:
        key = (query["doc_id"], query["question_id"])
        expected = {unit["unit_id"] for unit in prepared["documents"][query["doc_id"]]}
        if key not in found or len(found[key]) != len(expected) or set(found[key]) != expected:
            raise ValueError("original dense ranking is not a full native-unit permutation")
        if [uid for uid in found[key] if uid in set(query["candidate_ids"])] != query["ranked_ids"]:
            raise ValueError("prepared ranking differs from original dense order")
    return found


def baseline_records(prepared, documents, annotations, rankings, tokenizer, deadline=math.inf):
    records, coverage = [], []
    for query in prepared["queries"]:
        check_time(deadline)
        doc, qid = query["doc_id"], query["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        by_id = {unit.unit_id: index for index, unit in enumerate(units)}
        count = PackCounter(tokenizer, units, deadline=deadline)
        capped_indices = [by_id[uid] for uid in query["candidate_ids"]]
        capped_units = [units[index] for index in capped_indices]
        references = [reference["evidence"] for reference in references_from_annotations(gold)]
        local_count = lambda indices: count([capped_indices[index] for index in indices])
        options = oracle_candidates(capped_units, references, local_count, deadline=deadline)
        # The oracle obeys the same <=3 units as the deployed selector.
        options = [(indices, tokens) for indices, tokens in options if len(indices) <= SELECTOR["max_units"]]
        local_oracle = choose_oracle(options, SELECTOR["budget"], capped_units, gold, deadline)
        choices = {
            "dense_top3_capped": pack_ranked(units, [by_id[uid] for uid in query["ranked_ids"]],
                                               SELECTOR["budget"], count, max_units=3),
            "dense_top3_full_document": pack_ranked(units, [by_id[uid] for uid in rankings[(doc, qid)]],
                                                      SELECTOR["budget"], count, max_units=3),
            "gold_subset_oracle_capped_max3": [capped_indices[index] for index in local_oracle],
            "empty": [],
        }
        capped_texts = {unit.native_text for unit in capped_units}
        all_texts = {unit.native_text for unit in units}
        coverage.append({"doc_id": doc, "family_id": query["family_id"], "question_id": qid,
            "candidate_count": len(capped_units), "full_document_count": len(units),
            "reference_sizes": [len(ref) for ref in references],
            "capped_reachable_reference_items": [sum(text in capped_texts for text in ref) for ref in references],
            "full_reachable_reference_items": [sum(text in all_texts for text in ref) for ref in references],
            "oracle_max_units": 3, "oracle_subsets_within_unit_cap": len(options)})
        for method, chosen in choices.items():
            records.append({"doc_id": doc, "family_id": query["family_id"], "question_id": qid,
                "method": method, "budget": SELECTOR["budget"],
                **score_selection(units, chosen, gold, count, SELECTOR["budget"])})
    return records, coverage


def tokenizer_files(directory):
    directory = Path(directory).resolve()
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "sentencepiece.bpe.model")
    files = [directory / name for name in names if (directory / name).is_file()]
    if not (directory / "tokenizer.json").is_file() or not (directory / "tokenizer_config.json").is_file():
        raise ValueError("requires the fixed local tokenizer files")
    return files


def plan(args):
    started = time.monotonic()
    prepared_dir, sidecar, tokenizer_path, output = (Path(getattr(args, key)).resolve()
        for key in ("prepared", "sidecar", "tokenizer", "output"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        prepared, manifest, documents = load_prepared(prepared_dir)
        inputs = dict(manifest["input_sha256"])
        if inputs.get(str(sidecar)) != digest(sidecar):
            raise ValueError("sidecar is not bound to prepared manifest")
        local_paths = [prepared_dir / "prepared.json", prepared_dir / "manifest.json", sidecar,
                       *tokenizer_files(tokenizer_path)]
        for path in tokenizer_files(tokenizer_path):
            if inputs.get(str(path)) != digest(path):
                raise ValueError("tokenizer differs from frozen dense baseline")
        local_paths.extend(Path(__file__).resolve().parent / name for name in CODE_FILES)
        inputs.update({str(path): digest(path) for path in local_paths})
        batches = freeze_batches(prepared)
        schedule, prediction = schedule_requests(batches)
        references = selected_gold(sidecar, prepared)  # References never enter batches or payloads.
        rankings = full_rankings(prepared, manifest)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        records, coverage = baseline_records(prepared, documents, references, rankings, tokenizer,
                                            deadline=started + CAPS["max_seconds"])
        verify_hashes(inputs)
        write_json(output / "batches.json", batches)
        write_rows(output / "baseline_per_question.jsonl", records)
        write_rows(output / "candidate_coverage.jsonl", coverage)
        write_json(output / "baseline_summary.json", {"status": "completed_offline", "api_calls": 0,
            "metrics": aggregate(records), "oracle_scope": "Gold-reference subsets restricted to the same capped pool, <=3 distinct native units, BGE evidence budget1024; not deployable.",
            "limits": LIMITS})
        config = {"schema": "slac-qasper-relation-experiment-v1", "status": "planned",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_dir": str(prepared_dir), "sidecar": str(sidecar), "tokenizer": str(tokenizer_path),
            "input_sha256": inputs, "plan_files_sha256": {
                name: digest(output / name) for name in ("batches.json", "baseline_per_question.jsonl", "candidate_coverage.jsonl", "baseline_summary.json")},
            "models": client.MODELS, "prompt_version": client.PROMPT_VERSION,
            "caps": dict(CAPS), "selector": dict(SELECTOR), "schedule": schedule,
            "predicted_reservations": prediction, "batch_count": len(batches),
            "query_count": len(prepared["queries"]), "family_count": len(prepared["documents"]),
            "static_grouping": "by document, <=8 questions and common byte-admissible batches",
            "support_grouping": "by document/question, <=8 questions and common byte-admissible batches",
            "visible_context": "Full batch state.items is visible to both backends, despite per-item instructions; static state contains no query.",
            "replay": "One unique result per backend/task, reused for I/C/S; I and C are identical offline policies.",
            "byte_cap_per_payload": client.BYTE_CAP, "limits": LIMITS,
            "preflight_cap_revision": "Before any API execution or model results, total conservative input allowance increased from 4M to 8M after whole-batch repeated-per-JEV-question reservation measured 6.50M; USD2/request160/question1600 caps unchanged.",
            "api_calls": 0, "test_payload_read": False, "independent_evaluation": False,
            "source_hashes_unchanged": True, "elapsed_seconds": time.monotonic() - started}
        write_json(output / "experiment_config.json", config)
        write_json(output / "plan_manifest.json", {"experiment_config_sha256": digest(output / "experiment_config.json"),
                                                   "status": "planned", "api_calls": 0})
        return config
    except Exception as exc:
        failure(output, "plan", exc, started)
        raise


def paired_differences(records):
    by_method = defaultdict(dict)
    for row in records:
        key = (row["family_id"], row["doc_id"], row["question_id"])
        if key in by_method[row["method"]]:
            raise ValueError("duplicate paired metric identity")
        by_method[row["method"]][key] = row
    comparisons = [("S_jev", "C_jev"), ("S_general", "C_general")]
    comparisons.extend((f"{mode}_jev", f"{mode}_general") for mode in ("I", "C", "S"))
    summaries = []
    for plus, minus in comparisons:
        if not by_method[plus] or set(by_method[plus]) != set(by_method[minus]):
            raise ValueError("paired methods require identical complete question coverage")
        item = {"comparison": f"{plus} - {minus}", "statistics": {}}
        for metric in ("official_evidence_f1", "reference_evidence_recall", "actual_evidence_tokens"):
            families, documents = defaultdict(list), defaultdict(list)
            for key, row in by_method[plus].items():
                delta = row[metric] - by_method[minus][key][metric]
                families[key[0]].append(delta)
                documents[key[1]].append(delta)
            family_values = {name: sum(values) / len(values) for name, values in families.items()}
            item["statistics"][metric] = {"family_deltas": family_values,
                "family_macro_delta": sum(family_values.values()) / len(family_values),
                "document_macro_delta": sum(sum(v) / len(v) for v in documents.values()) / len(documents),
                "families": len(family_values), "confidence_interval": None,
                "interpretation": "descriptive paired development comparison; no significance claim"}
        summaries.append(item)
    return summaries


def replay_records(prepared, documents, annotations, labels, tokenizer, deadline=math.inf):
    expected = {task["id"] for name in ("static_tasks", "support_tasks") for task in prepared[name]}
    if set(labels) != set(client.MODELS) or any(set(values) != expected for values in labels.values()):
        raise ValueError("incomplete actual model labels; no partial-run scores")
    records, traces = [], []
    support_lookup = {(task["doc_id"], task["question_id"], task["unit_id"]): task["id"]
                      for task in prepared["support_tasks"]}
    for query in prepared["queries"]:
        check_time(deadline)
        doc, qid = query["doc_id"], query["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        candidate_ids = set(query["candidate_ids"])
        for backend in client.MODELS:
            relevance = {uid: labels[backend][support_lookup[(doc, qid, uid)]] for uid in candidate_ids}
            relations = [AdjacentRelation(task["left_id"], task["right_id"], labels[backend][task["id"]])
                for task in prepared["static_tasks"] if task["doc_id"] == doc
                and task["left_id"] in candidate_ids and task["right_id"] in candidate_ids]
            independent = None
            for mode in ("I", "C", "S"):
                trace = replay_policy(units, query["candidate_ids"], relevance, relations, query["ranked_ids"],
                    mode=mode, tokenizer=tokenizer, budget=SELECTOR["budget"],
                    chunk_budget=SELECTOR["chunk_budget"], max_units=SELECTOR["max_units"], deadline=deadline)
                if mode == "I":
                    independent = trace["selected_ids"]
                elif mode == "C" and trace["selected_ids"] != independent:
                    raise ValueError("I/C replay disagreement")
                method = f"{mode}_{backend}"
                traces.append({"doc_id": doc, "family_id": query["family_id"], "question_id": qid,
                               "method": method, **trace})
                records.append({"doc_id": doc, "family_id": query["family_id"], "question_id": qid,
                    "method": method, "budget": SELECTOR["budget"],
                    **score_selection(units, trace["selected_indices"], gold,
                                      PackCounter(tokenizer, units, deadline=deadline), SELECTOR["budget"])})
    return records, traces


def load_plan(directory):
    directory = Path(directory).resolve()
    seal = read_json(directory / "plan_manifest.json")
    if digest(directory / "experiment_config.json") != seal["experiment_config_sha256"]:
        raise ValueError("frozen experiment configuration digest mismatch")
    config = read_json(directory / "experiment_config.json")
    if config.get("status") != "planned" or config.get("schema") != "slac-qasper-relation-experiment-v1":
        raise ValueError("invalid frozen plan")
    if config["caps"] != CAPS or config["selector"] != SELECTOR or config["models"] != client.MODELS:
        raise ValueError("frozen plan differs from current locked implementation")
    if config["prompt_version"] != client.PROMPT_VERSION:
        raise ValueError("prompt version differs from frozen plan")
    verify_hashes(config["input_sha256"])
    verify_hashes({str(directory / name): value for name, value in config["plan_files_sha256"].items()})
    batches = read_json(directory / "batches.json")
    schedule, totals = schedule_requests(batches)
    if schedule != config["schedule"] or totals != config["predicted_reservations"]:
        raise ValueError("regenerated requests differ from frozen plan")
    return config, batches


def run(args, *, client_factory=client.BoundedClient):
    started = time.monotonic()
    deadline = started + CAPS["max_seconds"]
    plan_dir, output = Path(args.plan).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bounded = None
    try:
        initial_plan_hashes = {str(plan_dir / name): digest(plan_dir / name)
                               for name in ("experiment_config.json", "plan_manifest.json")}
        config, batches = load_plan(plan_dir)
        verify_hashes(initial_plan_hashes)
        prepared, _, documents = load_prepared(config["prepared_dir"])
        annotations = selected_gold(config["sidecar"], prepared)
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
        by_batch = {batch["id"]: batch for batch in batches}
        labels = {backend: {} for backend in client.MODELS}
        bounded = client_factory(output / "provider_calls", key_file=args.key_file, proxy=args.proxy,
            budget_usd=CAPS["budget_usd"], request_cap=CAPS["request_cap"],
            question_cap=CAPS["question_cap"], token_cap=CAPS["input_allowance_cap"])
        for index, item in enumerate(config["schedule"]):
            # Provider transport has a 60-second timeout. Do not start a call
            # that could knowingly exceed the overall 30-minute run window.
            if deadline - time.monotonic() <= 65:
                raise TimeoutError("insufficient remaining provider time allowance")
            batch = by_batch[item["batch_id"]]
            decisions = bounded.submit(batch["tasks"], batch["kind"], item["backend"])
            if set(decisions) != set(item["task_ids"]) or set(decisions) & set(labels[item["backend"]]):
                raise ValueError("provider result identities differ from frozen schedule")
            labels[item["backend"]].update(decisions)
            temporary = output / "labels.tmp"
            temporary.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output / "labels.json")
            print(json.dumps({"completed_requests": index + 1, "total_requests": len(config["schedule"]),
                "actual_reported_cost_usd": bounded.ledger["actual_reported_cost_usd"]}), flush=True)
            check_time(deadline)
        records, traces = replay_records(prepared, documents, annotations, labels, tokenizer, deadline)
        comparisons = paired_differences(records)
        verify_hashes(config["input_sha256"])
        # Bind to the bytes seen at run start, not a potentially replaced seal.
        verify_hashes(initial_plan_hashes)
        verify_hashes({str(plan_dir / name): value for name, value in config["plan_files_sha256"].items()})
        check_time(deadline)
        write_rows(output / "per_question.jsonl", records)
        write_rows(output / "traces.jsonl", traces)
        summary = {"status": "completed", "scope": "actual unique backend judgments plus six-cell policy replay",
            "plan_sha256": initial_plan_hashes[str(plan_dir / "experiment_config.json")], "input_hashes_unchanged": True,
            "question_count": len(prepared["queries"]), "family_count": len(documents),
            "record_count": len(records), "metrics": aggregate(records), "paired_differences": comparisons,
            "actual_requests": len(bounded.ledger["attempts"]),
            "actual_reported_cost_usd": bounded.ledger["actual_reported_cost_usd"],
            "resolved_models": bounded.ledger["resolved_models"],
            "elapsed_seconds": time.monotonic() - started, "limits": LIMITS,
            "answer_generation_performed": False, "independent_evaluation": False,
            "cache_savings_measured": False, "test_payload_read": False,
            "all_packs_within_budget": all(row["actual_evidence_tokens"] <= row["budget"] for row in records),
            "all_results_available": True, "I_C_same_policy": True}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        failure(output, "run", exc, started)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("plan")
    for name in ("prepared", "sidecar", "tokenizer", "output"):
        offline.add_argument(f"--{name}", required=True)
    execute = commands.add_parser("run")
    for name in ("plan", "key-file", "output"):
        execute.add_argument(f"--{name}", required=True)
    execute.add_argument("--proxy")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = plan(arguments) if arguments.command == "plan" else run(arguments)
    print(json.dumps({"status": result["status"],
        "predicted_reservations": result.get("predicted_reservations"),
        "actual_reported_cost_usd": result.get("actual_reported_cost_usd")}, indent=2))
