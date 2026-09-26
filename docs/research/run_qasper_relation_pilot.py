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
                "cache_key": client.object_hash({"endpoint": client.MODELS[backend]["endpoint"],
                    "payload": payload, "prompt_version": client.PROMPT_VERSION}),
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


def nonnegative_decimal(value):
    if type(value) not in (str, int, float) or len(str(value)) > 128:
        raise ValueError("invalid prior accounting amount")
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("invalid prior accounting amount")
    return amount


def accounting():
    return {"attempts": 0, "questions": 0, "input_allowance": 0, "output_allowance": 0,
            "reservation_usd": "0", "known_cost_usd": "0", "unknown_cost_attempts": 0}


def add_attempt(totals, record):
    for target, source in (("questions", "question_count"), ("input_allowance", "input_allowance"),
                           ("output_allowance", "output_allowance")):
        value = record[source]
        if type(value) is not int or value < 1:
            raise ValueError("invalid prior allowance")
        totals[target] += value
    totals["attempts"] += 1
    totals["reservation_usd"] = str(nonnegative_decimal(totals["reservation_usd"]) +
                                    nonnegative_decimal(record["reserved_usd"]))
    if record.get("actual_cost_usd") is None:
        totals["unknown_cost_attempts"] += 1
    else:
        cost = nonnegative_decimal(record["actual_cost_usd"])
        if cost > nonnegative_decimal(record["reserved_usd"]):
            raise ValueError("prior cost exceeds reservation")
        totals["known_cost_usd"] = str(nonnegative_decimal(totals["known_cost_usd"]) + cost)


def prior_request(path, record, prompt_version):
    """Validate saved request accounting using its own frozen route prices."""
    payload = read_json(path)
    backend = record["backend"]
    if backend not in client.MODELS or record["kind"] not in client.CRITERIA:
        raise ValueError("invalid prior backend or kind")
    ids = client.payload_task_ids(payload, backend)
    if (not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids) or set(ids) != set(record["task_ids"])
            or len(ids) != record["question_count"] or digest(path) != record["request_sha256"]
            or client.object_hash(payload) != record["request_sha256"]):
        raise ValueError("prior request identity or hash mismatch")
    key = client.object_hash({"endpoint": client.MODELS[backend]["endpoint"],
                              "payload": payload, "prompt_version": prompt_version})
    if record["cache_key"] != key:
        raise ValueError("prior endpoint/payload/prompt cache key mismatch")
    count = len(ids) if backend == "jev" else 1
    input_allowance = (len(client.canonical_bytes(payload)) + 2048) * count
    price = payload["provider"]["max_price"]
    expected = max(Decimal("0.005"), (Decimal(input_allowance) * nonnegative_decimal(price["prompt"])
        + Decimal(1024) * nonnegative_decimal(price["completion"])) / Decimal(1000000) * Decimal("1.5"))
    if (record["input_allowance"] != input_allowance or record["output_allowance"] != 1024
            or nonnegative_decimal(record["reserved_usd"]) != expected):
        raise ValueError("prior request reservation mismatch")
    return payload


def validate_reused_response(response, payload, record):
    backend, kind = record["backend"], record["kind"]
    if record["status"] != "completed":
        raise ValueError("only completed prior requests can be reused")
    if backend == "general":
        profiles = [name for name, config in client.GENERAL_PROFILES.items()
                    if config["id"] == payload.get("model")
                    and payload.get("provider", {}).get("only") == [config["provider"]]]
        allowed_models = set().union(*(client.GENERAL_RESPONSE_MODELS[name] for name in profiles))
        allowed_providers = {client.GENERAL_PROFILES[name]["provider"] for name in profiles}
    else:
        allowed_models, allowed_providers = client.RESPONSE_MODELS[backend], {client.MODELS[backend]["provider"]}
    model, provider, response_id = response.get("model"), response.get("provider"), response.get("id")
    if (model not in allowed_models or model != record.get("response_model")
            or not isinstance(response_id, str) or not response_id.strip()
            or response_id != record.get("response_id") or provider != record.get("provider")):
        raise ValueError("prior response model or identity mismatch")
    if provider is not None and (not isinstance(provider, str) or provider.casefold() not in allowed_providers):
        raise ValueError("prior response provider mismatch")
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage != record.get("usage"):
        raise ValueError("prior response usage mismatch")
    for field, alternate, allowance in (("input_tokens", "prompt_tokens", "input_allowance"),
                                       ("output_tokens", "completion_tokens", "output_allowance")):
        value = usage.get(field, usage.get(alternate))
        if type(value) is not int or not 0 <= value <= record[allowance] or value != record.get(field):
            raise ValueError("prior response token usage mismatch")
    cost = nonnegative_decimal(usage.get("cost"))
    if cost != nonnegative_decimal(record.get("actual_cost_usd")) or cost > nonnegative_decimal(record["reserved_usd"]):
        raise ValueError("prior response cost mismatch")
    labels = client.parse_labels(response, payload, backend, kind)
    if labels != record.get("labels"):
        raise ValueError("prior saved labels differ from reparsed response")
    return labels


def freeze_prior_run(schedule, batches, directory, *, bind, allow_reuse):
    """Validate one run independently; only the designated run supplies labels."""
    directory = Path(directory).resolve() / "provider_calls"
    ledger_path = bind(directory / "ledger.json")
    ledger = read_json(ledger_path)
    if ledger.get("prompt_version") != client.PROMPT_VERSION or ledger.get("automatic_retries") != 0:
        raise ValueError("prior run prompt or retry policy mismatch")
    by_batch = {batch["id"]: batch for batch in batches}
    by_key = {item["cache_key"]: item for item in schedule}
    if len(by_key) != len(schedule):
        raise ValueError("duplicate scheduled cache key")
    request_records, seen_keys, expected_files = {}, set(), {ledger_path}
    totals, reused = accounting(), {}
    for number, record in enumerate(ledger["attempts"], 1):
        if (record["attempt"] != number or record["cache_key"] in seen_keys
                or record["status"] not in ("completed", "halted", "in_flight")):
            raise ValueError("prior attempts are not uniquely numbered")
        seen_keys.add(record["cache_key"])
        request_path = bind(directory / f"request_{number:03d}.json")
        expected_files.add(request_path)
        payload = prior_request(request_path, record, ledger["prompt_version"])
        request_records[str(request_path)] = record
        add_attempt(totals, record)
        response_path = directory / f"response_{number:03d}.json"
        if response_path.exists():
            response_path = bind(response_path)
            expected_files.add(response_path)
        error_name = record.get("error_response_file")
        if error_name:
            if error_name != f"error_response_{number:03d}.json":
                raise ValueError("invalid prior diagnostic filename")
            expected_files.add(bind(directory / error_name))
        if record["status"] != "completed":
            continue  # Failures consume admission budget but never produce reusable labels.
        response_path = bind(response_path)
        validate_reused_response(read_json(response_path), payload, record)
        if not allow_reuse:
            continue
        key = record["cache_key"]
        if key not in by_key:
            raise ValueError("completed prior request has no exact match in new schedule")
        item = by_key[key]
        batch = by_batch[item["batch_id"]]
        fresh_payload = client.make_payload(batch["tasks"], batch["kind"], item["backend"])
        if (record["backend"] != item["backend"] or record["kind"] != item["kind"]
                or record["task_ids"] != item["task_ids"] or payload != fresh_payload
                or record["request_sha256"] != item["payload_sha256"]):
            raise ValueError("prior request differs from frozen new request")
        reused[key] = {"request_path": str(request_path),
            "response_path": str(response_path), "ledger_path": str(ledger_path),
            "attempt": number, "backend": record["backend"], "kind": record["kind"],
            "original_actual_cost_usd": record["actual_cost_usd"], "response_model": record["response_model"]}
    actual_files = {p.resolve() for pattern in ("request_*.json", "response_*.json", "error_response_*.json")
                    for p in directory.glob(pattern)} | {ledger_path}
    if actual_files != expected_files:
        raise ValueError("prior request/response files are not completely accounted")
    if (nonnegative_decimal(ledger["reservation_total_usd"]) != nonnegative_decimal(totals["reservation_usd"])
            or nonnegative_decimal(ledger["actual_reported_cost_usd"]) != nonnegative_decimal(totals["known_cost_usd"])):
        raise ValueError("prior ledger aggregate mismatch")
    return request_records, totals, reused


def normalize_prior_directories(value):
    if value is None:
        return []
    values = [value] if isinstance(value, (str, Path)) else value
    paths = [Path(item).resolve() for item in values]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate prior source directory")
    return [str(path) for path in paths]


def freeze_prior(schedule, batches, reuse_run=None, prior_diagnostic=None, prior_runs=None):
    """Bind explicit historical runs and diagnostics without repeating charges."""
    reuse = str(Path(reuse_run).resolve()) if reuse_run else None
    extras = normalize_prior_directories(prior_runs)
    diagnostics = normalize_prior_directories(prior_diagnostic)
    directories = normalize_prior_directories(([reuse] if reuse else []) + extras)
    if set(map(Path, directories)) & set(map(Path, diagnostics)):
        raise ValueError("run and diagnostic cannot share a prior source directory")
    if diagnostics and not directories:
        raise ValueError("prior diagnostic requires the original run")
    prior = {"reuse_run": reuse, "prior_runs": extras, "prior_diagnostic": diagnostics,
             "input_sha256": {}, "accounting": accounting(), "reused_responses": {}}
    bound, totals, request_records = prior["input_sha256"], prior["accounting"], {}
    def bind(path):
        path = Path(path).resolve()
        bound[str(path)] = digest(path)
        return path
    for directory in directories:
        requests, run_totals, reused = freeze_prior_run(schedule, batches, directory,
            bind=bind, allow_reuse=directory == reuse)
        if set(requests) & set(request_records):
            raise ValueError("duplicate historical request source")
        request_records.update(requests)
        prior["reused_responses"].update(reused)
        for name in ("attempts", "questions", "input_allowance", "output_allowance", "unknown_cost_attempts"):
            totals[name] += run_totals[name]
        for name in ("reservation_usd", "known_cost_usd"):
            totals[name] = str(nonnegative_decimal(totals[name]) + nonnegative_decimal(run_totals[name]))
    for directory in diagnostics:
        diagnostic_path = bind(Path(directory) / "ledger.json")
        diagnostic = read_json(diagnostic_path)
        response = read_json(bind(Path(directory) / "response.json"))
        request_path = Path(diagnostic["request_path"]).resolve()
        if str(request_path) not in request_records:
            raise ValueError("diagnostic is not bound to a prior request")
        original = request_records[str(request_path)]
        if original["status"] == "completed":
            raise ValueError("diagnostic must identify the rejected prior request")
        if (type(diagnostic.get("http_status")) is not int or not 400 <= diagnostic["http_status"] <= 599
                or response.get("error", {}).get("code") != diagnostic["http_status"]):
            raise ValueError("diagnostic rejection status mismatch")
        for field in ("reserved_usd", "input_allowance", "output_allowance"):
            if diagnostic[field] != original[field]:
                raise ValueError("diagnostic reservation differs from original request")
        reported = response.get("usage", {}).get("cost")
        if ((reported is None) != (diagnostic.get("actual_cost_usd") is None)
                or reported is not None and nonnegative_decimal(reported) != nonnegative_decimal(diagnostic["actual_cost_usd"])):
            raise ValueError("diagnostic reported cost mismatch")
        add_attempt(totals, {**original, **diagnostic, "question_count": original["question_count"]})
    verify_hashes(bound)
    return prior


def cumulative_prediction(schedule, prior):
    reused = prior["reused_responses"]
    new = accounting()
    for item in schedule:
        if item["cache_key"] not in reused:
            add_attempt(new, {**item, "question_count": len(item["task_ids"]), "actual_cost_usd": "0"})
    old = prior["accounting"]
    total = {name: old[name] + new[name] for name in ("attempts", "questions", "input_allowance", "output_allowance")}
    total["reservation_usd"] = str(nonnegative_decimal(old["reservation_usd"]) + nonnegative_decimal(new["reservation_usd"]))
    if (total["attempts"] > CAPS["request_cap"] or total["questions"] > CAPS["question_cap"]
            or total["input_allowance"] > CAPS["input_allowance_cap"]
            or nonnegative_decimal(total["reservation_usd"]) > Decimal(CAPS["budget_usd"])):
        raise ValueError("cumulative prior plus new prediction exceeds pilot caps")
    return {"new_requests": new["attempts"], "new_questions": new["questions"],
            "new_input_allowance": new["input_allowance"], "new_reservation_usd": new["reservation_usd"],
            "reused_count": len(reused), "prior": old, "cumulative": total}


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
        general_profile = getattr(args, "general_profile", "gpt41mini")
        client.select_general_profile(general_profile)
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
        prior = freeze_prior(schedule, batches, getattr(args, "reuse_run", None),
                             getattr(args, "prior_diagnostic", None), getattr(args, "prior_run", None))
        cumulative = cumulative_prediction(schedule, prior)
        inputs.update(prior["input_sha256"])
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
            "models": json.loads(client.canonical_bytes(client.MODELS)), "prompt_version": client.PROMPT_VERSION,
            "general_profile": general_profile, "prior_execution": prior,
            "cumulative_prediction": cumulative,
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
    client.select_general_profile(config["general_profile"])
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
    saved_prior = config["prior_execution"]
    prior = freeze_prior(schedule, batches, saved_prior["reuse_run"], saved_prior["prior_diagnostic"],
                         saved_prior["prior_runs"])
    if prior != saved_prior or cumulative_prediction(schedule, prior) != config["cumulative_prediction"]:
        raise ValueError("prior response provenance or cumulative budget changed")
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
        prior = config["prior_execution"]
        used = prior["accounting"]
        reused_events = []
        if config["cumulative_prediction"]["new_requests"]:
            bounded = client_factory(output / "provider_calls", key_file=args.key_file, proxy=args.proxy,
                budget_usd=str(Decimal(CAPS["budget_usd"]) - Decimal(used["reservation_usd"])),
                request_cap=CAPS["request_cap"] - used["attempts"],
                question_cap=CAPS["question_cap"] - used["questions"],
                token_cap=CAPS["input_allowance_cap"] - used["input_allowance"])
        for index, item in enumerate(config["schedule"]):
            # Provider transport has a 60-second timeout. Do not start a call
            # that could knowingly exceed the overall 30-minute run window.
            if deadline - time.monotonic() <= 65:
                raise TimeoutError("insufficient remaining provider time allowance")
            batch = by_batch[item["batch_id"]]
            provenance = prior["reused_responses"].get(item["cache_key"])
            if provenance:
                verify_hashes(prior["input_sha256"])
                record = read_json(provenance["ledger_path"])["attempts"][provenance["attempt"] - 1]
                payload = client.make_payload(batch["tasks"], batch["kind"], item["backend"])
                decisions = validate_reused_response(read_json(provenance["response_path"]), payload, record)
                reused_events.append({"schedule_index": index, "cache_key": item["cache_key"],
                    "source": provenance, "new_api_call": False,
                    "accounting": "original known cost retained once in cumulative prior cost"})
                write_json(output / f"reuse_{len(reused_events):03d}.json", reused_events[-1])
            else:
                decisions = bounded.submit(batch["tasks"], batch["kind"], item["backend"])
                for source in prior["reused_responses"].values():
                    if (source["backend"] == item["backend"]
                            and bounded.ledger["resolved_models"][item["backend"]] != source["response_model"]):
                        raise ValueError("reused and new response model identity differs")
            if set(decisions) != set(item["task_ids"]) or set(decisions) & set(labels[item["backend"]]):
                raise ValueError("provider result identities differ from frozen schedule")
            labels[item["backend"]].update(decisions)
            temporary = output / "labels.tmp"
            temporary.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(output / "labels.json")
            print(json.dumps({"completed_requests": index + 1, "total_requests": len(config["schedule"]),
                "new_api_calls": len(bounded.ledger["attempts"]) if bounded else 0,
                "reused_count": len(reused_events),
                "new_reported_cost_usd": bounded.ledger["actual_reported_cost_usd"] if bounded else "0"}), flush=True)
            check_time(deadline)
        records, traces = replay_records(prepared, documents, annotations, labels, tokenizer, deadline)
        comparisons = paired_differences(records)
        verify_hashes(config["input_sha256"])
        # Bind to the bytes seen at run start, not a potentially replaced seal.
        verify_hashes(initial_plan_hashes)
        verify_hashes({str(plan_dir / name): value for name, value in config["plan_files_sha256"].items()})
        check_time(deadline)
        new_ledger = bounded.ledger if bounded else {"attempts": [], "actual_reported_cost_usd": "0", "resolved_models": {}}
        cumulative_usage = dict(used)
        for attempt in new_ledger["attempts"]:
            add_attempt(cumulative_usage, attempt)
        resolved = dict(new_ledger["resolved_models"])
        for value in prior["reused_responses"].values():
            if resolved.setdefault(value["backend"], value["response_model"]) != value["response_model"]:
                raise ValueError("reused and new response model identity differs")
        verify_hashes(prior["input_sha256"])
        write_rows(output / "per_question.jsonl", records)
        write_rows(output / "traces.jsonl", traces)
        summary = {"status": "completed", "scope": "actual unique backend judgments plus six-cell policy replay",
            "plan_sha256": initial_plan_hashes[str(plan_dir / "experiment_config.json")], "input_hashes_unchanged": True,
            "question_count": len(prepared["queries"]), "family_count": len(documents),
            "record_count": len(records), "metrics": aggregate(records), "paired_differences": comparisons,
            "actual_requests": len(new_ledger["attempts"]), "new_api_calls": len(new_ledger["attempts"]),
            "reused_count": len(reused_events), "prior_attempts": used["attempts"],
            "actual_reported_cost_usd": new_ledger["actual_reported_cost_usd"],
            "cumulative_accounting": cumulative_usage,
            "cumulative_known_cost_usd": cumulative_usage["known_cost_usd"],
            "cumulative_unknown_cost_attempts": cumulative_usage["unknown_cost_attempts"],
            "prior_input_sha256": prior["input_sha256"],
            "resolved_models": resolved, "general_profile": config["general_profile"],
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
    offline.add_argument("--general-profile", choices=tuple(client.GENERAL_PROFILES), default="gpt41mini")
    offline.add_argument("--reuse-run")
    offline.add_argument("--prior-run", action="append", default=[])
    offline.add_argument("--prior-diagnostic", action="append", default=[])
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
