"""Freeze, execute and audit support-only extended development judgments.

This is a new stage. The sealed 15-question pilot is never edited. Provider
clients are bounded segments of one stage budget, not independent allowances.
Only an explicitly requested, verified completed prefix can be reused. Halted
or uncertain provider attempts are never retried by this runner.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time

import analyze_qasper_jev_probability as probability
import analyze_qasper_relation_pilot as pilot_audit
import openrouter_decision_client as client
import prepare_qasper_extended_development as preparation
import run_qasper_relation_pilot as pilot
from run_qasper_evidence_baselines import PackCounter, aggregate, digest, pack_ranked, score_selection


SCHEMA = "slac-qasper-extended-support-plan-v1"
PROFILE = "qwen36plus-json"
CAPS = {"night_budget_usd": "5", "support_budget_usd": "4", "generation_minimum_reserve_usd": "1",
        "request_cap": 512, "question_cap": 2600, "input_allowance_cap": 40000000,
        "max_seconds": 5400, "stop_at_utc": "2026-09-27T01:00:00+00:00"}
SEGMENT_CAPS = {"budget_usd": "2", "request_cap": 160, "question_cap": 1600,
                "input_allowance_cap": 8000000}
SELECTOR = {"budget": 1024, "sizes": [1, 2, 3], "score_contract": "reported-scores",
            "score_sum_range": [0.985, 1.015], "score_rules": list(probability.RULES),
            "no_excluded": True, "gold_used_for_selection": False}
METHODS = tuple(f"{base}_k{k}" for k in SELECTOR["sizes"]
                for base in ("dense", "I_jev", "I_general", *probability.RULES))
LIMITS = [
    "The 77 questions are exposed development data, not independent confirmation.",
    "Support-only fixed-pool selection does not test a shared relation architecture.",
    "All k=1/2/3 and both fixed raw-score rules are retained; no best setting replaces the pilot.",
    "Reported JEV values are unnormalized raw scores, not calibrated probabilities.",
    "Equal evidence caps do not imply equal actual evidence lengths.",
    "Reservations are conservative local admission accounting, not a provider-enforced account cap.",
    "Historical pilot accounting is retained separately from this new stage's budget.",
    "No answer generation, training, measured cache savings or test QA reading occurs here.",
]


def read(path):
    return pilot.read_json(path)


def amount(value):
    return pilot.nonnegative_decimal(value)


def utc_now():
    return datetime.now(timezone.utc)


def ensure_admission_time(started):
    # The fixed provider timeout is 60 seconds. Do not knowingly start a call
    # that could overrun either the stage limit or the user's 09:00 deadline.
    remaining = (datetime.fromisoformat(CAPS["stop_at_utc"]) - utc_now()).total_seconds()
    if remaining <= 65 or time.monotonic() - started >= CAPS["max_seconds"] - 65:
        raise TimeoutError("stage or overnight deadline reached before another request")


def freeze_batches(prepared):
    """Common backend batches, by query and original support item order."""
    groups, seen = defaultdict(list), set()
    for task in prepared["support_tasks"]:
        if task["id"] in seen:
            raise ValueError("duplicate support task")
        seen.add(task["id"])
        client.validate_item(task["item"], "support")
        groups[(task["doc_id"], task["question_id"])].append({"id": task["id"], "item": task["item"]})
    if set(groups) != {(q["doc_id"], q["question_id"]) for q in prepared["queries"]}:
        raise ValueError("support group coverage differs from frozen queries")
    batches = []
    for query in prepared["queries"]:
        key = (query["doc_id"], query["question_id"])
        for tasks in client.task_batches(groups[key], "support"):
            batch = {"kind": "support", "group": list(key), "tasks": tasks}
            batches.append({**batch, "id": "batch-" + client.object_hash(batch)})
    return batches


def totals(schedule):
    return {"requests": len(schedule), "questions": sum(len(x["task_ids"]) for x in schedule),
            "input_allowance": sum(x["input_allowance"] for x in schedule),
            "output_allowance": sum(x["output_allowance"] for x in schedule),
            "reservation_usd": str(sum((amount(x["reserved_usd"]) for x in schedule), Decimal("0")))}


def schedule_requests(batches):
    schedule = []
    for batch_index, batch in enumerate(batches):
        for backend in (("jev", "general") if batch_index % 2 == 0 else ("general", "jev")):
            payload = client.make_payload(batch["tasks"], "support", backend)
            reserved, tokens_in, tokens_out = client.reservation(payload, backend)
            schedule.append({"schedule_index": len(schedule), "batch_id": batch["id"], "backend": backend,
                "kind": "support", "task_ids": [task["id"] for task in batch["tasks"]],
                "payload_sha256": client.object_hash(payload),
                "cache_key": client.object_hash({"endpoint": client.MODELS[backend]["endpoint"],
                    "payload": payload, "prompt_version": client.PROMPT_VERSION}),
                "reserved_usd": str(reserved), "input_allowance": tokens_in, "output_allowance": tokens_out})
    if len({x["cache_key"] for x in schedule}) != len(schedule):
        raise ValueError("duplicate scheduled request")
    prediction = totals(schedule)
    if (prediction["requests"] > CAPS["request_cap"] or prediction["questions"] > CAPS["question_cap"]
            or prediction["input_allowance"] > CAPS["input_allowance_cap"]
            or amount(prediction["reservation_usd"]) > amount(CAPS["support_budget_usd"])):
        raise ValueError("support prediction exceeds stage cap; no calls admitted")
    return schedule, prediction


def segment_schedule(schedule):
    """Split only transport accounting; request payloads/order never change."""
    segments, pending = [], []
    for request in schedule:
        proposed = [*pending, request]
        value = totals(proposed)
        fits = (value["requests"] <= SEGMENT_CAPS["request_cap"]
                and value["questions"] <= SEGMENT_CAPS["question_cap"]
                and value["input_allowance"] <= SEGMENT_CAPS["input_allowance_cap"]
                and amount(value["reservation_usd"]) <= amount(SEGMENT_CAPS["budget_usd"]))
        if not fits:
            if not pending:
                raise ValueError("one frozen request exceeds a client segment cap")
            segments.append(pending)
            pending = [request]
            if len(segment_schedule(pending)) != 1:
                raise ValueError("invalid single-request segment")
        else:
            pending = proposed
    if pending:
        segments.append(pending)
    return [{"segment": i + 1, "start": items[0]["schedule_index"],
             "stop": items[-1]["schedule_index"] + 1, **totals(items)} for i, items in enumerate(segments)]


def historical_snapshot(path):
    path = Path(path).resolve()
    value = read(path)
    if (value.get("status") != "completed" or not value.get("all_results_available")
            or not isinstance(value.get("cumulative_accounting"), dict)):
        raise ValueError("historical pilot summary must be complete")
    accounting = value["cumulative_accounting"]
    for name in ("reservation_usd", "known_cost_usd"):
        amount(accounting[name])
    for name in ("attempts", "unknown_cost_attempts"):
        if type(accounting[name]) is not int or accounting[name] < 0:
            raise ValueError("invalid historical accounting")
    return {"summary_path": str(path), "summary_sha256": digest(path), "accounting": accounting,
            "scope": "Previously completed pilot summary; preserved separately, not charged to the new stage cap."}


def baseline_records(prepared, documents, annotations, tokenizer):
    result = []
    for q in prepared["queries"]:
        units = documents[q["doc_id"]]
        by_id = {unit.unit_id: i for i, unit in enumerate(units)}
        count = PackCounter(tokenizer, units)
        for k in SELECTOR["sizes"]:
            chosen = pack_ranked(units, [by_id[uid] for uid in q["ranked_ids"]], 1024, count, max_units=k)
            result.append({**{name: q[name] for name in pilot_audit.IDENTITY}, "method": f"dense_k{k}", "budget": 1024,
                **score_selection(units, chosen, annotations[(q["doc_id"], q["question_id"])], count, 1024)})
    return result


def plan(args):
    output, prepared_dir, sidecar, tokenizer_path = (Path(getattr(args, name)).resolve()
        for name in ("output", "prepared", "sidecar", "tokenizer"))
    if output.exists():
        raise FileExistsError("plan output already exists")
    client.select_general_profile(PROFILE)
    prepared, manifest, documents = preparation.load_prepared(prepared_dir)
    inputs = {str(Path(path).resolve()): value for path, value in manifest["input_sha256"].items()}
    for path in (sidecar, *pilot.tokenizer_files(tokenizer_path)):
        if inputs.get(str(path)) != digest(path):
            raise ValueError("QA sidecar/tokenizer differs from prepared lineage")
    code_files = {Path(__file__).resolve(), Path(preparation.__file__).resolve(), Path(probability.__file__).resolve(),
                  Path(pilot_audit.__file__).resolve(), *(Path(pilot.__file__).parent / name for name in pilot.CODE_FILES)}
    paths = [prepared_dir / "prepared.json", prepared_dir / "manifest.json", *code_files]
    inputs.update({str(path): digest(path) for path in paths})
    historical = historical_snapshot(args.historical_summary)
    inputs[historical["summary_path"]] = historical["summary_sha256"]
    batches = freeze_batches(prepared)
    schedule, prediction = schedule_requests(batches)
    segments = segment_schedule(schedule)
    annotations = pilot.selected_gold(sidecar, prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    baseline = baseline_records(prepared, documents, annotations, tokenizer)
    pilot.verify_hashes(inputs)
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "batches.json", batches)
    pilot.write_rows(output / "baseline_per_question.jsonl", baseline)
    pilot.write_json(output / "baseline_summary.json", {"status": "completed_offline", "api_calls": 0,
        "metrics": aggregate(baseline), "gold_used_for_selection": False})
    config = {"schema": SCHEMA, "status": "planned", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(prepared_dir), "sidecar": str(sidecar), "tokenizer": str(tokenizer_path),
        "input_sha256": inputs, "plan_files_sha256": {name: digest(output / name) for name in (
            "batches.json", "baseline_per_question.jsonl", "baseline_summary.json")},
        "general_profile": PROFILE, "models": json.loads(client.canonical_bytes(client.MODELS)),
        "response_model_allowlists": {key: sorted(value) for key, value in client.RESPONSE_MODELS.items()},
        "prompt_version": client.PROMPT_VERSION, "caps": CAPS, "segment_caps": SEGMENT_CAPS,
        "selector": SELECTOR, "schedule": schedule, "segments": segments, "predicted_reservations": prediction,
        "generation_reservation_remaining_usd": str(amount(CAPS["night_budget_usd"]) - amount(prediction["reservation_usd"])),
        "historical_pilot": historical, "question_count": len(prepared["queries"]),
        "family_count": len({q["family_id"] for q in prepared["queries"]}), "methods": list(METHODS),
        "batching": "same query grouped for both backends; <=8 items and <=24000-byte payloads; alternate backend order by batch",
        "score_contract": "reported-scores; all support values required; invalid values halt the complete ranking experiment",
        "api_calls": 0, "test_payload_read": False, "static_api_calls_planned": 0,
        "answer_generation_performed": False, "independent_confirmation": False, "limits": LIMITS}
    pilot.write_json(output / "experiment_config.json", config)
    pilot.write_json(output / "plan_manifest.json", {"status": "planned", "experiment_config_sha256": digest(output / "experiment_config.json")})
    return config


def load_plan(directory):
    directory = Path(directory).resolve()
    seal = read(directory / "plan_manifest.json")
    if digest(directory / "experiment_config.json") != seal["experiment_config_sha256"]:
        raise ValueError("stage plan seal mismatch")
    config = read(directory / "experiment_config.json")
    client.select_general_profile(PROFILE)
    if (config.get("schema") != SCHEMA or config.get("status") != "planned"
            or config["general_profile"] != PROFILE or config["caps"] != CAPS
            or config["segment_caps"] != SEGMENT_CAPS or config["selector"] != SELECTOR
            or config["models"] != client.MODELS or config["prompt_version"] != client.PROMPT_VERSION
            or config["response_model_allowlists"] != {key: sorted(value) for key, value in client.RESPONSE_MODELS.items()}):
        raise ValueError("stage plan differs from frozen implementation")
    pilot.verify_hashes(config["input_sha256"])
    pilot.verify_hashes({str(directory / name): value for name, value in config["plan_files_sha256"].items()})
    prepared, _, _ = preparation.load_prepared(config["prepared_dir"])
    batches = read(directory / "batches.json")
    if batches != freeze_batches(prepared):
        raise ValueError("batches differ from prepared support items")
    schedule, prediction = schedule_requests(batches)
    if (config["schedule"] != schedule or config["predicted_reservations"] != prediction
            or config["segments"] != segment_schedule(schedule)
            or config["historical_pilot"] != historical_snapshot(config["historical_pilot"]["summary_path"])
            or config["question_count"] != len(prepared["queries"])
            or config["family_count"] != len({q["family_id"] for q in prepared["queries"]})
            or config["methods"] != list(METHODS)):
        raise ValueError("stage prediction or prepared identity mismatch")
    return config, batches


def support_scores(response, task_ids):
    if set(response.get("answers", {})) != set(task_ids):
        raise ValueError("JEV raw-score task coverage mismatch")
    scores = {key: answer.get("probabilities") for key, answer in response["answers"].items()}
    if any(probability.validate_probabilities(value, "reported-scores") for value in scores.values()):
        raise ValueError("JEV returned scores outside frozen reported-scores contract")
    return scores


def collect_execution(config, batches, run_dir, plan_sha256, *, require_complete=False, ancestry=()):
    """Verify a completed prefix. Any failed/in-flight provider attempt refuses reuse."""
    run_dir = Path(run_dir).resolve()
    if str(run_dir) in ancestry or len(ancestry) >= 8:
        raise ValueError("cyclic or excessive stage resume lineage")
    hashes = {}

    def checked(path):
        path = Path(path).resolve()
        current = digest(path)
        if str(path) in hashes and hashes[str(path)] != current:
            raise ValueError("execution source changed while reading")
        hashes[str(path)] = current
        return path

    manifest = read(checked(run_dir / "run_manifest.json"))
    if manifest["plan_sha256"] != plan_sha256:
        raise ValueError("resume source belongs to a different stage plan")
    labels, scores, accounting, resolved, completed = {b: {} for b in client.MODELS}, {}, pilot.accounting(), {}, 0
    if manifest["parent_run"]:
        pilot.verify_hashes(manifest["parent_input_sha256"])
        prior = collect_execution(config, batches, manifest["parent_run"], plan_sha256,
                                  ancestry=(*ancestry, str(run_dir)))
        if prior["input_sha256"] != manifest["parent_input_sha256"]:
            raise ValueError("resume source inventory changed")
        labels, scores, accounting, resolved, completed = (prior[key] for key in (
            "labels", "scores", "accounting", "resolved_models", "completed_requests"))
        hashes.update(prior["input_sha256"])
    elif manifest["parent_input_sha256"]:
        raise ValueError("unexpected resume bindings without a parent run")
    expected_segments = segment_schedule(config["schedule"][completed:])
    if manifest["segments"] != expected_segments or manifest["reused_prefix_requests"] != completed:
        raise ValueError("execution segments differ from frozen remaining schedule")
    by_batch = {batch["id"]: batch for batch in batches}
    saw_incomplete = False
    present = set()
    for segment in expected_segments:
        call_dir = run_dir / "provider_calls" / f"segment_{segment['segment']:03d}"
        if not call_dir.exists():
            saw_incomplete = True
            continue
        if saw_incomplete:
            raise ValueError("execution is not a contiguous scheduled prefix")
        present.add(call_dir.name)
        ledger = read(checked(call_dir / "ledger.json"))
        expected_caps = {"budget_usd": segment["reservation_usd"], "request_cap": segment["requests"],
                         "question_cap": segment["questions"], "conservative_input_token_cap": segment["input_allowance"]}
        if any(ledger.get(key) != value for key, value in expected_caps.items()):
            raise ValueError("provider segment cap mismatch")
        if ledger["halt_reason"] is not None or any(a["status"] != "completed" for a in ledger["attempts"]):
            raise ValueError("failed or uncertain provider attempt cannot be retried automatically")
        expected_files = {"ledger.json"}
        local_reserve, local_cost = Decimal("0"), Decimal("0")
        for local_index, attempt in enumerate(ledger["attempts"], start=1):
            if completed >= segment["stop"] or attempt["attempt"] != local_index:
                raise ValueError("segment request overflow or ordinal mismatch")
            item = config["schedule"][completed]
            request_path = call_dir / f"request_{local_index:03d}.json"
            response_path = call_dir / f"response_{local_index:03d}.json"
            expected_files.update((request_path.name, response_path.name))
            payload = pilot.prior_request(checked(request_path), attempt, config["prompt_version"])
            response = read(checked(response_path))
            batch = by_batch[item["batch_id"]]
            if (attempt["cache_key"] != item["cache_key"] or attempt["backend"] != item["backend"]
                    or attempt["kind"] != "support" or attempt["task_ids"] != item["task_ids"]
                    or client.object_hash(payload) != item["payload_sha256"]
                    or payload != client.make_payload(batch["tasks"], "support", item["backend"])):
                raise ValueError("saved request differs from the exact frozen schedule")
            parsed = pilot.validate_reused_response(response, payload, attempt)
            backend = item["backend"]
            if set(parsed) & set(labels[backend]):
                raise ValueError("duplicate executed support task")
            model = attempt["response_model"]
            if resolved.setdefault(backend, model) != model:
                raise ValueError("model changed across provider segments or resume sources")
            labels[backend].update(parsed)
            if backend == "jev":
                scores.update(support_scores(response, item["task_ids"]))
            pilot.add_attempt(accounting, attempt)
            local_reserve += amount(attempt["reserved_usd"])
            local_cost += amount(attempt["actual_cost_usd"])
            completed += 1
        if ({p.name for p in call_dir.iterdir()} != expected_files
                or amount(ledger["reservation_total_usd"]) != local_reserve
                or amount(ledger["actual_reported_cost_usd"]) != local_cost):
            raise ValueError("unaccounted provider artifact or segment accounting mismatch")
        if completed != segment["stop"]:
            saw_incomplete = True
    parent = run_dir / "provider_calls"
    if parent.exists() and {p.name for p in parent.iterdir()} != present:
        raise ValueError("unexpected provider segment")
    if require_complete and completed != len(config["schedule"]):
        raise ValueError("stage execution is incomplete")
    if amount(accounting["reservation_usd"]) > amount(CAPS["support_budget_usd"]):
        raise ValueError("cumulative stage reservation exceeds support cap")
    pilot.verify_hashes(hashes)
    return {"labels": labels, "scores": scores, "accounting": accounting, "resolved_models": resolved,
            "completed_requests": completed, "input_sha256": hashes}


def replay_records(prepared, documents, annotations, labels, scores, tokenizer):
    expected = {task["id"] for task in prepared["support_tasks"]}
    if (set(labels) != set(client.MODELS) or any(set(value) != expected for value in labels.values())
            or set(scores) != expected):
        raise ValueError("incomplete judgment coverage; no partial main scores")
    records = baseline_records(prepared, documents, annotations, tokenizer)
    traces = []
    lookup = {(t["doc_id"], t["question_id"], t["unit_id"]): t["id"] for t in prepared["support_tasks"]}
    for q in prepared["queries"]:
        doc, qid = q["doc_id"], q["question_id"]
        units, gold = documents[doc], annotations[(doc, qid)]
        count = PackCounter(tokenizer, units)
        identity = {name: q[name] for name in pilot_audit.IDENTITY}
        for backend in client.MODELS:
            relevance = {uid: labels[backend][lookup[(doc, qid, uid)]] for uid in q["candidate_ids"]}
            rankings = {}
            if backend == "jev":
                values = {uid: scores[lookup[(doc, qid, uid)]] for uid in q["candidate_ids"]}
                rankings = {rule: probability.probability_ranking(units, q["candidate_ids"], relevance, values,
                    q["ranked_ids"], rule, "reported-scores") for rule in probability.RULES}
            for k in SELECTOR["sizes"]:
                coarse = pilot.replay_policy(units, q["candidate_ids"], relevance, [], q["ranked_ids"],
                    mode="I", tokenizer=tokenizer, budget=1024, chunk_budget=384, max_units=k)
                choices = {f"I_{backend}_k{k}": coarse["selected_indices"]}
                choices.update({f"{rule}_k{k}": pack_ranked(units, ranking, 1024, count, max_units=k)
                                for rule, ranking in rankings.items()})
                for method, chosen in choices.items():
                    records.append({**identity, "method": method, "budget": 1024,
                                    **score_selection(units, chosen, gold, count, 1024)})
                    traces.append({**identity, "method": method, "max_units": k, "budget": 1024,
                        "candidate_information_sha256": coarse["candidate_information_sha256"],
                        "retrieval_ranking": q["ranked_ids"], "selected_ids": [units[i].unit_id for i in chosen],
                        "excluded_no_ids": coarse["excluded_no_ids"], "gold_used_for_selection": False})
    return records, traces


def summarize(prepared, records):
    questions = [tuple(q[name] for name in pilot_audit.IDENTITY) for q in prepared["queries"]]
    indexed = pilot_audit.index_rows(records, METHODS, questions)
    tables = {method: {key: indexed[(method, *key)] for key in questions} for method in METHODS}
    comparisons = []
    for k in SELECTOR["sizes"]:
        pairs = [(f"I_jev_k{k}", f"dense_k{k}"), (f"I_general_k{k}", f"dense_k{k}"),
                 (f"I_jev_k{k}", f"I_general_k{k}")]
        pairs.extend((f"{rule}_k{k}", f"{baseline}_k{k}")
                     for rule in probability.RULES for baseline in ("I_jev", "I_general", "dense"))
        for plus, minus in pairs:
            comparisons.append({"plus": plus, "minus": minus, **pilot_audit.compare(tables[plus], tables[minus], questions)})
    return {"record_count": len(records), "method_count": len(METHODS), "metrics": aggregate(records),
            "paired_comparisons": comparisons}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def attempted_accounting(run_dir, prior):
    result = dict(prior)
    result["cost_over_reservation_attempts"] = 0
    for path in sorted((Path(run_dir) / "provider_calls").glob("segment_*/ledger.json")):
        for attempt in read(path)["attempts"]:
            # Even the very failure being diagnosed (provider cost above the
            # reservation) must remain visible instead of breaking this report.
            result["attempts"] += 1
            for target, source in (("questions", "question_count"), ("input_allowance", "input_allowance"),
                                   ("output_allowance", "output_allowance")):
                result[target] += attempt[source]
            result["reservation_usd"] = str(amount(result["reservation_usd"]) + amount(attempt["reserved_usd"]))
            if attempt.get("actual_cost_usd") is None:
                result["unknown_cost_attempts"] += 1
            else:
                cost = amount(attempt["actual_cost_usd"])
                result["known_cost_usd"] = str(amount(result["known_cost_usd"]) + cost)
                result["cost_over_reservation_attempts"] += cost > amount(attempt["reserved_usd"])
    return result


def register_run(plan_dir, output, parent_run, plan_sha):
    """One continuation chain per plan; fresh output cannot reset spent budget.

    The lock is deliberately not auto-cleared after process death. A stale lock
    requires inspection of the registered run before an operator removes it.
    """
    lock = plan_dir / "execution.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    registry_path = plan_dir / "execution_registry.json"
    try:
        if registry_path.exists():
            registry = read(registry_path)
            if (registry["plan_sha256"] != plan_sha or not parent_run
                    or str(Path(parent_run).resolve()) != registry["runs"][-1]):
                raise ValueError("stage already started; use explicit resume from its latest verified prefix")
        else:
            if parent_run:
                raise ValueError("unregistered resume source")
            registry = {"plan_sha256": plan_sha, "runs": []}
        if str(output) in registry["runs"]:
            raise ValueError("run output was already registered")
        registry["runs"].append(str(output))
        atomic_json(registry_path, registry)
    except BaseException:
        lock.unlink()
        raise
    return lock


def run(args, *, client_factory=client.BoundedClient):
    started = time.monotonic()
    output, plan_dir = Path(args.output).resolve(), Path(args.plan).resolve()
    if output.exists():
        raise FileExistsError("run output already exists")
    config, batches = load_plan(plan_dir)
    plan_sha = digest(plan_dir / "experiment_config.json")
    parent_run = getattr(args, "resume_from", None)
    prior = collect_execution(config, batches, parent_run, plan_sha) if parent_run else {
        "labels": {b: {} for b in client.MODELS}, "scores": {}, "accounting": pilot.accounting(),
        "resolved_models": {}, "completed_requests": 0, "input_sha256": {}}
    completed, resolved = prior["completed_requests"], dict(prior["resolved_models"])
    segments = segment_schedule(config["schedule"][completed:])
    lock = register_run(plan_dir, output, parent_run, plan_sha)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "slac-qasper-extended-support-run-v1", "plan_sha256": plan_sha,
        "parent_run": str(Path(parent_run).resolve()) if parent_run else None,
        "parent_input_sha256": prior["input_sha256"], "reused_prefix_requests": completed, "segments": segments}
    pilot.write_json(output / "run_manifest.json", manifest)
    by_batch = {batch["id"]: batch for batch in batches}
    admitted = amount(prior["accounting"]["reservation_usd"])
    new_accounting = pilot.accounting()
    active_client = None
    stage = "provider_calls"
    try:
        for segment in segments:
            ensure_admission_time(started)
            active_client = client_factory(output / "provider_calls" / f"segment_{segment['segment']:03d}",
                key_file=args.key_file, proxy=getattr(args, "proxy", None), budget_usd=segment["reservation_usd"],
                request_cap=segment["requests"], question_cap=segment["questions"], token_cap=segment["input_allowance"])
            for item in config["schedule"][segment["start"]:segment["stop"]]:
                ensure_admission_time(started)
                if admitted + amount(item["reserved_usd"]) > amount(CAPS["support_budget_usd"]):
                    raise ValueError("global support reservation exhausted before request")
                admitted += amount(item["reserved_usd"])
                batch = by_batch[item["batch_id"]]
                decisions = active_client.submit(batch["tasks"], "support", item["backend"])
                attempt = active_client.ledger["attempts"][-1]
                if attempt["cache_key"] != item["cache_key"] or set(decisions) != set(item["task_ids"]):
                    raise ValueError("provider returned a different frozen request")
                backend, model = item["backend"], attempt["response_model"]
                if resolved.setdefault(backend, model) != model:
                    raise ValueError("response model changed across stage segments")
                if backend == "jev":
                    response = read(active_client.output / f"response_{attempt['attempt']:03d}.json")
                    support_scores(response, item["task_ids"])
                pilot.add_attempt(new_accounting, attempt)
                completed += 1
                progress = {"completed_requests": completed, "total_requests": len(config["schedule"]),
                    "new_api_calls": new_accounting["attempts"], "reused_prefix_requests": prior["completed_requests"],
                    "stage_reserved_usd": str(admitted), "new_reported_cost_usd": new_accounting["known_cost_usd"]}
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            active_client = None
        stage = "verify_and_score"
        verified = collect_execution(config, batches, output, plan_sha, require_complete=True)
        prepared, _, documents = preparation.load_prepared(config["prepared_dir"])
        annotations = pilot.selected_gold(config["sidecar"], prepared)
        tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
        records, traces = replay_records(prepared, documents, annotations, verified["labels"], verified["scores"], tokenizer)
        baseline = pilot_audit.read_rows(plan_dir / "baseline_per_question.jsonl")
        if [row for row in records if row["method"].startswith("dense_")] != baseline:
            raise ValueError("offline dense baselines changed")
        result = summarize(prepared, records)
        load_plan(plan_dir)
        pilot.verify_hashes(verified["input_sha256"])
        if digest(plan_dir / "experiment_config.json") != plan_sha:
            raise ValueError("plan changed during execution")
        score_values = list(verified["scores"].values())
        result.update(status="completed", schema="slac-qasper-extended-support-result-v1", plan_sha256=plan_sha,
            all_results_available=True, question_count=len(prepared["queries"]), family_count=len(documents),
            new_api_calls=len(config["schedule"]) - prior["completed_requests"], reused_prefix_requests=prior["completed_requests"],
            stage_accounting=verified["accounting"], historical_pilot=config["historical_pilot"],
            generation_reservation_remaining_usd=str(amount(CAPS["night_budget_usd"]) - amount(verified["accounting"]["reservation_usd"])),
            resolved_models=verified["resolved_models"], score_contract="reported-scores",
            raw_score_coverage={"tasks": len(score_values), "reported_scores_valid": len(score_values),
                "strict_distribution_invalid": sum(bool(probability.validate_probabilities(value)) for value in score_values),
                "renormalized": False, "confidence_used": False},
            input_binding_sha256=client.object_hash(verified["input_sha256"]), source_hashes_unchanged=True,
            answer_generation_performed=False, test_payload_read=False, independent_confirmation=False,
            elapsed_seconds=time.monotonic() - started, limits=LIMITS)
        pilot.write_json(output / "labels.json", verified["labels"])
        pilot.write_json(output / "raw_scores.json", verified["scores"])
        pilot.write_rows(output / "per_question.jsonl", records)
        pilot.write_rows(output / "traces.jsonl", traces)
        result["execution_input_sha256"] = verified["input_sha256"]
        result["output_files_sha256"] = {str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()}
        pilot.write_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        # A killed process may leave only the client ledger; explicit resume still
        # accepts solely a completely validated prefix and refuses in-flight calls.
        failed = {"status": "failed", "stage": stage, "error_class": type(exc).__name__,
                  "main_results_available": False, "completed_requests": completed,
                  "admitted_stage_reservation_usd": str(admitted), "automatic_retries": 0,
                  "prior_accounting": prior["accounting"], "new_completed_accounting": new_accounting}
        failed["stage_accounting"] = attempted_accounting(output, prior["accounting"])
        if active_client is not None:
            failed["active_segment_attempts"] = len(active_client.ledger["attempts"])
            failed["active_segment_reservation_usd"] = active_client.ledger["reservation_total_usd"]
            failed["active_segment_reported_cost_usd"] = active_client.ledger["actual_reported_cost_usd"]
            failed["active_segment_unknown_cost_attempts"] = sum("actual_cost_usd" not in row for row in active_client.ledger["attempts"])
        pilot.write_json(output / "failure.json", failed)
        raise
    finally:
        lock.unlink()


def verify_completed_run(plan_dir, run_dir):
    """Read-only generator prerequisite; returns config/prepared/docs/rows/summary."""
    plan_dir, run_dir = Path(plan_dir).resolve(), Path(run_dir).resolve()
    config, batches = load_plan(plan_dir)
    plan_sha = digest(plan_dir / "experiment_config.json")
    if (run_dir / "failure.json").exists():
        raise ValueError("failed stage has no complete main result")
    verified = collect_execution(config, batches, run_dir, plan_sha, require_complete=True)
    saved = read(run_dir / "summary.json")
    inventory = {str(path.relative_to(run_dir)): digest(path) for path in run_dir.rglob("*")
                 if path.is_file() and path != run_dir / "summary.json"}
    if (saved.get("output_files_sha256") != inventory
            or saved.get("execution_input_sha256") != verified["input_sha256"]):
        raise ValueError("completed stage output seal or response lineage differs")
    prepared, _, documents = preparation.load_prepared(config["prepared_dir"])
    annotations = pilot.selected_gold(config["sidecar"], prepared)
    tokenizer = pilot.AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True, trust_remote_code=False)
    records, traces = replay_records(prepared, documents, annotations, verified["labels"], verified["scores"], tokenizer)
    run_manifest = read(run_dir / "run_manifest.json")
    score_values = list(verified["scores"].values())
    metadata = {"status": "completed", "schema": "slac-qasper-extended-support-result-v1", "plan_sha256": plan_sha,
        "all_results_available": True, "question_count": len(prepared["queries"]), "family_count": len(documents),
        "new_api_calls": len(config["schedule"]) - run_manifest["reused_prefix_requests"],
        "reused_prefix_requests": run_manifest["reused_prefix_requests"],
        "stage_accounting": verified["accounting"], "historical_pilot": config["historical_pilot"],
        "generation_reservation_remaining_usd": str(amount(CAPS["night_budget_usd"]) - amount(verified["accounting"]["reservation_usd"])),
        "resolved_models": verified["resolved_models"], "score_contract": "reported-scores",
        "raw_score_coverage": {"tasks": len(score_values), "reported_scores_valid": len(score_values),
            "strict_distribution_invalid": sum(bool(probability.validate_probabilities(value)) for value in score_values),
            "renormalized": False, "confidence_used": False},
        "input_binding_sha256": client.object_hash(verified["input_sha256"]), "source_hashes_unchanged": True,
        "answer_generation_performed": False, "test_payload_read": False, "independent_confirmation": False,
        "limits": LIMITS}
    if (read(run_dir / "labels.json") != verified["labels"] or read(run_dir / "raw_scores.json") != verified["scores"]
            or pilot_audit.read_rows(run_dir / "per_question.jsonl") != records
            or pilot_audit.read_rows(run_dir / "traces.jsonl") != traces
            or any(saved.get(key) != value for key, value in summarize(prepared, records).items())
            or any(saved.get(key) != value for key, value in metadata.items())):
        raise ValueError("saved stage result differs from independent response replay")
    bindings = dict(verified["input_sha256"])
    for name in ("summary.json", "labels.json", "raw_scores.json", "per_question.jsonl", "traces.jsonl"):
        bindings[str(run_dir / name)] = digest(run_dir / name)
    load_plan(plan_dir)
    pilot.verify_hashes(bindings)
    return config, prepared, documents, records, saved


def audit(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("audit output already exists")
    config, prepared, documents, records, saved = verify_completed_run(args.plan, args.run)
    result = {"status": "verified", "api_calls": 0, "test_payload_read": False,
              "question_count": len(prepared["queries"]), "record_count": len(records),
              "input_binding_sha256": saved["input_binding_sha256"], "metrics": aggregate(records)}
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "audit.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in ("prepared", "sidecar", "tokenizer", "historical-summary", "output"):
        p.add_argument("--" + name, required=True)
    r = sub.add_parser("run")
    for name in ("plan", "key-file", "output"):
        r.add_argument("--" + name, required=True)
    r.add_argument("--proxy")
    r.add_argument("--resume-from")
    a = sub.add_parser("audit")
    for name in ("plan", "run", "output"):
        a.add_argument("--" + name, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        report = {"plan": plan, "run": run, "audit": audit}[arguments.command](arguments)
    except Exception as exc:
        print(json.dumps({"status": "refused", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({key: report[key] for key in ("status", "question_count", "predicted_reservations", "stage_accounting") if key in report}))
