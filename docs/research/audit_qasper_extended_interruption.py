"""Read-only audit of an initial support run halted by one uncertain timeout.

This does not resume, recreate requests, score partial results, or authorize a
retry. It validates the saved prefix against its original frozen plan and keeps
the terminal unknown-cost attempt in the accounting denominator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_qasper_extended_development as stage
import run_qasper_relation_pilot as pilot
from analyze_qasper_extended_results import snapshot
from run_qasper_evidence_baselines import digest


def validate_timeout_prefix(config, batches, run_dir, plan_hash):
    run_dir = Path(run_dir)
    manifest = pilot.read_json(run_dir / "run_manifest.json")
    if (manifest["plan_sha256"] != plan_hash or manifest["parent_run"] is not None
            or manifest["parent_input_sha256"] or manifest["reused_prefix_requests"] != 0
            or manifest["segments"] != config["segments"]):
        raise ValueError("requires the initial frozen execution without a resume parent")
    by_batch = {row["id"]: row for row in batches}
    totals, completed = pilot.accounting(), pilot.accounting()
    backend_totals = {backend: pilot.accounting() for backend in stage.client.MODELS}
    seen_tasks = {backend: set() for backend in stage.client.MODELS}
    resolved, halted, count = {}, None, 0
    expected_segments = set()
    for segment in config["segments"]:
        call_dir = run_dir / "provider_calls" / f"segment_{segment['segment']:03d}"
        if not call_dir.exists():
            if halted is None:
                raise ValueError("missing terminal timeout before absent segment")
            continue
        if halted is not None:
            raise ValueError("requests continued after terminal timeout")
        expected_segments.add(call_dir.name)
        ledger = pilot.read_json(call_dir / "ledger.json")
        caps = {"budget_usd": segment["reservation_usd"], "request_cap": segment["requests"],
                "question_cap": segment["questions"], "conservative_input_token_cap": segment["input_allowance"]}
        if (any(ledger.get(key) != value for key, value in caps.items())
                or ledger.get("automatic_retries") != 0
                or ledger.get("prompt_version") != config["prompt_version"]
                or not ledger["attempts"] or len(ledger["attempts"]) > segment["requests"]):
            raise ValueError("provider segment contract mismatch")
        local = pilot.accounting()
        local_models = {}
        files = {"ledger.json"}
        for local_index, attempt in enumerate(ledger["attempts"], start=1):
            if count >= segment["stop"] or halted is not None or attempt["attempt"] != local_index:
                raise ValueError("attempt ordering differs from a contiguous prefix")
            item = config["schedule"][count]
            request = call_dir / f"request_{local_index:03d}.json"
            files.add(request.name)
            payload = pilot.prior_request(request, attempt, config["prompt_version"])
            batch = by_batch[item["batch_id"]]
            if (attempt["cache_key"] != item["cache_key"] or attempt["backend"] != item["backend"]
                    or attempt["kind"] != "support" or attempt["task_ids"] != item["task_ids"]
                    or stage.client.object_hash(payload) != item["payload_sha256"]
                    or payload != stage.client.make_payload(batch["tasks"], "support", item["backend"])):
                raise ValueError("saved request differs from the frozen schedule")
            backend = item["backend"]
            if seen_tasks[backend].intersection(item["task_ids"]):
                raise ValueError("duplicate task attempt in the scheduled backend")
            seen_tasks[backend].update(item["task_ids"])
            if attempt["status"] == "completed":
                response_path = call_dir / f"response_{local_index:03d}.json"
                files.add(response_path.name)
                response = pilot.read_json(response_path)
                pilot.validate_reused_response(response, payload, attempt)
                if backend == "jev":
                    stage.support_scores(response, item["task_ids"])
                model = attempt["response_model"]
                if resolved.setdefault(backend, model) != model:
                    raise ValueError("resolved model changed within execution")
                local_models[backend] = model
                pilot.add_attempt(completed, attempt)
            elif (attempt["status"] == "halted" and attempt.get("error_class") == "TimeoutError"
                  and ledger["halt_reason"] == "TimeoutError" and local_index == len(ledger["attempts"])
                  and attempt.get("actual_cost_usd") is None and attempt.get("response_id") is None
                  and not any(key in attempt for key in ("labels", "usage", "response_model", "error_response_file"))):
                halted = {"request_ordinal": count + 1, "backend": backend,
                          "error_class": "TimeoutError", "response_received": False,
                          "cost_known": False, "reserved_usd": attempt["reserved_usd"],
                          "elapsed_seconds": attempt["elapsed_seconds"]}
            else:
                raise ValueError("unsupported failure or in-flight state; no assumed outcome")
            for total in (totals, local, backend_totals[backend]):
                pilot.add_attempt(total, attempt)
            count += 1
        if (set(path.name for path in call_dir.iterdir()) != files
                or ledger["resolved_models"] != local_models
                or stage.amount(ledger["reservation_total_usd"]) != stage.amount(local["reservation_usd"])
                or stage.amount(ledger["actual_reported_cost_usd"]) != stage.amount(local["known_cost_usd"])
                or ledger["halt_reason"] != ("TimeoutError" if halted else None)):
            raise ValueError("segment inventory or accounting differs from verified attempts")
        if halted is None and count != segment["stop"]:
            raise ValueError("unfinished nonterminal segment")
    if (halted is None or count >= len(config["schedule"])
            or {path.name for path in (run_dir / "provider_calls").iterdir()} != expected_segments):
        raise ValueError("not the expected incomplete timeout run")
    if stage.amount(totals["reservation_usd"]) > stage.amount(stage.CAPS["support_budget_usd"]):
        raise ValueError("support reservation exceeded")
    failure = pilot.read_json(run_dir / "failure.json")
    expected_accounting = {**totals, "cost_over_reservation_attempts": 0}
    if (failure["status"] != "failed" or failure["main_results_available"] is not False
            or failure["completed_requests"] != completed["attempts"]
            or failure["admitted_stage_reservation_usd"] != totals["reservation_usd"]
            or failure["new_completed_accounting"] != completed
            or failure["stage_accounting"] != expected_accounting
            or failure["prior_accounting"] != pilot.accounting()
            or failure["automatic_retries"] != 0):
        raise ValueError("failure summary differs from independently reconstructed accounting")
    return {"schema": "slac-extended-support-timeout-audit-v1", "status": "verified_incomplete",
        "planned_questions": config["question_count"], "planned_families": config["family_count"],
        "planned_requests": len(config["schedule"]), "completed_requests": completed["attempts"],
        "attempted_requests": count, "unattempted_requests": len(config["schedule"]) - count,
        "new_accounting": totals, "completed_accounting": completed, "backend_accounting": backend_totals,
        "terminal_attempt": halted, "resolved_models": resolved, "api_calls_by_audit": 0,
        "automatic_retries": 0, "main_results_available": False, "partial_quality_metrics_computed": False,
        "answer_stage_not_performed_by_audit": True, "independent_confirmation": False,
        "limits": ["Timeout does not establish provider rejection or zero charge.",
                   "The terminal reservation is retained and its billed cost is unknown.",
                   "Incomplete support coverage prevents the full paired quality and answer protocol.",
                   "This audit does not grant reuse, retry, a new budget, or permission to omit questions."]}


def audit(args):
    plan, run, output = (Path(getattr(args, name)).resolve() for name in ("plan", "run", "output"))
    if output.exists():
        raise FileExistsError("interruption audit output already exists")
    before = {**snapshot(plan), **snapshot(run)}
    config, batches = stage.load_plan(plan)
    report = validate_timeout_prefix(config, batches, run, digest(plan / "experiment_config.json"))
    if {**snapshot(plan), **snapshot(run)} != before:
        raise ValueError("frozen execution changed during interruption audit")
    bindings = {**before, **config["input_sha256"], str(Path(__file__).resolve()): digest(__file__)}
    pilot.verify_hashes(bindings)
    report["input_hashes_unchanged"] = True
    report["input_binding_sha256"] = stage.client.object_hash(bindings)
    output.mkdir(parents=True, exist_ok=False)
    pilot.write_json(output / "source_binding.json", {"input_sha256": bindings})
    pilot.write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=False, indent=2))
