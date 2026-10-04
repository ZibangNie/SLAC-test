"""One separately admitted eight-request historical JEV support diagnostic.

Import/prepare/verify never read credentials. No old experiment runner is called.
The sealed SupportClient supplies transport, accounting, scores and per-call guards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "docs/research"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_qasper_confirmation_support as support
from docs.research.conditional_probe_client import _proxy
from docs.research.run_conditional_probe_bounded import supervise

client = support.client
canonical = client.canonical_bytes
LIVE_ROOT = ROOT / "artifacts/research-foundation/offline-20261004/score-transfer-live-01"
PREP_DIR = ROOT / "artifacts/research-foundation/offline-20261004/score-transfer-prep-01"
HANDOFF_SHA = "1bc7f3716b1a2c23635371956cfe53f8feb128786f38bffbf39f813b7ca7aebc"
PACKETS_SHA = "7c6da126d36de36b130b0d675832c125bbf2b00e8d2ada2725f5b8470882b804"
HISTORICAL_SHA = "d43852b1511f7dbec1f1822466cb0aba8fa50ed3713ccb8b154ceb04af8f4739"
SCHEMA = "slac-score-transfer-microdiagnostic-v1"
MODEL = "typesafe/jev-1.13-20260917"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
LIMITS = {"requests": 8, "judgments": 64, "budget_usd": "0.10",
          "input_allowance": 892880, "output_allowance": 8192,
          "wire_bytes": 24000, "runtime_seconds": 180, "request_seconds": 65,
          "admission_window_seconds": 3600, "metadata_age_seconds": 86400,
          "automatic_retries": 0}
EXTRA_SOURCES = ("docs/research/SCORE_TRANSFER_MICRODIAGNOSTIC_PROTOCOL_20261004.md",
                 "docs/research/evaluate_score_transfer_microdiagnostic.py",
                 "tests/research/test_score_transfer_evaluation.py",
                 "docs/research/qasper_metrics.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc)


def timestamp(value):
    require(isinstance(value, str), "explicit UTC timestamp required")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset() == timedelta(0), "UTC required")
    return result.astimezone(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=client.unique_object)


def write(path, value, *, replace=False):
    path = Path(path)
    target = path.with_name(path.name + ".tmp") if replace else path
    with target.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        target.replace(path)


def contained(path):
    path = Path(path).resolve()
    require(path != LIVE_ROOT.resolve() and path.is_relative_to(LIVE_ROOT.resolve()),
            "output must be within the separate live-stage directory")
    return path


def _inputs():
    handoff = PREP_DIR / "handoff.json"
    packets = PREP_DIR / "request_packets.json"
    historical = PREP_DIR / "transport_feasibility.json"
    require(digest(handoff) == HANDOFF_SHA and digest(packets) == PACKETS_SHA
            and digest(historical) == HISTORICAL_SHA, "frozen preparation changed")
    bindings = {str(p.resolve()): digest(p) for p in (handoff, packets, historical)}
    for name, expected in read(handoff)["sealed_inputs"].items():
        require(Path(name).name == name and not name.startswith("."), "invalid prep filename")
        path = PREP_DIR / name
        require(digest(path) == expected, "sealed preparation input changed")
        bindings[str(path.resolve())] = expected
    # These old sources stay sealed; no copied/revised network implementation.
    for name, expected in read(historical)["source_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == expected, "historical source changed")
        bindings[str(path)] = expected
    return bindings


def _sources():
    paths = {Path(__file__), ROOT / "tests/research/test_score_transfer_microdiagnostic.py",
             Path(support.__file__), Path(client.__file__), Path(support.stage.__file__),
             Path(support.stage.probability.__file__), Path(support.pilot.__file__),
             Path(support.budget.__file__),
             ROOT / "docs/research/conditional_probe_client.py",
             ROOT / "docs/research/run_conditional_probe_bounded.py",
             *(ROOT / name for name in EXTRA_SOURCES)}
    return {str(p.resolve()): digest(p) for p in paths}


def _provider(path, now, *, fresh):
    path = contained(path)
    value = read(path)
    require(value.get("schema") == "slac-score-transfer-provider-check-v1"
            and value.get("status") == "verified_current_public_metadata", "provider review missing")
    require(len(value["models"]) == 1, "exact JEV metadata required")
    model = value["models"][0]
    endpoint = model["endpoint_metadata"]
    require(model["role"] == "jev" and model["request_endpoint"] == ENDPOINT
            and model["request_model_id"] == "typesafe/jev-1.13"
            and model["http_status"] == 200 and model["endpoint_count"] == 1
            and endpoint["name"] == "TypeSafe | " + MODEL
            and endpoint["model_id"] == "typesafe/jev-1.13"
            and endpoint["provider_name"] == "TypeSafe" and endpoint["tag"] == "typesafe"
            and endpoint["status"] == 0 and endpoint["context_length"] == 32000,
            "provider identity or availability changed")
    require(model["base_prices_usd_per_million"] == {"prompt": "0.042", "completion": "0"}
            and Decimal(endpoint["pricing"]["prompt"]) == Decimal("0.000000042")
            and Decimal(endpoint["pricing"]["completion"]) == 0
            and set(endpoint["pricing"]) <= {"prompt", "completion", "discount"}
            and endpoint["pricing"].get("discount", 0) == 0, "provider prices changed")
    if fresh:
        age = (now - timestamp(model["received_at_utc"])).total_seconds()
        require(0 <= age <= LIMITS["metadata_age_seconds"], "provider metadata stale or future-dated")
    require(model["raw_response_file"] == "provider_endpoints.json", "unexpected metadata file")
    raw = path.parent / "provider_endpoints.json"
    require(digest(raw) == model["raw_response_sha256"], "raw provider metadata changed")
    data = read(raw)["data"]
    require(raw.stat().st_size == model["raw_response_bytes"]
            and data["id"] == "typesafe/jev-1.13"
            and data["architecture"]["modality"] == "text->decisions"
            and data["architecture"]["input_modalities"] == ["text"]
            and data["architecture"]["output_modalities"] == ["decisions"]
            and data["endpoints"] == [endpoint], "provider extraction differs from raw metadata")
    require(not any("overrid" in key.casefold() for item in (value, model, data, endpoint)
                    for key in item), "provider overrides not admitted")
    return {str(path): digest(path), str(raw.resolve()): digest(raw)}


def _jobs():
    packets = read(PREP_DIR / "request_packets.json")
    require(type(packets) is list and len(packets) == 8, "exactly eight packets required")
    jobs, seen, groups = [], set(), {}
    for ordinal, packet in enumerate(packets, 1):
        batch, payload = packet["batch"], packet["payload"]
        tasks = batch["tasks"]
        require(packet["ordinal"] == ordinal and batch["kind"] == "support" and len(tasks) == 8,
                "packet count/order/schema changed")
        require(payload == client.make_payload(tasks, "support", "jev"), "historical payload changed")
        body = canonical(payload)
        reserve, inp, out = client.reservation(payload, "jev")
        require(len(body) <= LIMITS["wire_bytes"] and len(body) == packet["canonical_payload_bytes"]
                and hashlib.sha256(body).hexdigest() == packet["payload_sha256"]
                and str(reserve) == packet["reserved_usd"]
                and inp == packet["input_allowance"] and out == packet["output_allowance"],
                "packet hash or reservation changed")
        group = tuple(batch["group"])
        require(len(group) == 2, "invalid query group")
        groups[group] = groups.get(group, 0) + len(tasks)
        for task in tasks:
            identity = {"kind": "support", "doc_id": group[0], "question_id": group[1], "item": task["item"]}
            require(task["id"] == "support:" + client.object_hash(identity) and task["id"] not in seen,
                    "task identity or coverage changed")
            seen.add(task["id"])
        jobs.append({"ordinal": ordinal, "backend": "jev", "kind": "support",
                     "task_ids": [t["id"] for t in tasks], "tasks": tasks, "payload": payload,
                     "payload_sha256": packet["payload_sha256"], "reserved_usd": str(reserve),
                     "input_allowance": inp, "output_allowance": out})
    require(len(groups) == 4 and set(groups.values()) == {16} and len(seen) == 64,
            "four-query/64-judgment contract changed")
    require(len({j["payload_sha256"] for j in jobs}) == 8, "duplicate physical packet")
    require(sum(Decimal(j["reserved_usd"]) for j in jobs) <= Decimal(LIMITS["budget_usd"])
            and sum(j["input_allowance"] for j in jobs) <= LIMITS["input_allowance"]
            and sum(j["output_allowance"] for j in jobs) == LIMITS["output_allowance"], "admission cap exceeded")
    return jobs


def _window(plan, now, *, fresh):
    start, end, created = map(timestamp, (plan["valid_from_utc"], plan["valid_until_utc"], plan["created_at_utc"]))
    require(start <= created < end and 0 < (end-start).total_seconds() <= 3600, "invalid admission window")
    if fresh:
        require(start <= now and (end-now).total_seconds() > 65, "admission not active or expired")


def prepare(output, run_output, provider_snapshot, valid_from_utc, valid_until_utc):
    output, run_output = contained(output), contained(run_output)
    require(not output.exists() and not run_output.exists() and not (LIVE_ROOT / "consumed.json").exists(),
            "new unconsumed stage required")
    require(not output.is_relative_to(run_output) and not run_output.is_relative_to(output), "plan/run overlap")
    now = utc_now()
    plan = {"schema": SCHEMA, "limits": LIMITS, "created_at_utc": now.isoformat(),
            "valid_from_utc": valid_from_utc, "valid_until_utc": valid_until_utc,
            "plan_dir": str(output), "run_dir": str(run_output),
            "provider_snapshot": str(contained(provider_snapshot))}
    _window(plan, now, fresh=True)
    plan["inputs_sha256"] = _inputs() | _provider(provider_snapshot, now, fresh=True)
    plan["source_sha256"], plan["jobs"] = _sources(), _jobs()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    write(output / "seal.json", {"plan_sha256": digest(output / "plan.json")})
    return {"plan_dir": str(output), "plan_sha256": digest(output / "plan.json"), "requests": 8,
            "judgments": 64, "reserved_usd": str(sum(Decimal(j["reserved_usd"]) for j in plan["jobs"]))}


def verify_plan(directory, *, fresh=False):
    directory = contained(directory)
    plan = read(directory / "plan.json")
    sha = digest(directory / "plan.json")
    require(read(directory / "seal.json") == {"plan_sha256": sha}, "plan seal changed")
    require(plan["schema"] == SCHEMA and plan["limits"] == LIMITS
            and plan["plan_dir"] == str(directory), "plan identity/caps changed")
    contained(plan["run_dir"])
    _window(plan, utc_now(), fresh=fresh)
    require(plan["inputs_sha256"] == _inputs() | _provider(plan["provider_snapshot"], utc_now(), fresh=fresh),
            "input or provider binding changed")
    require(plan["source_sha256"] == _sources(), "source drift")
    jobs = _jobs()
    require(plan["jobs"] == jobs, "frozen jobs changed")
    return plan, jobs, sha


def _mirror(ledger):
    value = json.loads(canonical(ledger))
    for attempt in value["attempts"]:
        attempt["cost_status"] = ("provider_reported" if attempt.get("actual_cost_usd") is not None
                                  else "cost_unknown")
    return value


class _Client(support.SupportClient):
    """Mirror the persisted ledger for the existing outer watchdog observer."""
    def save(self):
        try:
            super().save()
        finally:
            path = self.output / "ledger.json"
            if path.exists():
                write(self.run_output / "ledger.json", _mirror(read(path)), replace=True)


def _accounting(ledger):
    attempts = ledger["attempts"]
    return {"attempts": len(attempts), "reserved_usd": ledger["reservation_total_usd"],
            "known_cost_usd": ledger["actual_reported_cost_usd"],
            "unknown_cost_attempts": sum(a.get("actual_cost_usd") is None for a in attempts),
            "completed_requests": sum(a["status"] == "completed" for a in attempts), "automatic_retries": 0}


def _collect(plan, jobs):
    run_dir = Path(plan["run_dir"])
    ledger, bindings = support.event_ledger(run_dir / "provider_calls")
    require(not ledger["halt_reason"] and len(ledger["attempts"]) == 8, "run incomplete")
    caps = {"budget_usd": "0.10", "request_cap": 8, "question_cap": 64,
            "conservative_input_token_cap": 892880, "prompt_version": client.PROMPT_VERSION,
            "automatic_retries": 0}
    require(all(ledger.get(k) == v for k, v in caps.items()), "executed ledger caps differ")
    require(read(run_dir / "ledger.json") == _mirror(ledger), "watchdog ledger differs")
    claim = read(LIVE_ROOT / "consumed.json")
    started = timestamp(claim["claimed_at_utc"])
    end = min(timestamp(plan["valid_until_utc"]), started + timedelta(seconds=180))
    finished = timestamp(read(run_dir / "summary.json")["finished_at_utc"]) if (run_dir / "summary.json").exists() else utc_now()
    previous_end = started
    labels, scores = {}, {}
    for job, record in zip(jobs, ledger["attempts"], strict=True):
        ordinal = job["ordinal"]
        req, resp = (run_dir / "provider_calls" / f"{prefix}_{ordinal:03d}.json" for prefix in ("request", "response"))
        require(req.read_bytes() == canonical(job["payload"]) and record["request_sha256"] == job["payload_sha256"]
                and record["attempt"] == ordinal and record["question_count"] == 8
                and record["status"] == "completed"
                and record["cache_key"] == client.object_hash({"endpoint": ENDPOINT, "payload": job["payload"],
                                                               "prompt_version": client.PROMPT_VERSION})
                and record["backend"] == "jev" and record["kind"] == "support"
                and record["task_ids"] == job["task_ids"] and record["response_model"] == MODEL
                and record["reserved_usd"] == job["reserved_usd"]
                and record["input_allowance"] == job["input_allowance"]
                and record["output_allowance"] == job["output_allowance"]
                and type(record["elapsed_seconds"]) in (int, float)
                and math.isfinite(record["elapsed_seconds"]) and 0 <= record["elapsed_seconds"] <= 65,
                "executed packet differs")
        attempt_start = timestamp(record["started_at"])
        attempt_end = attempt_start + timedelta(seconds=record["elapsed_seconds"])
        require(previous_end <= attempt_start and (end-attempt_start).total_seconds() > 65
                and attempt_end <= min(end, finished), "executed attempt outside timing bounds")
        previous_end = attempt_end
        response = read(resp)
        support.pilot.validate_reused_response(response, job["payload"], record)
        labels.update(client.parse_labels(response, job["payload"], "jev", "support"))
        scores.update(support.stage.support_scores(response, job["task_ids"]))
    require(len(labels) == len(scores) == 64 and not (run_dir / "hard_timeout.json").exists(), "incomplete timed run")
    return {"complete": True, "labels": labels, "reported_scores": scores,
            "accounting": _accounting(ledger), "bindings": bindings}


def run(directory, *, key_file=None, proxy=None, live=False, transport=None, guard_factory=support.Watchdog):
    plan, jobs, sha = verify_plan(directory, fresh=True)
    require(type(live) is bool and ((live and transport is None and key_file is not None)
                                  or (not live and callable(transport))), "explicit execution mode required")
    proxy = _proxy(proxy)
    output = contained(plan["run_dir"])
    require(not output.exists(), "run directory already exists")
    started = utc_now()
    # Stage-wide claim prevents a replacement plan from retrying these packets.
    write(LIVE_ROOT / "consumed.json", {"schema": SCHEMA, "plan_sha256": sha,
                                      "run_dir": str(output), "claimed_at_utc": started.isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    end = min(timestamp(plan["valid_until_utc"]), started + timedelta(seconds=180))
    guard = guard_factory(output, (end-utc_now()).total_seconds())
    bounded = None
    try:
        # Repeat after the exclusive claim and immediately before constructor key access.
        verify_plan(directory, fresh=True)
        bounded = _Client(output / "provider_calls", jobs=jobs, deadline=end.isoformat(), run_output=output,
                          guard_factory=guard_factory, key_file=key_file, proxy=proxy, transport=transport,
                          budget_usd="0.10", request_cap=8, question_cap=64, token_cap=892880)
        for job in jobs:
            verify_plan(directory, fresh=True)
            bounded.submit(job["tasks"], "support", "jev")
        verify_plan(directory)
        result = _collect(plan, jobs)
        finished = utc_now()
        require(finished < end, "runtime deadline exhausted")
        write(output / "judgments.json", {k: result[k] for k in ("complete", "labels", "reported_scores")})
        write(output / "summary.json", {"schema": SCHEMA, "status": "completed", "plan_sha256": sha,
              "finished_at_utc": finished.isoformat(), "accounting": result["accounting"],
              "judgments_sha256": digest(output / "judgments.json")})
        return result
    except BaseException as error:
        value = {"schema": SCHEMA, "status": "halted", "plan_sha256": sha,
                 "error_class": type(error).__name__, "automatic_retries": 0}
        if bounded is not None:
            value["accounting"] = _accounting(bounded.ledger)
        write(output / "failure.json", value)
        raise RuntimeError("score-transfer run halted; inspect private ledger") from None
    finally:
        guard.close()
        if bounded is not None:
            bounded.key = ""


def verify_run(directory):
    plan, jobs, sha = verify_plan(directory)
    run_dir = Path(plan["run_dir"])
    claim = read(LIVE_ROOT / "consumed.json")
    require(set(claim) == {"schema", "plan_sha256", "run_dir", "claimed_at_utc"}
            and claim["schema"] == SCHEMA and claim["plan_sha256"] == sha
            and claim["run_dir"] == str(run_dir), "execution claim differs")
    require(not (run_dir / "failure.json").exists(), "failed run has no complete judgments")
    result = _collect(plan, jobs)
    wanted = {k: result[k] for k in ("complete", "labels", "reported_scores")}
    require(read(run_dir / "judgments.json") == wanted, "saved judgments changed")
    summary = read(run_dir / "summary.json")
    finished, started = timestamp(summary["finished_at_utc"]), timestamp(claim["claimed_at_utc"])
    require(timestamp(plan["valid_from_utc"]) <= started <= finished < timestamp(plan["valid_until_utc"])
            and (finished-started).total_seconds() < 180, "completed run outside admission/runtime window")
    require(summary == {"schema": SCHEMA, "status": "completed", "plan_sha256": sha,
            "finished_at_utc": summary["finished_at_utc"],
            "accounting": result["accounting"], "judgments_sha256": digest(run_dir / "judgments.json")},
            "completion summary changed")
    for path in (Path(directory)/"plan.json", Path(directory)/"seal.json", LIVE_ROOT/"consumed.json",
                 run_dir/"judgments.json", run_dir/"summary.json", run_dir/"ledger.json"):
        result["bindings"][str(path.resolve())] = digest(path)
    result["bindings"].update(plan["inputs_sha256"] | plan["source_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("output", "run-output", "provider-snapshot", "valid-from-utc", "valid-until-utc"):
        p.add_argument("--"+name, required=True)
    for command in ("verify", "verify-run", "run", "bounded"):
        p = commands.add_parser(command); p.add_argument("--plan", required=True, type=Path)
        if command in ("run", "bounded"):
            p.add_argument("--key-file", required=True, type=Path); p.add_argument("--proxy")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.run_output, args.provider_snapshot, args.valid_from_utc, args.valid_until_utc)
    elif args.command == "verify":
        _, jobs, sha = verify_plan(args.plan, fresh=True)
        result = {"status": "verified", "plan_sha256": sha, "requests": len(jobs)}
    elif args.command == "verify-run":
        result = verify_run(args.plan)["accounting"]
    elif args.command == "run":
        result = run(args.plan, key_file=args.key_file, proxy=args.proxy, live=True)["accounting"]
    else:
        plan, _, sha = verify_plan(args.plan, fresh=True)
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan.resolve()),
                   "--key-file", str(args.key_file.resolve())]
        if args.proxy:
            command += ["--proxy", _proxy(args.proxy)]
        result = supervise(command, Path(plan["run_dir"]), LIVE_ROOT/"controller.json", timeout_seconds=180, plan_sha256=sha)
    print(json.dumps(result, sort_keys=True))
    if args.command == "bounded":
        return 0 if result["status"] == "worker_exited" and result.get("returncode") == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
