"""One explicitly amended continuation of the frozen primary support experiment.

The failed parent remains immutable. Its timeout remains an unknown-cost physical
attempt; only the amendment's one exact replacement and untouched suffix run here.
This module has its own schema, collector and verifier, never an old-style resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

import audit_qasper_extended_interruption as interruption
import analyze_qasper_extended_results as statistics
import run_qasper_extended_development as original
import run_qasper_local_answer_evaluation as local_answers
import run_qasper_owner_order_answers as owner_answers

client, pilot = original.client, original.pilot
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/research-foundation"
SCHEMA = "slac-qasper-primary-support-recovery-v1"
SPEC = {"original_logical_requests": 308, "inherited_completed_requests": 164,
    "previous_unknown_ordinal": 165, "explicit_replacements_allowed": 1,
    "previously_unattempted_requests": 143, "new_requests": 144,
    "question_count": 77, "family_count": 24, "night_cap_usd": "5",
    "new_reservation_usd": "1.3027567750", "support_after_reservation_usd": "2.7720199175",
    "night_after_reservation_usd": "4.1353945925", "future_generation_headroom_usd": "0.8646054075",
    "stop_at_utc": "2026-09-27T01:00:00+00:00", "max_execution_seconds": 5400,
    "automatic_retries": 0, "subsequent_recovery_allowed": False,
    "payload_prompt_schema_route_prices_changed": False, "partial_quality_scores": False,
    "generation_minimum_reserve_interpretation": "original planning headroom, not refundable money; subsequent whole answer plan must fit remaining exact reservation",
    "answer_generation_performed": False, "official_test_used": False,
    "independent_confirmation": False}
PRIOR = {"attempts": 539, "reservation_usd": "2.8326378175",
    "known_cost_usd": "0.273198707", "unknown_cost_attempts": 1}
MODELS = {"jev": "typesafe/jev-1.13-20260917", "general": "qwen/qwen3.6-plus"}
SOURCES = {"original_plan": "qasper-extended-development-plan-01",
    "original_run": "qasper-extended-development-run-01", "interruption_audit": "qasper-extended-interruption-audit-01",
    "local_plan": "qasper-local-answer-plan-02", "local_run": "qasper-local-answer-run-01",
    "local_audit": "qasper-local-answer-audit-01", "owner_plan": "qasper-owner-order-answer-plan-01",
    "owner_run": "qasper-owner-order-answer-run-01", "owner_audit": "qasper-owner-order-answer-audit-01"}
PLAN_FILES = ("experiment_config.json", "jobs.json", "inheritance.json", "plan_manifest.json")
OUTPUT_FILES = ("labels.json", "raw_scores.json", "per_question.jsonl", "traces.jsonl")


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=client.unique_object)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def merge(*values):
    return local_answers.merge(*values)


def tree(path):
    return {str(p.resolve()): original.digest(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def snapshot_files(path, names):
    path = Path(path).resolve()
    if {p.name for p in path.iterdir()} != set(names):
        raise ValueError("frozen artifact inventory differs")
    return {str((path / name).resolve()): original.digest(path / name) for name in names}


def compact_accounting(value):
    return {key: value[key] for key in PRIOR}


def add_totals(left, right):
    return {key: str(Decimal(left[key]) + Decimal(right[key])) if key.endswith("usd")
            else left[key] + right[key] for key in left}


def released_generation(plan_path, run_path, audit_path, *, extension):
    """Replay transport/accounting only; no old scientific score/model audit."""
    config = read(plan_path / "experiment_config.json")
    seal = read(plan_path / "plan_manifest.json")
    if seal["experiment_config_sha256"] != original.digest(plan_path / "experiment_config.json"):
        raise ValueError("completed answer plan seal differs")
    summary, audit, release = (read(run_path / "summary.json"), read(audit_path / "audit.json"),
                               read(audit_path / "source_binding.json"))
    if (summary.get("status") != "completed" or audit.get("status") != "verified_complete"
            or release.get("root_release") != "complete_run_reviewed_for_publication"
            or audit.get("all_bound_inputs_outputs_unchanged") is not True):
        raise ValueError("complete released generation accounting required")
    release_inputs = release["input_sha256"]
    for path in (plan_path / "experiment_config.json", run_path / "summary.json", audit_path / "audit.json"):
        if release_inputs.get(str(path.resolve())) != original.digest(path):
            raise ValueError("generation release is not bound to this completed source")
    expected = {str(Path(path).relative_to(run_path)): sha for path, sha in tree(run_path).items()
                if Path(path) != run_path / "summary.json"}
    if summary["output_sha256"] != expected:
        raise ValueError("completed generation output inventory differs")
    jobs = read(plan_path / "jobs.json")
    inspect = owner_answers.inspect_calls if extension else local_answers.inspect_calls
    _, ledger, calls, complete = inspect(config, jobs, run_path, require_complete=True)
    if not complete or len(ledger["attempts"]) != (7 if extension else 367):
        raise ValueError("completed generation physical denominator differs")
    accounting_function = owner_answers.accounting if extension else local_answers.accounting
    expected_accounting = accounting_function(config, ledger, True)
    if audit["accounting"] != expected_accounting or any(summary.get(k) != v for k, v in expected_accounting.items()):
        raise ValueError("generation accounting does not replay")
    result = {"attempts": len(ledger["attempts"]), "reservation_usd": ledger["reservation_total_usd"],
        "known_cost_usd": ledger["actual_reported_cost_usd"], "unknown_cost_attempts": 0}
    bindings = merge(tree(plan_path), tree(run_path), tree(audit_path), release_inputs, config["input_sha256"], calls)
    pilot.verify_hashes(bindings)
    return result, bindings


def source_data(protocol):
    paths = {name: (ARTIFACTS / value).resolve() for name, value in SOURCES.items()}
    before = merge(*(tree(path) for path in paths.values()))
    config, batches = original.load_plan(paths["original_plan"])
    audit = read(paths["interruption_audit"] / "audit.json")
    audit_inputs = read(paths["interruption_audit"] / "source_binding.json")["input_sha256"]
    replay = interruption.validate_timeout_prefix(config, batches, paths["original_run"],
        original.digest(paths["original_plan"] / "experiment_config.json"))
    if (replay != {k: v for k, v in audit.items() if k not in ("input_hashes_unchanged", "input_binding_sha256")}
            or audit["input_binding_sha256"] != client.object_hash(audit_inputs)
            or replay["completed_requests"] != SPEC["inherited_completed_requests"]
            or replay["attempted_requests"] != SPEC["previous_unknown_ordinal"]
            or replay["planned_requests"] != SPEC["original_logical_requests"]
            or replay["resolved_models"] != MODELS):
        raise ValueError("only the exact original one-timeout prefix is eligible")
    prior = compact_accounting(replay["new_accounting"])
    extra = []
    for tag, extension in (("local", False), ("owner", True)):
        account, hashes = released_generation(paths[tag + "_plan"], paths[tag + "_run"], paths[tag + "_audit"], extension=extension)
        prior = add_totals(prior, account); extra.append(hashes)
    if prior != PRIOR:
        raise ValueError("whole-night accounting changed; no budget reset")
    inheritance = []
    terminal = None
    for segment in config["segments"]:
        directory = paths["original_run"] / "provider_calls" / f"segment_{segment['segment']:03d}"
        if not directory.exists(): continue
        ledger_path = directory / "ledger.json"
        for attempt in read(ledger_path)["attempts"]:
            index = segment["start"] + attempt["attempt"] - 1
            request = directory / f"request_{attempt['attempt']:03d}.json"
            item = {"schedule_index": index, "ledger_path": str(ledger_path), "ledger_sha256": original.digest(ledger_path),
                "attempt": attempt["attempt"], "request_path": str(request), "request_sha256": original.digest(request),
                "cache_key": attempt["cache_key"]}
            if attempt["status"] == "completed":
                response = directory / f"response_{attempt['attempt']:03d}.json"
                inheritance.append({**item, "response_path": str(response), "response_sha256": original.digest(response)})
            else:
                terminal = item
    if [item["schedule_index"] for item in inheritance] != list(range(SPEC["inherited_completed_requests"])) or terminal["schedule_index"] != SPEC["previous_unknown_ordinal"] - 1:
        raise ValueError("inherited prefix/explicit replacement identity differs")
    code = [Path(__file__).resolve(), ROOT / "tests/research/test_qasper_primary_support_recovery.py",
        ROOT / "docs/research/analyze_qasper_recovered_support.py", ROOT / "tests/research/test_qasper_recovered_support_analysis.py",
        Path(statistics.__file__).resolve(), ROOT / "tests/research/test_analyze_qasper_extended_results.py",
        Path(interruption.__file__).resolve(), Path(local_answers.__file__).resolve(), Path(owner_answers.__file__).resolve(),
        Path(protocol).resolve()]
    bindings = merge(before, audit_inputs, config["input_sha256"], *extra,
                     {str(path): original.digest(path) for path in code})
    if before != merge(*(tree(path) for path in paths.values())):
        raise ValueError("parent source inventory changed")
    pilot.verify_hashes(bindings)
    return {"original_config": config, "batches": batches, "inheritance": inheritance,
        "explicit_replacement_of": terminal, "prior_night_accounting": prior,
        "prior_support_accounting": replay["new_accounting"], "input_sha256": bindings}


def build_jobs(data):
    config = data["original_config"]
    batches = {batch["id"]: batch for batch in data["batches"]}
    jobs = []
    for item in config["schedule"][SPEC["inherited_completed_requests"]:]:
        payload = client.make_payload(batches[item["batch_id"]]["tasks"], "support", item["backend"])
        if client.object_hash(payload) != item["payload_sha256"]:
            raise ValueError("old frozen payload changed")
        jobs.append({**item, "payload": payload,
            "origin": "explicit_one_time_replacement" if not jobs else "previously_unattempted"})
    if (len(jobs) != SPEC["new_requests"] or len({j["cache_key"] for j in jobs}) != len(jobs)
            or jobs[0]["cache_key"] != data["explicit_replacement_of"]["cache_key"]
            or client.canonical_bytes(jobs[0]["payload"]) != Path(data["explicit_replacement_of"]["request_path"]).read_bytes()):
        raise ValueError("replacement or exact untouched suffix differs")
    totals = original.totals(jobs)
    support = Decimal(data["prior_support_accounting"]["reservation_usd"]) + Decimal(totals["reservation_usd"])
    night = Decimal(data["prior_night_accounting"]["reservation_usd"]) + Decimal(totals["reservation_usd"])
    if (Decimal(totals["reservation_usd"]) != Decimal(SPEC["new_reservation_usd"])
            or support != Decimal(SPEC["support_after_reservation_usd"]) or support > Decimal(original.CAPS["support_budget_usd"])
            or night != Decimal(SPEC["night_after_reservation_usd"]) or night > Decimal(SPEC["night_cap_usd"])
            or Decimal(SPEC["night_cap_usd"]) - night != Decimal(SPEC["future_generation_headroom_usd"])):
        raise ValueError("amended cumulative reservation differs")
    for name, cap in (("requests", "request_cap"), ("questions", "question_cap"), ("input_allowance", "input_allowance_cap")):
        prior_key = "attempts" if name == "requests" else name
        if data["prior_support_accounting"][prior_key] + totals[name] > original.CAPS[cap]:
            raise ValueError("original cumulative support allowance exhausted")
    return jobs


def registration_path():
    return ARTIFACTS / "qasper-primary-support-recovery-registration-01.json"


def prepare(args):
    output, run_output, protocol = map(lambda x: Path(x).resolve(), (args.output, args.run_output, args.protocol))
    if output.exists() or run_output.exists() or registration_path().exists():
        raise FileExistsError("recovery is single-use; plan/run/registration must be unused")
    protected = [ARTIFACTS / path for path in SOURCES.values()]
    if (output.is_relative_to(run_output) or run_output.is_relative_to(output)
            or any(new.is_relative_to(old) or old.is_relative_to(new) for new in (output, run_output) for old in protected)):
        raise ValueError("recovery paths overlap immutable parents")
    data = source_data(protocol); jobs = build_jobs(data)
    config = {"schema": SCHEMA, "status": "prepared_not_executed", "specification": SPEC,
        "statistical_specification": statistics.SPECIFICATION,
        "statistical_specification_sha256": client.object_hash(statistics.SPECIFICATION),
        "protocol": str(protocol), "run_output": str(run_output), "single_use_registration": str(registration_path()),
        "original_config": data["original_config"], "resolved_model_locks": MODELS,
        **{key: data[key] for key in ("prior_night_accounting", "prior_support_accounting", "explicit_replacement_of", "input_sha256")},
        **{key: data["original_config"][key] for key in ("prepared_dir", "sidecar", "tokenizer", "methods", "selector")},
        "prediction": original.totals(jobs), "segments": original.segment_schedule(jobs),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "api_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "jobs.json", jobs); write(output / "inheritance.json", data["inheritance"])
    config["plan_files_sha256"] = {name: original.digest(output / name) for name in ("jobs.json", "inheritance.json")}
    write(output / "experiment_config.json", config)
    write(output / "plan_manifest.json", {"experiment_config_sha256": original.digest(output / "experiment_config.json")})
    return {"status": config["status"], "new_requests": len(jobs), "prediction": config["prediction"], "api_calls": 0}


def load_plan(directory):
    directory = Path(directory).resolve(); own = snapshot_files(directory, PLAN_FILES)
    config, seal = read(directory / "experiment_config.json"), read(directory / "plan_manifest.json")
    if (seal != {"experiment_config_sha256": own[str(directory / "experiment_config.json")]}
            or config["schema"] != SCHEMA or config["status"] != "prepared_not_executed"
            or config["specification"] != SPEC or config["resolved_model_locks"] != MODELS
            or config["statistical_specification"] != statistics.SPECIFICATION
            or config["statistical_specification_sha256"] != client.object_hash(statistics.SPECIFICATION)
            or config["single_use_registration"] != str(registration_path())):
        raise ValueError("frozen recovery contract/seal differs")
    data = source_data(config["protocol"]); jobs = build_jobs(data)
    for name in ("original_config", "prior_night_accounting", "prior_support_accounting", "explicit_replacement_of", "input_sha256"):
        if config[name] != data[name]: raise ValueError("recovery parent source differs")
    if (read(directory / "jobs.json") != jobs or read(directory / "inheritance.json") != data["inheritance"]
            or config["prediction"] != original.totals(jobs) or config["segments"] != original.segment_schedule(jobs)
            or config["plan_files_sha256"] != {name: own[str(directory / name)] for name in ("jobs.json", "inheritance.json")}
            or any(config[key] != data["original_config"][key] for key in ("prepared_dir", "sidecar", "tokenizer", "methods", "selector"))):
        raise ValueError("recovery jobs/inheritance/selection differ")
    pilot.verify_hashes(merge(own, data["input_sha256"]))
    return config, data, jobs, own


def utc_now():
    return datetime.now(timezone.utc)


def ensure_time(started, *, dispatch=False):
    margin = 65 if dispatch else 0
    if ((datetime.fromisoformat(SPEC["stop_at_utc"]) - utc_now()).total_seconds() <= margin
            or time.monotonic() - started >= SPEC["max_execution_seconds"] - margin):
        raise TimeoutError("recovery execution or absolute 09:00 deadline reached")


class HardDeadline:
    """The CLI process cannot outlive the wall deadline, even in a blocked read."""
    def __init__(self, output, started):
        self.cancel = threading.Event()
        seconds = min((datetime.fromisoformat(SPEC["stop_at_utc"]) - utc_now()).total_seconds(),
                      SPEC["max_execution_seconds"] - (time.monotonic() - started))
        def stop():
            if not self.cancel.wait(max(0, seconds)):
                try:
                    write(Path(output) / "hard_deadline.json", {"schema": SCHEMA, "status": "failed_hard_deadline",
                        "main_results_available": False, "automatic_retries": 0})
                finally:
                    os._exit(124)
        self.thread = threading.Thread(target=stop, daemon=True)
        self.thread.start()

    def close(self):
        self.cancel.set(); self.thread.join(timeout=1)


class RecoveryClient(client.BoundedClient):
    def __init__(self, *args, jobs, model_locks, started, **kwargs):
        self.jobs, self.model_locks, self.started = jobs, model_locks, started
        super().__init__(*args, **kwargs)

    def save(self):
        if not hasattr(self, "_sequence"):
            self._sequence, self._previous, self._artifacts = 0, None, {}
            self.ledger["resolved_models"] = dict(self.model_locks)
            self.ledger["resolved_model_locks"] = dict(self.model_locks)
        events = self.output.parent / (self.output.name + "_events")
        events.mkdir(exist_ok=True)
        additions = {}
        for prefix in ("request", "response", "error_response"):
            name = f"{prefix}_{len(self.ledger['attempts']):03d}.json"
            path = self.output / name
            if path.exists() and name not in self._artifacts: additions[name] = original.digest(path)
        event = {"schema": SCHEMA + "-event", "sequence": self._sequence, "previous_sha256": self._previous,
            "new_artifacts": additions, "ledger": self.redacted(self.ledger)}
        path = events / f"event_{self._sequence:05d}.json"; write(path, event)
        self._previous = original.digest(path); self._sequence += 1; self._artifacts.update(additions)
        super().save()
        if self.ledger["attempts"] and self.ledger["attempts"][-1]["status"] == "in_flight" and "elapsed_seconds" not in self.ledger["attempts"][-1]:
            ensure_time(self.started, dispatch=True)

    def submit(self, tasks, kind, backend):
        ensure_time(self.started, dispatch=True)
        attempts = self.ledger["attempts"]
        if getattr(self, "_blocked", False): raise ValueError("recovery client failed; no continuation")
        if attempts and attempts[-1]["status"] != "completed": raise ValueError("unresolved attempt; recovery cannot continue")
        if len(attempts) >= len(self.jobs): raise ValueError("frozen recovery segment exhausted")
        job = self.jobs[len(attempts)]
        payload = client.make_payload(tasks, kind, backend)
        if (kind != "support" or backend != job["backend"] or payload != job["payload"]
                or client.object_hash(payload) != job["payload_sha256"]):
            raise ValueError("dispatch differs from next exact frozen job")
        try:
            result = super().submit(tasks, kind, backend)
            record = self.ledger["attempts"][-1]
            response = read(self.output / f"response_{record['attempt']:03d}.json")
            pilot.validate_reused_response(response, payload, record)
            if backend == "jev": original.support_scores(response, job["task_ids"])
            return result
        except BaseException:
            self._blocked = True
            raise


def verify_events(directory, locks):
    directory = Path(directory); events = directory.parent / (directory.name + "_events")
    files = sorted(events.iterdir()); previous = None; last = None; artifacts = {}; hashes = {}
    if not files or [p.name for p in files] != [f"event_{i:05d}.json" for i in range(len(files))]:
        raise ValueError("recovery event stream is not contiguous")
    for sequence, path in enumerate(files):
        event = read(path); ledger = event["ledger"]
        if (set(event) != {"schema", "sequence", "previous_sha256", "new_artifacts", "ledger"}
                or event["schema"] != SCHEMA + "-event" or event["sequence"] != sequence
                or event["previous_sha256"] != previous or ledger["resolved_model_locks"] != locks
                or ledger["resolved_models"] != locks):
            raise ValueError("event hash chain/model identity differs")
        if last is None:
            if ledger["attempts"] or ledger["reservation_total_usd"] != "0": raise ValueError("initial event not empty")
        else:
            old, new = last["attempts"], ledger["attempts"]
            mutable = {"attempts", "reservation_total_usd", "actual_reported_cost_usd", "halt_reason"}
            if {k:v for k,v in ledger.items() if k not in mutable} != {k:v for k,v in last.items() if k not in mutable}:
                raise ValueError("fixed segment contract changed")
            if len(new) == len(old) + 1:
                if new[:-1] != old or (old and old[-1]["status"] != "completed") or new[-1]["status"] != "in_flight":
                    raise ValueError("attempt appended after failure or without reservation")
            elif len(new) == len(old) and old and old[-1]["status"] == "in_flight":
                if (new[:-1] != old[:-1] or new[-1]["status"] not in {"completed", "halted", "in_flight"}
                        or any(new[-1].get(k) != v for k,v in old[-1].items() if k != "status")):
                    raise ValueError("reserved attempt changed at completion")
            else: raise ValueError("invalid event transition")
        reserve = sum((Decimal(r["reserved_usd"]) for r in ledger["attempts"]), Decimal(0))
        known = sum((pilot.nonnegative_decimal(r["actual_cost_usd"]) for r in ledger["attempts"] if r.get("actual_cost_usd") is not None), Decimal(0))
        if Decimal(ledger["reservation_total_usd"]) != reserve or Decimal(ledger["actual_reported_cost_usd"]) != known:
            raise ValueError("event costs/reservations do not reconcile")
        for name, sha in event["new_artifacts"].items():
            match = re.fullmatch(r"(?:request|response|error_response)_(\d{3})\.json", name)
            if not match or int(match.group(1)) != len(ledger["attempts"]) or name in artifacts:
                raise ValueError("event artifact identity/reintroduction differs")
            artifacts[name] = sha
        previous = original.digest(path); hashes[str(path.resolve())] = previous; last = ledger
    if read(directory / "ledger.json") != last or {p.name for p in directory.iterdir()} != set(artifacts) | {"ledger.json"}:
        raise ValueError("current ledger/artifacts differ from append-only events")
    hashes = merge(hashes, {str((directory/name).resolve()):sha for name,sha in artifacts.items()},
                   {str((directory / "ledger.json").resolve()): original.digest(directory / "ledger.json")})
    pilot.verify_hashes(hashes)
    return last, hashes


def inherited_decisions(data):
    labels = {backend:{} for backend in MODELS}; scores = {}
    for entry in data["inheritance"]:
        record = read(entry["ledger_path"])["attempts"][entry["attempt"]-1]
        payload = pilot.prior_request(entry["request_path"], record, data["original_config"]["prompt_version"])
        response = read(entry["response_path"]); parsed = pilot.validate_reused_response(response,payload,record)
        backend = record["backend"]
        if set(parsed) & set(labels[backend]): raise ValueError("duplicate inherited support decisions")
        labels[backend].update(parsed)
        if backend == "jev": scores.update(original.support_scores(response,record["task_ids"]))
    return labels,scores


def collect(config, data, jobs, output, *, require_complete):
    labels,scores = inherited_decisions(data); accounting = pilot.accounting(); present = set()
    registration=Path(config["single_use_registration"])
    hashes={str(registration):original.digest(registration)}
    by_index = {j["schedule_index"]: j for j in jobs}; completed = 0; stopped = False
    base = Path(output) / "provider_calls"
    for segment in config["segments"]:
        directory = base / f"segment_{segment['segment']:03d}"
        if not directory.exists(): stopped = True; continue
        if stopped: raise ValueError("segment follows missing/failed prefix")
        present.update({directory.name,directory.name+"_events"})
        ledger, bound = verify_events(directory, config["resolved_model_locks"]); hashes = merge(hashes,bound)
        expected_caps = {"budget_usd":segment["reservation_usd"],"request_cap":segment["requests"],
            "question_cap":segment["questions"],"conservative_input_token_cap":segment["input_allowance"],
            "prompt_version":config["original_config"]["prompt_version"],"automatic_retries":0}
        if any(ledger.get(k)!=v for k,v in expected_caps.items()) or len(ledger["attempts"])>segment["requests"]:
            raise ValueError("new segment cap/contract differs")
        for number,record in enumerate(ledger["attempts"],1):
            job=by_index[segment["start"]+number-1]
            request=directory/f"request_{number:03d}.json"
            payload=pilot.prior_request(request,record,config["original_config"]["prompt_version"])
            if (record["attempt"]!=number or record["backend"]!=job["backend"] or record["kind"]!="support"
                    or record["task_ids"]!=job["task_ids"] or record["cache_key"]!=job["cache_key"]
                    or request.read_bytes()!=client.canonical_bytes(job["payload"]) or payload!=job["payload"]):
                raise ValueError("new attempt differs from exact next original schedule item")
            started_at=datetime.fromisoformat(record["started_at"])
            if started_at.tzinfo is None or (datetime.fromisoformat(SPEC["stop_at_utc"])-started_at).total_seconds()<=65:
                raise ValueError("saved attempt began outside deadline admission window")
            elapsed=record.get("elapsed_seconds")
            if ((elapsed is None and record["status"]!="in_flight")
                    or elapsed is not None and (type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0)):
                raise ValueError("invalid saved request elapsed time")
            # Cost-over-reservation failures still belong in the forensic ledger.
            for k,s in (("questions","question_count"),("input_allowance","input_allowance"),("output_allowance","output_allowance")):
                accounting[k]+=record[s]
            accounting["attempts"]+=1
            accounting["reservation_usd"]=str(Decimal(accounting["reservation_usd"])+Decimal(record["reserved_usd"]))
            if record.get("actual_cost_usd") is None: accounting["unknown_cost_attempts"]+=1
            else: accounting["known_cost_usd"]=str(Decimal(accounting["known_cost_usd"])+pilot.nonnegative_decimal(record["actual_cost_usd"]))
            if record["status"]=="completed":
                if stopped: raise ValueError("request followed terminal failure")
                response=read(directory/f"response_{number:03d}.json")
                parsed=pilot.validate_reused_response(response,payload,record); backend=job["backend"]
                if record["response_model"]!=config["resolved_model_locks"][backend] or set(parsed)&set(labels[backend]):
                    raise ValueError("cross-stage model drift or duplicate logical decision")
                labels[backend].update(parsed)
                if backend=="jev": scores.update(original.support_scores(response,job["task_ids"]))
                completed+=1
            elif record["status"] not in {"halted","in_flight"} or number!=len(ledger["attempts"]):
                raise ValueError("failed/unknown attempt is not terminal")
            else:
                stopped=True
                response_path=directory/f"response_{number:03d}.json"
                error_path=directory/f"error_response_{number:03d}.json"
                if response_path.exists() and error_path.exists(): raise ValueError("conflicting failed response artifacts")
                known=None
                for path in (response_path,error_path):
                    if path.exists():
                        value=read(path)
                        if isinstance(value.get("usage"),dict) and "cost" in value["usage"]:
                            try: known=pilot.nonnegative_decimal(value["usage"]["cost"])
                            except (ValueError,TypeError,ArithmeticError): pass
                if ((known is not None)!=(record.get("actual_cost_usd") is not None)
                        or known is not None and known!=pilot.nonnegative_decimal(record["actual_cost_usd"])):
                    raise ValueError("failed attempt cost differs from saved usage evidence")
        if ledger["halt_reason"] is not None: stopped=True
        if len(ledger["attempts"])!=segment["requests"]: stopped=True
    if base.exists() and {p.name for p in base.iterdir()}!=present:
        raise ValueError("unexpected provider segment/event artifacts")
    complete=completed==len(jobs) and not stopped
    if require_complete and not complete: raise ValueError("incomplete support recovery; no quality scores")
    if Decimal(accounting["reservation_usd"])>Decimal(config["prediction"]["reservation_usd"]):
        raise ValueError("new attempted reservation exceeds frozen plan")
    return {"labels":labels,"scores":scores,"new_accounting":accounting,"completed_new_requests":completed,
            "complete":complete,"input_sha256":hashes}


def accounting(config, execution):
    new=execution["new_accounting"]
    return {"new_accounting":new,"support_accounting":add_totals(config["prior_support_accounting"],new),
        "night_accounting":add_totals(config["prior_night_accounting"],compact_accounting(new)),
        "inherited_completed_requests":SPEC["inherited_completed_requests"],
        "completed_logical_requests":SPEC["inherited_completed_requests"]+execution["completed_new_requests"],
        "explicit_replacement_attempts":int(new["attempts"]>0),"old_unknown_attempt_refunded":False,
        "inherited_requests_charged_again":0,"automatic_retries":0}


def score(config, execution):
    if not execution["complete"]: raise ValueError("no partial scientific scoring")
    prepared,_,documents=original.preparation.load_prepared(config["prepared_dir"])
    annotations=pilot.selected_gold(config["sidecar"],prepared)
    tokenizer=pilot.AutoTokenizer.from_pretrained(config["tokenizer"],local_files_only=True,trust_remote_code=False)
    records,traces=original.replay_records(prepared,documents,annotations,execution["labels"],execution["scores"],tokenizer)
    baseline=original.pilot_audit.read_rows(ARTIFACTS/SOURCES["original_plan"]/"baseline_per_question.jsonl")
    if [r for r in records if r["method"].startswith("dense_")]!=baseline:
        raise ValueError("original dense records changed")
    return prepared,documents,records,traces,original.summarize(prepared,records)


def summary(config, execution, metrics, plan_hash):
    return {**metrics,"schema":SCHEMA,"status":"completed","all_results_available":True,
        "plan_sha256":plan_hash,"specification":SPEC,"original_plan_schema":config["original_config"]["schema"],
        "statistical_specification_sha256":config["statistical_specification_sha256"],
        "question_count":SPEC["question_count"],"family_count":SPEC["family_count"],
        "resolved_models":config["resolved_model_locks"],"input_binding_sha256":client.object_hash(merge(config["input_sha256"],execution["input_sha256"])),
        "execution_input_sha256":execution["input_sha256"],
        "answer_generation_performed":False,"test_payload_read":False,"independent_confirmation":False,
        **accounting(config,execution)}


def run(args, *, client_factory=RecoveryClient, guard_factory=HardDeadline):
    started=time.monotonic(); config,data,jobs,own=load_plan(args.plan)
    output=Path(config["run_output"]); plan_hash=own[str(Path(args.plan).resolve()/"experiment_config.json")]
    if output.exists(): raise FileExistsError("fixed recovery run exists; no second recovery")
    ensure_time(started,dispatch=True)
    write(config["single_use_registration"],{"schema":SCHEMA,"plan_sha256":plan_hash,"run_output":str(output),
        "registered_at_utc":utc_now().isoformat(),"explicit_replacement_limit":1,"subsequent_recovery_allowed":False})
    output.mkdir(parents=True,exist_ok=False); guard=guard_factory(output,started)
    batches={b["id"]:b for b in data["batches"]}; completed=0
    try:
        for segment in config["segments"]:
            ensure_time(started,dispatch=True)
            current=[j for j in jobs if segment["start"]<=j["schedule_index"]<segment["stop"]]
            bounded=client_factory(output/"provider_calls"/f"segment_{segment['segment']:03d}",jobs=current,
                model_locks=config["resolved_model_locks"],started=started,key_file=args.key_file,proxy=getattr(args,"proxy",None),
                budget_usd=segment["reservation_usd"],request_cap=segment["requests"],question_cap=segment["questions"],token_cap=segment["input_allowance"])
            for job in current:
                ensure_time(started,dispatch=True)
                bounded.submit(batches[job["batch_id"]]["tasks"],"support",job["backend"])
                completed+=1
                print(json.dumps({"new_completed_requests":completed,"new_requests":len(jobs),"inherited_requests":SPEC["inherited_completed_requests"]}),flush=True)
        pilot.verify_hashes(merge(config["input_sha256"],own))
        if {p.name for p in output.iterdir()}!={"provider_calls"}:
            raise ValueError("unexpected artifact before scientific scoring")
        execution=collect(config,data,jobs,output,require_complete=True)
        _,_,records,traces,metrics=score(config,execution)
        pilot.verify_hashes(merge(config["input_sha256"],own,execution["input_sha256"]));ensure_time(started)
        write(output/"labels.json",execution["labels"]);write(output/"raw_scores.json",execution["scores"])
        for name,rows in (("per_question.jsonl",records),("traces.jsonl",traces)):
            with (output/name).open("x",encoding="utf-8") as stream:
                for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        result=summary(config,execution,metrics,plan_hash)
        result["output_sha256"]={str(Path(p).relative_to(output)):h for p,h in tree(output).items()}
        ensure_time(started)
        result.update(completed_at_utc=utc_now().isoformat(),elapsed_seconds=time.monotonic()-started)
        write(output/"summary.json",result)
        return {"status":"completed","new_requests":completed,"logical_requests":SPEC["original_logical_requests"],"night_accounting":result["night_accounting"]}
    except BaseException as error:
        failure={"schema":SCHEMA,"status":"failed","error_class":type(error).__name__,"main_results_available":False,
                 "automatic_retries":0,"subsequent_recovery_allowed":False}
        try: failure.update(accounting(config,collect(config,data,jobs,output,require_complete=False)))
        except BaseException as audit_error:
            failure["accounting_verification_error_class"]=type(audit_error).__name__
            # Keep physical reservations even when response semantics prevent a
            # verified collector result. This is explicitly forensic, not reuse.
            try:
                forensic=original.attempted_accounting(output,pilot.accounting())
                failure["forensic_accounting_not_response_verified"]=True
                failure.update(accounting(config,{"new_accounting":forensic,"completed_new_requests":0}))
            except BaseException as accounting_error:
                failure["forensic_accounting_error_class"]=type(accounting_error).__name__
        write(output/"failure.json",failure);raise
    finally: guard.close()


def verify_completed_run(plan_dir,run_dir):
    config,data,jobs,own=load_plan(plan_dir);output=Path(run_dir).resolve()
    if str(output)!=config["run_output"]: raise ValueError("run differs from single fixed recovery output")
    registered=read(config["single_use_registration"]);plan_hash=own[str(Path(plan_dir).resolve()/"experiment_config.json")]
    if (set(registered)!={"schema","plan_sha256","run_output","registered_at_utc","explicit_replacement_limit","subsequent_recovery_allowed"}
            or registered["schema"]!=SCHEMA or registered["plan_sha256"]!=plan_hash or registered["run_output"]!=str(output)
            or registered["explicit_replacement_limit"]!=1 or registered["subsequent_recovery_allowed"] is not False):
        raise ValueError("registration differs")
    if (output/"failure.json").exists() or (output/"hard_deadline.json").exists(): raise ValueError("failed recovery has no full result")
    if {p.name for p in output.iterdir()} != {"provider_calls","summary.json",*OUTPUT_FILES}:
        raise ValueError("complete recovery top-level inventory differs")
    before=tree(output); saved=read(output/"summary.json")
    if saved["output_sha256"]!={str(Path(p).relative_to(output)):h for p,h in before.items() if Path(p)!=output/"summary.json"}:
        raise ValueError("recovery output inventory/hash differs")
    execution=collect(config,data,jobs,output,require_complete=True)
    prepared,documents,records,traces,metrics=score(config,execution)
    completed_at=datetime.fromisoformat(saved["completed_at_utc"]);registered_at=datetime.fromisoformat(registered["registered_at_utc"])
    elapsed=saved["elapsed_seconds"]
    if (completed_at.tzinfo is None or registered_at.tzinfo is None or not registered_at<=completed_at<datetime.fromisoformat(SPEC["stop_at_utc"])
            or type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=SPEC["max_execution_seconds"]):
        raise ValueError("complete recovery exceeded its execution deadline")
    expected=summary(config,execution,metrics,plan_hash)
    expected.update(output_sha256=saved["output_sha256"],completed_at_utc=saved["completed_at_utc"],elapsed_seconds=elapsed)
    if (saved!=expected or read(output/"labels.json")!=execution["labels"] or read(output/"raw_scores.json")!=execution["scores"]
            or local_answers.rows(output/"per_question.jsonl")!=records or local_answers.rows(output/"traces.jsonl")!=traces):
        raise ValueError("complete recovery decisions/metrics do not replay")
    pilot.verify_hashes(merge(config["input_sha256"],own,before,execution["input_sha256"]))
    if tree(output)!=before: raise ValueError("recovery output changed during audit")
    return config,prepared,documents,records,saved


def audit(args):
    config,prepared,documents,records,saved=verify_completed_run(args.plan,args.run)
    return {"schema":SCHEMA+"-audit","status":"verified_complete","records":len(records),
        "question_count":len(prepared["queries"]),"accounting":{k:saved[k] for k in ("new_accounting","support_accounting","night_accounting")},
        "original_failed_parent_preserved":True,"all_inputs_outputs_unchanged":True,"api_calls":0,"key_read":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for name in ("protocol","output","run-output"):p.add_argument("--"+name,required=True)
    r=sub.add_parser("run");r.add_argument("--plan",required=True);r.add_argument("--key-file",required=True);r.add_argument("--proxy")
    a=sub.add_parser("audit");a.add_argument("--plan",required=True);a.add_argument("--run",required=True)
    args=parser.parse_args();print(json.dumps({"prepare":prepare,"run":run,"audit":audit}[args.command](args),ensure_ascii=False,indent=2))
