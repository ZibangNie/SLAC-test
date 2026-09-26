"""Posthoc complete local-baseline Answer F1, separate from the halted JEV chain.

Plan/audit are offline. Run is single-use, keeps all 77 questions and six methods,
and reuses the frozen answer prompt, generator and bounded transport. No resume,
automatic retry, provider fallback, truncation or partial quality score.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import analyze_qasper_extended_results as bootstrap
import audit_qasper_extended_interruption as interruption
import audit_qasper_reranker_metadata as metadata
import run_qasper_answer_evaluation as legacy
import run_qasper_native_dual_index_v2 as native
import run_qasper_lexical_baseline as lexical
from run_qasper_evidence_baselines import PackCounter, Unit, render_pack


SCHEMA = "slac-qasper-local-answer-v1"
METHODS = ("dense_k3", "reranker_k3", "bm25_k3", "leaf_owner_k3", "dual_owner_k3", "empty")
PAIRS = (("reranker_k3", "dense_k3"), ("bm25_k3", "dense_k3"),
         ("leaf_owner_k3", "dense_k3"), ("dual_owner_k3", "dense_k3"),
         ("dense_k3", "empty"), ("dual_owner_k3", "leaf_owner_k3"))
EXPECTED = {"questions": 77, "families": 24, "logical_predictions": 462, "unique_requests": 367}
PRIOR = {"attempts": 165, "reservation_usd": "1.4692631425", "known_cost_usd": "0.212802982", "unknown_cost_attempts": 1}
RESERVATION = "1.3374991500"
SPEC = {"methods": list(METHODS), "pairs": [list(pair) for pair in PAIRS],
    "primary_k": 3, "posthoc_development": True, "official_test_used": False,
    "generator": legacy.MODEL, "prompt_version": legacy.PROMPT_VERSION,
    "maximum_output_tokens": legacy.MAX_OUTPUT, "prices": legacy.PRICES,
    "stop_at_utc": legacy.STOP_AT, "night_cap_usd": str(legacy.NIGHT_CAP),
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "bootstrap": "shared whole-family PCG64 multinomial draws; two-sided linear percentile 95%",
    "multiple_comparison_adjustment": "none", "partial_quality_scores": False,
    "automatic_retries": 0, "empty": "same-prompt empty-evidence abstention control, not unrestricted closed-book QA"}
PLAN_FILES = ("experiment_config.json", "plan_manifest.json", "jobs.json", "mapping.jsonl")
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "research-foundation"
DEFAULTS = {"prepared": "qasper-extended-development-prepared-01", "reranker_plan": "qasper-reranker-02",
    "reranker_run": "qasper-reranker-run-02", "native_plan": "qasper-native-dual-index-plan-02",
    "native_run": "qasper-native-dual-index-run-02", "lexical_plan": "qasper-lexical-plan-01",
    "lexical_run": "qasper-lexical-run-01", "prior_audit": "qasper-extended-interruption-audit-01"}


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=legacy.client.unique_object)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    legacy.pilot.write_json(Path(path), value)


def hashes(paths):
    return {str(Path(path).resolve()): legacy.digest(path) for path in paths}


def merge(*mappings):
    result = {}
    for mapping in mappings:
        for path, value in mapping.items():
            path = str(Path(path).resolve())
            if path in result and result[path] != value:
                raise ValueError("conflicting local-answer source binding")
            result[path] = value
    return result


def prior_accounting(directory):
    directory = Path(directory)
    own = hashes([directory / "audit.json", directory / "source_binding.json"])
    report, binding = read(directory / "audit.json"), read(directory / "source_binding.json")
    inputs = binding["input_sha256"]
    if (report.get("status") != "verified_incomplete" or report.get("attempted_requests") != 165
            or report.get("input_binding_sha256") != legacy.client.object_hash(inputs)
            or any(report["new_accounting"].get(name) != value for name, value in PRIOR.items())
            or report["terminal_attempt"]["reserved_usd"] != "0.006398280"
            or report["terminal_attempt"]["cost_known"] is not False):
        raise ValueError("historical night accounting differs; unknown reservation cannot be reset")
    legacy.pilot.verify_hashes(inputs)
    old_plan, old_run = ARTIFACTS / "qasper-extended-development-plan-01", ARTIFACTS / "qasper-extended-development-run-01"
    before = merge(bootstrap.snapshot(old_plan), bootstrap.snapshot(old_run))
    if any(inputs.get(path) != value for path, value in before.items()):
        raise ValueError("old support execution inventory differs from interruption audit")
    config, batches = interruption.stage.load_plan(old_plan)
    replay = interruption.validate_timeout_prefix(config, batches, old_run, legacy.digest(old_plan / "experiment_config.json"))
    if replay != {k: v for k, v in report.items() if k not in ("input_hashes_unchanged", "input_binding_sha256")}:
        raise ValueError("old attempted reservation does not independently replay")
    if merge(bootstrap.snapshot(old_plan), bootstrap.snapshot(old_run)) != before:
        raise ValueError("old support execution changed during accounting verification")
    legacy.pilot.verify_hashes(merge(inputs, own))
    return report["new_accounting"], merge(inputs, own)


def sources(paths, *, scientific_audit=False):
    """Bind completed local baselines; no support-label dependency or model load."""
    reranker = metadata.runner
    index_path = ARTIFACTS / "qasper-dense-01" / "embedding_index.json"
    files = [index_path, Path(paths["prepared"]) / "prepared.json", Path(paths["prepared"]) / "manifest.json"]
    source_directories = ("reranker_plan", "reranker_run", "native_plan", "native_run", "lexical_plan", "lexical_run")
    inventory = {name: sorted(str(p.resolve()) for p in Path(paths[name]).iterdir() if p.is_file()) for name in source_directories}
    for values in inventory.values():
        files.extend(values)
    files.extend(Path(module.__file__).resolve() for module in
        (legacy, legacy.client, bootstrap, metadata, native, lexical, interruption))
    files.append(Path(__file__).resolve())
    files.append(ROOT / "tests" / "research" / "test_qasper_local_answer_evaluation.py")
    before = hashes(files)
    if scientific_audit:
        proof = reranker.audit_saved_run(paths["reranker_plan"], paths["reranker_run"])
        if proof["queries"] != 77 or proof["records"] != 231:
            raise ValueError("reranker is incomplete")
        nr = native.audit(SimpleNamespace(plan=paths["native_plan"], run=paths["native_run"]))
        lr = lexical.audit(SimpleNamespace(plan=paths["lexical_plan"], run=paths["lexical_run"]))
        if nr["records"] != 462 or lr["records"] != 308:
            raise ValueError("native/lexical baselines are incomplete")
    prepared, manifest, documents = reranker.preparation.load_prepared(paths["prepared"])
    rp = reranker.load_plan(paths["reranker_plan"])
    rr = rows(Path(paths["reranker_run"]) / "per_question.jsonl")
    metadata.validate_metadata(rp, read(Path(paths["reranker_run"]) / "summary.json"), rr,
        rows(Path(paths["reranker_run"]) / "pair_scores.jsonl"),
        rows(Path(paths["reranker_plan"]) / "pair_audit.jsonl"), prepared)
    nr = rows(Path(paths["native_run"]) / "per_question.jsonl")
    lr = rows(Path(paths["lexical_run"]) / "per_question.jsonl")
    records = []
    for target, source, method, scope in (("dense_k3", nr, "leaf_direct", "given_document"),
            ("reranker_k3", rr, "bge_reranker_v2_m3_k3", None),
            ("bm25_k3", lr, "bm25", "given_document"),
            ("leaf_owner_k3", nr, "leaf_owner", "given_document"),
            ("dual_owner_k3", nr, "dual_owner", "given_document")):
        records.extend({**row, "method": target} for row in source
                       if row["method"] == method and (scope is None or row["scope"] == scope))
    index = read(index_path)["candidates"]
    inputs = merge(manifest["input_sha256"], rp["input_sha256"],
        read(Path(paths["native_run"]) / "summary.json")["input_sha256"],
        read(Path(paths["lexical_run"]) / "summary.json")["input_sha256"])
    if inventory != {name: sorted(str(p.resolve()) for p in Path(paths[name]).iterdir() if p.is_file()) for name in source_directories}:
        raise ValueError("local source inventory changed during validation")
    legacy.pilot.verify_hashes(before)
    inputs = merge(inputs, before)
    return prepared, records, index, rp["bge_tokenizer"], manifest["source_paths"]["sidecar"], inputs


def build_jobs(prepared, evidence_records, global_index, tokenizer):
    queries = {(q["doc_id"], q["question_id"]): q for q in prepared["queries"]}
    expected = {(method, *key) for key in queries for method in METHODS[:-1]}
    indexed = {}
    for row in evidence_records:
        key = row["method"], row["doc_id"], row["question_id"]
        if key in indexed or key not in expected:
            raise ValueError("duplicate or unexpected local evidence record")
        indexed[key] = row
    if len(queries) != len(prepared["queries"]) or set(indexed) != expected:
        raise ValueError("incomplete six-method question coverage")
    jobs, mapping = {}, []
    for key, q in queries.items():
        units = [Unit(**unit) for unit in prepared["documents"][key[0]]]
        by_id = {unit.unit_id: i for i, unit in enumerate(units)}
        count = PackCounter(tokenizer, units)
        for method in METHODS:
            row = indexed.get((method, *key))
            ids = row["selected_ids"] if row else []
            if len(ids) != len(set(ids)) or len(ids) > 3 or not set(ids) <= set(by_id):
                raise ValueError("invalid local native evidence identity")
            if row and method in ("dense_k3", "reranker_k3") and not set(ids) <= set(q["candidate_ids"]):
                raise ValueError("fixed-pool baseline selected outside its own candidates")
            if row and "selected_global_indices" in row:
                selected, candidates = row["selected_global_indices"], row["candidate_global_indices"]
                if (any(type(i) is not int or not 0 <= i < len(global_index) for i in selected)
                        or not set(selected) <= set(candidates)
                        or [(global_index[i]["doc_id"], global_index[i]["unit_id"]) for i in selected]
                           != [(key[0], uid) for uid in ids]):
                    raise ValueError("global native identity or own candidate membership differs")
            positions = [by_id[uid] for uid in ids]
            if len({units[i].native_text for i in positions}) != len(positions):
                raise ValueError("duplicate source-native evidence")
            pack = render_pack(units, positions)
            pack_hash = hashlib.sha256(pack.encode()).hexdigest()
            tokens = count(positions)
            if tokens > 1024 or (row and (row["pack_sha256"] != pack_hash
                    or row["actual_evidence_tokens"] != tokens or row["family_id"] != q["family_id"])):
                raise ValueError("audited native pack or actual budget differs")
            payload = legacy.make_payload(q["query"], pack)
            cache_key = legacy.client.object_hash({"endpoint": legacy.ENDPOINT,
                "prompt_version": legacy.PROMPT_VERSION, "payload": payload})
            amount, allowance = legacy.reserve(payload)
            job = {"cache_key": cache_key, "payload": payload, "reserved_usd": str(amount),
                   "input_allowance": allowance, "output_allowance": legacy.MAX_OUTPUT}
            if cache_key in jobs and jobs[cache_key] != job:
                raise ValueError("same payload hash has different complete content")
            jobs[cache_key] = job
            mapping.append({name: q[name] for name in ("family_id", "doc_id", "question_id")} |
                {"method": method, "cache_key": cache_key, "selected_ids": list(ids),
                 "pack_sha256": pack_hash, "actual_evidence_tokens": tokens})
    return sorted(jobs.values(), key=lambda job: job["cache_key"]), mapping


def validate_scope(prepared, jobs, mapping):
    values = {"questions": len(prepared["queries"]),
        "families": len({q["family_id"] for q in prepared["queries"]}),
        "logical_predictions": len(mapping), "unique_requests": len(jobs)}
    total = sum((Decimal(job["reserved_usd"]) for job in jobs), Decimal("0"))
    if values != EXPECTED or total != Decimal(RESERVATION) or total + Decimal(PRIOR["reservation_usd"]) > legacy.NIGHT_CAP:
        raise ValueError("complete local-answer scope or fixed reservation changed")
    return values, total


def plan(args):
    directory, output = Path(args.output).resolve(), Path(args.run_output).resolve()
    registration = directory.with_name(directory.name + ".single_use.json")
    if directory.exists() or output.exists() or registration.exists():
        raise FileExistsError("local-answer plan/run/registration must be unused")
    paths = {name: str(Path(getattr(args, name)).resolve()) for name in DEFAULTS}
    if directory.is_relative_to(output) or output.is_relative_to(directory):
        raise ValueError("plan and run directories must be separate")
    for new in (directory, output):
        if any(new.is_relative_to(Path(old)) or Path(old).is_relative_to(new) for old in paths.values()):
            raise ValueError("local-answer outputs overlap frozen source directories")
    prepared, records, index, bge_path, sidecar, inputs = sources(paths, scientific_audit=True)
    prior, prior_inputs = prior_accounting(paths["prior_audit"])
    tokenizer = metadata.runner.AutoTokenizer.from_pretrained(bge_path, local_files_only=True, trust_remote_code=False)
    jobs, mapping = build_jobs(prepared, records, index, tokenizer)
    counts, amount = validate_scope(prepared, jobs, mapping)
    inputs = merge(inputs, prior_inputs)
    legacy.pilot.verify_hashes(inputs)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "jobs.json", jobs)
    legacy.pilot.write_rows(directory / "mapping.jsonl", mapping)
    config = {"schema": SCHEMA, "status": "planned", "specification": SPEC,
        "source_paths": paths, "input_sha256": inputs, "sidecar": sidecar,
        "run_output": str(output), "single_use_registration": str(registration),
        "plan_files_sha256": {name: legacy.digest(directory / name) for name in ("jobs.json", "mapping.jsonl")},
        **counts, "prior_night_accounting": prior, "answer_reservation_usd": str(amount),
        "night_reservation_usd": str(Decimal(prior["reservation_usd"]) + amount),
        "api_calls": 0, "key_read": False, "gold_in_payload": False,
        "independent_confirmation": False, "not_a_jev_primary_result": True}
    write(directory / "experiment_config.json", config)
    write(directory / "plan_manifest.json", {"schema": SCHEMA,
        "experiment_config_sha256": legacy.digest(directory / "experiment_config.json")})
    return config


def load_plan(directory):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != set(PLAN_FILES):
        raise ValueError("local-answer plan file inventory differs")
    own = hashes(directory / name for name in PLAN_FILES)
    config, seal = read(directory / "experiment_config.json"), read(directory / "plan_manifest.json")
    if (config.get("schema") != SCHEMA or config.get("status") != "planned"
            or config.get("specification") != SPEC or seal.get("schema") != SCHEMA
            or seal["experiment_config_sha256"] != own[str(directory / "experiment_config.json")]):
        raise ValueError("local-answer sealed specification differs")
    if (config["single_use_registration"] != str(directory.with_name(directory.name + ".single_use.json"))
            or Path(config["run_output"]).is_relative_to(directory)
            or any(config.get(k) is not v for k, v in {"api_calls": 0, "key_read": False,
                "gold_in_payload": False, "independent_confirmation": False, "not_a_jev_primary_result": True}.items())):
        raise ValueError("single-use paths or plan metadata differ")
    legacy.pilot.verify_hashes(config["input_sha256"])
    prepared, records, index, bge, sidecar, inputs = sources(config["source_paths"])
    prior, prior_inputs = prior_accounting(config["source_paths"]["prior_audit"])
    if merge(inputs, prior_inputs) != config["input_sha256"] or prior != config["prior_night_accounting"] or sidecar != config["sidecar"]:
        raise ValueError("local-answer source or historical accounting identity differs")
    tokenizer = metadata.runner.AutoTokenizer.from_pretrained(bge, local_files_only=True, trust_remote_code=False)
    jobs, mapping = build_jobs(prepared, records, index, tokenizer)
    counts, amount = validate_scope(prepared, jobs, mapping)
    if (any(config[name] != value for name, value in counts.items()) or config["answer_reservation_usd"] != str(amount)
            or config["night_reservation_usd"] != str(Decimal(prior["reservation_usd"]) + amount)
            or read(directory / "jobs.json") != jobs or rows(directory / "mapping.jsonl") != mapping
            or config["plan_files_sha256"] != {name: own[str(directory / name)] for name in ("jobs.json", "mapping.jsonl")}):
        raise ValueError("complete jobs, mapping or reservation changed")
    legacy.pilot.verify_hashes(merge(config["input_sha256"], own))
    return config, prepared, jobs, mapping, own


class LocalAnswerClient(legacy.AnswerClient):
    """Keep an exclusive append-only event beside the legacy mutable ledger."""
    def submit(self, job):
        if self.ledger["attempts"] and self.ledger["attempts"][-1]["status"] != "completed":
            raise ValueError("unresolved previous attempt; no retry or subsequent request")
        return super().submit(job)

    def save(self):
        directory = self.output.parent / "attempt_ledger"
        directory.mkdir(exist_ok=True)
        sequence = getattr(self, "_event_sequence", 0)
        new_files = {}
        known = getattr(self, "_event_files", {})
        if self.ledger["attempts"]:
            index = len(self.ledger["attempts"])
            for prefix in ("request", "response", "error_response"):
                name = f"{prefix}_{index:03d}.json"
                if (self.output / name).exists() and name not in known:
                    new_files[name] = legacy.digest(self.output / name)
        event = {"schema": SCHEMA + "-attempt-event", "sequence": sequence,
            "previous_sha256": getattr(self, "_previous_event", None),
            "new_provider_artifact_sha256": new_files, "ledger": self.redacted(self.ledger)}
        path = directory / f"event_{sequence:05d}.json"
        write(path, event)
        self._previous_event, self._event_sequence = legacy.digest(path), sequence + 1
        self._event_files = {**known, **new_files}
        super().save()


def verify_events(output):
    output = Path(output)
    event_dir, calls = output / "attempt_ledger", output / "provider_calls"
    files = sorted(event_dir.iterdir())
    if not files or [p.name for p in files] != [f"event_{i:05d}.json" for i in range(len(files))]:
        raise ValueError("attempt event stream is missing or non-contiguous")
    bindings, artifacts, previous, last = {}, {}, None, None
    for sequence, path in enumerate(files):
        raw = path.read_bytes(); value = hashlib.sha256(raw).hexdigest()
        event = json.loads(raw, object_pairs_hook=legacy.client.unique_object)
        if (set(event) != {"schema", "sequence", "previous_sha256", "new_provider_artifact_sha256", "ledger"}
                or event["schema"] != SCHEMA + "-attempt-event" or event["sequence"] != sequence
                or event["previous_sha256"] != previous):
            raise ValueError("immutable attempt event chain differs")
        ledger = event["ledger"]
        if last is None:
            if ledger["attempts"] or ledger["reservation_total_usd"] != "0":
                raise ValueError("initial attempt event is not empty")
        else:
            old, new = last["attempts"], ledger["attempts"]
            fixed = ("schema", "prior_night_reservation_usd", "night_cap_usd", "planned_answer_reservation_usd", "automatic_retries", "prompt_version")
            if any(ledger[k] != last[k] for k in fixed):
                raise ValueError("attempt stream fixed contract changed")
            if len(new) not in (len(old), len(old) + 1) or new[:max(0, len(old) - 1)] != old[:-1]:
                raise ValueError("attempt history was rewritten")
            if len(new) == len(old) + 1:
                if new[:-1] != old or (old and old[-1]["status"] != "completed") or new[-1]["status"] != "in_flight":
                    raise ValueError("attempt appended after a failure or without prior reservation")
            elif not old or old[-1]["status"] != "in_flight" or new[-1]["status"] not in ("completed", "halted", "in_flight"):
                raise ValueError("invalid request completion transition")
            elif any(new[-1].get(k) != v for k, v in old[-1].items() if k != "status"):
                raise ValueError("reserved request identity changed at completion")
        reserved = sum((legacy.pilot.nonnegative_decimal(r["reserved_usd"]) for r in ledger["attempts"]), Decimal("0"))
        cost = sum((legacy.pilot.nonnegative_decimal(r["actual_cost_usd"]) for r in ledger["attempts"] if "actual_cost_usd" in r), Decimal("0"))
        if Decimal(ledger["reservation_total_usd"]) != reserved or Decimal(ledger["actual_reported_cost_usd"]) != cost:
            raise ValueError("attempt event accounting does not reconcile")
        for name, sha in event["new_provider_artifact_sha256"].items():
            match = re.fullmatch(r"(request|response|error_response)_(\d{3})\.json", name)
            if (not match or int(match.group(2)) != len(ledger["attempts"])
                    or not ledger["attempts"] or name in artifacts):
                raise ValueError("provider artifact reintroduced in event stream")
            artifacts[name] = sha
        bindings[str(path.resolve())] = value
        previous, last = value, ledger
    ledger = read(calls / "ledger.json")
    if ledger != last:
        raise ValueError("mutable ledger differs from last immutable event")
    if {p.name for p in calls.iterdir()} != set(artifacts) | {"ledger.json"}:
        raise ValueError("provider artifacts differ from immutable attempt stream")
    bindings = merge(bindings, {str((calls / name).resolve()): sha for name, sha in artifacts.items()}, hashes([calls / "ledger.json"]))
    legacy.pilot.verify_hashes(bindings)
    return ledger, bindings


def completed_response(job, record, response):
    usage, model, provider = response["usage"], response["model"], response.get("provider")
    cost = legacy.pilot.nonnegative_decimal(usage["cost"])
    if (model not in legacy.ALLOWED_MODELS or (provider is not None and (not isinstance(provider, str) or provider.casefold() != "alibaba"))
            or record["response_model"] != model or record["provider"] != provider
            or record["response_id"] != response.get("id") or record["usage"] != usage
            or cost != Decimal(record["actual_cost_usd"]) or cost > Decimal(job["reserved_usd"])):
        raise ValueError("completed response route, identity or cost differs")
    for target, name, maximum in (("input_tokens", "prompt_tokens", job["input_allowance"]),
                                  ("output_tokens", "completion_tokens", legacy.MAX_OUTPUT)):
        if type(usage[name]) is not int or not 0 <= usage[name] <= maximum or usage[name] != record[target]:
            raise ValueError("completed response token allowance differs")
    answer = legacy.parse_answer(response)
    if answer != record["answer"]:
        raise ValueError("saved answer differs from response")
    return answer


def inspect_calls(config, jobs, output, *, require_complete):
    ledger, bindings = verify_events(output)
    if (ledger["prior_night_reservation_usd"] != config["prior_night_accounting"]["reservation_usd"]
            or ledger["planned_answer_reservation_usd"] != config["answer_reservation_usd"]
            or ledger["night_cap_usd"] != str(legacy.NIGHT_CAP) or len(ledger["attempts"]) > len(jobs)
            or Decimal(ledger["reservation_total_usd"]) + Decimal(ledger["prior_night_reservation_usd"]) > legacy.NIGHT_CAP
            or ledger.get("automatic_retries") != 0 or ledger.get("prompt_version") != legacy.PROMPT_VERSION):
        raise ValueError("attempt ledger exceeds this plan or the prior night allowance")
    answers, models = {}, set()
    calls = Path(output) / "provider_calls"
    for index, record in enumerate(ledger["attempts"], 1):
        job = jobs[index - 1]
        raw = (calls / f"request_{index:03d}.json").read_bytes()
        if (raw != legacy.client.canonical_bytes(job["payload"]) or record["request_sha256"] != hashlib.sha256(raw).hexdigest()
                or record["attempt"] != index or any(record[name] != job[name] for name in
                    ("cache_key", "reserved_usd", "input_allowance", "output_allowance"))):
            raise ValueError("attempt request differs from the next frozen job")
        response_path = calls / f"response_{index:03d}.json"
        error_path = calls / f"error_response_{index:03d}.json"
        if response_path.exists() and error_path.exists():
            raise ValueError("one attempt has both success and HTTP-error responses")
        if record["status"] == "completed":
            if error_path.exists():
                raise ValueError("completed attempt contains an HTTP-error response")
            response_raw = (calls / f"response_{index:03d}.json").read_bytes()
            if hashlib.sha256(response_raw).hexdigest() != record["response_sha256"]:
                raise ValueError("completed response bytes changed")
            response = json.loads(response_raw, object_pairs_hook=legacy.client.unique_object)
            answers[job["cache_key"]] = completed_response(job, record, response)
            models.add(record["response_model"])
        elif record["status"] not in ("halted", "in_flight") or index != len(ledger["attempts"]):
            raise ValueError("an incomplete attempt must be terminal")
        else:
            saved = None
            if response_path.exists():
                if record.get("response_sha256") != legacy.digest(response_path):
                    raise ValueError("failed-attempt response hash differs")
                saved = read(response_path)
            elif error_path.exists():
                if record.get("error_response_file") != error_path.name:
                    raise ValueError("failed-attempt HTTP-error identity differs")
                saved = read(error_path)
            known = None
            if saved is not None and isinstance(saved.get("usage"), dict) and "cost" in saved["usage"]:
                try:
                    known = legacy.pilot.nonnegative_decimal(saved["usage"]["cost"])
                except (ValueError, TypeError, ArithmeticError):
                    pass
            if ((known is not None) != ("actual_cost_usd" in record)
                    or (known is not None and known != Decimal(record["actual_cost_usd"]))):
                raise ValueError("failed-attempt known cost differs from saved accounting evidence")
    if len(models) > 1:
        raise ValueError("generator changed within completed requests")
    complete = len(answers) == len(jobs) and not ledger.get("halt_reason")
    if require_complete and not complete:
        raise ValueError("incomplete local answer execution; no main quality scores")
    if complete:
        checked, extra, _ = legacy.verify_calls(config, jobs, output)
        if answers != checked:
            raise ValueError("complete answer replay differs")
        bindings = merge(bindings, extra)
    return answers, ledger, bindings, complete


def score_all(prepared, mapping, answers, annotations):
    if set(answers) != {r["cache_key"] for r in mapping}:
        raise ValueError("all answers required; partial quality scoring prohibited")
    questions = bootstrap.questions_from(prepared)
    expected = {(method, *key) for method in METHODS for key in questions}
    if len(mapping) != len(expected) or {(r["method"],r["family_id"],r["doc_id"],r["question_id"]) for r in mapping} != expected:
        raise ValueError("all six methods and all question identities required")
    records = []
    for row in mapping:
        predicted = answers[row["cache_key"]]
        refs = legacy.references_from_annotations(annotations[row["doc_id"], row["question_id"]])
        records.append({**row, "predicted_answer": predicted,
            "official_answer_f1": max(legacy.token_f1_score(predicted, ref["answer"]) for ref in refs)})
    tables = bootstrap.validated_tables(records, METHODS, ("official_answer_f1",), questions)
    groups, draws = bootstrap.family_resamples(questions)
    metrics = []
    for method in METHODS:
        selected = [tables[method][key] for key in questions]
        values = [r["official_answer_f1"] for r in selected]
        predictions, gold = {}, {}
        for row in selected:
            uid = row["question_id"]
            if uid in predictions:
                raise ValueError("official question identifiers collide")
            units = {u["unit_id"]: u for u in prepared["documents"][row["doc_id"]]}
            predictions[uid] = {"predicted_answer": row["predicted_answer"],
                "predicted_evidence": [units[x]["native_text"] for x in row["selected_ids"]]}
            gold[uid] = annotations[row["doc_id"], uid]
        metrics.append({"method": method, "questions": len(questions), "families": len(groups),
            "official_answer_f1_question_weighted": sum(values) / len(values),
            "answer_f1_family_balanced": sum(sum(values[i] for i in group)/len(group) for group in groups)/len(groups),
            "actual_evidence_tokens_question_weighted": sum(r["actual_evidence_tokens"] for r in selected)/len(selected),
            "actual_evidence_tokens_family_balanced": sum(sum(selected[i]["actual_evidence_tokens"] for i in group)/len(group) for group in groups)/len(groups),
            "unanswerable_predictions": sum(r["predicted_answer"] == "Unanswerable" for r in selected),
            "official_metrics": legacy.evaluate_qa(gold, predictions)})
    paired = [{"plus": plus, "minus": minus, **bootstrap.clustered_delta(
        [tables[plus][key]["official_answer_f1"]-tables[minus][key]["official_answer_f1"] for key in questions],
        groups, draws, "official_answer_f1")} for plus, minus in PAIRS]
    return records, {"metrics": metrics, "paired_comparisons": paired,
        "shared_resamples_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}


def accounting(config, ledger, complete):
    reserved = Decimal(ledger["reservation_total_usd"])
    return {"schema": SCHEMA, "status": "completed" if complete else "verified_incomplete",
        "main_results_available": complete, "new_api_calls": len(ledger["attempts"]),
        "completed_requests": sum(r["status"] == "completed" for r in ledger["attempts"]),
        "known_generation_cost_usd": ledger["actual_reported_cost_usd"],
        "unknown_generation_cost_attempts": sum("actual_cost_usd" not in r for r in ledger["attempts"]),
        "generation_attempted_reservation_usd": str(reserved),
        "generation_planned_reservation_usd": config["answer_reservation_usd"],
        "prior_night_accounting": config["prior_night_accounting"],
        "night_attempted_reservation_usd": str(Decimal(PRIOR["reservation_usd"]) + reserved),
        "automatic_retries": 0, "independent_confirmation": False, "not_a_jev_primary_result": True,
        "test_payload_read": False, "specification": SPEC}


def run(args, *, client_factory=LocalAnswerClient):
    plan_dir = Path(args.plan).resolve()
    config, prepared, jobs, mapping, own = load_plan(plan_dir)
    output = Path(config["run_output"])
    if output.exists():
        raise FileExistsError("local-answer run exists; no restart or new-output retry")
    registration = Path(config["single_use_registration"])
    write(registration, {"schema": SCHEMA, "plan_sha256": own[str(plan_dir / "experiment_config.json")],
        "run_output": str(output), "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    bounded = None
    try:
        bounded = client_factory(output / "provider_calls", prior_reservation=config["prior_night_accounting"]["reservation_usd"],
            jobs=jobs, key_file=args.key_file, proxy=args.proxy)
        for index, job in enumerate(jobs, 1):
            bounded.submit(job)
            print(json.dumps({"completed_requests": index, "unique_requests": len(jobs),
                "known_generation_cost_usd": bounded.ledger["actual_reported_cost_usd"]}), flush=True)
        answers, ledger, calls, complete = inspect_calls(config, jobs, output, require_complete=True)
        records, summary = score_all(prepared, mapping, answers, legacy.pilot.selected_gold(config["sidecar"], prepared))
        legacy.pilot.verify_hashes(merge(config["input_sha256"], own, calls))
        write(output / "answers.json", answers)
        legacy.pilot.write_rows(output / "per_question.jsonl", records)
        summary.update(accounting(config, ledger, complete), question_count=len(prepared["queries"]),
            family_count=len({q["family_id"] for q in prepared["queries"]}), record_count=len(mapping),
            plan_sha256=own[str(plan_dir / "experiment_config.json")],
            input_binding_sha256=legacy.client.object_hash(merge(config["input_sha256"], own, calls)))
        summary["output_sha256"] = {str(p.relative_to(output)): legacy.digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        failure = {"schema": SCHEMA, "status": "failed", "error_class": type(exc).__name__,
            "main_results_available": False, "automatic_retries": 0}
        if bounded is not None:
            failure["accounting"] = accounting(config, bounded.ledger, False)
        write(output / "failure.json", failure)
        raise


def audit(args):
    plan_dir, output = Path(args.plan).resolve(), Path(args.run).resolve()
    config, prepared, jobs, mapping, own = load_plan(plan_dir)
    if str(output) != config["run_output"]:
        raise ValueError("audit run differs from the single-use planned output")
    registration = read(config["single_use_registration"])
    if registration["plan_sha256"] != own[str(plan_dir / "experiment_config.json")] or registration["run_output"] != str(output):
        raise ValueError("single-use registration differs")
    initial = hashes(p for p in output.rglob("*") if p.is_file())
    answers, ledger, bindings, complete = inspect_calls(config, jobs, output, require_complete=not args.allow_incomplete)
    if complete and not (output / "failure.json").exists() and (output / "summary.json").exists():
        records, summary = score_all(prepared, mapping, answers, legacy.pilot.selected_gold(config["sidecar"], prepared))
        summary.update(accounting(config, ledger, True), question_count=len(prepared["queries"]),
            family_count=len({q["family_id"] for q in prepared["queries"]}), record_count=len(mapping),
            plan_sha256=own[str(plan_dir / "experiment_config.json")],
            input_binding_sha256=legacy.client.object_hash(merge(config["input_sha256"], own, bindings)))
        summary["output_sha256"] = {str(p.relative_to(output)): legacy.digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "summary.json"}
        if summary != read(output / "summary.json") or records != rows(output / "per_question.jsonl") or answers != read(output / "answers.json"):
            raise ValueError("complete local answer scores or metadata do not replay")
        status = "verified_complete"
    else:
        if not args.allow_incomplete:
            raise ValueError("failed local answer run cannot expose main results")
        status = "verified_incomplete"
    legacy.pilot.verify_hashes(merge(config["input_sha256"], own, bindings, initial))
    return {"status": status, "accounting": accounting(config, ledger, status == "verified_complete"),
        "all_bound_inputs_outputs_unchanged": True, "api_calls_by_audit": 0, "key_read": False,
        "partial_quality_metrics_computed": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("plan")
    for name, default in DEFAULTS.items():
        freeze.add_argument("--" + name.replace("_", "-"), default=str(ARTIFACTS / default))
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--run-output", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--key-file", required=True)
    execute.add_argument("--proxy")
    inspect = commands.add_parser("audit")
    inspect.add_argument("--plan", required=True)
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = {"plan": plan, "run": run, "audit": audit}[arguments.command](arguments)
    print(json.dumps({"status": result["status"], "api_calls_by_planning_or_audit": 0
        if arguments.command != "run" else None, "unique_requests": result.get("unique_requests")}))
