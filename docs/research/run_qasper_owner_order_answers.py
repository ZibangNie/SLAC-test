"""Bounded given-document owner-order answers with exact inherited responses.

Five complete 77-question arms; only seven new unique payloads. Plan and audit
are offline. No parent rerun, mutable parent files, retry, or partial scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import run_qasper_local_answer_evaluation as parent
import run_qasper_native_owner_order as ordering

legacy = parent.legacy
SCHEMA = "slac-qasper-owner-order-answers-v1"
METHODS = ("dense_k3", "leaf_owner_k3", "dual_owner_k3", "leaf_owner_leaf_score_k3", "dual_owner_leaf_score_k3")
PAIRS = ((METHODS[3], METHODS[1]), (METHODS[4], METHODS[2]),
         (METHODS[3], METHODS[0]), (METHODS[4], METHODS[0]), (METHODS[4], METHODS[3]))
PRIOR = {"attempts": 532, "reservation_usd": "2.8067622925", "known_cost_usd": "0.271982232", "unknown_cost_attempts": 1}
RESERVATION = "0.0258755250"
SPEC = {"methods": list(METHODS), "pairs": [list(p) for p in PAIRS],
    "primary_pairs": [list(p) for p in PAIRS[:2]], "questions": 77, "families": 24,
    "logical_predictions": 385, "control_logical_predictions": 154, "new_unique_requests": 7,
    "inherited_unique_payloads": 155, "unique_payloads": 162,
    "inherited_logical_predictions": 377, "new_logical_predictions": 8,
    "scope": "given_document", "candidate_source": "frozen CPU native owner-order control",
    "prompt_version": legacy.PROMPT_VERSION, "generator": legacy.MODEL, "maximum_output_tokens": legacy.MAX_OUTPUT,
    "resolved_model": "must equal the parent's actual resolved model", "prices": legacy.PRICES,
    "stop_at_utc": legacy.STOP_AT, "night_cap_usd": str(legacy.NIGHT_CAP),
    "cache_identity": "endpoint, prompt version and full canonical payload; inherited responses are not charged again",
    "budget_bge_evidence_tokens": 1024, "max_selected_units": 3,
    "bootstrap_seed": 20260927, "bootstrap_replicates": 10000,
    "bootstrap": "shared whole-family PCG64 multinomial draws; two-sided linear percentile 95%",
    "multiple_comparison_adjustment": "none", "posthoc_development": True,
    "independent_confirmation": False, "not_a_jev_primary_result": True,
    "test_payload_read": False, "partial_quality_scores": False, "automatic_retries": 0}
PLAN_FILES = ("experiment_config.json", "plan_manifest.json", "jobs.json", "mapping.jsonl", "inheritance.json")
DEFAULTS = {"parent_plan": "qasper-local-answer-plan-02", "parent_run": "qasper-local-answer-run-01",
    "parent_audit": "qasper-local-answer-audit-01", "owner_plan": "qasper-native-owner-order-plan-01",
    "owner_run": "qasper-native-owner-order-run-01"}


def tree(directory):
    return parent.hashes(p for p in Path(directory).rglob("*") if p.is_file())


def source_data(paths):
    """Replay saved CPU/answer sources without generating or reading credentials."""
    initial = parent.merge(*(tree(paths[k]) for k in DEFAULTS))
    source_inventory = dict(initial)
    source_files = [Path(__file__), Path(parent.__file__), Path(ordering.__file__),
        parent.ROOT / "tests/research/test_qasper_owner_order_answers.py"]
    initial = parent.merge(initial, parent.hashes(source_files))
    audit_dir = Path(paths["parent_audit"])
    release = parent.read(audit_dir / "source_binding.json")
    saved_audit = parent.read(audit_dir / "audit.json")
    if release.get("root_release") != "complete_run_reviewed_for_publication" or saved_audit.get("status") != "verified_complete":
        raise ValueError("parent answers require the complete independently released audit")
    legacy.pilot.verify_hashes(release["input_sha256"])
    proof = parent.audit(SimpleNamespace(plan=paths["parent_plan"], run=paths["parent_run"], allow_incomplete=False))
    if proof != saved_audit:
        raise ValueError("parent complete answer audit does not replay")
    owner_proof = ordering.audit(SimpleNamespace(plan=paths["owner_plan"], run=paths["owner_run"]))
    if owner_proof.get("status") != "verified" or owner_proof.get("records") != 616:
        raise ValueError("complete owner-order CPU control does not replay")
    pc = parent.read(Path(paths["parent_plan"]) / "experiment_config.json")
    prepared_dir = Path(pc["source_paths"]["prepared"])
    prepared = parent.read(prepared_dir / "prepared.json")
    owner_plan = parent.read(Path(paths["owner_plan"]) / "plan.json")
    summary = parent.read(Path(paths["parent_run"]) / "summary.json")
    ledger = parent.read(Path(paths["parent_run"]) / "provider_calls/ledger.json")
    prior = {"attempts": pc["prior_night_accounting"]["attempts"] + summary["new_api_calls"],
        "reservation_usd": summary["night_attempted_reservation_usd"],
        "known_cost_usd": str(Decimal(pc["prior_night_accounting"]["known_cost_usd"]) + Decimal(summary["known_generation_cost_usd"])),
        "unknown_cost_attempts": pc["prior_night_accounting"]["unknown_cost_attempts"] + summary["unknown_generation_cost_attempts"]}
    if prior != PRIOR or set(ledger["resolved_models"]) != {"generator"}:
        raise ValueError("inherited night accounting or actual resolved model differs")
    model = ledger["resolved_models"]["generator"]
    if model not in legacy.ALLOWED_MODELS:
        raise ValueError("parent generator is outside the fixed prompt/route contract")
    inputs = parent.merge(initial, release["input_sha256"], pc["input_sha256"], owner_plan["input_sha256"])
    if parent.merge(*(tree(paths[k]) for k in DEFAULTS)) != source_inventory:
        raise ValueError("source inventory changed during complete replays")
    legacy.pilot.verify_hashes(inputs)
    bge = parent.read(Path(pc["source_paths"]["reranker_plan"]) / "experiment_config.json")["bge_tokenizer"]
    return {"prepared": prepared, "parent_jobs": parent.read(Path(paths["parent_plan"]) / "jobs.json"),
        "parent_mapping": parent.rows(Path(paths["parent_plan"]) / "mapping.jsonl"),
        "parent_answers": parent.read(Path(paths["parent_run"]) / "answers.json"), "parent_ledger": ledger,
        "owner_records": parent.rows(Path(paths["owner_run"]) / "per_question.jsonl"),
        "global_index": parent.read(parent.ARTIFACTS / "qasper-dense-01/embedding_index.json")["candidates"],
        "tokenizer": bge, "sidecar": pc["sidecar"], "parent_model": model, "prior": prior, "input_sha256": inputs}


def render_job(query, prepared, record, index, tokenizer):
    units = [parent.Unit(**u) for u in prepared["documents"][query["doc_id"]]]
    positions = {u.unit_id: i for i, u in enumerate(units)}
    ids = record["selected_ids"]
    if (len(ids) != len(set(ids)) or len(ids) > 3 or not set(ids) <= set(positions)
            or record["family_id"] != query["family_id"] or record["doc_id"] != query["doc_id"]
            or record["question_id"] != query["question_id"]):
        raise ValueError("whole-native local pack identity differs")
    if "selected_global_indices" in record:
        selected, candidates = record["selected_global_indices"], record["candidate_global_indices"]
        if (len(selected) != len(set(selected)) or not set(selected) <= set(candidates)
                or any(type(i) is not int or not 0 <= i < len(index) or index[i]["doc_id"] != query["doc_id"] for i in candidates)
                or [(index[i]["doc_id"], index[i]["unit_id"]) for i in selected] != [(query["doc_id"], uid) for uid in ids]):
            raise ValueError("given-document control changed source or own candidates")
    chosen = [positions[uid] for uid in ids]
    if len({units[i].native_text for i in chosen}) != len(chosen):
        raise ValueError("duplicate native evidence")
    pack = parent.render_pack(units, chosen)
    tokens = parent.PackCounter(tokenizer, units)(chosen)
    if tokens > 1024 or tokens != record["actual_evidence_tokens"] or hashlib.sha256(pack.encode()).hexdigest() != record["pack_sha256"]:
        raise ValueError("complete native rendering or actual token budget differs")
    payload = legacy.make_payload(query["query"], pack)
    key = legacy.client.object_hash({"endpoint": legacy.ENDPOINT, "prompt_version": legacy.PROMPT_VERSION, "payload": payload})
    amount, allowance = legacy.reserve(payload)
    return {"cache_key": key, "payload": payload, "reserved_usd": str(amount),
        "input_allowance": allowance, "output_allowance": legacy.MAX_OUTPUT}


def build_jobs(data, tokenizer):
    prepared = data["prepared"]
    queries = {(q["doc_id"], q["question_id"]): q for q in prepared["queries"]}
    old_jobs = {j["cache_key"]: j for j in data["parent_jobs"]}
    old_rows = {(r["method"], r["doc_id"], r["question_id"]): r for r in data["parent_mapping"]}
    controls = [r for r in data["owner_records"] if r["scope"] == "given_document"]
    indexed = {(r["owner_method"], r["ordering"], r["doc_id"], r["question_id"]): r for r in controls}
    expected = {(m, o, *key) for m in ordering.OWNERS for o in ordering.ORDERS for key in queries}
    if (len(queries) != len(prepared["queries"]) or set(indexed) != expected or len(controls) != len(expected)
            or len(old_jobs) != len(data["parent_jobs"]) or len(old_rows) != len(data["parent_mapping"])):
        raise ValueError("complete unique parent and owner-order mappings required")
    ordinals = {j["cache_key"]: i for i, j in enumerate(data["parent_jobs"], 1)}
    new_jobs, mapping, inheritance = {}, [], {}
    for key, q in queries.items():
        records = {m: old_rows[(m, *key)] for m in METHODS[:3]}
        for owner, method in zip(ordering.OWNERS, METHODS[3:], strict=True):
            original = indexed[(owner, "original_owner", *key)]
            control = indexed[(owner, "leaf_score", *key)]
            if original["candidate_global_indices"] != control["candidate_global_indices"]:
                raise ValueError("frozen order-only candidate pool changed")
            original_job = render_job(q, prepared, original, data["global_index"], tokenizer)
            old_row = old_rows[(owner + "_k3", *key)]
            if any(original[k] != old_row[k] for k in ("selected_ids", "pack_sha256", "actual_evidence_tokens")) or original_job != old_jobs.get(old_row["cache_key"]):
                raise ValueError("original owner source does not reproduce the parent payload")
            records[method] = control
        for method in METHODS:
            row = records[method]
            job = render_job(q, prepared, row, data["global_index"], tokenizer)
            cache = job["cache_key"]
            if method in METHODS[:3] and row["cache_key"] != cache:
                raise ValueError("inherited baseline cache identity differs")
            if cache in old_jobs:
                if old_jobs[cache] != job or cache not in data["parent_answers"]:
                    raise ValueError("cache hash cannot replace complete payload or answer equality")
                ordinal = ordinals[cache]
                attempt = data["parent_ledger"]["attempts"][ordinal - 1]
                if attempt["status"] != "completed" or attempt["cache_key"] != cache or attempt["response_model"] != data["parent_model"]:
                    raise ValueError("inherited response identity or model differs")
                inheritance[cache] = {"parent_request_ordinal": ordinal, "request_sha256": attempt["request_sha256"],
                    "response_sha256": attempt["response_sha256"], "resolved_model": data["parent_model"]}
                origin = "parent"
            else:
                if cache in new_jobs and new_jobs[cache] != job:
                    raise ValueError("new payload hash collision")
                new_jobs[cache] = job
                origin = "new"
            mapping.append({k: q[k] for k in ("family_id", "doc_id", "question_id")} |
                {"method": method, "cache_key": cache, "response_origin": origin,
                 "selected_ids": list(row["selected_ids"]), "pack_sha256": row["pack_sha256"],
                 "actual_evidence_tokens": row["actual_evidence_tokens"]})
    return sorted(new_jobs.values(), key=lambda j: j["cache_key"]), mapping, dict(sorted(inheritance.items()))


def scope(data, jobs, mapping, inheritance):
    counts = {"questions": len(data["prepared"]["queries"]),
        "families": len({q["family_id"] for q in data["prepared"]["queries"]}),
        "logical_predictions": len(mapping), "new_unique_requests": len(jobs),
        "inherited_unique_payloads": len(inheritance), "unique_payloads": len(inheritance) + len(jobs),
        "inherited_logical_predictions": sum(r["response_origin"] == "parent" for r in mapping),
        "new_logical_predictions": sum(r["response_origin"] == "new" for r in mapping)}
    amount = sum((Decimal(j["reserved_usd"]) for j in jobs), Decimal("0"))
    if any(counts[k] != SPEC[k] for k in counts) or amount != Decimal(RESERVATION) or data["prior"] != PRIOR:
        raise ValueError("fixed complete owner-order answer scope/budget differs")
    if amount + Decimal(PRIOR["reservation_usd"]) > legacy.NIGHT_CAP:
        raise ValueError("night budget exceeded")
    return counts, amount


def plan(args):
    directory, output = Path(args.output).resolve(), Path(args.run_output).resolve()
    registration = directory.with_name(directory.name + ".single_use.json")
    paths = {k: str(Path(getattr(args, k)).resolve()) for k in DEFAULTS}
    if directory.exists() or output.exists() or registration.exists():
        raise FileExistsError("new owner-order plan/run/registration must be unused")
    if directory.is_relative_to(output) or output.is_relative_to(directory) or any(
            new.is_relative_to(Path(old)) or Path(old).is_relative_to(new) for new in (directory, output) for old in paths.values()):
        raise ValueError("new plan/run cannot overlap each other or parent source directories")
    data = source_data(paths)
    tokenizer = parent.metadata.runner.AutoTokenizer.from_pretrained(data["tokenizer"], local_files_only=True, trust_remote_code=False)
    jobs, mapping, inheritance = build_jobs(data, tokenizer)
    counts, amount = scope(data, jobs, mapping, inheritance)
    legacy.pilot.verify_hashes(data["input_sha256"])
    directory.mkdir(parents=True, exist_ok=False)
    parent.write(directory / "jobs.json", jobs); legacy.pilot.write_rows(directory / "mapping.jsonl", mapping)
    parent.write(directory / "inheritance.json", inheritance)
    config = {"schema": SCHEMA, "status": "planned", "specification": SPEC, "source_paths": paths,
        "input_sha256": data["input_sha256"], "run_output": str(output), "single_use_registration": str(registration),
        "parent_resolved_model": data["parent_model"], "prior_night_accounting": PRIOR,
        "answer_reservation_usd": str(amount), "night_reservation_usd": str(Decimal(PRIOR["reservation_usd"]) + amount),
        **counts, "plan_files_sha256": {name: legacy.digest(directory / name) for name in PLAN_FILES[2:]},
        "api_calls": 0, "key_read": False, "gold_in_payload": False}
    parent.write(directory / "experiment_config.json", config)
    parent.write(directory / "plan_manifest.json", {"schema": SCHEMA, "experiment_config_sha256": legacy.digest(directory / "experiment_config.json")})
    return config


def load_plan(directory):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != set(PLAN_FILES):
        raise ValueError("owner-order answer plan inventory differs")
    own = parent.hashes(directory / name for name in PLAN_FILES)
    config, seal = parent.read(directory / "experiment_config.json"), parent.read(directory / "plan_manifest.json")
    if (config.get("schema") != SCHEMA or config.get("status") != "planned" or config.get("specification") != SPEC
            or seal != {"schema": SCHEMA, "experiment_config_sha256": own[str(directory / "experiment_config.json")]}
            or config["single_use_registration"] != str(directory.with_name(directory.name + ".single_use.json"))
            or config.get("api_calls") != 0 or config.get("key_read") is not False or config.get("gold_in_payload") is not False):
        raise ValueError("owner-order answer frozen specification differs")
    legacy.pilot.verify_hashes(config["input_sha256"])
    data = source_data(config["source_paths"])
    tokenizer = parent.metadata.runner.AutoTokenizer.from_pretrained(data["tokenizer"], local_files_only=True, trust_remote_code=False)
    jobs, mapping, inheritance = build_jobs(data, tokenizer)
    counts, amount = scope(data, jobs, mapping, inheritance)
    if (data["input_sha256"] != config["input_sha256"] or data["parent_model"] != config["parent_resolved_model"]
            or data["prior"] != config["prior_night_accounting"] or any(config[k] != v for k, v in counts.items())
            or config["answer_reservation_usd"] != str(amount)
            or config["night_reservation_usd"] != str(Decimal(PRIOR["reservation_usd"]) + amount)
            or parent.read(directory / "jobs.json") != jobs or parent.rows(directory / "mapping.jsonl") != mapping
            or parent.read(directory / "inheritance.json") != inheritance
            or config["plan_files_sha256"] != {name: own[str(directory / name)] for name in PLAN_FILES[2:]}):
        raise ValueError("owner-order payload, inheritance, source, or accounting replay differs")
    legacy.pilot.verify_hashes(parent.merge(config["input_sha256"], own))
    return config, data, jobs, mapping, inheritance, own


class InheritedAnswerClient(parent.LocalAnswerClient):
    def __init__(self, *args, parent_model, **kwargs):
        if parent_model not in legacy.ALLOWED_MODELS:
            raise ValueError("invalid parent resolved model")
        self.parent_model = parent_model
        super().__init__(*args, **kwargs)

    def save(self):
        if not hasattr(self, "_event_sequence"):
            self.ledger["resolved_model_lock"] = self.parent_model
            self.ledger["resolved_models"] = {"generator": self.parent_model}
        super().save()


def inspect_calls(config, jobs, output, *, require_complete):
    answers, ledger, bindings, complete = parent.inspect_calls(config, jobs, output, require_complete=require_complete)
    for path in sorted((Path(output) / "attempt_ledger").iterdir()):
        event = parent.read(path)["ledger"]
        if (event.get("resolved_model_lock") != config["parent_resolved_model"]
                or event["resolved_models"] != {"generator": config["parent_resolved_model"]}):
            raise ValueError("new generation did not preserve the parent's actual model")
    return answers, ledger, bindings, complete


def score_all(data, mapping, inherited, new_answers, annotations):
    required_new = {r["cache_key"] for r in mapping if r["response_origin"] == "new"}
    required_parent = {r["cache_key"] for r in mapping if r["response_origin"] == "parent"}
    if set(new_answers) != required_new or set(inherited) != required_parent or required_new & required_parent:
        raise ValueError("all inherited and new answers are required; partial quality prohibited")
    answers = {**inherited, **new_answers}
    prepared = data["prepared"]
    questions = parent.bootstrap.questions_from(prepared)
    expected = {(m, *q) for m in METHODS for q in questions}
    if len(mapping) != len(expected) or {(r["method"],r["family_id"],r["doc_id"],r["question_id"]) for r in mapping} != expected:
        raise ValueError("all five methods and all question identities required")
    records = []
    for row in mapping:
        answer = answers[row["cache_key"]]
        refs = legacy.references_from_annotations(annotations[row["doc_id"], row["question_id"]])
        records.append({**row, "predicted_answer": answer,
            "official_answer_f1": max(legacy.token_f1_score(answer, ref["answer"]) for ref in refs)})
    tables = parent.bootstrap.validated_tables(records, METHODS, ("official_answer_f1",), questions)
    groups, draws = parent.bootstrap.family_resamples(questions)
    metrics = []
    for method in METHODS:
        selected = [tables[method][key] for key in questions]
        values = [r["official_answer_f1"] for r in selected]
        predictions, gold = {}, {}
        for row in selected:
            qid = row["question_id"]
            if qid in predictions: raise ValueError("official question identifiers collide")
            units = {u["unit_id"]: u for u in prepared["documents"][row["doc_id"]]}
            predictions[qid] = {"predicted_answer": row["predicted_answer"], "predicted_evidence": [units[x]["native_text"] for x in row["selected_ids"]]}
            gold[qid] = annotations[row["doc_id"], qid]
        metrics.append({"method": method, "questions": len(questions), "families": len(groups),
            "official_answer_f1_question_weighted": sum(values)/len(values),
            "answer_f1_family_balanced": sum(sum(values[i] for i in group)/len(group) for group in groups)/len(groups),
            "actual_evidence_tokens_question_weighted": sum(r["actual_evidence_tokens"] for r in selected)/len(selected),
            "actual_evidence_tokens_family_balanced": sum(sum(selected[i]["actual_evidence_tokens"] for i in group)/len(group) for group in groups)/len(groups),
            "unanswerable_predictions": sum(r["predicted_answer"] == "Unanswerable" for r in selected),
            "official_metrics": legacy.evaluate_qa(gold, predictions)})
    pairs = [{"plus": plus, "minus": minus, **parent.bootstrap.clustered_delta(
        [tables[plus][q]["official_answer_f1"]-tables[minus][q]["official_answer_f1"] for q in questions],
        groups, draws, "official_answer_f1")} for plus, minus in PAIRS]
    return records, {"metrics": metrics, "paired_comparisons": pairs,
        "shared_resamples_sha256": hashlib.sha256(draws.tobytes()).hexdigest()}


def accounting(config, ledger, complete):
    attempts = len(ledger["attempts"])
    reserved = Decimal(ledger["reservation_total_usd"])
    known = Decimal(ledger["actual_reported_cost_usd"])
    unknown = sum("actual_cost_usd" not in r for r in ledger["attempts"])
    return {"schema": SCHEMA, "status": "completed" if complete else "verified_incomplete",
        "main_results_available": complete, "new_api_calls": attempts,
        "completed_requests": sum(r["status"] == "completed" for r in ledger["attempts"]),
        "known_generation_cost_usd": str(known), "unknown_generation_cost_attempts": unknown,
        "generation_attempted_reservation_usd": str(reserved), "generation_planned_reservation_usd": config["answer_reservation_usd"],
        "prior_night_accounting": config["prior_night_accounting"], "night_attempts": PRIOR["attempts"] + attempts,
        "night_attempted_reservation_usd": str(Decimal(PRIOR["reservation_usd"]) + reserved),
        "night_known_reported_cost_subtotal_usd": str(Decimal(PRIOR["known_cost_usd"]) + known),
        "night_unknown_cost_attempts": PRIOR["unknown_cost_attempts"] + unknown,
        "inherited_requests_charged_again": 0, "parent_resolved_model": config["parent_resolved_model"],
        "automatic_retries": 0, "specification": SPEC}


def results(config, data, jobs, mapping, inheritance, own, output, *, require_complete):
    answers, ledger, calls, complete = inspect_calls(config, jobs, output, require_complete=require_complete)
    if not complete: return None, None, answers, ledger, calls
    inherited = {key: data["parent_answers"][key] for key in inheritance}
    records, summary = score_all(data, mapping, inherited, answers, legacy.pilot.selected_gold(data["sidecar"], data["prepared"]))
    summary.update(accounting(config, ledger, True), question_count=len(data["prepared"]["queries"]),
        family_count=len({q["family_id"] for q in data["prepared"]["queries"]}), record_count=len(mapping),
        inherited_logical_predictions=sum(r["response_origin"] == "parent" for r in mapping),
        inherited_unique_payloads=len(inheritance), new_logical_predictions=sum(r["response_origin"] == "new" for r in mapping),
        plan_sha256=next(v for p,v in own.items() if Path(p).name == "experiment_config.json"),
        input_binding_sha256=legacy.client.object_hash(parent.merge(config["input_sha256"], own, calls)))
    return records, summary, {**inherited, **answers}, ledger, calls


def run(args, *, client_factory=InheritedAnswerClient):
    config, data, jobs, mapping, inheritance, own = load_plan(args.plan)
    output = Path(config["run_output"])
    if output.exists(): raise FileExistsError("run exists; no retry or new-output rerun")
    parent.write(config["single_use_registration"], {"schema": SCHEMA,
        "plan_sha256": own[str(Path(args.plan).resolve()/"experiment_config.json")],
        "run_output": str(output), "registered_at_utc": datetime.now(timezone.utc).isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    bounded = None
    try:
        bounded = client_factory(output / "provider_calls", prior_reservation=PRIOR["reservation_usd"],
            jobs=jobs, parent_model=config["parent_resolved_model"], key_file=args.key_file, proxy=args.proxy)
        for index, job in enumerate(jobs, 1):
            bounded.submit(job)
            print(json.dumps({"completed_new_requests": index, "new_unique_requests": len(jobs),
                "known_generation_cost_usd": bounded.ledger["actual_reported_cost_usd"]}), flush=True)
        records, summary, answers, ledger, calls = results(config, data, jobs, mapping, inheritance, own, output, require_complete=True)
        legacy.pilot.verify_hashes(parent.merge(config["input_sha256"], own, calls))
        parent.write(output / "answers.json", answers); legacy.pilot.write_rows(output / "per_question.jsonl", records)
        summary["output_sha256"] = {str(p.relative_to(output)): legacy.digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        parent.write(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        failure = {"schema": SCHEMA, "status": "failed", "error_class": type(exc).__name__, "main_results_available": False, "automatic_retries": 0}
        if bounded is not None: failure["accounting"] = accounting(config, bounded.ledger, False)
        parent.write(output / "failure.json", failure)
        raise


def audit(args):
    config, data, jobs, mapping, inheritance, own = load_plan(args.plan)
    output = Path(args.run).resolve()
    if str(output) != config["run_output"]: raise ValueError("audit output differs from fixed single-use run")
    registered = parent.read(config["single_use_registration"])
    if registered["run_output"] != str(output) or registered["plan_sha256"] != own[str(Path(args.plan).resolve()/"experiment_config.json")]:
        raise ValueError("single-use registration differs")
    before = tree(output)
    answers, ledger, calls, complete = inspect_calls(config, jobs, output, require_complete=not args.allow_incomplete)
    if complete and (output / "summary.json").exists() and not (output / "failure.json").exists():
        records, expected, combined, ledger, calls = results(config, data, jobs, mapping, inheritance, own, output, require_complete=True)
        expected["output_sha256"] = {str(p.relative_to(output)): legacy.digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "summary.json"}
        if expected != parent.read(output / "summary.json") or records != parent.rows(output / "per_question.jsonl") or combined != parent.read(output / "answers.json"):
            raise ValueError("complete inherited/new scores and metadata fail replay")
        status = "verified_complete"
    else:
        if not args.allow_incomplete: raise ValueError("incomplete answers cannot expose primary scores")
        status = "verified_incomplete"
    legacy.pilot.verify_hashes(parent.merge(config["input_sha256"], own, calls, before))
    return {"status": status, "accounting": accounting(config, ledger, status == "verified_complete"),
        "all_bound_inputs_outputs_unchanged": True, "api_calls_by_audit": 0, "key_read": False,
        "parent_responses_validated_not_regenerated": True, "partial_quality_metrics_computed": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("plan")
    for name, default in DEFAULTS.items(): freeze.add_argument("--"+name.replace("_","-"), default=str(parent.ARTIFACTS/default))
    freeze.add_argument("--output", required=True); freeze.add_argument("--run-output", required=True)
    execute = commands.add_parser("run"); execute.add_argument("--plan", required=True)
    execute.add_argument("--key-file", required=True); execute.add_argument("--proxy")
    inspect = commands.add_parser("audit"); inspect.add_argument("--plan", required=True)
    inspect.add_argument("--run", required=True); inspect.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = {"plan": plan, "run": run, "audit": audit}[args.command](args)
    print(json.dumps({"status": report["status"], "new_unique_requests": report.get("new_unique_requests")}))
