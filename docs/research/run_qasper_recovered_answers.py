"""Complete primary answer stage after the explicit support recovery amendment.

Offline prepare requires a complete audited recovery. Exact completed generation
payloads may be inherited; only the whole remaining schedule can pass admission.
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
import threading
import time
from types import SimpleNamespace

import run_qasper_primary_support_recovery as recovery
import run_qasper_owner_order_answers as inherited

parent, legacy = inherited.parent, inherited.legacy
SCHEMA = "slac-qasper-recovered-primary-answers-v1"
METHODS, PAIRS = legacy.METHODS, parent.bootstrap.ANSWER_PAIRS
METRICS = ("official_answer_f1", "actual_evidence_tokens")
MODEL = "qwen/qwen3.6-plus"
PRIOR_RESERVATION, HEADROOM = Decimal("4.1353945925"), Decimal("0.8646054075")
SPEC = {"methods":list(METHODS),"pairs":[list(p) for p in PAIRS],"metrics":list(METRICS),
    "questions":77,"families":24,"logical_predictions":462,"paired_intervals":60,
    "prompt_version":legacy.PROMPT_VERSION,"generator":legacy.MODEL,"resolved_model":MODEL,
    "provider":"Alibaba","maximum_output_tokens":512,"reasoning":False,"prices":legacy.PRICES,
    "budget_bge_evidence_tokens":1024,"max_selected_units":3,"scope":"given_document",
    "prior_reservation_usd":str(PRIOR_RESERVATION),"new_generation_cap_usd":str(HEADROOM),"night_cap_usd":"5",
    "stop_at_utc":legacy.STOP_AT,"request_deadline_seconds":65,"dispatch_margin_seconds":65,
    "max_execution_seconds":5400,"automatic_retries":0,"partial_quality_scores":False,
    "cache_identity":"endpoint, prompt version, complete canonical payload bytes and audited completed response",
    "cache_interpretation":"experimental sharing, not measured production cache speedup or repeated-generation variation",
    "bootstrap_seed":20260927,"bootstrap_replicates":10000,"bootstrap":"shared whole-family PCG64 multinomial; two-sided linear percentile 95%",
    "multiple_comparison_adjustment":"none","independent_confirmation":False,"test_payload_read":False,
    "support_recovery_amendment_required":True,"old_unknown_attempt_refunded":False}
PLAN_FILES = ("experiment_config.json","plan_manifest.json","jobs.json","mapping.jsonl","inheritance.json")
CACHE_SOURCES = {key:recovery.SOURCES[key] for key in ("local_plan","local_run","local_audit","owner_plan","owner_run","owner_audit")}


def registration_path():
    return parent.ARTIFACTS/"qasper-recovered-primary-answers-registration-01.json"


def utc_now():
    return datetime.now(timezone.utc)


def ensure_time(started, *, dispatch=False):
    margin=SPEC["dispatch_margin_seconds"] if dispatch else 0
    if ((datetime.fromisoformat(SPEC["stop_at_utc"])-utc_now()).total_seconds() <= margin
            or time.monotonic()-started >= SPEC["max_execution_seconds"]-margin):
        raise TimeoutError("primary answer execution or absolute 09:00 deadline reached")


class Deadline:
    """Hard watchdog includes blocked reads; in-flight reservations remain intact."""
    def __init__(self,output,started,request=False):
        self.cancel=threading.Event()
        seconds=min((datetime.fromisoformat(SPEC["stop_at_utc"])-utc_now()).total_seconds(),
            SPEC["max_execution_seconds"]-(time.monotonic()-started))
        if request: seconds=min(seconds,SPEC["request_deadline_seconds"])
        def stop():
            if not self.cancel.wait(max(0,seconds)):
                try:
                    parent.write(Path(output)/"hard_deadline.json",{"schema":SCHEMA,"status":"failed_hard_deadline",
                        "request_deadline":request,"main_results_available":False,"automatic_retries":0})
                finally: os._exit(124)
        self.thread=threading.Thread(target=stop,daemon=True); self.thread.start()

    def close(self):
        self.cancel.set(); self.thread.join(timeout=1)


def validate_prior(prior):
    if (type(prior.get("attempts")) is not int or prior["attempts"]!=683 or Decimal(prior["reservation_usd"])!=PRIOR_RESERVATION
            or type(prior.get("unknown_cost_attempts")) is not int or prior["unknown_cost_attempts"]!=1
            or not Decimal("0.273198707") <= legacy.pilot.nonnegative_decimal(prior["known_cost_usd"]) <= PRIOR_RESERVATION):
        raise ValueError("complete recovery night accounting differs; no unknown refund")


def cache_source(paths,origin):
    """Transport replay validates all ancestors, even if only a subset is reused."""
    extension=origin=="owner"
    plan,run,audit=(Path(paths[f"{origin}_{kind}"]) for kind in ("plan","run","audit"))
    _,bindings=recovery.released_generation(plan,run,audit,extension=extension)
    config=parent.read(plan/"experiment_config.json"); jobs=parent.read(plan/"jobs.json")
    inspect=inherited.inspect_calls if extension else parent.inspect_calls
    answers,ledger,calls,complete=inspect(config,jobs,run,require_complete=True)
    if not complete or ledger["resolved_models"]!={"generator":MODEL}:
        raise ValueError("cached generation must use the exact inherited actual model")
    result={}
    for ordinal,(job,attempt) in enumerate(zip(jobs,ledger["attempts"],strict=True),1):
        key=job["cache_key"]
        if key in result: raise ValueError("duplicate ancestor payload key")
        raw=(run/"provider_calls"/f"request_{ordinal:03d}.json").read_bytes()
        if raw!=legacy.client.canonical_bytes(job["payload"]): raise ValueError("ancestor complete request bytes differ")
        result[key]={"job":job,"answer":answers[key],"origin":origin,
            "request_ordinal":ordinal,"request_sha256":attempt["request_sha256"],
            "response_sha256":attempt["response_sha256"],"resolved_model":MODEL}
    return result,parent.merge(bindings,calls)


def source_data(paths):
    before=parent.merge(*(inherited.tree(path) for path in paths.values()))
    inventory=dict(before)
    source_files=[Path(__file__),Path(recovery.__file__),Path(inherited.__file__),Path(parent.__file__),
        parent.ROOT/"tests/research/test_qasper_recovered_answers.py",
        parent.ROOT/"docs/research/PRIMARY_SUPPORT_RECOVERY_AMENDMENT_20260927.md"]
    before=parent.merge(before,parent.hashes(source_files))
    # This is the only admission to support records: it refuses partial quality.
    config,prepared,documents,records,summary=recovery.verify_completed_run(paths["recovery_plan"],paths["recovery_run"])
    saved_audit=parent.read(Path(paths["recovery_audit"])/"audit.json")
    expected_audit={"schema":recovery.SCHEMA+"-audit","status":"verified_complete","records":len(records),
        "question_count":len(prepared["queries"]),"accounting":{k:summary[k] for k in ("new_accounting","support_accounting","night_accounting")},
        "original_failed_parent_preserved":True,"all_inputs_outputs_unchanged":True,"api_calls":0,"key_read":False}
    if (summary.get("status")!="completed" or summary.get("all_results_available") is not True
            or len(records)!=1155 or saved_audit!=expected_audit):
        raise ValueError("complete independently audited 15-method support required")
    prior=summary["night_accounting"];validate_prior(prior)
    caches,bindings={},parent.merge(before,config["input_sha256"],summary["execution_input_sha256"])
    for origin in ("local","owner"):
        cache,source_hashes=cache_source(paths,origin)
        if caches.keys() & cache.keys(): raise ValueError("ancestor new-request caches unexpectedly overlap")
        caches.update(cache);bindings=parent.merge(bindings,source_hashes)
    if len(caches)!=374: raise ValueError("both complete generation ancestors required")
    if parent.merge(*(inherited.tree(path) for path in paths.values()))!=inventory:
        raise ValueError("completed sources changed during replay")
    legacy.pilot.verify_hashes(bindings)
    return {"prepared":prepared,"documents":documents,"records":records,"cache":caches,
        "tokenizer":config["tokenizer"],"sidecar":config["sidecar"],"prepared_dir":config["prepared_dir"],
        "prior":prior,"parent_model":MODEL,"input_sha256":bindings}


def build_jobs(data,tokenizer):
    jobs,mapping=legacy.build_jobs(data["prepared"],data["records"])
    for row in mapping:
        units=[parent.Unit(**u) for u in data["prepared"]["documents"][row["doc_id"]]]
        positions={u.unit_id:i for i,u in enumerate(units)}
        actual=parent.PackCounter(tokenizer,units)([positions[uid] for uid in row["selected_ids"]])
        if actual!=row["actual_evidence_tokens"] or actual>1024:
            raise ValueError("full native pack must reproduce actual BGE token count")
    new,inheritance=[],{}
    for job in jobs:
        key=job["cache_key"]
        if key not in data["cache"]: new.append(job);continue
        cached=data["cache"][key]
        if (cached["job"]!=job or legacy.client.canonical_bytes(cached["job"]["payload"])!=legacy.client.canonical_bytes(job["payload"])
                or cached["resolved_model"]!=MODEL or not isinstance(cached["answer"],str)):
            raise ValueError("cache key cannot replace full payload/model/answer equality")
        inheritance[key]={k:cached[k] for k in ("origin","request_ordinal","request_sha256","response_sha256","resolved_model")}
    for row in mapping: row["response_origin"]=inheritance[row["cache_key"]]["origin"] if row["cache_key"] in inheritance else "new"
    return new,mapping,dict(sorted(inheritance.items()))


def scope(data,jobs,mapping,inheritance):
    validate_prior(data["prior"])
    queries=data["prepared"]["queries"]
    counts={"questions":len(queries),"families":len({q["family_id"] for q in queries}),"logical_predictions":len(mapping),
        "new_unique_requests":len(jobs),"inherited_unique_payloads":len(inheritance),
        "unique_payloads":len(jobs)+len(inheritance),"new_logical_predictions":sum(r["response_origin"]=="new" for r in mapping)}
    counts["inherited_logical_predictions"]=len(mapping)-counts["new_logical_predictions"]
    if any(counts[k]!=SPEC[k] for k in ("questions","families","logical_predictions")):
        raise ValueError("all original six methods/questions required before whole-stage budget admission")
    amount=sum((legacy.pilot.nonnegative_decimal(job["reserved_usd"]) for job in jobs),Decimal("0"))
    if amount>HEADROOM or PRIOR_RESERVATION+amount>legacy.NIGHT_CAP:
        raise ValueError("whole recovered answer stage exceeds remaining cap; no reduction or partial execution")
    return counts,amount


def prepare(args):
    output,run_output=Path(args.output).resolve(),Path(args.run_output).resolve()
    paths={k:str((parent.ARTIFACTS/v).resolve()) for k,v in CACHE_SOURCES.items()}
    paths.update({k:str(Path(getattr(args,k)).resolve()) for k in ("recovery_plan","recovery_run","recovery_audit")})
    if output.exists() or run_output.exists() or registration_path().exists(): raise FileExistsError("single-use recovered answer stage already claimed")
    if output.is_relative_to(run_output) or run_output.is_relative_to(output) or any(
            new.is_relative_to(Path(old)) or Path(old).is_relative_to(new) for new in (output,run_output) for old in paths.values()):
        raise ValueError("new plan/run overlaps immutable parent chain")
    data=source_data(paths)
    tokenizer=parent.metadata.runner.AutoTokenizer.from_pretrained(data["tokenizer"],local_files_only=True,trust_remote_code=False)
    jobs,mapping,inheritance=build_jobs(data,tokenizer); counts,amount=scope(data,jobs,mapping,inheritance)
    legacy.pilot.verify_hashes(data["input_sha256"])
    output.mkdir(parents=True,exist_ok=False)
    parent.write(output/"jobs.json",jobs);legacy.pilot.write_rows(output/"mapping.jsonl",mapping);parent.write(output/"inheritance.json",inheritance)
    config={"schema":SCHEMA,"status":"planned_not_executed","specification":SPEC,"source_paths":paths,
        "input_sha256":data["input_sha256"],"prior_night_accounting":data["prior"],"parent_resolved_model":MODEL,
        "run_output":str(run_output),"single_use_registration":str(registration_path()),"answer_reservation_usd":str(amount),
        "night_reservation_usd":str(PRIOR_RESERVATION+amount),**counts,
        "plan_files_sha256":{name:legacy.digest(output/name) for name in PLAN_FILES[2:]},"api_calls":0,"key_read":False,"gold_in_payload":False}
    parent.write(output/"experiment_config.json",config)
    parent.write(output/"plan_manifest.json",{"schema":SCHEMA,"experiment_config_sha256":legacy.digest(output/"experiment_config.json")})
    return {"status":config["status"],**counts,"answer_reservation_usd":str(amount),"night_reservation_usd":config["night_reservation_usd"],"api_calls":0}


def load_plan(directory):
    directory=Path(directory).resolve()
    own=recovery.snapshot_files(directory,PLAN_FILES)
    config=parent.read(directory/"experiment_config.json")
    if (config.get("schema")!=SCHEMA or config.get("status")!="planned_not_executed" or config.get("specification")!=SPEC
            or parent.read(directory/"plan_manifest.json")!={"schema":SCHEMA,"experiment_config_sha256":own[str(directory/"experiment_config.json")]}
            or config.get("single_use_registration")!=str(registration_path()) or config.get("parent_resolved_model")!=MODEL
            or config.get("api_calls")!=0 or config.get("key_read") is not False or config.get("gold_in_payload") is not False):
        raise ValueError("recovered answer fixed plan contract differs")
    legacy.pilot.verify_hashes(config["input_sha256"])
    data=source_data(config["source_paths"])
    tokenizer=parent.metadata.runner.AutoTokenizer.from_pretrained(data["tokenizer"],local_files_only=True,trust_remote_code=False)
    jobs,mapping,inheritance=build_jobs(data,tokenizer);counts,amount=scope(data,jobs,mapping,inheritance)
    if (data["input_sha256"]!=config["input_sha256"] or data["prior"]!=config["prior_night_accounting"]
            or any(config[k]!=v for k,v in counts.items()) or config["answer_reservation_usd"]!=str(amount)
            or config["night_reservation_usd"]!=str(PRIOR_RESERVATION+amount)
            or parent.read(directory/"jobs.json")!=jobs or parent.rows(directory/"mapping.jsonl")!=mapping
            or parent.read(directory/"inheritance.json")!=inheritance
            or config["plan_files_sha256"]!={name:own[str(directory/name)] for name in PLAN_FILES[2:]}):
        raise ValueError("whole-stage payload/response inheritance/budget replay differs")
    legacy.pilot.verify_hashes(parent.merge(config["input_sha256"],own))
    return config,data,jobs,mapping,inheritance,own


class RecoveredAnswerClient(inherited.InheritedAnswerClient):
    def __init__(self,*args,started,**kwargs):
        self.started=started
        if kwargs.get("parent_model")!=MODEL: raise ValueError("fixed actual parent model required")
        super().__init__(*args,**kwargs)

    def submit(self,job):
        ensure_time(self.started,dispatch=True)
        return super().submit(job)


def empty_ledger(config):
    return {"schema":SCHEMA+"-cache-only","prior_night_reservation_usd":config["prior_night_accounting"]["reservation_usd"],
        "night_cap_usd":"5","planned_answer_reservation_usd":"0","reservation_total_usd":"0","actual_reported_cost_usd":"0",
        "attempts":[],"resolved_models":{"generator":MODEL},"resolved_model_lock":MODEL,"halt_reason":None,
        "automatic_retries":0,"prompt_version":legacy.PROMPT_VERSION}


def inspect_calls(config,jobs,output,*,require_complete):
    if jobs:
        answers,ledger,calls,complete=inherited.inspect_calls(config,jobs,output,require_complete=require_complete)
        cutoff=datetime.fromisoformat(SPEC["stop_at_utc"])
        for attempt in ledger["attempts"]:
            started=datetime.fromisoformat(attempt["started_at"])
            if started.utcoffset() is None or (cutoff-started).total_seconds()<=SPEC["dispatch_margin_seconds"]:
                raise ValueError("saved request started outside the absolute admission window")
            elapsed=attempt.get("elapsed_seconds")
            if attempt["status"]=="completed" or elapsed is not None:
                if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
                    raise ValueError("saved request elapsed time is not finite and nonnegative")
                if attempt["status"]=="completed" and elapsed>SPEC["request_deadline_seconds"]:
                    raise ValueError("completed request exceeded its hard deadline")
        return answers,ledger,calls,complete
    path=Path(output)/"cache_only_ledger.json"
    if parent.read(path)!=empty_ledger(config) or any((Path(output)/p).exists() for p in ("provider_calls","attempt_ledger")):
        raise ValueError("zero new requests must preserve a zero-cost cache-only ledger")
    return {},empty_ledger(config),parent.hashes([path]),True


def registration_binding(config,own):
    path=Path(config["single_use_registration"])
    raw=path.read_bytes();registered=json.loads(raw,object_pairs_hook=legacy.client.unique_object)
    if (set(registered)!={"schema","run_output","plan_sha256","registered_at_utc"}
            or registered["schema"]!=SCHEMA or registered["run_output"]!=config["run_output"]
            or registered["plan_sha256"]!=next(v for p,v in own.items() if Path(p).name=="experiment_config.json")
            or datetime.fromisoformat(registered["registered_at_utc"]).utcoffset() is None):
        raise ValueError("parent-chain single-use registration differs")
    return {str(path.resolve()):hashlib.sha256(raw).hexdigest()}


def score_all(data,mapping,answers,annotations):
    questions=parent.bootstrap.questions_from(data["prepared"])
    expected={(m,*q) for m in METHODS for q in questions}
    if (len(questions)!=77 or len({q[0] for q in questions})!=24 or len(mapping)!=462
            or {(r["method"],r["family_id"],r["doc_id"],r["question_id"]) for r in mapping}!=expected
            or set(answers)!={r["cache_key"] for r in mapping}):
        raise ValueError("all 462 complete predictions required before quality scoring")
    records,legacy_metrics=legacy.score_answers(data["prepared"],mapping,answers,annotations)
    groups,draws=parent.bootstrap.family_resamples(questions)
    domain,_=parent.bootstrap.summarize_domain("answer",records,METHODS,METRICS,PAIRS,questions,groups,draws)
    officials={m["method"]:m["official_metrics"] for m in legacy_metrics["metrics"]}
    for row in domain["method_means"]:
        row.update(questions=77,families=24,official_metrics=officials[row["method"]],
            unanswerable_predictions=sum(r["method"]==row["method"] and r["predicted_answer"]=="Unanswerable" for r in records))
    return records,{"metrics":domain["method_means"],"paired_comparisons":domain["paired_comparisons"],
        "shared_resamples_sha256":hashlib.sha256(draws.tobytes()).hexdigest()}


def accounting(config,ledger,complete):
    prior=config["prior_night_accounting"]; validate_prior(prior)
    attempted=len(ledger["attempts"]);reserved=Decimal(ledger["reservation_total_usd"]);known=Decimal(ledger["actual_reported_cost_usd"])
    unknown=sum("actual_cost_usd" not in r for r in ledger["attempts"])
    return {"schema":SCHEMA,"status":"completed" if complete else "verified_incomplete","main_results_available":complete,
        "new_api_calls":attempted,"completed_requests":sum(r["status"]=="completed" for r in ledger["attempts"]),
        "known_generation_cost_usd":str(known),"unknown_generation_cost_attempts":unknown,
        "generation_attempted_reservation_usd":str(reserved),"generation_planned_reservation_usd":config["answer_reservation_usd"],
        "prior_night_accounting":prior,"night_attempts":prior["attempts"]+attempted,
        "night_attempted_reservation_usd":str(Decimal(prior["reservation_usd"])+reserved),
        "night_known_reported_cost_subtotal_usd":str(Decimal(prior["known_cost_usd"])+known),
        "night_unknown_cost_attempts":prior["unknown_cost_attempts"]+unknown,
        "inherited_requests_charged_again":0,"old_unknown_attempt_refunded":False,"parent_resolved_model":MODEL,
        "automatic_retries":0,"specification":SPEC}


def results(config,data,jobs,mapping,inheritance,own,output,*,require_complete):
    new,ledger,calls,complete=inspect_calls(config,jobs,output,require_complete=require_complete)
    if not complete: return None,None,None,ledger,calls
    calls=parent.merge(calls,registration_binding(config,own))
    if any((Path(output)/name).exists() for name in ("failure.json","hard_deadline.json")):
        raise ValueError("failed execution cannot score even a completed response set")
    legacy.pilot.verify_hashes(parent.merge(config["input_sha256"],own,calls))
    required_new={r["cache_key"] for r in mapping if r["response_origin"]=="new"}
    if set(new)!=required_new or set(inheritance)!={r["cache_key"] for r in mapping if r["response_origin"]!="new"}:
        raise ValueError("only the complete required cache subset may be inherited")
    answers={key:data["cache"][key]["answer"] for key in inheritance}|new
    records,summary=score_all(data,mapping,answers,legacy.pilot.selected_gold(data["sidecar"],data["prepared"]))
    summary.update(accounting(config,ledger,True),question_count=77,family_count=24,record_count=462,
        **{k:config[k] for k in ("new_unique_requests","inherited_unique_payloads","new_logical_predictions","inherited_logical_predictions","unique_payloads")},
        plan_sha256=next(v for p,v in own.items() if Path(p).name=="experiment_config.json"),
        input_binding_sha256=legacy.client.object_hash(parent.merge(config["input_sha256"],own,calls)))
    return records,summary,answers,ledger,calls


def run(args,*,client_factory=RecoveredAnswerClient,guard_factory=Deadline):
    started=time.monotonic()
    config,data,jobs,mapping,inheritance,own=load_plan(args.plan)
    output=Path(config["run_output"])
    if output.exists(): raise FileExistsError("single fixed answer run already exists")
    ensure_time(started,dispatch=bool(jobs))
    parent.write(config["single_use_registration"],{"schema":SCHEMA,"run_output":str(output),
        "plan_sha256":own[str(Path(args.plan).resolve()/"experiment_config.json")],"registered_at_utc":utc_now().isoformat()})
    output.mkdir(parents=True,exist_ok=False);guard=guard_factory(output,started);bounded=None
    try:
        if not jobs: parent.write(output/"cache_only_ledger.json",empty_ledger(config))
        else:
            bounded=client_factory(output/"provider_calls",prior_reservation=config["prior_night_accounting"]["reservation_usd"],
                jobs=jobs,parent_model=MODEL,started=started,key_file=args.key_file,proxy=getattr(args,"proxy",None))
            for index,job in enumerate(jobs,1):
                ensure_time(started,dispatch=True)
                request_guard=guard_factory(output,started,request=True)
                try: bounded.submit(job)
                finally: request_guard.close()
                print(json.dumps({"completed_new_requests":index,"new_unique_requests":len(jobs)}),flush=True)
        records,summary,answers,ledger,calls=results(config,data,jobs,mapping,inheritance,own,output,require_complete=True)
        legacy.pilot.verify_hashes(parent.merge(config["input_sha256"],own,calls));ensure_time(started)
        parent.write(output/"answers.json",answers);legacy.pilot.write_rows(output/"per_question.jsonl",records)
        summary["output_sha256"]={str(Path(p).relative_to(output)):v for p,v in inherited.tree(output).items()}
        ensure_time(started);parent.write(output/"summary.json",summary)
        return {"status":"completed","new_api_calls":len(jobs),"logical_predictions":462}
    except BaseException as error:
        failure={"schema":SCHEMA,"status":"failed","error_class":type(error).__name__,"main_results_available":False,"automatic_retries":0}
        if bounded is not None: failure["accounting"]=accounting(config,bounded.ledger,False)
        parent.write(output/"failure.json",failure);raise
    finally: guard.close()


def audit(args):
    config,data,jobs,mapping,inheritance,own=load_plan(args.plan);output=Path(args.run).resolve()
    if str(output)!=config["run_output"]: raise ValueError("audit must use the single fixed output")
    registration=registration_binding(config,own)
    before=inherited.tree(output)
    _,ledger,calls,complete=inspect_calls(config,jobs,output,require_complete=not args.allow_incomplete)
    success=complete and (output/"summary.json").exists() and not any((output/f).exists() for f in ("failure.json","hard_deadline.json"))
    if success:
        required={"answers.json","per_question.jsonl","summary.json"}|({"provider_calls","attempt_ledger"} if jobs else {"cache_only_ledger.json"})
        if {p.name for p in output.iterdir()}!=required: raise ValueError("complete answer top-level inventory differs")
        records,expected,answers,ledger,calls=results(config,data,jobs,mapping,inheritance,own,output,require_complete=True)
        expected["output_sha256"]={str(Path(p).relative_to(output)):h for p,h in before.items() if Path(p)!=output/"summary.json"}
        if expected!=parent.read(output/"summary.json") or records!=parent.rows(output/"per_question.jsonl") or answers!=parent.read(output/"answers.json"):
            raise ValueError("complete inherited/new scientific result or metadata replay differs")
        status="verified_complete"
    else:
        if not args.allow_incomplete: raise ValueError("incomplete run has no primary answer scores")
        status="verified_incomplete"
    legacy.pilot.verify_hashes(parent.merge(config["input_sha256"],own,calls,before,registration))
    if inherited.tree(output)!=before: raise ValueError("answer output inventory changed during audit")
    return {"schema":SCHEMA+"-audit","status":status,"accounting":accounting(config,ledger,success),
        "all_bound_inputs_outputs_unchanged":True,"parent_responses_validated_not_regenerated":True,
        "api_calls_by_audit":0,"key_read":False,"partial_quality_metrics_computed":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    for name in ("recovery-plan","recovery-run","recovery-audit","output","run-output"):p.add_argument("--"+name,required=True)
    r=sub.add_parser("run");r.add_argument("--plan",required=True);r.add_argument("--key-file");r.add_argument("--proxy")
    a=sub.add_parser("audit");a.add_argument("--plan",required=True);a.add_argument("--run",required=True);a.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args();print(json.dumps({"prepare":prepare,"run":run,"audit":audit}[args.command](args),indent=2))
