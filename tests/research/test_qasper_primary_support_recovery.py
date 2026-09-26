"""Synthetic only: exact amended lineage, budget, transport and complete replay."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"docs/research"))
import run_qasper_primary_support_recovery as m


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    m.client.select_general_profile("qwen36plus-json")
    monkeypatch.setattr(m.client,"read_key",lambda *a:pytest.fail("key access forbidden"))
    monkeypatch.setattr(m,"utc_now",lambda:datetime(2026,9,26,20,tzinfo=timezone.utc))
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:pytest.fail("GPU query forbidden"))
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda *a,**k:pytest.fail("GPU init forbidden"))
    yield
    m.client.select_general_profile("gpt41mini")


def task(key):
    return {"id":key,"item":{"query":"Synthetic question?","unit":{"id":"u", "text":"Synthetic evidence."}}}


def response(backend,payload):
    ids=m.client.payload_task_ids(payload,backend)
    value={"id":"fake-response","provider":"TypeSafe" if backend=="jev" else "Alibaba",
           "model":m.MODELS[backend],"usage":{"cost":.00001,"input_tokens":10,"output_tokens":10}}
    if backend=="jev":
        value["answers"]={key:{"type":"choice","choice":"yes","probabilities":{"yes":.8,"no":.1,"unknown":.1}} for key in ids}
    else:
        value["choices"]=[{"finish_reason":"stop","message":{"content":json.dumps(dict.fromkeys(ids,"yes"))}}]
    return value


def job(index,batch,backend):
    payload=m.client.make_payload(batch["tasks"],"support",backend)
    reserved,inputs,outputs=m.client.reservation(payload,backend)
    return {"schedule_index":index,"batch_id":batch["id"],"backend":backend,"kind":"support",
        "task_ids":[t["id"] for t in batch["tasks"]],"payload_sha256":m.client.object_hash(payload),
        "cache_key":m.client.object_hash({"endpoint":m.client.MODELS[backend]["endpoint"],"payload":payload,"prompt_version":m.client.PROMPT_VERSION}),
        "reserved_usd":str(reserved),"input_allowance":inputs,"output_allowance":outputs}


class Guard:
    def __init__(self,*args):pass
    def close(self):pass


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    artifacts=tmp_path/"artifacts";artifacts.mkdir();monkeypatch.setattr(m,"ARTIFACTS",artifacts)
    batches=[{"id":"b0","tasks":[task("q0")]},{"id":"b1","tasks":[task("q1")]}]
    schedule=[job(0,batches[0],"general"),job(1,batches[0],"jev"),job(2,batches[1],"general")]
    old_dir=tmp_path/"old"
    def old_transport(backend,payload):
        if backend=="jev":raise TimeoutError()
        return response(backend,payload)
    old=m.client.BoundedClient(old_dir,transport=old_transport)
    old.submit(batches[0]["tasks"],"support","general")
    with pytest.raises(RuntimeError):old.submit(batches[0]["tasks"],"support","jev")
    old_total=m.pilot.accounting()
    for attempt in old.ledger["attempts"]:m.pilot.add_attempt(old_total,attempt)
    prior=m.compact_accounting(old_total)
    new=m.original.totals(schedule[1:]);night=Decimal(prior["reservation_usd"])+Decimal(new["reservation_usd"])
    spec={**m.SPEC,"original_logical_requests":3,"inherited_completed_requests":1,"previous_unknown_ordinal":2,
        "previously_unattempted_requests":1,"new_requests":2,"question_count":1,"family_count":1,
        "new_reservation_usd":new["reservation_usd"],"support_after_reservation_usd":str(night),
        "night_after_reservation_usd":str(night),"future_generation_headroom_usd":str(Decimal(5)-night)}
    monkeypatch.setattr(m,"SPEC",spec)
    def pointer(index):
        return {"schedule_index":index,"ledger_path":str(old_dir/"ledger.json"),"ledger_sha256":m.original.digest(old_dir/"ledger.json"),
            "attempt":index+1,"request_path":str(old_dir/f"request_{index+1:03d}.json"),"request_sha256":m.original.digest(old_dir/f"request_{index+1:03d}.json"),
            "cache_key":schedule[index]["cache_key"]}
    inherited={**pointer(0),"response_path":str(old_dir/"response_001.json"),"response_sha256":m.original.digest(old_dir/"response_001.json")}
    protocol=tmp_path/"protocol.md";protocol.write_text("Explicit single amended recovery.")
    config={"schema":"old-schema","schedule":schedule,"segments":m.original.segment_schedule(schedule),
        "prompt_version":m.client.PROMPT_VERSION,"prepared_dir":"synthetic-prepared","sidecar":"synthetic-sidecar",
        "tokenizer":"synthetic-tokenizer","methods":["dense_k3"],"selector":{"budget":1024}}
    data={"original_config":config,"batches":batches,"inheritance":[inherited],"explicit_replacement_of":pointer(1),
        "prior_night_accounting":prior,"prior_support_accounting":old_total,
        "input_sha256":{**m.tree(old_dir),str(protocol):m.original.digest(protocol)}}
    def source(_):
        m.pilot.verify_hashes(data["input_sha256"])
        return deepcopy(data)
    monkeypatch.setattr(m,"source_data",source)
    args=SimpleNamespace(protocol=protocol,output=tmp_path/"plan",run_output=artifacts/"run")
    m.prepare(args)
    monkeypatch.setattr(m,"score",lambda config,execution:({"queries":[{"id":"synthetic"}]},{},
        [{"method":"dense_k3","metric":1}],[],{"record_count":1,"metrics":{},"paired_comparisons":[]}))
    return SimpleNamespace(args=args,data=data,old_dir=old_dir,batches=batches,old_hashes=m.tree(old_dir))


def execute(f,transport=response):
    def factory(*args,**kwargs):
        return m.RecoveryClient(*args,**kwargs,transport=transport)
    return m.run(SimpleNamespace(plan=f.args.output,key_file="must-not-read",proxy=None),client_factory=factory,guard_factory=Guard)


def test_complete_recovery_inherits_once_replaces_unknown_once_and_preserves_parent(fixture):
    f=fixture;result=execute(f)
    assert result["new_requests"]==2 and result["logical_requests"]==3
    config,prepared,documents,rows,summary=m.verify_completed_run(f.args.output,f.args.run_output)
    assert summary["night_accounting"]["attempts"]==4
    assert summary["night_accounting"]["unknown_cost_attempts"]==1
    assert summary["support_accounting"]["questions"]==4
    assert summary["explicit_replacement_attempts"]==1
    assert summary["inherited_requests_charged_again"]==0 and summary["old_unknown_attempt_refunded"] is False
    assert summary["completed_logical_requests"]==3
    assert m.tree(f.old_dir)==f.old_hashes
    assert m.audit(SimpleNamespace(plan=f.args.output,run=f.args.run_output))["status"]=="verified_complete"
    assert config["schema"]==m.SCHEMA and config["original_config"]["schema"]=="old-schema"


def test_prepare_does_not_read_key_or_create_run(fixture):
    f=fixture;assert not f.args.run_output.exists()
    config,data,jobs,own=m.load_plan(f.args.output)
    assert len(data["inheritance"])==1 and [j["origin"] for j in jobs]==["explicit_one_time_replacement","previously_unattempted"]
    assert m.client.canonical_bytes(jobs[0]["payload"])==(f.old_dir/"request_002.json").read_bytes()
    assert not m.registration_path().exists()


@pytest.mark.parametrize("mutation",["jobs","inheritance","config","source","extra"])
def test_resealed_or_changed_inputs_fail_before_transport(fixture,mutation):
    f=fixture;p=f.args.output
    if mutation=="source":(f.old_dir/"response_001.json").write_text("changed")
    elif mutation=="extra":(p/"extra").write_text("extra")
    else:
        name={"jobs":"jobs.json","inheritance":"inheritance.json","config":"experiment_config.json"}[mutation]
        value=m.read(p/name)
        if mutation=="jobs":value[0]["payload"]["model"]="changed"
        elif mutation=="inheritance":value[0]["response_sha256"]="0"*64
        else:value["specification"]["new_requests"]=1
        (p/name).write_text(json.dumps(value),encoding="utf-8")
        config=m.read(p/"experiment_config.json")
        if mutation!="config":config["plan_files_sha256"][name]=m.original.digest(p/name)
        (p/"experiment_config.json").write_text(json.dumps(config),encoding="utf-8")
        (p/"plan_manifest.json").write_text(json.dumps({"experiment_config_sha256":m.original.digest(p/"experiment_config.json")}),encoding="utf-8")
    with pytest.raises(ValueError):execute(f,lambda *a:pytest.fail("transport forbidden"))


def test_second_run_or_renamed_resealed_run_cannot_reset_registration(fixture):
    f=fixture;execute(f)
    with pytest.raises(FileExistsError):execute(f)
    config=m.read(f.args.output/"experiment_config.json");config["run_output"]=str(f.args.run_output.parent/"renamed")
    (f.args.output/"experiment_config.json").write_text(json.dumps(config),encoding="utf-8")
    (f.args.output/"plan_manifest.json").write_text(json.dumps({"experiment_config_sha256":m.original.digest(f.args.output/"experiment_config.json")}),encoding="utf-8")
    with pytest.raises(FileExistsError):execute(f,lambda *a:pytest.fail("transport forbidden"))


@pytest.mark.parametrize("error",[TimeoutError,KeyboardInterrupt])
def test_failed_replacement_has_no_suffix_no_quality_and_keeps_unknown(fixture,monkeypatch,error):
    f=fixture;calls=[]
    def transport(*args):calls.append(1);raise error()
    monkeypatch.setattr(m,"score",lambda *a:pytest.fail("partial scores forbidden"))
    with pytest.raises((RuntimeError,KeyboardInterrupt)):execute(f,transport)
    assert len(calls)==1
    failure=m.read(f.args.run_output/"failure.json")
    assert failure["night_accounting"]["attempts"]==3
    assert failure["night_accounting"]["unknown_cost_attempts"]==2
    assert failure["subsequent_recovery_allowed"] is False
    assert not (f.args.run_output/"summary.json").exists()
    with pytest.raises(ValueError):m.verify_completed_run(f.args.output,f.args.run_output)
    with pytest.raises(FileExistsError):execute(f)
    assert m.tree(f.old_dir)==f.old_hashes


def test_failure_after_first_new_success_stops_no_partial_scores(fixture,monkeypatch):
    f=fixture;calls=[]
    def transport(b,p):
        calls.append(b)
        if len(calls)==2:raise TimeoutError()
        return response(b,p)
    monkeypatch.setattr(m,"score",lambda *a:pytest.fail("partial scores forbidden"))
    with pytest.raises(RuntimeError):execute(f,transport)
    fail=m.read(f.args.run_output/"failure.json")
    assert fail["night_accounting"]["attempts"]==4 and fail["night_accounting"]["unknown_cost_attempts"]==2
    assert fail["completed_logical_requests"]==2


@pytest.mark.parametrize("mutation",["model","provider","probabilities","task_ids"])
def test_invalid_new_response_stops_before_next_request(fixture,mutation):
    f=fixture;calls=[]
    def transport(b,p):
        calls.append(1);r=response(b,p)
        if mutation=="model":r["model"]="different-model"
        elif mutation=="provider":r["provider"]="different-provider"
        elif mutation=="probabilities":r["answers"][next(iter(r["answers"]))]["probabilities"]={"yes":.2,"no":.2,"unknown":.2}
        else:r["answers"]["extra"]={"type":"choice","choice":"yes"}
        return r
    with pytest.raises((RuntimeError,ValueError)):execute(f,transport)
    assert len(calls)==1 and not (f.args.run_output/"summary.json").exists()


def test_qwen_allowed_alias_drift_still_rejected_by_parent_model_lock(fixture):
    f=fixture;calls=[]
    def transport(b,p):
        calls.append(b);r=response(b,p)
        if b=="general":r["model"]="qwen/qwen3.6-plus-04-02"
        return r
    with pytest.raises(RuntimeError):execute(f,transport)
    assert len(calls)==2
    failure=m.read(f.args.run_output/"failure.json")
    assert failure["night_accounting"]["unknown_cost_attempts"]==1
    assert Decimal(failure["night_accounting"]["known_cost_usd"])==Decimal('.00003')


def test_changed_dispatch_job_and_duplicate_rejected_before_charge(fixture):
    f=fixture;config,data,jobs,own=m.load_plan(f.args.output)
    c=m.RecoveryClient(f.args.run_output,jobs=jobs,model_locks=m.MODELS,started=time.monotonic(),transport=response)
    with pytest.raises(ValueError):c.submit([task("wrong")],"support","jev")
    assert not c.ledger["attempts"]
    c.submit(f.batches[0]["tasks"],"support","jev")
    with pytest.raises(ValueError):c.submit(f.batches[0]["tasks"],"support","jev")
    assert len(c.ledger["attempts"])==1


def test_dispatch_deadline_prevents_key_read_registration_and_transport(fixture,monkeypatch):
    f=fixture;monkeypatch.setattr(m,"utc_now",lambda:datetime(2026,9,27,0,58,56,tzinfo=timezone.utc))
    with pytest.raises(TimeoutError):execute(f,lambda *a:pytest.fail("transport forbidden"))
    assert not m.registration_path().exists() and not f.args.run_output.exists()


def test_model_lock_present_in_initial_durable_event(fixture):
    f=fixture;execute(f)
    first=m.read(f.args.run_output/"provider_calls/segment_001_events/event_00000.json")
    assert first["ledger"]["resolved_models"]==m.MODELS
    assert first["ledger"]["attempts"]==[]


@pytest.mark.parametrize("mutation",["score","labels","fee","extra","event"])
def test_complete_audit_rejects_output_tampering_even_resealed(fixture,mutation):
    f=fixture;execute(f);out=f.args.run_output;saved=m.read(out/"summary.json")
    if mutation=="score":(out/"per_question.jsonl").write_text('{"method":"wrong"}\n')
    elif mutation=="labels":(out/"labels.json").write_text('{}')
    elif mutation=="fee":saved["night_accounting"]["reservation_usd"]="0"
    elif mutation=="extra":(out/"extra.json").write_text('{}')
    else:
        path=out/"provider_calls/segment_001_events/event_00001.json"
        event=m.read(path);event["ledger"]["attempts"][0]["cache_key"]="different"
        path.write_text(json.dumps(event),encoding="utf-8")
    saved["output_sha256"]={str(Path(p).relative_to(out)):h for p,h in m.tree(out).items() if Path(p)!=out/"summary.json"}
    (out/"summary.json").write_text(json.dumps(saved),encoding="utf-8")
    with pytest.raises(ValueError):m.verify_completed_run(f.args.output,out)


def test_registration_plan_mismatch_rejected(fixture):
    f=fixture;execute(f);path=m.registration_path();r=m.read(path);r["plan_sha256"]="changed"
    path.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(ValueError):m.verify_completed_run(f.args.output,f.args.run_output)


def test_hard_deadline_targets_own_process_and_records_failure(tmp_path,monkeypatch):
    callbacks=[];exit_codes=[]
    class FakeThread:
        def __init__(self,*,target,daemon):callbacks.append(target);assert daemon
        def start(self):pass
        def join(self,timeout):pass
    monkeypatch.setattr(m.threading,"Thread",FakeThread)
    monkeypatch.setattr(m.os,"_exit",exit_codes.append)
    guard=m.HardDeadline(tmp_path,time.monotonic())
    monkeypatch.setattr(guard.cancel,"wait",lambda seconds:False)
    callbacks[0]()
    assert exit_codes==[124]
    assert m.read(tmp_path/"hard_deadline.json")["main_results_available"] is False


def test_deadline_after_durable_reservation_preserves_unknown_without_transport(fixture,monkeypatch):
    f=fixture;original_check=m.ensure_time
    def expire(started,*,dispatch=False):
        ledger=f.args.run_output/"provider_calls/segment_001/ledger.json"
        if ledger.exists() and m.read(ledger)["attempts"]:raise TimeoutError()
        original_check(started,dispatch=dispatch)
    monkeypatch.setattr(m,"ensure_time",expire)
    with pytest.raises(TimeoutError):execute(f,lambda *a:pytest.fail("request must not leave process"))
    failure=m.read(f.args.run_output/"failure.json")
    assert failure["night_accounting"]["unknown_cost_attempts"]==2
    assert failure["new_accounting"]["attempts"]==1


def test_parent_change_during_calls_is_rejected_before_scoring(fixture,monkeypatch):
    f=fixture
    def transport(b,p):
        result=response(b,p)
        if b=="general":(f.old_dir/"response_001.json").write_text("changed")
        return result
    monkeypatch.setattr(m,"score",lambda *a:pytest.fail("changed source must not be scored"))
    with pytest.raises(ValueError):execute(f,transport)
    assert not (f.args.run_output/"per_question.jsonl").exists()


def test_invalid_raw_scores_block_later_submit_and_keep_known_fee(fixture):
    f=fixture;config,data,jobs,own=m.load_plan(f.args.output);calls=[]
    def transport(b,p):
        calls.append(b);result=response(b,p)
        result["answers"]["q0"]["probabilities"]={"yes":.2,"no":.2,"unknown":.2}
        return result
    c=m.RecoveryClient(f.args.run_output,jobs=jobs,model_locks=m.MODELS,started=time.monotonic(),transport=transport)
    with pytest.raises(ValueError):c.submit(f.batches[0]["tasks"],"support","jev")
    with pytest.raises(ValueError):c.submit(f.batches[1]["tasks"],"support","general")
    assert calls==["jev"] and c.ledger["actual_reported_cost_usd"]=="0.00001"


@pytest.mark.parametrize("field,value",[("completed_at_utc","2026-09-27T01:00:00+00:00"),("elapsed_seconds",5401)])
def test_completed_timing_tamper_rejected(fixture,field,value):
    f=fixture;execute(f);path=f.args.run_output/"summary.json";saved=m.read(path);saved[field]=value
    path.write_text(json.dumps(saved),encoding="utf-8")
    with pytest.raises(ValueError,match="deadline"):m.verify_completed_run(f.args.output,f.args.run_output)
