"""Synthetic exact inheritance, billing, model-lock and completeness tests."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"docs/research"))
import run_qasper_owner_order_answers as extension
from test_qasper_local_answer_evaluation import Tokenizer, fixture as parent_fixture, response


def fixture():
    p, old_rows, index, gold = parent_fixture()
    jobs, mapping = extension.parent.build_jobs(p, old_rows, index, Tokenizer())
    model = extension.legacy.MODEL
    ledger = {"resolved_models": {"generator": model}, "attempts": [
        {"status": "completed", "cache_key": j["cache_key"], "response_model": model,
         "request_sha256": extension.legacy.client.object_hash(j["payload"]), "response_sha256": "a"*64}
        for j in jobs]}
    rows = []
    for owner in extension.ordering.OWNERS:
        for order in extension.ordering.ORDERS:
            source = deepcopy(old_rows[0])
            source.update(scope="given_document", method=owner+"__"+order, owner_method=owner, ordering=order)
            if order == "leaf_score":
                source["selected_ids"] = ["u2"]
                source["selected_global_indices"] = [1]
                pack = extension.parent.render_pack([extension.parent.Unit(**u) for u in p["documents"]["doc"]], [1])
                source["pack_sha256"] = hashlib.sha256(pack.encode()).hexdigest()
                source["actual_evidence_tokens"] = len(pack)
            rows.append(source)
    return {"prepared": p, "parent_jobs": jobs, "parent_mapping": mapping,
        "parent_answers": {j["cache_key"]: "Blue" for j in jobs}, "parent_ledger": ledger,
        "owner_records": rows, "global_index": index, "tokenizer": "unused", "sidecar": "unused",
        "parent_model": model, "prior": extension.PRIOR.copy(), "input_sha256": {}}, gold


def built():
    data, gold = fixture()
    jobs, mapping, inherited = extension.build_jobs(data, Tokenizer())
    return data, gold, jobs, mapping, inherited


def config(jobs):
    return {"parent_resolved_model": extension.legacy.MODEL, "prior_night_accounting": extension.PRIOR.copy(),
        "answer_reservation_usd": str(sum((Decimal(j["reserved_usd"]) for j in jobs), Decimal("0")))}


def bounded(tmp_path, jobs, transport):
    return extension.InheritedAnswerClient(tmp_path/"provider_calls", jobs=jobs,
        prior_reservation=extension.PRIOR["reservation_usd"], parent_model=extension.legacy.MODEL, transport=transport)


def test_five_full_arms_inherit_only_exact_parent_payloads_and_dedup_new():
    data, _, jobs, mapping, inherited = built()
    assert len(mapping) == 5 and {r["method"] for r in mapping} == set(extension.METHODS)
    assert len(jobs) == 1 and len(inherited) == 1
    assert [r["response_origin"] for r in mapping] == ["parent"]*3+["new"]*2
    assert mapping[-1]["cache_key"] == mapping[-2]["cache_key"] == jobs[0]["cache_key"]
    assert set(inherited) < set(data["parent_answers"])
    payload = jobs[0]["payload"]
    user = json.loads(payload["messages"][1]["content"])
    assert set(user) == {"question", "evidence"} and "Red" in user["evidence"]
    assert "Blue" not in user["evidence"]


@pytest.mark.parametrize("mutation", ["missing_control", "duplicate_control", "candidate_change", "wrong_source", "pack_hash",
    "tokens", "parent_payload", "parent_cachekey", "parent_response_model", "missing_parent_answer", "duplicate_selected"])
def test_inheritance_source_or_full_payload_mismatch_rejected(mutation):
    data, _ = fixture()
    rows = data["owner_records"]
    if mutation == "missing_control": rows.pop()
    elif mutation == "duplicate_control": rows.append(deepcopy(rows[0]))
    elif mutation == "candidate_change": rows[1]["candidate_global_indices"] = [1]
    elif mutation == "wrong_source": data["global_index"][1]["doc_id"] = "other"
    elif mutation == "pack_hash": rows[1]["pack_sha256"] = "bad"
    elif mutation == "tokens": rows[1]["actual_evidence_tokens"] += 1
    elif mutation == "parent_payload":
        for job in data["parent_jobs"]: job["payload"]["temperature"] = .5
    elif mutation == "parent_cachekey": data["parent_mapping"][0]["cache_key"] = "bad"
    elif mutation == "parent_response_model":
        for r in data["parent_ledger"]["attempts"]: r["response_model"] = "qwen/qwen3.6-plus-04-02"
    elif mutation == "missing_parent_answer": data["parent_answers"] = {}
    elif mutation == "duplicate_selected": rows[1]["selected_ids"] *= 2
    with pytest.raises((ValueError, KeyError)): extension.build_jobs(data, Tokenizer())


def test_parent_resolved_model_is_locked_before_first_new_attempt(tmp_path):
    _, _, jobs, _, _ = built()
    def transport(payload):
        ledger, _ = extension.parent.verify_events(tmp_path)
        assert ledger["resolved_models"] == {"generator": extension.legacy.MODEL}
        assert ledger["resolved_model_lock"] == extension.legacy.MODEL
        assert ledger["attempts"][0]["status"] == "in_flight"
        assert Decimal(ledger["reservation_total_usd"]) > 0
        return response("Red")
    client = bounded(tmp_path, jobs, transport)
    initial = extension.parent.read(tmp_path/"attempt_ledger/event_00000.json")["ledger"]
    assert initial["attempts"] == [] and initial["resolved_models"]["generator"] == extension.legacy.MODEL
    client.submit(jobs[0])
    answers, ledger, _, complete = extension.inspect_calls(config(jobs), jobs, tmp_path, require_complete=True)
    assert complete and answers[jobs[0]["cache_key"]] == "Red"
    totals = extension.accounting(config(jobs), ledger, True)
    assert totals["new_api_calls"] == 1 and totals["night_attempts"] == 533
    assert totals["inherited_requests_charged_again"] == 0 and totals["night_unknown_cost_attempts"] == 1
    assert Decimal(totals["night_attempted_reservation_usd"]) == Decimal(extension.PRIOR["reservation_usd"])+Decimal(jobs[0]["reserved_usd"])
    assert totals["night_known_reported_cost_subtotal_usd"] == "0.272082232"


def test_allowed_alias_different_from_parent_stops_on_first_call_without_refund(tmp_path):
    _, _, jobs, _, _ = built()
    calls=[]
    def transport(payload):
        calls.append(payload)
        return response("Red", model="qwen/qwen3.6-plus-04-02")
    client=bounded(tmp_path,jobs,transport)
    with pytest.raises(RuntimeError): client.submit(jobs[0])
    with pytest.raises(ValueError): client.submit(jobs[0])
    answers, ledger, _, complete=extension.inspect_calls(config(jobs),jobs,tmp_path,require_complete=False)
    assert not complete and not answers and len(calls)==1
    assert ledger["attempts"][0]["status"]=="halted" and ledger["attempts"][0]["actual_cost_usd"]=="0.0001"
    assert Decimal(ledger["reservation_total_usd"]) == Decimal(jobs[0]["reserved_usd"])


@pytest.mark.parametrize("error", [TimeoutError, KeyboardInterrupt])
def test_unknown_new_failure_is_added_to_prior_unknown_and_never_retried(tmp_path,error):
    _, _, jobs, _, _=built()
    def transport(payload): raise error()
    client=bounded(tmp_path,jobs,transport)
    with pytest.raises((RuntimeError,KeyboardInterrupt)): client.submit(jobs[0])
    with pytest.raises(ValueError): client.submit(jobs[0])
    _, ledger, _, complete=extension.inspect_calls(config(jobs),jobs,tmp_path,require_complete=False)
    totals=extension.accounting(config(jobs),ledger,complete)
    assert not complete and totals["night_unknown_cost_attempts"]==2
    assert totals["known_generation_cost_usd"]=="0" and totals["night_known_reported_cost_subtotal_usd"]==extension.PRIOR["known_cost_usd"]


def test_parent_model_lock_tampering_is_rejected_even_with_valid_empty_chain(tmp_path):
    _, _, jobs, _, _=built()
    bounded(tmp_path,jobs,lambda _:response())
    event=extension.parent.read(tmp_path/"attempt_ledger/event_00000.json")
    event["ledger"]["resolved_model_lock"]="qwen/qwen3.6-plus-04-02"
    (tmp_path/"attempt_ledger/event_00000.json").write_text(json.dumps(event),encoding="utf-8")
    (tmp_path/"provider_calls/ledger.json").write_text(json.dumps(event["ledger"]),encoding="utf-8")
    with pytest.raises(ValueError,match="parent's actual model"):
        extension.inspect_calls(config(jobs),jobs,tmp_path,require_complete=False)


def test_official_scoring_requires_exact_needed_parent_subset_and_all_new_answers():
    data,gold,jobs,mapping,inherited=built()
    parent_answers={key:data["parent_answers"][key] for key in inherited}
    new_answers={j["cache_key"]:"Red" for j in jobs}
    records,summary=extension.score_all(data,mapping,parent_answers,new_answers,gold)
    assert len(records)==5 and len(summary["metrics"])==5 and len(summary["paired_comparisons"])==5
    assert [r["official_answer_f1"] for r in records]==[1,1,1,0,0]
    assert [(r["plus"],r["minus"]) for r in summary["paired_comparisons"]]==list(extension.PAIRS)
    assert summary["paired_comparisons"][0]["question_weighted"]["delta"]==-1
    with pytest.raises(ValueError,match="partial quality"): extension.score_all(data,mapping,parent_answers,{},gold)
    with pytest.raises(ValueError): extension.score_all(data,mapping,data["parent_answers"],new_answers,gold)
    with pytest.raises(ValueError): extension.score_all(data,mapping[:-1],parent_answers,new_answers,gold)


def fake_plan(monkeypatch,tmp_path):
    data,gold,jobs,mapping,inherited=built()
    directory=tmp_path/"plan"; directory.mkdir()
    (directory/"experiment_config.json").write_text("{}")
    own=extension.parent.hashes([directory/"experiment_config.json"])
    c={**config(jobs),"input_sha256":{},"run_output":str(tmp_path/"run"),"single_use_registration":str(tmp_path/"once.json")}
    monkeypatch.setattr(extension,"load_plan",lambda _: (c,data,jobs,mapping,inherited,own))
    monkeypatch.setattr(extension.legacy.pilot,"selected_gold",lambda *a:gold)
    return directory,Path(c["run_output"]),jobs


def test_complete_run_charges_only_new_job_and_audits_all_five_arms(tmp_path,monkeypatch):
    plan,out,jobs=fake_plan(monkeypatch,tmp_path)
    calls=[]
    def transport(payload): calls.append(payload); return response("Red")
    def factory(path,**kw): return extension.InheritedAnswerClient(path,**kw,transport=transport)
    args=SimpleNamespace(plan=plan,key_file="never-read",proxy=None)
    summary=extension.run(args,client_factory=factory)
    verified=extension.audit(SimpleNamespace(plan=plan,run=out,allow_incomplete=False))
    assert verified["status"]=="verified_complete" and len(calls)==1
    assert summary["record_count"]==5 and summary["inherited_logical_predictions"]==3 and summary["new_logical_predictions"]==2
    assert summary["inherited_unique_payloads"]==1 and summary["new_api_calls"]==1
    assert len(extension.parent.read(out/"answers.json"))==2
    with pytest.raises(FileExistsError): extension.run(args,client_factory=factory)
    changed=extension.parent.read(out/"summary.json"); changed["night_unknown_cost_attempts"]=0
    (out/"summary.json").write_text(json.dumps(changed),encoding="utf-8")
    with pytest.raises(ValueError,match="metadata"): extension.audit(SimpleNamespace(plan=plan,run=out,allow_incomplete=False))


def test_failed_extension_does_not_expose_inherited_only_quality(tmp_path,monkeypatch):
    plan,out,jobs=fake_plan(monkeypatch,tmp_path)
    def transport(payload): raise TimeoutError()
    def factory(path,**kw): return extension.InheritedAnswerClient(path,**kw,transport=transport)
    monkeypatch.setattr(extension,"score_all",lambda *a:pytest.fail("partial extension must not score inherited subset"))
    args=SimpleNamespace(plan=plan,key_file="never-read",proxy=None)
    with pytest.raises(RuntimeError): extension.run(args,client_factory=factory)
    assert not (out/"summary.json").exists() and not (out/"per_question.jsonl").exists()
    verified=extension.audit(SimpleNamespace(plan=plan,run=out,allow_incomplete=True))
    assert verified["status"]=="verified_incomplete" and verified["accounting"]["night_unknown_cost_attempts"]==2
    with pytest.raises(FileExistsError): extension.run(args,client_factory=factory)


def test_inherited_budget_applies_before_key_read_or_new_client_creation(tmp_path):
    _,_,jobs,_,_=built()
    with pytest.raises(ValueError):
        extension.InheritedAnswerClient(tmp_path/"calls",prior_reservation="4.999999",jobs=jobs,
            parent_model=extension.legacy.MODEL,key_file="must-not-read",transport=lambda _:response())
    assert not (tmp_path/"calls").exists()


def test_fixed_cutoff_precedes_new_network_dispatch(tmp_path,monkeypatch):
    _,_,jobs,_,_=built()
    client=bounded(tmp_path,jobs,lambda _:response())
    client.transport=None
    class Ended:
        @staticmethod
        def now(tz): return extension.datetime.fromisoformat(extension.legacy.STOP_AT)
        @staticmethod
        def fromisoformat(value): return extension.datetime.fromisoformat(value)
    monkeypatch.setattr(extension.legacy,"datetime",Ended)
    monkeypatch.setattr(client.opener,"open",lambda *a,**k:pytest.fail("network after cutoff"))
    with pytest.raises(TimeoutError): client.submit(jobs[0])
    assert client.ledger["attempts"]==[] and client.ledger["reservation_total_usd"]=="0"


def prepared_plan(monkeypatch,tmp_path):
    data,gold,jobs,mapping,inherited=built()
    spec={**extension.SPEC,"questions":1,"families":1,"logical_predictions":5,"control_logical_predictions":2,
        "new_unique_requests":1,"inherited_logical_predictions":3,"new_logical_predictions":2,
        "inherited_unique_payloads":1,"unique_payloads":2}
    monkeypatch.setattr(extension,"SPEC",spec)
    monkeypatch.setattr(extension,"RESERVATION",config(jobs)["answer_reservation_usd"])
    monkeypatch.setattr(extension,"source_data",lambda _:data)
    monkeypatch.setattr(extension.parent.metadata.runner.AutoTokenizer,"from_pretrained",lambda *a,**k:Tokenizer())
    paths={name:str(tmp_path/"sources"/name) for name in extension.DEFAULTS}
    for p in paths.values(): Path(p).mkdir(parents=True)
    args=SimpleNamespace(**paths,output=tmp_path/"plan",run_output=tmp_path/"run")
    planned=extension.plan(args)
    return args,planned


def test_plan_roundtrip_binds_all_logical_predictions_and_only_new_jobs(tmp_path,monkeypatch):
    args,planned=prepared_plan(monkeypatch,tmp_path)
    config,data,jobs,mapping,inherited,own=extension.load_plan(args.output)
    assert planned==config and len(mapping)==5 and len(jobs)==1 and len(inherited)==1
    assert set(p.name for p in args.output.iterdir())==set(extension.PLAN_FILES)
    assert not args.run_output.exists() and not Path(config["single_use_registration"]).exists()
    assert Decimal(config["night_reservation_usd"])==Decimal(extension.PRIOR["reservation_usd"])+Decimal(config["answer_reservation_usd"])
    with pytest.raises(FileExistsError): extension.plan(args)


@pytest.mark.parametrize("mutation",["new_payload","inheritance","mapping","prior_reset","model_lock","counts"])
def test_plan_reseal_cannot_change_frozen_inheritance_or_accounting(tmp_path,monkeypatch,mutation):
    args,config=prepared_plan(monkeypatch,tmp_path)
    path=args.output
    if mutation=="new_payload":
        jobs=extension.parent.read(path/"jobs.json");jobs[0]["payload"]["temperature"]=.8
        (path/"jobs.json").write_text(json.dumps(jobs),encoding="utf-8")
    elif mutation=="inheritance":
        inheritance=extension.parent.read(path/"inheritance.json")
        next(iter(inheritance.values()))["parent_request_ordinal"]+=1
        (path/"inheritance.json").write_text(json.dumps(inheritance),encoding="utf-8")
    elif mutation=="mapping":
        rows=extension.parent.rows(path/"mapping.jsonl");rows[0]["response_origin"]="new"
        (path/"mapping.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n",encoding="utf-8")
    elif mutation=="prior_reset": config["prior_night_accounting"]={**config["prior_night_accounting"],"reservation_usd":"0"}
    elif mutation=="model_lock": config["parent_resolved_model"]="qwen/qwen3.6-plus-04-02"
    elif mutation=="counts": config["inherited_unique_payloads"]+=1
    config["plan_files_sha256"]={n:extension.legacy.digest(path/n) for n in extension.PLAN_FILES[2:]}
    (path/"experiment_config.json").write_text(json.dumps(config),encoding="utf-8")
    (path/"plan_manifest.json").write_text(json.dumps({"schema":extension.SCHEMA,
        "experiment_config_sha256":extension.legacy.digest(path/"experiment_config.json")}),encoding="utf-8")
    with pytest.raises(ValueError): extension.load_plan(path)
