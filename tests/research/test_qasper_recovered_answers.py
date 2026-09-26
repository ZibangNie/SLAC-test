"""Synthetic primary answers: exact inheritance, complete budget and no retries."""
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"docs/research"))
import run_qasper_recovered_answers as stage
from test_qasper_local_answer_evaluation import Tokenizer,fixture as native_fixture,response


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(stage,'utc_now',lambda:datetime(2026,9,27,0,0,tzinfo=timezone.utc))
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return stage.utc_now()
    monkeypatch.setattr(stage.legacy,'datetime',Clock)


def fixture():
    prepared,_,_,gold=native_fixture()
    prepared['queries'][0]['candidate_ids']=['u1','u2']
    units=[stage.parent.Unit(**u) for u in prepared['documents']['doc']]
    records=[]
    for i,method in enumerate(stage.METHODS[:-1]):
        selected=[1] if i in (1,3,4) else [0]
        pack=stage.parent.render_pack(units,selected)
        records.append({k:prepared['queries'][0][k] for k in ('doc_id','question_id','family_id')}|
            {'method':method,'selected_ids':[units[s].unit_id for s in selected],'actual_evidence_tokens':len(pack),
             'pack_sha256':hashlib.sha256(pack.encode()).hexdigest(),'budget':1024})
    jobs,_=stage.legacy.build_jobs(prepared,records)
    cache={}
    for job in jobs:
        evidence=json.loads(job['payload']['messages'][1]['content'])['evidence']
        if 'Red' in evidence: continue
        cache[job['cache_key']]={'job':deepcopy(job),'answer':'Blue' if evidence else 'Unanswerable',
            'origin':'local' if evidence else 'owner','request_ordinal':1,'request_sha256':stage.legacy.client.object_hash(job['payload']),
            'response_sha256':'a'*64,'resolved_model':stage.MODEL}
    return {'prepared':prepared,'records':records,'cache':cache,'parent_model':stage.MODEL,'input_sha256':{},
        'prior':{'attempts':683,'reservation_usd':str(stage.PRIOR_RESERVATION),'known_cost_usd':'0.3','unknown_cost_attempts':1},
        'tokenizer':'unused','sidecar':'unused'},gold


def built():
    data,gold=fixture();jobs,mapping,inheritance=stage.build_jobs(data,Tokenizer())
    return data,gold,jobs,mapping,inheritance


def config(data,jobs):
    return {'prior_night_accounting':data['prior'],'parent_resolved_model':stage.MODEL,
        'answer_reservation_usd':str(sum((Decimal(j['reserved_usd']) for j in jobs),Decimal('0')))}


def bounded(tmp_path,data,jobs,transport):
    return stage.RecoveredAnswerClient(tmp_path/'provider_calls',prior_reservation=data['prior']['reservation_usd'],
        jobs=jobs,parent_model=stage.MODEL,transport=transport,started=time.monotonic())


def test_six_groups_keep_complete_payload_identity_and_reuse_both_ancestors():
    data,_,jobs,mapping,inheritance=built()
    assert len(mapping)==6 and len(jobs)==1 and len(inheritance)==2
    assert {r['method'] for r in mapping}==set(stage.METHODS)
    assert [r['response_origin'] for r in mapping]==['local','new','local','new','new','owner']
    assert set(inheritance)==set(data['cache'])
    assert set(json.loads(jobs[0]['payload']['messages'][1]['content']))=={'question','evidence'}
    assert jobs[0]['payload']==stage.legacy.make_payload('What color?','[u2]\nRed')
    assert next(r for r in mapping if r['method']=='empty')['actual_evidence_tokens']==0


@pytest.mark.parametrize('mutation',['tokens','hash','family','missing','duplicate','outside','cache_payload','cache_model'])
def test_selection_or_payload_cannot_be_inherited_by_loose_identity(mutation):
    data,_=fixture()
    if mutation=='tokens': data['records'][0]['actual_evidence_tokens']+=1
    elif mutation=='hash': data['records'][0]['pack_sha256']='bad'
    elif mutation=='family': data['records'][0]['family_id']='other'
    elif mutation=='missing': data['records'].pop()
    elif mutation=='duplicate': data['records'].append(deepcopy(data['records'][0]))
    elif mutation=='outside': data['prepared']['queries'][0]['candidate_ids']=['u1']
    elif mutation=='cache_payload': next(iter(data['cache'].values()))['job']['payload']['temperature']=.5
    elif mutation=='cache_model': next(iter(data['cache'].values()))['resolved_model']='qwen/qwen3.6-plus-04-02'
    with pytest.raises(ValueError): stage.build_jobs(data,Tokenizer())


def test_whole_budget_is_admitted_only_with_all_77_and_24_families(monkeypatch):
    data,_,jobs,mapping,inheritance=built()
    with pytest.raises(ValueError,match='six methods'): stage.scope(data,jobs,mapping,inheritance)
    monkeypatch.setattr(stage,'SPEC',{**stage.SPEC,'questions':1,'families':1,'logical_predictions':6})
    counts,amount=stage.scope(data,jobs,mapping,inheritance)
    assert amount==Decimal(jobs[0]['reserved_usd']) and counts['inherited_logical_predictions']==3
    jobs[0]['reserved_usd']=str(stage.HEADROOM+Decimal('.0000000001'))
    with pytest.raises(ValueError,match='whole recovered'): stage.scope(data,jobs,mapping,inheritance)


@pytest.mark.parametrize('mutation',['attempts','reservation','unknown','known'])
def test_prior_whole_night_accounting_cannot_be_reset(mutation):
    data,_=fixture();prior=data['prior']
    if mutation=='attempts':prior['attempts']=374
    elif mutation=='reservation':prior['reservation_usd']='1.3633746750'
    elif mutation=='unknown':prior['unknown_cost_attempts']=0
    elif mutation=='known':prior['known_cost_usd']='0'
    with pytest.raises(ValueError):stage.validate_prior(prior)


def test_reservation_and_exact_actual_model_are_saved_before_transport(tmp_path):
    data,_,jobs,_,_=built()
    def transport(payload):
        ledger,_=stage.parent.verify_events(tmp_path)
        assert ledger['resolved_models']=={'generator':stage.MODEL}
        assert ledger['attempts'][0]['status']=='in_flight'
        assert Decimal(ledger['reservation_total_usd'])==Decimal(jobs[0]['reserved_usd'])
        return response('Red')
    client=bounded(tmp_path,data,jobs,transport);client.submit(jobs[0])
    answers,ledger,_,complete=stage.inspect_calls(config(data,jobs),jobs,tmp_path,require_complete=True)
    assert complete and answers[jobs[0]['cache_key']]=='Red'
    totals=stage.accounting(config(data,jobs),ledger,True)
    assert totals['night_attempts']==684 and totals['night_unknown_cost_attempts']==1
    assert totals['night_known_reported_cost_subtotal_usd']=='0.3001'
    assert totals['inherited_requests_charged_again']==0
    assert Decimal(totals['night_attempted_reservation_usd'])==stage.PRIOR_RESERVATION+Decimal(jobs[0]['reserved_usd'])


@pytest.mark.parametrize('error',[TimeoutError,KeyboardInterrupt])
def test_new_uncertain_request_preserves_old_and_new_unknown_and_never_retries(tmp_path,error):
    data,_,jobs,_,_=built();sent=[]
    def transport(payload):sent.append(payload);raise error()
    client=bounded(tmp_path,data,jobs,transport)
    with pytest.raises((RuntimeError,KeyboardInterrupt)):client.submit(jobs[0])
    with pytest.raises(ValueError):client.submit(jobs[0])
    answers,ledger,_,complete=stage.inspect_calls(config(data,jobs),jobs,tmp_path,require_complete=False)
    assert not complete and not answers and len(sent)==1
    totals=stage.accounting(config(data,jobs),ledger,False)
    assert totals['night_unknown_cost_attempts']==2 and totals['main_results_available'] is False
    assert Decimal(totals['generation_attempted_reservation_usd'])==Decimal(jobs[0]['reserved_usd'])


def test_other_allowed_model_alias_halts_with_known_cost_retained(tmp_path):
    data,_,jobs,_,_=built()
    client=bounded(tmp_path,data,jobs,lambda p:response('Red',model='qwen/qwen3.6-plus-04-02'))
    with pytest.raises(RuntimeError):client.submit(jobs[0])
    _,ledger,_,complete=stage.inspect_calls(config(data,jobs),jobs,tmp_path,require_complete=False)
    assert not complete and ledger['actual_reported_cost_usd']=='0.0001'
    assert stage.accounting(config(data,jobs),ledger,False)['night_unknown_cost_attempts']==1


def test_65second_margin_stops_before_reserving_or_sending(tmp_path,monkeypatch):
    data,_,jobs,_,_=built();sent=[]
    client=bounded(tmp_path,data,jobs,lambda p:sent.append(p))
    monkeypatch.setattr(stage,'utc_now',lambda:datetime(2026,9,27,0,58,55,tzinfo=timezone.utc))
    with pytest.raises(TimeoutError):client.submit(jobs[0])
    assert not sent and not client.ledger['attempts'] and client.ledger['reservation_total_usd']=='0'


def test_zero_missing_payloads_use_zero_cost_ledger_without_key_or_client(tmp_path):
    data,_,_,_,_=built();c=config(data,[])
    stage.parent.write(tmp_path/'cache_only_ledger.json',stage.empty_ledger(c))
    answers,ledger,_,complete=stage.inspect_calls(c,[],tmp_path,require_complete=True)
    assert complete and not answers and stage.accounting(c,ledger,True)['night_attempts']==683
    assert stage.accounting(c,ledger,True)['night_unknown_cost_attempts']==1
    (tmp_path/'provider_calls').mkdir()
    with pytest.raises(ValueError):stage.inspect_calls(c,[],tmp_path,require_complete=True)


def all_questions():
    base,gold=fixture();prepared={'documents':{},'queries':[]};records=[];annotations={}
    for i in range(77):
        doc,family,qid=f'd{i%24}',f'f{i%24}',f'q{i}'
        prepared['documents'][doc]=deepcopy(base['prepared']['documents']['doc'])
        q=deepcopy(base['prepared']['queries'][0]);q.update(doc_id=doc,family_id=family,question_id=qid,query=f'Color {i}?')
        prepared['queries'].append(q)
        for row in base['records']:records.append({**deepcopy(row),'doc_id':doc,'family_id':family,'question_id':qid})
        annotations[doc,qid]=deepcopy(gold['doc','q'])
    data={**base,'prepared':prepared,'records':records,'cache':{}}
    jobs,mapping,_=stage.build_jobs(data,Tokenizer())
    answers={job['cache_key']:('Unanswerable' if not json.loads(job['payload']['messages'][1]['content'])['evidence'] else 'Blue') for job in jobs}
    return data,mapping,answers,annotations


def test_all_462_predictions_six_means_fifteen_pairs_sixty_intervals():
    data,mapping,answers,annotations=all_questions()
    rows,summary=stage.score_all(data,mapping,answers,annotations)
    assert len(rows)==462 and len(summary['metrics'])==6 and len(summary['paired_comparisons'])==15
    assert [(p['plus'],p['minus']) for p in summary['paired_comparisons']]==list(stage.PAIRS)
    assert sum(len(p['metrics'])*2 for p in summary['paired_comparisons'])==60
    for pair in summary['paired_comparisons']:
        for metric in stage.METRICS:
            result=pair['metrics'][metric]
            assert result['questions']==77 and result['families']==24
            assert result['question_positive']+result['question_ties']+result['question_negative']==77
    assert next(m for m in summary['metrics'] if m['method']=='empty')['unanswerable_predictions']==77
    with pytest.raises(ValueError):stage.score_all(data,mapping[:-1],answers,annotations)
    answers.pop(next(iter(answers)))
    with pytest.raises(ValueError):stage.score_all(data,mapping,answers,annotations)


def test_partial_results_never_load_gold_or_score_inherited_answers(tmp_path,monkeypatch):
    data,_,jobs,mapping,inheritance=built()
    client=bounded(tmp_path,data,jobs,lambda p:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError):client.submit(jobs[0])
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:pytest.fail('partial quality is prohibited'))
    result=stage.results(config(data,jobs),data,jobs,mapping,inheritance,{},tmp_path,require_complete=False)
    assert result[:3]==(None,None,None)


@pytest.mark.parametrize('mutation',['changed_source','failure_marker'])
def test_known_failed_or_changed_input_is_rejected_before_gold_scoring(tmp_path,monkeypatch,mutation):
    data,_,jobs,mapping,inheritance=built();c=config(data,jobs)
    source=tmp_path/'frozen_source';source.write_text('original')
    c['input_sha256']={str(source):stage.legacy.digest(source)}
    monkeypatch.setattr(stage,'inspect_calls',lambda *a,**k:({}, {}, {}, True))
    monkeypatch.setattr(stage,'registration_binding',lambda *a:{})
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:pytest.fail('known invalid sources must not reach gold'))
    if mutation=='changed_source':source.write_text('changed')
    else:(tmp_path/'hard_deadline.json').write_text('{}')
    with pytest.raises(ValueError):stage.results(c,data,jobs,mapping,inheritance,{},tmp_path,require_complete=True)


def prepared_plan(tmp_path,monkeypatch):
    data,_,jobs,mapping,inheritance=built()
    monkeypatch.setattr(stage,'SPEC',{**stage.SPEC,'questions':1,'families':1,'logical_predictions':6})
    monkeypatch.setattr(stage,'registration_path',lambda:tmp_path/'parent-chain-registration.json')
    monkeypatch.setattr(stage,'source_data',lambda p:deepcopy(data))
    monkeypatch.setattr(stage.parent.metadata.runner.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    args=SimpleNamespace(recovery_plan=tmp_path/'support-plan',recovery_run=tmp_path/'support-run',recovery_audit=tmp_path/'support-audit',
        output=tmp_path/'plan',run_output=tmp_path/'run')
    stage.prepare(args)
    return args


def test_plan_roundtrip_rebuilds_inheritance_payloads_and_budget_without_calls(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    c,d,j,m,i,own=stage.load_plan(args.output)
    assert len(j)==1 and len(m)==6 and len(i)==2 and c['gold_in_payload'] is False
    assert not args.run_output.exists() and not Path(c['single_use_registration']).exists()
    with pytest.raises(FileExistsError):stage.prepare(args)


@pytest.mark.parametrize('mutation',['budget','model','inheritance','jobs','count'])
def test_resealed_plan_cannot_change_complete_scope_or_inheritance(tmp_path,monkeypatch,mutation):
    args=prepared_plan(tmp_path,monkeypatch);path=args.output/'experiment_config.json';c=stage.parent.read(path)
    if mutation=='budget':c['answer_reservation_usd']='0'
    elif mutation=='model':c['parent_resolved_model']='qwen/qwen3.6-plus-04-02'
    elif mutation=='count':c['new_unique_requests']=0
    else:
        target=args.output/('inheritance.json' if mutation=='inheritance' else 'jobs.json')
        target.write_text('{}' if mutation=='inheritance' else '[]')
        c['plan_files_sha256'][target.name]=stage.legacy.digest(target)
    path.write_text(json.dumps(c))
    (args.output/'plan_manifest.json').write_text(json.dumps({'schema':stage.SCHEMA,'experiment_config_sha256':stage.legacy.digest(path)}))
    with pytest.raises(ValueError):stage.load_plan(args.output)


class Guard:
    def __init__(self,*a,**k):pass
    def close(self):pass


def test_failed_run_cannot_be_restarted_through_another_output_plan(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    def factory(*a,**k):
        k.pop('key_file');k.pop('proxy')
        return stage.RecoveredAnswerClient(*a,**k,transport=lambda p:(_ for _ in ()).throw(TimeoutError()))
    run_args=SimpleNamespace(plan=args.output,key_file='not-read',proxy=None)
    with pytest.raises(RuntimeError):stage.run(run_args,client_factory=factory,guard_factory=Guard)
    failure=stage.parent.read(args.run_output/'failure.json')
    assert failure['main_results_available'] is False and failure['accounting']['night_unknown_cost_attempts']==2
    assert not (args.run_output/'summary.json').exists()
    assert stage.audit(SimpleNamespace(plan=args.output,run=args.run_output,allow_incomplete=True))['status']=='verified_incomplete'
    with pytest.raises(FileExistsError):stage.run(run_args,client_factory=factory,guard_factory=Guard)
    args.output=tmp_path/'plan2';args.run_output=tmp_path/'run2'
    with pytest.raises(FileExistsError):stage.prepare(args)


def test_hard_request_watchdog_marks_terminal_unknown_without_retry(tmp_path,monkeypatch):
    observed={}
    class Event:
        def wait(self,seconds):observed['seconds']=seconds;return False
        def set(self):pass
    class Thread:
        def __init__(self,target,daemon):self.target=target
        def start(self):self.target()
        def join(self,timeout):pass
    monkeypatch.setattr(stage.threading,'Event',Event);monkeypatch.setattr(stage.threading,'Thread',Thread)
    monkeypatch.setattr(stage.os,'_exit',lambda code:observed.update(exit_code=code))
    stage.Deadline(tmp_path,time.monotonic(),request=True).close()
    assert observed=={'seconds':65,'exit_code':124}
    marker=stage.parent.read(tmp_path/'hard_deadline.json')
    assert marker['request_deadline'] is True and marker['main_results_available'] is False


def complete_run(tmp_path,monkeypatch,all_cached=False):
    data,_,answers,gold=all_questions()
    jobs,_=stage.legacy.build_jobs(data['prepared'],data['records'])
    for i,job in enumerate(jobs):
        if i==0 and not all_cached:continue
        data['cache'][job['cache_key']]={'job':job,'answer':answers[job['cache_key']],'origin':'local',
            'request_ordinal':i+1,'request_sha256':stage.legacy.client.object_hash(job['payload']),
            'response_sha256':'a'*64,'resolved_model':stage.MODEL}
    monkeypatch.setattr(stage,'registration_path',lambda:tmp_path/'registration.json')
    monkeypatch.setattr(stage,'source_data',lambda p:deepcopy(data))
    monkeypatch.setattr(stage.parent.metadata.runner.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:deepcopy(gold))
    args=SimpleNamespace(recovery_plan=tmp_path/'support-plan',recovery_run=tmp_path/'support-run',recovery_audit=tmp_path/'support-audit',
        output=tmp_path/'plan',run_output=tmp_path/'run')
    stage.prepare(args)
    def factory(*a,**k):
        k.pop('key_file');k.pop('proxy')
        return stage.RecoveredAnswerClient(*a,**k,transport=lambda p:response(answers[stage.legacy.client.object_hash(
            {'endpoint':stage.legacy.ENDPOINT,'prompt_version':stage.legacy.PROMPT_VERSION,'payload':p})]))
    stage.run(SimpleNamespace(plan=args.output,key_file='not-read',proxy=None),client_factory=factory,guard_factory=Guard)
    return args


def test_zero_new_full_run_and_audit_never_construct_a_paid_client(tmp_path,monkeypatch):
    # complete_run's factory is not needed because every exact payload is inherited.
    args=complete_run(tmp_path,monkeypatch,all_cached=True)
    result=stage.audit(SimpleNamespace(plan=args.output,run=args.run_output,allow_incomplete=False))
    assert result['status']=='verified_complete' and result['accounting']['new_api_calls']==0
    assert result['accounting']['generation_attempted_reservation_usd']=='0'
    assert result['accounting']['night_attempted_reservation_usd']==str(stage.PRIOR_RESERVATION)
    assert not (args.run_output/'provider_calls').exists()


@pytest.mark.parametrize('mutation',['schema','extra','timestamp'])
def test_registration_identity_and_bytes_are_bound_in_complete_result(tmp_path,monkeypatch,mutation):
    args=complete_run(tmp_path,monkeypatch)
    path=stage.registration_path();registered=stage.parent.read(path)
    if mutation=='schema':registered['schema']='wrong'
    elif mutation=='extra':registered['new_plan_allowed']=True
    elif mutation=='timestamp':registered['registered_at_utc']='2026-09-26T23:00:00+00:00'
    path.write_text(json.dumps(registered))
    with pytest.raises(ValueError):stage.audit(SimpleNamespace(plan=args.output,run=args.run_output,allow_incomplete=False))


@pytest.mark.parametrize('mutation',['late','no_timezone','negative','nan','infinite','over65'])
def test_saved_complete_request_must_satisfy_admission_and_elapsed_contract(monkeypatch,mutation):
    attempt={'status':'completed','started_at':'2026-09-27T00:00:00+00:00','elapsed_seconds':1.0}
    if mutation=='late':attempt['started_at']='2026-09-27T00:58:55+00:00'
    elif mutation=='no_timezone':attempt['started_at']='2026-09-27T00:00:00'
    elif mutation=='negative':attempt['elapsed_seconds']=-1
    elif mutation=='nan':attempt['elapsed_seconds']=float('nan')
    elif mutation=='infinite':attempt['elapsed_seconds']=float('inf')
    elif mutation=='over65':attempt['elapsed_seconds']=65.1
    monkeypatch.setattr(stage.inherited,'inspect_calls',lambda *a,**k:({}, {'attempts':[attempt]}, {}, True))
    with pytest.raises(ValueError):stage.inspect_calls({},[{}],'unused',require_complete=True)


def test_complete_run_replays_462_predictions_and_full_metadata_audit(tmp_path,monkeypatch):
    args=complete_run(tmp_path,monkeypatch)
    result=stage.audit(SimpleNamespace(plan=args.output,run=args.run_output,allow_incomplete=False))
    assert result['status']=='verified_complete' and result['accounting']['new_api_calls']==1
    assert result['accounting']['night_unknown_cost_attempts']==1
    saved=stage.parent.read(args.run_output/'summary.json')
    assert saved['record_count']==462 and len(saved['paired_comparisons'])==15
    assert len(stage.parent.read(args.run_output/'answers.json'))==saved['unique_payloads']


@pytest.mark.parametrize('mutation',['metadata','paired','answers','extra_file','failure_marker'])
def test_complete_audit_rejects_mutated_or_extra_outputs_even_with_new_output_hashes(tmp_path,monkeypatch,mutation):
    args=complete_run(tmp_path,monkeypatch);path=args.run_output/'summary.json';summary=stage.parent.read(path)
    if mutation=='metadata':summary['night_unknown_cost_attempts']=0
    elif mutation=='paired':summary['paired_comparisons'][0]['metrics']['official_answer_f1']['question_weighted']['delta']=-.5
    elif mutation=='answers':
        p=args.run_output/'answers.json';value=stage.parent.read(p);value[next(iter(value))]='changed';p.write_text(json.dumps(value))
    elif mutation=='extra_file':(args.run_output/'unaccounted.json').write_text('{}')
    elif mutation=='failure_marker':(args.run_output/'hard_deadline.json').write_text('{}')
    # The legitimate full-audit output inventory must remain exact, not merely self-consistent.
    summary['output_sha256']={str(Path(p).relative_to(args.run_output)):v for p,v in stage.inherited.tree(args.run_output).items() if Path(p)!=path}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):stage.audit(SimpleNamespace(plan=args.output,run=args.run_output,allow_incomplete=False))


def test_source_entry_requires_complete_recovery_before_any_cached_answers(monkeypatch):
    monkeypatch.setattr(stage.inherited,'tree',lambda p:{})
    monkeypatch.setattr(stage.parent,'hashes',lambda p:{})
    def incomplete(*a):raise ValueError('incomplete support')
    monkeypatch.setattr(stage.recovery,'verify_completed_run',incomplete)
    monkeypatch.setattr(stage,'cache_source',lambda *a:pytest.fail('partial support must not reach answer planning'))
    with pytest.raises(ValueError,match='incomplete support'):
        stage.source_data({'recovery_plan':'unused','recovery_run':'unused','recovery_audit':'unused'})
