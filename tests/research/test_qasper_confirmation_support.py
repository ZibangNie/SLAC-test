"""Synthetic-only execution contract tests; no real key, QA, model, or API."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/research'))
spec=importlib.util.spec_from_file_location('confirmation_support_test',ROOT/'docs/research/run_qasper_confirmation_support.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DEADLINE='2099-01-01T12:00:00+08:00'


class Guard:
    def __init__(self,*args):pass
    def close(self):pass


@pytest.fixture(autouse=True)
def blocked(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('real key/API/old execution forbidden')
    monkeypatch.setattr(m.client,'read_key',forbidden)
    monkeypatch.setattr(m.client.urllib.request,'urlopen',forbidden)
    monkeypatch.setattr(m.os,'_exit',forbidden)
    monkeypatch.setattr(m.stage,'ensure_admission_time',forbidden)
    monkeypatch.setattr(m.pilot,'selected_gold',forbidden)


def response(backend,payload):
    ids=m.client.payload_task_ids(payload,backend)
    value={'id':'synthetic-response','model':m.MODEL_LOCKS[backend],
           'provider':'TypeSafe' if backend=='jev' else 'Alibaba',
           'usage':{'cost':.00001,'input_tokens':10,'output_tokens':10}}
    if backend=='jev':value['answers']={i:{'type':'choice','choice':'yes','probabilities':{'yes':.8,'no':.1,'unknown':.1}} for i in ids}
    else:value['choices']=[{'finish_reason':'stop','message':{'content':json.dumps(dict.fromkeys(ids,'yes'))}}]
    return value


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    art=tmp_path/'art';art.mkdir();monkeypatch.setattr(m,'ART',art)
    parent=art/'budget';parent.mkdir()
    batches=[];jobs=[]
    with m.budget.qwen_profile():
        for i in range(2):
            tasks=[{'id':f'support:synthetic{i}','item':{'query':'Synthetic question?','unit':{'id':f'u{i}','text':'Synthetic evidence.'}}}]
            batch={'id':f'batch-{i}','kind':'support','group':[f'd{i}',f'q{i}'],'tasks':tasks};batches.append(batch)
            for backend in (('jev','general') if i%2==0 else ('general','jev')):
                payload=m.client.make_payload(tasks,'support',backend);reserve,inp,out=m.client.reservation(payload,backend)
                jobs.append({'ordinal':len(jobs)+1,'batch_id':batch['id'],'backend':backend,'task_ids':[tasks[0]['id']],
                    'payload':payload,'payload_sha256':m.client.object_hash(payload),'canonical_payload_bytes':len(m.client.canonical_bytes(payload)),
                    'input_allowance':inp,'output_allowance':out,'reserved_usd':str(reserve)})
        public={'schema':m.budget.SCHEMA,'paid_execution_admitted':False,'Q_questions':2,'C_support_query_unit_pairs':2,'B_common_support_batches':2,
            'support_total_reserved_usd':m.totals(jobs)['reserved_usd'],'generation_upper_bound':{'request_count':10,'reserved_usd':str(Decimal('.014196')*10)},
            'provider_snapshot_sha256':m.budget.SNAPSHOT_SHA256,'implementation_sha256':m.budget.digest(m.budget.__file__)}
        private={'schema':m.budget.SCHEMA,'public_aggregate':public,'specification':m.budget.SPEC,'source_sha256':{},
                 'models':deepcopy(m.client.MODELS),'prompt_version':m.client.PROMPT_VERSION,'jobs':jobs,'batches':batches}
    for name,value in [('private_plan.json',private),('public_aggregate.json',public),('source_binding.json',{'input_sha256':{}})]:m.write(parent/name,value)
    m.write(parent/'seal.json',{n:m.budget.digest(parent/n) for n in m.budget.OUTPUT_FILES-{'seal.json'}})
    plan,run=art/'execution-plan',art/'run'
    m.prepare(parent,plan,run,'1',DEADLINE)
    return {'art':art,'budget':parent,'plan':plan,'run':run,'jobs':jobs,'batches':batches}


def test_prepare_and_roundtrip_no_key_with_separate_answer_lock(fixture):
    config,jobs,_,_=m.load_plan(fixture['plan'])
    assert len(jobs)==config['physical_attempt_cap']==4
    assert Decimal(config['generation_locked_reserve_usd'])==Decimal('.141960')
    assert Decimal(config['unallocated_reserve_usd'])==Decimal('1')-Decimal('.141960')-Decimal(config['support_committed_reserve_usd'])
    assert not fixture['run'].exists()


@pytest.mark.parametrize('deadline',['2099-01-01T12:00:00','bad','2000-01-01T00:00:00+00:00'])
def test_invalid_or_expired_deadline_refused(fixture,deadline):
    with pytest.raises(ValueError):m.prepare(fixture['budget'],fixture['art']/'other-plan',fixture['art']/'other-run','1',deadline)


@pytest.mark.parametrize('cap',['.001','0','NaN','Infinity','-1'])
def test_whole_budget_and_invalid_money_rejected(fixture,cap):
    with pytest.raises(ValueError):m.prepare(fixture['budget'],fixture['art']/'other-plan',fixture['art']/'other-run',cap,DEADLINE)


def test_complete_run_durable_reserve_before_each_transport_and_audit(fixture):
    calls=[]
    def transport(backend,payload):
        ledger=m.read(fixture['run']/'provider_calls/segment_001/ledger.json')
        assert ledger['attempts'][-1]['status']=='in_flight'
        assert Decimal(ledger['reservation_total_usd'])>0
        assert ledger['resolved_models']==m.MODEL_LOCKS
        calls.append(backend);return response(backend,payload)
    summary=m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    assert len(calls)==4 and summary['status']=='completed'
    assert summary['accounting']['unknown_cost_attempts']==0
    judgments=m.read(fixture['run']/'judgments.json')
    assert len(judgments['labels']['jev'])==len(judgments['reported_scores'])==2
    audit=m.audit(fixture['plan'],fixture['art']/'audit')
    assert audit==summary and audit['quality_metrics_computed'] is False


@pytest.mark.parametrize('error',[TimeoutError,KeyboardInterrupt])
def test_unknown_failure_preserved_no_partial_results_or_restart(fixture,error):
    calls=[]
    def transport(backend,payload):calls.append(backend);raise error()
    with pytest.raises(BaseException):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    assert len(calls)==1 and not (fixture['run']/'judgments.json').exists() and not (fixture['run']/'summary.json').exists()
    report=m.audit(fixture['plan'],fixture['art']/'audit',allow_incomplete=True)
    assert report['status']=='stopped_incomplete' and report['accounting']['unknown_cost_attempts']==1
    assert Decimal(report['accounting']['reserved_usd'])>0
    with pytest.raises(ValueError):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)


@pytest.mark.parametrize('case',['model','provider','cost','tokens','scores','finish','responseid'])
def test_bad_response_stops_and_keeps_known_cost(fixture,case):
    calls=[]
    def transport(backend,payload):
        calls.append(backend);value=response(backend,payload)
        if case=='model':value['model']='changed'
        elif case=='provider':value['provider']='other'
        elif case=='cost':value['usage']['cost']=5
        elif case=='tokens':value['usage']['input_tokens']=999999999
        elif case=='scores':next(iter(value['answers'].values()))['probabilities']['yes']=2
        elif case=='responseid':value['id']=''
        elif case=='finish':
            if backend=='jev':return value
            value['choices'][0]['finish_reason']='length'
        return value
    with pytest.raises(RuntimeError):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    assert len(calls)==(2 if case=='finish' else 1)
    report=m.audit(fixture['plan'],fixture['art']/'audit',allow_incomplete=True)
    assert report['status']=='stopped_incomplete' and report['accounting']['unknown_cost_attempts']==0
    assert Decimal(report['accounting']['known_cost_usd'])>0


def test_deadline_after_durable_reserve_prevents_dispatch(fixture,monkeypatch):
    original=m.ensure_time
    def checked(deadline,*,dispatch=False):
        path=fixture['run']/'provider_calls/segment_001/ledger.json'
        if dispatch and path.exists() and m.read(path)['attempts']:raise ValueError('late')
        return original(deadline,dispatch=dispatch)
    monkeypatch.setattr(m,'ensure_time',checked)
    def transport(*args):pytest.fail('request dispatched after reserve cutoff')
    with pytest.raises(ValueError):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    report=m.audit(fixture['plan'],fixture['art']/'audit',allow_incomplete=True)
    assert report['accounting']['attempts']==report['accounting']['unknown_cost_attempts']==1


def test_second_plan_same_budget_family_cannot_reset_registration(fixture):
    second=fixture['art']/'second-plan'
    m.prepare(fixture['budget'],second,fixture['art']/'second-run','1',DEADLINE)
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    with pytest.raises(FileExistsError):m.run(second,'never-read',transport=response,guard_factory=Guard)
    assert not (fixture['art']/'second-run').exists()


def test_modified_raw_response_refused_by_event_audit(fixture):
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    path=fixture['run']/'provider_calls/segment_001/response_001.json'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):m.audit(fixture['plan'],fixture['art']/'audit')


def test_source_drift_stops_before_full_judgments(fixture):
    count=0
    def transport(backend,payload):
        nonlocal count
        count+=1
        if count==4:
            path=fixture['budget']/'public_aggregate.json';path.write_bytes(path.read_bytes()+b'\n')
        return response(backend,payload)
    with pytest.raises(ValueError):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    assert not (fixture['run']/'judgments.json').exists()


def test_segment_partition_exact_coverage_and_global_amount(fixture):
    jobs=deepcopy(fixture['jobs'])*100
    parts=m.segments(jobs)
    assert parts[0]['start']==0 and parts[-1]['stop']==400
    assert all(a['stop']==b['start'] for a,b in zip(parts,parts[1:]))
    assert sum(Decimal(p['reserved_usd']) for p in parts)==Decimal(m.totals(jobs)['reserved_usd'])
    assert all(p['requests']<=160 and p['judgments']<=1600 and p['input_allowance']<=8000000 and Decimal(p['reserved_usd'])<=2 for p in parts)


@pytest.mark.parametrize('drift',[False,True])
def test_new_segments_keep_global_model_locks_and_single_accounting(fixture,monkeypatch,drift):
    monkeypatch.setattr(m,'SEGMENT_CAPS',{**m.SEGMENT_CAPS,'requests':2})
    plan,run=fixture['art']/'segmented-plan',fixture['art']/'segmented-run'
    m.prepare(fixture['budget'],plan,run,'1',DEADLINE)
    calls=[]
    def transport(backend,payload):
        calls.append(backend);value=response(backend,payload)
        if drift and len(calls)==3:value['model']='qwen/qwen3.6-plus-04-02'
        return value
    if drift:
        with pytest.raises(RuntimeError):m.run(plan,'never-read',transport=transport,guard_factory=Guard)
        report=m.audit(plan,fixture['art']/'segmented-audit',allow_incomplete=True)
        assert len(calls)==3 and report['accounting']['completed_requests']==2
    else:
        report=m.run(plan,'never-read',transport=transport,guard_factory=Guard)
        assert len(calls)==4
        assert Decimal(report['accounting']['reserved_usd'])==Decimal(m.totals(fixture['jobs'])['reserved_usd'])
        assert m.audit(plan,fixture['art']/'segmented-audit')==report
    for segment in (1,2):
        assert m.read(run/f'provider_calls/segment_{segment:03d}/ledger.json')['resolved_models']==m.MODEL_LOCKS


def test_wrong_payload_rejected_without_attempt_or_transport(fixture):
    def forbidden(*args):pytest.fail('wrong frozen payload dispatched')
    with m.budget.qwen_profile():
        bounded=m.SupportClient(fixture['art']/'isolated-client',jobs=fixture['jobs'],deadline=DEADLINE,
            run_output=fixture['art'],guard_factory=Guard,transport=forbidden,budget_usd='1',request_cap=4,question_cap=4,token_cap=100000)
        tasks=deepcopy(fixture['batches'][0]['tasks']);tasks[0]['item']['query']='changed query'
        with pytest.raises(ValueError,match='next frozen'):bounded.submit(tasks,'support','jev')
        assert bounded.ledger['attempts']==[]


def test_request_watchdog_is_65_seconds_and_global_guard_is_separate(fixture):
    seen=[]
    class RecordingGuard(Guard):
        def __init__(self,output,seconds):seen.append(seconds)
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=RecordingGuard)
    assert seen[0]>65 and seen[1:]==[65]*4


def test_echoed_credentials_are_redacted_in_raw_response_and_events(fixture):
    fake='sk-or-v1-'+'A'*40
    def transport(backend,payload):
        value=response(backend,payload);value['echo']='test-credential '+fake
        return value
    m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    for path in fixture['run'].rglob('*.json'):
        raw=path.read_text('utf-8')
        assert fake not in raw and 'test-credential' not in raw


def test_public_summary_change_cannot_pass_complete_audit(fixture):
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    path=fixture['run']/'public_aggregate.json';value=m.read(path)
    value['accounting']['known_cost_usd']='0'
    path.write_bytes(m.budget.canonical(value))
    with pytest.raises(ValueError):m.audit(fixture['plan'],fixture['art']/'audit')


@pytest.mark.parametrize('change',['deadline','source','hard_marker'])
def test_late_completion_gate_removes_success_summary(fixture,monkeypatch,change):
    original=m.write
    def late_write(path,value):
        original(path,value)
        if Path(path).name=='summary.json':
            if change=='deadline':monkeypatch.setattr(m,'utc_now',lambda:datetime(2100,1,1,tzinfo=timezone.utc))
            elif change=='source':
                victim=fixture['budget']/'public_aggregate.json';victim.write_bytes(victim.read_bytes()+b'\n')
            else:original(fixture['run']/'hard_timeout.json',{'status':'failed_hard_timeout'})
    monkeypatch.setattr(m,'write',late_write)
    with pytest.raises(ValueError):m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    assert not (fixture['run']/'summary.json').exists()
    assert (fixture['run']/'failure.json').exists()


@pytest.mark.parametrize('field',['support_commitment_usd','generation_locked_reserve_usd','total_budget_usd'])
def test_registration_global_amount_tampering_is_rejected(fixture,field):
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    config,_,_,_=m.load_plan(fixture['plan']);path=Path(config['registration'])
    value=m.read(path);value[field]='0';path.write_bytes(m.budget.canonical(value))
    with pytest.raises(ValueError,match='registration'):m.audit(fixture['plan'],fixture['art']/'audit')


def test_registration_binding_in_receipt_and_end_check(fixture,monkeypatch):
    m.run(fixture['plan'],'never-read',transport=response,guard_factory=Guard)
    config,_,_,_=m.load_plan(fixture['plan']);path=Path(config['registration'])
    m.audit(fixture['plan'],fixture['art']/'audit')
    receipt=m.read(fixture['art']/'audit/verification.json')
    assert receipt['input_sha256'][str(path.resolve())]==m.budget.digest(path)
    original=m.collect
    def changing(*args,**kwargs):
        result=original(*args,**kwargs);path.write_bytes(path.read_bytes()+b'\n');return result
    monkeypatch.setattr(m,'collect',changing)
    with pytest.raises(ValueError,match='bound input changed'):m.audit(fixture['plan'],fixture['art']/'audit2')


def test_http_error_reported_cost_is_kept_and_not_retried(fixture):
    calls=[]
    def transport(backend,payload):
        calls.append(backend)
        body=json.dumps({'error':{'code':503,'message':'synthetic'},
                         'usage':{'cost':.00003,'input_tokens':10,'output_tokens':0}}).encode()
        raise m.client.urllib.error.HTTPError('https://synthetic.invalid',503,'synthetic',{},io.BytesIO(body))
    with pytest.raises(RuntimeError):m.run(fixture['plan'],'never-read',transport=transport,guard_factory=Guard)
    report=m.audit(fixture['plan'],fixture['art']/'audit',allow_incomplete=True)
    assert len(calls)==1 and report['accounting']['unknown_cost_attempts']==0
    assert Decimal(report['accounting']['known_cost_usd'])==Decimal('.00003')
