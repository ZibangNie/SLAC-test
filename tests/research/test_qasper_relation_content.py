"""Synthetic-only static executor and complete cache replay tests."""
from copy import deepcopy
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_content as m

class NoGuard:
    def __init__(self,*a):pass
    def close(self):pass

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m.client,'read_key',lambda *a:pytest.fail('no keys'))
    monkeypatch.setattr(m,'utc_now',lambda:datetime(2026,9,27,0,0,tzinfo=timezone.utc))

def response(payload,label='dependent',model=m.MODEL):
    return {'model':model,'provider':'Typesafe','id':'synthetic','usage':{'input_tokens':10,'output_tokens':1,'cost':'0.00001'},
        'answers':{k:{'type':'choice','choice':label} for k in payload['questions']}}

def make_data(tmp):
    d={k:{} for k in ('tasks','queries','symbolic','maskmap','packs','answers','cache')}
    d.update(jobs=[],baselines=[],prepared={'queries':[]},prior=deepcopy(m.PRIOR));ann=[]
    for i in range(77):
        q=(f'f{i%24:02}',f'd{i:03}',f'q{i:03}');ident=dict(zip(m.packs.IDENTITY,q))
        edges=[[q[1],'u0','u1']] if i<20 else [];outcomes=[]
        for mask in range(2**len(edges)):
            text='[u0]\nBlue' if mask==0 else '[u1]\nRed';job=m.packs.job_for('Question '+str(i),text);key=job['cache_key']
            d['answers'][key]='Blue' if mask==0 else 'Red';d['cache'][key]={'job':job,'answer':d['answers'][key]};d['maskmap'][q,mask]=key
            h=m.hashlib.sha256(text.encode()).hexdigest();uid='u0' if mask==0 else 'u1'
            outcomes.append({'selected_identities':[[q[1],uid]],'rendered_pack':text,'pack_sha256':h,'actual_tokens':10+mask})
            d['packs'][q,key]={**ident,'cache_key':key,'selected_ids':[uid],'pack_sha256':h,'actual_evidence_tokens':10+mask}
        d['queries'][q]={**ident,'edges':edges,'outcomes':outcomes}
        d['symbolic'][q]={'edges':edges,'source_by_target':list(range(len(edges))),'required_indices':list(range(len(edges)))}
        for method in m.packs.BASELINES:d['baselines'].append(d['packs'][q,d['maskmap'][q,0]]|{'method':method})
        d['prepared']['queries'].append(ident|{'query':'Question '+str(i)})
        ann.append(ident|{'question':'Question '+str(i),'official_split':'validation','answer_annotations':[{'native_answer':{
            'unanswerable':False,'extractive_spans':['Blue'],'free_form_answer':'','yes_no':None,'evidence':[]}}]})
        if i<20:
            t={'id':f't_{i}','doc_id':q[1],'left_id':'u0','right_id':'u1','item':{'unit_a':{'id':'u0','text':'Blue'},'unit_b':{'id':'u1','text':'Red'}}}
            d['tasks'][tuple(edges[0])]=t;p=m.client.make_payload([t],'static','jev');r,a,o=m.client.reservation(p,'jev')
            d['jobs'].append({'edge':edges[0],'task_id':t['id'],'endpoint':m.client.MODELS['jev']['endpoint'],'payload':p,'payload_sha256':m.object_hash(p),'reservation_usd':str(r),'input_allowance':a,'output_allowance':o})
    side=tmp/'references.jsonl';m.pilot.write_rows(side,ann);src=tmp/'source.txt';src.write_text('fixed')
    d['references']={'path':str(side),'sha256':m.pilot.digest(side)};d['input_sha256']={str(src):m.pilot.digest(src)}
    return d

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    d=make_data(tmp_path);monkeypatch.setattr(m,'source_data',lambda:deepcopy(d))
    monkeypatch.setattr(m,'run_path',lambda:tmp_path/'run');monkeypatch.setattr(m,'registration_path',lambda:tmp_path/'registration.json')
    p=tmp_path/'plan';c=m.prepare(SimpleNamespace(output=p));return p,d,c

def factory(*a,**k):return m.StaticClient(*a,transport=lambda b,p:response(p),guard_factory=NoGuard,**k)
def ra(p):return SimpleNamespace(plan=p,key_file=None,proxy=None)
def aa(p):return SimpleNamespace(plan=p,run=m.run_path(),allow_incomplete=False)

def test_plan_roundtrip_no_score_and_full_budget(prepared,monkeypatch):
    p,d,c=prepared;monkeypatch.setattr(m,'score',lambda *a:pytest.fail('no planning quality'))
    loaded,data,jobs,own=m.load_plan(p)
    assert loaded==c and jobs==d['jobs'] and len(own)==3 and not m.run_path().exists()
    assert c['prediction']['prospective_night_reservation_usd']=='4.7586873300' and c['prediction']['new_generation_requests']==0

@pytest.mark.parametrize('mutation',['jobs','payload','prior','source','run','extra'])
def test_resealed_plan_rejects_changed_contract(prepared,mutation):
    p,d,c=prepared;c=deepcopy(c)
    if mutation=='extra':(p/'extra.json').write_text('{}')
    elif mutation=='source':Path(next(iter(d['input_sha256']))).write_text('changed')
    elif mutation in ('jobs','payload'):
        j=m.read(p/'jobs.json')
        if mutation=='jobs':j.pop()
        else:j[0]['payload']['state']['items']['t_0']['unit_a']['text']='changed'
        (p/'jobs.json').write_bytes(m.canonical(j));s=m.read(p/'plan_manifest.json');s['jobs_sha256']=m.pilot.digest(p/'jobs.json');(p/'plan_manifest.json').write_bytes(m.canonical(s))
    else:
        if mutation=='prior':c['prior_night_accounting']['unknown_cost_attempts']=0
        else:c['run_output']=str(p/'wrong')
        (p/'experiment_config.json').write_bytes(m.canonical(c));s=m.read(p/'plan_manifest.json');s['experiment_config_sha256']=m.pilot.digest(p/'experiment_config.json');(p/'plan_manifest.json').write_bytes(m.canonical(s))
    with pytest.raises((ValueError,AssertionError)):m.load_plan(p)

def test_complete_run_audit_all_539_24_and_old_unknown(prepared):
    p,d,c=prepared;s=m.run(ra(p),client_factory=factory,guard_factory=NoGuard)
    assert s['record_count']==539 and len(s['metrics'])==7 and len(s['paired_comparisons'])==6
    assert sum(2*len(x['metrics']) for x in s['paired_comparisons'])==24
    assert s['night_attempts']==837 and s['night_reservation_usd']=='4.7586873300' and s['night_unknown_cost_attempts']==1
    assert s['generation_calls']==0 and s['content_placebo_equal_packs']==77 and s['semantic_content_claim_must_stop']
    assert m.audit(aa(p))['status']=='verified_complete'
    with pytest.raises(ValueError):m.run(ra(p),client_factory=factory,guard_factory=NoGuard)

@pytest.mark.parametrize('label',['independent','unknown'])
def test_legal_inactive_labels_keep_all_questions(prepared,label):
    _,d,_=prepared;rows=m.resolve(d,{j['task_id']:label for j in d['jobs']});assert len(rows)==539
    assert all(r['cache_key']==d['maskmap'][m.identity(r),0] for r in rows if r['method']=='Rcontent')

@pytest.mark.parametrize('mutation',['missing','extra','invalid','bool'])
def test_partial_or_bad_labels_rejected(prepared,mutation):
    _,d,_=prepared;l={j['task_id']:'dependent' for j in d['jobs']}
    if mutation=='missing':l.pop('t_0')
    elif mutation=='extra':l['extra']='unknown'
    elif mutation=='invalid':l['t_0']='yes'
    else:l['t_0']=True
    with pytest.raises((ValueError,TypeError)):m.resolve(d,l)

def test_unobserved_completion_not_imputed(prepared):
    _,d,_=prepared;q=next(iter(d['queries']));d['jobs']=d['jobs'][1:];d['symbolic'][q]['required_indices']=[]
    with pytest.raises(ValueError,match='unobserved completion'):m.resolve(d,{j['task_id']:'independent' for j in d['jobs']})

@pytest.mark.parametrize('failure',['timeout','interrupt','model','label','missing_id'])
def test_failure_never_scores_or_continues(prepared,monkeypatch,failure):
    p,d,c=prepared;calls=[];clients=[];monkeypatch.setattr(m,'score',lambda *a:pytest.fail('partial cannot score'))
    def transport(backend,payload):
        calls.append(backend)
        if failure=='timeout':raise TimeoutError('synthetic')
        if failure=='interrupt':raise KeyboardInterrupt()
        v=response(payload,model='typesafe/jev-1.13' if failure=='model' else m.MODEL)
        if failure=='label':next(iter(v['answers'].values()))['choice']='yes'
        if failure=='missing_id':v.pop('id')
        return v
    def local(*a,**k):
        cl=m.StaticClient(*a,transport=transport,guard_factory=NoGuard,**k);clients.append(cl);return cl
    with pytest.raises(BaseException):m.run(ra(p),client_factory=local,guard_factory=NoGuard)
    assert calls==['jev'] and (m.run_path()/'failure.json').exists() and not (m.run_path()/'summary.json').exists()
    with pytest.raises(ValueError):clients[0].submit([d['tasks'][tuple(d['jobs'][0]['edge'])]],'static','jev')
    a=aa(p);a.allow_incomplete=True
    v=m.audit(a);assert v['status']=='verified_incomplete' and v['accounting']['night_reservation_usd']=='4.6636873300'
    expected_unknown=2 if failure in ('timeout','interrupt') else 1
    assert v['accounting']['night_unknown_cost_attempts']==expected_unknown
    assert Decimal(v['accounting']['new_static_known_cost_usd'])==(Decimal(0) if expected_unknown==2 else Decimal('.00001'))

def test_static_only_and_exact_next_payload(prepared,tmp_path):
    _,d,_=prepared;c=m.StaticClient(tmp_path/'calls',jobs=d['jobs'],started=m.time.monotonic(),transport=lambda *a:pytest.fail('no dispatch'),guard_factory=NoGuard)
    t=d['tasks'][tuple(d['jobs'][0]['edge'])]
    for kind,backend in [('support','jev'),('static','general')]:
        with pytest.raises(ValueError):c.submit([t],kind,backend)
    changed=deepcopy(t);changed['item']['unit_a']['text']='other'
    with pytest.raises(ValueError):c.submit([changed],'static','jev')

def test_65_second_margin_is_strict(monkeypatch):
    monkeypatch.setattr(m,'utc_now',lambda:datetime(2026,9,27,0,58,55,tzinfo=timezone.utc))
    with pytest.raises(ValueError):m.ensure_time(m.time.monotonic(),True)

@pytest.mark.parametrize('target',['record','summary','labels','registration','event','extra'])
def test_resealed_saved_result_rejected(prepared,target):
    p,d,c=prepared;m.run(ra(p),client_factory=factory,guard_factory=NoGuard);out=m.run_path()
    if target=='record':
        rr=m.packs.parent.rows(out/'per_question.jsonl');rr[0]['official_answer_f1']=.123;(out/'per_question.jsonl').write_bytes(b''.join(m.canonical(r)+b'\n' for r in rr))
    elif target=='summary':
        s=m.read(out/'summary.json');s['night_unknown_cost_attempts']=0;(out/'summary.json').write_bytes(m.canonical(s))
    elif target=='labels':
        l=m.read(out/'labels.json');l['t_0']='independent';(out/'labels.json').write_bytes(m.canonical(l))
    elif target=='registration':m.registration_path().write_text('{}')
    elif target=='event':(out/'attempt_ledger/event_00000.json').write_text('{}')
    else:(out/'extra.json').write_text('{}')
    s=m.read(out/'summary.json');s['output_sha256']={str(Path(k).relative_to(out)):v for k,v in m.packs.inherited.tree(out).items() if Path(k)!=out/'summary.json'};(out/'summary.json').write_bytes(m.canonical(s))
    with pytest.raises((ValueError,KeyError)):m.audit(aa(p))

def test_final_hash_failure_removes_summary(prepared,monkeypatch):
    p,d,_=prepared;original=m.pilot.verify_hashes
    def check(hashes):
        if (m.run_path()/'summary.json').exists():raise ValueError('changed')
        return original(hashes)
    monkeypatch.setattr(m.pilot,'verify_hashes',check)
    with pytest.raises(ValueError):m.run(ra(p),client_factory=factory,guard_factory=NoGuard)
    assert not (m.run_path()/'summary.json').exists() and (m.run_path()/'failure.json').exists()

def test_explicit_prepare_forbids_resolve_and_score(tmp_path,monkeypatch):
    d=make_data(tmp_path);monkeypatch.setattr(m,'source_data',lambda:deepcopy(d))
    monkeypatch.setattr(m,'run_path',lambda:tmp_path/'run');monkeypatch.setattr(m,'registration_path',lambda:tmp_path/'register.json')
    for name in ('resolve','score'):monkeypatch.setattr(m,name,lambda *a:pytest.fail('no labels or quality during prepare'))
    c=m.prepare(SimpleNamespace(output=tmp_path/'plan'));assert c['generation_calls']==0

def test_event_missing_field_rejected_even_with_rechained_events(prepared):
    p,d,c=prepared;m.run(ra(p),client_factory=factory,guard_factory=NoGuard);events=sorted((m.run_path()/'attempt_ledger').iterdir());previous=None
    for i,path in enumerate(events):
        ev=m.read(path);ev['previous_sha256']=previous
        if i>=1:ev['ledger'].pop('scope',None)
        path.write_bytes(m.canonical(ev));previous=m.pilot.digest(path)
    ledger=m.read(m.run_path()/'provider_calls/ledger.json');ledger.pop('scope');(m.run_path()/'provider_calls/ledger.json').write_bytes(m.canonical(ledger))
    with pytest.raises(ValueError,match='field set'):m.inspect_calls(c,d['jobs'],m.run_path(),True)

def test_global_registration_blocks_second_plan(prepared,tmp_path):
    p,d,c=prepared;m.registration_path().write_text('{}')
    with pytest.raises(ValueError):m.prepare(SimpleNamespace(output=tmp_path/'second'))

def test_entire_budget_rejected_not_trimmed(prepared):
    _,d,_=prepared;d['jobs'][0]['reservation_usd']='0.101'
    with pytest.raises(ValueError):m.prediction(d)
