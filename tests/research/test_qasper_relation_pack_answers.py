"""Synthetic full-pack, cache, budget, failure and descriptive-statistics checks."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_pack_answers as stage
from test_qasper_local_answer_evaluation import fixture as native_fixture,response


@pytest.fixture(autouse=True)
def offline_clock(monkeypatch):
    monkeypatch.setattr(stage,'utc_now',lambda:datetime(2026,9,27,0,0,tzinfo=timezone.utc))
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return stage.utc_now()
    monkeypatch.setattr(stage.legacy,'datetime',Clock)
    monkeypatch.setattr(stage.legacy.client,'read_key',lambda *a,**k:pytest.fail('real key forbidden'))


def fixture(n=1):
    original,_,_,gold=native_fixture();units=deepcopy(original['documents']['doc'])
    units.append({**units[1],'unit_id':'u3','order':2,'text':'Green','native_text':'Green','start':9,'end':14})
    data={'cases':[],'queries':[],'masks':[],'ancestor_mappings':{'primary':[],'local':[]},'cache':{},
          'input_sha256':{},'prior':deepcopy(stage.PRIOR),'references':{'path':'unused','sha256':'unused'},'old_records':{}}
    annotations={}
    for i in range(n):
        identity={'family_id':f'f{i%24}','doc_id':f'd{i%24}','question_id':f'q{i}'};question=f'Color {i}?'
        count=2 if n==1 else 1 if i<60 else 2 if i<75 else 4
        selecteds=[[0]] if count==1 else [[0],[1]] if count==2 else [[0],[1],[2],[2]]
        case={**identity,'units':deepcopy(units),'candidates':[0,1,2]};data['cases'].append(case)
        data['queries'].append({**identity,'mask_count':count,'eligible_edges':[] if count==1 else [[0,1]] if count==2 else [[0,1],[1,2]],'distinct_pack_count':len({tuple(s) for s in selecteds})})
        for mask,selected in enumerate(selecteds):
            pack=stage.parent.render_pack([stage.parent.Unit(**u) for u in units],selected)
            row={**identity,'mask':mask,'selected_indices':selected,'selected_ids':[units[j]['unit_id'] for j in selected],
                'actual_tokens':len(pack),'pack_sha256':hashlib.sha256(pack.encode()).hexdigest()}
            data['masks'].append(row)
            if mask==0:
                job=stage.job_for(question,pack);key=job['cache_key']
                data['cache'][key]={'job':job,'answer':'Blue','origin':'primary','request_ordinal':i+1,'request_sha256':stage.object_hash(job['payload']),
                    'response_sha256':'a'*64,'resolved_model':stage.MODEL}
                for method in stage.BASELINES:
                    data['ancestor_mappings']['local' if method=='reranker_k3' else 'primary'].append({**identity,'method':method,
                        'cache_key':key,'selected_ids':row['selected_ids'],'pack_sha256':row['pack_sha256'],'actual_evidence_tokens':len(pack)})
        annotations[identity['doc_id'],identity['question_id']]=deepcopy(gold['doc','q'])
    return data,annotations


def built(n=1):
    data,gold=fixture(n);parts=stage.build_jobs(data)
    return data,gold,*parts


def small_spec(monkeypatch,data,jobs,packs,masks,baselines,inheritance):
    monkeypatch.setattr(stage,'SPEC',{**stage.SPEC,'questions':1,'families':1,'mask_mappings':2,'distinct_query_packs':2,
        'distinct_pack_histogram':{'2':1},'unique_pack_payloads':2,'cached_pack_payloads':1,'new_unique_requests':1,
        'new_input_allowance':jobs[0]['input_allowance'],'new_output_allowance':512,'new_reservation_usd':jobs[0]['reserved_usd']})


def cfg(data,jobs):
    return {'prior_night_accounting':data['prior'],'parent_resolved_model':stage.MODEL,
        'answer_reservation_usd':str(sum((Decimal(j['reserved_usd']) for j in jobs),Decimal(0)))}


def client(tmp_path,data,jobs,transport):
    return stage.RelationPackClient(tmp_path/'provider_calls',prior_reservation=data['prior']['reservation_usd'],
        jobs=jobs,parent_model=stage.MODEL,transport=transport,started=time.monotonic())


def test_all_masks_deduplicate_only_exact_full_payload_and_bridge_original_I():
    data,_,jobs,packs,masks,baselines,inheritance=built()
    assert len(jobs)==1 and len(packs)==2 and len(masks)==2 and len(baselines)==4 and len(inheritance)==1
    assert len({r['cache_key'] for r in baselines})==1
    assert {r['cache_key'] for r in packs}==set(inheritance)|{jobs[0]['cache_key']}
    content=json.loads(jobs[0]['payload']['messages'][1]['content'])
    assert content=={'question':'Color 0?','evidence':'[u2]\nRed'}
    assert jobs[0]==stage.job_for(content['question'],content['evidence'])


@pytest.mark.parametrize('mutation',['missing_mask','duplicate_mask','pack_hash','selected_ids','over1024','missing_baseline','query','cache_payload','cache_model'])
def test_corrupted_pack_cube_or_inexact_cache_is_rejected(mutation):
    data,_=fixture()
    if mutation=='missing_mask':data['masks'].pop()
    elif mutation=='duplicate_mask':data['masks'][1]['mask']=0
    elif mutation=='pack_hash':data['masks'][1]['pack_sha256']='wrong'
    elif mutation=='selected_ids':data['masks'][1]['selected_ids']=['u1']
    elif mutation=='over1024':data['masks'][1]['actual_tokens']=1025
    elif mutation=='missing_baseline':data['ancestor_mappings']['local'].clear()
    elif mutation=='query':data['ancestor_mappings']['local'][0]['question_id']='outside'
    elif mutation=='cache_payload':next(iter(data['cache'].values()))['job']['payload']['temperature']=1
    elif mutation=='cache_model':next(iter(data['cache'].values()))['resolved_model']='qwen/qwen3.6-plus-04-02'
    with pytest.raises(ValueError):stage.build_jobs(data)


def test_complete_scope_exact_reservation_and_never_refund_unknown(monkeypatch):
    data,_,*parts=built();jobs,packs,masks,baselines,inheritance=parts
    with pytest.raises(ValueError,match='complete fixed'):stage.scope(data,*parts)
    small_spec(monkeypatch,data,*parts)
    counts,amount=stage.scope(data,*parts)
    assert counts['all_required_payloads']==2 and amount==Decimal(jobs[0]['reserved_usd'])
    data['prior']['unknown_cost_attempts']=0
    with pytest.raises(ValueError,match='historical'):stage.scope(data,*parts)
    data['prior']=deepcopy(stage.PRIOR);jobs[0]['reserved_usd']=str(stage.HEADROOM+Decimal('.0000000001'))
    with pytest.raises(ValueError,match='over remaining'):stage.scope(data,*parts)


@pytest.mark.parametrize('error',[TimeoutError,KeyboardInterrupt])
def test_failed_request_reservation_is_retained_and_no_followup_or_retry(tmp_path,error):
    data,_,jobs,*_=built();sent=[]
    def transport(payload):sent.append(payload);raise error()
    bounded=client(tmp_path,data,jobs,transport)
    with pytest.raises((RuntimeError,KeyboardInterrupt)):bounded.submit(jobs[0])
    with pytest.raises(ValueError):bounded.submit(jobs[0])
    _,ledger,_,complete=stage.inspect_calls(cfg(data,jobs),jobs,tmp_path,require_complete=False)
    totals=stage.accounting(cfg(data,jobs),ledger,False)
    assert not complete and len(sent)==1 and totals['night_unknown_cost_attempts']==2
    assert Decimal(totals['night_attempted_reservation_usd'])==Decimal(stage.PRIOR['reservation_usd'])+Decimal(jobs[0]['reserved_usd'])


@pytest.mark.parametrize('change',[{'model':'qwen/qwen3.6-plus-04-02'},{'choices':[{'finish_reason':'length','message':{'content':'{"answer":"Red"}'}}]}])
def test_actual_model_alias_or_truncation_fail_closed_but_keep_known_fee(tmp_path,change):
    data,_,jobs,*_=built();bounded=client(tmp_path,data,jobs,lambda p:response('Red',**change))
    with pytest.raises(RuntimeError):bounded.submit(jobs[0])
    _,ledger,_,complete=stage.inspect_calls(cfg(data,jobs),jobs,tmp_path,require_complete=False)
    assert not complete and ledger['actual_reported_cost_usd']=='0.0001'
    assert stage.accounting(cfg(data,jobs),ledger,False)['night_unknown_cost_attempts']==1


def test_cutoff_margin_stops_without_reservation_or_dispatch(tmp_path,monkeypatch):
    data,_,jobs,*_=built();sent=[];bounded=client(tmp_path,data,jobs,lambda p:sent.append(p))
    monkeypatch.setattr(stage,'utc_now',lambda:datetime(2026,9,27,0,58,55,tzinfo=timezone.utc))
    with pytest.raises(TimeoutError):bounded.submit(jobs[0])
    assert not sent and not bounded.ledger['attempts']


def test_partial_response_never_loads_references_or_scores(tmp_path,monkeypatch):
    data,_,jobs,packs,_,baselines,inheritance=built()
    bounded=client(tmp_path,data,jobs,lambda p:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError):bounded.submit(jobs[0])
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:pytest.fail('partial references forbidden'))
    monkeypatch.setattr(stage,'load_references',lambda *a:pytest.fail('partial references forbidden'))
    out=stage.results(cfg(data,jobs),data,jobs,packs,baselines,inheritance,{},tmp_path,require_complete=False)
    assert out[:5]==(None,None,None,None,None)


@pytest.mark.parametrize('mutation',['source','failure'])
def test_complete_but_invalid_source_rejected_before_gold(tmp_path,monkeypatch,mutation):
    data,_,jobs,packs,_,baselines,inheritance=built();config=cfg(data,jobs)
    source=tmp_path/'source';source.write_text('original');config['input_sha256']={str(source):stage.legacy.digest(source)}
    monkeypatch.setattr(stage,'inspect_calls',lambda *a,**k:({}, {}, {}, True))
    monkeypatch.setattr(stage,'registration_binding',lambda *a:{})
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:pytest.fail('invalid inputs must not reach references'))
    monkeypatch.setattr(stage,'load_references',lambda *a:pytest.fail('invalid inputs must not reach references'))
    if mutation=='source':source.write_text('changed')
    else:(tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError):stage.results(config,data,jobs,packs,baselines,inheritance,{},tmp_path,require_complete=True)


def test_all_96_packs_385_method_records_16_intervals_and_descriptive_envelope():
    data,gold,jobs,packs,masks,baselines,inheritance=built(77)
    answers={k:v['answer'] for k,v in data['cache'].items()}|{j['cache_key']:'Red' for j in jobs}
    pack_records,records,envelope,summary=stage.score_all(data,packs,baselines,answers,gold)
    assert len(pack_records)==96 and len(records)==385 and len(envelope)==77
    assert len(summary['metrics'])==5 and len(summary['paired_comparisons'])==4
    assert sum(2*len(p['metrics']) for p in summary['paired_comparisons'])==16
    assert [(r['plus'],r['minus']) for r in summary['paired_comparisons']]==list(stage.PAIRS)
    for pair in summary['paired_comparisons']:
        for metric in stage.METRICS:
            value=pair['metrics'][metric]
            assert value['question_positive']+value['question_ties']+value['question_negative']==77
            assert value['questions']==77 and value['families']==24
    f1=summary['paired_comparisons'][0]['metrics']['official_answer_f1']
    assert f1['question_weighted']['delta']!=f1['family_balanced']['delta']
    assert summary['observed_envelope']['maximum_minus_I']['question_ties']==77
    assert summary['observed_envelope']['minimum_minus_I']['question_negative']==17
    assert 'bootstrap_percentile_95' not in json.dumps(summary['observed_envelope'])
    assert sum(x['count'] for x in summary['distinct_pack_distribution']['answer_f1'])==96
    with pytest.raises(ValueError):stage.score_all(data,packs[:-1],baselines,answers,gold)
    with pytest.raises(ValueError):stage.score_all(data,packs,baselines[:-1],answers,gold)


def prepared_plan(tmp_path,monkeypatch):
    data,_,*parts=built();small_spec(monkeypatch,data,*parts)
    monkeypatch.setattr(stage,'source_data',lambda:deepcopy(data))
    monkeypatch.setattr(stage,'registration_path',lambda:tmp_path/'registration.json')
    monkeypatch.setattr(stage,'run_path',lambda:tmp_path/'run')
    args=SimpleNamespace(output=tmp_path/'plan');stage.prepare(args)
    return args


def test_prepare_rebuild_is_gold_free_and_not_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:pytest.fail('prepare references forbidden'))
    args=prepared_plan(tmp_path,monkeypatch)
    config,*_=stage.load_plan(args.output)
    assert config['new_unique_requests']==1 and config['gold_loaded'] is False
    assert not stage.run_path().exists() and not stage.registration_path().exists()
    with pytest.raises(FileExistsError):stage.prepare(args)


@pytest.mark.parametrize('field,value',[('answer_reservation_usd','0'),('new_unique_requests',0),('parent_resolved_model','other'),('gold_loaded',True),('run_output','other')])
def test_resealed_plan_cannot_change_budget_model_scope_or_fixed_run(tmp_path,monkeypatch,field,value):
    args=prepared_plan(tmp_path,monkeypatch);path=args.output/'experiment_config.json';config=stage.parent.read(path)
    config[field]=value;path.write_text(json.dumps(config))
    (args.output/'plan_manifest.json').write_text(json.dumps({'schema':stage.SCHEMA,'experiment_config_sha256':stage.legacy.digest(path)}))
    with pytest.raises(ValueError):stage.load_plan(args.output)


class Guard:
    def __init__(self,*a,**k):pass
    def close(self):pass


def test_failed_run_single_use_cannot_be_reset_through_new_plan(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    def factory(*a,**k):
        k.pop('key_file');k.pop('proxy')
        return stage.RelationPackClient(*a,**k,transport=lambda p:(_ for _ in ()).throw(TimeoutError()))
    run_args=SimpleNamespace(plan=args.output)
    with pytest.raises(RuntimeError):stage.run(run_args,client_factory=factory,guard_factory=Guard)
    assert not (stage.run_path()/'summary.json').exists()
    assert stage.audit(SimpleNamespace(plan=args.output,run=stage.run_path(),allow_incomplete=True))['status']=='verified_incomplete'
    with pytest.raises(FileExistsError):stage.run(run_args,client_factory=factory,guard_factory=Guard)
    with pytest.raises(FileExistsError):stage.prepare(SimpleNamespace(output=tmp_path/'plan2'))


def test_hard_watchdog_is_65_seconds_and_only_terminates_own_python(tmp_path,monkeypatch):
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
    assert stage.parent.read(tmp_path/'hard_deadline.json')['main_results_available'] is False


def complete_run(tmp_path,monkeypatch):
    data,gold,*parts=built();jobs,packs,masks,baselines,inheritance=parts
    small_spec(monkeypatch,data,*parts)
    reference=tmp_path/'references'
    stage.legacy.pilot.write_rows(reference,[{'doc_id':q['doc_id'],'question_id':q['question_id'],'family_id':q['family_id'],
        'question':q['query'],'official_split':'validation','answer_annotations':gold[q['doc_id'],q['question_id']]} for q in data['prepared']['queries']])
    data['references']={'path':str(reference),'sha256':stage.legacy.digest(reference)}
    for origin,methods in [('primary',{'I_jev_k3','dense_k3','p_yes_only_k3'}),('local',{'reranker_k3'})]:
        path=tmp_path/(origin+'_records.jsonl')
        stage.legacy.pilot.write_rows(path,[{**r,'predicted_answer':'Blue','official_answer_f1':1.0} for r in baselines if r['method'] in methods])
        data['old_records'][origin]={'path':str(path),'sha256':stage.legacy.digest(path)}
    monkeypatch.setattr(stage,'source_data',lambda:deepcopy(data))
    monkeypatch.setattr(stage,'registration_path',lambda:tmp_path/'registration.json')
    monkeypatch.setattr(stage,'run_path',lambda:tmp_path/'run')
    monkeypatch.setattr(stage.legacy.pilot,'selected_gold',lambda *a:deepcopy(gold))
    plan=tmp_path/'plan';stage.prepare(SimpleNamespace(output=plan))
    def factory(*a,**k):
        k.pop('key_file');k.pop('proxy')
        return stage.RelationPackClient(*a,**k,transport=lambda p:response('Red'))
    stage.run(SimpleNamespace(plan=plan),client_factory=factory,guard_factory=Guard)
    return plan


def test_complete_generation_then_full_raw_response_official_score_and_metadata_replay(tmp_path,monkeypatch):
    plan=complete_run(tmp_path,monkeypatch)
    result=stage.audit(SimpleNamespace(plan=plan,run=stage.run_path(),allow_incomplete=False))
    assert result['status']=='verified_complete' and result['accounting']['new_api_calls']==1
    assert result['accounting']['night_unknown_cost_attempts']==1 and result['accounting']['night_attempts']==804
    summary=stage.parent.read(stage.run_path()/'summary.json')
    assert summary['pack_record_count']==2 and summary['record_count']==5 and summary['baseline_records_verified']==4
    assert len(summary['paired_comparisons'])==4


@pytest.mark.parametrize('mutation',['accounting','comparison','envelope','answers','extra','failure','registration'])
def test_complete_audit_rejects_self_resealed_scientific_or_metadata_change(tmp_path,monkeypatch,mutation):
    plan=complete_run(tmp_path,monkeypatch);output=stage.run_path();path=output/'summary.json';summary=stage.parent.read(path)
    if mutation=='accounting':summary['night_unknown_cost_attempts']=0
    elif mutation=='comparison':summary['paired_comparisons'][0]['metrics']['official_answer_f1']['question_weighted']['delta']=1
    elif mutation=='envelope':summary['observed_envelope']['maximum_minus_I']['question_weighted']=.5
    elif mutation=='answers':
        p=output/'answers.json';value=stage.parent.read(p);value[next(iter(value))]='changed';p.write_text(json.dumps(value))
    elif mutation=='extra':(output/'extra.json').write_text('{}')
    elif mutation=='failure':(output/'failure.json').write_text('{}')
    elif mutation=='registration':
        p=stage.registration_path();value=stage.parent.read(p);value['registered_at_utc']='2026-09-26T23:00:00+00:00';p.write_text(json.dumps(value))
    summary['output_sha256']={str(Path(p).relative_to(output)):v for p,v in stage.inherited.tree(output).items() if Path(p)!=path}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):stage.audit(SimpleNamespace(plan=plan,run=output,allow_incomplete=False))


def test_hash_bound_reader_rejects_change_and_duplicate_json_keys(tmp_path):
    source=tmp_path/'source.json';source.write_text('{"a":1}')
    reader=stage.Reader();reader.json(source);source.write_text('{"a":2}')
    with pytest.raises(ValueError):reader.verify()
    source.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):stage.Reader().json(source)
