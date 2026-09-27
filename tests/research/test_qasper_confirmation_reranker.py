"""Synthetic parent receipts and model stubs only; no real QA/GPU/API."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import prepare_qasper_confirmation_reranker as m


class PackTokenizer:
    def encode(self,text,add_special_tokens=True,truncation=False):
        assert add_special_tokens and not truncation
        return [0,*[3+ord(c)%100 for c in text],2]


class PairTokenizer:
    def __call__(self,query,passage,add_special_tokens=True,truncation=False,padding=False):
        assert add_special_tokens and not truncation and not padding
        ids=[0,*[3+ord(c)%100 for c in query],2,2,*[3+ord(c)%100 for c in passage],2]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}


class Guard:
    def __init__(self,*args):pass
    def close(self):pass


def forbidden(*args,**kwargs):raise AssertionError('real GPU/model/tokenizer forbidden')
def overwrite(path,value):path.write_bytes(m.canonical(value)+b'\n')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m.base.gpu_gate,'confirm_idle',forbidden)
    monkeypatch.setattr(m.AutoModelForSequenceClassification,'from_pretrained',forbidden)
    monkeypatch.setattr(m.AutoTokenizer,'from_pretrained',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ARTIFACTS',tmp_path)
    run=tmp_path/'candidate';run.mkdir();audit=tmp_path/'candidate-audit';audit.mkdir()
    documents=[];questions=[]
    for di in range(2):
        identity={'doc_id':f'd{di}','family_id':f'c{di}','source_id':f's{di}','original_family_id':f'o{di}'}
        units=[{'unit_id':f'u{i}','order':i,'kind':'paragraph','start':i*5,'end':i*5+3,'text':f'text {i}','native_text':f'text {i}'} for i in range(5)]
        documents.append(identity|{'units':units})
        questions.extend(identity|{'question_id':f'q{j}','question':f'Question {di} {j}?'} for j in range(3))
    data={'documents':documents,'questions':questions};v=torch.zeros((16,1024));v[:,0]=1
    prepared,rankings,packs,index=m.base.produce(data,v,PackTokenizer(),float('inf'))
    sample=m.base.sample_check(data,v,prepared,rankings,packs,PackTokenizer(),float('inf'))
    m.write(run/'prepared.json',prepared);m.write(run/'sample_check.json',sample)
    (run/'references.jsonl').mkdir()  # Poisoned inherited-only path.
    summary={'schema':m.base.SCHEMA,'status':'completed_candidates','gold_read':False,'api_calls':0,
        'input_sha256':{'inherited-only-source':'a'*64},'plan_sha256':{'inherited-only-plan':'b'*64},
        'output_sha256':{n:m.digest(run/n) for n in ('prepared.json','sample_check.json')}}
    summary['output_sha256']['references.jsonl']='c'*64;m.write(run/'summary.json',summary)
    receipt={'schema':m.base.SCHEMA,'status':'verified_complete_with_sample','gold_read':False,'api_calls':0,
        'counts':{'questions':6,'documents':2,'units':10,'components_with_questions':2},'sample':sample,
        'plan_sha256':summary['plan_sha256'],'source_sha256':summary['input_sha256'],
        'run_sha256':{str((run/n).resolve()):m.digest(run/n) for n in ('summary.json','prepared.json','sample_check.json')}}
    m.write(audit/'verification.json',receipt)
    model=tmp_path/'models'/m.REVISION;model.mkdir(parents=True);bge=tmp_path/'bge';bge.mkdir()
    (model/'model.safetensors').write_bytes(b'synthetic reranker weights')
    small={'config.json':{'architectures':['XLMRobertaForSequenceClassification'],'max_position_embeddings':8194,'id2label':{'0':'score'}},
           'tokenizer_config.json':{'model_max_length':8192},'tokenizer.json':{'synthetic':True},'special_tokens_map.json':{},'sentencepiece.bpe.model':{}}
    for name,value in small.items():m.write(model/name,value)
    files={p.name:(p.stat().st_size,'sha256',m.digest(p)) for p in model.iterdir()}
    monkeypatch.setattr(m,'FILES',files);monkeypatch.setattr(m,'WEIGHTS_BYTES',files['model.safetensors'][0]);monkeypatch.setattr(m,'WEIGHTS_SHA256',files['model.safetensors'][2])
    for name in m.base.TOKENIZER_FILES:(bge/name).write_bytes(b'synthetic BGE tokenizer')
    monkeypatch.setattr(m.base,'TOKENIZER_HASHES',{name:m.digest(bge/name) for name in m.base.TOKENIZER_FILES})
    source=tmp_path/'source.py';source.write_text('synthetic source',encoding='utf-8')
    monkeypatch.setattr(m,'source_paths',lambda:[source]);monkeypatch.setattr(m,'tokenizer_for',lambda p:PairTokenizer() if Path(p).name==m.REVISION else PackTokenizer())
    return SimpleNamespace(parent=run,parent_audit=audit,model=model,bge=bge,plan=tmp_path/'plan',run=tmp_path/'run',audit=tmp_path/'audit',source=source)


def parent_reseal(f):
    summary=m.read(f.parent/'summary.json')
    for name in ('prepared.json','sample_check.json'):summary['output_sha256'][name]=m.digest(f.parent/name)
    overwrite(f.parent/'summary.json',summary);receipt=m.read(f.parent_audit/'verification.json')
    receipt['sample']=m.read(f.parent/'sample_check.json')
    receipt['run_sha256']={str((f.parent/n).resolve()):m.digest(f.parent/n) for n in ('summary.json','prepared.json','sample_check.json')}
    overwrite(f.parent_audit/'verification.json',receipt)


def prepare(f):return m.prepare(f.parent,f.parent_audit,f.plan,f.run,f.model,f.bge)
def infer(model,encoded,tokenizer,deadline):return [float(i%5) for i in range(len(encoded))],{
    'device':'synthetic','peak_allocated_bytes':100,'peak_reserved_bytes':200,
    'backbone_dtype':'float16','saved_score_dtype':'float32',**m.padding_metadata(encoded)}
def run(f,monkeypatch,inference=infer):
    monkeypatch.setattr(m.base.gpu_gate,'confirm_idle',lambda:[{'synthetic_idle':True}]*3)
    return m.run(f.plan,True,inference=inference,guard_factory=Guard)
def reseal(f):overwrite(f.plan/'seal.json',{n:m.digest(f.plan/n) for n in m.PLAN_FILES-{'seal.json'}})
def reseal_run(f):
    summary=m.read(f.run/'summary.json');summary['output_sha256']={n:m.digest(f.run/n) for n in m.RUN_FILES-{'summary.json'}}
    overwrite(f.run/'summary.json',summary)


def test_prepare_and_load_no_weight_hash_gold_or_model(fixture,monkeypatch):
    f=fixture;original=m.digest
    def digest(path):
        assert Path(path).name not in ('model.safetensors','references.jsonl')
        return original(path)
    monkeypatch.setattr(m,'digest',digest)
    result=prepare(f);plan,prepared,pairs,encoded,audit,own=m.load_plan(f.plan)
    assert result['questions']==6 and result['support_pairs']==30 and len(encoded)==len(audit)==30
    assert len(own)==6 and not f.run.exists() and plan['gold_read'] is False
    assert not any(Path(p).name in ('model.safetensors','references.jsonl') for p in plan['input_sha256'])


@pytest.mark.parametrize('kind',['summary','audit','failure','sha','sample'])
def test_parent_complete_receipt_contract(fixture,kind):
    f=fixture
    if kind=='failure':(f.parent/'failure.json').write_bytes(b'{}')
    elif kind=='sha':
        with (f.parent/'prepared.json').open('ab') as h:h.write(b' ')
    else:
        path=f.parent/'summary.json' if kind=='summary' else f.parent_audit/'verification.json'
        value=m.read(path)
        if kind=='sample':value['sample']['sample_questions']=0
        else:value['status']='partial'
        overwrite(path,value)
    with pytest.raises(ValueError):prepare(f)
    assert (f.plan/'not_ready.json').exists()


@pytest.mark.parametrize('kind',['missing_pair','duplicate_pair','changed_text','duplicate_query','wrong_candidate'])
def test_full_parent_pair_identity_preserved(fixture,kind):
    f=fixture;p=m.read(f.parent/'prepared.json')
    if kind=='missing_pair':p['support_tasks'].pop()
    elif kind=='duplicate_pair':p['support_tasks'][-1]=deepcopy(p['support_tasks'][0])
    elif kind=='changed_text':p['support_tasks'][0]['item']['unit']['text']='changed'
    elif kind=='duplicate_query':p['queries'][-1]=deepcopy(p['queries'][0])
    else:p['queries'][0]['candidate_ids'][0]='missing'
    overwrite(f.parent/'prepared.json',p);parent_reseal(f)
    with pytest.raises(ValueError):prepare(f)


@pytest.mark.parametrize('text_length,ready',[(1100,True),(8200,False)])
def test_complete_pair_guard_and_full_overlength_inventory(fixture,text_length,ready):
    f=fixture;p=m.read(f.parent/'prepared.json');p['documents']['d0'][0]['text']='x'*text_length
    for t in p['support_tasks']:
        if t['doc_id']=='d0' and t['unit_id']=='u0':t['item']['unit']['text']='x'*text_length
    overwrite(f.parent/'prepared.json',p);parent_reseal(f)
    if ready:prepare(f)
    else:
        with pytest.raises(ValueError,match='overlength'):prepare(f)
    audit=m.read(f.plan/'length_audit.json')
    assert audit['pairs']==30 and audit['pairs_over_1024']==3 and audit['maximum']>text_length
    assert len(m.read(f.plan/'token_ids.json'))==30 and (f.plan/'not_ready.json').exists()!=ready
    if ready:
        plan,_,_,encoded,_,_=m.load_plan(f.plan)
        assert plan['config']['max_pair_tokens']==8192 and max(len(x['input_ids']) for x in encoded)>1024
        assert plan['config']['evidence_budget']==1024
    else:
        with pytest.raises(ValueError,match='inventory'):m.load_plan(f.plan)


@pytest.mark.parametrize('kind',['extra','config','source','pairs','tokens','length'])
def test_plan_resealed_tampering_rejected(fixture,kind):
    f=fixture;prepare(f)
    if kind=='extra':(f.plan/'extra').write_text('x')
    elif kind=='source':f.source.write_text('changed')
    elif kind=='config':
        p=m.read(f.plan/'plan.json');p['config']['max_pair_tokens']=1024;overwrite(f.plan/'plan.json',p)
    elif kind=='pairs':
        p=m.rows((f.plan/'pair_audit.jsonl').read_bytes());p[0]['task_id']='changed'
        (f.plan/'pair_audit.jsonl').unlink();m.write_rows(f.plan/'pair_audit.jsonl',p)
    elif kind=='tokens':
        p=m.read(f.plan/'token_ids.json');p.pop();overwrite(f.plan/'token_ids.json',p)
    else:
        p=m.read(f.plan/'length_audit.json');p['actual_pair_tokens']+=1;overwrite(f.plan/'length_audit.json',p)
    reseal(f)
    with pytest.raises(ValueError):m.load_plan(f.plan)


def test_resealed_same_length_sample_token_change_rejected(fixture):
    f=fixture;prepare(f);plan,p,pairs,encoded,audit,own=m.load_plan(f.plan)
    keys=set(m.sample_identities(p));i=next(i for i,pair in enumerate(pairs) if (pair['doc_id'],pair['question_id']) in keys)
    encoded[i]['input_ids'][0]+=1;audit[i]['encoding_sha256']=hashlib.sha256(m.canonical(encoded[i])).hexdigest()
    overwrite(f.plan/'token_ids.json',encoded);(f.plan/'pair_audit.jsonl').unlink();m.write_rows(f.plan/'pair_audit.jsonl',audit);reseal(f)
    with pytest.raises(ValueError,match='sampled pair'):m.load_plan(f.plan)


def test_full_synthetic_infer_pack_and_sample_audit(fixture,monkeypatch):
    f=fixture;prepare(f);original=m.digest;calls=[]
    def digest(path):
        if Path(path).name=='model.safetensors':calls.append(path)
        assert Path(path).name!='references.jsonl'
        return original(path)
    monkeypatch.setattr(m,'digest',digest)
    public=run(f,monkeypatch);report=m.audit(f.plan,f.audit)
    assert len(calls)==1 and public['method']=='reranker_k3' and public['counts']['questions']==6
    assert report['sample']['sample_questions']==4 and report['sample']['model_rerun'] is False
    assert report['status']=='verified_complete_with_sample'
    assert len(m.rows((f.run/'pair_scores.jsonl').read_bytes()))==30
    assert len(m.rows((f.run/'packs.jsonl').read_bytes()))==6
    with pytest.raises(ValueError,match='single-use'):run(f,monkeypatch)


def test_busy_gpu_does_not_create_run(fixture):
    f=fixture;prepare(f)
    with pytest.raises(AssertionError):m.run(f.plan,True,inference=infer,guard_factory=Guard)
    assert not f.run.exists()


def test_explicit_admission_required(fixture):
    f=fixture;prepare(f)
    with pytest.raises(ValueError,match='explicit'):m.run(f.plan)


@pytest.mark.parametrize('kind',['oom','timeout','nonfinite','missing'])
def test_failure_stops_no_success_no_retry(fixture,monkeypatch,kind):
    f=fixture;prepare(f)
    def failed(*args):
        if kind=='oom':raise torch.OutOfMemoryError('synthetic')
        if kind=='timeout':raise TimeoutError('synthetic')
        scores,execution=infer(*args)
        if kind=='nonfinite':scores[0]=float('inf')
        else:scores.pop()
        return scores,execution
    with pytest.raises((ValueError,torch.OutOfMemoryError,TimeoutError)):run(f,monkeypatch,failed)
    assert (f.run/'failure.json').exists() and not (f.run/'summary.json').exists()
    with pytest.raises(ValueError,match='single-use'):run(f,monkeypatch)


def test_once_weight_hash_stops_before_model_on_drift(fixture,monkeypatch):
    f=fixture;prepare(f);p=f.model/'model.safetensors';p.write_bytes(b'x'*p.stat().st_size)
    with pytest.raises(ValueError,match='weight hash'):run(f,monkeypatch,forbidden)


def test_ranking_ties_and_whole_native_dedup_overflow():
    q={'doc_id':'d','question_id':'q','family_id':'f','candidate_ids':['a','b','c','d'],'ranked_ids':['b','c','a','d']}
    units=[m.Unit('a',0,'paragraph',0,2000,'x'*2000,'x'*2000),m.Unit('b',1,'paragraph',0,1,'b','same'),
           m.Unit('c',2,'paragraph',0,1,'c','same'),m.Unit('d',3,'paragraph',0,1,'d','d')]
    rank,pack=m.ranked_pack(q,units,dict.fromkeys(q['candidate_ids'],1.),PackTokenizer(),float('inf'))
    assert rank['ranked_ids']==['b','c','a','d'] and pack['selected_ids']==['b','d']


@pytest.mark.parametrize('kind',['raw_identity','raw_nonfinite','missing_query','pack_budget','public','sample'])
def test_resealed_saved_results_rejected(fixture,monkeypatch,kind):
    f=fixture;prepare(f);run(f,monkeypatch)
    if kind.startswith('raw'):
        p=m.rows((f.run/'pair_scores.jsonl').read_bytes())
        if kind=='raw_identity':p[0]['unit_id']='wrong'
        else:p[0]['score']='not-a-number'
        (f.run/'pair_scores.jsonl').unlink();m.write_rows(f.run/'pair_scores.jsonl',p)
    elif kind in ('missing_query','pack_budget'):
        p=m.rows((f.run/'packs.jsonl').read_bytes())
        if kind=='missing_query':p.pop()
        else:p[0]['actual_evidence_tokens']=1025
        (f.run/'packs.jsonl').unlink();m.write_rows(f.run/'packs.jsonl',p)
    elif kind=='public':
        p=m.read(f.run/'public_aggregate.json');p['evidence_token_total']+=1;overwrite(f.run/'public_aggregate.json',p)
    else:
        p=m.read(f.run/'sample_check.json');p['sample_questions']+=1;overwrite(f.run/'sample_check.json',p)
    reseal_run(f)
    with pytest.raises(ValueError):m.audit(f.plan,f.audit)


def test_source_drift_during_inference_no_complete(fixture,monkeypatch):
    f=fixture;prepare(f)
    def drift(*args):
        scores,execution=infer(*args);f.source.write_text('drift');return scores,execution
    with pytest.raises(ValueError,match='source changed'):run(f,monkeypatch,drift)
    assert not (f.run/'summary.json').exists()


def test_final_deadline_after_summary_removes_success(fixture,monkeypatch):
    f=fixture;prepare(f);original=m.base.check_time
    def check(deadline):
        if (f.run/'summary.json').exists():raise TimeoutError('final check')
        original(deadline)
    monkeypatch.setattr(m.base,'check_time',check)
    with pytest.raises(TimeoutError):run(f,monkeypatch)
    assert not (f.run/'summary.json').exists() and (f.run/'failure.json').exists()


def test_private_directory_and_overlap_guards(fixture):
    f=fixture
    with pytest.raises(ValueError,match='ignored'):m.prepare(f.parent,f.parent_audit,f.plan.parent.parent/'outside',f.run,f.model,f.bge)
    with pytest.raises(ValueError,match='overlaps'):m.prepare(f.parent,f.parent_audit,f.plan,f.plan/'run',f.model,f.bge)
    assert not f.plan.exists()


def test_final_audit_inventory_drift_rejected(fixture,monkeypatch):
    f=fixture;prepare(f);run(f,monkeypatch);original=m.sample_check
    def changed(*args):
        result=original(*args);(f.run/'extra').write_text('x');return result
    monkeypatch.setattr(m,'sample_check',changed)
    with pytest.raises(ValueError,match='final run inventory'):m.audit(f.plan,f.audit)


@pytest.mark.parametrize('field,value',[
    ('elapsed_seconds',float('nan')),('elapsed_seconds',-1.),('elapsed_seconds',1801.),
    ('peak_allocated_bytes',6*1024**3+1),('peak_reserved_bytes',7*1024**3+1),
    ('peak_allocated_bytes',-1),('peak_reserved_bytes',float('inf')),
    ('backbone_dtype','float32'),('saved_score_dtype','float16'),('max_microbatch',5),
    ('padded_input_tokens',-1),('max_padded_tokens_per_batch',-1)])
def test_resealed_execution_metadata_rejected(fixture,monkeypatch,field,value):
    f=fixture;prepare(f);run(f,monkeypatch);public=m.read(f.run/'public_aggregate.json')
    if field=='elapsed_seconds':public[field]=value
    else:public['execution'][field]=value
    (f.run/'public_aggregate.json').write_text(json.dumps(public),encoding='utf-8');reseal_run(f)
    with pytest.raises(ValueError):m.audit(f.plan,f.audit)
