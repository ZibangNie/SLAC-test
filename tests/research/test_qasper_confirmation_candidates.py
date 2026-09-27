"""Synthetic-only export/token/embedding fixtures; no real QA, GPU or network."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import prepare_qasper_confirmation_candidates as m


class Tokenizer:
    def encode(self,text,add_special_tokens=True,truncation=False):
        assert add_special_tokens and not truncation
        return [0,*[3+ord(c)%100 for c in text],2]


class Guard:
    def __init__(self,*args):pass
    def close(self):pass


def overwrite(path,value):path.write_bytes(m.canonical(value)+b'\n')


def forbidden(*args,**kwargs):raise AssertionError('real GPU/model/tokenizer call forbidden')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(m.gpu_gate,'confirm_idle',forbidden)
    monkeypatch.setattr(m.AutoModel,'from_pretrained',forbidden)
    monkeypatch.setattr(m.AutoTokenizer,'from_pretrained',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ARTIFACTS',tmp_path)
    export=tmp_path/'export';export.mkdir();documents=[];questions=[]
    for di,count in enumerate([19,4]):
        ident={'doc_id':f'doc-{di}','source_id':f'source-{di}','family_id':f'component-{di}','original_family_id':f'original-{di}'}
        units=[{'unit_id':f'u{i}','order':i,'kind':'paragraph','start':i*10,'end':i*10+4,'text':f'unit {i}', 'native_text':f'unit {i}'} for i in range(count)]
        documents.append(ident|{'units':units})
        for qi in range(3):questions.append(ident|{'question_id':f'q{qi}','question':f'Question {di} {qi}?'})
    m.write_rows(export/'documents.jsonl',documents);m.write_rows(export/'questions.jsonl',questions)
    # Poisoned references are never opened, even for a hash.
    (export/'references.jsonl').mkdir()
    summary={'schema':'slac-qasper-confirmation-export-v1','status':'completed','ready_for_candidate_preparation':True,
             'documents':len(documents),'questions':len(questions),'output_sha256':{
                 name:m.digest(export/name) for name in ('documents.jsonl','questions.jsonl')}}
    summary['output_sha256']['references.jsonl']='a'*64;m.write(export/'summary.json',summary)
    model=tmp_path/'model'/m.MODEL_REVISION;model.mkdir(parents=True)
    (model/'model.safetensors').write_bytes(b'synthetic weights')
    for name in m.TOKENIZER_FILES:(model/name).write_bytes(b'synthetic small tokenizer')
    monkeypatch.setattr(m,'WEIGHTS_BYTES',(model/'model.safetensors').stat().st_size)
    monkeypatch.setattr(m,'WEIGHTS_SHA256',m.digest(model/'model.safetensors'))
    monkeypatch.setattr(m,'TOKENIZER_HASHES',{name:m.digest(model/name) for name in m.TOKENIZER_FILES})
    source=tmp_path/'source.py';source.write_text('synthetic source',encoding='utf-8')
    monkeypatch.setattr(m,'source_paths',lambda:[source]);monkeypatch.setattr(m,'tokenizer_for',lambda path:Tokenizer())
    return SimpleNamespace(export=export,model=model,documents=documents,questions=questions,plan=tmp_path/'plan',run=tmp_path/'run',audit=tmp_path/'audit',source=source)


def refresh(f):
    s=m.read(f.export/'summary.json')
    for name in ('documents.jsonl','questions.jsonl'):s['output_sha256'][name]=m.digest(f.export/name)
    overwrite(f.export/'summary.json',s)


def prepare(f):return m.prepare(f.export,f.plan,f.run,f.model)


def vectors(data):
    n=sum(len(d['units']) for d in data['documents'])+len(data['questions'])
    value=torch.zeros((n,1024),dtype=torch.float32)
    for i in range(n):value[i,i%3]=1
    return value


def fake_encoder(model,tokenizer,ids,deadline):
    value=torch.zeros((len(ids),1024),dtype=torch.float32)
    for i in range(len(ids)):value[i,i%3]=1
    return value,{'device':'synthetic','torch':torch.__version__,'cuda_runtime':'synthetic','peak_allocated_bytes':100,'peak_reserved_bytes':200}


def run(f,monkeypatch,encoder=fake_encoder):
    monkeypatch.setattr(m.gpu_gate,'confirm_idle',lambda: [{'synthetic_idle':True}]*3)
    return m.run(f.plan,True,encoder=encoder,guard_factory=Guard)


def reseal(f):overwrite(f.plan/'seal.json',{n:m.digest(f.plan/n) for n in m.PLAN_FILES-{'seal.json'}})


def reseal_run(f):
    summary=m.read(f.run/'summary.json')
    summary['output_sha256']={n:m.digest(f.run/n) for n in m.RUN_FILES-{'summary.json'}}
    overwrite(f.run/'summary.json',summary)


def test_prepare_never_reads_weights_or_references_and_retains_all(fixture,monkeypatch):
    f=fixture;original=m.digest
    def digest(path):
        assert Path(path).name not in ('model.safetensors','references.jsonl')
        return original(path)
    monkeypatch.setattr(m,'digest',digest)
    result=prepare(f);plan,data,tokens,own=m.load_plan(f.plan)
    assert result['questions']==6 and result['documents']==2 and result['units']==23
    assert len(tokens['units'])==23 and len(tokens['queries'])==6 and len(own)==5
    assert data['questions']==f.questions and plan['gold_read'] is False and not f.run.exists()
    assert not any(Path(p).name in ('references.jsonl','model.safetensors') for p in plan['input_sha256'])


@pytest.mark.parametrize('which',['documents','questions'])
def test_changed_consumed_export_hash_stops_prepare(fixture,which):
    f=fixture
    with (f.export/(which+'.jsonl')).open('ab') as h:h.write(b' ')
    with pytest.raises(ValueError,match='hash'):prepare(f)
    assert m.read(f.plan/'not_ready.json')['status']=='not_ready'


@pytest.mark.parametrize('field,value',[('status','partial'),('ready_for_candidate_preparation',False),('questions',5),('schema','other')])
def test_complete_export_contract(fixture,field,value):
    f=fixture;s=m.read(f.export/'summary.json');s[field]=value;overwrite(f.export/'summary.json',s)
    with pytest.raises(ValueError):prepare(f)
    assert (f.plan/'not_ready.json').exists() and not f.run.exists()


@pytest.mark.parametrize('mutation',['duplicate_doc','duplicate_q','wrong_component','extra_gold_field','bad_order','duplicate_unit','missing_doc'])
def test_structure_failure_retained_not_ready(fixture,mutation):
    f=fixture;docs=deepcopy(f.documents);qs=deepcopy(f.questions)
    if mutation=='duplicate_doc':docs[1]['doc_id']=docs[0]['doc_id']
    elif mutation=='duplicate_q':qs[1]=deepcopy(qs[0])
    elif mutation=='wrong_component':qs[0]['family_id']='other'
    elif mutation=='extra_gold_field':qs[0]['answer_annotations']=[]
    elif mutation=='bad_order':docs[0]['units'][1]['order']=True
    elif mutation=='duplicate_unit':docs[0]['units'][1]['unit_id']='u0'
    else:qs[0]['doc_id']='missing'
    (f.export/'documents.jsonl').unlink();m.write_rows(f.export/'documents.jsonl',docs)
    (f.export/'questions.jsonl').unlink();m.write_rows(f.export/'questions.jsonl',qs);refresh(f)
    with pytest.raises(ValueError):prepare(f)
    assert (f.plan/'not_ready.json').exists()


def test_overlength_keeps_full_length_audit_and_blocks_run(fixture):
    f=fixture;qs=deepcopy(f.questions);qs[0]['question']='x'*8192
    (f.export/'questions.jsonl').unlink();m.write_rows(f.export/'questions.jsonl',qs);refresh(f)
    with pytest.raises(ValueError,match='overlength'):prepare(f)
    assert m.read(f.plan/'length_audit.json')['queries']['max']==8194
    with pytest.raises(ValueError,match='inventory'):m.load_plan(f.plan)


@pytest.mark.parametrize('mutation',['extra','config','projection','source','token_count'])
def test_plan_tampering_even_with_new_seal(fixture,mutation):
    f=fixture;prepare(f)
    if mutation=='extra':(f.plan/'extra').write_text('x')
    elif mutation=='config':
        p=m.read(f.plan/'plan.json');p['config']['microbatch']=8;overwrite(f.plan/'plan.json',p);reseal(f)
    elif mutation=='projection':
        p=m.read(f.plan/'input.json');p['questions'][0]['question']='changed';overwrite(f.plan/'input.json',p);reseal(f)
    elif mutation=='source':f.source.write_text('changed')
    else:
        p=m.read(f.plan/'token_ids.json');p['queries'].pop();overwrite(f.plan/'token_ids.json',p);reseal(f)
    with pytest.raises(ValueError):m.load_plan(f.plan)


def test_nonoverlap_and_exclusive_prepare(fixture):
    f=fixture
    with pytest.raises(ValueError,match='overlaps'):m.prepare(f.export,f.plan,f.plan/'run',f.model)
    assert not f.plan.exists()
    prepare(f)
    with pytest.raises(ValueError,match='new'):prepare(f)


def test_full_synthetic_run_and_bounded_audit(fixture,monkeypatch):
    f=fixture;prepare(f);calls=[];original=m.digest
    def digest(path):
        if Path(path).name=='model.safetensors':calls.append(str(path))
        assert Path(path).name!='references.jsonl'
        return original(path)
    monkeypatch.setattr(m,'digest',digest)
    result=run(f,monkeypatch)
    assert result['status']=='completed_candidates' and len(calls)==1
    assert result['counts']['questions']==6 and result['support_pairs']==60
    prepared=m.read(f.run/'prepared.json');assert len(prepared['queries'])==6 and not prepared['static_tasks']
    assert all(len(q['candidate_ids'])<=16 for q in prepared['queries'])
    assert all(p['actual_evidence_tokens']<=1024 and len(p['selected_ids'])<=3 for p in m.rows((f.run/'dense_packs.jsonl').read_bytes()))
    report=m.audit(f.plan,f.audit)
    assert report['status']=='verified_complete_with_sample' and report['sample']['sample_questions']==4 and len(calls)==1
    assert report['sample']['full_numerical_recomputation'] is False
    with pytest.raises(ValueError,match='single-use'):run(f,monkeypatch)


def test_busy_gpu_does_not_create_run(fixture):
    f=fixture;prepare(f)
    with pytest.raises(AssertionError):m.run(f.plan,True,encoder=fake_encoder,guard_factory=Guard)
    assert not f.run.exists()


def test_missing_explicit_admission(fixture):
    f=fixture;prepare(f)
    with pytest.raises(ValueError,match='explicit'):m.run(f.plan)
    assert not f.run.exists()


@pytest.mark.parametrize('kind',['oom','nonfinite','shape','timeout'])
def test_inference_failure_no_success_no_retry(fixture,monkeypatch,kind):
    f=fixture;prepare(f)
    def bad(*args):
        if kind=='oom':raise torch.OutOfMemoryError('synthetic')
        if kind=='timeout':raise TimeoutError('synthetic')
        values,metadata=fake_encoder(*args)
        if kind=='nonfinite':values[0,0]=float('nan')
        else:values=values[:1]
        return values,metadata
    with pytest.raises((ValueError,torch.OutOfMemoryError,TimeoutError)):run(f,monkeypatch,bad)
    assert (f.run/'failure.json').exists() and not (f.run/'summary.json').exists()
    with pytest.raises(ValueError,match='single-use'):run(f,monkeypatch)


def test_weight_changed_stops_before_encoder(fixture,monkeypatch):
    f=fixture;prepare(f);w=f.model/'model.safetensors';w.write_bytes(b'x'*w.stat().st_size)
    with pytest.raises(ValueError,match='weight hash'):run(f,monkeypatch,forbidden)
    assert not (f.run/'summary.json').exists()


@pytest.mark.parametrize('name',['rankings.jsonl','prepared.json','dense_packs.jsonl','embedding_index.json','public_aggregate.json'])
def test_saved_output_tamper_rejected(fixture,monkeypatch,name):
    f=fixture;prepare(f);run(f,monkeypatch)
    with (f.run/name).open('ab') as h:h.write(b' ')
    with pytest.raises(ValueError,match='output hash'):m.audit(f.plan,f.audit)


@pytest.mark.parametrize('kind',['support','index','sample_pack','public'])
def test_resealed_output_semantic_rejection(fixture,monkeypatch,kind):
    f=fixture;prepare(f);run(f,monkeypatch)
    if kind=='support':
        p=m.read(f.run/'prepared.json');p['support_tasks'][0]['item']['query']='changed';overwrite(f.run/'prepared.json',p)
    elif kind=='index':
        p=m.read(f.run/'embedding_index.json');p['queries'].reverse();overwrite(f.run/'embedding_index.json',p)
    elif kind=='sample_pack':
        p=m.rows((f.run/'dense_packs.jsonl').read_bytes());p[0]['actual_evidence_tokens']=1025
        (f.run/'dense_packs.jsonl').unlink();m.write_rows(f.run/'dense_packs.jsonl',p)
    else:
        p=m.read(f.run/'public_aggregate.json');p['support_pairs']+=1;overwrite(f.run/'public_aggregate.json',p)
    reseal_run(f)
    with pytest.raises(ValueError):m.audit(f.plan,f.audit)


def test_sample_is_identity_only_and_bounded():
    data={'questions':[{'doc_id':f'd{i}','question_id':f'q{j}','question':'a'} for i in range(20) for j in range(5)]}
    keys=m.sample_keys(data);changed=deepcopy(data)
    for row in changed['questions']:row['question']='entirely changed text'
    assert len(keys)==16 and len({d for d,q in keys})==8 and m.sample_keys(changed)==keys


def test_whole_pack_overflow_skips_and_exact_native_dedup():
    question={'family_id':'f','doc_id':'d','question_id':'q','source_id':'s','original_family_id':'o','question':'q?'}
    units=[m.Unit('u0',0,'paragraph',0,2000,'x'*2000,'x'*2000),m.Unit('u1',1,'paragraph',0,1,'a','same'),
           m.Unit('u2',2,'paragraph',0,1,'b','same'),m.Unit('u3',3,'paragraph',0,1,'c','c')]
    query,tasks,pack=m.candidate_row(question,units,[0,1,2,3],Tokenizer(),float('inf'))
    assert pack['selected_ids']==['u1','u3'] and len(tasks)==4


def test_watchdog_only_own_process_exit(tmp_path,monkeypatch):
    captured=[];thread_targets=[]
    class Thread:
        def __init__(self,target,daemon):thread_targets.append(target)
        def start(self):pass
        def join(self,timeout):pass
    monkeypatch.setattr(m.threading,'Thread',Thread);monkeypatch.setattr(m.os,'_exit',captured.append)
    guard=m.Watchdog(tmp_path,1800);monkeypatch.setattr(guard.cancel,'wait',lambda seconds:False)
    thread_targets[0]()
    assert captured==[124] and m.read(tmp_path/'hard_timeout.json')['complete'] is False


@pytest.mark.parametrize('kind',['token','length'])
def test_resealed_token_semantics_or_length_audit_rejected(fixture,kind):
    f=fixture;prepare(f)
    if kind=='token':
        data=m.read(f.plan/'input.json');keys=set(m.sample_keys(data));i=next(i for i,q in enumerate(data['questions']) if (q['doc_id'],q['question_id']) in keys)
        tokens=m.read(f.plan/'token_ids.json');tokens['queries'][i][0]+=1;overwrite(f.plan/'token_ids.json',tokens)
    else:
        audit=m.read(f.plan/'length_audit.json');audit['queries']['total_tokens']+=1;overwrite(f.plan/'length_audit.json',audit)
    reseal(f)
    with pytest.raises(ValueError):m.load_plan(f.plan)


def test_private_output_scope(fixture):
    with pytest.raises(ValueError,match='ignored artifact'):m.prepare(fixture.export,fixture.plan.parent.parent/'outside-private',fixture.run,fixture.model)


def test_source_change_during_encoding_prevents_success(fixture,monkeypatch):
    f=fixture;prepare(f)
    def drift(*args):
        result=fake_encoder(*args);f.source.write_text('changed');return result
    with pytest.raises(ValueError,match='source changed'):run(f,monkeypatch,drift)
    assert not (f.run/'summary.json').exists() and (f.run/'failure.json').exists()


def test_deadline_after_summary_write_removes_success(fixture,monkeypatch):
    f=fixture;prepare(f);original=m.check_time
    def check(deadline):
        if (f.run/'summary.json').exists():raise TimeoutError('final deadline')
        original(deadline)
    monkeypatch.setattr(m,'check_time',check)
    with pytest.raises(TimeoutError):run(f,monkeypatch)
    assert not (f.run/'summary.json').exists() and (f.run/'failure.json').exists()


def test_extra_run_file_during_sample_audit_rejected(fixture,monkeypatch):
    f=fixture;prepare(f);run(f,monkeypatch);original=m.sample_check
    def change(*args):
        result=original(*args);(f.run/'extra.json').write_bytes(b'{}');return result
    monkeypatch.setattr(m,'sample_check',change)
    with pytest.raises(ValueError,match='final run inventory'):m.audit(f.plan,f.audit)
