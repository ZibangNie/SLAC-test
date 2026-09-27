"""Gold-free restricted export -> complete dense candidates, with bounded sample audit."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

import run_qasper_dense_baseline as dense
import run_qasper_native_dual_index_v2 as gpu_gate
from prepare_qasper_relation_pilot import expand_candidates, stable_hash
from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT/'artifacts/research-foundation'
SCHEMA = 'slac-qasper-confirmation-candidates-v1'
MODEL_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
WEIGHTS_SHA256 = '993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e'
WEIGHTS_BYTES = 2271064456
DEFAULT_MODEL = Path('D:/code/Github/SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots') / MODEL_REVISION
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model')
TOKENIZER_HASHES = {'tokenizer.json':'4829dfefc91c9f9839cf1b554a99243c8911c43439668e2f974176b1925cd138',
    'tokenizer_config.json':'3e5e7d91646b2277098e245f3e73ba565de0e3c28b9e7ee7a9fcc3cd68ae4121',
    'special_tokens_map.json':'66715ae6a0dd4aff4fe228bcaaccac6e52c83fd0ba80992c2be7d0e43b362307',
    'config.json':'aef03cacaae68933fe96fc8b9a673601d30a8b20158f0684a0e51295920714a0',
    'sentencepiece.bpe.model':'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865'}
UNIT_FIELDS = {'unit_id','order','kind','start','end','text','native_text'}
IDENTITY = ('family_id','doc_id','question_id')
DOC_FIELDS = {'doc_id','source_id','original_family_id','family_id','units'}
QUESTION_FIELDS = {'doc_id','source_id','original_family_id','family_id','question_id','question'}
CONFIG = {'model_revision':MODEL_REVISION, 'max_input_tokens':8192, 'microbatch':4,
    'max_runtime_seconds':1800, 'seed':13, 'dtype':'float16', 'pooling':'CLS then FP32 L2',
    'ranking':'FP32 dot product; native-order ties', 'dense_seeds':8, 'candidate_cap':16,
    'max_selected_units':3, 'evidence_budget':1024, 'truncation':False,
    'automatic_retries':0, 'cpu_fallback':False, 'sample_documents':8, 'sample_questions_per_document':2,
    'sample_salt':'SLAC-CONFIRMATION-CANDIDATE-SAMPLE-v1'}
PLAN_FILES = {'plan.json','input.json','token_ids.json','length_audit.json','seal.json'}
RUN_FILES = {'registration.json','embeddings.safetensors','embedding_index.json','rankings.jsonl',
    'prepared.json','dense_packs.jsonl','sample_check.json','public_aggregate.json','summary.json'}


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())
def rows(raw):return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write(path,value):
    with Path(path).open('xb') as handle:handle.write(canonical(value)+b'\n')


def write_rows(path,values):
    with Path(path).open('xb') as handle:
        for value in values:handle.write(canonical(value)+b'\n')


def verify(bindings):
    for path,expected in bindings.items():require(digest(path)==expected,'direct source changed')


def overlap(output,paths):
    output=Path(output).resolve()
    for path in paths:
        path=Path(path).resolve()
        require(not(output==path or output.is_relative_to(path) or path.is_relative_to(output)),'output overlaps input')


def private_output(path):
    path=Path(path).resolve();root=ARTIFACTS.resolve()
    require(path!=root and path.is_relative_to(root),'private output must be below ignored artifact root')
    return path


def check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('candidate runtime exhausted')


def source_paths():
    return [Path(__file__),Path(__file__).with_name('CONFIRMATION_CANDIDATE_PROTOCOL_20260927.md'),
        ROOT/'tests/research/test_qasper_confirmation_candidates.py',Path(dense.__file__),
        Path(gpu_gate.__file__),Path(__file__).with_name('prepare_qasper_relation_pilot.py'),
        Path(__file__).with_name('run_qasper_evidence_baselines.py')]


def validate_input(documents,questions):
    require(bool(documents) and bool(questions),'empty exported population')
    by_doc={};seen=set()
    for row in documents:
        require(set(row)==DOC_FIELDS,'document fields differ')
        require(all(isinstance(row[k],str) and row[k] for k in DOC_FIELDS-{'units'}),'invalid document identity')
        require(row['doc_id'] not in by_doc,'duplicate document')
        require(isinstance(row['units'],list) and bool(row['units']),'empty document units')
        ids=set()
        for index,unit in enumerate(row['units']):
            require(set(unit)==UNIT_FIELDS,'unit fields differ')
            require(type(unit['order']) is int and unit['order']==index,'noncontiguous source order')
            require(all(isinstance(unit[k],str) and unit[k].strip() for k in ('unit_id','kind','text','native_text')),'invalid unit text')
            require(type(unit['start']) is int and type(unit['end']) is int and 0<=unit['start']<=unit['end'],'invalid unit span')
            require(unit['unit_id'] not in ids,'duplicate unit identity');ids.add(unit['unit_id'])
        by_doc[row['doc_id']]=row
    for row in questions:
        require(set(row)==QUESTION_FIELDS,'question fields differ')
        require(all(isinstance(v,str) and v.strip() for v in row.values()),'invalid question or identity')
        key=(row['doc_id'],row['question_id']);require(key not in seen,'duplicate question');seen.add(key)
        require(row['doc_id'] in by_doc,'question document missing')
        doc=by_doc[row['doc_id']]
        require(all(row[k]==doc[k] for k in ('family_id','original_family_id','source_id')),'question identity join differs')
    return {'documents':len(documents),'questions':len(questions),'components':len({r['family_id'] for r in documents}),
        'components_with_questions':len({r['family_id'] for r in questions}),
        'documents_without_questions':len(set(by_doc)-{r['doc_id'] for r in questions}),
        'units':sum(len(r['units']) for r in documents)}


def export_inputs(directory):
    directory=Path(directory).resolve()
    buffers={name:(directory/name).read_bytes() for name in ('summary.json','documents.jsonl','questions.jsonl')}
    bindings={str((directory/name).resolve()):hashlib.sha256(raw).hexdigest() for name,raw in buffers.items()}
    summary=json.loads(buffers['summary.json'])
    require(summary['schema']=='slac-qasper-confirmation-export-v1' and summary['status']=='completed'
            and summary['ready_for_candidate_preparation'] is True,'export is not complete/ready')
    for name in ('documents.jsonl','questions.jsonl'):
        require(summary['output_sha256'][name]==bindings[str((directory/name).resolve())],'export output hash differs')
    documents,questions=rows(buffers['documents.jsonl']),rows(buffers['questions.jsonl'])
    counts=validate_input(documents,questions)
    require(summary['documents']==counts['documents'] and summary['questions']==counts['questions'],'export denominator differs')
    data={'documents':sorted(documents,key=lambda r:r['doc_id']),
          'questions':sorted(questions,key=lambda r:(r['doc_id'],r['question_id']))}
    verify(bindings)
    return data,counts,bindings,summary['output_sha256']


def tokenizer_for(model):return AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)


def model_metadata(model):
    model=Path(model).resolve();weight=model/'model.safetensors'
    require(model.name==MODEL_REVISION,'model revision directory differs')
    require(weight.is_file() and weight.stat().st_size==WEIGHTS_BYTES,'model weight size differs')
    bindings={str((model/name).resolve()):digest(model/name) for name in TOKENIZER_FILES}
    require(all(bindings[str((model/name).resolve())]==TOKENIZER_HASHES[name] for name in TOKENIZER_FILES),'pinned tokenizer differs')
    return bindings


def prepare(export,output,run_output,model=DEFAULT_MODEL):
    output,run_output=private_output(output),private_output(run_output);model=Path(model).resolve()
    require(not output.exists() and not run_output.exists(),'outputs must be new')
    overlap(output,[export,model,run_output]);overlap(run_output,[export,model])
    output.mkdir(parents=True,exist_ok=False)
    try:
        data,counts,bindings,inherited=export_inputs(export)
        bindings.update(model_metadata(model));bindings.update({str(p.resolve()):digest(p) for p in source_paths()})
        tokenizer=tokenizer_for(model)
        unit_texts=[u['text'] for d in data['documents'] for u in d['units']]
        question_texts=[q['question'] for q in data['questions']]
        unit_ids,unit_audit=dense.audit_token_lengths(tokenizer,unit_texts,CONFIG['max_input_tokens'])
        query_ids,query_audit=dense.audit_token_lengths(tokenizer,question_texts,CONFIG['max_input_tokens'])
        audit={'units':unit_audit,'queries':query_audit,'truncation':False}
        write(output/'length_audit.json',audit)
        require(not unit_audit['above_max_length'] and not query_audit['above_max_length'],'overlength input; no dropping or truncation')
        plan={'schema':SCHEMA,'status':'prepared_no_model_inference','config':CONFIG,'counts':counts,
              'export_dir':str(Path(export).resolve()),'run_output':str(run_output),'model_dir':str(model),
              'weights_sha256':WEIGHTS_SHA256,'weights_bytes':WEIGHTS_BYTES,'input_sha256':bindings,
              'inherited_export_output_commitments':inherited,'gold_read':False,'api_calls':0}
        write(output/'input.json',data);write(output/'token_ids.json',{'units':unit_ids,'queries':query_ids});write(output/'plan.json',plan)
        verify(bindings)
        write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
        return {'status':plan['status'],**counts,'gold_read':False,'api_calls':0}
    except BaseException as error:
        write(output/'not_ready.json',{'status':'not_ready','error_type':type(error).__name__,'model_loaded':False,'api_calls':0})
        raise


def load_plan(directory):
    directory=Path(directory).resolve();require({p.name for p in directory.iterdir()}==PLAN_FILES,'plan inventory differs')
    raw={n:(directory/n).read_bytes() for n in PLAN_FILES};own={str((directory/n).resolve()):hashlib.sha256(b).hexdigest() for n,b in raw.items()}
    require(json.loads(raw['seal.json'])=={n:own[str((directory/n).resolve())] for n in PLAN_FILES-{'seal.json'}},'plan seal differs')
    plan,data,tokens=(json.loads(raw[n]) for n in ('plan.json','input.json','token_ids.json'))
    require(plan['schema']==SCHEMA and plan['status']=='prepared_no_model_inference' and plan['config']==CONFIG,'plan contract differs')
    require(plan['weights_sha256']==WEIGHTS_SHA256 and plan['weights_bytes']==WEIGHTS_BYTES and plan['gold_read'] is False and plan['api_calls']==0,'model/gold contract differs')
    require(validate_input(data['documents'],data['questions'])==plan['counts'],'plan count differs')
    actual,counts,bindings,inherited=export_inputs(plan['export_dir'])
    bindings.update(model_metadata(plan['model_dir']));bindings.update({str(p.resolve()):digest(p) for p in source_paths()})
    require(actual==data and counts==plan['counts'] and bindings==plan['input_sha256']
            and inherited==plan['inherited_export_output_commitments'],'source projection or closure differs')
    verify(plan['input_sha256']);verify(own)
    for key,count in [('units',plan['counts']['units']),('queries',plan['counts']['questions'])]:
        require(len(tokens[key])==count,'token count differs')
        require(all(isinstance(v,list) and 0<len(v)<=CONFIG['max_input_tokens'] and all(type(i) is int and i>=0 for i in v) for v in tokens[key]),'invalid token IDs')
    overlap(private_output(plan['run_output']),[directory,plan['export_dir'],plan['model_dir'],*plan['input_sha256']])
    require(read(directory/'length_audit.json')=={'units':token_summary(tokens['units']),
        'queries':token_summary(tokens['queries']),'truncation':False},'saved length audit differs')
    validate_sample_tokens(data,tokens,tokenizer_for(plan['model_dir']))
    verify(own);verify(plan['input_sha256'])
    require({p.name for p in directory.iterdir()}==PLAN_FILES,'final plan inventory differs')
    return plan,data,tokens,own


def sample_keys(data):
    by_doc={}
    for q in data['questions']:by_doc.setdefault(q['doc_id'],[]).append(q)
    rank=lambda value:hashlib.sha256(canonical([CONFIG['sample_salt'],value])).hexdigest()
    docs=sorted(by_doc,key=rank)[:CONFIG['sample_documents']]
    return [(doc,q['question_id']) for doc in docs for q in sorted(by_doc[doc],key=lambda r:rank([doc,r['question_id']]))[:CONFIG['sample_questions_per_document']]]


def token_summary(values):
    lengths=sorted(map(len,values));count=len(lengths)
    return {'count':count,'total_tokens':sum(lengths),'min':min(lengths,default=0),'max':max(lengths,default=0),
        'p50':lengths[(count-1)//2] if count else 0,'p95':lengths[math.ceil(count*.95)-1] if count else 0,
        'above_max_length':sum(n>8192 for n in lengths),'max_length_includes_special_tokens':8192,'truncation':False}


def validate_sample_tokens(data,tokens,tokenizer):
    """Up to sixteen questions and sixteen source units; no full retokenization."""
    keys=set(sample_keys(data));docs={d for d,q in keys};offset=0
    rank=lambda doc,uid:hashlib.sha256(canonical([CONFIG['sample_salt'],doc,uid])).hexdigest()
    for document in data['documents']:
        if document['doc_id'] in docs:
            selected=sorted(range(len(document['units'])),key=lambda i:rank(document['doc_id'],document['units'][i]['unit_id']))[:2]
            for i in selected:
                require(tokens['units'][offset+i]==tokenizer.encode(document['units'][i]['text'],add_special_tokens=True,truncation=False),'sampled unit token IDs differ')
        offset+=len(document['units'])
    for i,q in enumerate(data['questions']):
        if (q['doc_id'],q['question_id']) in keys:
            require(tokens['queries'][i]==tokenizer.encode(q['question'],add_special_tokens=True,truncation=False),'sampled query token IDs differ')


def candidate_row(question,units,ranking,tokenizer,deadline):
    full=[units[i].unit_id for i in ranking];seeds,candidates=expand_candidates(units,full)
    candidate_set=set(candidates);ranked=[uid for uid in full if uid in candidate_set]
    positions={u.unit_id:i for i,u in enumerate(units)}
    counter=PackCounter(tokenizer,units,deadline)
    selected=pack_ranked(units,[positions[uid] for uid in ranked],1024,counter,max_units=3)
    identity={k:question[k] for k in IDENTITY}
    query={**identity,'source_id':question['source_id'],'original_family_id':question['original_family_id'],
           'query':question['question'],'seed_ids':seeds,'candidate_ids':candidates,'ranked_ids':ranked}
    tasks=[]
    for unit in units:
        if unit.unit_id in candidate_set:
            item={'query':query['query'],'unit':{'id':unit.unit_id,'text':unit.text}}
            tasks.append({'id':'support:'+stable_hash({'kind':'support','doc_id':question['doc_id'],'question_id':question['question_id'],'item':item}),
                          'doc_id':question['doc_id'],'question_id':question['question_id'],'unit_id':unit.unit_id,'item':item})
    rendered=render_pack(units,selected)
    pack={**identity,'method':'dense_k3','budget':1024,'selected_ids':[units[i].unit_id for i in selected],
          'actual_evidence_tokens':counter(selected),'pack_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'rendered_pack':rendered}
    return query,tasks,pack


def produce(data,vectors,tokenizer,deadline):
    n=sum(len(d['units']) for d in data['documents']);q=len(data['questions'])
    require(vectors.ndim==2 and tuple(vectors.shape)==(n+q,1024) and torch.isfinite(vectors).all().item(),'embedding shape or values differ')
    require(torch.allclose(torch.linalg.vector_norm(vectors.float(),dim=1),torch.ones(n+q),atol=1e-4,rtol=1e-4),'embedding norms differ')
    offset=0;docs={};index=[]
    for d in data['documents']:
        units=[Unit(**u) for u in d['units']];docs[d['doc_id']]=(units,vectors[offset:offset+len(units)])
        index.extend({'doc_id':d['doc_id'],'unit_id':u.unit_id} for u in units);offset+=len(units)
    rankings=[];queries=[];tasks=[];packs=[]
    for i,question in enumerate(data['questions']):
        check_time(deadline);units,embeddings=docs[question['doc_id']]
        ranking=dense.dense_ranking(vectors[n+i],embeddings,units)
        query,task,pack=candidate_row(question,units,ranking,tokenizer,deadline)
        rankings.append({k:question[k] for k in IDENTITY}|{'ranked_ids':[units[j].unit_id for j in ranking]})
        queries.append(query);tasks.extend(task);packs.append(pack)
    require(len({t['id'] for t in tasks})==len(tasks),'duplicate support task identity')
    prepared={'schema':SCHEMA,'documents':{d['doc_id']:d['units'] for d in data['documents']},'queries':queries,'support_tasks':tasks,'static_tasks':[]}
    embedding_index={'candidates':index,'queries':[{k:r[k] for k in IDENTITY} for r in data['questions']]}
    return prepared,rankings,packs,embedding_index


def sample_check(data,vectors,prepared,rankings,packs,tokenizer,deadline):
    keys=sample_keys(data);wanted=set(keys);n=sum(len(d['units']) for d in data['documents'])
    offset=0;docs={}
    for d in data['documents']:
        units=[Unit(**u) for u in d['units']];docs[d['doc_id']]=(units,vectors[offset:offset+len(units)]);offset+=len(units)
    query_map={(r['doc_id'],r['question_id']):r for r in prepared['queries']}
    rank_map={(r['doc_id'],r['question_id']):r for r in rankings};pack_map={(r['doc_id'],r['question_id']):r for r in packs}
    for i,row in enumerate(data['questions']):
        key=(row['doc_id'],row['question_id'])
        if key not in wanted:continue
        check_time(deadline);units,embeddings=docs[row['doc_id']]
        ranking=dense.dense_ranking(vectors[n+i],embeddings,units)
        query,_,pack=candidate_row(row,units,ranking,tokenizer,deadline)
        require(rank_map[key]['ranked_ids']==[units[j].unit_id for j in ranking] and query_map[key]==query and pack_map[key]==pack,'sample rank/candidate/pack differs')
    return {'sample_documents':len({k[0] for k in keys}),'sample_questions':len(keys),
            'sample_identity_sha256':hashlib.sha256(canonical(keys)).hexdigest(),'sample_rank_pack_equal':True,
            'full_numerical_recomputation':False,'selection':'fixed salted identity hash; max8 documents x max2 questions'}


def validate_saved(data,vectors,prepared,rankings,packs,index):
    """Linear structural checks only; numerical ranking/token replay is sampled."""
    docs={d['doc_id']:[Unit(**u) for u in d['units']] for d in data['documents']}
    n=sum(map(len,docs.values()));questions=data['questions'];expected={(q['doc_id'],q['question_id']) for q in questions}
    require(vectors.dtype==torch.float32 and tuple(vectors.shape)==(n+len(questions),1024)
            and torch.isfinite(vectors).all().item(),'saved embedding shape/dtype/values differ')
    require(torch.allclose(torch.linalg.vector_norm(vectors,dim=1),torch.ones(len(vectors)),atol=1e-4,rtol=1e-4),'saved embedding norms differ')
    expected_index={'candidates':[{'doc_id':doc,'unit_id':u.unit_id} for doc,units in docs.items() for u in units],
                    'queries':[{k:q[k] for k in IDENTITY} for q in questions]}
    require(index==expected_index,'embedding index differs')
    for records in (prepared['queries'],rankings,packs):
        require(len(records)==len(expected) and {(r['doc_id'],r['question_id']) for r in records}==expected,'complete question coverage differs')
    require(prepared['schema']==SCHEMA and prepared['documents']=={d['doc_id']:d['units'] for d in data['documents']}
            and prepared['static_tasks']==[],'prepared document/static contract differs')
    qm={(r['doc_id'],r['question_id']):r for r in prepared['queries']};rm={(r['doc_id'],r['question_id']):r for r in rankings}
    pm={(r['doc_id'],r['question_id']):r for r in packs};expected_tasks=[]
    for question in questions:
        key=(question['doc_id'],question['question_id']);units=docs[key[0]];query,rank,pack=qm[key],rm[key],pm[key]
        require(all(all(record[k]==question[k] for k in IDENTITY) for record in (query,rank,pack)),'record identity differs')
        seeds,candidates=expand_candidates(units,rank['ranked_ids']);candidate_set=set(candidates)
        require(query=={**{k:question[k] for k in IDENTITY},'source_id':question['source_id'],
            'original_family_id':question['original_family_id'],'query':question['question'],'seed_ids':seeds,
            'candidate_ids':candidates,'ranked_ids':[u for u in rank['ranked_ids'] if u in candidate_set]},'candidate projection differs')
        positions={u.unit_id:i for i,u in enumerate(units)};selected=pack['selected_ids']
        require(len(selected)==len(set(selected))<=3 and set(selected)<=candidate_set,'invalid selected identities')
        indices=[positions[u] for u in selected]
        require(indices==sorted(indices) and len({units[i].native_text for i in indices})==len(indices),'selected order/duplicates differ')
        rendered=render_pack(units,indices)
        require(pack['method']=='dense_k3' and pack['budget']==1024 and type(pack['actual_evidence_tokens']) is int
            and 0<=pack['actual_evidence_tokens']<=1024 and (bool(indices) or pack['actual_evidence_tokens']==0)
            and pack['rendered_pack']==rendered and pack['pack_sha256']==hashlib.sha256(rendered.encode()).hexdigest(),'pack structure differs')
        for unit in units:
            if unit.unit_id in candidate_set:
                item={'query':question['question'],'unit':{'id':unit.unit_id,'text':unit.text}}
                expected_tasks.append({'id':'support:'+stable_hash({'kind':'support','doc_id':key[0],'question_id':key[1],'item':item}),
                    'doc_id':key[0],'question_id':key[1],'unit_id':unit.unit_id,'item':item})
    require(prepared['support_tasks']==expected_tasks,'support task content/order differs')


class Watchdog:
    def __init__(self,output,seconds):
        self.cancel=threading.Event()
        def stop():
            if not self.cancel.wait(max(0,seconds)):
                try:write(Path(output)/'hard_timeout.json',{'status':'failed','complete':False})
                finally:os._exit(124)
        self.thread=threading.Thread(target=stop,daemon=True);self.thread.start()
    def close(self):self.cancel.set();self.thread.join(timeout=1)


def encode(model_path,tokenizer,ids,deadline):
    require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
    torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(CONFIG['seed']);torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    model=None
    try:
        model=AutoModel.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False,
            use_safetensors=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda:0')
        model.requires_grad_(False)
        def memory_check(*args):
            require(torch.cuda.max_memory_allocated(0)<=6*1024**3 and torch.cuda.max_memory_reserved(0)<=7*1024**3,'GPU memory limit exceeded')
        vectors=dense.encode_token_ids(model,tokenizer,ids,batch_size=4,device='cuda:0',deadline=deadline,max_length=8192,progress=memory_check)
        return vectors,{'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'cuda_runtime':torch.version.cuda,'backbone_dtype':'float16','pooling_dtype':'float32','saved_vector_dtype':'float32',
            'pooling':'last_hidden_state[:,0] then FP32 L2','peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def run(directory,confirm_idle=False,encoder=encode,guard_factory=Watchdog):
    require(confirm_idle,'explicit idle admission required')
    plan,data,tokens,own=load_plan(directory);output=Path(plan['run_output'])
    require(not output.exists(),'single-use run already exists')
    idle=gpu_gate.confirm_idle()
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+CONFIG['max_runtime_seconds']
    guard=guard_factory(output,CONFIG['max_runtime_seconds'])
    write(output/'registration.json',{'schema':SCHEMA,'plan_sha256':own,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    try:
        # Exactly one weight stream hash per attempted run, immediately before load.
        require(digest(Path(plan['model_dir'])/'model.safetensors')==plan['weights_sha256'],'model weight hash differs')
        check_time(deadline);tokenizer=tokenizer_for(plan['model_dir'])
        vectors,execution=encoder(plan['model_dir'],tokenizer,[*tokens['units'],*tokens['queries']],deadline)
        check_time(deadline);vectors=vectors.float().contiguous()
        prepared,rankings,packs,index=produce(data,vectors,tokenizer,deadline)
        sample=sample_check(data,vectors,prepared,rankings,packs,tokenizer,deadline)
        verify(plan['input_sha256']);verify(own)
        save_file({'embeddings':vectors},str(output/'embeddings.safetensors'),metadata={'revision':MODEL_REVISION,'pooling':'CLS_FP32_L2'})
        write(output/'embedding_index.json',index);write(output/'prepared.json',prepared)
        write_rows(output/'rankings.jsonl',rankings);write_rows(output/'dense_packs.jsonl',packs);write(output/'sample_check.json',sample)
        public={'schema':SCHEMA,'status':'completed_candidates','counts':plan['counts'],'support_pairs':len(prepared['support_tasks']),
            'candidate_count_histogram':dict(Counter(len(q['candidate_ids']) for q in prepared['queries'])),
            'dense_empty_packs':sum(not p['selected_ids'] for p in packs),'dense_evidence_token_total':sum(p['actual_evidence_tokens'] for p in packs),
            'sample':sample,'execution':execution,'elapsed_seconds':time.monotonic()-started,
            'api_calls':0,'gold_read':False,'quality_metrics_computed':False,'training_performed':False,'paid_execution_admitted':False}
        write(output/'public_aggregate.json',public)
        verify(plan['input_sha256']);verify(own);check_time(deadline)
        summary={'schema':SCHEMA,'status':'completed_candidates','plan_sha256':own,'input_sha256':plan['input_sha256'],
            'output_sha256':{n:digest(output/n) for n in RUN_FILES-{'summary.json'}},'weights_sha256_verified_once':plan['weights_sha256'],
            'idle_samples':idle,'elapsed_seconds':time.monotonic()-started,'api_calls':0,'gold_read':False}
        check_time(deadline);write(output/'summary.json',summary);check_time(deadline)
        return public
    except BaseException as error:
        if (output/'summary.json').exists():(output/'summary.json').unlink()
        write(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'complete':False,'retry':False})
        raise
    finally:guard.close()


def audit(directory,output):
    plan,data,tokens,own=load_plan(directory);run=Path(plan['run_output']);output=private_output(output)
    require(not output.exists(),'audit output must be new');overlap(output,[directory,run,*plan['input_sha256']])
    require({p.name for p in run.iterdir()}==RUN_FILES,'incomplete run inventory')
    bindings={str((run/n).resolve()):digest(run/n) for n in RUN_FILES};summary=read(run/'summary.json')
    require(summary['schema']==SCHEMA and summary['status']=='completed_candidates' and summary['api_calls']==0 and summary['gold_read'] is False,'run contract differs')
    require(summary['plan_sha256']==own and summary['input_sha256']==plan['input_sha256'],'run source differs')
    require(summary['output_sha256']=={n:bindings[str((run/n).resolve())] for n in RUN_FILES-{'summary.json'}},'run output hash differs')
    require(summary['weights_sha256_verified_once']==WEIGHTS_SHA256,'weight verification differs')
    require(type(summary['elapsed_seconds']) in (int,float) and math.isfinite(summary['elapsed_seconds']) and 0<=summary['elapsed_seconds']<=1800,'runtime differs')
    vectors=load_file(str(run/'embeddings.safetensors'),device='cpu')['embeddings']
    prepared=read(run/'prepared.json');rankings=rows((run/'rankings.jsonl').read_bytes());packs=rows((run/'dense_packs.jsonl').read_bytes())
    validate_saved(data,vectors,prepared,rankings,packs,read(run/'embedding_index.json'))
    tokenizer=tokenizer_for(plan['model_dir']);sample=sample_check(data,vectors,prepared,rankings,packs,tokenizer,time.monotonic()+1800)
    require(sample==read(run/'sample_check.json'),'sample report differs')
    public=read(run/'public_aggregate.json')
    require(public['status']=='completed_candidates' and public['counts']==plan['counts'] and public['sample']==sample
        and public['support_pairs']==len(prepared['support_tasks']) and public['candidate_count_histogram']==dict(Counter(str(len(q['candidate_ids'])) for q in prepared['queries']))
        and public['dense_empty_packs']==sum(not p['selected_ids'] for p in packs)
        and public['dense_evidence_token_total']==sum(p['actual_evidence_tokens'] for p in packs)
        and all(public[k] is False for k in ('gold_read','quality_metrics_computed','training_performed','paid_execution_admitted'))
        and public['api_calls']==0,'public aggregate differs')
    registration=read(run/'registration.json');require(registration['schema']==SCHEMA and registration['plan_sha256']==own,'registration differs')
    verify(bindings);verify(own);verify(plan['input_sha256'])
    require({p.name for p in run.iterdir()}==RUN_FILES,'final run inventory differs')
    receipt={'schema':SCHEMA,'status':'verified_complete_with_sample','counts':plan['counts'],'sample':sample,
             'source_sha256':plan['input_sha256'],'plan_sha256':own,'run_sha256':bindings,'gold_read':False,'api_calls':0,
             'verification_scope':'complete identity/hash checks; bounded same-function rank/pack sample, not full independent numeric audit'}
    output.mkdir(parents=True,exist_ok=False);write(output/'verification.json',receipt)
    return {'status':receipt['status'],'sample':sample,'gold_read':False,'api_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--export',required=True);p.add_argument('--output',required=True);p.add_argument('--run-output',required=True);p.add_argument('--model',default=str(DEFAULT_MODEL))
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--confirm-idle',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.export,args.output,args.run_output,args.model)
    elif args.command=='run':result=run(args.plan,args.confirm_idle)
    else:result=audit(args.plan,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
