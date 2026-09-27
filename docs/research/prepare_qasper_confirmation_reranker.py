"""Gold-free reranker_k3 on a complete confirmation candidate inventory."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import prepare_qasper_confirmation_candidates as base
import run_qasper_reranker_baseline as old
from run_qasper_evidence_baselines import Unit, PackCounter, pack_ranked, render_pack

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/'artifacts/research-foundation'
SCHEMA='slac-qasper-confirmation-reranker-v1'
REVISION='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
DEFAULT_MODEL=Path('C:/Environment/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots')/REVISION
WEIGHTS_BYTES,_,WEIGHTS_SHA256=old.FILES['model.safetensors']
FILES=old.FILES
IDENTITY=('family_id','doc_id','question_id')
PAIR_FIELDS=('task_id','doc_id','question_id','unit_id')
CONFIG={'model_id':'BAAI/bge-reranker-v2-m3','revision':REVISION,'max_pair_tokens':8192,
    'model_context_tokens':8192,'max_runtime_seconds':1800,'microbatch':4,'dtype':'float16','attention':'sdpa',
    'score':'single classification logit converted to FP32; no sigmoid or normalization',
    'query_instruction':None,'passage_instruction':None,'truncation':False,'tie_break':'original dense rank then native source order',
    'method':'reranker_k3','max_units':3,'evidence_budget':1024,'automatic_retries':0,'cpu_fallback':False,
    'sample_salt':base.CONFIG['sample_salt'],'sample_documents':8,'sample_questions_per_document':2}
PLAN_FILES={'plan.json','prepared.json','token_ids.json','pair_audit.jsonl','length_audit.json','seal.json'}
RUN_FILES={'registration.json','pair_scores.jsonl','rankings.jsonl','packs.jsonl','sample_check.json','public_aggregate.json','summary.json'}
canonical,digest,read,write,write_rows,rows,require,verify=base.canonical,base.digest,base.read,base.write,base.write_rows,base.rows,base.require,base.verify


def private_output(path):
    path=Path(path).resolve();root=ARTIFACTS.resolve()
    require(path!=root and path.is_relative_to(root),'private output outside ignored artifact root')
    return path


def source_paths():
    return [Path(__file__),Path(__file__).with_name('CONFIRMATION_RERANKER_PROTOCOL_20260927.md'),
        ROOT/'tests/research/test_qasper_confirmation_reranker.py',Path(base.__file__),Path(old.__file__),
        Path(__file__).with_name('run_qasper_evidence_baselines.py'),Path(base.gpu_gate.__file__)]


def validate_prepared(prepared):
    require(prepared['schema']==base.SCHEMA and prepared['static_tasks']==[],'candidate schema/static contract differs')
    documents={doc:[Unit(**u) for u in units] for doc,units in prepared['documents'].items()}
    require(bool(documents) and bool(prepared['queries']),'empty candidate population')
    for doc,units in documents.items():
        require(bool(units) and len({u.unit_id for u in units})==len(units) and [u.order for u in units]==list(range(len(units))),'unit order/identity differs')
    seen=set()
    for q in prepared['queries']:
        key=(q['doc_id'],q['question_id']);require(key not in seen and key[0] in documents,'query identity differs');seen.add(key)
        require(all(isinstance(q[k],str) and q[k].strip() for k in (*IDENTITY,'query','source_id','original_family_id')),'invalid query fields')
        ids={u.unit_id for u in documents[key[0]]};c=q['candidate_ids'];rank=q['ranked_ids']
        require(0<len(c)<=16 and len(c)==len(set(c)) and set(c)<=ids and len(rank)==len(c) and set(rank)==set(c),'candidate identity/ranking differs')
        require(c==[u.unit_id for u in documents[key[0]] if u.unit_id in set(c)],'candidate native order differs')
    pairs=old.support_pairs(prepared,documents)
    counts={'questions':len(seen),'documents':len(documents),'components_with_questions':len({q['family_id'] for q in prepared['queries']}),
            'units':sum(map(len,documents.values())),'support_pairs':len(pairs)}
    return documents,pairs,counts


def source_data(candidate_run,candidate_audit):
    run,audit=Path(candidate_run).resolve(),Path(candidate_audit).resolve()
    paths=[run/'summary.json',run/'prepared.json',run/'sample_check.json',audit/'verification.json']
    raw={str(p):p.read_bytes() for p in paths};bindings={p:hashlib.sha256(v).hexdigest() for p,v in raw.items()}
    summary,prepared,sample,receipt=[base.json.loads(raw[str(p)]) for p in paths]
    require(summary['schema']==base.SCHEMA and summary['status']=='completed_candidates' and summary['gold_read'] is False and summary['api_calls']==0,'candidate run incomplete')
    require(not any((run/n).exists() for n in ('failure.json','hard_timeout.json')),'failed candidate run')
    require(receipt['schema']==base.SCHEMA and receipt['status']=='verified_complete_with_sample'
        and receipt['gold_read'] is False and receipt['api_calls']==0,'candidate audit incomplete')
    for path in paths[:3]:require(receipt['run_sha256'][str(path)]==bindings[str(path)],'candidate receipt file differs')
    for name in ('prepared.json','sample_check.json'):require(summary['output_sha256'][name]==bindings[str(run/name)],'candidate output differs')
    require(receipt['plan_sha256']==summary['plan_sha256'] and receipt['source_sha256']==summary['input_sha256'],'candidate receipt ancestry differs')
    require(receipt['sample']==sample and sample['sample_rank_pack_equal'] is True and sample['full_numerical_recomputation'] is False,'candidate sample differs')
    documents,pairs,counts=validate_prepared(prepared)
    require(all(receipt['counts'][k]==counts[k] for k in ('questions','documents','units','components_with_questions')),'candidate complete denominator differs')
    expected_sample=sample_identities(prepared)
    require(sample['sample_questions']==len(expected_sample) and sample['sample_documents']==len({d for d,q in expected_sample})
        and sample['sample_identity_sha256']==hashlib.sha256(canonical(expected_sample)).hexdigest(),'candidate sample identity differs')
    inherited={'candidate_source_sha256':summary['input_sha256'],'candidate_plan_sha256':summary['plan_sha256'],
               'candidate_outputs_sha256':summary['output_sha256'],'candidate_audit_run_sha256':receipt['run_sha256']}
    verify(bindings)
    return prepared,pairs,counts,bindings,inherited


def model_metadata(model,bge):
    model,bge=Path(model).resolve(),Path(bge).resolve();require(model.name==REVISION,'reranker revision differs')
    bindings={}
    for name,(size,kind,expected) in FILES.items():
        path=model/name;require(path.is_file() and path.stat().st_size==size,'model file size differs')
        if name=='model.safetensors':continue
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        actual=sha if kind=='sha256' else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(actual==expected,'pinned model metadata differs');bindings[str(path)]=sha
    config=read(model/'config.json')
    require(config.get('architectures')==['XLMRobertaForSequenceClassification'] and config.get('max_position_embeddings')==8194
            and set(config.get('id2label',{}))=={'0'},'classification architecture differs')
    require(read(model/'tokenizer_config.json')['model_max_length']==8192,'tokenizer context differs')
    for name,expected in base.TOKENIZER_HASHES.items():
        path=bge/name;actual=digest(path);require(actual==expected,'BGE evidence tokenizer differs');bindings[str(path)]=actual
    return bindings


def tokenizer_for(path):return AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)


def encode_pairs(pairs,tokenizer):
    """Record all complete lengths before rejecting an overlength inventory."""
    encoded=[];audit=[]
    for pair in pairs:
        item=dict(tokenizer(pair['query'],pair['passage'],add_special_tokens=True,truncation=False,padding=False))
        ids=item.get('input_ids')
        require(isinstance(ids,list) and ids and all(type(i) is int and i>=0 for i in ids),'invalid token IDs')
        require(all(isinstance(v,list) and len(v)==len(ids) for v in item.values()),'tokenized pair fields differ')
        encoded.append(item);audit.append({k:pair[k] for k in PAIR_FIELDS}|{'pair_tokens':len(ids),'encoding_sha256':hashlib.sha256(canonical(item)).hexdigest()})
    return encoded,audit


def sample_identities(prepared):
    return base.sample_keys({'questions':[{'doc_id':q['doc_id'],'question_id':q['question_id']} for q in prepared['queries']]})


def prepare(candidate_run,candidate_audit,output,run_output,model=DEFAULT_MODEL,bge_tokenizer=base.DEFAULT_MODEL):
    output,run_output=private_output(output),private_output(run_output)
    require(not output.exists() and not run_output.exists(),'outputs must be new')
    model,bge_tokenizer=Path(model).resolve(),Path(bge_tokenizer).resolve()
    for dest in (output,run_output):base.overlap(dest,[candidate_run,candidate_audit,model,bge_tokenizer])
    base.overlap(output,[run_output]);output.mkdir(parents=True,exist_ok=False)
    try:
        prepared,pairs,counts,bindings,inherited=source_data(candidate_run,candidate_audit)
        bindings.update(model_metadata(model,bge_tokenizer));bindings.update({str(p.resolve()):digest(p) for p in source_paths()})
        encoded,audit=encode_pairs(pairs,tokenizer_for(model));lengths=old.length_summary(audit)
        write(output/'token_ids.json',encoded);write_rows(output/'pair_audit.jsonl',audit);write(output/'length_audit.json',lengths)
        require(lengths['maximum']<=CONFIG['max_pair_tokens'],'overlength complete pair; no dropping/truncation')
        plan={'schema':SCHEMA,'status':'prepared_no_model_inference','config':CONFIG,'counts':counts,
              'candidate_run':str(Path(candidate_run).resolve()),'candidate_audit':str(Path(candidate_audit).resolve()),
              'run_output':str(run_output),'model_dir':str(model),'bge_tokenizer':str(bge_tokenizer),
              'weights_sha256':WEIGHTS_SHA256,'weights_bytes':WEIGHTS_BYTES,'input_sha256':bindings,
              'inherited_parent_commitments':inherited,'api_calls':0,'gold_read':False}
        write(output/'prepared.json',prepared);write(output/'plan.json',plan);verify(bindings)
        write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
        return {'status':plan['status'],**counts,'length_audit':lengths,'api_calls':0,'gold_read':False}
    except BaseException as error:
        write(output/'not_ready.json',{'status':'not_ready','error_type':type(error).__name__,'model_loaded':False,'api_calls':0})
        raise


def load_plan(directory):
    directory=Path(directory).resolve();require({p.name for p in directory.iterdir()}==PLAN_FILES,'plan inventory differs')
    raw={n:(directory/n).read_bytes() for n in PLAN_FILES};own={str((directory/n).resolve()):hashlib.sha256(v).hexdigest() for n,v in raw.items()}
    require(base.json.loads(raw['seal.json'])=={n:own[str((directory/n).resolve())] for n in PLAN_FILES-{'seal.json'}},'plan seal differs')
    plan,prepared,encoded=[base.json.loads(raw[n]) for n in ('plan.json','prepared.json','token_ids.json')];audit=rows(raw['pair_audit.jsonl'])
    require(plan['schema']==SCHEMA and plan['status']=='prepared_no_model_inference' and plan['config']==CONFIG
        and plan['api_calls']==0 and plan['gold_read'] is False and plan['weights_sha256']==WEIGHTS_SHA256 and plan['weights_bytes']==WEIGHTS_BYTES,'plan contract differs')
    actual,pairs,counts,bindings,inherited=source_data(plan['candidate_run'],plan['candidate_audit'])
    bindings.update(model_metadata(plan['model_dir'],plan['bge_tokenizer']));bindings.update({str(p.resolve()):digest(p) for p in source_paths()})
    require(actual==prepared and counts==plan['counts'] and bindings==plan['input_sha256'] and inherited==plan['inherited_parent_commitments'],'source projection differs')
    require(len(encoded)==len(pairs)==len(audit),'pair coverage differs')
    for pair,item,row in zip(pairs,encoded,audit,strict=True):
        ids=item.get('input_ids');require(isinstance(ids,list) and 0<len(ids)<=CONFIG['max_pair_tokens'] and all(type(i) is int and i>=0 for i in ids),'invalid complete token IDs')
        require(all(isinstance(v,list) and len(v)==len(ids) for v in item.values()),'token field lengths differ')
        require(row=={k:pair[k] for k in PAIR_FIELDS}|{'pair_tokens':len(ids),'encoding_sha256':hashlib.sha256(canonical(item)).hexdigest()},'pair audit identity/encoding differs')
    require(base.json.loads(raw['length_audit.json'])==old.length_summary(audit),'length summary differs')
    chosen=set(sample_identities(prepared));tokenizer=tokenizer_for(plan['model_dir'])
    for pair,item in zip(pairs,encoded,strict=True):
        if (pair['doc_id'],pair['question_id']) in chosen:
            actual=dict(tokenizer(pair['query'],pair['passage'],add_special_tokens=True,truncation=False,padding=False))
            require(actual==item,'sampled pair text/token IDs differ')
    base.overlap(private_output(plan['run_output']),[directory,plan['candidate_run'],plan['candidate_audit'],plan['model_dir'],plan['bge_tokenizer']])
    verify(bindings);verify(own);require({p.name for p in directory.iterdir()}==PLAN_FILES,'final plan inventory differs')
    return plan,prepared,pairs,encoded,audit,own


def ranked_pack(query,units,values,tokenizer,deadline):
    require(set(values)==set(query['candidate_ids']),'scores must cover all candidates')
    positions={u.unit_id:i for i,u in enumerate(units)};dense_rank={uid:i for i,uid in enumerate(query['ranked_ids'])}
    ranked=sorted(values,key=lambda uid:(-values[uid],dense_rank[uid],units[positions[uid]].order))
    counter=PackCounter(tokenizer,units,deadline);selected=pack_ranked(units,[positions[uid] for uid in ranked],1024,counter,max_units=3)
    rendered=render_pack(units,selected);identity={k:query[k] for k in IDENTITY}
    return identity|{'ranked_ids':ranked},identity|{'method':'reranker_k3','budget':1024,'selected_ids':[units[i].unit_id for i in selected],
        'actual_evidence_tokens':counter(selected),'rendered_pack':rendered,'pack_sha256':hashlib.sha256(rendered.encode()).hexdigest()}


def score_map(pairs,scores):
    require(len(pairs)==len(scores) and all(type(x) in (float,int) and math.isfinite(x) for x in scores),'score coverage/finiteness differs')
    result=defaultdict(dict)
    for pair,score in zip(pairs,scores,strict=True):
        key=(pair['doc_id'],pair['question_id']);require(pair['unit_id'] not in result[key],'duplicate pair score')
        result[key][pair['unit_id']]=score
    return result


def select(prepared,pairs,scores,tokenizer,deadline):
    documents,_,_=validate_prepared(prepared);values=score_map(pairs,scores);rankings=[];packs=[]
    for query in prepared['queries']:
        base.check_time(deadline);key=(query['doc_id'],query['question_id'])
        rank,pack=ranked_pack(query,documents[key[0]],values[key],tokenizer,deadline);rankings.append(rank);packs.append(pack)
    return rankings,packs


def sample_check(prepared,pairs,scores,rankings,packs,tokenizer,deadline):
    documents,_,_=validate_prepared(prepared);values=score_map(pairs,scores);keys=sample_identities(prepared);chosen=set(keys)
    ranks={(r['doc_id'],r['question_id']):r for r in rankings};saved={(r['doc_id'],r['question_id']):r for r in packs}
    for query in prepared['queries']:
        key=(query['doc_id'],query['question_id'])
        if key in chosen:
            base.check_time(deadline);rank,pack=ranked_pack(query,documents[key[0]],values[key],tokenizer,deadline)
            require(rank==ranks[key] and pack==saved[key],'sampled reranker ranking/pack differs')
    return {'sample_documents':len({k[0] for k in keys}),'sample_questions':len(keys),
        'sample_identity_sha256':hashlib.sha256(canonical(keys)).hexdigest(),'same_function_saved_score_pack_replay':True,
        'model_rerun':False,'full_numerical_recomputation':False}


def infer(model_path,encoded,tokenizer,deadline):
    require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
    torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(13);torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(min(1.,7*1024**3/torch.cuda.get_device_properties(0).total_memory),0);torch.cuda.reset_peak_memory_stats(0)
    model=None
    try:
        model=AutoModelForSequenceClassification.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda:0')
        class CheckedModel:
            def eval(self):model.eval()
            def requires_grad_(self,value):model.requires_grad_(value)
            def __call__(self,**kwargs):
                result=model(**kwargs)
                require(torch.cuda.max_memory_allocated(0)<=6*1024**3 and torch.cuda.max_memory_reserved(0)<=7*1024**3,'GPU memory exceeded')
                return result
        scores,compute=old.infer_pairs(encoded,tokenizer,CheckedModel(),device='cuda:0',deadline=deadline)
        return scores,{**compute,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'backbone_dtype':'float16','saved_score_dtype':'float32','peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def aggregate(plan,packs,sample):
    return {'schema':SCHEMA,'status':'completed_reranker_packs','method':'reranker_k3','counts':plan['counts'],
        'sample':sample,'evidence_token_total':sum(p['actual_evidence_tokens'] for p in packs),
        'empty_packs':sum(not p['selected_ids'] for p in packs),'selected_units_histogram':dict(Counter(str(len(p['selected_ids'])) for p in packs)),
        'api_calls':0,'gold_read':False,'quality_metrics_computed':False,'training_performed':False,'paid_execution_admitted':False}


def padding_metadata(encoded):
    lengths=sorted((len(item['input_ids']) for item in encoded),reverse=True)
    batches=[lengths[start]*len(lengths[start:start+CONFIG['microbatch']]) for start in range(0,len(lengths),CONFIG['microbatch'])]
    return {'padded_input_tokens':sum(batches),'max_padded_tokens_per_batch':max(batches),'max_microbatch':CONFIG['microbatch']}


def validate_execution(execution,encoded):
    require(execution['backbone_dtype']=='float16' and execution['saved_score_dtype']=='float32','execution dtype differs')
    require({k:execution[k] for k in padding_metadata(encoded)}==padding_metadata(encoded),'execution padding differs')
    for field,limit in [('peak_allocated_bytes',6*1024**3),('peak_reserved_bytes',7*1024**3)]:
        require(type(execution[field]) is int and 0<=execution[field]<=limit,'execution memory differs')
    require(execution['peak_allocated_bytes']<=execution['peak_reserved_bytes'],'execution memory ordering differs')


def run(directory,confirm_idle=False,inference=infer,guard_factory=base.Watchdog):
    require(confirm_idle,'explicit idle admission required');plan,prepared,pairs,encoded,audit,own=load_plan(directory)
    output=Path(plan['run_output']);require(not output.exists(),'single-use run already exists')
    idle=base.gpu_gate.confirm_idle();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+1800;guard=guard_factory(output,1800)
    write(output/'registration.json',{'schema':SCHEMA,'plan_sha256':own,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    try:
        require(digest(Path(plan['model_dir'])/'model.safetensors')==WEIGHTS_SHA256,'weight hash differs')
        base.check_time(deadline);tokenizer=tokenizer_for(plan['model_dir'])
        scores,execution=inference(plan['model_dir'],encoded,tokenizer,deadline);validate_execution(execution,encoded);base.check_time(deadline)
        bge=tokenizer_for(plan['bge_tokenizer']);rankings,packs=select(prepared,pairs,scores,bge,deadline)
        sample=sample_check(prepared,pairs,scores,rankings,packs,bge,deadline)
        verify(plan['input_sha256']);verify(own)
        raw=[{**row,'score':score} for row,score in zip(audit,scores,strict=True)]
        write_rows(output/'pair_scores.jsonl',raw);write_rows(output/'rankings.jsonl',rankings);write_rows(output/'packs.jsonl',packs)
        write(output/'sample_check.json',sample)
        public={**aggregate(plan,packs,sample),'execution':execution,'elapsed_seconds':time.monotonic()-started,
                'length_audit':old.length_summary(audit)};write(output/'public_aggregate.json',public)
        verify(plan['input_sha256']);verify(own);base.check_time(deadline)
        summary={'schema':SCHEMA,'status':'completed_reranker_packs','plan_sha256':own,'input_sha256':plan['input_sha256'],
            'output_sha256':{n:digest(output/n) for n in RUN_FILES-{'summary.json'}},'weights_sha256_verified_once':WEIGHTS_SHA256,
            'idle_samples':idle,'elapsed_seconds':time.monotonic()-started,'api_calls':0,'gold_read':False}
        write(output/'summary.json',summary);base.check_time(deadline)
        return public
    except BaseException as error:
        if (output/'summary.json').exists():(output/'summary.json').unlink()
        write(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'complete':False,'retry':False});raise
    finally:guard.close()


def validate_saved(prepared,pairs,audit,raw,rankings,packs):
    documents,_,_=validate_prepared(prepared)
    require(len(raw)==len(audit),'raw score coverage differs')
    for record,expected in zip(raw,audit,strict=True):require(set(record)==set(expected)|{'score'} and {k:record[k] for k in expected}==expected,'raw score identity differs')
    scores=[r['score'] for r in raw];score_map(pairs,scores)
    queries={(q['doc_id'],q['question_id']):q for q in prepared['queries']}
    for output in (rankings,packs):require(len(output)==len(queries) and {(r['doc_id'],r['question_id']) for r in output}==set(queries),'complete query coverage differs')
    for rank in rankings:
        q=queries[rank['doc_id'],rank['question_id']]
        require(rank['family_id']==q['family_id'] and len(rank['ranked_ids'])==len(q['candidate_ids'])
            and set(rank['ranked_ids'])==set(q['candidate_ids']),'saved ranking identity differs')
    for pack in packs:
        q=queries[pack['doc_id'],pack['question_id']];units=documents[pack['doc_id']];positions={u.unit_id:i for i,u in enumerate(units)};ids=pack['selected_ids']
        require(len(ids)==len(set(ids))<=3 and set(ids)<=set(q['candidate_ids']),'pack identity differs')
        indices=[positions[uid] for uid in ids];rendered=render_pack(units,indices)
        require(indices==sorted(indices) and len({units[i].native_text for i in indices})==len(indices),'pack order/dedup differs')
        require(pack['family_id']==q['family_id'] and pack['method']=='reranker_k3' and pack['budget']==1024
            and type(pack['actual_evidence_tokens']) is int and 0<=pack['actual_evidence_tokens']<=1024
            and (bool(ids) or pack['actual_evidence_tokens']==0) and pack['rendered_pack']==rendered
            and pack['pack_sha256']==hashlib.sha256(rendered.encode()).hexdigest(),'pack content differs')
    return scores


def audit(directory,output):
    plan,prepared,pairs,encoded,pair_audit,own=load_plan(directory);run=Path(plan['run_output']);output=private_output(output)
    require(not output.exists(),'audit output must be new');base.overlap(output,[directory,run,*plan['input_sha256']])
    require({p.name for p in run.iterdir()}==RUN_FILES,'run inventory differs')
    bindings={str((run/n).resolve()):digest(run/n) for n in RUN_FILES};summary=read(run/'summary.json')
    require(summary['schema']==SCHEMA and summary['status']=='completed_reranker_packs' and summary['api_calls']==0 and summary['gold_read'] is False,'run contract differs')
    require(summary['plan_sha256']==own and summary['input_sha256']==plan['input_sha256'],'run sources differ')
    require(summary['output_sha256']=={n:bindings[str((run/n).resolve())] for n in RUN_FILES-{'summary.json'}},'run output hash differs')
    require(summary['weights_sha256_verified_once']==WEIGHTS_SHA256 and type(summary['elapsed_seconds']) in (int,float)
        and math.isfinite(summary['elapsed_seconds']) and 0<=summary['elapsed_seconds']<=1800,'run time/weight differs')
    raw=rows((run/'pair_scores.jsonl').read_bytes());rankings=rows((run/'rankings.jsonl').read_bytes());packs=rows((run/'packs.jsonl').read_bytes())
    scores=validate_saved(prepared,pairs,pair_audit,raw,rankings,packs)
    sample=sample_check(prepared,pairs,scores,rankings,packs,tokenizer_for(plan['bge_tokenizer']),time.monotonic()+1800)
    require(sample==read(run/'sample_check.json'),'sample report differs')
    public=read(run/'public_aggregate.json');expected=aggregate(plan,packs,sample)
    require({k:public[k] for k in expected}==expected and public['length_audit']==old.length_summary(pair_audit),'aggregate differs')
    require(type(public['elapsed_seconds']) in (int,float) and math.isfinite(public['elapsed_seconds'])
        and 0<=public['elapsed_seconds']<=summary['elapsed_seconds'],'public elapsed time differs')
    validate_execution(public['execution'],encoded)
    registration=read(run/'registration.json');require(registration['schema']==SCHEMA and registration['plan_sha256']==own,'registration differs')
    verify(bindings);verify(own);verify(plan['input_sha256']);require({p.name for p in run.iterdir()}==RUN_FILES,'final run inventory differs')
    result={'schema':SCHEMA,'status':'verified_complete_with_sample','counts':plan['counts'],'sample':sample,
        'source_sha256':plan['input_sha256'],'plan_sha256':own,'run_sha256':bindings,'gold_read':False,'api_calls':0,
        'verification_scope':'complete identity/hash checks; bounded same-function saved-score pack replay; no model rerun or full independent numeric audit'}
    output.mkdir(parents=True,exist_ok=False);write(output/'verification.json',result)
    return {'status':result['status'],'counts':plan['counts'],'sample':sample,'gold_read':False,'api_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--candidate-run',required=True);p.add_argument('--candidate-audit',required=True)
    p.add_argument('--output',required=True);p.add_argument('--run-output',required=True);p.add_argument('--model',default=str(DEFAULT_MODEL));p.add_argument('--bge-tokenizer',default=str(base.DEFAULT_MODEL))
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--confirm-idle',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.candidate_run,args.candidate_audit,args.output,args.run_output,args.model,args.bge_tokenizer)
    elif args.command=='run':result=run(args.plan,args.confirm_idle)
    else:result=audit(args.plan,args.output)
    print(base.json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
