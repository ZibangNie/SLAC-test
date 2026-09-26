"""Complete all gold-free reachable packs; no partial quality or learned-edge claim.

Prepare replays completed request caches and saved mask/pack identities without
opening references. Only a complete, audited new response set may enter scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import run_qasper_recovered_answers as previous

parent, legacy, inherited = previous.parent, previous.legacy, previous.inherited
ROOT, A = parent.ROOT, parent.ARTIFACTS
SCHEMA = 'slac-qasper-relation-pack-answers-v1'
MODEL = 'qwen/qwen3.6-plus'
PRIOR = {'attempts':803,'reservation_usd':'4.6053728675','known_cost_usd':'0.487667787','unknown_cost_attempts':1}
HEADROOM = Decimal('0.3946271325')
RESERVATION = Decimal('0.0533144625')
IDENTITY = ('family_id','doc_id','question_id')
BASELINES = ('I_jev_k3','dense_k3','reranker_k3','p_yes_only_k3')
METHODS = BASELINES + ('full_adjacency',)
PAIRS = tuple(('full_adjacency',m) for m in BASELINES)
METRICS = ('official_answer_f1','actual_evidence_tokens')
SPEC = {'questions':77,'families':24,'mask_mappings':501,'distinct_query_packs':96,
    'distinct_pack_histogram':{'1':60,'2':15,'3':2},'unique_pack_payloads':96,
    'cached_pack_payloads':82,'new_unique_requests':14,'new_input_allowance':66355,'new_output_allowance':7168,
    'methods':list(METHODS),'primary_pair':list(PAIRS[0]),'secondary_pairs':[list(p) for p in PAIRS[1:]],
    'metrics':list(METRICS),'paired_intervals':16,'generator':MODEL,'required_actual_model':MODEL,
    'provider':'Alibaba','prompt_version':legacy.PROMPT_VERSION,'temperature':0,'reasoning':False,'max_output_tokens':512,
    'budget':1024,'max_units':3,'prices':legacy.PRICES,'prior_night_accounting':PRIOR,
    'new_reservation_usd':str(RESERVATION),'night_cap_usd':'5','remaining_headroom_before_usd':str(HEADROOM),
    'stop_at_utc':legacy.STOP_AT,'request_deadline_seconds':65,'dispatch_margin_seconds':65,'max_execution_seconds':1800,
    'automatic_retries':0,'bootstrap_seed':20260927,'bootstrap_replicates':10000,
    'bootstrap':'shared whole-family PCG64 multinomial; linear percentile 95%; QW and FB',
    'posthoc_exposed_development':True,'independent_confirmation':False,'multiple_comparison_control':False,
    'min_max_envelope':'single saved response per distinct payload; descriptive only; no intervals or sign tests',
    'mask_weighting':'masks map to deduplicated payloads, never treated as independent questions',
    'no_oracle_regeneration':True,'static_562_stage_closed':True,'learned_relations_evaluated':False,
    'source_containers':'old JSON containers are parsed then projected; old quality fields are not used during preparation'}
PLAN_FILES = ('experiment_config.json','plan_manifest.json','jobs.json','packs.jsonl','mask_mapping.jsonl','baselines.jsonl','inheritance.json')
RUN_FILES = {'answers.json','pack_records.jsonl','per_question.jsonl','envelope_per_question.jsonl','summary.json','provider_calls','attempt_ledger'}
ANCESTORS = (
 ('local','qasper-local-answer','02',367,462,'2db10e49db82f37b3b4999a594a38f7cc57b4856e1248e1ccc716d73cae6cc9a','c8447df4a13bdf772ff8ecadf1987983b06dc3720ff3ff4e31cd911e4e0dcffa'),
 ('owner','qasper-owner-order-answer','01',7,385,'0bdcd1674c305238b761361cf12d010bfc4222197a68615f74b2a876f84edab5','1a6d7df5f370c47b6463b22970c7db0cef8374bc6759f56cfe575a684a4e25cd'),
 ('primary','qasper-recovered-primary-answer','01',120,462,'a2b5fdce093c0d6f0d95c075b5923fa61294b7b742c6efaa9a80794ae6a2b790','ed04454828b368175fb0520b272dc71dd1939053076578c82e75539f08553c0d'))

def check(condition, reason):
    if not condition: raise ValueError(reason)

def identity(row): return tuple(row[k] for k in IDENTITY)
def canonical(value): return legacy.client.canonical_bytes(value)
def object_hash(value): return legacy.client.object_hash(value)
def registration_path(): return A/'qasper-relation-pack-answers-registration-01.json'
def run_path(): return A/'qasper-relation-pack-answer-run-01'
def utc_now(): return datetime.now(timezone.utc)

class Reader:
    def __init__(self): self.bindings={}
    def raw(self,path,expected=None):
        path=Path(path).resolve(); raw=path.read_bytes(); value=hashlib.sha256(raw).hexdigest()
        check(expected is None or value==expected,'bound source changed: '+path.name)
        check(str(path) not in self.bindings or self.bindings[str(path)]==value,'source changed during read')
        self.bindings[str(path)]=value
        return raw
    def json(self,path,expected=None): return json.loads(self.raw(path,expected),object_pairs_hook=legacy.client.unique_object)
    def rows(self,path,expected=None): return [json.loads(s,object_pairs_hook=legacy.client.unique_object) for s in self.raw(path,expected).splitlines() if s.strip()]
    def merge(self,bindings): self.bindings=parent.merge(self.bindings,bindings)
    def verify(self): legacy.pilot.verify_hashes(self.bindings)

def payload_key(payload):
    return object_hash({'endpoint':legacy.ENDPOINT,'prompt_version':legacy.PROMPT_VERSION,'payload':payload})

def job_for(question,pack):
    payload=legacy.make_payload(question,pack); amount,allowance=legacy.reserve(payload)
    return {'cache_key':payload_key(payload),'payload':payload,'reserved_usd':str(amount),'input_allowance':allowance,'output_allowance':512}

def load_ancestors(reader):
    cache={}; mappings={}; configs={}; deferred_records={}; summaries={}
    for origin,stem,version,n,records,summary_sha,audit_sha in ANCESTORS:
        plan=A/(stem+'-plan-'+version); run=A/(stem+'-run-01')
        summary=reader.json(run/'summary.json',summary_sha)
        audit=reader.json(A/(stem+'-audit-01')/'audit.json',audit_sha)
        check(audit['status']=='verified_complete' and audit['all_bound_inputs_outputs_unchanged'] is True,
              'completed audited ancestor required')
        check(summary['status']=='completed' and summary['main_results_available'] is True
            and (summary['question_count'],summary['family_count'],summary['record_count'])==(77,24,records),'ancestor coverage differs')
        cfg=reader.json(plan/'experiment_config.json',summary['plan_sha256'])
        manifest=reader.json(plan/'plan_manifest.json')
        check(manifest['experiment_config_sha256']==summary['plan_sha256'],'ancestor manifest changed')
        jobs=reader.json(plan/'jobs.json',cfg['plan_files_sha256']['jobs.json'])
        mapping=reader.rows(plan/'mapping.jsonl',cfg['plan_files_sha256']['mapping.jsonl'])
        check(len(jobs)==n and len(mapping)==records,'ancestor request/mapping incomplete')
        # Freeze executable dependencies, without following any QA or corpus path.
        for path,expected in cfg['input_sha256'].items():
            if Path(path).suffix=='.py': reader.raw(path,expected)
        inspector=parent.inspect_calls if origin=='local' else inherited.inspect_calls if origin=='owner' else previous.inspect_calls
        answers,ledger,calls,complete=inspector(cfg,jobs,run,require_complete=True)
        check(complete and ledger['resolved_models']=={'generator':MODEL},'actual cached model differs')
        reader.merge(calls)
        for path,value in calls.items():
            check(summary['output_sha256'][str(Path(path).relative_to(run))]==value,'audited cache seal changed')
        for ordinal,job in enumerate(jobs,1):
            payload=job['payload']; content=json.loads(payload['messages'][1]['content'],object_pairs_hook=legacy.client.unique_object)
            check(set(content)=={'question','evidence'} and job==job_for(content['question'],content['evidence']),
                  'cache requires complete frozen payload and budget')
            key=job['cache_key']; check(key not in cache,'ancestor physical caches overlap')
            attempt=ledger['attempts'][ordinal-1]
            check(attempt['response_model']==MODEL,'cached actual model differs')
            cache[key]={'job':job,'answer':answers[key],'origin':origin,'request_ordinal':ordinal,
                'request_sha256':attempt['request_sha256'],'response_sha256':attempt['response_sha256'],'resolved_model':MODEL}
        check(Decimal(ledger['reservation_total_usd'])==Decimal(audit['accounting']['generation_attempted_reservation_usd'])
            and Decimal(ledger['actual_reported_cost_usd'])==Decimal(audit['accounting']['known_generation_cost_usd']),
            'complete ancestor accounting differs')
        mappings[origin]=mapping;configs[origin]=cfg;summaries[origin]=summary
        deferred_records[origin]={'path':str(run/'per_question.jsonl'),'sha256':summary['output_sha256']['per_question.jsonl']}
    check(len(cache)==494,'all three completed caches required')
    for mapping in mappings.values(): check(all(r['cache_key'] in cache for r in mapping),'unresolved ancestor cache inheritance')
    primary=summaries['primary']
    prior={'attempts':primary['night_attempts'],'reservation_usd':primary['night_attempted_reservation_usd'],
        'known_cost_usd':primary['night_known_reported_cost_subtotal_usd'],'unknown_cost_attempts':primary['night_unknown_cost_attempts']}
    check(prior==PRIOR,'all historical costs and unknown must be retained')
    support_plan=Path(configs['primary']['source_paths']['recovery_plan'])/'experiment_config.json'
    support=reader.json(support_plan,configs['primary']['input_sha256'][str(support_plan)])
    sidecar=support['sidecar']; expected=support['input_sha256'][sidecar]
    check(configs['primary']['input_sha256'][sidecar]==expected,'reference lineage differs')
    return cache,mappings,prior,{'path':sidecar,'sha256':expected},deferred_records

def source_data():
    reader=Reader();cache,mappings,prior,references,old_records=load_ancestors(reader)
    plan_dir=A/'qasper-relation-opportunity-plan-01';run=A/'qasper-relation-opportunity-run-01'
    seal=reader.json(plan_dir/'seal.json');plan=reader.json(plan_dir/'plan.json',seal['plan.json'])
    cases=reader.json(plan_dir/'cases.json',seal['cases.json'])
    summary=reader.json(run/'summary.json','bdaa241d38fef03f176e85176da2a3f796b58f1ca0b6a4accb81d3807b576c88')
    audit=reader.json(A/'qasper-relation-opportunity-root-execution-01/audit.json','aed6acc868fbfdac795c1d363029e7cd138d416a47874a77286ed76bd8f94d1b')
    review=reader.json(A/'qasper-relation-opportunity-independent-verification-01/verification.json','ebf959058e67a2ea3285f70c985f0a94bda2d6912cd6eab24d4de1a33037e32c')
    check(summary['status']=='completed' and summary['all_masks_available'] is True and summary['quality_metrics_computed'] is False
          and audit['status']==review['status']=='verified_complete','complete independently verified gold-free cube required')
    for p in (plan_dir/'plan.json',plan_dir/'cases.json',plan_dir/'seal.json'):
        check(summary['plan_sha256'][str(p)]==reader.bindings[str(p)],'gate plan binding differs')
    masks=reader.rows(run/'per_mask.jsonl',summary['output_sha256']['per_mask.jsonl'])
    queries=reader.rows(run/'per_question.jsonl',summary['output_sha256']['per_question.jsonl'])
    for path in (Path(__file__),ROOT/'tests/research/test_qasper_relation_pack_answers.py',ROOT/'docs/research/RELATION_PACK_ANSWER_PROTOCOL_20260927.md'):
        reader.raw(path)
    reader.verify()
    return {'cache':cache,'ancestor_mappings':mappings,'prior':prior,'references':references,'old_records':old_records,
        'cases':cases,'masks':masks,'queries':queries,'input_sha256':reader.bindings}

def build_jobs(data):
    cases={identity(c):c for c in data['cases']}; qmap={identity(q):q for q in data['queries']}
    check(len(cases)==len(data['cases'])==len(qmap) and set(cases)==set(qmap),'duplicate/missing cube question')
    cache=data['cache'];questions={};baselines={}
    for origin,mapping in data['ancestor_mappings'].items():
        for row in mapping:
            key=identity(row);check(key in cases,'ancestor question outside cube')
            value=json.loads(cache[row['cache_key']]['job']['payload']['messages'][1]['content'])['question']
            check(key not in questions or questions[key]==value,'different inherited query string')
            questions[key]=value
            if (origin=='primary' and row['method'] in ('I_jev_k3','dense_k3','p_yes_only_k3')) or (origin=='local' and row['method']=='reranker_k3'):
                token=(row['method'],*key);check(token not in baselines,'duplicate baseline')
                baselines[token]={k:row[k] for k in (*IDENTITY,'method','cache_key','selected_ids','pack_sha256','actual_evidence_tokens')}
    check(set(questions)==set(cases) and set(baselines)=={(m,*key) for key in cases for m in BASELINES},'four complete baselines required')
    grouped=defaultdict(list)
    for row in data['masks']:grouped[identity(row)].append(row)
    check(set(grouped)==set(cases),'mask question coverage differs')
    jobs={};packs=[];mask_mapping=[];queries=[];documents={}
    for key in sorted(cases):
        case=cases[key];units=[parent.Unit(**u) for u in case['units']];group=grouped[key];q=qmap[key]
        check(len(group)==q['mask_count']==2**len(q['eligible_edges'])
            and sorted(r['mask'] for r in group)==list(range(len(group))),'mask cube incomplete')
        check(key[1] not in documents or documents[key[1]]==case['units'],'document text differs between questions')
        documents[key[1]]=case['units'];queries.append(dict(zip(IDENTITY,key))|{'query':questions[key]})
        seen={}
        for r in group:
            selected=r['selected_indices'];check(selected==sorted(set(selected)) and len(selected)<=3
                and set(selected)<=set(case['candidates']) and type(r['actual_tokens']) is int and 0<=r['actual_tokens']<=1024,'invalid whole-pack selection')
            check(len({units[j].native_text for j in selected})==len(selected),'duplicate same-source native text')
            pack=parent.render_pack(units,selected);pack_hash=hashlib.sha256(pack.encode()).hexdigest()
            ids=[units[j].unit_id for j in selected]
            check(pack_hash==r['pack_sha256'] and ids==r['selected_ids'],'saved mask pack differs')
            job=job_for(questions[key],pack);ck=job['cache_key']
            check(ck not in jobs or jobs[ck]==job,'payload hash collision');jobs[ck]=job
            if ck in cache:check(cache[ck]['job']==job and cache[ck]['resolved_model']==MODEL,'cached payload/model mismatch')
            m=dict(zip(IDENTITY,key))|{'mask':r['mask'],'cache_key':ck}
            mask_mapping.append(m)
            if ck not in seen:seen[ck]=dict(zip(IDENTITY,key))|{'cache_key':ck,'selected_ids':ids,'pack_sha256':pack_hash,
                'actual_evidence_tokens':r['actual_tokens'],'masks':[],'response_origin':cache[ck]['origin'] if ck in cache else 'new'}
            check(seen[ck]['actual_evidence_tokens']==r['actual_tokens'] and seen[ck]['selected_ids']==ids,'equal payload metadata differs')
            seen[ck]['masks'].append(r['mask'])
            if r['mask']==0:
                baseline=baselines['I_jev_k3',*key]
                check(ck==baseline['cache_key'] and ids==baseline['selected_ids'] and pack_hash==baseline['pack_sha256']
                    and r['actual_tokens']==baseline['actual_evidence_tokens'],'exact I baseline pack bridge failed')
            if r['mask']==len(group)-1:seen[ck]['full_adjacency']=True
        check(len(seen)==q['distinct_pack_count'],'distinct pack count differs')
        packs.extend({**row,'masks':sorted(row['masks']),'full_adjacency':row.get('full_adjacency',False)} for ck,row in sorted(seen.items()))
    required={r['cache_key'] for r in packs}|{r['cache_key'] for r in baselines.values()}
    inheritance={k:{f:cache[k][f] for f in ('origin','request_ordinal','request_sha256','response_sha256','resolved_model')}
                 for k in sorted(required&cache.keys())}
    new=[j for k,j in sorted(jobs.items()) if k not in cache]
    data['prepared']={'queries':queries,'documents':documents}
    return new,packs,sorted(mask_mapping,key=lambda r:(identity(r),r['mask'])),[baselines[k] for k in sorted(baselines)],inheritance

def scope(data,jobs,packs,masks,baselines,inheritance):
    questions=parent.bootstrap.questions_from(data['prepared']);keys={r['cache_key'] for r in packs}
    counts={'questions':len(questions),'families':len({q[0] for q in questions}),'mask_mappings':len(masks),'distinct_query_packs':len(packs),
        'unique_pack_payloads':len(keys),'cached_pack_payloads':len(keys&inheritance.keys()),'new_unique_requests':len(jobs),
        'new_input_allowance':sum(j['input_allowance'] for j in jobs),'new_output_allowance':sum(j['output_allowance'] for j in jobs)}
    histogram={str(k):v for k,v in sorted(Counter(Counter(identity(r) for r in packs).values()).items())}
    check(data['prior']==PRIOR,'historical cost/unknown changed')
    amount=sum((legacy.pilot.nonnegative_decimal(j['reserved_usd']) for j in jobs),Decimal(0))
    check(amount<=HEADROOM and Decimal(PRIOR['reservation_usd'])+amount<=Decimal(5),'whole stage over remaining cap')
    check(all(counts[k]==SPEC[k] for k in counts) and histogram==SPEC['distinct_pack_histogram']
        and len(baselines)==4*len(questions),'complete fixed 77/96/501/14 scope required')
    check(amount==Decimal(SPEC['new_reservation_usd']),'exact whole-stage reservation differs')
    counts.update(baseline_records=len(baselines),inherited_required_payloads=len(inheritance),
        all_required_payloads=len(inheritance)+len(jobs),distinct_pack_histogram=histogram)
    return counts,amount

def disjoint(output,parents):
    output=Path(output).resolve()
    if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in parents):
        raise ValueError('new output overlaps immutable ancestor')

def prepare(args):
    output=Path(args.output).resolve();run_output=run_path().resolve()
    if output.exists() or run_output.exists() or registration_path().exists(): raise FileExistsError('single-use plan/run already exists')
    data=source_data(); jobs,packs,masks,baselines,inheritance=build_jobs(data)
    counts,amount=scope(data,jobs,packs,masks,baselines,inheritance)
    parents={Path(p).parent for p in data['input_sha256']};disjoint(output,parents|{run_output});disjoint(run_output,parents)
    legacy.pilot.verify_hashes(data['input_sha256'])
    output.mkdir(parents=True,exist_ok=False)
    parent.write(output/'jobs.json',jobs);parent.write(output/'inheritance.json',inheritance)
    for name,rows in [('packs.jsonl',packs),('mask_mapping.jsonl',masks),('baselines.jsonl',baselines)]:legacy.pilot.write_rows(output/name,rows)
    config={'schema':SCHEMA,'status':'planned_not_executed','specification':SPEC,'input_sha256':data['input_sha256'],
        'deferred_reference_binding':data['references'],'deferred_baseline_record_bindings':data['old_records'],
        'prior_night_accounting':data['prior'],'parent_resolved_model':MODEL,'run_output':str(run_output),
        'single_use_registration':str(registration_path()),'answer_reservation_usd':str(amount),
        'night_reservation_usd':str(Decimal(PRIOR['reservation_usd'])+amount),**counts,
        'plan_files_sha256':{n:legacy.digest(output/n) for n in PLAN_FILES[2:]},'api_calls':0,'key_read':False,
        'gold_loaded':False,'new_quality_computed':False,'paid_execution_released':False}
    parent.write(output/'experiment_config.json',config)
    parent.write(output/'plan_manifest.json',{'schema':SCHEMA,'experiment_config_sha256':legacy.digest(output/'experiment_config.json')})
    return {k:config[k] for k in ('status','new_unique_requests','unique_pack_payloads','cached_pack_payloads','answer_reservation_usd','night_reservation_usd','gold_loaded','api_calls')}

def load_plan(directory):
    directory=Path(directory).resolve();own=previous.recovery.snapshot_files(directory,PLAN_FILES)
    reader=Reader();config=reader.json(directory/'experiment_config.json',own[str(directory/'experiment_config.json')])
    check(config.get('schema')==SCHEMA and config.get('status')=='planned_not_executed' and config.get('specification')==SPEC
        and reader.json(directory/'plan_manifest.json',own[str(directory/'plan_manifest.json')])=={'schema':SCHEMA,'experiment_config_sha256':own[str(directory/'experiment_config.json')]}
        and config.get('run_output')==str(run_path().resolve()) and config.get('single_use_registration')==str(registration_path())
        and config.get('parent_resolved_model')==MODEL
        and all(config.get(k) is v for k,v in {'api_calls':0,'key_read':False,'gold_loaded':False,'new_quality_computed':False,'paid_execution_released':False}.items()),
        'frozen complete-answer plan differs')
    legacy.pilot.verify_hashes(config['input_sha256'])
    data=source_data();jobs,packs,masks,baselines,inheritance=build_jobs(data);counts,amount=scope(data,jobs,packs,masks,baselines,inheritance)
    check(data['input_sha256']==config['input_sha256'] and data['references']==config['deferred_reference_binding']
        and data['old_records']==config['deferred_baseline_record_bindings'] and data['prior']==config['prior_night_accounting']
        and all(config[k]==v for k,v in counts.items()) and config['answer_reservation_usd']==str(amount)
        and config['night_reservation_usd']==str(Decimal(PRIOR['reservation_usd'])+amount),'source/coverage/budget replay differs')
    for name,expected in [('jobs.json',jobs),('inheritance.json',inheritance),('packs.jsonl',packs),('mask_mapping.jsonl',masks),('baselines.jsonl',baselines)]:
        value=reader.rows(directory/name,own[str(directory/name)]) if name.endswith('.jsonl') else reader.json(directory/name,own[str(directory/name)])
        check(value==expected,'saved full payload/pack mapping changed')
    check(config['plan_files_sha256']=={n:own[str(directory/n)] for n in PLAN_FILES[2:]},'plan file seals differ')
    legacy.pilot.verify_hashes(parent.merge(config['input_sha256'],own))
    return config,data,jobs,packs,masks,baselines,inheritance,own

def ensure_time(started,*,dispatch=False):
    margin=SPEC['dispatch_margin_seconds'] if dispatch else 0
    if (datetime.fromisoformat(SPEC['stop_at_utc'])-utc_now()).total_seconds()<=margin or time.monotonic()-started>=SPEC['max_execution_seconds']-margin:
        raise TimeoutError('whole-stage or absolute 09:00 cutoff reached')

class Deadline:
    def __init__(self,output,started,request=False):
        self.cancel=threading.Event()
        seconds=min((datetime.fromisoformat(SPEC['stop_at_utc'])-utc_now()).total_seconds(),SPEC['max_execution_seconds']-(time.monotonic()-started))
        if request:seconds=min(seconds,SPEC['request_deadline_seconds'])
        def stop():
            if not self.cancel.wait(max(0,seconds)):
                try:parent.write(Path(output)/'hard_deadline.json',{'schema':SCHEMA,'status':'failed_hard_deadline','main_results_available':False,'request_deadline':request,'automatic_retries':0})
                finally:os._exit(124)
        self.thread=threading.Thread(target=stop,daemon=True);self.thread.start()
    def close(self):self.cancel.set();self.thread.join(timeout=1)

class RelationPackClient(inherited.InheritedAnswerClient):
    def __init__(self,*args,started,**kwargs):
        check(kwargs.get('parent_model')==MODEL,'exact prior actual model required')
        self.started=started;super().__init__(*args,**kwargs)
    def submit(self,job):ensure_time(self.started,dispatch=True);return super().submit(job)

def inspect_calls(config,jobs,output,*,require_complete):
    # Same immutable event, raw usage/finish/model and elapsed-time checks as the completed ancestor.
    return previous.inspect_calls(config,jobs,output,require_complete=require_complete)

def registration_binding(config,own):
    reader=Reader();path=Path(config['single_use_registration']);value=reader.json(path)
    check(set(value)=={'schema','run_output','plan_sha256','registered_at_utc'} and value['schema']==SCHEMA
        and value['run_output']==config['run_output'] and value['plan_sha256']==next(v for p,v in own.items() if Path(p).name=='experiment_config.json')
        and datetime.fromisoformat(value['registered_at_utc']).utcoffset() is not None,'single-use registration differs')
    return reader.bindings

def accounting(config,ledger,complete):
    check(config['prior_night_accounting']==PRIOR,'prior total or unknown changed')
    attempted=len(ledger['attempts']);reserve=Decimal(ledger['reservation_total_usd']);known=Decimal(ledger['actual_reported_cost_usd'])
    unknown=sum('actual_cost_usd' not in r for r in ledger['attempts'])
    return {'schema':SCHEMA,'status':'completed' if complete else 'verified_incomplete','main_results_available':complete,
        'new_api_calls':attempted,'completed_requests':sum(r['status']=='completed' for r in ledger['attempts']),
        'known_generation_cost_usd':str(known),'unknown_generation_cost_attempts':unknown,
        'generation_attempted_reservation_usd':str(reserve),'generation_planned_reservation_usd':config['answer_reservation_usd'],
        'prior_night_accounting':PRIOR,'night_attempts':PRIOR['attempts']+attempted,
        'night_attempted_reservation_usd':str(Decimal(PRIOR['reservation_usd'])+reserve),
        'night_known_reported_cost_subtotal_usd':str(Decimal(PRIOR['known_cost_usd'])+known),
        'night_unknown_cost_attempts':1+unknown,'old_unknown_attempt_refunded':False,'inherited_requests_charged_again':0,
        'parent_resolved_model':MODEL,'automatic_retries':0,'specification':SPEC}

def complete_prediction_gate(data,packs,baselines,answers):
    questions=parent.bootstrap.questions_from(data['prepared'])
    check(len(questions)==SPEC['questions'] and len({q[0] for q in questions})==SPEC['families']
        and len(packs)==SPEC['distinct_query_packs'] and len({(identity(r),r['cache_key']) for r in packs})==len(packs)
        and {identity(r) for r in packs}==set(questions)
        and len(baselines)==4*len(questions) and {(r['method'],*identity(r)) for r in baselines}=={(m,*q) for m in BASELINES for q in questions}
        and set(answers)=={r['cache_key'] for r in packs+baselines}
        and all(isinstance(v,str) and v.strip() for v in answers.values()),'all distinct packs and four complete baselines required before gold')
    check(Counter(identity(r) for r in packs if r['full_adjacency'])==Counter(questions),'exactly one full-adjacency pack per question required')
    return questions

def score_all(data,packs,baselines,answers,annotations):
    questions=complete_prediction_gate(data,packs,baselines,answers)
    check(set(annotations)=={(q[1],q[2]) for q in questions},'references must cover exactly the admitted 77 questions')
    def scored(row):
        refs=legacy.references_from_annotations(annotations[row['doc_id'],row['question_id']])
        answer=answers[row['cache_key']]
        return {**row,'predicted_answer':answer,'official_answer_f1':max(legacy.token_f1_score(answer,r['answer']) for r in refs)}
    pack_records=[scored(r) for r in packs];records=[scored(r) for r in baselines]
    records.extend({k:v for k,v in r.items() if k not in ('masks','full_adjacency','response_origin')}|{'method':'full_adjacency'}
                   for r in pack_records if r['full_adjacency'])
    groups,draws=parent.bootstrap.family_resamples(questions)
    domain,_=parent.bootstrap.summarize_domain('answer',records,METHODS,METRICS,PAIRS,questions,groups,draws)
    baseline={identity(r):r for r in records if r['method']=='I_jev_k3'};grouped=defaultdict(list)
    for r in pack_records:grouped[identity(r)].append(r)
    envelope=[]
    for q in questions:
        values=[r['official_answer_f1'] for r in grouped[q]];low,high=min(values),max(values);original=baseline[q]['official_answer_f1']
        check(low<=original+1e-12 and high>=original-1e-12,'I response must belong to its observed envelope')
        envelope.append(dict(zip(IDENTITY,q))|{'distinct_packs':len(values),'I_answer_f1':original,
            'minimum_observed_answer_f1':low,'maximum_observed_answer_f1':high,'minimum_minus_I':low-original,'maximum_minus_I':high-original})
    def descriptive(field):
        values=[r[field] for r in envelope]
        return {'question_weighted':sum(values)/len(values),
            'family_balanced':sum(sum(values[int(i)] for i in group)/len(group) for group in groups)/len(groups),
            'question_positive':sum(v>1e-12 for v in values),'question_ties':sum(abs(v)<=1e-12 for v in values),'question_negative':sum(v< -1e-12 for v in values)}
    summary={'metrics':domain['method_means'],'paired_comparisons':domain['paired_comparisons'],
        'shared_resamples_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
        'observed_envelope':{name:descriptive(name) for name in ('minimum_observed_answer_f1','maximum_observed_answer_f1','minimum_minus_I','maximum_minus_I')},
        'distinct_pack_distribution':{'scope':'pooled descriptive 96-pack inventory; not question-weighted performance',
            'answer_f1':[{ 'value':v,'count':n} for v,n in sorted(Counter(r['official_answer_f1'] for r in pack_records).items())],
            'actual_evidence_tokens':[{'value':v,'count':n} for v,n in sorted(Counter(r['actual_evidence_tokens'] for r in pack_records).items())],
            'selected_units':[{'value':v,'count':n} for v,n in sorted(Counter(len(r['selected_ids']) for r in pack_records).items())]},
        'envelope_inference':'descriptive single-realization permissive envelope; no intervals/sign tests; no joint optimal-token claim',
        'independent_confirmation':False,'learned_relation_effect_claimed':False,'quality_bound_on_expected_generation':False}
    return pack_records,records,envelope,summary

def baseline_parity(data,records,reader):
    old={}
    for origin in ('primary','local'):
        source=data['old_records'][origin]
        for row in reader.rows(source['path'],source['sha256']):
            if (origin=='primary' and row['method'] in ('I_jev_k3','dense_k3','p_yes_only_k3')) or (origin=='local' and row['method']=='reranker_k3'):
                key=(row['method'],*identity(row));check(key not in old,'duplicate saved baseline');old[key]=row
    selected=[r for r in records if r['method'] in BASELINES]
    check(len(old)==len(selected)==4*SPEC['questions'],'all original baseline records required')
    fields=(*IDENTITY,'method','cache_key','selected_ids','pack_sha256','actual_evidence_tokens','predicted_answer','official_answer_f1')
    for row in selected:check(all(row[k]==old[row['method'],*identity(row)][k] for k in fields),'saved baseline response/official F1 parity failed')

def load_references(data,reader):
    """Same admitted-Qasper contract, parsed from the one hash-checked buffer."""
    source=data['references'];raw=reader.raw(source['path'],source['sha256'])
    wanted={(q['doc_id'],q['question_id']):q for q in data['prepared']['queries']};found={}
    for line in raw.splitlines():
        if not line.strip():continue
        row=json.loads(line,object_pairs_hook=legacy.client.unique_object);key=(row['doc_id'],row['question_id'])
        if key not in wanted:continue
        query=wanted[key]
        check(key not in found and row['official_split']=='validation' and row['family_id']==query['family_id']
            and row['question']==query['query'],'admitted reference identity differs')
        found[key]=row['answer_annotations']
    check(set(found)==set(wanted),'admitted reference coverage incomplete')
    return found

def results(config,data,jobs,packs,baselines,inheritance,own,output,*,require_complete):
    new,ledger,calls,complete=inspect_calls(config,jobs,output,require_complete=require_complete)
    if not complete:return None,None,None,None,None,ledger,calls
    calls=parent.merge(calls,registration_binding(config,own))
    check(not any((Path(output)/f).exists() for f in ('failure.json','hard_deadline.json')),'failed execution cannot score')
    legacy.pilot.verify_hashes(parent.merge(config['input_sha256'],own,calls))
    check(set(new)=={j['cache_key'] for j in jobs} and set(inheritance)==({r['cache_key'] for r in packs+baselines}-set(new)),
          'complete new and inherited payload partition required')
    answers={key:data['cache'][key]['answer'] for key in inheritance}|new
    complete_prediction_gate(data,packs,baselines,answers)
    # This is the first reference read. Its ancestor commitment was carried without opening it during prepare/run admission.
    reader=Reader();annotations=load_references(data,reader)
    reader.verify()
    pack_records,records,envelope,summary=score_all(data,packs,baselines,answers,annotations)
    baseline_parity(data,records,reader);reader.verify();calls=parent.merge(calls,reader.bindings)
    summary.update(accounting(config,ledger,True),question_count=SPEC['questions'],family_count=SPEC['families'],
        pack_record_count=len(pack_records),record_count=len(records),envelope_record_count=len(envelope),
        unique_pack_payloads=config['unique_pack_payloads'],cached_pack_payloads=config['cached_pack_payloads'],
        new_unique_requests=config['new_unique_requests'],baseline_records_verified=4*SPEC['questions'],
        plan_sha256=next(v for p,v in own.items() if Path(p).name=='experiment_config.json'),
        input_binding_sha256=object_hash(parent.merge(config['input_sha256'],own,calls)))
    return pack_records,records,envelope,summary,answers,ledger,calls

def run(args,*,client_factory=RelationPackClient,guard_factory=Deadline):
    started=time.monotonic();config,data,jobs,packs,masks,baselines,inheritance,own=load_plan(args.plan)
    output=Path(config['run_output'])
    if output.exists():raise FileExistsError('fixed run already exists')
    ensure_time(started,dispatch=True)
    parent.write(config['single_use_registration'],{'schema':SCHEMA,'run_output':str(output),
        'plan_sha256':own[str(Path(args.plan).resolve()/'experiment_config.json')],'registered_at_utc':utc_now().isoformat()})
    output.mkdir(parents=True,exist_ok=False);guard=guard_factory(output,started);bounded=None
    try:
        bounded=client_factory(output/'provider_calls',prior_reservation=PRIOR['reservation_usd'],jobs=jobs,parent_model=MODEL,
            started=started,key_file=getattr(args,'key_file',None),proxy=getattr(args,'proxy',None))
        for index,job in enumerate(jobs,1):
            legacy.pilot.verify_hashes(parent.merge(config['input_sha256'],own));ensure_time(started,dispatch=True)
            request_guard=guard_factory(output,started,request=True)
            try:bounded.submit(job)
            finally:request_guard.close()
            print(json.dumps({'completed_new_requests':index,'new_unique_requests':len(jobs)}),flush=True)
        pack_records,records,envelope,summary,answers,ledger,calls=results(config,data,jobs,packs,baselines,inheritance,own,output,require_complete=True)
        legacy.pilot.verify_hashes(parent.merge(config['input_sha256'],own,calls));ensure_time(started)
        parent.write(output/'answers.json',answers)
        for name,rows in [('pack_records.jsonl',pack_records),('per_question.jsonl',records),('envelope_per_question.jsonl',envelope)]:legacy.pilot.write_rows(output/name,rows)
        summary['output_sha256']={str(Path(p).relative_to(output)):value for p,value in inherited.tree(output).items()}
        ensure_time(started);parent.write(output/'summary.json',summary)
        return {'status':'completed','new_api_calls':len(jobs),'distinct_pack_predictions':len(pack_records),'method_records':len(records)}
    except BaseException as error:
        failure={'schema':SCHEMA,'status':'failed','error_class':type(error).__name__,'main_results_available':False,'automatic_retries':0}
        if bounded is not None:failure['accounting']=accounting(config,bounded.ledger,False)
        parent.write(output/'failure.json',failure);raise
    finally:guard.close()

def audit(args):
    config,data,jobs,packs,masks,baselines,inheritance,own=load_plan(args.plan);output=Path(args.run).resolve()
    check(str(output)==config['run_output'],'audit output is not the single fixed run')
    registration=registration_binding(config,own);before=inherited.tree(output)
    _,ledger,calls,complete=inspect_calls(config,jobs,output,require_complete=not args.allow_incomplete)
    success=complete and (output/'summary.json').exists() and not any((output/f).exists() for f in ('failure.json','hard_deadline.json'))
    if success:
        check({p.name for p in output.iterdir()}==RUN_FILES,'complete output inventory differs')
        pack_records,records,envelope,expected,answers,ledger,calls=results(config,data,jobs,packs,baselines,inheritance,own,output,require_complete=True)
        expected['output_sha256']={str(Path(p).relative_to(output)):value for p,value in before.items() if Path(p)!=output/'summary.json'}
        check(expected==parent.read(output/'summary.json') and answers==parent.read(output/'answers.json')
            and pack_records==parent.rows(output/'pack_records.jsonl') and records==parent.rows(output/'per_question.jsonl')
            and envelope==parent.rows(output/'envelope_per_question.jsonl'),'scientific outputs or metadata do not replay')
    else:check(args.allow_incomplete,'incomplete run has no complete answer result')
    legacy.pilot.verify_hashes(parent.merge(config['input_sha256'],own,calls,before,registration))
    check(inherited.tree(output)==before,'output changed during audit')
    return {'schema':SCHEMA+'-audit','status':'verified_complete' if success else 'verified_incomplete',
        'accounting':accounting(config,ledger,success),'all_bound_inputs_outputs_unchanged':True,
        'all_96_packs_scored':success,'original_308_baseline_records_replayed':success,'api_calls_by_audit':0,'key_read':False,
        'partial_quality_metrics_computed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--plan',required=True);r.add_argument('--key-file');r.add_argument('--proxy')
    a=sub.add_parser('audit');a.add_argument('--plan',required=True);a.add_argument('--run',required=True);a.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();print(json.dumps({'prepare':prepare,'run':run,'audit':audit}[args.command](args),indent=2))
