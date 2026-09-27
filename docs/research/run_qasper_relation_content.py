"""Single-use 20 static JEV judgments; complete cache-only answer replay.

Preparation proves every possible content/placebo pack has an exact completed
answer payload. No Qwen dispatch exists in this runner. References are deferred
until all twenty new static responses and their accounting are complete.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import openrouter_decision_client as client
import run_qasper_relation_pack_answers as packs
import run_qasper_relation_pilot as pilot

ROOT, A = packs.ROOT, packs.A
SCHEMA = 'slac-qasper-relation-content-v1'
MODEL = 'typesafe/jev-1.13-20260917'
PRIOR = {'attempts':817,'reservation_usd':'4.6586873300','known_cost_usd':'0.490420862','unknown_cost_attempts':1}
METHODS = ('R0','Rcontent','Radjacent','Rplacebo','dense_k3','reranker_k3','p_yes_only_k3')
PAIRS = tuple(('Rcontent',m) for m in ('R0','Radjacent','Rplacebo','dense_k3','reranker_k3','p_yes_only_k3'))
METRICS = ('official_answer_f1','actual_evidence_tokens')
SPEC = {'questions':77,'families':24,'static_requests':20,'mask_coverage':501,'distinct_packs':96,
    'cache_payloads':262,'methods':list(METHODS),'pairs':[list(x) for x in PAIRS],'metrics':list(METRICS),
    'records':539,'intervals':24,'bootstrap_seed':20260927,'bootstrap_replicates':10000,
    'stop_at_utc':'2026-09-27T01:00:00+00:00','max_seconds':1800,'request_seconds':65,
    'generation_calls':0,'automatic_retries':0,'posthoc_exposed_development':True}
SYMBOLIC_PLAN = A/'qasper-relation-placebo-demand-plan-01'
SYMBOLIC_RUN = A/'qasper-relation-placebo-demand-run-01'
SYMBOLIC_EXEC = A/'qasper-relation-placebo-demand-root-execution-01'
ANSWER_PLAN = A/'qasper-relation-pack-answer-plan-01'
ANSWER_RUN = A/'qasper-relation-pack-answer-run-01'
PINS = {'symbolic_plan':'cec5164a94b616ced56f31d087ff2aedeac34d2a013564a4054b99babd94e9c5',
    'symbolic_summary':'5eafc87a7f7dfbd78afdb276a15d5a0769e7e94d2e095d612494515e53881c80',
    'symbolic_audit':'a98d232df42b5531268c094bfd323365fc6d98c20744f5659549e2f855c2c6ac',
    'symbolic_complete':'8233cf600251cffdf29be0cd2af9932df4b98ebdf9b5827c4c4d19e203ade310',
    'symbolic_independent':'69b4af662ebae5f032eec0b44647338f8ee553cc208ba0ce4cf3da0c3f42cfc5',
    'answer_summary':'0202bb7cf16b398d4e989784887e8410c826faa85dd5b9dc03123fe92134acd0',
    'answer_review':'9649198eb82fb15fca034348b64a20607e28736662c06df374f9051e05e41dee'}
PLAN_FILES = {'experiment_config.json','jobs.json','plan_manifest.json'}
RUN_FILES = {'provider_calls','attempt_ledger','labels.json','resolved_mapping.jsonl','per_question.jsonl','summary.json'}
read, write, merge = packs.parent.read, packs.parent.write, packs.parent.merge
canonical, object_hash, check = client.canonical_bytes, client.object_hash, packs.check
identity = packs.identity

def registration_path(): return A/'qasper-relation-content-registration-01.json'
def run_path(): return A/'qasper-relation-content-run-01'
def utc_now(): return datetime.now(timezone.utc)
def own_paths(): return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_content.py',ROOT/'docs/research/RELATION_CONTENT_EXECUTION_PROTOCOL_20260927.md']

def source_data():
    reader=packs.Reader()
    sc=reader.json(SYMBOLIC_PLAN/'plan.json',PINS['symbolic_plan'])
    seal=reader.json(SYMBOLIC_PLAN/'seal.json')
    check(seal['plan.json']==PINS['symbolic_plan'],'symbolic plan seal changed')
    inputs=reader.json(SYMBOLIC_PLAN/'inputs.json',seal['inputs.json'])
    sm=reader.json(SYMBOLIC_RUN/'summary.json',PINS['symbolic_summary'])
    audit=reader.json(SYMBOLIC_EXEC/'audit.json',PINS['symbolic_audit'])
    completed=reader.json(SYMBOLIC_EXEC/'completed.json',PINS['symbolic_complete'])
    independent=reader.json(A/'qasper-relation-placebo-demand-independent-verification-01/verification.json',PINS['symbolic_independent'])
    check(independent['status']=='verified_complete' and (independent['questions'],independent['families'],independent['masks'])==(77,24,501)
        and all(independent[k] is True for k in ('all_demanded_completion_invariance_and_degeneracy_gates_equal',
            'all_strata_permutations_full_compositions_derivatives_required_sets_and_aggregates_equal','conditional_payload_bytes_and_decimal_reservations_recomputed'))
        and independent['quality_computed'] is False and independent['api_calls']==0,'complete independent symbolic proof required')
    check(completed['status']=='completed_and_audited' and completed['summary_sha256']==PINS['symbolic_summary']
        and completed['audit_sha256']==PINS['symbolic_audit'] and audit['status']=='verified_complete'
        and sm['all_symbolic_results_available'] is True,'complete symbolic execution required')
    reader.raw(SYMBOLIC_EXEC/'root_release.json',completed['root_release_sha256'])
    for phase in ('run','audit'):
        event=reader.json(SYMBOLIC_EXEC/(phase+'_execution.json'))
        check(event['exit_code']==0 and event['external_timeout'] is False,'symbolic child failed')
        for stream in ('stdout','stderr'):reader.raw(SYMBOLIC_EXEC/(phase+'.'+stream),event[stream+'_sha256'])
    for path,h in sc['input_sha256'].items():reader.raw(path,h)
    check({p.name for p in SYMBOLIC_PLAN.iterdir()}=={'plan.json','inputs.json','seal.json'}
        and {p.name for p in SYMBOLIC_RUN.iterdir()}==set(sm['output_sha256'])|{'summary.json'},'symbolic inventory changed')
    for n,h in sm['output_sha256'].items():reader.raw(SYMBOLIC_RUN/n,h)
    for p in [SYMBOLIC_PLAN/'plan.json',SYMBOLIC_PLAN/'inputs.json',SYMBOLIC_PLAN/'seal.json',SYMBOLIC_RUN/'summary.json',*(SYMBOLIC_RUN/n for n in sm['output_sha256'])]:
        check(independent['input_output_sha256'].get(str(p))==reader.bindings[str(p)],'independent symbolic source commitment differs')
    pub=reader.json(SYMBOLIC_RUN/'public_aggregate.json')
    check(pub['semantic_content_paid_phase_must_stop'] is False and pub['required_unique_edges']==20
        and pub['function_different_questions']==1 and pub['different_output_assignments']==4,'frozen nondegenerate population differs')
    symbolic={identity(q):q for q in reader.rows(SYMBOLIC_RUN/'per_question.jsonl')}
    jobs=reader.json(SYMBOLIC_RUN/'static_payloads.json');required=reader.json(SYMBOLIC_RUN/'required_edges.json')
    tasks={tuple(t[k] for k in ('doc_id','left_id','right_id')):t for t in inputs['static_tasks']}
    check(len(jobs)==len(required)==len(set(map(tuple,required)))==20,'all twenty unique tasks required')
    for job in jobs:
        task=tasks[tuple(job['edge'])];payload=client.make_payload([task],'static','jev');r,i,o=client.reservation(payload,'jev')
        check(job=={'edge':job['edge'],'task_id':task['id'],'endpoint':client.MODELS['jev']['endpoint'],'payload':payload,
            'payload_sha256':object_hash(payload),'reservation_usd':str(r),'input_allowance':i,'output_allowance':o},'frozen singleton task differs')
    check([j['edge'] for j in jobs]==required and len({j['task_id'] for j in jobs})==20,'job order/ID changed')
    ac,data,oldjobs,oldpacks,masks,baselines,inheritance,own=packs.load_plan(ANSWER_PLAN)
    answers,ledger,calls,complete=packs.inspect_calls(ac,oldjobs,ANSWER_RUN,require_complete=True)
    oldsummary=reader.json(ANSWER_RUN/'summary.json',PINS['answer_summary'])
    review=reader.json(A/'qasper-relation-pack-answer-independent-verification-01/result-01/verification.json',PINS['answer_review'])
    check(complete and review['status']=='independently_verified_complete' and review['summary_sha256']==PINS['answer_summary']
        and review['night_accounting']==PRIOR,'complete prior response/accounting review required')
    reader.merge(data['input_sha256']);reader.merge(own);reader.merge(calls);reader.merge(packs.registration_binding(ac,own))
    for p,h in calls.items():check(oldsummary['output_sha256'][str(Path(p).relative_to(ANSWER_RUN))]==h,'parent call output seal differs')
    cache=dict(data['cache'])
    for ordinal,job in enumerate(oldjobs,1):
        attempt=ledger['attempts'][ordinal-1];key=job['cache_key'];check(key not in cache,'physical cache overlap')
        cache[key]={'job':job,'answer':answers[key],'origin':'relation_pack','request_ordinal':ordinal,
            'request_sha256':attempt['request_sha256'],'response_sha256':attempt['response_sha256'],'resolved_model':packs.MODEL}
    maskmap={(identity(r),r['mask']):r['cache_key'] for r in masks};queries={identity(q['cube']):q['cube'] for q in inputs['queries']}
    check(len(queries)==len(symbolic)==77 and set(queries)==set(symbolic) and len(maskmap)==501,'full cube coverage differs')
    frozen={}
    for row in oldpacks:
        key=(identity(row),row['cache_key']);check(key not in frozen,'duplicate query pack');frozen[key]=row
    for q,cube in queries.items():
        request=cache[maskmap[q,0]]['job']['payload'];question=json.loads(request['messages'][1]['content'])['question']
        for mask,outcome in enumerate(cube['outcomes']):
            key=maskmap[q,mask];job=packs.job_for(question,outcome['rendered_pack']);row=frozen[q,key]
            check(job==cache[key]['job'] and canonical(job['payload'])==canonical(cache[key]['job']['payload'])
                and row['pack_sha256']==outcome['pack_sha256'] and row['actual_evidence_tokens']==outcome['actual_tokens']
                and row['selected_ids']==[x[1] for x in outcome['selected_identities']],'full-domain cache payload proof failed')
        s=symbolic[q];check(s['edges']==cube['edges'] and sorted(s['source_by_target'])==list(range(len(cube['edges']))),'permutation domain changed')
    keys={r['cache_key'] for r in oldpacks+baselines}
    check(len(keys)==262 and keys<=cache.keys() and len(frozen)==96 and Counter(r['method'] for r in baselines)==Counter({m:77 for m in packs.BASELINES}),'all cached answers/baselines required')
    fullanswers={k:cache[k]['answer'] for k in keys}
    reader.json(ANSWER_RUN/'answers.json',oldsummary['output_sha256']['answers.json'])
    check(fullanswers==read(ANSWER_RUN/'answers.json'),'saved parent answers differ')
    for p in own_paths()+[Path(packs.parent.bootstrap.__file__)]:reader.raw(p)
    reader.verify()
    return {'jobs':jobs,'tasks':tasks,'queries':queries,'symbolic':symbolic,'maskmap':maskmap,'packs':frozen,
        'baselines':baselines,'answers':fullanswers,'cache':{k:cache[k] for k in keys},'prepared':data['prepared'],
        'references':data['references'],'prior':PRIOR,'input_sha256':reader.bindings}

def prediction(data):
    jobs=data['jobs'];amount=sum((Decimal(j['reservation_usd']) for j in jobs),Decimal(0))
    check(len(jobs)==20 and amount==Decimal('.100') and data['prior']==PRIOR,'full stage budget differs')
    total=Decimal(PRIOR['reservation_usd'])+amount;check(total<=5,'whole night budget exceeded')
    return {'requests':20,'questions':20,'reservation_usd':str(amount),'input_allowance':sum(j['input_allowance'] for j in jobs),
        'output_allowance':sum(j['output_allowance'] for j in jobs),'prospective_night_reservation_usd':str(total),
        'new_generation_requests':0,'complete_possible_payloads_proven':262}

def prepare(args):
    output=Path(args.output).resolve();run=run_path()
    check(not output.exists() and not run.exists() and not registration_path().exists(),'single-use plan/run already exists')
    packs.disjoint(output,[SYMBOLIC_PLAN,SYMBOLIC_RUN,ANSWER_PLAN,ANSWER_RUN,run])
    data=source_data();pred=prediction(data)
    config={'schema':SCHEMA,'status':'prepared_no_static_calls','specification':SPEC,'created_at_utc':utc_now().isoformat(),
        'prior_night_accounting':PRIOR,'prediction':pred,'input_sha256':data['input_sha256'],
        'data_object_sha256':object_hash(serializable_data(data)),'jobs_sha256':object_hash(data['jobs']),
        'resolved_model_locks':{'jev':MODEL},'run_output':str(run),'single_use_registration':str(registration_path()),
        'deferred_reference_binding':data['references'],'generation_calls':0}
    output.mkdir(parents=True);write(output/'jobs.json',data['jobs']);write(output/'experiment_config.json',config)
    write(output/'plan_manifest.json',{'schema':SCHEMA,'experiment_config_sha256':pilot.digest(output/'experiment_config.json'),'jobs_sha256':pilot.digest(output/'jobs.json')})
    pilot.verify_hashes(data['input_sha256']);return config

def serializable_data(data):
    return {k:([{'key':list(key) if not isinstance(key,tuple) or not isinstance(key[0],tuple) else [list(key[0]),key[1]],'value':v} for key,v in sorted(value.items())] if k in ('tasks','queries','symbolic','maskmap','packs') else value) for k,value in data.items()}

def load_plan(directory):
    directory=Path(directory).resolve();check({p.name for p in directory.iterdir()}==PLAN_FILES,'plan inventory differs')
    reader=packs.Reader();seal=reader.json(directory/'plan_manifest.json');config=reader.json(directory/'experiment_config.json',seal['experiment_config_sha256']);jobs=reader.json(directory/'jobs.json',seal['jobs_sha256'])
    check(seal['schema']==SCHEMA and config['schema']==SCHEMA and config['status']=='prepared_no_static_calls'
        and config['specification']==SPEC and config['run_output']==str(run_path()) and config['single_use_registration']==str(registration_path())
        and config['resolved_model_locks']=={'jev':MODEL} and config['generation_calls']==0,'frozen experiment contract differs')
    pilot.verify_hashes(config['input_sha256'])
    data=source_data();check(config['input_sha256']==data['input_sha256'] and config['prior_night_accounting']==PRIOR
        and config['prediction']==prediction(data) and jobs==data['jobs'] and config['jobs_sha256']==object_hash(jobs)
        and config['data_object_sha256']==object_hash(serializable_data(data)) and config['deferred_reference_binding']==data['references'],'plan source reconstruction differs')
    reader.verify();pilot.verify_hashes(config['input_sha256']);return config,data,jobs,reader.bindings

def ensure_time(started,dispatch=False):
    margin=65 if dispatch else 0
    check(time.monotonic()-started<SPEC['max_seconds']-margin and (datetime.fromisoformat(SPEC['stop_at_utc'])-utc_now()).total_seconds()>margin,'execution/09:00 deadline exhausted')

class Guard:
    def __init__(self,output,seconds):
        self.cancel=threading.Event()
        def stop():
            if not self.cancel.wait(max(0,seconds)):
                try:write(Path(output)/'hard_deadline.json',{'status':'failed','main_results_available':False})
                finally:os._exit(124)
        self.thread=threading.Thread(target=stop,daemon=True);self.thread.start()
    def close(self):self.cancel.set();self.thread.join(timeout=1)

class StaticClient(client.BoundedClient):
    def __init__(self,*args,jobs,started,guard_factory=Guard,**kwargs):
        self.jobs=jobs;self.started=started;self.guard_factory=guard_factory;self.blocked=False
        super().__init__(*args,**kwargs)
    def save(self):
        invalid=False
        if self.ledger['attempts'] and self.ledger['attempts'][-1]['status']=='completed':
            record=self.ledger['attempts'][-1]
            try:
                pilot.validate_reused_response(read(self.output/f'response_{record["attempt"]:03d}.json'),self.jobs[record['attempt']-1]['payload'],record)
            except (ValueError,TypeError,KeyError):
                record.update(status='halted',error_class='InvalidResponseIdentity');self.ledger['halt_reason']='InvalidResponseIdentity';self.blocked=True;invalid=True
        if not hasattr(self,'sequence'):
            self.sequence=0;self.previous=None;self.artifacts={};self.ledger['resolved_models']={'jev':MODEL};self.ledger['resolved_model_locks']={'jev':MODEL}
        events=self.output.parent/'attempt_ledger';events.mkdir(exist_ok=True);added={}
        for prefix in ('request','response','error_response'):
            n=f'{prefix}_{len(self.ledger["attempts"]):03d}.json';p=self.output/n
            if p.exists() and n not in self.artifacts:added[n]=pilot.digest(p)
        event={'schema':SCHEMA+'-event','sequence':self.sequence,'previous_sha256':self.previous,'new_artifacts':added,'ledger':self.redacted(self.ledger)}
        path=events/f'event_{self.sequence:05d}.json';check(not path.exists(),'immutable event collision');write(path,event)
        self.previous=pilot.digest(path);self.sequence+=1;self.artifacts.update(added);super().save()
        if invalid:raise ValueError('completed response failed strict identity validation')
        if self.ledger['attempts'] and self.ledger['attempts'][-1]['status']=='in_flight' and 'elapsed_seconds' not in self.ledger['attempts'][-1]:ensure_time(self.started,True)
    def submit(self,tasks,kind,backend):
        ensure_time(self.started,True);attempts=self.ledger['attempts']
        check(not self.blocked and (not attempts or attempts[-1]['status']=='completed'),'terminal failure prohibits continuation')
        check(len(attempts)<len(self.jobs) and kind=='static' and backend=='jev','static-only frozen stage')
        job=self.jobs[len(attempts)];payload=client.make_payload(tasks,kind,backend)
        check(payload==job['payload'] and canonical(payload)==canonical(job['payload']),'next exact frozen request differs')
        guard=self.guard_factory(self.output.parent,65)
        try:
            labels=super().submit(tasks,kind,backend);record=self.ledger['attempts'][-1]
            pilot.validate_reused_response(read(self.output/f'response_{record["attempt"]:03d}.json'),payload,record)
            check(record['response_model']==MODEL,'dated JEV model changed');return labels
        except BaseException:self.blocked=True;raise
        finally:guard.close()

def inspect_calls(config,jobs,output,require_complete):
    output=Path(output);calls=output/'provider_calls';reader=packs.Reader();ledger=reader.json(calls/'ledger.json')
    eventpaths=sorted((output/'attempt_ledger').iterdir());check([p.name for p in eventpaths]==[f'event_{i:05d}.json' for i in range(len(eventpaths))] and eventpaths,'event inventory differs')
    previous=None;last=None;artifacts={}
    for i,path in enumerate(eventpaths):
        ev=reader.json(path);state=ev['ledger']
        check(set(ev)=={'schema','sequence','previous_sha256','new_artifacts','ledger'} and ev['schema']==SCHEMA+'-event' and ev['sequence']==i and ev['previous_sha256']==previous
            and state['resolved_models']==state['resolved_model_locks']=={'jev':MODEL},'event/model lock differs')
        if last is None:check(state['attempts']==[] and state['reservation_total_usd']=='0' and state['actual_reported_cost_usd']=='0' and state['halt_reason'] is None,'initial event must be empty and unpaid')
        else:
            check(set(state)==set(last),'event ledger field set changed')
            old,new=last['attempts'],state['attempts'];fixed=set(state)-{'attempts','reservation_total_usd','actual_reported_cost_usd','halt_reason'}
            check({k:state[k] for k in fixed}=={k:last[k] for k in fixed},'fixed event contract differs')
            if len(new)==len(old)+1:check(new[:-1]==old and (not old or old[-1]['status']=='completed') and new[-1]['status']=='in_flight','append after failure')
            else:check(len(old)==len(new) and bool(old) and old[-1]['status']=='in_flight' and new[-1]['status'] in ('completed','halted','in_flight') and new[:-1]==old[:-1] and all(new[-1].get(k)==v for k,v in old[-1].items() if k!='status'),'invalid terminal transition')
        check(Decimal(state['reservation_total_usd'])==sum((Decimal(r['reserved_usd']) for r in state['attempts']),Decimal(0)) and Decimal(state['actual_reported_cost_usd'])==sum((pilot.nonnegative_decimal(r['actual_cost_usd']) for r in state['attempts'] if r.get('actual_cost_usd') is not None),Decimal(0)),'event accounting differs')
        for name,h in ev['new_artifacts'].items():
            check(name not in artifacts and name in {f'{prefix}_{len(state["attempts"]):03d}.json' for prefix in ('request','response','error_response')},'invalid artifact introduction');reader.raw(calls/name,h);artifacts[name]=h
        previous=reader.bindings[str(path.resolve())];last=state
    check(last==ledger and {p.name for p in calls.iterdir()}==set(artifacts)|{'ledger.json'},'final event/provider inventory differs')
    pred=config['prediction'];check(len(ledger['attempts'])<=len(jobs) and all(ledger[k]==v for k,v in {'budget_usd':pred['reservation_usd'],'request_cap':20,'question_cap':20,'conservative_input_token_cap':pred['input_allowance'],'prompt_version':client.PROMPT_VERSION,'automatic_retries':0}.items()),'client cap changed')
    labels={};complete=len(ledger['attempts'])==len(jobs) and ledger['halt_reason'] is None
    for i,r in enumerate(ledger['attempts'],1):
        j=jobs[i-1];payload=pilot.prior_request(calls/f'request_{i:03d}.json',r,client.PROMPT_VERSION)
        check(r['attempt']==i and r['backend']=='jev' and r['kind']=='static' and r['task_ids']==[j['task_id']] and payload==j['payload'] and (calls/f'request_{i:03d}.json').read_bytes()==canonical(payload),'saved request differs')
        for k in ('input_allowance','output_allowance'):check(r[k]==j[k],'request allowance differs')
        check(Decimal(r['reserved_usd'])==Decimal(j['reservation_usd']),'request reservation differs')
        started=datetime.fromisoformat(r['started_at']);elapsed=r.get('elapsed_seconds')
        check(started.utcoffset() is not None and (datetime.fromisoformat(SPEC['stop_at_utc'])-started).total_seconds()>65,'saved deadline admission differs')
        check((elapsed is None and r['status']=='in_flight') or type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,'elapsed invalid')
        if r['status']=='completed':
            check(elapsed<=65 and r['response_model']==MODEL,'completed model or deadline differs')
            value=pilot.validate_reused_response(reader.json(calls/f'response_{i:03d}.json'),payload,r);check(not labels.keys()&value.keys(),'duplicate decision');labels.update(value)
        else:
            check(i==len(ledger['attempts']) and r['status'] in ('halted','in_flight'),'failure not terminal');complete=False
            rawcost=None
            for prefix in ('response','error_response'):
                p=calls/f'{prefix}_{i:03d}.json'
                if p.exists():
                    body=reader.json(p)
                    try:rawcost=pilot.nonnegative_decimal(body.get('usage',{}).get('cost'))
                    except (ValueError,TypeError,ArithmeticError):pass
            check((rawcost is None)==(r.get('actual_cost_usd') is None) and (rawcost is None or rawcost==pilot.nonnegative_decimal(r['actual_cost_usd'])),'failed fee lacks raw evidence')
    if require_complete:check(complete and len(labels)==20,'all twenty responses required before score')
    reader.verify();return labels,ledger,reader.bindings,complete

def registration_binding(config,own):
    reader=packs.Reader();v=reader.json(registration_path())
    check(set(v)=={'schema','run_output','plan_sha256','registered_at_utc'} and v['schema']==SCHEMA and v['run_output']==config['run_output'] and v['plan_sha256']==next(h for p,h in own.items() if Path(p).name=='experiment_config.json') and datetime.fromisoformat(v['registered_at_utc']).utcoffset() is not None,'single-use registration differs')
    return reader.bindings

def resolve(data,labels):
    check(set(labels)=={j['task_id'] for j in data['jobs']} and all(v in client.CRITERIA['static'] for v in labels.values()),'complete legal static labels required')
    observed={tuple(j['edge']):labels[j['task_id']]=='dependent' for j in data['jobs']};mapping=[]
    for q,cube in sorted(data['queries'].items()):
        s=data['symbolic'][q];edges=cube['edges'];known={i:observed[tuple(e)] for i,e in enumerate(edges) if tuple(e) in observed}
        check(set(s['required_indices'])<=set(known),'required original label missing')
        matches=[mask for mask in range(len(cube['outcomes'])) if all(bool(mask&(1<<i))==v for i,v in known.items())]
        check(matches,'no consistent hypothetical completion');content=set();placebo=set()
        for mask in matches:
            content.add(data['maskmap'][q,mask]);p=sum(((mask>>src)&1)<<dst for dst,src in enumerate(s['source_by_target']));placebo.add(data['maskmap'][q,p])
        check(len(content)==len(placebo)==1,'unobserved completion changes output')
        chosen={'R0':data['maskmap'][q,0],'Rcontent':next(iter(content)),'Radjacent':data['maskmap'][q,len(cube['outcomes'])-1],'Rplacebo':next(iter(placebo))}
        for method,key in chosen.items():
            r=data['packs'][q,key];mapping.append({k:r[k] for k in (*packs.IDENTITY,'cache_key','selected_ids','pack_sha256','actual_evidence_tokens')}|{'method':method})
    for r in data['baselines']:
        if r['method']!='I_jev_k3':mapping.append(dict(r))
    expected={(m,*q) for m in METHODS for q in data['queries']}
    check(len(mapping)==len(expected)==539 and {(r['method'],*identity(r)) for r in mapping}==expected,'all seven complete arms required')
    return mapping

def score(data,mapping):
    reader=packs.Reader();annotations=packs.load_references(data,reader);records=[]
    for r in mapping:
        refs=packs.legacy.references_from_annotations(annotations[r['doc_id'],r['question_id']]);answer=data['answers'][r['cache_key']]
        records.append({**r,'predicted_answer':answer,'official_answer_f1':max(packs.legacy.token_f1_score(answer,v['answer']) for v in refs)})
    qs=sorted(data['queries']);groups,draws=packs.parent.bootstrap.family_resamples(qs)
    domain,_=packs.parent.bootstrap.summarize_domain('answer',records,METHODS,METRICS,PAIRS,qs,groups,draws)
    reader.verify();return records,{'metrics':domain['method_means'],'paired_comparisons':domain['paired_comparisons'],'shared_resamples_sha256':hashlib.sha256(draws.tobytes()).hexdigest()},reader.bindings

def accounting(ledger,complete):
    n=len(ledger['attempts']);unknown=sum(r.get('actual_cost_usd') is None for r in ledger['attempts'])
    return {'prior_night_accounting':PRIOR,'new_static_attempts':n,'new_static_completed':sum(r['status']=='completed' for r in ledger['attempts']),
        'new_static_reserved_usd':ledger['reservation_total_usd'],'new_static_known_cost_usd':ledger['actual_reported_cost_usd'],'new_static_unknown_cost_attempts':unknown,
        'night_attempts':817+n,'night_reservation_usd':str(Decimal(PRIOR['reservation_usd'])+Decimal(ledger['reservation_total_usd'])),
        'night_known_cost_subtotal_usd':str(Decimal(PRIOR['known_cost_usd'])+Decimal(ledger['actual_reported_cost_usd'])),'night_unknown_cost_attempts':1+unknown,
        'old_unknown_refunded':False,'inherited_generation_charged_again':0,'generation_calls':0,'automatic_retries':0,'main_results_available':complete}

def complete_results(config,data,jobs,own,output):
    labels,ledger,bound,complete=inspect_calls(config,jobs,output,True);bound=merge(bound,registration_binding(config,own))
    check(not any((Path(output)/n).exists() for n in ('failure.json','hard_deadline.json')),'failed run cannot score')
    pilot.verify_hashes(merge(config['input_sha256'],own,bound));mapping=resolve(data,labels)
    records,metrics,refs=score(data,mapping);bound=merge(bound,refs)
    a={identity(r):r['cache_key'] for r in mapping if r['method']=='Rcontent'};b={identity(r):r['cache_key'] for r in mapping if r['method']=='Rplacebo'}
    summary={'schema':SCHEMA,'status':'completed','question_count':77,'family_count':24,'record_count':539,'specification':SPEC,**metrics,**accounting(ledger,True),
        'content_placebo_equal_packs':sum(a[q]==b[q] for q in a),'semantic_content_claim_must_stop':a==b,
        'observed_label_counts':dict(Counter(labels.values())),'observed_label_scope':'twenty demanded unique edges only; not full original strata or 562 labels',
        'unobserved_labels_imputed':False,'independent_confirmation':False,'plan_sha256':next(h for p,h in own.items() if Path(p).name=='experiment_config.json'),
        'input_binding_sha256':object_hash(merge(config['input_sha256'],own,bound))}
    pilot.verify_hashes(merge(config['input_sha256'],own,bound));return labels,mapping,records,summary,bound

def run(args,client_factory=StaticClient,guard_factory=Guard):
    start=time.monotonic();config,data,jobs,own=load_plan(args.plan);ensure_time(start)
    check(not run_path().exists() and not registration_path().exists(),'single-use stage already attempted')
    write(registration_path(),{'schema':SCHEMA,'run_output':str(run_path()),'plan_sha256':next(h for p,h in own.items() if Path(p).name=='experiment_config.json'),'registered_at_utc':utc_now().isoformat()})
    output=run_path();output.mkdir();guard=guard_factory(output,min(SPEC['max_seconds']-(time.monotonic()-start),(datetime.fromisoformat(SPEC['stop_at_utc'])-utc_now()).total_seconds()))
    try:
        c=client_factory(output/'provider_calls',jobs=jobs,started=start,key_file=args.key_file,proxy=args.proxy,
            budget_usd=config['prediction']['reservation_usd'],request_cap=20,question_cap=20,token_cap=config['prediction']['input_allowance'])
        for j in jobs:c.submit([data['tasks'][tuple(j['edge'])]],'static','jev')
        ensure_time(start);labels,mapping,records,summary,bound=complete_results(config,data,jobs,own,output);ensure_time(start)
        write(output/'labels.json',labels);pilot.write_rows(output/'resolved_mapping.jsonl',mapping);pilot.write_rows(output/'per_question.jsonl',records)
        summary['output_sha256']={str(Path(p).relative_to(output)):h for p,h in packs.inherited.tree(output).items()}
        write(output/'summary.json',summary);pilot.verify_hashes(merge(config['input_sha256'],own,bound));ensure_time(start)
        check({p.name for p in output.iterdir()}==RUN_FILES,'final output inventory differs');return summary
    except BaseException as exc:
        if (output/'summary.json').exists():(output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_quality','error_type':type(exc).__name__,'main_results_available':False,'automatic_retries':0});raise
    finally:guard.close()

def audit(args):
    config,data,jobs,own=load_plan(args.plan);output=Path(args.run).resolve();check(str(output)==config['run_output'],'wrong fixed run')
    before=packs.inherited.tree(output);labels,ledger,bound,complete=inspect_calls(config,jobs,output,not args.allow_incomplete)
    bound=merge(bound,registration_binding(config,own));success=complete and (output/'summary.json').exists() and not any((output/n).exists() for n in ('failure.json','hard_deadline.json'))
    if success:
        check({p.name for p in output.iterdir()}==RUN_FILES,'complete inventory differs');labels,mapping,records,summary,bound=complete_results(config,data,jobs,own,output)
        summary['output_sha256']={str(Path(p).relative_to(output)):h for p,h in before.items() if Path(p)!=output/'summary.json'}
        check(read(output/'labels.json')==labels and packs.parent.rows(output/'resolved_mapping.jsonl')==mapping and packs.parent.rows(output/'per_question.jsonl')==records and read(output/'summary.json')==summary,'complete science replay differs')
    else:check(args.allow_incomplete,'incomplete run cannot report quality')
    pilot.verify_hashes(merge(config['input_sha256'],own,bound,before));check(before==packs.inherited.tree(output),'output changed during audit')
    return {'schema':SCHEMA+'-audit','status':'verified_complete' if success else 'verified_incomplete','accounting':accounting(ledger,success),
        'all_bound_inputs_outputs_unchanged':True,'all_539_records_and_24_intervals_replayed':success,'api_calls_by_audit':0,'key_read':False,'partial_quality_metrics_computed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--key-file');p.add_argument('--proxy')
    p=sub.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--run',required=True);p.add_argument('--allow-incomplete',action='store_true')
    a=parser.parse_args();print(json.dumps({'prepare':prepare,'run':run,'audit':audit}[a.command](a),indent=2))
