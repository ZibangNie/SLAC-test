"""Independent stdlib reconstruction of saved relation-gate behavior.

Does not import the official gate, selector, renderer or aggregator. The only
non-stdlib component, loaded at verification time, is the same local tokenizer.
Never run against partial output. No QA, labels API, answers or quality scoring.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
PLAN_SHA='c67285f13543a7b633be87aa9e961bc6a9866482acba96e9defed25b873542b3'
PROTOCOL_SHA='c4c2a294cdb2c7db64949267cddb027520aa6bafd27980f01aad75452d9ce65a'
PROTOCOL=ROOT/'docs/research/results/qasper_relation_opportunity_protocol_20260927.json'
COMPLETION=ROOT/'artifacts/research-foundation/qasper-relation-opportunity-root-execution-01/completed.json'
UNIT_KEYS={'unit_id','order','kind','start','end','text','native_text'}
CASE_KEYS={'family_id','doc_id','question_id','units','candidates','ranking','labels',
           'baseline_selected','baseline_tokens','baseline_pack_sha256'}
RUN_FILES={'per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'}


def check(deadline):
    if time.monotonic()>deadline:raise TimeoutError('independent verification deadline exceeded')


def encode(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def render(case,indices):
    return '\n\n'.join('['+case['units'][i]['unit_id']+']\n'+case['units'][i]['text']
                        for i in sorted(indices,key=lambda j:case['units'][j]['order']))


def edge_domain(case):
    if set(case)!=CASE_KEYS:raise ValueError('independent core whitelist mismatch')
    candidates=case['candidates']; rank=case['ranking']; labels=case['labels']; units=case['units']
    if (not 1<=len(candidates)<=16 or any(type(i) is not int or not 0<=i<len(units) for i in candidates)
        or candidates!=sorted(set(candidates)) or len(rank)!=len(candidates) or set(rank)!=set(candidates)
        or any(type(i) is not int for i in rank) or len(labels)!=len(candidates)
        or any(v not in ('yes','no','unknown') for v in labels)):
        raise ValueError('invalid independent case candidates/labels')
    if (any(set(u)!=UNIT_KEYS for u in units) or
        any(type(u['order']) is not int or u['order']!=i for i,u in enumerate(units))):
        raise ValueError('native source order differs')
    if (len({u['unit_id'] for u in units})!=len(units) or
        any(not isinstance(u[k],str) for u in units for k in ('unit_id','text','native_text'))):
        raise ValueError('invalid native unit identity/text')
    yes={i for i,l in zip(candidates,labels) if l!='no'}
    return [(a,a+1) for a in candidates if a in yes and a+1 in yes and units[a]['native_text']!=units[a+1]['native_text']]


def replay(case,mask,token_count,deadline=math.inf):
    """Use integer half-scores and max priority, independently of official code."""
    domain=edge_domain(case)
    if type(mask) is not int or not 0<=mask<2**len(domain):raise ValueError('invalid complete-mask index')
    active={a:b for bit,(a,b) in enumerate(domain) if (mask>>bit)&1}
    units=case['units']; rank={i:r for r,i in enumerate(case['ranking'])}
    score={i:(2 if label=='yes' else 1) for i,label in zip(case['candidates'],case['labels']) if label!='no'}
    remaining=list(score); selected=[]; steps=[]
    def tokens(indices):
        value=token_count(tuple(sorted(indices)))
        if type(value) is not int or value<0:raise ValueError('invalid independent whole-pack tokens')
        return value
    while remaining and len(selected)<3:
        check(deadline)
        base_first=max(remaining,key=lambda i:(score[i],-rank[i],-units[i]['order']))
        bonuses={i:int(i in active and active[i] in selected) for i in remaining}
        winner=max(remaining,key=lambda i:(score[i]+2*bonuses[i],-rank[i],-units[i]['order']))
        remaining.remove(winner)
        row={'step':len(steps),'candidate_index':winner,'selected_before':sorted(selected),
             'base_score':score[winner]/2,'relation_bonus':bonuses[winner],
             'active_triggers':[[winner,active[winner]]] if bonuses[winner] else [],
             'priority_changed':winner!=base_first,'proposed_tokens':None,'accepted':False}
        if any(units[j]['native_text']==units[winner]['native_text'] for j in selected):
            row['action']='skip_exact_native_duplicate'
        else:
            row['proposed_tokens']=tokens(selected+[winner])
            if row['proposed_tokens']>1024:row['action']='skip_evidence_budget'
            else:
                selected.append(winner);row['accepted']=True;row['action']='select'
        steps.append(row)
    selected.sort()
    return {'selected_indices':selected,'selected_ids':[units[i]['unit_id'] for i in selected],
            'actual_tokens':tokens(selected),'pack_sha256':hashlib.sha256(render(case,selected).encode('utf-8')).hexdigest(),
            'trace':steps}


def hist(values):return {str(k):v for k,v in sorted(Counter(values).items())}


def reconstruct(cases,tokenizer,deadline=math.inf):
    masks=[];questions=[]
    for case in cases:
        cache={():0}
        def count(indices):
            check(deadline)
            if indices not in cache:
                cache[indices]=len(tokenizer.encode(render(case,indices),add_special_tokens=True,truncation=False))
            return cache[indices]
        edges=edge_domain(case); results=[replay(case,m,count,deadline) for m in range(2**len(edges))]
        zero=results[0]; full=results[-1]
        if (zero['selected_indices']!=case['baseline_selected'] or zero['actual_tokens']!=case['baseline_tokens']
            or zero['pack_sha256']!=case['baseline_pack_sha256']):raise ValueError('original audited I witness differs')
        identity={k:case[k] for k in ('family_id','doc_id','question_id')};local=[]
        for mask,result in enumerate(results):
            item={**identity,'mask':mask,'active_edges':[list(e) for bit,e in enumerate(edges) if (mask>>bit)&1],**result,
                  'added_indices':sorted(set(result['selected_indices'])-set(zero['selected_indices'])),
                  'removed_indices':sorted(set(zero['selected_indices'])-set(result['selected_indices'])),
                  'tokens_minus_zero':result['actual_tokens']-zero['actual_tokens'],
                  'selected_count_minus_zero':len(result['selected_indices'])-len(zero['selected_indices']),
                  'pack_changed':result['pack_sha256']!=zero['pack_sha256']}
            masks.append(item);local.append(item)
        questions.append({**identity,'eligible_edges':[list(e) for e in edges],'mask_count':len(local),
            'distinct_pack_count':len({(r['pack_sha256'],tuple(r['selected_indices'])) for r in local}),
            'any_pack_change':any(r['pack_changed'] for r in local),'changed_masks':sum(r['pack_changed'] for r in local),
            'minimum_actual_tokens':min(r['actual_tokens'] for r in local),'maximum_actual_tokens':max(r['actual_tokens'] for r in local),
            'zero_tokens':zero['actual_tokens'],'zero_pack_sha256':zero['pack_sha256'],
            'zero_selected_indices':zero['selected_indices'],'full_tokens':full['actual_tokens'],
            'full_pack_changed':full['pack_sha256']!=zero['pack_sha256'],
            'zero_independent_trace_parity':True,'zero_frozen_pack_parity':True,'full_adjacency_trace_parity':True})
    computed={'zero_independent_trace_parity_questions':len(questions),'zero_frozen_pack_parity_questions':len(questions),
        'full_adjacency_trace_parity_questions':len(questions),
        'queries_with_any_eligible_edge':sum(bool(q['eligible_edges']) for q in questions),
        'queries_with_any_possible_pack_change':sum(q['any_pack_change'] for q in questions),
        'full_adjacency_changed_pack_questions':sum(q['full_pack_changed'] for q in questions),
        'changed_masks':sum(r['pack_changed'] for r in masks),'unchanged_masks':sum(not r['pack_changed'] for r in masks),
        'distinct_pack_count_per_question_histogram':hist(q['distinct_pack_count'] for q in questions),
        'added_removed_count_histogram':dict(sorted(Counter(str(len(r['added_indices']))+','+str(len(r['removed_indices'])) for r in masks).items())),
        'selected_count_minus_zero_histogram':hist(r['selected_count_minus_zero'] for r in masks),
        'actual_tokens_range':[min(r['actual_tokens'] for r in masks),max(r['actual_tokens'] for r in masks)],
        'tokens_minus_zero_range':[min(r['tokens_minus_zero'] for r in masks),max(r['tokens_minus_zero'] for r in masks)],
        'tokens_minus_zero_histogram':hist(r['tokens_minus_zero'] for r in masks),
        'per_question_actual_token_range_width_histogram':hist(q['maximum_actual_tokens']-q['minimum_actual_tokens'] for q in questions)}
    return masks,questions,computed


def verify(args,tokenizer_factory=None):
    started=time.monotonic();deadline=started+300
    plan=Path(args.plan).resolve();run=Path(args.run).resolve();output=Path(args.output).resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (plan,run)):
        raise ValueError('verification output must be new and separate')
    if {p.name for p in plan.iterdir()}!={'plan.json','cases.json','seal.json'} or {p.name for p in run.iterdir()}!=RUN_FILES:
        raise ValueError('complete plan/run inventory required')
    completion=Path(getattr(args,'completion',COMPLETION)).resolve()
    direct=[PROTOCOL,completion,completion.parent/'audit.json',Path(__file__).resolve(),
            ROOT/'tests/research/test_qasper_relation_opportunity_independent.py',
            *(plan/n for n in ('plan.json','cases.json','seal.json')),*(run/n for n in RUN_FILES)]
    buffers={str(p):p.read_bytes() for p in direct};hashes={p:hashlib.sha256(b).hexdigest() for p,b in buffers.items()}
    if hashes[str(PROTOCOL)]!=PROTOCOL_SHA or hashes[str(plan/'plan.json')]!=PLAN_SHA:
        raise ValueError('external protocol/plan anchor mismatch')
    public_protocol=json.loads(buffers[str(PROTOCOL)]);config=json.loads(buffers[str(plan/'plan.json')])
    seal=json.loads(buffers[str(plan/'seal.json')]);cases=json.loads(buffers[str(plan/'cases.json')])
    if seal!={n:hashes[str(plan/n)] for n in ('plan.json','cases.json')}:raise ValueError('plan seal mismatch')
    if str(run)!=config['run_output'] or hashlib.sha256(encode(config['input_sha256'])).hexdigest()!=config['input_binding_sha256']:
        raise ValueError('fixed output/input commitment differs')
    for path,expected in config['input_sha256'].items():
        check(deadline)
        if sha(path)!=expected:raise ValueError('direct source changed')
        hashes[path]=expected
    summary=json.loads(buffers[str(run/'summary.json')]);public=json.loads(buffers[str(run/'public_aggregate.json')])
    completed=json.loads(buffers[str(completion)]);audit=json.loads(buffers[str(completion.parent/'audit.json')])
    if (completed.get('status')!='completed_and_audited' or completed.get('all_direct_bindings_unchanged') is not True
        or completed.get('plan_sha256')!=PLAN_SHA or completed.get('summary_sha256')!=hashes[str(run/'summary.json')]
        or completed.get('audit_sha256')!=hashes[str(completion.parent/'audit.json')]
        or audit.get('status')!='verified_complete' or audit.get('questions')!=77 or audit.get('masks')!=501
        or audit.get('all_masks_and_reference_traces_recomputed') is not True
        or audit.get('all_direct_input_hashes_unchanged') is not True
        or audit.get('input_binding_sha256')!=config['input_binding_sha256']
        or audit.get('run_sha256')!={str(run/n):hashes[str(run/n)] for n in RUN_FILES}):
        raise ValueError('root completed audit receipt required')
    if summary.get('status')!='completed' or summary.get('all_masks_available') is not True:
        raise ValueError('only completed gate may be verified')
    if summary['output_sha256']!={n:hashes[str(run/n)] for n in RUN_FILES-{'summary.json'}}:
        raise ValueError('run output seal differs')
    if config['inventory']!=public_protocol['inventory'] or config['specification']!=public_protocol['specification']:
        raise ValueError('public frozen scope differs')
    if (len(cases)!=77 or len({c['family_id'] for c in cases})!=24
        or len({(c['doc_id'],c['question_id']) for c in cases})!=77):raise ValueError('independent denominator differs')
    edge_hist=Counter(len(edge_domain(c)) for c in cases)
    if {str(k):v for k,v in sorted(edge_hist.items())}!=config['inventory']['eligible_edge_histogram']:
        raise ValueError('independent eligible-edge inventory differs')
    if tokenizer_factory is None:
        import os
        os.environ['TOKENIZERS_PARALLELISM']='false'
        from transformers import AutoTokenizer
        tokenizer_factory=lambda p:AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
    masks,questions,computed=reconstruct(cases,tokenizer_factory(config['tokenizer']),deadline)
    if len(masks)!=501:raise ValueError('independent complete mask count differs')
    if ([json.loads(line) for line in buffers[str(run/'per_mask.jsonl')].splitlines() if line.strip()]!=masks
        or [json.loads(line) for line in buffers[str(run/'per_question.jsonl')].splitlines() if line.strip()]!=questions):
        raise ValueError('independent mask/step/identity/token reconstruction differs')
    if any(public.get(k)!=v for k,v in computed.items()):raise ValueError('independent aggregate reconstruction differs')
    for k in ('specification','inventory','runtime','limits'):
        if public[k]!=public_protocol[k]:raise ValueError('public protocol metadata differs')
    if type(public['api_calls']) is not int or public['api_calls']!=0:raise ValueError('invalid API metadata')
    for key in ('quality_metrics_computed','best_mask_selected','gpu_used','encoder_model_loaded','qa_sidecar_opened','private_ids_in_public_output'):
        if public[key] is not False:raise ValueError('invalid side-effect/quality metadata')
    for path,expected in hashes.items():
        check(deadline)
        if sha(path)!=expected:raise ValueError('input/output changed during independent verification')
    if {p.name for p in plan.iterdir()}!={'plan.json','cases.json','seal.json'} or {p.name for p in run.iterdir()}!=RUN_FILES:
        raise ValueError('plan/run inventory changed during verification')
    result={'schema':'slac-qasper-relation-opportunity-independent-verification-v1','status':'verified_complete',
        'questions':77,'families':24,'masks':501,'every_selected_identity_pack_token_and_step_recomputed':True,
        'all_behavioral_aggregate_fields_recomputed':True,'official_selector_or_aggregator_imported':False,
        'qa_or_answers_read':False,'quality_computed':False,'api_calls':0,
        'elapsed_seconds':time.monotonic()-started,'plan_sha256':PLAN_SHA,
        'public_protocol_sha256':PROTOCOL_SHA,'input_output_sha256':hashes}
    check(deadline);output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_bytes(encode(result)+b'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','run','output'):parser.add_argument('--'+n,required=True)
    parser.add_argument('--completion',default=str(COMPLETION))
    result=verify(parser.parse_args())
    print(json.dumps({k:result[k] for k in ('status','questions','families','masks','quality_computed','api_calls')},indent=2))
