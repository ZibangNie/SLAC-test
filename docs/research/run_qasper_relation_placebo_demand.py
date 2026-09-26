"""Pure CPU original-domain placebo composition and optional fee projection.

No label/API execution, QA or answer reading. Prepared containers are projected
immediately to static tasks and selection metadata; old query fields are unused.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import math
from pathlib import Path
import platform
import time

import run_qasper_relation_demand_compilation as io

ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/research-foundation'
SCHEMA='slac-qasper-relation-placebo-demand-v1'
PROTOCOL_SHA='38f8d45fcfe1c16af71ff0ec2e3c36ed7d19b8af776c7b9831a978bd6b976ab0'
RECEIPT_SHA='164199b78cdb3475693e99ec42f5612d374507ce3f0a5f2a46873b9410897c4c'
PATHS={'receipt':ART/'qasper-relation-demand-independent-verification-01/verification.json',
    'demand_plan':ART/'qasper-relation-demand-plan-01/plan.json','cubes':ART/'qasper-relation-demand-plan-01/cubes.json',
    'demand_seal':ART/'qasper-relation-demand-plan-01/seal.json',
    'demand_summary':ART/'qasper-relation-demand-run-01/summary.json','demand_public':ART/'qasper-relation-demand-run-01/public_aggregate.json',
    'gate_cases':ART/'qasper-relation-opportunity-plan-01/cases.json','gate_masks':ART/'qasper-relation-opportunity-run-01/per_mask.jsonl',
    'prepared':ART/'qasper-extended-development-prepared-01/prepared.json',
    'client':ROOT/'docs/research/openrouter_decision_client.py','helper':Path(io.__file__).resolve()}
ANCHORS={'receipt':RECEIPT_SHA,'prepared':'b3bc8a3272d2898246aa7295aba7b366881368188c3a008b0f456e4a838cc885',
         'client':'1400d8fd16d65da1640cd0bb3ff3bb4e7ea4e3a80cb75356f0be2327bf1c97ba'}
DIRECTORIES={PATHS['demand_plan'].parent:{'plan.json','cubes.json','seal.json'},
             PATHS['demand_summary'].parent:{'decisions.jsonl','per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'},
             PATHS['gate_cases'].parent:{'plan.json','cases.json','seal.json'},
             PATHS['gate_masks'].parent:{'per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'},
             PATHS['prepared'].parent:{'prepared.json','manifest.json'}}
SPEC={'questions':77,'families':24,'masks':501,'eligible_edge_occurrences':107,'eligible_unique_edges':101,
      'static_tasks':562,'content_essential_occurrences':19,'content_essential_unique':19,'max_required_edges':38,
      'seed':20260927,'permutation_version':'SLAC-local-dependency-placebo-v1','max_seconds':300,
      'domain':'entire original E-star per query, before demand reduction',
      'stratum':['A.kind','B.kind','support[A]','support[B]','zero-mask baseline_trigger_class'],
      'orientation':'placebo[target_order[i]]=original[source_order[i]]; g(target)=source',
      'primary':['Rcontent-R0','Rcontent-Radjacent','Rcontent-Rplacebo'],
      'secondary':['Rcontent-dense_k3','Rcontent-BGE_reranker_k3','Rcontent-p_yes_only_k3'],
      'future_metrics':['official Answer F1','actual evidence tokens'],'future_weights':['question_weighted','family_balanced'],
      'future_bootstrap_draws':10000,'future_exploratory_intervals':24,'api_calls':0,'quality_computed':False,
      'stop_if_all_functions_equal':True,'budget_estimation_only_if_nondegenerate':True,'batching':'singleton original static prompt'}
LIMITS=[
    'This is a newly frozen exposed-development coverage contract; the original 562-label paid stage stays closed.',
    'Permutation uses the entire original eligible strata, never a reshuffle of the demanded subset.',
    'Only symbolic outputs are evaluated; no relation labels, QA, generated answers or quality values are read.',
    'Unobserved labels remain unobserved; complete label counts or classifier quality cannot be inferred from U.',
    'Full-domain stratum bijection preserves counts for every hypothetical completion, not observed label statistics.',
    'Structural content/placebo equality stops semantic-content payment and admission-oriented payload estimation.',
    'Singleton labels need not equal labels from different batch contexts or stochastic model realizations.',
    'Any reserve projection is not admission; the complete night ledger including new 14-answer commitments and unknown costs must be checked separately.',
    'No fees are scaled proportionally from the old 562-edge batch plan; no original budget is reset or refunded.',
    'Exact output equality does not imply trace equality, quality gains, new BDD theory or cross-stage sharing.',
    'Only consumed files are rehashed; unused ancestor commitments are inherited.',
    'The 300-second serial CPU deadline is cooperative, not an OS hard kill.',
]
RUN_FILES={'per_question.jsonl','per_mask.jsonl','required_edges.json','static_payloads.json','public_aggregate.json','summary.json'}


def own_paths():return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_placebo_demand.py',ROOT/'docs/research/RELATION_PLACEBO_DEMAND_PROTOCOL_20260927.md']
def check_dirs():
    for p,names in DIRECTORIES.items():
        if {x.name for x in p.iterdir()}!=names:raise ValueError('source directory inventory differs')


def identity(obj):return (obj['doc_id'],obj['question_id'])


def project(cubes,cases,masks,prepared):
    """Metadata-only projection: no P, influence, composition or fee calculation."""
    case_by={identity(c):c for c in cases};zero={}
    if len(case_by)!=len(cases):raise ValueError('duplicate gate source query')
    for row in masks:
        if row['mask']==0:
            if identity(row) in zero:raise ValueError('duplicate zero-mask trace')
            zero[identity(row)]=row
    if set(zero)!=set(case_by) or set(map(identity,cubes))!=set(case_by):raise ValueError('incomplete source query coverage')
    tasks=[];task_keys=set();units_by_doc={}
    for case in cases:
        units=case['units'];doc=case['doc_id'];by_id={u['unit_id']:u for u in units}
        if len(by_id)!=len(units) or any(u['order']!=i for i,u in enumerate(units)):raise ValueError('source unit order/identity differs')
        if doc in units_by_doc and units_by_doc[doc]!=by_id:raise ValueError('inconsistent shared document text')
        units_by_doc[doc]=by_id
    for task in prepared['static_tasks']:
        if set(task)!={'id','doc_id','left_id','right_id','item'}:raise ValueError('static task whitelist differs')
        key=(task['doc_id'],task['left_id'],task['right_id'])
        if key in task_keys:raise ValueError('duplicate static task edge')
        task_keys.add(key);doc,a,b=key;us=units_by_doc.get(doc,{})
        if (a not in us or b not in us or us[b]['order']!=us[a]['order']+1
            or task['item']!={'unit_a':{'id':a,'text':us[a]['text']},'unit_b':{'id':b,'text':us[b]['text']}}):raise ValueError('static prompt source differs')
        tasks.append(task)
    queries=[]
    for cube in cubes:
        io.validate_cube(cube);case=case_by[identity(cube)];z=zero[identity(cube)]
        if case['family_id']!=cube['family_id'] or z['family_id']!=cube['family_id']:raise ValueError('source family differs')
        labels=dict(zip(case['candidates'],case['labels']));units=case['units'];by_id={u['unit_id']:i for i,u in enumerate(units)}
        accepted=[];trace=[]
        for step in z['trace']:
            item={k:step[k] for k in ('candidate_index','selected_before','accepted','action')}
            if item['selected_before']!=sorted(accepted) or type(item['accepted']) is not bool:raise ValueError('zero trace selected prefix differs')
            if item['accepted']:
                if item['action']!='select' or item['candidate_index'] in accepted:raise ValueError('zero trace success differs')
                accepted.append(item['candidate_index'])
            trace.append(item)
        if sorted(accepted)!=z['selected_indices'] or z['pack_sha256']!=cube['outcomes'][0]['pack_sha256']:raise ValueError('zero trace witness differs')
        metadata=[]
        for edge in cube['edges']:
            doc,a,b=edge
            if tuple(edge) not in task_keys or doc!=cube['doc_id'] or a not in by_id or b not in by_id:raise ValueError('unknown source boundary')
            ia,ib=by_id[a],by_id[b]
            if ib!=ia+1 or labels.get(ia) not in ('yes','unknown') or labels.get(ib) not in ('yes','unknown'):raise ValueError('ineligible source boundary')
            ordinal=0;trigger=0
            for step in trace:
                if step['accepted']:
                    ordinal+=1
                    if step['candidate_index']==ib and ordinal in (1,2) and ia not in step['selected_before']:trigger=ordinal
            metadata.append({'edge':edge,'order':[ia,ib],'stratum':[units[ia]['kind'],units[ib]['kind'],labels[ia],labels[ib],trigger]})
        if [x['order'] for x in metadata]!=sorted(x['order'] for x in metadata):raise ValueError('native edge order differs')
        queries.append({'cube':cube,'edge_metadata':metadata})
    if (len(queries)!=77 or len({q['cube']['family_id'] for q in queries})!=24 or sum(len(q['cube']['outcomes']) for q in queries)!=501
        or sum(len(q['edge_metadata']) for q in queries)!=107 or len({tuple(e['edge']) for q in queries for e in q['edge_metadata']})!=101
        or len(tasks)!=562):raise ValueError('fixed metadata inventory differs')
    return {'queries':queries,'static_tasks':tasks}


def snapshot(deadline=math.inf):
    check_dirs();blobs={k:p.read_bytes() for k,p in PATHS.items()};hashes={str(PATHS[k]):hashlib.sha256(b).hexdigest() for k,b in blobs.items()}
    for k,h in ANCHORS.items():
        if hashes[str(PATHS[k])]!=h:raise ValueError('fixed external source anchor differs: '+k)
    receipt=io.parse(blobs['receipt']);commitment=receipt['input_output_sha256']
    if (receipt.get('status')!='verified_complete' or receipt.get('questions')!=77 or receipt.get('families')!=24 or receipt.get('masks')!=501
        or receipt.get('partial_cache_subcubes')!=4075 or receipt.get('assignment_known_subset_paths')!=23915
        or receipt.get('all_essential_derivatives_dag_outputs_depths_and_aggregates_equal') is not True
        or receipt.get('all_cache_restrictions_recomputed_via_ordered_cofactor_vectors') is not True
        or receipt.get('quality_computed') is not False or receipt.get('qa_or_answers_read') is not False):raise ValueError('completed independent demand audit required')
    for k,p in PATHS.items():
        if k not in ANCHORS and commitment.get(str(p))!=hashes[str(p)]:raise ValueError('source not bound by completed receipt')
    plan=io.parse(blobs['demand_plan']);summary=io.parse(blobs['demand_summary']);public=io.parse(blobs['demand_public'])
    if (io.parse(blobs['demand_seal'])!={'plan.json':hashes[str(PATHS['demand_plan'])],'cubes.json':hashes[str(PATHS['cubes'])]}
        or summary.get('status')!='completed' or summary.get('all_cubes_compiled_and_verified') is not True
        or summary['output_sha256'].get('public_aggregate.json')!=hashes[str(PATHS['demand_public'])]
        or public.get('essential_edge_query_occurrences')!=19 or public.get('essential_static_edge_union_count')!=19):raise ValueError('completed source seals/counts differ')
    prepared=io.parse(blobs['prepared'])
    if prepared.get('schema')!='slac-qasper-extended-development-prepared-v1':raise ValueError('prepared version differs')
    data=project(io.parse(blobs['cubes']),io.parse(blobs['gate_cases']),[io.parse(l) for l in blobs['gate_masks'].splitlines() if l.strip()],prepared)
    for p in own_paths():hashes[str(p)]=io.digest(p,deadline)
    if hashes[str(own_paths()[2])]!=PROTOCOL_SHA:raise ValueError('pre-answer frozen protocol changed')
    meta={'independent_demand_receipt_sha256':RECEIPT_SHA,'inherited_commitment_count':len(commitment),
          'inherited_commitment_sha256':io.object_hash(commitment),'unused_ancestors_rehashed':False,
          'source_demand_input_binding_sha256':plan['input_binding_sha256'],'original_static_stage_executed':False,
          'old_query_container_fields_projected_away':True}
    io.verify_hashes(hashes,deadline);check_dirs();io.check(deadline);return data,meta,hashes


def permutation(query):
    cube=query['cube'];meta=query['edge_metadata'];groups=defaultdict(list)
    if len(meta)!=len(cube['edges']):raise ValueError('edge metadata incomplete')
    for i,m in enumerate(meta):
        if set(m)!={'edge','order','stratum'} or m['edge']!=cube['edges'][i] or len(m['stratum'])!=5:raise ValueError('edge metadata differs')
        groups[io.canonical(m['stratum'])].append(i)
    g=list(range(len(meta)));strata=[]
    for key in sorted(groups):
        source=sorted(groups[key],key=lambda i:tuple(meta[i]['order']))
        def order(i):
            token=[SPEC['permutation_version'],SPEC['seed'],cube['family_id'],cube['question_id'],meta[i]['stratum'],meta[i]['edge']]
            return hashlib.sha256(io.canonical(token)).digest(),tuple(meta[i]['order'])
        target=sorted(source,key=order)
        for s,t in zip(source,target):g[t]=s
        strata.append({'stratum':io.parse(key),'source_order':source,'target_order':target})
    if sorted(g)!=list(range(len(meta))) or any(meta[t]['stratum']!=meta[s]['stratum'] for t,s in enumerate(g)):raise ValueError('invalid full-domain stratum bijection')
    return g,strata


def apply_permutation(mask,g):
    if type(mask) is not int or not 0<=mask<2**len(g) or any(type(i) is not int for i in g) or sorted(g)!=list(range(len(g))):raise ValueError('invalid assignment/permutation')
    return sum(((mask>>source)&1)<<target for target,source in enumerate(g))


def influences(values,n):
    keys=list(map(io.canonical,values))
    if len(keys)!=2**n:raise ValueError('incomplete function cube')
    return [bit for bit in range(n) if any(keys[m]!=keys[m^(1<<bit)] for m in range(2**n) if not m>>bit&1)]


def inspect_query(query,deadline=math.inf):
    cube=query['cube'];io.validate_cube(cube);n=len(cube['edges']);g,strata=permutation(query)
    content=cube['outcomes'];composed=[];rows=[];s=influences(content,n)
    for mask in range(2**n):
        io.check(deadline);permuted=apply_permutation(mask,g)
        # Independent source->target dictionary, separate from bit-shift implementation.
        target_bits={target:bool(mask&(1<<source)) for group in strata for source,target in zip(group['source_order'],group['target_order'])}
        if sum((1<<t) for t,b in target_bits.items() if b)!=permuted:raise ValueError('composition orientation parity failed')
        for group in strata:
            if sum((mask>>i)&1 for i in group['source_order'])!=sum((permuted>>i)&1 for i in group['target_order']):raise ValueError('stratum count preservation failed')
        value=content[permuted];composed.append(value)
        rows.append({'mask':mask,'permuted_mask':permuted,'content_output_sha256':io.object_hash(content[mask]),
                     'placebo_output_sha256':io.object_hash(value),'outputs_equal':content[mask]==value})
    placebo_s=influences(composed,n);inverse=sorted(g[t] for t in s)
    if placebo_s!=inverse:raise ValueError('essential inverse image differs')
    required=sorted(set(s)|set(inverse));completion_groups={}
    for mask,row in enumerate(rows):
        key=tuple((mask>>i)&1 for i in required);value=(io.canonical(content[mask]),io.canonical(composed[mask]))
        if key in completion_groups and completion_groups[key]!=value:raise ValueError('unobserved completion changed demanded outputs')
        completion_groups[key]=value
    result={k:cube[k] for k in ('family_id','doc_id','question_id')}
    result.update(edges=cube['edges'],source_by_target=g,strata=strata,content_essential_indices=s,placebo_essential_indices=placebo_s,
        required_indices=required,mask_count=len(rows),all_composition_and_inverse_parity=True,
        completion_groups_checked=len(completion_groups),all_unobserved_completions_equal=True,
        function_equal=all(r['outputs_equal'] for r in rows),different_output_assignments=sum(not r['outputs_equal'] for r in rows),
        moved_positions=sum(i!=v for i,v in enumerate(g)))
    return result,rows


def singleton_budget(required,tasks):
    """Metadata only. No client construction, credentials or network transport."""
    import openrouter_decision_client as client
    lookup={(t['doc_id'],t['left_id'],t['right_id']):t for t in tasks};jobs=[];reserve=Decimal(0);inputs=outputs=0
    if len(lookup)!=len(tasks) or len(required)>38:raise ValueError('static task coverage/cap differs')
    for edge in required:
        if tuple(edge) not in lookup:raise ValueError('required static source task missing')
        task=lookup[tuple(edge)];payload=client.make_payload([task],'static','jev');r,i,o=client.reservation(payload,'jev')
        jobs.append({'edge':edge,'task_id':task['id'],'endpoint':client.MODELS['jev']['endpoint'],'payload':payload,
                     'payload_sha256':client.object_hash(payload),'reservation_usd':str(r),'input_allowance':i,'output_allowance':o})
        reserve+=r;inputs+=i;outputs+=o
    public={'status':'singleton_metadata_estimate_only_not_admitted','singleton_requests':len(jobs),'items':len(jobs),
        'reservation_usd':str(reserve),'input_allowance':inputs,'output_allowance':outputs,
        'max_payload_bytes':max((len(client.canonical_bytes(j['payload'])) for j in jobs),default=0),
        'prompt_version':client.PROMPT_VERSION,'route':client.MODELS['jev'],'byte_cap':client.BYTE_CAP,
        'historical_night_reservation_before_new_14_answers_usd':'4.6053728675',
        'historical_headroom_before_new_14_answers_usd':'0.3946271325','current_global_headroom_evaluated':False,
        'future_answer_reservation_evaluated':False,'unknown_prior_cost_attempts_retained':1,'paid_admitted':False}
    return jobs,public


def evaluate(data,deadline=math.inf,budget_fn=None):
    questions=[];rows=[];required=set();content_union=set();placebo_union=set();eligible_union=set();strata_count=identity_strata=0
    for query in data['queries']:
        q,local=inspect_query(query,deadline);questions.append(q);ident={k:q[k] for k in ('family_id','doc_id','question_id')}
        rows.extend({**ident,**r} for r in local);edges=q['edges'];eligible_union.update(map(tuple,edges))
        content_union.update(tuple(edges[i]) for i in q['content_essential_indices']);placebo_union.update(tuple(edges[i]) for i in q['placebo_essential_indices']);required.update(tuple(edges[i]) for i in q['required_indices'])
        strata_count+=len(q['strata']);identity_strata+=sum(s['source_order']==s['target_order'] for s in q['strata'])
    if (len(questions)!=77 or len({q['family_id'] for q in questions})!=24 or len(rows)!=501 or len(eligible_union)!=101
        or sum(len(q['content_essential_indices']) for q in questions)!=19 or len(content_union)!=19 or not 19<=len(required)<=38):raise ValueError('fixed full-population demand differs')
    ordered=[list(e) for e in sorted(required)];degenerate=all(q['function_equal'] for q in questions)
    io.check(deadline)
    if degenerate:
        jobs=[];budget={'status':'not_estimated_structurally_degenerate','singleton_requests':0,'paid_admitted':False}
    else:jobs,budget=(budget_fn or singleton_budget)(ordered,data['static_tasks'])
    io.check(deadline)
    public={'schema':SCHEMA+'-public','status':'complete_symbolic_composition','specification':SPEC,'limits':LIMITS,
        'questions':77,'families':24,'masks':501,'eligible_unique_edges':len(eligible_union),'eligible_edge_occurrences':107,
        'content_essential_unique':len(content_union),'content_essential_occurrences':19,
        'placebo_essential_unique':len(placebo_union),'placebo_essential_occurrences':sum(len(q['placebo_essential_indices']) for q in questions),
        'required_unique_edges':len(required),'required_edge_query_occurrences':sum(len(q['required_indices']) for q in questions),
        'additional_unique_edges_for_placebo':len(required-content_union),'strata_count':strata_count,
        'identity_strata':identity_strata,'nonidentity_strata':strata_count-identity_strata,
        'moved_eligible_positions':sum(q['moved_positions'] for q in questions),
        'fixed_eligible_positions':107-sum(q['moved_positions'] for q in questions),
        'identity_permutation_questions':sum(not q['moved_positions'] for q in questions),
        'function_equal_questions':sum(q['function_equal'] for q in questions),'function_different_questions':sum(not q['function_equal'] for q in questions),
        'different_output_assignments':sum(not r['outputs_equal'] for r in rows),'equal_output_assignments':sum(r['outputs_equal'] for r in rows),
        'required_count_per_question_histogram':io.hist(len(q['required_indices']) for q in questions),
        'all_501_composition_parity':True,'all_77_inverse_image_parity':True,'all_unobserved_completion_invariance':True,
        'all_controls_structurally_degenerate':degenerate,'semantic_content_paid_phase_must_stop':degenerate,
        'budget_projection':budget,'api_calls':0,'key_read':False,'qa_or_answers_read':False,'quality_computed':False,
        'labels_observed':0,'full_label_counts_estimated':False,'original_static_stage_executed':False,'paid_admitted':False,
        'private_ids_in_public_output':False}
    return questions,rows,ordered,jobs,public


def runtime():return {'python':platform.python_version(),'execution':'serial CPU symbolic computation; no model/tokenizer'}
def validate_data(data):
    if set(data)!={'queries','static_tasks'} or len(data['queries'])!=77 or len(data['static_tasks'])!=562:raise ValueError('input metadata shape/count differs')
    cubes=[q['cube'] for q in data['queries']];io.inventory(cubes)
    for q in data['queries']:
        if set(q)!={'cube','edge_metadata'} or len(q['edge_metadata'])!=len(q['cube']['edges']):raise ValueError('query metadata shape differs')


def prepare(args):
    output=Path(args.output).resolve();run=Path(args.run_output).resolve()
    if output.exists() or run.exists():raise FileExistsError('plan/run must be new single-use directories')
    if output.is_relative_to(run) or run.is_relative_to(output) or any(new.is_relative_to(p.parent) or p.parent.is_relative_to(new) for new in (output,run) for p in PATHS.values()):raise ValueError('outputs overlap sources or each other')
    data,meta,hashes=snapshot();validate_data(data)
    config={'schema':SCHEMA,'status':'prepared_not_composed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'specification':SPEC,'limits':LIMITS,'runtime':runtime(),'source_metadata':meta,'input_sha256':hashes,'input_binding_sha256':io.object_hash(hashes),
        'data_object_sha256':io.object_hash(data),'run_output':str(run),'real_permutation_demand_or_budget_analyzed':False,'api_calls':0,'quality_computed':False}
    output.mkdir(parents=True,exist_ok=False);io.write(output/'plan.json',config);io.write(output/'inputs.json',data)
    io.write(output/'seal.json',{n:io.digest(output/n) for n in ('plan.json','inputs.json')});io.verify_hashes(hashes);check_dirs();return config


def load_plan(path,deadline=math.inf):
    path=Path(path).resolve()
    if {p.name for p in path.iterdir()}!={'plan.json','inputs.json','seal.json'}:raise ValueError('plan inventory differs')
    blobs={n:(path/n).read_bytes() for n in ('plan.json','inputs.json','seal.json')};own={str(path/n):hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    config=io.parse(blobs['plan.json']);data=io.parse(blobs['inputs.json'])
    if (io.parse(blobs['seal.json'])!={n:own[str(path/n)] for n in ('plan.json','inputs.json')} or config.get('schema')!=SCHEMA
        or config.get('status')!='prepared_not_composed' or config.get('specification')!=SPEC or config.get('limits')!=LIMITS or config.get('runtime')!=runtime()
        or config.get('real_permutation_demand_or_budget_analyzed') is not False or config.get('api_calls')!=0 or config.get('quality_computed') is not False
        or config.get('data_object_sha256')!=io.object_hash(data) or config.get('input_binding_sha256')!=io.object_hash(config['input_sha256'])):raise ValueError('frozen plan contract differs')
    fresh,meta,hashes=snapshot(deadline)
    if fresh!=data or meta!=config['source_metadata'] or hashes!=config['input_sha256']:raise ValueError('source reconstruction differs')
    validate_data(data);io.verify_hashes(own,deadline);return config,data,own


def public_metadata(public,config,timing):return {**public,'runtime':config['runtime'],'execution_timing':timing,'input_binding_sha256':config['input_binding_sha256'],'frozen_protocol_sha256':PROTOCOL_SHA,'source_independent_receipt_sha256':RECEIPT_SHA}


def summary_for(config,own,hashes,timing):return {'schema':SCHEMA,'status':'completed','all_symbolic_results_available':True,
    'specification':SPEC,'limits':LIMITS,'runtime':config['runtime'],'execution_timing':timing,'source_metadata':config['source_metadata'],
    'input_sha256':config['input_sha256'],'input_binding_sha256':config['input_binding_sha256'],'plan_sha256':own,'output_sha256':hashes,
    'api_calls':0,'key_read':False,'qa_or_answers_read':False,'quality_computed':False,'paid_admitted':False}


def run(args):
    start=time.monotonic();deadline=start+300;config,data,own=load_plan(args.plan,deadline);output=Path(config['run_output']);output.mkdir(parents=True,exist_ok=False)
    phase='symbolic_composition'
    try:
        loaded=time.monotonic();questions,rows,required,jobs,public=evaluate(data,deadline);evaluated=time.monotonic()
        phase='final_source_verification';io.verify_hashes({**config['input_sha256'],**own},deadline);check_dirs();phase='output_writing'
        io.write_rows(output/'per_question.jsonl',questions,deadline);io.write_rows(output/'per_mask.jsonl',rows,deadline)
        io.write(output/'required_edges.json',required);io.write(output/'static_payloads.json',jobs)
        timing={'load_validation_seconds':loaded-start,'compilation_verification_seconds':evaluated-loaded,'post_compilation_seconds_before_summary':time.monotonic()-evaluated};timing['total_seconds_before_summary']=sum(timing.values())
        io.validate_timing(timing);io.write(output/'public_aggregate.json',public_metadata(public,config,timing))
        hashes={n:io.digest(output/n,deadline) for n in RUN_FILES-{'summary.json'}};io.check(deadline);io.write(output/'summary.json',summary_for(config,own,hashes,timing))
        io.verify_hashes({**config['input_sha256'],**own},deadline);check_dirs();io.check(deadline)
        return {'status':'completed','api_calls':0,'paid_admitted':False}
    except BaseException as exc:
        if (output/'summary.json').exists():(output/'summary.json').unlink()
        io.write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_result','phase':phase,'error_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-start,'api_calls':0});raise


def audit(args):
    deadline=time.monotonic()+300;config,data,own=load_plan(args.plan,deadline);output=Path(args.run).resolve()
    if str(output)!=config['run_output'] or {p.name for p in output.iterdir()}!=RUN_FILES:raise ValueError('complete fixed-run inventory required')
    blobs={n:(output/n).read_bytes() for n in RUN_FILES};hashes={str(output/n):hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    summary=io.parse(blobs['summary.json']);timing=summary['execution_timing'];io.validate_timing(timing)
    if summary!=summary_for(config,own,{n:hashes[str(output/n)] for n in RUN_FILES-{'summary.json'}},timing):raise ValueError('complete summary metadata differs')
    questions,rows,required,jobs,public=evaluate(data,deadline)
    for n,value in [('per_question.jsonl',questions),('per_mask.jsonl',rows)]:
        if [io.parse(l) for l in blobs[n].splitlines() if l.strip()]!=value:raise ValueError('symbolic records differ')
    if io.parse(blobs['required_edges.json'])!=required or io.parse(blobs['static_payloads.json'])!=jobs or io.parse(blobs['public_aggregate.json'])!=public_metadata(public,config,timing):raise ValueError('demand/budget/public differs')
    io.verify_hashes({**config['input_sha256'],**own,**hashes},deadline);check_dirs();io.check(deadline)
    if {p.name for p in output.iterdir()}!=RUN_FILES:raise ValueError('run inventory changed')
    return {'schema':SCHEMA+'-audit','status':'verified_complete','questions':77,'masks':501,'all_original_domain_compositions_and_inverse_images_recomputed':True,
        'all_completion_invariance_and_stop_gates_recomputed':True,'all_direct_hashes_unchanged':True,'input_binding_sha256':config['input_binding_sha256'],
        'run_sha256':hashes,'api_calls':0,'key_read':False,'qa_or_answers_read':False,'quality_computed':False,'paid_admitted':False,'timing_independently_proven':False}


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--run-output',required=True)
    p=subs.add_parser('run');p.add_argument('--plan',required=True)
    p=subs.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    args=parser.parse_args();result={'prepare':prepare,'run':run,'audit':audit}[args.command](args)
    print(json.dumps(result if args.command=='audit' else {k:result[k] for k in ('status','api_calls')},indent=2))
