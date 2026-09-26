"""Compile audited finite relation cubes into exact-output decision DAGs.

Stdlib only. Prepare validates metadata and complete cubes without influence or
depth analysis. Run/audit require a separately sealed plan. No label oracle is
called in a real run; all hypothetical per-query assignments are retained.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/'artifacts/research-foundation'
SCHEMA='slac-qasper-relation-demand-compilation-v1'
RECEIPT_SHA='ebf959058e67a2ea3285f70c985f0a94bda2d6912cd6eab24d4de1a33037e32c'
SOURCE_PATHS={
    'receipt':ARTIFACTS/'qasper-relation-opportunity-independent-verification-01/verification.json',
    'plan':ARTIFACTS/'qasper-relation-opportunity-plan-01/plan.json',
    'cases':ARTIFACTS/'qasper-relation-opportunity-plan-01/cases.json',
    'seal':ARTIFACTS/'qasper-relation-opportunity-plan-01/seal.json',
    **{n:ARTIFACTS/'qasper-relation-opportunity-run-01'/n for n in ('per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json')},
    'protocol':ROOT/'docs/research/results/qasper_relation_opportunity_protocol_20260927.json',
}
SOURCE_INVENTORIES={
    SOURCE_PATHS['receipt'].parent:{'verification.json'},
    SOURCE_PATHS['plan'].parent:{'plan.json','cases.json','seal.json'},
    SOURCE_PATHS['summary.json'].parent:{'per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'},
}
SPEC={'questions':77,'families':24,'masks':501,'eligible_edge_occurrences':107,
      'partial_cache_subcubes':4075,'assignment_known_subset_paths':23915,
      'eligible_edge_histogram':{'0':31,'1':19,'2':8,'3':10,'4':6,'5':1,'6':1,'7':1},
      'original_static_edges':562,'max_edges_per_query':7,'max_seconds':300,
      'terminal_identity':'ordered source-qualified selected identities + exact rendered pack + SHA256 + actual tokens',
      'variable_order':'original native boundary order, first conditional-essential variable',
      'essential':'exists full-cube output difference when only this bit is flipped',
      'early_stop':'all outputs in current subcube exactly equal',
      'reduction':'intern equal full terminals and equal ordered decision triples',
      'order_optimization':False,'trace_equivalence':False,'api_calls':0,'quality_computed':False,
      'real_global_label_assignment_selected':False,'global_cache_schedule_executed':False}
LIMITS=[
    'Exploratory compilation after observing a development opportunity gate; not independent confirmation.',
    'Exact output/pack equivalence is stronger than matching text alone, and does not imply trace equivalence.',
    'Equivalence assumes the same fixed binary relation oracle; online batching and stochastic models can return different labels.',
    'Each query cube covers all assignments; per-query assignments need not jointly describe one global document labeling.',
    'Global edge identities are shared, but no real global labeling or cross-query cache schedule is chosen here.',
    'The essential static-edge union and sum of separate query worst depths are safe request upper bounds, not an exact shared-cache worst case.',
    'The 501-assignment depth histogram is combinatorial; it is not an expected request count or a label probability distribution.',
    'No API payload, provider fee estimate, proportional cost extrapolation, answer, QA or quality metric is computed.',
    'Fixed candidate/support/packer cubes do not establish retrieval, relation accuracy, downstream quality or cross-stage sharing benefit.',
    'Finite truth-table compilation costs grow exponentially with eligible variables; the observed maximum is seven.',
    'This applies established ordered finite-terminal decision diagrams; no new BDD algorithm is claimed.',
    'Content-essential edges do not preserve the original full-domain placebo label counts or its permuted outputs.',
    'Only directly consumed files are rehashed; unused ancestor commitments remain inherited provenance.',
    'The serial CPU deadline is cooperative between bounded operations; it is not an OS hard kill.',
]
CUBE_KEYS={'family_id','doc_id','question_id','edges','outcomes'}
OUTCOME_KEYS={'selected_identities','rendered_pack','pack_sha256','actual_tokens'}
RUN_FILES={'decisions.jsonl','per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'}


def check(deadline):
    if time.monotonic()>deadline:raise TimeoutError('demand compilation deadline exceeded')


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def object_hash(value):return hashlib.sha256(canonical(value)).hexdigest()


def unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON key')
        result[k]=v
    return result


def parse(blob):return json.loads(blob,object_pairs_hook=unique)


def digest(path,deadline=math.inf):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):check(deadline);h.update(chunk)
    check(deadline);return h.hexdigest()


def write(path,obj):
    with Path(path).open('xb') as f:f.write(canonical(obj)+b'\n')


def verify_hashes(mapping,deadline=math.inf):
    for p,h in mapping.items():
        if digest(p,deadline)!=h:raise ValueError('bound input changed: '+Path(p).name)


def verify_source_inventory():
    for p,names in SOURCE_INVENTORIES.items():
        if {x.name for x in p.iterdir()}!=names:raise ValueError('source directory inventory changed')


def own_paths():return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_demand_compilation.py',ROOT/'docs/research/RELATION_DEMAND_COMPILATION_PROTOCOL_20260927.md']


def validate_outcome(outcome,doc):
    if set(outcome)!=OUTCOME_KEYS:raise ValueError('terminal whitelist differs')
    ids=outcome['selected_identities'];text=outcome['rendered_pack'];tokens=outcome['actual_tokens']
    if (not isinstance(ids,list) or len(ids)>3 or any(not isinstance(i,list) or len(i)!=2 or i[0]!=doc or not isinstance(i[1],str) for i in ids)
        or len({tuple(i) for i in ids})!=len(ids) or not isinstance(text,str)
        or type(tokens) is not int or not 0<=tokens<=1024
        or outcome['pack_sha256']!=hashlib.sha256(text.encode('utf-8')).hexdigest()
        or (not ids and (text!='' or tokens!=0)) or (ids and (not text or tokens==0))):
        raise ValueError('invalid terminal identity/pack/token')


def validate_cube(cube):
    if set(cube)!=CUBE_KEYS:raise ValueError('cube whitelist differs')
    if any(not isinstance(cube[k],str) or not cube[k] for k in ('family_id','doc_id','question_id')):raise ValueError('invalid query identity')
    edges=cube['edges'];outcomes=cube['outcomes']
    if (not isinstance(edges,list) or len(edges)>SPEC['max_edges_per_query']
        or any(not isinstance(e,list) or len(e)!=3 or e[0]!=cube['doc_id'] or any(not isinstance(x,str) or not x for x in e) or e[1]==e[2] for e in edges)
        or len({tuple(e) for e in edges})!=len(edges) or len(outcomes)!=2**len(edges)):
        raise ValueError('invalid edge identities or incomplete cube')
    for value in outcomes:validate_outcome(value,cube['doc_id'])


def inventory(cubes):
    for cube in cubes:validate_cube(cube)
    result={'questions':len(cubes),'families':len({c['family_id'] for c in cubes}),
            'masks':sum(len(c['outcomes']) for c in cubes),'eligible_edge_occurrences':sum(len(c['edges']) for c in cubes),
            'partial_cache_subcubes':sum(3**len(c['edges']) for c in cubes),
            'assignment_known_subset_paths':sum(4**len(c['edges']) for c in cubes),
            'eligible_edge_histogram':hist(len(c['edges']) for c in cubes)}
    if len({(c['doc_id'],c['question_id']) for c in cubes})!=len(cubes):raise ValueError('duplicate query')
    if any(result[k]!=SPEC[k] for k in result):raise ValueError('fixed development inventory differs')
    return result


def project(cases,mask_rows,question_rows):
    """Validate saved identities/bytes; do not compare outputs across masks."""
    key=lambda r:(r['doc_id'],r['question_id'])
    by_case={key(c):c for c in cases};by_q={key(q):q for q in question_rows};by_mask={}
    if len(by_case)!=len(cases) or len(by_q)!=len(question_rows) or set(by_q)!=set(by_case):raise ValueError('incomplete/duplicate source query')
    for row in mask_rows:
        if key(row) not in by_case:raise ValueError('unknown source query')
        pair=(key(row),row['mask'])
        if type(row['mask']) is not int or pair in by_mask:raise ValueError('duplicate or invalid source mask')
        by_mask[pair]=row
    cubes=[]
    for case in cases:
        d=case['doc_id'];q=by_q[key(case)];us=case['units'];cand=case['candidates']
        if q['family_id']!=case['family_id'] or any(u['order']!=i for i,u in enumerate(us)) or len({u['unit_id'] for u in us})!=len(us):raise ValueError('source identity/order differs')
        eligible={i for i,l in zip(cand,case['labels']) if l!='no'}
        edges=[(i,i+1) for i in cand if i in eligible and i+1 in eligible and us[i]['native_text']!=us[i+1]['native_text']]
        if q['eligible_edges']!=[list(e) for e in edges] or q['mask_count']!=2**len(edges):raise ValueError('source edge domain differs')
        values=[]
        for mask in range(2**len(edges)):
            row=by_mask.pop((key(case),mask),None)
            if row is None or row['family_id']!=case['family_id']:raise ValueError('incomplete source cube')
            selected=row['selected_indices']
            if (any(type(i) is not int or i not in cand for i in selected) or selected!=sorted(set(selected))
                or any(i not in eligible for i in selected) or len({us[i]['native_text'] for i in selected})!=len(selected)
                or row['selected_ids']!=[us[i]['unit_id'] for i in selected]
                or row['active_edges']!=[list(e) for j,e in enumerate(edges) if mask>>j&1]):raise ValueError('unknown source selection/edge')
            pack='\n\n'.join('['+us[i]['unit_id']+']\n'+us[i]['text'] for i in selected)
            outcome={'selected_identities':[[d,us[i]['unit_id']] for i in selected],'rendered_pack':pack,
                     'pack_sha256':row['pack_sha256'],'actual_tokens':row['actual_tokens']}
            validate_outcome(outcome,d);values.append(outcome)
        cube={k:case[k] for k in ('family_id','doc_id','question_id')}
        cube.update(edges=[[d,us[a]['unit_id'],us[b]['unit_id']] for a,b in edges],outcomes=values)
        cubes.append(cube)
    if by_mask:raise ValueError('unexpected source masks')
    inventory(cubes);return cubes


def source_snapshot(deadline=math.inf):
    verify_source_inventory()
    blobs={name:p.read_bytes() for name,p in SOURCE_PATHS.items()};hashes={str(SOURCE_PATHS[n]):hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    if hashes[str(SOURCE_PATHS['receipt'])]!=RECEIPT_SHA:raise ValueError('fixed independent receipt anchor differs')
    receipt=parse(blobs['receipt']);commitments=receipt['input_output_sha256']
    if (receipt.get('status')!='verified_complete' or receipt.get('questions')!=77 or receipt.get('families')!=24 or receipt.get('masks')!=501
        or receipt.get('every_selected_identity_pack_token_and_step_recomputed') is not True
        or receipt.get('all_behavioral_aggregate_fields_recomputed') is not True
        or receipt.get('official_selector_or_aggregator_imported') is not False
        or receipt.get('qa_or_answers_read') is not False or receipt.get('quality_computed') is not False or receipt.get('api_calls')!=0):raise ValueError('complete independent verification required')
    for name,p in SOURCE_PATHS.items():
        if name!='receipt' and commitments.get(str(p))!=hashes[str(p)]:raise ValueError('direct source absent or changed in independent receipt')
    plan=parse(blobs['plan']);summary=parse(blobs['summary.json']);seal=parse(blobs['seal']);protocol=parse(blobs['protocol'])
    if (seal!={n:hashes[str(SOURCE_PATHS[k])] for n,k in [('plan.json','plan'),('cases.json','cases')]}
        or receipt['plan_sha256']!=hashes[str(SOURCE_PATHS['plan'])]
        or summary.get('status')!='completed' or summary.get('all_masks_available') is not True
        or summary['output_sha256']!={n:hashes[str(SOURCE_PATHS[n])] for n in ('per_mask.jsonl','per_question.jsonl','public_aggregate.json')}
        or plan['inventory']!=protocol['inventory'] or summary['inventory']!=plan['inventory']):raise ValueError('completed gate provenance differs')
    cubes=project(parse(blobs['cases']),[parse(x) for x in blobs['per_mask.jsonl'].splitlines() if x.strip()],
                  [parse(x) for x in blobs['per_question.jsonl'].splitlines() if x.strip()])
    for p in own_paths():hashes[str(p)]=digest(p,deadline)
    meta={'independent_receipt_sha256':RECEIPT_SHA,'inherited_direct_commitment_count':len(commitments),
          'inherited_direct_commitment_sha256':object_hash(commitments),'unused_ancestors_rehashed':False,
          'upstream_ancestor_commitment_count':plan['source_metadata']['inherited_upstream_commitment_count'],
          'upstream_ancestor_commitment_sha256':plan['source_metadata']['upstream_input_binding_sha256'],
          'source_plan_sha256':hashes[str(SOURCE_PATHS['plan'])],'source_summary_sha256':hashes[str(SOURCE_PATHS['summary.json'])]}
    verify_hashes(hashes,deadline);verify_source_inventory();check(deadline)
    return cubes,meta,hashes


def essential_indices(keys,masks,variables,deadline=math.inf):
    available=set(masks);result=[]
    for bit in variables:
        check(deadline)
        if any(not mask>>bit&1 and (mask^(1<<bit)) in available and keys[mask]!=keys[mask^(1<<bit)] for mask in masks):result.append(bit)
    return result


def compile_cube(cube,deadline=math.inf):
    validate_cube(cube);keys=[canonical(o) for o in cube['outcomes']];nodes=[];intern={}
    def add(node,signature):
        if signature not in intern:intern[signature]=len(nodes);nodes.append(node)
        return intern[signature]
    def build(masks,remaining):
        check(deadline)
        if all(keys[m]==keys[masks[0]] for m in masks):
            return add({'kind':'terminal','outcome':parse(keys[masks[0]])},('terminal',keys[masks[0]]))
        essential=essential_indices(keys,masks,remaining,deadline)
        if not essential:raise ValueError('nonconstant subcube has no variable')
        bit=essential[0];rest=[b for b in remaining if b>bit]
        zero=build([m for m in masks if not m>>bit&1],rest);one=build([m for m in masks if m>>bit&1],rest)
        if zero==one:return zero
        return add({'kind':'decision','edge_index':bit,'zero':zero,'one':one},('decision',bit,zero,one))
    root=build(list(range(len(keys))),list(range(len(cube['edges']))))
    return {'root':root,'nodes':nodes}


def validate_dag(cube,dag):
    if set(dag)!={'root','nodes'} or not isinstance(dag['nodes'],list):raise ValueError('invalid DAG schema')
    nodes=dag['nodes'];visited=set();active=set()
    def walk(i,last):
        if type(i) is not int or not 0<=i<len(nodes):raise ValueError('unknown DAG node')
        if i in active:raise ValueError('DAG cycle')
        node=nodes[i]
        if node.get('kind')=='terminal':
            if set(node)!={'kind','outcome'}:raise ValueError('invalid terminal node')
            validate_outcome(node['outcome'],cube['doc_id']);visited.add(i);return
        if (set(node)!={'kind','edge_index','zero','one'} or node.get('kind')!='decision'
            or type(node['edge_index']) is not int or not last<node['edge_index']<len(cube['edges'])
            or node['zero']==node['one']):raise ValueError('nonordered or unreduced decision')
        active.add(i);walk(node['zero'],node['edge_index']);walk(node['one'],node['edge_index']);active.remove(i);visited.add(i)
    walk(dag['root'],-1)
    if visited!=set(range(len(nodes))):raise ValueError('unreachable DAG node')


def traverse(dag,mask,edge_count):
    """Independent pointer traversal: no influence test or compilation call."""
    if type(mask) is not int or not 0<=mask<2**edge_count:raise ValueError('invalid traversal assignment')
    node_id=dag['root'];seen=set();asked=[];last=-1
    while True:
        if type(node_id) is not int or not 0<=node_id<len(dag['nodes']) or node_id in seen:raise ValueError('bad node or cycle')
        seen.add(node_id);node=dag['nodes'][node_id]
        if node.get('kind')=='terminal':return node['outcome'],asked
        bit=node.get('edge_index')
        if node.get('kind')!='decision' or type(bit) is not int or not last<bit<edge_count:raise ValueError('bad ordered decision')
        asked.append(bit);last=bit;node_id=node['one' if mask>>bit&1 else 'zero']


def cached_evaluate(cube,oracle,cache,deadline=math.inf):
    """Condition on all cached bits; the real stage uses hypothetical oracles only."""
    validate_cube(cube)
    return _cached_output(cube,[canonical(o) for o in cube['outcomes']],oracle,cache,deadline)


def _cached_output(cube,keys,oracle,cache,deadline):
    masks=list(range(len(keys)));remaining=[]
    for bit,edge in enumerate(cube['edges']):
        identity=tuple(edge)
        if identity in cache:
            if type(cache[identity]) is not bool:raise ValueError('cache requires exact Boolean values')
            masks=[m for m in masks if bool(m>>bit&1)==cache[identity]]
        else:remaining.append(bit)
    while not all(keys[m]==keys[masks[0]] for m in masks):
        check(deadline)
        bit=essential_indices(keys,masks,remaining,deadline)[0];edge=tuple(cube['edges'][bit]);value=oracle(edge)
        if type(value) is not bool:raise ValueError('fixed label oracle requires Boolean values')
        cache[edge]=value;remaining.remove(bit);masks=[m for m in masks if bool(m>>bit&1)==value]
    return cube['outcomes'][masks[0]]


def verify_all_cache_restrictions(cube,deadline=math.inf):
    """Independent callback checks every assignment/known-subset runtime path.

    The callback computes cofactors directly and never calls essential_indices.
    All oracle values are hypothetical bits of the enumerated assignment.
    """
    validate_cube(cube);keys=[canonical(o) for o in cube['outcomes']];edges=list(map(tuple,cube['edges']));n=len(edges)
    depths=[];states=set();next_edge_by_restriction={}
    for mask in range(2**n):
        for known in range(2**n):
            check(deadline)
            initial={e:bool(mask>>bit&1) for bit,e in enumerate(edges) if known>>bit&1}
            cache=dict(initial);observed=dict(initial);asked=[]
            states.add(tuple(int(initial[e]) if e in initial else -1 for e in edges))
            def oracle(edge):
                if edge not in edges or edge in observed:raise ValueError('cache runtime reread known/unknown edge')
                known_bits=sum(1<<i for i,e in enumerate(edges) if e in observed)
                value_bits=sum(1<<i for i,e in enumerate(edges) if observed.get(e,False))
                restriction=(known_bits,value_bits)
                if restriction not in next_edge_by_restriction:
                    compatible=[m for m in range(2**n) if m&known_bits==value_bits]
                    # Each candidate is checked as a complete cofactor pair, never just endpoints.
                    expected=None
                    for bit,e in enumerate(edges):
                        if e in observed:continue
                        if any(not m>>bit&1 and keys[m]!=keys[m|(1<<bit)] for m in compatible):
                            expected=e;break
                    next_edge_by_restriction[restriction]=expected
                expected=next_edge_by_restriction[restriction]
                if edge!=expected:raise ValueError('runtime did not ask first conditional-essential edge')
                bit=edges.index(edge);value=bool(mask>>bit&1);observed[edge]=value;asked.append(edge);return value
            actual=_cached_output(cube,keys,oracle,cache,deadline)
            if (actual!=cube['outcomes'][mask] or cache!=observed or any(cache[e]!=value for e,value in initial.items())
                or any(value!=bool(mask>>edges.index(e)&1) for e,value in cache.items())):raise ValueError('cached runtime output or global bit consistency differs')
            known_bits=sum(1<<i for i,e in enumerate(edges) if e in observed)
            value_bits=sum(1<<i for i,e in enumerate(edges) if observed.get(e,False))
            compatible=[m for m in range(2**n) if m&known_bits==value_bits];actual_key=canonical(actual)
            if any(keys[m]!=actual_key for m in compatible):raise ValueError('cached runtime stopped before constant subcube')
            depths.append(len(asked))
    if len(states)!=3**n or len(depths)!=4**n:raise ValueError('partial-cache completeness differs')
    return {'partial_cache_subcubes':len(states),'assignment_known_subset_paths':len(depths),'new_requests_histogram':hist(depths)}


def hist(values):return {str(k):v for k,v in sorted(Counter(values).items())}


def evaluate(cubes,deadline=math.inf):
    inv=inventory(cubes);decisions=[];rows=[];questions=[];essential_union=set();eligible_union=set();cache_hist=Counter()
    for cube in cubes:
        check(deadline);identity={k:cube[k] for k in ('family_id','doc_id','question_id')}
        keys=[canonical(o) for o in cube['outcomes']];masks=list(range(len(keys)))
        essential=essential_indices(keys,masks,range(len(cube['edges'])),deadline)
        dag=compile_cube(cube,deadline);validate_dag(cube,dag)
        cached=verify_all_cache_restrictions(cube,deadline)
        cache_hist.update({int(k):v for k,v in cached['new_requests_histogram'].items()})
        depths=[]
        for mask,expected in enumerate(cube['outcomes']):
            check(deadline);actual,asked=traverse(dag,mask,len(cube['edges']))
            if actual!=expected:raise ValueError('compiled output differs from audited eager cube')
            depths.append(len(asked));rows.append({**identity,'mask':mask,'asked_edge_indices':asked,'depth':len(asked),
                                                  'terminal_identity_sha256':object_hash(actual),'eager_output_equal':True})
        decisions.append({**identity,'edges':cube['edges'],'dag':dag})
        questions.append({**identity,'eligible_edges':len(cube['edges']),'essential_edge_indices':essential,
            'local_irrelevant_edge_indices':[i for i in range(len(cube['edges'])) if i not in essential],
            'mask_count':len(keys),'depth_min':min(depths),'depth_max':max(depths),'depth_histogram':hist(depths),
            'decision_nodes':sum(n['kind']=='decision' for n in dag['nodes']),
            'terminal_nodes':sum(n['kind']=='terminal' for n in dag['nodes']),'all_masks_equal':True,
            'cache_restriction_verification':cached})
        eligible_union.update(map(tuple,cube['edges']));essential_union.update(tuple(cube['edges'][i]) for i in essential)
    occurrence=sum(len(q['essential_edge_indices']) for q in questions);maxsum=sum(q['depth_max'] for q in questions)
    public={'schema':SCHEMA+'-public','status':'complete_exact_output_compilation','specification':SPEC,'limits':LIMITS,'inventory':inv,
        'eligible_static_edge_union_count':len(eligible_union),'essential_static_edge_union_count':len(essential_union),
        'globally_irrelevant_within_eligible_union_count':len(eligible_union-essential_union),
        'essential_edge_query_occurrences':occurrence,'local_irrelevant_edge_query_occurrences':inv['eligible_edge_occurrences']-occurrence,
        'essential_count_per_question_histogram':hist(len(q['essential_edge_indices']) for q in questions),
        'minimum_depth_per_question_histogram':hist(q['depth_min'] for q in questions),
        'maximum_depth_per_question_histogram':hist(q['depth_max'] for q in questions),
        'all_assignments_depth_histogram':hist(r['depth'] for r in rows),
        'all_partial_cache_subcubes_verified':sum(q['cache_restriction_verification']['partial_cache_subcubes'] for q in questions),
        'all_assignment_known_subset_paths_equal':sum(q['cache_restriction_verification']['assignment_known_subset_paths'] for q in questions),
        'assignment_known_subset_new_requests_histogram':{str(k):v for k,v in sorted(cache_hist.items())},
        'separate_query_worst_depth_sum_upper_bound':maxsum,'shared_cache_request_upper_bound':len(essential_union),
        'combined_safe_request_upper_bound':min(maxsum,len(essential_union)),
        'total_decision_nodes':sum(q['decision_nodes'] for q in questions),'total_terminal_nodes':sum(q['terminal_nodes'] for q in questions),
        'all_mask_output_equal_count':len(rows),'all_query_output_equal_count':len(questions),
        'trace_equivalence_claimed':False,'exact_global_cache_worst_case_computed':False,'expected_api_cost_computed':False,
        'api_calls':0,'qa_sidecar_opened':False,'quality_metrics_computed':False,'private_ids_in_public_output':False}
    return decisions,rows,questions,public


def runtime():return {'python':platform.python_version(),'implementation':'stdlib serial CPU; no tokenizer/model loaded'}


def prepare(args):
    output=Path(args.output).resolve();run_output=Path(args.run_output).resolve()
    if output.exists() or run_output.exists():raise FileExistsError('plan/run must be new single-use directories')
    forbidden=[p.parent for p in SOURCE_PATHS.values()]
    if (output.is_relative_to(run_output) or run_output.is_relative_to(output)
        or any(x.is_relative_to(p) or p.is_relative_to(x) for x in (output,run_output) for p in forbidden)):
        raise ValueError('new outputs overlap sources or one another')
    cubes,meta,hashes=source_snapshot();inv=inventory(cubes)
    config={'schema':SCHEMA,'status':'prepared_not_compiled','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'specification':SPEC,'limits':LIMITS,'inventory':inv,'runtime':runtime(),'source_metadata':meta,
        'input_sha256':hashes,'input_binding_sha256':object_hash(hashes),'cubes_object_sha256':object_hash(cubes),
        'run_output':str(run_output),'essential_or_depth_analysis_performed':False,'api_calls':0,'quality_computed':False}
    verify_hashes(hashes);verify_source_inventory();output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',config);write(output/'cubes.json',cubes)
    write(output/'seal.json',{n:digest(output/n) for n in ('plan.json','cubes.json')})
    verify_hashes(hashes);verify_source_inventory()
    return config


def load_plan(path,deadline=math.inf):
    path=Path(path).resolve()
    if {p.name for p in path.iterdir()}!={'plan.json','cubes.json','seal.json'}:raise ValueError('plan inventory differs')
    blobs={n:(path/n).read_bytes() for n in ('plan.json','cubes.json','seal.json')};own={str(path/n):hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    config=parse(blobs['plan.json']);cubes=parse(blobs['cubes.json'])
    if (parse(blobs['seal.json'])!={n:own[str(path/n)] for n in ('plan.json','cubes.json')}
        or config.get('schema')!=SCHEMA or config.get('status')!='prepared_not_compiled'
        or config.get('specification')!=SPEC or config.get('limits')!=LIMITS or config.get('runtime')!=runtime()
        or config.get('essential_or_depth_analysis_performed') is not False or config.get('api_calls')!=0 or config.get('quality_computed') is not False
        or config.get('input_binding_sha256')!=object_hash(config['input_sha256']) or config.get('cubes_object_sha256')!=object_hash(cubes)):
        raise ValueError('plan seal/contract differs')
    fresh,meta,hashes=source_snapshot(deadline)
    if (fresh!=cubes or meta!=config['source_metadata'] or hashes!=config['input_sha256'] or inventory(cubes)!=config['inventory']):raise ValueError('plan source reconstruction differs')
    verify_hashes(own,deadline);return config,cubes,own


def public_with_metadata(public,config,timing):
    return {**public,'runtime':config['runtime'],'execution_timing':timing,'input_binding_sha256':config['input_binding_sha256'],
            'source_independent_receipt_sha256':config['source_metadata']['independent_receipt_sha256']}


def summary_for(config,own,output_hashes,timing):
    return {'schema':SCHEMA,'status':'completed','all_cubes_compiled_and_verified':True,'specification':SPEC,'limits':LIMITS,
        'inventory':config['inventory'],'input_sha256':config['input_sha256'],'input_binding_sha256':config['input_binding_sha256'],
        'source_metadata':config['source_metadata'],'plan_sha256':own,'runtime':config['runtime'],'execution_timing':timing,
        'output_sha256':output_hashes,'api_calls':0,'qa_sidecar_opened':False,'quality_metrics_computed':False,
        'global_label_assignment_selected':False,'private_outputs_local_only':True}


def write_rows(path,rows,deadline):
    with Path(path).open('xb') as f:
        for row in rows:check(deadline);f.write(canonical(row)+b'\n')


def validate_timing(timing):
    names={'load_validation_seconds','compilation_verification_seconds','post_compilation_seconds_before_summary','total_seconds_before_summary'}
    if (set(timing)!=names or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in timing.values())
        or timing['total_seconds_before_summary']>SPEC['max_seconds']
        or abs(sum(timing[n] for n in names-{'total_seconds_before_summary'})-timing['total_seconds_before_summary'])>1e-8):raise ValueError('invalid measured timing')


def run(args):
    start=time.monotonic();deadline=start+SPEC['max_seconds'];config,cubes,own=load_plan(args.plan,deadline)
    output=Path(config['run_output']);output.mkdir(parents=True,exist_ok=False);phase='compilation'
    try:
        loaded=time.monotonic();decisions,rows,questions,public=evaluate(cubes,deadline);evaluated=time.monotonic()
        phase='source_verification';verify_hashes({**config['input_sha256'],**own},deadline);verify_source_inventory()
        phase='output_writing'
        for name,value in [('decisions.jsonl',decisions),('per_mask.jsonl',rows),('per_question.jsonl',questions)]:write_rows(output/name,value,deadline)
        timing={'load_validation_seconds':loaded-start,'compilation_verification_seconds':evaluated-loaded,
                'post_compilation_seconds_before_summary':time.monotonic()-evaluated};timing['total_seconds_before_summary']=sum(timing.values())
        validate_timing(timing);write(output/'public_aggregate.json',public_with_metadata(public,config,timing))
        hashes={n:digest(output/n,deadline) for n in RUN_FILES-{'summary.json'}}
        check(deadline);write(output/'summary.json',summary_for(config,own,hashes,timing))
        verify_hashes({**config['input_sha256'],**own},deadline);verify_source_inventory();check(deadline)
        return {'status':'completed','questions':len(questions),'masks':len(rows),'api_calls':0,'quality_computed':False}
    except BaseException as exc:
        if (output/'summary.json').exists():(output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_result','phase':phase,'error_type':type(exc).__name__,
                                    'elapsed_seconds':time.monotonic()-start,'api_calls':0,'quality_computed':False})
        raise


def audit(args):
    deadline=time.monotonic()+SPEC['max_seconds'];config,cubes,own=load_plan(args.plan,deadline);output=Path(args.run).resolve()
    if str(output)!=config['run_output'] or {p.name for p in output.iterdir()}!=RUN_FILES:raise ValueError('complete fixed-run inventory required')
    blobs={n:(output/n).read_bytes() for n in RUN_FILES};hashes={str(output/n):hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    summary=parse(blobs['summary.json']);timing=summary['execution_timing'];validate_timing(timing)
    if summary!=summary_for(config,own,{n:hashes[str(output/n)] for n in RUN_FILES-{'summary.json'}},timing):raise ValueError('complete summary differs')
    decisions,rows,questions,public=evaluate(cubes,deadline)
    for name,value in [('decisions.jsonl',decisions),('per_mask.jsonl',rows),('per_question.jsonl',questions)]:
        if [parse(line) for line in blobs[name].splitlines() if line.strip()]!=value:raise ValueError('saved compilation/traversal differs')
    if parse(blobs['public_aggregate.json'])!=public_with_metadata(public,config,timing):raise ValueError('saved aggregate differs')
    verify_hashes({**config['input_sha256'],**own,**hashes},deadline);verify_source_inventory();check(deadline)
    if {p.name for p in output.iterdir()}!=RUN_FILES:raise ValueError('run directory changed')
    return {'schema':SCHEMA+'-audit','status':'verified_complete','questions':len(questions),'masks':len(rows),
            'all_compilation_and_traversal_outputs_recomputed':True,'all_partial_cache_paths_recomputed':True,'all_direct_hashes_unchanged':True,
            'input_binding_sha256':config['input_binding_sha256'],'run_sha256':hashes,
            'timing_independently_proven':False,'api_calls':0,'quality_computed':False,'qa_sidecar_opened':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--run-output',required=True)
    p=subs.add_parser('run');p.add_argument('--plan',required=True)
    p=subs.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    args=parser.parse_args();result={'prepare':prepare,'run':run,'audit':audit}[args.command](args)
    print(json.dumps(result if args.command=='audit' else {k:result[k] for k in ('status','api_calls')},indent=2))
