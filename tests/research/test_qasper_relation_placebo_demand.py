"""Synthetic-only F o P, original-domain projection, gate and sealed I/O tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_placebo_demand as m


def value(doc,indices):
    text='\n\n'.join(f'[u{i}]\ntext{i}' for i in indices)
    return {'selected_identities':[[doc,f'u{i}'] for i in indices],'rendered_pack':text,
            'pack_sha256':hashlib.sha256(text.encode()).hexdigest(),'actual_tokens':len(text)+2 if text else 0}


def query(n=3,fn=lambda x:x&1):
    doc='doc';edges=[[doc,f'u{i}',f'u{i+1}'] for i in range(n)]
    cube={'family_id':'family','doc_id':doc,'question_id':'question','edges':edges,'outcomes':[value(doc,[fn(x)]) for x in range(2**n)]}
    return {'cube':cube,'edge_metadata':[{'edge':edge,'order':[i,i+1],'stratum':['paragraph','paragraph','yes','yes',0]} for i,edge in enumerate(edges)]}


def forced_rotation(query):
    n=len(query['cube']['edges']);source=list(range(n));target=source[1:]+source[:1];g=list(range(n))
    for s,t in zip(source,target):g[t]=s
    return g,([{'stratum':['paragraph','paragraph','yes','yes',0],'source_order':source,'target_order':target}] if n else [])


def identity_permutation(query):
    n=len(query['cube']['edges']);s=list(range(n))
    return s,([{'stratum':['paragraph','paragraph','yes','yes',0],'source_order':s,'target_order':s}] if n else [])


def test_nonselfinverse_three_cycle_orientation_and_essential_inverse(monkeypatch):
    monkeypatch.setattr(m,'permutation',forced_rotation)
    q,rows=m.inspect_query(query())
    assert q['source_by_target']==[2,0,1]
    assert q['content_essential_indices']==[0] and q['placebo_essential_indices']==[2]
    assert q['required_indices']==[0,2]
    assert rows[1]['permuted_mask']==2 and rows[4]['permuted_mask']==1
    assert q['all_unobserved_completions_equal']


def test_original_domain_not_essential_subset_shuffled(monkeypatch):
    monkeypatch.setattr(m,'permutation',forced_rotation)
    q,_=m.inspect_query(query(3,lambda x:(x>>1)&1))
    assert q['content_essential_indices']==[1] and q['placebo_essential_indices']==[0]
    assert len(q['source_by_target'])==3 and q['required_indices']==[0,1]


def test_seeded_strata_hash_rule_matches_independent_definition():
    q=query(5);q['edge_metadata'][4]['stratum'][-1]=1
    g,strata=m.permutation(q)
    assert g[4]==4 and len(strata)==2
    expected={}
    for s in strata:
        ids=s['source_order']
        ranked=sorted(ids,key=lambda i:(hashlib.sha256(json.dumps([
            'SLAC-local-dependency-placebo-v1',20260927,'family','question',q['edge_metadata'][i]['stratum'],q['cube']['edges'][i]],
            sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).digest(),tuple(q['edge_metadata'][i]['order'])))
        assert ranked==s['target_order']
        expected.update({t:source for source,t in zip(ids,ranked)})
    assert g==[expected[i] for i in range(5)]


def test_fixed_hash_tie_uses_native_order(monkeypatch):
    q=query(3)
    monkeypatch.setattr(m.hashlib,'sha256',lambda _:SimpleNamespace(digest=lambda:b'constant'))
    assert m.permutation(q)[0]==[0,1,2]


def test_zero_domain_and_structurally_invariant_nonidentity(monkeypatch):
    assert m.inspect_query(query(0,lambda _:0))[0]['function_equal']
    monkeypatch.setattr(m,'permutation',forced_rotation)
    q,_=m.inspect_query(query(3,lambda x:x.bit_count()%2))
    assert q['moved_positions']==3 and q['function_equal'] and q['required_indices']==[0,1,2]


def test_same_pack_hash_different_tokens_remains_essential(monkeypatch):
    q=query(2,lambda _:0);q['cube']['outcomes'][1]['actual_tokens']+=1
    monkeypatch.setattr(m,'permutation',forced_rotation)
    out,_=m.inspect_query(q)
    assert out['content_essential_indices']==[0,1] and not out['function_equal']


@pytest.mark.parametrize('mask,g',[(True,[0]),(-1,[0]),(2,[0]),(0,[0,0]),(0,[1,2]),(0,[False]),(0,[0.0])])
def test_invalid_permutation_assignment(mask,g):
    with pytest.raises(ValueError):m.apply_permutation(mask,g)


def test_timeout_and_readonly_determinism():
    q=query(3);saved=deepcopy(q)
    assert m.inspect_query(q)==m.inspect_query(q) and q==saved
    with pytest.raises(TimeoutError):m.inspect_query(q,deadline=-1)


def synthetic_sources():
    cases=[];cubes=[];masks=[];docs={}
    for n,count in [(0,31),(1,19),(2,8),(3,10),(4,6),(5,1),(6,1),(7,1)]:
        for _ in range(count):
            idx=len(cubes);doc='shared' if 31<=idx<=37 else f'd{idx:03}'
            units=[{'unit_id':f'u{i}','order':i,'kind':'paragraph','start':i,'end':i+1,'text':f'text{i}','native_text':f'text{i}'} for i in range(17)]
            docs[doc]=units;ident={'family_id':f'f{idx%24}','doc_id':doc,'question_id':f'q{idx}'}
            edges=[[doc,f'u{i}',f'u{i+1}'] for i in range(n)]
            function=(lambda mask:mask&1) if idx in ([31]+list(range(38,52))) else ((lambda mask:(mask&1)^((mask>>1)&1)) if idx in (52,53) else (lambda _:0))
            outcomes=[value(doc,[function(mask)]) for mask in range(2**n)]
            cube={**ident,'edges':edges,'outcomes':outcomes};cubes.append(cube)
            cases.append({**ident,'units':units,'candidates':list(range(n+1)),'labels':['yes']*(n+1)})
            for mask,o in enumerate(outcomes):
                selected=[int(o['selected_identities'][0][1][1:])]
                masks.append({**ident,'mask':mask,'selected_indices':selected,'pack_sha256':o['pack_sha256'],
                    'trace':[{'candidate_index':selected[0],'selected_before':[],'accepted':True,'action':'select'}]})
    tasks=[]
    for doc in sorted(docs):
        for i in range(16):
            if len(tasks)==562:break
            tasks.append({'id':f't_{doc}_{i}','doc_id':doc,'left_id':f'u{i}','right_id':f'u{i+1}',
                'item':{'unit_a':{'id':f'u{i}','text':f'text{i}'},'unit_b':{'id':f'u{i+1}','text':f'text{i+1}'}}})
    # Ensure every actually eligible edge is in the static inventory before filling extras.
    lookup={(t['doc_id'],t['left_id'],t['right_id']):t for t in tasks}
    needed={tuple(e) for c in cubes for e in c['edges']}
    all_tasks={}
    for doc in docs:
        for i in range(16):
            t={'id':f't_{doc}_{i}','doc_id':doc,'left_id':f'u{i}','right_id':f'u{i+1}',
               'item':{'unit_a':{'id':f'u{i}','text':f'text{i}'},'unit_b':{'id':f'u{i+1}','text':f'text{i+1}'}}}
            all_tasks[(doc,f'u{i}',f'u{i+1}')]=t
    keys=sorted(needed)+[k for k in sorted(all_tasks) if k not in needed][:562-len(needed)]
    prepared={'schema':'slac-qasper-extended-development-prepared-v1','static_tasks':[all_tasks[k] for k in keys],
              'queries':[{'query':'unused old query text'}],'answers':['never project']}
    return cubes,cases,masks,prepared


def test_projection_baseline_class_uses_success_count_not_attempt_count():
    cubes,cases,masks,p=synthetic_sources();q=next(c for c in cubes if len(c['edges'])==3);qid=q['question_id']
    z=next(r for r in masks if r['question_id']==qid and r['mask']==0)
    q['outcomes'][0]=value(q['doc_id'],[1,3]);z.update(selected_indices=[1,3],pack_sha256=q['outcomes'][0]['pack_sha256'],trace=[
        {'candidate_index':0,'selected_before':[],'accepted':False,'action':'skip_evidence_budget'},
        {'candidate_index':1,'selected_before':[],'accepted':True,'action':'select'},
        {'candidate_index':2,'selected_before':[1],'accepted':False,'action':'skip_evidence_budget'},
        {'candidate_index':3,'selected_before':[1],'accepted':True,'action':'select'}])
    projected=m.project(cubes,cases,masks,p);found=next(x for x in projected['queries'] if x['cube']['question_id']==qid)
    assert [e['stratum'][-1] for e in found['edge_metadata']]==[1,0,2]
    assert b'unused old query text' not in m.io.canonical(projected) and b'never project' not in m.io.canonical(projected)


@pytest.mark.parametrize('mutation',['duplicate_zero','missing_zero','unknown_unit','prompt_text','trace_prefix','eligible_no','family'])
def test_projection_rejects_wrong_provenance(mutation):
    cubes,cases,masks,p=synthetic_sources()
    if mutation=='duplicate_zero':masks.append(deepcopy(masks[0]))
    elif mutation=='missing_zero':masks.pop(0)
    elif mutation=='unknown_unit':cubes[31]['edges'][0][1]='missing'
    elif mutation=='prompt_text':p['static_tasks'][0]['item']['unit_a']['text']='changed'
    elif mutation=='trace_prefix':masks[0]['trace'][0]['selected_before']=[2]
    elif mutation=='eligible_no':cases[31]['labels'][0]='no'
    else:cases[0]['family_id']='other'
    with pytest.raises(ValueError):m.project(cubes,cases,masks,p)


def test_structurally_degenerate_must_not_estimate_or_call(monkeypatch):
    monkeypatch.setattr(m,'permutation',identity_permutation);data=m.project(*synthetic_sources())
    def forbidden(*a):raise AssertionError('degenerate control must stop before payload estimate')
    q,r,u,j,public=m.evaluate(data,budget_fn=forbidden)
    assert public['all_controls_structurally_degenerate'] and public['semantic_content_paid_phase_must_stop']
    assert len(q)==77 and len(r)==501 and len(u)==19 and j==[]
    assert public['budget_projection']['status']=='not_estimated_structurally_degenerate'


def test_nondegenerate_all_U_singleton_estimate_without_key_or_transport(monkeypatch):
    import openrouter_decision_client as client
    monkeypatch.setattr(m,'permutation',forced_rotation)
    monkeypatch.setattr(client,'read_key',lambda *a:pytest.fail('no key'))
    monkeypatch.setattr(client,'BoundedClient',lambda *a,**k:pytest.fail('no execution client'))
    q,r,u,j,pub=m.evaluate(m.project(*synthetic_sources()))
    assert not pub['all_controls_structurally_degenerate'] and 19<=len(u)<=38 and len(j)==len(u)
    assert len({x['task_id'] for x in j})==len(j)
    assert pub['budget_projection']['reservation_usd']==str(m.Decimal('.005')*len(j))
    assert all(len(x['payload']['questions'])==1 for x in j) and pub['paid_admitted'] is False
    assert sum(pub['required_count_per_question_histogram'].values())==77


def test_singleton_never_truncates_oversized_input():
    task={'id':'t','doc_id':'doc','left_id':'a','right_id':'b','item':{'unit_a':{'id':'a','text':'x'*25000},'unit_b':{'id':'b','text':'ok'}}}
    with pytest.raises(ValueError,match='byte cap'):m.singleton_budget([['doc','a','b']],[task])


def rewrite(path,obj):path.write_bytes(m.io.canonical(obj)+b'\n')
def lines(path,rows):path.write_bytes(b''.join(m.io.canonical(r)+b'\n' for r in rows))


@pytest.fixture
def source_fixture(tmp_path,monkeypatch):
    cubes,cases,masks,p=synthetic_sources();src=tmp_path/'sources';src.mkdir()
    names=['receipt','demand_plan','cubes','demand_seal','demand_summary','demand_public','gate_cases','gate_masks','prepared','client','helper']
    paths={n:src/(n+'.json') for n in names}
    plan={'input_binding_sha256':'old-binding'};rewrite(paths['demand_plan'],plan);rewrite(paths['cubes'],cubes)
    rewrite(paths['demand_seal'],{'plan.json':m.io.digest(paths['demand_plan']),'cubes.json':m.io.digest(paths['cubes'])})
    rewrite(paths['demand_public'],{'essential_edge_query_occurrences':19,'essential_static_edge_union_count':19})
    rewrite(paths['demand_summary'],{'status':'completed','all_cubes_compiled_and_verified':True,'output_sha256':{'public_aggregate.json':m.io.digest(paths['demand_public'])}})
    rewrite(paths['gate_cases'],cases);lines(paths['gate_masks'],masks);rewrite(paths['prepared'],p)
    rewrite(paths['client'],{'frozen':'client'});rewrite(paths['helper'],{'frozen':'helper'})
    receipt={'status':'verified_complete','questions':77,'families':24,'masks':501,'partial_cache_subcubes':4075,'assignment_known_subset_paths':23915,
        'all_essential_derivatives_dag_outputs_depths_and_aggregates_equal':True,'all_cache_restrictions_recomputed_via_ordered_cofactor_vectors':True,
        'quality_computed':False,'qa_or_answers_read':False,'input_output_sha256':{str(p):m.io.digest(p) for k,p in paths.items() if k!='receipt'}}
    rewrite(paths['receipt'],receipt)
    monkeypatch.setattr(m,'PATHS',paths);monkeypatch.setattr(m,'DIRECTORIES',{src:{p.name for p in paths.values()}})
    monkeypatch.setattr(m,'ANCHORS',{k:m.io.digest(paths[k]) for k in ('receipt','prepared','client')})
    monkeypatch.setattr(m,'RECEIPT_SHA',m.io.digest(paths['receipt']))
    return paths


def test_source_snapshot_and_prepare_never_permute_analyze_or_price(source_fixture,tmp_path,monkeypatch):
    for name in ('permutation','apply_permutation','influences','inspect_query','singleton_budget','evaluate'):
        monkeypatch.setattr(m,name,lambda *a,**k:pytest.fail('metadata prepare called symbolic analysis'))
    args=SimpleNamespace(output=tmp_path/'plan',run_output=tmp_path/'run');config=m.prepare(args)
    loaded,data,_=m.load_plan(args.output);assert loaded==config and len(data['queries'])==77
    assert len(config['input_sha256'])==14 and not args.run_output.exists()
    with pytest.raises(FileExistsError):m.prepare(args)


@pytest.mark.parametrize('target',['receipt','prepared','gate_cases','extra'])
def test_source_anchors_and_inventory_rejected(source_fixture,target):
    if target=='extra':rewrite(source_fixture['receipt'].parent/'extra.json',{})
    else:source_fixture[target].write_bytes(b'{}')
    with pytest.raises(ValueError):m.snapshot()


def test_source_toctou_rejected(source_fixture,monkeypatch):
    old=m.project
    def mutate(*a):
        x=old(*a);source_fixture['gate_cases'].write_bytes(b'[]');return x
    monkeypatch.setattr(m,'project',mutate)
    with pytest.raises(ValueError,match='changed'):m.snapshot()


@pytest.fixture
def prepared(source_fixture,tmp_path,monkeypatch):
    # Identity synthetic permutation makes the I/O fixture exercise the stop branch.
    monkeypatch.setattr(m,'permutation',identity_permutation)
    args=SimpleNamespace(output=tmp_path/'plan',run_output=tmp_path/'run');m.prepare(args);return args


def test_full_run_audit_single_use_and_public_whitelist(prepared):
    assert m.run(SimpleNamespace(plan=prepared.output))['status']=='completed'
    result=m.audit(SimpleNamespace(plan=prepared.output,run=prepared.run_output));assert result['status']=='verified_complete'
    public=m.io.parse((prepared.run_output/'public_aggregate.json').read_bytes())
    assert public['semantic_content_paid_phase_must_stop'] and public['equal_output_assignments']==501
    assert public['labels_observed']==0 and public['full_label_counts_estimated'] is False
    assert not any(k in public for k in ('question_id','doc_id','required_edges','strata','payload'))
    with pytest.raises(FileExistsError):m.run(SimpleNamespace(plan=prepared.output))


@pytest.mark.parametrize('target',['question','mask','required','payload','public','summary','extra'])
def test_resealed_complete_run_tampering_rejected(prepared,target):
    m.run(SimpleNamespace(plan=prepared.output));r=prepared.run_output
    if target in ('question','mask'):
        path=r/('per_question.jsonl' if target=='question' else 'per_mask.jsonl');rows=[m.io.parse(l) for l in path.read_bytes().splitlines()]
        if target=='question':rows[0]['function_equal']=False
        else:rows[0]['permuted_mask']=1
        lines(path,rows)
    elif target=='required':rewrite(r/'required_edges.json',[])
    elif target=='payload':rewrite(r/'static_payloads.json',[{'unexpected':'request'}])
    elif target=='public':
        p=m.io.parse((r/'public_aggregate.json').read_bytes());p['semantic_content_paid_phase_must_stop']=False;rewrite(r/'public_aggregate.json',p)
    elif target=='extra':rewrite(r/'extra.json',{})
    summary=m.io.parse((r/'summary.json').read_bytes())
    if target=='summary':summary['paid_admitted']=True
    summary['output_sha256']={n:m.io.digest(r/n) for n in m.RUN_FILES-{'summary.json'}};rewrite(r/'summary.json',summary)
    with pytest.raises(ValueError):m.audit(SimpleNamespace(plan=prepared.output,run=r))


def test_fail_closed_deadline_after_summary(prepared,monkeypatch):
    original=m.check_dirs
    def fail():
        original()
        if (prepared.run_output/'summary.json').exists():raise TimeoutError('after-summary deadline')
    monkeypatch.setattr(m,'check_dirs',fail)
    with pytest.raises(TimeoutError):m.run(SimpleNamespace(plan=prepared.output))
    assert not (prepared.run_output/'summary.json').exists() and (prepared.run_output/'failure.json').exists()
    with pytest.raises(FileExistsError):m.run(SimpleNamespace(plan=prepared.output))
    with pytest.raises(ValueError):m.audit(SimpleNamespace(plan=prepared.output,run=prepared.run_output))
