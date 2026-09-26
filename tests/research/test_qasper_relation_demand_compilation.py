"""Synthetic finite functions and sealed I/O; never analyze the real gate."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[2]/'docs/research/run_qasper_relation_demand_compilation.py'
spec=importlib.util.spec_from_file_location('demand_compilation',SOURCE)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def outcome(label='a',doc='d',tokens=10):
    text=f'[{label}]\ntext {label}'
    return {'selected_identities':[[doc,label]],'rendered_pack':text,
            'pack_sha256':hashlib.sha256(text.encode()).hexdigest(),'actual_tokens':tokens}


def cube(bits,fn,doc='d'):
    return {'family_id':'f','doc_id':doc,'question_id':'q',
            'edges':[[doc,f'u{i}',f'u{i+1}'] for i in range(bits)],
            'outcomes':[outcome(str(fn(mask)),doc) for mask in range(2**bits)]}


def necessary(c):return d.essential_indices([d.canonical(o) for o in c['outcomes']],list(range(len(c['outcomes']))),range(len(c['edges'])))


def depths(c):
    dag=d.compile_cube(c);d.validate_dag(c,dag);result=[]
    for mask,expected in enumerate(c['outcomes']):
        actual,asked=d.traverse(dag,mask,len(c['edges']));assert actual==expected;result.append(len(asked))
    return dag,result


def test_xor_equal_output_sets_are_not_equal_cofactors():
    c=cube(2,lambda m:(m&1)^((m>>1)&1));assert necessary(c)==[0,1]
    dag,ds=depths(c);assert ds==[2,2,2,2]
    assert dag['nodes'][dag['root']]['edge_index']==0
    assert sum(n['kind']=='terminal' for n in dag['nodes'])==2


def test_and_conditional_irrelevance_is_not_global_irrelevance():
    c=cube(2,lambda m:int(m==3));assert necessary(c)==[0,1]
    assert depths(c)[1]==[1,2,1,2]
    keys=[d.canonical(o) for o in c['outcomes']]
    assert d.essential_indices(keys,[0,2],[1])==[]
    assert d.essential_indices(keys,[1,3],[1])==[1]


def test_complete_cube_does_not_use_only_zero_full_or_one_branch():
    c=cube(3,lambda m:int(m==2));assert necessary(c)==[0,1,2]
    assert c['outcomes'][0]==c['outcomes'][-1]
    depths(c)


def test_constant_cube_early_stops_including_zero_variables():
    for bits in (0,3):
        c=cube(bits,lambda _:0);dag,ds=depths(c)
        assert necessary(c)==[] and ds==[0]*(2**bits) and len(dag['nodes'])==1


def test_order_is_native_not_optimized():
    c=cube(3,lambda m:((m>>2)&1) if m&1 else ((m>>1)&1))
    dag,_=depths(c);assert dag['nodes'][dag['root']]['edge_index']==0
    c=cube(3,lambda m:(m>>2)&1);dag,ds=depths(c)
    assert necessary(c)==[2] and dag['nodes'][dag['root']]['edge_index']==2 and ds==[1]*8


@pytest.mark.parametrize('field',['selected_identities','actual_tokens'])
def test_same_pack_hash_different_output_contract_is_not_merged(field):
    c=cube(1,lambda _:0)
    c['outcomes'][1][field]=[['d','another-identity']] if field=='selected_identities' else 11
    assert c['outcomes'][0]['pack_sha256']==c['outcomes'][1]['pack_sha256']
    assert necessary(c)==[0] and depths(c)[1]==[1,1]


def test_no_trace_in_terminal_contract():
    c=cube(1,lambda _:0);c['outcomes'][0]['trace']=[0]
    with pytest.raises(ValueError,match='whitelist'):d.compile_cube(c)
    c['outcomes'][0].pop('trace');assert depths(c)[1]==[0,0]


def test_placebo_swap_needs_content_irrelevant_source_edge():
    content=cube(2,lambda m:m&1);swapped=cube(2,lambda m:(m>>1)&1)
    assert necessary(content)==[0] and necessary(swapped)==[1]


def test_cache_conditions_all_known_bits_before_request():
    c=cube(2,lambda m:int(m==3));cache={tuple(c['edges'][1]):False}
    def forbidden(_):raise AssertionError('a is irrelevant after cached b=false')
    assert d.cached_evaluate(c,forbidden,cache)==c['outcomes'][0]


def test_source_qualified_cache_shared_consistent_value_once():
    a=cube(1,lambda m:m);b=deepcopy(a);b['question_id']='second';b['outcomes']=[outcome('x'),outcome('y')]
    other=cube(1,lambda m:m,doc='other');cache={};calls=[]
    def oracle(e):calls.append(e);return True
    assert d.cached_evaluate(a,oracle,cache)==a['outcomes'][1]
    assert d.cached_evaluate(b,oracle,cache)==b['outcomes'][1]
    assert d.cached_evaluate(other,oracle,cache)==other['outcomes'][1]
    assert calls==[tuple(a['edges'][0]),tuple(other['edges'][0])]


def test_invalid_oracle_cache_boolean_rejected():
    c=cube(1,lambda m:m)
    with pytest.raises(ValueError):d.cached_evaluate(c,lambda _:1,{})
    with pytest.raises(ValueError):d.cached_evaluate(c,lambda _:True,{tuple(c['edges'][0]):0})


def test_all_partial_cache_paths_are_exhaustive():
    for c in (cube(0,lambda _:0),cube(2,lambda m:int(m==3)),cube(3,lambda m:m%3)):
        result=d.verify_all_cache_restrictions(c);m=len(c['edges'])
        assert result['partial_cache_subcubes']==3**m and result['assignment_known_subset_paths']==4**m
        assert sum(result['new_requests_histogram'].values())==4**m


def test_cache_verifier_detects_wrong_first_question(monkeypatch):
    c=cube(2,lambda m:m&1)
    monkeypatch.setattr(d,'_cached_output',lambda cube,keys,oracle,cache,deadline:oracle(tuple(cube['edges'][1])))
    with pytest.raises(ValueError,match='first conditional'):d.verify_all_cache_restrictions(c)


@pytest.mark.parametrize('mutation',['incomplete','duplicate_edge','unknown_source','terminal_hash','terminal_bool','query','edge_bool'])
def test_invalid_cube_fails(mutation):
    c=cube(2,lambda m:m)
    if mutation=='incomplete':c['outcomes'].pop()
    elif mutation=='duplicate_edge':c['edges'][1]=c['edges'][0]
    elif mutation=='unknown_source':c['edges'][0][0]='other'
    elif mutation=='terminal_hash':c['outcomes'][0]['pack_sha256']='0'*64
    elif mutation=='terminal_bool':c['outcomes'][0]['actual_tokens']=True
    elif mutation=='query':c['raw_query']='forbidden'
    else:c['edges'][0][1]=True
    with pytest.raises(ValueError):d.compile_cube(c)


@pytest.mark.parametrize('mutation',['cycle','unknown_node','bad_terminal','nonordered','unreachable','equal_branches'])
def test_malformed_dag_fails(mutation):
    c=cube(2,lambda m:(m&1)^((m>>1)&1));dag=d.compile_cube(c);root=dag['nodes'][dag['root']]
    if mutation=='cycle':root['zero']=dag['root']
    elif mutation=='unknown_node':root['one']=999
    elif mutation=='bad_terminal':next(n for n in dag['nodes'] if n['kind']=='terminal')['outcome']['actual_tokens']=True
    elif mutation=='nonordered':next(n for n in dag['nodes'] if n['kind']=='decision' and n['edge_index']==1)['edge_index']=0
    elif mutation=='unreachable':dag['nodes'].append({'kind':'terminal','outcome':outcome('unused')})
    else:root['one']=root['zero']
    with pytest.raises(ValueError):d.validate_dag(c,dag)


def test_compiler_deterministic_readonly_and_traversal_independent(monkeypatch):
    c=cube(3,lambda m:m%3);before=deepcopy(c);a=d.compile_cube(c);b=d.compile_cube(c)
    assert a==b and c==before
    monkeypatch.setattr(d,'compile_cube',lambda *a,**k:pytest.fail('traversal called compiler'))
    monkeypatch.setattr(d,'essential_indices',lambda *a,**k:pytest.fail('traversal called influence'))
    for m,o in enumerate(c['outcomes']):assert d.traverse(a,m,3)[0]==o


def test_deadline():
    with pytest.raises(TimeoutError):d.compile_cube(cube(1,lambda m:m),deadline=-1)


def synthetic_sources():
    cases=[];rows=[];questions=[]
    for bits,count in [(0,31),(1,19),(2,8),(3,10),(4,6),(5,1),(6,1),(7,1)]:
        for _ in range(count):
            num=len(cases);doc=f'd{num}';identity={'family_id':f'f{num%24}','doc_id':doc,'question_id':f'q{num}'}
            units=[{'unit_id':f'u{i}','order':i,'text':f'text{i}','native_text':f'text{i}'} for i in range(bits+1)]
            case={**identity,'units':units,'candidates':list(range(bits+1)),'labels':['yes']*(bits+1)};cases.append(case)
            edges=[[i,i+1] for i in range(bits)];questions.append({**identity,'eligible_edges':edges,'mask_count':2**bits})
            for m in range(2**bits):
                i=bits if m.bit_count()%2 else 0;text=f'[u{i}]\ntext{i}'
                rows.append({**identity,'mask':m,'selected_indices':[i],'selected_ids':[f'u{i}'],
                    'active_edges':[e for bit,e in enumerate(edges) if m>>bit&1],
                    'pack_sha256':hashlib.sha256(text.encode()).hexdigest(),'actual_tokens':len(text)+2})
    return cases,rows,questions


def write_rows(path,rows):path.write_bytes(b''.join(d.canonical(r)+b'\n' for r in rows))
def rewrite(path,obj):path.write_bytes(d.canonical(obj)+b'\n')


@pytest.fixture
def sources(tmp_path,monkeypatch):
    cases,rows,questions=synthetic_sources()
    p=tmp_path/'gate-plan';r=tmp_path/'gate-run';receipt_dir=tmp_path/'receipt';protocol_dir=tmp_path/'protocol'
    for path in (p,r,receipt_dir,protocol_dir):path.mkdir()
    paths={'receipt':receipt_dir/'verification.json','plan':p/'plan.json','cases':p/'cases.json','seal':p/'seal.json',
           **{n:r/n for n in ('per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json')},'protocol':protocol_dir/'protocol.json'}
    inv=d.inventory(d.project(cases,rows,questions));plan={'inventory':inv,'source_metadata':{'inherited_upstream_commitment_count':2564,'upstream_input_binding_sha256':'synthetic'}}
    rewrite(paths['cases'],cases);rewrite(paths['plan'],plan);rewrite(paths['seal'],{n:d.digest(p/n) for n in ('plan.json','cases.json')})
    write_rows(paths['per_mask.jsonl'],rows);write_rows(paths['per_question.jsonl'],questions);rewrite(paths['public_aggregate.json'],{'synthetic':True})
    rewrite(paths['summary.json'],{'status':'completed','all_masks_available':True,'inventory':inv,
        'output_sha256':{n:d.digest(r/n) for n in ('per_mask.jsonl','per_question.jsonl','public_aggregate.json')}})
    rewrite(paths['protocol'],{'inventory':inv})
    receipt={'status':'verified_complete','questions':77,'families':24,'masks':501,
             'every_selected_identity_pack_token_and_step_recomputed':True,'all_behavioral_aggregate_fields_recomputed':True,
             'official_selector_or_aggregator_imported':False,'qa_or_answers_read':False,'quality_computed':False,'api_calls':0,
             'input_output_sha256':{str(path):d.digest(path) for n,path in paths.items() if n!='receipt'},'plan_sha256':d.digest(paths['plan'])}
    rewrite(paths['receipt'],receipt)
    monkeypatch.setattr(d,'SOURCE_PATHS',paths);monkeypatch.setattr(d,'RECEIPT_SHA',d.digest(paths['receipt']))
    monkeypatch.setattr(d,'SOURCE_INVENTORIES',{p:{'plan.json','cases.json','seal.json'},r:{'per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'},receipt_dir:{'verification.json'}})
    return paths


def test_source_projection_does_not_compute_influence_or_depth(sources,monkeypatch):
    for name in ('essential_indices','compile_cube','traverse','evaluate','verify_all_cache_restrictions','cached_evaluate'):
        monkeypatch.setattr(d,name,lambda *a,**k:pytest.fail('prepare analyzed real behavior'))
    cubes,meta,bindings=d.source_snapshot();assert len(cubes)==77 and sum(len(c['outcomes']) for c in cubes)==501
    assert meta['unused_ancestors_rehashed'] is False and len(bindings)==12


@pytest.mark.parametrize('mutation',['duplicate','missing','unknown','selected','hash','edge'])
def test_source_record_corruption_rejected(mutation):
    cases,rows,questions=synthetic_sources()
    if mutation=='duplicate':rows.append(deepcopy(rows[0]))
    elif mutation=='missing':rows.pop()
    elif mutation=='unknown':rows[0]['doc_id']='absent'
    elif mutation=='selected':rows[0]['selected_ids']=['unknown']
    elif mutation=='hash':rows[0]['pack_sha256']='0'*64
    else:rows[-1]['active_edges']=[]
    with pytest.raises(ValueError):d.project(cases,rows,questions)


@pytest.mark.parametrize('mutation',['receipt','direct_file','extra_inventory'])
def test_source_binding_rejections(sources,mutation):
    if mutation=='receipt':rewrite(sources['receipt'],{'status':'verified_complete'})
    elif mutation=='direct_file':sources['cases'].write_bytes(b'[]')
    else:rewrite(sources['cases'].parent/'extra.json',{})
    with pytest.raises(ValueError):d.source_snapshot()


def test_source_toctou_rejected(sources,monkeypatch):
    original=d.project
    def mutate(*args):
        value=original(*args);sources['cases'].write_bytes(b'[]');return value
    monkeypatch.setattr(d,'project',mutate)
    with pytest.raises(ValueError,match='changed'):d.source_snapshot()


@pytest.fixture
def prepared(sources,tmp_path):
    args=SimpleNamespace(output=tmp_path/'new-plan',run_output=tmp_path/'new-run')
    d.prepare(args);return args


def test_prepare_no_analysis_and_sealed_roundtrip(sources,tmp_path,monkeypatch):
    for name in ('essential_indices','compile_cube','traverse','evaluate','verify_all_cache_restrictions','cached_evaluate'):
        monkeypatch.setattr(d,name,lambda *a,**k:pytest.fail('metadata prepare invoked analysis'))
    args=SimpleNamespace(output=tmp_path/'new-plan',run_output=tmp_path/'new-run');config=d.prepare(args)
    saved,cubes,_=d.load_plan(args.output);assert saved==config and len(cubes)==77
    assert config['essential_or_depth_analysis_performed'] is False and not args.run_output.exists()
    with pytest.raises(FileExistsError):d.prepare(args)


def test_plan_changed_or_overlap_rejected(sources,tmp_path):
    with pytest.raises(ValueError):d.prepare(SimpleNamespace(output=tmp_path/'a',run_output=tmp_path/'a'/'run'))
    args=SimpleNamespace(output=tmp_path/'new-plan',run_output=tmp_path/'new-run');d.prepare(args)
    rewrite(args.output/'cubes.json',[])
    with pytest.raises(ValueError):d.load_plan(args.output)


def test_full_synthetic_run_audit_all_masks_and_safe_bounds(prepared):
    args=prepared;assert d.run(SimpleNamespace(plan=args.output))['status']=='completed'
    result=d.audit(SimpleNamespace(plan=args.output,run=args.run_output));assert result['status']=='verified_complete'
    public=d.parse((args.run_output/'public_aggregate.json').read_bytes())
    assert public['all_mask_output_equal_count']==501 and public['all_query_output_equal_count']==77
    assert sum(public['all_assignments_depth_histogram'].values())==501
    assert public['all_partial_cache_subcubes_verified']==4075 and public['all_assignment_known_subset_paths_equal']==23915
    assert public['essential_edge_query_occurrences']==107 and public['local_irrelevant_edge_query_occurrences']==0
    assert public['combined_safe_request_upper_bound']==107
    assert public['exact_global_cache_worst_case_computed'] is False and public['trace_equivalence_claimed'] is False
    assert not any(name in d.canonical(public).decode() for name in ('question_id','rendered_pack','selected_identities'))
    with pytest.raises(FileExistsError):d.run(SimpleNamespace(plan=args.output))


@pytest.mark.parametrize('target',['diagram','mask','question','public','summary','extra','nan_time'])
def test_resealed_outputs_are_recomputed(prepared,target):
    args=prepared;d.run(SimpleNamespace(plan=args.output));r=args.run_output
    if target in ('diagram','mask','question'):
        name={'diagram':'decisions.jsonl','mask':'per_mask.jsonl','question':'per_question.jsonl'}[target]
        records=[d.parse(line) for line in (r/name).read_bytes().splitlines()]
        if target=='diagram':records[0]['dag']['nodes'][0]['outcome']['actual_tokens']+=1
        elif target=='mask':records[0]['depth']+=1
        else:records[0]['depth_max']+=1
        write_rows(r/name,records)
    elif target=='public':
        value=d.parse((r/'public_aggregate.json').read_bytes());value['essential_static_edge_union_count']+=1;rewrite(r/'public_aggregate.json',value)
    elif target=='extra':rewrite(r/'extra.json',{})
    summary=d.parse((r/'summary.json').read_bytes())
    if target=='summary':summary['qa_sidecar_opened']=True
    if target=='nan_time':summary['execution_timing']['total_seconds_before_summary']=-1
    summary['output_sha256']={n:d.digest(r/n) for n in d.RUN_FILES-{'summary.json'}};rewrite(r/'summary.json',summary)
    with pytest.raises(ValueError):d.audit(SimpleNamespace(plan=args.output,run=r))


def test_timeout_failure_has_no_complete_summary_or_resume(prepared,monkeypatch):
    def timeout(*a,**k):raise TimeoutError('synthetic timeout')
    monkeypatch.setattr(d,'evaluate',timeout)
    with pytest.raises(TimeoutError):d.run(SimpleNamespace(plan=prepared.output))
    assert not (prepared.run_output/'summary.json').exists() and (prepared.run_output/'failure.json').exists()
    with pytest.raises(FileExistsError):d.run(SimpleNamespace(plan=prepared.output))
    with pytest.raises(ValueError):d.audit(SimpleNamespace(plan=prepared.output,run=prepared.run_output))


def test_after_summary_deadline_removes_complete(prepared,monkeypatch):
    original=d.verify_source_inventory
    def expire():
        original()
        if (prepared.run_output/'summary.json').exists():raise TimeoutError('deadline after summary')
    monkeypatch.setattr(d,'verify_source_inventory',expire)
    with pytest.raises(TimeoutError):d.run(SimpleNamespace(plan=prepared.output))
    assert not (prepared.run_output/'summary.json').exists() and (prepared.run_output/'failure.json').exists()
