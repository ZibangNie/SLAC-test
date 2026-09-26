"""Synthetic selection, isolation, provenance and complete-only gate contracts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_opportunity as gate


class Tokenizer:
    def __init__(self,counts=None): self.counts=counts; self.calls=[]
    def encode(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':True,'truncation':False}
        indices=tuple(int(x) for x in re.findall(r'^\[u(\d+)\]',text,re.M))
        self.calls.append(indices)
        n=self.counts[indices] if self.counts is not None else len(text)+2
        return [0]*n


def case(texts=('A','B','C','D'),ranking=(2,3,1,0),labels=None,counts=None):
    us=tuple(gate.Unit(f'u{i}',i,'paragraph',i,i+1,t,t) for i,t in enumerate(texts))
    c=gate.Case('f','d','q',us,tuple(range(len(us))),tuple(ranking),
                tuple(labels or ['yes']*len(us)),(),0,hashlib.sha256(b'').hexdigest())
    result=gate.select(c,(),gate.PackCounter(Tokenizer(counts),us))
    return replace(c,baseline_selected=tuple(result['selected_indices']),baseline_tokens=result['actual_tokens'],
                   baseline_pack_sha256=result['pack_sha256'])


def small_spec(monkeypatch,cases):
    hist=gate.Counter(len(gate.eligible_edges(c)) for c in cases)
    changes={'questions':len(cases),'families':len({c.family_id for c in cases}),
             'static_edges':len({(c.doc_id,i,i+1) for c in cases for i in c.candidates if i+1 in c.candidates}),
             'edge_query_occurrences':sum(i+1 in c.candidates for c in cases for i in c.candidates),
             'eligible_edge_occurrences':sum(k*v for k,v in hist.items()),
             'eligible_edge_histogram':{str(k):v for k,v in sorted(hist.items())},
             'masks':sum((1<<k)*v for k,v in hist.items())}
    monkeypatch.setattr(gate,'SPEC',{**gate.SPEC,**changes})


def test_backward_direction_dynamic_chain_and_trace():
    c=case(); out=gate.select(c,[(0,1),(1,2)],lambda s:len(s)*20)
    assert [t['candidate_index'] for t in out['trace']]==[2,1,0]
    assert [t['relation_bonus'] for t in out['trace']]==[0,1,1]
    assert out['trace'][2]['active_triggers']==[[0,1]]
    assert out['selected_indices']==[0,1,2]
    forward=case(ranking=(0,3,2,1))
    result=gate.select(forward,[(0,1)],lambda s:len(s))
    assert [t['candidate_index'] for t in result['trace']]==[0,3,2]


def test_no_never_rescued_and_unknown_remains_eligible():
    c=case(labels=('no','unknown','yes','yes'))
    assert gate.eligible_edges(c)==((1,2),(2,3))
    with pytest.raises(ValueError): gate.select(c,[(0,1)],lambda s:len(s))
    out=gate.select(c,[(1,2)],lambda s:len(s))
    assert [t['candidate_index'] for t in out['trace']]==[2,1,3]
    assert out['trace'][1]['base_score']==.5 and out['trace'][1]['relation_bonus']==1
    assert 0 not in out['selected_indices']


@pytest.mark.parametrize('edges',[[(0,1),(0,1)],[(0,2)],[(1,0)],[(True,2)],[(0,-1)]])
def test_duplicate_or_nonadjacent_edges_cannot_accumulate_bonus(edges):
    with pytest.raises(ValueError): gate.select(case(),edges,lambda s:len(s))


def test_source_duplicate_skips_and_does_not_enable_duplicate_edge():
    c=case(texts=('A','A','B','C'),ranking=(0,1,2,3))
    assert (0,1) not in gate.eligible_edges(c)
    r=gate.select(c,gate.eligible_edges(c),lambda s:len(s))
    assert r['selected_indices']==[0,2,3]
    assert r['trace'][1]['action']=='skip_exact_native_duplicate'
    assert r['trace'][1]['proposed_tokens'] is None


def test_whole_pack_nonadditive_budget_and_skipped_item_never_revisited(monkeypatch):
    monkeypatch.setattr(gate,'SPEC',{**gate.SPEC,'max_units':2})
    c=case(texts=('A','B'),ranking=(1,0))
    counts={():0,(0,):1100,(1,):700,(0,1):1024}
    result=gate.select(c,[(0,1)],lambda s:counts[tuple(s)])
    assert result['selected_indices']==[0,1] and result['actual_tokens']==1024
    c2=replace(c,ranking=(0,1))
    skipped=gate.select(c2,[(0,1)],lambda s:counts[tuple(s)])
    assert skipped['selected_indices']==[1] and skipped['trace'][0]['action']=='skip_evidence_budget'


def test_budget_and_dense_tie_order_match_independent_reference():
    c=case(ranking=(3,2,1,0),labels=('unknown','yes','unknown','yes'))
    t=Tokenizer()
    actual=gate.select(c,(),gate.PackCounter(t,c.units))
    assert actual==gate.independent_reference(c,t,float('inf'))
    assert [x['candidate_index'] for x in actual['trace']]==[3,1,2]


def test_fullmask_reference_is_independent_of_select(monkeypatch):
    c=case(); expected=gate.select(c,gate.eligible_edges(c),lambda s:len(s)*10)
    monkeypatch.setattr(gate,'select',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not call select')))
    assert gate.adjacency_reference(c,lambda s:len(s)*10)==expected


def test_enumerates_every_mask_keeps_unchanged_questions_and_no_quality(monkeypatch):
    c1=case(); c2=replace(case(labels=('no','no','no','yes')),family_id='f2',doc_id='d2',question_id='q2')
    small_spec(monkeypatch,[c1,c2])
    masks,qs,public=gate.evaluate([c1,c2],Tokenizer())
    assert len(masks)==9 and [r['mask'] for r in masks if r['question_id']=='q']==list(range(8))
    assert len(qs)==2 and qs[1]['mask_count']==1 and qs[1]['any_pack_change'] is False
    assert public['queries_with_any_possible_pack_change']==1
    assert public['zero_independent_trace_parity_questions']==2
    assert public['full_adjacency_trace_parity_questions']==2
    assert public['changed_masks']+public['unchanged_masks']==9
    forbidden={'family_id','doc_id','question_id','trace','selected_ids','f1','recall','answer','query','gold'}
    def inspect(value):
        if isinstance(value,dict):
            assert not set(value)&forbidden
            for v in value.values():inspect(v)
        elif isinstance(value,list):
            for v in value:inspect(v)
    inspect(public)


def test_bad_saved_baseline_rejected_before_mask_enumeration(monkeypatch):
    c=case(); small_spec(monkeypatch,[c]); bad=replace(c,baseline_tokens=c.baseline_tokens+1)
    with pytest.raises(ValueError,match='frozen I'):gate.evaluate([bad],Tokenizer())


def test_independent_reference_mismatch_fails_closed(monkeypatch):
    c=case();small_spec(monkeypatch,[c])
    monkeypatch.setattr(gate,'independent_reference',lambda *a,**k:{})
    with pytest.raises(ValueError,match='step trace'):gate.evaluate([c],Tokenizer())


def test_invalid_counter_timeout_and_core_extra_fields():
    c=case()
    for bad in (-1,True,1.2,float('nan')):
        with pytest.raises(ValueError):gate.select(c,(),lambda s:bad)
    with pytest.raises(TimeoutError):gate.select(c,(),lambda s:0,deadline=0)
    with pytest.raises(ValueError):gate.case_from_json({**gate.case_to_json(c),'query':'forbidden'})
    with pytest.raises(ValueError):gate.select(replace(c,candidates=(False,1,2,3)),(),lambda s:0)


def test_project_drops_query_task_bodies_quality_and_answers(monkeypatch):
    c=case();small_spec(monkeypatch,[c])
    monkeypatch.setitem(gate.SPEC,'support_tasks',4);monkeypatch.setitem(gate.SPEC,'upstream_records',1)
    prepared={'schema':'slac-qasper-extended-development-prepared-v1','documents':{'d':[gate.asdict(u) for u in c.units]},
        'queries':[{'family_id':'f','doc_id':'d','question_id':'q','query':'DO NOT PASS QUERY',
                    'candidate_ids':['u0','u1','u2','u3'],'ranked_ids':['u2','u3','u1','u0']}],
        'support_tasks':[{'id':f's{i}','doc_id':'d','question_id':'q','unit_id':f'u{i}','item':{'answer':'DO NOT PASS'}} for i in range(4)],
        'static_tasks':[{'doc_id':'d','left_id':f'u{i}','right_id':f'u{i+1}','item':{}} for i in range(3)]}
    row={'family_id':'f','doc_id':'d','question_id':'q','method':'I_jev_k3',
         'selected_ids':[f'u{i}' for i in c.baseline_selected],'actual_evidence_tokens':c.baseline_tokens,
         'pack_sha256':c.baseline_pack_sha256,'official_evidence_f1':object(),'answer':'FORBIDDEN'}
    out=gate.project(prepared,{f's{i}':'yes' for i in range(4)},[row])
    assert out==[c]
    assert 'DO NOT PASS' not in gate.canonical([gate.case_to_json(x) for x in out]).decode()
    assert set(gate.case_to_json(out[0]))==set(gate.Case.__dataclass_fields__)


def test_core_never_calls_metrics_or_qa_loaders(monkeypatch):
    import qasper_metrics
    import run_qasper_evidence_baselines as old
    for module,names in [(qasper_metrics,('evidence_metrics','references_from_annotations','answer_f1')),
                         (old,('evidence_metrics','references_from_annotations','load_frozen_pool','choose_oracle'))]:
        for name in names:
            if hasattr(module,name):monkeypatch.setattr(module,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('quality called')))
    c=case();small_spec(monkeypatch,[c])
    gate.evaluate([c],Tokenizer())


def plan_fixture(tmp_path,monkeypatch):
    c=case();small_spec(monkeypatch,[c])
    source=tmp_path/'source.txt';source.write_text('frozen')
    bound={str(source):gate.digest(source)}
    metadata={'support_summary_sha256':'a'*64,'support_audit_sha256':'b'*64}
    def snapshot(deadline=float('inf')):
        gate.check_time(deadline);gate.verify_hashes(bound,deadline)
        return [c],metadata,str(tmp_path/'tokenizer'),bound
    monkeypatch.setattr(gate,'source_snapshot',snapshot)
    monkeypatch.setattr(gate,'verify_source_directories',lambda:None)
    plan=tmp_path/'plan';run=tmp_path/'run'
    gate.prepare(SimpleNamespace(output=plan,run_output=run))
    return plan,run,source


def test_prepare_never_selects_and_plan_roundtrip(tmp_path,monkeypatch):
    c=case();small_spec(monkeypatch,[c]); source=tmp_path/'f';source.write_text('f')
    monkeypatch.setattr(gate,'source_snapshot',lambda *a:([c],{'source':'test'},str(tmp_path/'tok'),{str(source):gate.digest(source)}))
    monkeypatch.setattr(gate,'select',lambda *a,**k:(_ for _ in ()).throw(AssertionError('prepare enumerated')))
    plan=tmp_path/'plan';run=tmp_path/'run'
    result=gate.prepare(SimpleNamespace(output=plan,run_output=run))
    assert result['selector_executed'] is False and not run.exists()
    loaded,cs,_=gate.load_plan(plan)
    assert cs==[c] and loaded['inventory']['masks']==8


def test_single_use_complete_run_and_full_audit(tmp_path,monkeypatch):
    plan,run,_=plan_fixture(tmp_path,monkeypatch)
    assert gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())['status']=='completed'
    report=gate.audit(SimpleNamespace(plan=plan,run=run),lambda _:Tokenizer())
    assert report['status']=='verified_complete' and report['masks']==8
    with pytest.raises(FileExistsError):gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())


def test_source_hash_change_rejected_before_tokenizer(tmp_path,monkeypatch):
    plan,run,source=plan_fixture(tmp_path,monkeypatch);source.write_text('changed')
    with pytest.raises(ValueError,match='hash changed'):
        gate.run(SimpleNamespace(plan=plan),lambda _:(_ for _ in ()).throw(AssertionError('loaded tokenizer')))
    assert not run.exists()


@pytest.mark.parametrize('mutation',['extra_file','wrong_count','false_api','changed_trace','private_public','nan_time'])
def test_audit_rejects_resealed_metadata_or_trace_tampering(tmp_path,monkeypatch,mutation):
    plan,run,_=plan_fixture(tmp_path,monkeypatch);gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())
    summary=gate.parse((run/'summary.json').read_bytes())
    if mutation=='extra_file':
        (run/'extra.json').write_text('{}');summary['output_sha256']['extra.json']=gate.digest(run/'extra.json')
    elif mutation=='wrong_count':summary['inventory']['masks']+=1
    elif mutation=='false_api':summary['api_calls']=1
    elif mutation=='changed_trace':
        rows=[gate.parse(line) for line in (run/'per_mask.jsonl').read_bytes().splitlines()]
        rows[0]['trace'][0]['accepted']=False
        (run/'per_mask.jsonl').write_bytes(b'\n'.join(gate.canonical(r) for r in rows)+b'\n')
        summary['output_sha256']['per_mask.jsonl']=gate.digest(run/'per_mask.jsonl')
    elif mutation=='private_public':
        public=gate.parse((run/'public_aggregate.json').read_bytes());public['question_id']='private'
        (run/'public_aggregate.json').write_bytes(gate.canonical(public))
        summary['output_sha256']['public_aggregate.json']=gate.digest(run/'public_aggregate.json')
    else:summary['execution_timing']['total_seconds_before_summary']=float('nan')
    (run/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    with pytest.raises(ValueError):gate.audit(SimpleNamespace(plan=plan,run=run),lambda _:Tokenizer())


def test_terminal_timeout_keeps_failure_no_complete_and_no_resume(tmp_path,monkeypatch):
    plan,run,_=plan_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(gate,'evaluate',lambda *a,**k:(_ for _ in ()).throw(TimeoutError('synthetic')))
    with pytest.raises(TimeoutError):gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())
    assert (run/'failure.json').exists() and not (run/'summary.json').exists()
    with pytest.raises(FileExistsError):gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())
    with pytest.raises(ValueError,match='inventory'):gate.audit(SimpleNamespace(plan=plan,run=run),lambda _:Tokenizer())


def test_final_deadline_after_summary_removes_completion(tmp_path,monkeypatch):
    plan,run,_=plan_fixture(tmp_path,monkeypatch)
    real=gate.check_time
    def check(deadline):
        if (run/'summary.json').exists():raise TimeoutError('late completion')
        real(deadline)
    monkeypatch.setattr(gate,'check_time',check)
    with pytest.raises(TimeoutError):gate.run(SimpleNamespace(plan=plan),lambda _:Tokenizer())
    assert not (run/'summary.json').exists() and (run/'failure.json').exists()


def test_plan_extra_fields_and_resealed_core_query_rejected(tmp_path,monkeypatch):
    plan,run,_=plan_fixture(tmp_path,monkeypatch)
    cases=gate.parse((plan/'cases.json').read_bytes()); cases[0]['query']='private text'
    (plan/'cases.json').write_bytes(gate.canonical(cases))
    config=gate.parse((plan/'plan.json').read_bytes());config['cases_object_sha256']=gate.object_hash(cases)
    (plan/'plan.json').write_bytes(gate.canonical(config))
    (plan/'seal.json').write_bytes(gate.canonical({n:gate.digest(plan/n) for n in ('plan.json','cases.json')}))
    with pytest.raises(ValueError,match='whitelist'):gate.load_plan(plan)


def source_fixture(tmp_path,monkeypatch):
    c=case();small_spec(monkeypatch,[c])
    monkeypatch.setitem(gate.SPEC,'support_tasks',4)
    monkeypatch.setitem(gate.SPEC,'upstream_records',15)
    monkeypatch.setattr(gate,'ARTIFACTS',tmp_path/'artifacts')
    paths={k:gate.ARTIFACTS/p for k,p in gate.SOURCE_PATHS.items()}
    for directory,names in gate.SOURCE_DIRECTORIES.items():
        d=gate.ARTIFACTS/directory;d.mkdir(parents=True)
        for name in names:
            if name=='provider_calls':(d/name).mkdir()
            else:(d/name).write_text('{}',encoding='utf-8')
    def put(name,obj):paths[name].write_bytes(gate.canonical(obj))
    prepared={'schema':'slac-qasper-extended-development-prepared-v1','documents':{'d':[gate.asdict(u) for u in c.units]},
      'queries':[{'family_id':'f','doc_id':'d','question_id':'q','query':'PRIVATE QUERY',
                  'candidate_ids':['u0','u1','u2','u3'],'ranked_ids':['u2','u3','u1','u0']}],
      'support_tasks':[{'id':f's{i}','doc_id':'d','question_id':'q','unit_id':f'u{i}','item':{'query':'PRIVATE QUERY'}} for i in range(4)],
      'static_tasks':[{'doc_id':'d','left_id':f'u{i}','right_id':f'u{i+1}'} for i in range(3)]}
    put('prepared',prepared)
    put('manifest',{'schema':'slac-qasper-extended-development-manifest-v1','status':'prepared',
                    'prepared_sha256':gate.digest(paths['prepared'])})
    put('labels',{'jev':{f's{i}':'yes' for i in range(4)}})
    row={'family_id':'f','doc_id':'d','question_id':'q','selected_ids':[f'u{i}' for i in c.baseline_selected],
         'actual_evidence_tokens':c.baseline_tokens,'pack_sha256':c.baseline_pack_sha256,
         'official_evidence_f1':'PRIVATE QUALITY NOT USED','answer':'PRIVATE ANSWER'}
    methods=['I_jev_k3']+[f'unused_{i}' for i in range(14)]
    paths['records'].write_bytes(b'\n'.join(gate.canonical({**row,'method':m}) for m in methods)+b'\n')
    tokenizer=tmp_path/'tokenizer';tokenizer.mkdir()
    for name in gate.TOKENIZER_FILES:(tokenizer/name).write_text('{}',encoding='utf-8')
    bound={str(p.resolve()):gate.digest(p) for p in [paths['prepared'],paths['manifest'],
           *[gate.ROOT/'docs/research'/n for n in gate.HELPERS],*[tokenizer/n for n in gate.TOKENIZER_FILES]]}
    config={'schema':'slac-qasper-primary-support-recovery-v1','status':'prepared_not_executed',
            'prepared_dir':str(paths['prepared'].parent),'tokenizer':str(tokenizer),'input_sha256':bound}
    put('config',config);put('plan_manifest',{'experiment_config_sha256':gate.digest(paths['config'])})
    summary={'schema':'slac-qasper-primary-support-recovery-v1','status':'completed','all_results_available':True,
        'question_count':1,'family_count':1,'record_count':15,'method_count':15,
        'plan_sha256':gate.digest(paths['config']),'output_sha256':{paths[k].name:gate.digest(paths[k]) for k in ('labels','records')},
        'execution_input_sha256':{},'input_binding_sha256':gate.object_hash(bound),
        'metrics':{'private_quality':'NOT USED'}}
    put('summary',summary)
    audit={'schema':'slac-qasper-primary-support-recovery-v1-audit','status':'verified_complete','question_count':1,
           'records':15,'all_inputs_outputs_unchanged':True,'api_calls':0,'key_read':False}
    put('audit',audit)
    put('execution',{'schema':'slac-recovery-external-execution-receipt-v1','status':'completed_and_audited',
          'child_pid':None,'audit_exit_code':0,'run_exit_code':0,
          'audit_sha256':gate.digest(paths['audit']),'summary_sha256':gate.digest(paths['summary'])})
    monkeypatch.setattr(gate,'ANCHORS',{k:gate.digest(paths[k]) for k in gate.ANCHORS})
    return c,paths


def test_source_snapshot_validates_real_adapter_without_quality_propagation(tmp_path,monkeypatch):
    c,paths=source_fixture(tmp_path,monkeypatch)
    def no_selection(*a,**kw):raise AssertionError('source snapshot ran selector')
    monkeypatch.setattr(gate,'select',no_selection)
    cs,meta,_,bound=gate.source_snapshot()
    assert cs==[c] and meta['old_quality_fields_projected_away'] is True
    assert 'PRIVATE' not in gate.canonical([gate.case_to_json(x) for x in cs]).decode()
    assert meta['rehashed_upstream_all'] is False and all(str(p) in bound for p in paths.values())


@pytest.mark.parametrize('mutation',['audit_anchor','summary_seal','source_inventory','tokenizer','parent_commitment'])
def test_direct_input_receipt_and_ancestor_chain_fail_closed(tmp_path,monkeypatch,mutation):
    c,paths=source_fixture(tmp_path,monkeypatch)
    if mutation=='audit_anchor':paths['audit'].write_text('{}',encoding='utf-8')
    elif mutation=='summary_seal':paths['records'].write_bytes(paths['records'].read_bytes()+b'{}\n')
    elif mutation=='source_inventory':(paths['summary'].parent/'failure.json').write_text('{}')
    elif mutation=='tokenizer':(tmp_path/'tokenizer/tokenizer.json').write_text('changed')
    else:
        x=gate.parse(paths['summary'].read_bytes());x['input_binding_sha256']='x'*64
        paths['summary'].write_bytes(gate.canonical(x))
        # Simulate an external release that anchored invalid commitment metadata.
        e=gate.parse(paths['execution'].read_bytes());e['summary_sha256']=gate.digest(paths['summary'])
        paths['execution'].write_bytes(gate.canonical(e))
        monkeypatch.setattr(gate,'ANCHORS',{k:gate.digest(paths[k]) for k in gate.ANCHORS})
    with pytest.raises(ValueError):gate.source_snapshot()


def test_source_changed_after_projection_is_not_adopted(tmp_path,monkeypatch):
    c,paths=source_fixture(tmp_path,monkeypatch);original=gate.project
    def project_then_mutate(*args):
        result=original(*args)
        paths['labels'].write_bytes(b'{}')
        return result
    monkeypatch.setattr(gate,'project',project_then_mutate)
    with pytest.raises(ValueError,match='hash changed'):gate.source_snapshot()


def test_actual_local_bge_tokenizer_on_synthetic_text_only():
    model=(gate.ROOT.parent/'SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/'
           '5617a9f61b028005a4858fdac845db406aefb181')
    if not (model/'tokenizer.json').exists():pytest.skip('pinned local tokenizer not installed')
    tokenizer=gate.load_tokenizer(str(model))
    c=case(texts=('Synthetic definition: x is an integer.','This x has value two.',
                  'A separate synthetic observation.','Unicode synthetic: alpha β 中。'))
    zero=gate.select(c,(),gate.PackCounter(tokenizer,c.units))
    assert zero==gate.independent_reference(c,tokenizer,float('inf'))
    full=gate.select(c,gate.eligible_edges(c),gate.PackCounter(tokenizer,c.units))
    assert full==gate.adjacency_reference(c,gate.PackCounter(tokenizer,c.units))
    text=gate.render_pack(c.units,full['selected_indices'])
    assert full['actual_tokens']==len(tokenizer.encode(text,add_special_tokens=True,truncation=False))
    assert full['pack_sha256']==hashlib.sha256(text.encode('utf-8')).hexdigest()
