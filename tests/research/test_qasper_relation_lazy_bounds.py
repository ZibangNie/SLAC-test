"""Synthetic-only lazy bounds correctness, isolation and complete replay contracts."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_lazy_bounds as lazy
REAL_SOURCE_DATA = lazy.source_data


@pytest.fixture(autouse=True)
def no_real_inputs_or_tokenizer(monkeypatch):
    monkeypatch.setattr(lazy,'source_data',lambda *a: (_ for _ in ()).throw(AssertionError('real inputs forbidden')))
    monkeypatch.setattr(lazy,'load_tokenizer',lambda *a: (_ for _ in ()).throw(AssertionError('real tokenizer forbidden')))


class Tokenizer:
    def __init__(self,counts=None): self.counts=counts; self.calls=[]
    def encode(self,text,**kwargs):
        assert kwargs == {'add_special_tokens':True,'truncation':False}
        ids=tuple(int(i) for i in re.findall(r'^\[u(\d+)\]',text,re.M)); self.calls.append(ids)
        return [0]*(self.counts[ids] if self.counts is not None else len(text)+2)


def core(texts=('A','B','C','D'),ranking=(2,3,1,0),labels=None,candidates=None):
    us=tuple(lazy.Unit(f'u{i}',i,'paragraph',i,i+1,t,t) for i,t in enumerate(texts))
    cand=tuple(range(len(us))) if candidates is None else tuple(candidates)
    return lazy.Core(us,cand,tuple(ranking),tuple(labels or ['yes']*len(cand)))


def eager(c,mask,count):
    """Direct complete-oracle scan, independent of production bounds/selector."""
    edges=lazy.edges_of(c); selected=[]; pending={i:l for i,l in zip(c.candidates,c.labels) if l!='no'}
    while pending and len(selected)<3:
        choices=[]
        for i,l in pending.items():
            bonus=any(a==i and b in selected and mask & (1<<j) for j,(a,b) in enumerate(edges))
            choices.append((-(2 if l=='yes' else 1)-2*bonus,c.ranking.index(i),c.units[i].order,i))
        i=min(choices)[3]; pending.pop(i)
        if c.units[i].native_text in {c.units[j].native_text for j in selected}: continue
        if count(sorted([*selected,i]))<=1024: selected.append(i)
    selected.sort(); rendered=lazy.render_pack(c.units,selected)
    return {'selected_indices':selected,'selected_ids':[c.units[i].unit_id for i in selected],
            'rendered_pack':rendered,'pack_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'actual_tokens':count(selected)}


def wrapper(c,identity='q'):
    return {'family_id':'f','doc_id':'d','question_id':identity,'core':lazy.parse(lazy.canonical(lazy.asdict(c)))}


def targets(cases,tokenizer):
    result={}
    for w in cases:
        c=lazy.from_json(w['core']); count=lazy.PackCounter(tokenizer,c.units)
        for mask in range(2**len(lazy.edges_of(c))): result[(w['doc_id'],w['question_id'],mask)]=eager(c,mask,count)
    return result


def test_runtime_input_has_no_witness_cube_or_quality():
    assert set(lazy.Core.__dataclass_fields__)=={'units','candidates','ranking','labels'}
    raw=wrapper(core())['core']; raw['outcomes']=[]
    with pytest.raises(ValueError,match='whitelist'): lazy.from_json(raw)
    source=Path(lazy.__file__).read_text(encoding='utf-8')
    assert 'import run_qasper_relation_demand' not in source
    assert 'import run_qasper_relation_opportunity' not in source


@pytest.mark.parametrize('labels,ranking',[
    (('yes','yes','yes','yes'),(2,3,1,0)),
    (('unknown','yes','unknown','yes'),(2,3,1,0)),
    (('yes','no','unknown','yes'),(3,0,2,1)),
    (('no','no','no','no'),(0,1,2,3)),
])
def test_every_synthetic_assignment_and_cache_subset(labels,ranking):
    c=core(labels=labels,ranking=ranking); m=len(lazy.edges_of(c)); count=lambda s:20*len(s)
    for assignment,subset in itertools.product(range(1<<m),repeat=2):
        known={j:bool(assignment&(1<<j)) for j in range(m) if subset&(1<<j)}
        out=lazy.select(c,lambda j:bool(assignment&(1<<j)),count,known)
        expected=eager(c,assignment,count)
        assert {k:out[k] for k in expected}==expected
        lazy.validate_path(c,assignment,known,out,count)
        assert not set(out['requests'])&set(known)
        assert len(out['requests'])<=m-len(known)


def test_own_unknown_edge_can_remain_unread_when_winner_certified():
    c=core(ranking=(1,0,3,2))
    out=lazy.select(c,lambda j: (_ for _ in ()).throw(AssertionError('no read needed')),lambda s:len(s))
    winner=out['events'][1]
    assert winner['incumbent']==0 and winner['kind']=='winner'
    assert winner['bounds'][0]['active_edge']==0
    assert winner['bounds'][0]['best_key']!=winner['bounds'][0]['worst_key']
    assert out['requests']==[]
    assert all('relation_bonus' not in e and 'priority_changed' not in e for e in out['events'])


def test_only_active_conflicting_native_edges_requested_not_earlier_inactive():
    c=core(texts=('A','B','C','D','E','F'),ranking=(4,5,3,2,1,0))
    out=lazy.select(c,lambda j:False,lambda s:len(s))
    assert out['requests'][0]==3  # earlier variables exist, but their B is not selected
    first=next(e for e in out['events'] if e['kind']=='request')
    assert first['conflicts']==[3] and first['selected_before']==[4]
    lazy.validate_path(c,0,{},out,lambda s:len(s))


def test_known_inactive_edge_is_used_later_without_request_or_mutation():
    c=core(); known={1:True,0:True}; copy=known.copy(); asked=[]
    out=lazy.select(c,lambda j:asked.append(j) or False,lambda s:len(s),known)
    assert known==copy and not set(asked)&{0,1}
    assert out['selected_indices']==[0,1,2]


def test_full_cache_never_reads_and_uses_all_values():
    c=core(); out=lazy.select(c,lambda j:pytest.fail('full cache asked'),lambda s:len(s),{0:True,1:True,2:False})
    assert out['requests']==[] and out['selected_indices']==[0,1,2]


@pytest.mark.parametrize('value',[0,1,None,'true',[],{},1.0])
def test_callback_bool_is_strict(value):
    with pytest.raises(ValueError,match='strict bool'): lazy.select(core(),lambda j:value,lambda s:len(s))


@pytest.mark.parametrize('known',[{True:True},{0:1},{0:None},{-1:False},{3:True},[],{0.0:True}])
def test_initial_cache_strict_types_and_domain(known):
    with pytest.raises(ValueError,match='cache requires'): lazy.select(core(),lambda j:False,lambda s:len(s),known)


def test_budget_rejection_does_not_stop_remaining_candidates():
    c=core(ranking=(0,1,2,3)); sizes={(0,):2000,(1,):100,(1,2):200,(1,2,3):300}
    out=lazy.select(c,lambda j:pytest.fail('inactive forward edges'),lambda s:sizes[tuple(s)])
    assert out['selected_indices']==[1,2,3]
    assert out['events'][0]['action']=='skip_evidence_budget'


def test_nonmonotonic_tokenization_is_not_pruned_by_singleton_estimate():
    c=core(texts=('A','B','C'),ranking=(0,1,2)); sizes={(0,):100,(1,):2000,(0,1):80,(0,1,2):90}
    calls=[]
    out=lazy.select(c,lambda j:False,lambda s:calls.append(tuple(s)) or sizes[tuple(s)])
    assert out['selected_indices']==[0,1,2] and (1,) not in calls and (0,1) in calls


def test_duplicate_native_text_skips_after_certification():
    c=core(texts=('A','B','A','D'),ranking=(0,2,1,3)); count=lambda s:len(s)
    out=lazy.select(c,lambda j:False,count)
    assert out['selected_indices']==[0,1,3]
    assert out['events'][1]['action']=='skip_exact_native_duplicate'
    assert out['events'][1]['proposed_tokens'] is None
    lazy.validate_path(c,0,{},out,count)


def test_no_exclusion_gap_and_duplicate_adjacent_edges():
    c=core(texts=('A','A','C','D'),labels=('yes','unknown','no','yes'))
    assert lazy.edges_of(c)==()
    assert 2 not in lazy.select(c,lambda j:pytest.fail('no eligible edges'),lambda s:len(s))['selected_indices']
    gap=core(candidates=(0,2,3),ranking=(2,3,0),labels=('yes','yes','yes'))
    assert lazy.edges_of(gap)==((2,3),)


@pytest.mark.parametrize('mutator',[
    lambda r:r['events'][0].update(incumbent=0),
    lambda r:r['events'][0]['bounds'][0].update(worst_key=[-999,0,0]),
    lambda r:r['events'][1].update(variable=0),
    lambda r:r['events'].pop(),
    lambda r:r.update(actual_tokens=999),
    lambda r:r['requests'].append(r['requests'][0]),
])
def test_certificate_tampering_rejected(mutator):
    c=core(); count=lambda s:20*len(s); out=lazy.select(c,lambda j:False,count)
    mutator(out)
    with pytest.raises(ValueError): lazy.validate_path(c,0,{},out,count)


def test_wrong_initial_oracle_value_rejected():
    c=core(); out=lazy.select(c,lambda j:False,lambda s:len(s),{0:True})
    with pytest.raises(ValueError,match='contradicts'): lazy.validate_path(c,0,{0:True},out,lambda s:len(s))


def test_numeric_certificate_types_cannot_be_replaced_by_equal_bool():
    c=core(); count=lambda s:len(s); out=lazy.select(c,lambda j:False,count)
    out['events'][0]['bounds'][0]['active_edge']=False  # None differs, as does a bool in an integer field
    with pytest.raises(ValueError): lazy.validate_path(c,0,{},out,count)
    out=lazy.select(c,lambda j:False,count)
    out['events'][0]['bounds'][0]['worst_key'][2]=False  # False == 0 in Python, but not in the sealed contract
    with pytest.raises(ValueError): lazy.validate_path(c,0,{},out,count)


def test_deadline_and_invalid_counter_fail_closed():
    with pytest.raises(TimeoutError): lazy.select(core(),lambda j:False,lambda s:len(s),deadline=-1)
    with pytest.raises(ValueError,match='token count'): lazy.select(core(),lambda j:False,lambda s:True)


def test_question_cache_shared_only_for_exact_tokenization():
    cases=[wrapper(core())]; tok=Tokenizer(); expected=targets(cases,Tokenizer())
    empty,allrows,queries,pub=lazy.evaluate(cases,tok,expected,fixed=False)
    assert len(empty)==8 and len(allrows)==64
    assert pub['empty_cache']['output_equivalent_paths']==8
    assert pub['all_known_subsets']['output_equivalent_paths']==64
    assert pub['all_known_subsets']['selector_tokenizer_encodes']==0
    assert pub['total_tokenizer_encodes']==len(tok.calls)
    assert pub['total_token_cache_entries']<=15
    assert pub['total_counter_calls']==sum(pub[k]['selector_counter_calls']+pub[k]['validator_counter_calls'] for k in ('empty_cache','all_known_subsets'))
    assert all(not r['result']['requests'] for r in allrows if r['known_subset']==7)
    assert [r['result'] for r in allrows if r['known_subset']==0]==[r['result'] for r in empty]


def test_target_mismatch_stops_before_complete_aggregate():
    cases=[wrapper(core())]; expected=targets(cases,Tokenizer()); expected[('d','q',0)]['pack_sha256']='0'*64
    with pytest.raises(ValueError,match='saved gate'): lazy.evaluate(cases,Tokenizer(),expected,fixed=False)


def test_essentiality_is_posthoc_and_nonessential_extra_reads_retained():
    # All four short units have identical native text, so eligibility itself is empty.
    c=core(texts=('A','A','A','A')); cases=[wrapper(c)]
    empty,allrows,queries,pub=lazy.evaluate(cases,Tokenizer(),targets(cases,Tokenizer()),fixed=False)
    assert pub['essential_edge_occurrences_posthoc']==0
    assert pub['empty_cache']['total_reads']==0 and len(allrows)==1


def test_runtime_may_read_nonessential_edge_without_hiding_it():
    # The budget rejects every item, so all complete outputs coincide. A dynamic
    # nonessential read instead needs one selected anchor and insufficient space
    # for every remaining candidate. Its ordering can still be unresolved.
    c=core(); sizes=lambda s:100 if s==[2] else (0 if not s else 2000)
    tokenizer=Tokenizer({(2,):100,(3,):2000,(1,):2000,(0,):2000,
                         (0,2):2000,(1,2):2000,(2,3):2000})
    cases=[wrapper(c)]; target={('d','q',m):eager(c,m,sizes) for m in range(8)}
    empty,allrows,queries,pub=lazy.evaluate(cases,tokenizer,target,fixed=False)
    assert pub['essential_edge_occurrences_posthoc']==0
    assert pub['empty_cache']['nonessential_reads']>0
    assert pub['empty_cache']['paths_reading_nonessential']>0


def synthetic_plan(tmp_path,monkeypatch):
    cases=[wrapper(core())]; inv=lazy.inventory(cases,False)
    monkeypatch.setattr(lazy,'SPEC',{**lazy.SPEC,**inv})
    source=tmp_path/'source.txt'; source.write_text('fixed',encoding='utf-8'); hashes={str(source.resolve()):lazy.digest(source)}
    monkeypatch.setattr(lazy,'source_data',lambda *a:(cases,'synthetic-tokenizer',hashes,inv))
    expected=targets(cases,Tokenizer()); monkeypatch.setattr(lazy,'load_targets',lambda *a:expected)
    plan=tmp_path/'plan'; out=tmp_path/'run'
    lazy.prepare(SimpleNamespace(output=plan,run_output=out))
    return plan,out,source,expected


def test_prepare_is_metadata_only_and_load_binds_source(tmp_path,monkeypatch):
    monkeypatch.setattr(lazy,'select',lambda *a:pytest.fail('prepare selected'))
    plan,out,source,_=synthetic_plan(tmp_path,monkeypatch)
    config,cases,own=lazy.load_plan(plan)
    assert config['selector_executed'] is False and not out.exists() and len(own)==3
    source.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='source changed'): lazy.load_plan(plan)


def test_plan_seal_and_configuration_tamper_rejected(tmp_path,monkeypatch):
    plan,_,_,_=synthetic_plan(tmp_path,monkeypatch)
    p=plan/'plan.json'; cfg=lazy.parse(p.read_bytes()); cfg['tokenizer']='unbound'; p.write_bytes(lazy.canonical(cfg))
    with pytest.raises(ValueError,match='seal'): lazy.load_plan(plan)
    (plan/'seal.json').write_bytes(lazy.canonical({n:lazy.digest(plan/n) for n in ('plan.json','cores.json')}))
    with pytest.raises(ValueError,match='projection'): lazy.load_plan(plan)


def test_complete_run_audit_and_exclusive_execution(tmp_path,monkeypatch):
    plan,out,_,_=synthetic_plan(tmp_path,monkeypatch)
    result=lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())
    assert result['status']=='completed'
    audit=lazy.audit(SimpleNamespace(plan=plan,run=out),lambda p:Tokenizer())
    assert audit['status']=='verified_complete' and audit['assignment_known_subset_paths']==64
    with pytest.raises(FileExistsError): lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())


def test_failure_retained_without_complete_summary_and_cannot_restart(tmp_path,monkeypatch):
    plan,out,_,expected=synthetic_plan(tmp_path,monkeypatch); expected[('d','q',0)]['actual_tokens']=999
    with pytest.raises(ValueError): lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())
    assert (out/'failure.json').exists() and not (out/'summary.json').exists()
    with pytest.raises(FileExistsError): lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())
    with pytest.raises(ValueError,match='inventory'): lazy.audit(SimpleNamespace(plan=plan,run=out),lambda p:Tokenizer())


def test_tampered_aggregate_resealed_output_still_rejected(tmp_path,monkeypatch):
    plan,out,_,_=synthetic_plan(tmp_path,monkeypatch); lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())
    p=out/'public_aggregate.json'; value=lazy.parse(p.read_bytes()); value['empty_cache']['total_reads']+=1
    p.write_bytes(lazy.canonical(value)); summary=lazy.parse((out/'summary.json').read_bytes())
    summary['output_sha256']['public_aggregate.json']=lazy.digest(p); (out/'summary.json').write_bytes(lazy.canonical(summary))
    with pytest.raises(ValueError,match='aggregate'): lazy.audit(SimpleNamespace(plan=plan,run=out),lambda p:Tokenizer())


def test_source_change_during_evaluation_blocks_summary(tmp_path,monkeypatch):
    plan,out,source,_=synthetic_plan(tmp_path,monkeypatch); old=lazy.evaluate
    def change(*a,**kw):
        result=old(*a,**kw); source.write_text('changed',encoding='utf-8'); return result
    monkeypatch.setattr(lazy,'evaluate',change)
    with pytest.raises(ValueError,match='source changed'): lazy.run(SimpleNamespace(plan=plan),lambda p:Tokenizer())
    assert not (out/'summary.json').exists() and (out/'failure.json').exists()


def test_metadata_projection_binds_masks_without_parsing_them(tmp_path,monkeypatch):
    cases=[wrapper(core())]; inv=lazy.inventory(cases,False)
    monkeypatch.setattr(lazy,'SPEC',{**lazy.SPEC,**inv})
    parent=tmp_path/'parent'; parent.mkdir(); tok=tmp_path/'tokenizer'; tok.mkdir()
    paths={k:parent/(k+'.json') for k in lazy.SOURCE_PATHS}
    paths['gate_plan']=parent/'plan.json'; paths['gate_cases']=parent/'cases.json'
    monkeypatch.setattr(lazy,'SOURCE_PATHS',paths)
    raw=[{**{k:cases[0][k] for k in ('family_id','doc_id','question_id')},**cases[0]['core'],
          'baseline_selected':[2,3,1],'baseline_tokens':99,'baseline_pack_sha256':'0'*64}]
    lazy.write(paths['gate_cases'],raw)
    lazy.write(paths['gate_plan'],{'tokenizer':str(tok),'cases_object_sha256':lazy.object_hash(raw)})
    lazy.write(paths['gate_seal'],{paths[k].name:lazy.digest(paths[k]) for k in ('gate_plan','gate_cases')})
    paths['gate_masks'].write_bytes(b'not JSON: preparation must only hash this')
    lazy.write(paths['gate_summary'],{'status':'completed','all_masks_available':True,'quality_metrics_computed':False,
        'output_sha256':{'per_mask.jsonl':lazy.digest(paths['gate_masks'])}})
    for n in lazy.TOKEN_FILES: (tok/n).write_text('tokenizer metadata',encoding='utf-8')
    hashes={str(p.resolve()):lazy.digest(p) for k,p in paths.items() if k!='receipt'}
    hashes.update({str((tok/n).resolve()):lazy.digest(tok/n) for n in lazy.TOKEN_FILES})
    hashes.update({str((lazy.ROOT/'docs/research'/n).resolve()):lazy.digest(lazy.ROOT/'docs/research'/n) for n in lazy.HELPERS})
    lazy.write(paths['receipt'],{'status':'verified_complete','questions':77,'masks':501,
        'every_selected_identity_pack_token_and_step_recomputed':True,'qa_or_answers_read':False,'input_output_sha256':hashes})
    monkeypatch.setattr(lazy,'RECEIPT_SHA',lazy.digest(paths['receipt']))
    result,tokenizer,bindings,got=REAL_SOURCE_DATA()
    assert result==cases and tokenizer==str(tok.resolve()) and got==inv
    assert str(paths['gate_masks'].resolve()) in bindings
    assert set(result[0]['core'])==lazy.CORE_KEYS
    paths['gate_masks'].write_text('source changed',encoding='utf-8')
    with pytest.raises(ValueError,match='independent receipt'): REAL_SOURCE_DATA()


def test_output_overlap_refused_and_timing_validated(tmp_path):
    with pytest.raises(ValueError): lazy.avoid_overlap([tmp_path/'source'/'new'],[tmp_path/'source'])
    with pytest.raises(ValueError): lazy.avoid_overlap([tmp_path/'a',tmp_path/'a'/'b'],[])
    for bad in [float('nan'),-1,301,True]:
        with pytest.raises(ValueError): lazy.valid_timing({'loading_seconds':bad,'validation_seconds':0,
            'writing_and_verification_seconds':0,'total_seconds_before_summary':bad})


def test_denominator_formula_includes_empty_paths():
    hist={int(k):v for k,v in lazy.SPEC['eligible_edge_histogram'].items()}
    assert sum(v*2**m for m,v in hist.items())==501
    assert sum(v*4**m for m,v in hist.items())==23915
    assert 1+sum(lazy.math.comb(16,k) for k in (1,2,3))==697
