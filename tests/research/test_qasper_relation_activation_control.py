"""Synthetic-only attribution control and immutable complete replay contracts."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/research'))
import run_qasper_relation_activation_control as a

REAL_SOURCE_DATA = a.source_data
REAL_EVALUATE = a.evaluate


@pytest.fixture(autouse=True)
def no_real_sources(monkeypatch):
    def deny(*args,**kwargs): raise AssertionError('real source/tokenizer forbidden')
    monkeypatch.setattr(a,'source_data',deny)
    monkeypatch.setattr(a.parent,'source_data',deny)
    monkeypatch.setattr(a.parent,'load_tokenizer',deny)


def core(texts=('A','B','C'),ranking=(2,1,0),labels=('yes','yes','yes'),candidates=None):
    units=tuple(a.parent.Unit('u'+str(i),i,'paragraph',i,i+1,t,t) for i,t in enumerate(texts))
    return a.parent.Core(units,tuple(range(len(units))) if candidates is None else tuple(candidates),tuple(ranking),tuple(labels))


def make_saved(c,count=lambda ids:len(ids)*30):
    case={'family_id':'f','doc_id':'d','question_id':'q','core':asdict(c)}
    m=len(a.parent.edges_of(c)); saved={'empty_cache':{},'all_known_subsets':{}}
    for suite in saved:
        for mask in range(1<<m):
            for subset in ([0] if suite=='empty_cache' else range(1<<m)):
                known={j:bool(mask&(1<<j)) for j in range(m) if subset&(1<<j)}
                z=a.parent.select(c,lambda j:bool(mask&(1<<j)),count,known)
                saved[suite][('d','q',mask,subset)]={'family_id':'f','doc_id':'d','question_id':'q',
                    'assignment':mask,'known_subset':subset,'eligible_count':m,'initial_known_count':len(known),
                    'output_equivalent':True,'result':z}
    signatures={mask:tuple(saved['empty_cache'][('d','q',mask,0)]['result']['selected_indices']) for mask in range(1<<m)}
    essential=[j for j in range(m) if any(signatures[mask]!=signatures[mask^(1<<j)] for mask in range(1<<m))]
    questions={('d','q'):{'doc_id':'d','question_id':'q','family_id':'f','essential_variables_after_selection':essential}}
    return [a.parse(a.canonical(case))],saved,questions


@pytest.mark.parametrize('c,count',[
    (core(),lambda s:30*len(s)),
    (core(('A','B','C','D','E'),(4,1,2,3,0),('no','yes','yes','unknown','yes')),lambda s:30*len(s)),
    (core(ranking=(2,0,1),labels=('yes','unknown','yes')),lambda s:30 if len(s)<=1 else 2048),
    (core(('same','B','same','D','E'),(2,1,0,4,3),('yes',)*5),lambda s:30*len(s)),
    (core(('A','B','C','D','E'),(2,1,0,4,3),('yes',)*5),lambda s:2048 if tuple(sorted(s)) in ((1,2),(0,2)) else 30*len(s)),
    (core(('A','B','C','D','E'),(4,2,3,0),('yes','unknown','no','yes'),(0,2,3,4)),lambda s:30*len(s)),
    (core(labels=('no','no','no')),lambda s:0),
])
def test_every_synthetic_assignment_and_initial_subset(c,count,monkeypatch):
    cases,saved,questions=make_saved(c,count)
    monkeypatch.setattr(a.parent,'select',lambda *x,**k:(_ for _ in ()).throw(AssertionError('old selector forbidden')))
    e,r,q,p=a.evaluate(cases,saved,questions,fixed=False)
    for row in [*e,*r]:
        assert row['lazy_reads']<=row['activation_reads']<=2
        assert set(row['lazy_requested_variables'])<=set(row['activation_requested_variables'])
        assert row['exact_rendered_pack_parity'] and row['winner_trace_parity']
    assert p['all_known_subsets']['full_cache_paths']==p['all_known_subsets']['full_cache_zero_read_paths']


def test_bound_is_tight_and_not_final_selected_minus_one():
    c=core(('A','B','C','D','E'),(4,1,2,3,0),('no','yes','yes','unknown','yes'))
    z=a.select(c,lambda j:True,lambda s:30*len(s))
    assert len(z['requests'])==2
    c=core(ranking=(2,0,1),labels=('yes','unknown','yes'))
    z=a.select(c,lambda j:True,lambda s:30 if len(s)<=1 else 2048)
    assert len(z['selected_indices'])==1 and len(z['requests'])==1


def test_certification_savings_and_nonessential_negative_retained():
    cases,saved,questions=make_saved(core())
    e,_,_,p=a.evaluate(cases,saved,questions,fixed=False)
    assert all(r['activation_reads']==2 and r['lazy_reads']==0 for r in e)
    assert p['empty_cache']['activation_nonessential_reads']['total']==8
    assert p['empty_cache']['activation_minus_lazy']['total']==8


@pytest.mark.parametrize('known',[{True:True},{0:1},{-1:False},{2:True},[]])
def test_cache_strict_types(known):
    with pytest.raises(ValueError): a.select(core(),lambda j:True,lambda s:30,known)


@pytest.mark.parametrize('value',[1,0,None,'true'])
def test_oracle_strict_bool(value):
    with pytest.raises(ValueError,match='strict bool'): a.select(core(),lambda j:value,lambda s:30)


def test_native_order_and_active_only_not_recursive():
    result=a.select(core(),lambda j:True,lambda s:30)
    assert result['requests']==[1,0]
    assert [e['selected_before'] for e in result['events'] if e['kind']=='request']==[[2],[1,2]]


def test_token_lookup_missing_and_inconsistent():
    _,saved,_=make_saved(core())
    rows=list(saved['all_known_subsets'].values()); lookup,_=a.token_lookup(rows)
    with pytest.raises(ValueError,match='missing'): lookup([99])
    bad=deepcopy(rows[0]); bad['result']['actual_tokens']+=1
    with pytest.raises(ValueError,match='inconsistent'): a.token_lookup([*rows,bad])


@pytest.mark.parametrize('change',[None,'missing','duplicate','bool_mask','bad_subset','family','overlap','question'])
def test_load_saved_exact_coverage(tmp_path,monkeypatch,change):
    cases,saved,questions=make_saved(core()); paths={}; hashes={}
    for suite,index in saved.items():
        rows=deepcopy(list(index.values()))
        if suite=='all_known_subsets':
            if change=='missing': rows.pop()
            elif change=='duplicate': rows.append(deepcopy(rows[0]))
            elif change=='bool_mask': rows[0]['assignment']=False
            elif change=='bad_subset': rows[0]['known_subset']=99
            elif change=='family': rows[0]['family_id']='another'
            elif change=='overlap': rows[0]['result']['actual_tokens']+=1
        p=tmp_path/(suite+'.jsonl'); a.rows_write(p,rows,float('inf'))
        paths['run_'+suite+'.jsonl']=p; hashes[str(p.resolve())]=a.digest(p)
    rows=list(questions.values()) if change!='question' else []
    p=tmp_path/'per_question.jsonl'; a.rows_write(p,rows,float('inf'))
    paths['run_per_question.jsonl']=p; hashes[str(p.resolve())]=a.digest(p)
    monkeypatch.setattr(a,'SOURCE_PATHS',paths)
    if change is None:
        actual,q=a.load_saved(cases,hashes,float('inf')); assert actual==saved and q==questions
    else:
        with pytest.raises(ValueError): a.load_saved(cases,hashes,float('inf'))


@pytest.mark.parametrize('change',['trace','pack','reads','negative_token'])
def test_parent_replay_mismatch_rejected(change):
    cases,saved,questions=make_saved(core())
    row=saved['empty_cache'][('d','q',0,0)]
    if change=='trace': next(e for e in row['result']['events'] if e['kind']=='winner')['action']='skip_evidence_budget'
    elif change=='pack': row['result']['pack_sha256']='bad'
    elif change=='reads': row['result']['requests']=[0,0]
    else: saved['all_known_subsets'][('d','q',0,0)]['result']['actual_tokens']=-1
    with pytest.raises(ValueError): a.evaluate(cases,saved,questions,fixed=False)


def fixture_plan(tmp_path,monkeypatch):
    cases,saved,questions=make_saved(core())
    dep=tmp_path/'immutable.txt'; dep.write_text('bound',encoding='utf-8')
    hashes={str(dep.resolve()):a.digest(dep)}
    inv=a.parent.inventory(cases,False); meta={'inherited':False}
    monkeypatch.setattr(a,'source_data',lambda deadline=float('inf'):(cases,hashes,inv,meta))
    monkeypatch.setattr(a,'load_saved',lambda *args:(saved,questions))
    monkeypatch.setattr(a,'evaluate',lambda cases,saved,questions,deadline=float('inf'):REAL_EVALUATE(cases,saved,questions,deadline,False))
    plan=tmp_path/'plan'; run=tmp_path/'run'
    a.prepare(SimpleNamespace(output=plan,run_output=run))
    return plan,run,dep


def test_prepare_load_metadata_only(tmp_path,monkeypatch):
    plan,run,_=fixture_plan(tmp_path,monkeypatch)
    for name in ('select','evaluate','load_saved','token_lookup'):
        monkeypatch.setattr(a,name,lambda *x,**k:(_ for _ in ()).throw(AssertionError('analysis forbidden')))
    cfg,cases,own=a.load_plan(plan)
    assert cfg['selector_executed'] is False and not run.exists()


def test_complete_single_use_run_and_audit(tmp_path,monkeypatch):
    plan,run,_=fixture_plan(tmp_path,monkeypatch)
    result=a.run(SimpleNamespace(plan=plan))
    audit=a.audit(SimpleNamespace(plan=plan,run=run))
    assert result['status']=='completed' and audit['status']=='verified_complete'
    with pytest.raises(FileExistsError): a.run(SimpleNamespace(plan=plan))


@pytest.mark.parametrize('change',['missing','extra','row','public','summary'])
def test_saved_run_tamper_rejected_even_resealed(tmp_path,monkeypatch,change):
    plan,run,_=fixture_plan(tmp_path,monkeypatch); a.run(SimpleNamespace(plan=plan))
    if change=='missing': (run/'empty_cache.jsonl').unlink()
    elif change=='extra': (run/'extra').write_text('x')
    elif change=='row':
        p=run/'empty_cache.jsonl'; rows=[a.parse(l) for l in p.read_bytes().splitlines()]; rows.pop(); p.write_bytes(b'\n'.join(a.canonical(r) for r in rows)+b'\n')
    elif change=='public':
        p=run/'public_aggregate.json'; obj=a.parse(p.read_bytes()); obj['empty_cache']['activation_reads']['total']+=1; p.write_bytes(a.canonical(obj))
    else:
        p=run/'summary.json'; obj=a.parse(p.read_bytes()); obj['elapsed_seconds']=-1; p.write_bytes(a.canonical(obj))
    if change in ('row','public'):
        p=run/'summary.json'; obj=a.parse(p.read_bytes()); obj['output_sha256']={n:a.digest(run/n) for n in a.RUN_FILES-{'summary.json'}}; p.write_bytes(a.canonical(obj))
    with pytest.raises(ValueError): a.audit(SimpleNamespace(plan=plan,run=run))


def test_source_changed_after_plan(tmp_path,monkeypatch):
    plan,run,dep=fixture_plan(tmp_path,monkeypatch); dep.write_text('changed')
    with pytest.raises(ValueError): a.run(SimpleNamespace(plan=plan))
    assert not run.exists()


def test_failure_no_complete_summary_or_retry(tmp_path,monkeypatch):
    plan,run,_=fixture_plan(tmp_path,monkeypatch)
    monkeypatch.setattr(a,'evaluate',lambda *x,**k:(_ for _ in ()).throw(TimeoutError('deadline')))
    with pytest.raises(TimeoutError): a.run(SimpleNamespace(plan=plan))
    assert (run/'failure.json').exists() and not (run/'summary.json').exists()
    with pytest.raises(FileExistsError): a.run(SimpleNamespace(plan=plan))


def test_final_deadline_removes_completed_summary(tmp_path,monkeypatch):
    plan,run,_=fixture_plan(tmp_path,monkeypatch); real=a.check
    def check(deadline):
        if (run/'summary.json').exists(): raise TimeoutError('final deadline')
        real(deadline)
    monkeypatch.setattr(a,'check',check)
    with pytest.raises(TimeoutError): a.run(SimpleNamespace(plan=plan))
    assert not (run/'summary.json').exists() and (run/'failure.json').exists()


def test_final_audit_inventory_rechecked(tmp_path,monkeypatch):
    plan,run,_=fixture_plan(tmp_path,monkeypatch); a.run(SimpleNamespace(plan=plan))
    real=a.verify_hashes
    def verify(hashes,deadline=float('inf')):
        real(hashes,deadline)
        if str((run/'summary.json').resolve()) in hashes: (run/'extra').write_text('during audit')
    monkeypatch.setattr(a,'verify_hashes',verify)
    with pytest.raises(ValueError,match='inventory changed'): a.audit(SimpleNamespace(plan=plan,run=run))


def test_true_metadata_adapter_without_parsing_paths(tmp_path,monkeypatch):
    pp=tmp_path/'parent-plan'; pr=tmp_path/'parent-run'; pe=tmp_path/'parent-exec'
    for d in (pp,pr,pe): d.mkdir()
    paths={'receipt':tmp_path/'receipt.json','audit':pe/'audit.json','completed':pe/'completed.json',
           **{'plan_'+n:pp/n for n in a.parent.PLAN_FILES},**{'run_'+n:pr/n for n in a.parent.RUN_FILES}}
    cases=[]
    for m,n in [(0,31),(1,19),(2,8),(3,10),(4,6),(5,1),(6,1),(7,1)]:
        for _ in range(n):
            i=len(cases); c=core(tuple('t'+str(x) for x in range(m+1)),tuple(range(m+1)),('yes',)*(m+1))
            cases.append({'family_id':'f'+str(i%24),'doc_id':'d','question_id':'q'+str(i),'core':a.parse(a.canonical(asdict(c)))})
    inv=a.parent.inventory(cases)
    a.write(pp/'cores.json',cases)
    cfg={'core_object_sha256':a.object_hash(cases),'inventory':inv,'input_sha256':{'inherited-only':'0'*64}}
    a.write(pp/'plan.json',cfg); a.write(pp/'seal.json',{n:a.digest(pp/n) for n in ('cores.json','plan.json')})
    for n in a.parent.RUN_FILES-{'summary.json'}: (pr/n).write_bytes(b'not JSON; forbidden to parse in prepare')
    summary={'status':'completed','all_paths_available':True,'quality_computed':False,'qa_or_answers_read':False,'inventory':inv,
             'runtime':{'test':True},'plan_sha256':{str((pp/n).resolve()):a.digest(pp/n) for n in a.parent.PLAN_FILES},
             'output_sha256':{n:a.digest(pr/n) for n in a.parent.RUN_FILES-{'summary.json'}}}
    a.write(pr/'summary.json',summary)
    audit={'status':'verified_complete','full_output_and_certificates_recomputed':True,
           'run_sha256':{str((pr/n).resolve()):a.digest(pr/n) for n in a.parent.RUN_FILES}}
    a.write(pe/'audit.json',audit)
    completed={'status':'completed_and_audited','all_direct_bindings_unchanged':True,
               'summary_sha256':a.digest(pr/'summary.json'),'audit_sha256':a.digest(pe/'audit.json')}
    a.write(pe/'completed.json',completed)
    bindings={str(p.resolve()):a.digest(p) for k,p in paths.items() if k!='receipt'}
    for p in [Path(a.parent.__file__).resolve(),*(a.ROOT/'docs/research'/n for n in ('run_qasper_evidence_baselines.py','qasper_metrics.py','qasper_alignment_v2.py'))]:
        bindings[str(p.resolve())]=a.digest(p)
    receipt={'status':'verified_complete','questions':77,'families':24,'empty_cache_paths':501,'assignment_known_subset_paths':23915,
        'independent_eager_completion_certificates_and_true_tokens_verified':True,'all_public_efficiency_aggregates_equal':True,
        'qa_or_answers_read':False,'quality_computed':False,'input_output_sha256':bindings}
    a.write(paths['receipt'],receipt)
    for name,value in [('SOURCE_PATHS',paths),('PARENT_PLAN',pp),('PARENT_RUN',pr),('PARENT_EXEC',pe),
                       ('RECEIPT_SHA',a.digest(paths['receipt'])),('AUDIT_SHA',a.digest(paths['audit'])),('COMPLETED_SHA',a.digest(paths['completed']))]:
        monkeypatch.setattr(a,name,value)
    for name in ('select','load_saved','token_lookup','evaluate'):
        monkeypatch.setattr(a,name,lambda *x,**k:(_ for _ in ()).throw(AssertionError('not metadata')))
    cases2,hashes,inv2,meta=REAL_SOURCE_DATA()
    assert cases2==cases and inv2==inv and meta['token_counts_freshly_remeasured'] is False
    assert 'inherited-only' not in hashes
    (pr/'extra').write_text('x')
    with pytest.raises(ValueError,match='inventory'): REAL_SOURCE_DATA()
