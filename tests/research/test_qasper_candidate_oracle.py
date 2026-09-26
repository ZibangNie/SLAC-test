"""Synthetic exact-oracle semantics and freeze/complete-only contracts."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"docs/research"))
import run_qasper_candidate_oracle as oracle
from run_qasper_evidence_baselines import Unit


def units(texts):
    return [Unit(f"u{i}",i,"paragraph",i,i+1,text,text) for i,text in enumerate(texts)]


def annotations(evidence,unanswerable=False):
    return [{"native_answer":{"unanswerable":unanswerable,"extractive_spans":["answer"],
        "free_form_answer":"","yes_no":None,"evidence":evidence}}]


def test_exact_fraction_preserves_duplicates_empty_and_independent_reference_maxima():
    assert oracle.fraction_metrics([], [[]])==(Fraction(1),Fraction(1))
    assert oracle.fraction_metrics([], [[("d","A")]])==(Fraction(0),Fraction(0))
    assert oracle.fraction_metrics([("d","A")], [[("d","A"),("d","A")]])==(Fraction(2,3),Fraction(1,2))
    predicted=[("d",s) for s in ("A","B","C")]
    refs=[[('d',s) for s in ('A','B','D','E')],[('d','C')]]
    assert oracle.fraction_metrics(predicted,refs)==(Fraction(4,7),Fraction(1))
    assert oracle.references(annotations(["FLOAT SELECTED: figure"]),"d")==[[('d','FLOAT SELECTED: figure')]]
    assert oracle.references(annotations(["ignored"],True),"d")==[[]]


def test_at_most_three_all_subsets_and_global_lex_tie():
    u=units(['A','B','C','D']); keys=[('d',x.unit_id) for x in u]
    best,stats=oracle.exact_oracle([3,1,2,0],u,keys,[[('d',x) for x in ['A','B','C','D']]],lambda s:len(s)*10)
    assert best==(tuple([0,1,2]),Fraction(6,7),Fraction(3,4),30)
    assert stats=={'enumerated_subsets':15,'duplicate_rejected_subsets':0,'overbudget_subsets':0,'feasible_subsets':15}


def test_same_text_different_source_never_matches_gold():
    u=units(['Heading','Heading']);keys=[('wrong','u0'),('right','u1')]
    best,_=oracle.exact_oracle([0,1],u,keys,[[('right','Heading')]],lambda s:sum(i+1 for i in s))
    assert best[0]==(1,) and best[1]==1
    best,_=oracle.exact_oracle([0],u,keys,[[('right','Heading')]],lambda s:len(s))
    assert best[0]==() and best[1]==0


def test_same_doc_text_duplicates_rejected_but_alternative_cheaper_location_kept():
    u=units(['A','A']);keys=[('d','u0'),('d','u1')]
    best,stats=oracle.exact_oracle([0,1],u,keys,[[('d','A')]],lambda s:sum(100 if i==0 else 10 for i in s))
    assert best[0]==(1,) and best[3]==10 and stats['duplicate_rejected_subsets']==1
    best,_=oracle.exact_oracle([1,0],u,keys,[[('d','A')]],lambda s:len(s)*10)
    assert best[0]==(0,)


def test_full_enumeration_does_not_assume_removing_nongold_cannot_increase_tokens():
    u=units(['A','Z']); keys=[('d',x.unit_id) for x in u]
    # Synthetic nonmonotone whole-pack counter: a reference-only shortcut is invalid.
    counts={():0,(0,):1025,(1,):5,(0,1):1024}
    best,stats=oracle.exact_oracle([0,1],u,keys,[[('d','A')]],lambda s:counts[s])
    assert best[0]==(0,1) and best[1]==Fraction(2,3) and stats['overbudget_subsets']==1


def test_empty_reference_forces_empty_optimum_without_filtering_other_references():
    u=units(['A']);keys=[('d','u0')]
    best,stats=oracle.exact_oracle([0],u,keys,[[('d','A')],[]],lambda s:len(s)*10)
    assert best==((),Fraction(1),Fraction(1),0) and stats['enumerated_subsets']==2
    best,_=oracle.exact_oracle([0],u,keys,[[('d','FLOAT SELECTED: chart')]],lambda s:len(s)*10)
    assert best[0]==() and best[1]==0


@pytest.mark.parametrize('indices',[[-1],[2],[True],[0,0]])
def test_invalid_candidate_indices_rejected(indices):
    with pytest.raises(ValueError): oracle.exact_oracle(indices,units(['A']),[('d','u0')],[[]],lambda s:0)


def test_timeout_and_invalid_budget_counter_fail_without_result():
    with pytest.raises(TimeoutError): oracle.exact_oracle([0],units(['A']),[('d','u0')],[[]],lambda s:0,deadline=0)
    with pytest.raises(ValueError): oracle.exact_oracle([0],units(['A']),[('d','u0')],[[]],lambda s:-1)


class Tokenizer:
    def encode(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':True,'truncation':False}
        return list(text)


def evaluate_fixture():
    u=units(['A','B']);keys=[('d',x.unit_id) for x in u]
    gold=annotations(['A'])
    source={'units':u,'keys':keys,'tokenizer':Tokenizer(),
        'qa_by_key':{('d','q'):{'answer_annotations':gold}},
        'prepared':{'queries':[{'family_id':'f','doc_id':'d','question_id':'q'}]}}
    pack=oracle.bridge.render_pack(u,[1])
    row={'family_id':'f','doc_id':'d','question_id':'q','scope':'given_document','method':'leaf_direct',
        'candidate_global_indices':[0,1],'selected_global_indices':[1],'actual_evidence_tokens':len(pack),
        'pack_sha256':hashlib.sha256(pack.encode()).hexdigest(),'source_qualified_evidence_f1':0,
        'source_qualified_evidence_recall':0,'candidate_source_qualified_evidence_recall':1}
    return source,[row]


def test_evaluate_real_render_counter_validates_actual_before_gold_and_dominates(monkeypatch):
    source,rows=evaluate_fixture()
    monkeypatch.setattr(oracle,'CONFIG',{**oracle.CONFIG,'enumerated_subsets_including_empty':4})
    records,stats=oracle.evaluate(source,rows)
    r=records[0]
    assert r['oracle_selected_global_indices']==[0] and r['oracle_source_qualified_evidence_f1']==1
    assert r['oracle_minus_actual_f1']==1 and r['oracle_evidence_tokens']==len('[u0]\nA')
    assert stats['enumerated_subsets']==4 and stats['feasible_subsets']==4 and stats['token_cache_entries']==4


@pytest.mark.parametrize('mutation',['outside','tokens','hash','duplicate','boolean','negative'])
def test_invalid_actual_witness_rejected_before_gold_access(monkeypatch,mutation):
    source,rows=evaluate_fixture();source['qa_by_key']={}
    if mutation=='outside': rows[0]['candidate_global_indices']=[0]
    elif mutation=='tokens': rows[0]['actual_evidence_tokens']+=1
    elif mutation=='hash': rows[0]['pack_sha256']='bad'
    elif mutation=='duplicate': rows[0]['selected_global_indices']=[1,1]
    elif mutation=='boolean': rows[0]['selected_global_indices']=[True]
    elif mutation=='negative': rows[0]['selected_global_indices']=[-1]
    with pytest.raises(ValueError,match='feasible oracle witness'): oracle.evaluate(source,rows)


def test_every_annotation_is_preserved_and_float_interface_matches(monkeypatch):
    source,rows=evaluate_fixture()
    source['qa_by_key']['d','q']['answer_annotations']=annotations(['A','A'])+annotations(['ignored'],True)
    rows[0]['candidate_source_qualified_evidence_recall']=.5
    monkeypatch.setattr(oracle,'CONFIG',{**oracle.CONFIG,'enumerated_subsets_including_empty':4})
    records,stats=oracle.evaluate(source,rows)
    r=records[0]
    assert r['reference_count']==2 and r['empty_reference_count']==1 and r['any_empty_reference']==1
    assert r['oracle_empty_pack']==1 and r['oracle_source_qualified_evidence_f1']==1


def aggregate_fixture():
    queries=[{'family_id':f'f{i%24}','doc_id':f'd{i%24}','question_id':f'q{i}'} for i in range(77)]
    rows=[]
    for scope in oracle.SCOPES:
        for m,method in enumerate(oracle.METHODS):
            for q in queries:
                rows.append({**q,'scope':scope,'method':method,
                    'candidate_source_qualified_evidence_recall':.8,
                    'actual_source_qualified_evidence_f1':.2,'actual_source_qualified_evidence_recall':.3,
                    'oracle_source_qualified_evidence_f1':.5+.1*m,'oracle_source_qualified_evidence_recall':.7+.1*m,
                    'oracle_minus_actual_f1':.3+.1*m,'actual_evidence_tokens':400,'oracle_evidence_tokens':300+m,
                    'oracle_selected_units':2,'oracle_empty_pack':0,'any_empty_reference':0})
    return queries,rows


def test_aggregate_requires_all_six_arms_all_questions_and_four_fixed_pairs():
    queries,rows=aggregate_fixture()
    result=oracle.summarize(rows,queries)
    assert len(result['methods'])==6 and len(result['comparisons'])==4
    assert [(r['scope'],r['plus'],r['minus']) for r in result['comparisons']]==[
        (s,p,m) for s in oracle.SCOPES for p,m in oracle.PAIRS]
    assert all(r['questions']==77 and r['families']==24 for r in result['comparisons'])
    assert all(r['question_weighted']['delta']==pytest.approx(.1) for r in result['comparisons'])
    with pytest.raises(ValueError): oracle.summarize(rows[:-1],queries)


def test_subset_inventory_counts_without_reading_gold():
    rows=[{'scope':'given_document','method':'leaf_direct','candidate_global_indices':[0,1,2]},
          {'scope':'corpus_32','method':'dual_owner','candidate_global_indices':[0,1]}]
    result=oracle.subset_inventory(rows)
    assert result['enumerated_subsets_including_empty']==12 and result['unique_global_index_subsets']==8


def prepared_plan(tmp_path,monkeypatch):
    binding_path=tmp_path/'source';binding_path.write_text('frozen')
    binding={str(binding_path):oracle.native.digest(binding_path)}
    inventory={'enumerated_subsets_including_empty':311028,'unique_global_index_subsets':150461,'groups':[]}
    monkeypatch.setattr(oracle,'source_snapshot',lambda *a,**k:({},[],inventory,binding))
    monkeypatch.setattr(oracle,'evaluate',lambda *a,**k:pytest.fail('prepare must not calculate oracle scores'))
    args=SimpleNamespace(native_plan=tmp_path/'native-plan',native_run=tmp_path/'native-run',native_audit=tmp_path/'native-audit.json',
        output=tmp_path/'plan',run_output=tmp_path/'run')
    oracle.prepare(args)
    return args


def test_prepare_and_load_seal_without_scoring_or_create_run(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    plan,hashes=oracle.load_plan(args.output)
    assert plan['status']=='prepared_not_oracle_scored' and plan['oracle_scores_computed'] is False
    assert not args.run_output.exists()
    with pytest.raises(FileExistsError): oracle.prepare(args)


@pytest.mark.parametrize('mutation',['source','budget','subset_contract','extra_file'])
def test_changed_source_or_resealed_contract_rejected(tmp_path,monkeypatch,mutation):
    args=prepared_plan(tmp_path,monkeypatch)
    if mutation=='source': (tmp_path/'source').write_text('changed')
    elif mutation=='extra_file': (args.output/'extra').write_text('bad')
    else:
        path=args.output/'plan.json';plan=json.loads(path.read_text())
        if mutation=='budget': plan['config']['budget_bge_tokens']=2048
        else: plan['config']['enumerated_subsets_including_empty']=1
        path.write_text(json.dumps(plan),encoding='utf-8')
        (args.output/'plan_seal.json').write_text(json.dumps({'plan.json':oracle.native.digest(path)}),encoding='utf-8')
    with pytest.raises(ValueError): oracle.load_plan(args.output)


def test_timeout_run_saves_failure_without_any_complete_oracle(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    monkeypatch.setattr(oracle,'verified_source',lambda p:({},[]))
    def expired(*a,**k): raise TimeoutError()
    monkeypatch.setattr(oracle,'evaluate',expired)
    with pytest.raises(TimeoutError): oracle.run(SimpleNamespace(plan=args.output))
    failure=json.loads((args.run_output/'failure.json').read_text())
    assert failure['complete_oracle_available'] is False
    assert not (args.run_output/'summary.json').exists() and not (args.run_output/'public_aggregate.json').exists()
    with pytest.raises(FileExistsError): oracle.run(SimpleNamespace(plan=args.output))


def completed_run(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    queries,rows=aggregate_fixture()
    source={'prepared':{'queries':queries}}
    stats={'enumerated_subsets':311028,'duplicate_rejected_subsets':0,'overbudget_subsets':0,
        'feasible_subsets':311028,'token_cache_entries':150461}
    monkeypatch.setattr(oracle,'verified_source',lambda p:(source,[]))
    monkeypatch.setattr(oracle,'evaluate',lambda *a,**k:(deepcopy(rows),deepcopy(stats)))
    oracle.run(SimpleNamespace(plan=args.output))
    return args


def test_complete_run_audit_replays_all_public_metadata_and_requires_fixed_directory(tmp_path,monkeypatch):
    args=completed_run(tmp_path,monkeypatch)
    assert oracle.audit(SimpleNamespace(plan=args.output,run=args.run_output))['status']=='verified'
    with pytest.raises(ValueError,match='fixed output'):
        oracle.audit(SimpleNamespace(plan=args.output,run=tmp_path/'renamed-run'))
    with pytest.raises(FileExistsError): oracle.run(SimpleNamespace(plan=args.output))


@pytest.mark.parametrize('mutation',['denominator','cost','limits','paired','witness','elapsed','extra_file'])
def test_completed_audit_rejects_resealed_scientific_or_metadata_changes(tmp_path,monkeypatch,mutation):
    args=completed_run(tmp_path,monkeypatch)
    summary_path=args.run_output/'summary.json';summary=json.loads(summary_path.read_text())
    public_path=args.run_output/'public_aggregate.json';public=json.loads(public_path.read_text())
    if mutation=='denominator': public['question_count']=76
    elif mutation=='cost': public['api_calls']=1
    elif mutation=='limits': public['limits']=[]
    elif mutation=='paired': public['comparisons'][0]['question_weighted']['delta']=100
    elif mutation=='witness':
        path=args.run_output/'per_question.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['oracle_source_qualified_evidence_f1']=.12345
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    elif mutation=='elapsed': summary['elapsed_seconds']=1201
    elif mutation=='extra_file': (args.run_output/'unaccounted').write_text('extra')
    public_path.write_text(json.dumps(public),encoding='utf-8')
    summary['public']=public
    summary['output_sha256']={name:oracle.native.digest(args.run_output/name) for name in oracle.RUN_FILES[:-1]}
    summary_path.write_text(json.dumps(summary),encoding='utf-8')
    with pytest.raises(ValueError): oracle.audit(SimpleNamespace(plan=args.output,run=args.run_output))


def test_timeout_after_output_materialization_never_commits_complete_summary(tmp_path,monkeypatch):
    args=prepared_plan(tmp_path,monkeypatch)
    queries,rows=aggregate_fixture()
    monkeypatch.setattr(oracle,'verified_source',lambda p:({'prepared':{'queries':queries}},[]))
    monkeypatch.setattr(oracle,'evaluate',lambda *a,**k:(rows,{}))
    def check(deadline):
        if (args.run_output/'public_aggregate.json').exists(): raise TimeoutError()
    monkeypatch.setattr(oracle.native,'check_time',check)
    with pytest.raises(TimeoutError): oracle.run(SimpleNamespace(plan=args.output))
    assert not (args.run_output/'summary.json').exists()
    assert json.loads((args.run_output/'failure.json').read_text())['complete_oracle_available'] is False
    with pytest.raises(ValueError): oracle.audit(SimpleNamespace(plan=args.output,run=args.run_output))
