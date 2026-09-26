"""Synthetic aggregate-only publication gates; never reads actual oracle output."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"docs/research"))
import publish_qasper_candidate_oracle as publisher


def fixture():
    raw=deepcopy(publisher.FIXED)
    raw.update(input_binding_sha256='a'*64,shared_bootstrap_draws_sha256='b'*64,
        subset_accounting={'enumerated_subsets':311028,'duplicate_rejected_subsets':28,
            'overbudget_subsets':1000,'feasible_subsets':310000,'token_cache_entries':150000},methods=[],comparisons=[])
    for scope in publisher.experiment.SCOPES:
        for i,method in enumerate(publisher.experiment.METHODS):
            values=(.8,.2,.3,.5+.1*i,.7,.3+.1*i,400,300+i,2)
            raw['methods'].append({'scope':scope,'method':method,'questions':77,'families':24,
                'metrics':{m:{w:value for w in publisher.WEIGHTS} for m,value in zip(publisher.experiment.METRICS,values)},
                'oracle_empty_questions':10,'any_empty_reference_questions':7,
                'actual_witness_dominated_or_tied_questions':77})
        for plus,minus in publisher.experiment.PAIRS:
            raw['comparisons'].append({'scope':scope,'plus':plus,'minus':minus,'metric':publisher.experiment.CONFIG['primary_metric'],
                'questions':77,'families':24,'question_positive':20,'question_ties':50,'question_negative':7,
                'question_wins':20,'question_losses':7,
                **{w:{'delta':.1,'bootstrap_percentile_95':[-.01,.3]} for w in publisher.WEIGHTS}})
    return raw,deepcopy(publisher.AUDIT)


def test_whitelist_preserves_every_group_metric_pair_and_negative_ci():
    raw,audit=fixture();raw['private_question']='secret question'
    raw['methods'][0]['original_id']='private id'
    result=publisher.project(raw,audit,55.0)
    assert len(result['methods'])==6 and len(result['comparisons'])==4
    assert all(set(r['metrics'])==set(publisher.experiment.METRICS) for r in result['methods'])
    assert result['comparisons'][0]['family_balanced']['bootstrap_percentile_95']==[-.01,.3]
    assert 'secret question' not in json.dumps(result) and 'private id' not in json.dumps(result)
    assert result['publication']['new_api_cost_usd']=='0'
    assert result['publication']['prior_unknown_charge_resolved_by_this_diagnostic'] is False


@pytest.mark.parametrize('mutation',['partial_audit','audit_false','records','groups','pairs','partition',
    'cache','empty_reference','different_reference_denominator','dominance','gap','wins','delta','ci','nan','elapsed'])
def test_publication_rejects_incomplete_or_inconsistent_aggregates(mutation):
    raw,audit=fixture();elapsed=55
    if mutation=='partial_audit': audit['status']='partial'
    elif mutation=='audit_false': audit['all_subsets_enumerated']=False
    elif mutation=='records': raw['record_count']=461
    elif mutation=='groups': raw['methods'].pop()
    elif mutation=='pairs': raw['comparisons'].pop()
    elif mutation=='partition': raw['subset_accounting']['feasible_subsets']-=1
    elif mutation=='cache': raw['subset_accounting']['token_cache_entries']=150462
    elif mutation=='empty_reference': raw['methods'][0]['oracle_empty_questions']=6
    elif mutation=='different_reference_denominator': raw['methods'][0]['any_empty_reference_questions']=6
    elif mutation=='dominance': raw['methods'][0]['metrics']['oracle_source_qualified_evidence_f1']['question_weighted']=.1
    elif mutation=='gap': raw['methods'][0]['metrics']['oracle_minus_actual_f1']['question_weighted']=.9
    elif mutation=='wins': raw['comparisons'][0]['question_wins']=19
    elif mutation=='delta': raw['comparisons'][0]['family_balanced']['delta']=.2
    elif mutation=='ci': raw['comparisons'][0]['family_balanced']['bootstrap_percentile_95']=[.1,-.1]
    elif mutation=='nan': raw['methods'][0]['metrics']['actual_evidence_tokens']['question_weighted']=float('nan')
    elif mutation=='elapsed': elapsed=1200.1
    with pytest.raises(ValueError): publisher.project(raw,audit,elapsed)


def test_report_retains_all_weightings_empty_denominators_timing_and_boundaries():
    raw,audit=fixture();result=publisher.project(raw,audit,55.125)
    report=publisher.markdown(result)
    assert report.count('+0.100000 [-0.010000, +0.300000]')==8
    for fragment in ('311,028','150,461','55.125','Fraction','empty reference','NumPy/BLAS','未知收费','gold-guided','Answer F1'):
        assert fragment in report
    assert '[+0.000000, +0.000000]' in publisher.interval_text({'delta':0,'bootstrap_percentile_95':[0,0]})


def test_report_distinguishes_cross_zero_and_endpoint_zero_without_answerability_claim():
    raw,audit=fixture()
    for weight in publisher.WEIGHTS:
        raw['comparisons'][-1][weight]['bootstrap_percentile_95']=[0,.3]
    report=publisher.markdown(publisher.project(raw,audit,55))
    assert '双权重区间均跨 0' in report
    assert '双权重区间下界恰为 0' in report
    assert '所有预算合法子集的 F1 均为 0' in report
    assert 'oracle empty 数不等于不可回答问题数' in report


def publication_fixture(tmp_path,monkeypatch):
    raw,audit=fixture()
    plan_dir=tmp_path/'plan';plan_dir.mkdir()
    run_dir=tmp_path/'run';run_dir.mkdir()
    plan={'input_sha256':{},'input_binding_sha256':'a'*64,'run_output':str(run_dir)}
    (plan_dir/'plan.json').write_text(json.dumps(plan))
    (plan_dir/'plan_seal.json').write_text('{}')
    digest=publisher.experiment.native.digest
    plan_hashes={str(p):digest(p) for p in plan_dir.iterdir()}
    (run_dir/'public_aggregate.json').write_text(json.dumps(raw))
    (run_dir/'per_question.jsonl').write_text('PRIVATE CONTENT MUST NOT BE PARSED\n')
    summary={'schema':publisher.experiment.SCHEMA,'status':'completed','plan_sha256':plan_hashes,
        'input_sha256':{},'public':raw,'elapsed_seconds':55,
        'output_sha256':{name:digest(run_dir/name) for name in publisher.experiment.RUN_FILES[:-1]}}
    (run_dir/'summary.json').write_text(json.dumps(summary))
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps(audit))
    required=[Path(publisher.__file__).resolve(),Path(__file__).resolve(),Path(publisher.experiment.__file__).resolve(),
        Path(publisher.experiment.__file__).resolve().with_name('CANDIDATE_ORACLE_PROTOCOL_20260927.md'),
        audit_path,*plan_dir.iterdir(),*run_dir.iterdir()]
    bindings={str(path):digest(path) for path in required}
    release={'schema':publisher.RELEASE_SCHEMA,'root_release':publisher.RELEASE,'input_sha256':bindings}
    receipt=tmp_path/'source_binding.json';receipt.write_text(json.dumps(release))
    monkeypatch.setattr(publisher.experiment,'load_plan',lambda p:(deepcopy(plan),deepcopy(plan_hashes)))
    def forbidden(*a,**k): pytest.fail('publication must not perform oracle scoring or audit')
    for name in ('run','audit','evaluate','verified_source'): monkeypatch.setattr(publisher.experiment,name,forbidden)
    return SimpleNamespace(plan=plan_dir,run=run_dir,audit=audit_path,receipt=receipt,
        output_json=tmp_path/'public.json',output_report=tmp_path/'report.md')


def test_publication_reads_only_released_aggregate_and_refuses_overwrite(tmp_path,monkeypatch):
    args=publication_fixture(tmp_path,monkeypatch)
    assert publisher.publish(args)['status']=='published_complete_aggregate'
    result=json.loads(args.output_json.read_text())
    assert result['methods'][0]['metrics']['oracle_source_qualified_evidence_f1']['question_weighted']==.5
    assert 'PRIVATE' not in args.output_json.read_text()
    with pytest.raises(FileExistsError): publisher.publish(args)


@pytest.mark.parametrize('mutation',['release','missing_binding','changed_source','extra_run_file','unsealed_public'])
def test_root_release_and_exact_output_seals_gate_publication(tmp_path,monkeypatch,mutation):
    args=publication_fixture(tmp_path,monkeypatch)
    release=json.loads(args.receipt.read_text())
    if mutation=='release': release['root_release']='not released'
    elif mutation=='missing_binding': release['input_sha256'].pop(str(args.audit))
    elif mutation=='changed_source': args.audit.write_text('{}')
    elif mutation=='extra_run_file': (args.run/'failure.json').write_text('{}')
    elif mutation=='unsealed_public':
        p=args.run/'public_aggregate.json';raw=json.loads(p.read_text());raw['question_count']=76;p.write_text(json.dumps(raw))
        release['input_sha256'][str(p)]=publisher.experiment.native.digest(p)
    args.receipt.write_text(json.dumps(release))
    with pytest.raises(ValueError): publisher.publish(args)
    assert not args.output_json.exists() and not args.output_report.exists()
