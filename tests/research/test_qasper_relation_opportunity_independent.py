"""Synthetic behavior and sealed-I/O tests; no official selector import."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[2]/'docs/research/verify_qasper_relation_opportunity_independently.py'
spec=importlib.util.spec_from_file_location('independent_relation',SOURCE)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class Tokenizer:
    def encode(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':True,'truncation':False}
        return list(range(len(text)+2))


def make_case(n=5,ranking=None,labels=None):
    units=[dict(unit_id=f'u{i}',order=i,kind='paragraph',start=i,end=i+1,text=f'text{i}',native_text=f'text{i}') for i in range(n)]
    return dict(family_id='f',doc_id='d',question_id='q',units=units,candidates=list(range(n)),
        ranking=ranking or list(range(n)),labels=labels or ['yes']*n,
        baseline_selected=[],baseline_tokens=0,baseline_pack_sha256=hashlib.sha256(b'').hexdigest())


def with_baseline(case):
    result=v.replay(case,0,lambda indices:len(v.render(case,indices))+2 if indices else 0)
    case.update(baseline_selected=result['selected_indices'],baseline_tokens=result['actual_tokens'],baseline_pack_sha256=result['pack_sha256'])
    return case


def chosen(result):return [s['candidate_index'] for s in result['trace'] if s['accepted']]


def test_direction_and_dynamic_chain():
    case=make_case(ranking=[2,4,3,1,0])
    assert chosen(v.replay(case,0,lambda _:10))==[2,4,3]
    linked=v.replay(case,3,lambda _:10)
    assert chosen(linked)==[2,1,0]
    assert [x['relation_bonus'] for x in linked['trace']]==[0,1,1]
    assert linked['trace'][1]['active_triggers']==[[1,2]]
    assert linked['trace'][2]['selected_before']==[1,2]
    reverse=make_case(ranking=[0,4,3,2,1])
    assert chosen(v.replay(reverse,15,lambda _:10))==[0,4,3]
    assert v.replay(reverse,15,lambda _:10)['trace'][1]['relation_bonus']==0


def test_no_is_never_rescued_and_unknown_bonus_is_boolean():
    case=make_case(ranking=[1,4,3,2,0],labels=['no','yes','unknown','yes','yes'])
    assert (0,1) not in v.edge_domain(case)
    assert 0 not in v.replay(case,7,lambda _:1)['selected_indices']
    case=make_case(ranking=[1,4,3,2,0],labels=['unknown','yes','yes','yes','yes'])
    result=v.replay(case,15,lambda _:1)
    assert chosen(result)==[1,0,4]
    assert result['trace'][1]['base_score']==.5 and result['trace'][1]['relation_bonus']==1


def test_exact_native_duplicate_and_nonadditive_whole_pack_skip():
    case=make_case(4,ranking=[2,0,1,3])
    case['units'][0]['native_text']=case['units'][2]['native_text']
    counts={():0,(2,):800,(1,2):1025,(2,3):900}
    seen=[]
    result=v.replay(case,0,lambda xs:seen.append(xs) or counts[xs])
    assert chosen(result)==[2,3]
    assert [r['action'] for r in result['trace']]==['select','skip_exact_native_duplicate','skip_evidence_budget','select']
    assert (0,2) not in seen and (1,2) in seen
    assert result['actual_tokens']==900


def test_empty_no_special_tokens_and_source_order_render():
    case=make_case(2,ranking=[1,0],labels=['no','no'])
    def fail(*args,**kwargs):raise AssertionError('empty pack must not tokenize')
    with_baseline(case)
    masks,questions,aggregate=v.reconstruct([case],SimpleNamespace(encode=fail))
    assert masks[0]['actual_tokens']==0 and masks[0]['trace']==[]
    assert questions[0]['mask_count']==1 and aggregate['unchanged_masks']==1
    assert v.render(make_case(2),[1,0])=='[u0]\ntext0\n\n[u1]\ntext1'


@pytest.mark.parametrize('bad',[True,-1,16])
def test_mask_domain_rejects(bad):
    with pytest.raises(ValueError):v.replay(make_case(),bad,lambda _:1)


@pytest.mark.parametrize('value',[True,-1,1.5,float('nan')])
def test_invalid_whole_pack_count(value):
    with pytest.raises(ValueError):v.replay(make_case(),0,lambda _:value)


def test_core_whitelist_and_deadline():
    case=make_case();case['question']='must not enter core'
    with pytest.raises(ValueError,match='whitelist'):v.replay(case,0,lambda _:1)
    case=make_case();case['units'][0]['answer']='forbidden'
    with pytest.raises(ValueError):v.replay(case,0,lambda _:1)
    with pytest.raises(TimeoutError):v.replay(make_case(),0,lambda _:1,deadline=-1)


def test_all_masks_and_whole_trace_inventory():
    case=with_baseline(make_case(4,ranking=[2,3,1,0]))
    masks,questions,aggregate=v.reconstruct([case],Tokenizer())
    assert [r['mask'] for r in masks]==list(range(8))
    assert [len(r['active_edges']) for r in masks]==[0,1,1,2,1,2,2,3]
    assert questions[0]['mask_count']==8
    assert aggregate['changed_masks']+aggregate['unchanged_masks']==8
    assert all(len(r['selected_indices'])==3 for r in masks)


def write(path,value):path.write_bytes(v.encode(value)+b'\n')
def lines(path,rows):path.write_bytes(b''.join(v.encode(r)+b'\n' for r in rows))


@pytest.fixture
def sealed(tmp_path,monkeypatch):
    plan=tmp_path/'plan';run=tmp_path/'run';execution=tmp_path/'execution'
    for p in (plan,run,execution):p.mkdir()
    cases=[]
    for m,num in [(0,31),(1,19),(2,8),(3,10),(4,6),(5,1),(6,1),(7,1)]:
        for _ in range(num):
            q=len(cases);case=with_baseline(make_case(m+1,ranking=list(reversed(range(m+1)))))
            case.update(family_id=f'f{q%24}',doc_id=f'd{q%24}',question_id=f'q{q}')
            cases.append(case)
    masks,questions,aggregate=v.reconstruct(cases,Tokenizer())
    source=tmp_path/'source.txt';source.write_text('fixed source')
    bindings={str(source):v.sha(source)}
    inventory={'eligible_edge_histogram':v.hist(len(v.edge_domain(c)) for c in cases)}
    config={'run_output':str(run),'input_sha256':bindings,'input_binding_sha256':hashlib.sha256(v.encode(bindings)).hexdigest(),
            'inventory':inventory,'specification':{'fixed':True},'tokenizer':'synthetic'}
    write(plan/'cases.json',cases);write(plan/'plan.json',config)
    write(plan/'seal.json',{n:v.sha(plan/n) for n in ('plan.json','cases.json')})
    protocol={'inventory':inventory,'specification':config['specification'],'runtime':{},'limits':[]}
    public_protocol=tmp_path/'protocol.json';write(public_protocol,protocol)
    monkeypatch.setattr(v,'PROTOCOL',public_protocol);monkeypatch.setattr(v,'PROTOCOL_SHA',v.sha(public_protocol))
    monkeypatch.setattr(v,'PLAN_SHA',v.sha(plan/'plan.json'))
    public={**aggregate,**protocol,'api_calls':0,**{k:False for k in ('quality_metrics_computed','best_mask_selected','gpu_used','encoder_model_loaded','qa_sidecar_opened','private_ids_in_public_output')}}
    lines(run/'per_mask.jsonl',masks);lines(run/'per_question.jsonl',questions);write(run/'public_aggregate.json',public)
    summary={'status':'completed','all_masks_available':True,'output_sha256':{n:v.sha(run/n) for n in v.RUN_FILES-{'summary.json'}}}
    write(run/'summary.json',summary)
    def reseal():
        summary=json.loads((run/'summary.json').read_bytes())
        summary['output_sha256']={n:v.sha(run/n) for n in v.RUN_FILES-{'summary.json'}};write(run/'summary.json',summary)
        audit={'status':'verified_complete','questions':77,'masks':501,'all_masks_and_reference_traces_recomputed':True,
               'all_direct_input_hashes_unchanged':True,'input_binding_sha256':config['input_binding_sha256'],
               'run_sha256':{str(run/n):v.sha(run/n) for n in v.RUN_FILES}}
        write(execution/'audit.json',audit)
        completed={'status':'completed_and_audited','all_direct_bindings_unchanged':True,'plan_sha256':v.PLAN_SHA,
                   'summary_sha256':v.sha(run/'summary.json'),'audit_sha256':v.sha(execution/'audit.json')}
        write(execution/'completed.json',completed)
    reseal()
    args=SimpleNamespace(plan=plan,run=run,completion=execution/'completed.json',output=tmp_path/'verification')
    return args,reseal


def test_complete_synthetic_verification_and_single_use(sealed):
    args,_=sealed;result=v.verify(args,lambda _:Tokenizer())
    assert result['questions']==77 and result['families']==24 and result['masks']==501
    assert result['all_behavioral_aggregate_fields_recomputed']
    assert result['official_selector_or_aggregator_imported'] is False
    with pytest.raises(ValueError,match='new and separate'):v.verify(args,lambda _:Tokenizer())


@pytest.mark.parametrize('target',['trace','tokens','mask','question','aggregate','metadata','missing','extra','partial','receipt','source'])
def test_resealed_semantic_or_complete_input_tamper_fails(sealed,target):
    args,reseal=sealed;run=args.run
    if target in ('trace','tokens','mask'):
        rows=[json.loads(l) for l in (run/'per_mask.jsonl').read_bytes().splitlines()]
        if target=='trace':rows[0]['trace'][0]['base_score']+=.5
        elif target=='tokens':rows[0]['actual_tokens']+=1
        else:rows[0]['mask']=1
        lines(run/'per_mask.jsonl',rows);reseal()
    elif target=='question':
        rows=[json.loads(l) for l in (run/'per_question.jsonl').read_bytes().splitlines()];rows[0]['distinct_pack_count']+=1
        lines(run/'per_question.jsonl',rows);reseal()
    elif target in ('aggregate','metadata'):
        p=run/'public_aggregate.json';data=json.loads(p.read_bytes())
        if target=='aggregate':data['changed_masks']+=1
        else:data['quality_metrics_computed']=0
        write(p,data);reseal()
    elif target=='missing':(run/'per_mask.jsonl').unlink()
    elif target=='extra':write(run/'extra.json',{})
    elif target=='partial':
        p=run/'summary.json';data=json.loads(p.read_bytes());data['all_masks_available']=False;write(p,data);reseal()
    elif target=='receipt':
        p=args.completion;data=json.loads(p.read_bytes());data['status']='running';write(p,data)
    elif target=='source':(args.plan.parent/'source.txt').write_text('changed')
    with pytest.raises(ValueError):v.verify(args,lambda _:Tokenizer())
    assert not args.output.exists()


def test_source_toctou_rejected(sealed):
    args,_=sealed
    def factory(_):
        (args.plan.parent/'source.txt').write_text('changed after source verification')
        return Tokenizer()
    with pytest.raises(ValueError,match='changed during'):v.verify(args,factory)
    assert not args.output.exists()


def test_no_official_imports_or_quality_functions():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    assert not any(name and (name.startswith('run_qasper') or name.startswith('qasper_')) for name in imports)
    assert not any(isinstance(node,ast.Attribute) and node.attr in ('select','evaluate','aggregate','replay_policy','answer_f1') for node in ast.walk(tree))
