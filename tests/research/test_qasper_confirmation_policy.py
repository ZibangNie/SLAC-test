"""Synthetic metadata only: no real QA, text, corpus files, models or APIs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'docs/research/qasper_confirmation_policy.py'
spec=importlib.util.spec_from_file_location('confirmation_policy',PATH)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def row(doc,split='validation',source_id=None,family=None,body=None,**extra):
    suffix=sum(ord(c) for c in doc)
    return {'source':'qasper','split':split,'doc_id':doc,'source_id':source_id or f'2101.{suffix:05d}',
            'family_id':family or 'family-'+doc,'normalized_body_sha256':body or hashlib.sha256(doc.encode()).hexdigest(),**extra}


def exposure(**overrides):
    return {**{k:set() for k in p.EXPOSURE_FIELDS},**overrides}


def edge(a,b,severity='moderate',scope='canonical_train_validation'):
    return {'left_doc_id':a,'right_doc_id':b,'severity':severity,'scope':scope}


def review(a,b,value=True):
    return {'left_doc_id':a,'right_doc_id':b,'explicit_material_reuse':value}


def evaluate(rows,known=None,prior=None,edges=(),reviews=()):
    return p.evaluate_metadata(rows,exposure() if known is None else known,set() if prior is None else prior,edges,reviews)


def states(result):return {r['doc_id']:r['status'] for r in result['validation_ledger']}


def test_all_validation_sorted_retained_and_proposal_only():
    result=evaluate([row('train','train'),row('b'),row('a')])
    assert [x['doc_id'] for x in result['validation_ledger']]==['a','b']
    assert result['proposed_cohort_doc_ids']==['a','b']
    public=result['public_aggregate']
    assert public['canonical_documents']==3 and public['canonical_validation_documents']==2
    assert public['validation_status_counts']=={'quarantine':0,'needs_review':0,'operational_candidate_proposed':2}
    assert all(public[k] is False for k in p.FLAGS)
    assert public['source_rows_removed']==public['source_family_ids_modified']==0


def test_transitive_graph_keeps_train_bridge_before_quarantine():
    rows=[row('t','train',family='shared'),row('a',family='shared'),row('b'),row('c')]
    result=evaluate(rows,edges=[edge('a','b',scope='canonical_validation_validation')],reviews=[review('b','c')])
    assert states(result)==dict.fromkeys(['a','b','c'],'quarantine')
    assert result['proposed_cohort_doc_ids']==[]
    component=next(c for c in result['components'] if 't' in c['member_doc_ids'])
    assert component['member_doc_ids']==['a','b','c','t']
    assert {'declared_family','canonical_lexical_moderate','explicit_material_reuse','touches_canonical_train'}<=set(component['component_reasons'])
    assert [r['family_id'] for r in result['validation_ledger']]==['shared','family-b','family-c']


@pytest.mark.parametrize('field', ['family_id','normalized_body_sha256'])
def test_exact_family_or_body_links_train(field):
    a,b=row('t','train'),row('v');b[field]=a[field]
    result=evaluate([a,b]);assert states(result)['v']=='quarantine'
    assert result['public_aggregate']['validation_connected_to_train']==1


def test_arxiv_version_link_preserves_exact_raw_case_and_version():
    a=row('t','train',source_id='arXiv:2101.12345v1');b=row('v',source_id='https://arxiv.org/abs/2101.12345v12')
    result=evaluate([a,b]);r=result['validation_ledger'][0]
    assert r['source_id']==b['source_id'] and r['strict_arxiv_base']=='2101.12345' and r['strict_arxiv_version']==12
    assert r['status']=='quarantine'
    assert 'strict_arxiv_base' in r['component_reasons']


@pytest.mark.parametrize('source,expected',[
    ('2101.12345',('2101.12345',None)),('2101.12345v2',('2101.12345',2)),
    ('arxiv:2101.12345v1',('2101.12345',1)),('https://arxiv.org/abs/2101.12345',('2101.12345',None)),
    ('http://arxiv.org/abs/2101.12345',('2101.12345',None)),('hep-th/9701001v3',('hep-th/9701001',3)),
    ('math.AG/0601001',('math.AG/0601001',None)),('ARXIV:2101.12345',None),
    ('https://arxiv.org.evil/abs/2101.12345',None),('https://arxiv.org/abs/2101.12345?x=1',None),
    ('2100.12345',None),('2101.12345V2',None),('2101.12345v01',None),('2101.12345.pdf',None),
    ('private-id',None),('unknown/9701001',None),
])
def test_strict_arxiv_policy(source,expected):assert p.strict_arxiv_identity(source)==expected


def test_ambiguous_source_retained_and_propagates_needs_review():
    result=evaluate([row('a',source_id='publisher:unrecognized',family='shared'),row('b',family='shared'),row('c')])
    assert states(result)=={'a':'needs_review','b':'needs_review','c':'operational_candidate_proposed'}
    assert result['proposed_cohort_doc_ids']==['c']
    assert result['validation_ledger'][1]['source_identity_needs_review'] is False
    assert result['validation_ledger'][1]['component_source_review_required'] is True
    assert result['public_aggregate']['unrecognized_source_identity_documents']==1


def test_quarantine_precedes_ambiguous_review_without_losing_reason():
    result=evaluate([row('a',source_id='unrecognized')],prior={'a'})
    r=result['validation_ledger'][0]
    assert r['status']=='quarantine' and r['component_source_review_required'] is True
    assert set(r['component_reasons'])=={'touches_known_exposure','unrecognized_source_identity'}


@pytest.mark.parametrize('key,column', [('doc_ids','doc_id'),('source_ids','source_id'),('family_ids','family_id'),('normalized_body_sha256','normalized_body_sha256')])
def test_all_explicit_exposure_namespaces_propagate_to_component(key,column):
    a=row('a',family='shared');b=row('b',family='shared')
    result=evaluate([a,b],known=exposure(**{key:{a[column]}}))
    assert set(states(result).values())=={'quarantine'}
    assert result['public_aggregate']['validation_connected_to_known_exposure']==2


def test_known_arxiv_other_version_not_in_index_still_quarantines():
    result=evaluate([row('a',source_id='2101.12345v2')],known=exposure(source_ids={'arxiv:2101.12345v1'}))
    assert states(result)['a']=='quarantine'
    assert result['validation_ledger'][0]['direct_exposure_reasons']==['known_arxiv_base']


def test_out_of_index_exposure_allowed_and_exact_identity_not_casefolded():
    result=evaluate([row('a',family='MixedCase')],known=exposure(doc_ids={'other-dataset:outside'},source_ids={'other-dataset:source'},family_ids={'mixedcase'}))
    assert states(result)['a']=='operational_candidate_proposed'
    assert result['validation_ledger'][0]['family_id']=='MixedCase'


def test_development_bool_and_prior_only_validation():
    result=evaluate([row('a',development_exposed=True),row('b',development_exposed=False),row('c')],prior={'b'})
    assert states(result)=={'a':'quarantine','b':'quarantine','c':'operational_candidate_proposed'}
    assert result['validation_ledger'][2]['declared_development_exposed'] is None
    for prior in [{'absent'},{'t'}]:
        with pytest.raises(ValueError):evaluate([row('t','train'),row('v')],prior=prior)


def test_moderate_high_link_low_does_not_and_false_review_is_not_clearance():
    rows=[row('t','train'),row('low'),row('moderate'),row('high'),row('false')]
    result=evaluate(rows,edges=[edge('t','low','low'),edge('t','moderate'),edge('t','high','high')],reviews=[review('t','false',False)])
    assert states(result)=={'false':'operational_candidate_proposed','high':'quarantine','low':'operational_candidate_proposed','moderate':'quarantine'}
    assert result['public_aggregate']['source_review_explicit_reuse_records']==0
    assert result['public_aggregate']['human_reviewed'] is False
    assert result['public_aggregate']['overlap_input_severity_counts']=={'low':1,'moderate':1,'high':1}


def test_determinism_input_order_endpoint_orientation_and_no_mutation():
    rows=[row('t','train'),row('a'),row('b'),row('c')]
    known=exposure(doc_ids={'outside','b'});e=[edge('t','a'),edge('b','c',scope='canonical_validation_validation')];r=[review('a','b')]
    original=deepcopy((rows,known,e,r))
    first=evaluate(rows,known,edges=e,reviews=r)
    second=evaluate(list(reversed(rows)),known,edges=[{**x,'left_doc_id':x['right_doc_id'],'right_doc_id':x['left_doc_id']} for x in reversed(e)],reviews=[review('b','a')])
    assert first==second and (rows,known,e,r)==original
    component=first['components'][0]
    expected='component-'+hashlib.sha256(p._canonical({'schema':p.SCHEMA,'doc_ids':component['member_doc_ids']})).hexdigest()
    assert component['component_key']==expected


def test_duplicate_evidence_preserved_as_metadata_but_not_repeated_graph_link():
    result=evaluate([row('t','train'),row('v')],edges=[edge('t','v'),edge('v','t')],reviews=[review('t','v'),review('v','t')])
    public=result['public_aggregate']
    assert public['overlap_input_records']==2 and public['source_review_input_records']==2
    assert public['linking_evidence_counts']['canonical_lexical_moderate']==1
    assert public['linking_evidence_counts']['explicit_material_reuse']==1


def test_input_digest_distinguishes_declared_false_absent_and_case():
    a=evaluate([row('v')]);b=evaluate([row('v',development_exposed=False)]);c=evaluate([row('v',family='FAMILY-v')])
    assert len({r['input_metadata_sha256'] for r in (a,b,c)})==3


def test_public_fixed_keys_have_no_identifiers_digests_or_nested_identity_lists():
    result=evaluate([row('PRIVATE_DOC_123',family='PRIVATE_FAMILY_456',source_id='2101.99999')])
    public=result['public_aggregate'];text=json.dumps(public)
    for value in ('PRIVATE_DOC_123','PRIVATE_FAMILY_456','2101.99999',result['input_metadata_sha256'],result['components'][0]['component_key']):assert value not in text
    def walk(value):
        if isinstance(value,dict):
            for v in value.values():walk(v)
        else:assert type(value) in (str,int,bool)
    walk(public)
    assert public.keys()==evaluate([row('another')])['public_aggregate'].keys()


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(source='other'),lambda r:r.update(split='test'),lambda r:r.update(split=None),
    lambda r:r.update(doc_id=''),lambda r:r.update(doc_id=' white'),lambda r:r.update(doc_id='x\ny'),
    lambda r:r.update(normalized_body_sha256='A'*64),lambda r:r.update(normalized_body_sha256='x'),
    lambda r:r.update(development_exposed=1),lambda r:r.update(development_exposed=None),
    lambda r:r.update(qas=[]),lambda r:r.update(query_count=1),lambda r:r.update(quality=0.5),lambda r:r.pop('family_id'),
])
def test_invalid_closed_index_schema_fails(mutate):
    r=row('v');mutate(r)
    with pytest.raises(ValueError):evaluate([r])


def test_duplicate_doc_id_empty_and_no_validation_rejected():
    for rows in ([],[row('v'),row('v')],[row('t','train')]):
        with pytest.raises(ValueError):evaluate(rows)


@pytest.mark.parametrize('known',[
    {},{**exposure(),'extra':set()},exposure(doc_ids=['v']),exposure(doc_ids={1}),exposure(normalized_body_sha256={'invalid'}),
])
def test_invalid_exposure_schema(known):
    with pytest.raises(ValueError):evaluate([row('v')],known=known)


@pytest.mark.parametrize('bad',[
    edge('t','v',severity='unknown'),edge('t','v',scope='canonical_validation_validation'),edge('absent','v'),edge('v','v'),
    {**edge('t','v'),'score':0.4},{k:v for k,v in edge('t','v').items() if k!='scope'},
])
def test_invalid_overlap_schema(bad):
    with pytest.raises(ValueError):evaluate([row('t','train'),row('v')],edges=[bad])


@pytest.mark.parametrize('bad',[
    review('t','v',1),review('t','v',None),review('absent','v'),review('v','v'),{**review('t','v'),'title':'private'},
])
def test_invalid_review_schema(bad):
    with pytest.raises(ValueError):evaluate([row('t','train'),row('v')],reviews=[bad])


def test_conflicting_explicit_reuse_is_not_silently_resolved():
    with pytest.raises(ValueError):evaluate([row('t','train'),row('v')],reviews=[review('t','v'),review('v','t',False)])


def test_training_training_scope_can_bridge_then_reach_validation():
    rows=[row('t1','train'),row('t2','train'),row('v')]
    result=evaluate(rows,edges=[edge('t1','t2','high','canonical_train_train'),edge('v','t2')])
    assert states(result)['v']=='quarantine'
    assert result['components'][0]['member_doc_ids']==['t1','t2','v']


def test_failures_do_not_echo_private_identity():
    with pytest.raises(ValueError) as error:evaluate([row('PRIVATE\nSECRET')])
    assert 'PRIVATE' not in str(error.value) and 'SECRET' not in str(error.value)
