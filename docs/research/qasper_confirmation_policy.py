"""Pure metadata policy for a proposed Qasper confirmation cohort.

No I/O, QA, text, model, training or API access occurs in this module. The caller
must establish source completeness/provenance and any history of machine decoding
QA containers. Absence of a metadata match is not evidence of independence.

Input contract for evaluate_metadata:
* index_rows: list/tuple of exact required fields source, split, doc_id, source_id,
  family_id, normalized_body_sha256, plus optional development_exposed (bool).
  Only canonical source='qasper' and split='train'/'validation' are accepted.
* known_exposure: exactly doc_ids/source_ids/family_ids/normalized_body_sha256,
  each a set/frozenset of exact strings. Out-of-index history is permitted.
* prior_development_ids: set/frozenset; every ID must be in validation.
* overlap_edges: list/tuple of left_doc_id, right_doc_id, severity, scope.
  Severity is low/moderate/high. Scope is canonical_train_train,
  canonical_train_validation or canonical_validation_validation, checked against
  the endpoints. Only moderate/high link components; low records are retained.
* source_reviews: list/tuple of left_doc_id, right_doc_id,
  explicit_material_reuse (strict bool). False asserts no positive reuse finding;
  it does not certify independence. Conflicting reviews for a pair are rejected.

Every identity, original case/version and component membership stays in the
private result fields. Only public_aggregate is safe for aggregate publication.
The graph is an operational quarantine device, not a new ground-truth family map.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re

SCHEMA = 'slac-qasper-confirmation-metadata-policy-v1'
ROW_FIELDS = frozenset({'source','split','doc_id','source_id','family_id','normalized_body_sha256'})
EXPOSURE_FIELDS = frozenset({'doc_ids','source_ids','family_ids','normalized_body_sha256'})
EDGE_FIELDS = frozenset({'left_doc_id','right_doc_id','severity','scope'})
REVIEW_FIELDS = frozenset({'left_doc_id','right_doc_id','explicit_material_reuse'})
LINK_REASONS = ('declared_family','normalized_body_hash','strict_arxiv_base',
                'canonical_lexical_moderate','canonical_lexical_high','explicit_material_reuse')
EXPOSURE_REASONS = ('prior_development_id','index_development_exposed','known_doc_id',
                    'known_source_id','known_family_id','known_normalized_body_hash','known_arxiv_base')
STATUS = ('quarantine','needs_review','operational_candidate_proposed')
FLAGS = {'new_qa_read':False,'api_called':False,'training_started':False,
         'cleared':False,'cleared_for_training':False,'cohort_admitted':False,
         'human_reviewed':False,'independence_established':False,'model_contamination_free':False}
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_MODERN = re.compile(r'(?P<base>[0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5})(?:v(?P<version>[1-9][0-9]{0,8}))?\Z')
_LEGACY = re.compile(r'(?P<archive>[a-z][a-z-]{0,19})(?:\.(?P<class>[A-Z]{2}))?/(?P<number>[0-9]{2}(?:0[1-9]|1[0-2])[0-9]{3})(?:v(?P<version>[1-9][0-9]{0,8}))?\Z')
_ARCHIVES = frozenset({'acc-phys','adap-org','alg-geom','ao-sci','astro-ph','atom-ph',
    'bayes-an','chao-dyn','chem-ph','cmp-lg','cond-mat','cs','dg-ga','funct-an','gr-qc',
    'hep-ex','hep-lat','hep-ph','hep-th','math','math-ph','mtrl-th','nlin','nucl-ex',
    'nucl-th','patt-sol','physics','plasm-ph','q-alg','q-bio','quant-ph','solv-int','supr-con'})


def _fail(message):
    # Do not include potentially private identity values in exception messages.
    raise ValueError(message)


def _identity(value):
    if (type(value) is not str or not 0 < len(value) <= 1024
            or not value.isprintable() or any(c.isspace() for c in value)):
        _fail('identity must be a nonempty exact printable string without whitespace')
    return value


def _body_hash(value):
    if type(value) is not str or not _HASH.fullmatch(value):
        _fail('normalized body hash must be lowercase SHA-256')
    return value


def _object(value, required, optional=frozenset()):
    if type(value) is not dict or not required <= value.keys() or value.keys()-required-optional:
        _fail('metadata object fields do not match the closed schema')


def _sequence(value):
    if type(value) not in (list,tuple):
        _fail('metadata records must be a list or tuple')


def _string_set(value, validator=_identity):
    if type(value) not in (set,frozenset):
        _fail('exposure identities must be a set or frozenset')
    return {validator(v) for v in value}


def _canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def strict_arxiv_identity(source_id):
    """Return syntactically recognized (base, version); never query a registry.

    The original string is never normalized in the ledger or exact exposure
    matching. The base is used only for conservative version-family linking.
    Unsupported/ambiguous URLs, suffixes and case variants return None.
    Recognition is an operational syntax check, not proof of a real publication.
    """
    value = _identity(source_id)
    for prefix in ('https://arxiv.org/abs/','http://arxiv.org/abs/','arxiv:','arXiv:'):
        if value.startswith(prefix):
            value=value[len(prefix):]
            break
    match=_MODERN.fullmatch(value)
    if match:
        return match['base'], int(match['version']) if match['version'] else None
    match=_LEGACY.fullmatch(value)
    if match and match['archive'] in _ARCHIVES:
        base=match['archive'] + ('.'+match['class'] if match['class'] else '') + '/' + match['number']
        return base, int(match['version']) if match['version'] else None
    return None


def evaluate_metadata(index_rows, known_exposure, prior_development_ids, overlap_edges, source_reviews):
    """Return the complete sorted validation ledger, private proposal and counts.

    Quarantine takes precedence over needs_review, which takes precedence over
    operational_candidate_proposed. Unknown source syntax propagates a review
    requirement to the entire connected component. No source row is dropped,
    no existing family_id is rewritten, and no candidate is admitted by this
    function. Callers should publish only public_aggregate, never the full return.
    """
    _sequence(index_rows);_sequence(overlap_edges);_sequence(source_reviews)
    _object(known_exposure,EXPOSURE_FIELDS)
    exposure={k:_string_set(known_exposure[k],_body_hash if k=='normalized_body_sha256' else _identity) for k in EXPOSURE_FIELDS}
    prior=_string_set(prior_development_ids)
    rows={}
    for raw in index_rows:
        _object(raw,ROW_FIELDS,{'development_exposed'})
        if raw['source']!='qasper' or type(raw['source']) is not str or raw['split'] not in ('train','validation') or type(raw['split']) is not str:
            _fail('only canonical qasper train/validation metadata is allowed')
        row={k:_identity(raw[k]) for k in ('doc_id','source_id','family_id')}
        row.update(source='qasper',split=raw['split'],normalized_body_sha256=_body_hash(raw['normalized_body_sha256']))
        if 'development_exposed' in raw:
            if type(raw['development_exposed']) is not bool:_fail('development_exposed must be a strict bool')
            row['development_exposed']=raw['development_exposed']
        if row['doc_id'] in rows:_fail('duplicate canonical doc_id')
        rows[row['doc_id']]=row
    if not rows:_fail('canonical metadata must not be empty')
    validation={d for d,r in rows.items() if r['split']=='validation'}
    if not validation:_fail('canonical validation denominator must not be empty')
    if not prior<=validation:_fail('prior development identities must be present in validation')

    def endpoints(raw):
        left,right=_identity(raw['left_doc_id']),_identity(raw['right_doc_id'])
        if left==right or left not in rows or right not in rows:_fail('edge endpoints must be distinct canonical documents')
        return tuple(sorted((left,right)))

    edges=[]
    for raw in overlap_edges:
        _object(raw,EDGE_FIELDS);left,right=endpoints(raw)
        severity=raw['severity']
        if type(severity) is not str or severity not in ('low','moderate','high'):_fail('invalid canonical overlap severity')
        expected_scope='canonical_'+'_'.join(sorted((rows[left]['split'],rows[right]['split'])))
        if type(raw['scope']) is not str or raw['scope']!=expected_scope:_fail('overlap scope disagrees with canonical endpoints')
        edges.append({'left_doc_id':left,'right_doc_id':right,'severity':severity,'scope':raw['scope']})
    reviews=[];review_values={}
    for raw in source_reviews:
        _object(raw,REVIEW_FIELDS);left,right=endpoints(raw);value=raw['explicit_material_reuse']
        if type(value) is not bool:_fail('explicit material reuse must be a strict bool')
        if (left,right) in review_values and review_values[left,right]!=value:_fail('conflicting explicit reuse metadata')
        review_values[left,right]=value
        reviews.append({'left_doc_id':left,'right_doc_id':right,'explicit_material_reuse':value})
    edges.sort(key=_canonical);reviews.sort(key=_canonical)
    source_identity={d:strict_arxiv_identity(r['source_id']) for d,r in rows.items()}
    known_bases={parsed[0] for value in exposure['source_ids'] if (parsed:=strict_arxiv_identity(value)) is not None}
    direct_exposure={}
    for doc,row in rows.items():
        reasons=[]
        if doc in prior:reasons.append('prior_development_id')
        if row.get('development_exposed') is True:reasons.append('index_development_exposed')
        for key,column,reason in (('doc_ids','doc_id','known_doc_id'),('source_ids','source_id','known_source_id'),
                                  ('family_ids','family_id','known_family_id'),('normalized_body_sha256','normalized_body_sha256','known_normalized_body_hash')):
            if row[column] in exposure[key]:reasons.append(reason)
        if source_identity[doc] is not None and source_identity[doc][0] in known_bases:reasons.append('known_arxiv_base')
        direct_exposure[doc]=sorted(reasons)

    parent={d:d for d in rows};links=set()
    def find(doc):
        while parent[doc]!=doc:
            parent[doc]=parent[parent[doc]];doc=parent[doc]
        return doc
    def connect(left,right,reason):
        left,right=sorted((left,right));links.add((left,right,reason))
        a,b=find(left),find(right)
        if a!=b:parent[max(a,b)]=min(a,b)
    for field,reason in (('family_id','declared_family'),('normalized_body_sha256','normalized_body_hash')):
        groups=defaultdict(list)
        for doc,row in rows.items():groups[row[field]].append(doc)
        for group in groups.values():
            group.sort()
            for doc in group[1:]:connect(group[0],doc,reason)
    groups=defaultdict(list)
    for doc,identity in source_identity.items():
        if identity is not None:groups[identity[0]].append(doc)
    for group in groups.values():
        group.sort()
        for doc in group[1:]:connect(group[0],doc,'strict_arxiv_base')
    for edge in edges:
        if edge['severity'] in ('moderate','high'):connect(edge['left_doc_id'],edge['right_doc_id'],'canonical_lexical_'+edge['severity'])
    for review in reviews:
        if review['explicit_material_reuse']:connect(review['left_doc_id'],review['right_doc_id'],'explicit_material_reuse')
    components=defaultdict(list)
    for doc in sorted(rows):components[find(doc)].append(doc)
    component_rows=[];component_for={}
    for members in components.values():
        # Identity is based on policy version and exact sorted membership, not a mutable family rename.
        key='component-'+_hash({'schema':SCHEMA,'doc_ids':members})
        member_set=set(members)
        linking=[{'left_doc_id':a,'right_doc_id':b,'reason':r} for a,b,r in sorted(links) if a in member_set]
        train=[d for d in members if rows[d]['split']=='train']
        exposed=[d for d in members if direct_exposure[d]]
        ambiguous=[d for d in members if source_identity[d] is None]
        reasons=sorted({x['reason'] for x in linking})
        reasons += [r for condition,r in ((train,'touches_canonical_train'),(exposed,'touches_known_exposure'),(ambiguous,'unrecognized_source_identity')) if condition]
        state='quarantine' if train or exposed else 'needs_review' if ambiguous else 'operational_candidate_proposed'
        component={'component_key':key,'member_doc_ids':members,'train_doc_ids':train,'validation_doc_ids':[d for d in members if d in validation],
            'exposed_member_doc_ids':exposed,'source_identity_needs_review_doc_ids':ambiguous,
            'status':state,'component_reasons':sorted(reasons),'linking_evidence':linking}
        component_rows.append(component)
        for doc in members:component_for[doc]=component
    component_rows.sort(key=lambda c:c['member_doc_ids'])
    ledger=[]
    for doc in sorted(validation):
        row=rows[doc];component=component_for[doc];parsed=source_identity[doc]
        ledger.append({**row,'declared_development_exposed':row.get('development_exposed'),
            'strict_arxiv_base':parsed[0] if parsed else None,'strict_arxiv_version':parsed[1] if parsed else None,
            'component_key':component['component_key'],'status':component['status'],
            'component_reasons':list(component['component_reasons']),'direct_exposure_reasons':direct_exposure[doc],
            'source_identity_needs_review':parsed is None,'component_source_review_required':bool(component['source_identity_needs_review_doc_ids']),
            'original_family_id_unchanged':True})
    cohort=[r['doc_id'] for r in ledger if r['status']=='operational_candidate_proposed']
    counts=Counter(r['status'] for r in ledger)
    public={'schema':SCHEMA,'status':'metadata_proposal_only','canonical_documents':len(rows),
        'canonical_train_documents':len(rows)-len(validation),'canonical_validation_documents':len(validation),
        'prior_development_validation_documents':len(prior),'all_components':len(component_rows),
        'validation_components':sum(bool(c['validation_doc_ids']) for c in component_rows),
        'validation_status_counts':{k:counts[k] for k in STATUS},'proposed_validation_documents':len(cohort),
        'proposed_validation_components':sum(c['status']=='operational_candidate_proposed' and bool(c['validation_doc_ids']) for c in component_rows),
        'validation_connected_to_train':sum(bool(component_for[d]['train_doc_ids']) for d in validation),
        'validation_connected_to_known_exposure':sum(bool(component_for[d]['exposed_member_doc_ids']) for d in validation),
        'validation_components_requiring_source_review':sum(bool(c['validation_doc_ids']) and bool(c['source_identity_needs_review_doc_ids']) for c in component_rows),
        'unrecognized_source_identity_documents':sum(x is None for x in source_identity.values()),
        'direct_exposure_matched_canonical_documents':sum(bool(x) for x in direct_exposure.values()),
        'direct_exposure_reason_counts':{reason:sum(reason in reasons for reasons in direct_exposure.values()) for reason in EXPOSURE_REASONS},
        'linking_evidence_counts':{reason:sum(r==reason for _,_,r in links) for reason in LINK_REASONS},
        'overlap_input_records':len(edges),'overlap_input_severity_counts':{s:sum(e['severity']==s for e in edges) for s in ('low','moderate','high')},
        'source_review_input_records':len(reviews),'source_review_explicit_reuse_records':sum(r['explicit_material_reuse'] for r in reviews),
        'source_rows_removed':0,'source_family_ids_modified':0,**FLAGS}
    if sum(counts.values())!=len(validation):_fail('validation denominator mismatch')
    canonical_input={'schema':SCHEMA,'index_rows':[rows[d] for d in sorted(rows)],
        'known_exposure':{k:sorted(exposure[k]) for k in sorted(exposure)},'prior_development_ids':sorted(prior),
        'overlap_edges':edges,'source_reviews':reviews}
    return {'schema':SCHEMA,'input_metadata_sha256':_hash(canonical_input),'validation_ledger':ledger,
        'proposed_cohort_doc_ids':cohort,'components':component_rows,
        'validated_overlap_metadata':edges,'validated_source_review_metadata':reviews,
        'public_aggregate':public}
