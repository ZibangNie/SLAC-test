"""Separate complete confirmation inputs from references; inspect a fixed sample.

No model/API access. All cohort questions are copied mechanically; only eight
documents and at most two questions per sampled document receive content checks.
The user explicitly requested sampled checking instead of a full content audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from qasper_alignment_v2 import native_text_blocks, normalize_evidence
from qasper_metrics import references_from_annotations

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/research-foundation'
SCHEMA='slac-qasper-confirmation-export-v1'
MEMBER='qasper-dev-v0.3.json'
SALT='SLAC-QASPER-CONFIRMATION-EXPORT-20260927-v1'
ANSWER_FIELDS=('unanswerable','extractive_spans','free_form_answer','yes_no','evidence','highlighted_evidence')
PLAN_FILES={'plan.json','cohort.json','seal.json'}
OUTPUT_FILES={'documents.jsonl','questions.jsonl','question_manifest.jsonl','references.jsonl',
              'document_inventory.jsonl','sample_check.json','public_aggregate.json',
              'registration.json','summary.json'}
PINS={
 'qasper-confirmation-metadata-run-01/summary.json':'f47b0b5c88fc8b9f180ccf947171e1fafa9a76c6d106dbf8d3e4046a9b47987d',
 'qasper-confirmation-metadata-independent-01/verification.json':'a226347e0dc6d459af4833d3b51fefb8d406ca58caaf7216c45c0832300a9bd8',
 'qasper-pool/native_qa_alignment.json':'22e69af6d338736fc3a21e29f05d80248da2e54237aeeca20e814894bfd58d36'}
PROTOCOL_SHA='6824284907383333d826f658e6876f33c0b7f7e188519547d0fa4fc82a542a24'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def load(path):return json.loads(Path(path).read_bytes())


def json_rows(path):
    with Path(path).open('r',encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def write(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical(value)+b'\n')


def write_rows(path,values):
    with Path(path).open('xb') as stream:
        for value in values:stream.write(canonical(value)+b'\n')


def verify(bindings):
    for path,h in bindings.items():require(digest(path)==h,'bound input changed')


def sample_ids(values,limit,channel):
    require(type(limit)is int and limit>=0,'invalid sample limit')
    require(all(type(v)is str and v for v in values),'sample identities must be nonempty strings')
    require(len(values)==len(set(values)),'sample identities must be unique')
    return sorted(values,key=lambda v:(hashlib.sha256(f'{SALT}|{channel}|{v}'.encode()).hexdigest(),v))[:limit]


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON object key');result[key]=value
    return result


def read_native(archive):
    """Read only the exact validation member, never extract paths to the disk."""
    with tarfile.open(fileobj=io.BytesIO(archive) if isinstance(archive,bytes) else None,
                      name=None if isinstance(archive,bytes) else archive,mode='r:gz') as tar:
        matches=[member for member in tar.getmembers() if member.name==MEMBER]
        require(len(matches)==1,'validation archive member must be unique')
        member=matches[0]
        require(member.isfile() and 0<member.size<=32*1024*1024,'invalid validation member')
        with tar.extractfile(member) as stream:payload=stream.read(32*1024*1024+1)
        require(len(payload)==member.size,'validation member size differs')
    native=json.loads(payload,object_pairs_hook=_unique_object)
    require(type(native)is dict,'native member must be an object')
    return native,hashlib.sha256(payload).hexdigest()


def project_dataset(cohort,native,canonical_documents,archive_sha256,sample_doc_ids):
    require(type(cohort)is list and cohort,'empty cohort')
    wanted={r['doc_id']:r for r in cohort}
    require(len(wanted)==len(cohort),'duplicate cohort document')
    require(len(sample_doc_ids)==len(set(sample_doc_ids)) and set(sample_doc_ids)<=set(wanted) and len(sample_doc_ids)<=8,'invalid sample document inventory')
    require(set(canonical_documents)==set(wanted),'canonical cohort coverage differs')
    documents=[];questions=[];manifest=[];references=[];inventory=[];sample_questions=[]
    seen=set();counts=Counter();sample_units=0;omissions=Counter()
    for doc in sorted(wanted):
        c=wanted[doc];source=c['source_id']
        require(source in native,'cohort source absent from native member')
        paper=native[source];row=canonical_documents[doc]
        require(c['split']=='validation' and c['status']=='operational_candidate_proposed','noncandidate or wrong split')
        require(row['doc_id']==doc and row['source_id']==source and row['family_id']==c['family_id'] and row['original_split']=='validation','canonical identity differs')
        require(row['raw_locator'].get('member')==MEMBER and row['raw_locator'].get('json_pointer')==f'/{source}','canonical locator differs')
        require(row.get('extra',{}).get('raw_archive_sha256')==archive_sha256,'canonical archive commitment differs')
        identity={'doc_id':doc,'source_id':source,'original_family_id':c['family_id'],'family_id':c['component_key']}
        block_map={}
        for block in row['blocks']:
            key=(block['source_locator'].get('json_pointer'),block['kind'])
            block_map.setdefault(key,[]).append(block)
        units=[]
        for kind,_,locator,text in native_text_blocks(paper,source):
            require(type(text)is str,'native unit is not text')
            if not text.strip():continue
            matches=block_map.get((locator,'heading' if kind=='title' else kind),[])
            require(len(matches)==1,'native-to-canonical locator is not unique')
            block=matches[0];start,end=block['char_span']
            require(type(start)is int and type(end)is int and 0<=start<end<=len(row['canonical_text']),'invalid canonical span')
            selected_text=row['canonical_text'][start:end]
            require(selected_text.strip(),'native nonempty block has empty canonical text')
            if doc in sample_doc_ids:
                require(normalize_evidence(selected_text)==normalize_evidence(text),'sample native/canonical text differs')
                sample_units+=1
            units.append({'unit_id':block['block_id'],'order':len(units),'kind':kind,'start':start,'end':end,'text':selected_text,'native_text':text})
        require(units and len({u['unit_id'] for u in units})==len(units),'invalid model-visible unit identities')
        documents.append({**identity,'units':units})
        qas=paper['qas'];require(type(qas)is list,'questions must be a list')
        qids=[q['question_id'] for q in qas]
        require(len(qids)==len(set(qids)) and all(type(q)is str and q for q in qids),'invalid question identities')
        sampled=sample_ids(qids,2,f'question:{doc}') if doc in sample_doc_ids else []
        valid_queries=0;annotation_count=0
        for qa in qas:
            qid=qa['question_id'];question=qa['question'];require(type(question)is str,'question must be text')
            key=(doc,qid);require(key not in seen,'duplicate question key');seen.add(key)
            q_identity={**identity,'question_id':qid}
            questions.append({**q_identity,'question':question})
            manifest.append({k:q_identity[k] for k in ('doc_id','question_id','family_id')})
            require(type(qa['answers'])is list,'answer annotations must be a list')
            answer_annotations=[]
            for annotation in qa['answers']:
                require(type(annotation)is dict and type(annotation.get('answer'))is dict,'invalid reference container')
                answer=annotation['answer']
                kept={k:deepcopy(answer[k]) for k in ANSWER_FIELDS if k in answer}
                omissions.update(k for k in answer if k not in ANSWER_FIELDS)
                answer_annotations.append({'native_answer':kept})
            references.append({**q_identity,'answer_annotations':answer_annotations})
            valid_queries+=int(bool(question.strip()));annotation_count+=len(answer_annotations)
            counts['questions_without_annotations']+=int(not answer_annotations)
            if qid in sampled:
                # Only this fixed sample enters the reference schema adapter.
                converted=references_from_annotations(answer_annotations)
                require(len(converted)==len(qa['answers']),'sample annotation count differs')
                sample_questions.append({'doc_id':doc,'question_id':qid,'reference_annotations':len(converted),
                    'question_text_preserved':question==qa['question'],'reference_schema_passed':True})
        inventory.append({**identity,'question_count':len(qas),'annotation_count':annotation_count,'unit_count':len(units),
                          'invalid_query_count':len(qas)-valid_queries,'sampled_for_content_check':doc in sample_doc_ids})
        counts.update(questions=len(qas),annotations=annotation_count,units=len(units),invalid_queries=len(qas)-valid_queries,
                      documents_without_questions=int(not qas))
    order=lambda r:(r['doc_id'],r['question_id'])
    for values in (questions,manifest,references,sample_questions):values.sort(key=order)
    ready=counts['invalid_queries']==0 and counts['questions_without_annotations']==0 and counts['questions']>0
    public={'schema':SCHEMA,'status':'complete_export_sample_checked','documents':len(cohort),
        'components':len({c['component_key'] for c in cohort}),
        'question_bearing_components':len({r['family_id'] for r in inventory if r['question_count']}),
        'questions':counts['questions'],'reference_annotations':counts['annotations'],'units':counts['units'],
        'invalid_queries':counts['invalid_queries'],'questions_without_annotations':counts['questions_without_annotations'],
        'documents_without_questions':counts['documents_without_questions'],
        'sample_documents':len(sample_doc_ids),'sample_questions':len(sample_questions),'sample_native_units':sample_units,
        'omitted_non_evaluator_answer_field_occurrences':sum(omissions.values()),
        'sample_text_and_reference_schema_passed':True,'full_reference_schema_audit_performed':False,
        'all_questions_retained':True,'outcome_based_exclusions':0,'ready_for_candidate_preparation':ready,
        'new_qa_machine_read':True,'machine_decoded_validation_member_documents':len(native),
        'raw_text_or_qa_shown_to_agent':False,'new_model_inputs_submitted':False,'quality_scores_computed':False,
        'api_calls':0,'training_started':False,'official_test_payload_read':False,
        'cohort_admitted_for_export':True,'paid_execution_admitted':False,
        'independence_established':False,'model_contamination_free':False,'human_reviewed':False}
    spot={'schema':SCHEMA,'sample_policy':{'salt':SALT,'documents':8,'questions_per_document':2,'selection':'SHA256 bottom-k on identifiers only'},
          'sample_doc_ids':list(sample_doc_ids),'sample_questions':sample_questions,'sample_native_units':sample_units,
          'omitted_non_evaluator_answer_fields':dict(sorted(omissions.items())),
          'full_reference_schema_audit_performed':False,'sample_checks_passed':True}
    return {'documents':documents,'questions':questions,'question_manifest':manifest,'references':references,
            'document_inventory':inventory,'spot_check':spot,'public_aggregate':public}


def prepare(output,run_output):
    output,run_output=Path(output).resolve(),Path(run_output).resolve()
    require(output.is_relative_to(ART.resolve()) and run_output.is_relative_to(ART.resolve()),'outputs must remain in ignored research artifacts')
    require(not output.exists() and not run_output.exists() and not output.is_relative_to(run_output) and not run_output.is_relative_to(output),'new disjoint output paths required')
    bindings={str((ART/n).resolve()):h for n,h in PINS.items()};verify(bindings)
    protocol=ROOT/'docs/research/results/qasper_confirmation_protocol_v2_20260927.json'
    require(digest(protocol)==PROTOCOL_SHA,'current analysis protocol differs');bindings[str(protocol)]=PROTOCOL_SHA
    summary=load(ART/'qasper-confirmation-metadata-run-01/summary.json')
    require(summary['status']=='completed_metadata_proposal','metadata proposal incomplete')
    proposed=ART/'qasper-confirmation-metadata-run-01/proposed_cohort.jsonl'
    expected=summary['output_sha256'][proposed.name];require(digest(proposed)==expected,'proposal changed');bindings[str(proposed)]=expected
    cohort=json_rows(proposed)
    require(len(cohort)==248 and len({r['doc_id'] for r in cohort})==248,'full 248-document proposal required')
    require(all(r['status']=='operational_candidate_proposed' and r['split']=='validation' for r in cohort),'cohort status differs')
    alignment=load(ART/'qasper-pool/native_qa_alignment.json')
    raw={}
    for role,name in [('archive','qasper-train-dev-v0.3.tgz'),('canonical','documents-00000.jsonl.gz')]:
        matches=[(p,h) for p,h in alignment['input_sha256'].items() if Path(p).name==name]
        require(len(matches)==1,'raw source commitment not unique');path,h=matches[0]
        require(Path(path).is_file(),'committed raw source missing');raw[role]={'path':str(Path(path).resolve()),'sha256':h}
    for path in [Path(__file__),ROOT/'docs/research/CONFIRMATION_EXPORT_PROTOCOL_20260927.md',
                 ROOT/'tests/research/test_qasper_confirmation_export.py',
                 ROOT/'docs/research/qasper_alignment_v2.py',ROOT/'docs/research/qasper_metrics.py']:
        bindings[str(path.resolve())]=digest(path)
    sample=sample_ids([r['doc_id'] for r in cohort],8,'documents')
    plan={'schema':SCHEMA,'status':'prepared_export_not_executed','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'run_output':str(run_output),'input_sha256':bindings,'raw_content_expected_sha256':raw,
          'cohort_documents':248,'sample_doc_ids':sample,'sample_salt':SALT,'sample_questions_per_document':2,
          'all_questions_to_be_retained':True,'new_qa_read_in_preparation':False,
          'cohort_admitted_for_export':True,'independence_established':False,'paid_execution_admitted':False,'api_calls':0}
    verify(bindings);output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan);write(output/'cohort.json',cohort)
    write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
    return {'status':plan['status'],'documents':248,'sample_documents':8,'maximum_sample_questions':16,'new_qa_read':False,'api_calls':0}


def load_plan(directory):
    directory=Path(directory).resolve();require({p.name for p in directory.iterdir()}==PLAN_FILES,'plan inventory differs')
    raw={n:(directory/n).read_bytes() for n in PLAN_FILES}
    bindings={str(directory/n):hashlib.sha256(b).hexdigest() for n,b in raw.items()}
    require(json.loads(raw['seal.json'])=={n:bindings[str(directory/n)] for n in PLAN_FILES-{'seal.json'}},'plan seal differs')
    plan=json.loads(raw['plan.json']);cohort=json.loads(raw['cohort.json'])
    require(plan['schema']==SCHEMA and plan['status']=='prepared_export_not_executed' and plan['api_calls']==0,'plan contract differs')
    require(len(cohort)==plan['cohort_documents']==248 and plan['sample_doc_ids']==sample_ids([r['doc_id'] for r in cohort],8,'documents'),'cohort/sample denominator differs')
    require(plan['sample_salt']==SALT and plan['sample_questions_per_document']==2 and plan['all_questions_to_be_retained'] is True,'sampling policy differs')
    require(plan['new_qa_read_in_preparation'] is False and plan['cohort_admitted_for_export'] is True and plan['independence_established'] is False and plan['paid_execution_admitted'] is False,'admission scope differs')
    for name,h in PINS.items():require(plan['input_sha256'].get(str((ART/name).resolve()))==h,'historical metadata binding differs')
    verify(plan['input_sha256']);verify(bindings)
    proposed=ART/'qasper-confirmation-metadata-run-01/proposed_cohort.jsonl'
    source_summary=load(ART/'qasper-confirmation-metadata-run-01/summary.json')
    require(plan['input_sha256'].get(str(proposed))==source_summary['output_sha256'][proposed.name] and cohort==json_rows(proposed),'prepared cohort differs from original proposal')
    alignment=load(ART/'qasper-pool/native_qa_alignment.json')
    for role,name in [('archive','qasper-train-dev-v0.3.tgz'),('canonical','documents-00000.jsonl.gz')]:
        matches=[(p,h) for p,h in alignment['input_sha256'].items() if Path(p).name==name]
        require(len(matches)==1,'raw source commitment not unique')
        p,h=matches[0];require(plan['raw_content_expected_sha256'][role]=={'path':str(Path(p).resolve()),'sha256':h},'raw source projection differs')
    run_output=Path(plan['run_output']).resolve()
    require(run_output.is_relative_to(ART.resolve()) and not run_output.is_relative_to(directory) and not directory.is_relative_to(run_output),'run overlaps plan or is not private')
    return plan,cohort,bindings


def run(directory):
    plan,cohort,plan_bindings=load_plan(directory);output=Path(plan['run_output']).resolve()
    require(output.is_relative_to(ART.resolve()) and not output.exists(),'run output must be new and private')
    output.mkdir(parents=True,exist_ok=False);write(output/'registration.json',{'schema':SCHEMA,'plan_sha256':plan_bindings,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    # Hash the exact buffers consumed, once, rather than re-auditing old datasets.
    content={}
    for role,item in plan['raw_content_expected_sha256'].items():
        require(Path(item['path']).stat().st_size<=256*1024*1024,'raw compressed source exceeds bound')
        content[role]=Path(item['path']).read_bytes()
        require(hashlib.sha256(content[role]).hexdigest()==item['sha256'],'raw content commitment differs')
    native,member_hash=read_native(content['archive']);require(len(native)==281,'validation member document count differs')
    wanted={r['doc_id'] for r in cohort};canonical_documents={}
    with gzip.GzipFile(fileobj=io.BytesIO(content['canonical'])) as stream:
        for line in stream:
            row=json.loads(line)
            if row['doc_id'] in wanted:
                require(row['doc_id'] not in canonical_documents,'duplicate canonical document')
                canonical_documents[row['doc_id']]=row
    result=project_dataset(cohort,native,canonical_documents,plan['raw_content_expected_sha256']['archive']['sha256'],plan['sample_doc_ids'])
    verify(plan['input_sha256']);verify(plan_bindings)
    for name in ('documents','questions','question_manifest','references','document_inventory'):write_rows(output/f'{name}.jsonl',result[name])
    write(output/'sample_check.json',result['spot_check']);write(output/'public_aggregate.json',result['public_aggregate'])
    summary={'schema':SCHEMA,'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'documents':len(cohort),'questions':len(result['questions']),'ready_for_candidate_preparation':result['public_aggregate']['ready_for_candidate_preparation'],
        'plan_sha256':plan_bindings,'input_sha256':plan['input_sha256'],'consumed_raw_content_sha256':plan['raw_content_expected_sha256'],
        'native_member_sha256':member_hash,'output_sha256':{n:digest(output/n) for n in OUTPUT_FILES-{'summary.json'}},
        'new_qa_machine_read':True,'new_qa_shown_to_agent':False,'sample_only_content_check':True,'api_calls':0,'quality_scores_computed':False}
    write(output/'summary.json',summary)
    return result['public_aggregate']


def audit_sample(directory,output):
    """Mechanical coverage + sampled saved projections; no raw source reread."""
    plan,cohort,plan_bindings=load_plan(directory);run_dir=Path(plan['run_output']);output=Path(output).resolve()
    require(output.is_relative_to(ART.resolve()) and not output.exists() and not output.is_relative_to(run_dir),'new private audit directory required')
    require({p.name for p in run_dir.iterdir()}==OUTPUT_FILES,'run inventory differs')
    initial={str(run_dir/n):digest(run_dir/n) for n in OUTPUT_FILES}
    summary=load(run_dir/'summary.json');require(summary['schema']==SCHEMA and summary['status']=='completed' and summary['plan_sha256']==plan_bindings,'export summary differs')
    require(set(summary['output_sha256'])==OUTPUT_FILES-{'summary.json'},'output seal inventory differs')
    require(summary['output_sha256']=={n:initial[str(run_dir/n)] for n in OUTPUT_FILES-{'summary.json'}},'output commitment differs')
    require(summary['input_sha256']==plan['input_sha256'] and summary['consumed_raw_content_sha256']==plan['raw_content_expected_sha256'],'consumed source commitment differs')
    docs=json_rows(run_dir/'documents.jsonl');queries=json_rows(run_dir/'questions.jsonl');refs=json_rows(run_dir/'references.jsonl')
    manifest=json_rows(run_dir/'question_manifest.jsonl');inventory=json_rows(run_dir/'document_inventory.jsonl')
    keys=lambda rows:[(r['doc_id'],r['question_id'],r['family_id']) for r in rows]
    expected={r['doc_id']:r for r in cohort};by_doc={r['doc_id']:r for r in docs}
    require(len(docs)==len(by_doc)==len(inventory)==248 and set(by_doc)==set(expected),'document coverage differs')
    qkeys=keys(queries);require(len(qkeys)==len(set(qkeys)) and qkeys==keys(refs)==keys(manifest),'question/reference inventory differs')
    require(len(queries)==summary['questions'],'question count differs')
    for row in docs+queries+refs:
        c=expected[row['doc_id']]
        require(row['family_id']==c['component_key'] and row['original_family_id']==c['family_id'] and row['source_id']==c['source_id'],'export family mapping differs')
    require(all(set(q)=={'doc_id','source_id','original_family_id','family_id','question_id','question'} for q in queries),'query projection is not gold-free')
    require(all(set(r)=={'doc_id','source_id','original_family_id','family_id','question_id','answer_annotations'} for r in refs),'reference projection fields differ')
    require(all(set(r)=={'doc_id','question_id','family_id'} for r in manifest),'manifest projection fields differ')
    require(all(set(r)=={'doc_id','source_id','original_family_id','family_id','units'} for r in docs),'document projection fields differ')
    require(all(set(u)=={'unit_id','order','kind','start','end','text','native_text'} for d in docs for u in d['units']),'unit projection fields differ')
    require(len({r['doc_id'] for r in inventory})==len(inventory) and {r['doc_id'] for r in inventory}==set(expected),'document inventory identities differ')
    qcounts=Counter(q['doc_id'] for q in queries)
    badcounts=Counter(q['doc_id'] for q in queries if not q['question'].strip())
    acounts=Counter()
    for r in refs:acounts[r['doc_id']]+=len(r['answer_annotations'])
    for row in inventory:
        d=row['doc_id'];c=expected[d]
        actual={'doc_id':d,'source_id':c['source_id'],'original_family_id':c['family_id'],'family_id':c['component_key'],
                'question_count':qcounts[d],'annotation_count':acounts[d],'unit_count':len(by_doc[d]['units']),
                'invalid_query_count':badcounts[d],'sampled_for_content_check':d in plan['sample_doc_ids']}
        require(row==actual,'per-document mechanical counts differ')
    reference_index={(r['doc_id'],r['question_id']):r for r in refs}
    sampled=[];unit_count=0
    for doc in plan['sample_doc_ids']:
        for unit in by_doc[doc]['units']:
            require(normalize_evidence(unit['text'])==normalize_evidence(unit['native_text']),'sample text fidelity differs');unit_count+=1
        qids=sample_ids([q['question_id'] for q in queries if q['doc_id']==doc],2,f'question:{doc}')
        for qid in qids:
            references_from_annotations(reference_index[doc,qid]['answer_annotations']);sampled.append((doc,qid))
    spot=load(run_dir/'sample_check.json')
    require(spot['sample_doc_ids']==plan['sample_doc_ids'] and sorted(sampled)==sorted((r['doc_id'],r['question_id']) for r in spot['sample_questions']) and unit_count==spot['sample_native_units'],'sample inventory differs')
    require(spot['sample_policy']=={'salt':SALT,'documents':8,'questions_per_document':2,'selection':'SHA256 bottom-k on identifiers only'} and spot['sample_checks_passed'] is True and spot['full_reference_schema_audit_performed'] is False,'sample check policy differs')
    public=load(run_dir/'public_aggregate.json')
    no_annotations=sum(not r['answer_annotations'] for r in refs)
    ready=not sum(badcounts.values()) and not no_annotations and bool(queries)
    expected_public={'schema':SCHEMA,'status':'complete_export_sample_checked','documents':248,
        'components':len({r['component_key'] for r in cohort}),
        'question_bearing_components':len({r['family_id'] for r in queries}),
        'questions':len(queries),'reference_annotations':sum(acounts.values()),'units':sum(len(r['units']) for r in docs),
        'invalid_queries':sum(badcounts.values()),'questions_without_annotations':no_annotations,
        'documents_without_questions':sum(qcounts[d]==0 for d in expected),
        'sample_documents':8,'sample_questions':len(sampled),'sample_native_units':unit_count,
        'omitted_non_evaluator_answer_field_occurrences':sum(spot['omitted_non_evaluator_answer_fields'].values()),
        'sample_text_and_reference_schema_passed':True,'full_reference_schema_audit_performed':False,
        'all_questions_retained':True,'outcome_based_exclusions':0,'ready_for_candidate_preparation':ready,
        'new_qa_machine_read':True,'machine_decoded_validation_member_documents':281,
        'raw_text_or_qa_shown_to_agent':False,'new_model_inputs_submitted':False,'quality_scores_computed':False,
        'api_calls':0,'training_started':False,'official_test_payload_read':False,
        'cohort_admitted_for_export':True,'paid_execution_admitted':False,
        'independence_established':False,'model_contamination_free':False,'human_reviewed':False}
    require(public==expected_public and summary['ready_for_candidate_preparation']==ready,'public mechanical counts or flags differ')
    receipt={'schema':SCHEMA,'status':'verified_complete_coverage_and_fixed_sample','documents':248,'questions':len(queries),
        'sample_documents':8,'sample_questions':len(sampled),'sample_native_units':unit_count,
        'full_content_audit_performed':False,'raw_archive_reopened_for_audit':False,'quality_scores_computed':False,'api_calls':0,
        'summary_sha256':digest(run_dir/'summary.json'),'public_aggregate_sha256':digest(run_dir/'public_aggregate.json'),
        'plan_sha256':plan_bindings,'cohort_admitted_for_export':True,'independence_established':False,
        'scope':'complete mechanical identity/count/hash checks and fixed sample replay; no exhaustive content inspection'}
    verify(plan['input_sha256']);verify(plan_bindings);verify(initial)
    require({p.name for p in run_dir.iterdir()}==OUTPUT_FILES,'run inventory changed during audit')
    output.mkdir(parents=True,exist_ok=False);write(output/'verification.json',receipt)
    return {k:v for k,v in receipt.items() if k!='plan_sha256'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--run-output',required=True)
    a=sub.add_parser('run');a.add_argument('--plan',required=True)
    a=sub.add_parser('audit-sample');a.add_argument('--plan',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    result=prepare(a.output,a.run_output) if a.command=='prepare' else run(a.plan) if a.command=='run' else audit_sample(a.plan,a.output)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
