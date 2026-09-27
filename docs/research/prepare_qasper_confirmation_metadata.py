"""Sealed metadata-only preparation; no archive, QA, canonical text or API read."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import qasper_confirmation_policy as policy

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/research-foundation'
SCHEMA = 'slac-qasper-confirmation-metadata-v1'
PINNED = {
 'qasper-pool/pool_manifest.json':'02fb2f08cb22ef4be58d31b4ec59f0153bcf08aed3fb2acce636dda04698e9b3',
 'qasper-pool/candidates.jsonl':'0ed606e859251e265292c8fa9da430b4d6aa27554455ef2c88afc5fc2c962456',
 'qasper-pool/eligibility_audit.json':'8b498cbae0bf08b59e4d6410f898c36f4f5353a65c528b3ff3d779be58563946',
 'qasper-pool/native_qa_alignment.json':'22e69af6d338736fc3a21e29f05d80248da2e54237aeeca20e814894bfd58d36',
 'qasper-remaining-validation-screen-01/plan.json':'4900de642c74d08454f36216e06b829590705ef76efb49e2c4b94544b8c04197',
 'qasper-remaining-validation-screen-01/completion.json':'448637910f0cfad44dea986cb88b9dd16be8dfca8fe8a0b2c7629f234093f233',
 'qasper-remaining-validation-screen-01/audit_receipt.json':'f160683e787c68cf68eb25b2328b86375f1d588f8bf87408dab776a850409e26',
 'qasper-remaining-validation-screen-01/public_aggregate.json':'7c2b5be39c161dc7e139849daf756af48398782bde5ac555314a11188f742e5b',
 'qasper-remaining-validation-screen-01/flagged_pairs_unique.jsonl':'6859d4c422a2cc417e66b1d08c321128c04bf9b93a72e3e07ea7ac36e304f062',
 'qasper-validation-source-review-01/source_evidence_private.json':'bdb5056e7b2a254b210f0584da2ebf504bd35964983177eb8175da461c3cd890',
 'qasper-validation-source-review-01/review_seal.json':'3023abbf9fc575f0b405b1c80c5b02b2c1b24567a925e3befd8f0076180cfbd1',
}
EXPORTER_SHA = '5dd900d7b58d941ec544fa62d949edc6f4f1f9a0625d35899051bb69b359e028'
PLAN_FILES = {'plan.json', 'input.json', 'seal.json'}
RUN_FILES = {'validation_ledger.jsonl', 'proposed_cohort.jsonl', 'components.json',
             'public_aggregate.json', 'run_registration.json', 'summary.json'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value) + b'\n')


def rows(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(bindings):
    for p, h in bindings.items():
        require(digest(p) == h, 'bound file changed')


def own_files():
    return [Path(__file__), Path(policy.__file__), ROOT/'docs/research/CONFIRMATION_METADATA_PROTOCOL_20260927.md',
            ROOT/'tests/research/test_qasper_confirmation_policy.py',
            ROOT/'tests/research/test_qasper_confirmation_metadata.py']


def source_data():
    bindings = {str((ART/n).resolve()): h for n, h in PINNED.items()}
    verify(bindings)
    pool = load(ART/'qasper-pool/pool_manifest.json')
    named = {}
    for name in ('index.jsonl.gz', 'development_reservations.json', 'public_pilot_manifest.jsonl'):
        matches = [(p, h) for p, h in pool['input_sha256'].items() if Path(p).name == name]
        require(len(matches) == 1, 'metadata source name not unique')
        p, h = matches[0]; require(re.fullmatch('[0-9a-f]{64}', h), 'invalid metadata commitment')
        require(digest(p) == h, 'metadata source changed'); bindings[str(Path(p).resolve())] = h; named[name] = Path(p)
    original = rows(named['index.jsonl.gz'])
    require(Counter(r['original_split'] for r in original) == {'train':888, 'validation':281}, 'index split denominator differs')
    require(all(r['source'] == 'qasper' for r in original), 'non-Qasper source in index')
    require(len({r['doc_id'] for r in original}) == len(original), 'duplicate index identity')
    index = [{**{k:r[k] for k in ('source','doc_id','source_id','family_id','normalized_body_sha256','development_exposed')}, 'split':r['original_split']} for r in original]
    by_id = {r['doc_id']: r for r in index}; validation = {r['doc_id'] for r in index if r['split']=='validation'}
    candidates = rows(ART/'qasper-pool/candidates.jsonl')
    prior = {r['doc_id'] for r in candidates}
    require(len(candidates) == len(prior) == 32 and prior <= validation, 'development denominator differs')
    for r in candidates:
        require(r['official_split']=='validation' and all(r[k]==by_id[r['doc_id']][k] for k in ('source_id','family_id','normalized_body_sha256')), 'development metadata differs')
    eligibility = load(ART/'qasper-pool/eligibility_audit.json')['eligible_ids']
    require(len(eligibility) == len(set(eligibility)) and set(eligibility)==validation, 'original eligible denominator differs')
    reservation = load(named['development_reservations.json'])
    exposure = {'doc_ids':sorted(set(reservation['doc_ids'])), 'source_ids':sorted({r['paper_id'] for r in rows(named['public_pilot_manifest.jsonl'])}),
                'family_ids':sorted(set(reservation['family_ids'])), 'normalized_body_sha256':sorted(set(reservation['normalized_body_sha256']))}
    require(set(exposure['source_ids']) == set(pool['known_exposed_qasper_source_ids']), 'historical exposure scope differs')
    screen_plan = load(ART/'qasper-remaining-validation-screen-01/plan.json')
    screen = load(ART/'qasper-remaining-validation-screen-01/public_aggregate.json')
    complete = load(ART/'qasper-remaining-validation-screen-01/completion.json')
    audit = load(ART/'qasper-remaining-validation-screen-01/audit_receipt.json')
    flattened = [d for batch in screen_plan['batch_doc_ids'] for d in batch]
    require(len(flattened)==len(set(flattened))==249 and set(flattened)==validation-prior and set(screen_plan['current_doc_ids'])==prior, 'remaining coverage differs')
    require(complete['status']=='completed' and screen['status']=='complete_document_screen_with_limits' and screen['completed_batches']==8 and screen['remaining_documents']==249 and screen['flagged_pairs_unique']==1, 'complete screen missing')
    for name in ('public_aggregate.json','flagged_pairs_unique.jsonl'):
        require(complete['output_sha256'][name]==digest(ART/'qasper-remaining-validation-screen-01'/name), 'completed screen output differs')
    require(audit['status']=='verified_complete_document_screen' and audit['public_aggregate_sha256']==PINNED['qasper-remaining-validation-screen-01/public_aggregate.json'] and audit['remaining_documents']==249 and audit['qa_payload_read'] is False, 'screen audit scope differs')
    flags = rows(ART/'qasper-remaining-validation-screen-01/flagged_pairs_unique.jsonl')
    require(len(flags)==1, 'unique flag denominator differs')
    edges = []
    for r in flags:
        p = r['representative']; require(r['comparison_scope']=='qasper_train' and p['reference_scope']=='qasper_train', 'unexpected source flag scope')
        require(p['query_doc_id'] in validation-prior and by_id[p['reference_doc_id']]['split']=='train', 'source flag identity differs')
        reasons = set(p['review_reasons']); require(reasons=={'moderate_lexical_overlap_review'}, 'new flag class requires new projection')
        edges.append({'left_doc_id':p['query_doc_id'], 'right_doc_id':p['reference_doc_id'], 'severity':'moderate', 'scope':'canonical_train_validation'})
    relation = load(ART/'qasper-validation-source-review-01/source_evidence_private.json')
    review_seal = load(ART/'qasper-validation-source-review-01/review_seal.json')
    require(review_seal['files_sha256']['source_evidence_private.json']==PINNED['qasper-validation-source-review-01/source_evidence_private.json'], 'source relation seal differs')
    require(relation['action_flags']['human_review_completed'] is False, 'source review provenance changed')
    pair = relation['pair']; left,right=pair['validation']['doc_id'],pair['train']['doc_id']
    require({left,right}=={edges[0]['left_doc_id'],edges[0]['right_doc_id']}, 'review is not the screened pair')
    for key in ('validation','train'):
        require(all(pair[key][k] == by_id[pair[key]['doc_id']][k] for k in ('source_id','family_id')), 'source review identity differs')
    declared = relation['relationship']['explicit_material_reuse_and_additional_research_statement']
    mapped = relation['relationship']['reference_maps_to_earlier_document']
    require(declared is True and mapped is True, 'documented reuse not established')
    reviews = [{'left_doc_id':left,'right_doc_id':right,'explicit_material_reuse':True}]
    alignment = load(ART/'qasper-pool/native_qa_alignment.json')
    require(alignment['all_input_hashes_unchanged'] is True and alignment['raw_member_read']=='qasper-dev-v0.3.json' and alignment['test_payload_read'] is False, 'historical export receipt differs')
    require(alignment['documents']==32 and alignment['counts']['questions']==104 and {r['doc_id'] for r in alignment['per_document']}==prior, 'historical exported denominator differs')
    exporter = ROOT/'docs/research/export_qasper_native_sidecar.py'
    require(digest(exporter)==EXPORTER_SHA, 'historical exporter source differs'); bindings[str(exporter.resolve())] = EXPORTER_SHA
    for p in own_files():bindings[str(p.resolve())] = digest(p)
    history = {'historical_machine_decoded_entire_validation_member':True,'validation_documents_in_member':281,
       'historical_exported_documents':32,'historical_exported_questions':104,'at_least_one_complete_member_decode':True,
       'execution_time_source_hash_in_alignment_report':False,'evidence_type':'unchanged historical Git source plus sealed completed export report; reconstructed, not new telemetry',
       'exporter_git_commit':'ab2bfcbc837f53cd9e0841857c268b2e7a176e17','exporter_sha256':EXPORTER_SHA,
       'remaining_model_prompt_metric_and_human_visibility':'not established universally; no membership in bound 32-document export',
       'model_pretraining_contamination':'unknown','legacy_orig_split_counts_inherited':screen['legacy_orig_split_counts_unique_source_rows'],
       'source_review_human_completed':False,'explicit_version_field_documents':sum(any(k in r for k in ('version','source_version','arxiv_version')) for r in original)}
    payload = {'index_rows':index,'known_exposure':exposure,'prior_development_ids':sorted(prior),'overlap_edges':edges,'source_reviews':reviews,'exposure_history':history}
    inherited = {'canonical_and_legacy_content':pool['input_sha256'],'completed_screen_sources':screen_plan['input_sha256'],'native_archive_and_sidecar':alignment['input_sha256'],'native_sidecar_sha256':alignment['sidecar_sha256']}
    verify(bindings)
    return payload, bindings, inherited


def policy_input(payload):
    return {**{k:payload[k] for k in ('index_rows','overlap_edges','source_reviews')},
            'known_exposure':{k:set(v) for k,v in payload['known_exposure'].items()},
            'prior_development_ids':set(payload['prior_development_ids'])}


def no_overlap(output, paths):
    for p in paths:
        p=Path(p).resolve()
        require(not(output==p or p.is_relative_to(output) or output.is_relative_to(p)), 'output overlaps a sealed input')


def prepare(output, run_output):
    output,run_output=Path(output).resolve(),Path(run_output).resolve()
    require(not output.exists() and not run_output.exists(), 'plan and run must be new')
    payload,bindings,inherited=source_data()
    no_overlap(output,[*bindings,run_output]); no_overlap(run_output,bindings)
    plan={'schema':SCHEMA,'status':'prepared_metadata_policy_not_executed','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'run_output':str(run_output),'input_sha256':bindings,'inherited_commitments_not_freshly_rehashed':inherited,
          'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest(),'policy_executed':False,'new_qa_read':False,'api_calls':0}
    output.mkdir(parents=True,exist_ok=False);write(output/'input.json',payload);write(output/'plan.json',plan)
    write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
    return {'status':plan['status'],'direct_bindings':len(bindings),'canonical_documents':1169,'validation_documents':281,'prior_documents':32,'remaining_documents':249,'policy_executed':False,'new_qa_read':False,'api_calls':0}


def load_plan(directory):
    directory=Path(directory).resolve();require({p.name for p in directory.iterdir()}==PLAN_FILES,'plan inventory differs')
    buffers={n:(directory/n).read_bytes() for n in PLAN_FILES}
    captured={str((directory/n).resolve()):hashlib.sha256(raw).hexdigest() for n,raw in buffers.items()}
    require(json.loads(buffers['seal.json'])=={n:captured[str((directory/n).resolve())] for n in PLAN_FILES-{'seal.json'}},'plan seal differs')
    plan=json.loads(buffers['plan.json']);payload=json.loads(buffers['input.json'])
    require(plan['schema']==SCHEMA and plan['status']=='prepared_metadata_policy_not_executed' and plan['policy_executed'] is False and plan['api_calls']==0 and plan['new_qa_read'] is False,'plan contract differs')
    verify(plan['input_sha256']); actual,bindings,inherited=source_data()
    require(actual==payload and bindings==plan['input_sha256'] and inherited==plan['inherited_commitments_not_freshly_rehashed'],'source projection changed')
    require(hashlib.sha256(canonical(payload)).hexdigest()==plan['payload_sha256'],'payload differs')
    run_output=Path(plan['run_output'])
    require(run_output.is_absolute() and str(run_output.resolve())==plan['run_output'], 'run output is not an absolute canonical path')
    no_overlap(run_output,[*bindings,directory])
    verify(captured)
    return plan,payload,captured


def execute_policy(payload):
    result = policy.evaluate_metadata(**policy_input(payload))
    return result


def output_bytes(result, history):
    ledger=result['validation_ledger'];proposed=set(result['proposed_cohort_doc_ids'])
    require(len(proposed)==len(result['proposed_cohort_doc_ids']), 'duplicate proposal identity')
    require(proposed=={r['doc_id'] for r in ledger if r['status']=='operational_candidate_proposed'}, 'proposal identity differs')
    public={**result['public_aggregate'],'exposure_history':history}
    return {'validation_ledger.jsonl':b''.join(canonical(r)+b'\n' for r in ledger),
            'proposed_cohort.jsonl':b''.join(canonical(r)+b'\n' for r in ledger if r['doc_id'] in proposed),
            'components.json':canonical(result['components'])+b'\n',
            'public_aggregate.json':canonical(public)+b'\n'}


def run(directory):
    plan,payload,plan_bindings=load_plan(directory)
    output=Path(plan['run_output']);require(not output.exists(), 'run output already exists; no retry or overwrite')
    output.mkdir(parents=True,exist_ok=False)
    registration={'schema':SCHEMA,'status':'registered_single_use_metadata_run',
                  'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':plan_bindings,
                  'new_qa_read':False,'api_calls':0}
    write(output/'run_registration.json',registration)
    result=execute_policy(payload);serialized=output_bytes(result,payload['exposure_history'])
    verify(plan['input_sha256']);verify(plan_bindings)
    for name,raw in serialized.items():
        with (output/name).open('xb') as f:f.write(raw)
    summary={'schema':SCHEMA,'status':'completed_metadata_proposal','completed_at_utc':datetime.now(timezone.utc).isoformat(),
             'plan_sha256':plan_bindings,'input_sha256':plan['input_sha256'],
             'policy_input_sha256':result['input_metadata_sha256'],
             'output_sha256':{n:digest(output/n) for n in RUN_FILES-{'summary.json'}},
             'new_qa_read':False,'api_calls':0,'cleared':False,'cohort_admitted':False}
    verify(plan['input_sha256']);verify(plan_bindings);write(output/'summary.json',summary)
    return {**result['public_aggregate'],'status':summary['status'],
            'summary_sha256':digest(output/'summary.json')}


def audit(directory, output):
    plan,payload,plan_bindings=load_plan(directory);run_output=Path(plan['run_output']);output=Path(output).resolve()
    require(not output.exists(),'audit output must be new')
    no_overlap(output,[*plan['input_sha256'],Path(directory).resolve(),run_output])
    require({p.name for p in run_output.iterdir()}==RUN_FILES,'run inventory differs')
    buffers={n:(run_output/n).read_bytes() for n in RUN_FILES}
    run_bindings={str((run_output/n).resolve()):hashlib.sha256(raw).hexdigest() for n,raw in buffers.items()}
    summary=json.loads(buffers['summary.json']);registration=json.loads(buffers['run_registration.json'])
    require(summary['schema']==SCHEMA and summary['status']=='completed_metadata_proposal' and summary['new_qa_read'] is False and summary['api_calls']==0 and summary['cleared'] is False and summary['cohort_admitted'] is False, 'run contract differs')
    require(registration['schema']==SCHEMA and registration['status']=='registered_single_use_metadata_run' and registration['new_qa_read'] is False and registration['api_calls']==0 and registration['plan_sha256']==plan_bindings, 'registration differs')
    require(summary['plan_sha256']==plan_bindings and summary['input_sha256']==plan['input_sha256'],'run source commitments differ')
    require(summary['output_sha256']=={n:run_bindings[str((run_output/n).resolve())] for n in RUN_FILES-{'summary.json'}},'run seal differs')
    result=execute_policy(payload);expected=output_bytes(result,payload['exposure_history'])
    require(summary['policy_input_sha256']==result['input_metadata_sha256'],'policy input commitment differs')
    require(all(buffers[n]==raw for n,raw in expected.items()),'run output does not match policy recomputation')
    verify(plan['input_sha256']);verify(plan_bindings);verify(run_bindings)
    receipt={'schema':SCHEMA,'status':'verified_metadata_proposal','verified_at_utc':datetime.now(timezone.utc).isoformat(),
             'direct_source_bindings':len(plan['input_sha256']),'source_sha256':plan['input_sha256'],
             'plan_sha256':plan_bindings,'run_sha256':run_bindings,
             'public_aggregate_sha256':run_bindings[str((run_output/'public_aggregate.json').resolve())],
             'new_qa_read':False,'api_calls':0,'cleared':False,'cohort_admitted':False,
             'verification_scope':'same implementation replay and complete byte/hash verification; not an independent algorithm'}
    output.mkdir(parents=True,exist_ok=False);write(output/'verification.json',receipt)
    return {'status':receipt['status'],'direct_source_bindings':receipt['direct_source_bindings'],
            'verification_sha256':digest(output/'verification.json'),'new_qa_read':False,'api_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--run-output',required=True)
    p=sub.add_parser('run');p.add_argument('--plan',required=True)
    p=sub.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.output,args.run_output)
    elif args.command=='run':result=run(args.plan)
    else:result=audit(args.plan,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
