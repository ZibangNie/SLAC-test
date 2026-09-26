"""Gold-free CPU opportunity gate for every fixed eligible relation mask.

Prepare consumes already-audited metadata and never executes the new selector.
Run/audit enumerate masks, compare independent reference traces, and report only
pack/identity/token changes. No scoring, QA loader, model/API or answer access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer
from run_qasper_evidence_baselines import PackCounter, Unit, render_pack
import qasper_relation_replay as legacy

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/research-foundation'
SCHEMA = 'slac-qasper-relation-opportunity-v1'
SPEC = {'questions': 77, 'families': 24, 'support_tasks': 1214, 'upstream_records': 1155,
    'static_edges': 562, 'edge_query_occurrences': 844, 'eligible_edge_occurrences': 107,
    'eligible_edge_histogram': {'0':31,'1':19,'2':8,'3':10,'4':6,'5':1,'6':1,'7':1},
    'masks': 501, 'max_candidates':16, 'max_units':3, 'budget':1024, 'max_seconds':300,
    'base_scores':{'yes':1.0,'unknown':0.5}, 'no_excluded':True,
    'direction':'selected B boosts preceding A', 'bonus':1, 'bonus_accumulation':False,
    'relation_filter':'both endpoints eligible, distinct exact native text',
    'tie_break':'frozen dense rank then source order',
    'render':'unchanged full [unit_id] headers, source order',
    'deduplication':'same document and exact native text',
    'candidate_scope':'given-document original frozen support pool',
    'cpu_execution':'serial mask enumeration; tokenizer parallelism disabled',
    'quality_computed':False, 'best_mask_selected':False, 'api_calls':0}
LIMITS = [
    'All 77 questions are exposed development data; no independent confirmation.',
    'Masks are hypothetical edge assignments, not model-generated relations or deployable selected masks.',
    'Per-query masks need not agree on a shared document edge; the opportunity envelope is permissive.',
    'A changed pack is a behavioral opportunity, not evidence or answer quality improvement.',
    'Fixed candidates cannot establish chunk/index/retrieval or cross-stage sharing value.',
    'Existing JSON containers may contain quality fields; the adapter immediately projects a selection whitelist.',
    'Only directly consumed inputs are rehashed; upstream provenance commitments and completed audit are inherited.',
    'No QA sidecar, reference scorer, query string, generated answer or oracle witness enters the core.',
    'The 300-second deadline is checked between bounded operations and before complete publication.',
]
SOURCE_PATHS = {
    'prepared':'qasper-extended-development-prepared-01/prepared.json',
    'manifest':'qasper-extended-development-prepared-01/manifest.json',
    'config':'qasper-primary-support-recovery-plan-01/experiment_config.json',
    'plan_manifest':'qasper-primary-support-recovery-plan-01/plan_manifest.json',
    'summary':'qasper-primary-support-recovery-run-01/summary.json',
    'labels':'qasper-primary-support-recovery-run-01/labels.json',
    'records':'qasper-primary-support-recovery-run-01/per_question.jsonl',
    'audit':'qasper-primary-support-recovery-audit-01/audit.json',
    'execution':'qasper-primary-support-recovery-audit-01/execution.json',
}
SOURCE_DIRECTORIES = {
    'qasper-extended-development-prepared-01':{'prepared.json','manifest.json'},
    'qasper-primary-support-recovery-plan-01':{'experiment_config.json','plan_manifest.json','jobs.json','inheritance.json'},
    'qasper-primary-support-recovery-run-01':{'labels.json','raw_scores.json','per_question.jsonl','traces.jsonl','summary.json','provider_calls'},
    'qasper-primary-support-recovery-audit-01':{'audit.json','execution.json'},
}
# Externally completed/audited ancestors, not same-run self-signed evidence.
ANCHORS = {
    'prepared':'b3bc8a3272d2898246aa7295aba7b366881368188c3a008b0f456e4a838cc885',
    'config':'a58e06f2b7822437eb97381cfaea72b79485995d6d1dbe67852d5b6253b6b3bd',
    'summary':'3382bd53d43980d71bb030a99f46573fbd0beb47e7464b9282521f34ca564ad3',
    'audit':'4a353f8259defe7f65230f2e1da6908e4ea60534a2df036df183832f92abdc80',
    'execution':'1481df0952b6f6aa206a8a0c8c1b07d6b3b26815a8e23ac66f37a1cfb2d17235',
}
HELPERS = ('qasper_relation_replay.py','run_qasper_evidence_baselines.py',
           'qasper_alignment_v2.py','qasper_metrics.py')
TOKENIZER_FILES = ('config.json','tokenizer.json','tokenizer_config.json',
                   'special_tokens_map.json','sentencepiece.bpe.model')
RUN_FILES = {'per_mask.jsonl','per_question.jsonl','public_aggregate.json','summary.json'}


def check_time(deadline):
    if time.monotonic() > deadline: raise TimeoutError('relation opportunity deadline exceeded')


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def object_hash(value): return hashlib.sha256(canonical(value)).hexdigest()


def digest(path, deadline=math.inf):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024): check_time(deadline); h.update(chunk)
    check_time(deadline)
    return h.hexdigest()


def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON key')
        out[key]=value
    return out


def parse(blob): return json.loads(blob,object_pairs_hook=unique_object)


def write(path, value):
    with Path(path).open('xb') as stream: stream.write(canonical(value)+b'\n')


def verify_hashes(hashes, deadline=math.inf):
    for name,sha in hashes.items():
        if digest(name,deadline)!=sha: raise ValueError('direct input hash changed: '+Path(name).name)


def snapshot(paths, deadline=math.inf):
    buffers={}; hashes={}
    for key,path in paths.items():
        check_time(deadline); blob=Path(path).read_bytes(); check_time(deadline)
        buffers[key]=blob; hashes[str(Path(path).resolve())]=hashlib.sha256(blob).hexdigest()
    return buffers,hashes


def runtime():
    return {'python':platform.python_version(), 'transformers':version('transformers'),
            'tokenizers':version('tokenizers'), 'tokenizer_parallelism':False}


@dataclass(frozen=True)
class Case:
    family_id: str
    doc_id: str
    question_id: str
    units: tuple[Unit,...]
    candidates: tuple[int,...]
    ranking: tuple[int,...]
    labels: tuple[str,...]  # aligned with candidates, never reference labels
    baseline_selected: tuple[int,...]
    baseline_tokens: int
    baseline_pack_sha256: str


def validate_case(case):
    if type(case) is not Case or any(not isinstance(getattr(case,k),str) or not getattr(case,k)
                                   for k in ('family_id','doc_id','question_id')):
        raise ValueError('invalid core case identity')
    if not case.units or any(type(u) is not Unit for u in case.units): raise ValueError('invalid units')
    if (any(type(u.order) is not int or u.order!=i or not isinstance(u.unit_id,str) or not u.unit_id for i,u in enumerate(case.units))
        or len({u.unit_id for u in case.units})!=len(case.units)):
        raise ValueError('units are not a complete ordered document')
    if any(not isinstance(u.text,str) or not u.text.strip() or not isinstance(u.native_text,str)
           or not u.native_text.strip() for u in case.units): raise ValueError('empty native unit')
    if (not 0<len(case.candidates)<=SPEC['max_candidates'] or any(type(i) is not int or not 0<=i<len(case.units) for i in case.candidates)
        or tuple(sorted(set(case.candidates)))!=case.candidates
        or any(type(i) is not int for i in case.ranking) or len(case.ranking)!=len(case.candidates)
        or set(case.ranking)!=set(case.candidates) or len(case.labels)!=len(case.candidates)
        or any(label not in ('yes','no','unknown') for label in case.labels)):
        raise ValueError('invalid candidates, ranking or support labels')
    if (any(type(i) is not int for i in case.baseline_selected) or tuple(sorted(set(case.baseline_selected)))!=case.baseline_selected
        or not set(case.baseline_selected)<=set(case.candidates) or len(case.baseline_selected)>SPEC['max_units']
        or type(case.baseline_tokens) is not int or not 0<=case.baseline_tokens<=SPEC['budget']
        or not isinstance(case.baseline_pack_sha256,str) or len(case.baseline_pack_sha256)!=64
        or any(c not in '0123456789abcdef' for c in case.baseline_pack_sha256)):
        raise ValueError('invalid saved baseline witness')


def eligible_edges(case):
    validate_case(case)
    eligible={i for i,label in zip(case.candidates,case.labels) if label!='no'}
    return tuple((i,i+1) for i in case.candidates if i in eligible and i+1 in eligible
                 and case.units[i].native_text!=case.units[i+1].native_text)


def inventory(cases):
    if len(cases)!=SPEC['questions'] or len({c.family_id for c in cases})!=SPEC['families']:
        raise ValueError('incomplete question/family denominator')
    identities=[(c.doc_id,c.question_id) for c in cases]
    if len(set(identities))!=len(cases): raise ValueError('duplicate core question')
    histogram=Counter(len(eligible_edges(c)) for c in cases)
    occurrences=sum(i+1 in c.candidates for c in cases for i in c.candidates)
    all_edges={(c.doc_id,i,i+1) for c in cases for i in c.candidates if i+1 in c.candidates}
    result={'questions':len(cases),'families':len({c.family_id for c in cases}),
        'static_edges':len(all_edges),'edge_query_occurrences':occurrences,
        'eligible_edge_occurrences':sum(k*v for k,v in histogram.items()),
        'eligible_edge_histogram':{str(k):v for k,v in sorted(histogram.items())},
        'masks':sum((1<<k)*v for k,v in histogram.items())}
    if any(result[k]!=SPEC[k] for k in result): raise ValueError('fixed mask inventory differs')
    return result


def safe_count(count, selected):
    value=count(sorted(selected))
    if type(value) is not int or value<0: raise ValueError('invalid complete-pack token count')
    return value


def finish(case, selected, trace, count):
    selected=sorted(selected)
    tokens=safe_count(count,selected)
    if tokens>SPEC['budget'] or len(selected)>SPEC['max_units']: raise ValueError('illegal final pack')
    if len({case.units[i].native_text for i in selected})!=len(selected): raise ValueError('duplicate final native text')
    return {'selected_indices':selected,'selected_ids':[case.units[i].unit_id for i in selected],
            'actual_tokens':tokens,'pack_sha256':hashlib.sha256(render_pack(case.units,selected).encode('utf-8')).hexdigest(),
            'trace':trace}


def select(case, active_edges, count, deadline=math.inf):
    """Pure selection: only Case fields, directed active edges and pack counter."""
    possible=set(eligible_edges(case)); edges=tuple(active_edges)
    if any(not isinstance(e,(tuple,list)) or len(e)!=2 or any(type(i) is not int for i in e) for e in edges):
        raise ValueError('invalid edge representation')
    edges=tuple(tuple(e) for e in edges)
    if len(set(edges))!=len(edges) or not set(edges)<=possible: raise ValueError('invalid active relation edges')
    scores={i:SPEC['base_scores'][l] for i,l in zip(case.candidates,case.labels) if l!='no'}
    rank={i:r for r,i in enumerate(case.ranking)}; pending=set(scores); selected=[]; text=set(); trace=[]
    while pending and len(selected)<SPEC['max_units']:
        check_time(deadline)
        linked={i:sorted([a,b] for a,b in edges if a==i and b in selected) for i in pending}
        independent=min(pending,key=lambda i:(-scores[i],rank[i],case.units[i].order))
        chosen=min(pending,key=lambda i:(-(scores[i]+bool(linked[i])),rank[i],case.units[i].order))
        pending.remove(chosen)
        row={'step':len(trace),'candidate_index':chosen,'selected_before':sorted(selected),
             'base_score':scores[chosen],'relation_bonus':int(bool(linked[chosen])),
             'active_triggers':linked[chosen],'priority_changed':chosen!=independent,
             'proposed_tokens':None,'accepted':False}
        if case.units[chosen].native_text in text: row['action']='skip_exact_native_duplicate'
        else:
            row['proposed_tokens']=safe_count(count,[*selected,chosen])
            if row['proposed_tokens']<=SPEC['budget']:
                selected.append(chosen); text.add(case.units[chosen].native_text)
                row.update(action='select',accepted=True)
            else: row['action']='skip_evidence_budget'
        trace.append(row)
    check_time(deadline)
    return finish(case,selected,trace,count)


def independent_reference(case, tokenizer, deadline):
    """Old pure I selector only; never old loaders, verification or scoring."""
    relevance={case.units[i].unit_id:l for i,l in zip(case.candidates,case.labels)}
    old=legacy.replay_policy(case.units,[case.units[i].unit_id for i in case.candidates],relevance,[],
        [case.units[i].unit_id for i in case.ranking],mode='I',tokenizer=tokenizer,
        budget=SPEC['budget'],chunk_budget=384,max_units=SPEC['max_units'],deadline=deadline)
    by_id={u.unit_id:i for i,u in enumerate(case.units)}
    trace=[{'step':r['step'],'candidate_index':by_id[r['unit_id']],
            'selected_before':[by_id[uid] for uid in r['selected_before']], 'base_score':r['base_score'],
            'relation_bonus':int(r['relation_bonus']),'active_triggers':[],
            'priority_changed':r['relation_changed_priority'],'proposed_tokens':r['proposed_tokens'],
            'accepted':r['accepted'],'action':r['action']} for r in old['selection_trace']]
    return finish(case,old['selected_indices'],trace,PackCounter(tokenizer,case.units,deadline))


def adjacency_reference(case, count, deadline=math.inf):
    """Independent list-scan implementation; no call to select or graph mask."""
    validate_case(case)
    base={i:1.0 if l=='yes' else 0.5 for i,l in zip(case.candidates,case.labels) if l!='no'}
    priority={i:p for p,i in enumerate(case.ranking)}
    todo=list(base); chosen=[]; trace=[]
    while todo and len(chosen)<SPEC['max_units']:
        check_time(deadline)
        score=[]
        for i in todo:
            linked=(i+1 in chosen and case.units[i].native_text!=case.units[i+1].native_text)
            score.append((-(base[i]+int(linked)),priority[i],case.units[i].order,i,linked))
        winner=min(score); i=winner[3]; linked=winner[4]
        first=sorted(todo,key=lambda j:(-base[j],priority[j],case.units[j].order))[0]
        todo.remove(i)
        row={'step':len(trace),'candidate_index':i,'selected_before':sorted(chosen),'base_score':base[i],
            'relation_bonus':int(linked),'active_triggers':[[i,i+1]] if linked else [],
            'priority_changed':i!=first,'proposed_tokens':None,'accepted':False}
        if any(case.units[j].native_text==case.units[i].native_text for j in chosen): row['action']='skip_exact_native_duplicate'
        else:
            row['proposed_tokens']=safe_count(count,chosen+[i])
            if row['proposed_tokens']<=SPEC['budget']:
                chosen.append(i); row.update(action='select',accepted=True)
            else: row['action']='skip_evidence_budget'
        trace.append(row)
    return finish(case,chosen,trace,count)


def case_to_json(case): return asdict(case)


def case_from_json(value):
    if set(value)!=set(Case.__dataclass_fields__): raise ValueError('core whitelist differs')
    fields={**value,'units':tuple(Unit(**u) for u in value['units'])}
    for k in ('candidates','ranking','labels','baseline_selected'): fields[k]=tuple(fields[k])
    case=Case(**fields); validate_case(case); return case


def project(prepared, labels, rows):
    """Immediately discard query text/task payloads and all per-question quality."""
    docs={d:tuple(Unit(**{k:u[k] for k in Unit.__dataclass_fields__}) for u in values)
          for d,values in prepared['documents'].items()}
    queries=[{k:q[k] for k in ('family_id','doc_id','question_id','candidate_ids','ranked_ids')}
             for q in prepared['queries']]
    tasks=[{k:t[k] for k in ('id','doc_id','question_id','unit_id')} for t in prepared['support_tasks']]
    static=[{k:t[k] for k in ('doc_id','left_id','right_id')} for t in prepared['static_tasks']]
    witnesses=[{k:r[k] for k in ('family_id','doc_id','question_id','selected_ids','actual_evidence_tokens','pack_sha256')}
               for r in rows if r['method']=='I_jev_k3']
    if len(rows)!=SPEC['upstream_records'] or len(tasks)!=SPEC['support_tasks']:
        raise ValueError('upstream complete counts differ')
    task_map={(t['doc_id'],t['question_id'],t['unit_id']):t['id'] for t in tasks}
    if len(task_map)!=len(tasks) or set(labels)!=set(task_map.values()): raise ValueError('incomplete support task coverage')
    expected={(q['doc_id'],q['question_id']) for q in queries}
    saved={(r['doc_id'],r['question_id']):r for r in witnesses}
    if len(saved)!=len(witnesses) or set(saved)!=expected: raise ValueError('incomplete original I baseline')
    cases=[]; seen_edges=set()
    for q in queries:
        d,qi=q['doc_id'],q['question_id']; us=docs[d]; by_id={u.unit_id:i for i,u in enumerate(us)}; r=saved[d,qi]
        if r['family_id']!=q['family_id']: raise ValueError('baseline family differs')
        indices=tuple(by_id[uid] for uid in q['candidate_ids'])
        case=Case(q['family_id'],d,qi,us,indices,tuple(by_id[uid] for uid in q['ranked_ids']),
            tuple(labels[task_map[d,qi,us[i].unit_id]] for i in indices),tuple(by_id[uid] for uid in r['selected_ids']),
            r['actual_evidence_tokens'],r['pack_sha256'])
        validate_case(case); cases.append(case)
        seen_edges.update((d,us[i].unit_id,us[i+1].unit_id) for i in indices if i+1 in indices)
    if len(static)!=SPEC['static_edges'] or {(e['doc_id'],e['left_id'],e['right_id']) for e in static}!=seen_edges:
        raise ValueError('static edge metadata coverage differs')
    inventory(cases)
    return cases


def own_paths():
    return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_opportunity.py',
            ROOT/'docs/research/RELATION_OPPORTUNITY_PROTOCOL_20260927.md']


def verify_source_directories():
    for directory,names in SOURCE_DIRECTORIES.items():
        if {p.name for p in (ARTIFACTS/directory).iterdir()}!=names:
            raise ValueError('audited direct-source directory inventory differs')


def source_snapshot(deadline=math.inf):
    verify_source_directories()
    paths={k:(ARTIFACTS/p).resolve() for k,p in SOURCE_PATHS.items()}
    buffers,hashes=snapshot(paths,deadline)
    for key,expected in ANCHORS.items():
        if hashes[str(paths[key])]!=expected: raise ValueError('external completed audit anchor differs: '+key)
    config=parse(buffers['config']); summary=parse(buffers['summary']); audit=parse(buffers['audit']); execution=parse(buffers['execution'])
    manifest=parse(buffers['manifest']); plan_manifest=parse(buffers['plan_manifest'])
    if (audit.get('status')!='verified_complete' or audit.get('schema')!='slac-qasper-primary-support-recovery-v1-audit'
        or audit.get('question_count')!=SPEC['questions'] or audit.get('records')!=SPEC['upstream_records']
        or audit.get('all_inputs_outputs_unchanged') is not True or audit.get('api_calls')!=0 or audit.get('key_read') is not False
        or execution.get('schema')!='slac-recovery-external-execution-receipt-v1'
        or execution.get('status')!='completed_and_audited' or execution.get('child_pid') is not None
        or execution.get('audit_exit_code')!=0 or execution.get('run_exit_code')!=0
        or execution.get('audit_sha256')!=hashes[str(paths['audit'])] or execution.get('summary_sha256')!=hashes[str(paths['summary'])]):
        raise ValueError('completed source audit/execution receipt required')
    if (summary.get('schema')!='slac-qasper-primary-support-recovery-v1' or summary.get('status')!='completed'
        or summary.get('all_results_available') is not True or summary.get('question_count')!=SPEC['questions']
        or summary.get('family_count')!=SPEC['families'] or summary.get('record_count')!=SPEC['upstream_records']
        or summary.get('method_count')!=15 or summary.get('plan_sha256')!=hashes[str(paths['config'])]
        or plan_manifest!={'experiment_config_sha256':hashes[str(paths['config'])]}
        or manifest.get('status')!='prepared' or manifest.get('schema')!='slac-qasper-extended-development-manifest-v1'
        or manifest.get('prepared_sha256')!=hashes[str(paths['prepared'])]
        or config.get('schema')!='slac-qasper-primary-support-recovery-v1' or config.get('status')!='prepared_not_executed'
        or config.get('prepared_dir')!=str(paths['prepared'].parent)):
        raise ValueError('source version, count or preparation lineage differs')
    for name in ('labels','records'):
        if summary['output_sha256'][paths[name].name]!=hashes[str(paths[name])]: raise ValueError('audited output seal differs')
    upstream={**config['input_sha256'],**summary['execution_input_sha256']}
    if any(k in config['input_sha256'] and config['input_sha256'][k]!=v for k,v in summary['execution_input_sha256'].items()):
        raise ValueError('conflicting upstream commitment')
    if object_hash(upstream)!=summary['input_binding_sha256']: raise ValueError('ancestor commitment differs')
    tokenizer=Path(config['tokenizer']).resolve()
    helper_paths=[ROOT/'docs/research'/name for name in HELPERS]
    tokenizer_paths=[tokenizer/name for name in TOKENIZER_FILES if (tokenizer/name).is_file()]
    if not {'tokenizer.json','tokenizer_config.json','config.json'}<=set(p.name for p in tokenizer_paths):
        raise ValueError('incomplete local tokenizer inventory')
    for p in [paths['prepared'],paths['manifest'],*helper_paths,*tokenizer_paths]:
        expected=config['input_sha256'].get(str(p.resolve()))
        actual=digest(p,deadline)
        if expected!=actual: raise ValueError('source/helper/tokenizer absent or changed in audited lineage: '+p.name)
        hashes[str(p.resolve())]=actual
    for p in own_paths(): hashes[str(p)]=digest(p,deadline)
    prepared=parse(buffers['prepared']); label_container=parse(buffers['labels'])
    if prepared.get('schema')!='slac-qasper-extended-development-prepared-v1':
        raise ValueError('prepared core source version differs')
    rows=[parse(line) for line in buffers['records'].splitlines() if line.strip()]
    cases=project(prepared,label_container['jev'],rows)
    meta={'prepared_sha256':hashes[str(paths['prepared'])], 'support_summary_sha256':hashes[str(paths['summary'])],
          'support_audit_sha256':hashes[str(paths['audit'])], 'support_execution_sha256':hashes[str(paths['execution'])],
          'upstream_input_binding_sha256':summary['input_binding_sha256'],
          'inherited_upstream_commitment_count':len(upstream), 'rehashed_upstream_all':False,
          'direct_source_inventory':{str(p):hashes[str(p)] for p in paths.values()},
          'source_directory_inventories':{name:sorted(files) for name,files in SOURCE_DIRECTORIES.items()},
          'tokenizer_file_names':sorted(p.name for p in tokenizer_paths),
          'legacy_helper_names':list(HELPERS),'old_quality_fields_projected_away':True}
    verify_hashes(hashes,deadline)
    verify_source_directories()
    return cases,meta,str(tokenizer),hashes


def prepare(args):
    output,run_output=Path(args.output).resolve(),Path(args.run_output).resolve()
    if output.exists() or run_output.exists(): raise FileExistsError('plan/run must be new single-use directories')
    forbidden={Path(ARTIFACTS/p).resolve().parent for p in SOURCE_PATHS.values()}
    if (output==run_output or output.is_relative_to(run_output) or run_output.is_relative_to(output)
        or any(new.is_relative_to(old) or old.is_relative_to(new) for new in (output,run_output) for old in forbidden)):
        raise ValueError('new outputs overlap audited sources or one another')
    cases,meta,tokenizer,hashes=source_snapshot()
    inv=inventory(cases)
    core=[case_to_json(c) for c in cases]
    config={'schema':SCHEMA,'status':'prepared_not_enumerated','specification':SPEC,'limits':LIMITS,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime(),'run_output':str(run_output),
        'tokenizer':tokenizer,'input_sha256':hashes,'input_binding_sha256':object_hash(hashes),
        'source_metadata':meta,'inventory':inv,'cases_object_sha256':object_hash(core),
        'api_calls':0,'qa_sidecar_opened':False,'query_strings_in_core':False,'selector_executed':False}
    verify_hashes(hashes)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'cases.json',core); write(output/'plan.json',config)
    write(output/'seal.json',{name:digest(output/name) for name in ('cases.json','plan.json')})
    return config


def load_plan(directory,deadline=math.inf):
    directory=Path(directory).resolve()
    if {p.name for p in directory.iterdir()}!={'plan.json','cases.json','seal.json'}:
        raise ValueError('plan file inventory differs')
    buffers,own=snapshot({n:directory/n for n in ('plan.json','cases.json','seal.json')},deadline)
    seal=parse(buffers['seal.json'])
    if seal!={n:own[str(directory/n)] for n in ('plan.json','cases.json')}:
        raise ValueError('plan seal mismatch')
    config=parse(buffers['plan.json'])
    if (config.get('schema')!=SCHEMA or config.get('status')!='prepared_not_enumerated'
        or config.get('specification')!=SPEC or config.get('limits')!=LIMITS or config.get('runtime')!=runtime()
        or config.get('api_calls')!=0 or config.get('selector_executed') is not False
        or config.get('qa_sidecar_opened') is not False or config.get('query_strings_in_core') is not False
        or object_hash(config['input_sha256'])!=config['input_binding_sha256']):
        raise ValueError('frozen plan contract differs')
    verify_hashes(config['input_sha256'],deadline)
    cases,meta,tokenizer,hashes=source_snapshot(deadline)
    core=parse(buffers['cases.json'])
    if (hashes!=config['input_sha256'] or meta!=config['source_metadata'] or tokenizer!=config['tokenizer']
        or object_hash(core)!=config['cases_object_sha256'] or core!=parse(canonical([case_to_json(c) for c in cases]))
        or inventory(cases)!=config['inventory']):
        raise ValueError('saved core inputs differ from audited whitelist')
    # Round-trip exact key/type checks ensure no hidden quality/query field survives.
    decoded=[case_from_json(c) for c in core]
    return config,decoded,own


def evaluate(cases,tokenizer,deadline=math.inf):
    inv=inventory(cases); mask_rows=[]; query_rows=[]
    for case in cases:
        check_time(deadline)
        edges=eligible_edges(case); count=PackCounter(tokenizer,case.units,deadline)
        zero=select(case,(),count,deadline)
        if zero!=independent_reference(case,tokenizer,deadline):
            raise ValueError('zero-mask independent step trace parity failed')
        if (zero['selected_indices']!=list(case.baseline_selected)
            or zero['pack_sha256']!=case.baseline_pack_sha256 or zero['actual_tokens']!=case.baseline_tokens):
            raise ValueError('zero mask differs from frozen I baseline pack')
        all_edges=select(case,edges,count,deadline)
        if all_edges!=adjacency_reference(case,PackCounter(tokenizer,case.units,deadline),deadline):
            raise ValueError('full-mask independent adjacency trace parity failed')
        identity={k:getattr(case,k) for k in ('family_id','doc_id','question_id')}
        q_masks=[]
        for mask in range(1<<len(edges)):
            check_time(deadline)
            active=[edge for bit,edge in enumerate(edges) if mask&(1<<bit)]
            result=select(case,active,count,deadline)
            if mask==0 and result!=zero or mask==(1<<len(edges))-1 and result!=all_edges:
                raise ValueError('mask endpoint parity changed during enumeration')
            before=set(zero['selected_indices']); after=set(result['selected_indices'])
            row={**identity,'mask':mask,'active_edges':[list(e) for e in active],**result,
                 'added_indices':sorted(after-before),'removed_indices':sorted(before-after),
                 'tokens_minus_zero':result['actual_tokens']-zero['actual_tokens'],
                 'selected_count_minus_zero':len(after)-len(before),
                 'pack_changed':result['pack_sha256']!=zero['pack_sha256']}
            mask_rows.append(row); q_masks.append(row)
        unique={(r['pack_sha256'],tuple(r['selected_indices'])) for r in q_masks}
        tokens=[r['actual_tokens'] for r in q_masks]
        query_rows.append({**identity,'eligible_edges':[list(e) for e in edges], 'mask_count':len(q_masks),
            'distinct_pack_count':len(unique),'any_pack_change':any(r['pack_changed'] for r in q_masks),
            'changed_masks':sum(r['pack_changed'] for r in q_masks),
            'minimum_actual_tokens':min(tokens),'maximum_actual_tokens':max(tokens),
            'zero_tokens':zero['actual_tokens'],'zero_pack_sha256':zero['pack_sha256'],
            'zero_selected_indices':zero['selected_indices'],'full_tokens':all_edges['actual_tokens'],
            'full_pack_changed':all_edges['pack_sha256']!=zero['pack_sha256'],
            'zero_independent_trace_parity':True,'zero_frozen_pack_parity':True,'full_adjacency_trace_parity':True})
    check_time(deadline)
    if len(mask_rows)!=inv['masks'] or len(query_rows)!=inv['questions']: raise ValueError('incomplete mask enumeration')
    return mask_rows,query_rows,aggregate(mask_rows,query_rows,inv)


def histogram(values): return {str(k):v for k,v in sorted(Counter(values).items())}


def aggregate(mask_rows,query_rows,inv):
    if len(mask_rows)!=inv['masks'] or len(query_rows)!=inv['questions']: raise ValueError('incomplete aggregate')
    tokens=[r['actual_tokens'] for r in mask_rows]; deltas=[r['tokens_minus_zero'] for r in mask_rows]
    return {'schema':SCHEMA+'-public','status':'complete_opportunity_enumeration','specification':SPEC,'limits':LIMITS,
        'inventory':inv,'zero_independent_trace_parity_questions':sum(r['zero_independent_trace_parity'] for r in query_rows),
        'zero_frozen_pack_parity_questions':sum(r['zero_frozen_pack_parity'] for r in query_rows),
        'full_adjacency_trace_parity_questions':sum(r['full_adjacency_trace_parity'] for r in query_rows),
        'queries_with_any_eligible_edge':sum(bool(r['eligible_edges']) for r in query_rows),
        'queries_with_any_possible_pack_change':sum(r['any_pack_change'] for r in query_rows),
        'full_adjacency_changed_pack_questions':sum(r['full_pack_changed'] for r in query_rows),
        'changed_masks':sum(r['pack_changed'] for r in mask_rows),
        'unchanged_masks':sum(not r['pack_changed'] for r in mask_rows),
        'distinct_pack_count_per_question_histogram':histogram(r['distinct_pack_count'] for r in query_rows),
        'added_removed_count_histogram':dict(sorted(Counter(f"{len(r['added_indices'])},{len(r['removed_indices'])}" for r in mask_rows).items())),
        'selected_count_minus_zero_histogram':histogram(r['selected_count_minus_zero'] for r in mask_rows),
        'actual_tokens_range':[min(tokens),max(tokens)],'tokens_minus_zero_range':[min(deltas),max(deltas)],
        'tokens_minus_zero_histogram':histogram(deltas),
        'per_question_actual_token_range_width_histogram':histogram(r['maximum_actual_tokens']-r['minimum_actual_tokens'] for r in query_rows),
        'api_calls':0,'encoder_model_loaded':False,'gpu_used':False,'qa_sidecar_opened':False,
        'quality_metrics_computed':False,'best_mask_selected':False,'private_ids_in_public_output':False}


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)


def public_with_execution(public,config,timing):
    return {**public,'runtime':config['runtime'],'execution_timing':timing,
            'input_binding_sha256':config['input_binding_sha256'],
            'source_summary_sha256':config['source_metadata']['support_summary_sha256'],
            'source_audit_sha256':config['source_metadata']['support_audit_sha256']}


def expected_summary(config,own,output_hashes,timing):
    return {'schema':SCHEMA,'status':'completed','all_masks_available':True,'specification':SPEC,
        'inventory':config['inventory'],'plan_sha256':own,'input_sha256':config['input_sha256'],
        'input_binding_sha256':config['input_binding_sha256'],'source_metadata':config['source_metadata'],
        'runtime':config['runtime'],'execution_timing':timing,'output_sha256':output_hashes,
        'api_calls':0,'quality_metrics_computed':False,'qa_sidecar_opened':False,'query_strings_in_core':False,
        'encoder_model_loaded':False,'gpu_used':False,'private_traces_local_only':True}


def write_rows(path,rows,deadline):
    with Path(path).open('xb') as stream:
        for row in rows: check_time(deadline); stream.write(canonical(row)+b'\n')


def run(args,tokenizer_factory=load_tokenizer):
    started=time.monotonic(); deadline=started+SPEC['max_seconds']
    config,cases,own=load_plan(args.plan,deadline)
    output=Path(config['run_output']).resolve()
    output.mkdir(parents=True,exist_ok=False)
    phase='tokenizer_loading'
    try:
        check_time(deadline); tokenizer=tokenizer_factory(config['tokenizer']); check_time(deadline)
        loaded=time.monotonic(); phase='enumeration'
        masks,queries,public=evaluate(cases,tokenizer,deadline)
        enumerated=time.monotonic(); phase='final_source_verification'
        verify_hashes({**config['input_sha256'],**own},deadline)
        phase='output_writing'
        write_rows(output/'per_mask.jsonl',masks,deadline); write_rows(output/'per_question.jsonl',queries,deadline)
        timing={'load_and_validation_seconds':loaded-started,'enumeration_seconds':enumerated-loaded,
                'post_enumeration_seconds_before_summary':time.monotonic()-enumerated}
        timing['total_seconds_before_summary']=sum(timing.values())
        write(output/'public_aggregate.json',public_with_execution(public,config,timing))
        hashes={name:digest(output/name,deadline) for name in RUN_FILES-{'summary.json'}}
        check_time(deadline)
        write(output/'summary.json',expected_summary(config,own,hashes,timing))
        verify_hashes({**config['input_sha256'],**own},deadline)
        verify_source_directories()
        check_time(deadline)
        return {'status':'completed','questions':len(queries),'masks':len(masks),
                'summary_sha256':digest(output/'summary.json',deadline),'quality_metrics_computed':False,'api_calls':0}
    except BaseException as exc:
        if (output/'summary.json').exists(): (output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_result','phase':phase,
              'error_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-started,
              'api_calls':0,'quality_metrics_computed':False})
        raise


def validate_timing(timing):
    names={'load_and_validation_seconds','enumeration_seconds','post_enumeration_seconds_before_summary','total_seconds_before_summary'}
    if (set(timing)!=names or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in timing.values())
        or timing['total_seconds_before_summary']>SPEC['max_seconds']
        or abs(sum(timing[k] for k in names-{'total_seconds_before_summary'})-timing['total_seconds_before_summary'])>1e-8):
        raise ValueError('invalid measured execution timing')


def audit(args,tokenizer_factory=load_tokenizer):
    started=time.monotonic();deadline=started+SPEC['max_seconds']
    config,cases,own=load_plan(args.plan,deadline)
    output=Path(args.run).resolve()
    if output!=Path(config['run_output']).resolve() or {p.name for p in output.iterdir()}!=RUN_FILES:
        raise ValueError('complete fixed-run file inventory required')
    buffers,run_hashes=snapshot({name:output/name for name in RUN_FILES},deadline)
    summary=parse(buffers['summary.json']); timing=summary.get('execution_timing',{})
    validate_timing(timing)
    files={n:run_hashes[str(output/n)] for n in RUN_FILES-{'summary.json'}}
    if summary!=expected_summary(config,own,files,timing): raise ValueError('summary metadata or output binding differs')
    tokenizer=tokenizer_factory(config['tokenizer']);check_time(deadline)
    masks,queries,public=evaluate(cases,tokenizer,deadline)
    if ([parse(line) for line in buffers['per_mask.jsonl'].splitlines() if line.strip()]!=masks
        or [parse(line) for line in buffers['per_question.jsonl'].splitlines() if line.strip()]!=queries
        or parse(buffers['public_aggregate.json'])!=public_with_execution(public,config,timing)):
        raise ValueError('saved mask/trace/aggregate differs from complete replay')
    verify_hashes({**config['input_sha256'],**own,**run_hashes},deadline)
    verify_source_directories()
    return {'schema':SCHEMA+'-audit','status':'verified_complete','questions':len(queries),'masks':len(masks),
            'all_masks_and_reference_traces_recomputed':True,'all_direct_input_hashes_unchanged':True,
            'source_audit_sha256':config['source_metadata']['support_audit_sha256'],
            'input_binding_sha256':config['input_binding_sha256'],'run_sha256':run_hashes,
            'measured_timing_reexecuted_or_independently_proven':False,
            'api_calls':0,'quality_metrics_computed':False,'qa_sidecar_opened':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare'); p.add_argument('--output',required=True);p.add_argument('--run-output',required=True)
    p=subs.add_parser('run');p.add_argument('--plan',required=True)
    p=subs.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    args=parser.parse_args()
    result={'prepare':prepare,'run':run,'audit':audit}[args.command](args)
    print(json.dumps(result if args.command=='audit' else {k:result[k] for k in ('status','api_calls')},indent=2))
