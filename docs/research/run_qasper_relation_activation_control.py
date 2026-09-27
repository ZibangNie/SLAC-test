"""Activation-only exact control using already verified whole-pack token measurements."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import run_qasper_relation_lazy_bounds as parent

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT/'artifacts/research-foundation'
SCHEMA = 'slac-qasper-relation-activation-control-v1'
RECEIPT_SHA = 'e8c69eea1f90f4bd0662971f3f36f5a298c5633c6ebf0e324a401f747798a57e'
AUDIT_SHA = '91c8ff9d675eed760cc04dc6a2aa780c050793cdefcf79a25088a70bc4f13b0a'
COMPLETED_SHA = '9aec128ce822dee0f32ba2aefbbd084c08a17cd8d1054ba1d31b971e5ac49ca2'
SPEC = {'questions':77,'families':24,'empty_cache_paths':501,'assignment_known_subset_paths':23915,
        'max_units':3,'budget':1024,'maximum_new_reads':2,'max_seconds':300,
        'policy':'read all unknown active edges in native order before each exact winner',
        'token_measurement':'exact replay of independently verified saved whole-pack BGE counts',
        'comparison':'eligible minus activation; activation minus lazy',
        'api_calls':0,'quality_computed':False,'new_tokenizer_encodes':0}
LIMITS = [
    'The two-read maximum follows from the directed adjacency activation structure, not interval certification.',
    'All-subset paths include the empty-cache paths; assignments are combinatorial and not label probabilities.',
    'Only activation-minus-lazy measures additional interval-certificate reduction against this fixed control.',
    'Saved whole-pack token measurements are replayed; timing is not serving latency or a tokenizer benchmark.',
    'Nonessential reads and zero savings are retained for both policies; no optimality is claimed.',
    'Logical reads are not HTTP calls, fees, quality or end-to-end savings.',
    'No cross-query shared-cache schedule or globally consistent labeling is executed.',
    'Unconsumed ancestor hashes are inherited commitments, not freshly rehashed source files.',
    'The in-process deadline is cooperative; external process timeout is supplied separately by Root.',
]
PARENT_PLAN = ARTIFACTS/'qasper-relation-lazy-bounds-plan-01'
PARENT_RUN = ARTIFACTS/'qasper-relation-lazy-bounds-run-01'
PARENT_EXEC = ARTIFACTS/'qasper-relation-lazy-bounds-root-execution-01'
SOURCE_PATHS = {
    'receipt':ARTIFACTS/'qasper-relation-lazy-bounds-independent-verification-01/result-01/verification.json',
    'audit':PARENT_EXEC/'audit.json','completed':PARENT_EXEC/'completed.json',
    **{'plan_'+n:PARENT_PLAN/n for n in parent.PLAN_FILES},
    **{'run_'+n:PARENT_RUN/n for n in parent.RUN_FILES},
}
PLAN_FILES = {'plan.json','cores.json','seal.json'}
RUN_FILES = {'empty_cache.jsonl','all_known_subsets.jsonl','per_question.jsonl','public_aggregate.json','summary.json'}
canonical = parent.canonical
parse = parent.parse
object_hash = parent.object_hash
digest = parent.digest
verify_hashes = parent.verify_hashes
bound_read = parent.bound_read
write = parent.write
rows_write = parent.rows_write
check = parent.check


def own_paths():
    return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_activation_control.py',
            ROOT/'docs/research/RELATION_ACTIVATION_CONTROL_PROTOCOL_20260927.md']


def runtime():
    return {'python':platform.python_version(),'execution':'CPU saved-token replay','tokenizer_loaded':False}


def parent_inventory():
    if {p.name for p in PARENT_PLAN.iterdir()} != parent.PLAN_FILES or {p.name for p in PARENT_RUN.iterdir()} != parent.RUN_FILES:
        raise ValueError('parent directory inventory differs')


def select(core, ask, count, known=None, deadline=math.inf):
    """No cube, lazy path, essentiality, assignment or quality enters this policy."""
    parent.validate_core(core)
    edges = parent.edges_of(core)
    cache = parent.validate_cache({} if known is None else known,len(edges))
    if not callable(ask) or not callable(count): raise ValueError('callbacks must be callable')
    pending = {i:l for i,l in zip(core.candidates,core.labels) if l != 'no'}
    ranks = {i:r for r,i in enumerate(core.ranking)}
    selected = []; texts = set(); requests = []; events = []
    while pending and len(selected) < SPEC['max_units']:
        check(deadline)
        for j,(a,b) in enumerate(edges):
            if a in pending and b in selected and j not in cache:
                check(deadline); value = ask(j); check(deadline)
                if type(value) is not bool: raise ValueError('oracle must return strict bool')
                cache[j] = value; requests.append(j)
                events.append({'kind':'request','variable':j,'value':value,'selected_before':sorted(selected)})
        def key(i):
            bonus = any(a == i and b in selected and cache[j] for j,(a,b) in enumerate(edges))
            return (-(2 if pending[i] == 'yes' else 1)-2*int(bonus),ranks[i],core.units[i].order)
        chosen = min(pending,key=key); del pending[chosen]
        event = {'kind':'winner','candidate':chosen,'selected_before':sorted(selected),
                 'proposed_tokens':None,'action':'skip_exact_native_duplicate'}
        if core.units[chosen].native_text not in texts:
            event['proposed_tokens'] = parent.safe_count(count,[*selected,chosen])
            if event['proposed_tokens'] <= SPEC['budget']:
                selected.append(chosen); texts.add(core.units[chosen].native_text); event['action'] = 'select'
            else: event['action'] = 'skip_evidence_budget'
        events.append(event)
    selected.sort(); rendered = parent.render_pack(core.units,selected)
    result = {'selected_indices':selected,'selected_ids':[core.units[i].unit_id for i in selected],
              'rendered_pack':rendered,'pack_sha256':hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
              'actual_tokens':parent.safe_count(count,selected),'requests':requests,
              'final_known':[[j,cache[j]] for j in sorted(cache)],'events':events}
    check(deadline)
    return result


def source_data(deadline=math.inf):
    """Metadata-only admission: never parse saved paths, essentiality or run a selector."""
    hashes = {str(p.resolve()):digest(p,deadline) for p in SOURCE_PATHS.values()}
    for name,pin in [('receipt',RECEIPT_SHA),('audit',AUDIT_SHA),('completed',COMPLETED_SHA)]:
        if hashes[str(SOURCE_PATHS[name].resolve())] != pin: raise ValueError('fixed completed source differs')
    receipt = bound_read(SOURCE_PATHS['receipt'],hashes,deadline)
    if (receipt.get('status') != 'verified_complete' or receipt.get('questions') != 77 or receipt.get('families') != 24
        or receipt.get('empty_cache_paths') != 501 or receipt.get('assignment_known_subset_paths') != 23915
        or receipt.get('independent_eager_completion_certificates_and_true_tokens_verified') is not True
        or receipt.get('all_public_efficiency_aggregates_equal') is not True
        or receipt.get('qa_or_answers_read') is not False or receipt.get('quality_computed') is not False):
        raise ValueError('independent parent completion missing')
    for key,p in SOURCE_PATHS.items():
        if key != 'receipt' and receipt['input_output_sha256'].get(str(p.resolve())) != hashes[str(p.resolve())]:
            raise ValueError('direct source not covered by independent verification')
    parent_inventory()
    config = bound_read(SOURCE_PATHS['plan_plan.json'],hashes,deadline)
    seal = bound_read(SOURCE_PATHS['plan_seal.json'],hashes,deadline)
    if seal != {n:hashes[str((PARENT_PLAN/n).resolve())] for n in parent.PLAN_FILES-{'seal.json'}}:
        raise ValueError('parent seal differs')
    cases = bound_read(SOURCE_PATHS['plan_cores.json'],hashes,deadline)
    inventory = parent.inventory(cases)
    if config['core_object_sha256'] != object_hash(cases) or config['inventory'] != inventory:
        raise ValueError('parent Core inventory differs')
    summary = bound_read(SOURCE_PATHS['run_summary.json'],hashes,deadline)
    audit = bound_read(SOURCE_PATHS['audit'],hashes,deadline)
    completed = bound_read(SOURCE_PATHS['completed'],hashes,deadline)
    run_hashes = {str((PARENT_RUN/n).resolve()):hashes[str((PARENT_RUN/n).resolve())] for n in parent.RUN_FILES}
    if (summary.get('status') != 'completed' or summary.get('all_paths_available') is not True
        or summary.get('quality_computed') is not False or summary.get('qa_or_answers_read') is not False
        or summary['inventory'] != inventory or audit.get('status') != 'verified_complete'
        or audit.get('full_output_and_certificates_recomputed') is not True or audit.get('run_sha256') != run_hashes
        or summary['output_sha256'] != {n:run_hashes[str((PARENT_RUN/n).resolve())] for n in parent.RUN_FILES-{'summary.json'}}
        or summary['plan_sha256'] != {str((PARENT_PLAN/n).resolve()):hashes[str((PARENT_PLAN/n).resolve())] for n in parent.PLAN_FILES}
        or completed.get('status') != 'completed_and_audited' or completed.get('all_direct_bindings_unchanged') is not True
        or completed.get('summary_sha256') != run_hashes[str((PARENT_RUN/'summary.json').resolve())]
        or completed.get('audit_sha256') != AUDIT_SHA): raise ValueError('complete parent binding differs')
    # Imported helpers only; tokenizer files and unconsumed ancestors stay inherited.
    for p in [Path(parent.__file__).resolve(),ROOT/'docs/research/run_qasper_evidence_baselines.py',
              ROOT/'docs/research/qasper_metrics.py',ROOT/'docs/research/qasper_alignment_v2.py']:
        h = digest(p,deadline)
        if receipt['input_output_sha256'].get(str(p.resolve())) != h: raise ValueError('consumed helper changed')
        hashes[str(p.resolve())] = h
    for p in own_paths(): hashes[str(p.resolve())] = digest(p,deadline)
    metadata = {'parent_independent_receipt_sha256':RECEIPT_SHA,'parent_runtime':summary['runtime'],
                'inherited_parent_input_commitments':config['input_sha256'],
                'inherited_parent_input_commitments_sha256':object_hash(config['input_sha256']),
                'token_counts_freshly_remeasured':False}
    verify_hashes(hashes,deadline)
    parent_inventory()
    return cases,hashes,inventory,metadata


def prepare(args):
    output,run = Path(args.output).resolve(),Path(args.run_output).resolve()
    if output.exists() or run.exists(): raise FileExistsError('new exclusive plan/run required')
    parent.avoid_overlap([output,run],{p.parent for p in SOURCE_PATHS.values()})
    cases,hashes,inventory,metadata = source_data()
    config = {'schema':SCHEMA,'status':'prepared_not_executed','specification':SPEC,'limits':LIMITS,
              'created_at_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime(),'run_output':str(run),
              'input_sha256':hashes,'input_binding_sha256':object_hash(hashes),'inventory':inventory,
              'core_object_sha256':object_hash(cases),'parent_metadata':metadata,
              'selector_executed':False,'qa_or_answers_read':False,'api_calls':0,'quality_computed':False}
    verify_hashes(hashes); output.mkdir(parents=True,exist_ok=False)
    write(output/'cores.json',cases); write(output/'plan.json',config)
    write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
    return config


def load_plan(directory,deadline=math.inf):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != PLAN_FILES: raise ValueError('plan inventory differs')
    own = {str(directory/n):digest(directory/n,deadline) for n in PLAN_FILES}
    if bound_read(directory/'seal.json',own,deadline) != {n:own[str(directory/n)] for n in PLAN_FILES-{'seal.json'}}:
        raise ValueError('plan seal differs')
    config = bound_read(directory/'plan.json',own,deadline)
    if (config.get('schema') != SCHEMA or config.get('status') != 'prepared_not_executed'
        or config.get('specification') != SPEC or config.get('limits') != LIMITS or config.get('runtime') != runtime()
        or config.get('selector_executed') is not False or config.get('qa_or_answers_read') is not False
        or config.get('api_calls') != 0 or config.get('quality_computed') is not False
        or object_hash(config['input_sha256']) != config['input_binding_sha256']): raise ValueError('plan contract differs')
    verify_hashes(config['input_sha256'],deadline)
    cases,hashes,inventory,metadata = source_data(deadline)
    if (config['input_sha256'] != hashes or config['inventory'] != inventory or config['parent_metadata'] != metadata
        or config['core_object_sha256'] != object_hash(cases)
        or bound_read(directory/'cores.json',own,deadline) != cases): raise ValueError('plan projection changed')
    parent.avoid_overlap([directory,config['run_output']],{p.parent for p in SOURCE_PATHS.values()})
    verify_hashes(own,deadline)
    return config,cases,own


def bound_rows(path,hashes,deadline):
    check(deadline); blob = Path(path).read_bytes(); check(deadline)
    if hashlib.sha256(blob).hexdigest() != hashes[str(Path(path).resolve())]: raise ValueError('saved paths changed')
    return [parse(line) for line in blob.splitlines() if line.strip()]


def load_saved(cases,hashes,deadline):
    suites = {n:bound_rows(SOURCE_PATHS['run_'+n+'.jsonl'],hashes,deadline) for n in ('empty_cache','all_known_subsets')}
    questions = bound_rows(SOURCE_PATHS['run_per_question.jsonl'],hashes,deadline)
    by_key = {(x['doc_id'],x['question_id']):x for x in cases}; query_map = {}
    for q in questions:
        key = (q['doc_id'],q['question_id'])
        if key not in by_key or key in query_map or q['family_id'] != by_key[key]['family_id']: raise ValueError('parent query identity differs')
        m = len(parent.edges_of(parent.from_json(by_key[key]['core'])))
        essential = q['essential_variables_after_selection']
        if any(type(j) is not int or not 0 <= j < m for j in essential) or sorted(set(essential)) != essential:
            raise ValueError('invalid inherited essentiality')
        query_map[key] = q
    if set(query_map) != set(by_key): raise ValueError('incomplete parent questions')
    indexed = {}
    for suite,rows in suites.items():
        index = {}
        for row in rows:
            check(deadline); key = (row['doc_id'],row['question_id']); wrapper = by_key.get(key)
            if wrapper is None or row['family_id'] != wrapper['family_id']: raise ValueError('parent path identity differs')
            m = len(parent.edges_of(parent.from_json(wrapper['core']))); a,s = row['assignment'],row['known_subset']
            if (type(a) is not int or type(s) is not int or not 0 <= a < 1<<m or not 0 <= s < 1<<m
                or (suite == 'empty_cache' and s != 0) or row['eligible_count'] != m
                or row['initial_known_count'] != s.bit_count() or (*key,a,s) in index
                or row['output_equivalent'] is not True): raise ValueError('parent path coverage differs')
            index[(*key,a,s)] = row
        expected = {(*key,a,s) for key,w in by_key.items()
                    for a in range(1<<len(parent.edges_of(parent.from_json(w['core']))))
                    for s in ([0] if suite == 'empty_cache' else range(1<<len(parent.edges_of(parent.from_json(w['core'])))))}
        if set(index) != expected: raise ValueError('incomplete parent path denominator')
        indexed[suite] = index
    for key,row in indexed['empty_cache'].items():
        other = indexed['all_known_subsets'][key]
        if row['result'] != other['result']: raise ValueError('parent empty-cache overlap differs')
    return indexed,query_map


def token_lookup(rows):
    values = {}
    def add(indices,value):
        key = tuple(sorted(indices))
        if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices) or type(value) is not int or value < 0:
            raise ValueError('invalid saved token measurement')
        if key in values and values[key] != value: raise ValueError('inconsistent saved whole-pack tokens')
        values[key] = value
    for row in rows:
        result = row['result']; add(result['selected_indices'],result['actual_tokens'])
        for event in result['events']:
            if event['kind'] == 'winner' and event['proposed_tokens'] is not None:
                add([*event['selected_before'],event['incumbent']],event['proposed_tokens'])
    def count(indices):
        key = tuple(sorted(indices))
        if key not in values: raise ValueError('missing saved whole-pack token measurement')
        return values[key]
    return count,len(values)


def winners(result,lazy=False):
    return [{'candidate':e['incumbent' if lazy else 'candidate'],'selected_before':e['selected_before'],
             'proposed_tokens':e['proposed_tokens'],'action':e['action']}
            for e in result['events'] if e['kind'] == 'winner']


def histogram(values): return {str(k):v for k,v in sorted(Counter(values).items())}


def aggregate(rows):
    def total(field): return sum(r[field] for r in rows)
    result = {'paths':len(rows),'exact_pack_and_winner_trace_parity_paths':len(rows),
              'lazy_request_subset_paths':len(rows),'activation_at_most_two_paths':len(rows)}
    for field in ('initially_unknown_eligible','activation_reads','lazy_reads','eligible_minus_activation','activation_minus_lazy'):
        values = [r[field] for r in rows]
        result[field] = {'total':sum(values),'histogram':histogram(values),'maximum':max(values,default=0)}
    for method in ('activation','lazy'):
        values = [len(r[method+'_nonessential_variables']) for r in rows]
        result[method+'_nonessential_reads'] = {'total':sum(values),'histogram':histogram(values),
            'paths':sum(v>0 for v in values)}
        result[method+'_unique_source_qualified_edges'] = len({tuple(e) for r in rows for e in r[method+'_requested_edge_identities']})
    result['additional_certification_saving_paths'] = sum(r['activation_minus_lazy']>0 for r in rows)
    result['equal_activation_and_lazy_paths'] = sum(r['activation_minus_lazy']==0 for r in rows)
    result['full_cache_paths'] = sum(r['initially_unknown_eligible']==0 for r in rows)
    result['full_cache_zero_read_paths'] = sum(r['initially_unknown_eligible']==0 and r['activation_reads']==r['lazy_reads']==0 for r in rows)
    if total('eligible_minus_activation')+total('activation_minus_lazy') != total('initially_unknown_eligible')-total('lazy_reads'):
        raise ValueError('read reduction decomposition differs')
    return result


def evaluate(cases,saved,query_map,deadline=math.inf,fixed=True):
    inventory = parent.inventory(cases,fixed); outputs = {'empty_cache':[],'all_known_subsets':[]}; queries = []
    for wrapper in cases:
        check(deadline); key = (wrapper['doc_id'],wrapper['question_id']); core = parent.from_json(wrapper['core'])
        edges = parent.edges_of(core); m = len(edges)
        local_saved = [r for k,r in saved['all_known_subsets'].items() if k[:2] == key]
        count,entries = token_lookup(local_saved)
        local = {name:[] for name in outputs}
        # Essentiality is used only after the comparator returns, for negative diagnostics.
        essential = set(query_map[key]['essential_variables_after_selection'])
        for suite in outputs:
            for assignment in range(1<<m):
                for subset in ([0] if suite == 'empty_cache' else range(1<<m)):
                    check(deadline); old = saved[suite][(*key,assignment,subset)]
                    initial = {j:bool(assignment&(1<<j)) for j in range(m) if subset&(1<<j)}
                    actual = select(core,lambda j:bool(assignment&(1<<j)),count,initial,deadline)
                    lazy = old['result']; outcome = ('selected_indices','selected_ids','rendered_pack','pack_sha256','actual_tokens')
                    if any(actual[k] != lazy[k] for k in outcome) or winners(actual) != winners(lazy,True):
                        raise ValueError('activation winner trace or exact pack differs')
                    asked,prior = actual['requests'],lazy['requests']
                    if (len(asked) != len(set(asked)) or set(asked)&set(initial) or len(prior) != len(set(prior))
                        or set(prior)&set(initial) or not set(prior)<=set(asked)
                        or not len(prior)<=len(asked)<=min(m-len(initial),SPEC['maximum_new_reads'])):
                        raise ValueError('query inclusion or structural bound violated')
                    identity = {k:wrapper[k] for k in ('doc_id','question_id','family_id')}
                    def edge_ids(js): return [[wrapper['doc_id'],core.units[edges[j][0]].unit_id,core.units[edges[j][1]].unit_id] for j in js]
                    row = {**identity,'assignment':assignment,'known_subset':subset,'eligible_count':m,
                           'initially_unknown_eligible':m-len(initial),'activation_reads':len(asked),'lazy_reads':len(prior),
                           'eligible_minus_activation':m-len(initial)-len(asked),'activation_minus_lazy':len(asked)-len(prior),
                           'activation_requested_variables':asked,'lazy_requested_variables':prior,
                           'activation_requested_edge_identities':edge_ids(asked),'lazy_requested_edge_identities':edge_ids(prior),
                           'activation_nonessential_variables':[j for j in asked if j not in essential],
                           'lazy_nonessential_variables':[j for j in prior if j not in essential],
                           'winner_removal_trace':winners(actual),'activation_events':actual['events'],
                           'activation_final_known':actual['final_known'],
                           'outcome':{k:actual[k] for k in outcome if k != 'rendered_pack'},
                           'exact_rendered_pack_parity':True,'winner_trace_parity':True}
                    local[suite].append(row)
        queries.append({**{k:wrapper[k] for k in ('doc_id','question_id','family_id')},
                        'eligible_count':m,'replayed_token_cache_entries':entries,
                        **{name:aggregate(rows) for name,rows in local.items()}})
        for suite in outputs: outputs[suite].extend(local[suite])
    if any(len(outputs[n]) != inventory['empty_cache_paths' if n=='empty_cache' else 'assignment_known_subset_paths'] for n in outputs):
        raise ValueError('incomplete full-domain evaluation')
    public = {'schema':SCHEMA+'-public','status':'complete_exact_control','specification':SPEC,'limits':LIMITS,
              'inventory':inventory,**{n:aggregate(rows) for n,rows in outputs.items()},
              'replayed_token_cache_entries':sum(q['replayed_token_cache_entries'] for q in queries),
              'new_tokenizer_encodes':0,'api_calls':0,'quality_computed':False,'qa_or_answers_read':False,
              'gpu_used':False,'global_cache_schedule_executed':False,'private_ids_in_public_output':False}
    check(deadline)
    return outputs['empty_cache'],outputs['all_known_subsets'],queries,public


def summary_for(config,own,outputs,elapsed):
    return {'schema':SCHEMA,'status':'completed','all_paths_available':True,'specification':SPEC,
            'inventory':config['inventory'],'plan_sha256':own,'input_sha256':config['input_sha256'],
            'input_binding_sha256':config['input_binding_sha256'],'parent_metadata':config['parent_metadata'],
            'output_sha256':outputs,'elapsed_seconds':elapsed,'runtime':runtime(),
            'timing_scope':'saved-token replay including validation and writing; not serving latency',
            'api_calls':0,'quality_computed':False,'qa_or_answers_read':False,'new_tokenizer_encodes':0}


def run(args):
    start = time.monotonic(); deadline = start+SPEC['max_seconds']
    config,cases,own = load_plan(args.plan,deadline); output = Path(config['run_output'])
    output.mkdir(parents=True,exist_ok=False)
    try:
        saved,queries = load_saved(cases,config['input_sha256'],deadline)
        empty,allrows,perq,public = evaluate(cases,saved,queries,deadline)
        for name,rows in [('empty_cache.jsonl',empty),('all_known_subsets.jsonl',allrows),('per_question.jsonl',perq)]:
            rows_write(output/name,rows,deadline)
        write(output/'public_aggregate.json',public)
        verify_hashes({**config['input_sha256'],**own},deadline)
        outputs = {n:digest(output/n,deadline) for n in RUN_FILES-{'summary.json'}}
        write(output/'summary.json',summary_for(config,own,outputs,time.monotonic()-start))
        verify_hashes({**config['input_sha256'],**own},deadline); check(deadline)
        if {p.name for p in output.iterdir()} != RUN_FILES: raise ValueError('unexpected run inventory')
        return {'status':'completed','empty_cache_paths':len(empty),'assignment_known_subset_paths':len(allrows),
                'summary_sha256':digest(output/'summary.json',deadline),'api_calls':0}
    except BaseException as exc:
        if (output/'summary.json').exists(): (output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_result','error_type':type(exc).__name__,
                                    'elapsed_seconds':time.monotonic()-start,'api_calls':0})
        raise


def audit(args):
    deadline = time.monotonic()+SPEC['max_seconds']; config,cases,own = load_plan(args.plan,deadline)
    output = Path(args.run).resolve()
    if output != Path(config['run_output']).resolve() or {p.name for p in output.iterdir()} != RUN_FILES:
        raise ValueError('complete exact output inventory required')
    hashes = {str(output/n):digest(output/n,deadline) for n in RUN_FILES}
    summary = bound_read(output/'summary.json',hashes,deadline); elapsed = summary.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=SPEC['max_seconds']:
        raise ValueError('invalid execution timing')
    if summary != summary_for(config,own,{n:hashes[str(output/n)] for n in RUN_FILES-{'summary.json'}},elapsed):
        raise ValueError('summary binding differs')
    saved,questions = load_saved(cases,config['input_sha256'],deadline)
    empty,allrows,perq,public = evaluate(cases,saved,questions,deadline)
    for name,expected in [('empty_cache.jsonl',empty),('all_known_subsets.jsonl',allrows),('per_question.jsonl',perq)]:
        if bound_rows(output/name,hashes,deadline) != expected: raise ValueError('saved comparator paths differ')
    if bound_read(output/'public_aggregate.json',hashes,deadline) != public: raise ValueError('public aggregate differs')
    verify_hashes({**config['input_sha256'],**own,**hashes},deadline)
    if {p.name for p in output.iterdir()} != RUN_FILES: raise ValueError('run inventory changed during audit')
    return {'schema':SCHEMA+'-audit','status':'verified_complete','empty_cache_paths':len(empty),
            'assignment_known_subset_paths':len(allrows),'all_paths_and_decomposition_recomputed':True,
            'all_source_hashes_unchanged':True,'run_sha256':hashes,'input_binding_sha256':config['input_binding_sha256'],
            'api_calls':0,'quality_computed':False,'qa_or_answers_read':False,'new_tokenizer_encodes':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output',required=True); p.add_argument('--run-output',required=True)
    p = sub.add_parser('run'); p.add_argument('--plan',required=True)
    p = sub.add_parser('audit'); p.add_argument('--plan',required=True); p.add_argument('--run',required=True)
    args = parser.parse_args(); result = {'prepare':prepare,'run':run,'audit':audit}[args.command](args)
    print(json.dumps(result if args.command!='prepare' else {k:result[k] for k in ('status','inventory','api_calls')},indent=2))
