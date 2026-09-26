"""Exact-pack lazy rank bounds. Runtime has no cube, essentiality or quality input."""
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
from run_qasper_evidence_baselines import PackCounter, Unit, render_pack

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/research-foundation'
SCHEMA = 'slac-qasper-relation-lazy-bounds-v1'
RECEIPT_SHA = 'ebf959058e67a2ea3285f70c985f0a94bda2d6912cd6eab24d4de1a33037e32c'
SPEC = {'questions':77, 'families':24, 'empty_cache_paths':501,
    'assignment_known_subset_paths':23915, 'eligible_edge_occurrences':107,
    'eligible_edge_histogram':{'0':31,'1':19,'2':8,'3':10,'4':6,'5':1,'6':1,'7':1},
    'max_candidates':16, 'budget':1024, 'max_units':3, 'max_seconds':300,
    'base_half_scores':{'yes':2,'unknown':1}, 'bonus_half_score':2,
    'query_order':'first native-order unknown active edge of incumbent or conflicting candidate',
    'certificate':'incumbent worst key <= every other best key',
    'tie_order':'dense rank then source order', 'initial_cache_values':'strict bool',
    'runtime_uses_cube':False, 'runtime_uses_essentiality':False, 'quality_computed':False,
    'api_calls':0, 'global_cache_schedule_executed':False}
LIMITS = [
    'Exposed development cases; exact output under a fixed hypothetical oracle, not independent confirmation.',
    'Assignment histograms are combinatorial, not a distribution of real labels or expected requests.',
    'Logical reads are not provider calls, batching fees, end-to-end savings or relation/answer quality.',
    'Equivalent API judgments in different batches need not return the same labels.',
    'No optimal query policy, new classical algorithm or compact optimal decision diagram is claimed.',
    'Runtime has no cube, essential set, question, answer, reference or quality input.',
    'Validation is exhaustive and exponential; the runtime selector itself does not enumerate assignments.',
    'All-subset paths include the 501 empty-cache paths; the two suites are not independent samples.',
    'Token-count cache is shared across each question replay; timings are not cold-start serving comparisons.',
    'Relation observations reset for each path; no actual global label assignment or shared-cache schedule.',
    'Unobserved bonus/priority-change traces are not fabricated; certificates prove exact pack output.',
    'Only directly consumed sources are rehashed; unused ancestor commitments remain inherited.',
    'The CPU deadline is cooperative between bounded operations, not an operating-system hard kill.',
]
SOURCE_PATHS = {
    'receipt':ARTIFACTS/'qasper-relation-opportunity-independent-verification-01/verification.json',
    'gate_plan':ARTIFACTS/'qasper-relation-opportunity-plan-01/plan.json',
    'gate_cases':ARTIFACTS/'qasper-relation-opportunity-plan-01/cases.json',
    'gate_seal':ARTIFACTS/'qasper-relation-opportunity-plan-01/seal.json',
    'gate_summary':ARTIFACTS/'qasper-relation-opportunity-run-01/summary.json',
    'gate_masks':ARTIFACTS/'qasper-relation-opportunity-run-01/per_mask.jsonl',
}
TOKEN_FILES = ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model')
HELPERS = ('run_qasper_evidence_baselines.py','qasper_metrics.py','qasper_alignment_v2.py')
PLAN_FILES = {'plan.json','cores.json','seal.json'}
RUN_FILES = {'empty_cache.jsonl','all_known_subsets.jsonl','per_question.jsonl','public_aggregate.json','summary.json'}
CORE_KEYS = {'units','candidates','ranking','labels'}


def check(deadline):
    if time.monotonic() > deadline: raise TimeoutError('lazy bounds deadline exceeded')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode('utf-8')


def object_hash(value): return hashlib.sha256(canonical(value)).hexdigest()


def unique(pairs):
    result = {}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate JSON key')
        result[k] = v
    return result


def parse(blob): return json.loads(blob, object_pairs_hook=unique)


def digest(path, deadline=math.inf):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024): check(deadline); h.update(chunk)
    check(deadline)
    return h.hexdigest()


def bound_read(path, hashes, deadline=math.inf):
    check(deadline); blob = Path(path).read_bytes(); check(deadline)
    if hashlib.sha256(blob).hexdigest() != hashes[str(Path(path).resolve())]: raise ValueError('bound input changed')
    return parse(blob)


def verify_hashes(hashes, deadline=math.inf):
    for p,h in hashes.items():
        if digest(p,deadline) != h: raise ValueError('source changed: '+Path(p).name)


def write(path,value):
    with Path(path).open('xb') as f: f.write(canonical(value)+b'\n')


def rows_write(path,rows,deadline):
    with Path(path).open('xb') as f:
        for row in rows: check(deadline); f.write(canonical(row)+b'\n')


def runtime():
    return {'python':platform.python_version(),'transformers':version('transformers'),
            'tokenizers':version('tokenizers'),'tokenizer_parallelism':False}


@dataclass(frozen=True)
class Core:
    units: tuple[Unit,...]
    candidates: tuple[int,...]
    ranking: tuple[int,...]
    labels: tuple[str,...]


def validate_core(core):
    if type(core) is not Core or not core.units or any(type(u) is not Unit for u in core.units):
        raise ValueError('invalid runtime core')
    if (any(type(u.order) is not int or u.order != i or not isinstance(u.unit_id,str) or not u.unit_id
            or not isinstance(u.text,str) or not u.text.strip() or not isinstance(u.native_text,str)
            or not u.native_text.strip() for i,u in enumerate(core.units))
        or len({u.unit_id for u in core.units}) != len(core.units)):
        raise ValueError('invalid native units')
    if (not 0 < len(core.candidates) <= SPEC['max_candidates']
        or any(type(i) is not int or not 0 <= i < len(core.units) for i in core.candidates)
        or tuple(sorted(set(core.candidates))) != core.candidates
        or any(type(i) is not int for i in core.ranking) or len(core.ranking) != len(core.candidates)
        or set(core.ranking) != set(core.candidates) or len(core.labels) != len(core.candidates)
        or any(type(l) is not str or l not in ('yes','unknown','no') for l in core.labels)):
        raise ValueError('invalid candidate/label/rank contract')


def from_json(value):
    if set(value) != CORE_KEYS: raise ValueError('runtime whitelist differs')
    if any(set(u) != set(Unit.__dataclass_fields__) for u in value['units']): raise ValueError('unit whitelist differs')
    core = Core(tuple(Unit(**u) for u in value['units']), *(tuple(value[k]) for k in ('candidates','ranking','labels')))
    validate_core(core)
    return core


def edges_of(core):
    eligible = {i for i,l in zip(core.candidates,core.labels) if l != 'no'}
    return tuple((i,i+1) for i in core.candidates if i in eligible and i+1 in eligible
                 and core.units[i].native_text != core.units[i+1].native_text)


def validate_cache(known,m):
    if type(known) is not dict or any(type(k) is not int or not 0 <= k < m or type(v) is not bool for k,v in known.items()):
        raise ValueError('cache requires strict integer variables and bool values')
    return dict(known)


def safe_count(count,selected):
    value = count(sorted(selected))
    if type(value) is not int or value < 0: raise ValueError('invalid actual token count')
    return value


def bounds(core,pending,selected,known,edges,rank):
    active = {a:j for j,(a,b) in enumerate(edges) if b in selected}
    base = {i:SPEC['base_half_scores'][l] for i,l in zip(core.candidates,core.labels) if l != 'no'}
    result = []
    for i in sorted(pending):
        edge = active.get(i); lower = upper = base[i]
        if edge is not None:
            if edge not in known: upper += 2
            elif known[edge]: lower += 2; upper += 2
        result.append({'candidate':i,'worst_key':[-lower,rank[i],core.units[i].order],
                       'best_key':[-upper,rank[i],core.units[i].order],'active_edge':edge})
    return result


def select(core, ask, count, known=None, deadline=math.inf):
    """Runtime policy. Neither complete assignments nor targets are arguments."""
    validate_core(core)
    if not callable(ask): raise ValueError('callback must be callable')
    edges = edges_of(core); cache = validate_cache({} if known is None else known,len(edges))
    rank = {i:r for r,i in enumerate(core.ranking)}
    pending = {i for i,l in zip(core.candidates,core.labels) if l != 'no'}
    selected = []; text = set(); events = []; requests = []
    while pending and len(selected) < SPEC['max_units']:
        check(deadline)
        table = bounds(core,pending,selected,cache,edges,rank)
        e = min(table,key=lambda r:r['worst_key'])
        conflicts = [r for r in table if r['candidate'] != e['candidate'] and r['best_key'] < e['worst_key']]
        row = {'selected_before':sorted(selected),'bounds':table,'incumbent':e['candidate'],
               'conflicts':[r['candidate'] for r in conflicts]}
        if conflicts:
            choices = {r['active_edge'] for r in [e,*conflicts] if r['active_edge'] is not None and r['active_edge'] not in cache}
            if not choices: raise AssertionError('uncertified winner without an unknown active edge')
            variable = min(choices); check(deadline); value = ask(variable); check(deadline)
            if type(value) is not bool: raise ValueError('callback must return strict bool')
            cache[variable] = value; requests.append(variable)
            events.append({**row,'kind':'request','variable':variable,'value':value})
            continue
        chosen = e['candidate']; pending.remove(chosen)
        event = {**row,'kind':'winner','proposed_tokens':None,'action':'skip_exact_native_duplicate'}
        if core.units[chosen].native_text not in text:
            event['proposed_tokens'] = safe_count(count,[*selected,chosen])
            if event['proposed_tokens'] <= SPEC['budget']:
                selected.append(chosen); text.add(core.units[chosen].native_text); event['action'] = 'select'
            else: event['action'] = 'skip_evidence_budget'
        events.append(event)
    selected.sort(); tokens = safe_count(count,selected); rendered = render_pack(core.units,selected)
    check(deadline)
    return {'selected_indices':selected,'selected_ids':[core.units[i].unit_id for i in selected],
            'rendered_pack':rendered,'pack_sha256':hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
            'actual_tokens':tokens,'requests':requests,'final_known':[[k,cache[k]] for k in sorted(cache)],'events':events}


def validate_path(core,assignment,initial,result,count,deadline=math.inf):
    """Validation only: independently enumerate consistent scores, never call bounds/select."""
    validate_core(core); edges = edges_of(core); m = len(edges)
    if type(assignment) is not int or not 0 <= assignment < 2**m: raise ValueError('invalid assignment')
    known = validate_cache(initial,m)
    if any(v != bool(assignment & (1<<j)) for j,v in known.items()): raise ValueError('initial cache contradicts oracle')
    pending = [i for i,l in zip(core.candidates,core.labels) if l != 'no']
    scores = {i:2 if l == 'yes' else 1 for i,l in zip(core.candidates,core.labels) if l != 'no'}
    ranks = {i:r for r,i in enumerate(core.ranking)}; selected = []; requests = []; completion_checks = 0
    for event in result['events']:
        check(deadline)
        if not pending or len(selected) == SPEC['max_units']: raise ValueError('event after termination')
        consistent = [mask for mask in range(1<<m) if all(bool(mask & (1<<j)) == v for j,v in known.items())]
        keysets = {i:[] for i in pending}
        for mask in consistent:
            for i in pending:
                added = any(a == i and b in selected and mask & (1<<j) for j,(a,b) in enumerate(edges))
                keysets[i].append((-(scores[i]+2*int(added)),ranks[i],core.units[i].order))
        completion_checks += len(consistent)
        table = [{'candidate':i,'worst_key':list(max(keysets[i])),'best_key':list(min(keysets[i])),
                  'active_edge':next((j for j,(a,b) in enumerate(edges) if a == i and b in selected),None)} for i in sorted(pending)]
        e = min(pending,key=lambda i:max(keysets[i]))
        conflict = [i for i in sorted(pending) if i != e and min(keysets[i]) < max(keysets[e])]
        expected = {'selected_before':sorted(selected),'bounds':table,'incumbent':e,'conflicts':conflict}
        if conflict:
            allowed = [j for j,(a,b) in enumerate(edges) if a in [e,*conflict] and b in selected and j not in known]
            if not allowed: raise ValueError('invalid conflict without an unknown edge')
            j = allowed[0]; value = bool(assignment & (1<<j))
            expected.update(kind='request',variable=j,value=value)
            if type(event.get('value')) is not bool or canonical(event) != canonical(expected): raise ValueError('request certificate/order/value differs')
            known[j] = value; requests.append(j)
        else:
            if any(min(pending,key=lambda i:keysets[i][k]) != e for k in range(len(consistent))):
                raise ValueError('winner differs under a consistent completion')
            pending.remove(e); expected.update(kind='winner',proposed_tokens=None,action='skip_exact_native_duplicate')
            if core.units[e].native_text not in {core.units[i].native_text for i in selected}:
                n = safe_count(count,[*selected,e]); expected['proposed_tokens'] = n
                if n <= SPEC['budget']: selected.append(e); expected['action'] = 'select'
                else: expected['action'] = 'skip_evidence_budget'
            if canonical(event) != canonical(expected): raise ValueError('winner/action certificate differs')
    if pending and len(selected) < SPEC['max_units']: raise ValueError('path truncated')
    selected.sort(); rendered = render_pack(core.units,selected)
    expected = {'selected_indices':selected,'selected_ids':[core.units[i].unit_id for i in selected],
        'rendered_pack':rendered,'pack_sha256':hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
        'actual_tokens':safe_count(count,selected),'requests':requests,'final_known':[[j,known[j]] for j in sorted(known)],
        'events':result['events']}
    if canonical(result) != canonical(expected) or len(requests) != len(set(requests)) or set(requests)&set(initial):
        raise ValueError('final output/cache/requests differ')
    return completion_checks


def inventory(cases, fixed=True):
    keys = [(x['doc_id'],x['question_id']) for x in cases]
    if len(set(keys)) != len(cases): raise ValueError('duplicate question')
    histogram = Counter(len(edges_of(from_json(x['core']))) for x in cases)
    result = {'questions':len(cases),'families':len({x['family_id'] for x in cases}),
        'empty_cache_paths':sum(v*2**m for m,v in histogram.items()),
        'assignment_known_subset_paths':sum(v*4**m for m,v in histogram.items()),
        'eligible_edge_occurrences':sum(m*v for m,v in histogram.items()),
        'eligible_edge_histogram':{str(m):v for m,v in sorted(histogram.items())}}
    if any(set(x) != {'family_id','doc_id','question_id','core'} or any(type(x[k]) is not str or not x[k] for k in ('family_id','doc_id','question_id')) for x in cases):
        raise ValueError('case wrapper whitelist differs')
    if fixed and any(result[k] != SPEC[k] for k in result): raise ValueError('complete fixed inventory differs')
    return result


def own_paths():
    return [Path(__file__).resolve(),ROOT/'tests/research/test_qasper_relation_lazy_bounds.py',
            ROOT/'docs/research/RELATION_LAZY_BOUNDS_PROTOCOL_20260927.md']


def source_data(deadline=math.inf):
    """Metadata and raw-input projection only: no selector, mask parsing or tokenizer."""
    receipt_blob = SOURCE_PATHS['receipt'].read_bytes()
    if hashlib.sha256(receipt_blob).hexdigest() != RECEIPT_SHA: raise ValueError('complete parent receipt differs')
    receipt = parse(receipt_blob)
    if (receipt.get('status') != 'verified_complete' or receipt.get('questions') != 77 or receipt.get('masks') != 501
        or receipt.get('every_selected_identity_pack_token_and_step_recomputed') is not True
        or receipt.get('qa_or_answers_read') is not False): raise ValueError('parent complete verification missing')
    hashes = {str(p.resolve()):digest(p,deadline) for p in SOURCE_PATHS.values()}
    for k,p in SOURCE_PATHS.items():
        if k != 'receipt' and receipt['input_output_sha256'].get(str(p.resolve())) != hashes[str(p.resolve())]:
            raise ValueError('parent source does not match independent receipt')
    plan = bound_read(SOURCE_PATHS['gate_plan'],hashes,deadline)
    seal = bound_read(SOURCE_PATHS['gate_seal'],hashes,deadline)
    if seal != {p.name:hashes[str(p.resolve())] for p in (SOURCE_PATHS['gate_plan'],SOURCE_PATHS['gate_cases'])}:
        raise ValueError('parent plan seal differs')
    summary = bound_read(SOURCE_PATHS['gate_summary'],hashes,deadline)
    if (summary.get('status') != 'completed' or summary.get('all_masks_available') is not True
        or summary.get('quality_metrics_computed') is not False
        or summary['output_sha256'].get('per_mask.jsonl') != hashes[str(SOURCE_PATHS['gate_masks'].resolve())]):
        raise ValueError('parent complete mask binding differs')
    raw = bound_read(SOURCE_PATHS['gate_cases'],hashes,deadline)
    if object_hash(raw) != plan['cases_object_sha256']: raise ValueError('parent cases object differs')
    cases = [{**{k:x[k] for k in ('family_id','doc_id','question_id')},'core':{k:x[k] for k in CORE_KEYS}} for x in raw]
    inv = inventory(cases)
    tokenizer = Path(plan['tokenizer']).resolve()
    for p in [*(tokenizer/n for n in TOKEN_FILES),*(ROOT/'docs/research'/n for n in HELPERS)]:
        h = digest(p,deadline)
        if receipt['input_output_sha256'].get(str(p.resolve())) != h: raise ValueError('helper/tokenizer binding differs')
        hashes[str(p.resolve())] = h
    for p in own_paths(): hashes[str(p.resolve())] = digest(p,deadline)
    verify_hashes(hashes,deadline)
    return cases,str(tokenizer),hashes,inv


def avoid_overlap(paths,protected):
    allpaths = [Path(p).resolve() for p in paths]
    for i,p in enumerate(allpaths):
        if any(p == q or p.is_relative_to(q) or q.is_relative_to(p) for q in [*allpaths[:i],*(Path(x).resolve() for x in protected)]):
            raise ValueError('new output overlaps a source or another output')


def prepare(args):
    output,run = Path(args.output).resolve(),Path(args.run_output).resolve()
    if output.exists() or run.exists(): raise FileExistsError('new exclusive plan/run required')
    avoid_overlap([output,run],{p.parent for p in SOURCE_PATHS.values()})
    cases,tokenizer,hashes,inv = source_data()
    config = {'schema':SCHEMA,'status':'prepared_not_executed','specification':SPEC,'limits':LIMITS,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'runtime':runtime(),'run_output':str(run),
        'tokenizer':tokenizer,'input_sha256':hashes,'input_binding_sha256':object_hash(hashes),
        'core_object_sha256':object_hash(cases),'inventory':inv,'selector_executed':False,
        'api_calls':0,'quality_computed':False,'qa_or_answers_read':False}
    verify_hashes(hashes)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'cores.json',cases); write(output/'plan.json',config)
    write(output/'seal.json',{n:digest(output/n) for n in PLAN_FILES-{'seal.json'}})
    return config


def load_plan(directory,deadline=math.inf):
    directory = Path(directory).resolve()
    if {p.name for p in directory.iterdir()} != PLAN_FILES: raise ValueError('plan inventory differs')
    own = {str(directory/n):digest(directory/n,deadline) for n in PLAN_FILES}
    seal = bound_read(directory/'seal.json',own,deadline)
    if seal != {n:own[str(directory/n)] for n in PLAN_FILES-{'seal.json'}}: raise ValueError('plan seal differs')
    config = bound_read(directory/'plan.json',own,deadline)
    if (config.get('schema') != SCHEMA or config.get('status') != 'prepared_not_executed'
        or config.get('specification') != SPEC or config.get('limits') != LIMITS or config.get('runtime') != runtime()
        or config.get('selector_executed') is not False or config.get('api_calls') != 0
        or config.get('quality_computed') is not False or config.get('qa_or_answers_read') is not False
        or object_hash(config['input_sha256']) != config['input_binding_sha256']): raise ValueError('plan contract differs')
    verify_hashes(config['input_sha256'],deadline)
    cases,tokenizer,hashes,inv = source_data(deadline)
    if (config['tokenizer'] != tokenizer or config['input_sha256'] != hashes or config['inventory'] != inv
        or config['core_object_sha256'] != object_hash(cases)
        or bound_read(directory/'cores.json',own,deadline) != cases): raise ValueError('plan input projection changed')
    avoid_overlap([directory,config['run_output']],{p.parent for p in SOURCE_PATHS.values()})
    verify_hashes(own,deadline)
    return config,cases,own


class MeasuredCounter:
    def __init__(self,tokenizer,units,deadline):
        self.counter = PackCounter(tokenizer,units,deadline); self.calls = 0
    def __call__(self,selected): self.calls += 1; return self.counter(selected)


def load_targets(cases,hashes,deadline):
    p = SOURCE_PATHS['gate_masks']; check(deadline); blob = p.read_bytes(); check(deadline)
    if hashlib.sha256(blob).hexdigest() != hashes[str(p.resolve())]: raise ValueError('gate target changed')
    rows = [parse(line) for line in blob.splitlines() if line.strip()]; targets = {}
    by_key = {(x['doc_id'],x['question_id']):x for x in cases}
    for row in rows:
        identity = (row['doc_id'],row['question_id']); wrapper = by_key[identity]; core = from_json(wrapper['core'])
        edges = edges_of(core); mask = row['mask']; key = (*identity,mask)
        if (type(mask) is not int or not 0 <= mask < 2**len(edges) or key in targets
            or row['family_id'] != wrapper['family_id']
            or row['active_edges'] != [list(e) for j,e in enumerate(edges) if mask & (1<<j)]):
            raise ValueError('gate target identity/assignment differs')
        indices = row['selected_indices']
        if (any(type(i) is not int for i in indices) or sorted(set(indices)) != indices
            or not set(indices) <= set(core.candidates) or len(indices) > 3): raise ValueError('illegal target indices')
        rendered = render_pack(core.units,indices)
        value = {k:row[k] for k in ('selected_indices','selected_ids','pack_sha256','actual_tokens')}
        value['rendered_pack'] = rendered
        if (value['selected_ids'] != [core.units[i].unit_id for i in indices]
            or value['pack_sha256'] != hashlib.sha256(rendered.encode('utf-8')).hexdigest()
            or type(value['actual_tokens']) is not int or not 0 <= value['actual_tokens'] <= 1024): raise ValueError('target pack differs')
        targets[key] = value
    expected = {(*key,m) for key,x in by_key.items() for m in range(2**len(edges_of(from_json(x['core']))))}
    if set(targets) != expected: raise ValueError('incomplete gate targets')
    return targets


def histogram(values): return {str(k):v for k,v in sorted(Counter(values).items())}


def suite_aggregate(rows):
    requested = [e for row in rows for e in row['requested_edge_identities']]
    return {'paths':len(rows),'output_equivalent_paths':sum(r['output_equivalent'] for r in rows),
        'request_count_histogram':histogram(len(r['result']['requests']) for r in rows),
        'maximum_reads':max((len(r['result']['requests']) for r in rows),default=0),
        'total_reads':sum(len(r['result']['requests']) for r in rows),
        'all_initially_unknown_eligible_reads':sum(r['eligible_count']-r['initial_known_count'] for r in rows),
        'saved_reads':sum(r['eligible_count']-r['initial_known_count']-len(r['result']['requests']) for r in rows),
        'fewer_reads_paths':sum(len(r['result']['requests']) < r['eligible_count']-r['initial_known_count'] for r in rows),
        'equal_reads_paths':sum(len(r['result']['requests']) == r['eligible_count']-r['initial_known_count'] for r in rows),
        'nonessential_reads':sum(len(r['nonessential_requested_variables']) for r in rows),
        'paths_reading_nonessential':sum(bool(r['nonessential_requested_variables']) for r in rows),
        'unique_source_qualified_edges_requested':len({tuple(e) for e in requested}),
        'unique_query_edge_occurrences_requested':len({(r['doc_id'],r['question_id'],j) for r in rows for j in r['result']['requests']}),
        'winner_certificates':sum(e['kind']=='winner' for r in rows for e in r['result']['events']),
        'consistent_completion_event_checks':sum(r['consistent_completion_event_checks'] for r in rows),
        'selector_counter_calls':sum(r['selector_counter_calls'] for r in rows),
        'selector_tokenizer_encodes':sum(r['selector_tokenizer_encodes'] for r in rows),
        'validator_counter_calls':sum(r['validator_counter_calls'] for r in rows),
        'validator_tokenizer_encodes':sum(r['validator_tokenizer_encodes'] for r in rows)}


def evaluate(cases,tokenizer,targets,deadline=math.inf,fixed=True):
    inv = inventory(cases,fixed); empty_rows = []; all_rows = []; query_rows = []
    for wrapper in cases:
        check(deadline); core = from_json(wrapper['core']); edges = edges_of(core); m = len(edges)
        identity = {k:wrapper[k] for k in ('family_id','doc_id','question_id')}; key = (wrapper['doc_id'],wrapper['question_id'])
        count = MeasuredCounter(tokenizer,core.units,deadline); local_empty = []; local_all = []
        # Essentiality is intentionally derived only after all runtime paths for this query.
        for suite,subsets,dest in [('empty',[0],local_empty),('all',range(1<<m),local_all)]:
            for assignment in range(1<<m):
                for subset in subsets:
                    check(deadline)
                    initial = {j:bool(assignment & (1<<j)) for j in range(m) if subset & (1<<j)}
                    calls0,enc0 = count.calls,len(count.counter.cache)
                    result = select(core,lambda j:bool(assignment & (1<<j)),count,initial,deadline)
                    calls1,enc1 = count.calls,len(count.counter.cache)
                    checks = validate_path(core,assignment,initial,result,count,deadline)
                    expected = targets[(*key,assignment)]
                    if {k:result[k] for k in expected} != expected: raise ValueError('lazy output differs from complete saved gate')
                    if subset == (1<<m)-1 and result['requests']: raise ValueError('full cache requested a label')
                    dest.append({**identity,'assignment':assignment,'known_subset':subset,'eligible_count':m,
                        'initial_known_count':len(initial),'result':result,'output_equivalent':True,
                        'selected_identities':[[wrapper['doc_id'],uid] for uid in result['selected_ids']],
                        'requested_edge_identities':[[wrapper['doc_id'],core.units[edges[j][0]].unit_id,core.units[edges[j][1]].unit_id] for j in result['requests']],
                        'consistent_completion_event_checks':checks,'selector_counter_calls':calls1-calls0,
                        'selector_tokenizer_encodes':enc1-enc0,'validator_counter_calls':count.calls-calls1,
                        'validator_tokenizer_encodes':len(count.counter.cache)-enc1})
        essential = {j for j in range(m) if any(targets[(*key,mask)] != targets[(*key,mask^(1<<j))] for mask in range(1<<m))}
        for row in [*local_empty,*local_all]:
            row['nonessential_requested_variables'] = [j for j in row['result']['requests'] if j not in essential]
        n = len(core.candidates); cache_bound = 1+sum(math.comb(n,k) for k in range(1,min(n,3)+1))
        if len(count.counter.cache) > cache_bound: raise ValueError('whole-pack cache bound exceeded')
        query_rows.append({**identity,'eligible_count':m,'essential_variables_after_selection':sorted(essential),
            'essential_edge_identities':[[wrapper['doc_id'],core.units[edges[j][0]].unit_id,core.units[edges[j][1]].unit_id] for j in sorted(essential)],
            'empty_cache':suite_aggregate(local_empty),'all_known_subsets':suite_aggregate(local_all),
            'token_cache_entries':len(count.counter.cache),'token_cache_bound':cache_bound,
            'total_tokenizer_encodes':len(count.counter.cache)-1,'total_counter_calls':count.calls})
        empty_rows.extend(local_empty); all_rows.extend(local_all)
    if len(empty_rows) != inv['empty_cache_paths'] or len(all_rows) != inv['assignment_known_subset_paths']:
        raise ValueError('incomplete replay denominator')
    essential_occ = sum(len(q['essential_variables_after_selection']) for q in query_rows)
    public = {'schema':SCHEMA+'-public','status':'complete_exact_output_validation','specification':SPEC,'limits':LIMITS,
        'inventory':inv,'empty_cache':suite_aggregate(empty_rows),'all_known_subsets':suite_aggregate(all_rows),
        'essential_edge_occurrences_posthoc':essential_occ,
        'unique_essential_edges_posthoc':len({tuple(e) for q in query_rows for e in q['essential_edge_identities']}),
        'prior_19_essential_occurrences_reproduced':essential_occ == 19 if fixed else None,
        'total_token_cache_entries':sum(q['token_cache_entries'] for q in query_rows),
        'total_token_cache_bound':sum(q['token_cache_bound'] for q in query_rows),
        'maximum_question_token_cache_entries':max(q['token_cache_entries'] for q in query_rows),
        'total_tokenizer_encodes':sum(q['total_tokenizer_encodes'] for q in query_rows),
        'total_counter_calls':sum(q['total_counter_calls'] for q in query_rows),
        'api_calls':0,'gpu_used':False,'encoder_model_loaded':False,'qa_or_answers_read':False,
        'quality_computed':False,'global_cache_schedule_executed':False,'private_ids_in_public_output':False}
    check(deadline)
    return empty_rows,all_rows,query_rows,public


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)


def summary_for(config,own,outputs,timing):
    return {'schema':SCHEMA,'status':'completed','all_paths_available':True,'specification':SPEC,
        'inventory':config['inventory'],'plan_sha256':own,'input_sha256':config['input_sha256'],
        'input_binding_sha256':config['input_binding_sha256'],'runtime':config['runtime'],
        'output_sha256':outputs,'timing':timing,'api_calls':0,'qa_or_answers_read':False,'quality_computed':False,
        'encoder_model_loaded':False,'gpu_used':False}


def public_metadata(public,config,timing):
    return {**public,'runtime':config['runtime'],'timing':timing,'input_binding_sha256':config['input_binding_sha256']}


def run(args,tokenizer_factory=load_tokenizer):
    start = time.monotonic(); deadline = start+SPEC['max_seconds']
    config,cases,own = load_plan(args.plan,deadline); output = Path(config['run_output'])
    output.mkdir(parents=True,exist_ok=False); phase = 'loading'
    try:
        tokenizer = tokenizer_factory(config['tokenizer']); targets = load_targets(cases,config['input_sha256'],deadline)
        loaded = time.monotonic(); phase = 'all_path_validation'
        empty,allrows,queries,public = evaluate(cases,tokenizer,targets,deadline)
        evaluated = time.monotonic(); phase = 'writing'
        rows_write(output/'empty_cache.jsonl',empty,deadline); rows_write(output/'all_known_subsets.jsonl',allrows,deadline)
        rows_write(output/'per_question.jsonl',queries,deadline)
        verify_hashes({**config['input_sha256'],**own},deadline)
        timing = {'loading_seconds':loaded-start,'validation_seconds':evaluated-loaded,
                  'writing_and_verification_seconds':time.monotonic()-evaluated}
        timing['total_seconds_before_summary'] = sum(timing.values())
        write(output/'public_aggregate.json',public_metadata(public,config,timing))
        outputs = {n:digest(output/n,deadline) for n in RUN_FILES-{'summary.json'}}
        write(output/'summary.json',summary_for(config,own,outputs,timing))
        verify_hashes({**config['input_sha256'],**own},deadline); check(deadline)
        return {'status':'completed','empty_cache_paths':len(empty),'assignment_known_subset_paths':len(allrows),
                'summary_sha256':digest(output/'summary.json',deadline),'api_calls':0}
    except BaseException as exc:
        if (output/'summary.json').exists(): (output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'failed_no_complete_result','phase':phase,
            'error_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-start,'api_calls':0,'quality_computed':False})
        raise


def valid_timing(timing):
    names = {'loading_seconds','validation_seconds','writing_and_verification_seconds','total_seconds_before_summary'}
    if (set(timing) != names or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in timing.values())
        or timing['total_seconds_before_summary'] > SPEC['max_seconds']
        or abs(sum(v for k,v in timing.items() if k != 'total_seconds_before_summary')-timing['total_seconds_before_summary']) > 1e-8):
        raise ValueError('invalid execution timing')


def audit(args,tokenizer_factory=load_tokenizer):
    deadline = time.monotonic()+SPEC['max_seconds']; config,cases,own = load_plan(args.plan,deadline)
    output = Path(args.run).resolve()
    if output != Path(config['run_output']).resolve() or {p.name for p in output.iterdir()} != RUN_FILES:
        raise ValueError('complete exact run inventory required')
    run_hashes = {str(output/n):digest(output/n,deadline) for n in RUN_FILES}
    summary = bound_read(output/'summary.json',run_hashes,deadline); timing = summary['timing']; valid_timing(timing)
    if summary != summary_for(config,own,{n:run_hashes[str(output/n)] for n in RUN_FILES-{'summary.json'}},timing):
        raise ValueError('summary binding/metadata differs')
    tokenizer = tokenizer_factory(config['tokenizer']); targets = load_targets(cases,config['input_sha256'],deadline)
    empty,allrows,queries,public = evaluate(cases,tokenizer,targets,deadline)
    for name,expected in [('empty_cache.jsonl',empty),('all_known_subsets.jsonl',allrows),('per_question.jsonl',queries)]:
        blob = (output/name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != run_hashes[str(output/name)]: raise ValueError('run source changed')
        if canonical([parse(line) for line in blob.splitlines() if line.strip()]) != canonical(expected):
            raise ValueError('saved full path/certificate differs')
    if bound_read(output/'public_aggregate.json',run_hashes,deadline) != public_metadata(public,config,timing):
        raise ValueError('aggregate differs from complete replay')
    verify_hashes({**config['input_sha256'],**own,**run_hashes},deadline)
    return {'schema':SCHEMA+'-audit','status':'verified_complete','empty_cache_paths':len(empty),
        'assignment_known_subset_paths':len(allrows),'full_output_and_certificates_recomputed':True,
        'source_hashes_unchanged':True,'measured_timing_independently_reproduced':False,
        'run_sha256':run_hashes,'input_binding_sha256':config['input_binding_sha256'],
        'api_calls':0,'quality_computed':False,'qa_or_answers_read':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output',required=True); p.add_argument('--run-output',required=True)
    p = sub.add_parser('run'); p.add_argument('--plan',required=True)
    p = sub.add_parser('audit'); p.add_argument('--plan',required=True); p.add_argument('--run',required=True)
    args = parser.parse_args(); result = {'prepare':prepare,'run':run,'audit':audit}[args.command](args)
    print(json.dumps(result if args.command != 'prepare' else {k:result[k] for k in ('status','inventory','api_calls')},indent=2))
