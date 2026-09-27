"""New-budget, support-only execution. No references, quality, or generation."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import plan_qasper_confirmation_budget as budget
import openrouter_decision_client as client
import run_qasper_relation_pilot as pilot
import run_qasper_extended_development as stage

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/research-foundation'
SCHEMA = 'slac-qasper-confirmation-support-v1'
PLAN_FILES = {'plan.json', 'jobs.json', 'batches.json', 'seal.json'}
SEGMENT_CAPS = {'requests': 160, 'judgments': 1600, 'input_allowance': 8000000, 'reserved_usd': '2'}
MODEL_LOCKS = {'jev': 'typesafe/jev-1.13-20260917', 'general': 'qwen/qwen3.6-plus'}
SPEC = {'scope': 'complete confirmation support only', 'automatic_retries': 0, 'fallbacks': 0,
        'generation_performed': False, 'references_read': False, 'quality_metrics_computed': False,
        'dispatch_deadline_margin_seconds': 65, 'per_request_hard_seconds': 65,
        'segment_caps': SEGMENT_CAPS, 'old_night_budget_or_deadline_inherited': False,
        'unknown_reservations_refunded': False, 'resolved_model_locks': MODEL_LOCKS}


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    require(Path(path).name not in {'references.jsonl', 'native_qa_sidecar.jsonl', 'native_qa_sidecar_v2.jsonl'}, 'references forbidden')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=client.unique_object)


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(budget.canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())


def amount(value):
    require(isinstance(value, (str, Decimal)), 'budget must be an explicit decimal string')
    result = Decimal(value)
    require(result.is_finite() and result >= 0, 'invalid money amount')
    return result


def deadline_time(value):
    require(isinstance(value, str), 'absolute timezone-aware deadline required')
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'deadline must include timezone')
    return parsed.astimezone(timezone.utc)


def utc_now(): return datetime.now(timezone.utc)


def ensure_time(deadline, *, dispatch=False):
    require((deadline_time(deadline) - utc_now()).total_seconds() > (65 if dispatch else 0), 'absolute deadline exhausted')


def totals(jobs):
    return {'requests': len(jobs), 'judgments': sum(len(j['task_ids']) for j in jobs),
            'input_allowance': sum(j['input_allowance'] for j in jobs),
            'output_allowance': sum(j['output_allowance'] for j in jobs),
            'reserved_usd': str(sum((amount(j['reserved_usd']) for j in jobs), Decimal(0)))}


def segments(jobs):
    result, start = [], 0
    while start < len(jobs):
        stop = start
        while stop < len(jobs):
            t = totals(jobs[start:stop+1])
            if any(t[k] > SEGMENT_CAPS[k] for k in ('requests', 'judgments', 'input_allowance')) or amount(t['reserved_usd']) > Decimal('2'): break
            stop += 1
        require(stop > start, 'one job exceeds client segment capacity')
        result.append({'segment': len(result)+1, 'start': start, 'stop': stop, **totals(jobs[start:stop])})
        start = stop
    return result


def load_budget(directory):
    directory = Path(directory).resolve()
    require({p.name for p in directory.iterdir()} == budget.OUTPUT_FILES, 'budget plan incomplete')
    bindings = {str(directory/n): budget.digest(directory/n) for n in budget.OUTPUT_FILES}
    seal = read(directory/'seal.json')
    require(seal == {n: bindings[str(directory/n)] for n in budget.OUTPUT_FILES-{'seal.json'}}, 'budget seal differs')
    private, public, sources = (read(directory/n) for n in ('private_plan.json', 'public_aggregate.json', 'source_binding.json'))
    require(private['schema'] == public['schema'] == budget.SCHEMA and private['public_aggregate'] == public
            and private['specification'] == budget.SPEC and public['paid_execution_admitted'] is False, 'budget contract differs')
    require(private['source_sha256'] == sources['input_sha256'], 'budget sources differ')
    for path, sha in private['source_sha256'].items(): budget.bind_hash(bindings, path, sha)
    require(public['provider_snapshot_sha256'] == budget.SNAPSHOT_SHA256
            and public['implementation_sha256'] == budget.digest(budget.__file__), 'budget implementation/snapshot changed')
    budget.verify(bindings)
    jobs, batches = private['jobs'], private['batches']
    expected = []
    with budget.qwen_profile():
        require(private['models'] == client.MODELS and private['prompt_version'] == client.PROMPT_VERSION, 'budget route/profile differs')
        for bi, batch in enumerate(batches):
            for backend in (('jev', 'general') if bi % 2 == 0 else ('general', 'jev')):
                payload = client.make_payload(batch['tasks'], 'support', backend)
                reserve, inputs, outputs = client.reservation(payload, backend)
                expected.append({'ordinal': len(expected)+1, 'batch_id': batch['id'], 'backend': backend,
                    'task_ids': [t['id'] for t in batch['tasks']], 'payload': payload,
                    'payload_sha256': client.object_hash(payload), 'canonical_payload_bytes': len(client.canonical_bytes(payload)),
                    'input_allowance': inputs, 'output_allowance': outputs, 'reserved_usd': str(reserve)})
    require(jobs == expected and len({b['id'] for b in batches}) == len(batches), 'budget jobs differ from frozen common batches')
    require(len(jobs) == 2*public['B_common_support_batches'] and totals(jobs)['reserved_usd'] == public['support_total_reserved_usd'], 'support reservation differs')
    generation = public['generation_upper_bound']
    require(generation['request_count'] == 5*public['Q_questions'] and amount(generation['reserved_usd']) == Decimal('.014196')*generation['request_count'], 'full answer commitment differs')
    budget.verify(bindings)
    return private, public, bindings


def source_paths():
    return [Path(__file__), ROOT/'tests/research/test_qasper_confirmation_support.py',
        ROOT/'docs/research/CONFIRMATION_SUPPORT_EXECUTION_PROTOCOL_20260927.md',
        Path(budget.__file__), Path(client.__file__), Path(pilot.__file__), Path(stage.__file__)]


def prepare(budget_plan, output, run_output, total_budget_usd, deadline):
    cap = amount(total_budget_usd); require(cap > 0, 'positive explicit new budget required')
    ensure_time(deadline, dispatch=True)
    output, run_output, parent = map(lambda p: Path(p).resolve(), (output, run_output, budget_plan))
    for path in (output, run_output):
        require(path.is_relative_to(ART.resolve()) and path != ART.resolve() and not path.exists(), 'new ignored directory required')
    require(not (output.is_relative_to(run_output) or run_output.is_relative_to(output)), 'plan/run overlap')
    private, public, bindings = load_budget(parent)
    for path in (parent, *map(Path, bindings)):
        for destination in (output, run_output):
            require(not (destination == path or destination.is_relative_to(path) or path.is_relative_to(destination)), 'output overlaps source')
    support, generation = amount(public['support_total_reserved_usd']), amount(public['generation_upper_bound']['reserved_usd'])
    require(support + generation <= cap, 'whole support plus full answer commitment exceeds new budget')
    for path in source_paths(): budget.bind_hash(bindings, path, budget.digest(path))
    registration = parent.with_name(parent.name+'-support-registration.json')
    require(not registration.exists(), 'this budget family already registered; no restart')
    jobs = private['jobs']
    config = {'schema': SCHEMA, 'status': 'prepared_support_execution', 'specification': SPEC,
        'budget_plan': str(parent), 'run_output': str(run_output), 'registration': str(registration),
        'total_budget_usd': str(cap), 'support_committed_reserve_usd': str(support),
        'generation_locked_reserve_usd': str(generation), 'unallocated_reserve_usd': str(cap-support-generation),
        'deadline_utc': deadline_time(deadline).isoformat(), 'physical_attempt_cap': len(jobs),
        'Q': public['Q_questions'], 'C': public['C_support_query_unit_pairs'], 'B': public['B_common_support_batches'],
        'segments': segments(jobs), 'input_sha256': bindings, 'api_calls_in_preparation': 0,
        'key_read_in_preparation': False, 'quality_metrics_computed': False}
    budget.verify(bindings); output.mkdir(parents=True, exist_ok=False)
    for name, value in [('plan.json', config), ('jobs.json', jobs), ('batches.json', private['batches'])]: write(output/name, value)
    write(output/'seal.json', {n: budget.digest(output/n) for n in PLAN_FILES-{'seal.json'}})
    return {k: config[k] for k in ('status', 'Q', 'C', 'B', 'total_budget_usd', 'support_committed_reserve_usd',
                                   'generation_locked_reserve_usd', 'unallocated_reserve_usd', 'deadline_utc', 'physical_attempt_cap')}


def load_plan(directory):
    directory = Path(directory).resolve()
    require({p.name for p in directory.iterdir()} == PLAN_FILES, 'execution plan incomplete')
    raw = {n: (directory/n).read_bytes() for n in PLAN_FILES}
    own = {str(directory/n): hashlib.sha256(b).hexdigest() for n, b in raw.items()}
    values = {n: json.loads(b, object_pairs_hook=client.unique_object) for n, b in raw.items()}
    require(values['seal.json'] == {n: own[str(directory/n)] for n in PLAN_FILES-{'seal.json'}}, 'execution plan seal differs')
    config, jobs, batches = values['plan.json'], values['jobs.json'], values['batches.json']
    require(config['schema'] == SCHEMA and config['status'] == 'prepared_support_execution' and config['specification'] == SPEC, 'execution contract differs')
    budget.verify(config['input_sha256']); budget.verify(own)
    private, public, parent = load_budget(config['budget_plan'])
    require(all(config['input_sha256'].get(p) == h for p,h in parent.items()), 'parent budget source differs')
    require(jobs == private['jobs'] and batches == private['batches'] and config['segments'] == segments(jobs), 'frozen schedule differs')
    require(config['physical_attempt_cap'] == len(jobs) and (config['Q'],config['C'],config['B']) ==
            (public['Q_questions'],public['C_support_query_unit_pairs'],public['B_common_support_batches']), 'complete denominator differs')
    support, generation, cap = amount(config['support_committed_reserve_usd']), amount(config['generation_locked_reserve_usd']), amount(config['total_budget_usd'])
    require(support == amount(public['support_total_reserved_usd']) and generation == amount(public['generation_upper_bound']['reserved_usd'])
            and support+generation <= cap and amount(config['unallocated_reserve_usd']) == cap-support-generation, 'global budget lock differs')
    require(Path(config['registration']) == Path(config['budget_plan']).with_name(Path(config['budget_plan']).name+'-support-registration.json'), 'single-use family registration differs')
    deadline_time(config['deadline_utc'])
    require(Path(config['run_output']).is_relative_to(ART.resolve()) and not Path(config['run_output']).is_relative_to(directory), 'invalid fixed run path')
    for path in source_paths(): require(config['input_sha256'].get(str(path.resolve())) == budget.digest(path), 'execution source changed')
    budget.verify(config['input_sha256']); budget.verify(own)
    return config, jobs, batches, own


class Watchdog:
    """Only exits this executor process; never kills another process."""
    def __init__(self, output, seconds):
        self.cancel = threading.Event()
        def expire():
            if not self.cancel.wait(max(0, seconds)):
                try: write(Path(output)/'hard_timeout.json', {'schema': SCHEMA, 'status': 'failed_hard_timeout'})
                finally: os._exit(124)
        self.thread = threading.Thread(target=expire, daemon=True); self.thread.start()
    def close(self): self.cancel.set(); self.thread.join(timeout=1)


class SupportClient(client.BoundedClient):
    def __init__(self, *args, jobs, deadline, run_output, guard_factory=Watchdog, **kwargs):
        self.jobs, self.deadline, self.run_output, self.guard_factory = jobs, deadline, run_output, guard_factory
        self._blocked = False
        super().__init__(*args, **kwargs)

    def save(self):
        if not hasattr(self, '_sequence'):
            self._sequence, self._previous, self._artifacts = 0, None, {}
            self.ledger['resolved_models'] = dict(MODEL_LOCKS)
            self.ledger['resolved_model_locks'] = dict(MODEL_LOCKS)
        attempts = self.ledger['attempts']; failure = None
        if attempts and attempts[-1]['status'] == 'completed':
            record = attempts[-1]; job = self.jobs[len(attempts)-1]
            try:
                response = read(self.output/f"response_{len(attempts):03d}.json")
                pilot.validate_reused_response(response, job['payload'], record)
                if job['backend'] == 'jev': stage.support_scores(response, job['task_ids'])
                require(record['response_model'] == MODEL_LOCKS[job['backend']], 'dated model changed')
                require(math.isfinite(record['elapsed_seconds']) and 0 <= record['elapsed_seconds'] <= 65, 'request elapsed exceeds bound')
                ensure_time(self.deadline)
            except BaseException as error:
                record['status'] = 'halted'; record['error_class'] = type(error).__name__
                self.ledger['halt_reason'] = type(error).__name__; failure = error
        events = self.output.parent/(self.output.name+'_events'); events.mkdir(exist_ok=True)
        additions = {}
        for prefix in ('request','response','error_response'):
            name = f'{prefix}_{len(attempts):03d}.json'; path = self.output/name
            if path.exists() and name not in self._artifacts: additions[name] = budget.digest(path)
        event = {'schema': SCHEMA+'-event', 'sequence': self._sequence, 'previous_sha256': self._previous,
                 'new_artifacts': additions, 'ledger': self.redacted(self.ledger)}
        path = events/f'event_{self._sequence:05d}.json'; write(path, event)
        self._previous = budget.digest(path); self._sequence += 1; self._artifacts.update(additions)
        super().save()
        with (self.output/'ledger.json').open('r+b') as stream: os.fsync(stream.fileno())
        if failure is not None: self._blocked = True; raise RuntimeError('support response contract failed') from None
        if attempts and attempts[-1]['status'] == 'in_flight' and 'elapsed_seconds' not in attempts[-1]:
            ensure_time(self.deadline, dispatch=True)

    def submit(self, tasks, kind, backend):
        ensure_time(self.deadline, dispatch=True)
        attempts = self.ledger['attempts']
        require(not self._blocked and (not attempts or attempts[-1]['status'] == 'completed'), 'failed/unknown client cannot continue')
        require(len(attempts) < len(self.jobs), 'frozen segment exhausted')
        job = self.jobs[len(attempts)]
        require(kind == 'support' and backend == job['backend'] and client.make_payload(tasks,kind,backend) == job['payload'], 'not the next frozen request')
        guard = self.guard_factory(self.run_output, min(65, (deadline_time(self.deadline)-utc_now()).total_seconds()))
        try: return super().submit(tasks, kind, backend)
        except BaseException: self._blocked = True; raise
        finally: guard.close()


def event_ledger(directory):
    directory = Path(directory); event_dir = directory.parent/(directory.name+'_events')
    files = sorted(event_dir.iterdir()); previous, last, artifacts = None, None, {}
    require(files and [p.name for p in files] == [f'event_{i:05d}.json' for i in range(len(files))], 'event sequence incomplete')
    bindings = {}
    for i,path in enumerate(files):
        event = read(path); ledger = event['ledger']; attempts = ledger['attempts']
        require(set(event) == {'schema','sequence','previous_sha256','new_artifacts','ledger'} and event['schema'] == SCHEMA+'-event'
                and event['sequence'] == i and event['previous_sha256'] == previous
                and ledger['resolved_models'] == ledger['resolved_model_locks'] == MODEL_LOCKS, 'event identity differs')
        if last is None: require(not attempts, 'initial ledger not empty')
        else:
            old = last['attempts']; mutable = {'attempts','reservation_total_usd','actual_reported_cost_usd','halt_reason'}
            require({k:v for k,v in ledger.items() if k not in mutable} == {k:v for k,v in last.items() if k not in mutable}, 'segment contract changed')
            if len(attempts) == len(old)+1:
                require(attempts[:-1] == old and (not old or old[-1]['status'] == 'completed') and attempts[-1]['status'] == 'in_flight', 'invalid reserve event')
            else:
                require(len(attempts) == len(old) and old and old[-1]['status'] == 'in_flight'
                        and attempts[:-1] == old[:-1] and attempts[-1]['status'] in {'completed','halted','in_flight'}
                        and all(attempts[-1].get(k) == v for k,v in old[-1].items() if k != 'status'), 'invalid terminal event')
        require(amount(ledger['reservation_total_usd']) == sum((amount(a['reserved_usd']) for a in attempts), Decimal(0))
                and amount(ledger['actual_reported_cost_usd']) == sum((amount(a['actual_cost_usd']) for a in attempts if a.get('actual_cost_usd') is not None), Decimal(0)), 'ledger money differs')
        for name, sha in event['new_artifacts'].items():
            require(name in {f'{prefix}_{len(attempts):03d}.json' for prefix in ('request','response','error_response')} and name not in artifacts, 'artifact not new current attempt')
            artifacts[name] = sha
        previous = budget.digest(path); bindings[str(path.resolve())] = previous; last = ledger
    require(read(directory/'ledger.json') == last and {p.name for p in directory.iterdir()} == set(artifacts)|{'ledger.json'}, 'final event/ledger inventory differs')
    bindings.update({str((directory/n).resolve()): h for n,h in artifacts.items()})
    bindings[str((directory/'ledger.json').resolve())] = budget.digest(directory/'ledger.json')
    budget.verify(bindings)
    return last, bindings


def collect(config, jobs, *, complete):
    output = Path(config['run_output']); base = output/'provider_calls'
    labels = {'jev': {}, 'general': {}}; scores = {}; bindings = {}; stopped = False; successful = 0
    account = {'attempts': 0, 'completed_requests': 0, 'unknown_cost_attempts': 0, 'reserved_usd': Decimal(0), 'known_cost_usd': Decimal(0)}
    seen_dirs = set()
    for segment in config['segments']:
        directory = base/f"segment_{segment['segment']:03d}"
        if not directory.exists(): stopped = True; continue
        require(not stopped, 'segment appeared after failed/missing prefix')
        seen_dirs.update({directory.name, directory.name+'_events'})
        ledger, bound = event_ledger(directory); bindings.update(bound)
        require(len(ledger['attempts']) <= segment['requests'], 'physical segment attempts exceeded')
        expected_caps = {'budget_usd': segment['reserved_usd'], 'request_cap': segment['requests'],
            'question_cap': segment['judgments'], 'conservative_input_token_cap': segment['input_allowance'],
            'prompt_version': client.PROMPT_VERSION, 'automatic_retries': 0}
        require(all(ledger[k] == v for k,v in expected_caps.items()), 'segment caps differ')
        for i, record in enumerate(ledger['attempts']):
            job = jobs[segment['start']+i]
            payload = pilot.prior_request(directory/f'request_{i+1:03d}.json', record, client.PROMPT_VERSION)
            require(record['attempt'] == i+1 and payload == job['payload'] and record['backend'] == job['backend']
                    and record['kind'] == 'support' and record['task_ids'] == job['task_ids'], 'physical attempt differs from frozen order')
            require((deadline_time(config['deadline_utc'])-deadline_time(record['started_at'])).total_seconds() > 65, 'attempt started too late')
            account['attempts'] += 1; account['reserved_usd'] += amount(record['reserved_usd'])
            if record.get('actual_cost_usd') is None: account['unknown_cost_attempts'] += 1
            else: account['known_cost_usd'] += amount(record['actual_cost_usd'])
            if record['status'] == 'completed':
                response = read(directory/f'response_{i+1:03d}.json')
                parsed = pilot.validate_reused_response(response, payload, record)
                require(record['response_model'] == MODEL_LOCKS[job['backend']] and not set(parsed)&set(labels[job['backend']]), 'model or duplicate judgment differs')
                elapsed = record.get('elapsed_seconds')
                require(type(elapsed) in (float,int) and math.isfinite(elapsed) and 0 <= elapsed <= 65
                        and deadline_time(record['started_at']).timestamp()+elapsed <= deadline_time(config['deadline_utc']).timestamp(), 'completed elapsed/deadline differs')
                labels[job['backend']].update(parsed)
                if job['backend'] == 'jev': scores.update(stage.support_scores(response,job['task_ids']))
                successful += 1
            else:
                require(record['status'] in {'halted','in_flight'} and i == len(ledger['attempts'])-1, 'failure not terminal')
                stopped = True
                known = None
                for prefix in ('response','error_response'):
                    path = directory/f'{prefix}_{i+1:03d}.json'
                    if path.exists():
                        value = read(path)
                        try: known = amount(str(value['usage']['cost']))
                        except (KeyError,TypeError,ValueError,ArithmeticError): pass
                require((known is not None) == (record.get('actual_cost_usd') is not None)
                        and (known is None or known == amount(record['actual_cost_usd'])), 'failed attempt known/unknown cost differs')
        if ledger['halt_reason'] is not None or len(ledger['attempts']) != segment['requests']: stopped = True
    require(not base.exists() or {p.name for p in base.iterdir()} == seen_dirs, 'unexpected provider artifacts')
    is_complete = successful == len(jobs) and not stopped
    require(account['attempts'] <= config['physical_attempt_cap'] and account['reserved_usd'] <= amount(config['support_committed_reserve_usd']), 'global support cap exceeded')
    if complete:
        require(is_complete and len(labels['jev']) == len(labels['general']) == len(scores) == config['C'], 'full support coverage required')
    account['completed_requests'] = successful
    account['reserved_usd'] = str(account['reserved_usd']); account['known_cost_usd'] = str(account['known_cost_usd'])
    return {'complete': is_complete, 'accounting': account, 'labels': labels, 'reported_scores': scores, 'bindings': bindings}


def execution_summary(config, result):
    attempted = amount(result['accounting']['reserved_usd'])
    return {'schema': SCHEMA, 'status': 'completed' if result['complete'] else 'stopped_incomplete',
        'Q': config['Q'], 'C': config['C'], 'B': config['B'], 'accounting': result['accounting'],
        'total_budget_usd': config['total_budget_usd'], 'support_committed_reserve_usd': config['support_committed_reserve_usd'],
        'generation_locked_reserve_usd': config['generation_locked_reserve_usd'], 'unallocated_reserve_usd': config['unallocated_reserve_usd'],
        'unattempted_support_commitment_usd': str(amount(config['support_committed_reserve_usd'])-attempted),
        'unattempted_commitment_automatically_released': False, 'specification': SPEC,
        'labels_available_for_complete_cohort': result['complete'], 'quality_metrics_computed': False}


def run(directory, key_file, *, proxy=None, transport=None, guard_factory=Watchdog):
    config,jobs,batches,own = load_plan(directory); output = Path(config['run_output'])
    ensure_time(config['deadline_utc'],dispatch=True)
    require(not output.exists(), 'fixed run directory already exists')
    write(config['registration'], {'schema': SCHEMA, 'plan_sha256': own, 'run_output': str(output),
        'registered_at_utc': utc_now().isoformat(), 'support_commitment_usd': config['support_committed_reserve_usd'],
        'generation_locked_reserve_usd': config['generation_locked_reserve_usd'], 'total_budget_usd': config['total_budget_usd']})
    registration_binding = {str(Path(config['registration']).resolve()): budget.digest(config['registration'])}
    output.mkdir(parents=True,exist_ok=False)
    guard = guard_factory(output,(deadline_time(config['deadline_utc'])-utc_now()).total_seconds())
    by_batch = {b['id']: b for b in batches}
    try:
        with budget.qwen_profile():
            for segment in config['segments']:
                ensure_time(config['deadline_utc'],dispatch=True)
                current = jobs[segment['start']:segment['stop']]
                bounded = SupportClient(output/'provider_calls'/f"segment_{segment['segment']:03d}", jobs=current,
                    deadline=config['deadline_utc'],run_output=output,guard_factory=guard_factory,
                    key_file=key_file,proxy=proxy,transport=transport,budget_usd=segment['reserved_usd'],
                    request_cap=segment['requests'],question_cap=segment['judgments'],token_cap=segment['input_allowance'])
                for job in current:
                    bounded.submit(by_batch[job['batch_id']]['tasks'],'support',job['backend'])
            budget.verify(config['input_sha256']);budget.verify(own);ensure_time(config['deadline_utc'])
            require(not (output/'hard_timeout.json').exists(), 'hard deadline failure exists')
            result = collect(config,jobs,complete=True)
            write(output/'judgments.json',{'labels':result['labels'],'reported_scores':result['reported_scores']})
            public = execution_summary(config,result);write(output/'public_aggregate.json',public)
            budget.verify(config['input_sha256']);budget.verify(own);budget.verify(result['bindings'])
            write(output/'summary.json',{**public,'plan_sha256':own,'completed_at_utc':utc_now().isoformat(),
                'output_sha256':{str(p.relative_to(output)):budget.digest(p) for p in output.rglob('*') if p.is_file()}})
            budget.verify(config['input_sha256']);budget.verify(own);budget.verify(result['bindings']);budget.verify(registration_binding)
            ensure_time(config['deadline_utc'])
            require(not (output/'hard_timeout.json').exists(), 'hard deadline failure exists')
            return public
    except BaseException as error:
        if (output/'summary.json').exists(): (output/'summary.json').unlink()
        write(output/'failure.json',{'schema':SCHEMA,'status':'stopped_incomplete','error_type':type(error).__name__,
              'automatic_retries':0,'no_partial_quality':True,'reservations_preserved':True})
        raise
    finally: guard.close()


def audit(directory, output, *, allow_incomplete=False):
    config,jobs,_,own=load_plan(directory);run_dir=Path(config['run_output']);output=Path(output).resolve()
    require(output.is_relative_to(ART.resolve()) and not output.exists() and not output.is_relative_to(run_dir), 'new private audit directory required')
    registration_path=Path(config['registration']).resolve()
    registration_raw=registration_path.read_bytes()
    registration_binding={str(registration_path):hashlib.sha256(registration_raw).hexdigest()}
    registration=json.loads(registration_raw,object_pairs_hook=client.unique_object)
    require(set(registration)=={'schema','plan_sha256','run_output','registered_at_utc','support_commitment_usd',
        'generation_locked_reserve_usd','total_budget_usd'} and registration['schema']==SCHEMA
        and registration['plan_sha256']==own and registration['run_output']==str(run_dir)
        and registration['support_commitment_usd']==config['support_committed_reserve_usd']
        and registration['generation_locked_reserve_usd']==config['generation_locked_reserve_usd']
        and registration['total_budget_usd']==config['total_budget_usd'], 'registration differs')
    require((deadline_time(config['deadline_utc'])-deadline_time(registration['registered_at_utc'])).total_seconds()>65,
            'registration deadline differs')
    initial={str(p.resolve()):budget.digest(p) for p in run_dir.rglob('*') if p.is_file()}
    with budget.qwen_profile(): result=collect(config,jobs,complete=not allow_incomplete)
    complete=result['complete'] and not any((run_dir/n).exists() for n in ('failure.json','hard_timeout.json'))
    if not allow_incomplete: require(complete,'failed/incomplete run cannot complete audit')
    public=execution_summary(config,{**result,'complete':complete})
    if complete:
        summary=read(run_dir/'summary.json')
        require(read(run_dir/'public_aggregate.json')==public and all(summary[k]==v for k,v in public.items())
                and summary['plan_sha256']==own,'summary metadata differs')
        expected={str(Path(p).relative_to(run_dir)):h for p,h in initial.items() if Path(p).name!='summary.json'}
        require(summary['output_sha256']==expected,'complete output seals differ')
        require(read(run_dir/'judgments.json')=={'labels':result['labels'],'reported_scores':result['reported_scores']},'saved judgments differ')
    budget.verify(initial);budget.verify(config['input_sha256']);budget.verify(own);budget.verify(registration_binding)
    require({str(p.resolve()) for p in run_dir.rglob('*') if p.is_file()}==set(initial),'run inventory changed')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'verification.json',{'schema':SCHEMA,'status':'verified_complete' if complete else 'verified_incomplete_prefix',
          'summary':public,'input_sha256':{**config['input_sha256'],**own,**initial,**registration_binding},'api_calls':0,'key_read':False,'references_read':False})
    return public


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--budget-plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--run-output',required=True);p.add_argument('--total-budget-usd',required=True);p.add_argument('--deadline',required=True)
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--key-file',required=True);p.add_argument('--proxy')
    p=sub.add_parser('audit');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    try:
        result=prepare(args.budget_plan,args.output,args.run_output,args.total_budget_usd,args.deadline) if args.command=='prepare' else \
            run(args.plan,args.key_file,proxy=args.proxy) if args.command=='run' else audit(args.plan,args.output,allow_incomplete=args.allow_incomplete)
    except BaseException as error:
        print(json.dumps({'status':'stopped','error_type':type(error).__name__,'automatic_retries':0}));raise SystemExit(1) from None
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
