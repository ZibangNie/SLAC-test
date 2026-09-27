"""Budget tests use synthetic candidates and the public frozen price snapshot."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'docs/research'))
spec = importlib.util.spec_from_file_location('confirmation_budget_tests', REPO / 'docs/research/plan_qasper_confirmation_budget.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic budget planning cannot use API/key/model/old runtime')
    monkeypatch.setattr(m.client, 'read_key', forbidden)
    monkeypatch.setattr(m.client, 'BoundedClient', forbidden)
    monkeypatch.setattr(m.client.urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(m.stage, 'schedule_requests', forbidden)
    monkeypatch.setattr(m.stage, 'plan', forbidden)
    monkeypatch.setattr(m.stage.pilot, 'selected_gold', forbidden)


@pytest.fixture
def snapshot():
    # Public provider metadata only; this contains no experimental QA.
    return json.loads(m.SNAPSHOT.read_bytes())


def prepared(counts=(3, 9), text='Synthetic source unit.', question='SYNTHETIC PRIVATE QUERY?'):
    documents, queries, tasks = {}, [], []
    for qi, count in enumerate(counts):
        doc = f'PRIVATE_DOC_{qi}'
        units = [{'unit_id': f'u{i}', 'order': i, 'kind': 'paragraph', 'start': i * 100,
                  'end': i * 100 + len(text), 'text': text, 'native_text': text} for i in range(count)]
        documents[doc] = units
        query = {'family_id': f'PRIVATE_COMPONENT_{qi // 2}', 'doc_id': doc, 'question_id': f'q{qi}',
                 'source_id': f'PRIVATE_SOURCE_{qi}', 'original_family_id': f'PRIVATE_FAMILY_{qi}',
                 'query': question, 'seed_ids': [u['unit_id'] for u in units[:8]],
                 'candidate_ids': [u['unit_id'] for u in units], 'ranked_ids': [u['unit_id'] for u in units]}
        queries.append(query)
        for unit in units:
            item = {'query': question, 'unit': {'id': unit['unit_id'], 'text': unit['text']}}
            identifier = 'support:' + m.client.object_hash({'kind': 'support', 'doc_id': doc,
                                                            'question_id': query['question_id'], 'item': item})
            tasks.append({'id': identifier, 'doc_id': doc, 'question_id': query['question_id'],
                          'unit_id': unit['unit_id'], 'item': item})
    return {'schema': m.CANDIDATE_SCHEMA, 'documents': documents, 'queries': queries,
            'support_tasks': tasks, 'static_tasks': []}


def test_complete_common_batches_and_single_jev_reuse(snapshot):
    private, public = m.build_budget(prepared(), snapshot)
    assert (public['Q_questions'], public['C_support_query_unit_pairs'], public['B_common_support_batches']) == (2, 12, 3)
    assert len(private['jobs']) == 6
    assert [len(b['tasks']) for b in private['batches']] == [3, 8, 1]
    assert len({b['group'][0] for b in private['batches']}) == 2
    by_batch = {}
    for job in private['jobs']:
        by_batch.setdefault(job['batch_id'], []).append(job)
    for jobs in by_batch.values():
        assert {j['backend'] for j in jobs} == {'jev', 'general'}
        assert jobs[0]['task_ids'] == jobs[1]['task_ids']
    assert all(v['requests'] == 3 and v['judgments'] == 12 for v in public['support'].values())
    assert public['generation_upper_bound']['request_count'] == 10
    assert 'one JEV response' in public['specification']['support_reuse']


def test_reservations_independent_decimal_hand_formula_and_floor(snapshot):
    private, public = m.build_budget(prepared((1, 8)), snapshot)
    sums = {'jev': Decimal(0), 'general': Decimal(0)}
    for job in private['jobs']:
        size = len(json.dumps(job['payload'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        count = len(job['task_ids']) if job['backend'] == 'jev' else 1
        allowance = (size + 2048) * count
        prompt, completion = (Decimal('.042'), Decimal(0)) if job['backend'] == 'jev' else (Decimal('.325'), Decimal('1.95'))
        cost = max(Decimal('.005'), (allowance * prompt + 1024 * completion) / 1000000 * Decimal('1.5'))
        assert job['canonical_payload_bytes'] == size
        assert job['input_allowance'] == allowance and job['output_allowance'] == 1024
        assert Decimal(job['reserved_usd']) == cost
        sums[job['backend']] += cost
    assert Decimal(public['support_total_reserved_usd']) == sum(sums.values())
    assert all(Decimal(public['support'][k]['reserved_usd']) == v for k, v in sums.items())


def test_qwen_json_explicit_and_global_profile_restored(snapshot):
    m.client.select_general_profile('gpt41mini')
    models, responses = deepcopy(m.client.MODELS), deepcopy(m.client.RESPONSE_MODELS)
    private, _ = m.build_budget(prepared((1,)), snapshot)
    qwen = next(j['payload'] for j in private['jobs'] if j['backend'] == 'general')
    assert qwen['model'] == 'qwen/qwen3.6-plus'
    assert qwen['provider']['only'] == ['alibaba'] and qwen['provider']['allow_fallbacks'] is False
    assert qwen['response_format'] == {'type': 'json_object'}
    assert qwen['reasoning'] == {'enabled': False}
    assert m.client.MODELS == models and m.client.RESPONSE_MODELS == responses


def test_profile_restored_when_planning_fails(snapshot):
    before = deepcopy(m.client.MODELS)
    bad = deepcopy(snapshot)
    bad['models'][1]['endpoint_metadata']['tag'] = 'different-provider'
    with pytest.raises(ValueError):
        m.build_budget(prepared((1,)), bad)
    assert m.client.MODELS == before


def test_unicode_byte_splitting_is_common_and_does_not_truncate(snapshot):
    data = prepared((8,), text='漢' * 1500)
    private, public = m.build_budget(data, snapshot)
    assert public['B_common_support_batches'] > 1
    assert sum(len(b['tasks']) for b in private['batches']) == 8
    assert all(j['canonical_payload_bytes'] <= 24000 for j in private['jobs'])
    assert all(t['item']['unit']['text'] == '漢' * 1500 for b in private['batches'] for t in b['tasks'])


def test_oversized_single_item_rejected_without_truncation(snapshot):
    with pytest.raises(ValueError, match='byte cap'):
        m.build_budget(prepared((1,), text='漢' * 9000), snapshot)


def test_five_arm_answer_bound_is_conditional_not_exact_generation(snapshot):
    _, public = m.build_budget(prepared((1, 1, 1)), snapshot)
    bound = public['generation_upper_bound']
    assert Decimal(bound['per_request_reserved_usd']) == Decimal('.014196')
    assert bound['request_count'] == 15
    assert Decimal(bound['reserved_usd']) == Decimal('.212940')
    assert all(bound[k] is False for k in ('actual_payloads_constructed', 'unique_request_count_known',
                                          'all_actual_payloads_fit_byte_cap_verified', 'deduplication_discount_applied'))
    assert Decimal(public['support_plus_conditional_generation_upper_usd']) == (
        Decimal(public['support_total_reserved_usd']) + Decimal(bound['reserved_usd']))
    assert public['specification']['new_stage_budget_cap_usd'] is None
    assert public['specification']['deadline'] is None
    assert public['specification']['old_night_deadline_or_budget_inherited'] is False
    assert public['paid_execution_admitted'] is False


def test_public_contains_no_synthetic_identity_or_payload(snapshot):
    _, public = m.build_budget(prepared(), snapshot)
    encoded = json.dumps(public)
    assert 'PRIVATE_' not in encoded and 'SYNTHETIC' not in encoded
    assert 'payload"' not in encoded
    assert public['references_read'] is False and public['key_read'] is False and public['api_calls'] == 0


@pytest.mark.parametrize('mutation', [
    lambda p: p['support_tasks'].pop(),
    lambda p: p['support_tasks'].append(deepcopy(p['support_tasks'][0])),
    lambda p: p['support_tasks'][0]['item'].__setitem__('query', 'changed'),
    lambda p: p['support_tasks'][0]['item']['unit'].__setitem__('text', 'changed'),
    lambda p: p['queries'][0]['candidate_ids'].append('unknown'),
    lambda p: p['queries'][0].__setitem__('ranked_ids', []),
    lambda p: p['queries'].append(deepcopy(p['queries'][0])),
    lambda p: p['static_tasks'].append({'anything': True}),
    lambda p: p['queries'][0].__setitem__('references', ['forbidden']),
    lambda p: p['documents']['PRIVATE_DOC_0'][0].__setitem__('answer', 'forbidden'),
    lambda p: p['documents']['PRIVATE_DOC_0'][0].__setitem__('order', True),
])
def test_missing_duplicate_changed_or_gold_containing_projection_rejected(snapshot, mutation):
    data = prepared()
    mutation(data)
    with pytest.raises(ValueError):
        m.build_budget(data, snapshot)


@pytest.mark.parametrize('mutation', [
    lambda s: s['models'][0].__setitem__('request_model_id', 'changed'),
    lambda s: s['models'][1]['endpoint_metadata'].__setitem__('provider_name', 'other'),
    lambda s: s['models'][1]['endpoint_metadata']['pricing'].__setitem__('prompt', '0.5'),
    lambda s: s['models'][0]['base_prices_usd_per_million'].__setitem__('prompt', '0.4'),
    lambda s: s['models'][0]['endpoint_metadata'].__setitem__('name', 'TypeSafe | other-revision'),
    lambda s: s['reservation_contract'].__setitem__('maximum_items_per_support_batch', 9),
    lambda s: s['reservation_contract'].__setitem__('upper_reservation_per_unique_generation_usd', '.1'),
])
def test_snapshot_model_provider_price_or_reservation_drift_rejected(snapshot, mutation):
    mutation(snapshot)
    with pytest.raises(ValueError):
        m.build_budget(prepared((1,)), snapshot)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value) + b'\n')


class CandidateFixture:
    def __init__(self, tmp_path):
        self.run, self.plan = tmp_path / 'candidate-run', tmp_path / 'candidate-plan'
        self.run.mkdir(); self.plan.mkdir()
        self.prepared = prepared((1,) * 248)
        counts = {'documents': 248, 'questions': 248}
        self.source = tmp_path / 'synthetic-source.py'
        self.source.write_text('# synthetic source', encoding='utf-8')
        inherited = {str(self.source): m.digest(self.source)}
        self.config = {'schema': m.CANDIDATE_SCHEMA, 'status': 'prepared_no_model_inference',
            'run_output': str(self.run), 'gold_read': False, 'api_calls': 0, 'input_sha256': inherited,
            'counts': counts, 'config': {'dense_seeds': 8, 'candidate_cap': 16, 'max_selected_units': 3,
                                      'evidence_budget': 1024, 'truncation': False}}
        put(self.plan / 'plan.json', self.config)
        for name in m.PLAN_FILES - {'plan.json', 'seal.json'}:
            put(self.plan / name, {'synthetic_never_parsed': True})
        put(self.plan / 'seal.json', {n: m.digest(self.plan / n) for n in m.PLAN_FILES - {'seal.json'}})
        parents = {str(self.plan / n): m.digest(self.plan / n) for n in m.PLAN_FILES}
        for name in m.RUN_FILES - {'summary.json'}:
            put(self.run / name, {'synthetic_not_a_model': True})
        put(self.run / 'prepared.json', self.prepared)
        put(self.run / 'public_aggregate.json', {'status': 'completed_candidates', 'counts': counts,
            'support_pairs': 248, 'gold_read': False, 'quality_metrics_computed': False,
            'paid_execution_admitted': False, 'api_calls': 0})
        self.summary = {'schema': m.CANDIDATE_SCHEMA, 'status': 'completed_candidates', 'gold_read': False,
            'api_calls': 0, 'plan_sha256': parents, 'input_sha256': inherited,
            'output_sha256': {n: m.digest(self.run / n) for n in m.RUN_FILES - {'summary.json'}}}
        put(self.run / 'summary.json', self.summary)
        self.audit = tmp_path / 'audit.json'
        self.receipt = {'schema': m.CANDIDATE_SCHEMA, 'status': 'verified_complete_with_sample',
            'gold_read': False, 'api_calls': 0, 'counts': counts, 'source_sha256': inherited,
            'plan_sha256': parents, 'run_sha256': {str(self.run / n): m.digest(self.run / n) for n in m.RUN_FILES}}
        put(self.audit, self.receipt)


def test_loader_reads_no_references_or_deferred_plan_containers(tmp_path, monkeypatch):
    f = CandidateFixture(tmp_path)
    original = Path.read_bytes

    def guard(path):
        assert path.name not in {'references.jsonl', 'input.json', 'token_ids.json', 'length_audit.json', 'embeddings.safetensors'}
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', guard)
    data, bindings, inherited = m.load_candidates(f.run, f.audit)
    assert data == f.prepared and inherited == f.config['input_sha256']
    assert str(f.audit) in bindings and not any(Path(p).name == 'references.jsonl' for p in bindings)


@pytest.mark.parametrize('target', ['prepared', 'incomplete_audit', 'bad_plan', 'extra_file'])
def test_loader_rejects_broken_completed_lineage(tmp_path, target):
    f = CandidateFixture(tmp_path)
    if target == 'prepared':
        (f.run / 'prepared.json').write_bytes((f.run / 'prepared.json').read_bytes() + b'\n')
    elif target == 'incomplete_audit':
        f.receipt['status'] = 'partial'; put(f.audit, f.receipt)
    elif target == 'bad_plan':
        (f.plan / 'plan.json').write_bytes((f.plan / 'plan.json').read_bytes() + b'\n')
    else:
        (f.run / 'failure.json').write_text('{}')
    with pytest.raises(ValueError):
        m.load_candidates(f.run, f.audit)


def test_reference_paths_rejected_before_open(tmp_path, monkeypatch):
    path = tmp_path / 'references.jsonl'
    def forbidden(*args, **kwargs):
        raise AssertionError('reference opened')
    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(ValueError, match='reference'):
        m.digest(path)
    with pytest.raises(ValueError, match='reference'):
        m.read_bound(path, {})


def test_output_is_private_sealed_and_nonoverwrite(tmp_path, monkeypatch, snapshot):
    f = CandidateFixture(tmp_path)
    art = tmp_path / 'ignored'
    art.mkdir()
    monkeypatch.setattr(m, 'ART', art)
    output = art / 'budget'
    public = m.plan_budget(f.run, f.audit, output)
    assert {p.name for p in output.iterdir()} == m.OUTPUT_FILES
    seal = json.loads((output / 'seal.json').read_bytes())
    assert seal == {n: m.digest(output / n) for n in m.OUTPUT_FILES - {'seal.json'}}
    assert public['Q_questions'] == 248 and public['paid_execution_admitted'] is False
    with pytest.raises(ValueError, match='new ignored'):
        m.plan_budget(f.run, f.audit, output)


def test_source_change_during_budget_calculation_prevents_output(tmp_path, monkeypatch):
    f = CandidateFixture(tmp_path)
    art = tmp_path / 'ignored'; art.mkdir(); monkeypatch.setattr(m, 'ART', art)
    original = m.build_budget
    def changing(data, snapshot):
        result = original(data, snapshot)
        (f.run / 'prepared.json').write_bytes((f.run / 'prepared.json').read_bytes() + b'\n')
        return result
    monkeypatch.setattr(m, 'build_budget', changing)
    with pytest.raises(ValueError, match='bound input changed'):
        m.plan_budget(f.run, f.audit, art / 'budget')
    assert not (art / 'budget').exists()


def test_previously_pinned_source_cannot_be_recaptured_with_new_hash(tmp_path, monkeypatch):
    f = CandidateFixture(tmp_path)
    art = tmp_path / 'ignored'; art.mkdir(); monkeypatch.setattr(m, 'ART', art)
    original = m.digest
    target = Path(m.client.__file__).resolve()
    calls = 0
    def changing_digest(path):
        nonlocal calls
        actual = original(path)
        if Path(path).resolve() == target:
            calls += 1
            if calls == 2:
                return '0' * 64
        return actual
    monkeypatch.setattr(m, 'digest', changing_digest)
    with pytest.raises(ValueError, match='conflicting source commitment'):
        m.plan_budget(f.run, f.audit, art / 'budget')
    assert calls == 2 and not (art / 'budget').exists()


def test_change_while_writing_leaves_no_final_plan_seal(tmp_path, monkeypatch):
    f = CandidateFixture(tmp_path)
    art = tmp_path / 'ignored'; art.mkdir(); monkeypatch.setattr(m, 'ART', art)
    output = art / 'budget'
    original = m.verify
    def changing_verify(bindings):
        if output.exists():
            (f.run / 'prepared.json').write_bytes((f.run / 'prepared.json').read_bytes() + b'\n')
        return original(bindings)
    monkeypatch.setattr(m, 'verify', changing_verify)
    with pytest.raises(ValueError, match='bound input changed'):
        m.plan_budget(f.run, f.audit, output)
    assert output.exists() and not (output / 'seal.json').exists()


def test_changed_input_inventory_during_calculation_is_rejected(tmp_path, monkeypatch):
    f = CandidateFixture(tmp_path)
    art = tmp_path / 'ignored'; art.mkdir(); monkeypatch.setattr(m, 'ART', art)
    original = m.build_budget
    def changing(data, snapshot):
        result = original(data, snapshot)
        (f.run / 'failure.json').write_text('{}')
        return result
    monkeypatch.setattr(m, 'build_budget', changing)
    with pytest.raises(ValueError, match='inventory changed'):
        m.plan_budget(f.run, f.audit, art / 'budget')


def test_old_night_request_and_money_caps_are_not_new_planner_caps(snapshot):
    private, public = m.build_budget(prepared((1,) * 600), snapshot)
    assert len(private['jobs']) == 1200
    assert Decimal(public['support_total_reserved_usd']) >= Decimal('6')
    assert public['specification']['new_stage_budget_cap_usd'] is None
    assert public['paid_execution_admitted'] is False
