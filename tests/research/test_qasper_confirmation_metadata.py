"""Metadata adapter tests: synthetic files only, no archive/QA/model/API access."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'docs/research'))
spec = importlib.util.spec_from_file_location('confirmation_metadata_test_adapter', REPO / 'docs/research/prepare_qasper_confirmation_metadata.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
POLICY_EVALUATE = m.policy.evaluate_metadata


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value) + b'\n')


def put_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b''.join(m.canonical(v) + b'\n' for v in values)
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(content, mtime=0))
    else:
        path.write_bytes(content)


def forbidden(*args, **kwargs):
    raise AssertionError('metadata preparation must not evaluate policy or read deferred data')


@pytest.fixture(autouse=True)
def no_evaluation(monkeypatch):
    monkeypatch.setattr(m.policy, 'evaluate_metadata', forbidden)


class SyntheticSources:
    """Full 888/281/32/249 denominators without source text or QA content."""
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / 'sources'
        self.art = self.root / 'artifacts'
        self.index = self.root / 'metadata/index.jsonl.gz'
        self.reservation = self.root / 'metadata/development_reservations.json'
        self.pilot = self.root / 'metadata/public_pilot_manifest.jsonl'
        self.rows = []
        for split, size in [('train', 888), ('validation', 281)]:
            for number in range(size):
                doc = f'{split}-{number:04d}'
                self.rows.append({'source': 'qasper', 'original_split': split, 'doc_id': doc,
                    'source_id': f'2101.{number + (0 if split == "train" else 10000):05d}',
                    'family_id': 'family-' + doc, 'normalized_body_sha256': hashlib.sha256(doc.encode()).hexdigest(),
                    'development_exposed': False, 'metadata_not_used': 'synthetic ignored field'})
        self.train = self.rows[:888]
        self.validation = self.rows[888:]
        self.prior = self.validation[:32]
        self.remaining = self.validation[32:]
        self.pool_name = 'qasper-pool/pool_manifest.json'
        self.screen_prefix = 'qasper-remaining-validation-screen-01/'
        self.review_prefix = 'qasper-validation-source-review-01/'
        put_rows(self.index, self.rows)
        put(self.reservation, {k: [] for k in ['doc_ids', 'family_ids', 'normalized_body_sha256']})
        put_rows(self.pilot, [{'paper_id': f'exposed-source-{i}'} for i in range(3)])
        # Deliberately nonexistent deferred artifacts: source_data must inherit commitments only.
        self.deferred = self.root / 'NEVER_OPEN_ARCHIVE_OR_QA'
        self.inherited = {str(self.deferred): 'a' * 64}
        self.pool = {'input_sha256': {**self.inherited,
            str(self.index): m.digest(self.index), str(self.reservation): m.digest(self.reservation),
            str(self.pilot): m.digest(self.pilot)},
            'known_exposed_qasper_source_ids': [f'exposed-source-{i}' for i in range(3)]}
        put(self.path(self.pool_name), self.pool)
        put_rows(self.path('qasper-pool/candidates.jsonl'), [
            {**{k: r[k] for k in ['doc_id', 'source_id', 'family_id', 'normalized_body_sha256']},
             'official_split': 'validation'} for r in self.prior])
        put(self.path('qasper-pool/eligibility_audit.json'), {'eligible_ids': [r['doc_id'] for r in self.validation]})
        put(self.path('qasper-pool/native_qa_alignment.json'), {
            'all_input_hashes_unchanged': True, 'raw_member_read': 'qasper-dev-v0.3.json',
            'test_payload_read': False, 'documents': 32, 'counts': {'questions': 104},
            'per_document': [{'doc_id': r['doc_id']} for r in self.prior],
            'input_sha256': self.inherited, 'sidecar_sha256': 'b' * 64})
        remaining = [r['doc_id'] for r in self.remaining]
        put(self.path(self.screen_prefix + 'plan.json'), {
            'batch_doc_ids': [remaining[i:i+32] for i in range(0, 249, 32)],
            'current_doc_ids': [r['doc_id'] for r in self.prior], 'input_sha256': self.inherited})
        put(self.path(self.screen_prefix + 'public_aggregate.json'), {
            'status': 'complete_document_screen_with_limits',
            'completed_batches': 8, 'remaining_documents': 249, 'flagged_pairs_unique': 1,
            'legacy_orig_split_counts_unique_source_rows': {'train-file': {'test': 857}, 'dev-file': {'test': 91}}})
        put_rows(self.path(self.screen_prefix + 'flagged_pairs_unique.jsonl'), [{
            'comparison_scope': 'qasper_train', 'representative': {
                'reference_scope': 'qasper_train', 'query_doc_id': self.remaining[0]['doc_id'],
                'reference_doc_id': self.train[0]['doc_id'],
                'review_reasons': ['moderate_lexical_overlap_review']}}])
        put(self.path(self.screen_prefix + 'completion.json'), {'status': 'completed', 'output_sha256': {}})
        put(self.path(self.screen_prefix + 'audit_receipt.json'), {
            'status': 'verified_complete_document_screen', 'remaining_documents': 249, 'qa_payload_read': False,
            'public_aggregate_sha256': ''})
        put(self.path(self.review_prefix + 'source_evidence_private.json'), {
            'action_flags': {'human_review_completed': False},
            'pair': {scope: {k: row[k] for k in ['doc_id', 'source_id', 'family_id']}
                     for scope, row in [('validation', self.remaining[0]), ('train', self.train[0])]},
            'relationship': {'explicit_material_reuse_and_additional_research_statement': True,
                             'reference_maps_to_earlier_document': True}})
        put(self.path(self.review_prefix + 'review_seal.json'), {'files_sha256': {}})
        exporter = self.root / 'docs/research/export_qasper_native_sidecar.py'
        exporter.parent.mkdir(parents=True, exist_ok=True)
        exporter.write_text('# synthetic historical exporter commitment\n', encoding='utf-8')
        own = self.root / 'adapter_source.py'
        own.write_text('# synthetic source\n', encoding='utf-8')
        monkeypatch.setattr(m, 'ROOT', self.root)
        monkeypatch.setattr(m, 'ART', self.art)
        monkeypatch.setattr(m, 'EXPORTER_SHA', m.digest(exporter))
        monkeypatch.setattr(m, 'own_files', lambda: [own])
        self.names = tuple(m.PINNED)
        self.monkeypatch = monkeypatch
        self.refresh()

    def path(self, name):
        return self.art / name

    def change(self, name, fn):
        value = m.load(self.path(name))
        fn(value)
        put(self.path(name), value)
        self.refresh()

    def refresh(self):
        pool = m.load(self.path(self.pool_name))
        for path in [self.index, self.reservation, self.pilot]:
            pool['input_sha256'][str(path)] = m.digest(path)
        put(self.path(self.pool_name), pool)
        completion = m.load(self.path(self.screen_prefix + 'completion.json'))
        completion['output_sha256'] = {name: m.digest(self.path(self.screen_prefix + name))
            for name in ['public_aggregate.json', 'flagged_pairs_unique.jsonl']}
        put(self.path(self.screen_prefix + 'completion.json'), completion)
        audit = m.load(self.path(self.screen_prefix + 'audit_receipt.json'))
        audit['public_aggregate_sha256'] = m.digest(self.path(self.screen_prefix + 'public_aggregate.json'))
        put(self.path(self.screen_prefix + 'audit_receipt.json'), audit)
        put(self.path(self.review_prefix + 'review_seal.json'), {'files_sha256': {
            'source_evidence_private.json': m.digest(self.path(self.review_prefix + 'source_evidence_private.json'))}})
        self.monkeypatch.setattr(m, 'PINNED', {name: m.digest(self.path(name)) for name in self.names})


@pytest.fixture
def sources(tmp_path, monkeypatch):
    return SyntheticSources(tmp_path, monkeypatch)


def test_complete_projection_never_opens_deferred_data_or_evaluates(sources):
    payload, bindings, inherited = m.source_data()
    assert Counter(r['split'] for r in payload['index_rows']) == {'train': 888, 'validation': 281}
    assert len(payload['prior_development_ids']) == 32
    assert len({r['doc_id'] for r in payload['index_rows'] if r['split'] == 'validation'} - set(payload['prior_development_ids'])) == 249
    assert all(set(r) == {'source', 'split', 'doc_id', 'source_id', 'family_id', 'normalized_body_sha256', 'development_exposed'} for r in payload['index_rows'])
    assert not sources.deferred.exists() and str(sources.deferred) not in bindings
    assert str(sources.deferred) in inherited['native_archive_and_sidecar']
    assert payload['source_reviews'][0]['explicit_material_reuse'] is True
    history = payload['exposure_history']
    assert history['source_review_human_completed'] is False
    assert history['historical_machine_decoded_entire_validation_member'] is True
    assert history['execution_time_source_hash_in_alignment_report'] is False
    assert history['model_pretraining_contamination'] == 'unknown'
    assert history['legacy_orig_split_counts_inherited']['train-file']['test'] == 857


def test_policy_input_projects_only_metadata_and_converts_slots(sources):
    payload, _, _ = m.source_data()
    projected = m.policy_input(payload)
    assert set(projected) == {'index_rows', 'known_exposure', 'prior_development_ids', 'overlap_edges', 'source_reviews'}
    assert isinstance(projected['prior_development_ids'], set)
    assert all(isinstance(v, set) for v in projected['known_exposure'].values())
    assert len(projected['known_exposure']['source_ids']) == 3


def test_changed_pinned_hash_rejected(sources):
    sources.path('qasper-pool/eligibility_audit.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='bound file changed'):
        m.source_data()


def test_changed_named_metadata_hash_rejected(sources):
    sources.index.write_bytes(b'changed')
    with pytest.raises(ValueError, match='metadata source changed'):
        m.source_data()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'invalid_split', 'wrong_source'])
def test_index_identity_and_denominator_rejected(sources, mutation):
    rows = deepcopy(sources.rows)
    if mutation == 'duplicate': rows[-1]['doc_id'] = rows[-2]['doc_id']
    elif mutation == 'missing': rows.pop()
    elif mutation == 'invalid_split': rows[-1]['original_split'] = 'test'
    else: rows[-1]['source'] = 'other'
    put_rows(sources.index, rows)
    sources.refresh()
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_family', 'invalid_split'])
def test_prior_cohort_contract(sources, mutation):
    name = 'qasper-pool/candidates.jsonl'
    rows = m.rows(sources.path(name))
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = deepcopy(rows[0])
    elif mutation == 'wrong_family': rows[0]['family_id'] = 'wrong-family'
    else: rows[0]['official_split'] = 'test'
    put_rows(sources.path(name), rows); sources.refresh()
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('mutation', ['missing', 'duplicate'])
def test_original_eligible_coverage(sources, mutation):
    def mutate(value):
        if mutation == 'missing': value['eligible_ids'].pop()
        else: value['eligible_ids'][-1] = value['eligible_ids'][0]
    sources.change('qasper-pool/eligibility_audit.json', mutate)
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'prior_leak', 'wrong_current'])
def test_remaining_coverage(sources, mutation):
    def mutate(value):
        if mutation == 'missing': value['batch_doc_ids'][-1].pop()
        elif mutation == 'duplicate': value['batch_doc_ids'][-1][-1] = value['batch_doc_ids'][0][0]
        elif mutation == 'prior_leak': value['batch_doc_ids'][-1][-1] = value['current_doc_ids'][0]
        else: value['current_doc_ids'][-1] = value['batch_doc_ids'][0][0]
    sources.change(sources.screen_prefix + 'plan.json', mutate)
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('name,field,value', [
    ('completion.json', 'status', 'partial'),
    ('public_aggregate.json', 'completed_batches', 7),
    ('public_aggregate.json', 'status', 'partial'),
    ('audit_receipt.json', 'status', 'partial'),
    ('audit_receipt.json', 'qa_payload_read', True),
])
def test_incomplete_or_changed_screen_contract(sources, name, field, value):
    sources.change(sources.screen_prefix + name, lambda obj: obj.__setitem__(field, value))
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('count', [0, 2])
def test_flag_actual_count_must_match_one(sources, count):
    name = sources.screen_prefix + 'flagged_pairs_unique.jsonl'
    values = m.rows(sources.path(name))
    put_rows(sources.path(name), values * count); sources.refresh()
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('mutation', ['wrong_pair', 'wrong_family', 'human_claim', 'missing_declaration', 'missing_reference'])
def test_source_review_join_and_provenance(sources, mutation):
    def mutate(value):
        if mutation == 'wrong_pair': value['pair']['validation']['doc_id'] = sources.remaining[1]['doc_id']
        elif mutation == 'wrong_family': value['pair']['train']['family_id'] = 'wrong-family'
        elif mutation == 'human_claim': value['action_flags']['human_review_completed'] = True
        elif mutation == 'missing_declaration': value['relationship']['explicit_material_reuse_and_additional_research_statement'] = False
        else: value['relationship']['reference_maps_to_earlier_document'] = False
    sources.change(sources.review_prefix + 'source_evidence_private.json', mutate)
    with pytest.raises(ValueError): m.source_data()


@pytest.mark.parametrize('field,value', [('documents',31), ('raw_member_read','qasper-test.json'), ('test_payload_read',True)])
def test_historical_decode_evidence_contract(sources, field, value):
    sources.change('qasper-pool/native_qa_alignment.json', lambda obj: obj.__setitem__(field,value))
    with pytest.raises(ValueError): m.source_data()


@pytest.fixture
def prepared(sources, tmp_path):
    directory, run = tmp_path / 'plan', tmp_path / 'run'
    m.prepare(directory, run)
    return directory, run


def reseal(directory):
    put(directory / 'seal.json', {name: m.digest(directory/name) for name in m.PLAN_FILES - {'seal.json'}})


def test_prepare_and_load_are_metadata_only(sources, prepared):
    directory, run = prepared
    plan, payload, bindings = m.load_plan(directory)
    assert plan['status'] == 'prepared_metadata_policy_not_executed'
    assert not plan['policy_executed'] and not plan['new_qa_read'] and plan['api_calls'] == 0
    assert len(payload['index_rows']) == 1169 and len(bindings) == 3 and not run.exists()


@pytest.mark.parametrize('which', ['plan', 'run'])
def test_prepare_refuses_existing_output(sources, tmp_path, which):
    plan, run = tmp_path / 'plan', tmp_path / 'run'
    (plan if which == 'plan' else run).mkdir()
    with pytest.raises(ValueError, match='must be new'): m.prepare(plan, run)
    assert not (run if which == 'plan' else plan).exists()


@pytest.mark.parametrize('case', ['equal', 'plan_contains_run', 'run_contains_plan', 'source_descendant'])
def test_prepare_overlap_refused_before_writes(sources, tmp_path, case):
    plan, run = tmp_path / 'plan', tmp_path / 'run'
    if case == 'equal': run = plan
    elif case == 'plan_contains_run': run = plan / 'run'
    elif case == 'run_contains_plan': plan = run / 'plan'
    else: plan = sources.index / 'plan'
    with pytest.raises(ValueError, match='overlaps'): m.prepare(plan, run)
    assert not plan.exists() and not run.exists()


def test_no_overlap_rejects_ancestor_of_source_even_without_source_file(tmp_path):
    with pytest.raises(ValueError, match='overlaps'):
        m.no_overlap(tmp_path / 'new', [tmp_path / 'new' / 'source.json'])


@pytest.mark.parametrize('file', ['plan.json','input.json'])
def test_plan_byte_tamper_without_new_seal_fails(prepared, file):
    directory, _ = prepared
    with (directory/file).open('ab') as handle: handle.write(b' ')
    with pytest.raises(ValueError, match='seal'): m.load_plan(directory)


@pytest.mark.parametrize('mutation', ['extra', 'missing'])
def test_plan_inventory_is_exact(prepared, mutation):
    directory, _ = prepared
    if mutation == 'extra': (directory/'extra.json').write_bytes(b'{}')
    else: (directory/'input.json').unlink()
    with pytest.raises(ValueError, match='inventory'): m.load_plan(directory)


@pytest.mark.parametrize('field,value', [('status','completed'), ('policy_executed',True), ('api_calls',1), ('new_qa_read',True)])
def test_resealed_plan_contract_changes_fail(prepared, field, value):
    directory, _ = prepared
    plan = m.load(directory/'plan.json'); plan[field] = value; put(directory/'plan.json',plan); reseal(directory)
    with pytest.raises(ValueError, match='contract'): m.load_plan(directory)


@pytest.mark.parametrize('mutation', ['projection','inherited','payload_hash','run_overlap'])
def test_resealed_plan_semantic_tamper_fails(prepared, mutation):
    directory, _ = prepared
    plan = m.load(directory/'plan.json')
    if mutation == 'projection':
        payload = m.load(directory/'input.json'); payload['prior_development_ids'].pop()
        put(directory/'input.json',payload); plan['payload_sha256'] = hashlib.sha256(m.canonical(payload)).hexdigest()
    elif mutation == 'inherited': plan['inherited_commitments_not_freshly_rehashed']['native_sidecar_sha256'] = 'c'*64
    elif mutation == 'payload_hash': plan['payload_sha256'] = 'c'*64
    else: plan['run_output'] = str(directory/'run')
    put(directory/'plan.json',plan); reseal(directory)
    with pytest.raises(ValueError): m.load_plan(directory)


def test_fresh_source_changed_after_prepare_rejected(prepared, sources):
    sources.index.write_bytes(b'changed after seal')
    with pytest.raises(ValueError): m.load_plan(prepared[0])


def test_load_plan_rechecks_captured_bytes_after_source_projection(prepared, monkeypatch):
    directory, _ = prepared
    original = m.source_data
    def mutate_plan_after_read():
        result = original()
        with (directory/'plan.json').open('ab') as handle: handle.write(b' ')
        return result
    monkeypatch.setattr(m, 'source_data', mutate_plan_after_read)
    with pytest.raises(ValueError, match='bound file changed'): m.load_plan(directory)


@pytest.fixture
def completed(prepared, monkeypatch):
    # Only this explicitly synthetic fixture enables the pure metadata policy.
    monkeypatch.setattr(m.policy, 'evaluate_metadata', POLICY_EVALUATE)
    directory, run = prepared
    result = m.run(directory)
    assert result['status'] == 'completed_metadata_proposal'
    return directory, run


def test_complete_run_and_recomputed_audit_keep_all_rows_without_admission(completed, tmp_path):
    directory, run = completed
    ledger = m.rows(run/'validation_ledger.jsonl')
    proposed = m.rows(run/'proposed_cohort.jsonl')
    public = m.load(run/'public_aggregate.json')
    assert len(ledger) == 281 and len({r['doc_id'] for r in ledger}) == 281
    assert public['canonical_documents'] == 1169 and public['canonical_validation_documents'] == 281
    assert all(public[k] is False for k in m.policy.FLAGS)
    assert {r['doc_id'] for r in proposed} == {r['doc_id'] for r in ledger if r['status'] == 'operational_candidate_proposed'}
    assert {p.name for p in run.iterdir()} == m.RUN_FILES
    receipt = m.audit(directory, tmp_path/'audit')
    assert receipt['status'] == 'verified_metadata_proposal'
    detail = m.load(tmp_path/'audit/verification.json')
    assert not detail['cohort_admitted'] and not detail['cleared'] and not detail['new_qa_read']
    assert 'not an independent algorithm' in detail['verification_scope']
    assert len(detail['run_sha256']) == len(m.RUN_FILES)


def test_completed_run_cannot_be_repeated(completed):
    with pytest.raises(ValueError, match='no retry or overwrite'): m.run(completed[0])


def test_failed_policy_leaves_registration_and_refuses_resume(prepared, monkeypatch):
    directory, run = prepared
    monkeypatch.setattr(m, 'execute_policy', forbidden)
    with pytest.raises(AssertionError): m.run(directory)
    assert {p.name for p in run.iterdir()} == {'run_registration.json'}
    with pytest.raises(ValueError, match='no retry or overwrite'): m.run(directory)


def test_source_mutation_during_policy_cannot_create_completed_output(prepared, sources, monkeypatch):
    directory, run = prepared
    def mutate(payload):
        result = POLICY_EVALUATE(**m.policy_input(payload))
        sources.index.write_bytes(b'changed during policy')
        return result
    monkeypatch.setattr(m, 'execute_policy', mutate)
    with pytest.raises(ValueError, match='bound file changed'): m.run(directory)
    assert {p.name for p in run.iterdir()} == {'run_registration.json'}


@pytest.mark.parametrize('mutation', ['extra', 'missing'])
def test_audit_requires_complete_run_inventory(completed, tmp_path, mutation):
    directory, run = completed
    if mutation == 'extra': (run/'extra').write_bytes(b'x')
    else: (run/'components.json').unlink()
    with pytest.raises(ValueError, match='inventory'): m.audit(directory, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('output', ['validation_ledger.jsonl','proposed_cohort.jsonl','components.json','public_aggregate.json'])
def test_audit_recomputes_all_output_bytes_after_attacker_reseals(completed, tmp_path, output):
    directory, run = completed
    with (run/output).open('ab') as handle: handle.write(b' ')
    summary = m.load(run/'summary.json'); summary['output_sha256'][output] = m.digest(run/output)
    put(run/'summary.json', summary)
    with pytest.raises(ValueError, match='policy recomputation'): m.audit(directory, tmp_path/'audit')


@pytest.mark.parametrize('field,value', [('status','partial'), ('cohort_admitted',True), ('cleared',True), ('new_qa_read',True), ('api_calls',1)])
def test_audit_rejects_false_completion_or_admission_claim(completed, tmp_path, field, value):
    directory, run = completed
    summary = m.load(run/'summary.json'); summary[field] = value; put(run/'summary.json',summary)
    with pytest.raises(ValueError, match='run contract'): m.audit(directory, tmp_path/'audit')


def test_audit_rejects_resealed_registration_change(completed, tmp_path):
    directory, run = completed
    registration = m.load(run/'run_registration.json'); registration['plan_sha256'] = {}
    put(run/'run_registration.json', registration)
    summary = m.load(run/'summary.json'); summary['output_sha256']['run_registration.json'] = m.digest(run/'run_registration.json')
    put(run/'summary.json', summary)
    with pytest.raises(ValueError, match='registration'): m.audit(directory, tmp_path/'audit')


def test_audit_refuses_existing_or_overlapping_output(completed, tmp_path):
    directory, run = completed
    audit = tmp_path/'audit'
    m.audit(directory, audit)
    with pytest.raises(ValueError, match='must be new'): m.audit(directory, audit)
    for target in [run/'audit', directory/'audit']:
        with pytest.raises(ValueError, match='overlaps'): m.audit(directory, target)


def test_audit_rechecks_captured_run_bytes_after_recomputation(completed, tmp_path, monkeypatch):
    directory, run = completed
    original = m.execute_policy
    def mutate(payload):
        result = original(payload)
        with (run/'public_aggregate.json').open('ab') as handle: handle.write(b' ')
        return result
    monkeypatch.setattr(m, 'execute_policy', mutate)
    with pytest.raises(ValueError, match='bound file changed'): m.audit(directory, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()
