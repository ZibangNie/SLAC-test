"""Offline common support batches and conditional five-arm answer budget.

This module has no execution command. It never reads references, credentials,
provider responses, or model weights. A completed candidate audit is inherited;
its retrieval is not rerun. Private payloads stay in a new ignored directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import openrouter_decision_client as client
import run_qasper_extended_development as stage
import run_qasper_answer_evaluation as answers

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/research-foundation'
SCHEMA = 'slac-qasper-confirmation-budget-v1'
CANDIDATE_SCHEMA = 'slac-qasper-confirmation-candidates-v1'
PROFILE = 'qwen36plus-json'
SNAPSHOT = ROOT / 'docs/research/results/qasper_confirmation_provider_snapshot_20260927.json'
SNAPSHOT_SHA256 = '772e39b49ea2ad7bc19120f6a435736480a4438d695237efe5ca504f4aecf8fc'
RUN_FILES = {'registration.json', 'embeddings.safetensors', 'embedding_index.json', 'rankings.jsonl',
             'prepared.json', 'dense_packs.jsonl', 'sample_check.json', 'public_aggregate.json', 'summary.json'}
PLAN_FILES = {'plan.json', 'input.json', 'token_ids.json', 'length_audit.json', 'seal.json'}
OUTPUT_FILES = {'private_plan.json', 'public_aggregate.json', 'source_binding.json', 'seal.json'}
METHODS = ('dense_k3', 'reranker_k3', 'I_jev_k3', 'I_general_k3', 'p_yes_only_k3')
SPEC = {
    'general_profile': PROFILE, 'maximum_support_items': 8, 'canonical_payload_byte_cap': 24000,
    'input_byte_allowance_extra': 2048, 'safety_multiplier': '1.5', 'support_output_allowance': 1024,
    'support_floor_usd': '0.005', 'generation_output_allowance': 512, 'generation_floor_usd': '0.001',
    'answer_methods': list(METHODS), 'support_reuse': 'one JEV response feeds I_jev and p_yes_only',
    'common_batches': True, 'automatic_retries_planned': 0, 'static_requests_planned': 0,
    'paid_execution_admitted': False, 'deadline': None, 'new_stage_budget_cap_usd': None,
    'old_night_deadline_or_budget_inherited': False,
}
LIMITS = [
    'Support reservations are exact applications of the frozen local formula to complete saved payloads, not exact provider charges.',
    'The answer upper bound is five requests per question at the byte cap, without deduplication or cache discounts.',
    'The answer bound is conditional on each eventual complete payload fitting 24000 canonical UTF-8 bytes. Packing/token limits alone do not establish that condition.',
    'Actual answer payloads, unique generation count, budget admission, model access, returned model and execution deadline remain unresolved.',
    'The dated public provider snapshot establishes metadata and price assumptions, not current inference availability or account access.',
    'No old overnight budget, deadline, spend refund, or paid authorization is inherited. There is no request execution entry point.',
    'This planning stage reads model-visible candidate containers; it never opens evaluator-only references or computes quality.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    require(Path(path).name not in {'references.jsonl', 'native_qa_sidecar.jsonl', 'native_qa_sidecar_v2.jsonl'},
            'reference files are forbidden inputs')
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_bound(path, bindings):
    path = Path(path).resolve()
    require(path.name not in {'references.jsonl', 'native_qa_sidecar.jsonl', 'native_qa_sidecar_v2.jsonl'},
            'reference files are forbidden inputs')
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(str(path) not in bindings or bindings[str(path)] == actual, 'input changed before parsing')
    bindings[str(path)] = actual
    return json.loads(raw, object_pairs_hook=client.unique_object)


def verify(bindings):
    for path, expected in bindings.items():
        require(digest(path) == expected, 'bound input changed')


def bind_hash(bindings, path, expected):
    path = str(Path(path).resolve())
    require(path not in bindings or bindings[path] == expected, 'conflicting source commitment')
    bindings[path] = expected


def verify_candidate_inventories(run, bindings):
    require({p.name for p in Path(run).iterdir()} == RUN_FILES, 'candidate run inventory changed')
    plan_dirs = {Path(p).parent for p in bindings if Path(p).name == 'seal.json'}
    require(len(plan_dirs) == 1, 'candidate plan directory not unique')
    require({p.name for p in next(iter(plan_dirs)).iterdir()} == PLAN_FILES, 'candidate plan inventory changed')


@contextmanager
def qwen_profile():
    models, responses = deepcopy(client.MODELS), deepcopy(client.RESPONSE_MODELS)
    try:
        client.select_general_profile(PROFILE)
        yield
    finally:
        client.MODELS.clear(); client.MODELS.update(models)
        client.RESPONSE_MODELS.clear(); client.RESPONSE_MODELS.update(responses)


def validate_snapshot(snapshot):
    require(snapshot['schema'] == 'slac-confirmation-provider-metadata-review-v1'
            and snapshot['status'] == 'read_only_public_metadata_snapshot_not_execution_admission',
            'provider snapshot schema/status differs')
    rows = {row['role']: row for row in snapshot['models']}
    require(len(rows) == len(snapshot['models']) == 2 and set(rows) == {'jev', 'qwen'}, 'provider roles differ')
    require(client.BYTE_CAP == 24000 and client.MAX_OUTPUT_TOKENS == 1024 and answers.MAX_OUTPUT == 512,
            'payload allowance implementation differs')
    require(client.MODELS['general'] == client.GENERAL_PROFILES[PROFILE]
            and client.MODELS['general']['output_format'] == 'json_object', 'explicit Qwen JSON profile required')
    for backend, role, display in [('jev', 'jev', 'TypeSafe'), ('general', 'qwen', 'Alibaba')]:
        model, row = client.MODELS[backend], rows[role]
        endpoint = row['endpoint_metadata']
        require(row['http_status'] == 200 and row['endpoint_count'] == 1
                and row['request_endpoint'] == model['endpoint'] and row['request_model_id'] == model['id']
                and endpoint['model_id'] == model['id'] and endpoint['tag'] == model['provider']
                and endpoint['provider_name'] == display, 'model/provider metadata differs')
        revision = model.get('observed_endpoint_revision', model.get('observed_canonical_slug'))
        require(endpoint['name'] == f'{display} | {revision}', 'observed provider revision differs')
        for key, field in [('prompt', 'prompt_per_million'), ('completion', 'completion_per_million')]:
            value = Decimal(model[field])
            require(value.is_finite() and value >= 0
                    and Decimal(endpoint['pricing'][key]) * 1000000 == value
                    and Decimal(row['base_prices_usd_per_million'][key]) == value, 'provider price differs')
    require(answers.MODEL == client.MODELS['general']['id']
            and all(Decimal(answers.PRICES[k]) == Decimal(client.MODELS['general'][f'{k}_per_million'])
                    for k in ('prompt', 'completion')) and answers.PRICES['request'] == '0', 'generator price/profile differs')
    expected = {'currency': 'USD', 'canonical_payload_byte_cap': 24000, 'input_byte_allowance_extra': 2048,
                'safety_multiplier': '1.5', 'maximum_items_per_support_batch': 8, 'support_output_allowance': 1024,
                'support_floor_usd': '0.005', 'generation_output_allowance': 512, 'generation_floor_usd': '0.001'}
    require(all(snapshot['reservation_contract'][k] == v for k, v in expected.items()), 'reservation metadata differs')
    return rows


def validate_prepared(prepared):
    require(set(prepared) == {'schema', 'documents', 'queries', 'support_tasks', 'static_tasks'}
            and prepared['schema'] == CANDIDATE_SCHEMA and prepared['static_tasks'] == [], 'candidate schema differs')
    queries, documents = prepared['queries'], prepared['documents']
    require(isinstance(queries, list) and queries and isinstance(documents, dict) and documents, 'empty candidate population')
    units = {}
    for doc, rows in documents.items():
        require(isinstance(doc, str) and doc and isinstance(rows, list) and rows, 'invalid candidate document')
        require(all(set(u) == {'unit_id', 'order', 'kind', 'start', 'end', 'text', 'native_text'} for u in rows),
                'unit projection fields differ')
        require(all(all(isinstance(u[k], str) and u[k].strip() for k in ('unit_id', 'kind', 'text', 'native_text'))
                    and type(u['start']) is int and type(u['end']) is int and 0 <= u['start'] <= u['end'] for u in rows),
                'invalid unit identity/text/span')
        by_id = {u['unit_id']: u for u in rows}
        require(len(by_id) == len(rows) and all(type(u['order']) is int and u['order'] == i
                and isinstance(u['text'], str) and u['text'].strip() for i, u in enumerate(rows)), 'unit identity/order differs')
        units[doc] = by_id
    lookup, expected = {}, []
    for q in queries:
        require(set(q) == {'family_id', 'doc_id', 'question_id', 'source_id', 'original_family_id', 'query',
                           'seed_ids', 'candidate_ids', 'ranked_ids'}, 'query projection fields differ')
        require(all(isinstance(q[k], str) and q[k].strip() for k in ('family_id', 'doc_id', 'question_id',
                'source_id', 'original_family_id', 'query')), 'invalid query identity/text')
        key = q['doc_id'], q['question_id']
        require(key not in lookup and q['doc_id'] in documents, 'duplicate query or unknown document')
        lookup[key] = q
        ids = q['candidate_ids']
        require(1 <= len(ids) <= 16 and len(set(ids)) == len(ids) and set(ids) <= set(units[q['doc_id']])
                and len(q['ranked_ids']) == len(ids) and set(q['ranked_ids']) == set(ids)
                and 1 <= len(q['seed_ids']) <= 8 and len(set(q['seed_ids'])) == len(q['seed_ids'])
                and set(q['seed_ids']) <= set(ids), 'candidate identity coverage differs')
        for unit in documents[q['doc_id']]:
            if unit['unit_id'] in ids:
                item = {'query': q['query'], 'unit': {'id': unit['unit_id'], 'text': unit['text']}}
                identity = {'kind': 'support', 'doc_id': q['doc_id'], 'question_id': q['question_id'], 'item': item}
                expected.append({'id': 'support:' + client.object_hash(identity), 'doc_id': q['doc_id'],
                                 'question_id': q['question_id'], 'unit_id': unit['unit_id'], 'item': item})
    require(prepared['support_tasks'] == expected, 'support tasks differ from complete query/candidate projection')
    require(len({t['id'] for t in expected}) == len(expected), 'duplicate support task identity')
    return len(queries), len(expected)


def build_budget(prepared, snapshot):
    """Pure plan calculation; the reused helpers are not execution functions."""
    questions, pairs = validate_prepared(prepared)
    with qwen_profile():
        validate_snapshot(snapshot)
        batches = stage.freeze_batches(prepared)
        jobs = []
        totals = {backend: {'requests': 0, 'judgments': 0, 'canonical_payload_bytes': 0,
                           'input_allowance': 0, 'output_allowance': 0, 'reserved_usd': Decimal(0)}
                  for backend in ('jev', 'general')}
        for index, batch in enumerate(batches):
            for backend in (('jev', 'general') if index % 2 == 0 else ('general', 'jev')):
                payload = client.make_payload(batch['tasks'], 'support', backend)
                reserved, tokens_in, tokens_out = client.reservation(payload, backend)
                size = len(client.canonical_bytes(payload))
                job = {'ordinal': len(jobs) + 1, 'batch_id': batch['id'], 'backend': backend,
                       'task_ids': [t['id'] for t in batch['tasks']], 'payload': payload,
                       'payload_sha256': client.object_hash(payload), 'canonical_payload_bytes': size,
                       'input_allowance': tokens_in, 'output_allowance': tokens_out, 'reserved_usd': str(reserved)}
                jobs.append(job)
                value = totals[backend]
                value['requests'] += 1; value['judgments'] += len(batch['tasks'])
                value['canonical_payload_bytes'] += size; value['input_allowance'] += tokens_in
                value['output_allowance'] += tokens_out; value['reserved_usd'] += reserved
        require(all(v['requests'] == len(batches) and v['judgments'] == pairs for v in totals.values()),
                'support schedule coverage differs')
        # Every future generation payload must separately satisfy this cap.
        allowance = client.BYTE_CAP + 2048
        generation_per = max(Decimal('0.001'),
            (Decimal(allowance) * Decimal(answers.PRICES['prompt'])
             + Decimal(answers.MAX_OUTPUT) * Decimal(answers.PRICES['completion'])) / 1000000 * Decimal('1.5'))
        require(generation_per == Decimal(snapshot['reservation_contract']['upper_reservation_per_unique_generation_usd']),
                'generation bound differs from provider snapshot')
        support_sum = sum((v['reserved_usd'] for v in totals.values()), Decimal(0))
        generation_upper = generation_per * len(METHODS) * questions
        for value in totals.values():
            value['reserved_usd'] = str(value['reserved_usd'])
        public = {'schema': SCHEMA, 'status': 'offline_budget_plan_not_paid_admitted',
                  'Q_questions': questions, 'C_support_query_unit_pairs': pairs, 'B_common_support_batches': len(batches),
                  'support': totals, 'support_total_reserved_usd': str(support_sum),
                  'generation_upper_bound': {'request_count': len(METHODS) * questions,
                      'per_request_reserved_usd': str(generation_per), 'reserved_usd': str(generation_upper),
                      'maximum_input_allowance_per_request': allowance, 'maximum_output_allowance_per_request': 512,
                      'actual_payloads_constructed': False, 'unique_request_count_known': False,
                      'all_actual_payloads_fit_byte_cap_verified': False, 'deduplication_discount_applied': False},
                  'support_plus_conditional_generation_upper_usd': str(support_sum + generation_upper),
                  'specification': deepcopy(SPEC), 'limits': list(LIMITS), 'api_calls': 0,
                  'key_read': False, 'references_read': False, 'quality_metrics_computed': False,
                  'paid_execution_admitted': False}
        private = {'schema': SCHEMA, 'specification': deepcopy(SPEC), 'models': deepcopy(client.MODELS),
                   'response_model_allowlists': {k: sorted(v) for k, v in client.RESPONSE_MODELS.items()},
                   'prompt_version': client.PROMPT_VERSION, 'batches': batches, 'jobs': jobs,
                   'public_aggregate': public}
    return private, public


def load_candidates(run_directory, audit_path):
    """Read saved gold-free containers; inherit the already completed sample audit."""
    run = Path(run_directory).resolve()
    require({p.name for p in run.iterdir()} == RUN_FILES, 'candidate run inventory incomplete')
    bindings = {str((run / n).resolve()): digest(run / n) for n in RUN_FILES}
    audit = read_bound(audit_path, bindings)
    summary = read_bound(run / 'summary.json', bindings)
    prepared = read_bound(run / 'prepared.json', bindings)
    public = read_bound(run / 'public_aggregate.json', bindings)
    require(audit['schema'] == CANDIDATE_SCHEMA and audit['status'] == 'verified_complete_with_sample'
            and audit['gold_read'] is False and audit['api_calls'] == 0, 'complete candidate audit required')
    require(summary['schema'] == CANDIDATE_SCHEMA and summary['status'] == 'completed_candidates'
            and summary['gold_read'] is False and summary['api_calls'] == 0, 'candidate run incomplete')
    require(audit['run_sha256'] == {str((run / n).resolve()): bindings[str((run / n).resolve())] for n in RUN_FILES},
            'candidate audit does not bind current complete run')
    require(summary['output_sha256'] == {n: bindings[str((run / n).resolve())] for n in RUN_FILES - {'summary.json'}},
            'candidate run seal differs')
    parent = summary['plan_sha256']
    directories = {Path(p).resolve().parent for p in parent}
    require(len(directories) == 1 and {Path(p).name for p in parent} == PLAN_FILES, 'candidate plan inventory differs')
    plan_dir = directories.pop()
    require({p.name for p in plan_dir.iterdir()} == PLAN_FILES, 'candidate plan directory incomplete')
    for path, expected in parent.items():
        bind_hash(bindings, path, expected)
    verify(parent)
    plan = read_bound(plan_dir / 'plan.json', bindings)
    seal = read_bound(plan_dir / 'seal.json', bindings)
    require(seal == {n: parent[str((plan_dir / n).resolve())] for n in PLAN_FILES - {'seal.json'}}, 'candidate plan seal differs')
    require(plan['schema'] == CANDIDATE_SCHEMA and plan['status'] == 'prepared_no_model_inference'
            and Path(plan['run_output']).resolve() == run and plan['gold_read'] is False and plan['api_calls'] == 0,
            'candidate plan contract differs')
    require(audit['plan_sha256'] == parent and audit['source_sha256'] == summary['input_sha256'] == plan['input_sha256'],
            'candidate source lineage differs')
    require(public['status'] == 'completed_candidates' and public['counts'] == plan['counts'] == audit['counts']
            and public['gold_read'] is False and public['quality_metrics_computed'] is False
            and public['paid_execution_admitted'] is False and public['api_calls'] == 0, 'candidate counts/flags differ')
    require(plan['counts']['documents'] == len(prepared['documents']) == 248, 'full 248-document cohort required')
    q, c = validate_prepared(prepared)
    require(q == plan['counts']['questions'] and c == public['support_pairs'], 'candidate Q/C counts differ')
    for key, value in {'dense_seeds': 8, 'candidate_cap': 16, 'max_selected_units': 3, 'evidence_budget': 1024,
                       'truncation': False}.items():
        require(plan['config'][key] == value, 'candidate selection contract differs')
    # Previous source audit remains an inherited commitment; do not reload
    # exported questions, references, tokenizer or multi-gigabyte model weights.
    verify(bindings)
    verify_candidate_inventories(run, bindings)
    return prepared, bindings, deepcopy(plan['input_sha256'])


def plan_budget(candidate_run, candidate_audit, output):
    output = Path(output).resolve()
    require(output != ART.resolve() and output.is_relative_to(ART.resolve()) and not output.exists(), 'new ignored output directory required')
    for path in (Path(candidate_run).resolve(), Path(candidate_audit).resolve()):
        require(not (output == path or output.is_relative_to(path) or path.is_relative_to(output)), 'output overlaps input')
    prepared, bindings, inherited = load_candidates(candidate_run, candidate_audit)
    require(digest(SNAPSHOT) == SNAPSHOT_SHA256, 'frozen provider snapshot changed')
    snapshot = read_bound(SNAPSHOT, bindings)
    for relative, expected in snapshot['old_contract_comparison']['source_sha256'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == expected, 'provider-profile source changed')
        bind_hash(bindings, path, expected)
    for path in (Path(__file__), Path(stage.__file__), Path(client.__file__), Path(answers.__file__),
                 ROOT / 'tests/research/test_qasper_confirmation_budget.py',
                 ROOT / 'docs/research/CONFIRMATION_BUDGET_PROTOCOL_20260927.md'):
        bind_hash(bindings, path, digest(path))
    private, public = build_budget(prepared, snapshot)
    private['source_sha256'] = bindings
    private['inherited_candidate_source_commitments_not_reopened'] = inherited
    public['provider_snapshot_sha256'] = SNAPSHOT_SHA256
    public['implementation_sha256'] = digest(__file__)
    verify(bindings)
    verify_candidate_inventories(candidate_run, bindings)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('private_plan.json', private), ('public_aggregate.json', public),
                        ('source_binding.json', {'schema': SCHEMA, 'input_sha256': bindings,
                         'inherited_candidate_source_commitments_not_reopened': inherited})]:
        with (output / name).open('xb') as stream:
            stream.write(canonical(value) + b'\n')
    # Keep failed construction unsealed. A new source/inventory check here
    # cannot reassign any commitment captured before calculation.
    verify(bindings)
    verify_candidate_inventories(candidate_run, bindings)
    with (output / 'seal.json').open('xb') as stream:
        stream.write(canonical({n: digest(output / n) for n in OUTPUT_FILES - {'seal.json'}}) + b'\n')
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--candidate-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        public = plan_budget(args.candidate_run, args.candidate_audit, args.output)
    except Exception as exc:
        print(json.dumps({'status': 'planning_failed', 'error_type': type(exc).__name__, 'api_calls': 0}))
        raise SystemExit(1) from None
    print(json.dumps(public, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
