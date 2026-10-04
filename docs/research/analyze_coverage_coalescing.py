"""Post-hoc packaging diagnostic on the same twelve saved coverage records."""
from __future__ import annotations

import json
import os
import socket
import sys

from probe_natural_partition_coverage import ROOT, PHASE, INPUTS, TOKENIZER, TOKEN_HASHES, sha, write


def main():
    paths = {
        'records': (PHASE / 'run-01/records.json', 'd2d157b56126be28d7b95717190f862c6b5caed833dbebefa146c07ee11382f5'),
        'report': (PHASE / 'run-01/report.json', '1f80f58e765e7bb7054734ef559befb4e2685dcb3fa08f39ff698c5e8a8e680a'),
        'bounded': INPUTS['bounded'],
    }
    raw = {name: path.read_bytes() for name, (path, _) in paths.items()}
    assert all(sha(raw[name]) == expected for name, (_, expected) in paths.items())
    records = json.loads(raw['records'])
    baseline = json.loads(raw['report'])
    assert len(records) == 12 and baseline['private_records_sha256'] == sha(raw['records'])
    docs = {d['doc_id']: d['native_document_text'] for d in json.loads(raw['bounded'])['native_documents']}
    assert len(docs) == 2 and sum(map(len, docs.values())) == 33109
    source_names = ('docs/research/analyze_coverage_coalescing.py',
                    'docs/research/probe_natural_partition_coverage.py',
                    'SLAC/llm/service/renderers.py', 'SLAC/llm/io/schemas.py')
    source_hashes = {name: sha((ROOT / name).read_bytes()) for name in source_names}
    for name in source_names[1:]:
        assert source_hashes[name] == baseline['plan']['source_sha256'][name]
    assert all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items())
    output = PHASE / 'coalescing-posthoc-01'
    output.mkdir()
    plan = {
        'design': 'posthoc_all_12_saved_records_no_selection_or_threshold_changes',
        'run_counts_inspected_before_this_plan': True,
        'operation': 'Merge touching selected source spans only; never add gap characters.',
        'max_output_runs': 3, 'max_evidence_block_proxy_tokens': 1024,
        'k_semantics_changed_from_retrieval_chunks_to_output_source_runs': True,
        'input_sha256': {name: expected for name, (_, expected) in paths.items()},
        'source_sha256': source_hashes, 'tokenizer_sha256': TOKEN_HASHES,
        'original_preregistered_result_unchanged': True,
    }
    write(output / 'plan.json', plan)
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in posthoc packaging diagnostic')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
        os.environ[name] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(ROOT))
    from SLAC.llm.io.schemas import EvidenceItem
    from SLAC.llm.service.renderers import render_evidence_block
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    def render(record, spans):
        return render_evidence_block([
            EvidenceItem(chunk_id=f'span-{a:05d}-{b:05d}', doc_id=record['doc_id'],
                         query_id=record['question_id'], passage_text=docs[record['doc_id']][a:b])
            for a, b in spans], preserve_source_text=True)
    cache = {}
    for record in records:
        if record['rendered_text'] is not None:
            text = render(record, record['required_spans'])
            assert text == record['rendered_text']
            assert sha(text.encode()) == record['summary']['rendered_sha256']
            tokens = record['summary']['evidence_block_proxy_tokens']
            assert text not in cache or cache[text] == tokens
            cache[text] = tokens
    baseline_texts = set(cache)
    initial_cache_size = len(cache)
    rows, private = [], []
    for record in records:
        summary, spans = record['summary'], record['required_spans']
        assert summary['annotation_status'] == 'mapped' and 0 < len(spans) <= 251
        runs = []
        for a, b in spans:
            assert type(a) is int and type(b) is int and 0 <= a < b <= len(docs[record['doc_id']])
            assert not runs or a >= runs[-1][1]
            if runs and a == runs[-1][1]:
                runs[-1][1] = b
            else:
                runs.append([a, b])
        # Independent character membership checks only on these saved selections.
        before = {i for a, b in spans for i in range(a, b)}
        after = {i for a, b in runs for i in range(a, b)}
        reference = {i for a, b in record['reference_intervals'] for i in range(a, b)}
        assert before == after and reference <= after
        assert len(after) == summary['required_source_chars']
        tokens, rendered_hash, rendered, origin = None, None, None, None
        if len(runs) <= 3:
            rendered = render(record, runs)
            origin = ('exact_string_baseline_cache' if rendered in baseline_texts else
                      'exact_string_posthoc_cache' if rendered in cache else 'new_exact_string_measurement')
            if rendered not in cache:
                cache[rendered] = len(tokenizer.encode(rendered, add_special_tokens=True, truncation=False))
            tokens = cache[rendered]
            rendered_hash = sha(rendered.encode())
        row = {key: summary[key] for key in ('ordinal', 'annotation_index', 'arm')}
        row.update(original_required_chunks=len(spans), output_runs=len(runs),
                   selected_source_chars=len(after), source_character_set_unchanged=True,
                   evidence_block_proxy_tokens=tokens, rendered_sha256=rendered_hash,
                   count_origin=origin,
                   witness_satisfies_output_run_budget=tokens is not None and tokens <= 1024)
        rows.append(row)
        private.append({'summary': row, 'runs': runs, 'rendered_text': rendered})
    assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules and 'jax' not in sys.modules
    assert all(sha(path.read_bytes()) == expected for path, expected in paths.values())
    assert source_hashes == {name: sha((ROOT / name).read_bytes()) for name in source_names}
    assert all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items())
    write(output / 'records.json', private)
    report = {'schema': 'slac-coverage-coalescing-posthoc-v1', 'plan': plan, 'rows': rows,
              'resources': {'api_calls': 0, 'model_calls': 0, 'new_documents': 0,
                            'new_reference_rows': 0, 'source_documents_reused': 2,
                            'new_unique_render_strings_tokenized': len(cache) - initial_cache_size},
              'private_records_sha256': sha((output / 'records.json').read_bytes()),
              'limits': ['Reference-assisted post-hoc packaging comparison; not a deployable selector.',
                         'Changes what k counts, not the saved original result or original feasibility policy.',
                         'No JEV labels, semantic sufficiency, answer quality or novelty were evaluated.']}
    write(output / 'report.json', report)
    print(json.dumps({'rows': rows, 'resources': report['resources']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
