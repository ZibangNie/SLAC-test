"""Fixed two-question character-coverage feasibility; no model or API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'artifacts/research-foundation/offline-20261005/natural-granularity-coverage-01'
BASE = ROOT / 'artifacts/research-foundation/offline-20261004'
INPUTS = {
    'bounded': (BASE / 'refiner-granularity-probe-01/bounded_inputs.json', 'e860ca0d755ce4f1ce24b73f64e89b4ad6e9e1eb1da0659efa333eb8fc7d603c'),
    'exports': (BASE / 'refiner-document-source-export-01/run-01/document_exports.json', 'e28538bf86ed10b452257c887067c21d5fdd1588baef3d964ac24229de87df87'),
    'references': (PHASE / 'selected_references.json', '721d092de114f29d17bbed4cd29d61c0399f90bbacc80271ff05b98f95fa3f0f'),
}
TOKENIZER = ROOT.parent / 'SLAC-test/SLAC/refiner/slac_refiner/models/bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
TOKEN_HASHES = {
    'tokenizer.json': '4829dfefc91c9f9839cf1b554a99243c8911c43439668e2f974176b1925cd138',
    'tokenizer_config.json': '3e5e7d91646b2277098e245f3e73ba565de0e3c28b9e7ee7a9fcc3cd68ae4121',
    'special_tokens_map.json': '66715ae6a0dd4aff4fe228bcaaccac6e52c83fd0ba80992c2be7d0e43b362307',
    'sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'config.json': 'aef03cacaae68933fe96fc8b9a673601d30a8b20158f0684a0e51295920714a0',
}
SOURCE_NAMES = (
    'docs/research/NATURAL_GRANULARITY_COVERAGE_PROTOCOL_20261005.md',
    'docs/research/probe_natural_partition_coverage.py',
    'docs/research/partition_coverage.py', 'tests/research/test_partition_coverage.py',
    'SLAC/llm/service/renderers.py', 'SLAC/llm/io/schemas.py',
)
ARMS = ('raw_b0_source', 'rule_projected_source', 'all_atoms_source')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write('\n')


def identity(row):
    return tuple(row[k] for k in ('family_id', 'doc_id', 'question_id'))


def run(output):
    output = output.resolve()
    assert output.parent == PHASE.resolve() and not output.exists()
    blobs = {name: path.read_bytes() for name, (path, _) in INPUTS.items()}
    assert all(sha(blobs[name]) == expected for name, (_, expected) in INPUTS.items())
    assert all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items())
    sources = {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}
    freeze = json.loads((PHASE / 'pre_reference_freeze.json').read_bytes())
    assert freeze['protocol_sha256'] == sources[SOURCE_NAMES[0]]
    assert sources['docs/research/partition_coverage.py'] == '40887f61ad5235c76aa64f1db90a61c77773bcc5cdbb7d892781c68cfed63337'
    data = {name: json.loads(raw) for name, raw in blobs.items()}
    assert [r['ordinal'] for r in data['references']] == [1, 2]
    output.mkdir()
    plan = {'source_sha256': sources, 'input_sha256': {name: expected for name, (_, expected) in INPUTS.items()},
            'tokenizer_sha256': TOKEN_HASHES, 'ordinals': [1, 2], 'arms': ARMS,
            'max_chunks': 3, 'max_tokens': 1024, 'coverage_measured_before_this_seal': False,
            'budget_scope': 'Complete production evidence block; BGE proxy tokens only; excludes other messages and output.'}
    write(output / 'execution_plan.json', plan)
    def denied(*args, **kwargs):
        raise RuntimeError('network forbidden in source-coverage feasibility probe')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = denied
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    for name in ('USE_TORCH', 'USE_TF', 'USE_FLAX'):
        os.environ[name] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'docs/research'))
    from partition_coverage import minimal_cover, assess_cover
    from SLAC.llm.io.schemas import EvidenceItem
    from SLAC.llm.service.renderers import render_evidence_block, SOURCE_RENDERER_VERSION
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=False)
    count_cache = {}
    def count(text):
        if text not in count_cache:
            count_cache[text] = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        return count_cache[text]
    bounded = data['bounded']
    native_docs = {d['doc_id']: d for d in bounded['native_documents']}
    exports = {d['doc_id']: d for d in data['exports']}
    query_metadata = {identity(q): q for q in bounded['queries']}
    mappings = {(m['doc_id'], m['native_unit_id']): m for m in bounded['unit_mapping']}
    assert len(native_docs) == len(exports) == len(query_metadata) == 2 and len(mappings) == 83
    summaries, private_records, document_summaries = [], [], []
    for case in data['references']:
        ordinal, query, reference = case['ordinal'], case['query'], case['reference']
        assert identity(query) == identity(reference) and query_metadata[identity(query)] == query
        doc_id = query['doc_id']
        source_record, exported = native_docs[doc_id], exports[doc_id]
        source = source_record['native_document_text']
        assert exported['document_ordinal'] == ordinal and exported['view']['source_text'] == source
        assert sha(source.encode()) == source_record['native_document_text_sha256']
        assert source_record['source_id'] == reference['source_id']
        units = bounded['documents'][doc_id]
        text_to_units = {}
        for unit in units:
            mapping = mappings[doc_id, unit['unit_id']]
            a, b = mapping['raw_native_char_span']
            assert source[a:b] == unit['native_text'] and sha(source[a:b].encode()) == mapping['raw_native_text_sha256']
            text_to_units.setdefault(unit['native_text'], []).append((unit['unit_id'], [a, b]))
        partitions = {}
        for arm in ARMS:
            if arm == 'all_atoms_source':
                spans = exported['view']['atom_char_spans']
            else:
                chunks = exported['exports'][arm]['refined_chunks']
                spans = [c['source_char_span'] for c in chunks]
                assert all(c['text'] == source[a:b] for c, (a, b) in zip(chunks, spans, strict=True))
            minimal_cover(spans, [], len(source))
            assert ''.join(source[a:b] for a, b in spans) == source
            partitions[arm] = spans
        expected_counts = {1: (60, 41, 177), 2: (23, 15, 74)}[ordinal]
        assert tuple(len(partitions[a]) for a in ARMS) == expected_counts
        document_summaries.append({'ordinal': ordinal, 'source_chars': len(source),
            'native_units': len(units), 'partition_chunk_counts': {a: len(partitions[a]) for a in ARMS},
            'all_partitions_cover_complete_original_source': True})
        annotations = reference['answer_annotations']
        assert len(annotations) == 2
        for annotation_index, annotation in enumerate(annotations):
            answer = annotation.get('native_answer', annotation.get('answer', annotation))
            evidence = answer.get('evidence')
            unanswerable = answer.get('unanswerable')
            annotation_status, mapping_details, intervals = 'mapped', [], []
            if type(unanswerable) is not bool or not isinstance(evidence, list):
                annotation_status = 'invalid_reference'
            elif unanswerable:
                annotation_status = 'unanswerable_reference'
            elif not evidence:
                annotation_status = 'empty_reference'
            else:
                for item_index, text in enumerate(evidence):
                    if not isinstance(text, str) or not text.strip():
                        annotation_status = 'invalid_reference'
                        mapping_details.append({'item_index': item_index, 'kind': 'invalid'})
                        continue
                    matches = text_to_units.get(text, [])
                    kind = 'single_exact_native_unit' if len(matches) == 1 else 'unmatched' if not matches else 'ambiguous'
                    mapping_details.append({'item_index': item_index, 'text_sha256': sha(text.encode()),
                        'kind': kind, 'matched_unit_ids': [u for u, _ in matches],
                        'matched_spans': [span for _, span in matches]})
                    if len(matches) == 1:
                        intervals.append(matches[0][1])
                    elif annotation_status != 'invalid_reference':
                        annotation_status = 'mapping_unresolved'
            mapping_counts = dict(Counter(m['kind'] for m in mapping_details))
            for arm in ARMS:
                row = {'ordinal': ordinal, 'annotation_index': annotation_index, 'arm': arm,
                       'annotation_status': annotation_status, 'original_evidence_items': len(evidence) if isinstance(evidence, list) else None,
                       'unanswerable': unanswerable if type(unanswerable) is bool else None,
                       'mapping_kind_counts': mapping_counts, 'status': annotation_status,
                       'partition_chunks': len(partitions[arm]), 'required_chunk_count': None,
                       'reference_chars': None, 'required_source_chars': None,
                       'evidence_block_proxy_tokens': None, 'rendered_sha256': None}
                detail = {'doc_id': doc_id, 'question_id': query['question_id'],
                          'mapping_details': mapping_details, 'reference_intervals': intervals,
                          'required_indices': None, 'required_spans': None, 'rendered_text': None}
                if annotation_status == 'mapped':
                    coverage = minimal_cover(partitions[arm], intervals, len(source))
                    required = coverage['required_indices']
                    row.update(required_chunk_count=len(required), reference_chars=coverage['reference_chars'],
                               required_source_chars=coverage['required_source_chars'])
                    detail.update(reference_union=coverage['reference_union'], required_indices=required,
                                  required_spans=[partitions[arm][i] for i in required])
                    tokens = None
                    if len(required) <= 3:
                        items = [EvidenceItem(chunk_id=f'span-{a:05d}-{b:05d}', doc_id=doc_id,
                                   passage_text=source[a:b], query_id=query['question_id'],
                                   query_text=reference['question']) for a, b in detail['required_spans']]
                        rendered = render_evidence_block(items, preserve_source_text=True)
                        tokens = count(rendered)
                        row.update(evidence_block_proxy_tokens=tokens, rendered_sha256=sha(rendered.encode()))
                        detail['rendered_text'] = rendered
                    row['status'] = assess_cover(len(required), len(partitions[arm]), tokens)
                summaries.append(row)
                private_records.append({'summary': row, **detail})
    assert len(summaries) == 12 and len(document_summaries) == 2
    assert 'torch' not in sys.modules and 'tensorflow' not in sys.modules and 'jax' not in sys.modules
    assert sources == {name: sha((ROOT / name).read_bytes()) for name in SOURCE_NAMES}
    assert all(sha(path.read_bytes()) == expected for path, expected in INPUTS.values())
    assert all(sha((TOKENIZER / name).read_bytes()) == expected for name, expected in TOKEN_HASHES.items())
    write(output / 'records.json', private_records)
    report = {'schema': 'slac-natural-partition-coverage-v1', 'status': 'completed_fixed_two_questions',
        'documents': document_summaries, 'rows': summaries,
        'state_counts_by_arm': {arm: dict(Counter(r['status'] for r in summaries if r['arm'] == arm)) for arm in ARMS},
        'counts': {'questions': 2, 'annotations': 4, 'annotation_partition_records': 12,
                   'full_render_measurements': sum(r['evidence_block_proxy_tokens'] is not None for r in summaries),
                   'unique_render_strings_tokenized': len(count_cache)},
        'resources': {'api_calls': 0, 'model_calls': 0, 'training_updates': 0, 'atomizer_calls': 0,
                      'projector_calls': 0, 'new_unique_documents': 0, 'source_documents_reused': 2,
                      'reference_rows_decoded': 2, 'upstream_corpus_reads': 0},
        'renderer_version': SOURCE_RENDERER_VERSION, 'plan': plan,
        'runtime': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
        'private_records_sha256': sha((output / 'records.json').read_bytes()),
        'limits': ['Exposed two-question descriptive feasibility, not independent evaluation or publication novelty.',
                   'Uses reference labels and every chunk of the fixed document, not a deployable selector or the old candidate pool.',
                   'Character coverage is not semantic sufficiency or answer quality; no old JEV labels were transferred.',
                   'Complete evidence block BGE proxy count excludes other messages, provider framing and output; not billing or full-context admission.',
                   'Over-budget required sets with room for supersets remain unresolved without a monotonicity assumption.',
                   'No learned Refiner partition was evaluated.']}
    write(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PHASE / 'run-01')
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({'status': report['status'], 'rows': report['rows'], 'counts': report['counts']}, ensure_ascii=False))
