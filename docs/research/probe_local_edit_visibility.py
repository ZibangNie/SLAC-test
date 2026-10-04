"""Fixed synthetic dependency witness; no models, ranking, or natural data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BLOCKED = {'torch', 'transformers', 'sentence_transformers', 'numpy', 'faiss', 'requests', 'httpx'}


class DenyBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise RuntimeError(f'Backend prohibited in synthetic witness: {fullname}')
        return None


def deny_external(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
        raise RuntimeError(f'External operation prohibited: {event}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def write_rows(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    base = (ROOT / 'artifacts/research-foundation').resolve()
    if not output.is_relative_to(base) or output == base:
        raise ValueError('Output must be a new subdirectory of research artifacts')
    output.mkdir(parents=True, exist_ok=False)
    sys.meta_path.insert(0, DenyBackends())
    sys.addaudithook(deny_external)
    sys.path.insert(0, str(ROOT))
    from SLAC.refiner.pipeline.assemble.export_refined_chunks import build_refined_chunks, build_leaf_records
    from SLAC.retrieval.dataio.readers import load_chunk_records, load_leaf_records
    from SLAC.retrieval.index.build_lookup_tables import enrich_all_records
    from SLAC.retrieval.index.build_leaf_dense import compose_leaf_retrieval_text
    from SLAC.retrieval.index.build_chunk_dense import compose_chunk_retrieval_text
    from SLAC.llm.io.schemas import EvidenceItem
    from SLAC.llm.service.renderers import render_evidence_block

    record = {
        'doc_id': 'synthetic-local-edit',
        'atoms': ['Amber hardware is available.', 'Blue services are available.',
                  'Cobalt services are available.', 'Delta hardware is available.'],
        'b0': [0, 0, 1],
        'chunk0_units': [
            {'unit_id': 0, 'path': ['Alpha'], 'depth': 1},
            {'unit_id': 1, 'path': ['Beta'], 'depth': 1},
            {'unit_id': 2, 'path': ['Gamma'], 'depth': 1},
        ],
        'unit2atom_span': [
            {'unit_id': 0, 'start_atom': 0, 'end_atom': 1},
            {'unit_id': 1, 'start_atom': 1, 'end_atom': 3},
            {'unit_id': 2, 'start_atom': 3, 'end_atom': 4},
        ],
    }
    partitions = {'baseline': [2], 'split': [0, 2], 'merge': []}
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == '.py':
                sources[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    protocol = ROOT / 'docs/research/LOCAL_EDIT_VISIBILITY_PROTOCOL_20261005.md'
    sources[protocol.relative_to(ROOT).as_posix()] = digest(protocol.read_bytes())
    fixture = {'record': record, 'partitions': partitions}
    write(output / 'fixture.json', fixture)
    freeze = {'fixture_sha256': digest((output / 'fixture.json').read_bytes()),
              'source_sha256': dict(sorted(sources.items()))}
    write(output / 'freeze.json', freeze)

    arms = {}
    objects = {}
    for name, gaps in partitions.items():
        candidate = {'candidate_id': name, 'candidate_type': 'synthetic',
                     'prediction': {'b_pred_sparse': gaps}}
        raw_chunks = build_refined_chunks(record, candidate)
        raw_leaves = build_leaf_records(raw_chunks, record['atoms'])
        chunk_path, leaf_path = output / f'{name}.chunks.jsonl', output / f'{name}.leaves.jsonl'
        write_rows(chunk_path, raw_chunks)
        write_rows(leaf_path, raw_leaves)
        chunks, leaves = enrich_all_records(load_chunk_records(chunk_path), load_leaf_records(leaf_path))
        owners = {chunk.chunk_id: chunk for chunk in chunks}
        evidence = [EvidenceItem(chunk_id=c.chunk_id, doc_id=c.doc_id,
                                 passage_text=c.text, path_text=c.path_text) for c in chunks]
        rendered = render_evidence_block(evidence)
        objects[name] = (chunks, leaves, evidence)
        arms[name] = {
            'chunks': [{'id': c.chunk_id, 'span': [c.atom_start, c.atom_end], 'text': c.text,
                        'path_text': c.path_text, 'anchor_text': c.anchor_text,
                        'encoder_text': compose_chunk_retrieval_text(c)} for c in chunks],
            'leaves': [{'id': leaf.leaf_id, 'text': leaf.text, 'owner': leaf.owner_chunk_id,
                        'path_text': leaf.path_text, 'owner_anchor': leaf.owner_chunk_anchor,
                        'encoder_text': compose_leaf_retrieval_text(leaf, owners[leaf.owner_chunk_id])}
                       for leaf in leaves],
            'normalized_complete_body': ' '.join(c.text for c in chunks),
            'forced_all_chunk_evidence': rendered,
            'forced_all_chunk_evidence_sha256': digest(rendered.encode('utf-8')),
        }
    comparisons = []
    baseline = arms['baseline']
    for name in ('split', 'merge'):
        changed = arms[name]
        assert len(changed['leaves']) == len(baseline['leaves']) == 4
        counts = {field: sum(a[field] != b[field] for a, b in
                            zip(baseline['leaves'], changed['leaves'], strict=True))
                  for field in ('id', 'text', 'owner', 'path_text', 'owner_anchor', 'encoder_text')}
        comparisons.append({'contrast': f'baseline_to_{name}', 'leaf_fields_changed': counts,
                            'normalized_complete_body_equal': baseline['normalized_complete_body'] == changed['normalized_complete_body'],
                            'forced_all_chunk_evidence_equal': baseline['forced_all_chunk_evidence'] == changed['forced_all_chunk_evidence']})
    tail_items = []
    for name in ('baseline', 'split'):
        chunks, _, evidence = objects[name]
        tail_items.append(next(ev for chunk, ev in zip(chunks, evidence, strict=True)
                               if (chunk.atom_start, chunk.atom_end) == (3, 4)))
    tail_rendered = [render_evidence_block([ev]) for ev in tail_items]
    report = {'schema': 'slac-local-edit-visibility-synthetic-v1', 'status': 'completed',
              'fixture_sha256': freeze['fixture_sha256'],
              'freeze_sha256': digest((output / 'freeze.json').read_bytes()),
              'source_sha256': freeze['source_sha256'], 'arms': arms, 'comparisons': comparisons,
              'unchanged_tail': {'body_equal': tail_items[0].passage_text == tail_items[1].passage_text,
                                 'ids_equal': tail_items[0].chunk_id == tail_items[1].chunk_id,
                                 'rendered_equal': tail_rendered[0] == tail_rendered[1],
                                 'rendered': tail_rendered},
              'resources': {'api_calls': 0, 'model_loads': 0, 'tokenizer_loads': 0,
                            'training_steps': 0, 'natural_records': 0, 'ranks_or_vectors_computed': 0},
              'limits': ['One designed synthetic document, not natural evidence or performance.',
                         'Default normalized export only; no source_document mode or full HTTP request audit.',
                         'All-chunk evidence is forced for renderer diagnosis, not produced by retrieval or budget selection.',
                         'Changed encoder input does not guarantee changed vectors, ranks or answers.']}
    assert not BLOCKED.intersection(sys.modules)
    write(output / 'report.json', report)
    print(json.dumps({'status': 'completed', 'comparisons': comparisons,
                      'unchanged_tail': {k: v for k, v in report['unchanged_tail'].items() if k != 'rendered'},
                      'report_sha256': digest((output / 'report.json').read_bytes())}))


if __name__ == '__main__':
    main()
