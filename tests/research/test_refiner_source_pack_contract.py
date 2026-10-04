"""Real preparation/packing to source requests, using authored text and toy bytes."""
import json

import pytest

from SLAC.refiner.pipeline.assemble.source_document_view import DocumentSourceView
from SLAC.refiner.pipeline.assemble.source_coverage import build_native_coverage_index
from SLAC.retrieval.dataio.readers import load_chunk_records
from SLAC.retrieval.dataio.source_records import resolve_refiner_source_selection
from SLAC.retrieval.decision.refiner_bridge import build_refiner_source_request
from SLAC.retrieval.index.build_lookup_tables import enrich_all_records
from SLAC.retrieval.pack.evidence_packer import pack_evidence
from SLAC.retrieval.schemas.records import RetrievalCandidate


@pytest.mark.parametrize('arm', ['standalone', 'plain_conditional'])
def test_real_pack_estimates_do_not_bypass_complete_source_request_budget(tmp_path, arm):
    atoms = ('Amber console has four ports.', 'Blue cabinet stays locked.')
    source_parts = (atoms[0] + '\r\n', atoms[1] + '  ')
    split = len(source_parts[0])
    source = ''.join(source_parts)
    view = DocumentSourceView(
        'authored', 'python-characters', source, atoms,
        ((0, split), (split, len(source))),
        (tuple((i, i + 1) for i in range(len(atoms[0]))),
         tuple((split + i, split + i + 1) for i in range(len(atoms[1])))),
    )
    index = build_native_coverage_index(view, [
        {'native_unit_id': 'left', 'source_span': [0, len(atoms[0])], 'source_text': atoms[0]},
        {'native_unit_id': 'right', 'source_span': [split, split + len(atoms[1])], 'source_text': atoms[1]},
    ])
    rows = []
    for i, text in enumerate(source_parts):
        rows.append({'doc_id': 'authored', 'chunk_id': f'chunk-{i}', 'chunk_index': i,
                     'atom_start': i, 'atom_end': i + 1, 'num_atoms': 1, 'text': text,
                     'source': 'refiner_source_view', 'text_mode': 'source_document',
                     'source_coordinate_system': view.coordinate_system,
                     'source_char_span': list(view.char_span(i, i + 1))})
    path = tmp_path / 'chunks.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    chunks = load_chunk_records(path, source_indexes={'authored': index})
    enrich_all_records(chunks, [])
    candidates = [RetrievalCandidate(c.chunk_id, c.doc_id, c.text, [], 0,
                                    token_est=1, best_chunk_score=2 - i)
                  for i, c in enumerate(chunks)]
    packed, summary = pack_evidence(candidates, {'pack': {'max_packed_items': 1,
                                                       'evidence_budget_tokens': 1}})
    assert [p.chunk_id for p in packed] == ['chunk-0']
    assert summary['used_tokens'] == 1
    assert packed[0].text == atoms[0] != source_parts[0]
    current, candidate, changed = resolve_refiner_source_selection(
        packed, candidates[1], {c.chunk_id: c for c in chunks}, {'authored': index},
        text_policy='reconstruct',
    )
    assert changed == ('chunk-0', 'chunk-1')
    assert current[0].unit.text == source_parts[0]
    assert candidate.unit.text == source_parts[1]

    # This counter deliberately counts UTF-8 bytes, not provider/BGE tokens.
    seen = []
    def toy_bytes(text):
        seen.append(text)
        return len(text.encode('utf-8'))
    config = dict(arm=arm, endpoint_id='offline-only', model_id='authored-model',
                  expected_response_model='authored-response', token_counter=toy_bytes,
                  counter_version='toy-utf8-bytes-v1')
    wrapped = build_refiner_source_request('Which devices are described?', current, candidate,
                                           max_tokens=1024, **config)
    expected = '[authored/chunk-1]\n' + source_parts[1]
    if arm == 'plain_conditional':
        expected = '[authored/chunk-0]\n' + source_parts[0] + '\n\n' + expected
    assert seen == [expected]
    assert wrapped.request.rendered_evidence == expected
    assert wrapped.request.evidence_tokens == len(expected.encode('utf-8'))
    assert wrapped.request.payload()['state']['candidate']['text'] == source_parts[1]
    with pytest.raises(ValueError, match='complete evidence exceeds max_tokens'):
        build_refiner_source_request('Which devices are described?', current, candidate,
                                      max_tokens=2, **config)
    assert seen == [expected, expected]
    assert summary['used_tokens'] == 1  # Legacy selection was neither mutated nor rerun.
