"""Bounded synthetic feasibility check; not a selector or a quality benchmark."""
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
            raise RuntimeError(f'Backend prohibited: {fullname}')
        return None


def deny_external(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
        raise RuntimeError(f'External operation prohibited: {event}')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def partitions(atoms):
    mandatory = {i for i in range(len(atoms) - 1) if atoms[i][1] != atoms[i + 1][0]}
    for cuts in range(1 << (len(atoms) - 1)):
        if not all(cuts & (1 << i) for i in mandatory):
            continue
        blocks, first = [], 0
        for last in range(1, len(atoms) + 1):
            if last == len(atoms) or cuts & (1 << (last - 1)):
                blocks.append(list(range(first, last)))
                first = last
        yield cuts, blocks


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
    sys.path[:0] = [str(ROOT), str(ROOT / 'docs/research')]
    from probe_candidate_granularity import merge_spans
    from SLAC.llm.io.schemas import EvidenceItem
    from SLAC.llm.service.renderers import render_evidence_block

    fixture = {
        'cases': [
            {'name': 'contiguous', 'source': 'a b c d e',
             'atoms': [[0, 2], [2, 4], [4, 6], [6, 8], [8, 9]]},
            {'name': 'gapped', 'source': 'a b |gap| c d e',
             'atoms': [[0, 2], [2, 4], [10, 12], [12, 14], [14, 15]]},
        ],
        'byte_budgets': [0, 512, 4096], 'run_caps': [1, 2],
        'item_cap_counterexample': {'source': 'ab', 'atoms': [[0, 1], [1, 2]], 'item_cap': 1},
    }
    names = ['docs/research/probe_boundary_identifiability.py',
             'docs/research/BOUNDARY_IDENTIFIABILITY_PROTOCOL_20261005.md',
             'docs/research/probe_candidate_granularity.py',
             'SLAC/llm/io/schemas.py', 'SLAC/llm/service/renderers.py']
    write(output / 'fixture.json', fixture)
    freeze = {'fixture_sha256': sha((output / 'fixture.json').read_bytes()),
              'source_sha256': {name: sha((ROOT / name).read_bytes()) for name in names}}
    write(output / 'freeze.json', freeze)

    def render(case, spans):
        runs = merge_spans(spans)
        text = render_evidence_block([
            EvidenceItem(chunk_id=f'span-{a:05d}-{b:05d}', doc_id=f"synthetic-{case['name']}",
                         query_id='fixed-query', passage_text=case['source'][a:b])
            for a, b in runs], preserve_source_text=True)
        return runs, text

    cases = []
    for case in fixture['cases']:
        atoms = case['atoms']
        outputs = {}
        for mask in range(1 << len(atoms)):
            spans = [span for i, span in enumerate(atoms) if mask & (1 << i)]
            runs, text = render(case, spans)
            coverage = {x for a, b in spans for x in range(a, b)}
            assert coverage == {x for a, b in runs for x in range(a, b)}
            outputs[mask] = {'runs': runs, 'rendered': text, 'bytes': len(text.encode('utf-8')),
                             'sha256': sha(text.encode('utf-8'))}
        families, render_checks = {}, 0
        for cuts, blocks in partitions(atoms):
            masks = set()
            for subset in range(1 << len(blocks)):
                selected = [block for i, block in enumerate(blocks) if subset & (1 << i)]
                mask = sum(1 << j for block in selected for j in block)
                spans = [[atoms[block[0]][0], atoms[block[-1]][1]] for block in selected]
                runs, text = render(case, spans)
                assert runs == outputs[mask]['runs'] and text == outputs[mask]['rendered']
                masks.add(mask)
                render_checks += 1
            families[cuts] = masks
        all_atoms_cuts = (1 << (len(atoms) - 1)) - 1
        assert families[all_atoms_cuts] == set(outputs)
        gated_rows, subset_checks, refinement_pairs = [], 0, []
        for coarse, coarse_masks in families.items():
            for fine, fine_masks in families.items():
                if coarse & fine != coarse:
                    continue
                assert coarse_masks <= fine_masks
                refinement_pairs.append([coarse, fine])
        for run_cap in fixture['run_caps']:
            for budget in fixture['byte_budgets']:
                feasible = {mask for mask, row in outputs.items()
                            if len(row['runs']) <= run_cap and row['bytes'] <= budget}
                by_partition = {cuts: masks & feasible for cuts, masks in families.items()}
                for coarse, fine in refinement_pairs:
                    assert by_partition[coarse] <= by_partition[fine]
                    subset_checks += 1
                assert set().union(*by_partition.values()) == by_partition[all_atoms_cuts]
                gated_rows.append({'run_cap': run_cap, 'byte_budget': budget,
                                   'all_atom_output_count': len(by_partition[all_atoms_cuts]),
                                   'counts_by_cuts': {str(cuts): len(masks) for cuts, masks in by_partition.items()}})
        cases.append({'name': case['name'], 'partitions': len(families),
                      'block_selection_render_checks': render_checks,
                      'refinement_pairs_including_identity': refinement_pairs,
                      'gated_inclusion_checks': subset_checks, 'gate_rows': gated_rows,
                      'output_by_atom_mask': {str(mask): row for mask, row in outputs.items()},
                      'masks_by_partition_cuts': {str(cuts): sorted(masks) for cuts, masks in families.items()}})
    counter_case = {'name': 'item-cap', 'source': 'ab'}
    coarse_runs, coarse_text = render(counter_case, [[0, 2]])
    fine_outputs = [render(counter_case, spans)[1] for spans in ([], [[0, 1]], [[1, 2]])]
    assert len(coarse_runs) == 1 and coarse_text not in fine_outputs
    report = {'schema': 'slac-boundary-identifiability-synthetic-v1', 'status': 'passed',
              'fixture_sha256': freeze['fixture_sha256'],
              'freeze_sha256': sha((output / 'freeze.json').read_bytes()),
              'source_sha256': freeze['source_sha256'], 'cases': cases,
              'item_cap_counterexample': {'item_cap': 1, 'coarse_complete_ab_feasible': True,
                                          'fine_complete_ab_feasible': False,
                                          'coarse_rendered': coarse_text, 'fine_rendered': fine_outputs},
              'resources': {'api_calls': 0, 'model_loads': 0, 'tokenizer_loads': 0,
                            'training_steps': 0, 'natural_records': 0, 'scorer_calls': 0},
              'limits': ['Finite synthetic validation accompanies, but does not replace, the general constructive proof.',
                         'Cost in this check is exact UTF-8 bytes, not tokens or billing.',
                         'Feasible output families do not imply equal algorithmic outputs, ranking, scoring cost or utility.',
                         'Canonical source-span renderer only; actual production metadata may depend on partition.',
                         'Atom endpoints and allowed source domain are fixed; no new within-atom cut or domain expansion.']}
    assert not BLOCKED.intersection(sys.modules)
    write(output / 'report.json', report)
    print(json.dumps({'status': 'passed', 'counts': [
        {key: row[key] for key in ('name', 'partitions', 'block_selection_render_checks', 'gated_inclusion_checks')}
        for row in cases], 'item_cap_counterexample': 'verified',
        'report_sha256': sha((output / 'report.json').read_bytes())}))


if __name__ == '__main__':
    main()
