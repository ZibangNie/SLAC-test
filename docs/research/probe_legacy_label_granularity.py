"""Inspect exactly legacy-dev ordinals 0/1 and their two named source documents.

Replays only the archived generator's pure atomizer and boundary helper. Never
executes its dataset builder, noise sampler, tokenizer, model or network code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / 'artifacts/research-foundation/probe-02-final/selected_dev.jsonl'
SELECTED_SHA = 'd99155a866907a69d1c4617c5b3cbe3d2d3bdc2297783587df305710afc45b78'
SOURCE_ROOT = ROOT.parent / 'SLAC-test/SLAC/refiner/data/real_dataset/raw_sources_flat'
SOURCE_NAMES = ('dev/llm_structured/A/A_dv_000004.json',
                'dev/llm_structured/A/A_dv_000005.json')
GENERATOR = ROOT / 'SLAC/refiner/data_backup_20260310_185454/build_refiner_from_real_dataset.py'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(pred: list[int], gold: list[int]) -> float:
    tp = sum(p == g == 1 for p, g in zip(pred, gold))
    denominator = sum(pred) + sum(gold)
    return 2 * tp / denominator if denominator else 0.0


def load_pure_helpers():
    # This archived module has only standard-library imports and a main guard.
    # It is bound below by SHA; invoking main/build_sample is outside this probe.
    spec = importlib.util.spec_from_file_location('slac_legacy_atomizer_probe', GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run() -> tuple[dict, dict]:
    assert sha(SELECTED) == SELECTED_SHA
    with SELECTED.open(encoding='utf-8') as file:
        rows = [json.loads(next(file)), json.loads(next(file))]
    helpers = load_pure_helpers()
    outputs, bindings = [], []
    for ordinal, (row, name) in enumerate(zip(rows, SOURCE_NAMES)):
        source = (SOURCE_ROOT / name).resolve()
        assert source.is_relative_to(SOURCE_ROOT.resolve())
        assert Path(row['source_path']).resolve() == source
        assert source.stat().st_size < 50000
        raw = json.loads(source.read_bytes())
        units, spans, atoms = row['chunk0_units'], row['unit2atom_span'], row['atoms']
        n, u = len(atoms), len(units)
        assert len(spans) == u == len(raw['units'])
        raw_texts = [unit['text'] for unit in raw['units']]
        seed_texts = [unit['text'] for unit in units]
        assert raw_texts == seed_texts
        for field in ('type', 'level'):
            assert [unit[field] for unit in raw['units']] == [unit[field] for unit in units]
        assert [str(unit['unit_id']) for unit in raw['units']] == [str(unit['unit_id']) for unit in units]
        assert [unit.get('parent_id') for unit in raw['units']] == [unit.get('meta', {}).get('parent_id') for unit in units]
        cursor, rebuilt_spans, rebuilt_texts = 0, [], []
        for unit in units:
            pieces = helpers.atomize_text(unit['text'], language=row['language'], **row['meta']['atomizer'])
            assert pieces
            rebuilt_spans.append({'unit_id': unit['unit_id'], 's': cursor, 'e': cursor + len(pieces)})
            cursor += len(pieces)
            rebuilt_texts.extend(pieces)
        assert rebuilt_texts == [atom['text'] for atom in atoms]
        assert rebuilt_spans == spans
        assert cursor == n
        assert spans[0]['s'] == 0 and spans[-1]['e'] == n
        assert all(a['e'] == b['s'] for a, b in zip(spans, spans[1:]))
        expected = [0] * (n - 1)
        for span in spans[:-1]:
            assert span['e'] > span['s']
            expected[span['e'] - 1] = 1
        archived = helpers.boundaries_from_spans(spans, n)
        gold = row['b_gold']
        assert expected == archived == gold
        assert sum(gold) == u - 1
        assert all(type(v) is int and v in (0, 1) for v in gold + row['b0'])
        atom_owners = [str(atom['meta']['chunk0_unit_id']) for atom in atoms]
        owner_change = [int(a != b) for a, b in zip(atom_owners, atom_owners[1:])]
        assert owner_change == gold
        lengths = [span['e'] - span['s'] for span in spans]
        b0_diff = sum(a != b for a, b in zip(row['b0'], gold))
        outputs.append({
            'local_dev_ordinal': ordinal, 'atoms': n, 'source_units': u,
            'gaps': n - 1, 'gold_positive_gaps': sum(gold),
            'gold_density': sum(gold) / (n - 1),
            'atoms_per_source_unit_histogram': dict(sorted(Counter(lengths).items())),
            'single_atom_units': lengths.count(1),
            'gold_chunks_below_declared_min_atoms': sum(length < row['meta']['length_constraints']['min_chunk_atoms'] for length in lengths),
            'declared_length_constraints': row['meta']['length_constraints'],
            'saved_atomizer_config': row['meta']['atomizer'],
            'source_units_match_saved_texts_types_levels_ids_parents': True,
            'archived_atomizer_reproduces_all_saved_atom_texts_and_spans': True,
            'gold_exactly_unit_end_boundaries': True,
            'gold_exactly_adjacent_atom_owner_changes': True,
            'b0_differs_from_gold_at_gaps': b0_diff,
            'b0_macro_component_f1': f1(row['b0'], gold),
            'all_boundary_macro_component_f1': f1([1] * (n - 1), gold),
            'unit_end_reference_reconstruction_f1': f1(expected, gold),
            'unit_end_reconstruction_is_label_derivation_not_learned_quality': True,
            'projection_fix_count': len(row['meta']['projection_fix']),
            'stored_lineage_flags': row['meta']['label_repair']['flags'],
            'selected_source_family': row['source_family'],
            'selected_source_domain': row['source_domain'],
            'selected_orig_split': row['orig_split'],
            'raw_source_declared_split': raw.get('split'),
            'raw_source_top_level_fields': sorted(raw),
            'raw_source_has_separate_boundary_field': any(key in raw for key in ('b_gold', 'gold_boundaries', 'boundaries')),
        })
        bindings.append({'local_dev_ordinal': ordinal,
                         'source_file_sha256': sha(source), 'source_file_bytes': source.stat().st_size,
                         'selected_row_sha256': hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                         'source_path_private': str(source), 'doc_id_private': row['doc_id'],
                         'unit_ids_private': [unit['unit_id'] for unit in units]})
    public_bindings = [{k: v for k, v in item.items() if not k.endswith('_private')} for item in bindings]
    report = {
        'schema': 'slac-legacy-label-granularity-two-doc-probe-v1',
        'status': 'completed_fixed_two_document_mechanical_reconstruction',
        'selection': 'First two rows of the already exposed selected_dev.jsonl; no reselection or expansion.',
        'scope': {'unique_existing_documents': 2, 'selected_rows_parsed': 2, 'named_flat_source_files': 2,
                  'upstream_jsonl_rows_parsed': 0, 'full_dataset_scans': 0, 'api_calls': 0,
                  'model_calls': 0, 'training_updates': 0, 'tokenizer_calls': 0,
                  'noise_sampler_calls': 0, 'semantic_reference_review': False},
        'bindings': {'selected_dev_sha256': SELECTED_SHA,
                     'archived_generator_sha256': sha(GENERATOR),
                     'probe_sha256': sha(Path(__file__)), 'documents': public_bindings},
        'documents': outputs,
        'interpretation': 'On these two rows, b_gold mechanically reconstructs the existing source-unit partition. The archived pure atomizer reproduces every saved atom and span. This does not certify historical code execution, human/LLM semantic quality, or downstream usefulness.',
        'limits': ['Two law documents are not representative of the entire eight-row dev set or corpus.',
                   'Current helper reproduction and current source byte matching do not authenticate historical execution or source authorship.',
                   'Unit-end or owner-change perfect reconstruction uses clean structural information that determines the weak target; it is not a learned or independent semantic score.',
                   'The historical b0 noise sampler was inspected but not rerun; stored noise provenance remains unverified.',
                   'No tokenizer ran: the archived atomizer uses an estimate despite stored tokenizer-name metadata.',
                   'No rows, labels, constraints, weights or training recipes were changed.'],
    }
    return report, {'bindings': bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result, private = run()
    for name, value in [('result.json', result), ('private_bindings.json', private)]:
        with (args.output_dir / name).open('x', encoding='utf-8', newline='\n') as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2)
            file.write('\n')
    print(json.dumps({'status': result['status'], 'documents': result['documents']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
