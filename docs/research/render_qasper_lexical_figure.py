"""Render a standalone development figure from public lexical aggregates only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    raw = Path(source).read_bytes()
    data = json.loads(raw)
    if (data.get('status') != 'completed' or data.get('question_count') != 77
            or data.get('family_count') != 24 or len(data.get('scopes', [])) != 2):
        raise ValueError('requires complete 77-question public lexical aggregate')
    scopes = data['scopes']
    if [item['scope'] for item in scopes] != ['given_document', 'corpus_32']:
        raise ValueError('scope inventory/order differs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metric = 'source_qualified_evidence_f1'
    dense, lexical, tokens = [], [], []
    for scope in scopes:
        methods = {item['method']: item['metrics'] for item in scope['methods']}
        if set(methods) != {'dense_cached', 'bm25'}:
            raise ValueError('method inventory differs')
        dense.append(methods['dense_cached'][metric]['question_weighted'])
        lexical.append(methods['bm25'][metric]['question_weighted'])
        tokens.append([methods[name]['actual_evidence_tokens']['question_weighted']
                       for name in ('dense_cached', 'bm25')])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': '#fbfbf8'})
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw={'width_ratios': [1.03, 1]})
    figure.patch.set_facecolor('#fbfbf8')
    for axis in axes:
        axis.set_facecolor('#fbfbf8')
    colors = {'dense': '#496c9b', 'bm25': '#bc7048', 'family': '#718672'}
    x = np.arange(2)
    for offset, values, color, label in [(-.18, dense, colors['dense'], 'Cached dense'),
                                        (.18, lexical, colors['bm25'], 'Fixed BM25')]:
        bars = axes[0].bar(x + offset, values, .32, color=color, label=label, zorder=3)
        axes[0].bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=5, fontsize=11)
    axes[0].set_xticks(x, ['Given paper', 'Query-only / 32 papers'])
    axes[0].set_ylim(0, .265)
    axes[0].set_ylabel('Source-qualified evidence F1')
    axes[0].set_title('Complete 77-question development comparison', loc='left', pad=18, fontsize=12)
    axes[0].grid(axis='y', color='#d9dedf', zorder=0, linewidth=.7)
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    for i, pair in enumerate(tokens):
        axes[0].text(i, -.039, f'Mean evidence tokens: {pair[0]:.0f} / {pair[1]:.0f}',
                     ha='center', fontsize=9, color='#444b51')

    axes[1].axvline(0, color='#555e67', linewidth=1, linestyle='--', zorder=1)
    for i, scope in enumerate(scopes):
        values = scope['comparison']['metrics'][metric]
        for shift, weighting, color, label in [(.12, 'question_weighted', colors['bm25'], 'Question weighted'),
                                               (-.12, 'family_balanced', colors['family'], 'Family balanced')]:
            point = values[weighting]
            low, high = values[weighting + '_percentile95']
            axes[1].errorbar(point, 1-i+shift, xerr=[[point-low], [high-point]],
                             fmt='o', color=color, capsize=4, markersize=6,
                             label=label if i == 0 else None, zorder=3)
    axes[1].set_yticks([1, 0], ['Given paper', 'Query-only\n32 papers'])
    axes[1].set_ylim(-.6, 1.65)
    axes[1].set_xlim(-.105, .065)
    axes[1].set_xlabel('Evidence F1 difference (BM25 minus dense)')
    axes[1].set_title('Paired intervals span zero in both scopes', loc='left', pad=18, fontsize=12)
    axes[1].legend(frameon=False, loc='upper left', fontsize=9)
    axes[1].grid(axis='x', color='#d9dedf', linewidth=.7, zorder=0)
    figure.suptitle('Fixed BM25 yields mixed evidence-F1 changes on this development set',
                    x=.05, ha='left', fontsize=15, fontweight='bold', y=.98)
    figure.text(.05, .895, '77 questions / 24 families  |  Whole native units  |  At most 3 units and 1,024 BGE tokens',
                fontsize=10, color='#444b51')
    figure.text(.05, .055,
        '10,000 whole-family bootstrap draws; descriptive, unadjusted 95% intervals. Fixed BM25 is not a tuned IR baseline.\n'
        'Query-only cross-paper retrieval changes the original given-paper Qasper task. No answer-quality or JEV claim.',
        fontsize=9, color='#444b51', linespacing=1.6)
    figure.subplots_adjust(left=.07, right=.97, bottom=.25, top=.79, wspace=.45)
    outputs = []
    for suffix in ('png', 'svg'):
        path = output / ('lexical_baseline.' + suffix)
        figure.savefig(path, dpi=180, bbox_inches='tight')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                            encoding='utf-8')
        outputs.append(path)
    plt.close(figure)
    manifest = {'status': 'rendered', 'public_source_sha256': hashlib.sha256(raw).hexdigest(),
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                'scope': 'Complete aggregate only; no per-question data or new model calculation.'}
    (output/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output), indent=2))
