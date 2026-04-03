"""
Plot raw eval_score (no parent fallback) per (type, position) across iterations.

Usage:
    python plot_raw_scores.py --archive-dir <path> --n-iters <N> --out <out.png> [--baseline 60.89]
"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_BASELINE = 60.89

POSITION_LABELS = {0: 'Before question', 1: 'After question', 2: 'After options'}
TYPE_LABELS = {
    'CODE_SNIPPET':   'Code Snippet',
    'ENCRYPTED_TEXT': 'Encrypted Text',
    'MARKUP_NOISE':   'Markup Noise',
    'MATH_FACT':      'Math Fact',
    'RANDOM_FACT':    'Random Fact',
}

COLORS = ['#4e79a7', '#e15759', '#59a14f']


def load_data(archive_dir, n_iters):
    """Load raw eval_score (no parent fallback) for each (type, position) x iteration."""
    data = {}
    for i in range(1, n_iters + 1):
        path = f'{archive_dir}/iter{i}.jsonl'
        try:
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    if 'eval_score' not in row:
                        continue
                    key = (row['type'], row['position'])
                    if key not in data:
                        data[key] = {}
                    data[key][i] = row['eval_score']
        except FileNotFoundError:
            pass
    return data


def plot(data, n_iters, out_path, baseline=DEFAULT_BASELINE):
    types = sorted({k[0] for k in data})
    positions = sorted({k[1] for k in data})
    iters = list(range(1, n_iters + 1))

    all_scores = [v for d in data.values() for v in d.values()]
    ymin = max(0, min(all_scores) - 4)
    ymax = min(100, max(max(all_scores), baseline) + 4)

    if n_iters <= 10:
        xtick_step = 1
    elif n_iters <= 30:
        xtick_step = 5
    else:
        xtick_step = 10
    xticks = [1] + [i for i in range(xtick_step, n_iters, xtick_step)]
    if n_iters not in xticks:
        xticks.append(n_iters)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    n_types = len(types)
    fig, axes = plt.subplots(1, n_types, figsize=(5.5 * n_types, 6), sharey=True)
    if n_types == 1:
        axes = [axes]

    mid = n_types // 2

    for ax_idx, (ax, dtype) in enumerate(zip(axes, types)):
        ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
        ax.axhspan(baseline - 1, baseline + 1, color='gray', alpha=0.08, zorder=0)

        for pos_idx, position in enumerate(positions):
            key = (dtype, position)
            if key not in data:
                continue
            xs = sorted(data[key].keys())
            ys = [data[key][x] for x in xs]
            ax.plot(xs, ys, color=COLORS[pos_idx], linewidth=1.8, alpha=0.9)

        ax.set_title(TYPE_LABELS.get(dtype, dtype), fontsize=15, fontweight='bold', pad=8)
        ax.set_xlim(0.5, n_iters + 0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], rotation=45, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.25, linestyle=':')

        if ax_idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=13)
        if ax_idx == mid:
            ax.set_xlabel('Iteration', fontsize=13)

    legend_elements = [
        Line2D([0], [0], color=COLORS[i], linewidth=2,
               label=POSITION_LABELS.get(positions[i], f'pos {positions[i]}'))
        for i in range(len(positions))
    ] + [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
               label=f'Baseline ({baseline:.1f}%)')
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=11,
                    framealpha=0.9, title='Distractor position', title_fontsize=11)

    fig.suptitle(
        'Adversarial Distractor Attack — Raw Accuracy by Type & Position\n'
        'Qwen3-8B · GPQA-Diamond',
        fontsize=15, fontweight='bold', y=1.04
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True, help='Path to archive/ directory')
    parser.add_argument('--n-iters', type=int, required=True)
    parser.add_argument('--out', default='raw_eval_scores.png')
    parser.add_argument('--baseline', type=float, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    data = load_data(args.archive_dir, args.n_iters)
    plot(data, args.n_iters, args.out, args.baseline)
