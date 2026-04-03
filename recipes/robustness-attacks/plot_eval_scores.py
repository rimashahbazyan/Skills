"""
Visualize effective eval score (min of eval_score, parent_score) per (type, position)
across iterations for robustness attack JSONL archives.

Usage:
    python plot_eval_scores.py --archive-dir <path> --n-iters <N> --out <out.png> [--baseline 60.89]
"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_BASELINE = 60.89  # clean pass@1 on GPQA-Diamond

POSITION_LABELS = {0: 'Before question', 1: 'After question', 2: 'After options'}
TYPE_LABELS = {
    'CODE_SNIPPET':   'Code Snippet',
    'ENCRYPTED_TEXT': 'Encrypted Text',
    'MARKUP_NOISE':   'Markup Noise',
    'MATH_FACT':      'Math Fact',
    'RANDOM_FACT':    'Random Fact',
}

# Muted, colorblind-friendly palette
COLORS = ['#4e79a7', '#e15759', '#59a14f']


def load_data(archive_dir, n_iters):
    """Load effective score (min of eval_score, parent_score) for each (type, position) x iteration."""
    data = {}
    for i in range(1, n_iters + 1):
        path = f'{archive_dir}/iter{i}.jsonl'
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                key = (row['type'], row['position'])
                if key not in data:
                    data[key] = {}
                eval_s = row['eval_score']
                parent_s = row['parent_score']
                score = eval_s if (parent_s is None or eval_s < parent_s) else parent_s
                data[key][i] = score
    return data


def plot(data, n_iters, out_path, baseline=DEFAULT_BASELINE):
    types = sorted({k[0] for k in data})
    positions = sorted({k[1] for k in data})
    iters = list(range(1, n_iters + 1))

    all_scores = [v for d in data.values() for v in d.values()]
    ymin = max(0, min(all_scores) - 4)
    ymax = min(100, max(max(all_scores), baseline) + 4)

    # Sparse x-ticks: use round steps (1, 5, or 10) to avoid crowding
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
    fig, axes = plt.subplots(
        1, n_types,
        figsize=(5.5 * n_types + 2.0, 7.0),
        sharey=True,
        gridspec_kw={'wspace': 0.08},
    )
    if n_types == 1:
        axes = [axes]

    fig.patch.set_facecolor('#fafafa')

    for ax, t in zip(axes, types):
        ax.set_facecolor('#fafafa')

        ax.axhline(baseline, color='#888888', linewidth=1.2, linestyle='--', zorder=2)
        ax.fill_between(
            [iters[0] - 0.5, iters[-1] + 0.5],
            baseline, ymax,
            color='#888888', alpha=0.06, zorder=1,
        )

        vline_x = iters[0] - 0.5
        ax.axvline(vline_x, color='#aaaaaa', linewidth=1.0, linestyle=':', zorder=2)

        for p, color in zip(positions, COLORS):
            key = (t, p)
            if key not in data:
                continue
            scores = [data[key].get(i, np.nan) for i in iters]
            ax.plot(
                iters, scores,
                color=color, marker=None,
                linewidth=2.5, zorder=3, clip_on=False,
            )
            final = next((scores[i] for i in range(len(iters) - 1, -1, -1)
                          if not np.isnan(scores[i])), None)
            if final is not None:
                ax.annotate(
                    f'{final:.1f}',
                    xy=(iters[-1], final),
                    xytext=(4, 0), textcoords='offset points',
                    fontsize=10, color=color, va='center', clip_on=False,
                )

        title = TYPE_LABELS.get(t, t.replace('_', ' ').title())
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color='#222222')
        if ax is axes[len(axes) // 2]:
            ax.set_xlabel('Iteration', fontsize=13, color='#444444', labelpad=6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=11, rotation=45, ha='right')
        ax.set_xlim(iters[0] - 0.5, iters[-1] + 0.5)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True, which='major', axis='y', color='#cccccc', linewidth=0.7, zorder=0)
        ax.grid(False, which='major', axis='x')
        ax.tick_params(axis='y', labelsize=11, colors='#444444')
        ax.tick_params(axis='x', colors='#444444')

    axes[0].set_ylabel('Accuracy (%)', fontsize=13, color='#444444', labelpad=6)

    pos_handles = [
        Line2D([0], [0], color=c, linewidth=2.5, label=POSITION_LABELS.get(p, f'pos={p}'))
        for p, c in zip(positions, COLORS)
    ]
    baseline_handle = Line2D([0], [0], color='#888888', linewidth=1.5, linestyle='--',
                              label=f'Baseline ({baseline:.1f}%)')

    fig.suptitle(
        'Adversarial Distractor Attack — Accuracy by Type & Position\n'
        'Qwen3-8B · GPQA-Diamond',
        fontsize=15, fontweight='bold', color='#111111', y=1.04,
    )

    fig.legend(
        handles=pos_handles + [baseline_handle],
        title='Distractor position', title_fontsize=12,
        loc='upper right', ncol=1,
        bbox_to_anchor=(1.0, 1.0),
        fontsize=12, frameon=True, framealpha=0.9,
        edgecolor='#cccccc',
    )

    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True, help='Path to archive/ directory')
    parser.add_argument('--n-iters', type=int, required=True)
    parser.add_argument('--out', default='eval_scores.png')
    parser.add_argument('--baseline', type=float, default=DEFAULT_BASELINE,
                        help='Clean baseline accuracy to draw as reference line')
    args = parser.parse_args()

    data = load_data(args.archive_dir, args.n_iters)
    plot(data, args.n_iters, args.out, baseline=args.baseline)
