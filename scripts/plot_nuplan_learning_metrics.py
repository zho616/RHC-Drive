"""Compare existing nuPlan periodic validation curves; one training seed each."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from plot_tintersection_learning_metrics import tensorboard_ema

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures/nuplan_learning_metrics_tb099_20261024'
RUNS = [
    ('R2Dreamer', '#194F6A', ROOT / 'logdir/nuplan_pittsburgh_baseline_500k_20261022/baseline/metrics.jsonl'),
    ('R2Dreamer+Context', '#ED3943', ROOT / 'logdir/nuplan_context_roundabout_queue_20261023/nuplan_r2dreamer_context/metrics.jsonl'),
]
METRICS = [('episode/eval_score', 'Episode Reward'),
           ('episode/eval_success', 'Success Rate'),
           ('episode/eval_route_completion', 'Route Completion Rate')]

def main():
    # Evidence: compare learning trajectories, without claiming statistical superiority.
    # Quantitative triptych; actual validation steps, debiased EMA .99, no error bands.
    # Export editable PDF/SVG plus PNG and raw/smoothed source values.
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    rows = []
    for label, color, path in RUNS:
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if '"episode/eval_score"' in line]
        assert records and all(a['step'] <= b['step'] for a, b in zip(records, records[1:]))
        for ax, (key, title) in zip(axes, METRICS):
            raw = [r[key] for r in records]
            smooth = tensorboard_ema(raw, .99)
            ax.plot([r['step'] for r in records], smooth, color=color, lw=1.8, label=label)
            rows.extend(dict(method=label, step=r['step'], metric=key, raw=value,
                             smoothed=float(y), smoothing=.99,
                             evaluation_episodes=r.get('episode/eval_episodes', ''))
                        for r, value, y in zip(records, raw, smooth))
        print(label, 'checkpoints', len(records), 'final step', records[-1]['step'],
              'episodes/checkpoint', sorted(set(r.get('episode/eval_episodes') for r in records)))
    for ax, (key, title) in zip(axes, METRICS):
        ax.set_title(title)
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel(title)
        ax.set_xlim(0, 500005)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x/1000:.0f}k'))
        if key != 'episode/eval_score':
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'nuplan_learning_metrics.{ext}', dpi=240)
    with (OUT / 'source_data.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'README.md').write_text('nuPlan training-time periodic validation curves. '
        'One training seed per method; TensorBoard debiased EMA weight 0.99; no uncertainty bands. '
        'These are validation metrics during training, not independent final-test results.\n')
    print(OUT)

if __name__ == '__main__':
    main()
