"""Compare random-weight and trained encoders on identical nominal fixed-condition histories."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from compare_fixed_vehicle_random_context import association
from plot_retrained_vehicle_context_pca import fit_pca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {'question': 'Is the vehicle-type separation already present with random encoder weights?',
                'archetype': 'quantitative grid', 'backend': 'python',
                'panels': ['Random initialization, prespecified seed20261015', 'Trained11 best120007'],
                'exports': ['png', 'pdf', 'svg', 'csv', 'npz'],
                'risks': ['Random init is not the original training step0.',
                          'Independent PCA bases; distances across panels are not comparable.',
                          'Identical raw histories already contain vehicle cues; separation alone is not learning evidence.']}
    (args.output / 'figure_contract.json').write_text(json.dumps(contract, indent=2))
    all_rows = list(csv.DictReader((args.input / 'source_data.csv').open()))
    selected = np.array([row['regime'] == 'nominal' for row in all_rows])
    rows = [row for row, keep in zip(all_rows, selected) if keep]
    data = np.load(args.input / 'representations.npz')
    history = data['history'][selected].copy()
    trained = data['context'][selected].copy()
    metadata = json.loads((args.input / 'results.json').read_text())
    types = ['default', 'm', 'l', 's', 'xl']
    y = np.array([types.index(row['vehicle']) for row in rows])
    profiles = np.array([int(row['profile']) for row in rows])
    folds = np.array_split(np.random.default_rng(20261016).permutation(np.unique(profiles)), 6)
    fit = ~np.isin(profiles, folds[0])
    assert history.shape == (120, 5, 6) and fit.sum() == 100
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('T_random_frozen_encoder', args.source / 'context.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    seeds = list(range(20261015, 20261035))
    random_outputs = []
    with torch.inference_mode():
        for seed in seeds:
            torch.manual_seed(seed)
            encoder = module.HistoryEncoder(state_dim=4).eval()
            random_outputs.append(encoder(torch.from_numpy(history), torch.ones(120, 5, dtype=torch.bool)).numpy())
    random_outputs = np.array(random_outputs)
    context = {'random': random_outputs[0], 'trained': trained}
    coords, ratios = {}, {}
    for name, x in context.items():
        coords[name], ratios[name], _, _, _ = fit_pca(x.astype(float), fit)
    np.testing.assert_allclose(coords['trained'], data['coordinates'][selected], atol=1e-8)
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    markers = ['o', 'o', 'o', '^', '^']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for ax, name, letter, title in zip(axs, ['random', 'trained'], 'ab',
                                      [f'Random encoder weights\nInitialization seed {seeds[0]}',
                                       f'Trained encoder\n11 best at {metadata["checkpoint"]["step"]:,} steps']):
        for k in range(5):
            ax.scatter(*coords[name][y == k].T, c=colors[k], marker=markers[k], s=38, alpha=.7, edgecolors='none')
        ax.set(xlabel=f'PC1 ({ratios[name][0]:.1%})', ylabel=f'PC2 ({ratios[name][1]:.1%})', title=title)
        ax.text(-.12, 1.08, letter, transform=ax.transAxes, fontsize=12, weight='bold')
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=markers[k], color=colors[k], linestyle='none', markersize=6, label=v.upper()) for k,v in enumerate(types)]
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle('Vehicle-type PCA | identical 120 physical histories\nInitial speed 10 m/s; measured front-wheel angle 15°; nominal mass / friction\nIndependent standardized PCA bases; 24 shared throttle profiles per type', fontsize=11)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / f'random_vs_trained_vehicle_pca.{ext}', dpi=300)
    plt.close(fig)
    random_eta = np.array([association(x, y, fit) for x in random_outputs])
    result = {'checkpoint': metadata['checkpoint'], 'source_histories': str(args.input), 'windows': 120,
              'random_seeds': seeds, 'displayed_random_seed': seeds[0],
              'random_eta_squared': {'mean': float(random_eta.mean()), 'sd': float(random_eta.std(ddof=1)),
                                     'displayed': float(random_eta[0]), 'all': random_eta.tolist()},
              'trained_eta_squared': association(trained, y, fit),
              'pca_variance_ratio': {name: ratio.tolist() for name, ratio in ratios.items()},
              'method': 'Same frozen encoder architecture, random weights without training. Identical nominal histories from preceding figure. Standardization and each separate PCA fitted on same20 profile groups; all24 groups displayed. Display seed fixed before encoding, not selected by separation. Twenty random seeds used for descriptive8D association sensitivity only.',
              'limitations': contract['risks']}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output / 'representations.npz', history=history, trained_context=trained,
                        random_contexts=random_outputs, random_coordinates=coords['random'], trained_coordinates=coords['trained'])
    with (args.output / 'source_data.csv').open('w') as f:
        fields = ['vehicle', 'profile', 'pca_fit'] + [f'{stage}_{field}{k}' for stage in context for field,n in [('context', 8), ('pc', 2)] for k in range(n)]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i,row in enumerate(rows):
            record = {'vehicle': row['vehicle'], 'profile': row['profile'], 'pca_fit': bool(fit[i])}
            for name, x in context.items():
                record.update({f'{name}_context{k}': float(x[i,k]) for k in range(8)})
                record.update({f'{name}_pc{k}': float(coords[name][i,k]) for k in range(2)})
            writer.writerow(record)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
