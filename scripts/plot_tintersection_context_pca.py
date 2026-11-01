"""Natural T-intersection rollouts and frozen state4 context PCA, without training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MODELS = ['default', 'm', 'l', 's', 'xl']


def collect(args):
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from dreamer import Dreamer
    from envs.metadrive import MetaDrive
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cfg = OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')
    cfg.device = args.device
    OmegaConf.resolve(cfg)
    cfg.model.compile = False
    assert cfg.env.map_block == 'T' and cfg.model.context.enabled
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    histories, rows, episodes = [], [], []
    model = None
    for vehicle in MODELS:
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        c.env.chassis.eval_models = [vehicle]
        c.env.eval_start_seed = args.seed_start - int(c.env.seed)
        c.env.eval_num_scenarios = args.maps
        c.env.eval_episode_num = args.maps
        c.env.eval_env_num = 1
        env = MetaDrive(c.env, evaluation=True)
        try:
            if model is None:
                model = Dreamer(c.model, env.observation_space, env.action_space).to(args.device)
                model.load_state_dict(payload['agent_state_dict'], strict=True)
                model.clone_and_freeze()
                model.eval()
                step = int(payload['step'])
                del payload
            with torch.inference_mode():
                for seed in range(args.seed_start, args.seed_start + args.maps):
                    torch.manual_seed(seed + 12345)
                    np.random.seed(seed + 12345)
                    obs = env.reset(seed=seed)
                    state = model.get_initial_state(1)
                    chassis = dict(env.chassis_parameters)
                    candidates, collision_history = [], []
                    score = 0.
                    for t in range(int(c.env.time_limit)):
                        collision_history.append(bool(obs['ctx_collision']))
                        if (t >= 5 and t % 5 == 0 and obs['ctx_mask'].all()
                                and not bool(obs['is_last']) and not any(collision_history[-5:])):
                            candidates.append((t, np.array(obs['ctx_history'], copy=True)))
                        td = TensorDict({k: torch.as_tensor(np.asarray(v), device=args.device).reshape(1, *np.asarray(v).shape)
                                         for k, v in obs.items()}, batch_size=[1])
                        action, state = model.act(td, state, eval=True)
                        obs, reward, done, info = env.step(action[0].cpu().numpy())
                        score += float(reward)
                        if done:
                            break
                    assert done
                    episode_steps = t + 1
                    eid = len(episodes)
                    ids = np.linspace(0, len(candidates)-1, min(30, len(candidates)), dtype=int)
                    for index in ids:
                        t, h = candidates[index]
                        assert h.shape == (5, 6) and np.isfinite(h).all()
                        histories.append(h)
                        rows.append({'episode': eid, 'seed': seed, 'vehicle': vehicle, 'step': t,
                                     'speed_mps': float(h[-1, 2]), 'wheel_deg_right': float(np.rad2deg(h[-1, 3])),
                                     'mass': chassis['mass'], 'wheel_friction': chassis['wheel_friction'],
                                     'mass_scale': chassis['mass_scale'], 'friction_scale': chassis['friction_scale']})
                    episode = {'episode': eid, 'seed': seed, 'vehicle': vehicle, 'chassis': chassis,
                               'steps': episode_steps,
                               'success': bool(info.get('arrive_dest', False)), 'reward': score,
                               'windows': len(ids)}
                    episodes.append(episode)
                    print(f'{len(episodes)}/{5*args.maps} {vehicle} seed={seed} windows={len(ids)}', flush=True)
                    (args.output / 'progress.json').write_text(json.dumps({'episodes': len(episodes), 'total': 5*args.maps}))
        finally:
            env.close()
    history = np.asarray(histories, dtype=np.float32)
    np.savez_compressed(args.output / 'histories.npz', history=history)
    (args.output / 'windows.json').write_text(json.dumps(rows, indent=2))
    (args.output / 'episodes.json').write_text(json.dumps(episodes, indent=2))
    (args.output / 'provenance.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_step': step,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'source': str(args.source.resolve()), 'maps': args.maps, 'seed_start': args.seed_start,
        'collection': 'Own frozen policy; actor mode; stochastic RSSM fixed per-map seed. Natural resets, no speed/steering overrides. Five vehicle types on the same map seeds; mass/friction retain training grids. Non-overlapping K5 windows, at most30 evenly spaced per episode; no speed/angle filters.',
        'sampling_unit': 'episode; leave-one-map-out keeps all five paired type rollouts and all their windows together',
    }, indent=2))


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from analyze_natural_vehicle_context import classify
    from analyze_swap_context_associations import load_context
    import torch
    torch.set_num_threads(1)
    rows = json.loads((args.output / 'windows.json').read_text())
    history = np.load(args.output / 'histories.npz')['history']
    _, context, metadata = load_context(args.source, args.checkpoint, history, 'T_pca')
    provenance = json.loads((args.output / 'provenance.json').read_text())
    assert metadata['sha256'] == provenance['checkpoint_sha256']
    y = np.array([MODELS.index(r['vehicle']) for r in rows])
    maps = np.array([r['seed'] for r in rows])
    episodes = np.array([r['episode'] for r in rows])
    scale = context.std(0).clip(1e-6)
    z = (context-context.mean(0))/scale
    _, singular, axes = np.linalg.svd(z, full_matrices=False)
    xy = z @ axes[:2].T
    variance = singular[:2]**2 / (singular**2).sum()
    total = (z*z).sum()
    eta2 = sum((y == k).sum() * (z[y == k].mean(0)**2).sum() for k in range(5)) / total
    # Descriptive adjustment only: nonlinear state effects can remain after this regression.
    conditions = history[..., 2:].reshape(len(y), -1).astype(float)
    conditions = (conditions-conditions.mean(0))/conditions.std(0).clip(1e-6)
    multipliers = np.array([[r['mass_scale'], r['friction_scale']] for r in rows])
    nuisance = np.column_stack((np.ones(len(y)), conditions, conditions**2, multipliers))
    type_design = np.eye(5)[y, 1:]
    residual = z-nuisance @ np.linalg.lstsq(nuisance, z, rcond=None)[0]
    full = np.column_stack((nuisance, type_design))
    residual_full = z-full @ np.linalg.lstsq(full, z, rcond=None)[0]
    partial_type = 1-(residual_full**2).sum()/(residual**2).sum()
    spaces = {'context8': context, 'speed_angle_history': history[..., 2:4].reshape(len(y), -1),
              'raw_history': history.reshape(len(y), -1)}
    probes = {}
    for name, x in spaces.items():
        pred = np.full(len(y), -1)
        for held in np.unique(maps):
            train, test = maps != held, maps == held
            pred[test] = classify(x, y, train, test, episodes, 'softmax')
        accuracies = [(pred[episodes == e] == y[episodes == e]).mean() for e in np.unique(episodes)]
        confusion = np.zeros((5, 5))
        for e in np.unique(episodes):
            select = episodes == e
            confusion[y[select][0]] += np.bincount(pred[select], minlength=5) / select.sum() / args.maps
        probes[name] = {'episode_balanced_accuracy': float(np.mean(accuracies)), 'confusion': confusion.tolist()}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    markers = ['o', 'o', 'o', '^', '^']
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
    order = np.random.default_rng(13).permutation(len(y))
    for k, vehicle in enumerate(MODELS):
        ids = order[y[order] == k]
        axs[0, 0].scatter(*xy[ids].T, color=colors[k], marker=markers[k], s=12, alpha=.45, edgecolors='none', rasterized=True)
    axs[0, 0].set_title('Vehicle type')
    factors = [('speed_mps', 'Speed (m/s)', 'viridis'), ('wheel_deg_right', 'Front-wheel angle (deg, right +)', 'coolwarm'),
               ('mass_scale', 'Mass multiplier', 'viridis'), ('friction_scale', 'Friction multiplier', 'viridis')]
    for ax, (key, title, cmap) in zip([axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]], factors):
        values = np.array([r[key] for r in rows])
        sc = ax.scatter(*xy[order].T, c=values[order], cmap=cmap, s=10, alpha=.5, edgecolors='none', rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=.75)
        ax.set_title(title)
    for ax in axs.flat[:5]:
        ax.set_xlabel(f'PC1 ({variance[0]:.1%})')
        ax.set_ylabel(f'PC2 ({variance[1]:.1%})')
        ax.set_xlim(xy[:, 0].min()-.2, xy[:, 0].max()+.2)
        ax.set_ylim(xy[:, 1].min()-.2, xy[:, 1].max()+.2)
        ax.grid(alpha=.12)
        ax.set_axisbelow(True)
    ax = axs[1, 2]
    image = ax.imshow(probes['context8']['confusion'], vmin=0, vmax=1, cmap='Blues')
    for i in range(5):
        for j in range(5):
            value = probes['context8']['confusion'][i][j]
            ax.text(j, i, f'{value:.0%}', ha='center', va='center', color='white' if value>.6 else '#222222', fontsize=8)
    ax.set_xticks(range(5), [v.upper() for v in MODELS], rotation=30, ha='right')
    ax.set_yticks(range(5), [v.upper() for v in MODELS])
    ax.set(xlabel='Predicted vehicle', ylabel='True vehicle', title=f'Leave-one-map-out linear probe\nAccuracy {probes["context8"]["episode_balanced_accuracy"]:.1%} (chance 20%)')
    fig.colorbar(image, ax=ax, shrink=.75)
    for label, ax in zip('abcdef', axs.flat):
        ax.text(-.14, 1.08, label, transform=ax.transAxes, weight='bold', fontsize=12)
    handles = [Line2D([], [], marker=markers[k], color=colors[k], linestyle='none', label=v.upper()+(' (unseen type)' if k>=3 else ''), markersize=6) for k, v in enumerate(MODELS)]
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle(f'T-intersection: 11 DALI shallow + swap1, best at {metadata["step"]:,} steps\n{len(y):,} natural K5 windows; {len(np.unique(episodes))} episodes; identical PCA coordinates in a–e', fontsize=11)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / f'context_vehicle_pca.{ext}', dpi=300)
    plt.close(fig)
    # A larger standalone vehicle panel makes overlapping classes easier to inspect.
    fig, ax = plt.subplots(figsize=(6.2, 4.6), layout='constrained')
    for k in range(5):
        ids = order[y[order] == k]
        ax.scatter(*xy[ids].T, s=18, marker=markers[k], color=colors[k], alpha=.5, edgecolors='none', rasterized=True)
    ax.set(xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})', title='Context PCA colored by vehicle type\nNatural T-intersection driving; best checkpoint')
    ax.grid(alpha=.12)
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / f'context_vehicle_pca_main.{ext}', dpi=300)
    plt.close(fig)
    result = {'samples': len(y), 'episodes': len(np.unique(episodes)), 'maps': len(np.unique(maps)),
              'counts': {v: int((y==k).sum()) for k,v in enumerate(MODELS)},
              'variance_ratio': variance.tolist(), 'vehicle_eta_squared_standardized_context': float(eta2),
              'vehicle_partial_r2': float(partial_type),
              'partial_r2_adjustment': 'Linear and squared K5 speed/angle/action channels plus mass/friction multipliers; descriptive in-sample regression, not proof of disentanglement.',
              'probes': probes, 'checkpoint': metadata,
              'limitations': ['PCA and eta squared are descriptive, not causal type information.',
                              'Natural speed, angle and policy trajectories can depend on vehicle type.',
                              'Supervised probe includes labels for all five types, including S/XL; it is not zero-shot recognition.',
                              'Only one trained checkpoint and ten map groups. No proof of world model use.']}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    with (args.output / 'source_data.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])+['pc1', 'pc2']+[f'context_{i}' for i in range(8)])
        writer.writeheader()
        for row, coordinates, ctx in zip(rows, xy, context):
            writer.writerow({**row, 'pc1': coordinates[0], 'pc2': coordinates[1], **{f'context_{i}': v for i,v in enumerate(ctx)}})
    np.savez_compressed(args.output / 'pca_arrays.npz', context=context, coordinates=xy, mean=context.mean(0), scale=scale, axes=axes[:2])
    (args.output / 'README.md').write_text('# T-intersection context PCA\n\n自然驾驶，不固定或筛选速度、转角。使用本次11的best模型及其自身策略，五车型各10回合，同地图种子配对，质量及摩擦倍率保留0.6–1.4。每回合至多30个不重叠K5窗口，剔除碰撞与不完整历史。\n\nPCA基于标准化8维context，使用所有展示点无监督拟合。五个散点面板使用完全相同坐标；颜色分别表示车型、速度、前轮角、质量倍率、摩擦倍率。车型关联eta²为标准化context总平方和中的类间差异占比，不是信息百分比。\n\n车型探针为标准化+逻辑回归，按整张地图留出，所有同地图的五车型回合一同进入测试折，报告回合等权准确率。速度/转角历史及原始历史作为参照，详见results.json。探针对S/XL使用监督标签，不是零样本分类。PCA聚类和探针准确率不能单独证明纯动力学参数辨识或世界模型利用。\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maps', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=10620)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    if not args.plot_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'figure_contract.json').write_text(json.dumps({
            'question': 'Does context organize by vehicle type during natural T-intersection driving?',
            'evidence': ['PCA by type', 'Identical projection recolored by operating conditions and parameter multipliers', 'Map-group held-out linear probe with kinematic references'],
            'archetype': 'quantitative grid', 'backend': 'python', 'exports': ['png', 'pdf', 'svg', 'csv'],
            'risk': 'Visual separation can reflect operating state and policy; no causal or world-model-use claim.'}, indent=2))
        collect(args)
    plot(args)


if __name__ == '__main__':
    main()
