"""Controlled initial-speed / physical wheel-angle probes for the T-trained encoder."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

MODELS = ['default', 'm', 'l', 's', 'xl']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', type=int, default=24)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'figure_contract.json').write_text(json.dumps({
        'question': 'Does T-trained context retain vehicle-type structure at matched initial speed and measured wheel angle?',
        'evidence': ['Nominal physical parameters', 'Paired random mass/friction multipliers', 'Profile-group held-out classification'],
        'archetype': 'quantitative grid', 'backend': 'python', 'exports': ['png', 'pdf', 'svg', 'csv', 'npz'],
        'risks': ['Actual wheel angle held fixed entails type-specific normalized steering input.',
                  'Only initial speed fixed; speeds diverge inside the K5 history.',
                  'Controlled straight-road probes are not driving-policy evaluation.']}, indent=2))
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from omegaconf import OmegaConf
    from envs.metadrive import MetaDrive
    from analyze_swap_context_associations import load_context
    from analyze_natural_vehicle_context import classify
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')
    cfg.device = 'cpu'
    OmegaConf.resolve(cfg)
    assert cfg.env.map_block == 'T'
    cfg.env.traffic_density = 0.
    cfg.env.spawn.randomize = False
    cfg.env.eval_start_seed = 20000
    cfg.env.eval_num_scenarios = 1
    cfg.env.time_limit = 200
    cfg.env.termination.out_of_road_done = False
    cfg.env.termination.on_continuous_line_done = False
    cfg.env.chassis.eval_mass_scales = [1.]
    cfg.env.chassis.eval_steering_scales = [1.]
    cfg.env.chassis.eval_friction_scales = [1.]
    env = MetaDrive(cfg.env, evaluation=True)
    # This diagnostic uses a common flat road; the checkpoint remains T-trained.
    env._env.config['map_config']['config'] = ''
    env._env.config['map_config']['lane_width'] = 20.
    histories, rows = [], []
    try:
        for regime in ('nominal', 'randomized'):
            for profile in range(args.profiles):
                rng = np.random.default_rng(910000 + profile)
                throttle = np.interp(np.arange(5), [0, 2, 4], rng.uniform(.15, .4, 3)).astype(np.float32)
                ms, fs = (1., 1.) if regime == 'nominal' else tuple(rng.choice(np.arange(6, 15)/10, 2))
                for vehicle_type in MODELS:
                    spec = env._env.config['chassis_spec']
                    for key, value in {'models': [vehicle_type], 'mass_scales': [float(ms)],
                                       'steering_scales': [1.], 'friction_scales': [float(fs)]}.items():
                        spec[key] = value
                    env.reset(seed=20000)
                    car = env._env.vehicle
                    for _ in range(30):
                        _, _, done, _ = env.step(np.zeros(2, np.float32))
                        assert not done, 'Unexpected settling termination'
                    car.set_velocity([1., 0.], value=10., in_local_frame=True)
                    steer = 15./car.max_steering
                    car._set_action(np.array([steer, throttle[0]], np.float32))
                    env._context_history.observe(car)
                    speed0 = float(np.linalg.norm(car.velocity))
                    wheel0 = [car.system.getSteeringValue(i) for i in (0, 1)]
                    np.testing.assert_allclose(speed0, 10., atol=1e-4)
                    np.testing.assert_allclose(wheel0, 15., atol=1e-4)
                    actual = dict(env.chassis_parameters)
                    np.testing.assert_allclose(car.body.getMass(), actual['mass'], atol=1e-4)
                    np.testing.assert_allclose([car.system.getWheel(i).getFrictionSlip() for i in range(4)], actual['wheel_friction'], rtol=1e-6)
                    for t in range(5):
                        obs, _, done, _ = env.step(np.array([steer, throttle[t]], np.float32))
                        assert not done and not obs['ctx_collision'], 'Invalid probe; no selective rejection'
                        np.testing.assert_allclose([car.system.getSteeringValue(i) for i in (0, 1)], 15., atol=1e-4)
                    assert obs['ctx_mask'].all()
                    h = obs['ctx_history'].copy()
                    assert h.shape == (5, 6)
                    np.testing.assert_allclose(h[:, 3], -np.deg2rad(15), atol=1e-5)
                    histories.append(h)
                    rows.append({'regime': regime, 'vehicle': vehicle_type, 'profile': profile,
                                 'initial_speed_mps': speed0, 'initial_wheel_angle_deg_left': float(wheel0[0]),
                                 'endpoint_speed_mps': float(h[-1, 2]), 'normalized_steering_left': float(steer),
                                 'mass_scale': float(ms), 'friction_scale': float(fs),
                                 'actual_mass': actual['mass'], 'actual_wheel_friction': actual['wheel_friction']})
                if (profile+1) % 6 == 0:
                    print(regime, profile+1, '/', args.profiles, 'profiles', flush=True)
    finally:
        env.close()
    history = np.array(histories)
    _, context, metadata = load_context(args.source, args.checkpoint, history, 'T_fixed')
    y = np.array([MODELS.index(r['vehicle']) for r in rows])
    groups = np.array([r['profile'] for r in rows])
    regimes = np.array([r['regime'] for r in rows])
    folds = np.array_split(np.random.default_rng(20261016).permutation(args.profiles), 6)
    coordinates = np.zeros((len(y), 2))
    results = {'checkpoint': metadata, 'initial_speed_mps': 10., 'physical_wheel_angle_deg_left': 15.,
               'profiles': args.profiles, 'window_count': len(y), 'conditions': {}}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#4C78A8', '#59A14F', '#F28E2B', '#E15759', '#B279A2']
    markers = ['o', 'o', 'o', '^', '^']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for ax, regime, letter in zip(axs, ('nominal', 'randomized'), 'ab'):
        ids = np.flatnonzero(regimes == regime)
        x = context[ids]
        fit = ~np.isin(groups[ids], folds[0])
        mean, sd = x[fit].mean(0), x[fit].std(0).clip(1e-6)
        z = (x-mean)/sd
        _, singular, axes = np.linalg.svd(z[fit], full_matrices=False)
        xy = z @ axes[:2].T
        coordinates[ids] = xy
        variance = singular[:2]**2/(singular**2).sum()
        zz = z-z.mean(0)
        eta2 = sum((y[ids] == k).sum()*(zz[y[ids] == k].mean(0)**2).sum() for k in range(5))/(zz**2).sum()
        probes = {}
        for name, features in {'context8': x, 'speed_angle_history': history[ids, :, 2:4].reshape(len(ids), -1),
                               'raw_history': history[ids].reshape(len(ids), -1)}.items():
            predictions = np.full(len(ids), -1)
            for held in folds:
                test = np.isin(groups[ids], held)
                predictions[test] = classify(features, y[ids], ~test, test, np.arange(len(ids)), 'softmax')
            confusion = np.zeros((5, 5), dtype=int)
            np.add.at(confusion, (y[ids], predictions), 1)
            probes[name] = {'accuracy': float((predictions == y[ids]).mean()), 'confusion': confusion.tolist()}
        for k, vehicle in enumerate(MODELS):
            select = y[ids] == k
            ax.scatter(*xy[select].T, color=colors[k], marker=markers[k], s=38, alpha=.7, edgecolors='none')
        ax.set(xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})')
        ax.set_title(('Nominal mass / friction' if regime == 'nominal' else 'Random mass / friction: 0.6–1.4×')+
                     f'\nHeld-out-profile type accuracy: {probes["context8"]["accuracy"]:.1%}', fontsize=10)
        ax.text(-.12, 1.08, letter, transform=ax.transAxes, weight='bold', fontsize=12)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
        results['conditions'][regime] = {'pca_variance_ratio': variance.tolist(), 'vehicle_eta_squared': float(eta2),
                                          'probes': probes, 'samples': len(ids),
                                          'endpoint_speed_range_mps': [float(history[ids, -1, 2].min()), float(history[ids, -1, 2].max())]}
    handles = [Line2D([], [], marker=markers[k], linestyle='none', color=colors[k], label=v.upper()+(' (unseen type)' if k>=3 else ''), markersize=6) for k,v in enumerate(MODELS)]
    fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=9)
    fig.suptitle(f'T-trained 11 context | best at {metadata["step"]:,} steps\nInitial speed 10 m/s; measured front-wheel angle held at 15°\n24 shared throttle profiles per type; speed evolves freely during each K5 window', fontsize=11)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / f'fixed_condition_context_pca.{ext}', dpi=300)
    plt.close(fig)
    results['methodology'] = 'Physical diagnostic on the same flat traffic-free wide road; independent resets with30 settling steps. Set initial speed once; actual wheel angle maintained at15deg by type-specific normalized steer. Five real10Hz decision steps per K5 history. Same24 throttle profiles per type and regime; random mass/friction multipliers paired across types. PCA standardized and fitted on20 profile groups, remaining4 projected; panels fit separately. Six profile-group held-out softmax probe with train-only scaling; all five types supervised, includingS/XL.'
    results['limitations'] = ['Fixed initial speed does not imply identical speed inside the history.',
                              'Normalized steer varies by type to maintain identical physical wheel angle; it can reveal max-steering/type.',
                              'High type separability does not prove pure parameter identification or world-model use.',
                              'Twenty-four randomized parameter pairs are a diagnostic sample, not an exhaustive243-chassis grid test.',
                              'Probe population differs from the natural-driving PCA; accuracies are not a controlled improvement comparison.']
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    np.savez_compressed(args.output / 'representations.npz', history=history, context=context, coordinates=coordinates)
    with (args.output / 'source_data.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])+['pc1', 'pc2']+[f'context_{k}' for k in range(8)])
        writer.writeheader()
        for row, xy, ctx in zip(rows, coordinates, context):
            writer.writerow({**row, 'pc1': xy[0], 'pc2': xy[1], **{f'context_{k}': v for k,v in enumerate(ctx)}})
    (args.output / 'README.md').write_text('# 固定初速和实际前轮角的车型context诊断\n\n初速10m/s仅设置一次，实际前轮角15度在5步窗口内保持不变，油门序列跨车型配对。左图为各车型标称质量/摩擦，右图为0.6–1.4倍率的24组配对随机质量/摩擦。每图五车型各24点，独立拟合PCA，不能比较两图绝对坐标。采用本次T路口训练11的120007步best编码器；采样环境为无车流宽直道，不是策略驾驶回合。\n\n车型探针按油门profile整组留出，同一profile下所有车型和全部参数配对样本均不跨训练/测试。S/XL虽为编码器训练未见车型，探针仍用其监督标签，不是零样本分类。实际轮角一致要求不同归一化转向输入，该线索和后续速度差异仍存在，不能单凭分群声称纯动力学辨识或世界模型利用。数值、检查点哈希及限制见results.json，完整输入和输出见representations.npz及source_data.csv。\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
