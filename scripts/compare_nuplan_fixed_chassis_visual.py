"""Paired full-episode top-down GIFs and shared navigation/trajectory figure."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=int, default=0)
    parser.add_argument('--scenario-dir', type=Path,
                        help='Optional complete split; preserves original scenario indices')
    parser.add_argument('--mass-scale', type=float, default=1.)
    parser.add_argument('--friction-scale', type=float, default=.6)
    parser.add_argument('--selection-note', default='Scene fixed before policy evaluation; no selection using policy outcomes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.queue / 'plan.json').read_text())
    jobs = {name: next(j for j in plan['jobs'] if j['name'] == 'nuplan_' + name)
            for name in ('r2dreamer', 'r2dreamer_context')}
    assert len({j['source'] for j in jobs.values()}) == 1
    source = Path(jobs['r2dreamer']['source'])
    sys.path.insert(0, str(source))
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from dreamer import Dreamer
    from envs.nuplan import NuPlanDrive

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    labels = {'r2dreamer': 'R2Dreamer', 'r2dreamer_context': 'R2Dreamer+Context'}
    colors = {'r2dreamer': '#194F6A', 'r2dreamer_context': '#ED3943'}
    records, animations, results = {}, {}, {}
    reference, polygons, initial = None, None, None

    # Figure contract: show one prespecified paired scenario, without an expected winner.
    # Evidence: complete policy trajectories against the actual shared navigation route.
    # Quantitative single panel, Python backend, editable PDF/SVG + PNG and raw traces.
    # No fitted reference, temporal cropping, or smoothing. Record selection provenance.
    for name, job in jobs.items():
        print('START', name, flush=True)
        (args.output / 'status.json').write_text(json.dumps(dict(status='running', method=name)))
        checkpoint = Path(job['logdir']) / 'best.pt'
        config = OmegaConf.load(checkpoint.parent / '.hydra/config.yaml')
        config.device = 'cpu'
        OmegaConf.resolve(config)
        config.model.compile = False
        config.env.scenario_eval_dir = config.env.scenario_test_dir
        config.env.eval_num_scenarios = 100
        if args.scenario_dir:
            config.env.scenario_eval_dir = str(args.scenario_dir.resolve())
            config.env.eval_num_scenarios = -1
        config.env.eval_episode_num = 1
        config.env.eval_env_num = 1
        config.env.chassis.eval_models = ['xl']
        config.env.chassis.eval_mass_scales = [args.mass_scale]
        config.env.chassis.eval_steering_scales = [1.]
        config.env.chassis.eval_friction_scales = [args.friction_scale]
        env = NuPlanDrive(config.env, evaluation=True)
        try:
            assert env.observation_space['state'].shape == (271,)
            model = Dreamer(config.model, env.observation_space, env.action_space).to('cpu')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(payload['agent_state_dict'], strict=True)
            model.clone_and_freeze()
            model.eval()
            checkpoint_step = int(payload['step'])
            del payload
            torch.manual_seed(args.scene + 12345)
            np.random.seed(args.scene + 12345)
            obs = env.reset(seed=args.scene)
            state = model.get_initial_state(1)
            car = env._env.vehicle
            chassis = dict(env.chassis_parameters)
            scenario_id = env._env.engine.data_manager.current_scenario_id
            route = car.navigation.reference_trajectory
            ref = np.array([route.position(float(s), 0.) for s in
                            np.linspace(0., route.length, max(2, int(route.length / .25) + 1))])[:, :2]
            init = dict(position=np.asarray(car.position).tolist(), velocity=np.asarray(car.velocity).tolist(),
                        heading=float(car.heading_theta), chassis=chassis, scenario_id=scenario_id)
            if reference is None:
                reference, initial = ref, init
                polygons = [np.asarray(lane.polygon).tolist()
                            for lane in env._env.current_map.road_network.get_all_lanes()
                            if lane.polygon is not None]
            else:
                np.testing.assert_allclose(reference, ref, atol=1e-8)
                np.testing.assert_allclose(initial['position'], init['position'], atol=1e-8)
                np.testing.assert_allclose(initial['velocity'], init['velocity'], atol=1e-8)
                assert initial['heading'] == init['heading'] and initial['chassis'] == chassis
                assert initial['scenario_id'] == scenario_id
            assert chassis['model'] == 'xl'
            assert np.isclose(chassis['mass_scale'], args.mass_scale)
            assert np.isclose(chassis['friction_scale'], args.friction_scale)
            dt = float(config.env.decision_repeat) * .02
            center = ((reference.min(axis=0) + reference.max(axis=0)) / 2).tolist()
            frames, trace = [], []
            score, info = 0., {}

            def capture(step, action, final=False):
                rgb = env.render(mode='topdown', window=False, screen_size=(700, 700),
                                 film_size=(3000, 3000), scaling=2.5, camera_position=center,
                                 draw_center_line=True, draw_target_vehicle_trajectory=True,
                                 semantic_map=False)
                image = Image.new('RGB', (700, 780), 'white')
                image.paste(Image.fromarray(np.asarray(rgb, dtype=np.uint8)), (0, 32))
                draw = ImageDraw.Draw(image)
                draw.text((12, 6), f'{labels[name]} | {int(config.env.steps)//1000}K | XL | scene {args.scene} | best {checkpoint_step}',
                          font=font, fill=colors[name])
                outcome = ('SUCCESS' if info.get('arrive_dest') else 'COLLISION' if info.get('crash')
                           else 'OUT OF ROAD' if info.get('out_of_road') else 'TIMEOUT' if final else 'RUNNING')
                draw.text((12, 738), f't={step * dt:.1f}s  v={car.speed_km_h:.1f} km/h  '
                          f'route={info.get("route_completion", 0):.1%}  {outcome}', font=font, fill='#202020')
                draw.text((12, 760), f'mass={chassis["mass"]} kg  wheel_friction={chassis["wheel_friction"]:.2f}',
                          font=font, fill='#454545')
                frames.append(image)
                trace.append(dict(step=step, time_s=step * dt, x=float(car.position[0]), y=float(car.position[1]),
                                  speed_km_h=float(car.speed_km_h), steering=float(action[0]),
                                  throttle_brake=float(action[1]), route_completion=float(info.get('route_completion', 0)),
                                  cumulative_reward=score))

            capture(0, [0., 0.])
            with torch.inference_mode():
                for step in range(1, int(config.env.time_limit) + 1):
                    tensors = {}
                    for key, value in obs.items():
                        value = torch.as_tensor(np.asarray(value))
                        tensors[key] = value.reshape(1, 1) if value.ndim == 0 else value.unsqueeze(0)
                    action, state = model.act(TensorDict(tensors, batch_size=(1,)), state, eval=True)
                    action = action[0].numpy()
                    obs, reward, done, info = env.step(action)
                    score += float(reward)
                    capture(step, action, done)
                    if step % 50 == 0:
                        print(name, 'step', step, flush=True)
                    if done:
                        break
                else:
                    raise RuntimeError('Episode did not finish within its native horizon')
            filename = args.output / (name + '_topdown.gif')
            frames[0].save(filename, save_all=True, append_images=frames[1:],
                           duration=[int(dt * 1000)] * (len(frames) - 1) + [1500], loop=0)
            with (args.output / (name + '_trajectory.csv')).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
            records[name], animations[name] = trace, frames
            results[name] = dict(checkpoint=str(checkpoint), checkpoint_step=checkpoint_step,
                                 training_steps=int(config.env.steps),
                                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 scenario_id=scenario_id, scene=args.scene, chassis=chassis,
                                 success=bool(info.get('arrive_dest')), crash=bool(info.get('crash')),
                                 out_of_road=bool(info.get('out_of_road')),
                                 timeout=bool(obs['is_last'] and not obs['is_terminal']),
                                 steps=step, reward=score, route_completion=float(info.get('route_completion', 0)),
                                 mean_speed_km_h=float(np.mean([r['speed_km_h'] for r in trace[1:]])),
                                 initial_state=init, gif=str(filename.resolve()))
            print('RESULT', name, json.dumps(results[name]), flush=True)
        finally:
            env.close()

    # Synchronized display uses equal simulation timestamps; after termination the
    # corresponding panel holds its actual final frame while the other continues.
    paired = []
    length = max(map(len, animations.values()))
    for index in range(length):
        image = Image.new('RGB', (1412, 780), '#FFFFFF')
        for x, name in [(0, 'r2dreamer'), (712, 'r2dreamer_context')]:
            image.paste(animations[name][min(index, len(animations[name]) - 1)], (x, 0))
        paired.append(image)
    paired[0].save(args.output / 'paired_topdown.gif', save_all=True, append_images=paired[1:],
                   duration=[100] * (len(paired) - 1) + [1500], loop=0)
    paired[-1].save(args.output / 'topdown_final.png')
    np.savetxt(args.output / 'reference_navigation.csv', reference, delimiter=',',
               header='world_x_m,world_y_m', comments='')
    (args.output / 'map_polygons.json').write_text(json.dumps(polygons))
    (args.output / 'results.json').write_text(json.dumps(dict(methods=results, pairing_verified=True,
        scene_selection=args.selection_note,
        source_root=str(source), inference_seed=args.scene + 12345, device='cpu',
        scenario_directory=str(config.env.scenario_eval_dir),
        reference='Actual MetaDrive navigation.reference_trajectory; sampled at 0.25m; not fitted to policy paths',
        mass_scale=args.mass_scale, friction_scale=args.friction_scale,
        scope='One paired scene and one inference seed; qualitative case, not aggregate benchmark'), indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    origin = np.asarray(initial['position'])
    all_points = np.concatenate([reference] + [np.array([[r['x'], r['y']] for r in trace])
                                              for trace in records.values()]) - origin
    lo, hi = all_points.min(axis=0) - 12, all_points.max(axis=0) + 12
    fig, ax = plt.subplots(figsize=(8.5, 5.7), layout='constrained')
    visible = []
    for polygon in polygons:
        poly = np.asarray(polygon)[:, :2] - origin
        if len(poly) >= 3 and np.all(poly.max(axis=0) >= lo) and np.all(poly.min(axis=0) <= hi):
            visible.append(poly)
    ax.add_collection(PolyCollection(visible, facecolors='#F4F5F6', edgecolors='#D9DDE1', linewidths=.45))
    ax.plot(*(reference - origin).T, color='#303842', linestyle='--', linewidth=1.6,
            label='Reference navigation', zorder=3)
    for name, trace in records.items():
        xy = np.array([[r['x'], r['y']] for r in trace]) - origin
        ax.plot(*xy.T, color=colors[name], linewidth=1.8,
                linestyle='-' if name.endswith('context') else '-.', label=labels[name], zorder=4)
        ax.scatter(*xy[-1], color=colors[name], s=38,
                   marker='o' if name.endswith('context') else 's', zorder=5)
    ax.scatter(0., 0., marker='D', color='black', s=28, label='Start', zorder=6)
    ax.scatter(*(reference[-1] - origin), marker='*', color='#303842', s=80, label='Reference end', zorder=5)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal',
           xlabel='World x relative to start (m)', ylabel='World y relative to start (m)',
           title=f'nuPlan {next(iter(results.values()))["training_steps"]//1000}K | scene {args.scene} | XL | mass {initial["chassis"]["mass"]} kg | '
                 f'wheel friction {initial["chassis"]["wheel_friction"]:.2f}')
    ax.legend(loc='best', frameon=True, framealpha=.95, fontsize=8)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output / ('trajectory_comparison.' + extension), dpi=220)
    plt.close(fig)
    (args.output / 'status.json').write_text(json.dumps(dict(status='completed', pairing_verified=True)))
    print('COMPLETED', args.output, flush=True)


if __name__ == '__main__':
    main()
