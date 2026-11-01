"""Record an explicit post-hoc search for an Ours-success/baseline-failure case."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario-dir', type=Path,
                        help='Optional complete split; preserves original scenario indices')
    parser.add_argument('--max-cases', type=int, default=1,
                        help='Stop after this many Context-success/baseline-failure pairs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.queue / 'plan.json').read_text())
    jobs = {name: next(j for j in plan['jobs'] if j['name'] == 'nuplan_' + name)
            for name in ('r2dreamer', 'r2dreamer_context')}
    sys.path.insert(0, jobs['r2dreamer']['source'])
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from dreamer import Dreamer
    from envs.nuplan import NuPlanDrive
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(Path(jobs['r2dreamer_context']['logdir']) / '.hydra/config.yaml')
    config.device = 'cpu'
    OmegaConf.resolve(config)
    config.model.compile = False
    config.env.scenario_eval_dir = config.env.scenario_test_dir
    config.env.eval_num_scenarios = 100
    if args.scenario_dir:
        config.env.scenario_eval_dir = str(args.scenario_dir.resolve())
        config.env.eval_num_scenarios = -1
    config.env.eval_env_num = 1
    config.env.eval_episode_num = 1
    config.env.chassis.eval_models = ['xl']
    config.env.chassis.eval_mass_scales = [1.]
    config.env.chassis.eval_steering_scales = [1.]
    config.env.chassis.eval_friction_scales = [.6]
    env = NuPlanDrive(config.env, evaluation=True)
    models = {}
    for name, job in jobs.items():
        mc = OmegaConf.load(Path(job['logdir']) / '.hydra/config.yaml')
        mc.device = 'cpu'
        OmegaConf.resolve(mc)
        mc.model.compile = False
        model = Dreamer(mc.model, env.observation_space, env.action_space).to('cpu')
        payload = torch.load(Path(job['logdir']) / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(payload['agent_state_dict'], strict=True)
        model.clone_and_freeze()
        model.eval()
        models[name] = model
        del payload
    candidates = json.loads(args.candidates.read_text())
    state = dict(status='running', selection='Post-hoc favorable-case search: Context success, baseline failure',
                 mass_scale=1., friction_scale=.6, model='xl', inference_seed_offset=12345,
                 candidates=candidates, cases=[], selected=None, selected_cases=[])
    state['scenario_directory'] = str(config.env.scenario_eval_dir)

    def save():
        tmp = args.output / 'search.json.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(args.output / 'search.json')

    def evaluate(name, seed):
        torch.manual_seed(seed + 12345)
        np.random.seed(seed + 12345)
        obs = env.reset(seed=seed)
        latent = models[name].get_initial_state(1)
        initial = np.asarray(env._env.vehicle.position).tolist()
        chassis = dict(env.chassis_parameters)
        positions, speeds, actions = [], [], []
        score = 0.
        route = env._env.vehicle.navigation.reference_trajectory
        ref = np.array([route.position(float(s), 0.) for s in np.linspace(0, route.length, max(2, int(route.length / .25) + 1))])
        for step in range(1, int(config.env.time_limit) + 1):
            data = {}
            for key, value in obs.items():
                tensor = torch.as_tensor(np.asarray(value))
                data[key] = tensor.reshape(1, 1) if tensor.ndim == 0 else tensor.unsqueeze(0)
            action, latent = models[name].act(TensorDict(data, batch_size=(1,)), latent, eval=True)
            action = action[0].numpy()
            obs, reward, done, info = env.step(action)
            score += float(reward)
            positions.append(np.asarray(env._env.vehicle.position).copy())
            speeds.append(float(env._env.vehicle.speed_km_h))
            actions.append(action.copy())
            if done:
                break
        xy = np.asarray(positions)
        errors = np.linalg.norm(xy[:, None, :2] - ref[None, :, :2], axis=-1).min(axis=1)
        return dict(success=bool(info.get('arrive_dest')), crash=bool(info.get('crash')),
                    out_of_road=bool(info.get('out_of_road')), timeout=bool(obs['is_last'] and not obs['is_terminal']),
                    route_completion=float(info.get('route_completion', 0)), reward=score, steps=step,
                    mean_speed_km_h=float(np.mean(speeds)), route_rmse_m=float(np.sqrt(np.mean(errors ** 2))),
                    action_mean_absolute_change=np.abs(np.diff(actions, axis=0)).mean(axis=0).tolist(),
                    chassis=chassis, initial_position=initial)

    try:
        save()
        with torch.inference_mode():
            for candidate in candidates:
                scene = candidate['scene']
                case = dict(scene=scene, geometry=candidate)
                case['r2dreamer_context'] = evaluate('r2dreamer_context', scene)
                state['cases'].append(case)
                save()
                print('OURS', scene, json.dumps(case['r2dreamer_context']), flush=True)
                if not case['r2dreamer_context']['success']:
                    case['baseline_not_run'] = 'Context failed; cannot satisfy favorable-case criterion'
                    save()
                    continue
                case['r2dreamer'] = evaluate('r2dreamer', scene)
                assert case['r2dreamer']['chassis'] == case['r2dreamer_context']['chassis']
                assert case['r2dreamer']['initial_position'] == case['r2dreamer_context']['initial_position']
                save()
                print('BASELINE', scene, json.dumps(case['r2dreamer']), flush=True)
                if not case['r2dreamer']['success']:
                    state['selected_cases'].append(case)
                    state['selected'] = state['selected_cases'][0]
                    state['selection_reason'] = 'Context arrived; baseline failed, with fixed parameters and matched inference seed'
                    if len(state['selected_cases']) >= args.max_cases:
                        break
            if state['selected'] is None:
                successful = [c for c in state['cases'] if c['r2dreamer_context']['success']]
                better_tracking = [c for c in successful if c['r2dreamer_context']['route_rmse_m'] < c['r2dreamer']['route_rmse_m']]
                if better_tracking:
                    state['selected'] = min(better_tracking, key=lambda c: c['r2dreamer_context']['route_rmse_m'] / c['r2dreamer']['route_rmse_m'])
                    state['selection_reason'] = 'Both arrived; Context has lower full-episode route RMSE (time weighted)'
            state['status'] = 'completed'
            save()
    finally:
        env.close()
    print('SELECTED', json.dumps(state['selected']), flush=True)


if __name__ == '__main__':
    main()
