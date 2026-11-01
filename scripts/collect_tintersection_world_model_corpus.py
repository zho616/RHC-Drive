"""Collect a common T-intersection trajectory corpus with two frozen reference policies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maps', type=int, default=10)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'episodes').mkdir()
    sys.path.insert(0, str((args.training_root/'source').resolve()))
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from dreamer import Dreamer
    from envs.metadrive import MetaDrive
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cfg = OmegaConf.load(args.training_root/'r2dreamer_dali_shallow/.hydra/config.yaml')
    cfg.device = args.device
    OmegaConf.resolve(cfg)
    cfg.model.compile = False
    assert cfg.env.map_block == 'T' and cfg.env.context_enabled
    cfg.env.eval_start_seed = 30000-int(cfg.env.seed)
    cfg.env.eval_num_scenarios = args.maps
    cfg.env.eval_env_num = 1
    cfg.env.eval_episode_num = args.maps
    env = MetaDrive(cfg.env, evaluation=True)
    models, provenance = {}, {}
    for label, folder in [('baseline', 'r2dreamer'), ('11_shallow', 'r2dreamer_dali_shallow')]:
        c = OmegaConf.load(args.training_root/folder/'.hydra/config.yaml')
        c.device = args.device
        OmegaConf.resolve(c)
        c.model.compile = False
        checkpoint = args.training_root/folder/'best.pt'
        payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model = Dreamer(c.model, env.observation_space, env.action_space).to(args.device)
        model.load_state_dict(payload['agent_state_dict'], strict=True)
        model.clone_and_freeze()
        model.eval()
        models[label] = model
        provenance[label] = {'checkpoint': str(checkpoint.resolve()), 'step': payload['step'],
                             'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        del payload
    (args.output/'provenance.json').write_text(json.dumps({
        'models': provenance, 'map_block': 'T', 'map_seeds': list(range(30000, 30000+args.maps)),
        'traffic_density': float(cfg.env.traffic_density), 'mass_scales': list(cfg.env.chassis.eval_mass_scales),
        'friction_scales': list(cfg.env.chassis.eval_friction_scales),
        'protocol': 'Two frozen policies; each sees five types on same map seeds and paired parameter multipliers. Original random mass/friction grids retained, max steering nominal. Natural rollouts; no speed/angle intervention. Each world model evaluated on the entire same corpus, including both collectors.'}, indent=2))
    episodes = []
    try:
        with torch.inference_mode():
            for map_id in range(args.maps):
                seed = 30000+map_id
                for vehicle in ['default', 'm', 'l', 's', 'xl']:
                    for policy, model in models.items():
                        env._env.config['chassis_spec']['models'] = [vehicle]
                        torch.manual_seed(seed+12345)
                        np.random.seed(seed+12345)
                        obs = env.reset(seed=seed)
                        chassis = dict(env.chassis_parameters)
                        state = model.get_initial_state(1)
                        arrays = {k: [] for k in obs}
                        arrays.update(executed_action=[], speed=[], wheel_degrees=[], reward=[])

                        def append(observation, reward=0.):
                            for k,v in observation.items():
                                arrays[k].append(np.asarray(v).copy())
                            car = env._env.vehicle
                            arrays['speed'].append(float(np.linalg.norm(car.velocity)))
                            arrays['wheel_degrees'].append(float(np.mean([car.system.getSteeringValue(i) for i in (0, 1)])))
                            arrays['reward'].append(float(reward))
                        append(obs)
                        for step in range(int(cfg.env.time_limit)):
                            td = TensorDict({k: torch.as_tensor(np.asarray(v), device=args.device).reshape(1, *np.asarray(v).shape)
                                             for k,v in obs.items()}, batch_size=[1])
                            action, state = model.act(td, state, eval=True)
                            act = action[0].cpu().numpy()
                            arrays['executed_action'].append(act.copy())
                            obs, reward, done, info = env.step(act)
                            append(obs, reward)
                            if done:
                                break
                        assert done
                        arrays['executed_action'].append(np.zeros(2, np.float32))
                        name = f'{len(episodes):04d}.npz'
                        np.savez_compressed(args.output/'episodes'/name, **{k: np.array(v) for k,v in arrays.items()})
                        episodes.append({'file': name, 'map_id': map_id, 'seed': seed, 'model': vehicle,
                                         'policy': policy, 'length': step+1, 'chassis': chassis,
                                         'success': bool(info.get('arrive_dest', False))})
                        (args.output/'episodes.json').write_text(json.dumps(episodes, indent=2))
                        print('collected', len(episodes), '/', args.maps*10, vehicle, policy, 'steps', step+1, flush=True)
    finally:
        env.close()
    origins = []
    for ei, episode in enumerate(episodes):
        d = np.load(args.output/'episodes'/episode['file'])
        end = episode['length']
        for t in sorted(set(range(20, end, 10)) | {end-1, end-5, end-10}):
            if t<20 or not d['ctx_mask'][t].all() or d['is_last'][t] or d['ctx_collision'][t]:
                continue
            origins.append({'episode': ei, 't': t, 'map_id': episode['map_id'], 'model': episode['model'], 'policy': episode['policy']})
    assert len(episodes) == args.maps*10 and len(origins)>0
    (args.output/'origins.json').write_text(json.dumps(origins, indent=2))
    print('Completed:', len(episodes), 'episodes;', len(origins), 'forecast origins', flush=True)


if __name__ == '__main__':
    main()
