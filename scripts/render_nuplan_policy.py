#!/usr/bin/env python3
"""Render a frozen nuPlan policy without replacing its vector observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--scenario-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--inference-seed-offset', type=int, default=12345)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Episodes/max-steps must be positive; seed-start must be nonnegative')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from PIL import Image, ImageDraw, ImageFont
    from dreamer import Dreamer
    import envs.nuplan as nuplan
    from metadrive.obs.state_obs import LidarStateObservation
    from panda3d.core import loadPrcFileData

    loadPrcFileData('', 'audio-library-name null')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    checkpoint = args.checkpoint.resolve()
    config = OmegaConf.load(checkpoint.parent/'.hydra/config.yaml')
    config.device = args.device
    OmegaConf.resolve(config)
    assert config.env.task == 'metadrive_nuplan'
    config.model.compile = False
    config.env.scenario_eval_dir = str(args.scenario_dir.resolve())
    config.env.eval_num_scenarios = max(100, args.seed_start + args.episodes)
    config.env.eval_episode_num = args.episodes
    config.env.eval_env_num = 1
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    width, height, side, header, footer = 1024, 640, 512, 40, 56
    canvas_size = (width + side, header + height + footer)
    fps = 1 / (float(config.env.decision_repeat) * .02)
    assert fps == 10
    native = nuplan.NuPlanPaperEnv
    reference = nuplan.NuPlanDrive(config.env, evaluation=True)
    expected_state_shape = reference.observation_space['state'].shape
    reference.close()

    class OffscreenNuPlan(native):
        def __init__(self, env_config):
            # Graphics are enabled through image_observation, but the explicit
            # observer keeps this policy's original LiDAR state unchanged.
            env_config.update(use_render=False, image_observation=True,
                              agent_observation=LidarStateObservation,
                              sensors={'main_camera': ('MainCamera', width, height)},
                              window_size=(width, height), show_interface=False,
                              show_fps=False, camera_height=4.0, camera_dist=11.0,
                              use_chase_camera_follow_lane=False)
            super().__init__(env_config)

    nuplan.NuPlanPaperEnv = OffscreenNuPlan
    env = nuplan.NuPlanDrive(config.env, evaluation=True)
    assert env.observation_space['state'].shape == expected_state_shape
    model = Dreamer(config.model, env.observation_space, env.action_space).to(args.device)
    model.load_state_dict(payload['agent_state_dict'], strict=True)
    model.clone_and_freeze()
    model.eval()
    best_step = payload.get('step')
    del payload
    OmegaConf.save(config, args.output/'policy_config.yaml')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    episodes = []
    videos = []
    started = time.monotonic()
    provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint,
                      checkpoint_step=best_step, source_root=str(source),
                      source_hashes={p:hashlib.sha256((source/p).read_bytes()).hexdigest()
                                     for p in ('dreamer.py', 'context.py', 'context_rssm.py', 'envs/nuplan.py')},
                      scenario_dir=str(args.scenario_dir.resolve()),
                      selection='Consecutive held-out scenario indices; no outcome-based selection',
                      seed_start=args.seed_start, episodes_requested=args.episodes,
                      inference_seed_offset=args.inference_seed_offset, inference_device=args.device,
                      observation=f'Original {expected_state_shape[0]}-D vector state plus original history/context keys; no image input',
                      policy='Actor mode; seeded stochastic RSSM posterior; frozen parameters',
                      render='Native MetaDrive offscreen chase view plus top-down view',
                      fps=fps, resolution=list(canvas_size))
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')

    def frame(seed, step, score, info, action):
        env._env.engine.graphicsEngine.renderFrame()
        texture = env._env.engine.win.getDisplayRegion(1).getScreenshot()
        rgb = np.frombuffer(texture.getRamImageAs('RGB'), dtype=np.uint8).reshape(
            texture.getYSize(), texture.getXSize(), 3)[::-1]
        view = Image.fromarray(np.ascontiguousarray(rgb)).resize((width, height))
        top = env.render(mode='topdown', window=False, screen_size=(side, height),
                         film_size=(2000, 2000), scaling=3, draw_center_line=True,
                         draw_target_vehicle_trajectory=True)
        output = Image.new('RGB', canvas_size, '#142030')
        output.paste(view, (0, header))
        output.paste(Image.fromarray(np.asarray(top, dtype=np.uint8)).convert('RGB'), (width, header))
        draw = ImageDraw.Draw(output)
        draw.text((14, 9), f'nuPlan Pittsburgh | R2Dreamer+Context | best {best_step} | test scene {seed}', font=font, fill='white')
        draw.text((width+14, 9), 'Top-down trajectory', font=small, fill='#C6DBF0')
        speed = float(env._env.vehicle.speed_km_h)
        route = float(info.get('route_completion', 0))
        y = header + height + 5
        draw.text((14, y), f't={step/fps:.1f}s   speed={speed:.1f} km/h   route={route:.1%}   return={score:.2f}', font=small, fill='white')
        outcome = ('SUCCESS' if info.get('arrive_dest') else 'COLLISION' if info.get('crash') else
                   'OUT OF ROAD' if info.get('out_of_road') else 'RUNNING')
        draw.text((14, y+24), f'steer={action[0]:+.3f}   throttle/brake={action[1]:+.3f}   {outcome}   | policy-controlled ego; replay traffic', font=small, fill='#C6DBF0')
        return output

    try:
        with torch.inference_mode(), (args.output/'trace.jsonl').open('w') as trace:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                torch.manual_seed(seed + args.inference_seed_offset)
                np.random.seed(seed + args.inference_seed_offset)
                obs = env.reset(seed=seed)
                assert obs['state'].shape == expected_state_shape and 'ctx_history' in obs
                state = model.get_initial_state(1)
                chassis = env.chassis_parameters
                scenario_id = env._env.engine.data_manager.current_scenario_id
                score, speeds, info, done, step = 0.0, [], {}, False, 0
                video = args.output/f'scene_{seed:03d}.mp4'
                command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                           '-s',f'{canvas_size[0]}x{canvas_size[1]}','-r',str(fps),'-i','-',
                           '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20',
                           '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
                try:
                    im = frame(seed, 0, score, info, [0,0])
                    im.save(args.output/f'scene_{seed:03d}_start.png')
                    encoder.stdin.write(im.tobytes())
                    limit = min(int(config.env.time_limit), args.max_steps or int(config.env.time_limit))
                    for step in range(1, limit+1):
                        tensors = {}
                        for key, value in obs.items():
                            value = torch.as_tensor(np.asarray(value), device=args.device)
                            tensors[key] = value.reshape(1,1) if value.ndim == 0 else value.unsqueeze(0)
                        batch = TensorDict(tensors, batch_size=(1,), device=args.device)
                        action, state = model.act(batch, state, eval=True)
                        action = action[0].cpu().numpy()
                        obs, reward, done, info = env.step(action)
                        score += float(reward)
                        speeds.append(float(env._env.vehicle.speed_km_h))
                        im = frame(seed, step, score, info, action)
                        encoder.stdin.write(im.tobytes())
                        position = np.asarray(env._env.vehicle.position).tolist()
                        trace.write(json.dumps(dict(seed=seed,step=step,action=action.tolist(),position=position,
                                                    reward=float(reward),speed_km_h=speeds[-1],
                                                    route_completion=float(info.get('route_completion',0))))+'\n')
                        if step == 30:
                            im.save(args.output/f'scene_{seed:03d}_preview.png')
                        if done:break
                    capped = not done
                    # Hold the actual final frame for one second for inspection.
                    for _ in range(int(fps)):encoder.stdin.write(im.tobytes())
                    im.save(args.output/f'scene_{seed:03d}_end.png')
                finally:
                    encoder.stdin.close()
                    if encoder.wait() != 0:raise RuntimeError('Video encoding failed')
                row=dict(seed=seed,scenario_id=scenario_id,chassis=chassis,steps=step,reward=score,
                         success=bool(info.get('arrive_dest',False)),crash=bool(info.get('crash',False)),
                         out_of_road=bool(info.get('out_of_road',False)),
                         timeout=bool(obs['is_last'] and not obs['is_terminal']),
                         visualization_step_cap=capped,route_completion=float(info.get('route_completion',0)),
                         mean_speed_km_h=float(np.mean(speeds)),video=str(video.resolve()))
                episodes.append(row);videos.append(video)
                (args.output/'episodes.json').write_text(json.dumps(episodes,indent=2)+'\n')
                print(json.dumps(row),flush=True)
        listing=args.output/'concat.txt'
        listing.write_text(''.join(f"file '{p.resolve()}'\n" for p in videos))
        combined=args.output/'nuplan_r2dreamer_context_best.mp4'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
                        '-c','copy','-movflags','+faststart',str(combined)],check=True)
        summary=dict(provenance,status='completed',elapsed_seconds=time.monotonic()-started,
                     completed_episodes=len(episodes),episodes=episodes,combined_video=str(combined.resolve()),
                     note='Qualitative consecutive-scene rendering, not a replacement for the independent 100-episode benchmark.')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('VIDEO',combined,flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
