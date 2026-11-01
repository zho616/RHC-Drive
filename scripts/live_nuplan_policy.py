"""Live native-window and browser visualization of a frozen nuPlan policy."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
import time


PAGE = '''<!doctype html><meta charset="utf-8"><title>nuPlan live policy</title>
<style>body{background:#142030;color:#edf4fb;font:16px system-ui;margin:20px}button,select,input{font:inherit;padding:7px;margin:4px}img{width:100%;max-width:1400px;border:1px solid #456}pre{white-space:pre-wrap}small{color:#b5c9dd}</style>
<h2>nuPlan · R2Dreamer+Context · best 925000 · LIVE</h2>
<label>Chassis <select id="pool"><option>ID: Default/M/L</option><option>S</option><option>XL</option></select></label>
<label>Scene <input id="seed" type="number" min="0" max="99" value="0" style="width:65px"></label>
<button onclick="send({seed:+seed.value,pool:pool.selectedIndex})">Load / restart</button>
<button onclick="send({next:true})">Next scene</button><button onclick="send({pause:true})">Pause / resume</button>
<p><small>Native 3D + top-down trajectory · frozen policy controls ego · recorded traffic replay · original vector observations</small></p>
<img src="/stream"><pre id="status">Loading policy...</pre>
<script>const seed=document.getElementById('seed'),pool=document.getElementById('pool');
async function send(data){await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
setInterval(async()=>{try{let s=await(await fetch('/state')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);}catch(e){}},1000);</script>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--scenario-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=6020)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_root.resolve()))
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from panda3d.core import loadPrcFileData
    from dreamer import Dreamer
    from envs.nuplan import NuPlanDrive

    loadPrcFileData('', 'audio-library-name null')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load(args.checkpoint.parent / '.hydra/config.yaml')
    config.device = args.device
    OmegaConf.resolve(config)
    config.model.compile = False
    config.env.scenario_eval_dir = str(args.scenario_dir.resolve())
    config.env.eval_num_scenarios = 100
    config.env.eval_episode_num = 100
    config.env.eval_env_num = 1
    # Enable the native window while retaining the native vector observer.
    env = NuPlanDrive(config.env, evaluation=True, use_render=True)
    assert env.observation_space['state'].shape == (271,)
    model = Dreamer(config.model, env.observation_space, env.action_space).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload['agent_state_dict'], strict=True)
    model.clone_and_freeze()
    model.eval()
    checkpoint_step = int(payload['step'])
    del payload
    shared = dict(jpeg=b'', state={}, pending={}, closed=False)
    condition = threading.Condition()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_GET(self):
            if self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    while not shared['closed']:
                        with condition:
                            condition.wait(timeout=2)
                            jpeg = shared['jpeg']
                        if jpeg:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                             str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == '/state':
                with condition:
                    content = json.dumps(shared['state']).encode()
                kind = 'application/json'
            elif self.path == '/frame.jpg':
                with condition:
                    content = shared['jpeg']
                kind = 'image/jpeg'
            else:
                content = PAGE.encode()
                kind = 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != '/control':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1024:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if 'seed' in data and not 0 <= int(data['seed']) < 100:
                    raise ValueError('Scene must be 0..99')
                if 'pool' in data and int(data['pool']) not in (0, 1, 2):
                    raise ValueError('Invalid chassis pool')
                with condition:
                    shared['pending'].update({k: v for k, v in data.items()
                                              if k in ('seed', 'pool', 'next', 'pause')})
                self.send_response(204)
                self.end_headers()
            except (ValueError, TypeError):
                self.send_error(400)

    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pools = [['default', 'm', 'l'], ['s'], ['xl']]
    seed, pool, paused = 0, 0, False
    reset, done, hold_until = True, False, 0.
    stopped = {'value': False}
    keys_bound = False
    period = float(config.env.decision_repeat) * .02
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=checkpoint_step,
        source_root=str(args.source_root.resolve()), scenario_dir=str(args.scenario_dir.resolve()),
        original_state_dim=271, device=args.device, inference_seed_offset=12345,
        target_fps=1 / period, url=f'http://192.168.118.78:{args.port}/',
        note='Live qualitative visualization; not a replacement for headless benchmark results.'), indent=2))
    try:
        with torch.inference_mode(), (args.output / 'episodes.jsonl').open('a', buffering=1) as log:
            while not stopped['value']:
                started = time.monotonic()
                with condition:
                    pending = shared['pending']
                    shared['pending'] = {}
                if pending.get('pause'):
                    paused = not paused
                if 'pool' in pending:
                    pool = int(pending['pool'])
                    reset = True
                if 'seed' in pending:
                    seed = int(pending['seed'])
                    reset = True
                if pending.get('next'):
                    seed = (seed + 1) % 100
                    reset = True
                if done and time.monotonic() >= hold_until and not paused:
                    seed = (seed + 1) % 100
                    reset = True
                if reset:
                    env._env.config['chassis_spec']['models'] = pools[pool]
                    torch.manual_seed(seed + 12345)
                    np.random.seed(seed + 12345)
                    obs = env.reset(seed=seed)
                    if not keys_bound:
                        env._env.engine.accept('q', lambda: stopped.update(value=True))
                        env._env.engine.accept('escape', lambda: stopped.update(value=True))
                        keys_bound = True
                    state = model.get_initial_state(1)
                    assert obs['state'].shape == (271,)
                    score, step, info, done = 0., 0, {}, False
                    chassis = dict(env.chassis_parameters)
                    action_array = np.zeros(2)
                    reset = False
                if not paused and not done:
                    tensors = {}
                    for key, value in obs.items():
                        tensor = torch.as_tensor(np.asarray(value), device=args.device)
                        tensors[key] = tensor.reshape(1, 1) if tensor.ndim == 0 else tensor.unsqueeze(0)
                    action, state = model.act(TensorDict(tensors, batch_size=(1,)), state, eval=True)
                    action_array = action[0].cpu().numpy()
                    obs, reward, done, info = env.step(action_array)
                    score += float(reward)
                    step += 1
                    if done:
                        hold_until = time.monotonic() + 2
                        row = dict(seed=seed, pool=pool, chassis=chassis, steps=step,
                                   reward=score, success=bool(info.get('arrive_dest')),
                                   crash=bool(info.get('crash')), out_of_road=bool(info.get('out_of_road')))
                        log.write(json.dumps(row) + '\n')
                        print(json.dumps(row), flush=True)
                env._env.engine.taskMgr.step()
                env._env.engine.graphicsEngine.renderFrame()
                texture = env._env.engine.win.getScreenshot()
                rgb = np.frombuffer(texture.getRamImageAs('RGB'), np.uint8).reshape(
                    texture.getYSize(), texture.getXSize(), 3)[::-1]
                top = env.render(mode='topdown', window=False, screen_size=(440, 600),
                                 film_size=(2000, 2000), scaling=3,
                                 draw_center_line=True, draw_target_vehicle_trajectory=True)
                frame = Image.new('RGB', (1400, 600))
                frame.paste(Image.fromarray(np.ascontiguousarray(rgb)).resize((960, 600)), (0, 0))
                frame.paste(Image.fromarray(np.asarray(top, dtype=np.uint8)).convert('RGB'), (960, 0))
                buffer = io.BytesIO()
                frame.save(buffer, format='JPEG', quality=82)
                current = dict(checkpoint_step=checkpoint_step, scene=seed, pool=['ID', 'S', 'XL'][pool],
                               chassis=chassis, step=step, speed_km_h=round(float(env._env.vehicle.speed_km_h), 2),
                               route_completion=round(float(info.get('route_completion', 0)), 4),
                               reward=round(score, 2), action=action_array.tolist(),
                               paused=paused, done=bool(done), success=bool(info.get('arrive_dest')),
                               crash=bool(info.get('crash')), out_of_road=bool(info.get('out_of_road')))
                with condition:
                    shared.update(jpeg=buffer.getvalue(), state=current)
                    condition.notify_all()
                (args.output / 'status.json').write_text(json.dumps(current, indent=2))
                if step == 30:
                    frame.save(args.output / 'preview.jpg')
                time.sleep(max(0., period - (time.monotonic() - started)))
    finally:
        shared['closed'] = True
        server.shutdown()
        env.close()


if __name__ == '__main__':
    main()
