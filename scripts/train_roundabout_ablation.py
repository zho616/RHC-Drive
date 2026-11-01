"""Portable matched roundabout ablations; select methods/seeds independently on each server."""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('history_only', 'response_only', 'full_h1', 'oracle', 'raw_history', 'full_h5', 'baseline')
TESTS = (('ID', 10020, None), ('S', 10320, 's'), ('XL', 10420, 'xl'))


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def hashes(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def build_config(root, method, seed, logdir, smoke=False):
    cfg = OmegaConf.load(root / 'configs/metadrive_roundabout_500k.yaml')
    cfg.seed = seed
    cfg.env.seed = '${seed}'
    cfg.env.start_seed = -seed
    cfg.env.eval_start_seed = 10000 - seed
    cfg.logdir = str(logdir)
    enabled = method != 'baseline'
    response = method in ('response_only', 'full_h1', 'full_h5')
    exchange = method in ('full_h1', 'full_h5')
    cfg.env.context_enabled = enabled
    cfg.model.context.update(dict(enabled=enabled, use_history_context=enabled,
        source=method if method in ('oracle', 'raw_history') else 'encoded_history',
        injection='dali_shallow', actor_critic=False,
        use_response_loss=response, use_context_exchange=exchange,
        prediction_horizon=1 if method == 'full_h1' else 5,
        aux_weight=1. if method in ('history_only', 'response_only', 'full_h1', 'full_h5') else 0.,
        swap_weight=1. if exchange else 0., wm_grad_scale=1. if method == 'history_only' else 0.,
        log_grad_norms=True))
    if method == 'oracle':
        cfg.env.oracle_parameters = True
    if smoke:
        cfg.device = 'cpu'
        cfg.env.steps = 96
        cfg.env.env_num = 2
        cfg.env.eval_env_num = 1
        cfg.env.eval_episode_num = 1
        cfg.env.time_limit = 32
        cfg.env.train_ratio = 8
        cfg.batch_size = 2
        cfg.batch_length = 8
        cfg.buffer.max_size = 512
        cfg.model.compile = False
        cfg.model.deter = 32
        cfg.model.hidden = 16
        cfg.model.units = 16
        cfg.model.discrete = 4
        cfg.model.rssm.stoch = 4
        cfg.model.rssm.blocks = 4
        cfg.model.imag_horizon = 3
        cfg.trainer.eval_every = 48
        cfg.trainer.update_log_every = 32
    assert cfg.env.seed + cfg.env.start_seed == 0
    assert cfg.env.seed + cfg.env.eval_start_seed == 10000
    return cfg


def prepare(root, output, methods, seeds, gpu, smoke):
    output.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'controls', 'configs', 'training', 'evaluation', 'tables'):
        (output / name).mkdir()
    for p in root.glob('*.py'):
        shutil.copy2(p, output / 'source' / p.name)
    for folder in ('configs', 'envs', 'optim'):
        shutil.copytree(root / folder, output / 'source' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(root / 'pyproject.toml', output / 'source/pyproject.toml')
    shutil.copy2(root / 'scripts/evaluate_metadrive.py', output / 'controls/evaluate_metadrive.py')
    shutil.copy2(Path(__file__), output / 'controls/train_roundabout_ablation.py')
    jobs = []
    for seed in seeds:
        for method in methods:
            name = f'seed{seed}_{method}'
            run = output / 'training' / name
            cfg = build_config(root, method, seed, run, smoke)
            OmegaConf.save(cfg, output / 'configs' / (name + '.yaml'))
            train = [sys.executable, '-u', str(output / 'source/train.py'),
                     '--config-path', str(output / 'configs'), '--config-name', name]
            evaluations = []
            for condition, start, model in TESTS:
                command = [sys.executable, '-u', str(output / 'controls/evaluate_metadrive.py'),
                    '--checkpoint', str(run / 'best.pt'), '--source-root', str(output / 'source'),
                    '--output', str(output / 'evaluation' / name / condition),
                    '--episodes', '2' if smoke else '100', '--seed-start', str(start),
                    '--environment-seed', '0', '--record-lane-metrics', '--device', 'cpu']
                if model:
                    command += ['--eval-models', model]
                evaluations.append(dict(condition=condition, command=command))
            jobs.append(dict(name=name, method=method, seed=seed, run=str(run),
                             budget=int(cfg.env.steps), train=train, evaluations=evaluations))
    from metadrive.envs import metadrive_env
    runtime_path = Path(metadrive_env.__file__).resolve().parents[1]
    runtime_hashes = {str(p.relative_to(runtime_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(runtime_path.rglob('*.py')) if '__pycache__' not in p.parts}
    plan = dict(created_at=now(), methods=methods, seeds=seeds, gpu=gpu, smoke=smoke, jobs=jobs,
        source_sha256=hashes(output / 'source'), control_sha256=hashes(output / 'controls'),
        config_sha256=hashes(output / 'configs'), metadrive_module=str(runtime_path),
        metadrive_source_sha256=runtime_hashes,
        selection='Validation success, completion, return; independent tests never select weights',
        test_episodes_per_condition=2 if smoke else 100)
    save(output / 'plan.json', plan)
    save(output / 'status.json', dict(phase='prepared',
        jobs={j['name']: dict(state='pending', stages={}) for j in jobs}))
    return plan


def verify(output, plan):
    for folder, key in [('source', 'source_sha256'), ('controls', 'control_sha256'), ('configs', 'config_sha256')]:
        if hashes(output / folder) != plan[key]:
            raise RuntimeError('Frozen files changed: ' + folder)


def summarize(output, plan):
    rows = []
    for job in plan['jobs']:
        for condition, _, _ in TESTS:
            path = output / 'evaluation' / job['name'] / condition / 'summary.json'
            if path.exists():
                value = json.loads(path.read_text())
                if value['status'] == 'completed':
                    rows.append(dict(method=job['method'], training_seed=job['seed'], condition=condition,
                        checkpoint_step=value['checkpoint_step'], **value['summary']))
    if rows:
        with (output / 'tables/test_results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(output, plan):
    lock = (output / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify(output, plan)
    state = json.loads((output / 'status.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['gpu']), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    # Ensure preflight sees the same visible GPU as the child training processes.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(plan['gpu'])
    import torch
    if not plan['smoke'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --smoke only for the CPU functional check')
    state.update(phase='running', runner_pid=os.getpid(), started_at=now())
    save(output / 'status.json', state)
    try:
        for job in plan['jobs']:
            entry = state['jobs'][job['name']]
            if entry['state'] == 'completed':
                continue
            verify(output, plan)
            state['current_job'] = job['name']
            entry['state'] = 'running'
            save(output / 'status.json', state)
            stages = [('training', job['train'])] + [('test_' + e['condition'], e['command']) for e in job['evaluations']]
            for stage, command in stages:
                info = entry['stages'].get(stage, {})
                if info.get('state') == 'verified':
                    continue
                if stage == 'training' and Path(job['run']).exists():
                    raise RuntimeError('Partial run requires review; no automatic restart: ' + job['run'])
                with (output / (job['name'] + '_' + stage + '.log')).open('a') as stream:
                    process = subprocess.Popen(command, cwd=output / 'source', env=env,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    info = dict(state='running', pid=process.pid, started_at=now(), command=command)
                    entry['stages'][stage] = info
                    save(output / 'status.json', state)
                    code = process.wait()
                if code:
                    info.update(state='failed', returncode=code)
                    raise RuntimeError(f'{job["name"]} {stage} exited {code}; see stage log')
                if stage == 'training':
                    checkpoint = torch.load(Path(job['run']) / 'latest.pt', map_location='cpu', weights_only=False)
                    if checkpoint['step'] < job['budget'] or checkpoint['update_count'] < 1:
                        raise RuntimeError('Incomplete training budget or updates')
                    if not all(torch.isfinite(v).all() for v in checkpoint['agent_state_dict'].values() if torch.is_tensor(v)):
                        raise RuntimeError('Nonfinite weights')
                    actual = [int(v['step']) for v in checkpoint['optims_state_dict']['_optimizer']['state'].values() if 'step' in v]
                    if not actual or max(actual) < 1:
                        raise RuntimeError('No actual optimizer steps')
                    info.update(final_step=checkpoint['step'], actual_optimizer_steps=max(actual))
                    del checkpoint
                else:
                    result = json.loads((output / 'evaluation' / job['name'] / stage[5:] / 'summary.json').read_text())
                    if result['status'] != 'completed' or result['completed_episodes'] != plan['test_episodes_per_condition']:
                        raise RuntimeError('Incomplete independent test')
                info.update(state='verified', returncode=code, finished_at=now())
                save(output / 'status.json', state)
            entry.update(state='completed', finished_at=now())
            save(output / 'status.json', state)
            summarize(output, plan)
        state.update(phase='completed', finished_at=now())
        save(output / 'status.json', state)
    except BaseException as error:
        state.update(phase='failed', reason=repr(error), failed_at=now())
        save(output / 'status.json', state)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=list(METHODS[:5]))
    parser.add_argument('--seeds', nargs='+', type=int, default=[2])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print recipe summary; create nothing')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze code/configs without training')
    parser.add_argument('--run-prepared', action='store_true', help='Run a prepared queue using its saved plan')
    args = parser.parse_args()
    if args.gpu < 0 or any(seed < 0 for seed in args.seeds):
        parser.error('GPU index and seeds must be nonnegative')
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Methods and seeds must not repeat')
    output = args.output.expanduser().resolve()
    if args.run_prepared:
        plan = json.loads((output / 'plan.json').read_text())
    else:
        if output.exists():
            parser.error('Use a fresh output directory')
        if args.dry_run:
            jobs = []
            for seed in args.seeds:
                for method in args.methods:
                    cfg = build_config(ROOT, method, seed, output / 'training' / f'seed{seed}_{method}', args.smoke)
                    jobs.append(dict(method=method, seed=seed, budget=cfg.env.steps,
                        train_start=cfg.env.seed + cfg.env.start_seed,
                        validation_start=cfg.env.seed + cfg.env.eval_start_seed,
                        context=OmegaConf.to_container(cfg.model.context, resolve=True)))
            print(json.dumps(jobs, indent=2))
            return
        plan = prepare(ROOT, output, args.methods, args.seeds, args.gpu, args.smoke)
    if not args.prepare_only:
        run(output, plan)


if __name__ == '__main__':
    main()
