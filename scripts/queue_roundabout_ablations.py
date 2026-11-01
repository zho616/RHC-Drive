"""Freeze and run five roundabout ablations, seeds 0/1/2, strictly serially."""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'dreamer.py').is_file() and (p / 'iv_output').is_dir())
TEMPLATE = ROOT / 'logdir/main_comparison_seed2_20261029/configs/roundabout_B_r2dreamer_context.yaml'
METHODS = ('history_only', 'response_only', 'full_h1', 'oracle', 'raw_history')
TESTS = (('ID', 10020, None), ('S', 10320, 's'), ('XL', 10420, 'xl'))


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def method_config(template, method, seed, logdir):
    cfg = copy.deepcopy(template)
    cfg.seed = seed
    cfg.env.seed = '${seed}'
    cfg.env.start_seed = -seed
    cfg.env.eval_start_seed = 10000 - seed
    cfg.logdir = str(logdir)
    cfg.hydra = dict(run=dict(dir='${logdir}'))
    cfg.model.context.update(dict(enabled=True, use_history_context=True,
        source='encoded_history', injection='dali_shallow', actor_critic=False,
        use_response_loss=method in ('response_only', 'full_h1'),
        use_context_exchange=method == 'full_h1',
        prediction_horizon=1 if method == 'full_h1' else 5,
        aux_weight=1. if method in ('history_only', 'response_only', 'full_h1') else 0.,
        swap_weight=1. if method == 'full_h1' else 0.,
        wm_grad_scale=1. if method == 'history_only' else 0., log_grad_norms=True))
    if method in ('oracle', 'raw_history'):
        cfg.model.context.source = method
    if method == 'oracle':
        cfg.env.oracle_parameters = True
    assert cfg.env.task == 'metadrive_roundabout' and cfg.env.map_block == 'O'
    assert cfg.env.steps == 500000 and cfg.batch_size == 16 and cfg.batch_length == 64
    assert cfg.env.num_scenarios == 100 and cfg.env.eval_num_scenarios == 20
    assert cfg.env.seed + cfg.env.start_seed == 0
    assert cfg.env.seed + cfg.env.eval_start_seed == 10000
    old_env = OmegaConf.to_container(template.env, resolve=True)
    new_env = OmegaConf.to_container(cfg.env, resolve=True)
    assert all(old_env[k] == new_env[k] for k in old_env
               if k not in ('seed', 'start_seed', 'eval_start_seed'))
    return cfg


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('source', 'configs', 'controls', 'validation', 'tables', 'training', 'evaluation', 'smoke'):
        (out / folder).mkdir()
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, out / 'source' / p.name)
    for folder in ('configs', 'envs', 'optim', 'tests'):
        shutil.copytree(ROOT / folder, out / 'source' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'pyproject.toml', out / 'source/pyproject.toml')
    shutil.copy2(ROOT / 'scripts/evaluate_metadrive.py', out / 'controls/evaluate_metadrive.py')
    shutil.copy2(Path(__file__), out / 'controls/queue_roundabout_ablations.py')
    template = OmegaConf.load(TEMPLATE)
    jobs = []
    for seed in (0, 1, 2):
        for number, method in enumerate(METHODS, 1):
            name = f'seed{seed}_{number}_{method}'
            run = out / 'training' / name
            cfg = method_config(template, method, seed, run)
            OmegaConf.save(cfg, out / 'configs' / (name + '.yaml'))
            train = [sys.executable, '-u', str(out / 'source/train.py'), '--config-path',
                     str(out / 'configs'), '--config-name', name]
            evaluations = []
            for condition, start, chassis in TESTS:
                command = [sys.executable, '-u', str(out / 'controls/evaluate_metadrive.py'),
                           '--checkpoint', str(run / 'best.pt'), '--source-root', str(out / 'source'),
                           '--output', str(out / 'evaluation' / name / condition), '--episodes', '100',
                           '--seed-start', str(start), '--environment-seed', '0',
                           '--record-lane-metrics', '--device', 'cpu']
                if chassis:
                    command += ['--eval-models', chassis]
                evaluations.append(dict(condition=condition, command=command))
            jobs.append(dict(name=name, number=number, method=method, seed=seed,
                             logdir=str(run), train=train, evaluations=evaluations))
    # Native environment + full network + actual optimizer updates, in fresh short runs.
    smoke_jobs = []
    for method in ('oracle', 'raw_history'):
        name = 'smoke_' + method
        cfg = method_config(template, method, 0, out / 'smoke' / method)
        cfg.env.steps = 256
        cfg.env.env_num = 2
        cfg.env.eval_env_num = 1
        cfg.env.eval_episode_num = 1
        cfg.env.train_ratio = 8
        cfg.batch_size = 2
        cfg.batch_length = 16
        cfg.buffer.max_size = 1000
        cfg.model.compile = False
        cfg.trainer.update_log_every = 1
        OmegaConf.save(cfg, out / 'configs' / (name + '.yaml'))
        smoke_jobs.append(dict(name=name, logdir=str(out / 'smoke' / method),
            command=[sys.executable, '-u', str(out / 'source/train.py'),
                     '--config-path', str(out / 'configs'), '--config-name', name]))
    shutil.copy2('/tmp/rhc_direct_tests.log', out / 'validation/unit_tests.log')
    plan = dict(created_at=now(), scenario='Roundabout-B, mass/friction protocol',
        seeds=[0, 1, 2], budget_per_training=500000, formal_training_count=15,
        formal_environment_steps=7500000, order='seed0 methods1-5; seed1 methods1-5; seed2 methods1-5',
        template=str(TEMPLATE), template_sha256=sha(TEMPLATE),
        source_sha256=fingerprints(out / 'source'),
        artifacts_sha256={str(p.relative_to(out)): sha(p) for folder in ('configs', 'controls')
                          for p in sorted((out / folder).rglob('*')) if p.is_file()},
        jobs=jobs, smoke_jobs=smoke_jobs,
        metadrive_root='/home/server/projects/metadrive',
        selection='Best on validation success, route completion, then return; test never selects weights',
        condition_definitions={
            'history_only': 'Encoded K5 history; WM losses -> encoder at scale1; no response/exchange predictor',
            'response_only': 'Encoded K5 history; H5 response loss; no exchange; WM -> encoder detached',
            'full_h1': 'Encoded K5 history; H1 response loss + exchange; WM -> encoder detached',
            'oracle': '10 physical parameters: mass, steering, friction, length, width, wheelbase, track, wheel radius, engine/brake limits; no model ID, encoder or auxiliaries',
            'raw_history': 'Normalized five 6D action-response tokens flattened (30D) plus 5 validity flags; no encoder or auxiliaries'},
        limitations=[
            'Oracle contains ten main physical parameters, not an exhaustive physics-engine state.',
            'Condition dimensions 8/10/35 differ, so posterior input parameter counts are not matched.',
            'All conditions, including oracle, are zeroed at episode-reset steps, matching the reference RSSM convention.',
            'Shallow changes only the posterior; future conditions are unused by prior/transition/actor/critic.',
            'History-only vs response-only changes both supervision and gradient routing.',
            'Existing baseline and Full H5 are references, not additional training jobs.'])
    save(out / 'plan.json', plan)
    save(out / 'status.json', dict(phase='prepared', created_at=now(),
        jobs={j['name']: dict(state='pending', stages={}) for j in jobs}))
    command = [sys.executable, '-u', str(out / 'controls/queue_roundabout_ablations.py'),
               'run', '--output', str(out)]
    (out / 'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
    (out / 'PLAN.zh.md').write_text(
        '# 环岛五组消融，三个训练种子\n\n'
        '顺序：seed0 第1–5组 → seed1 第1–5组 → seed2 第1–5组，全程串行。'
        '每项全新初始化、500K环境步；验证20回合/10K，按成功率、完成率、回报选best。'
        '每项结束后，固定ID/S/XL各100回合独立测试。\n\n'
        '1. History-only：世界模型损失训练历史编码器，无响应/交换损失。\n'
        '2. Response-only：H5响应预测，无交换，WM梯度截断。\n'
        '3. Full H1：单步响应预测＋交换，输入历史仍K5。\n'
        '4. Oracle：质量、转角、摩擦、车长/宽、轴距、轮距、轮半径、驱动/制动上限，10维直接输入后验。\n'
        '5. Raw history：5×6归一化历史按时序展开＋5维mask，35维直接输入后验，无编码器和辅助损失。\n\n'
        '所有组保持Shallow后验注入，不直接输入Actor/Critic。训练DEFAULT/M/L，'
        '质量与摩擦0.6–1.4（间隔0.1），最大转角保持车型名义值。'
        '训练地图0–99、验证10000–10019；训练seed偏移已补偿。'
        'Oracle为主要结构和控制参数的特权对照；输入维度不同，非参数量匹配。'
        '回合reset时条件统一清零。旧实验不修改、不续训。\n')
    print(out)


def verify(out):
    plan = read(out / 'plan.json')
    if sha(plan['template']) != plan['template_sha256']:
        raise RuntimeError('Reference template changed')
    if fingerprints(out / 'source') != plan['source_sha256']:
        raise RuntimeError('Frozen training implementation changed')
    for folder, hashes in plan.get('additional_sources', {}).items():
        if fingerprints(out / folder) != hashes:
            raise RuntimeError('Frozen additional training implementation changed: ' + folder)
    for rel, expected in plan['artifacts_sha256'].items():
        if sha(out / rel) != expected:
            raise RuntimeError('Frozen configuration/control changed: ' + rel)
    return plan


def checkpoint_check(folder, budget):
    import torch
    latest = torch.load(Path(folder) / 'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(Path(folder) / 'best.pt', map_location='cpu', weights_only=False)
    if latest['step'] < budget or latest['update_count'] < 1:
        raise RuntimeError('Training budget/update count incomplete')
    optimizer = latest['optims_state_dict']['_optimizer']['state']
    actual_steps = [int(value['step']) for value in optimizer.values() if 'step' in value]
    if not actual_steps or max(actual_steps) < 1:
        raise RuntimeError('AMP skipped all optimizer steps; attempted update count is insufficient')
    for checkpoint in (latest, best):
        if not all(torch.isfinite(v).all() for v in checkpoint['agent_state_dict'].values() if torch.is_tensor(v)):
            raise RuntimeError('Nonfinite trained weights')
    return dict(final_step=latest['step'], update_count=latest['update_count'],
                actual_optimizer_steps=max(actual_steps),
                best_step=best['step'], best_sha256=sha(Path(folder) / 'best.pt'))


def launch(command, log, env, update):
    with log.open('a', buffering=1) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        update(dict(state='running', pid=process.pid, started_at=now(), command=command, log=str(log)))
        code = process.wait()
    update(dict(state='completed' if code == 0 else 'failed', returncode=code, finished_at=now()))
    if code:
        raise RuntimeError(f'Command failed ({code}); see {log}')


def process_identity(pid):
    """PID reuse guard for adopting this queue's existing, uninterrupted child."""
    folder = Path('/proc') / str(pid)
    fields = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19],
        command=(folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
        process_state=fields[0])


def adopt_training(info, command, update):
    identity = info['adopt_identity']
    if identity['command'] != command:
        raise RuntimeError('Adopted command does not match the frozen training job')
    update(dict(state='adopted_running', adopted_at=now(), exit_code_available=False))
    while True:
        try:
            current = process_identity(identity['pid'])
        except FileNotFoundError:
            break
        if current['start_ticks'] != identity['start_ticks']:
            raise RuntimeError('Adopted training PID was reused')
        if current['process_state'] == 'Z':
            break
        if current['command'] != identity['command']:
            raise RuntimeError('Adopted training process changed identity')
        time.sleep(5)
    update(dict(state='completed', finished_at=now(),
        completion_check='Adopted process exited; require full-budget finite latest/best and actual optimizer steps'))


def report(out, plan):
    rows = []
    for job in plan['jobs']:
        for condition, _, _ in TESTS:
            path = out / 'evaluation' / job['name'] / condition / 'summary.json'
            if path.exists():
                result = read(path)
                if result['status'] == 'completed':
                    rows.append(dict(method=job['method'], training_seed=job['seed'], condition=condition,
                        checkpoint_step=result['checkpoint_step'], **result['summary']))
    if rows:
        with (out / 'tables/all_seed_condition_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(out):
    import torch
    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = verify(out)
    state = read(out / 'status.json')
    env = dict(os.environ, PYTHONPATH=plan['metadrive_root'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/rhc-roundabout-matplotlib', HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    state.update(phase='preflight', runner_pid=os.getpid(), started_at=now())
    save(out / 'status.json', state)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; formal CPU training is not substituted')
        save(out / 'validation/runtime.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, free_disk_bytes=shutil.disk_usage(out).free, checked_at=now()))
        if shutil.disk_usage(out).free < 15 * 1024 ** 3:
            raise RuntimeError('Insufficient free disk for 15 training runs')
        def smoke_update(name, info):
            state.setdefault('smoke', {}).setdefault(name, {}).update(info)
            save(out / 'status.json', state)
        for job in plan['smoke_jobs']:
            if state.get('smoke', {}).get(job['name'], {}).get('state') == 'verified':
                continue
            if Path(job['logdir']).exists():
                raise RuntimeError('Partial smoke directory requires review: ' + job['logdir'])
            launch(job['command'], out / 'validation' / (job['name'] + '.log'), env,
                   lambda info, name=job['name']: smoke_update(name, info))
            checked = checkpoint_check(job['logdir'], 256)
            smoke_update(job['name'], dict(state='verified', **checked))
        for job in plan['jobs']:
            entry = state['jobs'][job['name']]
            if entry['state'] == 'completed':
                continue
            verify(out)
            state.update(phase='running', current_job=job['name'])
            entry['state'] = 'running'
            save(out / 'status.json', state)
            def stage_update(stage, info):
                entry['stages'].setdefault(stage, {}).update(info)
                save(out / 'status.json', state)
            for preflight in job.get('preflights', []):
                stage = preflight['name']
                if entry['stages'].get(stage, {}).get('state') == 'verified':
                    continue
                if Path(preflight['logdir']).exists():
                    raise RuntimeError('Partial oracle preflight requires review')
                launch(preflight['command'], out / 'validation' / (stage + '.log'), env,
                       lambda info: stage_update(stage, info))
                checked = checkpoint_check(preflight['logdir'], preflight['budget'])
                stage_update(stage, dict(state='verified', **checked))
            training = entry['stages'].get('training', {})
            if training.get('state') != 'verified':
                if training.get('adopt_identity'):
                    adopt_training(training, job['train'], lambda info: stage_update('training', info))
                else:
                    if Path(job['logdir']).exists():
                        raise RuntimeError('Partial training requires review; no silent restart: ' + job['logdir'])
                    launch(job['train'], out / (job['name'] + '.log'), env,
                           lambda info: stage_update('training', info))
                stage_update('training', dict(state='verified', **checkpoint_check(job['logdir'], 500000)))
            for evaluation in job['evaluations']:
                stage = 'test_' + evaluation['condition']
                if entry['stages'].get(stage, {}).get('state') == 'verified':
                    continue
                launch(evaluation['command'], out / (job['name'] + '_' + stage + '.log'), env,
                       lambda info: stage_update(stage, info))
                result = read(out / 'evaluation' / job['name'] / evaluation['condition'] / 'summary.json')
                if result['status'] != 'completed' or result['completed_episodes'] != 100:
                    raise RuntimeError('Independent evaluation incomplete')
                stage_update(stage, dict(state='verified', episodes=100))
            entry.update(state='completed', finished_at=now())
            save(out / 'status.json', state)
            report(out, plan)
        state.update(phase='completed', finished_at=now())
        save(out / 'status.json', state)
    except BaseException as error:
        state.update(phase='failed', reason=repr(error), failed_at=now())
        save(out / 'status.json', state)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'run', 'report'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'run':
        run(out)
    else:
        plan = verify(out)
        if args.action == 'report':
            report(out, plan)
        print(json.dumps(dict(jobs=len(plan['jobs']), seeds=plan['seeds'], status='verified')))


if __name__ == '__main__':
    main()
