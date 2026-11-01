"""Add the compact geometry oracle to pending jobs without restarting active training."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import queue_roundabout_ablations as queue
from vehicle_parameters import ORACLE_SCHEMA


def prepare(out):
    plan = queue.verify(out)
    state = queue.read(out / 'status.json')
    oracle_jobs = [job for job in plan['jobs'] if job['method'] == 'oracle']
    assert len(oracle_jobs) == 3
    assert all(state['jobs'][job['name']]['state'] == 'pending' for job in oracle_jobs)
    revision = out / 'revisions/oracle10_v2'
    revision.mkdir(parents=True, exist_ok=False)
    for name in ('plan.json', 'status.json', 'launch.json', 'PLAN.zh.md'):
        shutil.copy2(out / name, revision / ('previous_' + name))
    source = out / 'source_oracle10_v2'
    shutil.copytree(out / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changes = ('context.py', 'vehicle_parameters.py', 'envs/metadrive.py',
               'tests/test_direct_conditions.py', 'tests/test_vehicle_parameters.py')
    for rel in changes:
        shutil.copy2(ROOT / rel, source / rel)
    control = out / 'controls/queue_roundabout_ablations_v2.py'
    shutil.copy2(ROOT / 'scripts/queue_roundabout_ablations.py', control)
    modified = copy.deepcopy(plan)
    modified.setdefault('additional_sources', {})[source.name] = queue.fingerprints(source)
    added_configs = []
    for job in modified['jobs']:
        if job['method'] != 'oracle':
            continue
        cfg = OmegaConf.load(out / 'configs' / (job['name'] + '.yaml'))
        cfg.env.oracle_parameters = True
        name = job['name'] + '10'
        config_path = out / 'configs' / (name + '.yaml')
        OmegaConf.save(cfg, config_path)
        added_configs.append(config_path)
        job['train'][2] = str(source / 'train.py')
        job['train'][-1] = name
        for evaluation in job['evaluations']:
            command = evaluation['command']
            command[command.index('--source-root') + 1] = str(source)
    # Check the expanded oracle on GPU just before its first formal job, preserving serial execution.
    cfg = OmegaConf.load(out / 'configs/seed0_4_oracle10.yaml')
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
    cfg.logdir = str(out / 'smoke/oracle10_v2')
    smoke_config = out / 'configs/smoke_oracle10_v2.yaml'
    OmegaConf.save(cfg, smoke_config)
    added_configs.append(smoke_config)
    first_oracle = next(job for job in modified['jobs'] if job['method'] == 'oracle')
    first_oracle['preflights'] = [dict(name='smoke_oracle10_v2', logdir=cfg.logdir, budget=256,
        command=[sys.executable, '-u', str(source / 'train.py'), '--config-path',
                 str(out / 'configs'), '--config-name', 'smoke_oracle10_v2'])]
    for p in [control] + added_configs:
        modified['artifacts_sha256'][str(p.relative_to(out))] = queue.sha(p)
    modified['condition_definitions']['oracle'] = (
        '10D actual mass, max steering, wheel friction, length, width, wheelbase, track, '
        'wheel radius, max engine force, max brake force; no learned encoder or model ID')
    modified['oracle_schema'] = [dict(name=name, scale=scale, unit=unit) for name, scale, unit in ORACLE_SCHEMA]
    modified['limitations'] = [line for line in modified['limitations']
                               if not line.startswith(('Oracle is privileged', 'Condition dimensions'))]
    modified['limitations'] += ['Oracle covers ten main structural/control parameters, not exhaustive vehicle physics.',
                                'Condition dimensions 8/10/35 differ; parameter counts are not matched.']
    modified['oracle_revision'] = dict(created_at=queue.now(), source=str(source),
        change='Expand pending oracle jobs from 3 to 10 physical parameters; preserve other jobs and active PID',
        audit_logs=['validation/oracle10_unit_tests.log', 'validation/oracle10_physics_tests.log'])
    queue.save(revision / 'plan.next.json', modified)
    shutil.copy2('/tmp/rhc_oracle10_tests.log', out / 'validation/oracle10_unit_tests.log')
    shutil.copy2('/tmp/rhc_oracle10_physics.log', out / 'validation/oracle10_physics_tests.log')
    queue.save(revision / 'preparation.json', dict(control=str(control), source=str(source), changes=changes,
        previous_plan_sha256=queue.sha(out / 'plan.json'), next_plan_sha256=queue.sha(revision / 'plan.next.json')))
    print(revision)


def handoff(out):
    revision = out / 'revisions/oracle10_v2'
    preparation = queue.read(revision / 'preparation.json')
    assert queue.sha(out / 'plan.json') == preparation['previous_plan_sha256']
    previous = queue.verify(out)
    modified = queue.read(revision / 'plan.next.json')
    for folder, hashes in modified['additional_sources'].items():
        assert queue.fingerprints(out / folder) == hashes
    for rel, expected in modified['artifacts_sha256'].items():
        assert queue.sha(out / rel) == expected
    controller = queue.read(out / 'launch.json')['pid']
    controller_id = queue.process_identity(controller)
    assert str(out / 'controls/queue_roundabout_ablations.py') in controller_id['command']
    # Pause only the waiting controller. Its training child and all env workers continue.
    os.kill(controller, signal.SIGSTOP)
    try:
        state = queue.read(out / 'status.json')
        assert state['phase'] == 'running'
        name = state['current_job']
        job = next(j for j in previous['jobs'] if j['name'] == name)
        assert job['method'] != 'oracle'
        assert all(state['jobs'][j['name']]['state'] == 'pending'
                   for j in previous['jobs'] if j['method'] == 'oracle')
        training = state['jobs'][name]['stages']['training']
        assert training['state'] == 'running'
        identity = queue.process_identity(training['pid'])
        assert identity['command'] == job['train'] and identity['process_state'] != 'Z'
        training['adopt_identity'] = identity
    except BaseException:
        os.kill(controller, signal.SIGCONT)
        raise
    # Retire the supervisor, leaving the exact training process running.
    os.kill(controller, signal.SIGTERM)
    os.kill(controller, signal.SIGCONT)
    for _ in range(50):
        try:
            if queue.process_identity(controller)['process_state'] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Old controller did not retire; active training remains unchanged')
    queue.save(out / 'plan.json', modified)
    state['controller_handoff'] = dict(previous_pid=controller, active_training=identity,
        reason='User requested compact geometry oracle; active training is not restarted', at=queue.now())
    queue.save(out / 'status.json', state)
    queue.verify(out)
    command = [sys.executable, '-u', preparation['control'], 'run', '--output', str(out)]
    log = out / 'runner_oracle10_v2.log'
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    launch = dict(pid=process.pid, command=command, log=str(log), adopted_training=identity)
    queue.save(out / 'launch.json', launch)
    queue.save(revision / 'handoff.json', launch)
    print(json.dumps(launch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'handoff'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    globals()[args.action](args.output.resolve())
