"""Queue nuPlan R2+context and matched roundabout SAC/PPO/DreamerV3 controls."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf


ORDER = ('nuplan_r2dreamer_context', 'roundabout_sac', 'roundabout_ppo',
         'roundabout_dreamerv3', 'roundabout_dreamerv3_context')
MASS_FRICTION_ORDER = ('nuplan_r2dreamer_context', 'roundabout_r2dreamer',
    'roundabout_r2dreamer_context', 'roundabout_sac', 'roundabout_ppo',
    'roundabout_dreamerv3', 'roundabout_dreamerv3_context')


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def fingerprint(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}


def freeze(source, destination):
    destination.mkdir()
    for file in source.glob('*.py'):
        shutil.copy2(file, destination/file.name)
    for name in ('envs', 'optim', 'configs'):
        shutil.copytree(source/name, destination/name, ignore=shutil.ignore_patterns('__pycache__'))
    if (source/'pyproject.toml').exists():
        shutil.copy2(source/'pyproject.toml', destination/'pyproject.toml')


def job_config(reference, run, steps, representation, context):
    config = copy.deepcopy(reference)
    config.logdir = str(run)
    config.env.steps = steps
    config.model.rep_loss = representation
    config.model.context.enabled = context
    config.model.context.injection = 'dali_shallow'
    config.model.context.actor_critic = False
    config.model.context.aux_weight = 1.
    config.model.context.swap_weight = 1. if context else 0.
    config.model.context.wm_grad_scale = 0.
    config.env.context_enabled = context
    # Keep interpolation until Hydra loads the job, including evaluation chassis pools.
    config.hydra = dict(run=dict(dir='${logdir}'))
    return config


def validate_environment(reference, candidate):
    left = OmegaConf.to_container(reference.env, resolve=True)
    right = OmegaConf.to_container(candidate.env, resolve=True)
    for env in (left, right):
        env.pop('context_enabled', None)
        env.pop('steps', None)
    if left != right:
        raise ValueError('Environment differs from its scenario reference beyond context/step budget')


def mass_friction_reference(reference):
    config = copy.deepcopy(reference)
    scales = [round(.6+.1*i,1) for i in range(9)]
    config.env.map_block = 'O'
    config.env.chassis.mass_scales = scales
    config.env.chassis.steering_scales = [1.]
    config.env.chassis.friction_scales = scales
    config.env.chassis.eval_models = '${env.chassis.models}'
    config.env.chassis.eval_mass_scales = '${env.chassis.mass_scales}'
    config.env.chassis.eval_steering_scales = '${env.chassis.steering_scales}'
    config.env.chassis.eval_friction_scales = '${env.chassis.friction_scales}'
    return config


def process_running(pid, expected_command):
    try:
        command = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    if state == 'Z' or not command:
        return False
    if command != expected_command:
        raise RuntimeError(f'PID {pid} no longer identifies the expected training command')
    return True


def prepare(args):
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_dir, controls = output/'configs', output/'controls'
    config_dir.mkdir(); controls.mkdir()
    for name in ('train_metadrive_sb3.py', 'evaluate_metadrive.py', 'run_nuplan_roundabout_queue.py'):
        shutil.copy2(repo/'scripts'/name, controls/name)
    nuplan_reference = args.nuplan_reference.resolve()
    roundabout_reference = args.roundabout_reference.resolve()
    nuplan_source, roundabout_source = output/'nuplan_source', output/'roundabout_source'
    freeze(nuplan_reference/'source', nuplan_source)
    # Mass/friction requires the physical wheel updates and four-key donor profile.
    freeze(nuplan_reference/'source' if args.roundabout_mass_friction else roundabout_reference/'source', roundabout_source)
    nuplan = OmegaConf.load(nuplan_reference/'baseline/.hydra/config.yaml')
    roundabout = OmegaConf.load(roundabout_reference/'r2dreamer_dali_shallow/.hydra/config.yaml')
    if args.roundabout_mass_friction:
        roundabout = mass_friction_reference(roundabout)
    if nuplan.env.task != 'metadrive_nuplan' or roundabout.env.task != 'metadrive_roundabout':
        raise ValueError('Wrong scenario reference')
    if not args.after_status.resolve().is_file():
        raise FileNotFoundError(args.after_status)
    jobs = []
    order = MASS_FRICTION_ORDER if args.roundabout_mass_friction else ORDER
    for name in order:
        is_nuplan, context = name.startswith('nuplan_'), name.endswith('context')
        source = nuplan_source if is_nuplan else roundabout_source
        reference = nuplan if is_nuplan else roundabout
        representation = 'r2dreamer' if 'r2dreamer' in name else 'dreamer'
        run = output/name
        config = job_config(reference, run, args.steps, representation, context)
        validate_environment(reference, config)
        config_path = config_dir/(name+'.yaml')
        OmegaConf.save(config, config_path)
        sb3 = name.rsplit('_', 1)[-1] in ('sac', 'ppo')
        base = [sys.executable, '-u']
        if sb3:
            command = base+[str(controls/'train_metadrive_sb3.py'), '--source', str(source),
                '--config', str(config_path), '--output', str(run), '--method', name.rsplit('_',1)[-1],
                '--steps', str(args.steps)]
            preflight = command.copy()
            preflight[preflight.index('--output')+1] = str(output/'validation'/name)
            preflight += ['--smoke']
        else:
            command = base+[str(source/'train.py'), '--config-path', str(config_dir), '--config-name', name]
            smoke_overrides = ['device=cuda:0', 'env.steps=128', 'env.env_num=2', 'env.eval_env_num=1',
                'env.eval_episode_num=2', 'env.time_limit=32', 'env.train_ratio=8',
                'buffer.max_size=512', 'batch_size=4', 'batch_length=8', 'model.compile=false',
                'model.deter=32', 'model.hidden=16', 'model.units=16', 'model.discrete=4',
                'model.rssm.stoch=4', 'model.rssm.blocks=4', 'model.imag_horizon=3',
                'trainer.eval_every=64', 'trainer.update_log_every=32',
                'env.chassis.models=[default]', 'env.chassis.mass_scales=[1.0]',
                'env.chassis.steering_scales=[1.0]', f'logdir={output/"validation"/name}']
            if 'friction_scales' in config.env.chassis:
                smoke_overrides.append('env.chassis.friction_scales=[1.0]')
            preflight = command+smoke_overrides
        evaluations = []
        if not sb3:
            if is_nuplan:
                tests = [(f'test_seed{i}', 0, [], i) for i in range(3)]
            else:
                tests = [('new_maps',10020,[],0), ('s',10320,['s'],0), ('xl',10420,['xl'],0)]
            for split, seed, pool, inference_seed in tests:
                evaluation = base+[str(controls/'evaluate_metadrive.py'), '--checkpoint', str(run/'best.pt'),
                    '--source-root', str(source), '--output', str(run/'evaluation'/split),
                    '--episodes', '100', '--seed-start', str(seed)]
                if is_nuplan:
                    evaluation += ['--scenario-dir', str(config.env.scenario_test_dir),
                                   '--inference-seed-offset', str(12345+100000*inference_seed)]
                if pool:
                    evaluation += ['--eval-models', *pool]
                evaluations.append(dict(name=split, command=evaluation))
        jobs.append(dict(name=name, source=str(source), config=str(config_path), logdir=str(run),
            context=context, sb3=sb3, steps=args.steps, command=command, preflight=preflight,
            evaluations=evaluations))
    adoption = None
    if args.adopt_queue:
        previous = args.adopt_queue.resolve()
        old_plan = json.loads((previous/'plan.json').read_text())
        old_status = json.loads((previous/'status.json').read_text())
        name = ORDER[0]
        if old_status['phase'] != name+'/training':
            raise ValueError('Only a currently running nuPlan context training can be adopted')
        adopted = copy.deepcopy(next(j for j in old_plan['jobs'] if j['name']==name))
        stage = old_status['jobs'][name]['stages']['training']
        if not process_running(stage['pid'],adopted['command']):
            raise ValueError('Training process is not running')
        adopted.update(adopted=True,adopted_pid=stage['pid'],adopted_started_at=stage['started_at'])
        jobs[0] = adopted
        adoption = dict(previous_queue=str(previous),previous_runner_pid=old_status['runner_pid'],
                        adopted_pid=stage['pid'],previous_job=old_status['jobs'][name])
    plan = dict(created_at=now(), after_status=str(args.after_status.resolve()),
        nuplan_reference=str(nuplan_reference), roundabout_reference=str(roundabout_reference),
        roundabout_mass_friction=args.roundabout_mass_friction,adoption=adoption,
        jobs=jobs, sources={str(p):fingerprint(p) for p in (nuplan_source,roundabout_source,controls,config_dir)})
    if adoption:
        for folder in old_plan['sources']:
            plan['sources'][folder] = fingerprint(Path(folder))
    write_json(output/'plan.json', plan)
    write_json(output/'status.json', dict(phase='prepared', created_at=now(),
        after_status=plan['after_status'], jobs={j['name']:dict(state='queued',logdir=j['logdir']) for j in jobs}))
    print(json.dumps(dict(output=str(output), order=list(order), steps=args.steps,
                         adoption=adoption), indent=2), flush=True)


def validate_checkpoint(folder, steps, context=False):
    import torch
    payload = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
    if payload['step'] < steps or payload['update_count'] < 1 or not (folder/'best.pt').exists():
        raise RuntimeError(f'Incomplete training: {folder}')
    if not all(bool(torch.isfinite(value).all()) for value in payload['agent_state_dict'].values()
               if value.is_floating_point()):
        raise RuntimeError(f'Nonfinite checkpoint: {folder}')
    if context and steps == 128:
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        if not any(row.get('train/context/swap_pairs', 0)>0 for row in rows):
            raise RuntimeError('Context preflight did not exercise valid exchange pairs')


def run(output):
    plan = json.loads((output/'plan.json').read_text())
    for folder, expected in plan['sources'].items():
        if fingerprint(Path(folder)) != expected:
            raise RuntimeError('Frozen source/config fingerprint changed: '+folder)
    state = json.loads((output/'status.json').read_text())
    if state['phase'] != 'prepared':
        raise RuntimeError('Refusing to rerun a queue that has already started')
    state.update(phase='waiting',runner_pid=os.getpid(),started_at=now())
    status_path = output/'status.json'
    write_json(status_path,state)
    current = None
    active_job = None

    def stop(signum, frame):
        if current is not None and current.poll() is None:
            try:
                os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HYDRA_FULL_ERROR='1',CUDA_VISIBLE_DEVICES='0')

    def execute(job, stage, command):
        nonlocal current
        name=job['name']; record=state['jobs'][name]
        state['phase']=name+'/'+stage
        record.update(state='running',stage=stage)
        detail=record.setdefault('stages',{}).setdefault(stage,{})
        detail.update(state='running',started_at=now(),command=command)
        with (output/(name+'_'+stage+'.log')).open('w') as stream:
            current=subprocess.Popen(command,cwd=job['source'],env=environment,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            detail['pid']=current.pid;write_json(status_path,state)
            print(now(),state['phase'],flush=True)
            code=current.wait()
        detail.update(returncode=code,finished_at=now(),state='completed' if code==0 else 'failed')
        write_json(status_path,state)
        if code:
            raise RuntimeError(f'{name}/{stage} failed with code {code}; later jobs not started')

    try:
        while True:
            prior=json.loads(Path(plan['after_status']).read_text())
            if prior['phase']=='completed':
                break
            if prior['phase'] in ('failed','interrupted','blocked','completed_with_failures','cancelled') or not Path(f'/proc/{prior["runner_pid"]}').exists():
                raise RuntimeError('Prerequisite training/evaluations did not finish successfully')
            if state.get('prerequisite_phase') != prior['phase']:
                state['prerequisite_phase']=prior['phase'];write_json(status_path,state)
            time.sleep(20)
        state['prerequisite_completed_at']=now();write_json(status_path,state)
        for job in plan['jobs']:
            name=job['name'];record=state['jobs'][name]
            active_job = name
            record['started_at']=now()
            if job.get('adopted'):
                record.update(copy.deepcopy(plan['adoption']['previous_job']))
                record.update(state='running',stage='training',adopted_by=os.getpid())
                record['stages']['training']['adopted']=True
                state['phase']=name+'/training'
                write_json(status_path,state)
                while process_running(job['adopted_pid'],job['command']):
                    time.sleep(10)
                validate_checkpoint(Path(job['logdir']),job['steps'])
                rows=[json.loads(line) for line in (Path(job['logdir'])/'metrics.jsonl').read_text().splitlines()]
                if not any(r['step']>=job['steps'] and 'episode/eval_score' in r for r in rows):
                    raise RuntimeError('Adopted training is missing its final validation')
                record['stages']['training'].update(state='completed',finished_at=now(),
                    returncode=None,completion_verified='full-budget finite latest checkpoint and final validation')
                write_json(status_path,state)
            else:
                execute(job,'preflight',job['preflight'])
                if job['sb3']:
                    summary=json.loads((output/'validation'/name/'training_summary.json').read_text())
                    if not summary['finite_parameters'] or summary['updates']<1:
                        raise RuntimeError('SB3 preflight did not pass')
                else:
                    validate_checkpoint(output/'validation'/name,128,job['context'])
                execute(job,'training',job['command'])
                if job['sb3']:
                    summary=json.loads((Path(job['logdir'])/'training_summary.json').read_text())
                    if summary['steps'] != job['steps'] or not summary['finite_parameters']:
                        raise RuntimeError('SB3 training did not complete its budget')
                else:
                    validate_checkpoint(Path(job['logdir']),job['steps'])
            for evaluation in job['evaluations']:
                execute(job,'evaluation_'+evaluation['name'],evaluation['command'])
            record.update(state='completed',finished_at=now());write_json(status_path,state)
        state.update(phase='completed',finished_at=now())
    except BaseException as error:
        state.update(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                     error=repr(error),finished_at=now())
        if active_job is not None:
            state['jobs'][active_job].update(state=state['phase'],error=repr(error),finished_at=now())
        traceback.print_exc()
        write_json(status_path,state)
        return 1
    write_json(status_path,state)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--after-status',type=Path)
    parser.add_argument('--nuplan-reference',type=Path)
    parser.add_argument('--roundabout-reference',type=Path)
    parser.add_argument('--steps',type=int,default=500000)
    parser.add_argument('--roundabout-mass-friction',action='store_true')
    parser.add_argument('--adopt-queue',type=Path,help='Prepared handoff of the existing nuPlan context process')
    args=parser.parse_args()
    if args.run:
        return run(args.output.resolve())
    if not all((args.after_status,args.nuplan_reference,args.roundabout_reference)) or args.steps<1:
        parser.error('Preparation requires all reference paths and a positive step budget')
    prepare(args)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
