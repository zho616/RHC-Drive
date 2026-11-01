"""Repeat seed0 T-intersection and roundabout-B methods at training seed1.

Training implementations and algorithm settings are copied from each seed0 run.
Training seed/output paths change, with seed-dependent map offsets compensated
to retain the original train/validation split. Evaluation uses the seed0 map,
spawn and chassis RNG to retain paired test conditions across training seeds.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback

from omegaconf import OmegaConf

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'dreamer.py').is_file() and (p/'iv_output').is_dir())
METHODS=('ppo','sac','r2dreamer_context','r2dreamer','dreamerv3')
TESTS=(('new_maps',10020,None),('s',10320,'s'),('xl',10420,'xl'))


def now():return datetime.datetime.now().astimezone().isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def flat(value,prefix=''):
    result={}
    for k,v in value.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):result.update(flat(v,name))
        else:result[name]=v
    return result


def seed_config(reference,run):
    config=copy.deepcopy(reference)
    original_env_seed=int(reference.env.seed)
    config.seed=1;config.env.seed='${seed}';config.logdir=str(run)
    # The original adapter adds env.seed to these offsets. Compensating the
    # offsets keeps both map sets identical to seed0, rather than shifting the
    # validation set into the first independent test map at10020.
    config.env.start_seed=int(reference.env.start_seed)+original_env_seed-1
    config.env.eval_start_seed=int(reference.env.eval_start_seed)+original_env_seed-1
    config.hydra=dict(run=dict(dir='${logdir}'))
    old=flat(OmegaConf.to_container(reference,resolve=True))
    new=flat(OmegaConf.to_container(config,resolve=True))
    diff={k:dict(seed0=old.get(k),seed1=new.get(k)) for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    allowed={'seed','env.seed','env.start_seed','env.eval_start_seed','logdir','hydra.run.dir'}
    if set(diff)-allowed:raise ValueError(f'Unexpected configuration changes: {diff}')
    if config.env.steps!=500000 or config.seed!=1:raise ValueError('Unexpected training budget/seed')
    assert int(config.env.seed)+int(config.env.start_seed)==original_env_seed+int(reference.env.start_seed)
    assert int(config.env.seed)+int(config.env.eval_start_seed)==original_env_seed+int(reference.env.eval_start_seed)
    return config,diff


def patch_sb3_tests(text):
    anchor="    p.add_argument('--steps',type=int,default=500000);p.add_argument('--smoke',action='store_true')"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"\n    p.add_argument('--test-environment-seed',type=int,default=None)")
    anchor='            ec.eval_start_seed=start;ec.eval_num_scenarios=count;ec.eval_env_num=1;ec.eval_episode_num=count'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"\n            if a.test_environment_seed is not None:\n                ec.seed=a.test_environment_seed;ec.eval_start_seed=start-int(ec.seed)")
    anchor="            folder=out/'evaluation'/name;folder.mkdir(parents=True)"
    assert text.count(anchor)==1
    return text.replace(anchor,anchor+"\n            OmegaConf.save(ec,folder/'config.yaml')")


def patch_dreamer_tests(text):
    anchor="    parser.add_argument('--device',default='cuda:0')"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"\n    parser.add_argument('--environment-seed',type=int,default=None)")
    anchor='    OmegaConf.resolve(config)'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"\n    if args.environment_seed is not None:config.env.seed=args.environment_seed")
    anchor="    (output/'provenance.json').write_text"
    assert text.count(anchor)==1
    return text.replace(anchor,"    provenance.update(training_seed=int(config.seed),environment_seed=int(config.env.seed))\n"+anchor)


def fingerprint(source):
    return {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    for sub in ('configs','references','sources','controls'):(output/sub).mkdir()
    refs={
        'r2dreamer':ROOT/'logdir/tintersection_mass_friction_500k_20261021/training/r2dreamer',
        'r2dreamer_context':ROOT/'logdir/tintersection_mass_friction_500k_20261021/training/r2dreamer_dali_shallow',
        'dreamerv3':ROOT/'logdir/tintersection_baselines_queue_20261022/dreamerv3/dreamerv3',
        'ppo':ROOT/'logdir/tintersection_baselines_queue_20261022/ppo',
        'sac':ROOT/'logdir/tintersection_baselines_queue_20261022/sac'}
    sources={};jobs=[]
    for group in ('tintersection','roundabout_B'):
        for method in METHODS:
            name=group+'_'+method
            reference=refs[method] if group=='tintersection' else ROOT/'logdir/roundabout_mass_friction_queue_20261023'/('roundabout_'+method)
            sb3=method in ('ppo','sac')
            config_path=reference/('config.yaml' if sb3 else '.hydra/config.yaml')
            config=OmegaConf.load(config_path)
            assert int(config.seed)==0 and config.env.chassis.enabled
            assert list(config.env.chassis.mass_scales)==[.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4]
            assert list(config.env.chassis.friction_scales)==list(config.env.chassis.mass_scales)
            if group=='roundabout_B':
                original_source=ROOT/'logdir/roundabout_mass_friction_queue_20261023/roundabout_source'
            elif sb3:original_source=ROOT/'logdir/tintersection_baselines_queue_20261022/source'
            else:original_source=reference.parent/'source'
            if str(original_source) not in sources:
                frozen=output/'sources'/f'source_{len(sources)}';frozen.mkdir()
                for p in original_source.glob('*.py'):shutil.copy2(p,frozen/p.name)
                for sub in ('envs','optim','configs'):
                    shutil.copytree(original_source/sub,frozen/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
                if (original_source/'pyproject.toml').exists():shutil.copy2(original_source/'pyproject.toml',frozen/'pyproject.toml')
                expected=fingerprint(frozen)
                assert all(sha(original_source/k)==v for k,v in expected.items())
                sources[str(original_source)]=dict(copy=str(frozen),sha256=expected)
            source=Path(sources[str(original_source)]['copy'])
            run=output/name
            candidate,diff=seed_config(config,run)
            saved_config=output/'configs'/(name+'.yaml');OmegaConf.save(candidate,saved_config)
            shutil.copy2(config_path,output/'references'/(name+'_seed0.yaml'))
            base=[sys.executable,'-u']
            evaluations=[]
            if sb3:
                original_control=(ROOT/'logdir/tintersection_baselines_queue_20261022/source/scripts/train_metadrive_sb3.py' if group=='tintersection'
                    else ROOT/'logdir/roundabout_mass_friction_queue_20261023/controls/train_metadrive_sb3.py')
                control=output/'controls'/(group+'_train_sb3.py')
                control.write_text(patch_sb3_tests(original_control.read_text()))
                command=base+[str(control),'--source',str(source),'--config',str(saved_config),'--output',str(run),
                    '--method',method,'--steps','500000','--test-environment-seed','0']
                hyper=json.loads((reference/'hyperparameters.json').read_text())
            else:
                command=base+[str(source/'train.py'),'--config-path',str(output/'configs'),'--config-name',name]
                for split,start,pool in TESTS:
                    e=base+[str(output/'controls/evaluate_metadrive.py'),'--checkpoint',str(run/'best.pt'),
                        '--source-root',str(source),'--output',str(run/'evaluation'/split),
                        '--episodes','100','--seed-start',str(start),'--environment-seed','0']
                    if pool:e+=['--eval-models',pool]
                    evaluations.append(dict(name=split,command=e))
                hyper=None
            jobs.append(dict(name=name,group=group,method=method,reference=str(reference),source=str(source),
                config=str(saved_config),logdir=str(run),sb3=sb3,steps=500000,training_seed=1,
                changes_from_seed0=diff,seed0_hyperparameters=hyper,command=command,evaluations=evaluations))
    (output/'controls/evaluate_metadrive.py').write_text(patch_dreamer_tests((ROOT/'scripts/evaluate_metadrive.py').read_text()))
    shutil.copy2(Path(__file__),output/'controls/queue_t_roundabout_seed1.py')
    plan=dict(created_at=now(),jobs=jobs,sources=sources,
        protocol='Five methods for T-intersection and roundabout-B only. No DreamerV3+Context, no A group. Each seed0 training implementation/settings is retained. Seed and output directory differ; map offsets compensate env.seed to preserve the exact original split. 500K steps, original10K validation20 episodes, validation-selected best, three100-episode tests with fixed seed0 environment RNG.',
        training_seed_semantics='Global/model RNG and per-worker environment/chassis streams become seed1. Compensate original adapter offsets: start_seed=-1 and eval_start_seed=9999 with env.seed=1 yield the same train0–99 and validation10000–10019 maps as seed0. Final test maps and chassis RNG remain exactly seed0 protocol, with no validation/test map overlap.',
        evaluation_seed=0,tests=[dict(name=n,seed_start=s,seed_end=s+99,models=m) for n,s,m in TESTS])
    plan['artifacts_sha256']={str(p.relative_to(output)):sha(p) for sub in ('configs','references','controls') for p in (output/sub).rglob('*') if p.is_file()}
    write(output/'plan.json',plan)
    write(output/'status.json',dict(phase='prepared',created_at=now(),jobs={j['name']:dict(state='queued') for j in jobs}))
    print(json.dumps(dict(output=str(output),jobs=[j['name'] for j in jobs]),indent=2))


def validate_training(job):
    folder=Path(job['logdir'])
    if job['sb3']:
        summary=json.loads((folder/'training_summary.json').read_text())
        if summary['steps']!=job['steps'] or not summary['finite_parameters']:raise RuntimeError('Invalid SB3 completion')
        old=copy.deepcopy(job['seed0_hyperparameters']['settings'])
        new=json.loads((folder/'hyperparameters.json').read_text())['settings']
        old['seed']=int(job.get('training_seed',1))
        if old!=new:raise RuntimeError('SB3 training hyperparameters differ beyond seed')
        checkpoint=folder/'best.zip'
    else:
        import torch
        payload=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if payload['step']<job['steps'] or payload['update_count']<1:raise RuntimeError('Incomplete Dreamer run')
        if not all(bool(torch.isfinite(v).all()) for v in payload['agent_state_dict'].values() if v.is_floating_point()):raise RuntimeError('Nonfinite weights')
        checkpoint=folder/'best.pt'
    if not checkpoint.exists():raise RuntimeError('Missing validation-selected best')
    return checkpoint


def retain_evaluation(output,job,checkpoint):
    folder=Path(job['logdir']);checkpoint_hash=sha(checkpoint)
    namespace=job.get('result_namespace','tintersection_roundabout_B_seed1_20261025')
    retained=ROOT/'iv_output/table/source_data'/namespace/job['name']
    retained.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split,start,_ in TESTS:
        evaluation=folder/'evaluation'/split
        summary=json.loads((evaluation/'summary.json').read_text())
        episodes=[json.loads(x) for x in (evaluation/'episodes.jsonl').read_text().splitlines()]
        if len(episodes)!=100 or sorted(x['seed'] for x in episodes)!=list(range(start,start+100)):
            raise RuntimeError('Test count/map seeds differ from seed0 protocol')
        values=summary['summary']
        provenance=dict(training_seed=int(job.get('training_seed',1)),evaluation_environment_seed=0,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
            source=job['source'],reference_seed0_run=job['reference'],seed_start=start,seed_end=start+99,episodes=100)
        write(evaluation/'repeat_provenance.json',provenance)
        shutil.copytree(evaluation,retained/'evaluation'/split,dirs_exist_ok=True)
        rows.append(dict(scene=job['group'],method=job['method'],training_seed=int(job.get('training_seed',1)),split=split,episodes=100,
            **{k:values[k] for k in ('success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps')},
            checkpoint_sha256=checkpoint_hash,summary_source=str(evaluation/'summary.json')))
    shutil.copy2(job['config'],retained/'training_config.yaml')
    write(retained/'provenance.json',dict(job=job,checkpoint_sha256=checkpoint_hash))
    write(output/(job['name']+'_results.json'),rows)
    for group in (job['group'],):
        combined=[]
        for f in output.glob(group+'_*_results.json'):combined.extend(json.loads(f.read_text()))
        if combined:
            filename=(group+f"_seed{int(job.get('training_seed',1))}_test_results.csv" if 'result_namespace' not in job else namespace+'_'+group+'_test_results.csv')
            dest=ROOT/'iv_output/table'/filename
            temp=dest.with_suffix('.tmp')
            with temp.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
            temp.replace(dest)


def run(output, resume_roundabout=False):
    import fcntl
    lock=(output/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((output/'plan.json').read_text());state=json.loads((output/'status.json').read_text())
    if resume_roundabout:
        if state['phase']!='paused':raise RuntimeError('Roundabout resume requires the paused queue')
        if any(state['jobs'][j['name']]['state']!='completed' for j in plan['jobs'] if j['group']=='tintersection'):
            raise RuntimeError('T-intersection must already be completed')
        jobs=[j for j in plan['jobs'] if j['group']=='roundabout_B']
        if [j['method'] for j in jobs]!=list(METHODS):raise RuntimeError('Unexpected Roundabout methods')
        for job in jobs:
            if state['jobs'][job['name']]['state']!='paused' or Path(job['logdir']).exists():
                raise RuntimeError('Roundabout resume requires untouched paused jobs')
    else:
        if state['phase']!='prepared':raise RuntimeError('Queue already started; refusing duplicate training')
        jobs=plan['jobs']
    for info in plan['sources'].values():
        if fingerprint(Path(info['copy']))!=info['sha256']:raise RuntimeError('Frozen training source changed')
    for name,expected in plan['artifacts_sha256'].items():
        if sha(output/name)!=expected:raise RuntimeError('Control/config fingerprint changed')
    state.update(runner_pid=os.getpid(),phase='starting',started_at=now());write(output/'status.json',state)
    current=None
    def stop(signum,frame):
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HYDRA_FULL_ERROR='1',CUDA_VISIBLE_DEVICES='0',
        SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',MPLCONFIGDIR='/tmp/r2dreamer-mpl',PYTHONPATH='/home/server/projects/metadrive')
    def execute(job,stage,command):
        nonlocal current
        state['phase']=job['name']+'/'+stage
        record=state['jobs'][job['name']];record.update(state='running',stage=stage)
        detail=dict(started_at=now(),state='running',command=command)
        record.setdefault('stages',{})[stage]=detail
        with (output/(job['name']+'_'+stage+'.log')).open('w') as stream:
            current=subprocess.Popen(command,cwd=job['source'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            detail['pid']=current.pid;write(output/'status.json',state)
            print(now(),state['phase'],flush=True);code=current.wait()
        detail.update(state='completed' if code==0 else 'failed',returncode=code,finished_at=now());write(output/'status.json',state)
        if code:raise RuntimeError(f'{stage} exited with {code}')
    try:
        for job in jobs:
            record=state['jobs'][job['name']];record['started_at']=now()
            try:
                if Path(job['logdir']).exists():raise RuntimeError('Training output already exists')
                execute(job,'training',job['command'])
                checkpoint=validate_training(job)
                for test in job['evaluations']:execute(job,'test_'+test['name'],test['command'])
                retain_evaluation(output,job,checkpoint)
                record.update(state='completed',finished_at=now(),checkpoint=str(checkpoint))
            except Exception as error:
                record.update(state='failed',error=repr(error),finished_at=now());traceback.print_exc()
            write(output/'status.json',state)
        state.update(phase='completed' if all(r['state']=='completed' for r in state['jobs'].values()) else 'completed_with_failures',finished_at=now())
    except BaseException as error:
        state.update(phase='interrupted',error=repr(error),finished_at=now());raise
    finally:write(output/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--resume-roundabout',action='store_true',help='Resume only the five untouched paused Roundabout jobs')
    args=p.parse_args()
    if args.prepare==args.run:p.error('Choose exactly one of --prepare or --run')
    if args.resume_roundabout and not args.run:p.error('--resume-roundabout requires --run')
    output=args.output.resolve()
    if args.prepare:prepare(output)
    else:run(output,args.resume_roundabout)


if __name__=='__main__':main()
