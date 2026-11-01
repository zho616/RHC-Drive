"""Repeat the completed five-method T-intersection seed1 experiment after nuPlan."""
import argparse
import copy
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

from omegaconf import OmegaConf
import queue_t_roundabout_seed1 as queue
from run_nuplan_roundabout_queue import freeze

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dreamer.py').is_file() and (p/'iv_output').is_dir())
ORDER=('ppo','sac','r2dreamer','r2dreamer_context','dreamerv3')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare(out,reference,after,seed=1,groups=('tintersection',)):
    old=json.loads((reference/'plan.json').read_text());oldstate=json.loads((reference/'status.json').read_text())
    originals={(j['group'],j['method']):j for j in old['jobs']}
    assert seed in (1,2) and len(set(groups))==len(groups)
    selected=[originals[g,m] for g in groups for m in ORDER]
    assert all(oldstate['jobs'][j['name']]['state']=='completed' for j in selected)
    assert after.is_file()
    out.mkdir(parents=True,exist_ok=False)
    for name in ('configs','controls','references','sources'):(out/name).mkdir()
    for name in (Path(__file__).name,'queue_t_roundabout_seed1.py','run_nuplan_roundabout_queue.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'controls'/name)
    for name in ('tintersection_train_sb3.py','roundabout_B_train_sb3.py','evaluate_metadrive.py'):
        shutil.copy2(reference/'controls'/name,out/'controls'/name)
    sources={};jobs=[];protected={}
    for original in selected:
        method=original['method'];job=copy.deepcopy(original);original_source=Path(job['source'])
        source_info=next(info for info in old['sources'].values() if info['copy']==str(original_source))
        assert queue.fingerprint(original_source)==source_info['sha256']
        if str(original_source) not in sources:
            frozen=out/'sources'/original_source.name;freeze(original_source,frozen)
            assert queue.fingerprint(frozen)==source_info['sha256']
            sources[str(original_source)]=dict(copy=str(frozen),sha256=queue.fingerprint(frozen))
        source=Path(sources[str(original_source)]['copy']);run=out/job['name']
        cfg=OmegaConf.load(job['config']);oldcfg=copy.deepcopy(cfg)
        assert cfg.seed==1 and cfg.env.steps==500000 and cfg.trainer.eval_every==10000
        old_seed=int(oldcfg.env.seed)
        cfg.seed=seed;cfg.env.seed='${seed}'
        cfg.env.start_seed=int(oldcfg.env.start_seed)+old_seed-seed
        cfg.env.eval_start_seed=int(oldcfg.env.eval_start_seed)+old_seed-seed
        cfg.logdir=str(run);cfg.hydra=dict(run=dict(dir='${logdir}'))
        a=queue.flat(OmegaConf.to_container(oldcfg,resolve=True));b=queue.flat(OmegaConf.to_container(cfg,resolve=True))
        diff={k:dict(previous=a.get(k),rerun=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        allowed={'logdir','hydra.run.dir'} if seed==int(oldcfg.seed) else {'logdir','hydra.run.dir','seed','env.seed','env.start_seed','env.eval_start_seed'}
        assert set(diff)==allowed
        assert cfg.model==oldcfg.model
        assert int(cfg.env.seed)+int(cfg.env.start_seed)==int(oldcfg.env.seed)+int(oldcfg.env.start_seed)
        assert int(cfg.env.seed)+int(cfg.env.eval_start_seed)==int(oldcfg.env.seed)+int(oldcfg.env.eval_start_seed)
        config=out/'configs'/(job['name']+'.yaml');OmegaConf.save(cfg,config)
        shutil.copy2(job['config'],out/'references'/(job['name']+'_previous_seed1.yaml'))
        job.update(source=str(source),config=str(config),logdir=str(run),result_namespace=out.name,
            reference_seed1=original['logdir'],changes_from_previous_seed1=diff,training_seed=seed)
        job.pop('changes_from_seed0',None)
        if job['sb3']:
            command=list(job['command']);command[0]=sys.executable
            command[2]=str(out/'controls'/(job['group']+'_train_sb3.py'))
            for flag,value in (('--source',source),('--config',config),('--output',run)):
                command[command.index(flag)+1]=str(value)
        else:
            command=[sys.executable,'-u',str(source/'train.py'),'--config-path',str(out/'configs'),'--config-name',job['name']]
        job['command']=command
        for evaluation in job['evaluations']:
            cmd=evaluation['command'];cmd[0]=sys.executable;cmd[2]=str(out/'controls/evaluate_metadrive.py')
            for flag,value in (('--checkpoint',run/'best.pt'),('--source-root',source),
                               ('--output',run/'evaluation'/evaluation['name'])):
                cmd[cmd.index(flag)+1]=str(value)
        for p in Path(original['logdir']).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
        jobs.append(job)
    for rel in ('iv_output/table/tintersection_seed1_test_results.csv','iv_output/table/roundabout_B_seed1_test_results.csv'):
        p=ROOT/rel
        if p.is_file():protected[str(p)]=sha(p)
    plan=dict(created_at=queue.now(),jobs=jobs,sources=sources,after_status=str(after),protected=protected,
        order=[j['name'] for j in jobs],groups=list(groups),training_seed=seed,budget=500000,repeat_reference=str(reference),
        protocol=f'Fresh training seed{seed} after nuPlan seed1 completes. Exact per-method frozen training implementations and settings retained. Map offsets compensate env.seed to retain train/validation splits; test environment RNG0 retained. Separate output namespace, no warm start.',
        artifacts_sha256={str(p.relative_to(out)):sha(p) for sub in ('configs','references','controls') for p in (out/sub).glob('*') if p.is_file()},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    queue.write(out/'plan.json',plan)
    queue.write(out/'status.json',dict(phase='prepared',created_at=queue.now(),jobs={j['name']:dict(state='queued') for j in jobs}))
    (out/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join([
        sys.executable,'-u',str(out/'controls/queue_tintersection_repeat.py'),'--output',str(out),'--run'])+'\n')
    (out/'PLAN.zh.md').write_text(f'等待nuPlan seed1两组500k训练、测试及报告完成后，按场景顺序 {", ".join(groups)}，每场景依次PPO、SAC、R2Dreamer、Ours（Full五步+Exchange）、DreamerV3。全部seed{seed}，各500k，随机初始化。\n\n训练模型/奖励/随机化/优化器等保持各自原冻结设置；seed2补偿start_seed=-2、eval_start_seed=9998，保留实际训练地图0–99和验证10000–10019。每10k验证20回合，ID/S/XL各100配对测试；测试环境seed0。新结果独立命名空间，已有结果不覆盖。\n')
    print(out,flush=True)


def verify(plan):
    for path,expected in plan['protected'].items():
        if sha(path)!=expected:raise RuntimeError('Previous result changed: '+path)


def run(out):
    lock=(out/'dependency.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((out/'plan.json').read_text());state=json.loads((out/'status.json').read_text())
    if state['phase']!='prepared':raise RuntimeError('Queue already started')
    verify(plan)
    state.update(phase='waiting_for_nuplan',runner_pid=os.getpid(),dependency_runner_pid=os.getpid(),after_status=plan['after_status'])
    queue.write(out/'status.json',state)
    after=Path(plan['after_status'])
    try:
        while True:
            prior=json.loads(after.read_text())
            alive=Path(f'/proc/{prior.get("runner_pid",0)}/cmdline').exists()
            if prior['phase']=='completed' and (after.parent/'REPORT.zh.md').is_file():break
            if prior['phase'] in ('failed','interrupted') or not alive:
                raise RuntimeError('nuPlan prerequisite failed or exited before final report')
            if state.get('prerequisite_phase')!=prior['phase']:
                state['prerequisite_phase']=prior['phase'];queue.write(out/'status.json',state)
            time.sleep(20)
        verify(plan)
        if any(Path(j['logdir']).exists() for j in plan['jobs']):raise RuntimeError('Refusing existing training outputs')
        # Delegate the existing trainer/evaluator queue after successful dependency.
        state.update(phase='prepared',prerequisite_completed_at=queue.now());queue.write(out/'status.json',state)
        os.environ.update(PYTHONPATH='/home/server/projects/metadrive',MPLCONFIGDIR='/tmp/rhc-ablation-matplotlib',
            XDG_CACHE_HOME='/tmp/rhc-ablation-cache')
        queue.run(out)
        verify(plan)
    except BaseException as error:
        state=json.loads((out/'status.json').read_text());state.update(phase='failed',error=repr(error),finished_at=queue.now())
        queue.write(out/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--reference',type=Path,default=ROOT/'logdir/tintersection_roundabout_B_seed1_20261025')
    parser.add_argument('--after-status',type=Path,default=ROOT/'logdir/nuplan_500k_seed1_20261028/status.json')
    parser.add_argument('--seed',type=int,choices=(1,2),default=1)
    parser.add_argument('--groups',nargs='+',choices=('tintersection','roundabout_B'),default=['tintersection'])
    args=parser.parse_args();out=args.output.resolve()
    if args.run:run(out)
    else:prepare(out,args.reference.resolve(),args.after_status.resolve(),args.seed,args.groups)


if __name__=='__main__':main()
