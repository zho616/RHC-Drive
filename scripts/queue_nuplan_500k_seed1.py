"""Repeat the paper's nuPlan 500K baseline and Full at training seed1."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
from run_nuplan_roundabout_queue import fingerprint, freeze, now, run, write_json
from queue_t_roundabout_seed1 import flat, patch_dreamer_tests

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dreamer.py').is_file() and (p/'iv_output').is_dir())
REF=ROOT/'logdir/nuplan_pittsburgh_baseline_500k_20261022'
FULL=ROOT/'logdir/nuplan_context_roundabout_queue_20261023'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def prepare(output,seed=1,after_status=None):
    if seed not in (1,2):raise ValueError('Only training seeds1/2 are supported')
    references={'r2dreamer':REF/'baseline','ours':FULL/'nuplan_r2dreamer_context'}
    baseline_source=REF/'source';full_source=FULL/'nuplan_source'
    assert all(sha(baseline_source/rel)==expected for rel,expected in fingerprint(full_source).items())
    output.mkdir(parents=True,exist_ok=False)
    for name in ('configs','controls','validation','tables'):(output/name).mkdir()
    source=output/'source';freeze(baseline_source,source)
    assert fingerprint(source)==fingerprint(full_source)
    for name in ('run_nuplan_roundabout_queue.py',Path(__file__).name):
        shutil.copy2(ROOT/'scripts'/name,output/'controls'/name)
    shutil.copy2(ROOT/'scripts/queue_t_roundabout_seed1.py',output/'controls/queue_t_roundabout_seed1.py')
    original_evaluator=baseline_source/'scripts/evaluate_metadrive.py'
    (output/'controls/evaluate_metadrive.py').write_text(patch_dreamer_tests(original_evaluator.read_text()))
    protected={};jobs=[];differences={};split_hashes={}
    for method,reference in references.items():
        old=OmegaConf.load(reference/'.hydra/config.yaml')
        assert old.seed==0 and old.env.steps==500000 and old.trainer.eval_every==10000
        assert old.env.eval_episode_num==100 and old.env.num_scenarios==8000
        cfg=copy.deepcopy(old);cfg.seed=seed;cfg.env.seed='${seed}'
        cfg.logdir=str(output/method);cfg.hydra=dict(run=dict(dir='${logdir}'))
        a=flat(OmegaConf.to_container(old,resolve=True));b=flat(OmegaConf.to_container(cfg,resolve=True))
        diff={k:dict(reference_seed0=a.get(k),candidate=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        assert set(diff)=={'seed','env.seed','logdir','hydra.run.dir'},diff
        assert cfg.env.start_seed==old.env.start_seed and cfg.env.eval_start_seed==old.env.eval_start_seed
        if method=='ours':
            assert cfg.model.context.enabled and cfg.model.context.injection=='dali_shallow'
            assert cfg.model.context.swap_weight==1 and cfg.model.context.aux_weight==1
            assert cfg.model.context.wm_grad_scale==0 and not cfg.model.context.actor_critic
            assert 'min(5, length - 1)' in (source/'context.py').read_text()
        else:assert not cfg.model.context.enabled
        for split in ('train','eval','test'):
            folder=Path(cfg.env['scenario_'+split+'_dir'])
            for filename in ('dataset_summary.pkl','dataset_mapping.pkl'):
                p=folder/filename
                if filename=='dataset_summary.pkl':assert p.is_file(),p
                if p.is_file():split_hashes[str(p)]=sha(p)
        path=output/'configs'/(method+'.yaml');OmegaConf.save(cfg,path);differences[method]=diff
        command=[sys.executable,'-u',str(source/'train.py'),'--config-path',str(output/'configs'),'--config-name',method]
        preflight=command+['device=cuda:0','env.steps=128','env.env_num=2','env.eval_env_num=1',
            'env.eval_episode_num=2','env.time_limit=32','env.train_ratio=8','buffer.max_size=512',
            'batch_size=4','batch_length=8','model.compile=false','model.deter=32','model.hidden=16',
            'model.units=16','model.discrete=4','model.rssm.stoch=4','model.rssm.blocks=4',
            'model.imag_horizon=3','trainer.eval_every=64','trainer.update_log_every=32',
            'env.chassis.models=[default]','env.chassis.mass_scales=[1.0]',
            'env.chassis.steering_scales=[1.0]','env.chassis.friction_scales=[1.0]',
            f'logdir={output/"validation"/method}']
        evaluations=[]
        for condition,pool,device in (('id',None,'cuda:0'),('s','s','cpu'),('xl','xl','cpu')):
            for inference_seed in (0,1,2):
                evaluation=[sys.executable,'-u',str(output/'controls/evaluate_metadrive.py'),
                    '--checkpoint',str(output/method/'best.pt'),'--source-root',str(source),
                    '--output',str(output/method/'evaluation'/condition/f'seed{inference_seed}'),
                    '--episodes','100','--seed-start','0','--environment-seed','0',
                    '--scenario-dir',str(cfg.env.scenario_test_dir),
                    '--inference-seed-offset',str(12345+100000*inference_seed),'--device',device]
                if pool:evaluation+=['--eval-models',pool]
                evaluations.append(dict(name=condition+f'_seed{inference_seed}',command=evaluation))
        jobs.append(dict(name=method,source=str(source),config=str(path),logdir=str(output/method),
            context=method=='ours',sb3=False,steps=500000,command=command,preflight=preflight,evaluations=evaluations))
        for p in (reference/'best.pt',reference/'latest.pt',reference/'metrics.jsonl',reference/'.hydra/config.yaml'):
            protected[str(p)]=sha(p)
    for original in (full_source,baseline_source):
        for rel,digest in fingerprint(original).items():protected[str(original/rel)]=digest
    protected.update(split_hashes)
    for p in (ROOT/'iv_output/table/nuplan_test_results.csv',ROOT/'iv_output/table/nuplan_by_inference_seed.csv'):
        protected[str(p)]=sha(p)
    write_json(output/'config_differences.json',differences)
    metadrive=Path('/home/server/projects/metadrive')
    dependency=dict(root=str(metadrive),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=metadrive,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=metadrive,text=True).strip()))
    for p in (REF/'split_audit.json',REF/'physics_audit.json'):
        shutil.copy2(p,output/'validation'/p.name)
    prerequisite=after_status or ROOT/'logdir/tintersection_roundabout_B_seed1_20261025/status.json'
    if not Path(prerequisite).is_file():raise FileNotFoundError(prerequisite)
    plan=dict(created_at=now(),after_status=str(prerequisite),
        jobs=jobs,sources={str(p):fingerprint(p) for p in (source,output/'configs',output/'controls')},
        protected=protected,seed=seed,budget=500000,eval_every=10000,validation_episodes=100,
        from_scratch=True,reference_seed0={name:str(path) for name,path in references.items()},metadrive=dependency,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol=f'Paper 500K recipe, not the later 1M/eval5K recipe. Separate train/val/test databases; scenario indices remain0-based. Training RNG becomes seed{seed}; tests force environment RNG0. Each condition has100 unique scenarios and3 inference RNG repeats, not3 training seeds.',
        context='Original Full: state4/history5/context8, posterior-only, Response horizon5 weight1, Exchange weight1, WM->Encoder0; unchanged Actor-Critic/reward.')
    write_json(output/'plan.json',plan)
    write_json(output/'status.json',dict(phase='prepared',created_at=now(),jobs={j['name']:dict(state='queued',logdir=j['logdir']) for j in jobs}))
    (output/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join([
        sys.executable,'-u',str(output/'controls'/Path(__file__).name),'--output',str(output),'--run'])+'\n')
    (output/'PLAN.zh.md').write_text(f'nuPlan Pittsburgh：依次R2Dreamer、Ours，seed{seed}，各500k，随机初始化；每10k验证100回合，按原success/route/return选best。\n\n'
        '训练源码与论文500k两组原冻结源码一致，只改变seed/env.seed及输出目录。Ours仍为五步Response+Exchange、WM不回传Context。训练/验证/测试为原独立数据库，适配器场景索引固定从0开始，无需补偿start_seed。\n\n'
        '每组完成后，ID/S/XL各100场景×3推理重复，环境seed固定0以匹配旧seed0测试底盘/出生随机流；ID用CUDA、S/XL用CPU，与原表一致。三次推理重复不当作三次训练。\n\n'
        '旧checkpoint、日志、论文表格及数据划分元数据受SHA256保护。短程GPU preflight不计正式结果；失败停止后续任务，不覆盖不完整运行。依赖使用当前已安装MetaDrive，其git标识/dirty状态单独记录，未替换为1M配方。\n')
    print(output,flush=True)


def verify_protected(output):
    plan=json.loads((output/'plan.json').read_text())
    for p,expected in plan['protected'].items():
        if sha(p)!=expected:raise RuntimeError('Protected paper artifact changed: '+p)
    return plan


def summarize(output):
    seed=json.loads((output/'plan.json').read_text())['seed']
    rows=[];metrics=('success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps')
    for method in ('r2dreamer','ours'):
        for condition in ('id','s','xl'):
            summaries=[]
            for inference_seed in (0,1,2):
                folder=output/method/'evaluation'/condition/f'seed{inference_seed}'
                summary=json.loads((folder/'summary.json').read_text())
                episodes=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
                assert summary['status']=='completed' and summary['completed_episodes']==len(episodes)==100
                assert [r['seed'] for r in episodes]==list(range(100))
                summaries.append(summary['summary'])
            rows.append(dict(method=method,training_seed=seed,condition=condition,unique_scenarios=100,
                inference_repeats=3,rollouts=300,**{k:sum(s[k] for s in summaries)/3 for k in metrics}))
    with (output/'tables'/f'nuplan_seed{seed}_summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f'nuPlan500k seed{seed}：R2Dreamer与Ours均完成训练与测试。三次推理重复均来自同一训练模型。','',
           '| Method | ID Success | S Success | XL Success |','|---|---|---|---|']
    for method in ('r2dreamer','ours'):
        values=[next(r['success'] for r in rows if r['method']==method and r['condition']==c) for c in ('id','s','xl')]
        lines.append('| '+method+' | '+' | '.join(f'{100*x:.1f}%' for x in values)+' |')
    (output/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')


def execute(output):
    lock=(output/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=verify_protected(output)
    if any(Path(j['logdir']).exists() for j in plan['jobs']):raise RuntimeError('Refuse existing formal training directories')
    os.environ.update(PYTHONPATH='/home/server/projects/metadrive',MPLCONFIGDIR='/tmp/rhc-ablation-matplotlib',
        XDG_CACHE_HOME='/tmp/rhc-ablation-cache',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    code=run(output)
    if code==0:summarize(output)
    verify_protected(output)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--seed',type=int,choices=(1,2),default=1)
    parser.add_argument('--after-status',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.run:return execute(output)
    prepare(output,args.seed,args.after_status.resolve() if args.after_status else None);return 0


if __name__=='__main__':raise SystemExit(main())
