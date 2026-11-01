"""Defer Full H1 ablation and the paused Roundabout seed1 queue until A/B/C/D finish."""
import argparse
import copy
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'dreamer.py').exists() and (p/'iv_output').is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_t_roundabout_seed1 as queue


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.write(path, value)


def verify(out):
    plan = read(out/'plan.json')
    if queue.fingerprint(out/'source') != plan['source_sha256']:
        raise RuntimeError('Frozen H1 source changed')
    for rel, expected in plan['artifacts_sha256'].items():
        if queue.sha(out/rel) != expected:
            raise RuntimeError('H1 config/control changed: '+rel)
    for path, expected in plan['protected'].items():
        if queue.sha(path) != expected:
            raise RuntimeError('Protected original artifact changed: '+path)
    return plan


def prepare(out, predecessor, roundabout):
    from omegaconf import OmegaConf
    old = read(predecessor/'manifest.json')
    for path, expected in old['protected'].items():
        if queue.sha(path) != expected: raise RuntimeError('Original artifact changed: '+path)
    if queue.fingerprint(predecessor/'source') != old['source_sha256']:
        raise RuntimeError('Original ablation source changed')
    state = read(roundabout/'status.json'); original = read(roundabout/'plan.json')
    assert state['phase'] == 'paused'
    pending = [j for j in original['jobs'] if j['group']=='roundabout_B']
    assert [j['method'] for j in pending] == list(queue.METHODS)
    assert all(state['jobs'][j['name']]['state']=='completed' for j in original['jobs'] if j['group']=='tintersection')
    assert all(state['jobs'][j['name']]['state']=='paused' and not Path(j['logdir']).exists() for j in pending)
    for info in original['sources'].values():
        assert queue.fingerprint(Path(info['copy'])) == info['sha256']
    for rel, expected in original['artifacts_sha256'].items():
        assert queue.sha(roundabout/rel) == expected
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('configs','controls','validation','evaluation','tables'):
        (out/folder).mkdir()
    shutil.copytree(predecessor/'source', out/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Start with the already audited ablation source, replace only horizon-related files.
    for rel in ('context.py','dreamer.py','configs/model/_base_.yaml'):
        shutil.copy2(ROOT/rel, out/'source'/rel)
    for path in (Path(__file__), ROOT/'scripts/queue_t_roundabout_seed1.py'):
        shutil.copy2(path, out/'controls'/path.name)
    shutil.copy2(predecessor/'evaluate_metadrive.py', out/'controls/evaluate_metadrive.py')
    shutil.copy2(ROOT/'tests/test_rhc_ablation.py', out/'controls/test_rhc_ablation.py')
    shutil.copy2(ROOT/'tests/test_context.py', out/'controls/test_context.py')
    test_helper=out/'controls/test_context.py'
    test_helper.write_text(test_helper.read_text().replace("config_path='../configs'",
                          "config_path='../source/configs'"))
    cfg = OmegaConf.load(predecessor/'configs/D.yaml')
    reference = queue.flat(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.context.prediction_horizon = 1
    cfg.logdir = str(out/'training/full_h1')
    assert cfg.seed == 1 and cfg.env.steps == 500000
    assert cfg.model.context.swap_weight == 1 and cfg.model.context.wm_grad_scale == 0
    assert cfg.model.context.use_context_exchange and cfg.model.context.use_response_loss
    OmegaConf.save(cfg, out/'configs/full_h1.yaml')
    changed = queue.flat(OmegaConf.to_container(cfg, resolve=True))
    differences = {k:dict(full_h5=reference.get(k), full_h1=changed.get(k))
                   for k in reference.keys()|changed.keys() if reference.get(k)!=changed.get(k)}
    assert set(differences)=={'model.context.prediction_horizon','logdir','hydra.run.dir'}
    source_changes = [p for p,h in queue.fingerprint(out/'source').items() if old['source_sha256'].get(p)!=h]
    assert set(source_changes)=={'context.py','dreamer.py','configs/model/_base_.yaml'}
    train = [sys.executable,'-u',str(out/'source/train.py'),'--config-path',str(out/'configs'),'--config-name','full_h1']
    evaluations = copy.deepcopy(old['jobs']['D']['evaluations'])
    for e in evaluations:
        command=e['command']; command[0]=sys.executable; command[2]=str(out/'controls/evaluate_metadrive.py')
        for flag,value in (('--checkpoint',out/'training/full_h1/best.pt'),('--source-root',out/'source'),
                           ('--output',out/'evaluation'/e['condition'])):
            command[command.index(flag)+1]=str(value)
    resume=[sys.executable,'-u',str(out/'controls/queue_t_roundabout_seed1.py'),
            '--output',str(roundabout),'--run','--resume-roundabout']
    protected=dict(old['protected'])
    for condition in ('ID','S','XL'):
        for filename in ('summary.json','episodes.jsonl'):
            path=predecessor/'evaluation/D'/condition/filename; protected[str(path)]=queue.sha(path)
    plan=dict(created_at=now(),seed=1,budget=500000,scenario='T-intersection',history_length=5,context_dim=8,
        prediction_horizon=1,exchange=True,wm_to_context=False,
        predecessor=str(predecessor),predecessor_launch=read(predecessor/'gpu_launch.json'),
        roundabout=str(roundabout),roundabout_methods=list(queue.METHODS),roundabout_horizon=5,
        reference=old['jobs']['D'],train=train,evaluations=evaluations,resume_roundabout=resume,
        configuration_differences=differences,source_changes=source_changes,
        source_sha256=queue.fingerprint(out/'source'),protected=protected,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        artifacts_sha256={str(p.relative_to(out)):queue.sha(p) for sub in ('configs','controls')
                          for p in (out/sub).rglob('*') if p.is_file()},
        order=['complete existing A/B/C/D training, 12 paired tests, and report',
               'Full H1 500k + 3 x 100 paired tests + H1/H5 comparison','resume original Roundabout-B five-method seed1 queue'])
    save(out/'plan.json',plan)
    save(out/'status.json',dict(phase='prepared',created_at=now(),formal_training='not_started'))
    command=[sys.executable,'-u',str(out/'controls/queue_rhc_followup.py'),'run','--output',str(out)]
    (out/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(command)+'\n')
    (out/'PLAN.zh.md').write_text('顺序：当前 T-intersection seed1 A/B/C/D 训练、配对测试和报告完成后，'
        'Full H1 训练 500k，再做 ID/S/XL 各 100 回合配对测试，最后恢复 Roundabout-B seed1 五方法。\n\n'
        'H1 只修改 Response 预测步数（5→1），历史窗口5、Context维度8、Exchange权重1、WM→Context关闭。'
        '全新训练，沿用 Full 的验证选择规则；旧 Full H5 checkpoint 和测试结果只读复用。\n\n'
        '环岛按 PPO、SAC、R2Dreamer+Context、R2Dreamer、DreamerV3 顺序，使用原队列冻结源码及配置。'
        '环岛 Full 仍为 H5。当前五组均未开始。\n\n'
        '启动命令见 commands.sh；进度见 status.json。前序失败时停止，已有不完整输出拒绝覆盖。\n')
    print(out,flush=True)


def smoke(out):
    verify(out)
    # CPU FP32 model update/strict reload test. Production CUDA/AMP config stays intact.
    # Run from the frozen source with the frozen tests; external legacy reference tests
    # are run separately from the repository so their original paths remain meaningful.
    driver=('import sys,unittest; sys.path[:0]=sys.argv[1:3]; '
            'suite=unittest.defaultTestLoader.loadTestsFromName('
            '"test_rhc_ablation.AblationTests.test_horizon_one_matches_single_transition_and_keeps_exchange"); '
            'result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())')
    command=[sys.executable,'-u','-c',driver,str(out/'source'),str(out/'controls')]
    with (out/'validation/smoke.log').open('x') as log:
        result=subprocess.run(command,cwd=out/'source',env=environment(),stdout=log,stderr=subprocess.STDOUT)
    save(out/'validation/smoke.json',dict(passed=result.returncode==0,returncode=result.returncode,command=command,
        precision='CPU FP32 eager',formal_result=False,
        checks=['H1 target/action alignment','nonzero encoder gradient','auxiliary isolated from RSSM',
                'Exchange positive pairs','finite full-model gradients','optimizer changes encoder',
                'strict state reload','finite policy action','history5/context8'],
        limitations='Short synthetic update; not a CUDA/compile or environment rollout performance result'))
    if result.returncode:raise RuntimeError('H1 frozen smoke failed')


def environment():
    env=dict(os.environ)
    env.update(PYTHONPATH='/home/server/projects/metadrive',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/rhc-ablation-matplotlib',XDG_CACHE_HOME='/tmp/rhc-ablation-cache',
        CUDA_VISIBLE_DEVICES='0',HYDRA_FULL_ERROR='1',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    return env


def completed_episodes(path, start):
    summary=read(path/'summary.json')
    if summary['status']!='completed' or summary['completed_episodes']!=100:
        raise RuntimeError('Incomplete paired evaluation: '+str(path))
    rows=[json.loads(x) for x in (path/'episodes.jsonl').read_text().splitlines()]
    if len(rows)!=100 or sorted(r['seed'] for r in rows)!=list(range(start,start+100)):
        raise RuntimeError('Invalid paired test seeds: '+str(path))
    return rows,summary


def comparison(out, plan):
    import numpy as np
    summaries=[]; paired=[]
    metrics=('success','route_completion','out_of_lane','crash','reward','mean_speed_km_h','steps','out_of_road')
    for condition,start in (('ID',10020),('S',10320),('XL',10420)):
        rows,summary=completed_episodes(out/'evaluation'/condition,start)
        old,reference=completed_episodes(Path(plan['predecessor'])/'evaluation/D'/condition,start)
        for label,s in (('Full H1',summary),('Full H5',reference)):
            summaries.append(dict(method=label,condition=condition,checkpoint_sha256=s['checkpoint_sha256'],
                checkpoint_step=s['checkpoint_step'],**{m:s['summary'][m] for m in metrics}))
        rows={r['seed']:r for r in rows}; old={r['seed']:r for r in old}; seeds=sorted(rows)
        if any(rows[s]['chassis']!=old[s]['chassis'] for s in seeds):raise RuntimeError('Chassis not paired')
        draws=np.random.default_rng(20261026).integers(0,100,(10000,100))
        for metric in metrics:
            delta=np.array([float(rows[s][metric])-float(old[s][metric]) for s in seeds])
            lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
            paired.append(dict(condition=condition,metric=metric,difference=float(delta.mean()),
                ci95_low=float(lo),ci95_high=float(hi),comparison='H1-H5'))
    for filename,rows in (('condition_metrics.csv',summaries),('paired_bootstrap.csv',paired)):
        with (out/'tables'/filename).open('x') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(out/'comparison.json',dict(results=summaries,paired=paired,bootstrap_replicates=10000,
        bootstrap_seed=20261026,limits='One training seed; paired episode percentile intervals, not training-seed uncertainty'))
    text=['Full 单步预测消融：T-intersection seed1，500k，保留 Exchange，历史窗口5。',
          '', '以下差异均为 H1−H5；每个条件100个相同地图/底盘测试回合，配对 bootstrap 95% CI。','']
    for row in paired:
        if row['metric']=='success':
            text.append(f"- {row['condition']}: {100*row['difference']:+.1f} pp [{100*row['ci95_low']:.1f}, {100*row['ci95_high']:.1f}]")
    text+=['','完整速度、碰撞、回报和路线完成率见 tables/condition_metrics.csv。只有一个训练种子，不能推断跨训练种子稳定性。']
    (out/'REPORT.zh.md').write_text('\n'.join(text)+'\n')


def predecessor_ready(plan):
    path=Path(plan['predecessor'])
    state=read(path/'status.json')
    pid=plan['predecessor_launch']['pid']
    proc=Path(f'/proc/{pid}/cmdline')
    try: alive=str(path/'commands.sh').encode() in proc.read_bytes().split(b'\0')
    except FileNotFoundError: alive=False
    if alive:return False
    if state['phase']!='completed':raise RuntimeError('Predecessor exited before successful training/evaluation/report completion')
    if read(path/'training_resource_status.json')['status']!='completed':raise RuntimeError('Predecessor training incomplete')
    for method in 'ABCD':
        for condition,start in (('ID',10020),('S',10320),('XL',10420)):
            completed_episodes(path/'evaluation'/method/condition,start)
    if not (path/'REPORT.zh.md').is_file():raise RuntimeError('Missing predecessor report')
    return True


def adopted_training_active(handoff):
    proc=Path('/proc')/str(handoff['training_pid'])
    try:
        fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z' or fields[19]!=handoff['training_start_ticks']:return False
        command=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if command!=handoff['training_command']:raise RuntimeError('Adopted training PID identity changed')
        return True
    except (FileNotFoundError,ProcessLookupError):return False


def continue_seed0(out, repeat):
    """Continue after completed H1 using a fresh, protected seed0 retry directory."""
    locks=[(directory/'runner.lock').open('a') for directory in (out,repeat)]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=verify(out);state=read(out/'status.json')
    if state['phase']!='failed' or state.get('h1_comparison')!='completed':
        raise RuntimeError('Continuation requires completed H1 and a failed downstream queue')
    if read(repeat/'status.json')['phase']!='prepared':raise RuntimeError('Retry must use a fresh prepared directory')
    schedule=read(repeat/'schedule.json')
    for relative,expected in schedule['control_sha256'].items():
        if queue.sha(repeat/relative)!=expected:raise RuntimeError('Retry control changed: '+relative)
    for condition,start in (('ID',10020),('S',10320),('XL',10420)):
        completed_episodes(out/'evaluation'/condition,start)
    queue.validate_training(dict(logdir=str(out/'training/full_h1'),sb3=False,steps=500000))
    if not (out/'comparison.json').is_file():raise RuntimeError('Missing completed H1 comparison')
    save(repeat/'previous_controller_status.json',state)
    state.update(phase='seed0_retry_starting',runner_pid=os.getpid(),seed0_repeat=str(repeat),
        recovery_started_at=now(),remaining_order=['seed0 B 500k fresh retry','seed0 C 500k',
        'seed0 paired tests and report','Roundabout-B five methods seed1'])
    for key in ('error','finished_at','child_pid'):state.pop(key,None)
    save(out/'status.json',state)
    current=None
    def stop(signum,frame):
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(stage,command,cwd):
        nonlocal current
        state.update(phase=stage,stage_started_at=now(),command=command)
        with (repeat/(stage+'.log')).open('x') as stream:
            current=subprocess.Popen(command,cwd=cwd,env=environment(),stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=current.pid;save(out/'status.json',state)
            print(now(),stage,flush=True);code=current.wait()
        state.pop('child_pid',None)
        if code:raise RuntimeError(f'{stage} exited with {code}')
        state.setdefault('completed_stages',[]).append(stage);save(out/'status.json',state)
    try:
        save(repeat/'status.json',dict(phase='training_in_progress',runner_pid=os.getpid(),
            formal_training={'A':'reused','B':'queued','C':'queued','D':'reused'}))
        for phase in ('train','evaluate','report'):
            execute('seed0_'+phase,[sys.executable,'-u',str(repeat/'controls/rhc_ablation.py'),
                phase,'--output',str(repeat)],ROOT)
        if read(repeat/'status.json')['phase']!='completed':raise RuntimeError('Seed0 tests/report incomplete')
        execute('roundabout_five_methods',plan['resume_roundabout'],ROOT)
        if read(Path(plan['roundabout'])/'status.json')['phase']!='completed':raise RuntimeError('Roundabout contains failures')
        state.update(phase='completed',finished_at=now());save(out/'status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=repr(error),finished_at=now());save(out/'status.json',state)
        save(repeat/'controller_failure.json',dict(error=repr(error),finished_at=now()))
        repeat_state=read(repeat/'status.json')
        if repeat_state['phase']!='completed':
            repeat_state.update(phase='failed',error=repr(error),finished_at=now());save(repeat/'status.json',repeat_state)
        traceback.print_exc();raise


def prepare_e_seed0(out, bc, reference):
    """Keep E's audited H1 implementation, change seed and compensate map offsets."""
    from omegaconf import OmegaConf
    original=verify(reference)
    if read(bc/'status.json')['phase']!='paused_by_user':raise RuntimeError('Expected the paused B/C seed0 run')
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('configs','controls','training','evaluation','tables','validation'):(out/folder).mkdir()
    shutil.copytree(reference/'source',out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cfg=OmegaConf.load(reference/'configs/full_h1.yaml')
    before=queue.flat(OmegaConf.to_container(cfg,resolve=True));old_seed=int(cfg.env.seed)
    cfg.seed=0;cfg.env.seed='${seed}'
    cfg.env.start_seed=int(cfg.env.start_seed)+old_seed
    cfg.env.eval_start_seed=int(cfg.env.eval_start_seed)+old_seed
    cfg.logdir=str(out/'training/E')
    after=queue.flat(OmegaConf.to_container(cfg,resolve=True))
    differences={k:dict(seed1=before.get(k),seed0=after.get(k)) for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    assert set(differences)=={'seed','env.seed','env.start_seed','env.eval_start_seed','logdir','hydra.run.dir'}
    assert cfg.model.context.prediction_horizon==1 and cfg.model.context.use_context_exchange
    assert cfg.model.context.swap_weight==1 and cfg.model.context.wm_grad_scale==0
    assert cfg.env.seed+cfg.env.start_seed==0 and cfg.env.seed+cfg.env.eval_start_seed==10000
    assert cfg.env.steps==500000
    OmegaConf.save(cfg,out/'configs/E.yaml')
    for name in ('queue_rhc_followup.py','queue_t_roundabout_seed1.py','rhc_ablation.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'controls'/name)
    shutil.copy2(reference/'controls/evaluate_metadrive.py',out/'controls/evaluate_metadrive.py')
    evaluations=copy.deepcopy(original['evaluations'])
    for evaluation in evaluations:
        command=evaluation['command'];command[0]=sys.executable;command[2]=str(out/'controls/evaluate_metadrive.py')
        for flag,value in (('--checkpoint',out/'training/E/best.pt'),('--source-root',out/'source'),
                           ('--output',out/'evaluation'/evaluation['condition'])):
            command[command.index(flag)+1]=str(value)
    protected=dict(original['protected'])
    for name in ('best.pt','latest.pt','metrics.jsonl','.hydra/config.yaml'):
        path=reference/'training/full_h1'/name;protected[str(path)]=queue.sha(path)
    plan=dict(seed=0,budget=500000,scenario='T-intersection',history_length=5,context_dim=8,
        prediction_horizon=1,exchange=True,wm_to_context=False,bc_output=str(bc),reference_seed1=str(reference),
        train=[sys.executable,'-u',str(out/'source/train.py'),'--config-path',str(out/'configs'),'--config-name','E'],
        evaluations=evaluations,resume_roundabout=original['resume_roundabout'],roundabout=original['roundabout'],
        configuration_differences=differences,source_sha256=queue.fingerprint(out/'source'),protected=protected,
        artifacts_sha256={str(p.relative_to(out)):queue.sha(p) for sub in ('configs','controls') for p in (out/sub).glob('*') if p.is_file()},
        order=['resume in-memory B seed0, finish B and C 500k','E Full H1 seed0 500k',
               'paired A/B/C/D/E seed0 evaluations and report','original Roundabout-B seed1 five methods'])
    assert plan['source_sha256']==original['source_sha256']
    save(out/'plan.json',plan);save(out/'status.json',dict(phase='prepared',B='paused_in_memory',C='queued',E='queued'))
    (out/'PLAN.zh.md').write_text('T-intersection seed0：B继续当前暂停的新运行，保留Replay/优化器；C随机初始化；E为Full单步预测，保留Exchange，随机初始化。三组各500k。B/C沿用上一轮实现，E与seed1单步源码完全一致。\n\n顺序B→C→E→统一配对评估（ID/S/XL各100）与汇总→原环岛seed1五方法。历史窗口5、Context维度8、奖励与Actor-Critic等不变。旧CUDA失败尝试单独保留，不混入完成结果。\n')
    print(out,flush=True)


def report_five_seed0(bc, out):
    import numpy as np
    manifest=read(bc/'manifest.json');data={};summary_rows=[]
    metrics=('success','route_completion','crash','out_of_lane','out_of_road','reward','mean_speed_km_h','steps')
    for method in 'ABCDE':
        for condition,start in (('ID',10020),('S',10320),('XL',10420)):
            directory=(out/'evaluation'/condition) if method=='E' else bc/'evaluation'/method/condition
            episodes,summary=completed_episodes(directory,start)
            data[method,condition]={r['seed']:r for r in episodes}
            summary_rows.append(dict(method=method,condition=condition,checkpoint_step=summary['checkpoint_step'],
                **{key:summary['summary'][key] for key in metrics}))
    paired=[]
    for newer,older in (('B','A'),('C','B'),('D','C'),('E','D'),('E','A')):
        for condition in ('ID','S','XL'):
            a,b=data[newer,condition],data[older,condition];seeds=sorted(a)
            assert a.keys()==b.keys() and all(a[s]['chassis']==b[s]['chassis'] for s in seeds)
            draws=np.random.default_rng(20261027).integers(0,100,(10000,100))
            for metric in metrics:
                delta=np.array([float(a[s][metric])-float(b[s][metric]) for s in seeds])
                low,high=np.quantile(delta[draws].mean(1),[.025,.975])
                paired.append(dict(comparison=newer+'-'+older,condition=condition,metric=metric,
                    difference=float(delta.mean()),ci95_low=float(low),ci95_high=float(high)))
    for filename,rows in (('five_method_metrics.csv',summary_rows),('paired_bootstrap.csv',paired)):
        with (out/'tables'/filename).open('x') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels={'A':'R2Dreamer','B':'History-only','C':'Response-guided','D':'Full H5','E':'Full H1'}
    lines=['T-intersection seed0，五组均完成500k预算，按原验证协议选择checkpoint。',
           'B/C/E为本次训练；A/D只读复用原seed0模型，统一重评估。每个条件100个配对回合。','',
           '| Method | ID Success | S Success | XL Success | Mean Return |',
           '|---|---|---|---|---|']
    latex=[r'\begin{tabular}{lrrrr}',r'\hline',r'Method & ID Success & S Success & XL Success & Return \\',r'\hline']
    for method in 'ABCDE':
        rows=[next(r for r in summary_rows if r['method']==method and r['condition']==c) for c in ('ID','S','XL')]
        values=[f"{100*r['success']:.1f}" for r in rows]+[f"{np.mean([r['reward'] for r in rows]):.1f}"]
        lines.append('| '+labels[method]+' | '+' | '.join(values)+' |')
        latex.append(labels[method]+' & '+' & '.join(values)+r' \\')
    lines+=['','成功率单位%，回报为三个条件等权均值。速度/碰撞等诊断见CSV。',
            'B→C同时改变监督和WM→Encoder梯度来源，不能作纯Response损失归因。配对CI仅反映测试回合不确定性。']
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    (out/'tables/five_method_summary.tex').write_text('\n'.join(latex+[r'\hline',r'\end{tabular}'])+'\n')


def run_bce_seed0(out):
    plan=verify(out);bc=Path(plan['bc_output'])
    locks=[(directory/'runner.lock').open('a') for directory in (out,bc)]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(out/'status.json');handoff=read(out/'handoff.json')
    if state['phase']!='prepared':raise RuntimeError('BCE queue already started')
    if not adopted_training_active(handoff):raise RuntimeError('Paused B/C process is no longer available')
    state.update(phase='BC_training',B='running',C='queued',E='queued',runner_pid=os.getpid(),
        adopted_pid=handoff['training_pid'],started_at=now())
    save(out/'status.json',state)
    current=None
    def stop(signum,frame):
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(stage,command,cwd):
        nonlocal current
        state.update(phase=stage,command=command,stage_started_at=now())
        with (out/(stage+'.log')).open('x') as stream:
            current=subprocess.Popen(command,cwd=cwd,env=environment(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=current.pid;save(out/'status.json',state)
            print(now(),stage,flush=True);code=current.wait()
        state.pop('child_pid',None)
        if code:raise RuntimeError(f'{stage} exited with {code}')
        state.setdefault('completed_stages',[]).append(stage);save(out/'status.json',state)
    try:
        print(now(),'Adopt existing B/C training process',handoff['training_pid'],flush=True)
        while adopted_training_active(handoff):
            resource=read(bc/'training_resource_status.json')
            state['current_method']=resource.get('current')
            for method in 'BC':
                state[method]='completed' if method in resource.get('completed',[]) else ('running' if resource.get('current')==method else 'queued')
            save(out/'status.json',state);time.sleep(30)
        for method in 'BC':
            queue.validate_training(dict(logdir=str(bc/'training'/method),sb3=False,steps=500000))
        if read(bc/'training_resource_status.json')['status']!='completed':raise RuntimeError('B/C helper exited without full completion')
        state.update(B='completed',C='completed');save(out/'status.json',state)
        verify(out)
        if (out/'training/E').exists():raise RuntimeError('E output already exists')
        execute('E_training',plan['train'],out/'source')
        queue.validate_training(dict(logdir=str(out/'training/E'),sb3=False,steps=500000))
        state['E']='completed';save(out/'status.json',state)
        for phase in ('evaluate','report'):
            execute('BC_'+phase,[sys.executable,'-u',str(bc/'controls/rhc_ablation.py'),phase,'--output',str(bc)],ROOT)
        for evaluation in plan['evaluations']:
            execute('E_test_'+evaluation['condition'],evaluation['command'],out/'source')
        report_five_seed0(bc,out);verify(out)
        state['seed0_five_method_report']='completed';save(out/'status.json',state)
        execute('roundabout_five_methods',plan['resume_roundabout'],ROOT)
        if read(Path(plan['roundabout'])/'status.json')['phase']!='completed':raise RuntimeError('Roundabout contains failures')
        state.update(phase='completed',finished_at=now());save(out/'status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=repr(error),finished_at=now());save(out/'status.json',state)
        traceback.print_exc();raise


def run(out, repeat=None):
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=verify(out); state=read(out/'status.json')
    handoff=None
    if repeat is not None:
        handoff=read(repeat/'handoff.json')
        if state['phase']!='h1_training' or state.get('child_pid')!=handoff['training_pid']:
            raise RuntimeError('Handoff requires the recorded active H1 training')
        if state['runner_pid']!=handoff['retired_runner_pid']:raise RuntimeError('Unexpected previous scheduler')
        if Path(f"/proc/{handoff['retired_runner_pid']}/cmdline").exists():
            command=Path(f"/proc/{handoff['retired_runner_pid']}/cmdline").read_bytes()
            if command:raise RuntimeError('Previous scheduler must be retired before adoption')
    elif state['phase']!='prepared':raise RuntimeError('Already started; refusing duplicate queue')
    if not read(out/'validation/smoke.json')['passed']:raise RuntimeError('Smoke did not pass')
    if handoff:
        state.update(phase='h1_training',formal_training='running',runner_pid=os.getpid(),
            adopted_at=now(),seed0_repeat=str(repeat),
            remaining_order=['finish current H1 training and tests','seed0 B 500k','seed0 C 500k',
                             'seed0 paired tests and report','Roundabout-B five methods seed1'])
    else:
        state.update(phase='waiting_for_abcd_report',runner_pid=os.getpid(),started_at=now())
    save(out/'status.json',state)
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(stage,command,cwd):
        nonlocal child
        state.update(phase=stage,stage_started_at=now(),command=command)
        with (out/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=cwd,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save(out/'status.json',state)
            print(now(),stage,flush=True);code=child.wait()
        if code:raise RuntimeError(f'{stage} exited with {code}')
        state.setdefault('completed_stages',[]).append(stage);state.pop('child_pid',None);save(out/'status.json',state)
    try:
        if handoff:
            print(now(),'Adopting H1 training without restarting it',handoff['training_pid'],flush=True)
            while adopted_training_active(handoff):time.sleep(30)
            state['h1_completion_check']='Adopted process exited; validate complete-budget finite checkpoint and best; exit code unavailable.'
            state.setdefault('completed_stages',[]).append('h1_training')
            state.pop('child_pid',None)
        else:
            while True:
                try: ready=predecessor_ready(plan)
                except json.JSONDecodeError: ready=False
                if ready: break
                time.sleep(30)
            verify(out)
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU production substitute')
            if (out/'training/full_h1').exists():raise RuntimeError('Refusing existing H1 training output')
            execute('h1_training',plan['train'],out/'source')
        job=dict(logdir=str(out/'training/full_h1'),sb3=False,steps=500000)
        checkpoint=queue.validate_training(job)
        state['formal_training']='completed';state['checkpoint']=str(checkpoint);save(out/'status.json',state)
        for evaluation in plan['evaluations']:
            directory=out/'evaluation'/evaluation['condition']
            if directory.exists():raise RuntimeError('Refusing existing evaluation output')
            execute('h1_test_'+evaluation['condition'],evaluation['command'],out/'source')
        comparison(out,plan);verify(out)
        state['h1_comparison']='completed';save(out/'status.json',state)
        if repeat is not None:
            schedule=read(repeat/'schedule.json')
            for relative,expected in schedule['control_sha256'].items():
                if queue.sha(repeat/relative)!=expected:raise RuntimeError('Seed0 control changed: '+relative)
            save(repeat/'status.json',dict(phase='training_in_progress',runner_pid=os.getpid(),
                formal_training={'A':'reused','B':'queued','C':'queued','D':'reused'}))
            for phase in ('train','evaluate','report'):
                command=[sys.executable,'-u',str(repeat/'controls/rhc_ablation.py'),phase,'--output',str(repeat)]
                execute('seed0_'+phase,command,ROOT)
            if read(repeat/'status.json')['phase']!='completed':raise RuntimeError('Seed0 tests/report incomplete')
        execute('roundabout_five_methods',plan['resume_roundabout'],ROOT)
        if read(Path(plan['roundabout'])/'status.json')['phase']!='completed':
            raise RuntimeError('Roundabout queue contains failures; inspect its status')
        state.update(phase='completed',finished_at=now());save(out/'status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=repr(error),finished_at=now());save(out/'status.json',state)
        traceback.print_exc();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','smoke','run','continue-seed0','prepare-e-seed0','run-bce-seed0'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--predecessor',type=Path,default=ROOT/'logdir/rhc_context_ablation_seed1_20261026')
    parser.add_argument('--roundabout',type=Path,default=ROOT/'logdir/tintersection_roundabout_B_seed1_20261025')
    parser.add_argument('--seed0-repeat',type=Path,help='Adopt the current H1 process, then insert the prepared seed0 B/C repeat')
    args=parser.parse_args();out=args.output.resolve()
    if args.phase=='prepare-e-seed0':
        if not args.seed0_repeat:parser.error('prepare-e-seed0 requires --seed0-repeat')
        return prepare_e_seed0(out,args.seed0_repeat.resolve(),args.predecessor.resolve())
    if args.phase=='run-bce-seed0':return run_bce_seed0(out)
    if args.phase=='continue-seed0':
        if not args.seed0_repeat:parser.error('continue-seed0 requires --seed0-repeat')
        return continue_seed0(out,args.seed0_repeat.resolve())
    if args.phase=='prepare':prepare(out,args.predecessor.resolve(),args.roundabout.resolve())
    elif args.phase=='smoke':smoke(out)
    else:run(out,args.seed0_repeat.resolve() if args.seed0_repeat else None)


if __name__=='__main__':main()
