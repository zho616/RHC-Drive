#!/usr/bin/env python3
"""After the active roundabout PPO, finish the old 75-combination cohort first."""
import argparse,copy,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
ACTIVE=ROOT/'logdir/roundabout_mass_friction_queue_20261023'
LEGACY=ROOT/'logdir/chassis_baselines_500k_20261015_020330'
OLD_CONTEXT=ROOT/'logdir/context_chassis_500k_20261015_102744'
OUT=ROOT/'logdir/roundabout_legacy_priority_queue_20261023'
PYTHON='/home/server/anaconda3/envs/r2dreamer/bin/python'
STEPS=500000
METHODS=['roundabout_sac_old75','roundabout_ppo_old75']

def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def checkpoint_progress(folder):
 import torch
 payload=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
 rows=[json.loads(line)for line in (folder/'metrics.jsonl').read_text().splitlines()if line.strip()]
 step=max((int(row.get('step',0))for row in rows),default=0)
 updates=max((int(row.get('train/opt/updates',0))for row in rows),default=0)
 finite=all(bool(torch.isfinite(v).all())for v in payload['agent_state_dict'].values()if v.is_floating_point())
 return step,updates,finite
def base_config():
 c=OmegaConf.load(LEGACY/'r2dreamer/.hydra/config.yaml')
 c.logdir=''
 c.env.steps=STEPS;c.env.env_num=8;c.env.eval_env_num=2;c.env.eval_episode_num=20;c.env.train_ratio=512
 c.buffer.max_size=500000;c.buffer.storage_device='cpu';c.seed=0;c.device='cuda:0';c.trainer.save_best=True
 c.env.reward.mode='paper';c.env.reward.lateral=0.
 c.env.chassis.enabled=True;c.env.chassis.models=['default','m','l'];c.env.chassis.mass_scales=[.8,.9,1.,1.1,1.2];c.env.chassis.steering_scales=[.8,.9,1.,1.1,1.2]
 c.env.chassis.eval_models='${env.chassis.models}';c.env.chassis.eval_mass_scales='${env.chassis.mass_scales}';c.env.chassis.eval_steering_scales='${env.chassis.steering_scales}'
 c.hydra=dict(run=dict(dir='${logdir}'))
 return c

def prepare():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'configs').mkdir();(OUT/'controls').mkdir()
 for name in ['train_metadrive_sb3.py','evaluate_metadrive.py']:
  source=ACTIVE/'controls'/name
  (OUT/'controls'/name).write_bytes(source.read_bytes())
 cfg=base_config();jobs=[]
 for name in METHODS:
  method=name.removeprefix('roundabout_').removesuffix('_old75')
  context=method.endswith('_context')
  rep='r2dreamer' if method.startswith('r2dreamer') else 'dreamer'
  conf=copy.deepcopy(cfg);conf.logdir=str(OUT/name);conf.model.rep_loss=rep
  conf.model.context.enabled=context;conf.env.context_enabled=context
  if context:
   # Match the legacy chassis-context experiment; eager execution avoids its prior compile dtype failure.
   ref=OmegaConf.load(OLD_CONTEXT/'r2dreamer_context/.hydra/config.yaml')
   for key in ('context','deter','hidden','discrete','depth','units','act','norm'):
    if key in ref.model:conf.model[key]=copy.deepcopy(ref.model[key])
   conf.model.context.enabled=True;conf.model.context.injection='recurrent_concat';conf.model.context.rollout='dynamic';conf.model.context.aux_weight=1.
   conf.model.compile=False;conf.env.context_enabled=True
  path=OUT/'configs'/(name+'.yaml');OmegaConf.save(conf,path)
  sb3=method in ('sac','ppo')
  source=LEGACY/'source' if not context else OLD_CONTEXT/'source'
  run=OUT/name
  if sb3:
   command=[PYTHON,'-u',str(OUT/'controls/train_metadrive_sb3.py'),'--source',str(source),'--config',str(path),'--output',str(run),'--method',method,'--steps',str(STEPS)]
   pre=command.copy();pre[pre.index('--output')+1]=str(OUT/'validation'/name);pre+=['--smoke']
   evals=[]
  else:
   command=[PYTHON,'-u',str(source/'train.py'),'--config-path',str(OUT/'configs'),'--config-name',name]
   pre=command+['device=cuda:0','env.steps=128','env.env_num=2','env.eval_env_num=1','env.eval_episode_num=2','env.time_limit=32','env.train_ratio=8','buffer.max_size=512','batch_size=4','batch_length=8','model.compile=false','model.deter=32','model.hidden=16','model.units=16','model.discrete=4','model.rssm.stoch=4','model.rssm.blocks=4','model.imag_horizon=3','trainer.eval_every=64','trainer.update_log_every=32','env.chassis.models=[default]','env.chassis.mass_scales=[1.0]','env.chassis.steering_scales=[1.0]',f'logdir={OUT/"validation"/name}']
   evals=[]
   for split,seed,pool in [('new_maps',10020,None),('s',10320,'s'),('xl',10420,'xl')]:
    ev=[PYTHON,'-u',str(OUT/'controls/evaluate_metadrive.py'),'--checkpoint',str(run/'best.pt'),'--source-root',str(source),'--output',str(run/'evaluation'/split),'--episodes','100','--seed-start',str(seed)]
    if pool:ev+=['--eval-models',pool]
    evals.append(ev)
  jobs.append(dict(name=name,method=method,source=str(source),config=str(path),sb3=sb3,command=command,preflight=pre,evaluations=evals,steps=STEPS))
 active=json.loads((ACTIVE/'plan.json').read_text())
 resume=[copy.deepcopy(next(j for j in active['jobs'] if j['name']==name))for name in ['roundabout_dreamerv3']]
 plan=dict(created_at=now(),waiting_for=str(ACTIVE/'status.json'),active_queue=str(ACTIVE),legacy_profile=dict(models=['default','m','l'],mass_scales=[.8,.9,1.,1.1,1.2],steering_scales=[.8,.9,1.,1.1,1.2],combinations=75,reward='paper',seed=0,steps=STEPS),legacy_jobs=jobs,resume_mass_friction_jobs=resume)
 save(OUT/'plan.json',plan)
 status=dict(phase='prepared_waiting_for_ppo',created_at=now(),jobs={j['name']:dict(state='queued',logdir=str(OUT/j['name']))for j in jobs},then_resume=['roundabout_dreamerv3'])
 save(OUT/'status.json',status)
 print('Prepared',OUT,flush=True)

def wait_for_update(path,last_mtime):
 # Linux inotify wakes on the status.json atomic rename used by the queue runner.
 import ctypes,select,struct
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
 if fd<0:time.sleep(.02);return
 wd=libc.inotify_add_watch(fd,os.fsencode(path.parent),0x00000080|0x00000008)
 poll=select.poll();poll.register(fd,select.POLLIN);poll.poll(1000)
 try:os.close(fd)
 except OSError:pass

def run():
 plan=json.loads((OUT/'plan.json').read_text());sp=Path(plan['waiting_for']);state=json.loads((OUT/'status.json').read_text())
 state.pop('error',None);state.pop('finished_at',None);state['runner_pid']=os.getpid()
 state['phase']='waiting_for_current_ppo';state['waiting_started_at']=now();save(OUT/'status.json',state)
 while True:
  active=json.loads(sp.read_text());ppo=active['jobs']['roundabout_ppo']
  stage=ppo.get('stages',{}).get('training',{})
  if ppo.get('state')=='completed' and stage.get('state')=='completed' and (stage.get('returncode') in (0,None)):
   break
  if active['phase'] in ('failed','interrupted','blocked','completed') and ppo.get('state')!='completed':
   raise RuntimeError('Active queue ended before PPO completed')
  wait_for_update(sp,0)
 state['ppo_completed_at']=now();state['phase']='pausing_mass_friction_queue';save(OUT/'status.json',state)
 runner=int(active.get('runner_pid',0))if active.get('phase')!='priority_handoff_paused'else 0
 try:
  if runner>0:os.kill(runner,signal.SIGTERM)
 except ProcessLookupError:pass
 deadline=time.time()+120
 def live(pid):
  try:return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
  except FileNotFoundError:return False
 while runner>0 and live(runner) and time.time()<deadline:time.sleep(.1)
 if runner>0 and live(runner):raise RuntimeError('Could not pause mass/friction queue after PPO')
 # Restore an accurate handoff record if SIGTERM landed between the PPO stage and queue bookkeeping.
 active=json.loads(sp.read_text());ppo=active['jobs']['roundabout_ppo'];ppo['state']='completed';ppo.pop('error',None);active['phase']='priority_handoff_paused';active.pop('error',None);save(sp,active)
 state['phase']='legacy75';save(OUT/'status.json',state)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HYDRA_FULL_ERROR='1',CUDA_VISIBLE_DEVICES='0')
 for job in plan['legacy_jobs']:
  record=state['jobs'][job['name']]
  if record.get('state')in('completed','cancelled_by_user','skipped_already_completed'):continue
  record['started_at']=now()
  stages=[('preflight',job['preflight']),('training',job['command'])]
  for stage,cmd in stages:
   record.update(state='running',stage=stage);record.setdefault('stages',{})[stage]=dict(state='running',started_at=now(),command=cmd);save(OUT/'status.json',state)
   with (OUT/f"{job['name']}_{stage}.log").open('w')as log:
    result=subprocess.run(cmd,cwd=job['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
   record['stages'][stage].update(state='completed'if result.returncode==0 else'failed',returncode=result.returncode,finished_at=now());save(OUT/'status.json',state)
   if result.returncode:raise RuntimeError(f"{job['name']}/{stage} returned {result.returncode}")
   if stage=='preflight':
    if job['sb3']:
     summary=json.loads((OUT/'validation'/job['name']/'training_summary.json').read_text())
     if not summary['finite_parameters'] or summary['updates']<1:raise RuntimeError('SB3 preflight invalid')
    else:
     folder=OUT/'validation'/job['name'];step,updates,finite=checkpoint_progress(folder)
     if step<128 or updates<1 or not finite:raise RuntimeError('Dreamer preflight checkpoint invalid')
     if job['method']=='r2dreamer_context':
      rows=[json.loads(line)for line in (folder/'metrics.jsonl').read_text().splitlines()]
      if not any(row.get('train/context/valid_targets',0)>0 and row.get('train/loss/context_aux',0)>=0 for row in rows):raise RuntimeError('Context preflight did not exercise valid context targets')
  if job['sb3']:
   summary=json.loads((OUT/job['name']/'training_summary.json').read_text())
   if summary['steps']!=STEPS or not summary['finite_parameters']:raise RuntimeError('SB3 training incomplete/nonfinite')
  else:
   step,updates,finite=checkpoint_progress(OUT/job['name'])
   if step<STEPS or updates<1 or not (OUT/job['name']/'best.pt').is_file() or not finite:raise RuntimeError('Dreamer training incomplete/nonfinite')
   for i,cmd in enumerate(job['evaluations']):
    record['stage']=f'evaluation_{i}';record['stages'][f'evaluation_{i}']=dict(state='running',started_at=now(),command=cmd);save(OUT/'status.json',state)
    with (OUT/f"{job['name']}_evaluation_{i}.log").open('w')as log:result=subprocess.run(cmd,cwd=job['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
    record['stages'][f'evaluation_{i}'].update(state='completed'if result.returncode==0 else'failed',returncode=result.returncode,finished_at=now());save(OUT/'status.json',state)
    if result.returncode:raise RuntimeError(f'{job["name"]} evaluation failed')
  record.update(state='completed',finished_at=now());save(OUT/'status.json',state)
 state['phase']='resume_mass_friction';save(OUT/'status.json',state)
 for job in plan['resume_mass_friction_jobs']:
  record=state['jobs'].setdefault(job['name'],dict(logdir=job['logdir']))
  if record.get('state')in('completed','cancelled_by_user'):continue
  # The prior runner is interrupted by the deliberate priority handoff; queue the untouched jobs here.
  for stage in ['preflight','training']:
   cmd=job['preflight'] if stage=='preflight' else job['command'];record.update(state='running',stage=stage);record.setdefault('stages',{})[stage]=dict(state='running',started_at=now(),command=cmd);save(OUT/'status.json',state)
   with (OUT/f"{job['name']}_{stage}.log").open('w')as log:result=subprocess.run(cmd,cwd=job['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
   record['stages'][stage].update(state='completed'if result.returncode==0 else'failed',returncode=result.returncode,finished_at=now());save(OUT/'status.json',state)
   if result.returncode:raise RuntimeError(f'{job["name"]}/{stage} returned {result.returncode}')
   if stage=='preflight':
    validation=Path(next(arg.split('=',1)[1]for arg in job['preflight']if arg.startswith('logdir=')))
    step,updates,finite=checkpoint_progress(validation)
    if step<128 or updates<1 or not finite:raise RuntimeError('Resume preflight checkpoint invalid')
   elif stage=='training':
    step,updates,finite=checkpoint_progress(Path(job['logdir']))
    if step<job['steps'] or updates<1 or not (Path(job['logdir'])/'best.pt').is_file() or not finite:raise RuntimeError('Resumed training is incomplete')
  for i,e in enumerate(job['evaluations']):
   record['stages'][f'evaluation_{i}']=dict(state='running',started_at=now(),command=e['command']);save(OUT/'status.json',state)
   with (OUT/f"{job['name']}_evaluation_{i}.log").open('w')as log:result=subprocess.run(e['command'],cwd=job['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
   record['stages'][f'evaluation_{i}'].update(state='completed'if result.returncode==0 else'failed',returncode=result.returncode,finished_at=now());save(OUT/'status.json',state)
   if result.returncode:raise RuntimeError('Mass/friction evaluation failed')
  record.update(state='completed',finished_at=now());save(OUT/'status.json',state)
 state.update(phase='completed',finished_at=now());save(OUT/'status.json',state)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.run:
  try:run()
  except BaseException as e:
   try:
    x=json.loads((OUT/'status.json').read_text());x.update(phase='failed',error=repr(e),finished_at=now());save(OUT/'status.json',x)
   except Exception:pass
   traceback.print_exc();return 1
 else:p.error('Use --prepare or --run')
 return 0
if __name__=='__main__':raise SystemExit(main())
