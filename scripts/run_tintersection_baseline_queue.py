"""Freeze and queue PPO, SAC, DreamerV3 and Shallow-context DreamerV3 after a run."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

def now():return datetime.datetime.now().astimezone().isoformat()
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--after-status',type=Path,required=True)
    p.add_argument('--env-config',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,source/f.name)
    for d in ['envs','optim','configs','scripts']:shutil.copytree(root/d,source/d,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root/'pyproject.toml',source/'pyproject.toml');shutil.copy2(a.env_config,out/'baseline_env_config.yaml')
    (out/'source.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=root))
    jobs=[]
    for name in ['ppo','sac']:
        cmd=[sys.executable,'-u',str(source/'scripts/train_metadrive_sb3.py'),'--source',str(source),'--config',str(out/'baseline_env_config.yaml'),'--output',str(out/name),'--method',name]
        if a.smoke:cmd+=['--smoke']
        jobs.append((name,cmd,out/name))
    for name in ['dreamerv3','dreamerv3_dali_shallow']:
        dest=out/name
        cmd=[sys.executable,'-u',str(source/'scripts/train_metadrive_compare.py'),'--logdir',str(dest),'--source-from',str(source),'--methods','dreamerv3','--steps','500000','--map-block','T','--chassis-randomization','--mass-friction-randomization']
        if name.endswith('shallow'):cmd+=['--context-variants','dali_shallow','--context-swap-weight','1']
        cmd+=['--smoke'] if a.smoke else ['--evaluate-best']
        jobs.append((name,cmd,dest/name))
    status=dict(phase='waiting',runner_pid=os.getpid(),created_at=now(),after_status=str(a.after_status.resolve()),source=str(source),jobs={name:dict(state='queued',command=cmd,logdir=str(run)) for name,cmd,run in jobs})
    statusfile=out/'status.json';write(statusfile,status)
    current=None
    def stop(signum,frame):
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            prior=json.loads(a.after_status.read_text())
            if prior['phase']=='completed':break
            if prior['phase'] in ['failed','interrupted','blocked'] or not Path(f"/proc/{prior['runner_pid']}").exists():
                raise RuntimeError('Prerequisite did not complete successfully; queue not started')
            time.sleep(20)
        status['prerequisite_completed_at']=now()
        for name,cmd,run in jobs:
            status['phase']=name;record=status['jobs'][name];record.update(state='running',started_at=now())
            with (out/(name+'.log')).open('w') as f:
                current=subprocess.Popen(cmd,cwd=source,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=current.pid;write(statusfile,status);code=current.wait()
            record.update(returncode=code,finished_at=now(),state='completed' if code==0 else 'failed');write(statusfile,status)
            if code:raise RuntimeError(f'{name} failed with exit code {code}; subsequent jobs not started')
        status.update(phase='completed',finished_at=now())
    except BaseException as e:
        status.update(phase='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e),finished_at=now())
        write(statusfile,status);raise
    write(statusfile,status)

if __name__=='__main__':main()
