"""Adopt the current T-intersection training, finish testing, then pause.

The prior scheduler is stopped before handoff so it cannot launch roundabout.
The training process has its own session and is neither stopped nor restarted.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();out=args.output.resolve()
    spec=importlib.util.spec_from_file_location('frozen_seed1_queue',out/'controls/queue_t_roundabout_seed1.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    handoff=json.loads((out/'pause_after_tintersection_handoff.json').read_text())
    plan=json.loads((out/'plan.json').read_text())
    job=next(j for j in plan['jobs'] if j['name']==handoff['job'])
    state=json.loads((out/'status.json').read_text())
    state.update(runner_pid=os.getpid(),phase=job['name']+'/training',
                 pause_requested=True,pause_after_group='tintersection',handoff_controller=True)
    for name,record in state['jobs'].items():
        if name.startswith('roundabout_B_'):
            assert record['state'] in ('queued','paused')
            record.update(state='paused',reason='User requested no further training after T-intersection')
    record=state['jobs'][job['name']]
    record['stages']['training']['adopted_by']=os.getpid()
    q.write(out/'status.json',state)
    training_pid=handoff['training_pid']
    def training_active():
        proc=Path('/proc')/str(training_pid)
        try:
            fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
            if fields[0]=='Z':return False
            if fields[19]!=handoff['training_start_ticks']:return False
            command=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if command!=job['command']:raise RuntimeError('Adopted PID identity changed')
            return True
        except (FileNotFoundError,ProcessLookupError):return False
    current=None
    def stop(signum,frame):
        if current is not None and current.poll() is None:
            try:os.killpg(current.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        print(q.now(),'Adopting existing training',training_pid,'without restarting it',flush=True)
        while training_active():time.sleep(2)
        checkpoint=q.validate_training(job)
        record['stages']['training'].update(state='completed',finished_at=q.now(),
            completion_check='Adopted process exited; complete-budget finite checkpoint and best verified. Exit code unavailable after scheduler handoff.')
        q.write(out/'status.json',state)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HYDRA_FULL_ERROR='1',
            CUDA_VISIBLE_DEVICES='0',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',
            MPLCONFIGDIR='/tmp/r2dreamer-mpl',PYTHONPATH='/home/server/projects/metadrive')
        for test in job['evaluations']:
            stage='test_'+test['name'];state['phase']=job['name']+'/'+stage;record['stage']=stage
            detail=dict(state='running',command=test['command'],started_at=q.now())
            record['stages'][stage]=detail
            with (out/(job['name']+'_'+stage+'.log')).open('w') as stream:
                current=subprocess.Popen(test['command'],cwd=job['source'],env=env,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                detail['pid']=current.pid;q.write(out/'status.json',state)
                print(q.now(),stage,flush=True);code=current.wait()
            detail.update(state='completed' if code==0 else 'failed',returncode=code,finished_at=q.now())
            q.write(out/'status.json',state)
            if code:raise RuntimeError(f'{stage} failed with {code}')
        q.retain_evaluation(out,job,checkpoint)
        record.update(state='completed',finished_at=q.now(),checkpoint=str(checkpoint))
        state.update(phase='paused',paused_at=q.now(),
            reason='T-intersection training and tests finished. Roundabout B held by user request.')
        print(q.now(),'T-intersection finished; queue paused before all roundabout jobs',flush=True)
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        state.update(phase='paused_with_error',error=repr(error),paused_at=q.now())
        traceback.print_exc()
    finally:q.write(out/'status.json',state)


if __name__=='__main__':main()
