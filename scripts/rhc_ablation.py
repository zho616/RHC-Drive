"""Prepare, validate, execute and summarize the protected four-group RHC ablation."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'dreamer.py').is_file() and (p/'iv_output').is_dir())
REF_ROOT = ROOT / 'logdir/tintersection_roundabout_B_seed1_20261025'
REFERENCES = {'A': REF_ROOT / 'tintersection_r2dreamer', 'D': REF_ROOT / 'tintersection_r2dreamer_context'}
VARIANTS = {'A': (False, False, False, 0.), 'B': (True, False, False, 1.),
            'C': (True, True, False, 0.), 'D': (True, True, True, 0.)}
LABELS = {'A': 'R2Dreamer', 'B': 'History-only', 'C': 'Response-guided', 'D': 'Full RHC-Drive'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def flatten(value, prefix=''):
    result = {}
    for key, v in value.items():
        name = prefix + key
        if isinstance(v, dict): result.update(flatten(v, name + '.'))
        else: result[name] = v
    return result


def verify_protection(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for path, expected in manifest['protected'].items():
        if sha(Path(path)) != expected: raise RuntimeError('Protected existing artifact changed: ' + path)
    for path,expected in manifest['source_sha256'].items():
        if sha(out/'source'/path)!=expected:raise RuntimeError('Frozen ablation source changed: '+path)
    return manifest


def prepare(out):
    from omegaconf import OmegaConf
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('source', 'configs', 'training', 'smoke', 'evaluation', 'tables', 'figures'):
        (out / folder).mkdir()
    for path in ROOT.glob('*.py'): shutil.copy2(path, out / 'source' / path.name)
    for folder in ('envs', 'optim', 'configs'):
        shutil.copytree(ROOT / folder, out / 'source' / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'pyproject.toml', out / 'source/pyproject.toml')
    shutil.copy2(ROOT / 'scripts/evaluate_metadrive.py', out / 'evaluate_metadrive.py')
    protected = {}
    for run in REFERENCES.values():
        for path in [run/'best.pt', run/'latest.pt', run/'metrics.jsonl', run/'.hydra/config.yaml']:
            protected[str(path)] = sha(path)
        for path in (run/'evaluation').rglob('*'):
            if path.is_file(): protected[str(path)] = sha(path)
    legacy_source = REF_ROOT / 'sources/source_1'
    for path in legacy_source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts: protected[str(path)] = sha(path)
    template = OmegaConf.load(REFERENCES['D'] / '.hydra/config.yaml')
    assert template.seed == 1 and template.env.steps == 500000
    base_recipe=flatten(OmegaConf.to_container(OmegaConf.load(REFERENCES['A']/'.hydra/config.yaml'),resolve=True))
    full_recipe=flatten(OmegaConf.to_container(template,resolve=True))
    reference_diff={k:dict(A=base_recipe.get(k),D=full_recipe.get(k)) for k in base_recipe.keys()|full_recipe.keys()
                    if base_recipe.get(k)!=full_recipe.get(k)}
    assert set(reference_diff)=={'logdir','model.context.enabled','model.context.swap_weight','env.context_enabled'}
    save(out/'reference_compatibility.json',dict(status='matched_recipe',differences=reference_diff,
        seed=1,configured_budget=500000,validation_episodes=20,validation_frequency=10000,
        selection='Existing trainer lexicographic (success, route completion, return); independent test not used'))
    jobs, differences = {}, {}
    for name, (history, response, exchange, wm) in VARIANTS.items():
        reference = OmegaConf.load(REFERENCES['A'] / '.hydra/config.yaml') if name == 'A' else template
        cfg = copy.deepcopy(reference)
        cfg.model.context.update(dict(use_history_context=history, enabled=history,
            use_response_loss=response, use_context_exchange=exchange, wm_grad_scale=wm,
            aux_weight=1., swap_weight=1. if exchange else 0., log_grad_norms=True))
        cfg.logdir = str(out / 'training' / name)
        cfg.hydra = dict(run=dict(dir='${logdir}'))
        OmegaConf.save(cfg, out / 'configs' / (name + '.yaml'))
        original = OmegaConf.to_container(reference, resolve=True)
        candidate = OmegaConf.to_container(cfg, resolve=True)
        a, b = flatten(original), flatten(candidate)
        diff = {key: dict(reference=a.get(key), variant=b.get(key)) for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        allowed = {'logdir', 'hydra.run.dir'}
        if any(key not in allowed and not key.startswith('model.context.') for key in diff):
            raise RuntimeError('Unexpected training recipe difference: ' + str(diff))
        differences[name] = diff
        for key in ('batch_size', 'batch_length', 'seed'):
            assert candidate[key] == original[key]
        assert candidate['env'] == original['env']
        run = REFERENCES[name] if name in REFERENCES else out / 'training' / name
        source = legacy_source if name in REFERENCES else out / 'source'
        jobs[name] = dict(label=LABELS[name], state='reused_checkpoint' if name in REFERENCES else 'not_trained',
            model=str(run), source=str(source), config=str(out/'configs'/(name+'.yaml')),
            train=[sys.executable, '-u', str(out/'source/train.py'), '--config-path', str(out/'configs'), '--config-name', name],
            evaluations=[])
        for condition, start, models in [('ID',10020,[]),('S',10320,['s']),('XL',10420,['xl'])]:
            command = [sys.executable, '-u', str(out/'evaluate_metadrive.py'), '--checkpoint', str(run/'best.pt'),
                '--source-root', str(source), '--output', str(out/'evaluation'/name/condition),
                '--episodes','100','--seed-start',str(start),'--environment-seed','0',
                '--record-lane-metrics','--device','cpu']
            if models: command += ['--eval-models'] + models
            jobs[name]['evaluations'].append(dict(condition=condition, command=command))
    from subprocess import check_output
    head = check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    version = {str(p.relative_to(out/'source')):sha(p) for p in (out/'source').rglob('*') if p.is_file()}
    manifest = dict(seed=1, budget=500000, git_head=head, dirty_working_tree=True, source_sha256=version,
        protected=protected, jobs=jobs, metadrive_root='/home/server/projects/metadrive',
        gradient_design={'A':'no context','B':'WM(dyn,rep,barlow,rew,con) -> E at scale 1; no predictor',
            'C':'Response -> E,D; WM detached; Exchange off', 'D':'Original Full: Response+Exchange -> E,D; WM detached'},
        causal_limits=['B->C changes both response supervision and gradient source, per explicit user choice; not a pure response-loss attribution.',
                       'C->D isolates Exchange with the same encoder, predictor, posterior, losses and stop-gradient rule.',
                       'A/B/C/D have the same main architecture except the explicitly added history branch and response predictor.'])
    save(out/'manifest.json',manifest); save(out/'config_differences.json',differences)
    if Path('/tmp/rhc_ablation_preflight.json').exists():
        shutil.copy2('/tmp/rhc_ablation_preflight.json', out/'preflight.json')
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', 'export PYTHONPATH=' + shlex.quote(manifest['metadrive_root']),
             'export OMP_NUM_THREADS=1', 'export MKL_NUM_THREADS=1', 'export MPLCONFIGDIR=/tmp/rhc-ablation-matplotlib']
    for name in ('B','C'): lines.append('# '+name+' raw training: '+shlex.join(jobs[name]['train']))
    for phase in ('train','evaluate','report'):
        lines.append(shlex.join([sys.executable,str(ROOT/'scripts/rhc_ablation.py'),phase,'--output',str(out)]))
    (out/'commands.sh').write_text('\n'.join(lines)+'\n')
    save(out/'status.json',dict(phase='prepared', formal_training={'A':'reused','B':'not_run','C':'not_run','D':'reused'}))
    print(out)


def runtime_env(manifest):
    env = dict(os.environ)
    env.update(PYTHONPATH=manifest['metadrive_root'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/rhc-ablation-matplotlib', XDG_CACHE_HOME='/tmp/rhc-ablation-cache')
    return env


def prepare_repeat(out, previous, seed):
    """Repeat B/C without changing their frozen algorithm; re-evaluate seed0 references."""
    from omegaconf import OmegaConf
    if seed != 0: raise ValueError('Only the user-requested seed0 repeat is supported')
    prior=verify_protection(previous)
    references={'A':ROOT/'logdir/tintersection_mass_friction_500k_20261021/training/r2dreamer',
                'D':ROOT/'logdir/tintersection_mass_friction_500k_20261021/training/r2dreamer_dali_shallow'}
    for run in references.values():
        cfg=OmegaConf.load(run/'.hydra/config.yaml')
        assert cfg.seed==seed and cfg.env.steps==500000
        assert (run/'best.pt').is_file() and (run/'latest.pt').is_file()
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('configs','controls','training','evaluation','tables','figures','validation'):
        (out/folder).mkdir()
    shutil.copytree(previous/'source',out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(previous/'evaluate_metadrive.py',out/'evaluate_metadrive.py')
    protected=dict(prior['protected']);jobs={};differences={}
    for name in 'ABCD':
        old=OmegaConf.load(previous/'configs'/(name+'.yaml'));cfg=copy.deepcopy(old)
        old_seed=int(old.env.seed)
        cfg.seed=seed;cfg.env.seed='${seed}'
        cfg.env.start_seed=int(old.env.start_seed)+old_seed-seed
        cfg.env.eval_start_seed=int(old.env.eval_start_seed)+old_seed-seed
        cfg.logdir=str(out/'training'/name)
        a=flatten(OmegaConf.to_container(old,resolve=True));b=flatten(OmegaConf.to_container(cfg,resolve=True))
        diff={k:dict(seed1=a.get(k),seed0=b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        assert set(diff)=={'seed','env.seed','env.start_seed','env.eval_start_seed','logdir','hydra.run.dir'}
        assert cfg.env.seed+cfg.env.start_seed==0 and cfg.env.seed+cfg.env.eval_start_seed==10000
        assert cfg.model==old.model
        assert cfg.model.context.wm_grad_scale==(1. if name=='B' else 0.)
        OmegaConf.save(cfg,out/'configs'/(name+'.yaml'));differences[name]=diff
        run=references[name] if name in references else out/'training'/name
        source=run.parent/'source' if name in references else out/'source'
        assert source.is_dir()
        if name in references:
            actual=OmegaConf.load(run/'.hydra/config.yaml')
            x=flatten(OmegaConf.to_container(actual,resolve=True))
            mismatch={k:dict(existing=x.get(k),repeat=b.get(k)) for k in x.keys()|b.keys() if x.get(k)!=b.get(k)}
            assert all(k in ('logdir','hydra.run.dir') or k.startswith('model.context.') for k in mismatch), mismatch
        jobs[name]=dict(label=LABELS[name],state='reused_checkpoint' if name in references else 'not_trained',
            model=str(run),source=str(source),config=str(out/'configs'/(name+'.yaml')),
            train=[sys.executable,'-u',str(out/'source/train.py'),'--config-path',str(out/'configs'),'--config-name',name],
            evaluations=[])
        for condition,start,models in [('ID',10020,[]),('S',10320,['s']),('XL',10420,['xl'])]:
            command=[sys.executable,'-u',str(out/'evaluate_metadrive.py'),'--checkpoint',str(run/'best.pt'),
                '--source-root',str(source),'--output',str(out/'evaluation'/name/condition),'--episodes','100',
                '--seed-start',str(start),'--environment-seed','0','--record-lane-metrics','--device','cpu']
            if models:command+=['--eval-models']+models
            jobs[name]['evaluations'].append(dict(condition=condition,command=command))
        original_run=references.get(name,Path(prior['jobs'][name]['model']))
        for p in [original_run/'best.pt',original_run/'latest.pt',original_run/'metrics.jsonl',original_run/'.hydra/config.yaml']:
            protected[str(p)]=sha(p)
        if name in references:
            for p in source.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':protected[str(p)]=sha(p)
    manifest=copy.deepcopy(prior)
    manifest.update(seed=seed,jobs=jobs,protected=protected,repeat_of=str(previous),
        source_sha256={str(p.relative_to(out/'source')):sha(p) for p in (out/'source').rglob('*') if p.is_file()},
        repeat_protocol='Fresh B/C initialization, seed0, unchanged frozen seed1 B/C source and model settings; compensated map offsets retain train0-99/validation10000-10019; existing seed0 A/D weights re-evaluated with matching native-output tests.')
    save(out/'manifest.json',manifest);save(out/'config_differences.json',differences)
    save(out/'status.json',dict(phase='prepared',formal_training={'A':'reused','B':'not_run','C':'not_run','D':'reused'}))
    shutil.copy2(Path(__file__),out/'controls/rhc_ablation.py')
    lines=['#!/usr/bin/env bash','set -euo pipefail']
    for phase in ('train','evaluate','report'):
        lines.append(shlex.join([sys.executable,str(out/'controls/rhc_ablation.py'),phase,'--output',str(out)]))
    (out/'commands.sh').write_text('\n'.join(lines)+'\n')
    print(out)


def audit(out):
    """Production-size FP32 gradients on real paired replay prefixes; no update."""
    manifest = verify_protection(out)
    sys.path.insert(0, str(out/'source'))
    import gymnasium as gym
    import numpy as np
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from dreamer import Dreamer
    torch.set_num_threads(1)
    corpus = ROOT/'logdir/tintersection_mass_friction_500k_20261021/world_model_common_corpus'
    metadata = json.loads((corpus/'episodes.json').read_text())
    ids = [i for i,e in enumerate(metadata) if e['length'] >= 64][:16]
    assert len(ids) == 16
    arrays = []
    for i in ids:
        with np.load(corpus/'episodes'/metadata[i]['file']) as z: d = {k:v.copy() for k,v in z.items()}
        executed = d['executed_action'][:-1].clip(-1, 1)
        np.testing.assert_array_equal(d['ctx_prev_action'][1:], np.stack((executed[:,1], -executed[:,0]), -1))
        for t in range(min(64,len(d['state']))):
            n=min(t,5)
            np.testing.assert_array_equal(d['ctx_mask'][t], np.arange(5)>=5-n)
            if n:
                tokens=np.concatenate((d['ctx_ego_state'][t-n+1:t+1], d['ctx_prev_action'][t-n+1:t+1]),-1)
                np.testing.assert_array_equal(d['ctx_history'][t,-n:], tokens)
        d['action'] = np.concatenate([np.zeros((1,2),np.float32),d['executed_action'][:-1]])
        d['ctx_episode_uid'] = np.tile([0,i+1],(len(d['state']),1))
        for k,v in list(d.items()):
            if v.ndim == 1: d[k] = v[:,None]
        arrays.append(d)
    excluded = {'executed_action','speed','wheel_degrees'}
    keys = [k for k in arrays[0] if k not in excluded]
    td = TensorDict({k:torch.as_tensor(np.stack([d[k][:64] for d in arrays])) for k in keys},batch_size=(16,64))
    td['reward'] = td['reward'].float()
    obs = gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=v.shape[2:],dtype=np.float32)
                          for k,v in td.items() if k not in ('action','reward')})
    act = gym.spaces.Box(-1,1,(2,),np.float32)
    results = []
    for name in 'ABCD':
        cfg = OmegaConf.load(out/'configs'/(name+'.yaml'));cfg.device='cpu';OmegaConf.resolve(cfg);cfg.model.compile=False
        torch.manual_seed(int(cfg.seed)); model=Dreamer(cfg.model,obs,act);model._scaler=torch.amp.GradScaler(enabled=False)
        torch.manual_seed(333);_, metrics=model._cal_grad(td,model.rssm.initial(16))
        norms={}
        for module_name in ('context_encoder','context_dynamics','rssm','actor','value'):
            module=getattr(model,module_name,None)
            grads=[p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None] if module is not None else []
            norms[module_name]=float(torch.stack(grads).norm()) if grads else None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if name!='A': assert norms['context_encoder']>0
        if name=='D': assert float(metrics['context/swap_pairs'])>0
        results.append(dict(method=name, optimized_parameters=sum(p.numel() for p in model._named_params.values()),
            module_parameters={k:sum(p.numel() for p in getattr(model,k).parameters()) if hasattr(model,k) else 0
                for k in ('encoder','rssm','actor','value','reward','cont','prj','context_encoder','context_dynamics')},
            gradient_norms=norms, losses={k:float(v.detach()) for k,v in metrics.items() if k.startswith('loss/')},
            swap_pairs=float(metrics.get('context/swap_pairs',0)), wm_input_grad_norm=float(metrics.get('context/wm_input_grad_norm',0))))
        print(name,results[-1],flush=True)
        del model, metrics
    save(out/'gradient_audit.json',dict(precision='FP32 CPU, before AGC/optimizer',batch=[16,64],episodes=ids,results=results,
        alignment='Exact canonical previous-action shift and every K5 causal history token checked on all 16 real prefixes.',
        limits='Real cached prefixes at fresh initialization; paired trajectories overrepresent exchange availability. No parameter updates or claims of training performance.'))
    verify_protection(out)


def smoke(out):
    manifest=verify_protection(out)
    from omegaconf import OmegaConf
    # Test harness only: CPU FP16 without CUDA loss scaling produced NaNs even
    # for baseline. Disable only update's outer autocast; production code/configs
    # and all formal training commands retain their original precision.
    driver = ('import contextlib,runpy,sys; source,configs,name=sys.argv[1:]; '
              'sys.path.insert(0,source); import dreamer; '
              'dreamer.autocast=lambda **kw: contextlib.nullcontext(); '
              'sys.argv=[source+"/train.py","--config-path",configs,"--config-name",name]; '
              'runpy.run_path(source+"/train.py",run_name="__main__")')
    save(out/'smoke_protocol.json',dict(precision='FP32 CPU; eager; diagnostic harness disables outer update autocast only',
         driver=driver, formal_result=False, steps=96, batch=[2,8], train_ratio=8,
         chassis='nominal Default for branch coverage, identical for all methods',
         limits='Does not validate CUDA FP16 / torch.compile or 500K performance.'))
    reports=[]
    for name in 'ABCD':
        cfg=OmegaConf.load(out/'configs'/(name+'.yaml'));cfg.device='cpu';cfg.model.compile=False
        cfg.logdir=str(out/'smoke'/name);cfg.batch_size=2;cfg.batch_length=8
        cfg.env.steps=96;cfg.env.train_ratio=8;cfg.env.env_num=2;cfg.env.eval_env_num=1;cfg.env.eval_episode_num=1
        cfg.env.num_scenarios=2;cfg.env.eval_num_scenarios=1;cfg.env.time_limit=24
        cfg.buffer.max_size=256;cfg.buffer.storage_device='cpu'
        cfg.trainer.eval_every=48;cfg.trainer.update_log_every=16;cfg.trainer.save_every=32
        # Identical branch-coverage setting across all smoke groups, not paper results.
        for prefix in ('','eval_'):
            cfg.env.chassis[prefix+'models']=['default']
            for key in ('mass_scales','steering_scales','friction_scales'):cfg.env.chassis[prefix+key]=[1.]
        path=out/'configs'/('smoke_'+name+'.yaml');OmegaConf.save(cfg,path)
        command=[sys.executable,'-u','-c',driver,str(out/'source'),str(out/'configs'),'smoke_'+name]
        if (out/'smoke'/name).exists():raise RuntimeError('Use a fresh smoke directory')
        with (out/('smoke_'+name+'.log')).open('w') as log:
            result=subprocess.run(command,env=runtime_env(manifest),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row=dict(method=name,returncode=result.returncode,formal_result=False)
        if result.returncode:
            reports.append(row);save(out/'smoke_results.json',reports)
            raise RuntimeError('Smoke failed; inspect '+str(out/('smoke_'+name+'.log')))
        records=[json.loads(line) for line in (out/'smoke'/name/'metrics.jsonl').open()]
        records=[r for r in records if 'train/loss/dyn' in r]
        if not records:raise RuntimeError('Smoke contained no optimizer updates')
        import math
        assert all(math.isfinite(v) for r in records for k,v in r.items() if isinstance(v,(int,float)))
        norms=[r['train/grad/context_encoder_norm'] for r in records if 'train/grad/context_encoder_norm' in r]
        if name!='A':assert max(norms)>0
        row.update(logged_updates=len(records),encoder_grad_norms=norms)
        reports.append(row);save(out/'smoke_results.json',reports)
    verify_protection(out)


def train(out):
    manifest=verify_protection(out)
    import torch
    cuda_available=torch.cuda.is_available()
    completed=[]
    for name in ('B','C'):
        directory=Path(manifest['jobs'][name]['model'])
        if directory.exists():
            if (directory/'latest.pt').exists() and (directory/'best.pt').exists():
                payload=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
                finished=payload['step']>=manifest['budget'];del payload
                if finished:completed.append(name);continue
            raise RuntimeError('Refuse to overwrite a partial/existing run; original trainer has no replay-preserving resume: '+str(directory))
        if not cuda_available:
            save(out/'training_resource_status.json',dict(status='not_run',reason='CUDA unavailable; no 500K CPU substitute',
                 methods=[n for n in ('B','C') if n not in completed],completed=completed))
            raise SystemExit('CUDA unavailable: pending 500K training not started. Reproducible commands are in commands.sh.')
        save(out/'training_resource_status.json',dict(status='running',current=name,completed=completed))
        try:
            with (out/('training_'+name+'.log')).open('w') as log:
                subprocess.run(manifest['jobs'][name]['train'],env=runtime_env(manifest),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        except BaseException as error:
            save(out/'training_resource_status.json',dict(status='failed',current=name,completed=completed,error=repr(error)))
            raise
        completed.append(name)
    save(out/'training_resource_status.json',dict(status='completed',completed=completed))
    verify_protection(out)


def evaluate(out, references_only=False, smoke_only=False):
    manifest=verify_protection(out)
    for name in ('AD' if references_only else 'ABCD'):
        job=manifest['jobs'][name]
        if not (Path(job['model'])/'best.pt').exists():continue
        if name not in REFERENCES:
            latest=Path(job['model'])/'latest.pt'
            if not latest.exists():continue
            import torch
            payload=torch.load(latest,map_location='cpu',weights_only=False)
            finished=payload['step']>=manifest['budget'];del payload
            if not finished:continue
        for evaluation in job['evaluations']:
            command=list(evaluation['command']);condition=evaluation['condition']
            if smoke_only:
                if condition!='ID':continue
                command[command.index('--episodes')+1]='1'
                command[command.index('--output')+1]=str(out/'evaluation_smoke'/name)
            existing=Path(command[command.index('--output')+1])/'summary.json'
            if existing.exists():
                result=json.loads(existing.read_text())
                if result['status']=='completed' and result['completed_episodes']==int(command[command.index('--episodes')+1]):
                    continue
                raise RuntimeError('Incomplete evaluation exists; preserve it and use a fresh output: '+str(existing.parent))
            with (out/('evaluation_'+name+'_'+condition+('_smoke' if smoke_only else '')+'.log')).open('w') as log:
                subprocess.run(command,env=runtime_env(manifest),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify_protection(out)


def checkpoints(out):
    """Strict loading of actual independent smoke checkpoints, with fresh models."""
    manifest=verify_protection(out)
    sys.path.insert(0,str(out/'source'))
    import gymnasium as gym
    import numpy as np
    import torch
    from tensordict import TensorDict
    from omegaconf import OmegaConf
    from dreamer import Dreamer
    torch.set_num_threads(1)
    corpus=ROOT/'logdir/tintersection_mass_friction_500k_20261021/world_model_common_corpus'
    metadata=json.loads((corpus/'episodes.json').read_text())
    with np.load(corpus/'episodes'/metadata[0]['file']) as z:
        fields={k:torch.as_tensor(z[k][0].copy()) for k in z.files if k not in ('executed_action','speed','wheel_degrees','reward')}
    for k,v in list(fields.items()):fields[k]=v.reshape(1,1) if v.ndim==0 else v.unsqueeze(0)
    obs=gym.spaces.Dict({k:gym.spaces.Box(-np.inf,np.inf,shape=v.shape[1:],dtype=np.float32) for k,v in fields.items()})
    act=gym.spaces.Box(-1,1,(2,),np.float32);td=TensorDict(fields,batch_size=(1,))
    results=[]
    for name in 'ABCD':
        cfg=OmegaConf.load(out/'configs'/('smoke_'+name+'.yaml'));OmegaConf.resolve(cfg)
        path=out/'smoke'/name/'latest.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model=Dreamer(cfg.model,obs,act)
        model.load_state_dict(payload['agent_state_dict'],strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model.eval();torch.manual_seed(333)
        action,_=model.act(td,model.get_initial_state(1),eval=True)
        assert torch.isfinite(action).all()
        results.append(dict(method=name,checkpoint=str(path),sha256=sha(path),step=payload['step'],
                            optimizer_updates=payload['update_count'],strict_load=True,
                            action=action.tolist(),payload_keys=list(payload)))
        del model,payload
    save(out/'checkpoint_roundtrip.json',dict(status='passed',results=results))
    verify_protection(out)


def sampling(out):
    """Actual resets with the original randomized training distributions."""
    manifest=verify_protection(out)
    sys.path.insert(0,str(out/'source'))
    import numpy as np
    from omegaconf import OmegaConf
    from envs.metadrive import MetaDrive
    sequences={}
    for name in 'ABCD':
        cfg=OmegaConf.load(out/'configs'/(name+'.yaml'));OmegaConf.resolve(cfg)
        env=MetaDrive(cfg.env)
        sequence=[]
        try:
            for _ in range(4):
                obs=env.reset()
                sequence.append(dict(seed=int(env._env.current_seed),chassis=env.chassis_parameters,
                                     state_sha256=hashlib.sha256(obs['state'].tobytes()).hexdigest()))
        finally:env.close()
        sequences[name]=sequence
    assert all(sequence==sequences['A'] for sequence in sequences.values())
    save(out/'sampling_reproducibility.json',dict(status='passed',training_resets_per_method=4,sequences=sequences))
    verify_protection(out)


def report(out):
    """Only completed 100-episode evaluations enter formal tables/intervals."""
    manifest=verify_protection(out)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    resource_path=out/'training_resource_status.json'
    resource_state=json.loads(resource_path.read_text()) if resource_path.exists() else {}
    training_states={n:'reused_500k' for n in REFERENCES}
    for n in ('B','C'):
        checkpoint=Path(manifest['jobs'][n]['model'])/'latest.pt'
        training_states[n]='not_run'
        if checkpoint.exists():
            import torch
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            training_states[n]='completed' if payload['step']>=manifest['budget'] else 'incomplete'
            del payload
        if resource_state.get('status')=='running' and resource_state.get('current')==n:
            training_states[n]='running'
    new_training_complete=all(training_states[n]=='completed' for n in ('B','C'))
    conditions=('ID','S','XL')
    metrics=('success','route_completion','out_of_lane','crash','reward','mean_speed_km_h','steps',
             'out_of_road','out_of_lane_step_fraction')
    data={}; summaries=[]
    for name in 'ABCD':
        for condition in conditions:
            path=out/'evaluation'/name/condition
            if not (path/'summary.json').exists():continue
            summary=json.loads((path/'summary.json').read_text())
            if summary['status']!='completed' or summary['completed_episodes']!=100:continue
            rows=[json.loads(line) for line in (path/'episodes.jsonl').read_text().splitlines()]
            assert len(rows)==100 and len({r['seed'] for r in rows})==100
            data[name,condition]=rows
            summaries.append(dict(method=name,label=LABELS[name],condition=condition,
                                  checkpoint_step=summary['checkpoint_step'],**summary['summary']))
    def table_csv(path,rows,fields=None):
        if not rows:return
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
    table_csv(out/'tables/condition_metrics.csv',summaries)
    comparisons=[]
    for other in 'ABC':
        for condition in conditions:
            if ('D',condition) not in data or (other,condition) not in data:continue
            base={r['seed']:r for r in data[other,condition]}
            full={r['seed']:r for r in data['D',condition]}
            assert base.keys()==full.keys()
            seeds=sorted(base)
            assert all(base[s]['chassis']==full[s]['chassis'] for s in seeds), 'Not chassis-paired'
            draws=np.random.default_rng(20261026).integers(0,len(seeds),(10000,len(seeds)))
            for metric in metrics:
                delta=np.array([float(full[s][metric])-float(base[s][metric]) for s in seeds])
                ci=np.quantile(delta[draws].mean(1),[.025,.975])
                comparisons.append(dict(comparison='D-'+other,condition=condition,metric=metric,n=100,
                    difference=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                    full_only_wins=sum(full[s]['success'] and not base[s]['success'] for s in seeds),
                    baseline_only_wins=sum(base[s]['success'] and not full[s]['success'] for s in seeds)))
    table_csv(out/'tables/paired_bootstrap.csv',comparisons)
    save(out/'paired_bootstrap.json',dict(replicates=10000,seed=20261026,unit='paired map/chassis episode',
        interval='percentile 95%; one training seed, no across-training-seed inference',results=comparisons))
    lookup={(r['method'],r['condition']):r for r in summaries}
    def metric(name,condition,key):return lookup.get((name,condition),{}).get(key)
    def fmt(value,scale=1,decimals=1):return '--' if value is None else f'{value*scale:.{decimals}f}'
    main_rows=[];latex=[]
    for name in 'ABCD':
        returns=[metric(name,c,'reward') for c in conditions]
        reward=float(np.mean(returns)) if all(v is not None for v in returns) else None
        values=[fmt(metric(name,c,'success'),100) for c in conditions]+[fmt(metric(name,c,'out_of_lane'),100) for c in conditions]+[fmt(reward)]
        main_rows.append(dict(Scenario='T-intersection',Method=LABELS[name],**dict(zip(
            ('ID Success','S Success','XL Success','ID OOL','S OOL','XL OOL','Return'),values))))
        latex.append('T-intersection & '+LABELS[name]+' & '+' & '.join(values)+r' \\')
    table_csv(out/'tables/ablation_summary.csv',main_rows)
    (out/'tables/ablation_summary.tex').write_text('\n'.join([
        r'\begin{table*}[t]',r'\centering',r'\caption{Context ablation on T-intersection. Success and out-of-lane (OOL) are percentages. Return is the equal-weight mean across ID/S/XL. Each completed condition has 100 paired episodes. Dashes indicate experiments not yet trained/evaluated; they are not zero scores.}',
        r'\label{tab:context-ablation}',r'\begin{tabular}{llrrrrrrr}',r'\hline',
        r'Scenario & Method & ID Succ. & S Succ. & XL Succ. & ID OOL & S OOL & XL OOL & Return \\',r'\hline',
        *latex,r'\hline',r'\end{tabular}',r'\end{table*}'])+'\n')
    audit_data=json.loads((out/'gradient_audit.json').read_text())
    params=[]
    for r in audit_data['results']:
        name=r['method'];history,response,exchange,wm=VARIANTS[name]
        params.append(dict(method=name,history=history,response=response,exchange=exchange,wm_to_encoder_scale=wm,
            optimized_parameters=r['optimized_parameters'],encoder_parameters=r['module_parameters']['context_encoder'],
            predictor_parameters=r['module_parameters']['context_dynamics'],rssm_parameters=r['module_parameters']['rssm'],
            encoder_gradient_norm=r['gradient_norms']['context_encoder'],predictor_gradient_norm=r['gradient_norms']['context_dynamics'],
            rssm_gradient_norm=r['gradient_norms']['rssm']))
    table_csv(out/'tables/parameters_gradients.csv',params)
    # Scientific figure contract: show measured differences, pending groups are
    # explicit; no invented variance or inferred monotonic module gains.
    save(out/'figure_contract.json',dict(backend='Python/matplotlib',exports=['PNG 200 dpi','vector PDF'],
        conclusion='Measured A/D performance and uncertainty; B/C pending, no monotonicity claim',
        evidence='Completed 100-episode paired evaluations; original formal A/D logs; CPU gradients separate',
        risks=['One training seed','B/C formal data missing','B->C gradient-source confound'],
        smoothing='Training raw traces plus EMA alpha=.08, no confidence band'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    colors={'A':'#707070','B':'#377EB8','C':'#4DAF4A','D':'#D95F02'}
    def export(fig,name):
        fig.savefig(out/'figures'/(name+'.png'),dpi=200);fig.savefig(out/'figures'/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),sharey=True,layout='constrained')
    for ax,condition in zip(axes,conditions):
        for i,name in enumerate('ABCD'):
            value=metric(name,condition,'success')
            if value is None:
                ax.scatter(i,3,marker='_',s=100,color='#b0b0b0')
                ax.text(i,8,'Not trained' if name not in REFERENCES else 'Not evaluated',
                        rotation=90,ha='center',va='bottom',color='#777777',fontsize=8)
            else:
                ax.bar(i,100*value,color=colors[name],width=.7)
                ax.text(i,100*value+2,f'{100*value:.0f}%',ha='center',fontsize=9)
        ax.set(title={'ID':'ID: Default / M / L','S':'OOD: S','XL':'OOD: XL'}[condition],xticks=range(4),xticklabels=list('ABCD'),ylim=(0,110))
        ax.set_yticks([0,25,50,75,100]);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Success rate (%)')
    fig.legend(handles=[Patch(facecolor=colors[n],label=n+': '+LABELS[n]) for n in 'ABCD'],loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    export(fig,'success_ablation')
    fig,ax=plt.subplots(figsize=(6.5,3.2),layout='constrained')
    dots=[r for r in comparisons if r['metric']=='success' and r['condition'] in ('S','XL')]
    for y,r in enumerate(dots):
        d=100*r['difference'];lo=100*r['ci95_low'];hi=100*r['ci95_high']
        ax.errorbar(d,y,xerr=[[max(0,d-lo)],[max(0,hi-d)]],fmt='o',color=colors['D'],capsize=4)
        ax.annotate(f'{d:+.1f} [{lo:+.1f}, {hi:+.1f}]',(d,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    ax.axvline(0,color='#999999',ls='--',lw=1)
    ax.set(yticks=range(len(dots)),yticklabels=[r['condition']+'  '+r['comparison'] for r in dots],
           xlabel='Paired success difference (percentage points), bootstrap 95% CI',ylim=(-.6,max(.6,len(dots)-.3)))
    if not dots:ax.text(.5,.5,'Paired evaluations pending',transform=ax.transAxes,ha='center')
    export(fig,'ood_paired_differences')
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout='constrained')
    keys=('episode/eval_score','episode/eval_success','train/loss/context_aux')
    curve_rows=[]
    for name in 'ABCD':
        log=Path(manifest['jobs'][name]['model'])/'metrics.jsonl'
        if not log.exists():continue
        records=[json.loads(line) for line in log.open()]
        for ax,key in zip(axes,keys):
            points=[(r['step'],r[key]) for r in records if key in r]
            if not points:continue
            x,y=np.array(points).T;ema=[];last=y[0]
            for value in y:last=.08*value+.92*last;ema.append(last)
            ax.plot(x/1000,y,color=colors[name],lw=.6,alpha=.2)
            ax.plot(x/1000,ema,color=colors[name],lw=1.3,label=name+': '+LABELS[name])
            curve_rows.extend(dict(method=name,metric=key,step=int(step),raw=float(value),ema=float(smooth)) for step,value,smooth in zip(x,y,ema))
    for ax,title in zip(axes,('Validation episode return','Validation success','Response normalized L1')):
        ax.set(title=title,xlabel='Training interactions (×1000)');ax.grid(alpha=.15)
        if ax.lines:ax.legend(fontsize=7,frameon=False)
    axes[1].set_ylim(-.03,1.03)
    axes[2].text(.98,.96,('' if new_training_complete else 'B/C: formal runs pending\n')+'Raw + EMA (α = 0.08)',transform=axes[2].transAxes,ha='right',va='top',fontsize=7,color='#555555')
    export(fig,'formal_training_curves');table_csv(out/'tables/training_curves.csv',curve_rows)
    lines=['# RHC-Drive Context 消融实验报告','',
        '本报告仅统计已完成的正式评估。短程 Smoke Test 不计入论文性能结果。'+('B/C 的 500k 训练已完成。' if new_training_complete else 'B/C 的 500k 训练尚未全部完成。'),'',
        '## 实验与梯度定义','',
        '| 组 | History | Response | Exchange | WM→Encoder | 可训练参数量 |',
        '|---|---|---|---|---|---|']
    for p in params:lines.append(f"| {p['method']} {LABELS[p['method']]} | {p['history']} | {p['response']} | {p['exchange']} | {p['wm_to_encoder_scale']} | {p['optimized_parameters']:,} |")
    lines += ['', '参数量按优化器中的唯一可训练参数统计，不重复计算冻结共享副本或目标网络。History Encoder 为 68,232，posterior 新增输入权重 2,048；C/D 的 Response Predictor 为 18,948。Actor-Critic 参数量四组相同。',
        'B 通过 posterior 的世界模型损失 dyn/rep/barlow/rew/con 训练 Context Encoder；不将 Actor-Critic 或 repval 的梯度作为其监督。B 不构造 Response Predictor，也不执行预测/交换路径。',
        'C/D 保留旧 Full 的 stop-gradient：WM 不更新 Context Encoder；Response/Exchange 更新 Encoder 和 Predictor，不更新 RSSM。C 与 D 仅交换损失不同。',
        'WM→Encoder 为零不等于世界模型不能优化：RSSM 及预测头仍由 WM 损失更新，Encoder 则由辅助监督更新。模块能否提升预测和驾驶性能需要实际结果支持。',
        '**因果限制：按用户指定的四组设计，B→C 同时改变 Response 监督和 WM→Encoder 梯度来源，不能将差异完全归因于新增 Response Loss。C→D 才是严格的交换监督消融。**','',
        '共同目标为 L_WM = L_dyn + 0.1 L_rep + 0.05 L_barlow + L_rew + L_con；再加原始 L_policy + L_value + 0.3 L_repval。C/D 增加 1×L_response，D 再加 1×L_exchange。B 以零值 surrogate 传递所选 WM 目标对 context 的一阶梯度，保持原世界模型参数的梯度不变。','',
        '## 实现审计','',
        '- History 输入为过去最多 5 个已完成动作—响应端点；每个 token 为加速度、横摆角速度、速度、实际前轮角和先前执行的两个动作。无底盘标签输入。',
        '- Encoder：两个 Transformer block，hidden=64，4 heads，FFN=128，masked mean pooling，context_dim=8。posterior 将 context 与 h_t、观测 embedding 拼接；transition/prior、reward/continuation 头及 Actor-Critic 无直接 context 输入。',
        '- Response：固定锚点 context、最多 5 步残差预测；每一步预测状态 detach，context 不 detach；L1 按 [10,1,30,π/3] 归一化，权重 1。后续误差直接更新本步 Predictor 和 Encoder，但不穿过前面所有预测状态。',
        '- Exchange：只选择不同真实 episode UID 且完整配置 (model_id,mass,max_steering,wheel_friction) 完全相同的 donor，须完整 K5 历史；使用 recipient 状态/动作做一步预测，donor context 不 detach，权重 1。不是只匹配车型。',
        '- 历史、动作移位、终止/碰撞屏蔽、无有效交换样本与梯度隔离均有检查；真实轨迹前缀用于生产规格梯度诊断。',
        '- Full 与冻结旧实现的同输入损失及参数梯度完全一致；已有 checkpoint、训练日志、评估文件、旧源码均 SHA256 保护，未改写。','',
        '## 已完成程度','']
    for name in 'ABCD':
        completed=[c for c in conditions if (name,c) in data]
        description=f"复用已有 seed={manifest['seed']}、500k 训练" if name in REFERENCES else '正式训练状态 '+training_states[name]
        lines.append(f"- {name}：{description}；完成评估：{', '.join(completed) or '无'}。")
    smoke_path=out/'smoke_results.json'
    if smoke_path.exists():
        for r in json.loads(smoke_path.read_text()):lines.append(f"- {r['method']} FP32 CPU Smoke：exit={r['returncode']}，记录更新窗口 {r.get('logged_updates',0)}，Encoder 梯度 {r.get('encoder_grad_norms',[])}。")
    checkpoint_path=out/'checkpoint_roundtrip.json'
    if checkpoint_path.exists():
        updates={r['method']:r['optimizer_updates'] for r in json.loads(checkpoint_path.read_text())['results']}
        lines.append('四组 Smoke 均从随机初始化开始，使用独立目录，未加载或微调既有 A/D 权重；每组 96 交互步，实际优化次数 '+str(updates)+'。')
    for path,label in [('sampling_reproducibility.json','四组真实训练环境同种子重置配置与初始观测一致（每组 4 次）'),
                       ('checkpoint_roundtrip.json','四组独立 Smoke checkpoint 严格加载并推断成功')]:
        if (out/path).exists():lines.append('- '+label+'。')
    lines += ['', ('本次复用已验证且 SHA256 完全一致的 B/C 源码；此前 seed1 的 Smoke 结果不冒充本次 seed0 测量。seed0 真实 batch 梯度诊断单独保存在 gradient_audit.json。' if 'repeat_of' in manifest else
        'CPU FP16 的首次 Baseline Smoke 出现 NaN，失败日志保留。随后统一使用仅作用于 Smoke 的 FP32/eager 测试适配器；未修改正式训练精度或旧 Full。'),
        ('补充 CUDA 审计：沙箱外 RTX 5060 Ti 可用，四组生产 batch=16×64 的 FP16/eager 损失与梯度均有限、B Encoder 梯度非零。正式 compile/长程运行状态以训练日志为准。' if (out/'validation/cuda_amp_audit.json').exists() else 'CUDA FP16/compile 与长程训练尚未验证。'),
        'B/C 正式 500k 训练状态见上文。初次 CUDA 不可用记录来自沙箱设备隔离；后续 GPU 检查与启动记录单独保存。完整配置、源码快照、训练/评估命令已保存。','',
        '## 性能汇总','',
        '| Scenario | Method | ID Success | S Success | XL Success | ID OOL | S OOL | XL OOL | Return |',
        '|---|---|---|---|---|---|---|---|---|']
    for row in main_rows:lines.append('| '+' | '.join(str(v) for v in row.values())+' |')
    lines += ['', 'Success/OOL 单位为 %；Return 为 ID/S/XL 三个条件等权平均。OOL 为回合内任一步 native `off_lane=true`；另记录逐步比例和 out_of_road，二者不能混用。缺失项 -- 表示未运行。',
        f"各条件 100 回合，同地图、同完整底盘、同环境种子 0、同推断 RNG、相同测试种子列表；seed={manifest['seed']}，模型均以原训练验证协议选 best，训练预算均 500k，不使用测试集选 checkpoint。",'',
        '评估用同一 CPU/FP32/eager 设置及各自原训练冻结源码；可能与既往 GPU 随机潜状态采样结果存在数值差异，应引用本次统一重评估，而非混用旧测试数值。','',
        '| 条件 | 对比 | 成功率差 pp | 配对 bootstrap 95% CI |', '|---|---|---|---|']
    for r in comparisons:
        if r['metric']=='success':lines.append(f"| {r['condition']} | {r['comparison']} | {100*r['difference']:+.1f} | [{100*r['ci95_low']:+.1f}, {100*r['ci95_high']:+.1f}] |")
    lines += ['', '区间为相同 100 个回合配对重采样 10000 次的 percentile bootstrap（seed=20261026），只表示测试回合的不确定性，不代表多训练种子的稳定性。','',
        '| 条件/方法 | Route completion | Collision % | Out-of-road % | Mean speed km/h | Episode length | Return |', '|---|---|---|---|---|---|---|']
    for r in summaries:lines.append(f"| {r['condition']}/{r['method']} | {r['route_completion']:.3f} | {100*r['crash']:.1f} | {100*r['out_of_road']:.1f} | {r['mean_speed_km_h']:.2f} | {r['steps']:.1f} | {r['reward']:.2f} |")
    if summaries and all(r['out_of_lane']==0 for r in summaries):
        lines += ['', '**指标限制：本次原生 off_lane 在所有已完成条件均为 0，OOL 无区分度。不能据此宣称没有越线或驶出道路；out_of_road 是另一项已单独报告的终止事件指标。**']
    lines += ['', '## 梯度诊断（真实轨迹，初始化时 FP32，AGC 前）','',
        '| Method | Encoder norm | Predictor norm | RSSM norm |', '|---|---|---|---|']
    for p in params:lines.append(f"| {p['method']} | {fmt(p['encoder_gradient_norm'],decimals=6)} | {fmt(p['predictor_gradient_norm'],decimals=6)} | {fmt(p['rssm_gradient_norm'],decimals=3)} |")
    lines += ['', '这验证了训练信号存在与路径正确，不证明长期梯度稳定或任务性能提高。真实 batch=16×64；交换候选来自成对采集的真实轨迹，不能把此 batch 的覆盖率当作在线 Replay 的典型覆盖率。',
        '旧 A/D 正式日志未记录逐模块梯度，不能追补为历史测量；新 B/C 配置已开启真实 unscale 后、AGC 前的梯度范数记录。','',
        '## 对方法创新性的支持程度','',
        ('完整四组结果已具备时，应按各条件的实际数值与置信区间判断模块贡献，不能预设 A<B<C<D。' if len(data)==12 else
         '当前能验证实现与梯度语义；A/D 的统一重评估只提供完整方法相对 Baseline 的证据，不能替代模块消融。History-only 是否优于 Baseline、Response-guided 是否优于 History-only、Full 是否进一步优于 Response-guided，均须等待 B/C 正式训练及配对评估。不能预设 A<B<C<D。')]
    for c in conditions:
        if ('A',c) in lookup and ('D',c) in lookup:
            a=lookup['A',c];d=lookup['D',c]
            lines.append(f"{c}：D−A 成功率 {100*(d['success']-a['success']):+.1f} pp，碰撞 {100*(d['crash']-a['crash']):+.1f} pp，速度 {d['mean_speed_km_h']-a['mean_speed_km_h']:+.2f} km/h，回合长度 {d['steps']-a['steps']:+.1f}；需结合诊断表判断保守驾驶和碰撞代价。")
            if d['success']>a['success']:
                lines.append('成功率提升同时平均速度降低，存在保守驾驶的替代解释。' if d['mean_speed_km_h']<a['mean_speed_km_h'] else '成功率提升未伴随平均速度降低，本条件不支持“仅靠低速保守驾驶”的解释。')
            lines.append('碰撞率点估计上升，需如实报告这一安全代价。' if d['crash']>a['crash'] else '未观察到碰撞率点估计上升；不等于已证明安全性提升。')
    if all((n,c) in lookup for n in 'AD' for c in conditions):
        gains={c:100*(lookup['D',c]['success']-lookup['A',c]['success']) for c in conditions}
        lines.append(f"OOD 相对 ID 的增益点估计：S 比 ID {gains['S']-gains['ID']:+.1f} pp，XL 比 ID {gains['XL']-gains['ID']:+.1f} pp；这不是跨条件差异的显著性检验。")
    lines += ['', '## 复现','', f'Python: `{sys.executable}`；MetaDrive: `{manifest["metadrive_root"]}`。',
        f'可直接运行 `bash {out}/commands.sh`（训练需要可用 CUDA；已完成评估会跳过）。',
        f'GPU 可用后依次执行 `python scripts/rhc_ablation.py train --output {out}`、`evaluate --output {out}`、`report --output {out}`（后两条沿用相同脚本前缀）。已完成评估目录受保护；再次 evaluate 会跳过完整的已有结果。',
        '`commands.sh` 提供每组原始独立命令。`manifest.json` 保存训练种子、源码 SHA、旧文件 SHA、代码 HEAD 和 checkpoint 路径；`config_differences.json` 保存相对原配方的全部配置差异。','']
    (out/'REPORT.zh.md').write_text('\n'.join(lines))
    save(out/'status.json',dict(phase='completed' if len(data)==12 else ('training_in_progress' if 'running' in training_states.values() else 'partial_formal_results'),formal_training=training_states,
         completed_evaluations=[dict(method=n,condition=c,episodes=len(rows)) for (n,c),rows in data.items()],
         protected_artifact_hashes_verified=True))
    save(out/'artifact_versions.json',dict(runner=str(Path(__file__).resolve()),runner_sha256=sha(Path(__file__)),
        evaluator_sha256=sha(out/'evaluate_metadrive.py'),
        tests={p.name:sha(p) for p in [ROOT/'tests/test_rhc_ablation.py',*(ROOT/'tests').glob('test_context*.py')]},
        configs={str(p.relative_to(out)):sha(p) for p in (out/'configs').glob('*.yaml')}))
    verify_protection(out)
    print(out/'REPORT.zh.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','prepare-repeat','audit','sampling','checkpoints','smoke','train','evaluate','evaluate-references','evaluate-smoke','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=ROOT/'logdir/rhc_context_ablation_seed1_20261026')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();out=args.output.resolve()
    if args.phase=='prepare':prepare(out)
    elif args.phase=='prepare-repeat':prepare_repeat(out,args.previous.resolve(),args.seed)
    elif args.phase=='audit':audit(out)
    elif args.phase=='sampling':sampling(out)
    elif args.phase=='checkpoints':checkpoints(out)
    elif args.phase=='smoke':smoke(out)
    elif args.phase=='train':train(out)
    elif args.phase.startswith('evaluate'):evaluate(out,args.phase!='evaluate',args.phase=='evaluate-smoke')
    else:report(out)


if __name__=='__main__':main()
