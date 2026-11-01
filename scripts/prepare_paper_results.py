#!/usr/bin/env python3
"""Export existing experimental evidence without rerunning or changing training."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT/'logdir'
METRICS = ('success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps')


def read(path):
    return json.loads(Path(path).read_text())


def csvfile(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def jsonfile(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def training_loss_exports(out):
    rows=[]
    keys=('dyn','rep','barlow','rew','con','context_aux','context_swap')
    for path in sorted((out/'provenance/metrics_snapshots').glob('*.jsonl')):
        for row in records(path):
            if 'train/loss/dyn' not in row:continue
            result=dict(run_id=path.stem,step=row['step'])
            result.update({k:row['train/loss/'+k]for k in keys if 'train/loss/'+k in row})
            rows.append(result)
    csvfile(out/'04_forecast/training_losses.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Refresh a partially written export directory')
    args=parser.parse_args(); out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    sources={}; registry=[]; curves=[]; summaries=[]; episodes=[]; pending=[]; paired=[]
    captured=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()

    def track(path):
        path=Path(path)
        if path.is_file():
            raw=path.read_bytes()
            sources[str(path.resolve())]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        return path

    def copy(source, target):
        source=track(source); target=out/target
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

    entries=[];eval_overrides={}
    def add(id, scene, method, model, evalroot=None, cohort='mass_friction_243', role='main'):
        entries.append(dict(run_id=id,scene=scene,method=method,model_dir=str(model),
            eval_dir=str(evalroot or model/'evaluation'),cohort=cohort,role=role))

    t=LOG/'tintersection_mass_friction_500k_20261021/training'
    q=LOG/'tintersection_baselines_queue_20261022'
    r=LOG/'roundabout_mass_friction_queue_20261023'
    add('T_R2','t_intersection','R2Dreamer',t/'r2dreamer')
    add('T_R2_context','t_intersection','R2Dreamer+context',t/'r2dreamer_dali_shallow')
    for name in ('ppo','sac'):
        add('T_'+name.upper(),'t_intersection',name.upper(),q/name)
    add('T_DV3','t_intersection','DreamerV3',q/'dreamerv3/dreamerv3')
    add('T_DV3_context','t_intersection','DreamerV3+context',q/'dreamerv3_dali_shallow/dreamerv3_dali_shallow')
    add('T_R2_wmgrad05','t_intersection','R2Dreamer+context WMgrad0.5',LOG/'tintersection_shallow_wmgrad05_500k_20261022/training/r2dreamer_dali_shallow',role='ablation')
    for name,method in [('r2dreamer','R2Dreamer'),('r2dreamer_context','R2Dreamer+context'),('sac','SAC'),('ppo','PPO'),('dreamerv3','DreamerV3'),('dreamerv3_context','DreamerV3+context')]:
        add('O_'+name,'roundabout',method,r/('roundabout_'+name))
    legacy=LOG/'world_model_injection_comparison_20261021'
    for name,method in [('baseline','R2Dreamer'),('03_recurrent','R2+recurrent'),('09_deep','R2+Deep+swap'),('10_joint','R2+Joint+swap'),('11_shallow','R2+Shallow+swap')]:
        prov=read(track(legacy/name/'provenance.json')); model=Path(prov['checkpoint']).parent
        ev=model/'evaluation'
        if name=='baseline':ev=model.parent/'best_eval100_20261015_095447/r2dreamer'
        if name=='03_recurrent':ev=model.parent/'comparison_20261016_011017'
        add('legacy_'+name,'roundabout',method,model,ev,cohort='legacy_mass_steering',role='historical_mechanism_comparison')
    old=read(track(LOG/'context_kinematics_500k_20261015_180340/comparison_20261016_011017/evaluation_comparison.json'))
    for split in ('new_maps','s','xl'):
        eval_overrides['legacy_baseline',split]=Path(old[split]['data_paths']['baseline'])

    byrun={}
    for e in entries:
        model=Path(e['model_dir']);cfgpath=model/'.hydra/config.yaml'
        if not cfgpath.exists():cfgpath=model/'config.yaml'
        # Queued jobs already have prepared configurations.
        if not cfgpath.exists() and e['run_id'].startswith('O_'):
            cfgpath=r/'configs'/('roundabout_'+e['run_id'][2:]+'.yaml')
        cfg=OmegaConf.to_container(OmegaConf.load(track(cfgpath)),resolve=True)
        chassis=cfg['env']['chassis']; models=chassis['models']
        e.update(training_seed=cfg['env']['seed'],budget=cfg['env']['steps'],map_block=cfg['env'].get('map_block'),
            model_pool=json.dumps(models),mass_scales=json.dumps(chassis['mass_scales']),
            steering_scales=json.dumps(chassis['steering_scales']),friction_scales=json.dumps(chassis.get('friction_scales',[1.])),
            combinations=len(models)*len(chassis['mass_scales'])*len(chassis['steering_scales'])*len(chassis.get('friction_scales',[1.])),
            config_path=str(cfgpath.resolve()))
        context=cfg['model'].get('context',{})
        e.update(context_enabled=context.get('enabled',False),context_injection=context.get('injection'),
            swap_weight=context.get('swap_weight'),wm_grad_scale=context.get('wm_grad_scale',0))
        if e['cohort']=='mass_friction_243':
            assert e['combinations']==243 and chassis['steering_scales']==[1.0]
        log=model/'metrics.jsonl';logs=[]
        if log.exists():
            # Capture once: live runs may append during export.
            raw=log.read_bytes()
            sources[str(log.resolve())]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            snapshot=out/'provenance/metrics_snapshots'/(e['run_id']+'.jsonl')
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(raw)
            logs=[json.loads(line) for line in raw.decode().splitlines() if line]
        last=max((x.get('step',0) for x in logs),default=0)
        e['observed_step']=last;e['training_state']='completed' if last>=e['budget'] else 'running_or_partial' if logs else 'queued'
        best=model/('best.zip' if e['method'] in ('PPO','SAC') else 'best.pt')
        best_step=None
        if (model/'best.json').exists():best_step=read(track(model/'best.json'))['step']
        if best.exists():e['checkpoint_sha256']=hashlib.sha256(best.read_bytes()).hexdigest()
        for index,row in enumerate(logs):
            for key in ('episode/score','episode/eval_score'):
                if key in row:
                    curves.append(dict(run_id=e['run_id'],scene=e['scene'],method=e['method'],cohort=e['cohort'],role=e['role'],training_seed=e['training_seed'],
                        series='training_return' if key=='episode/score' else 'validation_return',
                        step=row['step'],record_index=index,reward=row[key],length=row.get('episode/length' if key=='episode/score' else 'episode/eval_length'),
                        success=row.get('episode/success' if key=='episode/score' else 'episode/eval_success'),
                        source=str(log.resolve())))
        for split in ('new_maps','s','xl'):
            folder=eval_overrides.get((e['run_id'],split),Path(e['eval_dir'])/split)
            if not (folder/'summary.json').exists():
                pending.append(dict(run_id=e['run_id'],scene=e['scene'],method=e['method'],split=split,state='missing_or_queued',training_state=e['training_state']))
                continue
            summary=read(track(folder/'summary.json')); rows=records(track(folder/'episodes.jsonl'))
            assert len(rows)==100
            observed={k:sum(float(row[k]) for row in rows)/len(rows) for k in METRICS}
            for k,v in observed.items():assert abs(v-summary['summary'][k])<1e-6,(folder,k)
            if 'checkpoint_sha256' in summary:assert summary['checkpoint_sha256']==e['checkpoint_sha256']
            if summary.get('checkpoint_step') is not None:best_step=summary['checkpoint_step']
            base={k:e[k]for k in ('run_id','scene','method','cohort','role','training_seed')}
            summaries.append(dict(**base,split=split,episodes=100,inference_repeats=1,unique_scenarios=100,best_step=best_step,**observed,source=str((folder/'summary.json').resolve())))
            for row in rows:
                flat=dict(**base,split=split,seed=row['seed'],**{k:row[k]for k in METRICS})
                flat.update({'chassis_'+k:v for k,v in row['chassis'].items()});episodes.append(flat)
            byrun[e['run_id'],split]=rows
        e['best_step']=best_step;registry.append(e)

    for scene,baseline in [('t_intersection','T_R2'),('roundabout','O_r2dreamer'),('roundabout_legacy','legacy_baseline')]:
        for split in ('new_maps','s','xl'):
            br=byrun.get((baseline,split))
            if br is None:continue
            for e in registry:
                if e['run_id']==baseline or (e['run_id'],split) not in byrun:continue
                if e['cohort'] != next(v['cohort']for v in registry if v['run_id']==baseline):continue
                if e['scene'] != ('roundabout' if scene.startswith('roundabout') else scene):continue
                rr=byrun[e['run_id'],split]
                for a,b in zip(br,rr):
                    assert a['seed']==b['seed']
                    for key in ('model','mass','max_steering','wheel_friction'):
                        if key not in a['chassis'] and key not in b['chassis']:continue
                        av,bv=a['chassis'][key],b['chassis'][key]
                        assert av==bv if isinstance(av,str) else np.isclose(av,bv), (e['run_id'],split,key)
                delta=np.array([float(b['success'])-float(a['success'])for a,b in zip(br,rr)])
                rng=np.random.default_rng(20261023);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                paired.append(dict(scene=scene,cohort=e['cohort'],baseline=baseline,method=e['run_id'],split=split,n_pairs=100,
                    success_delta=delta.mean(),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975),pairing_verified=True))

    csvfile(out/'experiments.csv',registry)
    csvfile(out/'01_training/raw_return_curves.csv',curves)
    bins={}
    for row in curves:
        if row['series']!='training_return':continue
        bin_id=max(0,math.ceil(row['step']/10000)-1)
        bins.setdefault((row['run_id'],bin_id),[]).append(row)
    binned=[]
    for (id,bin_id),rows in bins.items():
        values=np.array([v['reward']for v in rows])
        binned.append(dict(run_id=id,scene=rows[0]['scene'],method=rows[0]['method'],cohort=rows[0]['cohort'],
            bin_start=bin_id*10000,bin_end=(bin_id+1)*10000,mean_observed_step=np.mean([v['step']for v in rows]),
            mean_reward=values.mean(),within_run_record_sd=values.std(ddof=1)if len(values)>1 else '',n_records=len(values)))
    csvfile(out/'01_training/return_bins_10k.csv',binned)
    csvfile(out/'03_driving/main_summary.csv',[v for v in summaries if v['cohort']=='mass_friction_243'])
    csvfile(out/'03_driving/legacy_ablation_summary.csv',[v for v in summaries if v['cohort']!='mass_friction_243'])
    csvfile(out/'03_driving/per_episode.csv',episodes)
    csvfile(out/'03_driving/paired_success_deltas.csv',paired)
    csvfile(out/'03_driving/pending.csv',pending)
    training_loss_exports(out)

    # PCA: copy actual coordinates and representations; never refit or cherry-pick points.
    pca=LOG/'tintersection_mass_friction_500k_20261021'
    for source_name,target in [('context_pca_best120k','natural'),('context_pca_fixed10mps15deg_validated','fixed'),('context_pca_fixed_random_vs_trained','random_vs_trained')]:
        source=pca/source_name
        for f in source.iterdir():
            if f.suffix in ('.csv','.npz','.json','.md','.svg','.pdf'):
                copy(f,Path('02_context_pca')/target/f.name)

    # Full 1..10-step errors already saved by the common-trajectory evaluation.
    forecast=[]; forecast_pairs=[]
    setups=[('t_intersection_mass_friction',pca/'world_model_comparison', ['baseline','11_shallow','12_shallow_wmgrad05']),
            ('roundabout_legacy_mass_steering',legacy,['baseline','03_recurrent','09_deep','10_joint','11_shallow'])]
    for cohort,root,methods in setups:
        arrays={};origins=None;origin_digest=None
        for method in methods:
            folder=(LOG/'tintersection_shallow_wmgrad05_500k_20261022/world_model_comparison'/method
                    if method=='12_shallow_wmgrad05' else root/method)
            prov=read(track(folder/'provenance.json'))
            if origin_digest is None:origin_digest=prov['origins_sha256']
            assert origin_digest==prov['origins_sha256']
            origins=read(track(Path(prov['corpus'])/'origins.json'))
            errors=np.load(track(folder/'errors.npz'));arrays[method]={k:errors[k]for k in errors.files}
            copy(folder/'errors.npz',Path('04_forecast/raw')/cohort/method/'errors.npz')
            copy(folder/'provenance.json',Path('04_forecast/raw')/cohort/method/'provenance.json')
            for group,select in [('all',np.ones(len(origins),bool)),('seen',np.array([v['model']in ('default','m','l')for v in origins])),('unseen',np.array([v['model']in ('s','xl')for v in origins]))]:
                for key in errors.files:
                    if key in ('valid','physical_valid'):continue
                    mask=errors['physical_valid']if key=='auxiliary_mae'else errors['valid']
                    values=errors[key]if key=='auxiliary_mae'else errors[key].mean(0)
                    for h in range(values.shape[1]):
                        valid=select&mask[:,h]
                        v=values[valid,h]
                        dims=enumerate(('acceleration_mps2','yaw_rate_radps','speed_mps','wheel_rad'))if key=='auxiliary_mae'else [(None,'scalar')]
                        for dim,label in dims:
                            mean=float(v.mean(axis=0)[dim])if dim is not None else float(v.mean())
                            forecast.append(dict(cohort=cohort,method=method,checkpoint_step=prov['step'],device=prov['device'],group=group,
                                head='context_auxiliary'if key=='auxiliary_mae'else 'RSSM',metric=key,channel=label,
                                horizon=h+1,time_seconds=(h+1)/10,n_origins=int(valid.sum()),mean=mean))
            assert prov['origins_sha256']==hashlib.sha256((Path(prov['corpus'])/'origins.json').read_bytes()).hexdigest()
        # Inference draws are averaged per origin; maps are the bootstrap clusters.
        base=arrays['baseline'];ctx=arrays['11_shallow']
        assert np.array_equal(base['valid'],ctx['valid'])
        for key in ('reward_mae','reward_nll','continue_brier','continue_nll'):
            diff=ctx[key].mean(0)-base[key].mean(0)
            for h in (0,4,9):
                mask=base['valid'][:,h]; groups=np.array([v['map_id']for v in origins])
                ids=np.unique(groups);sums=np.array([diff[(groups==g)&mask,h].sum()for g in ids]);counts=np.array([((groups==g)&mask).sum()for g in ids])
                rng=np.random.default_rng(20261023);idx=rng.integers(0,len(ids),(10000,len(ids)))
                boot=sums[idx].sum(1)/counts[idx].sum(1)
                forecast_pairs.append(dict(cohort=cohort,method='11_shallow_minus_baseline',metric=key,horizon=h+1,
                    delta=float(diff[mask,h].mean()),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975),n_map_clusters=len(ids)))
        copy(Path(prov['corpus'])/'origins.json',Path('04_forecast/raw')/cohort/'origins.json')
    csvfile(out/'04_forecast/metrics_by_horizon.csv',forecast)
    csvfile(out/'04_forecast/paired_map_bootstrap.csv',forecast_pairs)

    # Existing old-protocol traces are retained as an explicitly separate cohort.
    stability=LOG/'driving_stability_20261021'
    for f in ('episode_metrics.csv','results.json','REPORT.md','reproducibility_check.json'):
        copy(stability/f,Path('05_trajectories_legacy')/f)
    frame_rows=[]
    for method in ('baseline','09_deep','11_shallow'):
        source=stability/method
        for f in ('episodes.jsonl','provenance.json'):copy(source/f,Path('05_trajectories_legacy')/method/f)
        for row in records(track(source/'episodes.jsonl')):
            f=source/row['trace'];d=np.load(track(f));n=len(d['time'])
            for i in range(n):
                action=d['action'][i]if i<len(d['action'])else (float('nan'),float('nan'))
                frame_rows.append(dict(method=method,split=row['split'],seed=row['seed'],frame=i,time_s=d['time'][i],
                    x_m=d['position'][i,0],y_m=d['position'][i,1],speed_mps=d['speed_mps'][i],
                    wheel_rad_right=d['wheel_rad'][i],yaw_rate_radps_right=d['yaw_rate_right'][i],
                    accel_long_mps2=d['accel_long'][i],jerk_mps3=(d['accel_long'][i]-d['accel_long'][i-1])/(d['time'][i]-d['time'][i-1])if i else float('nan'),
                    steering_action_left=action[0],throttle_action=action[1],collision=bool(d['collision'][i]),is_last=bool(d['is_last'][i])))
            copy(f,Path('05_trajectories_legacy')/method/row['trace'])
        for f in (source/'maps').glob('*.json'):copy(f,Path('05_trajectories_legacy')/method/'maps'/f.name)
    csvfile(out/'05_trajectories_legacy/frame_data.csv',frame_rows)
    # Include nuPlan as additional validation, preserving paired inference repeats.
    nuplan=LOG/'nuplan_id_ood_comparison_20261023'
    for f in ('comparison.json','REPORT.md','world_model_training_loss_comparison.json'):
        if (nuplan/f).exists():copy(nuplan/f,Path('06_nuplan')/f)
    nurows=[]
    for split,item in read(track(nuplan/'comparison.json')).items():
        for method,v in item['methods'].items():
            nurows.append(dict(split=split,method=method,rollouts=300,unique_scenarios=100,inference_repeats=3,training_seeds=1,**v['mean']))
    csvfile(out/'06_nuplan/summary.csv',nurows)
    jsonfile(out/'source_manifest.json',dict(captured_at=captured,sources=sources))
    jsonfile(out/'export_counts.json',dict(experiments=len(registry),return_records=len(curves),driving_summary_rows=len(summaries),
        driving_episode_rows=len(episodes),forecast_rows=len(forecast),trajectory_frames=len(frame_rows),pending_rows=len(pending)))
    print(json.dumps(read(out/'export_counts.json')))


if __name__=='__main__':main()
