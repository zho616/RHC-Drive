#!/usr/bin/env python3
"""Frozen-checkpoint, paired-seed MetaDrive evaluation with per-episode results."""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import time


def summarize(rows):
    n=len(rows)
    if not n:
        return {}
    keys=('success','crash','out_of_road','timeout','reward','route_completion','steps','mean_speed_km_h')
    keys += tuple(key for key in ('out_of_lane', 'out_of_lane_step_fraction') if all(key in row for row in rows))
    result={key:sum(float(row[key]) for row in rows)/n for key in keys}
    successes=sum(row['success'] for row in rows)
    p=successes/n; z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    result.update(episodes=n,success_count=successes,success_wilson95=[center-half,center+half])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=pathlib.Path)
    parser.add_argument('--source-root',required=True,type=pathlib.Path)
    parser.add_argument('--output',required=True,type=pathlib.Path)
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=10020)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--environment-seed',type=int,default=None,
                        help='Keep independent-test chassis/map sampling fixed across training seeds')
    parser.add_argument('--record-lane-metrics',action='store_true',
                        help='Record any off-lane event and the fraction of off-lane control steps, separately from out-of-road')
    parser.add_argument('--scenario-dir',type=pathlib.Path,help='Held-out nuPlan ScenarioNet split')
    parser.add_argument('--inference-seed-offset',type=int,default=12345)
    parser.add_argument('--eval-models',nargs='+',choices=('s','default','m','l','xl'),
                        help='Override evaluation chassis pool without changing training config')
    args=parser.parse_args()
    if args.episodes<1: parser.error('episodes must be positive')
    checkpoint=args.checkpoint.resolve(); source=args.source_root.resolve(); output=args.output.resolve()
    if output.exists(): parser.error('Use a fresh output directory')
    if not checkpoint.is_file() or not (source/'dreamer.py').is_file():
        parser.error('Missing checkpoint or frozen source')
    output.mkdir(parents=True)
    # Import the exact training implementation, including environment adapters.
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from dreamer import Dreamer
    from envs.metadrive import MetaDrive

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config=OmegaConf.load(checkpoint.parent/'.hydra/config.yaml')
    config.device=args.device
    OmegaConf.resolve(config)
    if args.environment_seed is not None:
        config.env.seed=args.environment_seed
    config.model.compile=False
    config.env.eval_start_seed=args.seed_start-int(config.env.seed)
    config.env.eval_num_scenarios=args.episodes
    config.env.eval_episode_num=args.episodes
    config.env.eval_env_num=1
    nuplan=config.env.task=='metadrive_nuplan'
    if args.scenario_dir:
        if not nuplan:parser.error('--scenario-dir requires a nuPlan checkpoint')
        config.env.scenario_eval_dir=str(args.scenario_dir.resolve())
    if nuplan:
        if args.seed_start!=0:parser.error('Pass --seed-start 0 for nuPlan split indices')
        from envs.nuplan import NuPlanDrive
        MetaDrive=NuPlanDrive
    if args.eval_models:
        if not config.env.chassis.enabled:
            parser.error('--eval-models requires a checkpoint trained with chassis randomization')
        if len(set(args.eval_models)) != len(args.eval_models):
            parser.error('Do not repeat evaluation models')
        config.env.chassis.eval_models=list(args.eval_models)
    OmegaConf.save(config,output/'config.yaml')
    payload=torch.load(checkpoint,map_location=args.device,weights_only=False)
    fingerprint=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance=dict(checkpoint=str(checkpoint),checkpoint_sha256=fingerprint,
        checkpoint_step=payload.get('step'),source_root=str(source),episodes_requested=args.episodes,
        seed_start=args.seed_start,seed_end=args.seed_start+args.episodes-1,
        inference_seed_offset=args.inference_seed_offset,context_enabled=bool(config.model.context.enabled),
        context_injection=str(config.model.context.injection),
        context_actor_critic=bool(getattr(config.model.context, 'actor_critic', False)),
        scenario_dir=str(config.env.scenario_eval_dir)if nuplan else None,
        evaluation=('held-out nuPlan scenarios; overridden chassis pool; training scale distribution; frozen weights'
                    if nuplan and args.eval_models else
                    'held-out nuPlan scenarios; training chassis/scale distribution; frozen weights'if nuplan else
                    'new maps; overridden chassis pool; training scale distribution; frozen weights'
                    if args.eval_models else 'new maps; training chassis/scale distribution; frozen weights'),
        eval_models=list(config.env.chassis.eval_models) if config.env.chassis.enabled else None,
        eval_mass_scales=list(config.env.chassis.eval_mass_scales) if config.env.chassis.enabled else None,
        eval_steering_scales=list(config.env.chassis.eval_steering_scales) if config.env.chassis.enabled else None,
        eval_friction_scales=list(getattr(config.env.chassis, 'eval_friction_scales', [1.0])) if config.env.chassis.enabled else None,
        policy='actor mode; stochastic RSSM posterior with fixed per-episode RNG seed')
    provenance.update(training_seed=int(config.seed), environment_seed=int(config.env.seed),
        out_of_lane_definition='Any control step with native info.off_lane=true; also report step fraction' if args.record_lane_metrics else None)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[]; started=time.monotonic(); env=None

    def save_progress(status,error=None):
        result=dict(provenance,status=status,completed_episodes=len(rows),
                    elapsed_seconds=time.monotonic()-started,summary=summarize(rows),
                    per_model={name:summarize([r for r in rows if r['chassis']['model']==name])
                               for name in sorted({r['chassis']['model'] for r in rows})})
        if error: result['error']=error
        temporary=output/'summary.json.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(output/'summary.json')

    try:
        env=MetaDrive(config.env,evaluation=True)
        model=Dreamer(config.model,env.observation_space,env.action_space).to(args.device)
        model.load_state_dict(payload['agent_state_dict'],strict=True)
        model.clone_and_freeze(); model.eval()
        del payload
        print(json.dumps(provenance),flush=True)
        save_progress('running')
        with (output/'episodes.jsonl').open('w',buffering=1) as trace, torch.inference_mode():
            for seed in range(args.seed_start,args.seed_start+args.episodes):
                torch.manual_seed(seed+args.inference_seed_offset); np.random.seed(seed+args.inference_seed_offset)
                obs=env.reset(seed=seed)
                chassis=env.chassis_parameters or dict(model=env._env.vehicle.config["vehicle_model"])
                state=model.get_initial_state(1)
                total=0.; speeds=[]; lane_flags=[]; done=False; info={}
                for step in range(1,int(config.env.time_limit)+1):
                    tensors={}
                    for key,value in obs.items():
                        value=torch.as_tensor(np.asarray(value),device=args.device)
                        tensors[key]=value.reshape(1,1) if value.ndim==0 else value.unsqueeze(0)
                    batch=TensorDict(tensors,batch_size=(1,),device=args.device)
                    action,state=model.act(batch,state,eval=True)
                    obs,reward,done,info=env.step(action[0].cpu().numpy())
                    total+=float(reward); speeds.append(float(env._env.vehicle.speed_km_h))
                    if args.record_lane_metrics:
                        if 'off_lane' not in info:
                            raise RuntimeError('Native off_lane flag unavailable; do not substitute out_of_road')
                        lane_flags.append(bool(info['off_lane']))
                    if done: break
                if not done: raise RuntimeError('Environment did not terminate within its configured horizon')
                row=dict(episode=len(rows)+1,seed=seed,chassis=chassis,steps=step,reward=total,
                    mean_speed_km_h=float(np.mean(speeds)),success=bool(info.get('arrive_dest',False)),
                    crash=bool(info.get('crash',False)),out_of_road=bool(info.get('out_of_road',False)),
                    timeout=bool(obs['is_last'] and not obs['is_terminal']),
                    route_completion=float(info.get('route_completion',0.)))
                if args.record_lane_metrics:
                    row.update(out_of_lane=any(lane_flags), out_of_lane_step_fraction=float(np.mean(lane_flags)))
                rows.append(row)
                trace.write(json.dumps(row)+'\n')
                save_progress('running')
                print(f"{len(rows)}/{args.episodes} seed={seed} model={chassis['model']} "
                      f"success={int(row['success'])} score={total:.2f} running_success={summarize(rows)['success']:.3f}",flush=True)
        save_progress('completed')
        print('SUMMARY '+json.dumps(summarize(rows)),flush=True)
    except BaseException as error:
        save_progress('failed',repr(error))
        raise
    finally:
        if env is not None: env.close()


if __name__=='__main__': main()
