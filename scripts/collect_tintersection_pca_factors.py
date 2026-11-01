"""Controlled one-factor or full-factorial probes for the frozen T-trained best."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profiles',type=int,default=24)
    p.add_argument('--compound',action='store_true',help='Full 9x9 mass/friction grid')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source.resolve()))
    import torch
    from omegaconf import OmegaConf
    from context import HistoryEncoder
    from envs.metadrive import MetaDrive
    torch.set_num_threads(1)
    cfg=OmegaConf.load(a.checkpoint.parent/'.hydra/config.yaml');cfg.device='cpu'
    OmegaConf.resolve(cfg)
    assert cfg.env.map_block=='T' and cfg.model.context.injection=='dali_shallow'
    cfg.env.traffic_density=0.;cfg.env.spawn.randomize=False
    cfg.env.eval_start_seed=20000;cfg.env.eval_num_scenarios=1;cfg.env.time_limit=200
    cfg.env.termination.out_of_road_done=False;cfg.env.termination.on_continuous_line_done=False
    cfg.env.chassis.eval_models=['m'];cfg.env.chassis.eval_mass_scales=[1.]
    cfg.env.chassis.eval_steering_scales=[1.];cfg.env.chassis.eval_friction_scales=[1.]
    OmegaConf.save(cfg,a.output/'sampling_config.yaml')
    weight_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    encoder=HistoryEncoder(4).eval()
    encoder.load_state_dict({k[len('context_encoder.'):]:v for k,v in payload['agent_state_dict'].items()
                             if k.startswith('context_encoder.')},strict=True)
    step=int(payload['step']);del payload
    rows,histories,physical=[],[],[]
    scales=np.arange(6,15)/10
    env=MetaDrive(cfg.env,evaluation=True)
    env._env.config['map_config']['config']=''
    env._env.config['map_config']['lane_width']=20.
    try:
        for factor in (['joint'] if a.compound else ['mass','friction']):
            for profile in range(a.profiles):
                rng=np.random.default_rng(910000+profile)
                throttle=np.interp(np.arange(5),[0,2,4],rng.uniform(.15,.4,3)).astype(np.float32)
                levels=[(m,f) for m in scales for f in scales] if a.compound else scales
                for level in levels:
                    ms,fs=level if a.compound else (level if factor=='mass' else 1.,level if factor=='friction' else 1.)
                    spec=env._env.config['chassis_spec']
                    spec.update(dict(models=['m'],mass_scales=[float(ms)],
                                     steering_scales=[1.],friction_scales=[float(fs)]))
                    env.reset(seed=20000);car=env._env.vehicle
                    actual=dict(env.chassis_parameters)
                    for _ in range(30):
                        _,_,done,_=env.step(np.zeros(2,np.float32))
                        assert not done,'Settling terminated; do not selectively drop probes'
                    car.set_velocity([1.,0.],value=10.,in_local_frame=True)
                    steer=15./car.max_steering
                    car._set_action(np.array([steer,throttle[0]],np.float32))
                    env._context_history.observe(car)
                    np.testing.assert_allclose(np.linalg.norm(car.velocity),10.,atol=1e-4)
                    np.testing.assert_allclose([car.system.getSteeringValue(i) for i in (0,1)],15.,atol=1e-4)
                    path=[]
                    for t in range(5):
                        obs,_,done,_=env.step(np.array([steer,throttle[t]],np.float32))
                        assert not done and not obs['ctx_collision'],'Invalid probe; do not selectively filter'
                        np.testing.assert_allclose([car.system.getSteeringValue(i) for i in (0,1)],15.,atol=1e-4)
                        path.append(obs['ctx_ego_state'].copy())
                    assert obs['ctx_mask'].all()
                    np.testing.assert_allclose(car.body.getMass(),actual['mass'],atol=1e-4)
                    np.testing.assert_allclose([car.system.getWheel(i).getFrictionSlip() for i in range(4)],
                                               actual['wheel_friction'],rtol=1e-6)
                    np.testing.assert_allclose(actual['mass_scale'],ms,atol=1e-6)
                    np.testing.assert_allclose(actual['friction_scale'],fs,atol=1e-6)
                    histories.append(obs['ctx_history'].copy());physical.append(path)
                    rows.append(dict(factor=factor,scale='' if a.compound else float(level),
                        mass_scale=float(ms),friction_scale=float(fs),profile=profile,vehicle='m',
                        mass_kg=float(car.body.getMass()),wheel_friction=float(car.system.getWheel(0).getFrictionSlip()),
                        initial_speed_mps=10.,wheel_angle_deg=15.,normalized_steer=float(steer),
                        endpoint_speed_mps=float(obs['ctx_ego_state'][2])))
                if (profile+1)%6==0:
                    print(factor,profile+1,'/',a.profiles,'profiles',flush=True)
    finally:
        env.close()
    h=np.array(histories)
    assert h.shape==((81 if a.compound else 18)*a.profiles,5,6)
    with torch.inference_mode():
        c=encoder(torch.from_numpy(h),torch.ones(h.shape[:2],dtype=torch.bool)).numpy()
    assert np.isfinite(c).all()
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==weight_hash
    np.savez_compressed(a.output/'representations.npz',history=h,context=c,physical=np.array(physical))
    with(a.output/'windows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'provenance.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint.resolve()),
        checkpoint_step=step,checkpoint_sha256=weight_hash,source=str(a.source.resolve()),
        profiles=a.profiles,scales=scales.tolist(),windows=len(rows),state_dim=4,history_length=5,
        compound=a.compound,
        controls=('M vehicle only; full factorial9x9 mass/friction grid.' if a.compound else 'M vehicle only; one mass/friction factor changes at a time; other multiplier=1;')+' Same flat road, fixed initial speed10m/s, measured front wheel angle15deg and paired throttle profiles. Thirty settling steps then five real steps. Speed evolves freely during history.',
        limitations=['PCA is descriptive, not proof of parameter identification or world-model utilization.',
                     'Wheel friction is Bullet friction slip, not a measured road adhesion coefficient.',
                     'Controlled diagnostics are not on-policy T-intersection driving episodes.']),indent=2))
    print('COMPLETED',a.output,flush=True)


if __name__=='__main__':main()
