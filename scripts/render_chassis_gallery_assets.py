#!/usr/bin/env python3
"""Render native MetaDrive chassis and assemble a train/OOD figure."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures/chassis_gallery_verified_20261023'
MODELS=['default','m','l','s','xl']

def render(model):
    sys.path.insert(0,str(ROOT.parent/'metadrive'))
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from panda3d.core import PNMImage
    from metadrive.component.vehicle.vehicle_type import vehicle_type
    cls=vehicle_type[model]
    env=MetaDriveEnv(dict(map='T',use_render=False,image_observation=True,
        sensors={'main_camera':('MainCamera',1200,750)},window_size=(1200,750),
        show_interface=False,show_fps=False,traffic_density=0,num_scenarios=1,
        vehicle_config=dict(image_source='main_camera',vehicle_model=model,show_navi_mark=False,show_dest_mark=False)))
    try:
        env.reset(seed=0)
        import numpy as np
        history=[]
        stable=False
        for step in range(400):
            env.step([0,-1])
            position=np.array(env.agent.origin.getPos(env.engine.render),dtype=float)
            history.append(position)
            if step>=99 and np.max(np.ptp(np.array(history[-20:]),axis=0))<0.002:
                stable=True
                break
        assert stable, f'{model}: suspension did not settle: {np.ptp(np.array(history[-20:]),axis=0)}'
        wheel_checks=[]
        for index in range(env.agent.system.getNumWheels()):
            wheel=env.agent.system.getWheel(index)
            ray=wheel.getRaycastInfo()
            wheel_checks.append(dict(index=index,in_contact=bool(ray.isInContact()),radius=float(wheel.getWheelRadius())))
        assert len(wheel_checks)==4
        assert all(w['in_contact'] for w in wheel_checks), (model,wheel_checks)
        wheel_nodes=env.agent.origin.findAllMatches('**/wheel')
        assert len(wheel_nodes)==4
        assert all(not node.getTightBounds() is None for node in wheel_nodes)

        cam=env.engine.main_camera;cam.stop_track();env.engine.taskMgr.remove(cam.TOP_DOWN_TASK_NAME)
        pos=env.agent.origin.getPos(env.engine.render)
        env.engine.camera.setPos(pos.x+6.5,pos.y-7,pos.z+2.8)
        env.engine.camera.lookAt(pos.x,pos.y,pos.z+.35)
        for _ in range(8):env.engine.graphicsEngine.renderFrame()
        im=PNMImage();env.engine.win.getScreenshot(im);im.write(str(OUT/'raw'/f'{model}.png'))
        friction=cls.PARAMETER_SPACE['wheel_friction']
        meta=dict(model=model,mass_kg=cls.MASS,length_m=cls.DEFAULT_LENGTH,width_m=cls.DEFAULT_WIDTH,
            nominal_wheel_friction=float((friction.low[0]+friction.high[0])/2),asset=str(cls.path) if model!='s' else 'beetle/vehicle.gltf',
            body_mass_kg=float(env.agent.body.getMass()),map_seed=0,render='offscreen native MetaDrive 0.4.3',snapshot='nominal chassis, stationary with brake applied',wheel_render_two_sided=cls.TIRE_TWO_SIDED,physics_wheels=env.agent.system.getNumWheels(),settling_steps=step+1,stability_position_range=np.ptp(np.array(history[-20:]),axis=0).tolist(),wheel_checks=wheel_checks,body_hpr=list(env.agent.origin.getHpr(env.engine.render)),rendered_wheel_nodes=len(wheel_nodes))
        (OUT/'raw'/f'{model}.json').write_text(json.dumps(meta,indent=2))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-model',choices=MODELS,required=True);a=p.parse_args()
    (OUT/'raw').mkdir(parents=True,exist_ok=True)
    render(a.render_model)
