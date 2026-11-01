#!/usr/bin/env python3
"""Render native MetaDrive chassis and assemble a train/OOD figure."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures/chassis_randomization_20261023'
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
        for _ in range(8):env.step([0,0])
        cam=env.engine.main_camera;cam.stop_track();env.engine.taskMgr.remove(cam.TOP_DOWN_TASK_NAME)
        pos=env.agent.origin.getPos(env.engine.render)
        env.engine.camera.setPos(pos.x+6.5,pos.y-7,pos.z+3.7)
        env.engine.camera.lookAt(pos.x,pos.y,pos.z+.35)
        for _ in range(8):env.engine.graphicsEngine.renderFrame()
        im=PNMImage();env.engine.win.getScreenshot(im);im.write(str(OUT/'raw'/f'{model}.png'))
        friction=cls.PARAMETER_SPACE['wheel_friction']
        meta=dict(model=model,mass_kg=cls.MASS,length_m=cls.DEFAULT_LENGTH,width_m=cls.DEFAULT_WIDTH,
            nominal_wheel_friction=float((friction.low[0]+friction.high[0])/2),asset=str(cls.path) if model!='s' else 'beetle/vehicle.gltf',
            body_mass_kg=float(env.agent.body.getMass()),map_seed=0,render='offscreen native MetaDrive 0.4.3',snapshot='nominal chassis, zero control')
        (OUT/'raw'/f'{model}.json').write_text(json.dumps(meta,indent=2))
    finally:env.close()

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'logdir/tintersection_mass_friction_500k_20261021/training/r2dreamer/.hydra/config.yaml')
    spec=OmegaConf.to_container(cfg.env.chassis,resolve=True)
    assert spec['models']==MODELS[:3] and spec['steering_scales']==[1.]
    assert spec['mass_scales']==spec['friction_scales']==[.6,.7,.8,.9,1.,1.1,1.2,1.3,1.4]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(183/25.4,112/25.4),facecolor='white')
    for x,w,color in [(.012,.646,'#E3E3E3'),(.673,.315,'#CCD9E4')]:
        fig.add_artist(FancyBboxPatch((x,.075),w,.87,boxstyle='round,pad=0.006,rounding_size=0.025',facecolor=color,edgecolor='none',transform=fig.transFigure,zorder=0))
    fig.text(.03,.907,'(a) Training chassis',fontsize=10,weight='bold')
    fig.text(.69,.907,'(b) Unseen chassis',fontsize=10,weight='bold')
    slots={'default':(.025,.55),'m':(.348,.55),'l':(.025,.16),'s':(.685,.55),'xl':(.685,.16)}
    metadata=[]
    for model,(x,y) in slots.items():
        ax=fig.add_axes([x,y,.29,.31]);ax.imshow(plt.imread(OUT/'raw'/f'{model}.png'));ax.axis('off')
        meta=json.loads((OUT/'raw'/f'{model}.json').read_text());metadata.append(meta)
        title='Default' if model=='default' else model.upper()
        fig.text(x+.145,y-.018,title,ha='center',fontsize=11)
        fig.text(x+.145,y-.052,f"Nominal mass: {meta['mass_kg']} kg",ha='center',fontsize=7.5,color='#333333')
    x=.365
    fig.text(x,.424,'Episode-wise randomization',fontsize=9,weight='bold')
    lines=[r'$m = s_m\,m_0$',r'$f_{\mathrm{tire}} = s_f\,f_{\mathrm{tire},0}$',
           r'$s_m,s_f\ \in\ \{0.6,0.7,\ldots,1.4\}$',
           'Independent uniform sampling','3 types × 9 masses × 9 frictions',
           '= 243 training combinations','Steering fixed at nominal value']
    for i,line in enumerate(lines):fig.text(x,.38-i*.038,line,fontsize=8 if i<3 else 7.3)
    fig.text(.5,.027,'OOD: unseen vehicle types (S / XL), with the same mass and tire-friction scale ranges.',ha='center',fontsize=7.3)
    for ext in ['png','pdf','svg','tiff']:
        opts={'dpi':600} if ext in ['png','tiff'] else {}
        if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(OUT/f'chassis_randomization.{ext}',**opts)
    fig.savefig(OUT/'chassis_randomization_preview.png',dpi=220)
    (OUT/'source_data.json').write_text(json.dumps(dict(spec=spec,vehicles=metadata),indent=2))
    (OUT/'CAPTION.md').write_text('Chassis randomization in MetaDrive. (a) Default, M and L are sampled uniformly during training. Mass and tire friction are independently sampled from nine multipliers (0.6–1.4 in increments of 0.1), giving 243 combinations. Parameters remain fixed within each episode; maximum steering stays at its nominal value. (b) S and XL are held out from training and used for unseen-type evaluation, with the same multiplier ranges. Images show actual native simulator assets at nominal parameters on a common map and camera setup; the illustrative render contains no traffic. Tire friction denotes the simulator wheel friction parameter, not a measured road friction coefficient.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render-model',choices=MODELS);a=p.parse_args()
    (OUT/'raw').mkdir(parents=True,exist_ok=True)
    if a.render_model:render(a.render_model)
    else:
        (OUT/'figure_contract.json').write_text(json.dumps(dict(conclusion='Distinguish training and unseen chassis, and document physical randomization.',archetype='image plate plus parameter specification',backend='Python',statistics='Protocol illustration, not outcome evidence',exports=['PNG','PDF','SVG','TIFF'],size_mm=[183,112]),indent=2))
        for model in MODELS:subprocess.run([sys.executable,__file__,'--render-model',model],check=True)
        plot()
