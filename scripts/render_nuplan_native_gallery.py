"""Native ScenarioEnv topdown maps and objects with recorded policy paths.

Figure contract: paired qualitative comparisons, four groups in a 2x2 grid.
Use the same frozen CSV paths, native scenes and pairwise bounds as the prior
gallery; no policy inference or physical replay. Native raster maps are embedded
in PNG/PDF/SVG with editable method labels. Cases are selected illustrations,
including one training-split case, not an aggregate benchmark.
"""
import argparse
import hashlib
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.top_down_obs_impl import ObjectGraphics,WorldSurface,history_object
from metadrive.utils.utils import import_pygame
from metadrive.scenario.utils import read_dataset_summary
from plot_nuplan_trajectory_gallery import load_case,METHODS
from plot_nuplan_trafficgen_style import COLORS,SCENE_NAMES,TITLES,HIDDEN_BACKGROUND_AGENTS
from restyle_nuplan_trajectory_gallery import motion_heading

pygame=import_pygame()


def native_panels(case,context,out,side=1200):
    scene=context['scene'];split=context['split']
    dataset=Path(context['source']).parents
    # Locate the split root even when the source file is in a nested shard.
    folder=next(p for p in dataset if (p/'dataset_summary.pkl').exists())
    count=len(read_dataset_summary(str(folder))[1])
    env=ScenarioEnv(dict(data_directory=str(folder),num_scenarios=count,
        store_data=False,store_map=False,use_render=False,image_observation=False,
        force_destroy=True,reactive_traffic=False,set_static=False,use_bounding_box=False,
        top_down_show_real_size=True,log_level=40))
    try:
        env.reset(seed=scene)
        assert env.engine.data_manager.current_scenario['id']==context['scenario_id']
        np.testing.assert_allclose(env.vehicle.position,case['traces'][0][0,:2],atol=1e-5)
        x0,x1,y0,y1=case['bounds'];span=x1-x0;center=[(x0+x1)/2,(y0+y1)/2]
        bbox=env.current_map.road_network.get_bounding_box()
        scaling=side/span
        film_side=int(np.ceil(max(bbox[1]-bbox[0],bbox[3]-bbox[2])*(scaling+.12)))+100
        film_side=max(film_side,side+100)
        env.render(mode='topdown',window=False,screen_size=(side,side),
            film_size=(film_side,film_side),scaling=scaling,camera_position=center,
            semantic_map=False,draw_center_line=False,num_stack=1,
            draw_target_vehicle_trajectory=False,draw_contour=True)
        renderer=env.top_down_renderer
        assert np.isclose(renderer.scaling,scaling)
        hidden=[context['agents'][i] for i in HIDDEN_BACKGROUND_AGENTS.get(scene,set())]
        removed=[];kept=[]
        for obj in renderer.history_objects[-1]:
            if obj.name==env.vehicle.name:continue
            if any(np.linalg.norm(np.array(obj.position)-a['xy'])<.3 for a in hidden):
                removed.append(dict(name=obj.name,position=np.asarray(obj.position).tolist()))
                continue
            kept.append(obj)
        renderer.history_objects=deque([kept],maxlen=1)
        renderer.refresh();renderer._draw()
        background=renderer._frame_canvas.copy()
        center_pixel=background.pos2pix(*center)
        crop=(center_pixel[0]-side/2,center_pixel[1]-side/2,side,side)
        images=[]
        for index in range(2):
            surface=background.copy()
            ref=case['ref']
            # Full navigation route, sampled from the saved native route.
            distance=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))])
            for k in range(len(ref)-1):
                if distance[k]%4.5<3:
                    pygame.draw.line(surface,(221,161,47),surface.pos2pix(*ref[k]),
                                     surface.pos2pix(*ref[k+1]),max(1,round(.21*scaling)))
            trace=case['traces'][index];xy=trace[:,:2]
            points=[surface.pos2pix(*p) for p in xy]
            pygame.draw.lines(surface,(255,255,255),False,points,max(2,round(.9*scaling)))
            pale=np.array(to_rgb(COLORS[index][0]));dark=np.array(to_rgb(COLORS[index][1]))
            maximum=max(t[-1,2] for t in case['traces'])
            for k in range(len(points)-1):
                t=trace[k+1,2]/maximum;color=tuple(np.rint(255*(pale*(1-t)+dark*t)).astype(int))
                pygame.draw.line(surface,color,points[k],points[k+1],max(2,round(.6*scaling)))
            initial=case['result']['methods'][METHODS[index]]['initial_state']['heading']
            obj=history_object(name=METHODS[index],position=xy[-1],
                heading_theta=motion_heading(trace,len(trace)-1,initial),WIDTH=2.3,LENGTH=5.74,
                color=tuple(np.rint(dark*255).astype(int)),done=False,type='VEHICLE')
            ObjectGraphics.display(obj,surface,obj.color,obj.heading_theta,draw_contour=True,contour_width=2)
            pygame.draw.circle(surface,(255,255,255),points[0],max(2,round(.65*scaling)))
            pygame.draw.circle(surface,obj.color,points[0],max(2,round(.65*scaling)),max(1,round(.16*scaling)))
            screen=pygame.Surface((side,side));screen.fill((255,255,255))
            screen.blit(surface,(0,0),area=crop)
            rgb=WorldSurface.to_cv2_image(screen)
            Image.fromarray(rgb).save(out/f'scene_{scene}_{METHODS[index]}_native.png')
            images.append(rgb)
        return images,dict(scene=scene,split=split,scenario_id=context['scenario_id'],
            world_bounds=case['bounds'],screen_size=[side,side],scaling=scaling,
            semantic_map=False,native_t0_objects=len(kept),display_only_hidden_objects=removed,
            map_renderer='ScenarioEnv.render(mode=topdown) / TopDownRenderer',
            object_renderer='ObjectGraphics.display',source=context['source'])
    finally:env.close()


def compose(panels,target):
    cols=2 if len(panels)>2 else 1;rows=int(np.ceil(len(panels)/cols))
    fig=plt.figure(figsize=(12.4 if cols==2 else 6.2,rows*3.25+.12),facecolor='white')
    grid=fig.add_gridspec(rows,cols,left=.014,right=.986,bottom=.015,top=.94,wspace=.12,hspace=.16)
    for i,pair in enumerate(panels):
        sub=grid[i//cols,i%cols].subgridspec(1,2,wspace=.055)
        for j,img in enumerate(pair):
            ax=fig.add_subplot(sub[0,j]);ax.imshow(img,interpolation='none');ax.set(xticks=[],yticks=[])
            ax.set_title(TITLES[j],fontsize=9.4,pad=6,color='#444444')
            for spine in ax.spines.values():spine.set_color('#D9D9D9');spine.set_linewidth(.55)
    for ext in ('png','pdf','svg'):fig.savefig(target.with_suffix('.'+ext),dpi=450,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    old=json.loads((args.context_root/'provenance.json').read_text())['original_provenance']
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    cases=[load_case(Path(c['source']),110.,None,20.) for c in old['cases']]
    np.testing.assert_allclose([c['bounds'] for c in cases],old['panel_bounds'])
    panels=[];metadata=[]
    for case in cases:
        scene=case['result']['methods'][METHODS[0]]['scene']
        context=json.loads((args.context_root/f'scene_{scene}_render_context.json').read_text())
        imgs,meta=native_panels(case,context,args.output);panels.append(imgs);metadata.append(meta)
        compose([imgs],args.output/(SCENE_NAMES[scene]['slug']+'_native_paired'))
        print('RENDERED',scene,meta['native_t0_objects'],flush=True)
    compose(panels,args.output/'trajectory_gallery_native_scenarioenv')
    hashes={str(c['folder']/name):hashlib.sha256((c['folder']/name).read_bytes()).hexdigest()
        for c in cases for name in ('r2dreamer_trajectory.csv','r2dreamer_context_trajectory.csv','reference_navigation.csv')}
    result=dict(original_provenance=old,panels=metadata,sources_sha256=hashes,
        renderer_source=str(Path(__import__('metadrive.engine.top_down_renderer',fromlist=['a']).__file__)),
        note='Native ScenarioEnv t0 map and traffic render with complete recorded trajectories overlaid. No policy rerun. The requested background vehicle is omitted from both scene31 views for display only. Third case is train, selected qualitative examples do not estimate benchmark performance.',
        exports='Native map/object layers are raster; PDF/SVG retain editable labels, not vector map geometry.')
    (args.output/'provenance.json').write_text(json.dumps(result,indent=2))
    print('COMPLETED',args.output,flush=True)


if __name__=='__main__':main()
