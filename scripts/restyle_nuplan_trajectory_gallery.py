"""Rich vector BEV from original scenario geometry and unchanged policy traces."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from shapely.geometry import Polygon, box

from plot_nuplan_trajectory_gallery import load_case, METHODS

TITLES=['R2Dreamer','R2Dreamer + Context (Ours)']
COLORS=[('#73AED1','#17558E'),('#ECA198','#CC3347')]


def extract_context(case,dataset_root):
    result=case['result'];scene=result['methods'][METHODS[0]]['scene']
    split=Path(result.get('scenario_directory','test')).name
    folder=dataset_root/split
    with(folder/'dataset_summary.pkl').open('rb') as f:summary=pickle.load(f)
    with(folder/'dataset_mapping.pkl').open('rb') as f:mapping=pickle.load(f)
    name=list(summary)[scene];path=folder/mapping[name]/name
    with path.open('rb') as f:scenario=pickle.load(f)
    assert scenario['id']==result['methods'][METHODS[0]]['scenario_id']
    ego=scenario['metadata']['sdc_id']
    offset=np.array(scenario['tracks'][ego]['state']['position'][0,:2],float)
    # Current cache coordinates equal the native centralized scenario coordinates.
    offset+=np.array(result['methods'][METHODS[0]]['initial_state']['position'])
    bounds=case['bounds'];region=box(bounds[0]-10,bounds[2]-10,bounds[1]+10,bounds[3]+10)
    def nearby(p):
        return box(p[:,0].min(),p[:,1].min(),p[:,0].max(),p[:,1].max()).intersects(region)
    features=[]
    for feature in scenario['map_features'].values():
        item=dict(type=feature['type'],route=bool(feature.get('is_sdc_route',False)))
        for key in ['polyline','polygon']:
            if key in feature:
                p=np.asarray(feature[key])[:,:2]-offset
                if len(p)>1 and nearby(p):item[key]=p.tolist()
        if 'polyline' in item or 'polygon' in item:features.append(item)
    agents=[]
    for track_id,track in scenario['tracks'].items():
        if track_id==ego or not track['state']['valid'][0]:continue
        state=track['state'];p=np.asarray(state['position'][0,:2])-offset
        if not region.contains(box(p[0]-.01,p[1]-.01,p[0]+.01,p[1]+.01)):continue
        item=dict(type=track['type'],xy=p.tolist(),heading=float(state['heading'][0]),
                  length=float(np.asarray(state['length'][0]).reshape(-1)[0]),
                  width=float(np.asarray(state['width'][0]).reshape(-1)[0]))
        agents.append(item)
    return dict(scene=scene,split=split,scenario_id=scenario['id'],features=features,agents=agents,
        coordinate='Native ScenarioNet coordinates centralized exactly as MetaDrive, then translated by saved trace start.',
        traffic_timestamp_s=0.,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def car(ax,xy,heading,length,width,color,outline=False,alpha=1,zorder=6,detail=True):
    rotation=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
    def world(p):return np.asarray(p)@rotation.T+xy
    # Rounded-looking chamfered outline, preserving actual bounding dimensions.
    l,w=length/2,width/2;c=min(.25*width,.12*length)
    body=world([[-l+c,-w],[l-c,-w],[l,-w+c],[l,w-c],[l-c,w],[-l+c,w],[-l,w-c],[-l,-w+c]])
    ax.add_patch(MplPolygon(body,closed=True,facecolor='white' if outline else color,
                           edgecolor=color if outline else 'white',lw=.65 if outline else .5,
                           alpha=alpha,zorder=zorder,joinstyle='round'))
    if not detail:return
    if outline:
        # Heading tick is schematic; track position/box/heading remain native.
        line=world([[.25*length,-.32*width],[.25*length,.32*width]])
        ax.plot(*line.T,color=color,lw=.4,alpha=alpha,zorder=zorder+.1)
    else:
        roof=world([[-.11*length,-.34*width],[.24*length,-.34*width],
                    [.29*length,.34*width],[-.11*length,.34*width]])
        ax.fill(*roof.T,color='white',alpha=.82,zorder=zorder+.1,lw=0)
        for xx in [-.24*length,.25*length]:
            for yy in [-.49*width,.49*width]:
                p=world([[xx-.08*length,yy],[xx+.08*length,yy]])
                ax.plot(*p.T,color='#33404B',lw=.9,zorder=zorder+.2,solid_capstyle='round')


def motion_heading(trace,index,initial):
    left=max(0,index-2);right=min(len(trace)-1,index+2)
    delta=trace[right,:2]-trace[left,:2]
    return float(np.arctan2(delta[1],delta[0])) if np.linalg.norm(delta)>1e-6 else initial


def draw_panel(ax,case,context,index):
    shapes=list(case['roads'].geoms) if hasattr(case['roads'],'geoms') else [case['roads']]
    for road in shapes:
        if road.geom_type!='Polygon':continue
        ax.fill(*np.asarray(road.exterior.coords).T,facecolor='#E9ECEF',edgecolor='#8D959C',lw=.58,zorder=0)
        for hole in road.interiors:
            ax.fill(*np.asarray(hole.coords).T,facecolor='white',edgecolor='#8D959C',lw=.58,zorder=.1)
    # Recorded route-lane polygons, not an inferred corridor around a policy path.
    for f in context['features']:
        if f['route'] and 'polygon' in f and 'LANE' in f['type']:
            ax.add_patch(MplPolygon(np.array(f['polygon']),closed=True,facecolor='#C9DFEF',
                                   edgecolor='none',alpha=.65,zorder=.2))
        if f['type']=='CROSSWALK' and 'polygon' in f:
            ax.add_patch(MplPolygon(np.array(f['polygon']),closed=True,facecolor='#F8F9FA',
                                   edgecolor='#C3C8CD',lw=.25,hatch='////',zorder=.4))
    centers=[]
    for f in context['features']:
        if 'polyline' not in f:continue
        p=np.array(f['polyline']);kind=f['type']
        if 'LANE' in kind:
            centers.append(p)
        elif 'ROAD_EDGE' in kind:
            ax.plot(*p.T,color='#818B93',lw=.65,zorder=1.1)
        elif 'ROAD_LINE' in kind:
            broken='BROKEN' in kind
            ax.plot(*p.T,color='#A1AAB1' if broken else '#AAB2B8',lw=.45 if broken else .4,
                    ls=(0,(2.5,2.2)) if broken else '-',zorder=1)
    if centers:ax.add_collection(LineCollection(centers,colors='#8F9CA6',linewidths=.28,
                                               linestyles=(0,(1,2)),alpha=.65,zorder=.8))
    for agent in context['agents']:
        p=np.array(agent['xy']);kind=agent['type']
        if kind in ['VEHICLE','CYCLIST']:
            car(ax,p,agent['heading'],agent['length'],agent['width'],'#4C6D99',
                outline=True,alpha=.82,zorder=2,detail=True)
        elif kind=='PEDESTRIAN':
            ax.scatter(*p,s=5.5,c='#9862B3',edgecolors='white',linewidths=.25,zorder=2.1)
        elif kind=='TRAFFIC_CONE':
            ax.scatter(*p,s=7,marker='^',c='#D6A34B',edgecolors='white',linewidths=.25,zorder=2.1)
        elif kind=='TRAFFIC_BARRIER':
            car(ax,p,agent['heading'],agent['length'],agent['width'],'#BC9C64',outline=True,zorder=2,detail=False)
    ref=case['ref'];trace=case['traces'][index];xy=trace[:,:2]
    ax.plot(*ref.T,color='white',lw=2.1,zorder=3,solid_capstyle='round')
    ax.plot(*ref.T,color='#D39737',lw=1.05,ls=(0,(3.5,2.5)),zorder=3.1)
    cmap=LinearSegmentedColormap.from_list('trajectory',COLORS[index])
    norm=Normalize(0,max(t[-1,2] for t in case['traces']))
    segments=np.stack([xy[:-1],xy[1:]],axis=1)
    ax.add_collection(LineCollection(segments,colors='white',lw=3.25,capstyle='round',zorder=4))
    lc=LineCollection(segments,cmap=cmap,norm=norm,lw=2.15,capstyle='round',zorder=4.1)
    lc.set_array(trace[1:,2]);ax.add_collection(lc)
    initial=case['result']['methods'][METHODS[index]]['initial_state']['heading']
    # Sparse actual trajectory snapshots, with bounded metric separation.
    last=0
    for k in range(1,len(trace)-1):
        if np.linalg.norm(xy[k]-xy[last])<12 or trace[k,2]-trace[last,2]<2.:continue
        if np.linalg.norm(xy[k]-xy[-1])<8:continue
        car(ax,xy[k],motion_heading(trace,k,initial),5.74,2.3,COLORS[index][1],
            outline=True,alpha=.38,zorder=4.2,detail=False)
        last=k
    ax.scatter(*xy[0],s=13,facecolor='white',edgecolor=COLORS[index][1],lw=.8,zorder=5)
    car(ax,xy[-1],motion_heading(trace,len(trace)-1,initial),5.74,2.3,COLORS[index][1],
        outline=False,zorder=6,detail=True)
    x0,x1,y0,y1=case['bounds']
    ax.set(xlim=(x0,x1),ylim=(y0,y1),aspect='equal',xticks=[],yticks=[])
    for spine in ax.spines.values():spine.set_visible(False)


def render(cases,contexts,target):
    n=len(cases);pairs=2 if n>2 else 1;rows=int(np.ceil(n/pairs))
    fig=plt.figure(figsize=(12.4 if pairs==2 else 6.3,rows*3.25),facecolor='white')
    grid=fig.add_gridspec(rows,pairs*2,left=.012,right=.99,bottom=.008,top=.94,wspace=.055,hspace=.12)
    for i,(case,context) in enumerate(zip(cases,contexts)):
        row,pair=divmod(i,pairs)
        for j in range(2):
            ax=fig.add_subplot(grid[row,pair*2+j]);draw_panel(ax,case,context,j)
            ax.set_title(TITLES[j],color=COLORS[j][1],fontsize=10,pad=7,fontweight='medium')
            if j==0:
                scene=context['scene'];label=f'Train scene {scene}' if context['split']=='train' else f'Scene {scene}'
                ax.text(.025,.975,f'({chr(97+i)})  {label}',transform=ax.transAxes,va='top',ha='left',
                        fontsize=8.7,fontweight='bold',color='#303942',zorder=20,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    for ext in ['png','pdf','svg']:fig.savefig(target.with_suffix('.'+ext),dpi=450,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=json.loads(a.provenance.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'svg.fonttype':'none','hatch.linewidth':.25})
    cases=[load_case(Path(c['source']),110.,None,20.) for c in old['cases']]
    assert np.allclose(np.array([c['bounds'] for c in cases]),old['panel_bounds'])
    contexts=[extract_context(c,a.dataset_root) for c in cases]
    for c,x in zip(cases,contexts):
        (a.output/f'scene_{x["scene"]}_render_context.json').write_text(json.dumps(x))
        render([c],[x],a.output/f'scene_{x["scene"]}_paired')
    render(cases,contexts,a.output/'trajectory_gallery_rich_bev')
    sources={}
    for c in cases:
        for name in ['r2dreamer_trajectory.csv','r2dreamer_context_trajectory.csv','reference_navigation.csv','results.json','map_polygons.json']:
            path=c['folder']/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    note=dict(original_provenance=old,styles='Layered recorded road geometry, route lane polygons, native centerlines/crosswalks, shared t0 traffic boxes and small recorded pedestrians/cones; full time-gradient policy paths and native-sized XL glyphs.',
        sources_sha256=sources,contexts=[{k:v for k,v in x.items() if k not in ['features','agents']} for x in contexts],
        invariants='Same four cases, identical pairwise bounds as previous balanced gallery, equal metric aspect; full unsmoothed traces and reference; all native t0 traffic shared across methods; no simulations or training rerun.',
        glyphs='Traffic positions/headings/dimensions from native t0 tracks. Ego XL5.74x2.3m, orientation estimated from local recorded motion; sparse outlines are actual trajectory locations. Car roof/wheel strokes and crosswalk hatching are schematic render decoration.',
        sources=['https://arxiv.org/pdf/2306.12241.pdf','https://metadriverse.github.io/trafficgen/'],
        reference='User-provided planning BEV figure; visual inspiration only. Selected examples remain qualitative and scene5242 remains train split.')
    (a.output/'provenance.json').write_text(json.dumps(note,indent=2))
    print('COMPLETED',a.output,flush=True)


if __name__=='__main__':main()
