"""TrafficGen-inspired wire-map BEV: white roads, filled boxes, thin paths."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Polygon
import numpy as np

from plot_nuplan_trajectory_gallery import load_case,METHODS
from restyle_nuplan_trajectory_gallery import motion_heading

TITLES=['R2Dreamer','Ours']
COLORS=[('#8495EC','#4058CD'),('#EF9295','#CF444F')]
# Display-only omission requested for the paired upper-right case. The original
# t0 context, recorded traffic interactions, trajectories and results stay intact.
HIDDEN_BACKGROUND_AGENTS={31:{0}}
SCENE_NAMES={
    13:dict(title='Medium-speed\ndriving',name='Medium-speed driving',slug='medium_speed_driving',native_type='medium_magnitude_speed'),
    31:dict(title='Protected\nintersection turn',name='Protected intersection turn',slug='protected_intersection_turn',native_type='starting_protected_cross_turn'),
    5242:dict(title='Crosswalk traversal\n(train)',name='Crosswalk traversal (train)',slug='crosswalk_traversal_train',native_type='traversing_crosswalk'),
    57:dict(title='Stop-sign\nintersection traversal',name='Stop-sign intersection traversal',slug='stop_sign_intersection_traversal',native_type='starting_straight_stop_sign_intersection_traversal'),
}


def rectangle(ax,xy,heading,length,width,fill,edge='#303030',zorder=3):
    rot=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
    p=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])@rot.T+xy
    ax.add_patch(Polygon(p,closed=True,facecolor=fill,edgecolor=edge,lw=.32,zorder=zorder))


def panel(ax,case,context,index,draw_area_context=False):
    # Recorded edges and markings, with no road-surface or route-lane color fill.
    centers=[]
    for f in context['features']:
        if draw_area_context and 'polygon' in f:
            # ScenarioNet stores sidewalks and crosswalks as polygons, not
            # polylines. Preserve the exact observed boundary, including its
            # non-driving-side contour; do not invent building/land-use shapes.
            if f['type']=='ROAD_EDGE_SIDEWALK':
                ax.add_patch(Polygon(np.asarray(f['polygon']),closed=True,fill=False,
                    edgecolor='#585858',lw=.5,zorder=.6))
            elif f['type']=='CROSSWALK':
                ax.add_patch(Polygon(np.asarray(f['polygon']),closed=True,fill=False,
                    edgecolor='#A1A1A1',lw=.32,zorder=.65))
        if 'polyline' not in f:continue
        xy=np.array(f['polyline']);kind=f['type']
        if 'ROAD_EDGE' in kind:
            ax.plot(*xy.T,color='#292929',lw=.58,zorder=1)
        elif 'ROAD_LINE' in kind:
            ax.plot(*xy.T,color='#666666',lw=.36,
                    ls=(0,(3.3,2.5)) if 'BROKEN' in kind else '-',zorder=.8)
        elif 'LANE' in kind:
            centers.append(xy)
    if centers:
        ax.add_collection(LineCollection(centers,colors='#B5B5B5',lw=.24,
                                        linestyles=(0,(2.2,2.2)),zorder=.4))
    for agent_index,agent in enumerate(context['agents']):
        if agent_index in HIDDEN_BACKGROUND_AGENTS.get(context['scene'],set()):continue
        xy=np.array(agent['xy']);kind=agent['type']
        if kind in ['VEHICLE','CYCLIST','TRAFFIC_BARRIER']:
            rectangle(ax,xy,agent['heading'],agent['length'],agent['width'],'#252525',edge='#151515',zorder=2)
        elif kind=='PEDESTRIAN':
            ax.scatter(*xy,s=2,c='#333333',edgecolors='none',zorder=2)
        elif kind=='TRAFFIC_CONE':
            ax.scatter(*xy,s=3,marker='^',c='#333333',edgecolors='none',zorder=2)
    ref=case['ref'];trace=case['traces'][index];xy=trace[:,:2]
    ax.plot(*ref.T,color='#D9A531',lw=.85,ls=(0,(4,2.2)),zorder=3)
    segments=np.stack([xy[:-1],xy[1:]],1)
    cmap=LinearSegmentedColormap.from_list('method',COLORS[index])
    norm=Normalize(0,max(t[-1,2] for t in case['traces']))
    ax.add_collection(LineCollection(segments,colors='white',lw=2.05,capstyle='round',zorder=3.8))
    lc=LineCollection(segments,cmap=cmap,norm=norm,lw=1.35,capstyle='round',zorder=4)
    lc.set_array(trace[1:,2]);ax.add_collection(lc)
    ax.scatter(*xy[0],s=7,facecolors='white',edgecolors=COLORS[index][1],lw=.6,zorder=5)
    initial=case['result']['methods'][METHODS[index]]['initial_state']['heading']
    rectangle(ax,xy[-1],motion_heading(trace,len(trace)-1,initial),5.74,2.3,COLORS[index][1],zorder=6)
    x0,x1,y0,y1=case['bounds']
    ax.set(xlim=(x0,x1),ylim=(y0,y1),aspect='equal',xticks=[],yticks=[],facecolor='white')
    for spine in ax.spines.values():spine.set_color('#D9D9D9');spine.set_linewidth(.55)


def render(cases,contexts,target,draw_area_context=False):
    columns=2 if len(cases)>2 else 1
    rows=int(np.ceil(len(cases)/columns))
    fig=plt.figure(figsize=(12.4 if columns==2 else 6.2,rows*3.25+.12),facecolor='white')
    outer=fig.add_gridspec(rows,columns,left=.014,right=.986,
                          bottom=.015,top=.94,wspace=.12,hspace=.16)
    for i,(case,context) in enumerate(zip(cases,contexts)):
        row,col=divmod(i,columns)
        inner=outer[row,col].subgridspec(1,2,wspace=.055)
        for j in range(2):
            ax=fig.add_subplot(inner[0,j]);panel(ax,case,context,j,draw_area_context)
            ax.set_title(TITLES[j],fontsize=9.4,pad=6,color='#444444',fontweight='normal')
    for ext in ['png','pdf','svg']:fig.savefig(target.with_suffix('.'+ext),dpi=450,facecolor='white')
    plt.close(fig)


def compact_composition(cases):
    """Tighten real view bounds; keep the same crop for paired methods.

    Long approaches use the last 75m of the navigation route as a display focus.
    Source paths remain complete and are merely clipped by the axes. Always keep
    both actual terminal positions visible, including a possible early failure.
    """
    from shapely.geometry import LineString,Point
    notes=[]
    for case in cases:
        ref=case['ref'];traces=case['traces']
        points=np.concatenate([ref]+[t[:,:2] for t in traces])
        focus='Complete recorded paths and reference'
        if np.ptp(points,axis=0).max()+20>100:
            route=LineString(ref)
            threshold=max(0.,route.length-75.)
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))]
            parts=[ref[distance>=threshold]]
            for trace in traces:
                progress=np.array([route.project(Point(p)) for p in trace[:,:2]])
                parts.extend([trace[progress>=threshold,:2],trace[-1:,:2]])
            points=np.concatenate(parts)
            focus='Final75m of navigation route and corresponding policy segments; straight approach can lie outside view. Both terminal positions retained.'
        lo=points.min(axis=0);hi=points.max(axis=0)
        span=max(60.,float((hi-lo).max())+20.)
        center=(hi+lo)/2
        case['bounds']=[float(center[0]-span/2),float(center[0]+span/2),
                        float(center[1]-span/2),float(center[1]+span/2)]
        case['span']=span
        x0,x1,y0,y1=case['bounds']
        for trace in traces:
            assert x0<trace[-1,0]<x1 and y0<trace[-1,1]<y1
        notes.append(dict(scene=case['result']['methods'][METHODS[0]]['scene'],
            focus=focus,panel_bounds=case['bounds'],span_m=span,
            visible_recorded_positions=[int(((t[:,0]>=x0)&(t[:,0]<=x1)&(t[:,1]>=y0)&(t[:,1]<=y1)).sum()) for t in traces],
            total_recorded_positions=[len(t) for t in traces]))
    return notes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--draw-area-context',action='store_true',
                   help='Include native polygon-only sidewalk and crosswalk boundaries')
    p.add_argument('--compact-composition',action='store_true',
                   help='Tighter shared views; long approaches focus on the final75m of route')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.context_root/'provenance.json').read_text())['original_provenance']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    cases=[load_case(Path(c['source']),110.,None,20.) for c in old['cases']]
    assert np.allclose([c['bounds'] for c in cases],old['panel_bounds'])
    composition=compact_composition(cases) if a.compact_composition else None
    contexts=[json.loads((a.context_root/f'scene_{c["result"]["methods"][METHODS[0]]["scene"]}_render_context.json').read_text()) for c in cases]
    for case,context in zip(cases,contexts):
        assert context['scenario_id']==case['result']['methods'][METHODS[0]]['scenario_id']
        render([case],[context],a.output/(SCENE_NAMES[context['scene']]['slug']+'_paired'),a.draw_area_context)
    render(cases,contexts,a.output/'trajectory_gallery_trafficgen',a.draw_area_context)
    hashes={str(c['folder']/name):hashlib.sha256((c['folder']/name).read_bytes()).hexdigest()
            for c in cases for name in ['r2dreamer_trajectory.csv','r2dreamer_context_trajectory.csv','reference_navigation.csv']}
    (a.output/'provenance.json').write_text(json.dumps(dict(style='TrafficGen-inspired white wire-map, dark road edges, gray lane lines, filled black native t0 traffic rectangles, thin blue/red policy trajectories and colored terminal XL boxes.',
        context_source=str(a.context_root.resolve()),original_provenance=old,sources_sha256=hashes,
        scene_names={str(k):v for k,v in SCENE_NAMES.items()},
        naming='Only method labels are displayed. Scene names, identifiers and split labels are retained in provenance only; the third case is still train split.',
        layout='Four groups in a2x2 arrangement; R2Dreamer left and Ours right within each group. No scene titles, group numbers or separator lines. Identical square paired views.',
        invariants='Identical full source trajectories and reference. Shared pairwise views; actual framing recorded under composition. Native positions/headings/sizes used. No surface fills, route highlights, decorative vehicle details, or repeated ego footprints.',
        composition=composition,
        display_only_hidden_agents={str(scene):[dict(agent_index=i,agent=json.loads((a.context_root/f'scene_{scene}_render_context.json').read_text())['agents'][i]) for i in sorted(indices)] for scene,indices in HIDDEN_BACKGROUND_AGENTS.items()},
        traffic_display='For scene31 only, omit t0 vehicle index0 from both panels because its static glyph overlaps both policy trajectories. Display-only change; raw context and evaluation outcomes unchanged.',
        polygon_context=dict(enabled=a.draw_area_context,
            types=['ROAD_EDGE_SIDEWALK','CROSSWALK'] if a.draw_area_context else [],
            note='Exact recorded polygon outlines, no synthetic buildings or land-use. Native cache and paths unchanged; framing recorded under composition.'),
        reference='User-provided TrafficGen Figure1; visual style adaptation, not generated traffic or a TrafficGen experiment.'),indent=2))
    print('COMPLETED',a.output,flush=True)


if __name__=='__main__':main()
