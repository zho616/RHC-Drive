#!/usr/bin/env python3
"""Three no-band T-intersection validation learning curves with a shared legend."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

METHODS=[
    ('T_PPO','PPO','#4AA8E8','-'),
    ('T_SAC','SAC','#C99764','-'),
    ('T_DV3','DreamerV3','#747D8D','-'),
    ('T_DV3_context','DreamerV3+context','#9441BD','--'),
    ('T_R2','R2Dreamer','#194F6A','-'),
    ('T_R2_context','R2Dreamer+context','#ED3943','-'),
]
METRICS=[('reward','episode/eval_score','Episode Reward'),
         ('success','episode/eval_success','Success Rate'),
         ('route_completion','episode/eval_route_completion','Route Completion Rate')]
TENSORBOARD_RUNS={
    'T_PPO':'13_500K_PPO', 'T_SAC':'14_500K_SAC',
    'T_DV3':'15_500K_DreamerV3_baseline',
    'T_DV3_context':'16_500K_DreamerV3_DALI_shallow_swap1',
    'T_R2':'01_500K_baseline',
    'T_R2_context':'11_500K_state4_DALI_shallow_swap1',
}


def tensorboard_ema(values,weight=.98):
    """Match TensorBoard 2.21 Scalars resmoothDataset, including debiasing."""
    values=np.asarray(values,dtype=float)
    if not len(values):return values.copy()
    constant=bool(np.all(values==values[0]))
    result=[];last=0.;count=0
    for value in values:
        if constant or not np.isfinite(value):result.append(value);continue
        last=last*weight+(1-weight)*value;count+=1
        denominator=1-weight**count if weight!=1 else 1.
        result.append(last/denominator)
    return np.asarray(result)


def write_csv(path,rows):
    with path.open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path)
    p.add_argument('--exports',type=Path,help='Reuse saved TensorBoard scalar JSON exports')
    p.add_argument('--tensorboard-url',help='Read exact scalar point sequences from a running TensorBoard')
    p.add_argument('--smoothing',type=float,default=.98)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not args.tensorboard_url and not args.exports and not args.snapshots:p.error('Provide --tensorboard-url or --snapshots')
    if not 0<=args.smoothing<1:p.error('Smoothing must be in [0,1)')
    raw=[];data={};plotrows=[];hashes={};duplicates={};max_offset=0
    fetched={}
    if args.tensorboard_url or args.exports:
        (out/'tensorboard_exports').mkdir(exist_ok=True)
        def fetch(task):
            id,metric,tag=task
            query=urllib.parse.urlencode(dict(run=TENSORBOARD_RUNS[id],tag=tag,format='json'))
            if args.exports:
                values=json.loads((args.exports/(id+'_'+metric+'.json')).read_text())
            else:
                url=args.tensorboard_url.rstrip('/')+'/data/plugin/scalars/scalars?'+query
                with urllib.request.urlopen(url,timeout=30)as response:values=json.load(response)
            assert values,(id,metric)
            content=json.dumps(values,indent=2)+'\n'
            (out/'tensorboard_exports'/(id+'_'+metric+'.json')).write_text(content)
            return (id,metric),values,hashlib.sha256(content.encode()).hexdigest()
        tasks=[(id,m,key)for id,_,_,_ in METHODS for m,key,_ in METRICS]
        with ThreadPoolExecutor(max_workers=6)as pool:
            for (id,metric),values,digest in pool.map(fetch,tasks):
                fetched[id,metric]=values;hashes[id+'_'+metric]=digest
    for id,label,color,style in METHODS:
        points=[];seen=set();extra=[]
        if args.tensorboard_url or args.exports:
            reference=fetched[id,'reward']
            assert all([x[1]for x in fetched[id,m]]==[x[1]for x in reference]for m,_,_ in METRICS)
            records=[(i,dict(step=value[1],wall_time=value[0],**{key:fetched[id,m][i][2]for m,key,_ in METRICS}))for i,value in enumerate(reference)]
        else:
            source=args.snapshots/(id+'.jsonl');content=source.read_bytes();hashes[id]=hashlib.sha256(content).hexdigest()
            records=[(i,json.loads(line))for i,line in enumerate(content.decode().splitlines())if 'episode/eval_score'in line]
        for index,row in records:
            schedule=int(round(row['step']/10000)*10000)
            max_offset=max(max_offset,abs(schedule-row['step']))
            value=dict(run_id=id,method=label,record_index=index,recorded_step=row['step'],scheduled_step=schedule,
                wall_time=row.get('wall_time',''),episodes=20,used_for_plot=True,
                **{metric:float(row[key])for metric,key,_ in METRICS})
            assert all(np.isfinite(value[m])for m,_,_ in METRICS)
            assert all(-1e-6<=value[m]<=1+1e-6 for m in ('success','route_completion'))
            raw.append(value)
            if schedule in seen:extra.append(index)
            seen.add(schedule);points.append(value)
        assert sorted(seen)==list(range(0,500001,10000)),id
        assert all(a['recorded_step']<=b['recorded_step']for a,b in zip(points,points[1:]))
        smooth_values={m:tensorboard_ema([r[m]for r in points],args.smoothing)for m,_,_ in METRICS}
        for i,row in enumerate(points):
            result=dict(run_id=id,method=label,recorded_step=row['recorded_step'],step=row['recorded_step'],
                points_seen=i+1,smoothing_weight=args.smoothing,episodes_per_checkpoint=20)
            for metric,_,_ in METRICS:
                result['raw_'+metric]=row[metric];result['smoothed_'+metric]=float(smooth_values[metric][i])
            plotrows.append(result)
        data[id]=[r for r in plotrows if r['run_id']==id];duplicates[id]=extra
    assert max_offset<=16
    write_csv(out/'validation_source.csv',raw);write_csv(out/'plot_source_data.csv',plotrows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],
        'font.size':8,'axes.titlesize':8.7,'axes.labelsize':7.4,'xtick.labelsize':6.7,'ytick.labelsize':6.7,
        'axes.facecolor':'white','axes.edgecolor':'#444444','axes.linewidth':.6,
        'axes.axisbelow':True,'grid.color':'white','grid.linewidth':.65,
        'xtick.major.size':2,'ytick.major.size':2,'xtick.major.width':.4,'ytick.major.width':.4,
        'xtick.color':'#202020','ytick.color':'#202020',
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','text.usetex':False})

    def panel(ax,metric,title,smooth=True):
        for id,label,color,style in METHODS:
            points=data[id];x=np.array([v['step']for v in points]);y=np.array([v[('smoothed_'if smooth else'raw_')+metric]for v in points])
            if metric!='reward':y=np.clip(y,0,1)
            ax.plot(x,y,color=color,linestyle=style,lw=1.4 if id=='T_R2_context'else 1.15,label=label)
        ax.set_xlim(0,500010);ax.set_xticks(np.arange(0,500001,100000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:'0'if x==0 else f'{x/1000:.0f}k'))
        ax.set_xlabel('Environment Steps',labelpad=4);ax.set_ylabel(title,labelpad=4);ax.set_title(title,pad=6)
        if metric=='reward':ax.set_ylim(-20,105);ax.set_yticks([0,25,50,75,100])
        else:ax.set_ylim(-.025,1.025);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for name,spine in ax.spines.items():spine.set_visible(True)
        ax.grid(False);ax.tick_params(pad=2)
        assert len(ax.collections)==0,'No uncertainty or variability bands are permitted'

    handles=[Line2D([0],[0],color=c,linestyle=s,lw=1.3,label=l)for _,l,c,s in METHODS]
    # Fill the legend rows as PPO/SAC/DreamerV3, then context/R2/R2+context.
    legend_order=[0,3,1,4,2,5]
    exported=[]
    def export(fig,stem):
        fig.canvas.draw()
        for suffix in ('png','pdf','svg','tiff'):
            options={'dpi':600}if suffix in ('png','tiff')else{}
            if suffix=='tiff':options['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(out/(stem+'.'+suffix),**options);exported.append(stem+'.'+suffix)
        fig.savefig(out/(stem+'_preview.png'),dpi=240)

    for smooth in (True,False):
        fig,axes=plt.subplots(1,3,figsize=(183/25.4,86/25.4))
        fig.subplots_adjust(left=.065,right=.988,bottom=.32,top=.88,wspace=.34)
        for ax,(metric,key,title)in zip(axes,METRICS):panel(ax,metric,title,smooth)
        legend=fig.legend(handles=[handles[i]for i in legend_order],loc='upper center',bbox_to_anchor=(.52,.175),
            ncol=3,frameon=False,fontsize=7.5,columnspacing=1.6,handlelength=2.1,handletextpad=.5,labelspacing=.7)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert all(not legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))for ax in axes)
        export(fig,'tintersection_learning_metrics'if smooth else'tintersection_learning_metrics_raw')
        plt.close(fig)
    for metric,key,title in METRICS:
        fig=plt.figure(figsize=(89/25.4,83/25.4));ax=fig.add_axes([.17,.32,.79,.55])
        panel(ax,metric,title)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.175),ncol=2,
            frameon=False,fontsize=6.7,handlelength=1.7,columnspacing=1.25,labelspacing=.6)
        export(fig,'tintersection_'+metric);plt.close(fig)
    provenance=dict(metric_scope='Training-time fixed validation, not training rollouts or independent final test',
        training_seed=0,validation_episodes=20,evaluation_interval_steps=10000,
        smoothing='TensorBoard Scalars debiased exponential moving average',smoothing_weight=args.smoothing,
        tensorboard_url=args.tensorboard_url,tensorboard_runs=TENSORBOARD_RUNS,
        algorithm_source='https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/components/vz_line_chart2/line-chart.ts#L690-L724',
        uncertainty_bands=False,background='white',grid=False,method_styles=METHODS,source_exports=str(args.exports),source_hashes=hashes,retained_duplicate_record_indices=duplicates,
        max_schedule_offset_steps=max_offset,step_rounding_for_plot=False,all_metrics_same_checkpoints=True,formats=exported)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'CAPTION.md').write_text(
        'T-intersection learning curves under the shared mass/friction randomization protocol. '
        'All three panels use the same periodic validation records (20 fixed validation episodes every '
        '10,000 environment steps). Lines use TensorBoard Scalars exponential smoothing with weight '
        f'{args.smoothing:g}, including initial-value debiasing; no preliminary binning or rolling mean is used. '
        'One training seed per method (seed 0); no uncertainty bands are shown. '
        'Validation route completion is used because per-training-episode route completion was not logged. '
        'The raw companion uses identical records without smoothing. Duplicate-step records are retained '
        'in TensorBoard order and actual recorded steps are used without rounding. '
        'These curves are distinct from independent 100-episode best-checkpoint test results.\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
