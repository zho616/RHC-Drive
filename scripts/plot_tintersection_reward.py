#!/usr/bin/env python3
"""T-intersection reward curves from real training logs, with within-run SD."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

METHODS = [
    ('T_SAC', 'SAC', '#C69A69', '-'),
    ('T_DV3', 'DreamerV3', '#778195', '-'),
    ('T_R2', 'R2Dreamer', '#19516D', '-'),
    ('T_PPO', 'PPO', '#24A79D', '-.'),
    ('T_DV3_context', 'DreamerV3+context', '#50A6E0', '--'),
    ('T_R2_context', 'R2Dreamer+context', '#E7454B', '-'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with args.source.open() as stream: allrows=list(csv.DictReader(stream))
    raw=[]; plotted=[]; data={};counts={}
    grid=np.arange(2500,500001,2500)
    for id,label,color,style in METHODS:
        rows=[r for r in allrows if r['run_id']==id and r['series']=='training_return']
        assert rows and all(r['cohort']=='mass_friction_243' and r['training_seed']=='0' for r in rows)
        rows=sorted(rows,key=lambda r:(float(r['step']),int(r['record_index'])))
        steps=np.array([float(r['step'])for r in rows]);reward=np.array([float(r['reward'])for r in rows])
        assert np.isfinite(steps).all() and np.isfinite(reward).all()
        counts[id]=len(rows);raw.extend(rows);points=[]
        for step in grid:
            selected=(steps>max(0,step-20000))&(steps<=step)
            values=reward[selected]
            if len(values)<3:continue
            mean=float(values.mean());sd=float(values.std(ddof=1))
            point=dict(run_id=id,method=label,step=int(step),window_start=max(0,int(step)-20000),
                n_records=int(len(values)),mean_reward=mean,within_run_sd=sd,
                band_low=mean-sd,band_high=mean+sd)
            plotted.append(point);points.append(point)
        data[id]=points
    for filename,rows in [('training_source.csv',raw),('plot_source_data.csv',plotted)]:
        with (args.output/filename).open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],
        'font.size':9.5,'axes.labelsize':10.5,'axes.titlesize':10,'axes.linewidth':.8,
        'xtick.labelsize':9,'ytick.labelsize':9,'xtick.direction':'out','ytick.direction':'out',
        'xtick.major.width':.7,'ytick.major.width':.7,'svg.fonttype':'none','pdf.fonttype':42,
        'text.usetex':False,'savefig.facecolor':'white'})
    bounds=[p[k]for values in data.values()for p in values for k in ('band_low','band_high')]
    ylim=(np.floor(min(bounds)/20)*20-5,np.ceil(max(bounds)/20)*20+5)
    exported=[];layout=[]
    for band in (True,False):
        fig=plt.figure(figsize=(160/25.4,125/25.4),facecolor='white')
        ax=fig.add_axes([.13,.17,.835,.62])
        for id,label,color,style in METHODS:
            points=data[id];x=np.array([p['step']for p in points]);y=np.array([p['mean_reward']for p in points]);sd=np.array([p['within_run_sd']for p in points])
            if band:ax.fill_between(x,y-sd,y+sd,color=color,alpha=.10,linewidth=0,zorder=1)
            ax.plot(x,y,color=color,linestyle=style,linewidth=1.8 if id=='T_R2_context'else 1.5,zorder=3)
        ax.set_xlim(0,500000);ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(0,500001,100000));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:'0'if x==0 else f'{x/1000:.0f}k'))
        ax.yaxis.set_major_locator(MultipleLocator(40))
        ax.set_xlabel('Step',labelpad=5);ax.set_ylabel('Episode Return',labelpad=7)
        ax.set_title('T-intersection',pad=9)
        ax.grid(False)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#353535')
        handles=[Line2D([0],[0],color=c,linestyle=s,linewidth=1.7,label=l)for _,l,c,s in METHODS]
        # Matplotlib fills legend columns vertically; reorder into two method rows.
        order=[0,3,1,4,2,5]
        legend=fig.legend(handles=[handles[i]for i in order],ncol=3,loc='upper center',bbox_to_anchor=(.515,.985),
            frameon=True,fancybox=True,edgecolor='#DDDDDD',facecolor='white',framealpha=1,
            fontsize=8.2,handlelength=2.05,handletextpad=.55,columnspacing=1.35,borderpad=.5,labelspacing=.55)
        legend.get_frame().set_linewidth(.5)
        note=('1 training seed; shading: ±1 local SD (20K-step window).'
              if band else '1 training seed; trailing 20K-step mean; no uncertainty band.')
        footer=fig.text(.55,.035,note,ha='center',va='center',fontsize=7.2,color='#555555')
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert not legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer))
        assert fig.bbox.contains(*footer.get_window_extent(renderer).get_points()[0])
        stem='tintersection_reward'if band else 'tintersection_reward_no_band'
        for suffix in ('png','pdf','svg','tiff'):
            target=args.output/(stem+'.'+suffix)
            options={'dpi':600}if suffix in ('png','tiff')else{}
            if suffix=='tiff':options['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(target,**options);exported.append(target.name)
        # Inline-friendly preview, rendered by the same Python backend.
        fig.savefig(args.output/(stem+'_preview.png'),dpi=200)
        layout.append(dict(variant=stem,legend_clear=True,ylim=list(ylim)))
        plt.close(fig)
    provenance=dict(source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        input_records_per_method=counts,training_seeds=[0],window_steps=20000,grid_steps=2500,
        spread='within-run return-log sample SD; not across training seeds',
        budget=500000,extrapolation=False,backend='Python matplotlib',formats=exported)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.output/'qa.json').write_text(json.dumps(dict(layout=layout,finite_data=True,methods=6,series='training_return'),indent=2)+'\n')
    (args.output/'CAPTION.md').write_text(
        'T-intersection learning curves under the shared mass/friction randomization protocol. '
        'Lines show the trailing 20,000-environment-step mean of recorded episode returns; '
        'shaded regions show ±1 sample standard deviation of return records within the same window. '
        'Each method has one training seed (seed 0). Shading represents within-run variability, '
        'not uncertainty across independent training runs. All six methods use the same smoothing rule '
        'and a 500,000-environment-step budget. Validation and test returns are excluded.\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
