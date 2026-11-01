"""Mass-color/friction-size PCA for the full factorial frozen context probe."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_tintersection_pca_t2z_style import CMAP,export,pca


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'figures';out.mkdir(exist_ok=True)
    (a.root/'figure_contract.json').write_text(json.dumps(dict(
        question='How are simultaneous mass and wheel-friction changes organized in the frozen context?',
        evidence='Full9x9 factorial grid, identical24 throttle profiles at each cell; PCA fitted to all1944 native contexts before aggregation.',
        archetype='single quantitative scatter with dual parameter encodings',backend='python',
        exports=['PNG450dpi','editable PDF','editable SVG','all-window CSV','condition-mean CSV','NPZ'],
        risks=['Main points are means of24 projected contexts, not individual episodes.',
               'All-window projection is separately provided on the same fitted PCA basis.',
               'No label-conditioned coordinate transforms; no parameter disentanglement claim.']),indent=2))
    d=np.load(a.root/'probes/representations.npz')
    with(a.root/'probes/windows.csv').open() as f:rows=list(csv.DictReader(f))
    assert d['context'].shape==(1944,8) and len(rows)==1944
    xy,fit=pca(d['context'])
    means=[]
    for ms in np.arange(6,15)/10:
        for fs in np.arange(6,15)/10:
            ids=np.array([i for i,r in enumerate(rows) if np.isclose(float(r['mass_scale']),ms) and np.isclose(float(r['friction_scale']),fs)])
            assert len(ids)==24 and {int(rows[i]['profile']) for i in ids}==set(range(24))
            point=xy[ids].mean(0)
            row=dict(mass_scale=float(ms),friction_scale=float(fs),n=24,
                     mass_kg=float(np.mean([float(rows[i]['mass_kg']) for i in ids])),
                     wheel_friction=float(np.mean([float(rows[i]['wheel_friction']) for i in ids])),
                     pc1=float(point[0]),pc2=float(point[1]))
            row.update({f'context_{k}':float(d['context'][ids,k].mean()) for k in range(8)});means.append(row)
    for i,row in enumerate(rows):
        row.update(pc1=float(xy[i,0]),pc2=float(xy[i,1]))
        row.update({f'context_{k}':float(d['context'][i,k]) for k in range(8)})
    write_csv(a.root/'all_windows.csv',rows);write_csv(a.root/'condition_means.csv',means)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.linewidth':.75,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.size':3,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42,
        'legend.frameon':False})
    mass_norm=Normalize(min(r['mass_kg'] for r in means),max(r['mass_kg'] for r in means))
    fmin,fmax=min(r['wheel_friction'] for r in means),max(r['wheel_friction'] for r in means)
    for name,data,individual in [('mass_friction_compound_pca',means,False),
                                  ('mass_friction_compound_pca_all_windows',rows,True)]:
        fig=plt.figure(figsize=(4.8,3.35),layout='constrained')
        grid=fig.add_gridspec(2,2,width_ratios=[1,.25],height_ratios=[1.1,1],wspace=.08)
        ax=fig.add_subplot(grid[:,0]);barholder=fig.add_subplot(grid[0,1]);barholder.set_axis_off()
        cax=barholder.inset_axes([.18,.08,.17,.78])
        legendax=fig.add_subplot(grid[1,1]);legendax.set_axis_off()
        size=lambda f:(4+18*(np.asarray(f)-fmin)/(fmax-fmin)) if individual else (28+120*(np.asarray(f)-fmin)/(fmax-fmin))
        # Shuffle all cells with a predetermined seed; no favored parameter draws last.
        order=np.random.default_rng(20261025).permutation(len(data))
        mass=np.array([r['mass_kg'] for r in data],float);friction=np.array([r['wheel_friction'] for r in data],float)
        coords=np.array([[r['pc1'],r['pc2']] for r in data])
        sc=ax.scatter(coords[order,0],coords[order,1],c=mass[order],s=size(friction[order]),
                      cmap=CMAP,norm=mass_norm,alpha=.3 if individual else .9,
                      edgecolors='none' if individual else '#777777',linewidths=0 if individual else .35)
        ax.set_title('(d)  Compounded factors',loc='left',pad=8)
        ax.set_box_aspect(1);ax.margins(.1)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        cb=fig.colorbar(sc,cax=cax);cb.ax.set_title('Mass (kg)',fontsize=8,pad=8)
        cb.set_ticks([720,1200,1680]);cb.ax.tick_params(labelsize=7,length=2)
        # Solid colorbar even for the transparent all-window display.
        cb.solids.set_alpha(1)
        levels=[fmin,(fmin+fmax)/2,fmax]
        handles=[ax.scatter([],[],s=float(size(v)),facecolors='none',edgecolors='#777777',linewidths=.7,label=f'{v:.2f}') for v in levels]
        legendax.legend(handles=handles,title='Wheel friction',loc='upper left',bbox_to_anchor=(-.12,1),
                        fontsize=7,title_fontsize=7.5,labelspacing=1.1,borderaxespad=0,handletextpad=.65)
        export(fig,out/name)
    np.savez_compressed(a.root/'pca_arrays.npz',context=d['context'],coordinates=xy,
                        condition_mean_coordinates=np.array([[r['pc1'],r['pc2']] for r in means]))
    meta=dict(pca=fit,contexts=1944,conditions=81,profiles_per_condition=24,
        main_points='Mean of24 projected contexts at each physical parameter combination. PCA is fitted once to all1944 individual contexts; means do not influence fit or component sign.',
        encodings=dict(color='Actual vehicle mass, kg; red=low,blue=high',area='Linearly increasing with actual Bullet wheel-friction slip'),
        limits='Separate PCA basis from prior single-factor and type figures; no direct coordinate comparison. No claim that mass/friction are disentangled or world-model prediction improves.',
        checkpoint=json.loads((a.root/'probes/provenance.json').read_text()))
    (a.root/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETED',out,'variance',fit['explained_variance'])


if __name__=='__main__':main()
