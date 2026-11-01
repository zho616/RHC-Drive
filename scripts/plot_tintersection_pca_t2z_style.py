"""Reference-style single-factor strips and vehicle-type context PCA."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

TYPES=['default','m','l','s','xl']
NAMES=['Default','M','L','S (OOD)','XL (OOD)']
COLORS=['#426E97','#668B52','#C08C49','#B84953','#7F5B9B']
CMAP=LinearSegmentedColormap.from_list('parameter',['#B72136','#79527D','#123F70'])


def pca(x):
    x=np.asarray(x,np.float64)
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-10,scale,1.)
    z=(x-mean)/scale
    _,s,v=np.linalg.svd(z,full_matrices=False)
    # Deterministic component signs, chosen without parameter/type labels.
    signs=np.sign(v[np.arange(len(v)),np.abs(v).argmax(1)]);signs[signs==0]=1
    axes=v[:2]*signs[:2,None];xy=z@axes.T
    return xy,dict(mean=mean.tolist(),scale=scale.tolist(),axes=axes.tolist(),
                   explained_variance=(s[:2]**2/(s*s).sum()).tolist())


def export(fig,path):
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=450,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vehicle-data',type=Path,required=True)
    a=p.parse_args();figdir=a.root/'figures';figdir.mkdir(exist_ok=True)
    factor=a.root/'factor_probes'
    meta=json.loads((factor/'provenance.json').read_text())
    vehicle_meta=json.loads((a.vehicle_data/'results.json').read_text())
    assert meta['checkpoint_sha256']==vehicle_meta['checkpoint']['sha256']
    contract=dict(question='How does the same frozen T-trained context organize one-factor mass/friction changes and nominal vehicle types?',
        evidence=['Fixed-M one-factor mass sweep','Fixed-M one-factor wheel-friction sweep','Five nominal vehicle types at matched initial speed and physical front-wheel angle'],
        archetype='quantitative grid, two horizontal strips plus a larger type scatter',backend='python',
        exports=['PNG 450dpi','editable PDF','editable SVG','CSV','NPZ'],
        risks=['PCA is descriptive, not a causal parameter-identification result.',
               'All scatter coordinates are real PC1/PC2; no artificial y jitter or label-dependent rotation.',
               'Different panels fit PCA separately; their absolute coordinates cannot be compared.',
               'Vehicle types have different native physical parameters and normalized steering inputs.',
               'Initial speed is fixed once; subsequent speeds are allowed to diverge.'])
    (a.root/'figure_contract.json').write_text(json.dumps(contract,indent=2))
    with(factor/'windows.csv').open() as f:factor_rows=list(csv.DictReader(f))
    factors=np.load(factor/'representations.npz')
    with(a.vehicle_data/'source_data.csv').open() as f:all_vehicle_rows=list(csv.DictReader(f))
    vr=[r for r in all_vehicle_rows if r['regime']=='nominal']
    vx=np.array([[float(r[f'context_{i}']) for i in range(8)] for r in vr])
    panels={};source=[];arrays={}
    for name in ['mass','friction','vehicle']:
        if name=='vehicle':
            rows=vr;x=vx
        else:
            ids=np.array([i for i,r in enumerate(factor_rows) if r['factor']==name])
            rows=[factor_rows[i] for i in ids];x=factors['context'][ids]
        xy,fit=pca(x)
        panels[name]=dict(rows=rows,context=x,xy=xy,fit=fit)
        arrays[name+'_context']=x;arrays[name+'_coordinates']=xy
        for row,c,point in zip(rows,x,xy):
            out=dict(panel=name,vehicle=row['vehicle'],profile=int(row['profile']),
                     scale=float(row['scale']) if name!='vehicle' else 1.,
                     mass_kg=float(row['mass_kg'] if name!='vehicle' else row['actual_mass']),
                     wheel_friction=float(row['wheel_friction'] if name!='vehicle' else row['actual_wheel_friction']),
                     initial_speed_mps=10.,front_wheel_angle_deg=15.,pc1=float(point[0]),pc2=float(point[1]))
            out.update({f'context_{i}':float(c[i]) for i in range(8)});source.append(out)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.linewidth':.75,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.size':3,'ytick.major.size':3,'svg.fonttype':'none','pdf.fonttype':42,
        'legend.frameon':False})
    def factor_panel(ax,name,letter,strip=False):
        d=panels[name];xy=d['xy'];rows=d['rows'];var=d['fit']['explained_variance']
        key='mass_kg' if name=='mass' else 'wheel_friction'
        values=np.array([float(r[key]) for r in rows]);norm=Normalize(values.min(),values.max())
        order=np.random.default_rng(20261025).permutation(len(rows))
        artist=ax.scatter(xy[order,0],xy[order,1],c=values[order],cmap=CMAP,norm=norm,
                          s=17 if strip else 22,alpha=.78,edgecolors='white',linewidths=.18)
        title='Mass' if name=='mass' else 'Wheel friction'
        ax.set_title(f'({letter})  {title}',loc='left',pad=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.margins(.08);ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(3))
        cb=ax.figure.colorbar(artist,ax=ax,fraction=.042,pad=.025,aspect=14 if strip else 25)
        cb.set_label('Mass (kg)' if name=='mass' else 'Wheel friction',fontsize=7,labelpad=4)
        cb.set_ticks(np.linspace(values.min(),values.max(),3));cb.ax.tick_params(labelsize=7,length=2)
        return artist
    def vehicle_panel(ax,letter):
        d=panels['vehicle'];xy=d['xy'];rows=d['rows'];var=d['fit']['explained_variance']
        y=np.array([r['vehicle'] for r in rows])
        handles=[]
        for k,name in enumerate(TYPES):
            mask=y==name;marker='o' if k<3 else '^'
            ax.scatter(xy[mask,0],xy[mask,1],s=27 if k<3 else 36,marker=marker,
                       color=COLORS[k],alpha=.82,edgecolors='white',linewidths=.3)
            handles.append(Line2D([],[],linestyle='none',marker=marker,markersize=5,
                                  markerfacecolor=COLORS[k],markeredgecolor='none',label=NAMES[k]))
        ax.set_title(f'({letter})  Vehicle type',loc='left',pad=8)
        ax.set_xlabel('');ax.set_ylabel('')
        ax.margins(.14);ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ymin,ymax=xy[:,1].min(),xy[:,1].max();span=ymax-ymin
        ax.set_ylim(ymin-.12*span,ymax+.24*span)
        ax.set_box_aspect(1)
        ax.legend(handles=[handles[i] for i in [0,3,1,4,2]],loc='upper center',ncol=3,fontsize=6.7,
                  handletextpad=.3,columnspacing=.8,borderaxespad=.3)
    fig=plt.figure(figsize=(7.2,2.85),layout='constrained')
    grid=fig.add_gridspec(2,2,width_ratios=[1.15,1],hspace=.08,wspace=.14)
    factor_panel(fig.add_subplot(grid[0,0]),'mass','a',strip=True)
    factor_panel(fig.add_subplot(grid[1,0]),'friction','b',strip=True)
    vehicle_panel(fig.add_subplot(grid[:,1]),'c')
    export(fig,figdir/'context_pca_t2z_layout')
    for name in ['mass','friction','vehicle']:
        fig,ax=plt.subplots(figsize=(3.3,3.3) if name=='vehicle' else (3.7,3.15),layout='constrained')
        if name=='vehicle':vehicle_panel(ax,'c')
        else:factor_panel(ax,name,'a' if name=='mass' else 'b')
        export(fig,figdir/(name+'_context_pca'))
    with(a.root/'pca_source_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    np.savez_compressed(a.root/'pca_arrays.npz',**arrays)
    stats={}
    for name,d in panels.items():
        z=(d['context']-np.array(d['fit']['mean']))/np.array(d['fit']['scale'])
        labels=np.array([r['vehicle'] if name=='vehicle' else float(r['scale']) for r in d['rows']])
        eta=sum((labels==v).sum()*np.square(z[labels==v].mean(0)).sum() for v in np.unique(labels))/np.square(z).sum()
        stats[name]=dict(n=len(z),pca=d['fit'],descriptive_eta_squared=float(eta))
    (a.root/'metadata.json').write_text(json.dumps(dict(checkpoint=meta,vehicle_source=str(a.vehicle_data.resolve()),
        vehicle_checkpoint=vehicle_meta['checkpoint'],panels=stats,style='Reference-inspired two narrow continuous-color factor panels plus a larger categorical vehicle panel. White background. Colors encode real physical values; circles=training chassis, triangles=S/XL unseen chassis. No data-driven jitter or class-dependent coordinate transforms.'),indent=2))
    assert len(source)==432+120
    assert all(np.isfinite(v).all() for v in arrays.values())
    print(json.dumps({name:dict(n=v['n'],variance=v['pca']['explained_variance'],eta2=v['descriptive_eta_squared'])
                      for name,v in stats.items()},indent=2))
    print('COMPLETED',figdir)


if __name__=='__main__':main()
