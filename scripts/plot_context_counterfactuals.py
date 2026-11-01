"""Publication panels for native frozen Shallow-context interventions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Figure contract: test context dependence and propagation into native future
# predictions. Do not claim mass/friction identification or accuracy improvement.
# Quantitative grid; Python; 183-mm width; editable PDF/SVG, PNG and source CSV.
COLORS={'r2dreamer_context':'#ED3943','dreamerv3_context':'#9441BD'}
LABELS={'r2dreamer_context':'R2Dreamer + Context','dreamerv3_context':'DreamerV3 + Context'}


def export(fig,path):
    for ext in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+ext),dpi=450,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    out=a.root/'figures';out.mkdir(exist_ok=True)
    summary=json.loads((a.root/'summary.json').read_text())
    with(a.root/'sensitivity_curves.csv').open() as f:curves=list(csv.DictReader(f))
    with(a.root/'representative_predictions.csv').open() as f:examples=list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.75,'legend.frameon':False})
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.95),layout='constrained')
    for label,model in summary.items():
        rows=model['dimensions'];color=COLORS[label]
        x=np.arange(1,9)
        axs[0,0].plot(x,[r['posterior_kl'] for r in rows],color=color,marker='o',ms=3,lw=1.2,label=LABELS[label])
        dim=model['strongest_posterior_dimension']
        rr=[r for r in curves if r['method']==label and int(r['dimension'])==dim and r['metric']=='prior_js']
        xx=np.array([int(r['horizon']) for r in rr])
        yy=np.array([float(r['mean']) for r in rr])
        axs[0,1].plot(xx,yy,color=color,lw=1.2,label=f'{LABELS[label]} (c{dim})')
        axs[0,1].fill_between(xx,[float(r['ci_low']) for r in rr],[float(r['ci_high']) for r in rr],color=color,alpha=.12,lw=0)
    axs[0,0].set(xlabel='Perturbed context dimension',ylabel=r'$D_{KL}(q \,\Vert\, q_{cf})$, nats',xticks=range(1,9),
                 title='(a)  Context changes posterior inference')
    axs[0,0].legend(fontsize=7,loc='upper right')
    axs[0,1].set(xlabel='Imagination horizon (steps)',ylabel='Prior JS divergence, nats',
                 title='(b)  Effect propagates into future priors',xlim=(1,50))
    axs[0,1].legend(fontsize=6.5)
    for ax,label,metric,title,ylabel in [
        (axs[1,0],'r2dreamer_context','reward','(c)  R2Dreamer: native reward prediction','Predicted reward'),
        (axs[1,1],'dreamerv3_context','speed_kmh','(d)  DreamerV3: native speed prediction','Predicted speed (km/h)')]:
        rr=[r for r in examples if r['method']==label and r['example']=='median_nonzero']
        for branch,color,ls in [('Original','#747D8D','-'),('+1 SD context',COLORS[label],'--')]:
            selected=[r for r in rr if r['branch']==branch]
            ax.plot([int(r['horizon']) for r in selected],[float(r[metric]) for r in selected],
                    color=color,ls=ls,lw=1.2,label=branch)
        ax.set(xlabel='Imagination horizon (steps)',ylabel=ylabel,title=title,xlim=(1,50))
        ax.legend(fontsize=7)
    for ax in axs.flat:
        title=ax.get_title();ax.set_title('')
        ax.set_title(title,loc='left',fontsize=8.5,pad=9)
        ax.grid(axis='y',color='#ECEEF0',lw=.6)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
    export(fig,out/'context_counterfactual_native')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for ax,(label,model) in zip(axs,summary.items()):
        rows=model['dimensions'];y=np.array([r['auc'] for r in rows])
        lo=np.array([r['auc_ci_low'] for r in rows]);hi=np.array([r['auc_ci_high'] for r in rows])
        ax.errorbar(range(1,9),y,yerr=[y-lo,hi-y],fmt='o-',ms=3,lw=1,color=COLORS[label],capsize=2)
        ax.axhline(.5,color='#858A90',lw=.8,ls='--',label='Chance')
        ax.set(xlabel='Perturbed context dimension',ylabel='Original vs. perturbed AUC',
               title=LABELS[label],xticks=range(1,9),ylim=(.4,.8))
        ax.set_title('');ax.set_title(LABELS[label],loc='left',fontsize=8.5,pad=9)
        ax.grid(axis='y',color='#ECEEF0',lw=.6)
        ax.legend(fontsize=7)
    export(fig,out/'context_counterfactual_auc')
    print(out)


if __name__=='__main__':main()
