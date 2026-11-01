"""Publication panels for native common-corpus metrics and separate training logs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

# Figure contract: native metrics do not support improved world-model alignment
# or reconstruction after Context. Paired equal-episode evaluation is primary;
# own-policy training losses are a separate diagnostic, never pooled with it.
# Quantitative grid, Python backend, editable PDF/SVG plus PNG and source CSV.
METHODS=['dreamerv3','r2dreamer','dreamerv3_context','r2dreamer_context']
LABELS=['DreamerV3','R2Dreamer','DreamerV3\n+ Context','R2Dreamer\n+ Context']
COLORS=['#747D8D','#194F6A','#9441BD','#ED3943']


def export(fig,path):
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'figures';out.mkdir(exist_ok=True)
    data=json.loads((a.root/'comparison.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                         'xtick.major.size':3,'ytick.major.size':3})
    panels=[('kl_prior_post_raw','Prior–posterior KL','KL(p || q), nats'),
            ('state_native_recon_loss','Native state reconstruction loss','Summed symlog squared error'),
            ('state_psnr_from_mse','State reconstruction PSNR','PSNR, dB (peak = 1)'),
            ('state_mae','State reconstruction MAE','MAE, normalized state units')]
    for group in ['all','ood_s_xl']:
        fig,axs=plt.subplots(2,2,figsize=(8.1,5.35),layout='constrained')
        for i,(ax,(key,title,ylabel)) in enumerate(zip(axs.flat,panels)):
            entries=[data['summary'][group][m][key] for m in METHODS]
            ymax=max(v['ci95'][1] for v in entries if v is not None)*1.23
            for x,(entry,color) in enumerate(zip(entries,COLORS)):
                if entry is None:
                    ax.text(x,.04,'N/A',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8,color='#858A90')
                    continue
                mean=entry['mean'];lo,hi=entry['ci95']
                ax.bar(x,mean,width=.58,color=color,alpha=.9,zorder=3)
                ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='none',ecolor='#41464B',capsize=3,lw=.8,zorder=4)
                ax.text(x,hi+.025*ymax,f'{mean:.2f}' if key in ['kl_prior_post_raw','state_psnr_from_mse'] else f'{mean:.4f}',
                        ha='center',va='bottom',fontsize=7.5,color='#30353A')
            ax.set_title(f'({chr(97+i)})  {title}',fontsize=9,loc='left',pad=9)
            ax.set_ylabel(ylabel,fontsize=8);ax.set_xticks(range(4),LABELS,fontsize=7)
            ax.set_ylim(0,ymax);ax.set_xlim(-.6,3.6);ax.grid(axis='y',color='#ECEEF0',lw=.6,zorder=0)
            ax.yaxis.set_major_locator(MaxNLocator(5))
        export(fig,out/f'native_world_model_{group}')
    with (a.root/'training_curves.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axs=plt.subplots(2,2,figsize=(8.1,5.2),layout='constrained')
    metrics=[('train/loss/dyn','Training KL loss (free-nats clipped)'),
             ('train/loss/state','Native state reconstruction loss'),
             ('train/loss/rew','Reward prediction loss'),('train/loss/con','Continuation prediction loss')]
    handles=[]
    for i,(ax,(metric,title)) in enumerate(zip(axs.flat,metrics)):
        for m,label,color in zip(METHODS,LABELS,COLORS):
            rr=[r for r in rows if r['method']==m and r['metric']==metric]
            if not rr:continue
            x=np.array([float(r['step']) for r in rr]);y=np.array([float(r['value']) for r in rr])
            # TensorBoard-style bias-corrected EMA, with the user's 0.99 setting.
            numerator=0.;denominator=0.;smooth=[]
            for v in y:
                numerator=.99*numerator+.01*v;denominator=.99*denominator+.01;smooth.append(numerator/denominator)
            ax.plot(x,y,color=color,lw=.45,alpha=.18)
            line,=ax.plot(x,smooth,color=color,lw=1.25,ls='--' if m=='dreamerv3_context' else '-',label=label.replace('\n',' '))
            if i==0:handles.append(line)
        ax.set_title(f'({chr(97+i)})  {title}',fontsize=9,loc='left',pad=8)
        ax.set_xlabel('Environment steps');ax.set_xlim(0,500000);ax.set_ylabel('Native loss')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}k'))
        ax.grid(color='#ECEEF0',lw=.6);ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.legend(handles,[h.get_label() for h in handles],loc='outside lower center',ncol=4,frameon=False,fontsize=7)
    export(fig,out/'training_world_model_losses_ema099')
    print(out)


if __name__=='__main__':main()
