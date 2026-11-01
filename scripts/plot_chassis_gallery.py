#!/usr/bin/env python3
"""Five native simulator views in a staggered one-plus-four layout."""
import json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'output/figures/chassis_randomization_20261023'
OUT=ROOT/'output/figures/chassis_gallery_20261023'
OUT.mkdir(parents=True,exist_ok=True)
contract=dict(conclusion='Show the five native chassis used in the experiments.',archetype='image plate',backend='Python/matplotlib',layout='Default left centered; M/S upper right; L/XL lower right',processing='Identical crop of native render: rows70:745, columns180:1080; no color adjustments',text='Panel letters and model names only',size_mm=[183,96])
(OUT/'figure_contract.json').write_text(json.dumps(contract,indent=2))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(183/25.4,96/25.4),facecolor='white')
w=.32;h=w*183/96*675/900
slots=[('default','(a) Default',.02,.5-h/2),('m','(b) M',.34,.5),('s','(c) S',.66,.5),('l','(d) L',.34,.5-h),('xl','(e) XL',.66,.5-h)]
for model,title,x,y in slots:
    ax=fig.add_axes([x,y,w,h]);im=plt.imread(SRC/'raw'/f'{model}.png')
    ax.imshow(im[70:745,180:1080]);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('black');spine.set_linewidth(1.1)
    ax.text(.5,.94,title,transform=ax.transAxes,ha='center',va='top',color='white',fontsize=10,weight='bold',path_effects=[pe.withStroke(linewidth=1,foreground='#444444')])
for ext in ['png','pdf','svg','tiff']:
    opts={'dpi':600}if ext in ['png','tiff']else{}
    if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(OUT/f'chassis_gallery.{ext}',**opts)
fig.savefig(OUT/'chassis_gallery_preview.png',dpi=240)
(OUT/'CAPTION.md').write_text('Native MetaDrive vehicle models used in the experiments: (a) Default, (b) M, (c) S, (d) L and (e) XL. Default, M and L form the training pool; S and XL are held out for unseen-type evaluation. The images use the same illustrative map and relative camera setup. An identical crop is applied to each native simulator render; no color adjustments are applied.\n')
shutil.copy2(SRC/'source_data.json',OUT/'source_data.json')
shutil.copytree(SRC/'raw',OUT/'raw',dirs_exist_ok=True)
shutil.copy2(__file__,OUT/Path(__file__).name)
print(OUT)
