#!/usr/bin/env python3
"""Native chassis gallery with highlighted ID/OOD groups and white gutters."""
import json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures/chassis_gallery_verified_20261023'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42})
(OUT/'figure_contract.json').write_text(json.dumps(dict(conclusion='Identify the training (ID) and held-out (OOD) vehicle types.',backend='Python',archetype='image plate',layout='One left plus four right; gray ID group and blue OOD group; white image borders',processing='Same crop: rows70:745 columns180:1080. All five vehicles captured after suspension stabilization; all four wheels checked for ground contact.',size_mm=[183,110]),indent=2))
fig=plt.figure(figsize=(183/25.4,110/25.4),facecolor='white')
for x,w,color in [(.012,.642,'#DDDDDD'),(.676,.312,'#C5D5E3')]:
 fig.add_artist(FancyBboxPatch((x,.02),w,.96,boxstyle='round,pad=0.003,rounding_size=0.028',facecolor=color,edgecolor='none',transform=fig.transFigure,zorder=0))
fig.text(.333,.935,'ID · Training vehicle types',ha='center',fontsize=10,weight='bold')
fig.text(.832,.935,'OOD · Unseen types',ha='center',fontsize=9.3,weight='bold')
w=.285;h=w*183/110*675/900
slots=[('default','(a) Default',.03,.34),('m','(b) M',.351,.53),('s','(c) S',.69,.53),('l','(d) L',.351,.10),('xl','(e) XL',.69,.10)]
for model,title,x,y in slots:
 ax=fig.add_axes([x,y,w,h]);im=plt.imread(OUT/'raw'/f'{model}.png');ax.imshow(im[70:745,180:1080]);ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_color('white');spine.set_linewidth(3)
 fig.text(x+w/2,y-.036,title,ha='center',fontsize=10)
for ext in ['png','pdf','svg','tiff']:
 opts={'dpi':600}if ext in ['png','tiff']else{}
 if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
 fig.savefig(OUT/f'chassis_id_ood.{ext}',**opts)
fig.savefig(OUT/'chassis_id_ood_preview.png',dpi=240)
(OUT/'CAPTION.md').write_text('Native MetaDrive chassis: Default, M and L form the training/ID vehicle-type pool (gray); S and XL are unseen OOD vehicle types (blue). All views use native simulator assets with a common illustrative map and relative camera setup. Each vehicle is captured while stationary with the brake applied after at least 100 settling steps, with less than 2 mm position variation over the final 20 samples. All four wheel meshes and physics wheels are present; all four wheel raycasts report contact. No chassis geometry or physics parameters were altered. Identical image crops are applied; no color adjustments.\n')
shutil.copy2(__file__,OUT/Path(__file__).name)
shutil.copy2(ROOT/'scripts/render_chassis_gallery_assets.py',OUT/'render_chassis_gallery_assets.py')
