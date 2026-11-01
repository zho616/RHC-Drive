"""Booktabs-style T-intersection results across two independent training seeds.

Figure contract: compare measured reward, success, route completion, speed and
collision across methods separately for ID maps and S/XL OOD. Mean and sample
SD (ddof=1) are computed across seed-level test means, not across episodes.
Python-only static table plate; editable PDF/SVG, PNG450dpi, CSV and LaTeX.
No metrics or seed1 values are imputed for untrained methods.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS=['ppo','sac','dreamerv3','r2dreamer','r2dreamer_context']
LABELS=dict(ppo='PPO',sac='SAC',dreamerv3='DreamerV3',r2dreamer='R2Dreamer',
            r2dreamer_context='R2Dreamer+Context (Ours)')
ALIASES={'PPO':'ppo','SAC':'sac','DreamerV3':'dreamerv3','R2Dreamer':'r2dreamer',
         'R2Dreamer+context':'r2dreamer_context'}
SPLITS={'new_maps':'Unseen maps (ID vehicles)','s':'Vehicle S (OOD)','xl':'Vehicle XL (OOD)'}
METRICS=[('reward','RW ↑',1,'max'),('success','SR (%) ↑',100,'max'),
         ('route_completion','RC (%) ↑',100,'max'),
         ('mean_speed_km_h','AS (km/h)',1,None),('crash','CR (%) ↓',100,'min')]
RAW_METRICS=['reward','success','route_completion','mean_speed_km_h','crash','out_of_road','timeout','steps']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_records(inputs):
    records={};sources={};cohorts={}
    for expected_seed,path in enumerate(inputs):
        sources[str(path.resolve())]=sha(path)
        for row in csv.DictReader(path.open(encoding='utf-8-sig')):
            method=ALIASES.get(row['method'],row['method'])
            if method not in METHODS:continue
            seed=int(row['training_seed']);split=row['split']
            assert seed==expected_seed and int(row['episodes'])==100
            summary_path=Path(row['summary_source']);folder=summary_path.parent
            summary=json.loads(summary_path.read_text())['summary']
            episodes_path=folder/'episodes.jsonl'
            episodes=[json.loads(x) for x in episodes_path.read_text().splitlines()]
            assert len(episodes)==100
            normalized={k:float(row[k]) for k in RAW_METRICS}
            for k,v in normalized.items():
                assert np.isclose(v,summary[k],rtol=1e-10,atol=1e-10)
                assert np.isclose(v,np.mean([float(e[k]) for e in episodes]),rtol=1e-10,atol=1e-10)
            cohort={e['seed']:{k:e['chassis'][k] for k in
                ('model','mass_scale','steering_scale','friction_scale','mass','wheel_friction')} for e in episodes}
            if split in cohorts:assert cohorts[split]==cohort,(seed,method,split)
            else:cohorts[split]=cohort
            key=(split,method,seed);assert key not in records
            records[key]=dict(method=method,split=split,training_seed=seed,episodes=100,
                **normalized,checkpoint_sha256=row['checkpoint_sha256'],
                summary_source=str(summary_path),summary_sha256=sha(summary_path),
                episodes_source=str(episodes_path),episodes_sha256=sha(episodes_path))
    assert len(records)==30
    return records,sources


def aggregate(records):
    stats={};rows=[]
    for split in SPLITS:
        for method in METHODS:
            row=dict(method=method,split=split,n_training_seeds=2,episodes_per_seed=100)
            for metric in RAW_METRICS:
                values=[records[(split,method,seed)][metric] for seed in (0,1)]
                mean=statistics.mean(values);sd=statistics.stdev(values)
                row[metric+'_mean']=mean;row[metric+'_sample_sd']=sd
                stats[(split,method,metric)]=(mean,sd)
            rows.append(row)
    return stats,rows


def is_best(stats,split,method,metric,direction):
    if direction is None:return False
    best=(min if direction=='min' else max)(stats[(split,m,metric)][0] for m in METHODS)
    return np.isclose(stats[(split,method,metric)][0],best,rtol=0,atol=1e-12)


def formatted(stats,split,method,metric,scale):
    mean,sd=stats[(split,method,metric)]
    return f'{mean*scale:.2f} ± {sd*scale:.2f}'


def table_panel(ax,stats,split,letter=None):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    title=(f'({letter}) ' if letter else '')+SPLITS[split]
    ax.text(.008,.99,title,ha='left',va='top',fontsize=11.6,fontweight='bold')
    x=[.008,.374,.519,.658,.801,.938]
    ax.hlines([.85,.70,.035],xmin=.003,xmax=.997,colors='black',linewidths=[1.05,.65,1.05])
    ax.text(x[0],.773,'Method',ha='left',va='center',fontsize=10.8,fontweight='bold')
    for xx,(_,title,_,_) in zip(x[1:],METRICS):
        ax.text(xx,.773,title,ha='center',va='center',fontsize=10.8,fontweight='bold')
    for i,method in enumerate(METHODS):
        y=.635-i*.126
        ax.text(x[0],y,LABELS[method],ha='left',va='center',fontsize=10.55,
                fontweight='bold' if method=='r2dreamer_context' else 'normal')
        for xx,(metric,_,scale,direction) in zip(x[1:],METRICS):
            ax.text(xx,y,formatted(stats,split,method,metric,scale),ha='center',va='center',fontsize=10.5,
                fontweight='bold' if is_best(stats,split,method,metric,direction) else 'normal')


def render(stats,out):
    plt.rcParams.update({'font.family':'STIXGeneral','pdf.fonttype':42,'svg.fonttype':'none',
                         'mathtext.fontset':'stix'})
    fig,axes=plt.subplots(3,1,figsize=(9.7,5.65))
    fig.subplots_adjust(left=.018,right=.986,top=.985,bottom=.102,hspace=.10)
    for letter,(split,ax) in zip('abc',zip(SPLITS,axes)):table_panel(ax,stats,split,letter)
    fig.text(.025,.061,'Mean ± sample SD across training seeds 0 and 1 (n = 2); 100 test episodes per seed and condition.',
             ha='left',fontsize=9.1)
    fig.text(.025,.027,'RW: reward; SR: success rate; RC: route completion; AS: average speed; CR: collision rate. Bold: best mean.',
             ha='left',fontsize=9.1)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'tintersection_seed0_seed1_results.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    for split in SPLITS:
        fig,ax=plt.subplots(figsize=(9.7,1.88));fig.subplots_adjust(left=.018,right=.986,top=.98,bottom=.12)
        table_panel(ax,stats,split)
        for ext in ('png','pdf','svg'):fig.savefig(out/f'tintersection_{split}_seed0_seed1.{ext}',dpi=450,facecolor='white')
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed0',type=Path,default=Path('iv_output/table/tintersection_test_results.csv'))
    p.add_argument('--seed1',type=Path,default=Path('iv_output/table/tintersection_seed1_test_results.csv'))
    p.add_argument('--table-root',type=Path,default=Path('iv_output/table'))
    p.add_argument('--figure-dir',type=Path,default=Path('iv_output/fig/tintersection_two_seed_results'))
    a=p.parse_args();a.figure_dir.mkdir(parents=True,exist_ok=True)
    archive=a.table_root/'source_data/tintersection_seed0_seed1_results';archive.mkdir(parents=True,exist_ok=True)
    records,sources=load_records([a.seed0,a.seed1]);stats,rows=aggregate(records)
    dump_csv(a.table_root/'tintersection_seed0_seed1_results.csv',rows)
    dump_csv(archive/'seed_level_results.csv',list(records.values()))
    markdown='T-intersection：seed0 + seed1\n\n'
    markdown+='均值 ± 样本标准差（ddof=1），跨两个训练种子计算。每个种子、每个测试条件100回合。加粗为同条件最优均值；AS仅作速度报告，不作为越大越优的排名项。\n\n'
    latex=[]
    for split,title in SPLITS.items():
        markdown+=f'**{title}**\n\n| Method | RW ↑ | SR (%) ↑ | RC (%) ↑ | AS (km/h) | CR (%) ↓ |\n|---|---:|---:|---:|---:|---:|\n'
        tex=[r'\begin{table*}[t]',r'\centering',
            r'\caption{T-intersection: '+title+r'. Mean $\pm$ sample SD across two training seeds (0 and 1), 100 test episodes per seed.}',
            r'\begin{tabular}{lccccc}',r'\toprule',
            r'Method & RW $\uparrow$ & SR (\%) $\uparrow$ & RC (\%) $\uparrow$ & AS (km/h) & CR (\%) $\downarrow$ \\',r'\midrule']
        for method in METHODS:
            md_cells=[];tex_cells=[]
            for metric,_,scale,direction in METRICS:
                v=formatted(stats,split,method,metric,scale);best=is_best(stats,split,method,metric,direction)
                md_cells.append('**'+v+'**' if best else v)
                v=v.replace('±',r'$\pm$');tex_cells.append(r'\textbf{'+v+'}' if best else v)
            name=LABELS[method]
            markdown+='| '+name+' | '+' | '.join(md_cells)+' |\n'
            tex.append((r'\textbf{'+name+'}' if method=='r2dreamer_context' else name)+' & '+' & '.join(tex_cells)+r' \\')
        markdown+='\n'
        tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}']);latex.append('\n'.join(tex))
    markdown+='RW：平均回合奖励；SR：成功率；RC：路线完成率；AS：平均速度；CR：碰撞率。百分比指标的标准差单位是百分点。\n\n'
    markdown+='DreamerV3+Context没有seed1，不纳入两种子汇总。原seed0和seed1结果均保留。两个训练种子仍不足以据此声称统计显著性。\n'
    (a.table_root/'tintersection_seed0_seed1_results.md').write_text(markdown)
    (a.table_root/'tintersection_seed0_seed1_results.tex').write_text('\n\n'.join(latex)+'\n')
    render(stats,a.figure_dir)
    provenance=dict(training_seeds=[0,1],n_training_seeds=2,episodes_per_seed_per_condition=100,
        sample_sd_ddof=1,aggregation='Mean and sample SD of seed-level test means; no episode pooling for the SD.',
        method_scope=METHODS,conditions=SPLITS,inputs_sha256=sources,
        raw_episode_means_verified=True,paired_test_maps_and_chassis_verified=True,total_test_episodes=3000,
        image_contract='Three booktabs-style panels, no vertical rules or shading. Actual measured metrics only.',
        speed_ranking='Average speed reported without a best-mean ranking.',
        limitation='Only two independent training seeds; SD is not a confidence interval or standard error.')
    (archive/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    shutil.copy2(__file__,archive/Path(__file__).name)
    for suffix in ('csv','md','tex'):shutil.copy2(a.table_root/f'tintersection_seed0_seed1_results.{suffix}',archive/f'tintersection_seed0_seed1_results.{suffix}')
    assert all(sha(path)==fingerprint for path,fingerprint in sources.items())
    hashes={str(f.resolve()):sha(f) for folder in (archive,a.figure_dir) for f in folder.rglob('*') if f.is_file() and f.name!='sha256.json'}
    (archive/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(markdown)


if __name__=='__main__':main()
