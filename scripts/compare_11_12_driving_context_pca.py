"""Paired driving audit and identical-history vehicle-type PCA for runs 11 and 12."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analyze_swap_context_associations import load_context
from analyze_natural_vehicle_context import classify
from compare_fixed_vehicle_random_context import association
from plot_retrained_vehicle_context_pca import fit_pca

TYPES=['default','m','l','s','xl']

def main():
 root=Path(__file__).resolve().parents[1]
 old=root/'logdir/tintersection_mass_friction_500k_20261021'
 new=root/'logdir/tintersection_shallow_wmgrad05_500k_20261022'
 out=new/'comparison_11_12_driving_context';out.mkdir(parents=True,exist_ok=False)
 contract={'question':'Does WM-to-context feedback improve driving and vehicle-type organization compared with the same Shallow architecture?',
 'evidence':['100 paired driving scenarios per split','Both encoders receive identical common natural histories','Fixed initial speed/physical wheel angle, nominal and randomized mass/friction histories','Full8D association and grouped held-out softmax vehicle probes'],
 'archetype':'quantitative grid','backend':'python','exports':['png','pdf','svg','csv','npz','json','markdown'],
 'risks':['One training seed, different validation-selected best steps; no causal or multi-seed claim','Independent standardized PCA bases; coordinates across encoders cannot be compared','Common natural histories collected by baseline and11, not12','S/XL are supervised probe labels, so probe accuracy is not zero-shot classification','Fixed physical angle entails type-specific normalized steering; initial speed fixed, later speed allowed to diverge','PCA/eta describe associations, not pure parameter identification or WM utilization']}
 (out/'figure_contract.json').write_text(json.dumps(contract,indent=2))
 runs={'11':old/'training/r2dreamer_dali_shallow','12':new/'training/r2dreamer_dali_shallow'}
 sources={'11':old/'training/source','12':new/'training/source'}
 driving={};rng=np.random.default_rng(20261022);inds=rng.integers(0,100,(10000,100))
 for split in ['new_maps','s','xl']:
  rows={};driving[split]={'summaries':{},'provenance':{}}
  for name,p in runs.items():
   d=p/'evaluation'/split;s=json.loads((d/'summary.json').read_text());assert s['status']=='completed'
   rows[name]=sorted([json.loads(s)for s in(d/'episodes.jsonl').read_text().splitlines()],key=lambda r:r['seed'])
   assert len(rows[name])==len({r['seed']for r in rows[name]})==100
   meta=json.loads((d/'provenance.json').read_text())
   assert meta['checkpoint_sha256']==hashlib.sha256((p/'best.pt').read_bytes()).hexdigest()
   driving[split]['summaries'][name]=s['summary'];driving[split]['provenance'][name]=meta
  identity=lambda r:(r['seed'],*[r['chassis'][k]for k in ['model','mass','max_steering','wheel_friction']])
  assert [identity(r)for r in rows['11']]==[identity(r)for r in rows['12']]
  driving[split]['paired_12_minus_11']={}
  for k in ['success','crash','out_of_road','reward','route_completion','mean_speed_km_h','steps']:
   delta=np.array([float(b[k])-float(a[k])for a,b in zip(rows['11'],rows['12'])])
   driving[split]['paired_12_minus_11'][k]={'mean':float(delta.mean()),'paired_bootstrap95':np.quantile(delta[inds].mean(1),[.025,.975]).tolist()}
  driving[split]['success_pairs']={'both':sum(a['success']and b['success']for a,b in zip(rows['11'],rows['12'])),'11_only':sum(a['success']and not b['success']for a,b in zip(rows['11'],rows['12'])),'12_only':sum(not a['success']and b['success']for a,b in zip(rows['11'],rows['12']))}
 corpus=old/'world_model_common_corpus';episodes=json.loads((corpus/'episodes.json').read_text());histories=[];natural_rows=[]
 for ei,e in enumerate(episodes):
  d=dict(np.load(corpus/'episodes'/e['file']));candidate=[]
  for t in range(5,len(d['state']),5):
   sl=slice(t-4,t+1)
   if d['ctx_mask'][t].all()and not any(d[k][sl].any()for k in ['is_first','is_last','is_terminal','ctx_collision']):candidate.append(t)
  ids=np.linspace(0,len(candidate)-1,min(30,len(candidate)),dtype=int)
  for index in ids:
   t=candidate[index];h=d['ctx_history'][t].copy();sl=slice(t-4,t+1)
   assert h.shape==(5,6)
   np.testing.assert_allclose(h[:,:4],d['ctx_ego_state'][sl],atol=1e-6)
   np.testing.assert_allclose(h[:,4:],d['ctx_prev_action'][sl],atol=1e-6)
   histories.append(h);natural_rows.append({'vehicle':e['model'],'episode':ei,'map_id':e['map_id'],'collector':e['policy'],'t':t})
 natural=np.array(histories,np.float32)
 fixed_dir=old/'context_pca_fixed10mps15deg_validated'
 fixed_rows=list(csv.DictReader((fixed_dir/'source_data.csv').open()));fixed=np.load(fixed_dir/'representations.npz')['history']
 sets={'natural':(natural,natural_rows)}
 for regime in ['nominal','randomized']:
  select=np.array([r['regime']==regime for r in fixed_rows]);sets[regime]=(fixed[select],[r for r,keep in zip(fixed_rows,select)if keep])
 all_h=np.concatenate([v[0]for v in sets.values()]);contexts={};metadata={};torch.set_num_threads(1)
 for name in runs:
  _,contexts[name],metadata[name]=load_context(sources[name],runs[name]/'best.pt',all_h,'11_12_'+name)
 np.testing.assert_allclose(contexts['11'][len(natural):],np.load(fixed_dir/'representations.npz')['context'],atol=1e-6)
 result={'driving':driving,'checkpoints':metadata,'representations':{},'limits':contract['risks']};coords={};save={'history':all_h};offset=0;csvrows=[]
 folds=np.array_split(np.random.default_rng(20261016).permutation(24),6)
 for setting,(history,rows)in sets.items():
  n=len(history);sl=slice(offset,offset+n);y=np.array([TYPES.index(r['vehicle'])for r in rows])
  groups=np.array([int(r['map_id']if setting=='natural'else r['profile'])for r in rows])
  eps=np.array([int(r['episode'])if setting=='natural'else i for i,r in enumerate(rows)])
  folds_here=[np.array([g])for g in np.unique(groups)]if setting=='natural'else folds
  fit=~np.isin(groups,folds_here[0]);sp={name:contexts[name][sl]for name in contexts}
  sp.update(raw_history=history.reshape(n,-1),speed_angle_history=history[...,2:4].reshape(n,-1))
  metrics={};predictions={}
  for name,x in sp.items():
   pred=np.full(n,-1)
   for held in folds_here:
    test=np.isin(groups,held);pred[test]=classify(x,y,~test,test,eps,'softmax')
   assert (pred>=0).all();predictions[name]=pred
   acc_by_group=np.array([np.mean([(pred[eps==e]==y[eps==e]).mean()for e in np.unique(eps[groups==g])])for g in np.unique(groups)])
   metrics[name]={'eta_squared':association(x,y,fit),'group_balanced_accuracy':float(acc_by_group.mean()),'accuracy_by_group':acc_by_group.tolist(),'window_accuracy':float((pred==y).mean())}
   if name in contexts:
    coords[(setting,name)],ratio,mu,sd,axes=fit_pca(x,fit);metrics[name]['pca_variance_ratio']=ratio.tolist()
    save[f'{setting}_{name}_pc']=coords[(setting,name)];save[f'{setting}_{name}_context']=x
    save[f'{setting}_{name}_pca_mean']=mu;save[f'{setting}_{name}_pca_sd']=sd;save[f'{setting}_{name}_pca_axes']=axes
   print(setting,name,metrics[name]['eta_squared'],metrics[name]['group_balanced_accuracy'],flush=True)
  n_groups=len(metrics['11']['accuracy_by_group']);sample=np.random.default_rng(20261022).integers(0,n_groups,(10000,n_groups))
  delta=np.array(metrics['12']['accuracy_by_group'])-metrics['11']['accuracy_by_group']
  result['representations'][setting]={'n_windows':n,'n_groups':n_groups,'pca_fit_windows':int(fit.sum()),'metrics':metrics,'accuracy_difference_12_minus_11':{'mean':float(delta.mean()),'group_bootstrap95':np.quantile(delta[sample].mean(1),[.025,.975]).tolist()},'protocol':'leave-one-map-out, episode balanced'if setting=='natural'else 'six-fold shared throttle-profile holdout; fixed initial speed10m/s and physical wheel angle15deg; nominal or paired randomized mass/friction'}
  for i,r in enumerate(rows):
   row=dict(setting=setting,vehicle=r['vehicle'],group=int(groups[i]),episode=int(eps[i]),pca_fit=bool(fit[i]))
   for name in contexts:
    row.update({f'run{name}_context{k}':float(contexts[name][offset+i,k])for k in range(8)})
    row.update({f'run{name}_pc{k+1}':float(coords[(setting,name)][i,k])for k in range(2)})
    row[f'run{name}_predicted_vehicle']=TYPES[predictions[name][i]]
   csvrows.append(row)
  save[f'{setting}_labels']=y;save[f'{setting}_groups']=groups;offset+=n
 np.savez_compressed(out/'representations.npz',**save)
 with(out/'source_data.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 (out/'comparison.json').write_text(json.dumps(result,indent=2))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
 colors=['#4C78A8','#59A14F','#F28E2B','#E15759','#B279A2'];markers=['o','s','D','^','v']
 # Main grid: rows are encoders, columns are operating-condition protocols.
 fig,axs=plt.subplots(2,3,figsize=(11.6,7.1),layout='constrained')
 for col,setting in enumerate(sets):
  history,rows=sets[setting];y=np.array([TYPES.index(r['vehicle'])for r in rows]);order=np.random.default_rng(11).permutation(len(y))
  for row,name in enumerate(contexts):
   ax=axs[row,col];xy=coords[(setting,name)];metrics=result['representations'][setting]['metrics'][name];ratio=metrics['pca_variance_ratio']
   for k in range(5):
    ix=order[y[order]==k];ax.scatter(*xy[ix].T,color=colors[k],marker=markers[k],s=10 if setting=='natural'else 26,alpha=.4 if setting=='natural'else .7,edgecolors='none',rasterized=setting=='natural')
   ax.set(xlabel=f'PC1 ({ratio[0]:.1%})',ylabel=f'PC2 ({ratio[1]:.1%})')
   title={'natural':'Natural T-intersection histories','nominal':'Fixed initial conditions | nominal','randomized':'Fixed initial conditions | randomized'}[setting]
   ax.set_title(f'{title}\nRun {name} | type $\eta^2$={metrics["eta_squared"]:.3f} | probe={metrics["group_balanced_accuracy"]:.1%}',fontsize=8)
   ax.text(-.13,1.09,'abcdef'[row*3+col],transform=ax.transAxes,weight='bold',fontsize=10)
   ax.grid(alpha=.13);ax.set_axisbelow(True)
 handles=[Line2D([],[],color=colors[k],marker=markers[k],linestyle='none',markersize=5,label=t.upper())for k,t in enumerate(TYPES)]
 fig.legend(handles=handles,loc='outside lower center',ncol=5)
 fig.suptitle(f'Vehicle-type context PCA | 11 vs 12 (WM gradient scale 0.5)\nIdentical input histories per encoder; independent standardized PCA bases; best checkpoints 120K / 180K',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'vehicle_context_pca_11_vs_12.{ext}',dpi=240)
 plt.close(fig)
 # Fixed nominal two-panel version matches the earlier user-requested figure.
 fig,axs=plt.subplots(1,2,figsize=(9,4.6),layout='constrained');rows=sets['nominal'][1];y=np.array([TYPES.index(r['vehicle'])for r in rows])
 for ax,name,letter in zip(axs,contexts,'ab'):
  xy=coords[('nominal',name)];m=result['representations']['nominal']['metrics'][name];vr=m['pca_variance_ratio']
  for k in range(5):ax.scatter(*xy[y==k].T,color=colors[k],marker=markers[k],s=34,alpha=.72,edgecolors='none')
  ax.set(title=f'Run {name}: '+('no WM feedback'if name=='11'else 'WM feedback scale 0.5')+f'\nType $\eta^2$={m["eta_squared"]:.3f}; held-out probe={m["group_balanced_accuracy"]:.1%}',xlabel=f'PC1 ({vr[0]:.1%})',ylabel=f'PC2 ({vr[1]:.1%})')
  ax.text(-.12,1.07,letter,transform=ax.transAxes,weight='bold',fontsize=11);ax.grid(alpha=.13);ax.set_axisbelow(True)
 fig.legend(handles=handles,loc='outside lower center',ncol=5)
 fig.suptitle('Identical 120 histories | initial speed 10 m/s; front-wheel angle 15°\nNominal mass/friction; 24 shared throttle profiles per vehicle type',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(out/f'fixed_nominal_vehicle_pca_11_vs_12.{ext}',dpi=240)
 plt.close(fig)
 lines=['# 11 vs 12：驾驶表现与车型context PCA','','## 驾驶表现','','相同100个地图种子及实际车型、质量、最大前轮角、wheel_friction逐回合配对核验；检查点SHA256核验。best由各自训练验证选择：11为120007，12为180002。','','| 测试集 | 方法 | 成功率 | 碰撞率 | 出界率 | 奖励 | 路线完成率 | 平均速度km/h | 平均步数 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for split,d in driving.items():
  for name,s in d['summaries'].items():lines.append(f'| {split} | {name} | {s["success"]:.0%} | {s["crash"]:.0%} | {s["out_of_road"]:.0%} | {s["reward"]:.2f} | {s["route_completion"]:.1%} | {s["mean_speed_km_h"]:.2f} | {s["steps"]:.1f} |')
 lines+=['','成功率差值（12−11，配对场景bootstrap95%，仅反映场景抽样而非训练种子不确定性）：']
 for split,d in driving.items():
  v=d['paired_12_minus_11']['success'];lo,hi=v['paired_bootstrap95'];lines.append(f'- {split}: {v["mean"]*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}]；配对成功计数 {d["success_pairs"]}。')
 lines+=['','## PCA和完整8维context量化','','两个编码器接收完全相同的每一个历史窗口。PCA标准化和基底分别拟合：natural用9张地图、fixed用20种profile，其余样本仅投影；各图显示全部样本。坐标轴不能跨模型直接比较。eta²是标准化8维context中车型间均值差异占总变异的比例，并非“车型信息百分比”。分类为带L2正则的线性softmax，标准化在每个训练fold内拟合。natural按episode与地图平衡；fixed按profile分组6折。','','| 条件 | 窗口数 | 11 eta² | 12 eta² | 11车型probe | 12车型probe |','|---|---:|---:|---:|---:|---:|']
 for setting,d in result['representations'].items():
  m=d['metrics'];lines.append(f'| {setting} | {d["n_windows"]} | {m["11"]["eta_squared"]:.4f} | {m["12"]["eta_squared"]:.4f} | {m["11"]["group_balanced_accuracy"]:.2%} | {m["12"]["group_balanced_accuracy"]:.2%} |')
 lines+=['','对照：输入本身包含车型线索。raw history / speed-angle history的分类准确率：']
 for setting,d in result['representations'].items():lines.append(f'- {setting}: raw={d["metrics"]["raw_history"]["group_balanced_accuracy"]:.2%}, speed/angle={d["metrics"]["speed_angle_history"]["group_balanced_accuracy"]:.2%}。')
 lines+=['','自然历史来自前次公共语料，baseline和11各50回合，5车型×10地图×2采集策略，非12自身策略轨迹。不重新采集以避免策略分布差异。固定条件使用同一批已验证物理响应，初始速度10m/s、实际前轮角15度；速度在后续5步内自然变化，固定前轮角需要车型相关归一化转向输入。','','S/XL在训练世界模型时未见，但有标签probe训练包含这两类；上述准确率不是零样本分类。单训练seed、不同best步数，结果是描述性比较。车型分簇不能证明context只编码底盘，也不能证明世界模型正确利用context。当前12的reward/continue NLL异常与本PCA分析是不同问题。','']
 (out/'REPORT.md').write_text('\n'.join(lines));print(out,flush=True)
if __name__=='__main__':main()
