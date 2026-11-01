"""Verify and summarize paired best-checkpoint DreamerV3/Shallow driving tests."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 queue=json.loads((root/'status.json').read_text());assert queue['phase']=='completed'
 names=['dreamerv3','dreamerv3_dali_shallow'];labels=['baseline','context11'];runs={l:root/n/n for l,n in zip(labels,names)}
 result={'protocol':'T-intersection; paired100 seeds per split, identical physical chassis; best checkpoints selected on20 validation episodes; both500K training; one training seed0; paired bootstrap10000 resamples reflects scenario uncertainty only.','models':{},'splits':{}}
 cfgs={}
 for label,name in zip(labels,names):
  assert queue['jobs'][name]['state']=='completed';s=json.loads((root/name/'status.json').read_text());assert s['phase']=='completed';entry=s['methods'][name];assert entry['state']=='completed'
  cfg=OmegaConf.load(runs[label]/'.hydra/config.yaml');OmegaConf.resolve(cfg);cfgs[label]=cfg
  assert cfg.model.rep_loss=='dreamer'and cfg.env.map_block=='T'
  if label=='context11':assert cfg.model.context.injection=='dali_shallow'and cfg.model.context.aux_weight==cfg.model.context.swap_weight==1 and cfg.model.context.wm_grad_scale==0
  else:assert not cfg.model.context.enabled
  rows=[json.loads(s)for s in(runs[label]/'metrics.jsonl').read_text().splitlines()];evals=[r for r in rows if 'episode/eval_success'in r];last20=[r for r in rows if 'train/loss/dyn'in r][-20:]
  result['models'][label]={'training_end_step':rows[-1]['step'],'best_step':entry['best_step'],'checkpoint_sha256':hashlib.sha256((runs[label]/'best.pt').read_bytes()).hexdigest(),'best_validation':entry['best_eval_metrics'],'final_validation':evals[-1],'last20_training_mean':{k:float(np.mean([r[k]for r in last20]))for k in last20[-1]if k.startswith('train/loss/')}}
 # The observation history switch is the only environment-configuration difference.
 envs={l:OmegaConf.to_container(cfg.env,resolve=True)for l,cfg in cfgs.items()}
 for env in envs.values():env.pop('context_enabled',None)
 assert envs['baseline']==envs['context11']
 rng=np.random.default_rng(20261022);indices=rng.integers(0,100,(10000,100));allrecords=[]
 for split in ['new_maps','s','xl']:
  records={};info={'summaries':{},'paired_context_minus_baseline':{},'provenance':{}}
  for label in labels:
   folder=runs[label]/'evaluation'/split;summary=json.loads((folder/'summary.json').read_text());assert summary['status']=='completed'
   rows=sorted([json.loads(s)for s in(folder/'episodes.jsonl').read_text().splitlines()],key=lambda r:r['seed']);assert len(rows)==len({r['seed']for r in rows})==100
   meta=json.loads((folder/'provenance.json').read_text());assert meta['checkpoint_sha256']==result['models'][label]['checkpoint_sha256'];assert meta['checkpoint_step']==result['models'][label]['best_step']
   assert sum(r['success']for r in rows)==summary['summary']['success_count']
   for k in ['success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps']:
    np.testing.assert_allclose(np.mean([r[k]for r in rows]),summary['summary'][k],rtol=1e-6)
   records[label]=rows;info['summaries'][label]=summary['summary'];info['provenance'][label]=meta
   allrecords.extend(dict(method=label,split=split,**r)for r in rows)
  identity=lambda r:(r['seed'],*[r['chassis'][k]for k in ['model','mass','max_steering','wheel_friction']])
  assert [identity(r)for r in records['baseline']]==[identity(r)for r in records['context11']]
  for key in ['success','crash','out_of_road','timeout','reward','route_completion','mean_speed_km_h','steps']:
   delta=np.array([float(c[key])-float(b[key])for b,c in zip(records['baseline'],records['context11'])])
   info['paired_context_minus_baseline'][key]={'difference':float(delta.mean()),'paired_bootstrap95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
  info['success_pairs']={'both_success':sum(b['success']and c['success']for b,c in zip(records['baseline'],records['context11'])),'baseline_only':sum(b['success']and not c['success']for b,c in zip(records['baseline'],records['context11'])),'context_only':sum(not b['success']and c['success']for b,c in zip(records['baseline'],records['context11']))}
  result['splits'][split]=info
 (out/'comparison.json').write_text(json.dumps(result,indent=2));(out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in allrecords))
 lines=['# DreamerV3 baseline vs DreamerV3＋旧11 context：T-intersection测试','','结论：context组在新地图/S/XL成功率分别提高8/7/38个百分点，主要收益在XL；同时车速明显降低、回合更长，不能解释为相同行驶效率下的全面提升。','','## 已核验的测试协议','','两组已完成500K训练及自动独立测试，每个方法3×100回合。对已有完整结果做一致性和配对审计，没有重复运行相同测试。每个split逐回合核验地图seed、车型、实际质量、最大转角和wheel_friction；核验best权重SHA256、测试来源和逐回合统计。环境配置除context历史开关外完全一致。','','本仓库DreamerV3 reproduction，MLP向量观测；context参考旧11：state4/K5/context8、DALI Shallow posterior-only、aux=1/swap=1/wm_grad_scale=0。训练seed0，质量/摩擦倍率0.6–1.4，车流0.2。','','## best模型驾驶指标','','| 测试集 | 方法 | best step | 成功率 | 碰撞率 | 出界率 | 奖励 | 路线完成率 | 平均速度km/h | 平均步数 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for split,info in result['splits'].items():
  for label,m in info['summaries'].items():lines.append(f'| {split} | {label} | {result["models"][label]["best_step"]} | {m["success"]:.0%} | {m["crash"]:.0%} | {m["out_of_road"]:.0%} | {m["reward"]:.2f} | {m["route_completion"]:.1%} | {m["mean_speed_km_h"]:.2f} | {m["steps"]:.1f} |')
 lines+=['','所有测试超时率均为0。碰撞与出界标记可同时出现，因此各失败率不一定相加为1。','','## 配对成功率变化','','| 测试集 | context−baseline | 配对bootstrap95% | baseline独成功 | context独成功 |','|---|---:|---|---:|---:|']
 for split,info in result['splits'].items():
  d=info['paired_context_minus_baseline']['success'];lo,hi=d['paired_bootstrap95'];pairs=info['success_pairs'];lines.append(f'| {split} | {d["difference"]*100:+.1f} pp | [{lo*100:+.1f}, {hi*100:+.1f}] pp | {pairs["baseline_only"]} | {pairs["context_only"]} |')
 lines+=['','单训练seed：bootstrap只反映所测场景的不确定性，不是多次训练方差。新地图/S差值区间可能跨零；XL改善更明显。','','## 解释范围','','- 结果为各自best，非500K latest：baseline420003，context180002，选取依据均为训练验证。','- 此次核验驾驶表现，不是独立公共轨迹的世界模型预测精度比较，也不单独证明context只表征底盘或被WM正确利用。','- 同时改变context表征、注入和辅助损失；不能把增益归因于交换损失单个因素。','- 降低车速可能贡献驾驶稳定性收益，尚无速度匹配干预。','']
 (out/'REPORT.md').write_text('\n'.join(lines))
 for split,info in result['splits'].items():print(split,json.dumps({'summaries':info['summaries'],'success_delta':info['paired_context_minus_baseline']['success']}),flush=True)
if __name__=='__main__':main()
