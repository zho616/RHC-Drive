"""Report completed WM-gradient experiment against the common original corpus."""
import json
from pathlib import Path
import numpy as np

def main():
 root=Path(__file__).resolve().parents[1]
 old=root/'logdir/tintersection_mass_friction_500k_20261021';new=root/'logdir/tintersection_shallow_wmgrad05_500k_20261022'
 out=new/'world_model_comparison';labels=['01 baseline','11 Shallow','12 WMgrad0.5']
 forecasts=[old/'world_model_comparison/baseline',old/'world_model_comparison/11_shallow',out/'12_shallow_wmgrad05']
 runs=[old/'training/r2dreamer',old/'training/r2dreamer_dali_shallow',new/'training/r2dreamer_dali_shallow']
 summaries={label:json.loads((p/'summary.json').read_text())for label,p in zip(labels,forecasts)}
 provenance={label:json.loads((p/'provenance.json').read_text())for label,p in zip(labels,forecasts)}
 assert len({v['origins_sha256']for v in provenance.values()})==1
 errors=[dict(np.load(p/'errors.npz'))for p in forecasts]
 for e in errors:
  np.testing.assert_array_equal(e['valid'],errors[0]['valid'])
  assert all(np.isfinite(v).all()for v in e.values())
 train={}
 for label,p in zip(labels,runs):
  rows=[json.loads(s)for s in(p/'metrics.jsonl').read_text().splitlines()];rows=[r for r in rows if r['step']>=400000 and 'train/loss/dyn'in r]
  keys=[k for k in rows[-1]if k.startswith('train/loss/')or k.startswith('train/context/one_step_mae')]
  train[label]={'n':len(rows),'first_step':rows[0]['step'],'last_step':rows[-1]['step'],'mean':{k:float(np.mean([r[k]for r in rows]))for k in keys}}
 (out/'comparison_01_11_12.json').write_text(json.dumps(dict(forecasts=summaries,training_last100k=train,provenance=provenance),indent=2))
 lines=['# 12：WM→context=0.5 世界模型指标核查','','结论：奖励/continue头发生严重退化，不能将本次驾驶成功率改善解释为世界模型预测提升。尚不能确定根因，需另行核查梯度回传路径在编译、混合精度和优化器下的行为。','','## 末段训练损失','','约400K–495K，各20条均值。各自策略采集的训练分布不同，且不对应早期best。','','| 指标 | 01 baseline | 11 | 12 |','|---|---:|---:|---:|']
 for key in ['dyn','rep','barlow','rew','con','context_aux','context_swap']:
  lines.append('| '+key+' | '+' | '.join(f"{train[l]['mean']['train/loss/'+key]:.6f}"if 'train/loss/'+key in train[l]['mean']else '—'for l in labels)+' |')
 lines+=['','## 共同真实轨迹的best预测误差','','100回合、10地图、5车型、2种采集策略；2410预测起点、3次潜变量采样；相同观测前缀与未来动作，开环预测，无未来真值/context输入。checkpoint：01=310001，11=120007，12=180002。','','01/11使用原已有GPU FP32评估，12为避免竞争训练GPU使用CPU FP32；相同协议、样本和种子，但CPU/CUDA潜变量随机数实现不同，因此不是相同的潜变量样本。单训练seed，不能代表重复实验方差。','','| 指标 | 步长 | 01 baseline | 11 | 12 |','|---|---:|---:|---:|---:|']
 for key in ['reward_mae','reward_nll','continue_brier','continue_nll']:
  for h in ['1','5','10']:lines.append(f'| {key} | {h} | '+' | '.join(f"{summaries[l]['all'][key][h]:.6f}"for l in labels)+' |')
 lines+=['','reward MAE为原奖励单位，NLL越小越好；continue目标=1-is_terminal，超时仍可持续。12的Brier接近始终预测continue=1的简单参照，不能仅看Brier低值认定有效；极高NLL表明概率预测严重失准。','','## 辅助响应预测器：与RSSM世界模型分开报告','','以下是独立context辅助预测器的开环误差，不是RSSM解码状态误差。','','| 模型 | 步长 | 加速度 m/s² | 横摆角速度 rad/s | 速度 m/s | 前轮角 rad |','|---|---:|---:|---:|---:|---:|']
 for l in labels[1:]:
  for h in ['1','5','10']:lines.append(f'| {l} | {h} | '+' | '.join(f'{v:.6f}'for v in summaries[l]['all']['auxiliary_mae'][h])+' |')
 lines+=['','KL受到free-nats及梯度停止影响；Barlow是表征目标，均不能单独证明物理状态预测精度。本R2分支没有训练观测重建解码器。','','本报告未修改任何checkpoint或运行中实验，未触发重训。','']
 (out/'REPORT_01_11_12.md').write_text('\n'.join(lines))
 print(out/'REPORT_01_11_12.md')
if __name__=='__main__':main()
