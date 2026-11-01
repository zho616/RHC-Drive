"""Auditable native-metric comparison; absent decoders remain not applicable."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

METHODS = ['dreamerv3', 'r2dreamer', 'dreamerv3_context', 'r2dreamer_context']
LABELS = {'dreamerv3':'DreamerV3', 'r2dreamer':'R2Dreamer',
          'dreamerv3_context':'DreamerV3 + Context', 'r2dreamer_context':'R2Dreamer + Context'}
METRICS = ['kl_prior_post_raw','kl_post_prior_raw','state_native_recon_loss',
           'state_mae','state_mse','state_psnr_from_mse','state_psnr_frame',
           'prior_entropy','post_entropy','kl_prior_post_unimix','kl_post_prior_unimix']


def write_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--tables-only',action='store_true')
    a=p.parse_args();root=a.root
    rows={m:json.loads((root/m/'episodes.json').read_text()) for m in METHODS}
    prov={m:json.loads((root/m/'provenance.json').read_text()) for m in METHODS}
    assert len({v['corpus_manifest_sha256'] for v in prov.values()})==1
    masks=[]
    for m in METHODS:
        with np.load(root/m/'native_metrics.npz') as z:masks.append(z['valid'].copy())
    for z in masks[1:]:np.testing.assert_array_equal(z,masks[0])
    identifiers=lambda rr:[(r['episode'],r['map_id'],r['seed'],r['model'],r['policy'],r['chassis'],r['valid_frames']) for r in rr]
    for m in METHODS[1:]:assert identifiers(rows[m])==identifiers(rows[METHODS[0]])
    reference=rows[METHODS[0]]
    groups={
        'all':[i for i,r in enumerate(reference) if r['metrics']],
        'id':[i for i,r in enumerate(reference) if r['metrics'] and r['model'] in ['default','m','l']],
        'ood_s_xl':[i for i,r in enumerate(reference) if r['metrics'] and r['model'] in ['s','xl']],
        's':[i for i,r in enumerate(reference) if r['metrics'] and r['model']=='s'],
        'xl':[i for i,r in enumerate(reference) if r['metrics'] and r['model']=='xl'],
        'baseline_collector':[i for i,r in enumerate(reference) if r['metrics'] and r['policy']=='baseline'],
        'context_collector':[i for i,r in enumerate(reference) if r['metrics'] and r['policy']=='11_shallow']}
    unique=np.unique([r['map_id'] for r in reference]);rng=np.random.default_rng(20261025)
    draws=rng.integers(0,len(unique),(10000,len(unique)))
    summary={};table=[];paired=[]
    def estimate(method,ids,metric):
        source='state_mse' if metric=='state_psnr_from_mse' else metric
        if source not in rows[method][ids[0]]['metrics']:return None
        vals=np.array([rows[method][i]['metrics'][source] for i in ids],np.float64)
        maps=np.array([reference[i]['map_id'] for i in ids])
        sums=np.array([vals[maps==mi].sum() for mi in unique]);counts=np.array([(maps==mi).sum() for mi in unique])
        boots=sums[draws].sum(1)/counts[draws].sum(1)
        point=float(vals.mean())
        if metric=='state_psnr_from_mse':point=float(-10*np.log10(point));boots=-10*np.log10(boots)
        return dict(mean=point,ci95=np.quantile(boots,[.025,.975]).tolist(),bootstrap=boots)
    for group,ids in groups.items():
        summary[group]={}
        estimates={}
        for m in METHODS:
            record=dict(method=m,label=LABELS[m],split=group,checkpoint_step=prov[m]['checkpoint_step'],
                        episodes=len(ids),valid_frames=sum(reference[i]['valid_frames'] for i in ids),
                        state_dim=prov[m]['state_dim'],native_decoder=prov[m]['trained_decoder'])
            sm={}
            for metric in METRICS:
                v=estimate(m,ids,metric);estimates[m,metric]=v
                record[metric]=v['mean'] if v else None
                sm[metric]={k:x for k,x in v.items() if k!='bootstrap'} if v else None
            summary[group][m]=sm;table.append(record)
        for new,base in [('dreamerv3_context','dreamerv3'),('r2dreamer_context','r2dreamer')]:
            for metric in METRICS:
                x,y=estimates[new,metric],estimates[base,metric]
                if x is None or y is None:continue
                delta=x['mean']-y['mean'];lo,hi=np.quantile(x['bootstrap']-y['bootstrap'],[.025,.975])
                paired.append(dict(split=group,comparison=new+'_minus_'+base,metric=metric,
                    baseline=y['mean'],context=x['mean'],difference=delta,
                    relative_change_pct=100*delta/y['mean'],ci95_low=float(lo),ci95_high=float(hi)))
    runpaths={m:Path(v['checkpoint']).parent for m,v in prov.items()}
    training=[];curves=[]
    for m,path in runpaths.items():
        rr=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        rr=[r for r in rr if 'train/loss/dyn' in r]
        keys=sorted({k for r in rr for k in r if k.startswith('train/loss/')})
        tail=rr[-20:];record=dict(method=m,label=LABELS[m],training_seed=0,
             first_tail_step=tail[0]['step'],last_tail_step=tail[-1]['step'],records=len(tail))
        for key in ['dyn','rep','state','barlow','rew','con','context_aux','context_swap']:
            vs=[r['train/loss/'+key] for r in tail if 'train/loss/'+key in r]
            record[key]=float(np.mean(vs)) if vs else None
        training.append(record)
        for r in rr:
            for k in keys:
                if k in r:curves.append(dict(method=m,step=r['step'],metric=k,value=r[k]))
    write_csv(root/'native_comparison.csv',table);write_csv(root/'paired_differences.csv',paired)
    write_csv(root/'training_last20.csv',training);write_csv(root/'training_curves.csv',curves)
    result=dict(summary=summary,paired=paired,training_last20=training,provenance=prov,
        checkpoint_selection='best' if all(Path(v['checkpoint']).name=='best.pt' for v in prov.values()) else 'latest',
        aggregation='Average latent draws and valid frames within each episode, then equal-weight episodes. PSNR_from_mse=-10log10(episode-balanced MSE), peak=1. Frame-mean PSNR also exported separately.',
        confidence='10000 paired map-cluster bootstrap resamples of the same 10 maps. Scenario uncertainty only, not independent training-seed uncertainty.',
        counts={g:dict(episodes=len(ids),frames=sum(reference[i]['valid_frames'] for i in ids)) for g,ids in groups.items()},
        limitations=['100 fixed real trajectories from two R2-family collectors; collector composition is not neutral to all four methods, so stratified collector results are included.',
                     '98 episodes have valid t>=20 frames; two too-short episodes excluded identically.',
                     'Only one original training seed. Three latent draws are not three training seeds.',
                     ('Different validation-selected best steps; table describes deployed best checkpoints, not equal-step final checkpoints.' if all(Path(v['checkpoint']).name=='best.pt' for v in prov.values()) else 'Final saved checkpoints are all approximately 500K steps, differing by at most seven steps due to vectorized stepping.'),
                     'State reconstruction PSNR is not BEV image PSNR; R2 has no native decoder and entries remain null.',
                     'Posterior reconstruction observes the target state; it does not establish future open-loop prediction quality.',
                     'Lower KL alone does not prove better dynamics or exclude representation collapse.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    if a.tables_only:
        print(root/'comparison.json')
        return
    def f(v,d=5):return 'N/A' if v is None else f'{v:.{d}f}'
    lines=['# T-intersection：四方法原生世界模型指标', '',
        '结论：当前实际部署的 best 检查点不支持“加入 Context 后先验对齐和重建质量改善”的论证。两个 Context 版本在公共轨迹上的 KL 均升高；DreamerV3+Context 的状态重建损失、MAE、MSE 更高，PSNR 更低。驾驶成功率提升与这些指标并不等价。', '',
        '参考 [DynaDreamer §VI-D / Table IV](https://arxiv.org/html/2607.13410v2#S6.SS4)，该文表头为 prior‖posterior KL 及 BEV 图像 PSNR/MAE，并在未见车型合并条件评估。这里只借用评估思路；不直接比较论文绝对数值。当前四模型均输入 259 维 [0,1] 状态，R2 两组无原生观测解码器，按用户要求不增加探针，重建项标 N/A。', '',
        '## 配对公共语料评估', '',
        '100 条已有真实 T 路口轨迹，10 张新地图（30000–30009），5 个车型（default/M/L/S/XL），两种冻结采集策略；保持配对的质量和摩擦倍率。每个模型读取完全相同的观测与动作前缀，不使用未来观测。暖启动 20 步，排除结束/碰撞目标后剩 98 回合、20934 个有效状态，每个模型 3 次随机隐变量采样。统计先在每回合内平均，再等权平均回合，避免慢车长回合主导结果。', '',
        'KL 使用原生训练的未混匀 categorical logits，按随机变量维度求和，单位 nat；主表不应用 free-nats 截断。另导出与实际采样 unimix 分布一致的 KL、熵以及训练方向 posterior‖prior KL。重建调用原生已训练解码器，无任何新增训练。', '',
        '| 方法 | best step | KL prior‖post ↓ | KL post‖prior ↓ | 原生重建损失 ↓ | 状态 MAE ↓ | 状态 MSE ↓ | 状态 PSNR dB ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['all','id','ood_s_xl','s','xl']:
        lines+=['',f'### {group}', ''] if group!='all' else []
        if group!='all':lines+=['| 方法 | best step | KL prior‖post ↓ | KL post‖prior ↓ | 原生重建损失 ↓ | 状态 MAE ↓ | 状态 MSE ↓ | 状态 PSNR dB ↑ |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in [t for t in table if t['split']==group]:
            lines.append('| '+r['label']+' | '+str(r['checkpoint_step'])+' | '+' | '.join(f(r[k]) for k in ['kl_prior_post_raw','kl_post_prior_raw','state_native_recon_loss','state_mae','state_mse','state_psnr_from_mse'])+' |')
    lines+=['','原生重建损失为 259 维 symlog 空间的平方误差之和，不是像素 MAE、原状态 MSE 或有正规化常数的概率 NLL。原状态 MSE/MAE 为每维均值。PSNR=−10log10(MSE)，peak=1，不裁剪预测值。另保存先算每帧 PSNR 再平均的数值，二者因 log 非线性不会相同。', '',
            '## Context−baseline 配对变化', '',
            '| 分组 | 对比 | 指标 | baseline | Context | 相对变化 | 差值地图 bootstrap 95% |',
            '|---|---|---|---:|---:|---:|---|']
    for r in paired:
        if r['split'] in ['all','ood_s_xl'] and r['metric'] in ['kl_prior_post_raw','state_mae','state_mse','state_psnr_from_mse']:
            lines.append(f"| {r['split']} | {r['comparison']} | {r['metric']} | {r['baseline']:.5f} | {r['context']:.5f} | {r['relative_change_pct']:+.1f}% | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] |")
    lines+=['','PSNR 的差值单位为 dB；百分比不作为主要解释。区间只反映地图采样不确定性，不能当成三次训练的误差条。', '',
            '## 完整 500K 训练末段日志', '',
            '| 方法 | 最后20条 step 范围 | dyn/rep（截断） | state loss | Barlow | reward loss | continue loss |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in training:lines.append('| '+r['label']+f" | {r['first_tail_step']}–{r['last_tail_step']} | "+' | '.join(f(r[k]) for k in ['dyn','state','barlow','rew','con'])+' |')
    lines+=['','训练损失来自各方法自身策略产生的不同样本分布，且是 500K 末段；公共评估使用验证选出的 best。不能把这两种结果混成同一实验。dyn 与 rep 数值相同但梯度方向不同，并经过 free-nats 下限。Barlow 不是重建损失。', '',
            '## 可以支持的表述', '',
            '“Context 在本实验中改善了闭环驾驶表现，但在配对公共轨迹上，未观察到先验／后验 KL 缩小；DreamerV3 的原生状态重建也未改善。因此，当前证据支持任务层面的适应收益，尚不足以将收益归因于更精确的世界模型重建或更强的先验对齐。”', '',
            '## 解释边界', '',
            '- 当前两组 Context 是 DALI Shallow：当前 context 直接进入 posterior，prior 无当前 context 的直接入口；它仍可能经历史随机状态间接影响未来隐状态。这不同于 DynaDreamer 对 prior/posterior 的联合条件化，不能直接套用该文的理论结论。',
            '- 不同训练方法的潜变量语义和熵不同，即便维度相同，跨模型 KL 也不能独立作为能力排名；同骨干配对差值更有解释力。熵诊断见 JSON/CSV。',
            '- 语料只有两种 R2-family 采集策略，可能偏向其状态分布；按采集策略的分层结果已一并导出。要作更广泛的结论，需追加四方法均衡采集或独立行为策略语料。',
            '- 不宣称重建改善、未来预测改善或辅助交换损失的单独因果收益；未改动任何原始模型与既有驾驶测试。', '',
            '文件：native_comparison.csv、paired_differences.csv、training_last20.csv、training_curves.csv、comparison.json 和每模型 provenance/逐回合数据。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(root/'REPORT.md')
    for r in [x for x in table if x['split']=='all']:
        print(r['label'],{k:r[k] for k in ['kl_prior_post_raw','state_native_recon_loss','state_mae','state_mse','state_psnr_from_mse']})


if __name__=='__main__':main()
