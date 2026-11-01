"""Summarize common-corpus T world-model predictions separately from training losses."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    corpus = root/'world_model_common_corpus'
    out = root/'world_model_comparison'
    origins = json.loads((corpus/'origins.json').read_text())
    episodes = json.loads((corpus/'episodes.json').read_text())
    maps = np.array([o['map_id'] for o in origins])
    models = np.array([o['model'] for o in origins])
    collectors = np.array([o['policy'] for o in origins])
    arrays = {name: dict(np.load(out/name/'errors.npz')) for name in ('baseline', '11_shallow')}
    summary = {name: json.loads((out/name/'summary.json').read_text()) for name in arrays}
    provenance = {name: json.loads((out/name/'provenance.json').read_text()) for name in arrays}
    assert len({v['origins_sha256'] for v in provenance.values()}) == 1
    valid = arrays['baseline']['valid']
    for data in arrays.values():
        np.testing.assert_array_equal(valid, data['valid'])
        assert all(np.isfinite(v).all() for v in data.values())
    unique = np.unique(maps)
    draws = np.random.default_rng(20261021).integers(0, len(unique), (10000, len(unique)))
    metrics = ['reward_mae', 'reward_nll', 'continue_brier', 'continue_nll']
    paired = {}
    for key in metrics:
        paired[key] = {}
        delta = arrays['11_shallow'][key].mean(0)-arrays['baseline'][key].mean(0)
        for k in (0, 4, 9):
            sums = np.array([delta[(maps==mi)&valid[:, k], k].sum(dtype=np.float64) for mi in unique])
            counts = np.array([((maps==mi)&valid[:, k]).sum() for mi in unique])
            samples = sums[draws].sum(1)/counts[draws].sum(1)
            paired[key][str(k+1)] = {'difference': float(delta[valid[:,k],k].mean()),
                                     'map_cluster_bootstrap95': np.quantile(samples, [.025, .975]).tolist()}
    strata = {}
    for name, select in [('baseline_collector', collectors=='baseline'), ('11_collector', collectors=='11_shallow'),
                         ('seen_types', np.isin(models, ['default', 'm', 'l'])), ('unseen_types', np.isin(models, ['s', 'xl']))]:
        strata[name] = {label: {key: {str(k+1): float(data[key].mean(0)[select&valid[:,k],k].mean())
                                     for k in (0,4,9)} for key in metrics} for label,data in arrays.items()}
    train = {}
    for label, folder in [('baseline', 'r2dreamer'), ('11_shallow', 'r2dreamer_dali_shallow')]:
        rows = [json.loads(line) for line in (root/'training'/folder/'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if 'train/loss/dyn' in r][-20:]
        keys = [k for k in rows[-1] if k.startswith('train/loss/') or k.startswith('train/context/one_step_mae')
                or k in ('train/context/swap_coverage', 'train/context/swap_pairs')]
        train[label] = {'first_step': rows[0]['step'], 'last_step': rows[-1]['step'], 'n': len(rows),
                        'mean': {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}}
    n,H = valid.shape
    terminal = np.zeros((n,H), bool)
    persistence = np.zeros((n,H), np.float32)
    data = [dict(np.load(corpus/'episodes'/e['file'])) for e in episodes]
    for i,o in enumerate(origins):
        d = data[o['episode']]
        reward = sum(d['log_reward_'+key] for key in ('progress','speed','lateral','terminal'))
        np.testing.assert_allclose(reward, d['reward'], atol=1e-4)
        t = o['t']; length = min(H, episodes[o['episode']]['length']-t)
        terminal[i,:length] = d['is_terminal'][t+1:t+length+1]
        persistence[i,:length] = abs(reward[t]-reward[t+1:t+length+1])
    counts = {str(k+1): {'valid': int(valid[:,k].sum()), 'terminals': int(terminal[valid[:,k],k].sum()),
                          'always_continue_brier': float(terminal[valid[:,k],k].mean()),
                          'reward_persistence_mae': float(persistence[valid[:,k],k].mean())} for k in (0,4,9)}
    result = {'summary': summary, 'provenance': provenance, 'paired_11_minus_baseline': paired,
              'strata': strata, 'training_last20': train, 'counts_and_simple_controls': counts,
              'methodology': '100 real T-intersection trajectories, ten independent maps, five vehicle types, two frozen collectors, paired random mass/friction multipliers;2410 origins;three latent draws. Identical observation prefixes and recorded future actions for each model. Forecast horizon1/5/10 steps at10Hz. No future observations, states or true context. Single training seed; map-cluster bootstrap reflects scenario uncertainty only.'}
    (out/'comparison.json').write_text(json.dumps(result, indent=2))
    lines = ['# T-intersection：共同真实轨迹世界模型评估', '',
             '结论：这组best模型不支持世界模型全面提升。11奖励预测MAE在1/5/10步均退化，奖励NLL也更高；10步continue NLL改善，但Brier改善很小，地图bootstrap区间跨零。', '', result['methodology'], '',
             '两组均使用驾驶验证选出的best：baseline310001，11为120007。Shallow只在posterior注入context；先验滚动没有直接context入口。权重冻结，未训练观测解码器。', '',
             '## 共同轨迹预测误差（均越低越好）', '',
             '| 指标 | 步长 | baseline | 11 | 相对变化 | 差值地图bootstrap95%区间 |',
             '|---|---:|---:|---:|---:|---|']
    for metric in metrics:
        for k in (1,5,10):
            a = summary['baseline']['all'][metric][str(k)]
            b = summary['11_shallow']['all'][metric][str(k)]
            lo,hi = paired[metric][str(k)]['map_cluster_bootstrap95']
            lines.append(f'| {metric} | {k} | {a:.6f} | {b:.6f} | {(b/a-1)*100:+.1f}% | [{lo:+.6f}, {hi:+.6f}] |')
    lines += ['', '奖励MAE采用原始奖励单位；奖励NLL为离散symlog分布负对数似然。Continue目标为1-is_terminal，超时截断仍视为可持续；真实回合后的目标不计分，预测不按自身continue提前截断。Brier为概率均方误差，受终止类别不平衡影响。', '',
              '## 10步预测按轨迹采集策略分层', '',
              '| 轨迹来源 | 模型 | reward MAE | reward NLL | continue NLL |', '|---|---|---:|---:|---:|']
    for collector in ('baseline_collector', '11_collector'):
        for label,values in strata[collector].items():
            lines.append('| '+collector+' | '+label+' | '+' | '.join(f'{values[k]["10"]:.6f}' for k in ('reward_mae','reward_nll','continue_nll'))+' |')
    lines += ['', '奖励退化主要发生在baseline采集的轨迹；在11自身轨迹上，两组奖励误差相近，11仍未优于基线。考虑到两策略速度不同，可能涉及训练状态分布差异，但本测试不确定因果。', '',
              '## 末段训练日志（约400K–495.5K，最后20条均值）', '',
              '| 方法 | dyn/rep KL | Barlow | reward NLL | continue NLL | context aux | context swap |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label,v in train.items():
        m = v['mean']
        values = [f'{m["train/loss/"+k]:.6f}' if 'train/loss/'+k in m else '—' for k in ('dyn','barlow','rew','con','context_aux','context_swap')]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '各组训练样本来自自身策略，车速和状态分布不同；上述末段日志也不对应早期best权重。dyn/rep数值相同但梯度路径不同，KL有free-nats，不能将降低直接解释为更强动力学。Barlow为表征一致性目标，不是状态重建误差。', '',
              '## 11辅助响应预测器（独立于RSSM输出）', '',
              '| 步长 | 加速度MAE m/s² | 横摆角速度MAE rad/s | 速度MAE m/s | 前轮角MAE rad |', '|---|---:|---:|---:|---:|']
    for k, values in summary['11_shallow']['all']['auxiliary_mae'].items():
        lines.append('| '+k+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', 'Baseline无此辅助预测头，不能作同结构误差对比；这也不是RSSM未来物理状态的重建精度。', '',
              '## 计数与简单参照', '', '```json', json.dumps(counts, indent=2), '```', '',
              '按采集策略、训练已见/未见车型分层结果见comparison.json。两模型共享全部输入轨迹，采集策略不等于被评估模型。随机质量与摩擦只抽取十张地图对应的配对配置，并非遍历全部243种配置。', '',
              'PCA分离和驾驶成功率均不能代替世界模型预测测试；即使共同轨迹预测改善，也不能单独归因于swap损失，或证明context仅编码车辆参数。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for metric,values in paired.items():
        print(metric, {k: {'baseline': summary['baseline']['all'][metric][k], '11': summary['11_shallow']['all'][metric][k], **v} for k,v in values.items()})


if __name__ == '__main__':
    main()
