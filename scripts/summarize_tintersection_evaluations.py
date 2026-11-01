"""Audit paired baseline/shallow T-intersection driving evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Training folder with status.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.root / 'status.json').read_text())
    assert status['phase'] == 'completed', 'Wait for all evaluations to complete'
    methods = {'baseline': 'r2dreamer', '11_shallow': 'r2dreamer_dali_shallow'}
    result = {'summaries': {}, 'paired_differences': {}, 'provenance': {},
              'methodology': '100 paired scenario seeds per split; best checkpoint selected using training validation; paired bootstrap 10000 resamples. Single training seed: intervals reflect scenario uncertainty only.'}
    digests = {}
    for split in ('new_maps', 's', 'xl'):
        rows = {}
        result['summaries'][split], result['provenance'][split] = {}, {}
        for label, name in methods.items():
            assert status['methods'][name]['evaluations'][split]['state'] == 'completed'
            folder = args.root / name / 'evaluation' / split
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['status'] == 'completed'
            records = sorted([json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()], key=lambda r: r['seed'])
            assert len(records) == len({r['seed'] for r in records}) == 100
            assert sum(r['success'] for r in records) == summary['summary']['success_count']
            metadata = json.loads((folder / 'provenance.json').read_text())
            checkpoint = metadata['checkpoint']
            if checkpoint not in digests:
                digests[checkpoint] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            assert digests[checkpoint] == metadata['checkpoint_sha256']
            result['provenance'][split][label] = metadata
            result['summaries'][split][label] = summary['summary']
            rows[label] = records
        identity = lambda r: (r['seed'], *(r['chassis'][k] for k in ('model', 'mass', 'max_steering', 'wheel_friction')))
        assert [identity(r) for r in rows['baseline']] == [identity(r) for r in rows['11_shallow']]
        rng = np.random.default_rng(20261021)
        indices = rng.integers(0, 100, (10000, 100))
        comparison = {}
        for key in ('success', 'crash', 'out_of_road', 'timeout', 'reward', 'route_completion', 'mean_speed_km_h', 'steps'):
            delta = np.array([float(a[key]) - float(b[key]) for a, b in zip(rows['11_shallow'], rows['baseline'])])
            comparison[key] = {'difference': float(delta.mean()), 'paired_bootstrap95': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
        comparison['success_pairs'] = {
            'both_success': sum(a['success'] and b['success'] for a, b in zip(rows['11_shallow'], rows['baseline'])),
            '11_only_success': sum(a['success'] and not b['success'] for a, b in zip(rows['11_shallow'], rows['baseline'])),
            'baseline_only_success': sum(not a['success'] and b['success'] for a, b in zip(rows['11_shallow'], rows['baseline']))}
        result['paired_differences'][split] = comparison
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# T-intersection 500K 独立驾驶测试', '',
             '核验自动评估完成记录。两组均为 R2Dreamer，每类各100回合；逐回合地图种子、车型、实际质量、最大前轮转角、wheel_friction配对一致，检查点SHA256一致。', '',
             '| 测试集 | 方法 | best step | 成功率 | 碰撞率 | 出界率 | 超时率 | 奖励 | 路线完成 | 平均车速 km/h | 平均步数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in result['summaries']:
        for label, s in result['summaries'][split].items():
            step = result['provenance'][split][label]['checkpoint_step']
            lines.append(f'| {split} | {label} | {step} | {s["success"]:.0%} | {s["crash"]:.0%} | {s["out_of_road"]:.0%} | {s["timeout"]:.0%} | {s["reward"]:.2f} | {s["route_completion"]:.1%} | {s["mean_speed_km_h"]:.2f} | {s["steps"]:.1f} |')
    lines += ['', '## 11 相对基线的成功率差', '', '| 测试集 | 差值 | 配对bootstrap 95%区间 |', '|---|---:|---|']
    for split, d in result['paired_differences'].items():
        v = d['success']; lo, hi = v['paired_bootstrap95']
        lines.append(f'| {split} | {100*v["difference"]:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] pp |')
    lines += ['', '## 解释范围', '',
              '11 的成功率优势同时伴随更低车速、更长回合；不能据此声称同等行驶效率下的控制质量或安全性更好。',
              'best由20回合训练验证选择，baseline与11选择步数不同；这些不是500K final checkpoint结果。',
              '训练及测试质量和摩擦倍率为0.6–1.4，间隔0.1，转角倍率固定1；S/XL为训练未见车型。',
              '仅一个训练种子，bootstrap区间不反映不同训练种子的不确定性。这不是swap损失单独消融，也不是世界模型预测精度测试。']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result['summaries'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
