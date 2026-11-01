#!/usr/bin/env python3
"""Evaluate frozen nuPlan R2 checkpoints on paired held-out S/XL chassis."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'r2dreamer': ROOT/'logdir/nuplan_pittsburgh_baseline_500k_20261022',
    'r2dreamer_context': ROOT/'logdir/nuplan_context_roundabout_queue_20261023/nuplan_r2dreamer_context',
}
SOURCES = {
    'r2dreamer': RUNS['r2dreamer']/'source',
    'r2dreamer_context': RUNS['r2dreamer_context'].parent/'nuplan_source',
}
DATA = Path('/home/server/datasets/nuplan/scenarionet_pittsburgh_8000/test')
CHECKPOINTS = {name: run/('baseline/best.pt' if name == 'r2dreamer' else 'best.pt') for name,run in RUNS.items()}
ID_FOLDERS = {name: run if name == 'r2dreamer' else run/'evaluation' for name,run in RUNS.items()}
METRICS = ('success', 'crash', 'out_of_road', 'timeout', 'route_completion', 'reward', 'mean_speed_km_h')


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def read_results(folder):
    summary = json.loads((folder/'summary.json').read_text())
    rows = [json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
    assert summary['status'] == 'completed' and len(rows) == 100
    assert [r['seed'] for r in rows] == list(range(100))
    for key in METRICS:
        assert abs(sum(float(r[key]) for r in rows)/100-summary['summary'][key]) < 1e-8
    return summary, rows


def aggregate(output, splits):
    import numpy as np
    result = {}
    for split in splits:
        data = {}; metadata = {}
        for name in RUNS:
            seed_rows = []
            for i in range(3):
                folder = (ID_FOLDERS[name]/f'test_seed{i}'
                          if split == 'id' else output/name/split/f'seed{i}')
                summary, rows = read_results(folder)
                assert summary['checkpoint_sha256'] == hashlib.sha256(CHECKPOINTS[name].read_bytes()).hexdigest()
                assert summary['inference_seed_offset'] == 12345+100000*i
                assert Path(summary['scenario_dir']) == DATA
                seed_rows.append(rows)
                metadata[name] = {k: summary[k] for k in ('checkpoint_sha256','checkpoint_step','eval_models','eval_mass_scales','eval_friction_scales','eval_steering_scales')}
            data[name] = seed_rows
        for i in range(3):
            for a,b in zip(data['r2dreamer'][i], data['r2dreamer_context'][i]):
                assert a['seed'] == b['seed'] and a['chassis'] == b['chassis']
            for name in RUNS:
                for a,b in zip(data[name][0],data[name][i]):
                    assert a['chassis'] == b['chassis']
                    if split != 'id': assert a['chassis']['model'] == split
        methods = {}
        for name, rows in data.items():
            flat = [r for group in rows for r in group]
            methods[name] = dict(rollouts=len(flat), unique_scenarios=100,
                mean={k: sum(float(r[k]) for r in flat)/len(flat) for k in METRICS},
                success_by_inference_seed=[sum(r['success'] for r in group)/100 for group in rows])
        # Bootstrap scenario clusters, preserving all 3 inference repeats per scene.
        delta = np.array([[float(b['success'])-float(a['success']) for a,b in zip(data['r2dreamer'][i],data['r2dreamer_context'][i])] for i in range(3)]).mean(axis=0)
        rng = np.random.default_rng(20261023)
        boots = delta[rng.integers(0,100,size=(20000,100))].mean(axis=1)
        result[split] = dict(methods=methods, metadata=metadata,
            paired_success_delta=float(delta.mean()),
            paired_success_delta_scene_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),
            pairing_verified=True)
    write(output/'comparison.json', result)
    lines = ['# nuPlan Pittsburgh：R2Dreamer 与 R2Dreamer+context', '',
        '使用验证集选出的best权重；相同100个留出场景、相同底盘参数、3组推理种子。一个训练种子，推理重复不是独立训练。训练预算和配置以各训练目录记录为准。', '',
        'ID沿用已完成的独立测试；本次S/XL评估使用的设备见status.json，各best权重步数和指纹见comparison.json。', '',
        'S/XL为未见车型OOD；质量及轮胎摩擦倍率仍为0.6–1.4，最大转角取各车型名义值。车型变化同时包含几何、名义动力和质量等变化。未增加城市或倍率范围外OOD。', '',
        '| 测试 | 方法 | 成功率 | 碰撞率 | 出界率 | 超时率 | 路线完成率 | 回报 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, item in result.items():
        for name, values in item['methods'].items():
            m = values['mean']; vals = [f"{m[k]*100:.1f}%" for k in METRICS[:5]]
            lines.append('| '+' | '.join([split,name,*vals,f"{m['reward']:.2f}"])+' |')
    lines += ['', '成功率提升及按场景聚类的配对bootstrap 95%区间（百分点；仅反映本测试集及推理变异，不包含训练种子差异）：', '']
    for split,item in result.items():
        low,high=item['paired_success_delta_scene_bootstrap95']
        lines.append(f"- {split}: {item['paired_success_delta']*100:+.2f}，95%区间 [{low*100:+.2f}, {high*100:+.2f}]。")
    lines += ['', '测试场景仅来自一个留出采集session，不能据此断言跨城市或广泛交通条件的泛化能力。详细指纹、逐种子成功率和配对核验见comparison.json。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return result


def main():
    global RUNS, SOURCES, CHECKPOINTS, ID_FOLDERS, DATA
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--training-queue',type=Path,help='Evaluate the completed R2 runs from this newer queue')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.training_queue:
        from omegaconf import OmegaConf
        queue=args.training_queue.resolve()
        plan=json.loads((queue/'plan.json').read_text())
        status=json.loads((queue/'status.json').read_text())
        jobs={name:next(j for j in plan['jobs'] if j['name']=='nuplan_'+name) for name in RUNS}
        assert all(status['jobs'][j['name']]['state']=='completed' for j in jobs.values())
        RUNS={name:Path(j['logdir']) for name,j in jobs.items()}
        SOURCES={name:Path(j['source']) for name,j in jobs.items()}
        CHECKPOINTS={name:path/'best.pt' for name,path in RUNS.items()}
        ID_FOLDERS={name:path/'evaluation' for name,path in RUNS.items()}
        configs=[OmegaConf.load(path/'.hydra/config.yaml') for path in RUNS.values()]
        assert len({str(c.env.scenario_test_dir) for c in configs})==1
        DATA=Path(configs[0].env.scenario_test_dir)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    old=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
    previous={(j['method'],j['chassis'],j['seed']):j for j in old.get('jobs',[])}
    state=dict(status='running',pid=os.getpid(),device=args.device,jobs=[])
    lock=Lock()
    write(output/'status.json',state)
    aggregate(output,['id'])
    environment=os.environ.copy(); environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.device=='cpu':environment['CUDA_VISIBLE_DEVICES']=''
    environment['PYTHONPATH']='/home/server/projects/metadrive'
    def execute(chassis,seed,name):
        run=RUNS[name]
        folder=output/name/chassis/f'seed{seed}'
        command=[sys.executable,'-u',str(output/'controls/evaluate_metadrive.py'),
            '--checkpoint',str(CHECKPOINTS[name]),'--source-root',str(SOURCES[name]),
            '--output',str(folder),'--episodes','100','--seed-start','0',
            '--scenario-dir',str(DATA),'--inference-seed-offset',str(12345+seed*100000),
            '--eval-models',chassis,'--device',args.device]
        job=dict(method=name,chassis=chassis,seed=seed,command=command,status='running',started_at=datetime.datetime.now().astimezone().isoformat())
        with lock:
            state['jobs'].append(job); write(output/'status.json',state)
        if folder.exists():
            old_job=previous.get((name,chassis,seed),{})
            pid=old_job.get('pid')
            while pid and Path(f'/proc/{pid}/cmdline').exists():
                raw=Path(f'/proc/{pid}/cmdline').read_bytes()
                if not raw: break
                assert str(folder).encode() in raw, 'PID identity mismatch'
                time.sleep(2)
            read_results(folder)
            with lock:
                job['status']='completed'; job['adopted']=True; write(output/'status.json',state)
            return
        with (output/f'{name}_{chassis}_seed{seed}.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,cwd=SOURCES[name])
            with lock:
                job['pid']=process.pid; write(output/'status.json',state)
            code=process.wait()
        job['returncode']=code
        if code: raise RuntimeError(f'{name}/{chassis}/seed{seed} failed: {code}')
        read_results(folder)
        with lock:
            job['status']='completed'; write(output/'status.json',state)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute,chassis,seed,name) for chassis in ('s','xl') for seed in range(3) for name in RUNS]
            for future in as_completed(futures):
                future.result()
                complete=[s for s in ('s','xl') if sum(j['status']=='completed' and j['chassis']==s for j in state['jobs'])==6]
                if complete: aggregate(output,['id',*complete])
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error)); raise
    finally:
        write(output/'status.json',state)


if __name__=='__main__': main()
