"""Portable, isolated fresh training for the paper's driving experiments."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_nuplan_roundabout_queue import fingerprint, freeze, job_config


METHODS = ('r2dreamer_context', 'r2dreamer', 'dreamerv3',
           'dreamerv3_context', 'ppo', 'sac')


def build_config(root, args):
    name = ('metadrive_nuplan_pittsburgh' if args.task == 'nuplan'
            else 'metadrive_tintersection_mass_friction')
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        reference = compose(config_name=name)
    reference = OmegaConf.create(OmegaConf.to_container(reference, resolve=False))
    context = args.method.endswith('_context')
    steps = args.steps or (1000000 if args.task == 'nuplan' else 500000)
    config = job_config(reference, args.output.resolve() / 'training', steps,
                        'r2dreamer' if args.method.startswith('r2dreamer') else 'dreamer', context)
    config.seed = args.seed
    # Keep train/validation maps fixed across independent training seeds.
    config.env.start_seed = int(reference.env.start_seed) + int(reference.env.seed) - args.seed
    config.env.eval_start_seed = int(reference.env.eval_start_seed) + int(reference.env.seed) - args.seed
    config.device = args.device
    config.trainer.eval_every = args.eval_every or (5000 if args.task == 'nuplan' else 10000)
    config.env.eval_episode_num = args.eval_episodes or (100 if args.task == 'nuplan' else 20)
    if args.task == 'roundabout':
        config.env.task = 'metadrive_roundabout'
        config.env.map_block = 'O'
    if args.task == 'nuplan':
        data = args.data_root or os.environ.get('NUPLAN_DATA_ROOT')
        if not data:
            raise ValueError('nuPlan requires --data-root or NUPLAN_DATA_ROOT (train/val/test folders)')
        data = Path(data).expanduser().resolve()
        for split, directory in (('train', 'train'), ('eval', 'val'), ('test', 'test')):
            config.env[f'scenario_{split}_dir'] = str(data / directory)
    if args.smoke:
        config.device = 'cpu'
        config.env.steps = 128
        config.env.env_num = 2
        config.env.eval_env_num = 1
        config.env.eval_episode_num = 2
        config.env.time_limit = 32
        config.env.train_ratio = 8
        config.buffer.max_size = 512
        config.batch_size = 4
        config.batch_length = 8
        config.model.compile = False
        config.model.deter = 32
        config.model.hidden = 16
        config.model.units = 16
        config.model.discrete = 4
        config.model.rssm.stoch = 4
        config.model.rssm.blocks = 4
        config.model.imag_horizon = 3
        config.trainer.eval_every = 64
        config.trainer.update_log_every = 32
        config.env.chassis.models = ['default']
        config.env.chassis.mass_scales = [1.]
        config.env.chassis.steering_scales = [1.]
        config.env.chassis.friction_scales = [1.]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('nuplan', 'roundabout', 'tintersection'), required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for this one experiment')
    parser.add_argument('--data-root', type=Path, help='Converted nuPlan dataset with train/val/test subfolders')
    parser.add_argument('--gpu', type=int, default=0, help='Physical GPU index, remapped to cuda:0')
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--smoke', action='store_true', help='Tiny CPU functional check, not a paper run')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved config/command; create no files')
    args = parser.parse_args()
    for key in ('steps', 'eval_every', 'eval_episodes'):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    try:
        config = build_config(root, args)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh directory to avoid mixing experiments')
    if args.method in ('ppo', 'sac') and not args.smoke:
        if config.env.steps % config.env.env_num:
            parser.error('SB3 step budget must be divisible by the number of training environments')
        if args.method == 'ppo' and config.env.steps % (1250 * config.env.env_num):
            parser.error('PPO budget must be divisible by 1250 * env_num (complete rollout updates)')
    source = output / 'source'
    if args.method in ('ppo', 'sac'):
        command = [sys.executable, '-u', str(source / 'scripts/train_metadrive_sb3.py'),
                   '--source', str(source), '--config', str(output / 'configs/experiment.yaml'),
                   '--output', str(output / 'training'), '--method', args.method,
                   '--steps', str(config.env.steps), '--eval-every', str(config.trainer.eval_every)]
        if args.smoke:
            command.append('--smoke')
    else:
        command = [sys.executable, '-u', str(source / 'train.py'),
                   '--config-path', str(output / 'configs'), '--config-name', 'experiment']
    print(OmegaConf.to_yaml(config, resolve=True))
    print('Command:', shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if args.task == 'nuplan':
        for key in ('scenario_train_dir', 'scenario_eval_dir', 'scenario_test_dir'):
            if not (Path(config.env[key]) / 'dataset_summary.pkl').is_file():
                parser.error(f'Missing converted nuPlan dataset: {config.env[key]}')
    output.mkdir(parents=True)
    freeze(root, source)
    (source / 'scripts').mkdir()
    for name in ('train_metadrive_sb3.py', 'evaluate_metadrive.py'):
        (source / 'scripts' / name).write_bytes((root / 'scripts' / name).read_bytes())
    (output / 'configs').mkdir()
    OmegaConf.save(config, output / 'configs/experiment.yaml')
    (output / 'source_hashes.json').write_text(json.dumps(fingerprint(source), indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.call(command, cwd=source, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
