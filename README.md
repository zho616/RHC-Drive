# Context-Dreamer experiments

This fork adds vehicle dynamics context learning and chassis randomization to R2-Dreamer.
See the [experiment archive](experiments/README.md), [server reproduction guide](docs/server_training.md),
and [current T-intersection setup](docs/context_tintersection_mass_friction_500k.md).
The upstream implementation and credits are retained below.

For the current nuPlan / roundabout / T-intersection core code and independent
multi-server or multi-GPU runs, see [parallel server setup](docs/parallel_server_training.md).
The portable entry point is `python scripts/train_server.py --help`.

For the **current roundabout five-way ablations**, Full H5 and the no-context
baseline, use `python scripts/train_roundabout_ablation.py --help` and follow
[cloud training with the matched 500K recipe](docs/cloud_roundabout_training.md).
This entry point needs no local experiment archives or checkpoints.

## R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides the official implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a decoder-free, augmentation-free world model with redundancy-reducing regularization on latent representations. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

Installing via a virtual env like [uv](https://docs.astral.sh/uv/) is recommended. If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Select the environments you need, or use "all" to install everything.
# Available extras: dmc, atari, crafter, metaworld, metadrive, memorymaze, isaaclab, all
pip install -e ".[dmc,metaworld]"
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

MetaDrive roundabout training with LiDAR, ego state, and navigation input:

```bash
conda activate r2dreamer
python train.py --config-name metadrive
```

See [the MetaDrive guide](docs/metadrive.md) for configuration, smoke tests, and monitoring.
Optional K5 ego-dynamics context for R2Dreamer/DreamerV3: [migration guide](docs/context_dreamer.md).
Three-model chassis randomization with integer mass/steering grids: [training guide](docs/chassis_randomization.md).

For interactive 3D/top-down policy testing and custom model-loading interfaces,
see [the rendering test guide](docs/metadrive_render.md).

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | State / Image | Continuous | 0.5M–1M | GPU-parallelized robotic simulation. |
| [MetaDrive](https://github.com/metadriverse/metadrive) | LiDAR + State + Navigation | Continuous | 500K | Single-agent roundabout driving with background traffic. |

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Troubleshooting

If you encounter installation or runtime issues, please refer to our [Troubleshooting Guide](docs/troubleshooting.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
