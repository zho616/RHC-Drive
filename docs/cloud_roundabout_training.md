# 云服务器运行当前环岛消融

入口：`scripts/train_roundabout_ablation.py`。完整配方已保存于
`configs/metadrive_roundabout_500k.yaml`，不依赖本机 `logdir`、旧权重或绝对路径。
它与本机五组消融的生产配置逐字段核验一致（仅输出路径不同）。
本机正在运行的训练继续使用自己的冻结源码；Git 更新不改变它。

## 安装

使用 Python 3.11 和支持 CUDA 12.8 的 NVIDIA 驱动。当前配方使用约 32GB 主存；
每个实验保留 8 个训练环境，不能为了并发悄悄改小 batch 或环境数。

```bash
git clone https://github.com/zho616/RHC-Drive.git
cd RHC-Drive
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[metadrive]'
python scripts/server_preflight.py
```

模型依赖见 `pyproject.toml`，包括 torch2.8.0、torchrl0.9.2、tensordict0.9.1、
numpy1.26.0、gymnasium1.2.0；物理引擎固定 Panda3D1.10.16。
MetaDrive assets 必须存在；若安装包未包含 assets，先按 MetaDrive 安装流程准备好，
或直接使用下一节的本机运行时包。

## 对齐实际模拟器

相同版本号不保证实际源码相同。本机 MetaDrive0.4.3 使用源码快照
`f0a18fdf55c720e6e3023e97fae8b76beb652846`，该 revision 标识原训练机的源码快照，并非可从上游直接安装的提交。实际源码哈希已记录在仓库：

```bash
python scripts/export_metadrive_runtime.py --check configs/metadrive_runtime_reference.json
```

若核验不通过，使用本机导出的 `metadrive-runtime-20261101.tar.gz`，该包包含当前模拟器源码和
assets，单独传输，不放进 Git 仓库。已在本机 `deployment_bundle/` 准备好。
可用自己的 SSH 地址传输，例如在本机执行：

```bash
scp deployment_bundle/metadrive-runtime-20261101.tar.gz user@YOUR_SERVER:/data/
```

在服务器解压、安装并核验：

```bash
mkdir -p /data/matched-simulator
tar -xzf /data/metadrive-runtime-20261101.tar.gz -C /data/matched-simulator
python -m pip install -e /data/matched-simulator/metadrive-runtime
python scripts/export_metadrive_runtime.py --check configs/metadrive_runtime_reference.json
python scripts/server_preflight.py
```

如需重新导出，在原训练机运行
`python scripts/export_metadrive_runtime.py --output deployment_bundle/another-runtime.tar.gz`。
输出旁的 JSON 包含归档 SHA256。该脚本只打包，不上传文件。

## 按方法和 seed 分配训练

| 方法参数 | 实验 |
|---|---|
| `history_only` | 消融1，WM 损失训练编码器，无响应/交换 |
| `response_only` | 消融2，H5响应预测，无交换 |
| `full_h1` | 消融3，H1响应预测＋交换，输入历史仍K5 |
| `oracle` | 消融4，10项真实车辆参数直接注入 |
| `raw_history` | 消融5，5×6归一化历史＋5维mask直接注入 |
| `full_h5` | Ours，完整方法 |
| `baseline` | 无context的R2Dreamer |

例如让云服务器承担 seed2 五项，在 tmux 中执行：

```bash
tmux new -s rhc-seed2
source .venv/bin/activate
python scripts/train_roundabout_ablation.py --seeds 2 \
  --methods history_only response_only full_h1 oracle raw_history \
  --gpu 0 --output logdir/cloud_roundabout_seed2
```

同一命令中的任务按 seed、方法顺序串行运行。默认是 seed2 的五项消融。
两张卡可在两个 tmux 窗口分派不同任务（输出目录必须不同）：

```bash
python scripts/train_roundabout_ablation.py --seeds 2 \
  --methods history_only response_only full_h1 --gpu 0 \
  --output logdir/cloud_seed2_gpu0
python scripts/train_roundabout_ablation.py --seeds 2 \
  --methods oracle raw_history --gpu 1 \
  --output logdir/cloud_seed2_gpu1
```

云端分配方案不会自动停止本机同名任务。正式启动前明确两台机器各自负责的 seed/方法，
避免把重复任务的结果混在一起。当前本机队列没有因为部署代码而取消任何任务。

先核对命令而不创建文件：

```bash
python scripts/train_roundabout_ablation.py --seeds 2 --methods oracle raw_history \
  --output logdir/check_only --dry-run
```

CPU 功能检查使用新目录，配置会缩小，结果不可当作正式实验：

```bash
python scripts/train_roundabout_ablation.py --seeds 0 --methods oracle raw_history \
  --output logdir/smoke_roundabout --smoke
```

正式配置固定500K、8环境、batch16×64、ratio512、CPU replay500K；每10K验证20回合。
训练车型DEFAULT/M/L，质量与摩擦倍率0.6–1.4（步长0.1），转角倍率1。
训练地图0–99、验证10000–10019；seed偏移自动补偿。
每项训练后自动用验证选出的best，在固定ID/S/XL各测试100回合，environment_seed=0。
所有条件按相同地图/底盘采样测试，不按测试表现选择checkpoint。

## 查看与回传

输出包含冻结 `source/`、配置、模拟器源码哈希、状态、best/latest、TensorBoard和逐回合测试。
训练和测试失败会停止该队列；部分训练目录不会自动从头覆盖。
`--prepare-only` 可只保存任务，之后用 `--run-prepared --output PATH` 执行；
这个入口不会对中断的部分训练做伪续训。

```bash
tensorboard --logdir logdir/cloud_roundabout_seed2/training --host 127.0.0.1 --port 6006
# 从自己的电脑连接，按实际 SSH 地址替换：
ssh -N -L 6006:127.0.0.1:6006 user@YOUR_SERVER
```

浏览器访问 `http://127.0.0.1:6006`。独立测试汇总位于 `tables/test_results.csv`，
逐回合文件在 `evaluation/<job>/<ID|S|XL>/episodes.jsonl`。
回传每项的配置、source哈希、best.pt、metrics.jsonl、summary.json和episodes.jsonl；
跨服务器汇总时按训练 seed 等权统计，不能把测试回合数当作训练 seed 数。
