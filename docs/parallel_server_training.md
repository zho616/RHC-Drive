# 新服务器部署与并行实验

本仓库包含模型、Context 编码／辅助预测／交换损失、物理底盘随机化、
nuPlan 环境、PPO/SAC、评估和训练配置。数据、权重、replay、视频和运行日志不随 Git 上传。
以下命令从头训练，使用不同 output 可在多台服务器或多张卡独立运行。

## 1. 安装

Linux，Python 3.11，支持 CUDA 12.8 版 PyTorch 的 NVIDIA 驱动。
当前实测环境为 16GB 显存、约32GB内存；并发多个实验需要相应增加内存和 CPU。

```bash
git clone https://github.com/zho616/context-dreamer.git
cd context-dreamer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[metadrive,sb3]'
python scripts/server_preflight.py
```

依赖主版本见 `pyproject.toml`。如果服务器的 PyTorch 安装不是 CUDA 构建，
先按 PyTorch 官方安装说明安装与驱动匹配的 `torch==2.8.0`。
不要直接复制本机的 conda 环境目录。

## 2. 传输匹配的模拟器和 nuPlan 数据

当前 MetaDrive 为0.4.3，本机源码提交
`f0a18fdf55c720e6e3023e97fae8b76beb652846`，该提交是本地源码快照。
PyPI 同名版本不保证逐行一致。严格对齐时，复制当前服务器的完整 MetaDrive
源码目录（含 `metadrive/assets`），在新服务器运行：

```bash
python -m pip install -e /path/to/copied/metadrive
python scripts/server_preflight.py
```

nuPlan 复制已经转换好的 Pittsburgh 数据目录，保留完整子目录和索引文件：

```text
scenarionet_pittsburgh_8000/
  train/     # 8000 个训练场景
  val/       # 100 个验证场景
  test/      # 100 个独立测试场景
```

每个 split 的 `dataset_summary.pkl`、mapping 文件及全部场景文件必须一起传输。
训练使用转换后的 ScenarioNet 数据，不需要在新服务器重新转换原始数据库。
数据不通过 GitHub 分发。

## 3. nuPlan：1M步，每5000步验证100回合

```bash
export NUPLAN_DATA_ROOT=/data/nuplan/scenarionet_pittsburgh_8000

# 先检查命令，不创建输出、不启动训练。
python scripts/train_server.py --task nuplan --method r2dreamer_context \
  --output logdir/nuplan_context_seed0 --gpu 0 --dry-run

# 短 CPU 功能验证；输出目录必须与正式实验不同。
python scripts/train_server.py --task nuplan --method r2dreamer_context \
  --output logdir/nuplan_context_smoke --smoke

# 正式实验：放在 tmux 中运行，SSH断开后仍继续。
python scripts/train_server.py --task nuplan --method r2dreamer_context \
  --output logdir/nuplan_context_seed0 --gpu 0
```

方法选项：`r2dreamer_context`、`r2dreamer`、`dreamerv3`、
`dreamerv3_context`、`ppo`、`sac`。可用 `--seed`、`--steps`、`--eval-every`、
`--eval-episodes` 显式设置。nuPlan默认就是 seed0、1M步、验证间隔5000步、每次100回合。

两张 GPU 在两个 tmux 窗口分别启动，例如：

```bash
python scripts/train_server.py --task nuplan --method r2dreamer \
  --output logdir/nuplan_r2_seed0 --gpu 0
python scripts/train_server.py --task nuplan --method dreamerv3 \
  --output logdir/nuplan_dv3_seed0 --gpu 1
```

不同服务器也使用相同命令和相同数据版本。避免在同一路径写入多个实验。
每个 launcher 会冻结当次代码及配置到自己的 output，结果位于 `output/training`。
`--smoke` 使用小模型、少量步骤和缩小的随机化池，只验证运行链路，不能用于论文比较。

## 4. 环岛B组和 T-intersection

```bash
python scripts/train_server.py --task roundabout --method dreamerv3_context \
  --output logdir/roundabout_dv3_context_seed0 --gpu 0
python scripts/train_server.py --task tintersection --method r2dreamer_context \
  --output logdir/tintersection_context_seed0 --gpu 1
```

这两个任务默认50万步，每1万步验证20回合；可用参数覆盖。
三种场景均保留 Default/M/L 训练车型；质量和轮胎摩擦倍率0.6–1.4，步长0.1，
转向倍率固定1.0。Context使用state4、K5、8维编码、Shallow后验注入、aux1/swap1、
wm_grad_scale0，Actor/Critic不直接拼接context。nuPlan主观测271维，程序化道路259维。
PPO/SAC只接收同一环境的原生状态向量。

## 5. TensorBoard 与独立测试

```bash
tensorboard --logdir logdir --host 127.0.0.1 --port 6006
```

查看 `episode/eval_score`、`episode/eval_success`、`episode/eval_route_completion`。
它们是训练期间验证指标；最终独立测试使用验证集选出的 `best`。
五千步一次的验证点较稀疏，绘图建议保留原始点，并明确平滑系数。

Dreamer系列 nuPlan 单次100回合独立测试：

```bash
python scripts/evaluate_metadrive.py \
  --checkpoint logdir/nuplan_context_seed0/training/best.pt \
  --source-root logdir/nuplan_context_seed0/source \
  --scenario-dir "$NUPLAN_DATA_ROOT/test" \
  --output logdir/nuplan_context_seed0/test_seed0 \
  --episodes 100 --seed-start 0 --inference-seed-offset 12345
```

其余两组推理种子将 offset 改为112345和212345，并使用独立输出目录。
OOD加 `--eval-models s` 或 `--eval-models xl`，保留质量与摩擦倍率范围。
PPO/SAC训练脚本在训练完成后自动测试best的ID、S、XL，各100回合。
测试次数不是独立训练种子数。
