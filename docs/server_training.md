# 服务器环境与训练复现

当前仅整理并上传代码和实验记录，没有在新服务器启动训练。
此说明提供后续从头训练的入口，不是断点续训迁移；权重和 replay 不在 Git 中。

## 环境

使用 Linux、Python 3.11、支持 PyTorch 2.8 CUDA 构建的 NVIDIA 驱动。
本机参考环境为 RTX 5060 Ti 16 GB、约 32 GB 主存，replay 放在 CPU。
精确版本见 `experiments/runtime_reference.json`。

```bash
git clone https://github.com/zho616/RHC-Drive.git
cd RHC-Drive
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[metadrive]'
python scripts/server_preflight.py
```

本机 MetaDrive 来自独立源码目录，版本声明 0.4.3，提交为
`f0a18fdf55c720e6e3023e97fae8b76beb652846`，工作区无修改。
该目录没有配置 Git remote，不能保证 PyPI 同名版本与本机源码逐行相同。
要求严格复现时应另行复制这份 MetaDrive 源码及其 assets（不包含在本仓库），
并设置 `METADRIVE_ROOT=/path/to/metadrive`。预检查只能核对依赖、CUDA、assets 和
T 路口接口，不能替代源码一致性检查与 smoke test。

## 验证和后续启动命令

先做只读预检查并打印启动参数，以下命令不会启动训练：

```bash
bash scripts/run_tintersection_server.sh --output logdir/T_check --dry-run
```

准备正式训练时先运行 smoke（会进行很短的训练，须使用独立新目录）：

```bash
bash scripts/run_tintersection_server.sh --output logdir/T_smoke --smoke
```

通过后，单卡依次训练 baseline 和 11，每个 500K step，并分别执行 best checkpoint 评估：

```bash
bash scripts/run_tintersection_server.sh --output logdir/T_500k --variant both
```

多卡时可以分别启动 `--variant baseline --gpu 0` 和 `--variant shallow --gpu 1`，
使用不同 output。所有 output 都应是新目录。启动器默认使用当前环境的 `python`，
可通过 `PYTHON_BIN` 指定。正式任务建议放在 tmux 中，避免 SSH 断开中止。
不要将两张卡的日志目录写到同一路径。

配置：R2Dreamer、原生 T 路口、traffic_density=0.2、8 个训练环境、seed=0、
train_ratio=512、CPU replay 500K、batch 16×64。质量及 wheel_friction
倍率分别为 0.6–1.4、步长 0.1，最大前轮转角不随机化。
11 使用 state4、K=5、context_dim=8、DALI shallow posterior 注入、aux=1、swap=1。
baseline 关闭 context。runner 会在实验目录冻结当次源码。

```bash
tensorboard --logdir logdir/T_500k --port 6015 --bind_all
```

数值测试可以单独运行（需安装 pytest）：

```bash
python -m pytest tests/test_context.py tests/test_context_swap.py tests/test_chassis_friction.py tests/test_tintersection.py
```

分析绘图脚本还可能使用 matplotlib、scipy、scikit-learn；按对应脚本的输入要求安装。
本次上传前只验证了启动器语法、预检查与 dry-run，没有在另一台机器验证环境。
