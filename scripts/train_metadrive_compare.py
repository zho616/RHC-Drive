"""Run R2-Dreamer then this repository's DreamerV3 reproduction from one snapshot."""

import argparse
import datetime
import importlib.metadata
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


def now():
    return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True, type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=500000, help="Environment-step budget per method")
    parser.add_argument("--seed", type=int, default=0, help="Independent training seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Sequences per training batch")
    parser.add_argument("--batch-length", type=int, default=64, help="Sequence length per training batch")
    parser.add_argument("--train-ratio", type=int, default=512, help="Target replay training ratio")
    parser.add_argument("--smoke", action="store_true", help="Run a small validation of both methods")
    parser.add_argument("--methods", nargs="+", choices=("r2dreamer", "dreamerv3"),
                        default=["r2dreamer", "dreamerv3"])
    parser.add_argument("--context-comparison", action="store_true", help="Run baseline and context for each selected method")
    parser.add_argument("--context-only", action="store_true", help="Train only context variants using recurrent concatenation")
    parser.add_argument("--context-swap-weight", type=float, default=0.0,
                        help="One-step same-chassis cross-episode exchange loss weight")
    parser.add_argument("--context-wm-grad-scale", type=float, default=0.0,
                        help="Selected WM gradient multiplier into the shallow context encoder only")
    parser.add_argument("--map-block", choices=("O", "T"), default="O",
                        help="Native MetaDrive junction block; defaults to the existing roundabout")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Train an unconditioned baseline before the selected --context-variants")
    parser.add_argument("--context-variants", nargs="+", choices=("adarms_deter", "recurrent_actor_critic", "dali_deep", "dali_deep_joint", "dali_shallow"),
                        help="Run the requested R2Dreamer/DreamerV3 context variants sequentially")
    parser.add_argument("--chassis-randomization", action="store_true", help="Use three training chassis with discrete mass/steering scales")
    parser.add_argument("--mass-friction-randomization", action="store_true",
                        help="Replace steering variation with mass/friction scales0.6..1.4, step0.1")
    parser.add_argument("--reward-mode", choices=("additive", "metadrive", "paper"), default="paper")
    parser.add_argument("--ablations", nargs="+", choices=("ratio256", "buffer200k"))
    parser.add_argument("--source-from", type=pathlib.Path, help="Copy a prior experiment's frozen source")
    parser.add_argument("--after-status", type=pathlib.Path, help="Wait for this prerequisite queue to complete")
    parser.add_argument("--evaluate-best", action="store_true", help="After each run, test best on 100 new maps, S and XL episodes each")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if min(args.batch_size, args.batch_length, args.train_ratio) <= 0:
        parser.error("--batch-size, --batch-length and --train-ratio must be positive")
    if not 0 <= args.context_swap_weight < float("inf"):
        parser.error("--context-swap-weight must be finite and nonnegative")
    if not 0 <= args.context_wm_grad_scale < float("inf"):
        parser.error("--context-wm-grad-scale must be finite and nonnegative")
    if args.context_wm_grad_scale and (args.methods != ["r2dreamer"] or args.context_variants != ["dali_shallow"]):
        parser.error("WM context gradients require --methods r2dreamer --context-variants dali_shallow")
    if args.context_swap_weight and not ((args.context_only or args.context_variants) and args.chassis_randomization):
        parser.error("Context exchange requires a context variant and --chassis-randomization")
    if args.context_only and (args.context_comparison or args.ablations):
        parser.error("--context-only cannot be combined with --context-comparison or --ablations")
    if args.context_variants and (args.context_only or args.context_comparison or args.ablations):
        parser.error("--context-variants cannot be combined with other variant selectors")
    if args.context_variants and len(set(args.context_variants)) != len(args.context_variants):
        parser.error("Do not repeat a context variant")
    if args.include_baseline and not args.context_variants:
        parser.error("--include-baseline requires --context-variants")
    if args.mass_friction_randomization and not args.chassis_randomization:
        parser.error("--mass-friction-randomization requires --chassis-randomization")
    if args.evaluate_best and args.smoke:
        parser.error("Do not use --evaluate-best with --smoke")
    root = pathlib.Path(__file__).resolve().parents[1]
    logdir = args.logdir.expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    if (logdir / "status.json").exists() or (logdir / "source").exists():
        parser.error("Use a fresh log directory; this experiment already exists")
    source = logdir / "source"
    source.mkdir()
    source_root = args.source_from.resolve() if args.source_from else root
    for path in source_root.glob("*.py"):
        shutil.copy2(path, source / path.name)
    for name in ("configs", "envs", "optim"):
        shutil.copytree(source_root / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source_root / "pyproject.toml", source / "pyproject.toml")
    if args.evaluate_best:
        shutil.copy2(pathlib.Path(__file__).resolve().parent / "evaluate_metadrive.py", logdir / "evaluate_metadrive.py")

    if len(set(args.methods)) != len(args.methods):
        parser.error("Do not repeat a method")
    methods = [(name, "dreamer" if name == "dreamerv3" else "r2dreamer") for name in args.methods]
    variant_overrides = {}
    if args.ablations:
        if len(set(args.ablations)) != len(args.ablations):
            parser.error("Do not repeat an ablation")
        methods = [(f"dreamerv3_{name}", "dreamer") for name in args.ablations]
        for name in args.ablations:
            variant_overrides[f"dreamerv3_{name}"] = (
                ["env.train_ratio=256"] if name == "ratio256" else ["buffer.max_size=200000"]
            )
    if args.context_comparison:
        if args.ablations:
            parser.error("--context-comparison cannot be combined with --ablations")
        if not (source / "context.py").exists():
            parser.error("Selected source snapshot does not include context support")
        methods = [(label, rep) for name, rep in methods for label in (name, name + "_context")]
        for name, _ in methods:
            variant_overrides[name] = [f"model.context.enabled={str(name.endswith('_context')).lower()}"]
    if args.context_only:
        if not (source / "context.py").exists():
            parser.error("Selected source snapshot does not include context support")
        methods = [(name + "_context", rep) for name, rep in methods]
        for name, _ in methods:
            variant_overrides[name] = ["model.context.enabled=true", "model.context.injection=recurrent_concat"]
    if args.context_variants:
        methods = [(label, rep) for name, rep in methods
                   for label in ([name] if args.include_baseline else [])
                   + [name + "_" + variant for variant in args.context_variants]]
        for name, _ in methods:
            if name in args.methods:
                variant_overrides[name] = ["model.context.enabled=false", "model.context.swap_weight=0.0",
                                           "model.context.actor_critic=false"]
                continue
            adarms = name.endswith("_adarms_deter")
            joint = name.endswith("_dali_deep_joint")
            shallow = name.endswith("_dali_shallow")
            deep = name.endswith("_dali_deep") or joint
            variant_overrides[name] = ["model.context.enabled=true",
                f"model.context.injection={'dali_shallow' if shallow else 'dali_deep_joint' if joint else 'dali_deep' if deep else 'adarms_deter' if adarms else 'recurrent_concat'}",
                f"model.context.actor_critic={str(not (adarms or shallow)).lower()}"]
    if args.context_wm_grad_scale:
        variant_overrides["r2dreamer_dali_shallow"].append(f"model.context.wm_grad_scale={args.context_wm_grad_scale}")
    lateral = 0.2 if args.reward_mode == "additive" else 0.0
    common = [f"env.steps={args.steps}", f"env.train_ratio={args.train_ratio}",
              f"batch_size={args.batch_size}", f"batch_length={args.batch_length}",
              "buffer.max_size=500000",
              "buffer.storage_device=cpu", "env.env_num=8", "env.eval_episode_num=20", "env.eval_env_num=2",
              f"env.reward.mode={args.reward_mode}", f"env.reward.lateral={lateral}",
              f"seed={args.seed}", "trainer.save_best=true"]
    if args.map_block != "O":
        common += [f"env.map_block={args.map_block}", "env.task=metadrive_tintersection"]
    if args.chassis_randomization:
        if not (source / "envs/chassis.py").exists():
            parser.error("Selected source snapshot does not support chassis randomization")
        common += ["env.chassis.enabled=true"]
    # A frozen source may default to swap=1; an explicit zero must still override it.
    if args.context_variants or args.context_swap_weight:
        common += [f"model.context.swap_weight={args.context_swap_weight}"]
    if args.mass_friction_randomization:
        scales = "[0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]"
        common += [f"env.chassis.mass_scales={scales}", "env.chassis.steering_scales=[1.0]",
                   f"env.chassis.friction_scales={scales}"]
    if args.smoke:
        common += ["env.steps=96", "env.env_num=2", "env.eval_episode_num=4", "env.time_limit=16",
                   "batch_size=4", "batch_length=8", "env.train_ratio=8", "buffer.max_size=512",
                   "model.compile=false", "trainer.eval_every=48", "trainer.update_log_every=32"]
        if args.context_swap_weight:
            # Smoke only: guarantee cross-episode profile matches to exercise gradients.
            common += ["env.chassis.models=[default]", "env.chassis.mass_scales=[1.0]",
                       "env.chassis.steering_scales=[1.0]"]
            if args.mass_friction_randomization:
                common += ["env.chassis.friction_scales=[1.0]"]
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchrl", "tensordict", "metadrive-simulator", "gymnasium", "numpy")}
    status = {"runner_pid": os.getpid(), "created_at": now(), "phase": "queued", "smoke": args.smoke,
              "source": str(source), "python": sys.executable, "versions": versions,
              "source_from": str(source_root), "after_status": str(args.after_status) if args.after_status else None,
              "common_overrides": common,
              "methods": {name: {"state": "queued", "variant_overrides": variant_overrides.get(name, [])}
                          for name, _ in methods}}
    status_path = logdir / "status.json"
    write_json(status_path, status)
    # Text summaries make both runs visible before the queued method starts.
    from torch.utils.tensorboard import SummaryWriter
    from omegaconf import OmegaConf

    reward_config = OmegaConf.load(source / "configs/env/metadrive_lidar.yaml").reward

    def update_text(name, state):
        with SummaryWriter(str(logdir / name)) as writer:
            writer.add_text("experiment/status", state, 0)
            writer.add_text("experiment/reward",
                            f"mode={args.reward_mode}; progress={reward_config.progress}; "
                            f"speed={reward_config.speed}; lateral={lateral}; "
                            f"success={reward_config.success}; crash=-{reward_config.crash}; "
                            f"out_of_road=-{reward_config.out_of_road}", 0)
            writer.add_text("experiment/variant", str(variant_overrides.get(name, [])), 0)

    for name, _ in methods:
        update_text(name, "queued")
    current = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if current is not None and current.poll() is None:
            try:
                os.killpg(current.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.after_status:
            status["phase"] = "waiting_for_baseline"
            write_json(status_path, status)
            print(f"{now()} Waiting for {args.after_status}", flush=True)
            while not stopped:
                prerequisite = json.loads(args.after_status.read_text())
                if prerequisite["phase"] == "completed":
                    break
                if prerequisite["phase"] in ("failed", "interrupted") or not pathlib.Path(
                    f"/proc/{prerequisite['runner_pid']}"
                ).exists():
                    status.update(phase="blocked", error="Prerequisite did not complete successfully")
                    write_json(status_path, status)
                    return 1
                time.sleep(15)
        for name, rep_loss in methods:
            if stopped:
                break
            run = logdir / name
            command = [sys.executable, "-u", str(source / "train.py"), "--config-name", "metadrive",
                       *common, *variant_overrides.get(name, []), f"model.rep_loss={rep_loss}", f"logdir={run}"]
            status["phase"] = name
            record = status["methods"][name]
            record.update(state="starting", command=command, started_at=now(), logdir=str(run))
            write_json(status_path, status)
            print(f"{now()} Starting {name}: {run}", flush=True)
            update_text(name, "running")
            with (run / "launcher.log").open("w") as output:
                current = subprocess.Popen(command, cwd=source, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record.update(state="running", pid=current.pid)
                write_json(status_path, status)
                code = current.wait()
            record.update(returncode=code, finished_at=now())
            if code != 0 or stopped or not (run / "latest.pt").exists() or not (run / "best.pt").exists():
                record["state"] = "interrupted" if stopped else "failed"
                status["phase"] = record["state"]
                update_text(name, record["state"])
                write_json(status_path, status)
                print(f"{now()} {name} {record['state']}; next method will not start", flush=True)
                return 1
            record["state"] = "completed"
            import torch

            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            record["best_step"] = checkpoint["step"]
            record["best_eval_metrics"] = checkpoint["eval_metrics"]
            del checkpoint
            if args.evaluate_best:
                record["evaluations"] = {}
                for split, seed, pool in [("new_maps", 10020, []), ("s", 10320, ["s"]), ("xl", 10420, ["xl"])]:
                    if stopped:
                        break
                    status["phase"] = name + "/evaluation/" + split
                    eval_command = [sys.executable, "-u", str(logdir / "evaluate_metadrive.py"),
                        "--checkpoint", str(run / "best.pt"), "--source-root", str(source),
                        "--output", str(run / "evaluation" / split), "--episodes", "100", "--seed-start", str(seed)]
                    if pool:
                        eval_command += ["--eval-models", *pool]
                    evaluation = record["evaluations"][split] = {"state": "running", "command": eval_command}
                    write_json(status_path, status)
                    with (run / ("evaluation_" + split + ".log")).open("w") as output:
                        current = subprocess.Popen(eval_command, cwd=source, stdout=output, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        evaluation["pid"] = current.pid
                        write_json(status_path, status)
                        code = current.wait()
                    evaluation.update(returncode=code, state="completed" if code == 0 else "failed")
                    if code != 0 or stopped:
                        status["phase"] = "interrupted" if stopped else "failed"
                        write_json(status_path, status)
                        return 1
                    evaluation["summary"] = json.loads((run / "evaluation" / split / "summary.json").read_text())["summary"]
                    write_json(status_path, status)
            update_text(name, "completed")
            write_json(status_path, status)
            print(f"{now()} Completed {name}", flush=True)
        status["phase"] = "interrupted" if stopped else "completed"
        status["finished_at"] = now()
        write_json(status_path, status)
        return int(stopped)
    except Exception:
        status["phase"] = "failed"
        status["error"] = traceback.format_exc()
        write_json(status_path, status)
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGINT)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
