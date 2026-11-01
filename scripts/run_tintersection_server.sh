#!/usr/bin/env bash
# Reproducible fresh training; the Python runner freezes code per experiment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
variant="both"
gpu="0"
output=""
steps="500000"
seed="0"
batch_size="16"
batch_length="64"
train_ratio="512"
context_swap_weight="1"
source_from="$repo_dir"
smoke="false"
benchmark="false"
dry_run="false"

while (($#)); do
  case "$1" in
    --variant) variant="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --batch-length) batch_length="$2"; shift 2 ;;
    --train-ratio) train_ratio="$2"; shift 2 ;;
    --context-swap-weight) context_swap_weight="$2"; shift 2 ;;
    --source-from) source_from="$2"; shift 2 ;;
    --smoke) smoke="true"; shift ;;
    --benchmark) benchmark="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    --help|-h)
      echo "Usage: $0 --output FRESH_DIRECTORY [--variant both|baseline|shallow] [--gpu 0] [--steps 500000] [--seed 0] [--batch-size 16] [--batch-length 64] [--train-ratio 512] [--context-swap-weight 1] [--source-from FROZEN_SOURCE] [--smoke|--benchmark] [--dry-run]"
      echo "Optional environment: PYTHON_BIN, METADRIVE_ROOT. One GPU: both sequentially; separate GPUs: two separate variants/outputs."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be positive" >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be nonnegative" >&2; exit 2; }
for positive in "$batch_size" "$batch_length" "$train_ratio"; do
  [[ "$positive" =~ ^[1-9][0-9]*$ ]] || { echo "Batch and ratio values must be positive integers" >&2; exit 2; }
done
[[ "$smoke" != "true" || "$benchmark" != "true" ]] || { echo "--smoke and --benchmark cannot be combined" >&2; exit 2; }
[[ "$context_swap_weight" == "0" || "$context_swap_weight" == "1" ]] || {
  echo "--context-swap-weight must be 0 or 1 for the paper comparison" >&2; exit 2;
}
[[ -d "$source_from/configs" && -f "$source_from/train.py" ]] || {
  echo "--source-from must point to a source directory with train.py and configs/" >&2; exit 2;
}
source_from="$(cd -- "$source_from" && pwd)"
[[ "$variant" == "both" || "$variant" == "baseline" || "$variant" == "shallow" ]] || {
  echo "--variant must be both, baseline or shallow" >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -n "${METADRIVE_ROOT:-}" ]]; then
  [[ -d "$METADRIVE_ROOT/metadrive" ]] || { echo "METADRIVE_ROOT must contain metadrive/" >&2; exit 2; }
  export PYTHONPATH="$METADRIVE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
cd -- "$repo_dir"
"$python_bin" "$script_dir/server_preflight.py"
command=("$python_bin" -u "$script_dir/train_metadrive_compare.py"
  --logdir "$output" --source-from "$source_from" --methods r2dreamer
  --map-block T --chassis-randomization --mass-friction-randomization --steps "$steps" --seed "$seed"
  --batch-size "$batch_size" --batch-length "$batch_length" --train-ratio "$train_ratio")
if [[ "$variant" != "baseline" ]]; then
  command+=(--context-variants dali_shallow --context-swap-weight "$context_swap_weight")
fi
if [[ "$variant" == "both" ]]; then command+=(--include-baseline); fi
if [[ "$smoke" == "true" ]]; then command+=(--smoke); elif [[ "$benchmark" != "true" ]]; then command+=(--evaluate-best); fi
if [[ "$dry_run" == "true" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
