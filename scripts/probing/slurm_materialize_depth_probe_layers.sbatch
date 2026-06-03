#!/bin/bash
#SBATCH --job-name=vlm3r_depth_split_layers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --output=logs/depth_probe_materialize_%x_%j.out
#SBATCH --error=logs/depth_probe_materialize_%x_%j.err

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  export PATH="$CONDA_BASE/bin:$PATH"
fi
set +u
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK:-/leonardo_work/EUHPC_D32_006}/probing_experiment}"
MODEL_LABELS="${MODEL_LABELS:-zero_spatial,vlm3r_baseline}"
FEATURE_LEVELS="${FEATURE_LEVELS:-layer_0,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27}"
MATERIALIZE_EXTRA_ARGS="${MATERIALIZE_EXTRA_ARGS:-}"

mkdir -p logs

cmd=(python -u scripts/probing/materialize_depth_probe_layers.py
  --output-root "${OUTPUT_ROOT}"
  --model-labels "${MODEL_LABELS}"
  --feature-levels "${FEATURE_LEVELS}")

if [[ -n "$MATERIALIZE_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($MATERIALIZE_EXTRA_ARGS)
  cmd+=("${extra_args[@]}")
fi

echo "[INFO] ${cmd[*]}"
"${cmd[@]}"
