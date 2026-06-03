#!/bin/bash
#SBATCH --job-name=vlm3r_depth_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=logs/depth_probe_train_%x_%j.out
#SBATCH --error=logs/depth_probe_train_%x_%j.err

set -euo pipefail


CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
module load cuda/12.2
if [[ -z "${CUDA_HOME:-}" && -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="$EBROOTCUDA"
fi
if [[ -z "${CUDA_HOME:-}" && -x "$(command -v nvcc)" ]]; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  export PATH="$CONDA_BASE/bin:$PATH"
fi
set +u
conda activate "$CONDA_ENV"
set -u
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK:-/leonardo_work/EUHPC_D32_006}/probing_experiment}"
MODEL_LABELS="${MODEL_LABELS:-zero_spatial,vlm3r_baseline,geo_rope_fusion}"
DEVICE="${DEVICE:-cuda:0}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm3r-depth-probes}"
WANDB_DIR="${WANDB_DIR:-${WORK:-/leonardo_work/EUHPC_D32_006}/wandb}"
WANDB_RUN_TIME="${WANDB_RUN_TIME:-$(date +%Y%m%d_%H%M%S)}"
WANDB_RUN_NAME_DEFAULT="${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-manual}"
WANDB_RUN_NAME_DEFAULT="${WANDB_RUN_NAME_DEFAULT//[^A-Za-z0-9_.-]/_}"
WANDB_RUN_ID_DEFAULT="${WANDB_RUN_TIME}_${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-manual}"
WANDB_RUN_ID_DEFAULT="${WANDB_RUN_ID_DEFAULT//[^A-Za-z0-9_.-]/_}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$WANDB_RUN_ID_DEFAULT}"
WANDB_NAME="${WANDB_NAME:-$WANDB_RUN_NAME_DEFAULT}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FEATURE_LEVELS="${FEATURE_LEVELS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

mkdir -p logs

cmd=(python -u scripts/probing/train_depth_probes.py
  --output-root "${OUTPUT_ROOT}" \
  --sample-indices "${OUTPUT_ROOT}/sample_indices.json" \
  --model-labels "${MODEL_LABELS}" \
  --device "${DEVICE}")

if [[ -n "$FEATURE_LEVELS" ]]; then
  cmd+=(--feature-levels "$FEATURE_LEVELS")
fi
if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
  cmd+=(--skip-existing)
fi

if [[ "$WANDB_ENABLE" == "1" || "$WANDB_ENABLE" == "true" ]]; then
  cmd+=(--wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-dir "${WANDB_DIR}"
    --wandb-id "${WANDB_RUN_ID}"
    --wandb-name "${WANDB_NAME}"
  )
fi
if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=($TRAIN_EXTRA_ARGS)
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"

python scripts/probing/summarize_depth_probe_results.py \
  --output-root "${OUTPUT_ROOT}"
