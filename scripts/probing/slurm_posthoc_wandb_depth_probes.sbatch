#!/bin/bash
#SBATCH --job-name=vlm3r_depth_wandb_posthoc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --output=logs/depth_probe_wandb_posthoc_%x_%j.out
#SBATCH --error=logs/depth_probe_wandb_posthoc_%x_%j.err

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

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-${SCRATCH:-/leonardo_scratch/large/userexternal/shuang00}/probing_experiment}"
WORK_OUTPUT_ROOT="${WORK_OUTPUT_ROOT:-${WORK:-/leonardo_work/EUHPC_D32_006}/probing_experiment}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm3r-depth-probes}"
WANDB_DIR="${WANDB_DIR:-${WORK:-/leonardo_work/EUHPC_D32_006}/wandb}"
WANDB_NAME="${WANDB_NAME:-depth-probes-posthoc-${SLURM_JOB_ID:-manual}}"

mkdir -p logs "$WORK_OUTPUT_ROOT" "$WANDB_DIR"

rsync -a --partial "$SCRATCH_OUTPUT_ROOT/" "$WORK_OUTPUT_ROOT/"

python scripts/probing/wandb_posthoc_depth_probes.py \
  --output-root "$WORK_OUTPUT_ROOT" \
  --project "$WANDB_PROJECT" \
  --wandb-dir "$WANDB_DIR" \
  --name "$WANDB_NAME"
