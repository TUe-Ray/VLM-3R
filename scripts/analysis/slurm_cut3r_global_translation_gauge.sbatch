#!/bin/bash
#SBATCH --job-name=cut3r_gauge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=8G
#SBATCH --output=logs/cut3r_gauge_%x_%j.out
#SBATCH --error=logs/cut3r_gauge_%x_%j.err

# Evaluation-only: load existing CUT3R sidecars and never run training or inference.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"
mkdir -p logs

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
if command -v module >/dev/null 2>&1; then
  module load cuda/12.2 || true
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

: "${POINTMAP_SIDECAR_ROOT:?Set POINTMAP_SIDECAR_ROOT to the existing CUT3R point-map sidecar directory.}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/cut3r_global_translation_gauge_${SLURM_JOB_ID:-manual}}"

python -u scripts/analysis/verify_cut3r_global_translation_gauge.py \
  --sidecar-root "$POINTMAP_SIDECAR_ROOT" \
  --num-samples 3 \
  --seed 0 \
  --output-dir "$OUTPUT_DIR"
