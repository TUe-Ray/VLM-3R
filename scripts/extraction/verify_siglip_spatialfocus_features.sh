#!/usr/bin/env bash
#SBATCH --job-name=SIGLIP_sf_verify
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/extraction/%x_%j.out
#SBATCH --error=logs/extraction/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"

conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py verify-all \
  --manifest "$FAST_DATA_ROOT/siglip_features_dec_m2_training_index.json" \
  --output-root "$FAST_DATA_ROOT" \
  --run-id "verify-final-${SLURM_JOB_ID}"
