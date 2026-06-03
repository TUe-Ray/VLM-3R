#!/bin/bash
#SBATCH --job-name=spat_perturb_cat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset/logs/%x_%j.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset/logs/%x_%j.err

set -euo pipefail

if [[ -z "${CATEGORY:-}" ]]; then
  echo "[ERROR] CATEGORY must be exported, e.g. CATEGORY=room_size"
  exit 1
fi

export CATEGORIES="$CATEGORY"
export OUTPUT_DIR="${OUTPUT_DIR:-/leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset/by_category/$CATEGORY}"
export NUM_PER_CATEGORY="${NUM_PER_CATEGORY:-20}"

mkdir -p /leonardo_scratch/fast/EUHPC_D32_006/diag/spatial_perturbation_train_subset/logs "$OUTPUT_DIR"

/leonardo/home/userexternal/shuang00/VLM-3R/scripts/run_spatial_perturbation_diagnostic.sh
