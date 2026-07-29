#!/bin/bash
#SBATCH --job-name=DBG_siglip_spatialfocus
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/extraction/%x_%j.out
#SBATCH --error=logs/extraction/%x_%j.err

set -euo pipefail
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
MANIFEST="${MANIFEST:-$FAST_DATA_ROOT/siglip_features_dec_m2_alignment.json}"
mkdir -p logs/extraction
srun --kill-on-bad-exit=1 --wait=30 conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py extract --manifest "$MANIFEST" --output-root "$FAST_DATA_ROOT" --run-id "$SLURM_JOB_ID" --fail-on-error
conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py summarize --manifest "$MANIFEST" --output-root "$FAST_DATA_ROOT" --run-id "$SLURM_JOB_ID"
