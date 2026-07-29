#!/bin/bash
#SBATCH --job-name=SMOKE_siglip_spatialfocus
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
CUT3R_ROOT="${CUT3R_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
LOCAL_SIGLIP="${LOCAL_SIGLIP:-$MODEL_ROOT/siglip-so400m-patch14-384}"
MANIFEST="${MANIFEST:-$FAST_DATA_ROOT/siglip_features_dec_m2_alignment.json}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"

mkdir -p logs/extraction
conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py build-manifest \
  --manifest "$MANIFEST" --siglip-checkpoint "$LOCAL_SIGLIP" --vision-select-feature patch \
  --cut3r-layer-root "6=$CUT3R_ROOT" --cut3r-layer-root "9=$CUT3R_ROOT" --cut3r-layer-root "12=$FAST_DATA_ROOT" \
  --cut3r-subdir "6=spatial_features_dec_6" --cut3r-subdir "9=spatial_features_dec_9" --cut3r-subdir "12=spatial_features"
conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py validate-partition \
  --manifest "$MANIFEST"
srun --kill-on-bad-exit=1 --wait=30 conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py extract \
  --manifest "$MANIFEST" --output-root "$FAST_DATA_ROOT" --run-id "$SLURM_JOB_ID" --max-samples "$MAX_SAMPLES" --fail-on-error
conda run -n "$CONDA_ENV_NAME" python scripts/extraction/extract_siglip_spatialfocus_features.py summarize \
  --manifest "$MANIFEST" --output-root "$FAST_DATA_ROOT" --run-id "$SLURM_JOB_ID" --max-samples "$MAX_SAMPLES"
