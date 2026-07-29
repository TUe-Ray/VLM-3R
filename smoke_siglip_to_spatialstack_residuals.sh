#!/usr/bin/env bash
#SBATCH --job-name=SMOKE_SigLIPResidual
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-false}" != "true" ]]; then
  echo "Submit this GPU wrapper with: sbatch $0" >&2
  exit 2
fi
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
: "${SIGLIP_FEATURE_CACHE:?Set SIGLIP_FEATURE_CACHE.}"
: "${CUT3R_FEATURE_CACHE:?Set CUT3R_FEATURE_CACHE.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" "$REPO_DIR/scripts/train/train_siglip_to_spatialstack_residual.py" \
  --siglip_feature_cache "$SIGLIP_FEATURE_CACHE" --cut3r_feature_cache "$CUT3R_FEATURE_CACHE" \
  --cut3r_layer6_subdir "${CUT3R_LAYER6_SUBDIR:-spatial_features_dec_6}" \
  --cut3r_layer9_subdir "${CUT3R_LAYER9_SUBDIR:-spatial_features_dec_9}" \
  --cut3r_layer12_subdir "${CUT3R_LAYER12_SUBDIR:-spatial_features}" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" --output_dir "${OUTPUT_DIR:-$REPO_DIR/outputs/smoke_siglip_spatialstack}" \
  --residual_predictor_type "${RESIDUAL_PREDICTOR_TYPE:-token_mlp}" --startup_check_samples "${STARTUP_CHECK_SAMPLES:-8}" \
  --strict_cache_checks true --run_parity_check true --smoke_only
