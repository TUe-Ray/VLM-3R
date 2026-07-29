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
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
CUT3R_ROOT="${CUT3R_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
# Dataset-prefixed roots preserve the sample key while using the actual FAST
# SigLIP caches requested for ScanNet, ScanNet++, and ARKitScenes.
SIGLIP_FEATURE_CACHE="${SIGLIP_FEATURE_CACHE:-scannet=$FAST_DATA_ROOT/scannet/siglip_features_dec_m2;scannetpp=$FAST_DATA_ROOT/scannetpp/siglip_features_dec_m2;arkitscenes=$FAST_DATA_ROOT/arkitscenes/siglip_features_dec_m2}"
CUT3R_FEATURE_CACHE="${CUT3R_FEATURE_CACHE:-$CUT3R_ROOT}"
CUT3R_LAYER6_CACHE="${CUT3R_LAYER6_CACHE:-scannet=$CUT3R_ROOT/scannet/spatial_features_dec_6;scannetpp=$CUT3R_ROOT/scannetpp/spatial_features_dec_6;arkitscenes=$CUT3R_ROOT/arkitscenes/spatial_features_dec_6}"
CUT3R_LAYER9_CACHE="${CUT3R_LAYER9_CACHE:-scannet=$CUT3R_ROOT/scannet/spatial_features_dec_9;scannetpp=$CUT3R_ROOT/scannetpp/spatial_features_dec_9;arkitscenes=$CUT3R_ROOT/arkitscenes/spatial_features_dec_9}"
CUT3R_LAYER12_CACHE="${CUT3R_LAYER12_CACHE:-scannet=$FAST_DATA_ROOT/scannet/spatial_features;scannetpp=$FAST_DATA_ROOT/scannetpp/spatial_features;arkitscenes=$FAST_DATA_ROOT/arkitscenes/spatial_features}"
PYTHON_BIN="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/smoke_siglip_spatialstack_${SLURM_JOB_ID}}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-8}"
SMOKE_VALIDATION_SAMPLES="${SMOKE_VALIDATION_SAMPLES:-4}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-10}"
SMOKE_RESUME_EPOCHS="${SMOKE_RESUME_EPOCHS:-11}"
COMMON_ARGS=(
  --siglip_feature_cache "$SIGLIP_FEATURE_CACHE" --cut3r_feature_cache "$CUT3R_FEATURE_CACHE"
  --cut3r_layer6_subdir "${CUT3R_LAYER6_SUBDIR:-spatial_features_dec_6}"
  --cut3r_layer9_subdir "${CUT3R_LAYER9_SUBDIR:-spatial_features_dec_9}"
  --cut3r_layer12_subdir "${CUT3R_LAYER12_SUBDIR:-spatial_features}"
  --cut3r_layer6_cache "$CUT3R_LAYER6_CACHE" --cut3r_layer9_cache "$CUT3R_LAYER9_CACHE" --cut3r_layer12_cache "$CUT3R_LAYER12_CACHE"
  --teacher_checkpoint "$TEACHER_CHECKPOINT" --output_dir "$OUTPUT_DIR"
  --residual_predictor_type token_mlp --predictor_bottleneck_dim "${PREDICTOR_BOTTLENECK_DIM:-1024}"
  --max_train_samples "$SMOKE_TRAIN_SAMPLES" --max_validation_samples "$SMOKE_VALIDATION_SAMPLES"
  --startup_check_samples "${STARTUP_CHECK_SAMPLES:-8}" --strict_cache_checks true
  --batch_size "${BATCH_SIZE:-1}" --learning_rate "${LEARNING_RATE:-1e-4}" --weight_decay "${WEIGHT_DECAY:-0.01}"
  --smooth_l1_weight "${SMOOTH_L1_WEIGHT:-0.1}" --teacher_norm_eps "${TEACHER_NORM_EPS:-1e-6}"
  --validation_fraction "${VALIDATION_FRACTION:-0.25}" --split_seed "${SPLIT_SEED:-42}"
)
"$PYTHON_BIN" "$REPO_DIR/scripts/train/train_siglip_to_spatialstack_residual.py" "${COMMON_ARGS[@]}" \
  --epochs "$SMOKE_EPOCHS" --run_parity_check true
[[ -f "$OUTPUT_DIR/latest.pt" ]] || { echo "[SMOKE] missing checkpoint: $OUTPUT_DIR/latest.pt" >&2; exit 1; }
"$PYTHON_BIN" "$REPO_DIR/scripts/train/train_siglip_to_spatialstack_residual.py" "${COMMON_ARGS[@]}" \
  --epochs "$SMOKE_RESUME_EPOCHS" --resume "$OUTPUT_DIR/latest.pt" \
  --startup_check_samples 0 --run_parity_check false
