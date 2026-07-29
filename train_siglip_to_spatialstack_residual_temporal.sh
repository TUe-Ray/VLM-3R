#!/usr/bin/env bash
#SBATCH --job-name=SigLIPResidualTemporal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
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
: "${OUTPUT_DIR:?Set OUTPUT_DIR for predictor checkpoints.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
args=(
  --siglip_feature_cache "$SIGLIP_FEATURE_CACHE"
  --cut3r_feature_cache "$CUT3R_FEATURE_CACHE"
  --cut3r_layer6_subdir "${CUT3R_LAYER6_SUBDIR:-spatial_features_dec_6}"
  --cut3r_layer9_subdir "${CUT3R_LAYER9_SUBDIR:-spatial_features_dec_9}"
  --cut3r_layer12_subdir "${CUT3R_LAYER12_SUBDIR:-spatial_features}"
  --cut3r_layer6_cache "$CUT3R_LAYER6_CACHE"
  --cut3r_layer9_cache "$CUT3R_LAYER9_CACHE"
  --cut3r_layer12_cache "$CUT3R_LAYER12_CACHE"
  --teacher_checkpoint "$TEACHER_CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --residual_predictor_type temporal
  --predictor_bottleneck_dim "${PREDICTOR_BOTTLENECK_DIM:-1024}"
  --temporal_hidden_dim "${TEMPORAL_HIDDEN_DIM:-512}"
  --temporal_num_layers "${TEMPORAL_NUM_LAYERS:-2}"
  --temporal_num_heads "${TEMPORAL_NUM_HEADS:-8}"
  --temporal_ffn_dim "${TEMPORAL_FFN_DIM:-2048}"
  --temporal_dropout "${TEMPORAL_DROPOUT:-0.0}"
  --temporal_max_frames "${TEMPORAL_MAX_FRAMES:-128}"
  --validation_fraction "${VALIDATION_FRACTION:-0.1}"
  --split_seed "${SPLIT_SEED:-42}"
  --startup_check_samples "${STARTUP_CHECK_SAMPLES:-8}"
  --strict_cache_checks "${STRICT_CACHE_CHECKS:-false}"
  --run_parity_check "${RUN_PARITY_CHECK:-false}"
  --batch_size "${BATCH_SIZE:-1}"
  --epochs "${EPOCHS:-10}"
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --smooth_l1_weight "${SMOOTH_L1_WEIGHT:-0.1}"
)
[[ -z "${TRAIN_KEY_LIST:-}" ]] || args+=(--train_key_list "$TRAIN_KEY_LIST")
[[ -z "${VALIDATION_KEY_LIST:-}" ]] || args+=(--validation_key_list "$VALIDATION_KEY_LIST")
[[ -z "${DATASET_JSON:-}" ]] || args+=(--dataset_json "$DATASET_JSON")
[[ -z "${TRAIN_DATASET_JSON:-}" ]] || args+=(--train_dataset_json "$TRAIN_DATASET_JSON")
[[ -z "${VALIDATION_DATASET_JSON:-}" ]] || args+=(--validation_dataset_json "$VALIDATION_DATASET_JSON")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
exec "$PYTHON_BIN" "$REPO_DIR/scripts/train/train_siglip_to_spatialstack_residual.py" "${args[@]}"
