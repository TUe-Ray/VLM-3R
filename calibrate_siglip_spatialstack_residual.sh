#!/usr/bin/env bash
#SBATCH --job-name=CalibrateSigLIPResidual
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=32G
set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
CUT3R_ROOT="${CUT3R_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
: "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR.}"
SIGLIP_FEATURE_CACHE="${SIGLIP_FEATURE_CACHE:-scannet=$FAST_DATA_ROOT/scannet/siglip_features_dec_m2;scannetpp=$FAST_DATA_ROOT/scannetpp/siglip_features_dec_m2;arkitscenes=$FAST_DATA_ROOT/arkitscenes/siglip_features_dec_m2}"
CUT3R_LAYER6_CACHE="${CUT3R_LAYER6_CACHE:-scannet=$CUT3R_ROOT/scannet/spatial_features_dec_6;scannetpp=$CUT3R_ROOT/scannetpp/spatial_features_dec_6;arkitscenes=$CUT3R_ROOT/arkitscenes/spatial_features_dec_6}"
CUT3R_LAYER9_CACHE="${CUT3R_LAYER9_CACHE:-scannet=$CUT3R_ROOT/scannet/spatial_features_dec_9;scannetpp=$CUT3R_ROOT/scannetpp/spatial_features_dec_9;arkitscenes=$CUT3R_ROOT/arkitscenes/spatial_features_dec_9}"
CUT3R_LAYER12_CACHE="${CUT3R_LAYER12_CACHE:-scannet=$FAST_DATA_ROOT/scannet/spatial_features;scannetpp=$FAST_DATA_ROOT/scannetpp/spatial_features;arkitscenes=$FAST_DATA_ROOT/arkitscenes/spatial_features}"
args=(
  --siglip_feature_cache "$SIGLIP_FEATURE_CACHE"
  --cut3r_feature_cache "$CUT3R_ROOT"
  --cut3r_layer6_cache "$CUT3R_LAYER6_CACHE"
  --cut3r_layer9_cache "$CUT3R_LAYER9_CACHE"
  --cut3r_layer12_cache "$CUT3R_LAYER12_CACHE"
  --teacher_checkpoint "${TEACHER_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
  --predictor_checkpoint "$PREDICTOR_CHECKPOINT"
  --output_dir "$OUTPUT_DIR"
  --validation_fraction "${VALIDATION_FRACTION:-0.1}"
  --split_seed "${SPLIT_SEED:-42}"
)
[[ "${MAX_TRAIN_SAMPLES:-0}" == "0" ]] || args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
[[ "${MAX_VALIDATION_SAMPLES:-0}" == "0" ]] || args+=(--max_validation_samples "$MAX_VALIDATION_SAMPLES")
exec /leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python "$REPO_DIR/scripts/train/calibrate_siglip_spatialstack_residual.py" "${args[@]}"
