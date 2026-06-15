#!/bin/bash
# Eval Design 2: SpatialStack trained with world/reference xyz point-map supervision.
# Submit after the corresponding training output directory has the final LoRA save.
# Override PRETRAINED_LOCAL if the official training job id changes.
# Example:
#   PRETRAINED_LOCAL=/path/to/final/checkpoint sbatch eval_vlm3r_cut3r_spatialstack_pointmap_world_xyz_vsibench.sh
#SBATCH --job-name=Eval_CUT3R_SpatialStack_D2_Pointmap_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$TRAIN_MODEL_ROOT/cut3r_spatialstack_d2_pointmap_45457911}"
export RUN_NAME="${RUN_NAME:-eval_cut3r_spatialstack_d2_pointmap_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/cut3r_spatialstack_d2_pointmap_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_cut3r_spatialstack_d2_pointmap_vsibench}}"

export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
export CUT3R_TOKEN_FEATURES_ROOT="${CUT3R_TOKEN_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:$CUT3R_TOKEN_FEATURES_ROOT:spatial_features}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"

mkdir -p "$REPO_DIR/logs/eval" "$OUTPUT_PATH"

echo "==== CUT3R SpatialStack Design 2 pointmap VSI-Bench eval ===="
date
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "RUNTIME_ROOT=$RUNTIME_ROOT"
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "============================================================="

bash "$REPO_DIR/eval_spatialstack_vsibench.sh"

echo "[DONE] Design 2 eval artifacts are under: $OUTPUT_PATH"
