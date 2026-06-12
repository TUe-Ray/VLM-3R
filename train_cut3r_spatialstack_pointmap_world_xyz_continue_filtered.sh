#!/bin/bash
# Continue Design 2 from the latest good checkpoint, using filtered train JSONs
# and the verified large CUT3R point-map sidecar root.
#SBATCH --job-name=cut3r_spatialstack_d2_pointmap_continue
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322,lrdn0594
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FILTERED_DATA_YAML="$REPO_DIR/.codex-work/datasets/cut3r_spatialstack_d2_pointmap_continue_filtered/vsibench_data_filtered.yaml"
OUTPUT_ROOT="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_d2_pointmap_45457911"

export NOTE="${NOTE:-Continue CUT3R SpatialStack D2 point-map training from checkpoint-800 with filtered bad-video samples; mm_projector frozen.}"
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-cut3r_spatialstack_d2_pointmap_45457911}"
export OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT}"
export RESUME_MODE="${RESUME_MODE:-continue}"
export RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-$OUTPUT_ROOT/checkpoint-800}"

export DATA_PATH_YAML="${DATA_PATH_YAML:-$FILTERED_DATA_YAML}"
export DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
# Keep Decord's normal EOF tolerance; known bad videos are removed by the filtered dataset.
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export GEOMETRY_SPATIAL_FEATURES_ROOT="${GEOMETRY_SPATIAL_FEATURES_ROOT:-/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps}"
export GEOMETRY_SPATIAL_FEATURES_SUBDIR="${GEOMETRY_SPATIAL_FEATURES_SUBDIR:-spatial_features_points}"
export GEOMETRY_SPATIAL_TOWER_TYPE="${GEOMETRY_SPATIAL_TOWER_TYPE:-cut3r}"
export REQUIRE_GEOMETRY_SPATIAL_FEATURES="${REQUIRE_GEOMETRY_SPATIAL_FEATURES:-True}"

export MODEL_USE_CUT3R_CAMERA_TOKENS="${MODEL_USE_CUT3R_CAMERA_TOKENS:-False}"
export MODEL_USE_POINTMAP_SUPERVISION="${MODEL_USE_POINTMAP_SUPERVISION:-True}"
export MODEL_POINTMAP_HEAD_SOURCE="${MODEL_POINTMAP_HEAD_SOURCE:-llm_output}"
export MODEL_POINTMAP_POINT_MAP_KEY="${MODEL_POINTMAP_POINT_MAP_KEY:-point_maps_ref}"
export MODEL_LAMBDA_POINTMAP="${MODEL_LAMBDA_POINTMAP:-0.1}"
export MODEL_POINTMAP_COORD_SCALE="${MODEL_POINTMAP_COORD_SCALE:-10.0}"
export MODEL_POINTMAP_SMOOTH_L1_BETA="${MODEL_POINTMAP_SMOOTH_L1_BETA:-0.1}"
export MODEL_POINTMAP_DETACH_HIDDEN="${MODEL_POINTMAP_DETACH_HIDDEN:-False}"
export MODEL_POINTMAP_CONF_THRESHOLD="${MODEL_POINTMAP_CONF_THRESHOLD:-0.0}"

echo "==== D2 point-map continuation ===="
echo "FILTERED_DATA_YAML=$FILTERED_DATA_YAML"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RESUME_CHECKPOINT_PATH=$RESUME_CHECKPOINT_PATH"
echo "DECORD_EOF_RETRY_MAX=$DECORD_EOF_RETRY_MAX"
echo "GEOMETRY_SPATIAL_FEATURES_ROOT=$GEOMETRY_SPATIAL_FEATURES_ROOT"
echo "==================================="

exec bash "$REPO_DIR/train_cut3r_spatialstack.sh"
