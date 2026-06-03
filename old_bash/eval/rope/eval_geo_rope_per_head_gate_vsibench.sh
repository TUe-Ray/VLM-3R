#!/bin/bash
#SBATCH --job-name=EvalGeoRoPE_PerHeadGate
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"

export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/geo_rope_spherical_per_head_gate_init099_resume_fast_workfb_42445436}"
export RUN_NAME="${RUN_NAME:-eval_geo_rope_spherical_per_head_gate_init099_42445436}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/geo_rope_spherical_per_head_gate_init099_42445436}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_geo_rope_fusion_per_head_gate}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
export FORCE_GEO_ROPE_GATE_ZERO="${FORCE_GEO_ROPE_GATE_ZERO:-False}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"
export RUN_RUNTIME_IMPORT_CHECKS="${RUN_RUNTIME_IMPORT_CHECKS:-True}"
export LIMIT="${LIMIT:-0}"

echo "==== Per-head gated GeoRoPE eval ===="
date
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"
echo "MODEL_GEO_ROPE_FUSION_MODE=$MODEL_GEO_ROPE_FUSION_MODE"
echo "MODEL_GEO_ROPE_FUSION_GROUP_SPLIT=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
echo "MODEL_GEO_ROPE_POINT_MAP_KEY=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "FORCE_GEO_ROPE_GATE_ZERO=$FORCE_GEO_ROPE_GATE_ZERO"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "====================================="

exec bash "$REPO_DIR/eval_rope_fusion_cut3r.sh"
