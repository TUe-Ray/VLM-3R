#!/bin/bash
#SBATCH --job-name=EvalRoPE_Spherical_RefCoord
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

# Re-evaluate the original spherical RoPE Fusion checkpoint from
# logs/eval/eval_rope_spherical_100p_40951431.out with explicit coordinate
# consistency. This legacy CUT3R point-map model trained against the model-side
# default, which selected point_maps_ref/pts3d_in_other_view.
export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/rope_spherical_100p_40790070}"
export RUN_NAME="${RUN_NAME:-eval_rope_spherical_100p_40790070_refcoord_recheck}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/recheck_rope_spherical_100p_40790070_refcoord}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_spherical_rope}"
export MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
export MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
export MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"
export RUN_RUNTIME_IMPORT_CHECKS="${RUN_RUNTIME_IMPORT_CHECKS:-True}"
export LIMIT="${LIMIT:-0}"

echo "==== Spherical RoPE coordinate recheck ===="
date
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"
echo "MODEL_GEO_ROPE_FUSION_MODE=$MODEL_GEO_ROPE_FUSION_MODE"
echo "MODEL_GEO_ROPE_FUSION_GROUP_SPLIT=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
echo "MODEL_GEO_ROPE_POINT_MAP_KEY=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "=========================================="

exec bash "$REPO_DIR/eval_rope_fusion_cut3r.sh"
