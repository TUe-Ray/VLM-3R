#!/bin/bash
#SBATCH --job-name=eval_llm_vv_3d_rope_disabled
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

# A3 is an inference-time dependency control on the same trained checkpoint,
# not a separately trained no-RoPE baseline.
export RUN_NAME="${RUN_NAME:-eval_llm_visual_3d_rope_disabled}"
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_patch_only}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"

export MODEL_LLM_VISUAL_3D_ROPE_ENABLE="${MODEL_LLM_VISUAL_3D_ROPE_ENABLE:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_ALPHA="${MODEL_LLM_VISUAL_3D_ROPE_ALPHA:-0.0}"
export MODEL_LLM_VISUAL_3D_ROPE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_MODE:-spherical}"
export MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT="${MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT:-2,1,2}"
export MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH="${MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH:-10.0}"
export MODEL_LLM_VISUAL_3D_ROPE_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LAYERS:-all}"
export MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE="${MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE:-point_maps_ref}"
export MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE:-False}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS:-first_middle_last}"
export MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION="${MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION:-True}"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
exec bash "$REPO_DIR/eval_rope_fusion_cut3r.sh"
