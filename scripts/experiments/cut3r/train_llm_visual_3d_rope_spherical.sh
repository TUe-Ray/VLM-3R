#!/bin/bash
#SBATCH --job-name=llm_visual_3d_rope_spherical
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
#SBATCH --exclusive

set -euo pipefail

export NOTE="${NOTE:-LLM visual-token-only 3D RoPE, spherical, CUT3R reference geometry, no cross-attention GeoRoPE.}"

# Clean no-cross-attention-RoPE base.
export MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_patch_only}"
export MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
export MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"

# Single-shot LLM visual-token 3D RoPE settings.
export MODEL_LLM_VISUAL_3D_ROPE_ENABLE="${MODEL_LLM_VISUAL_3D_ROPE_ENABLE:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_ALPHA="${MODEL_LLM_VISUAL_3D_ROPE_ALPHA:-1.0}"
export MODEL_LLM_VISUAL_3D_ROPE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_MODE:-spherical}"
export MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT="${MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT:-2,1,2}"
export MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH="${MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH:-10.0}"
export MODEL_LLM_VISUAL_3D_ROPE_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LAYERS:-all}"
export MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE="${MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE:-point_maps_ref}"
export MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE:-False}"
export MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_MODE:-intra_sample_token_shuffle}"
export MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_SEED="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_SEED:-0}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS:-True}"
export MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS:-first_middle_last}"
export MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION="${MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION:-True}"

# The model code forces eager attention when LLM 3D RoPE is enabled; compile can still
# be toggled here for profiling this custom attention path.
export MODEL_TORCH_COMPILE="${MODEL_TORCH_COMPILE:-True}"
export MODEL_GRADIENT_CHECKPOINTING="${MODEL_GRADIENT_CHECKPOINTING:-True}"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
exec bash "$REPO_DIR/train_geo_rope_fusion_cut3r.sh"
