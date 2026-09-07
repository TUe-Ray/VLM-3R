#!/usr/bin/env bash
# Shared configuration for controlled fusion candidates B/C/D/E/H.
set -euo pipefail

ARCH_ID="${CONTROLLED_FUSION_ID:?Set CONTROLLED_FUSION_ID to B, C, D, E, or H}"
FAST_FEATURE_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"

export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_FEATURE_ROOT}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-12:spatial_features}"
export MODEL_CUT3R_SPATIALSTACK_LAYERS="12"
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE="token_mlp"
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_BINDING="source_specific"
export MODEL_CUT3R_SPATIALSTACK_ZERO_INIT="True"
export MODEL_USE_CUT3R_CAMERA_TOKENS="False"
export MODEL_USE_POINTMAP_SUPERVISION="False"
export MODEL_USE_AUXILIARY_GEOMETRY_HEAD="False"
export MODEL_USE_AUXILIARY_GEOMETRY_LOSS="False"
export MODEL_USE_BEV_SUPERVISION="False"
export MODEL_USE_DEPTH_SUPERVISION="False"

# Preserve the three requested controlled-comparison stages exactly. Periodic
# saving is disabled so save_total_limit=3 cannot rotate a milestone away.
export CHECKPOINT_MILESTONE_RATIOS="${CHECKPOINT_MILESTONE_RATIOS:-0.05,0.25,0.50}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

case "$ARCH_ID" in
    B)
        export NOTE="Controlled fusion B: CUT3R dec12 patch tokens add to SigLIP features before mm_projector."
        export SUFFIX="controlled_B_pre_projector_add_dec12_once"
        export MODEL_USE_CUT3R_SPATIALSTACK="False"
        export MODEL_TUNE_CUT3R_SPATIALSTACK="False"
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0"
        export MODEL_FUSION_BLOCK="pre_projector_add"
        export MODEL_TUNE_FUSION_BLOCK="True"
        export MODEL_PRE_PROJECTOR_ADD_SOURCE_LAYER="12"
        export MODEL_PRE_PROJECTOR_ADD_ZERO_INIT="True"
        export MODEL_SPATIAL_TOWER_SELECT_FEATURE="patch_tokens"
        ;;
    C)
        export NOTE="Controlled fusion C: CUT3R dec12 cross-attention once before LLM layer 0."
        export SUFFIX="controlled_C_cross_attn_dec12_llm0_once"
        export MODEL_USE_CUT3R_SPATIALSTACK="True"
        export MODEL_TUNE_CUT3R_SPATIALSTACK="True"
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0"
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE="cross_attn"
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK="False"
        ;;
    D)
        export NOTE="Controlled fusion D: CUT3R dec12 additive injection once before LLM layer 0."
        export SUFFIX="controlled_D_add_dec12_llm0_once"
        export MODEL_USE_CUT3R_SPATIALSTACK="True"
        export MODEL_TUNE_CUT3R_SPATIALSTACK="True"
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0"
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE="add"
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK="False"
        ;;
    E)
        export NOTE="Controlled fusion E: CUT3R dec12 additive injection before LLM layers 0/1/2 with independent site projectors."
        export SUFFIX="controlled_E_add_dec12x3_llm0_1_2_repeat_siteproj"
        export MODEL_USE_CUT3R_SPATIALSTACK="True"
        export MODEL_TUNE_CUT3R_SPATIALSTACK="True"
        export MODEL_CUT3R_SPATIALSTACK_LAYERS="12,12,12"
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0,1,2"
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE="add"
        export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_BINDING="site_specific"
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK="False"
        ;;
    H)
        export NOTE="Controlled fusion H: CUT3R dec12 cross-attention before LLM layers 0/1/2."
        export SUFFIX="controlled_H_cross_attn_dec12x3_llm0_1_2_repeat"
        export MODEL_USE_CUT3R_SPATIALSTACK="True"
        export MODEL_TUNE_CUT3R_SPATIALSTACK="True"
        export MODEL_CUT3R_SPATIALSTACK_LAYERS="12,12,12"
        export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="0,1,2"
        export MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE="cross_attn"
        export MODEL_FUSION_BLOCK=""
        export MODEL_TUNE_FUSION_BLOCK="False"
        ;;
    *)
        echo "[ERROR] Unsupported CONTROLLED_FUSION_ID=$ARCH_ID (expected B/C/D/E/H)."
        exit 2
        ;;
esac

export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${SUFFIX}_${SLURM_JOB_ID:-manual}}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
