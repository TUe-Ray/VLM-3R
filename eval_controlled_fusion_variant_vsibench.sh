#!/usr/bin/env bash
# Shared VSI-Bench handoff for controlled fusion candidates B/C/D/E/H.
set -euo pipefail

ARCH_ID="${CONTROLLED_FUSION_ID:?Set CONTROLLED_FUSION_ID to B, C, D, E, or H}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

export PRESERVE_CHECKPOINT_CONFIG=True
export EXPECTED_USE_CUT3R_SPATIALSTACK=True
export EXPECTED_FUSION_BLOCK=none
export EXPECTED_PRE_PROJECTOR_ADD_SOURCE_LAYER=""
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=""
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=""
export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=""
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE=token_mlp
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_BINDING=source_specific
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-12:spatial_features}"
export CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"

case "$ARCH_ID" in
    B)
        RUN_BASENAME="controlled_B_pre_projector_add_dec12_once"
        export EXPECTED_USE_CUT3R_SPATIALSTACK=False
        export EXPECTED_FUSION_BLOCK=pre_projector_add
        export EXPECTED_PRE_PROJECTOR_ADD_SOURCE_LAYER=12
        export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE=""
        export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_BINDING=""
        export CUT3R_SPATIALSTACK_LAYERS=12
        export CUT3R_SPATIALSTACK_LLM_LAYERS=0
        ;;
    C)
        RUN_BASENAME="controlled_C_cross_attn_dec12_llm0_once"
        export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=12
        export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0
        export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=cross_attn
        ;;
    D)
        RUN_BASENAME="controlled_D_add_dec12_llm0_once"
        export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=12
        export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0
        export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=add
        ;;
    E)
        RUN_BASENAME="controlled_E_add_dec12x3_llm0_1_2_repeat_siteproj"
        export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=12,12,12
        export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
        export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=add
        export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_BINDING=site_specific
        ;;
    H)
        RUN_BASENAME="controlled_H_cross_attn_dec12x3_llm0_1_2_repeat"
        export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=12,12,12
        export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
        export EXPECTED_CUT3R_SPATIALSTACK_FUSION_TYPE=cross_attn
        ;;
    *)
        echo "[ERROR] Unsupported CONTROLLED_FUSION_ID=$ARCH_ID (expected B/C/D/E/H)."
        exit 2
        ;;
esac

export CUT3R_SPATIALSTACK_LAYERS="${EXPECTED_CUT3R_SPATIALSTACK_LAYERS:-12}"
export CUT3R_SPATIALSTACK_LLM_LAYERS="${EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS:-0}"

if [[ -z "${PRETRAINED_LOCAL:-}" ]]; then
    shopt -s nullglob
    candidates=()
    for candidate in "$TRAIN_MODEL_ROOT/$RUN_BASENAME" "$TRAIN_MODEL_ROOT"/"${RUN_BASENAME}_"*; do
        [[ -f "$candidate/config.json" ]] && candidates+=("$candidate")
    done
    shopt -u nullglob
    if (( ${#candidates[@]} != 1 )); then
        echo "[ERROR] Expected exactly one checkpoint for $RUN_BASENAME, found ${#candidates[@]}."
        printf '  %s\n' "${candidates[@]}"
        echo "[ERROR] Set PRETRAINED_LOCAL explicitly."
        exit 2
    fi
    export PRETRAINED_LOCAL="${candidates[0]}"
fi

export RUN_NAME="${RUN_NAME:-eval_${RUN_BASENAME}_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/${RUN_BASENAME}_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}/.offline_runtime/${SLURM_JOB_ID:-$RUN_NAME}}"

REPO_ROOT="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
exec bash "$REPO_ROOT/eval_spatialstack_vsibench.sh"
