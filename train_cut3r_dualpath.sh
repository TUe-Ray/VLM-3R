#!/usr/bin/env bash
# Dedicated wrapper: it authors a job but never submits one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NOTE="${NOTE:-Question-aware dual-path CUT3R: SS 6/9/12 side path with one writeback after LLM layer 2.}"
export SUFFIX="${SUFFIX:-cut3r_dualpath}"
export MODEL_USE_CUT3R_SPATIALSTACK=False
export MODEL_TUNE_CUT3R_SPATIALSTACK=False
export MODEL_ENABLE_DUAL_PATH_SPATIAL=True
export MODEL_TUNE_DUAL_PATH_SPATIAL=True
export MODEL_SPATIAL_NUM_LAYERS=3
export MODEL_SPATIAL_SOURCE_LAYERS=0,1,2
export MODEL_SPATIAL_ATTENTION_MODE="${MODEL_SPATIAL_ATTENTION_MODE:-global}"
export MODEL_WRITEBACK_QUERY_SCOPE="${MODEL_WRITEBACK_QUERY_SCOPE:-all_tokens}"
export MODEL_WRITEBACK_VISIBILITY="${MODEL_WRITEBACK_VISIBILITY:-frame_local}"
export MODEL_WRITEBACK_OUTPUT_INIT_STD="${MODEL_WRITEBACK_OUTPUT_INIT_STD:-1.0e-5}"
export MODEL_SPATIAL_CHECKPOINT="${MODEL_SPATIAL_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
export MODEL_PRESERVE_DENSE_SPATIAL_TOKENS=True
export MODEL_DUAL_PATH_RAW_LAYER12_CONTROL="${MODEL_DUAL_PATH_RAW_LAYER12_CONTROL:-False}"
export MODEL_DUAL_PATH_POSITION_ALIGNMENT=exact_index
export MODEL_DUAL_PATH_GRADIENT_CHECKPOINTING="${MODEL_DUAL_PATH_GRADIENT_CHECKPOINTING:-True}"
export MODEL_SPATIAL_MLP_CHUNK_SIZE="${MODEL_SPATIAL_MLP_CHUNK_SIZE:-1024}"
export MODEL_WRITEBACK_QUERY_CHUNK_SIZE="${MODEL_WRITEBACK_QUERY_CHUNK_SIZE:-512}"
# exact_index maps each CUT3R 27x27 patch to the matching canonical visual
# token, so this dedicated path must not inherit the baseline 2x pooling.
export MODEL_MM_SPATIAL_POOL_STRIDE="${MODEL_MM_SPATIAL_POOL_STRIDE:-1}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
