#!/bin/bash
# Step B: official-style spatial merge projector plus timing ablation.
set -euo pipefail

export NOTE="${NOTE:-CUT3R SpatialStack projector ablation: merge_mlp, dec6/9/12 -> LLM layers 1/2/3.}"
export SUFFIX="${SUFFIX:-cut3r_spatialstack_merge_mlp_dec6_9_12_llm1_2_3}"
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-cut3r_spatialstack_merge_mlp_dec6_9_12_llm1_2_3_${SLURM_JOB_ID:-manual}}"
export MODEL_CUT3R_SPATIALSTACK_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS:-1,2,3}"
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE="${MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE:-merge_mlp}"
export MODEL_CUT3R_SPATIALSTACK_MERGE_SIZE="${MODEL_CUT3R_SPATIALSTACK_MERGE_SIZE:-2}"
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM="${MODEL_CUT3R_SPATIALSTACK_PROJECTOR_HIDDEN_DIM:-4096}"
export MODEL_USE_POINTMAP_SUPERVISION="${MODEL_USE_POINTMAP_SUPERVISION:-False}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
