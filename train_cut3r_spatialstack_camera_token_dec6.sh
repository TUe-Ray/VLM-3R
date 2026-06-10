#!/bin/bash
# Design 1: SpatialStack + one CUT3R dec6 camera token prepended per frame.
set -euo pipefail

export NOTE="${NOTE:-CUT3R SpatialStack plus explicit dec6 camera tokens; mm_projector frozen.}"
export SUFFIX="${SUFFIX:-vlm_3r_vsibench_cut3r_spatialstack_camera_token_dec6_lora}"
export MODEL_USE_CUT3R_CAMERA_TOKENS="${MODEL_USE_CUT3R_CAMERA_TOKENS:-True}"
export MODEL_CUT3R_CAMERA_TOKEN_LAYER="${MODEL_CUT3R_CAMERA_TOKEN_LAYER:-6}"
export MODEL_CUT3R_CAMERA_TOKEN_INIT_SCALE="${MODEL_CUT3R_CAMERA_TOKEN_INIT_SCALE:-1.0}"
export MODEL_CUT3R_CAMERA_TOKEN_PROJECTOR_TYPE="${MODEL_CUT3R_CAMERA_TOKEN_PROJECTOR_TYPE:-mlp}"
export MODEL_USE_POINTMAP_SUPERVISION="${MODEL_USE_POINTMAP_SUPERVISION:-False}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
