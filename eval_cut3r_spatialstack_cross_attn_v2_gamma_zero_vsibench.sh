#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SpatialStack_V2_GammaZero_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"

export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$TRAIN_MODEL_ROOT/cut3r_spatialstack_cross_attn_v2_resize_cam_gamma_47030066}"
export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-auto}"
export RUN_NAME="${RUN_NAME:-eval_cut3r_spatialstack_cross_attn_v2_gamma_zero_vsibench}"
export OUTPUT_PATH="${OUTPUT_PATH:-$FAST_ROOT/eval/logs/VLM3R/cut3r_spatialstack_cross_attn_v2_gamma_zero_vsibench}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$SCRIPT_DIR/.offline_runtime/${SLURM_JOB_ID:-eval_cut3r_spatialstack_cross_attn_v2_gamma_zero_vsibench}}"

export CUT3R_SPATIALSTACK_FUSION_TYPE="${CUT3R_SPATIALSTACK_FUSION_TYPE:-cross_attn_v2}"
export CUT3R_SPATIALSTACK_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS:-1,2,3}"
export CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY="${CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY:-True}"
export CUT3R_SPATIALSTACK_CROSS_ATTN_V2_FORCE_ZERO_GAMMA_AT_EVAL="${CUT3R_SPATIALSTACK_CROSS_ATTN_V2_FORCE_ZERO_GAMMA_AT_EVAL:-True}"

echo "==== CUT3R SpatialStack cross_attn_v2 gamma-zero VSI-Bench eval ===="
echo "TRAIN_OUTPUT_DIR=$TRAIN_OUTPUT_DIR"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "CUT3R_SPATIALSTACK_CROSS_ATTN_V2_FORCE_ZERO_GAMMA_AT_EVAL=$CUT3R_SPATIALSTACK_CROSS_ATTN_V2_FORCE_ZERO_GAMMA_AT_EVAL"

exec bash "$SCRIPT_DIR/eval_spatialstack_cross_attn_vsibench.sh"
