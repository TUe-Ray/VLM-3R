#!/bin/bash
#SBATCH --job-name=DBGSSZeroVisualVSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

# One real VSI example to validate checkpoint loading, final-layout patch zeroing,
# and layer-0/1/2 SpatialStack injections before full evaluation.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
export NUM_PROCESSES=1
export BATCH_SIZE=1
export LIMIT=1
export PRETRAINED_LOCAL="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963"
export PRESERVE_CHECKPOINT_CONFIG=True
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS="6,9,12"
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS="0,1,2"
export RUN_NAME="dbg_cut3r_spatialstack_45297963_zero_visual_patch_embeddings_vsibench"
export OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/debug_zero_visual_patch_embeddings_45297963}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+${EXTRA_MODEL_ARGS},}zero_visual_patch_embeddings=True"

exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
