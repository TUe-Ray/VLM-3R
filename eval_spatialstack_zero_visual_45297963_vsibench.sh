#!/bin/bash
#SBATCH --job-name=EvalSSZeroVisualVSI
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

# Inference-only ablation: preserve the checkpoint configuration, sidecars, and
# evaluator settings while zeroing final projected SigLIP patch embeddings once.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
export PRETRAINED_LOCAL="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963"
export PRESERVE_CHECKPOINT_CONFIG=True
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS="6,9,12"
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS="0,1,2"
export RUN_NAME="eval_cut3r_spatialstack_45297963_zero_visual_patch_embeddings_vsibench"
export OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/zero_visual_patch_embeddings_45297963}"
export EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:+${EXTRA_MODEL_ARGS},}zero_visual_patch_embeddings=True"

exec bash "$REPO_DIR/eval_spatialstack_vsibench.sh"
