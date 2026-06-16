#!/bin/bash
#SBATCH --job-name=cut3r_ss_token036
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
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

# Timing ablation: spread the three CUT3R injections across early LLM layers.
set -euo pipefail

export NOTE="${NOTE:-CUT3R SpatialStack timing ablation: token_mlp, dec6/9/12 -> LLM layers 0/3/6.}"
export SUFFIX="${SUFFIX:-cut3r_spatialstack_token_mlp_dec6_9_12_llm0_3_6}"
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${SUFFIX}_${SLURM_JOB_ID:-manual}}"
export MODEL_CUT3R_SPATIALSTACK_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
export MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS:-0,3,6}"
export MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE="${MODEL_CUT3R_SPATIALSTACK_PROJECTOR_TYPE:-token_mlp}"
export MODEL_USE_POINTMAP_SUPERVISION="${MODEL_USE_POINTMAP_SUPERVISION:-False}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "$SCRIPT_DIR/train_cut3r_spatialstack.sh"
