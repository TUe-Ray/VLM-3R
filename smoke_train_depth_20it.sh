#!/bin/bash
#SBATCH --job-name=SMOKE_depth20
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export NOTE="${NOTE:-SMOKE 20-step depth-only auxiliary supervision training check.}"
export MAX_STEPS="${MAX_STEPS:-20}"
export TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-4}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export REPORT_TO="${REPORT_TO:-none}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export MODEL_TORCH_COMPILE="${MODEL_TORCH_COMPILE:-True}"

exec bash "${SCRIPT_DIR}/train_depth_loss.sh" "$@"
