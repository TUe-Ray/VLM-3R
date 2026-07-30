#!/usr/bin/env bash
# One-step real-data dual-path smoke.  The four fixture records are deliberate:
# the distributed dataloader requires at least one record per ZeRO-3 rank.
#SBATCH --job-name=SMOKE_dualpath_train32
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=122880
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_DIR"

export TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/smoke}"
export DATA_ROOT="${DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannetpp}"
export DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/experiments/diagnostics/dualpath_smoke_4x_51bdbf173f.json}"
export SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_scratch/large/userexternal/shuang00/dualpath_smoke_frame_provenance/frames32}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6,9:spatial_features_dec_9,12:spatial_features_dec_12}"
export MODEL_FRAMES_UPBOUND=32
export MODEL_SPATIAL_ATTENTION_MODE="${MODEL_SPATIAL_ATTENTION_MODE:-frame_local}"
export MODEL_WRITEBACK_QUERY_SCOPE="${MODEL_WRITEBACK_QUERY_SCOPE:-all_tokens}"
export MODEL_WRITEBACK_VISIBILITY="${MODEL_WRITEBACK_VISIBILITY:-frame_local}"
export MODEL_WRITEBACK_OUTPUT_INIT_STD="${MODEL_WRITEBACK_OUTPUT_INIT_STD:-1e-5}"
export MODEL_TORCH_COMPILE=False
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero3.json}"
export TARGET_GLOBAL_BATCH_SIZE=4
export SAVE_STEPS=1
export MAX_STEPS=1
export TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${SLURM_JOB_NAME:-SMOKE_dualpath_train32}_${SLURM_JOB_ID:-manual}}"

exec bash train_cut3r_dualpath.sh
