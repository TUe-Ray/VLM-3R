#!/usr/bin/env bash
#SBATCH --job-name=SMOKE_CUT3RTokenOnly_Eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err
#SBATCH --mem=0
set -euo pipefail
REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
export CHECKPOINT="${CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/SMOKE_CUT3RTokenOnly_DeepSpeedPreflight_51080583/checkpoint-2}"
export EVAL_PREFLIGHT_ONLY=True
export CUT3R_TOKEN_MANIFEST_POLICY=warn
export RUN_NAME="${RUN_NAME:-smoke_cut3r_token_only_legacy_fallback}"
export OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/cut3r_token_only_smoke_${SLURM_JOB_ID}}"
exec bash "$REPO_DIR/eval_cut3r_token_only_vsibench.sh"
