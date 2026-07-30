#!/usr/bin/env bash
#SBATCH --job-name=cut3r_token_only_vsi_8n
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/cut3r_token_only/%x_%j.out
#SBATCH --error=logs/cut3r_token_only/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclusive

# Dedicated eight-node launch.  The shared CUT3R-only wrapper derives
# WORLD_SIZE and gradient accumulation from this allocation: 32 * 1 * 4 = 128.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
export TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export CUT3R_TOKEN_MANIFEST_POLICY="${CUT3R_TOKEN_MANIFEST_POLICY:-warn}"
exec bash "$REPO_DIR/train_cut3r_token_only_vsi.sh"
