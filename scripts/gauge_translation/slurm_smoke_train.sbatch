#!/bin/bash
#SBATCH --account=EUHPC_D32_006
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=SMOKE_cut3r_gauge_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err

set -euo pipefail
module purge
module load profile/deeplrn
module load cuda/12.1
REPO=${REPO:-/leonardo/home/userexternal/shuang00/VLM-3R}
cd "$REPO"
conda run --no-capture-output -n vlm3r python -u scripts/gauge_translation/train_cut3r_gauge_translation.py \
  --config configs/gauge_translation/smoke.json ${RESUME_FROM:+--resume-from "$RESUME_FROM"}
