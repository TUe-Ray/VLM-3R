#!/usr/bin/env bash
# One optimizer step for E on 2 nodes x 4 GPUs, using the official launcher.
# The target global batch remains 128, so this resolves to accumulation 16.
#SBATCH --job-name=SMOKE_controlled_E_2n8g
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_multinode/%x_%j.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_multinode/%x_%j.err

set -Eeuo pipefail

REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
export CONTROLLED_FUSION_ID=E
export CONTROLLED_FUSION_SMOKE=True
export CONTROLLED_FUSION_SMOKE_NODES=2

exec bash "$REPO_DIR/snellius/train_controlled_fusion_campaign.sbatch"
