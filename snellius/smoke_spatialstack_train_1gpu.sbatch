#!/usr/bin/env bash
# Smallest shared-node fallback for the same four-record, one-optimizer-step
# training smoke. One rank consumes four microbatches through accumulation.
#SBATCH --job-name=SMOKE_vlm3r_spatialstack_train_1gpu
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=110G
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_1gpu_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_1gpu_%j.err

set -Eeuo pipefail

export SMOKE_NUM_PROCESSES=1
export SMOKE_GRAD_ACCUMULATION_STEPS=4
export NUM_PROCESSES=1
exec bash /home/geusdd/VLM-3R/snellius/smoke_spatialstack_train.sbatch
