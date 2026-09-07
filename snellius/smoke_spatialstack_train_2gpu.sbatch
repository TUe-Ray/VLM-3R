#!/usr/bin/env bash
# Shared-node fallback for the same four-record, one-optimizer-step training
# smoke. Two ranks each consume two microbatches through gradient accumulation.
#SBATCH --job-name=SMOKE_vlm3r_spatialstack_train_2gpu
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=220G
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_2gpu_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_2gpu_%j.err

set -Eeuo pipefail

export SMOKE_NUM_PROCESSES=2
export SMOKE_GRAD_ACCUMULATION_STEPS=2
export NUM_PROCESSES=2
exec bash /home/geusdd/VLM-3R/snellius/smoke_spatialstack_train.sbatch
