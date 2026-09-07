#!/usr/bin/env bash
# Lower-footprint fallback for the trained SpatialStack VSI-Bench evaluation
# smoke.  It runs the same wrapper/configuration as the four-GPU smoke but
# launches one accelerate process and deliberately shares an A100 node.
#SBATCH --job-name=SMOKE_vlm3r_spatialstack_eval_1gpu
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=normal
#SBATCH --time=00:30:00
# Keep below the A100 partition's 120-GiB-per-GPU default so Slurm bills and
# schedules this as a true single-GPU shared-node smoke allocation.
#SBATCH --mem=110G
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_eval_1gpu_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_eval_1gpu_%j.err

set -Eeuo pipefail

# This survives the standard paths.env defaults sourced by the shared wrapper.
export SMOKE_NUM_PROCESSES=1
export CUDA_VISIBLE_DEVICES=0
exec bash /home/geusdd/VLM-3R/snellius/smoke_spatialstack_eval.sbatch
