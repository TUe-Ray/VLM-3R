#!/bin/bash
#SBATCH --job-name=vlm3r_depth_sync_work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --output=logs/depth_probe_sync_%x_%j.out
#SBATCH --error=logs/depth_probe_sync_%x_%j.err

set -euo pipefail

SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-${SCRATCH:-/leonardo_scratch/large/userexternal/shuang00}/probing_experiment}"
WORK_OUTPUT_ROOT="${WORK_OUTPUT_ROOT:-${WORK:-/leonardo_work/EUHPC_D32_006}/probing_experiment}"

mkdir -p logs "$WORK_OUTPUT_ROOT"
rsync -a --partial --info=progress2 "$SCRATCH_OUTPUT_ROOT/" "$WORK_OUTPUT_ROOT/"
