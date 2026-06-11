#!/bin/bash
#SBATCH --job-name=AggregateScanNetProbe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --output=logs/scannet_probe_aggregate_%x_%j.out
#SBATCH --error=logs/scannet_probe_aggregate_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  export PATH="$CONDA_BASE/bin:$PATH"
fi
set +u
conda activate "$CONDA_ENV"
set -u

MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
TASK_FILE="${TASK_FILE:?TASK_FILE is required}"
DEPTH_STEM="${DEPTH_STEM:-depth_probe_scannet_${MODEL_LABEL}}"
SEMANTIC_STEM="${SEMANTIC_STEM:-semantic_probe_scannet_${MODEL_LABEL}}"

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir depth_probes_scannet \
  --task-file "$TASK_FILE" \
  --stem "$DEPTH_STEM" \
  --kind depth

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir semantic_probes_scannet \
  --task-file "$TASK_FILE" \
  --stem "$SEMANTIC_STEM" \
  --kind semantic
