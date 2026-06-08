#!/bin/bash
#SBATCH --job-name=AggregateBEVLossProbe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --output=logs/cut3r_bev_loss_probe_aggregate_%x_%j.out
#SBATCH --error=logs/cut3r_bev_loss_probe_aggregate_%x_%j.err

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

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
NEW_TASK_FILE="${NEW_TASK_FILE:-$REPO_ROOT/scripts/probing/scannet_cut3r_bev_loss_probe_tasks.tsv}"
COMPARISON_TASK_FILE="${COMPARISON_TASK_FILE:-$REPO_ROOT/scripts/probing/scannet_five_model_probe_tasks.tsv}"

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir depth_probes_scannet \
  --task-file "$NEW_TASK_FILE" \
  --stem depth_probe_scannet_cut3r_bev_loss_8n32g_42837152 \
  --kind depth

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir semantic_probes_scannet \
  --task-file "$NEW_TASK_FILE" \
  --stem semantic_probe_scannet_cut3r_bev_loss_8n32g_42837152 \
  --kind semantic

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir depth_probes_scannet \
  --task-file "$COMPARISON_TASK_FILE" \
  --stem depth_probe_scannet_five_model_comparison \
  --kind depth

python scripts/probing/aggregate_scannet_probe_metrics.py \
  --output-root "$OUTPUT_ROOT" \
  --probe-subdir semantic_probes_scannet \
  --task-file "$COMPARISON_TASK_FILE" \
  --stem semantic_probe_scannet_five_model_comparison \
  --kind semantic
