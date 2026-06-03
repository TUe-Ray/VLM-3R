#!/bin/bash
#SBATCH --job-name=SemProbeScanNetLayer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=logs/semantic_probe_array_%x_%A_%a.out
#SBATCH --error=logs/semantic_probe_array_%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
module load cuda/12.2
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  export PATH="$CONDA_BASE/bin:$PATH"
fi
set +u
conda activate "$CONDA_ENV"
set -u
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/probing/scannet_baseline_probe_tasks.tsv}"
PROBE_SUBDIR="${PROBE_SUBDIR:-semantic_probes_scannet}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-7}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] This script must be submitted as a Slurm array." >&2
  exit 2
fi

task_line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")"
if [[ -z "$task_line" ]]; then
  echo "[ERROR] Empty task line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASK_FILE" >&2
  exit 2
fi
read -r MODEL_LABEL FEATURE_LEVEL <<< "$task_line"

echo "[INFO] Semantic task: model=$MODEL_LABEL feature=$FEATURE_LEVEL task_id=$SLURM_ARRAY_TASK_ID"
python -u scripts/probing/train_semantic_probes.py \
  --output-root "$OUTPUT_ROOT" \
  --sample-indices "$SAMPLE_INDICES" \
  --probe-subdir "$PROBE_SUBDIR" \
  --model-labels "$MODEL_LABEL" \
  --feature-levels "$FEATURE_LEVEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --skip-existing \
  --no-write-aggregate
