#!/bin/bash
#SBATCH --job-name=SMOKE_ScanNetProbe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=0
#SBATCH --output=logs/scannet_probe_smoke_%x_%j.out
#SBATCH --error=logs/scannet_probe_smoke_%x_%j.err

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
TASK_FILE="${TASK_FILE:?TASK_FILE must point to a probe task TSV}"
SMOKE_TASK_ID="${SMOKE_TASK_ID:-0}"
SMOKE_TAG="${SMOKE_TAG:-scannet_probe}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"

task_line="$(sed -n "$((SMOKE_TASK_ID + 1))p" "$TASK_FILE")"
if [[ -z "$task_line" ]]; then
  echo "[ERROR] Empty smoke task line for SMOKE_TASK_ID=$SMOKE_TASK_ID in $TASK_FILE" >&2
  exit 2
fi
read -r MODEL_LABEL FEATURE_LEVEL <<< "$task_line"

echo "[INFO] Smoke task: model=$MODEL_LABEL feature=$FEATURE_LEVEL task_id=$SMOKE_TASK_ID tag=$SMOKE_TAG"

python -u scripts/probing/train_depth_probes.py \
  --output-root "$OUTPUT_ROOT" \
  --sample-indices "$SAMPLE_INDICES" \
  --probe-subdir "depth_probes_scannet_smoke_${SMOKE_TAG}" \
  --model-labels "$MODEL_LABEL" \
  --feature-levels "$FEATURE_LEVEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --skip-existing \
  --no-write-aggregate

python -u scripts/probing/train_semantic_probes.py \
  --output-root "$OUTPUT_ROOT" \
  --sample-indices "$SAMPLE_INDICES" \
  --probe-subdir "semantic_probes_scannet_smoke_${SMOKE_TAG}" \
  --model-labels "$MODEL_LABEL" \
  --feature-levels "$FEATURE_LEVEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --skip-existing \
  --no-write-aggregate
