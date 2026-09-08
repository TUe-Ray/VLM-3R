#!/usr/bin/env bash
# Parallel local launcher for cache-only post-SFT MLP sample efficiency.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
INVENTORY="${INVENTORY:-$REPO_ROOT/scripts/probing/post_sft_depth_probe_sample_efficiency_inventory.json}"
SPLIT="${SPLIT:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/shaoruei/probe_outputs/post_sft_depth_probe_sample_efficiency_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/post_sft_depth_probe_sample_efficiency_v1}"
SIZES="${SIZES:-25,50,100,200,400,600,800,1000,1006}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
MODE="${1:-run}"

case "$MODE" in preflight|run|aggregate) ;; *) echo "Usage: $0 preflight|run|aggregate" >&2; exit 2;; esac
mkdir -p "$OUTPUT_ROOT/models" "$LOG_ROOT"

mapfile -t MODELS < <(python - "$INVENTORY" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1]))['models']:
    print(item['label'])
PY
)

preflight() {
  conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/probing/run_post_sft_depth_probe_sample_efficiency.py" \
    --inventory "$INVENTORY" --sample-indices "$SPLIT" --output-dir "$OUTPUT_ROOT/preflight" \
    --sample-sizes "$SIZES" --seeds "$SEEDS" --dry-run
}

aggregate() {
  conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/probing/aggregate_post_sft_depth_probe_sample_efficiency.py" \
    --inventory "$INVENTORY" --model-output-root "$OUTPUT_ROOT/models" --output-dir "$OUTPUT_ROOT"
}

if [[ "$MODE" == preflight ]]; then preflight; exit 0; fi
if [[ "$MODE" == aggregate ]]; then aggregate; exit 0; fi
preflight

run_one() {
  local label="$1"
  local physical_gpu="$2"
  local out="$OUTPUT_ROOT/models/$label"
  local log="$LOG_ROOT/$label.log"
  echo "[RUN] label=$label physical_gpu=$physical_gpu output=$out log=$log"
  env CUDA_VISIBLE_DEVICES="$physical_gpu" conda run --no-capture-output -n "$ENV_NAME" \
    python -u "$REPO_ROOT/scripts/probing/run_post_sft_depth_probe_sample_efficiency.py" \
    --inventory "$INVENTORY" --model-labels "$label" --sample-indices "$SPLIT" --output-dir "$out" \
    --sample-sizes "$SIZES" --seeds "$SEEDS" --device cuda:0 >"$log" 2>&1
}

for ((offset=0; offset<${#MODELS[@]}; offset+=2)); do
  pids=() labels=()
  for gpu in 0 1; do
    index=$((offset + gpu))
    (( index < ${#MODELS[@]} )) || continue
    run_one "${MODELS[$index]}" "$gpu" &
    pids+=("$!")
    labels+=("${MODELS[$index]}")
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || { echo "[ERROR] failed model=${labels[$index]}" >&2; exit 1; }
  done
done
aggregate
