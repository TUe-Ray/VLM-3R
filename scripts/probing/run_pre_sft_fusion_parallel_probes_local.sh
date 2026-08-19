#!/usr/bin/env bash
# Run independent pre-SFT fusion probes in two physical-GPU workers.
#
# This deliberately does not extract features.  Set WAIT_FOR_PID to the PID of
# the two-GPU extraction process when launching it alongside extraction; the
# workers start only after all expected feature files are present.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
CACHE_BASE="${CACHE_ROOT:-/home/shaoruei/probe_cache/pre_sft_fusion_multiseed_v1/full}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/pre_sft_fusion_multiseed_v1}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/pre_sft_fusion_multiseed_v1/parallel_probes}"
PROBE_SEED="${PROBE_SEED:-0}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-43200}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-1199}"

mkdir -p "$LOG_ROOT" "$DURABLE_ROOT/probes" "$DURABLE_ROOT/provenance"

if [[ -n "$WAIT_FOR_PID" ]]; then
  deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      echo "Timed out waiting for extraction PID $WAIT_FOR_PID" >&2
      exit 1
    fi
    sleep 30
  done
fi

seed_root() {
  printf '%s/vlm3r_native/seed_%s' "$CACHE_BASE" "$1"
}

feature_count_ok() {
  local seed="$1" label="pre_sft_vlm3r_native_seed$1" level count
  for level in fusion_output projected_features layer_0 layer_2 layer_9 layer_27; do
    count=$(find "$(seed_root "$seed")/features/$label/$level" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
    if [[ "$count" -ne $((EXPECTED_VIDEOS * 2)) ]]; then
      echo "Incomplete seed $seed/$level feature cache: $count files; expected $((EXPECTED_VIDEOS * 2))" >&2
      return 1
    fi
  done
}

# Seed 0 was already partly trained before this parallel worker was launched;
# --skip-existing makes this list safe to resume after an interrupted worker.
feature_count_ok 0
feature_count_ok 1

run_probe() {
  local gpu="$1" seed="$2" level="$3"
  local label="pre_sft_vlm3r_native_seed${seed}"
  local root="$(seed_root "$seed")"
  local log="$LOG_ROOT/seed${seed}_${level}_gpu${gpu}.log"
  echo "[START] physical_gpu=$gpu seed=$seed level=$level log=$log"
  env CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$root" \
    --sample-indices "$SAMPLE_INDICES" \
    --probe-subdir probes \
    --model-labels "$label" \
    --feature-levels "$level" \
    --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
    --num-workers 0 --device cuda:0 --skip-existing --no-write-aggregate \
    --probe-seed "$PROBE_SEED" --experiment-variant vlm3r_native \
    --fusion-init-seed "$seed" --shared-llm-layers 0,2,9,27 \
    >"$log" 2>&1
  echo "[DONE] physical_gpu=$gpu seed=$seed level=$level"
}

jobs=(
  "0 9"
  "0 27"
  "1 fusion_output"
  "1 projected_features"
  "1 layer_0"
  "1 layer_2"
  "1 layer_9"
  "1 layer_27"
)

# Run two jobs per wave, one on each physical GPU.  Keeping one process per
# GPU avoids allocator contention while still parallelizing independent probes.
pids=()
for job in "${jobs[@]}"; do
  read -r seed level <<<"$job"
  gpu=$(( ${#pids[@]} % 2 ))
  run_probe "$gpu" "$seed" "$level" &
  pids+=("$!")
  if [[ ${#pids[@]} -eq 2 ]]; then
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    [[ "$rc" -eq 0 ]] || exit "$rc"
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

for seed in 0 1; do
  label="pre_sft_vlm3r_native_seed${seed}"
  root="$(seed_root "$seed")"
  mkdir -p "$DURABLE_ROOT/probes/$label"
  cp -a "$root/probes/$label/." "$DURABLE_ROOT/probes/$label/"
  if [[ -f "$root/features/$label/extraction_provenance.json" ]]; then
    cp -a "$root/features/$label/extraction_provenance.json" \
      "$DURABLE_ROOT/provenance/${label}_extraction_provenance.json"
  fi
done
echo "[DONE] all parallel VLM3R probe workers completed"
