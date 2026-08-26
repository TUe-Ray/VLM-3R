#!/usr/bin/env bash
# Probe independently trained VLM3R baseline replications on the fixed ScanNet set.
set -euo pipefail

export PATH="/home/shaoruei/miniconda3/bin:${PATH:-}"
if [[ -f /home/shaoruei/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/shaoruei/miniconda3/etc/profile.d/conda.sh
fi

MODE="${1:-}"
LABEL="${2:-}"
case "$MODE" in
  preflight|summary) ;;
  smoke-one|run-one) [[ -n "$LABEL" ]] || { echo "model label required" >&2; exit 2; } ;;
  *) echo "Usage: $0 preflight|summary|smoke-one <label>|run-one <label>" >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/shaoruei/SpatialFocus}"
ENV_NAME="${ENV_NAME:-vlm3r}"
GPU="${GPU:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
BASE_MODEL="${BASE_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384}"
TEMPLATE="${TEMPLATE:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/Reproduction_2}"
APR30_CHECKPOINT="${APR30_CHECKPOINT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/selec_100pct_baseline_40390735}"
APR05_CHECKPOINT="${APR05_CHECKPOINT:-/mnt/DATA_SSD/shaoruei/models/vlm3r_runs/baseline_apr05_reproduction}"
FORWARD_ROOT="${FORWARD_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1}"
FEATURE_ROOT="${FEATURE_ROOT:-/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json}"
CACHE_ROOT="${CACHE_ROOT:-/home/shaoruei/probe_cache/scannet_baseline_replicates_v1}"
DURABLE_ROOT="${DURABLE_ROOT:-/home/shaoruei/probe_outputs/scannet_baseline_replicates_v1}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/scannet_baseline_replicates_v1}"
LOCAL_DATA_YAML="$REPO_ROOT/scripts/probing/scannet_depth_probe_local_data.yaml"
REPORT="$LOG_ROOT/preflight.json"
LAYERS=(0 1 2 3 6 9 12 15 18 21 24 27)
PRE_LLM_CSV="fusion_output,projected_features"
LAYER_CSV="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
LEVELS="${PRE_LLM_CSV},${LAYER_CSV}"

mkdir -p "$CACHE_ROOT" "$DURABLE_ROOT" "$LOG_ROOT"

configure_model() {
  case "$1" in
    baseline_apr30_40390735) CHECKPOINT="$APR30_CHECKPOINT" ;;
    baseline_apr05_reproduction) CHECKPOINT="$APR05_CHECKPOINT" ;;
    *) echo "Unsupported model label: $1" >&2; return 2 ;;
  esac
}

model_args=()
model_args+=(--model "baseline_apr30_40390735=$APR30_CHECKPOINT")
model_args+=(--model "baseline_apr05_reproduction=$APR05_CHECKPOINT")

preflight() {
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/summarize_scannet_baseline_replicates.py" \
    --durable-root "$DURABLE_ROOT" --template "$TEMPLATE" "${model_args[@]}" \
    --report "$REPORT" --write-summary
}

if [[ "$MODE" == preflight || "$MODE" == summary ]]; then
  preflight
  exit 0
fi

configure_model "$LABEL"
preflight
if ! jq -e --arg model_label "$LABEL" '.models[] | select(.label == $model_label and .ready == true)' "$REPORT" >/dev/null; then
  echo "Checkpoint failed baseline-setting validation: $LABEL ($CHECKPOINT)" >&2
  exit 1
fi

smoke_manifest() {
  local root="$1" manifest="$1/manifests/scannet_smoke_1train_1val.json"
  if [[ ! -f "$manifest" ]]; then
    mkdir -p "$(dirname "$manifest")"
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/make_depth_probe_smoke_manifest.py" \
      --sample-indices "$SAMPLE_INDICES" --output "$manifest" --train-videos 1 --val-videos 1 >/dev/null
  fi
  printf '%s\n' "$manifest"
}

extract() {
  local output_root="$1" manifest="$2" levels="$3" log="$4"
  local level pre_llm_csv
  local -a requested=() requested_layers=() requested_pre_llm=() extract_args=()
  IFS=, read -r -a requested <<< "$levels"
  for level in "${requested[@]}"; do
    case "$level" in
      layer_*) requested_layers+=("${level#layer_}") ;;
      fusion_output|projected_features) requested_pre_llm+=("$level") ;;
      *) echo "Unsupported probe feature level: $level" >&2; return 2 ;;
    esac
  done
  extract_args=(
    "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py"
    --model-label "$LABEL" --model-path "$CHECKPOINT" --feature-preset original
    --model-base "$BASE_MODEL" --siglip-path "$SIGLIP_MODEL" --output-root "$output_root"
    --sample-indices "$manifest" --data-yaml "$LOCAL_DATA_YAML" --feature-root "$FEATURE_ROOT"
    --spatial-features-subdir spatial_features --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT"
    --video-folder "$FORWARD_ROOT" --image-folder "$FORWARD_ROOT" --frames-upbound 32
    --dtype float16 --cache-dtype float16 --device cuda:0 --device-map auto
    --runtime-root "$output_root/runtime" --assert-first-video --resume
  )
  if (( ${#requested_layers[@]} )); then
    extract_args+=(--layers "${requested_layers[@]}")
  fi
  if (( ${#requested_pre_llm[@]} )); then
    pre_llm_csv=$(IFS=,; printf '%s' "${requested_pre_llm[*]}")
    extract_args+=(--pre-llm-features "$pre_llm_csv")
  fi
  env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" SPATIALFOCUS_CPU_MERGE_LORA=1 conda run -n "$ENV_NAME" python -u \
    "${extract_args[@]}" \
    2>&1 | tee "$log"
}

train() {
  local output_root="$1" manifest="$2" level="$3" epochs="$4" partial="$5" physical_gpu="$6" log="$7"
  args=(
    "$REPO_ROOT/scripts/probing/train_depth_probes.py" --output-root "$output_root"
    --sample-indices "$manifest" --probe-subdir probes --model-labels "$LABEL" --feature-levels "$level"
    --epochs "$epochs" --batch-size 32 --lr 1e-3 --early-stop-patience 10 --num-workers 0
    --device cuda:0 --no-write-aggregate
  )
  [[ "$partial" == true ]] && args+=(--allow-partial)
  env CUDA_VISIBLE_DEVICES="$physical_gpu" conda run -n "$ENV_NAME" python -u "${args[@]}" 2>&1 | tee "$log"
}

if [[ "$MODE" == smoke-one ]]; then
  ACTIVE_CACHE="$CACHE_ROOT/smoke/$LABEL"
  SMOKE_LOG="$LOG_ROOT/$LABEL.smoke.log"
  MANIFEST="$(smoke_manifest "$CACHE_ROOT/smoke")"
  mkdir -p "$ACTIVE_CACHE"
  echo "[SMOKE] label=$LABEL checkpoint=$CHECKPOINT GPUs=$CUDA_DEVICES log=$SMOKE_LOG"
  extract "$ACTIVE_CACHE" "$MANIFEST" "$LEVELS" "$SMOKE_LOG"
  IFS=, read -r -a requested <<< "$LEVELS"
  for level in "${requested[@]}"; do
    train "$ACTIVE_CACHE" "$MANIFEST" "$level" 2 true 0 "$LOG_ROOT/$LABEL.$level.smoke.log"
  done
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_scannet_final_layerwise_smoke.py" \
    --output-root "$ACTIVE_CACHE" --model-label "$LABEL" --feature-levels "$LEVELS" --manifest "$MANIFEST" \
    --report "$ACTIVE_CACHE/smoke_verification.json" 2>&1 | tee -a "$SMOKE_LOG"
  echo "[SMOKE DONE] $LABEL"
  exit 0
fi

mapfile -t MISSING < <(jq -r --arg model_label "$LABEL" '.models[] | select(.label == $model_label) | .missing[]' "$REPORT")
if (( ${#MISSING[@]} == 0 )); then
  echo "[DONE] All durable points already exist for $LABEL"
  exit 0
fi

ACTIVE_CACHE="$CACHE_ROOT/full/$LABEL"
FULL_LOG="$LOG_ROOT/$LABEL.full.log"
mkdir -p "$ACTIVE_CACHE" "$DURABLE_ROOT/provenance/$LABEL"
MISSING_CSV=$(IFS=,; printf '%s' "${MISSING[*]}")
echo "[RUN] label=$LABEL checkpoint=$CHECKPOINT GPUs=$CUDA_DEVICES levels=$MISSING_CSV log=$FULL_LOG"
extract "$ACTIVE_CACHE" "$SAMPLE_INDICES" "$MISSING_CSV" "$FULL_LOG"
conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
  --output-root "$ACTIVE_CACHE" --model-label "$LABEL" --sample-indices "$SAMPLE_INDICES" \
  --output "$DURABLE_ROOT/provenance/$LABEL/feature_completeness.json"

for ((start=0; start<${#MISSING[@]}; start+=2)); do
  pids=(); names=()
  for physical_gpu in 0 1; do
    idx=$((start + physical_gpu))
    (( idx < ${#MISSING[@]} )) || continue
    level="${MISSING[$idx]}"
    train "$ACTIVE_CACHE" "$SAMPLE_INDICES" "$level" 50 false "$physical_gpu" "$LOG_ROOT/$LABEL.$level.log" &
    pids+=("$!"); names+=("$level")
  done
  for idx in "${!pids[@]}"; do
    wait "${pids[$idx]}" || { echo "probe failed: $LABEL/${names[$idx]}" >&2; exit 1; }
  done
done

conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
  --output-root "$ACTIVE_CACHE" --model-label "$LABEL" --sample-indices "$SAMPLE_INDICES" --require-probes \
  --output "$DURABLE_ROOT/provenance/$LABEL/probe_completeness.json"

for level in "${MISSING[@]}"; do
  mkdir -p "$DURABLE_ROOT/probes/$LABEL/$level"
  cp -a "$ACTIVE_CACHE/probes/$LABEL/$level/." "$DURABLE_ROOT/probes/$LABEL/$level/"
done
cp -a "$ACTIVE_CACHE/features/$LABEL/extraction_provenance.json" "$DURABLE_ROOT/provenance/$LABEL/"
sha256sum "$CHECKPOINT"/{adapter_model.bin,non_lora_trainables.bin,adapter_config.json,config.json,generation_config.json} \
  > "$DURABLE_ROOT/provenance/$LABEL/checkpoint_sha256.txt"
preflight
if jq -e --arg model_label "$LABEL" '.models[] | select(.label == $model_label and .complete == true)' "$REPORT" >/dev/null; then
  case "$ACTIVE_CACHE" in
    "$CACHE_ROOT"/full/*) rm -rf -- "$ACTIVE_CACHE" ;;
    *) echo "refusing unexpected cleanup path: $ACTIVE_CACHE" >&2; exit 1 ;;
  esac
  echo "[DONE] Durable results verified; removed only $ACTIVE_CACHE"
else
  echo "[ERROR] Durable verification failed; cache retained at $ACTIVE_CACHE" >&2
  exit 1
fi
