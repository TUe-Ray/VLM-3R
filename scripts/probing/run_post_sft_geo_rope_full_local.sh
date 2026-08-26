#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/shaoruei/SpatialFocus"
ENV_NAME="vlm3r"
SAMPLE_INDICES="/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json"
# This full ScanNet manifest has every candidate video.  The authoritative
# sample-index manifest below is the sole selector of the fixed 1,199-video
# probe population; route-plan JSON only covers a subset of ScanNet videos.
TRAIN_JSON="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1/manifests/merged_qa_scannet_train.json"
FORWARD_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/forward_frames_32_v1"
TARGET_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/probe_targets_2f_v1"
FEATURE_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_features"
GEOMETRY_ROOT="/mnt/DATA_SSD/shaoruei/probing_data/cut3r_point_maps_32_v1"
BASE_MODEL="/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL="/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384"
# Keep this post-SFT comparison contract local to this wrapper.
LAYERS=(0 1 2 3 6 9 12 15 18 21 24 27)
PRE_LLM_CSV="fusion_output,projected_features"
LAYER_CSV="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
FULL_FEATURE_CSV="${PRE_LLM_CSV},${LAYER_CSV}"
LOG_ROOT="$REPO_ROOT/logs/post_sft_geometry_probes/full"
mkdir -p "$LOG_ROOT"

verify_runner_readiness() {
  local gpu
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/verify_titan_v_readiness.py" \
      --physical-gpu-id "$gpu" --output "$LOG_ROOT/gpu${gpu}_readiness.json"
  done
}

run_extract() {
  local label="$1" checkpoint="$2" output_root="$3"
  local log="$LOG_ROOT/${label}_extraction.log"
  local precheck="$output_root/${label}_pre_extract_completeness.json"
  local assert_args=(--assert-first-video)
  mkdir -p "$output_root"
  # A complete additive resume performs no forward pass, so asserting a first
  # video would be a false failure.  Any incomplete 14-level request must prove
  # the expanded runtime contract on its first actual forward.
  if conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
      --output-root "$output_root" --model-label "$label" --sample-indices "$SAMPLE_INDICES" \
      --output "$precheck" >/dev/null 2>&1; then
    assert_args=()
  fi
  env CUDA_VISIBLE_DEVICES=0,1 SPATIALFOCUS_CPU_MERGE_LORA=1 \
    SPATIALFOCUS_CPU_MERGE_GPU_BUDGETS=6GiB,10GiB \
    conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/extract_depth_probe_features.py" \
      --model-label "$label" --post-sft-architecture "$label" \
      --model-loading-mode adapter --model-path "$checkpoint" \
      --model-base "$BASE_MODEL" --model-name vlm-3r-llava-qwen2-lora \
      --sample-indices "$SAMPLE_INDICES" --train-data-json "$TRAIN_JSON" \
      --feature-root "$FEATURE_ROOT" --spatial-features-subdir spatial_features \
      --geometry-spatial-features-root "$GEOMETRY_ROOT" \
      --geometry-spatial-features-subdir spatial_features_points \
      --geometry-point-map-key point_maps_ref \
      --forward-frames-root "$FORWARD_ROOT" --probe-targets-root "$TARGET_ROOT" \
      --image-folder "$FORWARD_ROOT" --video-folder "$FORWARD_ROOT" \
      --siglip-path "$SIGLIP_MODEL" --skip-spatial-tower-load true \
      --device cuda:0 --device-map auto --dtype float16 --cache-dtype float16 \
      --layers "${LAYERS[@]}" \
      --pre-llm-features "$PRE_LLM_CSV" \
      "${assert_args[@]}" \
      --output-root "$output_root" --resume 2>&1 | tee "$log"
}

run_probes() {
  local label="$1" output_root="$2"
  local log="$LOG_ROOT/${label}_probes.log"
  # The current extractor already writes one tensor per layer under
  # features/<model>/layer_*.  Do not invoke the historical llm_layers
  # materializer, which would incorrectly report the cache as empty.
  # Run two independent layer probes at a time, pinned to the physical GPUs.
  local gpu=0
  local pids=()
  local layer_levels=()
  IFS=',' read -r -a layer_levels <<< "$FULL_FEATURE_CSV"
  for level in "${layer_levels[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" python -u \
      "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
      --output-root "$output_root" --sample-indices "$SAMPLE_INDICES" \
      --probe-subdir probes --model-labels "$label" --feature-levels "$level" \
      --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
      --num-workers 0 --device cuda:0 --no-write-aggregate --skip-existing \
      >> "$log" 2>&1 &
    pids+=("$!")
    gpu=$((1 - gpu))
    if ((${#pids[@]} == 2)); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done

  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/train_depth_probes.py" \
    --output-root "$output_root" --sample-indices "$SAMPLE_INDICES" \
    --probe-subdir probes --model-labels "$label" --feature-levels "$FULL_FEATURE_CSV" \
    --epochs 50 --batch-size 32 --lr 1e-3 --early-stop-patience 10 \
    --num-workers 0 --device cpu --skip-existing \
    --result-stem "depth_probe_scannet_${label}" 2>&1 | tee -a "$log"
}

verify_features() {
  local label="$1" output_root="$2"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$output_root" --model-label "$label" --sample-indices "$SAMPLE_INDICES" \
    --output "$output_root/${label}_feature_completeness.json"
}

verify_probes() {
  local label="$1" output_root="$2"
  conda run -n "$ENV_NAME" python -u "$REPO_ROOT/scripts/probing/verify_post_sft_depth_probe.py" \
    --output-root "$output_root" --model-label "$label" --sample-indices "$SAMPLE_INDICES" --require-probes \
    --output "$output_root/${label}_probe_completeness.json"
}

echo "[INFO] Intended extraction: one sharded CUDA_VISIBLE_DEVICES=0,1 model process; logs=$LOG_ROOT"
verify_runner_readiness

run_extract geo_rope_fusion \
  "$REPO_ROOT/.offline_runtime/post_sft_geometry_probes/rope_spherical_100p_40790070" \
  /home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823
verify_features geo_rope_fusion /home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823
run_probes geo_rope_fusion /home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823
verify_probes geo_rope_fusion /home/shaoruei/probe_outputs/post_sft_geo_rope_fusion_full_20260823

run_extract visual_3d_rope \
  "$REPO_ROOT/.offline_runtime/post_sft_geometry_probes/RoPE_Spherical_cut3r_100p_41520134" \
  /home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823
verify_features visual_3d_rope /home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823
run_probes visual_3d_rope /home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823
verify_probes visual_3d_rope /home/shaoruei/probe_outputs/post_sft_visual_3d_rope_full_20260823

echo "[DONE] post-SFT GeoRoPE extraction and probes completed"
