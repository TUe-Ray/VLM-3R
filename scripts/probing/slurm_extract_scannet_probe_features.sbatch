#!/bin/bash
#SBATCH --job-name=ScanNetProbeExtract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=logs/scannet_probe_extract_%x_%j.out
#SBATCH --error=logs/scannet_probe_extract_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
module load cuda/12.2
if [[ -z "${CUDA_HOME:-}" && -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="$EBROOTCUDA"
fi
if [[ -z "${CUDA_HOME:-}" && -x "$(command -v nvcc)" ]]; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
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

MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
ARCH_PRESET="${ARCH_PRESET:-original}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
if [[ -z "${FEATURE_ROOT:-}" ]]; then
  if [[ "$ARCH_PRESET" == "spatialstack" ]]; then
    FEATURE_ROOT="/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features"
  else
    FEATURE_ROOT="$FAST_DATA_ROOT"
  fi
fi
if [[ -z "${SPATIAL_FEATURES_SUBDIR:-}" ]]; then
  if [[ "$ARCH_PRESET" == "spatialstack" ]]; then
    SPATIAL_FEATURES_SUBDIR="6:spatial_features_dec_6,9:spatial_features_dec_9,12:$FAST_DATA_ROOT:spatial_features"
  else
    SPATIAL_FEATURES_SUBDIR="spatial_features"
  fi
fi
if [[ -z "${POINT_MAPS_ROOT:-}" ]]; then
  POINT_MAPS_ROOT="$FEATURE_ROOT"
  if [[ "$ARCH_PRESET" == "spatialstack" ]]; then
    POINT_MAPS_ROOT="$FAST_DATA_ROOT"
  fi
fi
POINT_MAPS_SUBDIR="${POINT_MAPS_SUBDIR:-spatial_features_points}"
DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
FEATURE_LEVELS="${FEATURE_LEVELS:-}"
LLM_LAYERS="${LLM_LAYERS:-}"
PRE_LLM_FEATURES="${PRE_LLM_FEATURES:-}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-}"
GPU_WORKERS="${GPU_WORKERS:-4}"
SHARD_COUNT="${SHARD_COUNT:-$GPU_WORKERS}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
ALLOW_EUCLIDEAN_DEPTH="${ALLOW_EUCLIDEAN_DEPTH:-0}"

mkdir -p logs

cmd=(python -u scripts/probing/extract_depth_probe_features.py
  --model-label "${MODEL_LABEL}"
  --model-path "${MODEL_PATH}"
  --feature-preset "${ARCH_PRESET}"
  --output-root "${OUTPUT_ROOT}"
  --sample-indices "${SAMPLE_INDICES}"
  --data-yaml "${DATA_YAML}"
  --feature-root "${FEATURE_ROOT}"
  --spatial-features-subdir "${SPATIAL_FEATURES_SUBDIR}"
  --point-maps-root "${POINT_MAPS_ROOT}"
  --point-maps-subdir "${POINT_MAPS_SUBDIR}"
  --video-folder "${FAST_DATA_ROOT}"
  --image-folder "${FAST_DATA_ROOT}"
  --dtype "${MODEL_DTYPE}"
  --cache-dtype "${CACHE_DTYPE}"
  --siglip-path "${SIGLIP_PATH}"
  --resume
)
if [[ -n "$FEATURE_LEVELS" ]]; then
  cmd+=(--feature-levels "$FEATURE_LEVELS")
fi
if [[ -n "$LLM_LAYERS" ]]; then
  cmd+=(--llm-layers "$LLM_LAYERS")
fi
if [[ -n "$PRE_LLM_FEATURES" ]]; then
  cmd+=(--pre-llm-features "$PRE_LLM_FEATURES")
fi
if [[ "$ALLOW_EUCLIDEAN_DEPTH" == "1" || "$ALLOW_EUCLIDEAN_DEPTH" == "true" ]]; then
  cmd+=(--allow-euclidean-depth)
fi

if [[ "$GPU_WORKERS" -lt 1 ]]; then
  echo "[ERROR] GPU_WORKERS must be >= 1, got $GPU_WORKERS" >&2
  exit 1
fi
if [[ "$SHARD_INDEX" -lt 0 || $((SHARD_INDEX + GPU_WORKERS)) -gt "$SHARD_COUNT" ]]; then
  echo "[ERROR] Invalid shard range: start=$SHARD_INDEX workers=$GPU_WORKERS count=$SHARD_COUNT" >&2
  exit 1
fi

pids=()
for worker in $(seq 0 $((GPU_WORKERS - 1))); do
  shard_index=$((SHARD_INDEX + worker))
  worker_runtime_root=".offline_runtime/depth_probe_${SLURM_JOB_ID:-manual}_${MODEL_LABEL}_worker${worker}"
  worker_cmd=("${cmd[@]}"
    --device "cuda:0"
    --shard-count "$SHARD_COUNT"
    --shard-index "$shard_index"
    --runtime-root "$worker_runtime_root"
  )
  if [[ -n "$LIMIT_VIDEOS" ]]; then
    worker_cmd+=(--limit-videos "$LIMIT_VIDEOS")
  fi
  echo "[INFO] Launching ${MODEL_LABEL} shard ${shard_index}/${SHARD_COUNT} on CUDA_VISIBLE_DEVICES=${worker}"
  CUDA_VISIBLE_DEVICES="$worker" "${worker_cmd[@]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
