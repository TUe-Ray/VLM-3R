#!/bin/bash
#SBATCH --job-name=SpatialStackProbeExtract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=0
#SBATCH --output=logs/cut3r_spatialstack_probe_extract_%x_%j.out
#SBATCH --error=logs/cut3r_spatialstack_probe_extract_%x_%j.err

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

MODEL_LABEL="${MODEL_LABEL:-cut3r_spatialstack_44323703}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
SPATIAL_FEATURE_ROOT="${SPATIAL_FEATURE_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6,9:spatial_features_dec_9,12:$FAST_DATA_ROOT:spatial_features}"
POINT_MAPS_ROOT="${POINT_MAPS_ROOT:-$FAST_DATA_ROOT}"
POINT_MAPS_SUBDIR="${POINT_MAPS_SUBDIR:-spatial_features_points}"
DATA_YAML="${DATA_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-}"
GPU_WORKERS="${GPU_WORKERS:-4}"
SHARD_COUNT="${SHARD_COUNT:-$GPU_WORKERS}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SIGLIP_PATH="${SIGLIP_PATH:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"

mkdir -p logs

cmd=(python -u scripts/probing/extract_depth_probe_features.py
  --model-label "${MODEL_LABEL}"
  --output-root "${OUTPUT_ROOT}"
  --sample-indices "${SAMPLE_INDICES}"
  --data-yaml "${DATA_YAML}"
  --feature-root "${SPATIAL_FEATURE_ROOT}"
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
