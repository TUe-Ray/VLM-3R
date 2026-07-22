#!/bin/bash
#SBATCH --job-name=PostHocGeoRoPE_CUT3R_Eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

NOTE="Zero-training/post-hoc Geometry-RoPE eval for a trained svf_patch_only CUT3R checkpoint."
echo "-------- Note --------"
echo "  note: $NOTE"

REPO_DIR="${REPO_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
SUBMODULE_DIR="${SUBMODULE_DIR:-$REPO_DIR/thinking-in-space}"
CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vsibench}"

FAST_ROOT="${FAST_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006}"
HF_HOME="${HF_HOME:-$FAST_ROOT/hf_cache}"
VSI_ROOT="${VSI_ROOT:-$FAST_ROOT/vsibench}"
VSI_MEDIA_ROOT="${VSI_MEDIA_ROOT:-$HF_HOME/vsibench}"
DATA_ROOT="${DATA_ROOT:-$FAST_ROOT/data/vlm3r}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-spatial_features_points}"
# Coordinate consistency rule:
# - Post-hoc GeoRoPE eval still needs an explicit coordinate choice.
# - point_maps_ref/pts3d_in_other_view = CUT3R reference/anchor-frame.
# - point_maps_cam/pts3d_in_self_view = per-frame camera coordinates.
# - Keep this choice fixed across comparable eval runs.
CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"

MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_eomt/selec_100%_baseline_40390735}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
RUN_NAME="${RUN_NAME:-posthoc_geo_rope_cut3r}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime}"
RUNTIME_ROOT="$RUNTIME_ROOT/${SLURM_JOB_ID:-$RUN_NAME}"
PRETRAINED_RUNTIME=""

TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
TASK_FILE="${TASK_FILE:-$TASK_DIR/vsibench.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
LIMIT="${LIMIT:-0}"

MODEL_SPATIAL_TOWER="${MODEL_SPATIAL_TOWER:-cut3r}"
MODEL_SPATIAL_FEATURE_DIM="${MODEL_SPATIAL_FEATURE_DIM:-768}"
MODEL_SPATIAL_TOWER_SELECT_FEATURE="${MODEL_SPATIAL_TOWER_SELECT_FEATURE:-all_tokens}"
MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_patch_only_geo_rope_eval}"
MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA="${MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA:-0.0}"
MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_Q="${MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_Q:-}"
MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_K="${MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_K:-}"

cd "$REPO_DIR"

echo "==== Job info ===="
date
echo "HOSTNAME=$(hostname)"
echo "REPO_DIR=$REPO_DIR"
echo "SUBMODULE_DIR=$SUBMODULE_DIR"
echo "TASK_DIR=$TASK_DIR"
echo "TASK_FILE=$TASK_FILE"
echo "FAST_ROOT=$FAST_ROOT"
echo "HF_HOME=$HF_HOME"
echo "VSI_ROOT=$VSI_ROOT"
echo "VSI_MEDIA_ROOT=$VSI_MEDIA_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "CHECK_SPATIAL_SIDECARS=$CHECK_SPATIAL_SIDECARS"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "MODEL_BASE_LOCAL=$MODEL_BASE_LOCAL"
echo "SIGLIP_LOCAL=$SIGLIP_LOCAL"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_FRAMES_NUM=$MAX_FRAMES_NUM"
echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_ATTN_IMPLEMENTATION=$MODEL_ATTN_IMPLEMENTATION"
echo "RUN_NAME=$RUN_NAME"
echo "LIMIT=$LIMIT"
echo "MODEL_SPATIAL_TOWER=$MODEL_SPATIAL_TOWER"
echo "MODEL_SPATIAL_FEATURE_DIM=$MODEL_SPATIAL_FEATURE_DIM"
echo "MODEL_SPATIAL_TOWER_SELECT_FEATURE=$MODEL_SPATIAL_TOWER_SELECT_FEATURE"
echo "MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"
echo "MODEL_GEO_ROPE_FUSION_MODE=$MODEL_GEO_ROPE_FUSION_MODE"
echo "MODEL_GEO_ROPE_FUSION_MAX_DEPTH=$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
echo "MODEL_GEO_ROPE_FUSION_GROUP_SPLIT=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
echo "MODEL_GEO_ROPE_FUSION_LOG_STATS=$MODEL_GEO_ROPE_FUSION_LOG_STATS"
echo "MODEL_GEO_ROPE_POINT_MAP_KEY=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA=$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA"
echo "MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_Q=$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_Q"
echo "MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_K=$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_K"
echo "=================="

for path in "$REPO_DIR" "$SUBMODULE_DIR" "$TASK_DIR" "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

if [[ ! -f "$PRETRAINED_LOCAL/config.json" ]]; then
  echo "[ERROR] Missing pretrained config: $PRETRAINED_LOCAL/config.json"
  exit 1
fi
if [[ ! -f "$PRETRAINED_LOCAL/adapter_config.json" ]]; then
  echo "[ERROR] Missing LoRA adapter config: $PRETRAINED_LOCAL/adapter_config.json"
  exit 1
fi
if [[ ! -f "$PRETRAINED_LOCAL/non_lora_trainables.bin" ]]; then
  echo "[ERROR] Missing non-LoRA trainables: $PRETRAINED_LOCAL/non_lora_trainables.bin"
  echo "[ERROR] Use the real checkpoint directory, not a stale runtime directory with broken symlinks."
  exit 1
fi
if [[ ! -f "$PRETRAINED_LOCAL/adapter_model.bin" && ! -f "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
  echo "[ERROR] Missing LoRA adapter weights: expected adapter_model.bin or adapter_model.safetensors under $PRETRAINED_LOCAL"
  exit 1
fi
if [[ ! -f "$SIGLIP_LOCAL/config.json" ]]; then
  echo "[ERROR] Missing local SigLIP config: $SIGLIP_LOCAL/config.json"
  exit 1
fi
if [[ ! -f "$TASK_FILE" ]]; then
  echo "[ERROR] Missing task yaml: $TASK_FILE"
  exit 1
fi

for parquet in "$VSI_ROOT/test_pruned.parquet" "$VSI_ROOT/test_debiased.parquet"; do
  if [[ ! -f "$parquet" ]]; then
    echo "[ERROR] Missing parquet file: $parquet"
    exit 1
  fi
done

for split in scannet arkitscenes scannetpp; do
  if [[ ! -e "$VSI_MEDIA_ROOT/$split" ]]; then
    echo "[ERROR] Missing video root used by task loader: $VSI_MEDIA_ROOT/$split"
    exit 1
  fi
  if [[ "$CHECK_SPATIAL_SIDECARS" == "True" && ! -d "$SPATIAL_FEATURES_ROOT/$split/$SPATIAL_FEATURES_SUBDIR" ]]; then
    echo "[ERROR] Missing CUT3R point-map sidecar directory: $SPATIAL_FEATURES_ROOT/$split/$SPATIAL_FEATURES_SUBDIR"
    exit 1
  fi
done

if command -v module >/dev/null 2>&1; then
  module purge || true
  unset LD_LIBRARY_PATH || true
  module load 2023 CUDA/12.1.1 || echo "[WARN] module load 2023 CUDA/12.1.1 failed; continuing"
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found and conda.sh missing under $CONDA_BASE"
  exit 1
fi

set +u
conda activate "$CONDA_ENV"
set -u

export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LMMS_EVAL_LAUNCHER=accelerate
export VLM3R_CODE_ROOT="$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -z "${CUDA_HOME:-}" && -d "/leonardo/prod/opt/compilers/cuda/12.1/none" ]]; then
  export CUDA_HOME="/leonardo/prod/opt/compilers/cuda/12.1/none"
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/bin" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R}"
mkdir -p "$OUTPUT_PATH"

echo "==== Runtime Info ===="
date
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "OUTPUT_PATH=$OUTPUT_PATH"
echo "======================"

nvidia-smi || true
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import lmms_eval; print('lmms_eval import ok')"

if [[ "$CHECK_SPATIAL_SIDECARS" == "True" ]]; then
  echo "==== VSiBench media/sidecar preflight ===="
  python - "$VSI_ROOT" "$VSI_MEDIA_ROOT" "$SPATIAL_FEATURES_ROOT" "$SPATIAL_FEATURES_SUBDIR" <<'PY'
import sys
from pathlib import Path

vsi_root = Path(sys.argv[1])
media_root = Path(sys.argv[2])
spatial_root = Path(sys.argv[3])
spatial_subdir = sys.argv[4]
parquets = [vsi_root / "test_pruned.parquet", vsi_root / "test_debiased.parquet"]

try:
    import pandas as pd
    rows = []
    for parquet in parquets:
        rows.extend(pd.read_parquet(parquet, columns=["dataset", "scene_name"]).to_dict("records"))
except Exception:
    import pyarrow.parquet as pq
    rows = []
    for parquet in parquets:
        rows.extend(pq.read_table(parquet, columns=["dataset", "scene_name"]).to_pylist())

missing_media = []
missing_sidecars = []
seen = set()

for row in rows:
    dataset = str(row["dataset"])
    scene = str(row["scene_name"])
    key = (dataset, scene)
    if key in seen:
        continue
    seen.add(key)

    media = media_root / dataset / f"{scene}.mp4"
    sidecar = spatial_root / dataset / spatial_subdir / f"{scene}.pt"
    if not media.is_file():
        missing_media.append(str(media))
    if not sidecar.is_file():
        missing_sidecars.append(str(sidecar))

print(f"Unique videos in eval parquet: {len(seen)}")
print(f"Missing media files: {len(missing_media)}")
print(f"Missing CUT3R point-map sidecars: {len(missing_sidecars)}")

if missing_media:
    print("First missing media files:")
    for path in missing_media[:20]:
        print(f"  {path}")
if missing_sidecars:
    print("First missing sidecars:")
    for path in missing_sidecars[:20]:
        print(f"  {path}")
if missing_media or missing_sidecars:
    raise SystemExit("Post-hoc GeoRoPE eval preflight failed.")
PY
  echo "=========================================="
fi

prepare_runtime_pretrained() {
  local runtime_dir="$RUNTIME_ROOT/pretrained_siglip_local"
  mkdir -p "$runtime_dir"

  local f base
  for f in "$PRETRAINED_LOCAL"/*; do
    base="$(basename "$f")"
    if [[ "$base" == "config.json" ]]; then
      continue
    fi
    ln -sfn "$f" "$runtime_dir/$base"
  done

  cp "$PRETRAINED_LOCAL/config.json" "$runtime_dir/config.json"
  python - "$runtime_dir/config.json" "$SIGLIP_LOCAL" "$MODEL_SPATIAL_TOWER" "$MODEL_SPATIAL_FEATURE_DIM" "$MODEL_SPATIAL_TOWER_SELECT_FEATURE" "$MODEL_FUSION_BLOCK" "$MODEL_GEO_ROPE_FUSION_MODE" "$MODEL_GEO_ROPE_FUSION_MAX_DEPTH" "$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT" "$MODEL_GEO_ROPE_FUSION_LOG_STATS" "$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA" "$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_Q" "$MODEL_GEO_ROPE_FUSION_EVAL_LAMBDA_K" "$MODEL_GEO_ROPE_POINT_MAP_KEY" <<'PY'
import json
import sys

def normalize_point_map_key(value):
    aliases = {
        "ref": "point_maps_ref",
        "reference": "point_maps_ref",
        "anchor": "point_maps_ref",
        "point_maps_ref": "point_maps_ref",
        "pts3d_in_other_view": "point_maps_ref",
        "cam": "point_maps_cam",
        "camera": "point_maps_cam",
        "self": "point_maps_cam",
        "point_maps_cam": "point_maps_cam",
        "pts3d_in_self_view": "point_maps_cam",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise SystemExit(f"Unsupported MODEL_GEO_ROPE_POINT_MAP_KEY={value!r}")
    return aliases[key]

cfg_path = sys.argv[1]
siglip_local = sys.argv[2]
spatial_tower = sys.argv[3]
spatial_feature_dim = int(sys.argv[4])
spatial_tower_select_feature = sys.argv[5]
fusion_block = sys.argv[6]
geo_rope_fusion_mode = sys.argv[7]
geo_rope_fusion_max_depth = float(sys.argv[8])
geo_rope_fusion_group_split = sys.argv[9]
geo_rope_fusion_log_stats = sys.argv[10].lower() in {"1", "true", "yes", "y", "on"}
geo_rope_fusion_eval_lambda = float(sys.argv[11])
geo_rope_fusion_eval_lambda_q = sys.argv[12]
geo_rope_fusion_eval_lambda_k = sys.argv[13]
geo_rope_point_map_key = normalize_point_map_key(sys.argv[14])

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

training_point_map_key = (
    cfg.get("geo_rope_training_point_map_key")
    or cfg.get("geometry_training_point_map_key")
    or cfg.get("geo_rope_point_map_key")
    or cfg.get("geometry_point_map_key")
)
training_point_map_key = normalize_point_map_key(training_point_map_key) if training_point_map_key else None
if training_point_map_key and training_point_map_key != geo_rope_point_map_key:
    raise SystemExit(
        "GeoRoPE point-map coordinate mismatch: checkpoint training used "
        f"{training_point_map_key}, eval requested {geo_rope_point_map_key}."
    )

cfg["mm_vision_tower"] = siglip_local
if "vision_tower" in cfg:
    cfg["vision_tower"] = siglip_local
cfg["spatial_tower"] = spatial_tower
cfg["spatial_feature_dim"] = spatial_feature_dim
cfg["spatial_tower_select_feature"] = spatial_tower_select_feature
cfg["fusion_block"] = fusion_block
cfg["geo_rope_fusion_mode"] = geo_rope_fusion_mode
cfg["geo_rope_fusion_max_depth"] = geo_rope_fusion_max_depth
cfg["geo_rope_fusion_group_split"] = geo_rope_fusion_group_split
cfg["geo_rope_fusion_log_stats"] = geo_rope_fusion_log_stats
cfg["geometry_rope_mode"] = geo_rope_fusion_mode
cfg["geometry_rope_max_depth"] = geo_rope_fusion_max_depth
cfg["geometry_rope_group_split"] = geo_rope_fusion_group_split
cfg["geometry_rope_log_stats"] = geo_rope_fusion_log_stats
cfg["geo_rope_fusion_eval_lambda"] = geo_rope_fusion_eval_lambda
cfg["geometry_rope_eval_lambda"] = geo_rope_fusion_eval_lambda
cfg["geo_rope_training_point_map_key"] = training_point_map_key or geo_rope_point_map_key
cfg["geo_rope_point_map_key"] = geo_rope_point_map_key
cfg["geometry_point_map_key"] = geo_rope_point_map_key
if geo_rope_fusion_eval_lambda_q:
    cfg["geo_rope_fusion_eval_lambda_q"] = float(geo_rope_fusion_eval_lambda_q)
if geo_rope_fusion_eval_lambda_k:
    cfg["geo_rope_fusion_eval_lambda_k"] = float(geo_rope_fusion_eval_lambda_k)

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

  PRETRAINED_RUNTIME="$runtime_dir"
}

prepare_runtime_pretrained

cd "$SUBMODULE_DIR"

MODEL_ARGS="pretrained=$PRETRAINED_RUNTIME,model_base=$MODEL_BASE_LOCAL"
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_ARGS+=",model_name=$MODEL_NAME"
fi
MODEL_ARGS+=",conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_FRAMES_NUM,attn_implementation=$MODEL_ATTN_IMPLEMENTATION"
MODEL_ARGS+=",spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR"
MODEL_ARGS+=",geo_rope_point_map_key=$MODEL_GEO_ROPE_POINT_MAP_KEY"

echo "Running post-hoc Geometry-RoPE offline evaluation"
echo "PRETRAINED_RUNTIME=$PRETRAINED_RUNTIME"
echo "MODEL_ARGS=$MODEL_ARGS"

cmd=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  -m lmms_eval
  --model vlm_3r
  --model_args "$MODEL_ARGS"
  --tasks "$TASK_DIR"
  --batch_size "$BATCH_SIZE"
  --log_samples
  --log_samples_suffix "$RUN_NAME"
  --output_path "$OUTPUT_PATH"
)

if [[ -n "$LIMIT" && "$LIMIT" != "0" ]]; then
  cmd+=(--limit "$LIMIT")
fi

printf '[CMD] %q ' "${cmd[@]}"
echo

cmd_pid=""
cleanup_cmd_group() {
  if [[ -n "${cmd_pid:-}" ]] && kill -0 "$cmd_pid" >/dev/null 2>&1; then
    kill -- -"${cmd_pid}" >/dev/null 2>&1 || true
  fi
  rm -rf "${RUNTIME_ROOT:-}" 2>/dev/null || true
}

trap cleanup_cmd_group EXIT

setsid "${cmd[@]}" &
cmd_pid=$!
if ! wait "$cmd_pid"; then
  status=$?
  echo "[ERROR] Evaluation command failed with exit code $status"
  cleanup_cmd_group
  exit "$status"
fi

trap - EXIT
echo "[DONE] Output path: $OUTPUT_PATH"
