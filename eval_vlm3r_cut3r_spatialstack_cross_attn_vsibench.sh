#!/bin/bash
#SBATCH --job-name=Eval_CUT3R_SpatialStack_CrossAttn_VSI
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

NOTE="Leonardo offline VSI-Bench eval for CUT3R SpatialStack cross-attn dec[6,9,12] -> LLM[0,1,2], no auxiliary losses."
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

TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$TRAIN_MODEL_ROOT/cut3r_spatialstack_cross_attn_45303862}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime}"
RUNTIME_ROOT="$RUNTIME_ROOT/${SLURM_JOB_ID:-eval_cut3r_spatialstack_cross_attn_vsibench}"
PRETRAINED_RUNTIME=""

TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
TASK_FILE="${TASK_FILE:-$TASK_DIR/vsibench.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
RUN_NAME="${RUN_NAME:-eval_cut3r_spatialstack_cross_attn_vsibench}"
LIMIT="${LIMIT:-0}"
LMMS_VERBOSITY="${LMMS_VERBOSITY:-DEBUG}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
CUT3R_TOKEN_FEATURES_ROOT="${CUT3R_TOKEN_FEATURES_ROOT:-$FAST_ROOT/data/vlm3r}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6;9:spatial_features_dec_9;12:$CUT3R_TOKEN_FEATURES_ROOT:spatial_features}"
CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"
CUT3R_SPATIALSTACK_LAYERS="${CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
CUT3R_SPATIALSTACK_LLM_LAYERS="${CUT3R_SPATIALSTACK_LLM_LAYERS:-0,1,2}"
CUT3R_SPATIALSTACK_FEATURE_DIM="${CUT3R_SPATIALSTACK_FEATURE_DIM:-768}"
CUT3R_SPATIALSTACK_FEATURE_KEY="${CUT3R_SPATIALSTACK_FEATURE_KEY:-cut3r_dec_layers}"
CUT3R_SPATIALSTACK_ZERO_INIT="${CUT3R_SPATIALSTACK_ZERO_INIT:-True}"
CUT3R_SPATIALSTACK_LOG_FIRST_N="${CUT3R_SPATIALSTACK_LOG_FIRST_N:-3}"
CUT3R_SPATIALSTACK_FUSION_TYPE="${CUT3R_SPATIALSTACK_FUSION_TYPE:-cross_attn}"
CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS="${CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS:-}"
CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT="${CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT:-0.0}"
CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT="${CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT:-True}"
CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY="${CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY:-True}"

cd "$REPO_DIR"

resolve_pretrained_local() {
  is_pretrained_dir() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    [[ -f "$dir/adapter_config.json" ]] || return 1
    [[ -f "$dir/non_lora_trainables.bin" ]] || return 1
    [[ -f "$dir/adapter_model.bin" || -f "$dir/adapter_model.safetensors" ]] || return 1
  }

  case "${PRETRAINED_LOCAL,,}" in
    auto|final)
      if [[ ! -d "$TRAIN_OUTPUT_DIR" ]]; then
        echo "[ERROR] TRAIN_OUTPUT_DIR does not exist for PRETRAINED_LOCAL=$PRETRAINED_LOCAL: $TRAIN_OUTPUT_DIR"
        exit 1
      fi

      if ! is_pretrained_dir "$TRAIN_OUTPUT_DIR"; then
        echo "[ERROR] Final checkpoint root is not ready: $TRAIN_OUTPUT_DIR"
        echo "[ERROR] Refusing to auto-select checkpoint-* because those are periodic step checkpoints, not the final post-training save."
        exit 1
      fi
      PRETRAINED_LOCAL="$TRAIN_OUTPUT_DIR"
      ;;
  esac
}

resolve_pretrained_local

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
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "CUT3R_TOKEN_FEATURES_ROOT=$CUT3R_TOKEN_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "CHECK_SPATIAL_SIDECARS=$CHECK_SPATIAL_SIDECARS"
echo "TRAIN_OUTPUT_DIR=$TRAIN_OUTPUT_DIR"
echo "CUT3R_SPATIALSTACK_LAYERS=$CUT3R_SPATIALSTACK_LAYERS"
echo "CUT3R_SPATIALSTACK_LLM_LAYERS=$CUT3R_SPATIALSTACK_LLM_LAYERS"
echo "CUT3R_SPATIALSTACK_FEATURE_DIM=$CUT3R_SPATIALSTACK_FEATURE_DIM"
echo "CUT3R_SPATIALSTACK_FEATURE_KEY=$CUT3R_SPATIALSTACK_FEATURE_KEY"
echo "CUT3R_SPATIALSTACK_ZERO_INIT=$CUT3R_SPATIALSTACK_ZERO_INIT"
echo "CUT3R_SPATIALSTACK_LOG_FIRST_N=$CUT3R_SPATIALSTACK_LOG_FIRST_N"
echo "CUT3R_SPATIALSTACK_FUSION_TYPE=$CUT3R_SPATIALSTACK_FUSION_TYPE"
echo "CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS=$CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS"
echo "CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT=$CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT"
echo "CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT=$CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT"
echo "CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY=$CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY"
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
echo "LMMS_VERBOSITY=$LMMS_VERBOSITY"
echo "EXTRA_MODEL_ARGS=$EXTRA_MODEL_ARGS"
echo "=================="

if [[ -z "$PRETRAINED_LOCAL" ]]; then
  echo "[ERROR] PRETRAINED_LOCAL is empty. Set it to the final completed checkpoint path before eval."
  exit 1
fi

for path in "$REPO_DIR" "$SUBMODULE_DIR" "$TASK_DIR" "$PRETRAINED_LOCAL" "$MODEL_BASE_LOCAL" "$SIGLIP_LOCAL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
    exit 1
  fi
done

for file in "$PRETRAINED_LOCAL/config.json" "$PRETRAINED_LOCAL/adapter_config.json" "$PRETRAINED_LOCAL/non_lora_trainables.bin" "$SIGLIP_LOCAL/config.json" "$TASK_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "[ERROR] Missing required file: $file"
    exit 1
  fi
done

if [[ ! -f "$PRETRAINED_LOCAL/adapter_model.bin" && ! -f "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
  echo "[ERROR] Missing LoRA adapter weights under $PRETRAINED_LOCAL"
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
  echo "==== VSI-Bench media/SpatialStack sidecar preflight ===="
  python - "$VSI_ROOT" "$VSI_MEDIA_ROOT" "$SPATIAL_FEATURES_ROOT" "$SPATIAL_FEATURES_SUBDIR" <<'PY'
import re
import sys
from pathlib import Path

vsi_root = Path(sys.argv[1])
media_root = Path(sys.argv[2])
default_spatial_root = Path(sys.argv[3])
subdir_spec = sys.argv[4]
parquets = [vsi_root / "test_pruned.parquet", vsi_root / "test_debiased.parquet"]


def parse_layer_key(value, infer_from_path=False):
    text = str(value).strip().strip("/\\")
    if not text:
        return None

    def parse_token(token):
        token = str(token).strip().lower()
        for prefix in ("decoder", "dec", "layer"):
            if token.startswith(prefix):
                token = token[len(prefix):].strip("_-")
        if token.startswith("m") and token[1:].isdigit():
            token = "-" + token[1:]
        try:
            return str(int(token))
        except ValueError:
            return None

    parsed = parse_token(text)
    if parsed is not None or not infer_from_path:
        return parsed
    match = re.search(r"(?:^|[_-])(?:decoder|dec|layer)?[_-]?(m?-?\d+)$", Path(text).name.lower())
    return parse_token(match.group(1)) if match else None


def split_layer_specs(spec):
    parts = [part.strip() for part in spec.replace(";", ",").split(",") if part.strip()]
    parsed = []
    for part in parts:
        colon_pieces = [piece.strip() for piece in part.split(":")]
        if len(colon_pieces) >= 3 and parse_layer_key(colon_pieces[0]) is not None:
            layer_key = parse_layer_key(colon_pieces[0])
            layer_root = Path(":".join(colon_pieces[1:-1]).strip())
            layer_subdir = colon_pieces[-1]
        elif ":" in part:
            left, right = [piece.strip() for piece in part.split(":", 1)]
            left_key = parse_layer_key(left)
            right_key = parse_layer_key(right)
            if left_key is not None:
                layer_key, layer_root, layer_subdir = left_key, default_spatial_root, right
            elif right_key is not None:
                layer_key, layer_root, layer_subdir = right_key, default_spatial_root, left
            else:
                raise ValueError(f"Cannot infer layer from spec: {part!r}")
        else:
            layer_key = parse_layer_key(part, infer_from_path=True)
            if layer_key is None:
                raise ValueError(f"Cannot infer layer from spec: {part!r}")
            layer_root, layer_subdir = default_spatial_root, part
        parsed.append((layer_key, layer_root, layer_subdir))
    return parsed


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

layer_specs = split_layer_specs(subdir_spec)
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
    if not media.is_file():
        missing_media.append(str(media))
    for layer_key, layer_root, layer_subdir in layer_specs:
        sidecar = layer_root / dataset / layer_subdir / f"{scene}.pt"
        if not sidecar.is_file():
            missing_sidecars.append(f"layer {layer_key}: {sidecar}")

print(f"Unique videos in eval parquet: {len(seen)}")
print("Layer sidecar specs:")
for layer_key, layer_root, layer_subdir in layer_specs:
    print(f"  layer {layer_key}: root={layer_root}, subdir={layer_subdir}")
print(f"Missing media files: {len(missing_media)}")
print(f"Missing layer sidecars: {len(missing_sidecars)}")
if missing_media:
    print("First missing media files:")
    for path in missing_media[:20]:
        print(f"  {path}")
if missing_sidecars:
    print("First missing sidecars:")
    for path in missing_sidecars[:30]:
        print(f"  {path}")
if missing_media or missing_sidecars:
    raise SystemExit("SpatialStack eval preflight failed.")
PY
  echo "========================================================"
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
  python - "$runtime_dir/config.json" "$SIGLIP_LOCAL" "$CUT3R_SPATIALSTACK_LAYERS" "$CUT3R_SPATIALSTACK_LLM_LAYERS" "$CUT3R_SPATIALSTACK_FEATURE_DIM" "$CUT3R_SPATIALSTACK_FEATURE_KEY" "$CUT3R_SPATIALSTACK_ZERO_INIT" "$CUT3R_SPATIALSTACK_LOG_FIRST_N" "$CUT3R_SPATIALSTACK_FUSION_TYPE" "$CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS" "$CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT" "$CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT" "$CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
siglip_local = sys.argv[2]
spatialstack_layers = sys.argv[3]
spatialstack_llm_layers = sys.argv[4]
spatialstack_feature_dim = int(sys.argv[5])
spatialstack_feature_key = sys.argv[6]
spatialstack_zero_init = sys.argv[7]
spatialstack_log_first_n = int(sys.argv[8])
spatialstack_fusion_type = sys.argv[9]
spatialstack_cross_attn_heads_raw = sys.argv[10]
spatialstack_cross_attn_dropout = float(sys.argv[11])
spatialstack_cross_attn_zero_init = sys.argv[12]
spatialstack_cross_attn_same_frame_only = sys.argv[13]


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value):
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return int(text)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["mm_vision_tower"] = siglip_local
if "vision_tower" in cfg:
    cfg["vision_tower"] = siglip_local

cfg["use_cut3r_spatialstack"] = True
cfg["tune_cut3r_spatialstack"] = False
cfg["cut3r_spatialstack_layers"] = spatialstack_layers
cfg["cut3r_spatialstack_llm_layers"] = spatialstack_llm_layers
cfg["cut3r_spatialstack_feature_dim"] = spatialstack_feature_dim
cfg["spatial_feature_dim"] = spatialstack_feature_dim
cfg["spatial_tower"] = "cut3r"
cfg["spatial_tower_preextracted_only"] = True
cfg["spatial_tower_select_feature"] = "all_tokens"
cfg["cut3r_spatialstack_feature_key"] = spatialstack_feature_key
cfg["cut3r_spatialstack_zero_init"] = as_bool(spatialstack_zero_init)
cfg["cut3r_spatialstack_log_first_n"] = spatialstack_log_first_n
cfg["cut3r_spatialstack_fusion_type"] = spatialstack_fusion_type
cfg["cut3r_spatialstack_cross_attn_heads"] = optional_int(spatialstack_cross_attn_heads_raw)
cfg["cut3r_spatialstack_cross_attn_dropout"] = spatialstack_cross_attn_dropout
cfg["cut3r_spatialstack_cross_attn_zero_init"] = as_bool(spatialstack_cross_attn_zero_init)
cfg["cut3r_spatialstack_cross_attn_same_frame_only"] = as_bool(spatialstack_cross_attn_same_frame_only)
cfg["use_auxiliary_geometry_head"] = False
cfg["use_auxiliary_geometry_loss"] = False
cfg["use_bev_supervision"] = False
cfg["use_depth_supervision"] = False
cfg["llm_visual_3d_rope_enable"] = False

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
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
  MODEL_ARGS+=",$EXTRA_MODEL_ARGS"
fi

echo "Running Leonardo offline SpatialStack VSI-Bench evaluation"
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
  --verbosity "$LMMS_VERBOSITY"
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
