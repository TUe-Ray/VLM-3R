#!/bin/bash
#SBATCH --job-name=cut3r_dec_layer_baseline_eval
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

NOTE="Leonardo offline VSI-Bench eval for trained CUT3R decoder-layer ablation checkpoints. Select the checkpoint and sidecars with CUT3R_LAYER=-2 or CUT3R_LAYER=-4."
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
CUT3R_LAYER="${CUT3R_LAYER:--2}"
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
SPATIAL_FEATURES_ROOT_DEFAULT="/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features"

case "$CUT3R_LAYER" in
  -2|m2|dec_m2|spatial_features_dec_m2)
    CUT3R_LAYER="-2"
    DEFAULT_PRETRAINED_LOCAL="$TRAIN_MODEL_ROOT/cut3r_dec_m2_baseline_41701410"
    DEFAULT_SPATIAL_FEATURES_SUBDIR="spatial_features_dec_m2"
    DEFAULT_RUN_NAME="eval_cut3r_dec_m2_baseline_41701410"
    ;;
  -4|m4|dec_m4|spatial_features_dec_m4)
    CUT3R_LAYER="-4"
    DEFAULT_PRETRAINED_LOCAL="$TRAIN_MODEL_ROOT/cut3r_dec_m4_baseline_41701424"
    DEFAULT_SPATIAL_FEATURES_SUBDIR="spatial_features_dec_m4"
    DEFAULT_RUN_NAME="eval_cut3r_dec_m4_baseline_41701424"
    ;;
  *)
    echo "[ERROR] Unsupported CUT3R_LAYER='$CUT3R_LAYER'. Use -2 or -4."
    exit 1
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$DEFAULT_PRETRAINED_LOCAL}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime}"
RUNTIME_ROOT="$RUNTIME_ROOT/${SLURM_JOB_ID:-$DEFAULT_RUN_NAME}"
PRETRAINED_RUNTIME=""
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$SPATIAL_FEATURES_ROOT_DEFAULT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-$DEFAULT_SPATIAL_FEATURES_SUBDIR}"
USE_RUNTIME_CUT3R_FEATURES="${USE_RUNTIME_CUT3R_FEATURES:-True}"
if [[ -z "${CHECK_SPATIAL_SIDECARS+x}" ]]; then
  if [[ "$USE_RUNTIME_CUT3R_FEATURES" == "True" ]]; then
    CHECK_SPATIAL_SIDECARS="False"
  else
    CHECK_SPATIAL_SIDECARS="True"
  fi
fi
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
TASK_FILE="${TASK_FILE:-$TASK_DIR/vsibench.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
LIMIT="${LIMIT:-0}"

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
echo "CUT3R_LAYER=$CUT3R_LAYER"
echo "SPATIAL_FEATURES_ROOT=$SPATIAL_FEATURES_ROOT"
echo "SPATIAL_FEATURES_SUBDIR=$SPATIAL_FEATURES_SUBDIR"
echo "USE_RUNTIME_CUT3R_FEATURES=$USE_RUNTIME_CUT3R_FEATURES"
echo "CHECK_SPATIAL_SIDECARS=$CHECK_SPATIAL_SIDECARS"
echo "CUT3R_WEIGHTS=$CUT3R_WEIGHTS"
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
if [[ "$USE_RUNTIME_CUT3R_FEATURES" == "True" && ! -f "$CUT3R_WEIGHTS" ]]; then
  echo "[ERROR] Runtime CUT3R features requested but weights are missing: $CUT3R_WEIGHTS"
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
    echo "[ERROR] The offline task resolves videos as HF_HOME/vsibench/<dataset>/<scene>.mp4."
    exit 1
  fi
  if [[ "$CHECK_SPATIAL_SIDECARS" == "True" && ! -d "$SPATIAL_FEATURES_ROOT/$split/$SPATIAL_FEATURES_SUBDIR" ]]; then
    echo "[ERROR] Missing CUT3R decoder-layer sidecar directory: $SPATIAL_FEATURES_ROOT/$split/$SPATIAL_FEATURES_SUBDIR"
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
print(f"Missing CUT3R decoder-layer sidecars: {len(missing_sidecars)}")
if missing_media:
    print("First missing media files:")
    for path in missing_media[:20]:
        print(f"  {path}")
if missing_sidecars:
    print("First missing sidecars:")
    for path in missing_sidecars[:20]:
        print(f"  {path}")
if missing_media or missing_sidecars:
    raise SystemExit(
        "CUT3R decoder-layer eval preflight failed. Extract the VSI-Bench eval sidecars first "
        "(for example with extract_vsibench_cut3r_layer_features.sh), or set "
        "CHECK_SPATIAL_SIDECARS=False to bypass this guard."
    )
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
  python - "$runtime_dir/config.json" "$SIGLIP_LOCAL" "$USE_RUNTIME_CUT3R_FEATURES" "$CUT3R_LAYER" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
siglip_local = sys.argv[2]
use_runtime_cut3r_features = sys.argv[3].lower() in {"1", "true", "yes", "y", "on"}
cut3r_layer = int(sys.argv[4])

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["mm_vision_tower"] = siglip_local
if "vision_tower" in cfg:
    cfg["vision_tower"] = siglip_local
if use_runtime_cut3r_features:
    cfg["spatial_tower"] = "cut3r"
    cfg["spatial_tower_select_layer"] = cut3r_layer
    cfg["spatial_tower_select_feature"] = "all_tokens"

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
if [[ "$USE_RUNTIME_CUT3R_FEATURES" == "True" ]]; then
  MODEL_ARGS+=",spatial_tower=cut3r,spatial_tower_select_feature=all_tokens"
else
  MODEL_ARGS+=",spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR"
fi

echo "Running Leonardo offline evaluation"
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
