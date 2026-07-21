#!/bin/bash
#SBATCH --job-name=DBGMetricGeoProj_CUT3R_Eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --mem=0

set -euo pipefail

NOTE="Leonardo offline VSI-Bench eval for MetricGroundedGeometryProjection CUT3R VLM-3R checkpoints. By default this reads pre-extracted CUT3R point-map sidecars under DATA_ROOT instead of running CUT3R on the fly."
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
# - This eval must use the same CUT3R point-map coordinate source as training.
# - point_maps_ref/pts3d_in_other_view = CUT3R reference/anchor-frame.
# - point_maps_cam/pts3d_in_self_view = per-frame camera coordinates.
# - Do not introduce eval-only aliases that change ref<->cam for a checkpoint.
MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
USE_RUNTIME_CUT3R_GEOMETRY="${USE_RUNTIME_CUT3R_GEOMETRY:-False}"
CHECK_SPATIAL_SIDECARS="${CHECK_SPATIAL_SIDECARS:-True}"
CUT3R_WEIGHTS="${CUT3R_WEIGHTS:-$REPO_DIR/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"

MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$MODEL_ROOT/Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE_LOCAL="${MODEL_BASE_LOCAL:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
SIGLIP_LOCAL="${SIGLIP_LOCAL:-$MODEL_ROOT/siglip-so400m-patch14-384}"
RUN_NAME="${RUN_NAME:-eval_metric_geometry_projection_cut3r_sidecars}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_DIR/.offline_runtime}"
# Use a job-specific subdir to prevent concurrent jobs from clobbering each other.
# SLURM_JOB_ID is unique per job; fall back to RUN_NAME for non-SLURM use.
RUNTIME_ROOT="$RUNTIME_ROOT/${SLURM_JOB_ID:-$RUN_NAME}"
PRETRAINED_RUNTIME=""

TASK_DIR="${TASK_DIR:-$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline}"
TASK_FILE="${TASK_FILE:-$TASK_DIR/vsibench.yaml}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MODEL_NAME="${MODEL_NAME:-vlm-3r-llava-qwen2-lora}"
LIMIT="${LIMIT:-0}"

MODEL_USE_GEOMETRY_AWARE_PROJECTION="${MODEL_USE_GEOMETRY_AWARE_PROJECTION:-True}"
MODEL_SPATIAL_ENCODER_TYPE="${MODEL_SPATIAL_ENCODER_TYPE:-cut3r}"
MODEL_SPATIAL_TOWER="${MODEL_SPATIAL_TOWER:-cut3r_points}"
MODEL_GEOMETRY_POSITION_MODE="${MODEL_GEOMETRY_POSITION_MODE:-spherical}"
MODEL_NUM_GEOMETRY_PROJECTION_LAYERS="${MODEL_NUM_GEOMETRY_PROJECTION_LAYERS:-1}"
MODEL_GEOMETRY_PROJECTION_NUM_HEADS="${MODEL_GEOMETRY_PROJECTION_NUM_HEADS:-16}"
MODEL_USE_AUXILIARY_GEOMETRY_HEAD="${MODEL_USE_AUXILIARY_GEOMETRY_HEAD:-True}"
MODEL_USE_AUXILIARY_GEOMETRY_LOSS="${MODEL_USE_AUXILIARY_GEOMETRY_LOSS:-False}"
MODEL_AUX_GEOMETRY_TARGETS="${MODEL_AUX_GEOMETRY_TARGETS:-azimuth,elevation,log_distance}"
MODEL_LAMBDA_GEO="${MODEL_LAMBDA_GEO:-0.1}"
MODEL_GEOMETRY_LOSS_TYPE="${MODEL_GEOMETRY_LOSS_TYPE:-smooth_l1}"
MODEL_DETACH_GEOMETRY_TARGETS="${MODEL_DETACH_GEOMETRY_TARGETS:-True}"
MODEL_GEOMETRY_GATE_INIT="${MODEL_GEOMETRY_GATE_INIT:-0.0}"
MODEL_USE_GEOMETRY_CONFIDENCE_MASK="${MODEL_USE_GEOMETRY_CONFIDENCE_MASK:-True}"
MODEL_ALLOW_MISSING_GEOMETRY_TARGETS="${MODEL_ALLOW_MISSING_GEOMETRY_TARGETS:-False}"
MODEL_GEOMETRY_POSITION_MAX_ABS="${MODEL_GEOMETRY_POSITION_MAX_ABS:-10.0}"
MODEL_GEOMETRY_FIXED_SCENE_SCALE="${MODEL_GEOMETRY_FIXED_SCENE_SCALE:-5.0}"
MODEL_GEOMETRY_PROJECTION_DROPOUT="${MODEL_GEOMETRY_PROJECTION_DROPOUT:-0.0}"
MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-}"

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
echo "MODEL_GEO_ROPE_POINT_MAP_KEY=$MODEL_GEO_ROPE_POINT_MAP_KEY"
echo "USE_RUNTIME_CUT3R_GEOMETRY=$USE_RUNTIME_CUT3R_GEOMETRY"
echo "CHECK_SPATIAL_SIDECARS=$CHECK_SPATIAL_SIDECARS"
echo "CUT3R_WEIGHTS=$CUT3R_WEIGHTS"
echo "PRETRAINED_LOCAL=$PRETRAINED_LOCAL"
echo "MODEL_BASE_LOCAL=$MODEL_BASE_LOCAL"
echo "SIGLIP_LOCAL=$SIGLIP_LOCAL"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_FRAMES_NUM=$MAX_FRAMES_NUM"
echo "MODEL_NAME=$MODEL_NAME"
echo "RUN_NAME=$RUN_NAME"
echo "LIMIT=$LIMIT"
echo "MODEL_USE_GEOMETRY_AWARE_PROJECTION=$MODEL_USE_GEOMETRY_AWARE_PROJECTION"
echo "MODEL_SPATIAL_ENCODER_TYPE=$MODEL_SPATIAL_ENCODER_TYPE"
echo "MODEL_SPATIAL_TOWER=$MODEL_SPATIAL_TOWER"
echo "MODEL_GEOMETRY_POSITION_MODE=$MODEL_GEOMETRY_POSITION_MODE"
echo "MODEL_NUM_GEOMETRY_PROJECTION_LAYERS=$MODEL_NUM_GEOMETRY_PROJECTION_LAYERS"
echo "MODEL_GEOMETRY_PROJECTION_NUM_HEADS=$MODEL_GEOMETRY_PROJECTION_NUM_HEADS"
echo "MODEL_USE_AUXILIARY_GEOMETRY_HEAD=$MODEL_USE_AUXILIARY_GEOMETRY_HEAD"
echo "MODEL_USE_AUXILIARY_GEOMETRY_LOSS=$MODEL_USE_AUXILIARY_GEOMETRY_LOSS"
echo "MODEL_AUX_GEOMETRY_TARGETS=$MODEL_AUX_GEOMETRY_TARGETS"
echo "MODEL_LAMBDA_GEO=$MODEL_LAMBDA_GEO"
echo "MODEL_GEOMETRY_LOSS_TYPE=$MODEL_GEOMETRY_LOSS_TYPE"
echo "MODEL_DETACH_GEOMETRY_TARGETS=$MODEL_DETACH_GEOMETRY_TARGETS"
echo "MODEL_GEOMETRY_GATE_INIT=$MODEL_GEOMETRY_GATE_INIT"
echo "MODEL_USE_GEOMETRY_CONFIDENCE_MASK=$MODEL_USE_GEOMETRY_CONFIDENCE_MASK"
echo "MODEL_ALLOW_MISSING_GEOMETRY_TARGETS=$MODEL_ALLOW_MISSING_GEOMETRY_TARGETS"
echo "MODEL_GEOMETRY_POSITION_MAX_ABS=$MODEL_GEOMETRY_POSITION_MAX_ABS"
echo "MODEL_GEOMETRY_FIXED_SCENE_SCALE=$MODEL_GEOMETRY_FIXED_SCENE_SCALE"
echo "MODEL_GEOMETRY_PROJECTION_DROPOUT=$MODEL_GEOMETRY_PROJECTION_DROPOUT"
echo "MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"
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
if [[ ! -f "$SIGLIP_LOCAL/config.json" ]]; then
  echo "[ERROR] Missing local SigLIP config: $SIGLIP_LOCAL/config.json"
  exit 1
fi
if [[ "$USE_RUNTIME_CUT3R_GEOMETRY" == "True" && ! -f "$CUT3R_WEIGHTS" ]]; then
  echo "[ERROR] Runtime CUT3R geometry requested but weights are missing: $CUT3R_WEIGHTS"
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
        table = pq.read_table(parquet, columns=["dataset", "scene_name"])
        rows.extend(table.to_pylist())

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
    raise SystemExit(
        "RoPE eval preflight failed. Generate/copy the missing videos and CUT3R point-map sidecars, "
        "or set CHECK_SPATIAL_SIDECARS=False to bypass this guard."
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
  python - "$runtime_dir/config.json" "$SIGLIP_LOCAL" "$CUT3R_WEIGHTS" "$USE_RUNTIME_CUT3R_GEOMETRY" "$MODEL_USE_GEOMETRY_AWARE_PROJECTION" "$MODEL_SPATIAL_ENCODER_TYPE" "$MODEL_SPATIAL_TOWER" "$MODEL_GEOMETRY_POSITION_MODE" "$MODEL_NUM_GEOMETRY_PROJECTION_LAYERS" "$MODEL_GEOMETRY_PROJECTION_NUM_HEADS" "$MODEL_USE_AUXILIARY_GEOMETRY_HEAD" "$MODEL_USE_AUXILIARY_GEOMETRY_LOSS" "$MODEL_AUX_GEOMETRY_TARGETS" "$MODEL_LAMBDA_GEO" "$MODEL_GEOMETRY_LOSS_TYPE" "$MODEL_DETACH_GEOMETRY_TARGETS" "$MODEL_GEOMETRY_GATE_INIT" "$MODEL_USE_GEOMETRY_CONFIDENCE_MASK" "$MODEL_ALLOW_MISSING_GEOMETRY_TARGETS" "$MODEL_GEOMETRY_POSITION_MAX_ABS" "$MODEL_GEOMETRY_FIXED_SCENE_SCALE" "$MODEL_GEOMETRY_PROJECTION_DROPOUT" "$MODEL_FUSION_BLOCK" "$MODEL_GEO_ROPE_POINT_MAP_KEY" <<'PY'
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
cut3r_weights_path = sys.argv[3]
use_runtime_cut3r_geometry = sys.argv[4].lower() in {"1", "true", "yes", "y", "on"}
use_geometry_aware_projection = sys.argv[5].lower() in {"1", "true", "yes", "y", "on"}
spatial_encoder_type = sys.argv[6]
spatial_tower = sys.argv[7]
geometry_position_mode = sys.argv[8]
num_geometry_projection_layers = int(sys.argv[9])
geometry_projection_num_heads = int(sys.argv[10])
use_auxiliary_geometry_head = sys.argv[11].lower() in {"1", "true", "yes", "y", "on"}
use_auxiliary_geometry_loss = sys.argv[12].lower() in {"1", "true", "yes", "y", "on"}
aux_geometry_targets = sys.argv[13]
lambda_geo = float(sys.argv[14])
geometry_loss_type = sys.argv[15]
detach_geometry_targets = sys.argv[16].lower() in {"1", "true", "yes", "y", "on"}
geometry_gate_init = float(sys.argv[17])
use_geometry_confidence_mask = sys.argv[18].lower() in {"1", "true", "yes", "y", "on"}
allow_missing_geometry_targets = sys.argv[19].lower() in {"1", "true", "yes", "y", "on"}
geometry_position_max_abs = float(sys.argv[20])
geometry_fixed_scene_scale = float(sys.argv[21])
geometry_projection_dropout = float(sys.argv[22])
fusion_block = sys.argv[23]
geo_rope_point_map_key = normalize_point_map_key(sys.argv[24])

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

training_point_map_key = (
    cfg.get("geo_rope_training_point_map_key")
    or cfg.get("geometry_training_point_map_key")
    or cfg.get("geo_rope_point_map_key")
    or cfg.get("geometry_point_map_key")
)
training_point_map_key = normalize_point_map_key(training_point_map_key) if training_point_map_key else None
if training_point_map_key is None and not use_runtime_cut3r_geometry:
    training_point_map_key = "point_maps_ref"
if training_point_map_key and training_point_map_key != geo_rope_point_map_key:
    raise SystemExit(
        "GeoRoPE point-map coordinate mismatch: checkpoint training used "
        f"{training_point_map_key}, eval requested {geo_rope_point_map_key}."
    )

cfg["mm_vision_tower"] = siglip_local
if "vision_tower" in cfg:
    cfg["vision_tower"] = siglip_local
cfg["use_geometry_aware_projection"] = use_geometry_aware_projection
cfg["spatial_encoder_type"] = spatial_encoder_type
cfg["geometry_position_mode"] = geometry_position_mode
cfg["num_geometry_projection_layers"] = num_geometry_projection_layers
cfg["geometry_projection_num_heads"] = geometry_projection_num_heads
cfg["use_auxiliary_geometry_head"] = use_auxiliary_geometry_head
cfg["use_auxiliary_geometry_loss"] = use_auxiliary_geometry_loss
cfg["aux_geometry_targets"] = aux_geometry_targets
cfg["lambda_geo"] = lambda_geo
cfg["geometry_loss_type"] = geometry_loss_type
cfg["detach_geometry_targets"] = detach_geometry_targets
cfg["geometry_gate_init"] = geometry_gate_init
cfg["use_geometry_confidence_mask"] = use_geometry_confidence_mask
cfg["allow_missing_geometry_targets"] = allow_missing_geometry_targets
cfg["geometry_position_max_abs"] = geometry_position_max_abs
cfg["geometry_fixed_scene_scale"] = geometry_fixed_scene_scale
cfg["geometry_projection_dropout"] = geometry_projection_dropout
cfg["geo_rope_training_point_map_key"] = training_point_map_key or geo_rope_point_map_key
cfg["geo_rope_point_map_key"] = geo_rope_point_map_key
cfg["geometry_point_map_key"] = geo_rope_point_map_key
cfg["cut3r_weights_path"] = cut3r_weights_path
cfg["weights_path"] = cut3r_weights_path

if use_runtime_cut3r_geometry:
    cfg["spatial_tower"] = spatial_tower
else:
    cfg.pop("spatial_tower", None)

if fusion_block:
    cfg["fusion_block"] = fusion_block
else:
    cfg.pop("fusion_block", None)
    cfg.pop("geo_rope_fusion_mode", None)
    cfg.pop("geo_rope_fusion_max_depth", None)
    cfg.pop("geo_rope_fusion_group_split", None)
    cfg.pop("geo_rope_fusion_log_stats", None)

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
MODEL_ARGS+=",conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_FRAMES_NUM"
if [[ "$USE_RUNTIME_CUT3R_GEOMETRY" == "True" ]]; then
  MODEL_ARGS+=",spatial_tower=$MODEL_SPATIAL_TOWER"
else
  MODEL_ARGS+=",spatial_features_root=$SPATIAL_FEATURES_ROOT,spatial_features_subdir=$SPATIAL_FEATURES_SUBDIR"
fi
MODEL_ARGS+=",geo_rope_point_map_key=$MODEL_GEO_ROPE_POINT_MAP_KEY"

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
