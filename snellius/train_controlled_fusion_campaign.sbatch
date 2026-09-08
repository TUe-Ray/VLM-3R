#!/usr/bin/env bash
# Official controlled-fusion campaign. Array indices: 0=B, 1=C, 2=D, 3=E, 4=H.
# Submit only after the dedicated 4-node E smoke has passed.
#SBATCH --job-name=controlled_fusion_BCDEH_4n16g
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --mem=0
#SBATCH --array=0-4%1
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/train/%x_%A_%a.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
SNELLIUS_REFERENCE_REPO="/home/geusdd/VLM-3R"
PRODUCTION_NODES=4
GPUS_PER_NODE=4
TARGET_GLOBAL_BATCH=128
PER_DEVICE_BATCH=1
MILESTONE_RATIOS="0.01,0.05,0.25,0.50"

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

cleanup_on_failure() {
    local status=$?
    trap - EXIT TERM INT ERR
    if [[ "$status" -ne 0 ]]; then
        echo "[CAMPAIGN ERROR] launcher failed with status $status" >&2
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scancel "$SLURM_JOB_ID" >/dev/null 2>&1 || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT TERM INT ERR

DRY_RUN="${CONTROLLED_FUSION_DRY_RUN:-False}"
SMOKE_RUN="${CONTROLLED_FUSION_SMOKE:-False}"
ARCH_IDS=(B C D E H)
if [[ -n "${CONTROLLED_FUSION_ID:-}" ]]; then
    ARCH_ID="$CONTROLLED_FUSION_ID"
else
    ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Set CONTROLLED_FUSION_ID or submit as an array task.}"
    if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
        echo "Unsupported array index: $ARRAY_INDEX" >&2
        exit 2
    fi
    ARCH_ID="${ARCH_IDS[$ARRAY_INDEX]}"
fi
case "$ARCH_ID" in
    B|C|D|E|H) ;;
    *) echo "Unsupported controlled-fusion architecture: $ARCH_ID" >&2; exit 2 ;;
esac

# Reuse the validated Snellius asset mapping, but execute the active feature
# branch rather than the reference checkout.
# shellcheck source=/dev/null
source "$SNELLIUS_REFERENCE_REPO/snellius/paths.env"
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR
SNELLIUS_CHECKPOINT_ROOT="$TRAIN_SAVE_ROOT"

export CONTROLLED_FUSION_ID="$ARCH_ID"
export CONTROLLED_FUSION_CONFIG_ONLY=True
export SPATIAL_FEATURES_ROOT="$CUT3R_DEC12_ROOT"
export SPATIAL_FEATURES_SUBDIR="12:$CUT3R_DEC12_ROOT:spatial_features"
# shellcheck source=/dev/null
source "$REPO_DIR/train_controlled_fusion_variant.sh"
unset CONTROLLED_FUSION_CONFIG_ONLY

export LOCAL_MODEL_BASE LOCAL_SIGLIP DATA_ROOT
export DATA_PATH_YAML="$REPO_DIR/snellius/vsibench_data.snellius.yaml"
export DEEPSPEED_CONFIG="$REPO_DIR/scripts/zero2.json"
export CUT3R_SOURCE_ROOT="$SNELLIUS_REFERENCE_REPO/third_party/CUT3R"
export PER_DEVICE_TRAIN_BATCH_SIZE="$PER_DEVICE_BATCH"
export TARGET_GLOBAL_BATCH_SIZE="$TARGET_GLOBAL_BATCH"
export NUM_TRAIN_EPOCHS=1
export MODEL_LORA_ENABLE=True
export MODEL_LORA_R=128
export MODEL_LORA_ALPHA=256
export MODEL_BF16=True
export MODEL_TF32=True
export MODEL_GRADIENT_CHECKPOINTING=True
export MODEL_TORCH_COMPILE=True
export MODEL_FRAMES_UPBOUND=32
export MODEL_ADD_TIME_INSTRUCTION=True
export MODEL_FORCE_SAMPLE=True
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0.0
export WARMUP_RATIO=0.03
export LR_SCHEDULER_TYPE=cosine
export LOGGING_STEPS=5
export DATALOADER_NUM_WORKERS=8
export DATALOADER_DROP_LAST=True
export DATA_GROUP_BY_MODALITY_LENGTH=True
export SPATIAL_RANK_LOSS_ENABLE=False
export CHECKPOINT_MILESTONE_RATIOS="$MILESTONE_RATIOS"
export SAVE_STRATEGY=no
export SAVE_TOTAL_LIMIT=4
export RESUME_MODE=fresh
export REQUIRE_SPATIAL_FEATURES=True
export STRICT_VIDEO_LOADING=True
export TRAIN_DATA_SHUFFLE=True
export DETERMINISTIC_DATA_ORDER=True
export NUM_GPUS_PER_NODE_OVERRIDE="$GPUS_PER_NODE"
export TRAIN_ENV_PREACTIVATED=True
export SEED=42

if is_true "$SMOKE_RUN"; then
    [[ "$ARCH_ID" == E ]] || { echo "The production-topology smoke is restricted to E." >&2; exit 2; }
    [[ "${SLURM_JOB_NAME:-}" == SMOKE* ]] || {
        echo "Smoke Slurm job names must start with SMOKE, got: ${SLURM_JOB_NAME:-unset}" >&2
        exit 2
    }
    RUN_TOKEN="${SLURM_JOB_ID:?The E smoke must run inside Slurm.}"
    export TRAIN_SAVE_ROOT="$SNELLIUS_CHECKPOINT_ROOT/controlled_fusion_smoke"
    export TRAIN_RUN_NAME="SMOKE_${SUFFIX}_4n4g_16gpu_${RUN_TOKEN}"
    export MAX_STEPS=1
    export TRAIN_DATA_MAX_SAMPLES=128
    export TRAIN_DATA_SHUFFLE=False
    export DETERMINISTIC_DATA_ORDER=True
    export REPORT_TO=none
    export LOGGING_STEPS=1
    export DATALOADER_NUM_WORKERS=2
    export MODEL_TORCH_COMPILE=False
    export MODEL_CUT3R_SPATIALSTACK_LOG_FIRST_N=1
else
    [[ -n "${SLURM_JOB_ID:-}" ]] || is_true "$DRY_RUN" || {
        echo "Official campaign training must be launched by Slurm." >&2
        exit 2
    }
    RUN_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry}}_${SLURM_ARRAY_TASK_ID:-$ARCH_ID}"
    export TRAIN_SAVE_ROOT="$SNELLIUS_CHECKPOINT_ROOT/controlled_fusion_official"
    export TRAIN_RUN_NAME="official_${SUFFIX}_4n4g_16gpu_${RUN_TOKEN}"
    export MAX_STEPS=""
    export TRAIN_DATA_MAX_SAMPLES=""
    export REPORT_TO=wandb
fi
export OUTPUT_DIR="$TRAIN_SAVE_ROOT/$TRAIN_RUN_NAME"

if is_true "$DRY_RUN"; then
    export DRY_RUN_PRINT_ARGS=True
    export NNODES="$PRODUCTION_NODES"
    export NUM_GPUS_PER_NODE="$GPUS_PER_NODE"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500
else
    [[ "${SLURM_JOB_NUM_NODES:-0}" == "$PRODUCTION_NODES" ]] || {
        echo "Expected $PRODUCTION_NODES allocated nodes, got ${SLURM_JOB_NUM_NODES:-unset}." >&2
        exit 2
    }
    [[ ! -e "$OUTPUT_DIR" ]] || {
        echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
        exit 2
    }

    module purge
    module load 2023
    module load CUDA/12.1.1

    VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
    [[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
        echo "Missing validated Snellius environment: $VLM3R_CONDA_PREFIX" >&2
        exit 2
    }
    export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
    export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$REPO_DIR:$CUT3R_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
    export TORCH_CUDA_ARCH_LIST=8.0
    export NCCL_NVLS_ENABLE=0
    export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE
    export WANDB_DIR="$VLM3R_LOG_ROOT/wandb"
    export WANDB_CACHE_DIR="$VLM3R_LOG_ROOT/wandb_cache"
    export WANDB_CONFIG_DIR="$VLM3R_LOG_ROOT/wandb_config"

    JOB_TMP_ROOT="/scratch-local/${SLURM_JOB_ID}/controlled_fusion_${SLURM_ARRAY_TASK_ID:-E}"
    export TMPDIR="$JOB_TMP_ROOT/tmp"
    export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg_cache"
    export TRITON_CACHE_DIR="$JOB_TMP_ROOT/triton_cache"
    export TORCH_EXTENSIONS_DIR="$JOB_TMP_ROOT/torch_extensions"

    srun --nodes="$PRODUCTION_NODES" --ntasks="$PRODUCTION_NODES" --ntasks-per-node=1 \
        --gpus-per-task="$GPUS_PER_NODE" --kill-on-bad-exit=1 --wait=30 --export=ALL \
        bash -c 'set -euo pipefail
            mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
            python -c '\''import socket, torch
expected = 4
actual = torch.cuda.device_count()
if not torch.cuda.is_available() or actual != expected:
    raise SystemExit(f"{socket.gethostname()}: expected {expected} GPUs, got {actual}")
print(f"[NODE PREFLIGHT] host={socket.gethostname()} gpus={actual}", flush=True)'\'''
fi

for required_dir in "$LOCAL_MODEL_BASE" "$LOCAL_SIGLIP" "$DATA_ROOT" "$CUT3R_DEC12_ROOT"; do
    [[ -d "$required_dir" ]] || { echo "Missing required directory: $required_dir" >&2; exit 2; }
done
[[ -f "$DATA_PATH_YAML" ]] || { echo "Missing formal data YAML: $DATA_PATH_YAML" >&2; exit 2; }

EXPECTED_WORLD_SIZE=$((PRODUCTION_NODES * GPUS_PER_NODE))
EXPECTED_GRAD_ACCUM=$((TARGET_GLOBAL_BATCH / (EXPECTED_WORLD_SIZE * PER_DEVICE_BATCH)))
[[ "$EXPECTED_GRAD_ACCUM" == 8 ]] || { echo "Expected gradient accumulation 8." >&2; exit 2; }
echo "[TOPOLOGY] nodes=$PRODUCTION_NODES gpus_per_node=$GPUS_PER_NODE world_size=$EXPECTED_WORLD_SIZE"
echo "[BATCH] per_device=$PER_DEVICE_BATCH target_global=$TARGET_GLOBAL_BATCH gradient_accumulation=$EXPECTED_GRAD_ACCUM"
echo "[ARCH] id=$ARCH_ID suffix=$SUFFIX output=$OUTPUT_DIR"
echo "[ARCH] layers=$MODEL_CUT3R_SPATIALSTACK_LAYERS llm_layers=$MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS fusion=${MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE:-none} projector_binding=$MODEL_CUT3R_SPATIALSTACK_PROJECTOR_BINDING tune_fusion=$MODEL_TUNE_FUSION_BLOCK tune_mm=$MODEL_TUNE_MM_MLP_ADAPTER"

python - "$DATA_PATH_YAML" "$EXPECTED_WORLD_SIZE" "$EXPECTED_GRAD_ACCUM" \
    "${TRAIN_DATA_MAX_SAMPLES:-}" "${MAX_STEPS:-}" "$NUM_TRAIN_EPOCHS" "$MILESTONE_RATIOS" <<'PY'
import json
import math
import sys
from pathlib import Path

yaml_path = Path(sys.argv[1])
world_size = int(sys.argv[2])
gradient_accumulation = int(sys.argv[3])
sample_cap = int(sys.argv[4]) if sys.argv[4] else None
max_steps_override = int(sys.argv[5]) if sys.argv[5] else None
epochs = float(sys.argv[6])
ratios = [float(value) for value in sys.argv[7].split(",")]
json_paths = []
for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith("- json_path:"):
        json_paths.append(Path(line.split(":", 1)[1].strip()))
if not json_paths:
    raise SystemExit(f"no json_path entries found in {yaml_path}")
counts = {}
for json_path in json_paths:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts[str(json_path)] = len(payload)
samples = sum(counts.values())
if sample_cap is not None:
    samples = min(samples, sample_cap)
micro_batches_per_rank = samples // world_size
updates_per_epoch = micro_batches_per_rank // gradient_accumulation
resolved_steps = max_steps_override or int(math.ceil(epochs * updates_per_epoch))
if resolved_steps <= 0:
    raise SystemExit("dataset/topology resolves to zero optimizer steps")
milestones = [
    {"ratio": ratio, "step": max(1, math.ceil(ratio * resolved_steps))}
    for ratio in ratios
]
print("[DATASET COUNTS] " + json.dumps(counts, sort_keys=True))
print(f"[TRAINING PLAN] samples={samples} optimizer_steps={resolved_steps}")
print("[MILESTONE PLAN] " + ", ".join(
    f"{100 * item['ratio']:g}%->step {item['step']}" for item in milestones
))
PY

if is_true "$SMOKE_RUN" && ! is_true "$DRY_RUN"; then
    SMOKE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    SMOKE_MASTER_PORT=$(shuf -i 30000-39999 -n 1)
    echo "[SMOKE NCCL] launching 16 ranks via $SMOKE_MASTER_ADDR:$SMOKE_MASTER_PORT"
    srun --nodes="$PRODUCTION_NODES" --ntasks="$PRODUCTION_NODES" --ntasks-per-node=1 \
        --gpus-per-task="$GPUS_PER_NODE" --kill-on-bad-exit=1 --wait=30 --export=ALL \
        torchrun --nnodes="$PRODUCTION_NODES" --nproc_per_node="$GPUS_PER_NODE" \
        --rdzv_id="${SLURM_JOB_ID}_nccl" --rdzv_backend=c10d \
        --rdzv_endpoint="$SMOKE_MASTER_ADDR:$SMOKE_MASTER_PORT" \
        "$REPO_DIR/snellius/nccl_smoke.py"
fi

echo "[GIT] commit=$(git -C "$REPO_DIR" rev-parse HEAD) branch=$(git -C "$REPO_DIR" branch --show-current)"
bash "$REPO_DIR/train_cut3r_spatialstack.sh"

if is_true "$SMOKE_RUN" && ! is_true "$DRY_RUN"; then
    python - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

output = Path(sys.argv[1])
for name in ("config.json", "adapter_config.json", "non_lora_trainables.bin", "trainer_state.json", "checkpoint_milestones.json"):
    if not (output / name).is_file():
        raise SystemExit(f"missing smoke artifact: {output / name}")
if not any((output / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
    raise SystemExit("missing final LoRA adapter")

trainer_state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
if trainer_state.get("global_step") != 1:
    raise SystemExit(f"expected global_step=1, got {trainer_state.get('global_step')!r}")
losses = [item["loss"] for item in trainer_state.get("log_history", []) if "loss" in item]
if not losses or not all(math.isfinite(float(loss)) for loss in losses):
    raise SystemExit(f"missing or non-finite training loss: {losses}")

config = json.loads((output / "config.json").read_text(encoding="utf-8"))
expected = {
    "use_cut3r_spatialstack": True,
    "cut3r_spatialstack_layers": "12,12,12",
    "cut3r_spatialstack_llm_layers": "0,1,2",
    "cut3r_spatialstack_fusion_type": "add",
    "cut3r_spatialstack_projector_binding": "site_specific",
    "tune_cut3r_spatialstack": True,
    "tune_mm_mlp_adapter": False,
}
for key, expected_value in expected.items():
    if config.get(key) != expected_value:
        raise SystemExit(f"config mismatch {key}: expected={expected_value!r}, actual={config.get(key)!r}")

state = torch.load(output / "non_lora_trainables.bin", map_location="cpu")
tensors = []
for site in range(3):
    suffix = f"cut3r_spatialstack_merger.branches.{site}.proj_out.weight"
    matches = [(key, value) for key, value in state.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"expected one tensor ending {suffix!r}, got {[key for key, _ in matches]}")
    key, value = matches[0]
    if not torch.isfinite(value).all() or torch.count_nonzero(value).item() == 0:
        raise SystemExit(f"projector did not receive a finite nonzero update: {key}")
    tensors.append((key, value))
if len({tensor.untyped_storage().data_ptr() for _, tensor in tensors}) != 3:
    raise SystemExit("E projectors do not have three distinct parameter storages")

manifest = json.loads((output / "checkpoint_milestones.json").read_text(encoding="utf-8"))
entries = manifest.get("milestones", [])
if manifest.get("max_steps") != 1 or [entry.get("ratio") for entry in entries] != [0.01, 0.05, 0.25, 0.5]:
    raise SystemExit(f"unexpected milestone manifest: {manifest}")
if any(entry.get("step") != 1 or entry.get("saved") is not True for entry in entries):
    raise SystemExit(f"milestone checkpoint was not saved correctly: {entries}")
checkpoint = output / "checkpoint-1"
for name in ("config.json", "non_lora_trainables.bin", "trainer_state.json"):
    if not (checkpoint / name).is_file():
        raise SystemExit(f"missing milestone artifact: {checkpoint / name}")
if not any((checkpoint / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
    raise SystemExit("missing milestone LoRA adapter")

print(json.dumps({
    "status": "pass",
    "global_step": 1,
    "losses": losses,
    "checkpoint": str(checkpoint),
    "site_projectors": [key for key, _ in tensors],
    "milestones": entries,
}, indent=2, sort_keys=True))
PY
    echo "[SMOKE PASS] E completed one optimizer step on 4 nodes x 4 GPUs: $OUTPUT_DIR"
fi

exit 0
