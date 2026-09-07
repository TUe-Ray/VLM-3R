#!/usr/bin/env bash
# Reload a trained controlled-fusion checkpoint and evaluate one real VSI-Bench sample.
# Array indices: 0=B, 1=E.
#SBATCH --job-name=SMOKE_controlled_reload
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=110G
#SBATCH --array=0-1
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_reload/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_reload/%x_%A_%a.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
REFERENCE_REPO="/home/geusdd/VLM-3R"
# shellcheck source=/dev/null
source "$REFERENCE_REPO/snellius/paths.env"
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR

ARCH_IDS=(B E)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This smoke must run as a Slurm array task.}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
    echo "Unsupported array index: $ARRAY_INDEX" >&2
    exit 2
fi
ARCH_ID="${ARCH_IDS[$ARRAY_INDEX]}"

: "${B_CHECKPOINT:?Set B_CHECKPOINT to the completed B 1-step smoke checkpoint.}"
E_CHECKPOINT="${E_CHECKPOINT:-$TRAIN_SAVE_ROOT/SMOKE_controlled_E_26453164_3}"
case "$ARCH_ID" in
    B) CHECKPOINT="$B_CHECKPOINT" ;;
    E) CHECKPOINT="$E_CHECKPOINT" ;;
esac

for required_file in config.json adapter_config.json non_lora_trainables.bin trainer_state.json; do
    [[ -f "$CHECKPOINT/$required_file" ]] || {
        echo "Missing checkpoint artifact: $CHECKPOINT/$required_file" >&2
        exit 3
    }
done
if [[ ! -f "$CHECKPOINT/adapter_model.bin" && ! -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
    echo "Missing LoRA adapter under $CHECKPOINT" >&2
    exit 3
fi

VLM3R_PYTHON="/home/geusdd/.conda/envs/vlm3r-snellius/bin/python"
"$VLM3R_PYTHON" - "$CHECKPOINT" "$ARCH_ID" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint = Path(sys.argv[1])
arch = sys.argv[2]
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
state = torch.load(checkpoint / "non_lora_trainables.bin", map_location="cpu")

if arch == "B":
    expected_config = {
        "fusion_block": "pre_projector_add",
        "pre_projector_add_source_layer": 12,
        "tune_fusion_block": True,
        "tune_mm_mlp_adapter": True,
    }
    required_fragments = (
        "fusion_block.spatial_proj_out.weight",
        "mm_projector.0.weight",
        "mm_projector.2.weight",
    )
else:
    expected_config = {
        "use_cut3r_spatialstack": True,
        "cut3r_spatialstack_layers": "12,12,12",
        "cut3r_spatialstack_llm_layers": "0,1,2",
        "cut3r_spatialstack_fusion_type": "add",
        "cut3r_spatialstack_projector_binding": "site_specific",
        "tune_mm_mlp_adapter": False,
    }
    required_fragments = tuple(
        f"cut3r_spatialstack_merger.branches.{site}.proj_out.weight"
        for site in range(3)
    )

for key, expected in expected_config.items():
    actual = config.get(key)
    if actual != expected:
        raise SystemExit(f"{arch} config mismatch for {key}: expected={expected!r}, actual={actual!r}")

matched = {}
for fragment in required_fragments:
    keys = [key for key in state if key.endswith(fragment)]
    if len(keys) != 1:
        raise SystemExit(f"expected exactly one state key ending {fragment!r}, got {keys}")
    tensor = state[keys[0]]
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"non-finite checkpoint tensor: {keys[0]}")
    matched[fragment] = keys[0]

print(json.dumps({
    "arch": arch,
    "checkpoint": str(checkpoint),
    "config_verified": expected_config,
    "required_saved_weights": matched,
}, indent=2, sort_keys=True))
PY

JOB_TMP_ROOT="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/controlled_reload_${SLURM_ARRAY_TASK_ID}"
TASK_DIR="$JOB_TMP_ROOT/vsibench_task"
mkdir -p "$TASK_DIR"
cp "$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_leonardo_offline/utils.py" "$TASK_DIR/utils.py"
cp "$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_leonardo_offline/vsibench.yaml" "$TASK_DIR/vsibench.yaml"
sed -i "s#/leonardo_scratch/fast/EUHPC_D32_006/vsibench#$VSI_BENCH_EVAL_ROOT#g" "$TASK_DIR/vsibench.yaml"

OUTPUT_PATH="$VLM3R_LOG_ROOT/smoke_controlled_reload/${ARCH_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_PATH"

export PRETRAINED_LOCAL="$CHECKPOINT"
export MODEL_BASE_LOCAL="$LOCAL_MODEL_BASE"
export SIGLIP_LOCAL="$LOCAL_SIGLIP"
export SUBMODULE_DIR="$REPO_DIR/thinking-in-space"
export TASK_DIR
export TASK_FILE="$TASK_DIR/vsibench.yaml"
export VSI_ROOT="$VSI_BENCH_EVAL_ROOT"
export VSI_MEDIA_ROOT="$VSI_BENCH_EVAL_MEDIA_ROOT"
export FAST_ROOT="$VLM3R_ROOT"
export HF_HOME
export SPATIAL_FEATURES_ROOT="$CUT3R_DEC12_ROOT"
export SPATIAL_FEATURES_SUBDIR="12:$CUT3R_DEC12_ROOT:spatial_features"
export CUT3R_SOURCE_ROOT="$REFERENCE_REPO/third_party/CUT3R"
export PYTHONPATH="$REPO_DIR:$REFERENCE_REPO/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export CONDA_BASE="/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1"
export CONDA_ENV="/home/geusdd/.conda/envs/vlm3r-snellius"
export NUM_PROCESSES=1
export BATCH_SIZE=1
export LIMIT=1
export MAX_FRAMES_NUM=32
export CHECK_SPATIAL_SIDECARS=False
export MODEL_ATTN_IMPLEMENTATION=sdpa
export RUNTIME_ROOT="$JOB_TMP_ROOT/runtime"
export OUTPUT_PATH
export RUN_NAME="SMOKE_controlled_reload_${ARCH_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[SMOKE RELOAD] arch=$ARCH_ID checkpoint=$CHECKPOINT output=$OUTPUT_PATH"
case "$ARCH_ID" in
    B) bash "$REPO_DIR/eval_controlled_fusion_B_pre_projector_add_dec12_vsibench.sh" ;;
    E) bash "$REPO_DIR/eval_controlled_fusion_E_add_dec12x3_llm0_1_2_vsibench.sh" ;;
esac

mapfile -t output_files < <(find "$OUTPUT_PATH" -type f -print -quit)
if (( ${#output_files[@]} == 0 )); then
    echo "No VSI-Bench output artifact was written under $OUTPUT_PATH" >&2
    exit 4
fi
echo "[SMOKE RELOAD PASS] arch=$ARCH_ID checkpoint=$CHECKPOINT output=$OUTPUT_PATH"
