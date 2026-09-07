#!/usr/bin/env bash
# One real optimizer step for controlled fusion variants B/C/D/E/H on Snellius.
# Array indices: 0=B, 1=C, 2=D, 3=E, 4=H.
#SBATCH --job-name=SMOKE_controlled_fusion
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=220G
#SBATCH --array=0-4%2
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_fusion/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/smoke_controlled_fusion/%x_%A_%a.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
SNELLIUS_REFERENCE_REPO="/home/geusdd/VLM-3R"
# shellcheck source=/dev/null
source "$SNELLIUS_REFERENCE_REPO/snellius/paths.env"
# paths.env belongs to the migrated reference checkout; execute the feature
# branch under test while reusing its verified Snellius asset mapping.
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR

ARCH_IDS=(B C D E H)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?This smoke must run as a Slurm array task.}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
    echo "Unsupported array index: $ARRAY_INDEX" >&2
    exit 2
fi
ARCH_ID="${ARCH_IDS[$ARRAY_INDEX]}"
RUN_STEM="SMOKE_controlled_${ARCH_ID}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="$TRAIN_SAVE_ROOT/$RUN_STEM"
DATA_YAML="$SNELLIUS_REFERENCE_REPO/snellius/vsibench_data_smoke.snellius.yaml"
NUM_PROCESSES="${SMOKE_NUM_PROCESSES:-2}"
GRAD_ACCUMULATION_STEPS="${SMOKE_GRAD_ACCUMULATION_STEPS:-2}"
for value_name in NUM_PROCESSES GRAD_ACCUMULATION_STEPS; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 2
    }
done
if (( NUM_PROCESSES * GRAD_ACCUMULATION_STEPS != 4 )); then
    echo "The four-record smoke requires NUM_PROCESSES * GRAD_ACCUMULATION_STEPS == 4" >&2
    exit 2
fi
export NUM_PROCESSES GRAD_ACCUMULATION_STEPS

module purge
module load 2023
module load CUDA/12.1.1

VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || {
    echo "Missing environment: $VLM3R_CONDA_PREFIX" >&2
    exit 2
}
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR:$SNELLIUS_REFERENCE_REPO/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export CUT3R_SOURCE_ROOT="$SNELLIUS_REFERENCE_REPO/third_party/CUT3R"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST=8.0

JOB_TMP_ROOT="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/controlled_fusion_${SLURM_ARRAY_TASK_ID}"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg_cache"
export TRITON_CACHE_DIR="$JOB_TMP_ROOT/triton_cache"
export TORCH_EXTENSIONS_DIR="$JOB_TMP_ROOT/torch_extensions"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$SNELLIUS_REFERENCE_REPO/snellius/validate_migration.sh"

for required_dir in \
    "$LOCAL_MODEL_BASE" \
    "$LOCAL_SIGLIP" \
    "$DATA_ROOT" \
    "$CUT3R_DEC12_ROOT" \
    "$TRAIN_SAVE_ROOT"; do
    [[ -d "$required_dir" ]] || { echo "Missing required directory: $required_dir" >&2; exit 2; }
done
[[ -f "$DATA_YAML" ]] || { echo "Missing smoke data YAML: $DATA_YAML" >&2; exit 2; }

python - "$REPO_DIR" <<'PY'
import pathlib
import sys

import llava
import torch
from src.dust3r.model import ARCroco3DStereo

repo = pathlib.Path(sys.argv[1]).resolve()
loaded = pathlib.Path(llava.__file__).resolve()
if repo not in loaded.parents:
    raise SystemExit(f"llava imported from {loaded}, expected current checkout {repo}")
expected = int(__import__("os").environ["NUM_PROCESSES"])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit(
        f"expected {expected} CUDA device(s), available={torch.cuda.is_available()}, "
        f"count={torch.cuda.device_count()}"
    )
print(f"[SMOKE] code={loaded}")
print(f"[SMOKE] cut3r_model={ARCroco3DStereo.__module__}")
print(f"[SMOKE] cuda={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

USE_SPATIALSTACK=True
TUNE_SPATIALSTACK=True
SPATIAL_SELECT_FEATURE=all_tokens
CUT3R_LAYERS=12
LLM_LAYERS=0
FUSION_TYPE=add
PROJECTOR_BINDING=source_specific
TUNE_FUSION_BLOCK=False
TUNE_MM_MLP_ADAPTER=False
FUSION_BLOCK_ARGS=()

case "$ARCH_ID" in
    B)
        USE_SPATIALSTACK=False
        TUNE_SPATIALSTACK=False
        SPATIAL_SELECT_FEATURE=patch_tokens
        TUNE_FUSION_BLOCK=True
        TUNE_MM_MLP_ADAPTER=True
        FUSION_BLOCK_ARGS=(
            --fusion_block pre_projector_add
            --pre_projector_add_source_layer 12
            --pre_projector_add_zero_init True
        )
        ;;
    C)
        FUSION_TYPE=cross_attn
        ;;
    D)
        FUSION_TYPE=add
        ;;
    E)
        CUT3R_LAYERS=12,12,12
        LLM_LAYERS=0,1,2
        FUSION_TYPE=add
        PROJECTOR_BINDING=site_specific
        ;;
    H)
        CUT3R_LAYERS=12,12,12
        LLM_LAYERS=0,1,2
        FUSION_TYPE=cross_attn
        ;;
esac

mkdir -p "$OUTPUT_DIR"
echo "[SMOKE] arch=$ARCH_ID output=$OUTPUT_DIR"
echo "[SMOKE] git=$(git -C "$REPO_DIR" rev-parse HEAD) branch=$(git -C "$REPO_DIR" branch --show-current)"

cmd=(
    torchrun --standalone --nproc_per_node="$NUM_PROCESSES"
    "$REPO_DIR/llava/train/train_mem.py"
    --model_name_or_path "$LOCAL_MODEL_BASE"
    --version qwen_1_5
    --vision_tower "$LOCAL_SIGLIP"
    --lora_enable True
    --lora_r 128
    --lora_alpha 256
    --spatial_tower cut3r
    --spatial_tower_preextracted_only True
    --spatial_tower_select_feature "$SPATIAL_SELECT_FEATURE"
    --spatial_feature_dim 768
    --use_cut3r_spatialstack "$USE_SPATIALSTACK"
    --tune_cut3r_spatialstack "$TUNE_SPATIALSTACK"
    --cut3r_spatialstack_layers "$CUT3R_LAYERS"
    --cut3r_spatialstack_llm_layers "$LLM_LAYERS"
    --cut3r_spatialstack_feature_dim 768
    --cut3r_spatialstack_feature_key cut3r_dec_layers
    --cut3r_spatialstack_zero_init True
    --cut3r_spatialstack_log_first_n 1
    --cut3r_spatialstack_projector_type token_mlp
    --cut3r_spatialstack_projector_binding "$PROJECTOR_BINDING"
    --cut3r_spatialstack_merge_size 2
    --cut3r_spatialstack_projector_hidden_dim 4096
    --cut3r_spatialstack_fusion_type "$FUSION_TYPE"
    --cut3r_spatialstack_cross_attn_dropout 0.0
    --cut3r_spatialstack_cross_attn_zero_init True
    --cut3r_spatialstack_cross_attn_same_frame_only True
    --cut3r_spatialstack_cross_attn_impl torch_mha
    --cut3r_spatialstack_cross_attn_patch_align resize
    --cut3r_spatialstack_cross_attn_use_camera_tokens False
    --cut3r_spatialstack_require_camera_tokens False
    --cut3r_spatialstack_cross_attn_use_mlp True
    --cut3r_spatialstack_cross_attn_norm_type qwen_rmsnorm
    --cut3r_spatialstack_cross_attn_pos_embed sincos2d
    --cut3r_spatialstack_cross_attn_gamma_attn_init 0.05
    --cut3r_spatialstack_cross_attn_gamma_mlp_init 0.05
    --cut3r_spatialstack_cross_attn_gamma_learnable True
    --use_cut3r_camera_tokens False
    --tune_spatial_tower False
    --tune_fusion_block "$TUNE_FUSION_BLOCK"
    --tune_mm_mlp_adapter "$TUNE_MM_MLP_ADAPTER"
    --llm_visual_3d_rope_enable False
    --use_geometry_aware_projection False
    --use_auxiliary_geometry_head False
    --use_auxiliary_geometry_loss False
    --use_bev_supervision False
    --use_depth_supervision False
    --use_pointmap_supervision False
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio anyres_max_9
    --mm_patch_merge_type spatial_unpad
    --mm_spatial_pool_stride 2
    --mm_newline_position grid
    --add_time_instruction True
    --force_sample True
    --frames_upbound 32
    --data_path "$DATA_YAML"
    --image_folder "$DATA_ROOT"
    --video_folder "$DATA_ROOT"
    --spatial_features_root "$CUT3R_DEC12_ROOT"
    --spatial_features_subdir "12:$CUT3R_DEC12_ROOT:spatial_features"
    --require_spatial_features True
    --strict_video_loading True
    --train_data_max_samples 4
    --train_data_shuffle False
    --deterministic_data_order True
    --group_by_modality_length True
    --deepspeed "$REPO_DIR/scripts/zero2.json"
    --bf16 True
    --tf32 True
    --attn_implementation flash_attention_2
    --model_max_length 32768
    --gradient_checkpointing True
    --lazy_preprocess True
    --torch_compile False
    --num_train_epochs 1
    --max_steps 1
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 1
    --save_total_limit 1
    --learning_rate 2e-5
    --weight_decay 0.0
    --warmup_ratio 0.0
    --lr_scheduler_type cosine
    --logging_steps 1
    --dataloader_num_workers 2
    --dataloader_drop_last True
    --report_to none
    --run_name "$RUN_STEM"
    --output_dir "$OUTPUT_DIR"
    --seed 42
    --data_seed 42
    "${FUSION_BLOCK_ARGS[@]}"
)

printf '[SMOKE TRAIN CMD] %q ' "${cmd[@]}"
printf '\n'
srun --kill-on-bad-exit=1 --wait=30 --export=ALL "${cmd[@]}"

for required_file in config.json adapter_config.json non_lora_trainables.bin trainer_state.json; do
    [[ -f "$OUTPUT_DIR/$required_file" ]] || {
        echo "Missing smoke training artifact: $OUTPUT_DIR/$required_file" >&2
        exit 3
    }
done
if [[ ! -f "$OUTPUT_DIR/adapter_model.safetensors" && ! -f "$OUTPUT_DIR/adapter_model.bin" ]]; then
    echo "Missing smoke LoRA adapter under $OUTPUT_DIR" >&2
    exit 3
fi

python - "$OUTPUT_DIR" "$ARCH_ID" <<'PY'
import json
import sys
from pathlib import Path

import torch

output = Path(sys.argv[1])
arch = sys.argv[2]
with (output / "trainer_state.json").open("r", encoding="utf-8") as handle:
    trainer_state = json.load(handle)
if trainer_state.get("global_step") != 1:
    raise SystemExit(f"expected global_step=1, got {trainer_state.get('global_step')!r}")

with (output / "config.json").open("r", encoding="utf-8") as handle:
    config = json.load(handle)
expected = {
    "B": {
        "fusion_block": "pre_projector_add",
        "pre_projector_add_source_layer": 12,
        "tune_fusion_block": True,
        "tune_mm_mlp_adapter": True,
    },
    "C": {
        "cut3r_spatialstack_layers": "12",
        "cut3r_spatialstack_llm_layers": "0",
        "cut3r_spatialstack_fusion_type": "cross_attn",
        "tune_mm_mlp_adapter": False,
    },
    "D": {
        "cut3r_spatialstack_layers": "12",
        "cut3r_spatialstack_llm_layers": "0",
        "cut3r_spatialstack_fusion_type": "add",
        "tune_mm_mlp_adapter": False,
    },
    "E": {
        "cut3r_spatialstack_layers": "12,12,12",
        "cut3r_spatialstack_llm_layers": "0,1,2",
        "cut3r_spatialstack_fusion_type": "add",
        "cut3r_spatialstack_projector_binding": "site_specific",
        "tune_mm_mlp_adapter": False,
    },
    "H": {
        "cut3r_spatialstack_layers": "12,12,12",
        "cut3r_spatialstack_llm_layers": "0,1,2",
        "cut3r_spatialstack_fusion_type": "cross_attn",
        "tune_mm_mlp_adapter": False,
    },
}[arch]
for key, expected_value in expected.items():
    actual = config.get(key)
    if actual != expected_value:
        raise SystemExit(
            f"{arch} config mismatch for {key}: expected {expected_value!r}, got {actual!r}"
        )
if arch == "B" and config.get("use_cut3r_spatialstack") not in (None, False):
    raise SystemExit(
        "B config unexpectedly enables SpatialStack: "
        f"{config.get('use_cut3r_spatialstack')!r}"
    )

state_path = output / "non_lora_trainables.bin"
if not state_path.is_file():
    raise SystemExit(f"missing non-LoRA trained state: {state_path}")
state = torch.load(state_path, map_location="cpu")
if arch == "B":
    suffixes = (
        "fusion_block.spatial_proj_out.weight",
        "mm_projector.0.weight",
        "mm_projector.2.weight",
    )
elif arch in {"C", "H"}:
    sites = ("0",) if arch == "C" else ("0", "1", "2")
    suffixes = tuple(
        f"cut3r_spatialstack_merger.cross_attn_blocks.{site}.out_proj.weight"
        for site in sites
    )
else:
    branches = ("12",) if arch == "D" else ("0", "1", "2")
    suffixes = tuple(
        f"cut3r_spatialstack_merger.branches.{branch}.proj_out.weight"
        for branch in branches
    )

matched = []
for suffix in suffixes:
    candidates = [(key, value) for key, value in state.items() if key.endswith(suffix)]
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one trained tensor ending {suffix!r}, got {[k for k, _ in candidates]}")
    key, value = candidates[0]
    if not torch.isfinite(value).all():
        raise SystemExit(f"non-finite trained tensor: {key}")
    if torch.count_nonzero(value).item() == 0:
        raise SystemExit(f"zero-init output weight did not update after optimizer step: {key}")
    matched.append((key, value))

if arch == "E":
    storage_ptrs = {value.untyped_storage().data_ptr() for _, value in matched}
    if len(storage_ptrs) != 3:
        raise SystemExit("E site-specific projector tensors do not have three distinct storages")

print(f"[SMOKE PASS] arch={arch} global_step=1 trained_tensors={[key for key, _ in matched]}")
PY

echo "[SMOKE PASS] Controlled fusion $ARCH_ID completed one optimizer step: $OUTPUT_DIR"
