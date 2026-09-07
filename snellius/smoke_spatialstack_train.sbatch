#!/usr/bin/env bash
# A real, one-step VLM-3R SpatialStack smoke. The default is four GPUs; small
# compatibility wrappers can preserve the four-record global batch with fewer
# ranks and additional gradient-accumulation microsteps.
#SBATCH --job-name=SMOKE_vlm3r_spatialstack_train
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_train_%j.err

set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
source "$REPO_DIR/snellius/paths.env"

NUM_PROCESSES="${SMOKE_NUM_PROCESSES:-4}"
GRAD_ACCUMULATION_STEPS="${SMOKE_GRAD_ACCUMULATION_STEPS:-1}"
for value_name in NUM_PROCESSES GRAD_ACCUMULATION_STEPS; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  }
done
if (( NUM_PROCESSES * GRAD_ACCUMULATION_STEPS != 4 )); then
  echo "the four-record smoke requires NUM_PROCESSES * GRAD_ACCUMULATION_STEPS == 4" >&2
  exit 2
fi
export NUM_PROCESSES GRAD_ACCUMULATION_STEPS

module purge
module load 2023
module load CUDA/12.1.1
VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || { echo "missing environment: $VLM3R_CONDA_PREFIX" >&2; exit 2; }
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST=8.0
VLM3R_JOB_TMPDIR="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/vlm3r"
export TMPDIR="$VLM3R_JOB_TMPDIR"
mkdir -p "$VLM3R_JOB_TMPDIR"
trap 'rm -rf -- "$VLM3R_JOB_TMPDIR"' EXIT

"$REPO_DIR/snellius/validate_migration.sh"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"
expected = int(__import__("os").environ["NUM_PROCESSES"])
assert torch.cuda.device_count() == expected, (torch.cuda.device_count(), expected)
print("[SMOKE] CUDA devices:", [torch.cuda.get_device_name(i) for i in range(expected)])
PY

DATA_YAML="$REPO_DIR/snellius/vsibench_data_smoke.snellius.yaml"
[[ -f "$DATA_YAML" ]] || { echo "missing rendered data YAML: $DATA_YAML" >&2; exit 2; }

# Four records yield one global batch across four ranks.  Frames_upbound remains
# 32 so sidecar/visual-frame alignment is the formal SpatialStack configuration.
OUTPUT_DIR="$TRAIN_SAVE_ROOT/SMOKE_spatialstack_dec6_9_12_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

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
  --spatial_tower_select_feature all_tokens
  --spatial_feature_dim 768
  --use_cut3r_spatialstack True
  --tune_cut3r_spatialstack True
  --cut3r_spatialstack_layers 6,9,12
  --cut3r_spatialstack_llm_layers 0,1,2
  --cut3r_spatialstack_feature_dim 768
  --cut3r_spatialstack_feature_key cut3r_dec_layers
  --cut3r_spatialstack_zero_init True
  --cut3r_spatialstack_log_first_n 1
  --cut3r_spatialstack_projector_type token_mlp
  --cut3r_spatialstack_merge_size 2
  --cut3r_spatialstack_projector_hidden_dim 4096
  --cut3r_spatialstack_fusion_type add
  --tune_spatial_tower False
  --tune_fusion_block False
  --tune_mm_mlp_adapter False
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
  --spatial_features_root "$CUT3R_DEC6_ROOT"
  --spatial_features_subdir "6:$CUT3R_DEC6_ROOT:spatial_features_dec_6;9:$CUT3R_DEC9_ROOT:spatial_features_dec_9;12:$CUT3R_DEC12_ROOT:spatial_features"
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
  --run_name "SMOKE_spatialstack_dec6_9_12_${SLURM_JOB_ID}"
  --output_dir "$OUTPUT_DIR"
  --seed 42
  --data_seed 42
)

printf '[SMOKE TRAIN CMD] %q ' "${cmd[@]}"
printf '\n'
srun --kill-on-bad-exit=1 --wait=30 --export=ALL "${cmd[@]}"

for required in config.json adapter_config.json non_lora_trainables.bin; do
  [[ -f "$OUTPUT_DIR/$required" ]] || { echo "missing smoke training artifact: $OUTPUT_DIR/$required" >&2; exit 3; }
done
if [[ ! -f "$OUTPUT_DIR/adapter_model.safetensors" && ! -f "$OUTPUT_DIR/adapter_model.bin" ]]; then
  echo "missing smoke LoRA adapter under $OUTPUT_DIR" >&2
  exit 3
fi

# A directory full of files is not enough: prove that all three trained
# SpatialStack branches were serialized, are finite, and moved away from their
# zero-initialized output weights during the optimizer step.
python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import torch

output = Path(sys.argv[1])
with (output / "config.json").open("r", encoding="utf-8") as handle:
    config = json.load(handle)

expected_config = {
    "use_cut3r_spatialstack": True,
    "cut3r_spatialstack_layers": "6,9,12",
    "cut3r_spatialstack_llm_layers": "0,1,2",
    "cut3r_spatialstack_feature_dim": 768,
    "cut3r_spatialstack_feature_key": "cut3r_dec_layers",
    "cut3r_spatialstack_projector_type": "token_mlp",
    "cut3r_spatialstack_fusion_type": "add",
}
for key, expected in expected_config.items():
    actual = config.get(key)
    if actual != expected:
        raise SystemExit(
            f"checkpoint config mismatch for {key}: expected {expected!r}, got {actual!r}"
        )

state = torch.load(output / "non_lora_trainables.bin", map_location="cpu")
branch_state = {
    key: value
    for key, value in state.items()
    if "cut3r_spatialstack_merger.branches." in key
}
if len(branch_state) != 18:
    raise SystemExit(f"expected 18 SpatialStack branch tensors, found {len(branch_state)}")
for layer in ("6", "9", "12"):
    prefix = f"cut3r_spatialstack_merger.branches.{layer}."
    tensors = {key: value for key, value in branch_state.items() if prefix in key}
    if len(tensors) != 6:
        raise SystemExit(f"expected 6 tensors for SpatialStack layer {layer}, found {len(tensors)}")
    for key, value in tensors.items():
        if not torch.isfinite(value).all():
            raise SystemExit(f"non-finite SpatialStack checkpoint tensor: {key}")
    proj_out = next(value for key, value in tensors.items() if key.endswith("proj_out.weight"))
    if torch.count_nonzero(proj_out).item() == 0:
        raise SystemExit(f"SpatialStack layer {layer} did not update from zero initialization")

print("[SMOKE TRAIN] checkpoint config and 18 trained SpatialStack tensors validated")
PY

CHECKPOINT_POINTER="$VLM3R_ROOT/migration/latest_spatialstack_smoke_checkpoint.txt"
POINTER_TMP="${CHECKPOINT_POINTER}.${SLURM_JOB_ID}.tmp"
printf '%s\n' "$OUTPUT_DIR" > "$POINTER_TMP"
mv -f -- "$POINTER_TMP" "$CHECKPOINT_POINTER"
echo "SpatialStack training smoke passed: $OUTPUT_DIR"
