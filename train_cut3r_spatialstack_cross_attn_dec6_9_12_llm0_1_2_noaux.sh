#!/bin/bash
#SBATCH --job-name=cut3r_spatialstack_cross_attn
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=16:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

set -euo pipefail

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

append_arg_map() {
    local -n source_map="$1"
    local key
    for key in "${!source_map[@]}"; do
        if [[ -z "${source_map[$key]}" ]]; then
            continue
        fi
        TORCHRUN_ARGS+=("--${key}" "${source_map[$key]}")
    done
}

assert_arg_value() {
    local key="$1"
    local expected="$2"
    local actual="${MODEL_ARGS[$key]:-}"
    if [[ "${actual,,}" != "${expected,,}" ]]; then
        echo "[SPATIALSTACK][ERROR] Expected MODEL_ARGS[$key]=$expected, got '$actual'."
        exit 1
    fi
}

assert_no_torchrun_arg() {
    local forbidden="$1"
    local arg
    for arg in "${TORCHRUN_ARGS[@]}"; do
        if [[ "$arg" == "$forbidden" ]]; then
            echo "[SPATIALSTACK][ERROR] Forbidden argument emitted: $forbidden"
            exit 1
        fi
    done
}

# ============================================================
# General
# ============================================================
NOTE="${NOTE:-CUT3R SpatialStack plain cross-attn ablation: dec[6,9,12] -> LLM[0,1,2], no auxiliary geometry losses.}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlm3r}"
DRY_RUN_PRINT_ARGS="${DRY_RUN_PRINT_ARGS:-False}"
SEED="${SEED:-42}"

# ============================================================
# Paths
# ============================================================
MODEL_ROOT="${MODEL_ROOT:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R}"
LOCAL_MODEL_BASE="${LOCAL_MODEL_BASE:-$MODEL_ROOT/LLaVA-NeXT-Video-7B-Qwen2}"
LOCAL_SIGLIP="${LOCAL_SIGLIP:-$MODEL_ROOT/siglip-so400m-patch14-384}"

WORK_DATA_ROOT="${WORK_DATA_ROOT:-/leonardo_work/EUHPC_D32_006/train_data/vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_ROOT="${DATA_ROOT:-$FAST_DATA_ROOT}"
CUT3R_SPATIALSTACK_FEATURE_ROOT="${CUT3R_SPATIALSTACK_FEATURE_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$CUT3R_SPATIALSTACK_FEATURE_ROOT}"
SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-6:spatial_features_dec_6,9:spatial_features_dec_9,12:$FAST_DATA_ROOT:spatial_features}"

TRAIN_SAVE_ROOT="${TRAIN_SAVE_ROOT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R}"
SUFFIX="${SUFFIX:-cut3r_spatialstack_cross_attn_dec6_9_12_llm0_1_2_noaux}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${SLURM_JOB_NAME:-$SUFFIX}_${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_SAVE_ROOT/$TRAIN_RUN_NAME}"
RESUME_MODE="${RESUME_MODE:-fresh}"
RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-none}"

WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE:-$OUTPUT_DIR/.wandb_run_id}"
WANDB_RUN_ID_SOURCE="env"
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    if [[ "$RESUME_MODE" == "continue" && -f "$WANDB_RUN_ID_FILE" ]]; then
        WANDB_RUN_ID="$(<"$WANDB_RUN_ID_FILE")"
        WANDB_RUN_ID_SOURCE="$WANDB_RUN_ID_FILE"
    elif [[ "$RESUME_MODE" == "continue" && "$RESUME_CHECKPOINT_PATH" != "none" && "$RESUME_CHECKPOINT_PATH" != "auto" ]]; then
        RESUME_CHECKPOINT_PARENT="${RESUME_CHECKPOINT_PATH%/*}"
        RESUME_WANDB_RUN_ID_FILE="$RESUME_CHECKPOINT_PARENT/.wandb_run_id"
        if [[ -f "$RESUME_WANDB_RUN_ID_FILE" ]]; then
            WANDB_RUN_ID="$(<"$RESUME_WANDB_RUN_ID_FILE")"
            WANDB_RUN_ID_SOURCE="$RESUME_WANDB_RUN_ID_FILE"
        fi
    fi
fi
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_TIME="${WANDB_RUN_TIME:-$(date +%Y%m%d_%H%M%S)}"
    WANDB_RUN_ID="${WANDB_RUN_TIME}_${TRAIN_RUN_NAME}"
    WANDB_RUN_ID_SOURCE="generated"
fi
WANDB_RUN_ID="${WANDB_RUN_ID//[^A-Za-z0-9_.-]/_}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_NAME="${WANDB_NAME:-$TRAIN_RUN_NAME}"
WANDB_DIR="${WANDB_DIR:-${WORK:-/tmp}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WORK:-/tmp}/wandb_cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WORK:-/tmp}/wandb_config}"

HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D32_006/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# ============================================================
# Model: explicit SpatialStack experiment
# ============================================================
MODEL_LORA_ENABLE="${MODEL_LORA_ENABLE:-True}"
MODEL_LORA_R="${MODEL_LORA_R:-128}"
MODEL_LORA_ALPHA="${MODEL_LORA_ALPHA:-256}"
MODEL_SPATIAL_TOWER="${MODEL_SPATIAL_TOWER:-cut3r}"
MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY="${MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY:-True}"
MODEL_SPATIAL_TOWER_SELECT_FEATURE="${MODEL_SPATIAL_TOWER_SELECT_FEATURE:-all_tokens}"
MODEL_SPATIAL_FEATURE_DIM="${MODEL_SPATIAL_FEATURE_DIM:-768}"

MODEL_USE_CUT3R_SPATIALSTACK="True"
MODEL_TUNE_CUT3R_SPATIALSTACK="True"
MODEL_CUT3R_SPATIALSTACK_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LAYERS:-6,9,12}"
MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS="${MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS:-0,1,2}"
MODEL_CUT3R_SPATIALSTACK_FEATURE_DIM="${MODEL_CUT3R_SPATIALSTACK_FEATURE_DIM:-$MODEL_SPATIAL_FEATURE_DIM}"
MODEL_CUT3R_SPATIALSTACK_FEATURE_KEY="${MODEL_CUT3R_SPATIALSTACK_FEATURE_KEY:-cut3r_dec_layers}"
MODEL_CUT3R_SPATIALSTACK_ZERO_INIT="${MODEL_CUT3R_SPATIALSTACK_ZERO_INIT:-True}"
MODEL_CUT3R_SPATIALSTACK_LOG_FIRST_N="${MODEL_CUT3R_SPATIALSTACK_LOG_FIRST_N:-0}"
MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE="${MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE:-cross_attn}"
MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS="${MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS:-}"
MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT="${MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT:-0.0}"
MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT="${MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT:-True}"
MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY="${MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY:-True}"

# Intentionally no MODEL_FUSION_BLOCK and no --fusion_block argument.
MODEL_TUNE_FUSION_BLOCK="False"
MODEL_USE_GEOMETRY_AWARE_PROJECTION="False"
MODEL_TUNE_GEOMETRY_AWARE_PROJECTION="False"
MODEL_USE_AUXILIARY_GEOMETRY_HEAD="False"
MODEL_USE_AUXILIARY_GEOMETRY_LOSS="False"
MODEL_USE_BEV_SUPERVISION="False"
MODEL_USE_DEPTH_SUPERVISION="False"
MODEL_LLM_VISUAL_3D_ROPE_ENABLE="False"

MODEL_TUNE_SPATIAL_TOWER="${MODEL_TUNE_SPATIAL_TOWER:-False}"
MODEL_TUNE_MM_MLP_ADAPTER="${MODEL_TUNE_MM_MLP_ADAPTER:-False}"
MODEL_VERSION="${MODEL_VERSION:-qwen_1_5}"
MODEL_MM_PROJECTOR_TYPE="${MODEL_MM_PROJECTOR_TYPE:-mlp2x_gelu}"
MODEL_MM_VISION_SELECT_LAYER="${MODEL_MM_VISION_SELECT_LAYER:--2}"
MODEL_MM_USE_IM_START_END="${MODEL_MM_USE_IM_START_END:-False}"
MODEL_MM_USE_IM_PATCH_TOKEN="${MODEL_MM_USE_IM_PATCH_TOKEN:-False}"
MODEL_IMAGE_ASPECT_RATIO="${MODEL_IMAGE_ASPECT_RATIO:-anyres_max_9}"
MODEL_IMAGE_GRID_PINPOINTS="${MODEL_IMAGE_GRID_PINPOINTS:-(1x1),...,(6x6)}"
MODEL_MM_PATCH_MERGE_TYPE="${MODEL_MM_PATCH_MERGE_TYPE:-spatial_unpad}"
MODEL_BF16="${MODEL_BF16:-True}"
MODEL_TF32="${MODEL_TF32:-True}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
MODEL_GRADIENT_CHECKPOINTING="${MODEL_GRADIENT_CHECKPOINTING:-True}"
MODEL_LAZY_PREPROCESS="${MODEL_LAZY_PREPROCESS:-True}"
MODEL_TORCH_COMPILE="${MODEL_TORCH_COMPILE:-True}"
MODEL_TORCH_COMPILE_BACKEND="${MODEL_TORCH_COMPILE_BACKEND:-inductor}"
MODEL_FRAMES_UPBOUND="${MODEL_FRAMES_UPBOUND:-32}"
MODEL_MM_NEWLINE_POSITION="${MODEL_MM_NEWLINE_POSITION:-grid}"
MODEL_ADD_TIME_INSTRUCTION="${MODEL_ADD_TIME_INSTRUCTION:-True}"
MODEL_FORCE_SAMPLE="${MODEL_FORCE_SAMPLE:-True}"
MODEL_MM_SPATIAL_POOL_STRIDE="${MODEL_MM_SPATIAL_POOL_STRIDE:-2}"

# Keep geometry-related parser values explicit but disabled.
MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
MODEL_SPATIAL_ENCODER_TYPE="${MODEL_SPATIAL_ENCODER_TYPE:-cut3r}"
MODEL_GEOMETRY_POSITION_MODE="${MODEL_GEOMETRY_POSITION_MODE:-spherical}"
MODEL_NUM_GEOMETRY_PROJECTION_LAYERS="${MODEL_NUM_GEOMETRY_PROJECTION_LAYERS:-1}"
MODEL_GEOMETRY_PROJECTION_NUM_HEADS="${MODEL_GEOMETRY_PROJECTION_NUM_HEADS:-16}"
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

# ============================================================
# Data / optimization
# ============================================================
DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}"
DATA_GROUP_BY_MODALITY_LENGTH="${DATA_GROUP_BY_MODALITY_LENGTH:-True}"
ZERO_SPATIAL_FEATURES="${ZERO_SPATIAL_FEATURES:-False}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-wandb}"
DATALOADER_DROP_LAST="${DATALOADER_DROP_LAST:-True}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"

SPATIAL_RANK_LOSS_ENABLE="${SPATIAL_RANK_LOSS_ENABLE:-False}"
LAMBDA_SIM="${LAMBDA_SIM:-0.01}"
SPATIAL_RANK_MARGIN="${SPATIAL_RANK_MARGIN:-0.2}"
ANCHORS_PER_FRAME="${ANCHORS_PER_FRAME:-128}"
POSITIVE_TOP_PERCENT="${POSITIVE_TOP_PERCENT:-10}"
NEGATIVE_BOTTOM_PERCENT="${NEGATIVE_BOTTOM_PERCENT:-30}"
SPATIAL_RANK_DEBUG_CHECKS="${SPATIAL_RANK_DEBUG_CHECKS:-False}"

echo "-------- Note --------"
echo "  note: $NOTE"
mkdir -p logs/train

if is_true "$DRY_RUN_PRINT_ARGS"; then
    JOB_TIME_LIMIT="DRY_RUN"
else
    JOB_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID:-}" -h -o "%l")
fi

echo "=== SLURM Job Specifications ==="
echo "Job Name: ${SLURM_JOB_NAME:-}"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Number of Nodes: ${SLURM_JOB_NUM_NODES:-}"
echo "Node List: ${SLURM_JOB_NODELIST:-}"
echo "GPUs per Node: ${SLURM_GPUS_PER_NODE:-}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK:-}"
echo "Tasks per Node: ${SLURM_NTASKS_PER_NODE:-}"
echo "Partition: ${SLURM_JOB_PARTITION:-}"
echo "QOS: ${SLURM_JOB_QOS:-}"
echo "Memory per Node: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Output: ${SLURM_STDOUT:-}"
echo "Error: ${SLURM_STDERR:-}"
echo "Job Time Limit: $JOB_TIME_LIMIT"

cleanup_on_training_failure() {
    local status=$?
    trap - EXIT TERM INT ERR
    if [[ "$status" -ne 0 ]]; then
        echo "[ERROR] SpatialStack training script failed with status $status."
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scancel "$SLURM_JOB_ID" >/dev/null 2>&1 || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_training_failure EXIT TERM INT ERR
SRUN_FAIL_FAST_ARGS=(--kill-on-bad-exit=1 --wait=30)
if [[ -n "${SRUN_EXTRA_ARGS:-}" ]]; then
    read -r -a SRUN_EXTRA_ARGS_ARRAY <<< "$SRUN_EXTRA_ARGS"
    SRUN_FAIL_FAST_ARGS+=("${SRUN_EXTRA_ARGS_ARRAY[@]}")
fi

if is_true "$DRY_RUN_PRINT_ARGS"; then
    echo "[DRY_RUN] Skipping module, conda, GPU, and Slurm discovery."
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_RUN_ID="$WANDB_RUN_ID"
    export WANDB_RESUME="$WANDB_RESUME"
    export WANDB_NAME="$WANDB_NAME"
    export WANDB_DIR="$WANDB_DIR"
    export WANDB_CACHE_DIR="$WANDB_CACHE_DIR"
    export WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR"
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
    NNODES="${NNODES:-1}"
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
else
    module load cuda/12.1
    module load cudnn
    module load profile/deeplrn

    echo "[DEBUG] after modules:"
    OUT=$(nvidia-smi -L 2>&1) || {
        echo "[ERROR] nvidia-smi failed on $(hostname)"
        echo "$OUT"
        exit 1
    }
    if echo "$OUT" | grep -q "Driver/library version mismatch"; then
        echo "[ERROR] NVML mismatch on $(hostname)"
        echo "$OUT"
        exit 1
    fi
    echo "$OUT"

    export PATH="$WORK/miniconda3/bin:$PATH"
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    set -u

    if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
    fi

    export WANDB_MODE="offline"
    export NCCL_NVLS_ENABLE=0
    export WANDB_RUN_ID="$WANDB_RUN_ID"
    export WANDB_RESUME="$WANDB_RESUME"
    export WANDB_NAME="$WANDB_NAME"
    export WANDB_DIR="$WANDB_DIR"
    export WANDB_CACHE_DIR="$WANDB_CACHE_DIR"
    export WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR"
    mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

    export HF_HOME="$HF_HOME"
    export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
    export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

    if [[ -v SLURM_GPUS_ON_NODE && -n "$SLURM_GPUS_ON_NODE" ]]; then
        NUM_GPUS_PER_NODE="$SLURM_GPUS_ON_NODE"
    elif [[ -v SLURM_GPUS_PER_NODE && -n "$SLURM_GPUS_PER_NODE" ]]; then
        NUM_GPUS_PER_NODE="$SLURM_GPUS_PER_NODE"
    else
        NUM_GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    fi

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [[ -v SLURM_JOB_NUM_NODES && -n "$SLURM_JOB_NUM_NODES" ]]; then
        NNODES="$SLURM_JOB_NUM_NODES"
    else
        NNODES=1
    fi
    MASTER_PORT=$(shuf -i 20000-29999 -n 1)
    export OMP_NUM_THREADS=2
fi

WORLD_SIZE=$((NNODES * NUM_GPUS_PER_NODE))
echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES"
echo "[DDP] NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"

if [[ "$RESUME_CHECKPOINT_PATH" == "none" && "$RESUME_MODE" == "continue" ]]; then
    RESUME_CHECKPOINT_PATH="auto"
fi
export RESUME_CHECKPOINT_PATH

if is_true "$DRY_RUN_PRINT_ARGS"; then
    echo "[DRY_RUN] Skipping output directory creation and local model path checks."
else
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
    if [[ ! -d "$LOCAL_MODEL_BASE" ]]; then
        echo "[ERROR] Local model base not found: $LOCAL_MODEL_BASE"
        exit 1
    fi
    if [[ ! -d "$LOCAL_SIGLIP" ]]; then
        echo "[ERROR] Local SigLIP not found: $LOCAL_SIGLIP"
        exit 1
    fi
fi

denom=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( TARGET_GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "[ERROR] TARGET_GLOBAL_BATCH_SIZE($TARGET_GLOBAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_TRAIN_BATCH_SIZE($denom)"
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denom))

declare -A MODEL_ARGS=(
    [model_name_or_path]="$LOCAL_MODEL_BASE"
    [lora_enable]="$MODEL_LORA_ENABLE"
    [lora_r]="$MODEL_LORA_R"
    [lora_alpha]="$MODEL_LORA_ALPHA"
    [spatial_tower]="$MODEL_SPATIAL_TOWER"
    [spatial_tower_preextracted_only]="$MODEL_SPATIAL_TOWER_PREEXTRACTED_ONLY"
    [spatial_tower_select_feature]="$MODEL_SPATIAL_TOWER_SELECT_FEATURE"
    [spatial_feature_dim]="$MODEL_SPATIAL_FEATURE_DIM"
    [llm_visual_3d_rope_enable]="$MODEL_LLM_VISUAL_3D_ROPE_ENABLE"
    [use_cut3r_spatialstack]="$MODEL_USE_CUT3R_SPATIALSTACK"
    [tune_cut3r_spatialstack]="$MODEL_TUNE_CUT3R_SPATIALSTACK"
    [cut3r_spatialstack_layers]="$MODEL_CUT3R_SPATIALSTACK_LAYERS"
    [cut3r_spatialstack_llm_layers]="$MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS"
    [cut3r_spatialstack_feature_dim]="$MODEL_CUT3R_SPATIALSTACK_FEATURE_DIM"
    [cut3r_spatialstack_feature_key]="$MODEL_CUT3R_SPATIALSTACK_FEATURE_KEY"
    [cut3r_spatialstack_zero_init]="$MODEL_CUT3R_SPATIALSTACK_ZERO_INIT"
    [cut3r_spatialstack_log_first_n]="$MODEL_CUT3R_SPATIALSTACK_LOG_FIRST_N"
    [cut3r_spatialstack_fusion_type]="$MODEL_CUT3R_SPATIALSTACK_FUSION_TYPE"
    [cut3r_spatialstack_cross_attn_heads]="$MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_HEADS"
    [cut3r_spatialstack_cross_attn_dropout]="$MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_DROPOUT"
    [cut3r_spatialstack_cross_attn_zero_init]="$MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_ZERO_INIT"
    [cut3r_spatialstack_cross_attn_same_frame_only]="$MODEL_CUT3R_SPATIALSTACK_CROSS_ATTN_SAME_FRAME_ONLY"
    [use_geometry_aware_projection]="$MODEL_USE_GEOMETRY_AWARE_PROJECTION"
    [spatial_encoder_type]="$MODEL_SPATIAL_ENCODER_TYPE"
    [geometry_position_mode]="$MODEL_GEOMETRY_POSITION_MODE"
    [num_geometry_projection_layers]="$MODEL_NUM_GEOMETRY_PROJECTION_LAYERS"
    [geometry_projection_num_heads]="$MODEL_GEOMETRY_PROJECTION_NUM_HEADS"
    [use_auxiliary_geometry_head]="$MODEL_USE_AUXILIARY_GEOMETRY_HEAD"
    [use_auxiliary_geometry_loss]="$MODEL_USE_AUXILIARY_GEOMETRY_LOSS"
    [aux_geometry_targets]="$MODEL_AUX_GEOMETRY_TARGETS"
    [lambda_geo]="$MODEL_LAMBDA_GEO"
    [geometry_loss_type]="$MODEL_GEOMETRY_LOSS_TYPE"
    [detach_geometry_targets]="$MODEL_DETACH_GEOMETRY_TARGETS"
    [geometry_gate_init]="$MODEL_GEOMETRY_GATE_INIT"
    [use_geometry_confidence_mask]="$MODEL_USE_GEOMETRY_CONFIDENCE_MASK"
    [allow_missing_geometry_targets]="$MODEL_ALLOW_MISSING_GEOMETRY_TARGETS"
    [geometry_position_max_abs]="$MODEL_GEOMETRY_POSITION_MAX_ABS"
    [geometry_fixed_scene_scale]="$MODEL_GEOMETRY_FIXED_SCENE_SCALE"
    [geometry_projection_dropout]="$MODEL_GEOMETRY_PROJECTION_DROPOUT"
    [tune_geometry_aware_projection]="$MODEL_TUNE_GEOMETRY_AWARE_PROJECTION"
    [geo_rope_fusion_mode]="$MODEL_GEO_ROPE_FUSION_MODE"
    [geo_rope_fusion_max_depth]="$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
    [geo_rope_fusion_group_split]="$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
    [geo_rope_fusion_log_stats]="$MODEL_GEO_ROPE_FUSION_LOG_STATS"
    [use_bev_supervision]="$MODEL_USE_BEV_SUPERVISION"
    [use_depth_supervision]="$MODEL_USE_DEPTH_SUPERVISION"
    [tune_spatial_tower]="$MODEL_TUNE_SPATIAL_TOWER"
    [tune_fusion_block]="$MODEL_TUNE_FUSION_BLOCK"
    [tune_mm_mlp_adapter]="$MODEL_TUNE_MM_MLP_ADAPTER"
    [version]="$MODEL_VERSION"
    [vision_tower]="$LOCAL_SIGLIP"
    [mm_projector_type]="$MODEL_MM_PROJECTOR_TYPE"
    [mm_vision_select_layer]="$MODEL_MM_VISION_SELECT_LAYER"
    [mm_use_im_start_end]="$MODEL_MM_USE_IM_START_END"
    [mm_use_im_patch_token]="$MODEL_MM_USE_IM_PATCH_TOKEN"
    [image_aspect_ratio]="$MODEL_IMAGE_ASPECT_RATIO"
    [image_grid_pinpoints]="$MODEL_IMAGE_GRID_PINPOINTS"
    [mm_patch_merge_type]="$MODEL_MM_PATCH_MERGE_TYPE"
    [bf16]="$MODEL_BF16"
    [tf32]="$MODEL_TF32"
    [model_max_length]="$MODEL_MAX_LENGTH"
    [gradient_checkpointing]="$MODEL_GRADIENT_CHECKPOINTING"
    [lazy_preprocess]="$MODEL_LAZY_PREPROCESS"
    [torch_compile]="$MODEL_TORCH_COMPILE"
    [torch_compile_backend]="$MODEL_TORCH_COMPILE_BACKEND"
    [frames_upbound]="$MODEL_FRAMES_UPBOUND"
    [mm_newline_position]="$MODEL_MM_NEWLINE_POSITION"
    [add_time_instruction]="$MODEL_ADD_TIME_INSTRUCTION"
    [force_sample]="$MODEL_FORCE_SAMPLE"
    [mm_spatial_pool_stride]="$MODEL_MM_SPATIAL_POOL_STRIDE"
)

declare -A DATA_ARGS=(
    [data_path]="$DATA_PATH_YAML"
    [image_folder]="$DATA_ROOT"
    [video_folder]="$DATA_ROOT"
    [spatial_features_root]="$SPATIAL_FEATURES_ROOT"
    [spatial_features_subdir]="$SPATIAL_FEATURES_SUBDIR"
    [zero_spatial_features]="$ZERO_SPATIAL_FEATURES"
    [group_by_modality_length]="$DATA_GROUP_BY_MODALITY_LENGTH"
)

declare -A TRAINING_ARGS=(
    [deepspeed]="$DEEPSPEED_CONFIG"
    [num_train_epochs]="$NUM_TRAIN_EPOCHS"
    [max_steps]="$MAX_STEPS"
    [save_total_limit]="$SAVE_TOTAL_LIMIT"
    [run_name]="$TRAIN_RUN_NAME"
    [output_dir]="$OUTPUT_DIR"
    [per_device_train_batch_size]="$PER_DEVICE_TRAIN_BATCH_SIZE"
    [per_device_eval_batch_size]="4"
    [gradient_accumulation_steps]="$GRADIENT_ACCUMULATION_STEPS"
    [evaluation_strategy]="no"
    [save_strategy]="$SAVE_STRATEGY"
    [save_steps]="$SAVE_STEPS"
    [learning_rate]="$LEARNING_RATE"
    [weight_decay]="$WEIGHT_DECAY"
    [warmup_ratio]="$WARMUP_RATIO"
    [lr_scheduler_type]="$LR_SCHEDULER_TYPE"
    [logging_steps]="$LOGGING_STEPS"
    [dataloader_num_workers]="$DATALOADER_NUM_WORKERS"
    [report_to]="$REPORT_TO"
    [dataloader_drop_last]="$DATALOADER_DROP_LAST"
    [seed]="$SEED"
    [data_seed]="$SEED"
    [spatial_rank_loss_enable]="$SPATIAL_RANK_LOSS_ENABLE"
    [lambda_sim]="$LAMBDA_SIM"
    [spatial_rank_margin]="$SPATIAL_RANK_MARGIN"
    [anchors_per_frame]="$ANCHORS_PER_FRAME"
    [positive_top_percent]="$POSITIVE_TOP_PERCENT"
    [negative_bottom_percent]="$NEGATIVE_BOTTOM_PERCENT"
    [spatial_rank_debug_checks]="$SPATIAL_RANK_DEBUG_CHECKS"
)

echo "========================================"
echo " SpatialStack Training Configuration"
echo "========================================"
echo "--- Resume ---"
echo "  TRAIN_SAVE_ROOT:             $TRAIN_SAVE_ROOT"
echo "  TRAIN_RUN_NAME:              $TRAIN_RUN_NAME"
echo "  OUTPUT_DIR:                  $OUTPUT_DIR"
echo "  RESUME_MODE:                 $RESUME_MODE"
echo "  RESUME_CHECKPOINT_PATH:      $RESUME_CHECKPOINT_PATH"
echo "  SEED:                        $SEED"
echo ""
echo "--- Weights & Biases ---"
echo "  WANDB_MODE:                  ${WANDB_MODE:-}"
echo "  WANDB_RUN_ID:                $WANDB_RUN_ID"
echo "  WANDB_RUN_ID_SOURCE:         $WANDB_RUN_ID_SOURCE"
echo "  WANDB_RUN_ID_FILE:           $WANDB_RUN_ID_FILE"
echo "  WANDB_RESUME:                $WANDB_RESUME"
echo "  WANDB_NAME:                  $WANDB_NAME"
echo "  WANDB_DIR:                   $WANDB_DIR"
echo ""
echo "--- ModelArguments ---"
for key in "${!MODEL_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${MODEL_ARGS[$key]}"
done
echo ""
echo "--- DataArguments ---"
for key in "${!DATA_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${DATA_ARGS[$key]}"
done
echo ""
echo "--- TrainingArguments ---"
for key in "${!TRAINING_ARGS[@]}"; do
    printf "  %-35s %s\n" "$key:" "${TRAINING_ARGS[$key]}"
done

declare -a TORCHRUN_ARGS=()
append_arg_map MODEL_ARGS
append_arg_map DATA_ARGS
append_arg_map TRAINING_ARGS

assert_arg_value use_cut3r_spatialstack True
assert_arg_value tune_cut3r_spatialstack True
assert_arg_value cut3r_spatialstack_fusion_type cross_attn
assert_arg_value cut3r_spatialstack_cross_attn_same_frame_only True
assert_arg_value tune_fusion_block False
assert_arg_value llm_visual_3d_rope_enable False
assert_arg_value use_geometry_aware_projection False
assert_arg_value use_auxiliary_geometry_head False
assert_arg_value use_auxiliary_geometry_loss False
assert_arg_value use_bev_supervision False
assert_arg_value use_depth_supervision False
assert_no_torchrun_arg "--fusion_block"
echo "[SPATIALSTACK-CROSS-ATTN] OK: fusion_type=${MODEL_ARGS[cut3r_spatialstack_fusion_type]}; same_frame_only=${MODEL_ARGS[cut3r_spatialstack_cross_attn_same_frame_only]}; no auxiliary losses."

if is_true "$DRY_RUN_PRINT_ARGS"; then
    echo "--- Final TORCHRUN_ARGS ---"
    printf '  %q' "${TORCHRUN_ARGS[@]}"
    echo ""
    echo "[DRY_RUN] Exiting before srun/torchrun."
    exit 0
fi

srun "${SRUN_FAIL_FAST_ARGS[@]}" --export=ALL torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    --rdzv_id="${SLURM_JOB_ID:-cut3r_spatialstack_cross_attn}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    llava/train/train_mem.py \
    "${TORCHRUN_ARGS[@]}"

exit 0
