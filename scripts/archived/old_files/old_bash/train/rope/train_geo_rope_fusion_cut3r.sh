#!/bin/bash
#SBATCH --job-name=geo_rope_fusion_cut3r_spherical_100p
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4             # 依你的叢集格式：也可能是 --gpus-per-node=1
#SBATCH --ntasks-per-node=1       # 通常 1 個 task，裡面用 torchrun 起多 GPU processes
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg # normal/boost_qos_dbg
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --mem=0
#SBATCH --exclude=lrdn0249,lrdn0612,lrdn0568,lrdn2400,lrdn0288,lrdn0418,lrdn0119,lrdn0159,lrdn0080,lrdn0843,lrdn3322
#SBATCH --exclusive

SUFFIX="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
NOTE="${NOTE:-CUT3R GeoRoPE Fusion: svf_geo_rope_fusion with point-map-derived spherical positions, 100% training data}"

TRAIN_DATA_PERCENTAGE="${TRAIN_DATA_PERCENTAGE:-100}"

MODEL_FUSION_BLOCK="${MODEL_FUSION_BLOCK:-svf_geo_rope_fusion}"
# Fusion options for this GeoRoPE Fusion experiment:
# - svf_patch_only
# - svf_geo_rope_fusion
# - svf_geo_rope_fusion_forced
# - svf_geo_rope_fusion_per_head_gate
# - svf_depth_geo_rope_fusion / svf_xyz_geo_rope_fusion / svf_spherical_geo_rope_fusion
MODEL_GEO_ROPE_FUSION_MODE="${MODEL_GEO_ROPE_FUSION_MODE:-spherical}"
MODEL_GEO_ROPE_FUSION_MAX_DEPTH="${MODEL_GEO_ROPE_FUSION_MAX_DEPTH:-10.0}"
MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="${MODEL_GEO_ROPE_FUSION_GROUP_SPLIT:-2,1,2}"
MODEL_GEO_ROPE_FUSION_LOG_STATS="${MODEL_GEO_ROPE_FUSION_LOG_STATS:-False}"
MODEL_GEO_ROPE_FUSION_LOG_ATTENTION_STATS="${MODEL_GEO_ROPE_FUSION_LOG_ATTENTION_STATS:-False}"
MODEL_GEO_ROPE_GATE_TYPE="${MODEL_GEO_ROPE_GATE_TYPE:-}"
MODEL_GEO_ROPE_HEAD_GATE_INIT="${MODEL_GEO_ROPE_HEAD_GATE_INIT:-0.99}"
MODEL_GEO_ROPE_POINT_MAP_KEY="${MODEL_GEO_ROPE_POINT_MAP_KEY:-point_maps_ref}"
MODEL_LLM_VISUAL_3D_ROPE_ENABLE="${MODEL_LLM_VISUAL_3D_ROPE_ENABLE:-False}"
MODEL_LLM_VISUAL_3D_ROPE_ALPHA="${MODEL_LLM_VISUAL_3D_ROPE_ALPHA:-1.0}"
MODEL_LLM_VISUAL_3D_ROPE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_MODE:-spherical}"
MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT="${MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT:-2,1,2}"
MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH="${MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH:-10.0}"
MODEL_LLM_VISUAL_3D_ROPE_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LAYERS:-all}"
MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE="${MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE:-point_maps_ref}"
MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE:-False}"
MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_MODE="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_MODE:-intra_sample_token_shuffle}"
MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_SEED="${MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_SEED:-0}"
MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS:-True}"
MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS="${MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS:-first_middle_last}"
MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION="${MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION:-True}"
# Group split rules:
# - svf_depth_geo_rope_fusion requires MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="1"
# - svf_xyz_geo_rope_fusion uses x,y,z split, e.g. "1,1,1" or "2,1,2"
# - svf_spherical_geo_rope_fusion uses theta,phi,log_r split, e.g. "1,1,1", "2,1,2", or "3,1,3"
# - svf_geo_rope_fusion uses MODEL_GEO_ROPE_FUSION_MODE to decide depth/xyz/spherical
# Coordinate consistency rule:
# - CUT3R point-map sidecars contain both point_maps_ref/pts3d_in_other_view
#   (reference/anchor-frame coordinates) and point_maps_cam/pts3d_in_self_view
#   (per-frame camera coordinates).
# - Whatever coordinate source is used for training must also be used for eval.
#   Do not let eval-only aliases silently switch ref<->cam.
# Run matrix:
#   baseline:  MODEL_FUSION_BLOCK="svf_patch_only"
#   depth:     MODEL_FUSION_BLOCK="svf_depth_geo_rope_fusion"     MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="1"
#   xyz:       MODEL_FUSION_BLOCK="svf_xyz_geo_rope_fusion"       MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="1,1,1" or "2,1,2"
#   spherical: MODEL_FUSION_BLOCK="svf_spherical_geo_rope_fusion" MODEL_GEO_ROPE_FUSION_GROUP_SPLIT="1,1,1" or "2,1,2" or "3,1,3"
# ============== Training percentage and shuffling (for ablation) ==============

# ============================================================
# User-defined variables: General
# ============================================================
# Data source roots. Default to FAST; keep WORK mirror as fallback.
WORK_DATA_ROOT="${WORK_DATA_ROOT:-/leonardo_work/EUHPC_D32_006/train_data/vlm3r}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DATA_ROOT="${DATA_ROOT:-$FAST_DATA_ROOT}"
# DATA_ROOT="${DATA_ROOT:-$WORK_DATA_ROOT}"  # WORK mirror fallback if FAST is unstable.
DATA_FALLBACK_ROOT="${DATA_FALLBACK_ROOT:-}"
# SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$FAST_DATA_ROOT}"  # FAST point-map sidecars.
SPATIAL_FEATURES_ROOT="${SPATIAL_FEATURES_ROOT:-$DATA_ROOT}"
SPATIAL_FEATURES_FALLBACK_ROOT="${SPATIAL_FEATURES_FALLBACK_ROOT:-$DATA_FALLBACK_ROOT}"
SPATIAL_FEATURES_SUBDIR="spatial_features_points"
CONDA_ENV_NAME="vlm3r"

MODEL_LORA_ENABLE="True"
MODEL_LORA_R="128"
MODEL_LORA_ALPHA="256"
MODEL_SPATIAL_TOWER="cut3r"
MODEL_SPATIAL_TOWER_SELECT_FEATURE="all_tokens"
MODEL_SPATIAL_FEATURE_DIM="768"

# ============================================================
# User-defined variables: Paths
# ============================================================
LOCAL_MODEL_BASE="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2"
LOCAL_SIGLIP="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384"

TRAIN_SAVE_ROOT="/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R"
LOG_DIR="/leonardo_scratch/fast/EUHPC_D32_006/hf_models/VLM3R/train_log"



WANDB_DIR="$WORK/wandb"
WANDB_CACHE_DIR="$WORK/wandb_cache"
WANDB_CONFIG_DIR="$WORK/wandb_config"

HF_HOME="/leonardo_scratch/fast/EUHPC_D32_006/hf_cache"
HF_DATASETS_CACHE="$HF_HOME/datasets"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"

# ============================================================
# User-defined variables: Resume / Ablation
# ============================================================
RESUME_MODE="${RESUME_MODE:-fresh}"                 # choices: fresh / continue
RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-none}"       # e.g. /path/to/checkpoint-1000
ZERO_SPATIAL_FEATURES="False"       # choices: False / True
STRICT_VIDEO_LOADING="${STRICT_VIDEO_LOADING:-True}"
REQUIRE_SPATIAL_FEATURES="${REQUIRE_SPATIAL_FEATURES:-True}"
PREFLIGHT_TRAIN_ASSETS="${PREFLIGHT_TRAIN_ASSETS:-True}"
SEED=42

# ============================================================
# User-defined variables: Model/Data/Training presets
# ============================================================


TRAIN_DATA_PERCENTAGE_SEED="$SEED"
TRAIN_DATA_SHUFFLE="True"
DETERMINISTIC_DATA_ORDER="True"


MODEL_TUNE_SPATIAL_TOWER="False"
MODEL_TUNE_FUSION_BLOCK="True"
MODEL_TUNE_MM_MLP_ADAPTER="True"
MODEL_VERSION="qwen_1_5"
MODEL_MM_PROJECTOR_TYPE="mlp2x_gelu"
MODEL_MM_VISION_SELECT_LAYER="-1"   #  从视觉编码器选择第几层特征
MODEL_MM_USE_IM_START_END="False"   #是否添加特殊的图像起止标记 <image>...</image>
MODEL_MM_USE_IM_PATCH_TOKEN="False" #是否使用图像补丁标记（如 <im_patch_0>）来保留空间位置信息，通常配合融合模块的空间池化使用
MODEL_IMAGE_ASPECT_RATIO="anyres_max_9"
MODEL_IMAGE_GRID_PINPOINTS="(1x1),...,(6x6)"
MODEL_MM_PATCH_MERGE_TYPE="spatial_unpad"
MODEL_BF16="True"
MODEL_TF32="True"
MODEL_MAX_LENGTH="32768"
MODEL_GRADIENT_CHECKPOINTING="${MODEL_GRADIENT_CHECKPOINTING:-True}"
MODEL_LAZY_PREPROCESS="True"
MODEL_TORCH_COMPILE="${MODEL_TORCH_COMPILE:-True}"
MODEL_TORCH_COMPILE_BACKEND="${MODEL_TORCH_COMPILE_BACKEND:-inductor}"
MODEL_FRAMES_UPBOUND="32"
MODEL_MM_NEWLINE_POSITION="grid"
MODEL_ADD_TIME_INSTRUCTION="True"
MODEL_FORCE_SAMPLE="True"
MODEL_MM_SPATIAL_POOL_STRIDE="2"

DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data.yaml}"  # FAST json_path entries.
# DATA_PATH_YAML="${DATA_PATH_YAML:-scripts/VLM_3R/vsibench_data_work.yaml}"  # WORK mirror fallback.
DATA_GROUP_BY_MODALITY_LENGTH="True"


PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LEARNING_RATE="2e-5"
WEIGHT_DECAY="0."
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"
REPORT_TO="${REPORT_TO:-wandb}"
DATALOADER_DROP_LAST="${DATALOADER_DROP_LAST:-True}"


# ========================================================================================

echo "-------- Note --------"
echo "  note: $NOTE"
mkdir -p logs/train
mkdir -p "$LOG_DIR"

JOB_TIME_LIMIT=$(squeue -j "$SLURM_JOB_ID" -h -o "%l")


echo "=== SLURM Job Specifications ==="
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per Node: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
MEMORY_PER_NODE="$SLURM_MEM_PER_NODE"
if [[ -z "$MEMORY_PER_NODE" ]]; then
    MEMORY_PER_NODE="N/A"
fi
echo "Memory per Node: $MEMORY_PER_NODE"
echo "Output: $SLURM_STDOUT"
echo "Error: $SLURM_STDERR"
echo "Job Time Limit: $JOB_TIME_LIMIT"
set -euo pipefail


# Cluster-specific environment setup (overridable via exported env vars)

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
# Some conda activation hooks assume optional vars exist and can fail under nounset.
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

# Prefer conda runtime libs to avoid system/libstdc++ mismatch.
if [[ -v LD_LIBRARY_PATH && -n "$LD_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi

export WANDB_MODE="offline"
export NCCL_NVLS_ENABLE=0
export WANDB_DIR="$WANDB_DIR"
export WANDB_CACHE_DIR="$WANDB_CACHE_DIR"
export WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Force local/offline Hugging Face resolution on compute nodes.
export HF_HOME="$HF_HOME"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DECORD_EOF_RETRY_MAX="$DECORD_EOF_RETRY_MAX"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"
echo "DECORD_EOF_RETRY_MAX=$DECORD_EOF_RETRY_MAX"



# set training parameters
# IMPORTANT: GPU process count is inferred from Slurm allocations.
# set training parameters
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
WORLD_SIZE=$((NNODES * NUM_GPUS_PER_NODE))
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export OMP_NUM_THREADS=2
echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"
echo "[DDP] NNODES=$NNODES"
echo "[DDP] NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"



# ============================================================
# Save/Resume Configuration
# ============================================================
MID_RUN_NAME="$SUFFIX"
OUTPUT_DIR="$TRAIN_SAVE_ROOT/$MID_RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# Derive resume behavior after OUTPUT_DIR is known.
if [[ "$RESUME_CHECKPOINT_PATH" == "none" ]]; then
    if [[ "$RESUME_MODE" == "continue" ]]; then
        RESUME_CHECKPOINT_PATH="auto"
    else
        RESUME_CHECKPOINT_PATH="none"
    fi
fi
export RESUME_CHECKPOINT_PATH

if [[ ! -d "$LOCAL_MODEL_BASE" ]]; then
    echo "[ERROR] Local model base not found: $LOCAL_MODEL_BASE"
    exit 1
fi
if [[ ! -d "$LOCAL_SIGLIP" ]]; then
    echo "[ERROR] Local SigLIP not found: $LOCAL_SIGLIP"
    exit 1
fi

echo "[LOCAL MODEL] model_name_or_path=$LOCAL_MODEL_BASE"
echo "[LOCAL MODEL] vision_tower=$LOCAL_SIGLIP"

denom=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( TARGET_GLOBAL_BATCH_SIZE % denom != 0 )); then
    echo "[ERROR] TARGET_GLOBAL_BATCH_SIZE($TARGET_GLOBAL_BATCH_SIZE) not divisible by WORLD_SIZE*PER_DEVICE_TRAIN_BATCH_SIZE($denom)"
    echo "Please adjust TARGET_GLOBAL_BATCH_SIZE or per-device batch size."
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / denom))
echo "[BATCH] PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "[BATCH] TARGET_GLOBAL_BATCH_SIZE=$TARGET_GLOBAL_BATCH_SIZE"
echo "[BATCH] GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"

# Ablation switch:
#   False -> use normal spatial_features (.pt)
#   True  -> load .pt and zero all tensor values
echo "[ABLATION] ZERO_SPATIAL_FEATURES=$ZERO_SPATIAL_FEATURES"

# Alias fusion blocks define their own GeoRoPE Fusion coordinate mode.
# This keeps Slurm submissions robust even when MODEL_GEO_ROPE_FUSION_MODE
# is not explicitly exported for depth/xyz/spherical aliases.
case "$MODEL_FUSION_BLOCK" in
    svf_depth_geo_rope_fusion)
        MODEL_GEO_ROPE_FUSION_MODE="depth"
        ;;
    svf_xyz_geo_rope_fusion)
        MODEL_GEO_ROPE_FUSION_MODE="xyz"
        ;;
    svf_spherical_geo_rope_fusion)
        MODEL_GEO_ROPE_FUSION_MODE="spherical"
        ;;
esac

# Validate fusion option and print method summary for reproducibility.
VALID_FUSION_BLOCKS=(
    "svf_patch_only"
    "svf_geo_rope_fusion"
    "svf_geo_rope_fusion_forced"
    "svf_geo_rope_fusion_per_head_gate"
    "svf_depth_geo_rope_fusion"
    "svf_xyz_geo_rope_fusion"
    "svf_spherical_geo_rope_fusion"
)
VALID_GEO_ROPE_FUSION_MODES=("depth" "xyz" "spherical")
IS_VALID_FUSION="False"
for fusion_name in "${VALID_FUSION_BLOCKS[@]}"; do
    if [[ "$MODEL_FUSION_BLOCK" == "$fusion_name" ]]; then
        IS_VALID_FUSION="True"
        break
    fi
done

if [[ "$IS_VALID_FUSION" != "True" ]]; then
    echo "[ERROR] Unsupported MODEL_FUSION_BLOCK: $MODEL_FUSION_BLOCK"
    echo "[ERROR] Supported values: ${VALID_FUSION_BLOCKS[*]}"
    exit 1
fi

IS_VALID_GEO_ROPE_FUSION_MODE="False"
for geo_rope_fusion_mode in "${VALID_GEO_ROPE_FUSION_MODES[@]}"; do
    if [[ "$MODEL_GEO_ROPE_FUSION_MODE" == "$geo_rope_fusion_mode" ]]; then
        IS_VALID_GEO_ROPE_FUSION_MODE="True"
        break
    fi
done

if [[ "$IS_VALID_GEO_ROPE_FUSION_MODE" != "True" ]]; then
    echo "[ERROR] Unsupported MODEL_GEO_ROPE_FUSION_MODE: $MODEL_GEO_ROPE_FUSION_MODE"
    echo "[ERROR] Supported values: ${VALID_GEO_ROPE_FUSION_MODES[*]}"
    exit 1
fi

echo "[FUSION] MODEL_FUSION_BLOCK=$MODEL_FUSION_BLOCK"
case "$MODEL_FUSION_BLOCK" in
    svf_patch_only)
        echo "[FUSION] svf_patch_only: baseline patch-only cross-attention, Q=2D, KV=patch_tokens"
        ;;
    svf_geo_rope_fusion)
        echo "[FUSION] svf_geo_rope_fusion: GeoRoPE Fusion cross-attention, mode=$MODEL_GEO_ROPE_FUSION_MODE, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
    svf_geo_rope_fusion_forced)
        echo "[FUSION] svf_geo_rope_fusion_forced: forced full GeoRoPE Q/K, mode=$MODEL_GEO_ROPE_FUSION_MODE, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
    svf_geo_rope_fusion_per_head_gate)
        echo "[FUSION] svf_geo_rope_fusion_per_head_gate: shared sigmoid per-head GeoRoPE gate, init=$MODEL_GEO_ROPE_HEAD_GATE_INIT, mode=$MODEL_GEO_ROPE_FUSION_MODE, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
    svf_depth_geo_rope_fusion)
        echo "[FUSION] svf_depth_geo_rope_fusion: GeoRoPE Fusion cross-attention, mode=depth, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
    svf_xyz_geo_rope_fusion)
        echo "[FUSION] svf_xyz_geo_rope_fusion: GeoRoPE Fusion cross-attention, mode=xyz, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
    svf_spherical_geo_rope_fusion)
        echo "[FUSION] svf_spherical_geo_rope_fusion: GeoRoPE Fusion cross-attention, mode=spherical, split=$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
        ;;
esac

declare -A MODEL_ARGS=(
    [model_name_or_path]="$LOCAL_MODEL_BASE"
    [lora_enable]="$MODEL_LORA_ENABLE"
    [lora_r]="$MODEL_LORA_R"
    [lora_alpha]="$MODEL_LORA_ALPHA"
    [spatial_tower]="$MODEL_SPATIAL_TOWER"
    [spatial_tower_select_feature]="$MODEL_SPATIAL_TOWER_SELECT_FEATURE"
    [spatial_feature_dim]="$MODEL_SPATIAL_FEATURE_DIM"
    [fusion_block]="$MODEL_FUSION_BLOCK"
    [geo_rope_fusion_mode]="$MODEL_GEO_ROPE_FUSION_MODE"
    [geo_rope_fusion_max_depth]="$MODEL_GEO_ROPE_FUSION_MAX_DEPTH"
    [geo_rope_fusion_group_split]="$MODEL_GEO_ROPE_FUSION_GROUP_SPLIT"
    [geo_rope_fusion_log_stats]="$MODEL_GEO_ROPE_FUSION_LOG_STATS"
    [geo_rope_fusion_log_attention_stats]="$MODEL_GEO_ROPE_FUSION_LOG_ATTENTION_STATS"
    [geo_rope_head_gate_init]="$MODEL_GEO_ROPE_HEAD_GATE_INIT"
    [geo_rope_point_map_key]="$MODEL_GEO_ROPE_POINT_MAP_KEY"
    [llm_visual_3d_rope_enable]="$MODEL_LLM_VISUAL_3D_ROPE_ENABLE"
    [llm_visual_3d_rope_alpha]="$MODEL_LLM_VISUAL_3D_ROPE_ALPHA"
    [llm_visual_3d_rope_mode]="$MODEL_LLM_VISUAL_3D_ROPE_MODE"
    [llm_visual_3d_rope_group_split]="$MODEL_LLM_VISUAL_3D_ROPE_GROUP_SPLIT"
    [llm_visual_3d_rope_max_depth]="$MODEL_LLM_VISUAL_3D_ROPE_MAX_DEPTH"
    [llm_visual_3d_rope_layers]="$MODEL_LLM_VISUAL_3D_ROPE_LAYERS"
    [llm_visual_3d_rope_geometry_source]="$MODEL_LLM_VISUAL_3D_ROPE_GEOMETRY_SOURCE"
    [llm_visual_3d_rope_shuffle]="$MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE"
    [llm_visual_3d_rope_shuffle_mode]="$MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_MODE"
    [llm_visual_3d_rope_shuffle_seed]="$MODEL_LLM_VISUAL_3D_ROPE_SHUFFLE_SEED"
    [llm_visual_3d_rope_log_stats]="$MODEL_LLM_VISUAL_3D_ROPE_LOG_STATS"
    [llm_visual_3d_rope_log_layers]="$MODEL_LLM_VISUAL_3D_ROPE_LOG_LAYERS"
    [llm_visual_3d_rope_force_eager_attention]="$MODEL_LLM_VISUAL_3D_ROPE_FORCE_EAGER_ATTENTION"
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
if [[ -n "$MODEL_GEO_ROPE_GATE_TYPE" ]]; then
    MODEL_ARGS[geo_rope_gate_type]="$MODEL_GEO_ROPE_GATE_TYPE"
fi

declare -A DATA_ARGS=(
    [data_path]="$DATA_PATH_YAML"
    [image_folder]="$DATA_ROOT"
    [video_folder]="$DATA_ROOT"
    [video_fallback_folder]="$DATA_FALLBACK_ROOT"
    [zero_spatial_features]="$ZERO_SPATIAL_FEATURES"
    [strict_video_loading]="$STRICT_VIDEO_LOADING"
    [spatial_features_root]="$SPATIAL_FEATURES_ROOT"
    [spatial_features_fallback_root]="$SPATIAL_FEATURES_FALLBACK_ROOT"
    [spatial_features_subdir]="$SPATIAL_FEATURES_SUBDIR"
    [require_spatial_features]="$REQUIRE_SPATIAL_FEATURES"
    [train_data_percentage]="$TRAIN_DATA_PERCENTAGE"
    [train_data_percentage_seed]="$TRAIN_DATA_PERCENTAGE_SEED"
    [train_data_shuffle]="$TRAIN_DATA_SHUFFLE"
    [deterministic_data_order]="$DETERMINISTIC_DATA_ORDER"
    [group_by_modality_length]="$DATA_GROUP_BY_MODALITY_LENGTH"   #控制 dataloader sampler 是否按模態長度分組（
                                        #通常可減少 padding、讓 batch 更穩定）
)


declare -A TRAINING_ARGS=(
    [deepspeed]="scripts/zero2.json"
    [num_train_epochs]="$NUM_TRAIN_EPOCHS"  # 1 epoch for ablation, adjust as needed
    [save_total_limit]="$SAVE_TOTAL_LIMIT"
    [run_name]="$SUFFIX"
    [output_dir]="$OUTPUT_DIR"
    [per_device_train_batch_size]="$PER_DEVICE_TRAIN_BATCH_SIZE"
    [per_device_eval_batch_size]="4"
    [gradient_accumulation_steps]="$GRADIENT_ACCUMULATION_STEPS"
    [evaluation_strategy]="no"
    #[save_strategy]="epoch"
    [save_strategy]="$SAVE_STRATEGY"
    [save_steps]="$SAVE_STEPS"  # Save every 100 steps, adjust as needed
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
)
if [[ -n "$MAX_STEPS" ]]; then
    TRAINING_ARGS[max_steps]="$MAX_STEPS"
fi

echo "========================================"
echo " Training Configuration"
echo "========================================"

echo "--- Resume ---"
echo "  TRAIN_SAVE_ROOT:                     $TRAIN_SAVE_ROOT"
echo "  OUTPUT_RUN_NAME (job+id):            $MID_RUN_NAME"
echo "  OUTPUT_DIR:                          $OUTPUT_DIR"
echo "  RESUME_MODE:                         $RESUME_MODE"
echo "  RESUME_CHECKPOINT_PATH:             $RESUME_CHECKPOINT_PATH"
echo "  SEED:                               $SEED"
if [[ "$RESUME_CHECKPOINT_PATH" != "none" ]]; then
    echo "  *** RESUMING TRAINING — weights will be updated in-place ***"
else
    echo "  *** FRESH TRAINING — new run ***"
fi
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

for key in "${!MODEL_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${MODEL_ARGS[$key]}")
done

for key in "${!DATA_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${DATA_ARGS[$key]}")
done

for key in "${!TRAINING_ARGS[@]}"; do
    TORCHRUN_ARGS+=("--${key}")
    TORCHRUN_ARGS+=("${TRAINING_ARGS[$key]}")
done

if [[ "$PREFLIGHT_TRAIN_ASSETS" == "True" ]]; then
    echo "[PREFLIGHT] Checking training videos and spatial feature sidecars before launching DDP..."
    python scripts/preflight_vlm3r_training_assets.py \
        --data-yaml "$DATA_PATH_YAML" \
        --video-root "$DATA_ROOT" \
        --spatial-features-root "$SPATIAL_FEATURES_ROOT" \
        --spatial-features-subdir "$SPATIAL_FEATURES_SUBDIR" \
        --require-videos \
        --require-spatial-features
fi

cleanup_on_signal() {
    local status=$?
    trap - EXIT TERM INT ERR
    if [[ "$status" -ne 0 ]]; then
        echo "[ERROR] Training command failed with status $status; canceling job ${SLURM_JOB_ID:-unknown} to avoid idle allocation."
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            scancel "$SLURM_JOB_ID" >/dev/null 2>&1 || true
        fi
    fi
    exit "$status"
}
trap cleanup_on_signal EXIT TERM INT ERR

srun --kill-on-bad-exit=1 --wait=30 --export=ALL torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NUM_GPUS_PER_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        llava/train/train_mem.py \
        "${TORCHRUN_ARGS[@]}"

exit 0
