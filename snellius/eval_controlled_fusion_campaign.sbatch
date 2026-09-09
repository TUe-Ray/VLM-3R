#!/usr/bin/env bash
# Official VSI-Bench evaluation array for the controlled B/C/D/E/H campaign.
# Submit with TRAIN_ARRAY_JOB_ID and --dependency=aftercorr:<training-array-id>.
#SBATCH --job-name=eval_controlled_fusion_BCDEH
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --array=0-4%5
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%A_%a.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%A_%a.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
SNELLIUS_REFERENCE_REPO="/home/geusdd/VLM-3R"
# shellcheck source=/dev/null
source "$SNELLIUS_REFERENCE_REPO/snellius/paths.env"
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR

TRAIN_ARRAY_JOB_ID="${TRAIN_ARRAY_JOB_ID:?Set TRAIN_ARRAY_JOB_ID to the official training array job ID.}"
[[ "$TRAIN_ARRAY_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAIN_ARRAY_JOB_ID must be numeric, got: $TRAIN_ARRAY_JOB_ID" >&2
    exit 2
}

ARCH_IDS=(B C D E H)
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this launcher as an array task.}"
if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#ARCH_IDS[@]} )); then
    echo "Unsupported array index: $ARRAY_INDEX" >&2
    exit 2
fi
ARCH_ID="${ARCH_IDS[$ARRAY_INDEX]}"

# Reuse the canonical controlled-candidate naming instead of maintaining a
# second architecture table in the Snellius wrapper.
export CONTROLLED_FUSION_ID="$ARCH_ID"
export CONTROLLED_FUSION_CONFIG_ONLY=True
# shellcheck source=/dev/null
source "$REPO_DIR/train_controlled_fusion_variant.sh"
unset CONTROLLED_FUSION_CONFIG_ONLY

PRETRAINED_LOCAL="$TRAIN_SAVE_ROOT/controlled_fusion_official/official_${SUFFIX}_4n4g_16gpu_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}"
export PRETRAINED_LOCAL
case "${CONTROLLED_FUSION_EVAL_CONFIG_ONLY:-False}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        echo "[OFFICIAL EVAL PLAN] arch=$ARCH_ID training_array=$TRAIN_ARRAY_JOB_ID task=$ARRAY_INDEX"
        echo "[OFFICIAL EVAL PLAN] checkpoint=$PRETRAINED_LOCAL"
        exit 0
        ;;
esac
for required_file in config.json adapter_config.json non_lora_trainables.bin generation_config.json; do
    [[ -s "$PRETRAINED_LOCAL/$required_file" ]] || {
        echo "Missing or empty official checkpoint artifact: $PRETRAINED_LOCAL/$required_file" >&2
        exit 3
    }
done
if [[ ! -s "$PRETRAINED_LOCAL/adapter_model.bin" && ! -s "$PRETRAINED_LOCAL/adapter_model.safetensors" ]]; then
    echo "Missing final LoRA adapter under $PRETRAINED_LOCAL" >&2
    exit 3
fi

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
export CONDA_BASE="/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1"
export CONDA_ENV="$VLM3R_CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR:$SNELLIUS_REFERENCE_REPO/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export CUT3R_SOURCE_ROOT="$SNELLIUS_REFERENCE_REPO/third_party/CUT3R"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST=8.0

JOB_TMP_ROOT="/scratch-local/${SLURM_JOB_ID}/controlled_fusion_eval_${ARRAY_INDEX}"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg_cache"
export TRITON_CACHE_DIR="$JOB_TMP_ROOT/triton_cache"
export TORCH_EXTENSIONS_DIR="$JOB_TMP_ROOT/torch_extensions"
TASK_DIR="$JOB_TMP_ROOT/vsibench_task"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TASK_DIR"

SOURCE_TASK_DIR="$REPO_DIR/thinking-in-space/lmms_eval/tasks/vsibench_leonardo_offline"
sed \
    's|cache_dir = os.path.join(base_cache_dir, cache_name)|cache_dir = os.path.expanduser(os.getenv("VSI_MEDIA_ROOT", os.path.join(base_cache_dir, cache_name)))|' \
    "$SOURCE_TASK_DIR/utils.py" > "$TASK_DIR/utils.py"
sed \
    -e "s|/leonardo_scratch/fast/EUHPC_D32_006/vsibench|$VSI_BENCH_EVAL_ROOT|g" \
    -e "s|cache_dir: vsibench|cache_dir: $HF_DATASETS_CACHE/vsibench|" \
    "$SOURCE_TASK_DIR/vsibench.yaml" > "$TASK_DIR/vsibench.yaml"
if grep -En '/leonardo_|cache_dir: vsibench$' "$TASK_DIR/vsibench.yaml"; then
    echo "Rendered task still contains a Leonardo or relative dataset path." >&2
    exit 2
fi

EVAL_TOKEN="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
OUTPUT_PATH="$VLM3R_LOG_ROOT/eval/controlled_fusion_official/${SUFFIX}_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}/eval_${EVAL_TOKEN}"
RUNTIME_ROOT="$JOB_TMP_ROOT/runtime"
RUN_NAME="eval_official_${SUFFIX}_${TRAIN_ARRAY_JOB_ID}_${ARRAY_INDEX}_${EVAL_TOKEN}"
export OUTPUT_PATH RUNTIME_ROOT RUN_NAME TASK_DIR
export TASK_FILE="$TASK_DIR/vsibench.yaml"
export SUBMODULE_DIR="$REPO_DIR/thinking-in-space"
export MODEL_BASE_LOCAL="$LOCAL_MODEL_BASE"
export SIGLIP_LOCAL="$LOCAL_SIGLIP"
export FAST_ROOT="$VLM3R_ROOT"
export VSI_ROOT="$VSI_BENCH_EVAL_ROOT"
export VSI_MEDIA_ROOT="$VSI_BENCH_EVAL_MEDIA_ROOT"
export SPATIAL_FEATURES_ROOT="$CUT3R_DEC12_ROOT"
export SPATIAL_FEATURES_SUBDIR="12:$CUT3R_DEC12_ROOT:spatial_features"
export NUM_PROCESSES=4
export BATCH_SIZE=1
export LIMIT=0
export MAX_FRAMES_NUM=32
export CHECK_SPATIAL_SIDECARS=True
export MODEL_ATTN_IMPLEMENTATION=sdpa

echo "[OFFICIAL EVAL] arch=$ARCH_ID training_array=$TRAIN_ARRAY_JOB_ID task=$ARRAY_INDEX"
echo "[OFFICIAL EVAL] checkpoint=$PRETRAINED_LOCAL"
echo "[OFFICIAL EVAL] output=$OUTPUT_PATH"

case "$ARCH_ID" in
    B) EVAL_WRAPPER="eval_controlled_fusion_B_pre_projector_add_dec12_vsibench.sh" ;;
    C) EVAL_WRAPPER="eval_controlled_fusion_C_cross_attn_dec12_llm0_vsibench.sh" ;;
    D) EVAL_WRAPPER="eval_controlled_fusion_D_add_dec12_llm0_vsibench.sh" ;;
    E) EVAL_WRAPPER="eval_controlled_fusion_E_add_dec12x3_llm0_1_2_vsibench.sh" ;;
    H) EVAL_WRAPPER="eval_controlled_fusion_H_cross_attn_dec12x3_llm0_1_2_vsibench.sh" ;;
esac
bash "$REPO_DIR/$EVAL_WRAPPER"

"$VLM3R_CONDA_PREFIX/bin/python" - "$OUTPUT_PATH" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
result_paths = list(output.rglob("results.json"))
sample_paths = list(output.rglob("vsibench.json"))
if not result_paths or not sample_paths:
    raise SystemExit(
        f"Missing evaluation artifacts under {output}: results={result_paths}, samples={sample_paths}"
    )
effective = 0
metrics = []
for path in result_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    effective += int(payload.get("n-samples", {}).get("vsibench", {}).get("effective", 0))
    metrics.append(payload.get("results", {}).get("vsibench", {}))
if effective <= 0:
    raise SystemExit(f"Evaluation completed without effective samples: {effective}")
for metric in metrics:
    for name, value in metric.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit(f"Non-finite evaluation metric: {name}={value}")
print(f"[OFFICIAL EVAL PASS] effective={effective} metrics={metrics}")
PY

echo "[OFFICIAL EVAL PASS] arch=$ARCH_ID output=$OUTPUT_PATH"
