#!/usr/bin/env bash
# Evaluate the migrated legacy Reproduction_2 LoRA checkpoint on VSI-Bench.
# Submit with LIMIT=1 and a SMOKE-prefixed job name before the full run.
#SBATCH --job-name=eval_reproduction2_vsibench
#SBATCH --account=tesei2748
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%j.out
#SBATCH --error=/gpfs/home4/geusdd/shuang/SpatialFocus/logs/eval/%x_%j.err

set -Eeuo pipefail

ACTIVE_REPO_DIR="/gpfs/home4/geusdd/shuang/SpatialFocus"
SNELLIUS_REFERENCE_REPO="/home/geusdd/VLM-3R"
# shellcheck source=/dev/null
source "$SNELLIUS_REFERENCE_REPO/snellius/paths.env"
REPO_DIR="$ACTIVE_REPO_DIR"
export REPO_DIR

PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-$VLM3R_ROOT/checkpoints/Reproduction_2}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
LIMIT="${LIMIT:-0}"
for value_name in NUM_PROCESSES; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 2
    }
done
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "LIMIT must be a non-negative integer, got: $LIMIT" >&2; exit 2; }
if (( LIMIT > 0 )) && [[ "${SLURM_JOB_NAME:-}" != SMOKE* ]]; then
    echo "A limited evaluation must use a SMOKE-prefixed Slurm job name." >&2
    exit 2
fi
if (( LIMIT > 0 && LIMIT < NUM_PROCESSES )); then
    echo "LIMIT=$LIMIT must provide at least one sample per process ($NUM_PROCESSES)." >&2
    exit 2
fi

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
export CONDA_BASE="/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0"
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

JOB_TMP_ROOT="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/reproduction2_eval"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg_cache"
export TRITON_CACHE_DIR="$JOB_TMP_ROOT/triton_cache"
export TORCH_EXTENSIONS_DIR="$JOB_TMP_ROOT/torch_extensions"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$SNELLIUS_REFERENCE_REPO/snellius/validate_migration.sh"
# validate_target_bundle.sh pins an older reference-checkout commit.  The
# migration validator plus the explicit checkpoint/runtime/eval checks below
# validate the assets this evaluation actually consumes without rejecting a
# newer code checkout solely because its Git SHA changed.

for required_dir in \
    "$PRETRAINED_LOCAL" \
    "$LOCAL_MODEL_BASE" \
    "$LOCAL_SIGLIP" \
    "$VSI_BENCH_EVAL_ROOT" \
    "$VSI_BENCH_EVAL_MEDIA_ROOT" \
    "$REPO_DIR/thinking-in-space" \
    "$CUT3R_SOURCE_ROOT/src"; do
    [[ -d "$required_dir" ]] || { echo "Missing required directory: $required_dir" >&2; exit 2; }
done
for required_file in \
    adapter_model.bin \
    non_lora_trainables.bin \
    adapter_config.json \
    config.json \
    generation_config.json; do
    [[ -s "$PRETRAINED_LOCAL/$required_file" ]] || {
        echo "Missing or empty checkpoint artifact: $PRETRAINED_LOCAL/$required_file" >&2
        exit 2
    }
done
[[ -s "$CUT3R_SOURCE_ROOT/src/cut3r_512_dpt_4_64.pth" ]] || {
    echo "Missing CUT3R runtime weights under $CUT3R_SOURCE_ROOT/src" >&2
    exit 2
}

python - "$PRETRAINED_LOCAL/config.json" "$NUM_PROCESSES" <<'PY'
import json
import sys

import torch

config_path, expected_processes = sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("fusion_block") != "cross_attention":
    raise SystemExit(
        "Reproduction_2 config must use fusion_block='cross_attention', got "
        f"{config.get('fusion_block')!r}"
    )
if config.get("spatial_tower") != "cut3r":
    raise SystemExit(f"Reproduction_2 config must use CUT3R, got {config.get('spatial_tower')!r}")
expected = int(expected_processes)
visible = torch.cuda.device_count()
if not torch.cuda.is_available() or visible != expected:
    raise SystemExit(
        f"GPU/rank mismatch: expected {expected} visible CUDA devices, found {visible}"
    )
print(
    "[REPRODUCTION2 PREFLIGHT] checkpoint topology and CUDA world size passed: "
    f"fusion=cross_attention, spatial_tower=cut3r, devices={visible}"
)
PY

SUBMODULE_DIR="$REPO_DIR/thinking-in-space"
SOURCE_TASK_DIR="$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline"
TASK_DIR="$VLM3R_ROOT/migration/eval_tasks/reproduction2_${SLURM_JOB_ID}"
mkdir -p "$TASK_DIR" "$HF_DATASETS_CACHE/vsibench"
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
grep -Fq 'os.getenv("VSI_MEDIA_ROOT"' "$TASK_DIR/utils.py" || {
    echo "Failed to render explicit VSI media root into task utils." >&2
    exit 2
}

RUN_NAME="${RUN_NAME:-Reproduction_2_${SLURM_JOB_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-$VLM3R_ROOT/logs/eval/$RUN_NAME}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$VLM3R_ROOT/migration/eval_runtime/$RUN_NAME}"
COMPAT_VSI_ROOT="$VLM3R_ROOT/migration/eval_roots/$RUN_NAME"
mkdir -p "$COMPAT_VSI_ROOT"
for parquet in test_pruned.parquet test_debiased.parquet; do
    [[ -s "$VSI_BENCH_EVAL_ROOT/$parquet" ]] || {
        echo "Missing VSI-Bench metadata: $VSI_BENCH_EVAL_ROOT/$parquet" >&2
        exit 2
    }
    ln -sfn "$VSI_BENCH_EVAL_ROOT/$parquet" "$COMPAT_VSI_ROOT/$parquet"
done
for split in scannet arkitscenes scannetpp; do
    [[ -d "$VSI_BENCH_EVAL_MEDIA_ROOT/$split" ]] || {
        echo "Missing VSI-Bench media split: $VSI_BENCH_EVAL_MEDIA_ROOT/$split" >&2
        exit 2
    }
    ln -sfn "$VSI_BENCH_EVAL_MEDIA_ROOT/$split" "$COMPAT_VSI_ROOT/$split"
done
export SUBMODULE_DIR PRETRAINED_LOCAL NUM_PROCESSES LIMIT TASK_DIR RUN_NAME OUTPUT_PATH RUNTIME_ROOT
export MODEL_BASE_LOCAL="$LOCAL_MODEL_BASE"
export SIGLIP_LOCAL="$LOCAL_SIGLIP"
export FAST_ROOT="$VLM3R_ROOT"
# eval_vlm3r_orig.sh predates separate metadata/media roots. Supply a job-local
# symlink view satisfying its legacy preflight while the rendered task retains
# explicit metadata and media roots.
export VSI_ROOT="$COMPAT_VSI_ROOT"
export VSI_MEDIA_ROOT="$VSI_BENCH_EVAL_MEDIA_ROOT"
export MODEL_NAME=vlm-3r-llava-qwen2-lora
export MODEL_ATTN_IMPLEMENTATION=flash_attention_2
export MAX_FRAMES_NUM=32
export BATCH_SIZE=1
export EXTRA_MODEL_ARGS=overwrite=False

echo "[REPRODUCTION2 EVAL] checkpoint=$PRETRAINED_LOCAL"
echo "[REPRODUCTION2 EVAL] limit=$LIMIT processes=$NUM_PROCESSES"
echo "[REPRODUCTION2 EVAL] output=$OUTPUT_PATH"
bash "$REPO_DIR/eval_vlm3r_orig.sh"

python - "$OUTPUT_PATH" "$LIMIT" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
limit = int(sys.argv[2])
result_paths = list(output.rglob("results.json"))
sample_paths = list(output.rglob("vsibench.json"))
if not result_paths or not sample_paths:
    raise SystemExit(
        f"Missing evaluation artifacts under {output}: results={result_paths}, samples={sample_paths}"
    )
effective = 0
metrics = []
for path in result_paths:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    effective += int(payload.get("n-samples", {}).get("vsibench", {}).get("effective", 0))
    metrics.append(payload.get("results", {}).get("vsibench", {}))
logged = 0
for path in sample_paths:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    logged += len(payload.get("logs", []))
if limit and (effective != limit or logged != limit):
    raise SystemExit(
        f"Smoke sample-count mismatch: expected={limit}, effective={effective}, logged={logged}"
    )
for metric in metrics:
    for name, value in metric.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit(f"Non-finite evaluation metric: {name}={value}")
print(
    f"[REPRODUCTION2 EVAL PASS] result_files={len(result_paths)} "
    f"effective={effective} logged={logged} metrics={metrics}"
)
PY

echo "Reproduction_2 VSI-Bench evaluation completed: $OUTPUT_PATH"
