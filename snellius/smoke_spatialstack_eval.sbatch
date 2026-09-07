#!/usr/bin/env bash
# One-node VSI-Bench evaluation smoke against the checkpoint emitted by the
# Snellius-only one-step SpatialStack smoke training wrapper.
#SBATCH --job-name=SMOKE_vlm3r_spatialstack_eval
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
#SBATCH --output=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_eval_%j.out
#SBATCH --error=/scratch-shared/geusdd/VLM3R/logs/SMOKE_vlm3r_spatialstack_eval_%j.err

set -Eeuo pipefail

REPO_DIR="/home/geusdd/VLM-3R"
source "$REPO_DIR/snellius/paths.env"

# `paths.env` provides the standard four-rank configuration.  Dedicated smoke
# wrappers may request a smaller world size without changing that shared file.
if [[ -n "${SMOKE_NUM_PROCESSES:-}" ]]; then
  [[ "$SMOKE_NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SMOKE_NUM_PROCESSES: $SMOKE_NUM_PROCESSES" >&2
    exit 2
  }
  export NUM_PROCESSES="$SMOKE_NUM_PROCESSES"
fi

module purge
module load 2023
module load CUDA/12.1.1
VLM3R_CONDA_PREFIX="/home/geusdd/.conda/envs/vlm3r-snellius"
[[ -x "$VLM3R_CONDA_PREFIX/bin/python" ]] || { echo "missing environment: $VLM3R_CONDA_PREFIX" >&2; exit 2; }
export CONDA_PREFIX="$VLM3R_CONDA_PREFIX"
export PATH="$VLM3R_CONDA_PREFIX/bin:$PATH"
# The shared evaluation launcher is also used on Leonardo and activates its
# configured conda environment itself.  Point it at Snellius's conda shim and
# the explicit prefix so this non-interactive batch shell never relies on a
# user-level `conda init`.
export CONDA_BASE="/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0"
export CONDA_ENV="$VLM3R_CONDA_PREFIX"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/CUT3R${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:?CUDA module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST=8.0
VLM3R_JOB_TMPDIR="${TMPDIR:-/scratch-local}/${SLURM_JOB_ID}/vlm3r"
export TMPDIR="$VLM3R_JOB_TMPDIR"
mkdir -p "$VLM3R_JOB_TMPDIR"
trap 'rm -rf -- "$VLM3R_JOB_TMPDIR"' EXIT

"$REPO_DIR/snellius/validate_migration.sh"
"$REPO_DIR/snellius/validate_target_bundle.sh"

CHECKPOINT_FILE="$VLM3R_ROOT/migration/latest_spatialstack_smoke_checkpoint.txt"
if [[ -n "${SMOKE_PRETRAINED_LOCAL:-}" ]]; then
  PRETRAINED_LOCAL="$SMOKE_PRETRAINED_LOCAL"
else
  [[ -f "$CHECKPOINT_FILE" ]] || { echo "missing smoke checkpoint pointer: $CHECKPOINT_FILE" >&2; exit 2; }
  PRETRAINED_LOCAL="$(<"$CHECKPOINT_FILE")"
fi
[[ -d "$PRETRAINED_LOCAL" ]] || { echo "missing smoke checkpoint: $PRETRAINED_LOCAL" >&2; exit 2; }

export SUBMODULE_DIR="$REPO_DIR/thinking-in-space"
export PRETRAINED_LOCAL
export MODEL_BASE_LOCAL="$LOCAL_MODEL_BASE"
export SIGLIP_LOCAL="$LOCAL_SIGLIP"
export FAST_ROOT="$VLM3R_ROOT"
export VSI_ROOT="$VSI_BENCH_EVAL_ROOT"
export VSI_MEDIA_ROOT="$VSI_BENCH_EVAL_MEDIA_ROOT"
export SPATIAL_FEATURES_ROOT="$CUT3R_DEC6_ROOT"
export CUT3R_TOKEN_FEATURES_ROOT="$CUT3R_DEC12_ROOT"
export SPATIAL_FEATURES_SUBDIR="6:$CUT3R_DEC6_ROOT:spatial_features_dec_6;9:$CUT3R_DEC9_ROOT:spatial_features_dec_9;12:$CUT3R_DEC12_ROOT:spatial_features"
SOURCE_TASK_DIR="$SUBMODULE_DIR/lmms_eval/tasks/vsibench_leonardo_offline"
# Four ranks are the standard smoke setting; the shared-node fallback passes
# SMOKE_NUM_PROCESSES=1 and must retain that valid single-GPU world size.
export NUM_PROCESSES="${SMOKE_NUM_PROCESSES:-4}"
export BATCH_SIZE=1
export MAX_FRAMES_NUM=32
export CONV_TEMPLATE=qwen_1_5
export MODEL_NAME=vlm-3r-llava-qwen2-lora
export MODEL_ATTN_IMPLEMENTATION=flash_attention_2
export PRESERVE_CHECKPOINT_CONFIG=True
export EXPECTED_CUT3R_SPATIALSTACK_LAYERS=6,9,12
export EXPECTED_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
export EXPECTED_CUT3R_SPATIALSTACK_PROJECTOR_TYPE=token_mlp
export CHECK_SPATIAL_SIDECARS=True
# Every distributed rank must receive at least one request.  The evaluator's
# request padding assumes this, and using one global sample with four ranks can
# leave ranks with no local request.
export LIMIT="${SMOKE_LIMIT:-$NUM_PROCESSES}"
if [[ ! "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT < NUM_PROCESSES )); then
  echo "SMOKE_LIMIT must be an integer >= NUM_PROCESSES ($NUM_PROCESSES), got: $LIMIT" >&2
  exit 2
fi
export RUN_NAME="SMOKE_spatialstack_dec6_9_12_${SLURM_JOB_ID}"
export OUTPUT_PATH="$VLM3R_ROOT/logs/$RUN_NAME"
export RUNTIME_ROOT="$VLM3R_ROOT/migration/eval_runtime"

# The shared task config names Leonardo's parquet location.  Render a
# job-local copy with the validated Snellius mirror while preserving the task
# utility module beside the YAML for lmms-eval's !function references.
TASK_DIR="$VLM3R_ROOT/migration/eval_tasks/${SLURM_JOB_ID}"
mkdir -p "$TASK_DIR"
mkdir -p "$HF_DATASETS_CACHE/vsibench"
# The upstream VSI utility historically derives the video root from the
# dataset cache name. Keep metadata caching and migrated media independent by
# rendering a job-local utility that honors the explicit VSI_MEDIA_ROOT.
sed \
  's|cache_dir = os.path.join(base_cache_dir, cache_name)|cache_dir = os.path.expanduser(os.getenv("VSI_MEDIA_ROOT", os.path.join(base_cache_dir, cache_name)))|' \
  "$SOURCE_TASK_DIR/utils.py" > "$TASK_DIR/utils.py"
sed \
  -e "s|/leonardo_scratch/fast/EUHPC_D32_006/vsibench|$VSI_ROOT|g" \
  -e "s|cache_dir: vsibench|cache_dir: $HF_DATASETS_CACHE/vsibench|" \
  "$SOURCE_TASK_DIR/vsibench.yaml" > "$TASK_DIR/vsibench.yaml"
export TASK_DIR
export TASK_FILE="$TASK_DIR/vsibench.yaml"
[[ -f "$TASK_FILE" ]] || { echo "failed to render Snellius eval task: $TASK_FILE" >&2; exit 2; }
grep -Fq 'os.getenv("VSI_MEDIA_ROOT"' "$TASK_DIR/utils.py" || {
  echo "failed to render explicit VSI media root into $TASK_DIR/utils.py" >&2
  exit 2
}
if grep -En '/leonardo_|cache_dir: vsibench$' "$TASK_FILE"; then
  echo "rendered task still contains a host-local or relative cache path: $TASK_FILE" >&2
  exit 2
fi

python - "$NUM_PROCESSES" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
visible = torch.cuda.device_count()
if not torch.cuda.is_available() or visible != expected:
    raise SystemExit(
        f"GPU/rank mismatch: expected {expected} visible CUDA devices, found {visible}"
    )
print(f"[SMOKE EVAL] CUDA preflight passed: {visible} devices for {expected} ranks")
PY

echo "[SMOKE EVAL] checkpoint=$PRETRAINED_LOCAL"
echo "[SMOKE EVAL] task=$TASK_DIR limit=$LIMIT output=$OUTPUT_PATH"
bash "$REPO_DIR/eval_spatialstack_vsibench.sh"

[[ -d "$OUTPUT_PATH" ]] || { echo "missing evaluation output directory: $OUTPUT_PATH" >&2; exit 3; }
python - "$OUTPUT_PATH" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_samples = int(sys.argv[2])
results = list(output.rglob("results.json"))
samples = list(output.rglob("vsibench.json"))
if not results:
    raise SystemExit(f"evaluation wrote no results.json under {output}")
if not samples:
    raise SystemExit(f"evaluation wrote no vsibench.json under {output}")
parsed_results = []
parsed_samples = []
for path in results:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload:
        raise SystemExit(f"evaluation artifact is empty: {path}")
    parsed_results.append(payload)
for path in samples:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload:
        raise SystemExit(f"evaluation artifact is empty: {path}")
    parsed_samples.append(payload)

effective = sum(
    int(payload.get("n-samples", {}).get("vsibench", {}).get("effective", 0))
    for payload in parsed_results
)
logged = sum(len(payload.get("logs", [])) for payload in parsed_samples)
if effective != expected_samples or logged != expected_samples:
    raise SystemExit(
        "evaluation sample-count mismatch: "
        f"expected={expected_samples}, effective={effective}, logged={logged}"
    )
print(
    f"[SMOKE EVAL] validated {len(results)} result file(s), "
    f"{len(samples)} sample file(s), and {logged} aggregated samples"
)
PY

echo "SpatialStack VSI-Bench evaluation smoke passed: $OUTPUT_PATH"
