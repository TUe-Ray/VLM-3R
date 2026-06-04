#!/bin/bash
#SBATCH --job-name=scannet_semantic_prep_align
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=32G
#SBATCH --output=logs/probing_scannet_%x_%j.out
#SBATCH --error=logs/probing_scannet_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-/leonardo_work/EUHPC_D32_006/miniconda3}"
CONDA_ENV="${CONDA_ENV:-vlm3r}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  export PATH="$CONDA_BASE/bin:$PATH"
fi
set +u
conda activate "$CONDA_ENV"
set -u
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p logs

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-/leonardo_work/EUHPC_D32_006/probing_experiment/sample_indices.json}"
SCANNET_LABEL_ROOT="${SCANNET_LABEL_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/scannet_semantic_labels/scans}"
SCANNET_LABEL_TSV="${SCANNET_LABEL_TSV:-/leonardo_scratch/fast/EUHPC_D32_006/scannet_semantic_labels/scannetv2-labels.combined.tsv}"
SCANNET20_CLASS_FILE="${SCANNET20_CLASS_FILE:-/leonardo_work/EUHPC_D32_006/data preparation back up/spar_workspace/src/ScanNet/Tasks/Benchmark/classes_SemVoxLabel-nyu40id.txt}"
MODEL_LABELS="${MODEL_LABELS:-zero_spatial,vlm3r_baseline}"
FEATURE_LEVELS="${FEATURE_LEVELS:-}"
LABEL_VALUE_SPACE="${LABEL_VALUE_SPACE:-auto}"
INSPECT_SAMPLES="${INSPECT_SAMPLES:-16}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
RUN_FEATURE_ALIGNMENT="${RUN_FEATURE_ALIGNMENT:-1}"
REQUIRE_DEPTH_GT="${REQUIRE_DEPTH_GT:-1}"
REQUIRE_ALL_FEATURE_LEVELS="${REQUIRE_ALL_FEATURE_LEVELS:-1}"

mkdir -p "$OUTPUT_ROOT"
ln -sfn /leonardo_work/EUHPC_D32_006/probing_experiment/features "$OUTPUT_ROOT/features"
ln -sfn /leonardo_work/EUHPC_D32_006/probing_experiment/metadata "$OUTPUT_ROOT/metadata"
ln -sfn /leonardo_work/EUHPC_D32_006/probing_experiment/gt_depth "$OUTPUT_ROOT/gt_depth"

prepare_cmd=(python -u scripts/probing/prepare_semantic_probe_scannet.py
  --output-root "$OUTPUT_ROOT"
  --sample-indices "$SAMPLE_INDICES"
  --scannet-label-root "$SCANNET_LABEL_ROOT"
  --scannet-label-tsv "$SCANNET_LABEL_TSV"
  --scannet20-class-file "$SCANNET20_CLASS_FILE"
  --label-value-space "$LABEL_VALUE_SPACE"
  --model-labels "$MODEL_LABELS"
  --inspect-samples "$INSPECT_SAMPLES")

if [[ -n "$FEATURE_LEVELS" ]]; then
  prepare_cmd+=(--feature-levels "$FEATURE_LEVELS")
fi
if [[ -n "$LIMIT_FRAMES" ]]; then
  prepare_cmd+=(--limit-frames "$LIMIT_FRAMES")
fi

printf '[INFO] Running prepare command:\n'
printf ' %q' "${prepare_cmd[@]}"
printf '\n'
"${prepare_cmd[@]}"

if [[ "$RUN_FEATURE_ALIGNMENT" == "1" || "$RUN_FEATURE_ALIGNMENT" == "true" ]]; then
  align_cmd=(python -u scripts/probing/check_semantic_feature_alignment.py
    --output-root "$OUTPUT_ROOT"
    --sample-indices "$OUTPUT_ROOT/semantic_probe_scannet_semantic_aligned_sample_indices.json"
    --model-labels "$MODEL_LABELS")

  if [[ -n "$FEATURE_LEVELS" ]]; then
    align_cmd+=(--feature-levels "$FEATURE_LEVELS")
  fi
  if [[ "$REQUIRE_DEPTH_GT" == "1" || "$REQUIRE_DEPTH_GT" == "true" ]]; then
    align_cmd+=(--require-depth-gt)
  else
    align_cmd+=(--no-require-depth-gt)
  fi
  if [[ "$REQUIRE_ALL_FEATURE_LEVELS" == "1" || "$REQUIRE_ALL_FEATURE_LEVELS" == "true" ]]; then
    align_cmd+=(--require-all-feature-levels)
  else
    align_cmd+=(--no-require-all-feature-levels)
  fi

  printf '[INFO] Running feature alignment command:\n'
  printf ' %q' "${align_cmd[@]}"
  printf '\n'
  "${align_cmd[@]}"
fi
