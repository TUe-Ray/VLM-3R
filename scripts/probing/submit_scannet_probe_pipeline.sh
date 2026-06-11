#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

MODEL_LABEL="${MODEL_LABEL:-}"
MODEL_PATH="${MODEL_PATH:-}"
ARCH_PRESET="${ARCH_PRESET:-original}"
TRAINING_JOB_ID="${TRAINING_JOB_ID:-}"
TRAINING_DEPENDENCY="${TRAINING_DEPENDENCY:-none}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
TASK_FILE="${TASK_FILE:-}"
FEATURE_LEVELS="${FEATURE_LEVELS:-}"
EXTRACT_JOB_ID="${EXTRACT_JOB_ID:-}"
SMOKE_TASK_ID="${SMOKE_TASK_ID:-0}"
SMOKE_TAG="${SMOKE_TAG:-}"
FAST_DATA_ROOT="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
DEPTH_PROBE_SUBDIR="${DEPTH_PROBE_SUBDIR:-depth_probes_scannet}"
SEMANTIC_PROBE_SUBDIR="${SEMANTIC_PROBE_SUBDIR:-semantic_probes_scannet}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/probing/submit_scannet_probe_pipeline.sh \
    --model-label LABEL \
    --model-path /path/to/checkpoint \
    --arch-preset original|zero_spatial|spatialstack|llm_only

Optional:
  --feature-levels fusion_output,projected_features,layer_0,layer_3
  --training-job-id JOBID
  --train-dependency auto|none|afterok:JOBID
  --output-root PATH
  --sample-indices PATH
  --task-file PATH
  --extract-job-id JOBID
  --smoke-task-id N
  --dry-run

Environment overrides for extraction:
  FAST_DATA_ROOT FEATURE_ROOT SPATIAL_FEATURES_SUBDIR POINT_MAPS_ROOT POINT_MAPS_SUBDIR
  LLM_LAYERS PRE_LLM_FEATURES ALLOW_EUCLIDEAN_DEPTH GPU_WORKERS SHARD_COUNT SHARD_INDEX
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-label) MODEL_LABEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --arch-preset) ARCH_PRESET="$2"; shift 2 ;;
    --feature-levels) FEATURE_LEVELS="$2"; shift 2 ;;
    --training-job-id) TRAINING_JOB_ID="$2"; shift 2 ;;
    --train-dependency|--training-dependency) TRAINING_DEPENDENCY="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --sample-indices) SAMPLE_INDICES="$2"; shift 2 ;;
    --task-file) TASK_FILE="$2"; shift 2 ;;
    --extract-job-id) EXTRACT_JOB_ID="$2"; shift 2 ;;
    --smoke-task-id) SMOKE_TASK_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_LABEL" || -z "$MODEL_PATH" ]]; then
  echo "[ERROR] --model-label and --model-path are required." >&2
  usage >&2
  exit 2
fi
if [[ "$MODEL_LABEL" == *"/"* || "$MODEL_LABEL" == *".."* ]]; then
  echo "[ERROR] MODEL_LABEL must be a simple path-safe name, got: $MODEL_LABEL" >&2
  exit 2
fi
case "$ARCH_PRESET" in
  original|zero_spatial|spatialstack|llm_only) ;;
  *) echo "[ERROR] Unsupported ARCH_PRESET=$ARCH_PRESET" >&2; exit 2 ;;
esac

if [[ "$TRAINING_DEPENDENCY" == "auto" && -z "$TRAINING_JOB_ID" ]]; then
  if [[ "$MODEL_LABEL" =~ ([0-9]{6,})$ ]]; then
    TRAINING_JOB_ID="${BASH_REMATCH[1]}"
  elif [[ "$(basename "$MODEL_PATH")" =~ ([0-9]{6,})$ ]]; then
    TRAINING_JOB_ID="${BASH_REMATCH[1]}"
  fi
fi

safe_label="${MODEL_LABEL//[^A-Za-z0-9_]/_}"
SMOKE_TAG="${SMOKE_TAG:-$safe_label}"

default_feature_levels() {
  case "$ARCH_PRESET" in
    original) echo "fusion_output,projected_features,layer_0,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27" ;;
    zero_spatial) echo "layer_0,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27" ;;
    spatialstack) echo "layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27" ;;
    llm_only) echo "layer_0,layer_3,layer_6,layer_9,layer_15,layer_21,layer_27" ;;
  esac
}

if [[ -z "$FEATURE_LEVELS" ]]; then
  FEATURE_LEVELS="$(default_feature_levels)"
fi
if [[ -z "$TASK_FILE" ]]; then
  TASK_FILE="$OUTPUT_ROOT/probe_task_files/scannet_${safe_label}_probe_tasks.tsv"
fi
mkdir -p "$(dirname "$TASK_FILE")"
: > "$TASK_FILE"
IFS=',' read -ra level_parts <<< "$FEATURE_LEVELS"
for raw_level in "${level_parts[@]}"; do
  level="$(echo "$raw_level" | xargs)"
  if [[ -n "$level" ]]; then
    printf "%s\t%s\n" "$MODEL_LABEL" "$level" >> "$TASK_FILE"
  fi
done
TASK_COUNT="$(awk 'NF && $1 !~ /^#/ {count++} END {print count + 0}' "$TASK_FILE")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "[ERROR] Empty task file: $TASK_FILE" >&2
  exit 1
fi
ARRAY_SPEC="0-$((TASK_COUNT - 1))"

job_id_only() {
  local raw="$1"
  raw="${raw%%;*}"
  echo "$raw"
}

submit() {
  echo "[SUBMIT] $*" >&2
  sbatch --parsable "$@"
}

active_job_id_by_name() {
  local name="$1"
  squeue -h -u "${USER:-$LOGNAME}" -n "$name" -o "%i" 2>/dev/null | head -n 1 || true
}

resolve_training_dependency() {
  local requested="$1"
  local job_id="$2"
  if [[ -z "$requested" || "$requested" == "none" ]]; then
    echo "none"
    return 0
  fi
  if [[ "$requested" != "auto" ]]; then
    echo "$requested"
    return 0
  fi
  if [[ -z "$job_id" ]]; then
    echo "none"
    return 0
  fi

  local queue_state
  queue_state="$(squeue -h -j "$job_id" -o "%T" 2>/dev/null | head -n 1 || true)"
  if [[ -n "$queue_state" ]]; then
    echo "afterok:$job_id"
    return 0
  fi

  local acct_line acct_state acct_exit
  acct_line="$(sacct -n -P -j "$job_id" --format=JobIDRaw,State,ExitCode 2>/dev/null | awk -F'|' -v id="$job_id" '$1 == id {print $2 "|" $3; exit}' || true)"
  if [[ -z "$acct_line" ]]; then
    echo "afterok:$job_id"
    return 0
  fi
  acct_state="${acct_line%%|*}"
  acct_exit="${acct_line#*|}"
  if [[ "$acct_state" == "COMPLETED" && "$acct_exit" == "0:0" ]]; then
    echo "none"
    return 0
  fi
  echo "[ERROR] Training job $job_id is not successful: state=$acct_state exit=$acct_exit" >&2
  return 1
}

RESOLVED_TRAINING_DEPENDENCY="$(resolve_training_dependency "$TRAINING_DEPENDENCY" "$TRAINING_JOB_ID")"
dependency_args=()
if [[ "$RESOLVED_TRAINING_DEPENDENCY" != "none" ]]; then
  dependency_args=(--dependency="$RESOLVED_TRAINING_DEPENDENCY")
fi
if [[ ! -e "$MODEL_PATH" ]]; then
  if [[ "$RESOLVED_TRAINING_DEPENDENCY" == "none" ]]; then
    echo "[ERROR] MODEL_PATH does not exist and no training dependency will wait for it: $MODEL_PATH" >&2
    exit 2
  fi
  echo "[WARN] MODEL_PATH does not exist yet; extraction will wait for $RESOLVED_TRAINING_DEPENDENCY: $MODEL_PATH" >&2
fi

echo "[INFO] Probe model: $MODEL_LABEL"
echo "[INFO] Model path: $MODEL_PATH"
echo "[INFO] Arch preset: $ARCH_PRESET"
echo "[INFO] Feature levels: $FEATURE_LEVELS"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Sample indices: $SAMPLE_INDICES"
echo "[INFO] Task file: $TASK_FILE ($TASK_COUNT tasks; array $ARRAY_SPEC)"
echo "[INFO] Training dependency request: ${TRAINING_DEPENDENCY:-none}"
echo "[INFO] Training job id: ${TRAINING_JOB_ID:-none}"
echo "[INFO] Training dependency resolved: $RESOLVED_TRAINING_DEPENDENCY"
echo "[INFO] Expected aggregate outputs:"
echo "       $OUTPUT_ROOT/depth_probe_scannet_${MODEL_LABEL}_results.csv"
echo "       $OUTPUT_ROOT/semantic_probe_scannet_${MODEL_LABEL}_results.csv"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] Wrote task file but did not submit Slurm jobs."
  echo "[DRY-RUN] Extraction script: scripts/probing/slurm_extract_scannet_probe_features.sbatch"
  echo "[DRY-RUN] Smoke script: scripts/probing/slurm_smoke_scannet_probe.sbatch"
  echo "[DRY-RUN] Depth array script: scripts/probing/slurm_train_depth_probe_scannet_array.sbatch"
  echo "[DRY-RUN] Semantic array script: scripts/probing/slurm_train_semantic_probe_scannet_array.sbatch"
  echo "[DRY-RUN] Aggregate script: scripts/probing/slurm_aggregate_scannet_probe.sbatch"
  exit 0
fi

export MODEL_LABEL MODEL_PATH ARCH_PRESET OUTPUT_ROOT SAMPLE_INDICES TASK_FILE FEATURE_LEVELS FAST_DATA_ROOT
export SMOKE_TASK_ID SMOKE_TAG
export FEATURE_ROOT="${FEATURE_ROOT:-}"
export SPATIAL_FEATURES_SUBDIR="${SPATIAL_FEATURES_SUBDIR:-}"
export POINT_MAPS_ROOT="${POINT_MAPS_ROOT:-}"
export POINT_MAPS_SUBDIR="${POINT_MAPS_SUBDIR:-}"
export LLM_LAYERS="${LLM_LAYERS:-}"
export PRE_LLM_FEATURES="${PRE_LLM_FEATURES:-}"
export ALLOW_EUCLIDEAN_DEPTH="${ALLOW_EUCLIDEAN_DEPTH:-0}"

extract_name="ProbeExtract_${safe_label}"
extract_id="$EXTRACT_JOB_ID"
if [[ -z "$extract_id" ]]; then
  extract_id="$(active_job_id_by_name "$extract_name")"
fi
if [[ -n "$extract_id" ]]; then
  echo "[INFO] Reusing active extraction_job_id=$extract_id"
else
  extract_raw="$(submit "${dependency_args[@]}" --job-name="$extract_name" --export=ALL scripts/probing/slurm_extract_scannet_probe_features.sbatch)"
  extract_id="$(job_id_only "$extract_raw")"
fi
echo "[INFO] extraction_job_id=$extract_id"

smoke_raw="$(submit --dependency="afterok:$extract_id" --job-name="SMOKE_Probe_${safe_label}" --export=ALL scripts/probing/slurm_smoke_scannet_probe.sbatch)"
smoke_id="$(job_id_only "$smoke_raw")"
echo "[INFO] smoke_job_id=$smoke_id"

export PROBE_SUBDIR="$DEPTH_PROBE_SUBDIR"
official_depth_raw="$(submit --dependency="afterok:$smoke_id" --job-name="DepthProbe_${safe_label}" --array="$ARRAY_SPEC" --export=ALL scripts/probing/slurm_train_depth_probe_scannet_array.sbatch)"
official_depth_id="$(job_id_only "$official_depth_raw")"
echo "[INFO] official_depth_job_id=$official_depth_id"

export PROBE_SUBDIR="$SEMANTIC_PROBE_SUBDIR"
official_sem_raw="$(submit --dependency="afterok:$smoke_id" --job-name="SemProbe_${safe_label}" --array="$ARRAY_SPEC" --export=ALL scripts/probing/slurm_train_semantic_probe_scannet_array.sbatch)"
official_sem_id="$(job_id_only "$official_sem_raw")"
echo "[INFO] official_semantic_job_id=$official_sem_id"

unset PROBE_SUBDIR
aggregate_raw="$(submit --dependency="afterok:$official_depth_id:$official_sem_id" --job-name="AggregateProbe_${safe_label}" --export=ALL scripts/probing/slurm_aggregate_scannet_probe.sbatch)"
aggregate_id="$(job_id_only "$aggregate_raw")"
echo "[INFO] aggregate_job_id=$aggregate_id"
