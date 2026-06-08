#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

TRAINING_JOB_ID="${TRAINING_JOB_ID:-44323703}"
TRAINING_DEPENDENCY="${TRAINING_DEPENDENCY:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/probing/scannet_cut3r_spatialstack_probe_tasks.tsv}"
EXTRACT_JOB_ID="${EXTRACT_JOB_ID:-}"
SMOKE_TASK_ID="${SMOKE_TASK_ID:-0}"
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
  echo "[ERROR] Training job $job_id is not a successful completed job: state=$acct_state exit=$acct_exit" >&2
  return 1
}

RESOLVED_TRAINING_DEPENDENCY="$(resolve_training_dependency "$TRAINING_DEPENDENCY" "$TRAINING_JOB_ID")"
dependency_args=()
if [[ "$RESOLVED_TRAINING_DEPENDENCY" != "none" ]]; then
  dependency_args=(--dependency="$RESOLVED_TRAINING_DEPENDENCY")
fi

echo "[INFO] Probe model: cut3r_spatialstack_44323703"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Sample indices: $SAMPLE_INDICES"
echo "[INFO] Task file: $TASK_FILE ($TASK_COUNT tasks; array $ARRAY_SPEC)"
echo "[INFO] Smoke task id: $SMOKE_TASK_ID"
echo "[INFO] Training dependency request: ${TRAINING_DEPENDENCY:-none}"
echo "[INFO] Training dependency resolved: $RESOLVED_TRAINING_DEPENDENCY"

extract_id="$EXTRACT_JOB_ID"
if [[ -z "$extract_id" ]]; then
  extract_id="$(active_job_id_by_name "SpatialStackProbeExtract")"
fi
if [[ -n "$extract_id" ]]; then
  echo "[INFO] Reusing active extraction_job_id=$extract_id"
else
  extract_raw="$(submit "${dependency_args[@]}" \
    --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES" \
    scripts/probing/slurm_extract_depth_probe_cut3r_spatialstack.sbatch)"
  extract_id="$(job_id_only "$extract_raw")"
fi
echo "[INFO] extraction_job_id=$extract_id"

smoke_raw="$(submit --dependency="afterok:$extract_id" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",SMOKE_TASK_ID="$SMOKE_TASK_ID" \
  scripts/probing/slurm_smoke_cut3r_spatialstack_probe.sbatch)"
smoke_id="$(job_id_only "$smoke_raw")"
echo "[INFO] smoke_job_id=$smoke_id"

official_dependency="afterok:$smoke_id"
official_depth_raw="$(submit --dependency="$official_dependency" \
  --job-name=DepthProbeSpatialStack \
  --array="$ARRAY_SPEC" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",PROBE_SUBDIR=depth_probes_scannet \
  scripts/probing/slurm_train_depth_probe_scannet_array.sbatch)"
official_depth_id="$(job_id_only "$official_depth_raw")"
echo "[INFO] official_depth_job_id=$official_depth_id"

official_sem_raw="$(submit --dependency="$official_dependency" \
  --job-name=SemProbeSpatialStack \
  --array="$ARRAY_SPEC" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",PROBE_SUBDIR=semantic_probes_scannet \
  scripts/probing/slurm_train_semantic_probe_scannet_array.sbatch)"
official_sem_id="$(job_id_only "$official_sem_raw")"
echo "[INFO] official_semantic_job_id=$official_sem_id"

aggregate_raw="$(submit --dependency="afterok:$official_depth_id:$official_sem_id" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" \
  scripts/probing/slurm_aggregate_cut3r_spatialstack_probe.sbatch)"
aggregate_id="$(job_id_only "$aggregate_raw")"
echo "[INFO] aggregate_job_id=$aggregate_id"
