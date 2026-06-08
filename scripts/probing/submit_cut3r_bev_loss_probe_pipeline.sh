#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/leonardo/home/userexternal/shuang00/VLM-3R}"
cd "$REPO_ROOT"

MODEL_LABEL="${MODEL_LABEL:-cut3r_bev_loss_8n32g_42837152}"
TRAINING_DEPENDENCY="${TRAINING_DEPENDENCY:-none}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/scannet_semantic_full}"
SAMPLE_INDICES="${SAMPLE_INDICES:-$OUTPUT_ROOT/semantic_probe_scannet_final_usable_sample_indices.json}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/probing/scannet_cut3r_bev_loss_probe_tasks.tsv}"
FEATURE_ROOT="${FEATURE_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
EXTRACT_JOB_ID="${EXTRACT_JOB_ID:-}"
SMOKE_TASK_ID="${SMOKE_TASK_ID:-0}"
SMOKE_TAG="${SMOKE_TAG:-$MODEL_LABEL}"
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

dependency_args=()
if [[ -n "$TRAINING_DEPENDENCY" && "$TRAINING_DEPENDENCY" != "none" ]]; then
  dependency_args=(--dependency="$TRAINING_DEPENDENCY")
fi

echo "[INFO] Probe model: $MODEL_LABEL"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Sample indices: $SAMPLE_INDICES"
echo "[INFO] Task file: $TASK_FILE ($TASK_COUNT tasks; array $ARRAY_SPEC)"
echo "[INFO] Feature root: $FEATURE_ROOT"
echo "[INFO] Smoke task id: $SMOKE_TASK_ID"
echo "[INFO] Training dependency: ${TRAINING_DEPENDENCY:-none}"

extract_id="$EXTRACT_JOB_ID"
if [[ -z "$extract_id" ]]; then
  extract_id="$(active_job_id_by_name "BEVLossProbeExtract")"
fi
if [[ -n "$extract_id" ]]; then
  echo "[INFO] Reusing active extraction_job_id=$extract_id"
else
  extract_raw="$(submit "${dependency_args[@]}" \
    --job-name=BEVLossProbeExtract \
    --export=ALL,MODEL_LABEL="$MODEL_LABEL",OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",FEATURE_ROOT="$FEATURE_ROOT" \
    scripts/probing/slurm_extract_depth_probe.sbatch)"
  extract_id="$(job_id_only "$extract_raw")"
fi
echo "[INFO] extraction_job_id=$extract_id"

smoke_raw="$(submit --dependency="afterok:$extract_id" \
  --job-name=SMOKE_BEVLossProbe \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",SMOKE_TASK_ID="$SMOKE_TASK_ID",SMOKE_TAG="$SMOKE_TAG" \
  scripts/probing/slurm_smoke_scannet_probe.sbatch)"
smoke_id="$(job_id_only "$smoke_raw")"
echo "[INFO] smoke_job_id=$smoke_id"

official_dependency="afterok:$smoke_id"
official_depth_raw="$(submit --dependency="$official_dependency" \
  --job-name=DepthProbeBEVLoss \
  --array="$ARRAY_SPEC" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",PROBE_SUBDIR=depth_probes_scannet \
  scripts/probing/slurm_train_depth_probe_scannet_array.sbatch)"
official_depth_id="$(job_id_only "$official_depth_raw")"
echo "[INFO] official_depth_job_id=$official_depth_id"

official_sem_raw="$(submit --dependency="$official_dependency" \
  --job-name=SemProbeBEVLoss \
  --array="$ARRAY_SPEC" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",SAMPLE_INDICES="$SAMPLE_INDICES",TASK_FILE="$TASK_FILE",PROBE_SUBDIR=semantic_probes_scannet \
  scripts/probing/slurm_train_semantic_probe_scannet_array.sbatch)"
official_sem_id="$(job_id_only "$official_sem_raw")"
echo "[INFO] official_semantic_job_id=$official_sem_id"

aggregate_raw="$(submit --dependency="afterok:$official_depth_id:$official_sem_id" \
  --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT" \
  scripts/probing/slurm_aggregate_cut3r_bev_loss_probe.sbatch)"
aggregate_id="$(job_id_only "$aggregate_raw")"
echo "[INFO] aggregate_job_id=$aggregate_id"
