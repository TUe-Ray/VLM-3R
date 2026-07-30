#!/usr/bin/env bash
# Submit all raw-feature experiment chains.  This script submits; it never
# cancels or changes any pre-existing residual-predictor jobs.
set -euo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
root="${RUN_ROOT:-$repo_dir/outputs/raw_siglip_cut3r_distillation_20260730}"
alignment_report="${ALIGNMENT_REPORT:-$root/alignment_report.json}"
online_parity_report="${ONLINE_PARITY_REPORT:-$root/alignment/online_offline_siglip_parity.json}"
train_wrapper="$repo_dir/train_raw_siglip_to_cut3r.sh"
eval_wrapper="$repo_dir/eval_raw_siglip_cut3r_vsibench.sh"
[[ -f "$alignment_report" ]] || { echo "Alignment report missing: $alignment_report" >&2; exit 2; }
python_bin="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"
"$python_bin" -c 'import json,sys; r=json.load(open(sys.argv[1])); ok=r.get("status") != "ALIGNMENT_UNRESOLVED" and r.get("frame_identity_evidence",{}).get("status") == "verified"; raise SystemExit(0 if ok else "Alignment report does not verify paired frame order; refusing submission.")' "$alignment_report"
[[ -f "$online_parity_report" ]] || { echo "Online/offline SigLIP parity report missing: $online_parity_report" >&2; exit 2; }
"$python_bin" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes") else "Online/offline SigLIP parity gate failed.")' "$online_parity_report"
mkdir -p "$root" "$repo_dir/logs/raw_siglip_cut3r"

submit() { sbatch --parsable "$@"; }

chain() {
  local slug="$1" predictor="$2" world="$3" gpus="$4" memory="$5" hours="$6"
  local output="$root/$slug" smoke_output="$output/smoke" eval_smoke="$output/eval_smoke"
  local common="REPO_DIR=$repo_dir,ALIGNMENT_REPORT=$alignment_report,PREDICTOR_TYPE=$predictor,TRAIN_WORLD_SIZE=$world,SEED=42,LEARNING_RATE=1e-4,WEIGHT_DECAY=0.01,WARMUP_FRACTION=0.05"
  local smoke train eval_job primary alternate
  smoke=$(submit --job-name="SMOKE_${slug}" --partition=boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 --mem="$memory" --gpus-per-node="$gpus" --ntasks=1 --export="ALL,$common,OUTPUT_DIR=$smoke_output,EPOCHS=1,MAX_TRAIN_SAMPLES=8,MAX_VALIDATION_SAMPLES=4" "$train_wrapper")
  train=$(submit --dependency="afterok:$smoke" --job-name="$slug" --partition=boost_usr_prod --qos=normal --time="$hours" --mem="$memory" --gpus-per-node="$gpus" --ntasks=1 --export="ALL,$common,OUTPUT_DIR=$output,EPOCHS=20,REQUIRE_EXPECTED_SPLIT=true" "$train_wrapper")
  eval_job=$(submit --dependency="afterok:$train" --job-name="SMOKE_Eval_${slug}" --partition=boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 --mem="$memory" --gpus-per-node=1 --ntasks=1 --export="ALL,REPO_DIR=$repo_dir,ONLINE_PARITY_REPORT=$online_parity_report,PREDICTOR_CHECKPOINT=$output/best_validation_residual_relative_l2.pt,OUTPUT_PATH=$eval_smoke,RUN_NAME=${slug}_scored_smoke,RESIDUAL_PREDICTOR_TYPE=$predictor,LIMIT=4,NUM_PROCESSES=1" "$eval_wrapper")
  primary=$(submit --dependency="afterok:$eval_job" --job-name="Eval_${slug}_RelL2" --partition=boost_usr_prod --qos=normal --time=12:00:00 --mem="$memory" --gpus-per-node=1 --ntasks=1 --export="ALL,REPO_DIR=$repo_dir,ONLINE_PARITY_REPORT=$online_parity_report,PREDICTOR_CHECKPOINT=$output/best_validation_residual_relative_l2.pt,OUTPUT_PATH=$output/eval_primary,RUN_NAME=${slug}_primary,RESIDUAL_PREDICTOR_TYPE=$predictor,NUM_PROCESSES=1" "$eval_wrapper")
  alternate=$(submit --dependency="afterok:$eval_job" --job-name="Eval_${slug}_Cosine" --partition=boost_usr_prod --qos=normal --time=12:00:00 --mem="$memory" --gpus-per-node=1 --ntasks=1 --export="ALL,REPO_DIR=$repo_dir,ONLINE_PARITY_REPORT=$online_parity_report,PREDICTOR_CHECKPOINT=$output/best_validation_residual_cosine.pt,DEDUPLICATE_AGAINST=$output/best_validation_residual_relative_l2.pt,OUTPUT_PATH=$output/eval_alternate,RUN_NAME=${slug}_alternate,RESIDUAL_PREDICTOR_TYPE=$predictor,NUM_PROCESSES=1" "$eval_wrapper")
  printf '%s smoke=%s train=%s eval_smoke=%s primary=%s alternate=%s\n' "$slug" "$smoke" "$train" "$eval_job" "$primary" "$alternate"
}

# The smoke resource values are deliberately conservative.  The official
# request is parameterized so measured smoke MaxRSS can be substituted before
# this driver is run on a production allocation.
chain raw_cut3r_token_mlp_1gpu raw_cut3r_token_mlp 1 1 "32G" "10:00:00"
chain raw_cut3r_token_mlp_4gpu raw_cut3r_token_mlp 4 4 "64G" "10:00:00"
chain raw_cut3r_spatial_temporal_1gpu raw_cut3r_spatial_temporal 1 1 "64G" "12:00:00"
chain raw_cut3r_spatial_temporal_4gpu raw_cut3r_spatial_temporal 4 4 "128G" "12:00:00"
