#!/usr/bin/env bash
#SBATCH --job-name=EvalRawSigLIPCut3R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --output=logs/raw_siglip_cut3r/%x_%j.out
#SBATCH --error=logs/raw_siglip_cut3r/%x_%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOGIN_NODE:-false}" != "true" ]]; then
  echo "Submit this GPU wrapper with sbatch." >&2
  exit 2
fi
repo_dir="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"
: "${PREDICTOR_CHECKPOINT:?Set PREDICTOR_CHECKPOINT}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH}"
: "${ONLINE_PARITY_REPORT:?Set ONLINE_PARITY_REPORT from verify_online_offline_siglip_parity.py}"
[[ -f "$PREDICTOR_CHECKPOINT" ]] || { echo "Predictor checkpoint not found: $PREDICTOR_CHECKPOINT" >&2; exit 1; }
[[ -f "$ONLINE_PARITY_REPORT" ]] || { echo "Online/offline SigLIP parity report not found: $ONLINE_PARITY_REPORT" >&2; exit 1; }
"$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r.get("passes") else "Online/offline SigLIP parity gate failed.")' "$ONLINE_PARITY_REPORT"
mkdir -p "$repo_dir/logs/raw_siglip_cut3r" "$OUTPUT_PATH"
export PRETRAINED_LOCAL="${PRETRAINED_LOCAL:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
export SPATIAL_FEATURES_ROOT=/dev/null
export CHECK_SPATIAL_SIDECARS=False
export RUN_NAME="${RUN_NAME:-raw_siglip_cut3r_predicted}"
checkpoint="$(readlink -f "$PREDICTOR_CHECKPOINT")"
if [[ -n "${DEDUPLICATE_AGAINST:-}" ]]; then
  [[ -f "$DEDUPLICATE_AGAINST" ]] || { echo "Dedup checkpoint not found: $DEDUPLICATE_AGAINST" >&2; exit 1; }
  hashes="$(PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'import sys,torch; from llava.model.raw_siglip_cut3r import raw_predictor_state_sha256; [print(raw_predictor_state_sha256(torch.load(path,map_location="cpu",weights_only=False)["predictor"])) for path in sys.argv[1:]]' "$checkpoint" "$DEDUPLICATE_AGAINST")"
  read -r candidate_hash reference_hash <<<"$hashes"
  if [[ "$candidate_hash" == "$reference_hash" ]]; then
    printf '{"status":"DEDUPLICATED","predictor_state_sha256":"%s","duplicate_of":"%s"}\n' "$candidate_hash" "$DEDUPLICATE_AGAINST" > "$OUTPUT_PATH/results.json"
    exit 0
  fi
fi
extra="${EXTRA_MODEL_ARGS:+$EXTRA_MODEL_ARGS,}use_predicted_spatialstack_residuals=true,residual_predictor_type=${RESIDUAL_PREDICTOR_TYPE:?Set RESIDUAL_PREDICTOR_TYPE},residual_predictor_checkpoint=$checkpoint,use_raw_siglip_cut3r_predictions=true,raw_cut3r_teacher_checkpoint=${RAW_CUT3R_TEACHER_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
export EXTRA_MODEL_ARGS="$extra"
exec bash "$repo_dir/eval_spatialstack_vsibench.sh"
