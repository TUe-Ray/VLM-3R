#!/usr/bin/env bash
#SBATCH --job-name=RawSigLIPCut3R
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
  echo "Submit with sbatch; raw feature training is not permitted on the login node." >&2
  exit 2
fi
repo_dir="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR}"
alignment_report="${ALIGNMENT_REPORT:?Set ALIGNMENT_REPORT}"
python_bin="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"
fast_root="${FAST_DATA_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}"
cut3r_root="${CUT3R_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}"
siglip_cache="${SIGLIP_FEATURE_CACHE:-scannet=$fast_root/scannet/siglip_features_dec_m2;scannetpp=$fast_root/scannetpp/siglip_features_dec_m2;arkitscenes=$fast_root/arkitscenes/siglip_features_dec_m2}"
layer6_cache="${CUT3R_LAYER6_CACHE:-scannet=$cut3r_root/scannet/spatial_features_dec_6;scannetpp=$cut3r_root/scannetpp/spatial_features_dec_6;arkitscenes=$cut3r_root/arkitscenes/spatial_features_dec_6}"
layer9_cache="${CUT3R_LAYER9_CACHE:-scannet=$cut3r_root/scannet/spatial_features_dec_9;scannetpp=$cut3r_root/scannetpp/spatial_features_dec_9;arkitscenes=$cut3r_root/arkitscenes/spatial_features_dec_9}"
layer12_cache="${CUT3R_LAYER12_CACHE:-scannet=$fast_root/scannet/spatial_features;scannetpp=$fast_root/scannetpp/spatial_features;arkitscenes=$fast_root/arkitscenes/spatial_features}"
world_size="${TRAIN_WORLD_SIZE:-1}"

mkdir -p "$output_dir" "$repo_dir/logs/raw_siglip_cut3r"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --siglip_feature_cache "$siglip_cache"
  --cut3r_layer6_cache "$layer6_cache"
  --cut3r_layer9_cache "$layer9_cache"
  --cut3r_layer12_cache "$layer12_cache"
  --teacher_checkpoint "${TEACHER_CHECKPOINT:-/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963}"
  --output_dir "$output_dir"
  --alignment_report "$alignment_report"
  --predictor_type "${PREDICTOR_TYPE:?Set PREDICTOR_TYPE}"
  --epochs "${EPOCHS:-20}"
  --seed "${SEED:-42}"
  --validation_fraction "${VALIDATION_FRACTION:-0.1}"
  --lr "${LEARNING_RATE:-1e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --warmup_fraction "${WARMUP_FRACTION:-0.05}"
)
[[ "${AUTOCAST:-false}" != "true" ]] || args+=(--autocast)
[[ "${REQUIRE_EXPECTED_SPLIT:-false}" != "true" ]] || args+=(--require_expected_split)
[[ -z "${MAX_TRAIN_SAMPLES:-}" ]] || args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
[[ -z "${MAX_VALIDATION_SAMPLES:-}" ]] || args+=(--max_validation_samples "$MAX_VALIDATION_SAMPLES")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")

cleanup() {
  code=$?
  if (( code != 0 )) && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scancel "$SLURM_JOB_ID" || true
  fi
  exit "$code"
}
trap cleanup EXIT
if [[ "$world_size" == "1" ]]; then
  "$python_bin" "$repo_dir/scripts/train/train_raw_siglip_to_cut3r.py" "${args[@]}"
elif [[ "$world_size" == "4" ]]; then
  torchrun --standalone --nnodes=1 --nproc_per_node=4 "$repo_dir/scripts/train/train_raw_siglip_to_cut3r.py" "${args[@]}"
else
  echo "TRAIN_WORLD_SIZE must be 1 or 4, got $world_size" >&2
  exit 2
fi
