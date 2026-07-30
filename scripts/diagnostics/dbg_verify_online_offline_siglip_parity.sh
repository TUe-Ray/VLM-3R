#!/usr/bin/env bash
#SBATCH --job-name=DBG_RawSigLIPParity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output=logs/raw_siglip_cut3r/%x_%j.out
#SBATCH --error=logs/raw_siglip_cut3r/%x_%j.err
set -euo pipefail

repo_dir="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cached="${CACHED_FEATURE:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/siglip_features_dec_m2/scene0000_00.pt}"
done="${SIGLIP_DONE:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/siglip_features_dec_m2/scene0000_00.pt.done.json}"
video="${VIDEO:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/videos/scene0000_00.mp4}"
model="${SIGLIP_MODEL:-/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/siglip-so400m-patch14-384}"
output="${OUTPUT:-$repo_dir/outputs/raw_siglip_cut3r_distillation_20260730/alignment/online_offline_siglip_parity.json}"
python_bin="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"

[[ -f "$cached" && -f "$done" && -f "$video" ]] || { echo "Missing cache, done record, or video" >&2; exit 2; }
[[ -d "$model" ]] || { echo "SigLIP model unavailable: $model" >&2; exit 2; }
mkdir -p "$(dirname "$output")" "$repo_dir/logs/raw_siglip_cut3r"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
"$python_bin" "$repo_dir/scripts/diagnostics/verify_online_offline_siglip_parity.py" \
  --cached_feature "$cached" --video "$video" --siglip_model "$model" \
  --siglip_done "$done" --output "$output" --device cuda
