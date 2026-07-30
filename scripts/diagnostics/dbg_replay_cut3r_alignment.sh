#!/usr/bin/env bash
#SBATCH --job-name=DBG_RawCut3RReplay
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
root="${OUTPUT_ROOT:-$repo_dir/outputs/raw_siglip_cut3r_distillation_20260730/alignment/cut3r_replay}"
video="${VIDEO:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/videos/scene0000_00.mp4}"
legacy="${LEGACY_SIDECAR:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features/scannet/spatial_features_dec_6/scene0000_00.pt}"
siglip_done="${SIGLIP_DONE:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r/scannet/siglip_features_dec_m2/scene0000_00.pt.done.json}"
base="${BASE_ALIGNMENT:-$repo_dir/outputs/raw_siglip_cut3r_distillation_20260730/alignment/alignment_evidence_v2.json}"
weights="${CUT3R_WEIGHTS:-/leonardo/home/userexternal/shuang00/VLM-3R/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}"
python_bin="${PYTHON_BIN:-/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python}"
mkdir -p "$root" "$repo_dir/logs/raw_siglip_cut3r"
export PYTHONPATH="$repo_dir/third_party/CUT3R:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$repo_dir/scripts/extraction/extract_cut3r_layer_features.py" \
  --input-file "$video" --output-root "$root" --layers 6 --cut3r-weights-path "$weights" \
  --processor-config-path "$repo_dir/processor_config.json" --gpu-ids 0 --batch-size 1 \
  --frames-upbound 32 --video-fps 1 --precision fp16
video_name="$(basename "${video%.*}")"
replayed="$root/spatial_features_dec_6/${video_name}.pt"
"$python_bin" "$repo_dir/scripts/diagnostics/verify_cut3r_sidecar_replay.py" \
  --legacy "$legacy" --replayed "$replayed" --siglip_done "$siglip_done" --base_alignment "$base" \
  --output "$root/replay_comparison.json" \
  --training_alignment_output "$repo_dir/outputs/raw_siglip_cut3r_distillation_20260730/alignment/training_alignment_report.json"
