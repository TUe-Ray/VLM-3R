#!/bin/bash
#SBATCH --account=EUHPC_D32_006
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=cut3r_gauge_context
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err

set -euo pipefail
module purge
module load profile/deeplrn
module load cuda/12.1
REPO=${REPO:-/leonardo/home/userexternal/shuang00/VLM-3R}
TOKEN_ROOT=${TOKEN_ROOT:-/leonardo_work/EUHPC_D32_006/VLM_3R_cut3r_min2N4_features}
FINAL_TOKEN_ROOT=${FINAL_TOKEN_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}
POINTMAP_ROOT=${POINTMAP_ROOT:-/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_pointmaps}
VIDEO_ROOT=${VIDEO_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/data/vlm3r}
CONTEXT_ROOT=${CONTEXT_ROOT:-/leonardo_scratch/large/userexternal/shuang00/VLM_3R_cut3r_gauge_context}
MANIFEST_DIR=${MANIFEST_DIR:-$REPO/outputs/gauge_translation_official_manifest}
ALIGNMENT_DIR=${ALIGNMENT_DIR:-$REPO/outputs/gauge_translation_official_alignment}
CHECKPOINT=${CHECKPOINT:-$REPO/third_party/CUT3R/src/cut3r_512_dpt_4_64.pth}
VSI_ROOT=${VSI_ROOT:-/leonardo_scratch/fast/EUHPC_D32_006/vsibench}
cd "$REPO"
: "${SMOKE_ACCEPTANCE:?Set SMOKE_ACCEPTANCE to a passing smoke_acceptance.json artifact}"
python -c 'import json,sys; payload=json.load(open(sys.argv[1])); assert payload.get("passed") is True, "smoke acceptance did not pass"' "$SMOKE_ACCEPTANCE"
conda run --no-capture-output -n vlm3r python scripts/gauge_translation/build_manifest.py \
  --token-root "$TOKEN_ROOT" --final-token-root "$FINAL_TOKEN_ROOT" --pointmap-root "$POINTMAP_ROOT" \
  --video-root "$VIDEO_ROOT" --context-root "$CONTEXT_ROOT" --output-dir "$MANIFEST_DIR" --mode source \
  --exclude "$VSI_ROOT/test_pruned.parquet" --exclude "$VSI_ROOT/test_debiased.parquet"
srun --kill-on-bad-exit=1 --wait=30 conda run --no-capture-output -n vlm3r python -u \
  scripts/extraction/extract_cut3r_gauge_context.py --manifest "$MANIFEST_DIR/all.jsonl" \
  --checkpoint "$CHECKPOINT" --alignment-report-dir "$ALIGNMENT_DIR" --precision fp16 --frames-upbound 32
conda run --no-capture-output -n vlm3r python scripts/gauge_translation/build_manifest.py \
  --token-root "$TOKEN_ROOT" --final-token-root "$FINAL_TOKEN_ROOT" --pointmap-root "$POINTMAP_ROOT" \
  --video-root "$VIDEO_ROOT" --context-root "$CONTEXT_ROOT" --output-dir "$MANIFEST_DIR" --mode official \
  --require-context --exclude "$VSI_ROOT/test_pruned.parquet" --exclude "$VSI_ROOT/test_debiased.parquet"
