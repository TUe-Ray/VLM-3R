#!/usr/bin/env bash
# Submit the approved single-GPU residual expansion chains without DDP.
set -euo pipefail
repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
run_root="${RUN_ROOT:-$repo_dir/outputs/siglip_residual_expansion_20260730}"
train_wrapper="$repo_dir/train_siglip_to_spatialstack_residual_temporal.sh"
eval_wrapper="$repo_dir/eval_siglip_predicted_residual_temporal.sh"
mkdir -p "$run_root" "$repo_dir/logs/train" "$repo_dir/logs/eval"

submit() { sbatch --parsable "$@"; }
chain() {
  local slug="$1" type="$2" epochs="$3" hours="$4" hidden="$5" layers="$6" heads="$7" ffn="$8" extra="$9"
  local smoke_out="$run_root/${slug}_smoke" out="$run_root/$slug"
  local common="REPO_DIR=$repo_dir,OUTPUT_DIR=$smoke_out,RESIDUAL_PREDICTOR_TYPE=$type,EPOCHS=1,BATCH_SIZE=1,MAX_TRAIN_SAMPLES=8,MAX_VALIDATION_SAMPLES=4,RUN_PARITY_CHECK=true,STARTUP_CHECK_SAMPLES=8,TEMPORAL_HIDDEN_DIM=$hidden,TEMPORAL_NUM_LAYERS=$layers,TEMPORAL_NUM_HEADS=$heads,TEMPORAL_FFN_DIM=$ffn,$extra"
  local smoke
  smoke=$(submit --job-name="SMOKE_${slug}" --qos=boost_qos_dbg --time=00:30:00 --mem=32G --gpus-per-node=1 --export="ALL,$common" "$train_wrapper")
  local official_env="REPO_DIR=$repo_dir,OUTPUT_DIR=$out,RESIDUAL_PREDICTOR_TYPE=$type,EPOCHS=$epochs,BATCH_SIZE=1,STARTUP_CHECK_SAMPLES=8,TEMPORAL_HIDDEN_DIM=$hidden,TEMPORAL_NUM_LAYERS=$layers,TEMPORAL_NUM_HEADS=$heads,TEMPORAL_FFN_DIM=$ffn,WANDB=true,WANDB_NAME=$slug,$extra"
  local train
  train=$(submit --dependency="afterok:$smoke" --job-name="SigLIPResidual_${slug}" --time="$hours" --mem=32G --gpus-per-node=1 --export="ALL,$official_env" "$train_wrapper")
  local eval_smoke_out="$run_root/eval_smoke_${slug}"
  local eval_smoke
  eval_smoke=$(submit --dependency="afterok:$train" --job-name="SMOKE_Eval_${slug}" --qos=boost_qos_dbg --time=00:30:00 --mem=32G --gpus-per-node=1 --export="ALL,PREDICTOR_CHECKPOINT=$out/best_validation_relative_l2.pt,OUTPUT_PATH=$eval_smoke_out,RUN_NAME=${slug}_scored_smoke,LIMIT=4,NUM_PROCESSES=1,RESIDUAL_PREDICTOR_TYPE=auto" "$eval_wrapper")
  local primary
  primary=$(submit --dependency="afterok:$eval_smoke" --job-name="Eval_${slug}_RelL2" --time=12:00:00 --mem=32G --gpus-per-node=1 --export="ALL,PREDICTOR_CHECKPOINT=$out/best_validation_relative_l2.pt,OUTPUT_PATH=$run_root/eval_${slug}_best_relative_l2,RUN_NAME=${slug}_best_relative_l2,NUM_PROCESSES=1,RESIDUAL_PREDICTOR_TYPE=auto" "$eval_wrapper")
  local alternate_wrap
  alternate_wrap="set -euo pipefail; primary='$out/best_validation_relative_l2.pt'; alternate='$out/best_validation_cosine.pt'; test -f \"\$primary\" && test -f \"\$alternate\"; primary_hash=\$(/leonardo_work/EUHPC_D32_006/miniconda3/envs/vlm3r/bin/python - \"\$primary\" \"\$alternate\" <<'PY'\nimport sys,torch\nfrom llava.model.siglip_spatialstack_residual import predictor_state_sha256\nfor p in sys.argv[1:]:\n print(predictor_state_sha256(torch.load(p,map_location='cpu',weights_only=False)['predictor']))\nPY\n); set -- \$primary_hash; if [ \"\$1\" = \"\$2\" ]; then echo DEDUPLICATED; exit 0; fi; export PREDICTOR_CHECKPOINT=\"\$alternate\" OUTPUT_PATH='$run_root/eval_${slug}_best_cosine' RUN_NAME='${slug}_best_cosine' NUM_PROCESSES=1 RESIDUAL_PREDICTOR_TYPE=auto; exec bash '$eval_wrapper'"
  local alternate
  alternate=$(submit --dependency="afterok:$eval_smoke" --partition=boost_usr_prod --qos=normal --job-name="Eval_${slug}_Cosine" --time=12:00:00 --mem=32G --gpus-per-node=1 --wrap="$alternate_wrap")
  printf '%s\n' "${slug}: smoke=$smoke train=$train eval_smoke=$eval_smoke primary=$primary alternate=$alternate"
}

baseline_ckpt="${BASELINE_CKPT:-$repo_dir/outputs/official_siglip_residual_temporal_mem246g_b1_20260730_retry1/best_validation_relative_l2.pt}"
if [[ "${SKIP_TEMPORAL_CONT20:-false}" != "true" ]]; then
  chain temporal_cont20 temporal 20 10:00:00 512 2 8 2048 "INIT_CHECKPOINT=$baseline_ckpt,LEARNING_RATE=2e-5,MIN_LEARNING_RATE=2e-6,WARMUP_RATIO=0"
fi
chain temporal_mag_smooth05 temporal 5 04:00:00 512 2 8 2048 "INIT_CHECKPOINT=$baseline_ckpt,LEARNING_RATE=1e-5,SMOOTH_L1_WEIGHT=0.5"
chain temporal_mag_rel_norm temporal 5 04:00:00 512 2 8 2048 "INIT_CHECKPOINT=$baseline_ckpt,LEARNING_RATE=1e-5,SMOOTH_L1_WEIGHT=0,RELATIVE_L2_LOSS_WEIGHT=0.25,LOG_NORM_LOSS_WEIGHT=0.05"
chain temporal_h512_l2_scratch20 temporal 20 10:00:00 512 2 8 2048 "LEARNING_RATE=1e-4"
chain spatial_temporal_h512_l2 spatial_temporal 20 12:00:00 512 2 8 2048 "LEARNING_RATE=1e-4,SPATIAL_NUM_BLOCKS=2"
chain temporal_wide_h768_l2 temporal 20 12:00:00 768 2 12 3072 "LEARNING_RATE=1e-4"
chain temporal_deep_h512_l4 temporal 20 14:00:00 512 4 8 2048 "LEARNING_RATE=1e-4"
chain temporal_wide_deep_h768_l4 temporal 20 18:00:00 768 4 12 3072 "LEARNING_RATE=1e-4"
chain target_adapter_h512 target_adapter_temporal 20 14:00:00 512 2 8 2048 "LEARNING_RATE=1e-4,SHARED_TEMPORAL_LAYERS=1,ADAPTER_NUM_LAYERS=1"
chain layer_conditioned_h512 layer_conditioned_temporal 20 16:00:00 512 2 8 2048 "LEARNING_RATE=1e-4,CONDITIONED_DECODER_LAYERS=1"
chain spatial_temporal_wide_deep_adapter_h768 spatial_temporal_target_adapter 20 22:00:00 768 4 12 3072 "LEARNING_RATE=1e-4,SPATIAL_NUM_BLOCKS=2,ADAPTER_NUM_LAYERS=1"
