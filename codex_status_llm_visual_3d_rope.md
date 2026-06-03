# Codex Handoff: LLM Visual-Token 3D RoPE

Date: 2026-05-27

## Current State

Implementation is complete enough for the single-shot run. The realistic/max-size VSiBench dry-run passed after adding nonfinite-logit sanitization in eager Qwen2 attention. Full training has been submitted, and the three eval jobs are queued with `afterok` dependency on the training job.

Active training job:

- `42815671` — `llm_visual_3d_rope_spherical`
- Expected checkpoint/output dir:
  `/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/llm_visual_3d_rope_spherical_42815671`

Dependent eval jobs:

- `42815705` — A1 correct geometry, `alpha=1.0`
- `42815706` — A2 shuffled geometry, `alpha=1.0`
- `42815707` — A3 disabled 3D RoPE, `alpha=0.0`

Queue check used:

```bash
squeue -j 42815671,42815705,42815706,42815707 -o '%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R'
```

At submission time, training was pending on priority and evals were pending on dependency.

## Files Modified

- `llava/model/language_model/llm_visual_3d_rope.py`
  - New Qwen2 eager attention subclass with visual-visual-only 3D RoPE logit replacement.
  - Uses raw Q and repeated raw K for grouped-query attention.
  - Replaces/interpolates only the visual-visual logit submatrix before the causal mask.
  - Sanitizes nonfinite attention logits after the causal/padding mask is applied, then reapplies the mask, so future-token attention remains blocked.
  - Logs first/middle/last layer stats by default.

- `llava/model/language_model/llava_qwen.py`
  - Installs the Qwen2 eager attention patch before Qwen2 layers are constructed.
  - Adds `llm_geo_pos` / `llm_geo_mask` through forward and generation.
  - Uses per-decoder-layer metadata attachment with `finally` cleanup, including during gradient-checkpoint recomputation.
  - Avoids the older `return_visual_metadata` path unless spatial-rank/metadata is explicitly requested, so this does not force `output_hidden_states=True`.

- `llava/model/llava_arch.py`
  - Adds lightweight dense metadata path:
    - `llm_geo_pos: [B, seq_len, 3]`
    - `llm_geo_mask: [B, seq_len]`
  - Uses requested CUT3R source, default `point_maps_ref`, with no silent camera-coordinate fallback.
  - Aligns pooled point maps to final visual token indices.
  - Excludes text, padding, newline, prefix, and invalid geometry tokens.
  - Adds deterministic eval-only intra-sample token shuffle.

- `llava/train/train.py`
  - Adds `llm_visual_3d_rope_*` config flags.
  - Forces eager attention when enabled.

- `llava/train/llava_trainer.py`
  - Adds rank-0 throttled JSONL stats at `$output_dir/llm_visual_3d_rope_stats.jsonl`.

- `thinking-in-space/lmms_eval/models/vlm_3r.py`
  - Adds matching eval-time flags and JSONL stats.

- Scripts:
  - `train_geo_rope_fusion_cut3r.sh`
  - `eval_rope_fusion_cut3r.sh`
  - `train_llm_visual_3d_rope_spherical.sh`
  - `eval_llm_visual_3d_rope_correct.sh`
  - `eval_llm_visual_3d_rope_shuffle.sh`
  - `eval_llm_visual_3d_rope_disabled.sh`
  - `dry_run_llm_visual_3d_rope_vsibench.sh`
  - `scripts/test_llm_visual_3d_rope.py`

## Dry-Run Validation

Synthetic attention validation passed:

- `alpha0_equivalence_max_abs`: `0.0`
- alpha=1 visual-visual logit deltas:
  - `0.014977857714793721`
  - `0.015228907592961045`
  - `0.01018826199359581`
- shuffle assignment changed while preserving value distribution:
  `shuffle_assignment_delta_abs_sum=28.30087661743164`
- causal leakage check: `causal_mask_masked_attention_prob_max=0.0`
- generation cached decode skip reason: `non_prefill_or_cached_decode`
- post-softmax attention nonfinite counts: zero

Final realistic/max-size VSiBench dry-run:

- Job: `42813999`
- Stats file:
  `/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/dry_llm_visual_3d_rope_fix5/dry_llm_visual_3d_rope_vsibench_fix5_llm_visual_3d_rope_stats.jsonl`
- `seq_len`: `6768`
- visual tokens: `6272`
- valid geometry tokens: `6272`
- frames: `32`
- peak allocated GPU memory: `43549694464` bytes, about `40.57 GiB`
- aggregate `attention_delta_mean_abs`: `4.5653767050760985`
- aggregate `visual_visual_logits_delta_mean_abs`: `4.5653767050760985`
- geometry source: `point_maps_ref`
- newline/prefix/padding/text overlaps: all `0`
- geometry NaN/Inf counts: all `0`
- masked attention probability max: `0.0`
- post-softmax attention NaN/Inf counts: all `0`
- decode layers skipped with `non_prefill_or_cached_decode`

Numerical caveat:

- The realistic prefill produced nonfinite fp16 logits before softmax, so `attention_logits_sanitized=true` in logged prefill layers. The sanitizer then produced zero NaN/Inf probabilities and reapplied the causal/padding mask. This should be monitored during training in `llm_visual_3d_rope_stats.jsonl`.

## Expected Outputs To Collect Later

Training stats:

- `/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/llm_visual_3d_rope_spherical_42815671/llm_visual_3d_rope_stats.jsonl`

Eval outputs:

- `/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eval_llm_visual_3d_rope_correct_ckpt42815671`
- `/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eval_llm_visual_3d_rope_shuffle_ckpt42815671`
- `/leonardo_scratch/fast/EUHPC_D32_006/eval/logs/VLM3R/eval_llm_visual_3d_rope_disabled_ckpt42815671`

Final comparison variants:

- A1 correct geometry, `alpha=1.0`
- A2 shuffled geometry, `alpha=1.0`
- A3 disabled 3D RoPE, `alpha=0.0`

Success requires A1 to beat both A2 and A3, preferably in spatial categories.
