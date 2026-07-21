# CUT3R SpatialStack Cross-Attn Summary

Implemented an opt-in `cut3r_spatialstack_fusion_type="cross_attn"` mode alongside the existing additive SpatialStack path.

The default remains `cut3r_spatialstack_fusion_type="add"`, preserving the dense residual behavior. In cross-attn mode, the merger reuses the existing CUT3R sidecar parsing, frame-order validation, visual-token metadata validation, square-grid resizing, and optional frame/token shuffle probes. It prepares aligned per-frame geometry token payloads for each selected LLM layer instead of dense residual tensors.

At each selected decoder layer, `LlavaQwenModel.forward()` applies a per-layer cross-attn block before the decoder block. Only visual token positions from `visual_metadata["visual_token_indices"]` are updated. Text, answer, padding, newline, special, and camera-prefix positions are never selected for updates because the existing metadata validation rejects overlap. Cached decode skips the update when `past_key_values_length > 0`.

Cross-attn uses:

- visual hidden-state LayerNorm and query projection
- geometry token LayerNorm and K/V projections
- multi-head scaled dot-product cross-attention
- output projection
- optional zero-init output projection, enabled by default

Default mapping remains CUT3R `6,9,12` to LLM `0,1,2`. Default alignment is same-frame only. All-frame attention is available only behind `cut3r_spatialstack_cross_attn_same_frame_only=False`.

New model args/config fields:

- `cut3r_spatialstack_fusion_type`
- `cut3r_spatialstack_cross_attn_heads`
- `cut3r_spatialstack_cross_attn_dropout`
- `cut3r_spatialstack_cross_attn_zero_init`
- `cut3r_spatialstack_cross_attn_same_frame_only`

Assumptions and unsupported cases:

- CUT3R sidecars still need pre-extracted `patch_tokens`; online CUT3R is unchanged and not used.
- 3D RoPE, reliability weighting, null tokens, global layout tokens, auxiliary losses, and pre-LLM fusion remain out of scope for this ablation.
- Ambiguous frame order still fails clearly unless sidecar `frame_indices`/`frame_order` can align visual frames by id.

Experiment script:

- `train_cut3r_spatialstack_cross_attn_dec6_9_12_llm0_1_2_noaux.sh`

Recommended smoke sequence:

- `DRY_RUN_PRINT_ARGS=True bash train_cut3r_spatialstack_cross_attn_dec6_9_12_llm0_1_2_noaux.sh`
- `pytest tests/test_cut3r_spatialstack.py`
- `MAX_STEPS=1` smoke training run before a full run
