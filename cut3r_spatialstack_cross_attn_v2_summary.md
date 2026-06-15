# CUT3R SpatialStack Cross-Attention V2

## Modified files

- `llava/model/cut3r_spatialstack.py`
- `llava/model/language_model/llava_qwen.py`
- `llava/train/train.py`
- `train_cut3r_spatialstack.sh`
- `train_cut3r_spatialstack_cross_attn_v2_resize_cam_gamma.sh`
- `train_cut3r_spatialstack_cross_attn_v2_merge_cam_gamma.sh`
- `tests/test_cut3r_spatialstack.py`

## Architecture

`cut3r_spatialstack_fusion_type=cross_attn_v2` adds an opt-in CUT3R SpatialStack block that builds per-frame geometry memory from:

- one projected CUT3R camera token
- aligned CUT3R patch tokens

The block uses Qwen RMSNorm when available, otherwise LayerNorm, PyTorch `nn.MultiheadAttention`, additive 2D sin-cos position embeddings, an FFN residual, and learnable positive residual scales for attention and MLP branches. It returns a delta that is added only at visual token positions by the existing LLM-layer injection path.

## Config flags

- `cut3r_spatialstack_fusion_type=cross_attn_v2`
- `cut3r_spatialstack_cross_attn_impl=torch_mha`
- `cut3r_spatialstack_cross_attn_patch_align=resize|merge`
- `cut3r_spatialstack_cross_attn_use_camera_tokens=True`
- `cut3r_spatialstack_require_camera_tokens=True`
- `cut3r_spatialstack_cross_attn_use_mlp=True`
- `cut3r_spatialstack_cross_attn_norm_type=qwen_rmsnorm`
- `cut3r_spatialstack_cross_attn_pos_embed=sincos2d`
- `cut3r_spatialstack_cross_attn_zero_init=False`
- `cut3r_spatialstack_cross_attn_gamma_attn_init=0.05`
- `cut3r_spatialstack_cross_attn_gamma_mlp_init=0.05`
- `cut3r_spatialstack_cross_attn_gamma_learnable=True`
- `cut3r_spatialstack_cross_attn_dropout=0.0`

## Smoke commands

Dry-run resize:

```bash
DRY_RUN_PRINT_ARGS=True bash train_cut3r_spatialstack_cross_attn_v2_resize_cam_gamma.sh
```

Dry-run merge:

```bash
DRY_RUN_PRINT_ARGS=True bash train_cut3r_spatialstack_cross_attn_v2_merge_cam_gamma.sh
```

One-step sanity run on the target training environment:

```bash
MAX_STEPS=1 MODEL_TORCH_COMPILE=False bash train_cut3r_spatialstack_cross_attn_v2_resize_cam_gamma.sh
```

## Verification

Local compile checks passed for the edited Python files. Dry-run argument printing passed for both V2 scripts. Full pytest/tiny torch runtime checks could not run in this local environment because the active Python lacks `pytest` and `torch`.

## Assumptions and risks

- V2 requires same-frame cross-attention because each memory contains one camera token for that frame.
- Required camera-token mode expects sidecars with `cut3r_dec_layers[layer]["camera_tokens"]`.
- `merge` alignment reuses existing `merge_frame_grid()` semantics.
- The default additive and old `cross_attn` paths are preserved unless `cross_attn_v2` is explicitly selected.
