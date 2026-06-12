# CUT3R SpatialStack Pre-Aggregation Ablation Summary

## Implemented config flags

- `cut3r_spatialstack_preagg_enable: bool = False`
- `cut3r_spatialstack_preagg_layers: str = "6,9,12"`
- `cut3r_spatialstack_preagg_type: str = "weighted_sum"`
- `cut3r_spatialstack_preagg_projector_sharing: str = "shared"`
- `cut3r_spatialstack_preagg_log_weights: bool = True`
- `cut3r_spatialstack_preagg_output_layer_key: str = "preagg"`
- `cut3r_spatialstack_preagg_use_layer_gamma: bool = True`
- `cut3r_spatialstack_preagg_layer_gamma_init: float = 1.0`

Default behavior is preserved: when `cut3r_spatialstack_preagg_enable=False`, SpatialStack still requires one-to-one `cut3r_spatialstack_layers` and `cut3r_spatialstack_llm_layers`, and still uses the existing per-layer `token_mlp` or `merge_mlp` branches.

## Experiment families

- `cut3r_spatialstack_preagg_wsum_sharedproj`
- `cut3r_spatialstack_preagg_concatlin_sharedproj`
- `cut3r_spatialstack_preagg_best_layerproj`

## Wrapper scripts

- `train_cut3r_spatialstack_preagg_wsum_sharedproj.sh`
- `train_cut3r_spatialstack_preagg_concatlin_sharedproj.sh`
- `train_cut3r_spatialstack_preagg_best_layerproj.sh`

## Eval scripts

- `eval_vlm3r_cut3r_spatialstack_preagg_wsum_sharedproj_vsibench.sh`
- `eval_vlm3r_cut3r_spatialstack_preagg_concatlin_sharedproj_vsibench.sh`
- `eval_vlm3r_cut3r_spatialstack_preagg_best_layerproj_vsibench.sh`

The generic `eval_vlm3r_cut3r_spatialstack_vsibench.sh` is kept as the baseline SpatialStack eval path. The pre-aggregation eval scripts create a temporary runtime checkpoint config with the corresponding pre-agg flags, then delegate to the generic eval script.

Target LLM layers are controlled by:

```bash
MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=0,1,2
```

or:

```bash
MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=1,2,3
```

The wrappers default to `1,2,3` and append sanitized layer suffixes such as `llm0_1_2` or `llm1_2_3` to the run name.

## Smoke test results

- Compile check passed:
  `python -m py_compile llava/model/cut3r_spatialstack.py llava/train/train.py thinking-in-space/lmms_eval/models/vlm_3r.py`
- Dry-run wrapper checks passed for all three wrappers with both `MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=1,2,3` and `0,1,2`.
- Direct tiny forward/backward smoke passed for:
  - weighted sum + shared projector with `1,2,3` and `0,1,2`
  - concat linear + shared projector with `1,2,3` and `0,1,2`
  - weighted sum + layer-specific projectors with `1,2,3` and `0,1,2`
  - concat linear + layer-specific projectors with `0,1,2`
- Smoke verified exact residual dict keys, residual shape `[batch, seq_len, hidden_size]`, finite loss, finite aggregator gradients, shared projector reuse, and distinct layer-specific projector instances.

`pytest` is not installed in this environment, so the new `tests/test_cut3r_spatialstack.py` tests were not run through pytest here. The direct Python smoke covered the new pre-aggregation behavior.

## Known assumptions

- Selected pre-aggregation source layers must all exist in the CUT3R sidecar.
- Selected source tensors must have identical `[frames, tokens, C]` shape before aggregation.
- Pre-aggregation is implemented for add-fusion SpatialStack, not the cross-attention SpatialStack variant.
- Existing visual metadata alignment is reused: `visual_token_indices`, `visual_frame_ids`, `frame_order`, optional `visual_grid_shapes`, and sidecar `frame_indices/frame_order`.

## Recommended next command

```bash
DRY_RUN_PRINT_ARGS=True MODEL_CUT3R_SPATIALSTACK_LLM_LAYERS=1,2,3 ./train_cut3r_spatialstack_preagg_wsum_sharedproj.sh
```

After confirming the printed config, launch E1 and E2 with the same target layer setting before choosing the aggregator for `preagg_best_layerproj`.
