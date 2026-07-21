# CUT3R Camera Token + Point-Map Infra Summary

## Implemented Flags

- `use_cut3r_camera_tokens`
- `cut3r_camera_token_layer`
- `cut3r_camera_token_init_scale`
- `cut3r_camera_token_projector_type`
- `use_pointmap_supervision`
- `pointmap_head_source`
- `pointmap_point_map_key`
- `lambda_pointmap`
- `pointmap_coord_scale`
- `pointmap_smooth_l1_beta`
- `pointmap_detach_hidden`
- `pointmap_conf_threshold`
- `use_spatial_bridge_tokens`
- `num_spatial_bridge_tokens`
- `use_spatial_bridge_aux_loss`
- `lambda_spatial_bridge`

## Modified Paths

- `llava/model/cut3r_spatialstack.py`
- `llava/model/llava_arch.py`
- `llava/model/geometry/pointmap_supervision.py`
- `llava/model/geometry/__init__.py`
- `llava/model/geometry/bev_supervision.py`
- `llava/model/language_model/llava_qwen.py`
- `llava/train/train.py`
- `llava/train/llava_trainer.py`
- `tests/test_cut3r_spatialstack.py`
- `tests/test_pointmap_supervision.py`
- `train_cut3r_spatialstack.sh`
- `train_cut3r_spatialstack_camera_token_dec6.sh`
- `train_cut3r_spatialstack_pointmap_world_xyz.sh`

## Default Behavior Preservation

All new behavior is disabled by default. The existing SpatialStack script still keeps `tune_mm_mlp_adapter=False`, `tune_fusion_block=False`, geometry projection disabled, BEV/depth disabled, and LLM visual 3D RoPE disabled.

## Shape And Mask Checks

- Camera tokens are inserted once per frame before that frame's visual sequence.
- `visual_token_indices` remains patch-token-only.
- Camera positions are tracked in `camera_prefix_token_indices` and `cut3r_camera_token_indices`.
- SpatialStack, BEV, depth, and point-map validators reject overlap between patch visual indices and camera/bridge/text/newline/padding tokens.
- Point-map supervision builds `[B, N_patch, 3]` world/reference-frame xyz targets only.
- Spatial bridge tokens are inserted before the first supervised answer token and receive `IGNORE_INDEX` labels.

## Known Risks

- Camera-token mode requires CUT3R layered sidecars containing `camera_tokens`; older patch-only sidecars fail loudly.
- Point-map supervision requires world/reference-frame sidecars with `point_maps_ref` or `pts3d_in_other_view`; raw tensors and camera-frame keys are rejected.
- Full `MAX_STEPS=1` training was not run in this environment because Torch/PyTest are not installed.

## Verification

- Passed `python -m py_compile` for touched Python modules.
- Passed `bash -n` for the SpatialStack baseline and new Design 1/2 scripts.
- `DRY_RUN_PRINT_ARGS=True` confirmed Design 1 emits camera-token flags and Design 2 emits point-map plus geometry-sidecar flags.
- `pytest` and direct Torch smoke tests could not run locally: `No module named pytest` and `No module named torch`.

## Recommended First Full Run

Run Design 1 first:

```bash
MAX_STEPS=1 DRY_RUN_PRINT_ARGS=False bash train_cut3r_spatialstack_camera_token_dec6.sh
```

Then run Design 2 for a short loss-ratio inspection:

```bash
MAX_STEPS=200 bash train_cut3r_spatialstack_pointmap_world_xyz.sh
```
