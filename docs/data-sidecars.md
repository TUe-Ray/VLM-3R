# Pi3X and VGGT Sidecars

Read this document before modifying Pi3X or VGGT decoded features, point maps,
their schemas, or their loader configuration. CUT3R sidecar guidance remains
in `AGENTS.md`.

## Pi3X decoded features

- Current Pi3X training sidecars are decoded-feature sidecars, not pre-sliced
  camera-token sidecars.
- Verified root:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `frames.decoded_features`, `frames.frame_idx`, and
  `meta.decoded_pos_template`, `meta.patch_start_idx`, `meta.num_frames`.
  `decoded_features` has shape `[F, T, 2048]`.
- Camera tokens must be computed at runtime from
  `pi3.camera_decoder(decoded_features, xpos=decoded_pos)`; do not use legacy
  flat `camera_tokens` Pi3X payloads.
- Use with:
  `SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features`,
  `SPATIAL_FEATURES_SUBDIR=.`, `MODEL_SPATIAL_TOWER=pi3x`, and
  `MODEL_SPATIAL_FEATURE_DIM=2048`.
- The archived wrapper
  `scripts/archived/old_files/old_bash/train/rope/train_geo_rope_fusion_cut3r_pi3x_pos.sh`
  keeps CUT3R as fusion/KV features while using Pi3X decoded features as the
  geometry provider via `GEOMETRY_SPATIAL_TOWER_TYPE=pi3x`,
  `GEOMETRY_SPATIAL_FEATURES_ROOT=/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_features`,
  and `GEOMETRY_SPATIAL_FEATURES_SUBDIR=.`.

## Pi3X point maps

- Pi3X point-map sidecars are world-space point maps decoded from the Pi3X
  decoded-feature root.
- Verified train/large root:
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_pi3x_pointmaps/{scannet,scannetpp,arkitscenes}/*.pt`.
- Verified VSI-Bench eval root:
  `/leonardo_work/EUHPC_D32_006/VLM_3R_pi3x_vsibench_eval_pointmaps/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `point_map` `[F,518,518,3]`, `camera_pose`
  `[F,4,4]`, `frame_idx`, and `meta`. `meta.coordinate_frame` is `world`; the
  schema is `pi3x_world_point_map_v1`.
- If consuming these directly, use root `.../VLM_3R_pi3x_pointmaps` and subdir
  `.` because files live directly under each dataset directory.

## VGGT features and diagnostics

- Current VGGT feature sidecars are aggregated-token sidecars, not depth-only
  sidecars.
- Verified extraction records point to:
  `/leonardo_scratch/large/userexternal/shuang00/VLM_3R_vggt_features/{scannet,scannetpp,arkitscenes}/*.pt`.
- Schema: `.pt` dict with `frames.aggregated_tokens`, `frames.frame_idx`, and
  `meta`. `meta.schema` is `vggt_aggregated_tokens_v1`; feature dim is 2048;
  extracted VGGT intermediate layers are `[4, 11, 17, 23]`.
- Use with:
  `SPATIAL_FEATURES_ROOT=/leonardo_scratch/large/userexternal/shuang00/VLM_3R_vggt_features`,
  `SPATIAL_FEATURES_SUBDIR=.`, `MODEL_SPATIAL_TOWER=vggt`, and
  `MODEL_SPATIAL_FEATURE_DIM=2048`.
- `scripts/extraction/export_vggt_point_cloud.py` is a diagnostic/export tool:
  it writes a PLY and manifest from an image folder, and only writes
  `depth_map` / `depth_conf` tensors when run with `--save-pt`.
