# Geometry and RoPE Design

Read this document before modifying geometry projection, GeoRoPE, or their
training and evaluation configuration.

## Metric-Grounded Geometry Projection

- Preserve the invariant: Q/K/V come from 2D visual tokens; geometry only
  rotates Q/K through Geometry-RoPE. Geometry is not used as K/V and is not
  concatenated into LLM tokens.
- Keep GeoRoPE Fusion and Metric-Grounded Geometry Projection conceptually
  separate.

## Coordinate consistency

- GeoRoPE point-map coordinates must be train/eval consistent. If training uses
  `point_maps_ref` / `pts3d_in_other_view` (CUT3R reference/anchor-frame
  coordinates), evaluation must use the same keys. If training uses
  `point_maps_cam` / `pts3d_in_self_view` (per-frame camera coordinates),
  evaluation must use the same keys.
- Never add an eval-only alias such as `point_maps = point_maps_cam` unless the
  matching training job used that same coordinate source.

## Baseline protection

- Default new experimental features to disabled unless requested.
- Avoid changing baseline behavior unless the wrapper or configuration
  explicitly enables the feature.
