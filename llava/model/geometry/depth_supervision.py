from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_supervision import (
    CAMERA_POINT_MAP_KEYS,
    CONFIDENCE_KEYS,
    _as_bool,
    _find_tensor,
    _frame_grid_shape,
    _metadata_list,
    _point_map_to_fhwc,
    _scalar_map_to_fhw,
    _shape_matches_fhw,
    _tensor_to_list_per_sample,
    _validate_visual_metadata,
)


GENERIC_POINT_MAP_KEYS = ("point_maps", "point_map", "points", "pts3d")


class DepthHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(hidden_dim), 1)

    def forward(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_hidden_states).squeeze(-1)


def _normalize_depth_point_map_key(key: Optional[str]) -> Tuple[Tuple[str, ...], str]:
    if key in (None, ""):
        return CAMERA_POINT_MAP_KEYS, "camera"
    normalized = str(key).strip().lower()
    aliases = {
        "cam": (CAMERA_POINT_MAP_KEYS, "camera"),
        "camera": (CAMERA_POINT_MAP_KEYS, "camera"),
        "self": (CAMERA_POINT_MAP_KEYS, "camera"),
        "point_maps_cam": (CAMERA_POINT_MAP_KEYS, "camera"),
        "pts3d_in_self_view": (("pts3d_in_self_view", "point_maps_cam"), "camera"),
    }
    generic_aliases = {
        "point_maps": (("point_maps",), "generic_camera_assumed"),
        "point_map": (("point_map",), "generic_camera_assumed"),
        "points": (("points",), "generic_camera_assumed"),
        "pts3d": (("pts3d",), "generic_camera_assumed"),
    }
    reference_aliases = {
        "ref",
        "reference",
        "anchor",
        "world",
        "point_maps_ref",
        "pts3d_in_other_view",
    }
    if normalized in reference_aliases:
        raise ValueError(
            "Depth supervision requires per-frame camera-space point maps "
            "(point_maps_cam or pts3d_in_self_view). Reference/world-frame "
            f"z is not valid depth; got depth_point_map_key={key!r}."
        )
    if normalized in generic_aliases:
        return generic_aliases[normalized]
    if normalized not in aliases:
        raise ValueError(
            "depth_point_map_key must be one of cam/point_maps_cam/"
            "pts3d_in_self_view. Generic point_maps aliases require "
            "depth_allow_generic_camera_assumed=True and are only valid "
            f"when they contain camera-space coordinates; got {key!r}."
        )
    return aliases[normalized]


def _find_depth_point_map(
    payload: Any,
    depth_point_map_key: str,
    *,
    depth_allow_generic_camera_assumed: bool,
    depth_allow_tensor_camera_assumed: bool,
) -> Tuple[torch.Tensor, str, str]:
    if isinstance(payload, torch.Tensor):
        if not depth_allow_tensor_camera_assumed:
            raise ValueError(
                "Depth supervision received a raw tensor payload, but raw tensors do not encode "
                "their coordinate space. Use geometry_spatial_features with point_maps_cam/"
                "pts3d_in_self_view, or set depth_allow_tensor_camera_assumed=True only after "
                "verifying the tensor is per-frame camera-space."
            )
        return payload, "tensor", "camera_tensor_assumed"
    target_keys, target_space = _normalize_depth_point_map_key(depth_point_map_key)
    if target_space == "generic_camera_assumed" and not depth_allow_generic_camera_assumed:
        raise ValueError(
            f"Depth supervision requested generic key {depth_point_map_key!r}, but generic point-map "
            "keys do not encode coordinate space. Use point_maps_cam/pts3d_in_self_view, or set "
            "depth_allow_generic_camera_assumed=True only after verifying the sidecar is camera-space."
        )
    tensor, target_key = _find_tensor(payload, target_keys)
    if tensor is not None:
        return tensor, target_key or str(depth_point_map_key), target_space
    available = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    raise KeyError(
        "Depth supervision requires camera-space point maps. Expected "
        f"{list(target_keys)} for depth_point_map_key={depth_point_map_key!r}; "
        f"available={available}. Do not use point_maps_ref/pts3d_in_other_view z as depth."
    )


def _build_dense_depth_validity(
    payload: Any,
    depth: torch.Tensor,
    *,
    target_key: str,
    target_space: str,
    use_geometry_confidence_mask: bool,
    depth_conf_threshold: float,
    depth_max_gt: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    frame_count, height, width = depth.shape
    valid = torch.isfinite(depth) & (depth > 0)
    max_gt = float(depth_max_gt)
    if max_gt > 0:
        valid = valid & (depth <= max_gt)

    debug: Dict[str, Any] = {
        "depth_target_key": target_key,
        "depth_target_space": target_space,
        "confidence_source": None,
        "confidence_threshold": float(depth_conf_threshold),
        "depth_max_gt": max_gt,
    }

    if use_geometry_confidence_mask and isinstance(payload, dict):
        conf_tensor, conf_key = _find_tensor(payload, CONFIDENCE_KEYS)
        if conf_tensor is not None:
            conf = _scalar_map_to_fhw(conf_tensor, name=conf_key or "confidence")
            if _shape_matches_fhw(conf, frame_count, height, width):
                valid = valid & torch.isfinite(conf) & (conf > float(depth_conf_threshold))
                debug["confidence_source"] = conf_key

    return valid, debug


def _pool_depth_frame(
    depth_frame: torch.Tensor,
    valid_frame: torch.Tensor,
    grid_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    depth = depth_frame.unsqueeze(0).unsqueeze(0).float()
    mask = valid_frame.unsqueeze(0).unsqueeze(0).float()
    values = torch.where(mask.bool(), depth, torch.zeros_like(depth))
    pooled_values = F.adaptive_avg_pool2d(values, (h_tok, w_tok))[0, 0]
    pooled_mask = F.adaptive_avg_pool2d(mask, (h_tok, w_tok))[0, 0]
    pooled_depth = pooled_values / pooled_mask.clamp_min(1e-8)
    pooled_depth = torch.where(pooled_mask > 0, pooled_depth, torch.zeros_like(pooled_depth))
    return pooled_depth.reshape(h_tok * w_tok), pooled_mask.reshape(-1) > 0


def _apply_existing_geometry_mask(
    valid_dense: torch.Tensor,
    existing_geometry_mask: Optional[torch.Tensor],
    debug: Dict[str, Any],
) -> torch.Tensor:
    if existing_geometry_mask is None:
        return valid_dense
    mask = existing_geometry_mask
    if mask.dim() == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.dim() == 3 and _shape_matches_fhw(mask, valid_dense.shape[0], valid_dense.shape[1], valid_dense.shape[2]):
        debug["existing_geometry_mask_source"] = "dense_fhw"
        return valid_dense & mask.to(dtype=torch.bool, device=valid_dense.device)
    debug["existing_geometry_mask_source"] = "provided_but_not_dense_fhw"
    return valid_dense


def _build_one_sample(
    payload: Any,
    metadata: Dict[str, Any],
    *,
    depth_point_map_key: str,
    use_geometry_confidence_mask: bool,
    depth_conf_threshold: float,
    depth_max_gt: float,
    depth_allow_generic_camera_assumed: bool,
    depth_allow_tensor_camera_assumed: bool,
    existing_geometry_mask: Optional[torch.Tensor],
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    _validate_visual_metadata(metadata, sample_idx)
    target_tensor, target_key, target_space = _find_depth_point_map(
        payload,
        depth_point_map_key,
        depth_allow_generic_camera_assumed=depth_allow_generic_camera_assumed,
        depth_allow_tensor_camera_assumed=depth_allow_tensor_camera_assumed,
    )
    point_map = _point_map_to_fhwc(target_tensor)
    depth = point_map[..., 2]
    valid_dense, validity_debug = _build_dense_depth_validity(
        payload,
        depth,
        target_key=target_key,
        target_space=target_space,
        use_geometry_confidence_mask=use_geometry_confidence_mask,
        depth_conf_threshold=depth_conf_threshold,
        depth_max_gt=depth_max_gt,
    )
    valid_dense = _apply_existing_geometry_mask(valid_dense, existing_geometry_mask, validity_debug)

    visual_indices = metadata["visual_token_indices"].detach().cpu().to(dtype=torch.long)
    frame_ids = metadata["visual_frame_ids"].detach().cpu().to(dtype=torch.long)
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if frame_order and len(frame_order) == int(depth.shape[0]):
        frame_to_source = {int(frame_id): idx for idx, frame_id in enumerate(frame_order)}
    else:
        frame_to_source = {idx: idx for idx in range(int(depth.shape[0]))}

    unique_frame_ids = [int(x) for x in torch.unique(frame_ids, sorted=True).tolist()]
    pooled_by_frame: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    frame_debug = []
    for frame_id in unique_frame_ids:
        token_count = int((frame_ids == frame_id).sum().item())
        grid_shape = _frame_grid_shape(metadata, frame_id, token_count=token_count)
        expected = int(grid_shape[0]) * int(grid_shape[1])
        if token_count != expected:
            raise ValueError(
                f"Frame {frame_id} has {token_count} visual tokens but metadata grid "
                f"{grid_shape} implies {expected}"
            )
        if frame_id not in frame_to_source:
            raise ValueError(f"Frame id {frame_id} is not present in point-map frame_order/source frames")
        source_idx = frame_to_source[frame_id]
        if source_idx < 0 or source_idx >= int(depth.shape[0]):
            raise ValueError(f"Frame id {frame_id} maps to out-of-range point-map frame {source_idx}")
        pooled_by_frame[frame_id] = _pool_depth_frame(depth[source_idx], valid_dense[source_idx], grid_shape)
        raw_shapes = metadata.get("raw_visual_grid_shapes") or [grid_shape]
        frame_debug.append({
            "frame_id": frame_id,
            "source_frame_index": int(source_idx),
            "source_hw": [int(depth.shape[1]), int(depth.shape[2])],
            "target_grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "raw_visual_grid_shape": list(raw_shapes[min(source_idx, len(raw_shapes) - 1)]),
            "resizing_or_pooling_applied": [int(depth.shape[1]), int(depth.shape[2])] != [
                int(grid_shape[0]),
                int(grid_shape[1]),
            ],
            "num_visual_tokens": token_count,
        })

    cursors = {frame_id: 0 for frame_id in unique_frame_ids}
    depth_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    for frame_id_tensor in frame_ids:
        frame_id = int(frame_id_tensor.item())
        pooled_depth, pooled_valid = pooled_by_frame[frame_id]
        cursor = cursors[frame_id]
        if cursor >= pooled_depth.shape[0]:
            raise ValueError(f"Frame {frame_id} cursor exceeded pooled depth token count")
        depth_rows.append(pooled_depth[cursor])
        mask_rows.append(pooled_valid[cursor])
        cursors[frame_id] = cursor + 1

    depth_gt_meter = torch.stack(depth_rows, dim=0) if depth_rows else torch.empty(0)
    valid_mask = torch.stack(mask_rows, dim=0).to(dtype=torch.bool) if mask_rows else torch.empty(0, dtype=torch.bool)
    depth_gt_log = torch.log1p(depth_gt_meter.clamp_min(0.0))
    valid_mask = valid_mask & torch.isfinite(depth_gt_log)
    debug = {
        **validity_debug,
        "num_visual_tokens": int(visual_indices.numel()),
        "num_valid_depth_tokens": int(valid_mask.sum().item()),
        "valid_depth_token_ratio": float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 0.0,
        "source_num_frames": int(depth.shape[0]),
        "frame_order": frame_order or list(range(int(depth.shape[0]))),
        "pooling_method": "masked_adaptive_avg_pool2d",
        "pooling_note": "Dense camera-space z is pooled to visual_grid_shapes before log1p.",
        "depth_meter_min": float(depth_gt_meter[valid_mask].min().item()) if valid_mask.any() else 0.0,
        "depth_meter_mean": float(depth_gt_meter[valid_mask].mean().item()) if valid_mask.any() else 0.0,
        "depth_meter_max": float(depth_gt_meter[valid_mask].max().item()) if valid_mask.any() else 0.0,
        "depth_log_min": float(depth_gt_log[valid_mask].min().item()) if valid_mask.any() else 0.0,
        "depth_log_mean": float(depth_gt_log[valid_mask].mean().item()) if valid_mask.any() else 0.0,
        "depth_log_max": float(depth_gt_log[valid_mask].max().item()) if valid_mask.any() else 0.0,
        "frames": frame_debug,
    }
    return depth_gt_log, valid_mask, debug


def build_depth_targets_from_point_maps(
    point_map_payloads: Any,
    visual_metadata: Any,
    *,
    depth_point_map_key: str = "point_maps_cam",
    use_geometry_confidence_mask: bool = True,
    depth_conf_threshold: float = 0.0,
    depth_max_gt: float = 20.0,
    depth_allow_generic_camera_assumed: bool = False,
    depth_allow_tensor_camera_assumed: bool = False,
    existing_geometry_mask: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build log-depth targets aligned to LLM visual-token metadata.

    Depth targets are camera-centric by construction. The default key searches
    point_maps_cam/pts3d_in_self_view and rejects reference-frame point maps so
    point_maps_ref[..., 2] cannot be used as per-frame camera depth.
    """

    payloads = _tensor_to_list_per_sample(point_map_payloads)
    metadata_items = _metadata_list(visual_metadata)
    if len(payloads) != len(metadata_items):
        raise ValueError(f"Batch size mismatch: {len(payloads)} point-map payloads vs {len(metadata_items)} metadata items")

    mask_items = _tensor_to_list_per_sample(existing_geometry_mask) if existing_geometry_mask is not None else [None] * len(payloads)
    if len(mask_items) not in {0, len(payloads)}:
        raise ValueError(f"existing_geometry_mask batch size mismatch: {len(mask_items)} vs {len(payloads)}")
    if not mask_items:
        mask_items = [None] * len(payloads)

    sample_targets = []
    sample_masks = []
    sample_debug = []
    max_tokens = 0
    for idx, (payload, metadata, mask) in enumerate(zip(payloads, metadata_items, mask_items)):
        depth_gt_log, valid_mask, debug = _build_one_sample(
            payload,
            metadata,
            depth_point_map_key=depth_point_map_key,
            use_geometry_confidence_mask=_as_bool(use_geometry_confidence_mask, True),
            depth_conf_threshold=float(depth_conf_threshold),
            depth_max_gt=float(depth_max_gt),
            depth_allow_generic_camera_assumed=_as_bool(depth_allow_generic_camera_assumed, False),
            depth_allow_tensor_camera_assumed=_as_bool(depth_allow_tensor_camera_assumed, False),
            existing_geometry_mask=mask,
            sample_idx=idx,
        )
        sample_targets.append(depth_gt_log)
        sample_masks.append(valid_mask)
        sample_debug.append(debug)
        max_tokens = max(max_tokens, int(depth_gt_log.shape[0]))

    if not sample_targets:
        return torch.empty(0, 0), torch.empty(0, 0, dtype=torch.bool), {"samples": []}

    dtype = sample_targets[0].dtype
    device = sample_targets[0].device
    batch_targets = torch.zeros(len(sample_targets), max_tokens, dtype=dtype, device=device)
    batch_masks = torch.zeros(len(sample_targets), max_tokens, dtype=torch.bool, device=device)
    for idx, (depth_gt_log, valid_mask) in enumerate(zip(sample_targets, sample_masks)):
        n = int(depth_gt_log.shape[0])
        batch_targets[idx, :n] = depth_gt_log.to(device=device, dtype=dtype)
        batch_masks[idx, :n] = valid_mask.to(device=device)

    total_tokens = int(batch_masks.numel())
    total_valid = int(batch_masks.sum().item())
    debug = {
        "depth_point_map_key_requested": depth_point_map_key,
        "depth_conf_threshold": float(depth_conf_threshold),
        "depth_max_gt": float(depth_max_gt),
        "depth_allow_generic_camera_assumed": bool(_as_bool(depth_allow_generic_camera_assumed, False)),
        "depth_allow_tensor_camera_assumed": bool(_as_bool(depth_allow_tensor_camera_assumed, False)),
        "num_samples": len(sample_targets),
        "num_total_visual_tokens": total_tokens,
        "num_valid_depth_tokens": total_valid,
        "valid_depth_token_ratio": float(total_valid / total_tokens) if total_tokens else 0.0,
        "samples": sample_debug,
    }
    if sample_debug:
        debug["depth_point_map_key_used"] = sample_debug[0].get("depth_target_key")
        debug["depth_target_space"] = sample_debug[0].get("depth_target_space")
    return batch_targets, batch_masks, debug
