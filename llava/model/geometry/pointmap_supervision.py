from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_supervision import (
    CONFIDENCE_KEYS,
    REFERENCE_POINT_MAP_KEYS,
    _as_bool,
    _build_dense_validity,
    _find_tensor,
    _frame_grid_shape,
    _metadata_list,
    _point_map_to_fhwc,
    _tensor_to_list_per_sample,
    _validate_visual_metadata,
)


class PointMapHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(visual_hidden_states)


def _normalize_pointmap_key(key: Optional[str]) -> Tuple[Tuple[str, ...], str]:
    if key in (None, ""):
        return REFERENCE_POINT_MAP_KEYS, "reference"
    normalized = str(key).strip().lower()
    aliases = {
        "ref": (REFERENCE_POINT_MAP_KEYS, "reference"),
        "reference": (REFERENCE_POINT_MAP_KEYS, "reference"),
        "anchor": (REFERENCE_POINT_MAP_KEYS, "reference"),
        "world": (REFERENCE_POINT_MAP_KEYS, "reference"),
        "point_maps_ref": (REFERENCE_POINT_MAP_KEYS, "reference"),
        "pts3d_in_other_view": (("pts3d_in_other_view", "point_maps_ref"), "reference"),
    }
    camera_aliases = {
        "cam",
        "camera",
        "self",
        "point_maps_cam",
        "pts3d_in_self_view",
    }
    generic_aliases = {"point_maps", "point_map", "points", "pts3d"}
    if normalized in camera_aliases:
        raise ValueError(
            "Point-map supervision is configured for world/reference-frame xyz. "
            f"Do not use camera-frame key {key!r}; use point_maps_ref or pts3d_in_other_view."
        )
    if normalized in generic_aliases:
        raise ValueError(
            "Generic point-map keys do not encode coordinate space. "
            f"Requested {key!r}; use point_maps_ref or pts3d_in_other_view explicitly."
        )
    if normalized not in aliases:
        raise ValueError(
            "pointmap_point_map_key must be one of ref/world/point_maps_ref/"
            f"pts3d_in_other_view; got {key!r}."
        )
    return aliases[normalized]


def _find_pointmap_target(payload: Any, pointmap_point_map_key: str) -> Tuple[torch.Tensor, str, str]:
    if isinstance(payload, torch.Tensor):
        raise ValueError(
            "Point-map supervision received a raw tensor payload, but raw tensors do not encode "
            "coordinate space. Use a sidecar dict with point_maps_ref or pts3d_in_other_view."
        )
    target_keys, target_space = _normalize_pointmap_key(pointmap_point_map_key)
    tensor, target_key = _find_tensor(payload, target_keys)
    if tensor is not None:
        return tensor, target_key or str(pointmap_point_map_key), target_space
    available = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
    raise KeyError(
        "Point-map supervision requires world/reference-frame point maps. Expected "
        f"{list(target_keys)} for pointmap_point_map_key={pointmap_point_map_key!r}; available={available}."
    )


def _pool_pointmap_frame(
    target_frame: torch.Tensor,
    valid_frame: torch.Tensor,
    grid_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    target = target_frame.permute(2, 0, 1).unsqueeze(0).float()
    mask = valid_frame.unsqueeze(0).unsqueeze(0).float()
    values = torch.where(mask.bool(), target, torch.zeros_like(target))
    pooled_values = F.adaptive_avg_pool2d(values, (h_tok, w_tok))[0].permute(1, 2, 0)
    pooled_mask = F.adaptive_avg_pool2d(mask, (h_tok, w_tok))[0, 0]
    pooled_xyz = pooled_values / pooled_mask.unsqueeze(-1).clamp_min(1e-8)
    pooled_xyz = torch.where(pooled_mask.unsqueeze(-1) > 0, pooled_xyz, torch.zeros_like(pooled_xyz))
    return pooled_xyz.reshape(h_tok * w_tok, 3), pooled_mask.reshape(-1) > 0


def _build_one_sample(
    payload: Any,
    metadata: Dict[str, Any],
    *,
    pointmap_point_map_key: str,
    use_geometry_confidence_mask: bool,
    pointmap_conf_threshold: float,
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    _validate_visual_metadata(metadata, sample_idx)
    target_tensor, target_key, target_space = _find_pointmap_target(payload, pointmap_point_map_key)
    target = _point_map_to_fhwc(target_tensor)
    valid_dense, validity_debug = _build_dense_validity(
        payload,
        target,
        target_key,
        target_space,
        use_geometry_confidence_mask,
        pointmap_conf_threshold,
    )

    visual_indices = metadata["visual_token_indices"].detach().cpu().to(dtype=torch.long)
    frame_ids = metadata["visual_frame_ids"].detach().cpu().to(dtype=torch.long)
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    if frame_order and len(frame_order) == int(target.shape[0]):
        frame_to_source = {int(frame_id): idx for idx, frame_id in enumerate(frame_order)}
    else:
        frame_to_source = {idx: idx for idx in range(int(target.shape[0]))}

    unique_frame_ids = [int(x) for x in torch.unique(frame_ids, sorted=True).tolist()]
    pooled_by_frame: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    frame_debug = []
    for frame_id in unique_frame_ids:
        token_count = int((frame_ids == frame_id).sum().item())
        grid_shape = _frame_grid_shape(metadata, frame_id, token_count=token_count)
        expected = int(grid_shape[0]) * int(grid_shape[1])
        if token_count != expected:
            raise ValueError(
                f"Frame {frame_id} has {token_count} visual tokens but metadata grid {grid_shape} implies {expected}"
            )
        if frame_id not in frame_to_source:
            raise ValueError(f"Frame id {frame_id} is not present in point-map frame_order/source frames")
        source_idx = frame_to_source[frame_id]
        if source_idx < 0 or source_idx >= int(target.shape[0]):
            raise ValueError(f"Frame id {frame_id} maps to out-of-range point-map frame {source_idx}")
        pooled_by_frame[frame_id] = _pool_pointmap_frame(target[source_idx], valid_dense[source_idx], grid_shape)
        frame_debug.append(
            {
                "frame_id": frame_id,
                "source_frame_index": int(source_idx),
                "source_hw": [int(target.shape[1]), int(target.shape[2])],
                "target_grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
                "num_visual_tokens": token_count,
            }
        )

    cursors = {frame_id: 0 for frame_id in unique_frame_ids}
    target_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    for frame_id_tensor in frame_ids:
        frame_id = int(frame_id_tensor.item())
        pooled_xyz, pooled_valid = pooled_by_frame[frame_id]
        cursor = cursors[frame_id]
        if cursor >= pooled_xyz.shape[0]:
            raise ValueError(f"Frame {frame_id} cursor exceeded pooled point-map token count")
        target_rows.append(pooled_xyz[cursor])
        mask_rows.append(pooled_valid[cursor])
        cursors[frame_id] = cursor + 1

    xyz_gt_meter = torch.stack(target_rows, dim=0) if target_rows else torch.empty(0, 3)
    valid_mask = torch.stack(mask_rows, dim=0).to(dtype=torch.bool) if mask_rows else torch.empty(0, dtype=torch.bool)
    valid_mask = valid_mask & torch.isfinite(xyz_gt_meter).all(dim=-1)
    debug = {
        **validity_debug,
        "pointmap_target_key": target_key,
        "pointmap_target_space": target_space,
        "confidence_threshold": float(pointmap_conf_threshold),
        "confidence_keys_checked": list(CONFIDENCE_KEYS),
        "num_visual_tokens": int(visual_indices.numel()),
        "num_valid_pointmap_tokens": int(valid_mask.sum().item()),
        "valid_pointmap_token_ratio": float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 0.0,
        "source_num_frames": int(target.shape[0]),
        "frame_order": frame_order or list(range(int(target.shape[0]))),
        "pooling_method": "masked_adaptive_avg_pool2d",
        "frames": frame_debug,
    }
    return xyz_gt_meter, valid_mask, debug


def build_pointmap_targets_from_point_maps(
    point_map_payloads: Any,
    visual_metadata: Any,
    *,
    pointmap_point_map_key: str = "point_maps_ref",
    use_geometry_confidence_mask: bool = True,
    pointmap_conf_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    payloads = _tensor_to_list_per_sample(point_map_payloads)
    metadata_items = _metadata_list(visual_metadata)
    if len(payloads) != len(metadata_items):
        raise ValueError(f"Batch size mismatch: {len(payloads)} point-map payloads vs {len(metadata_items)} metadata items")

    sample_targets = []
    sample_masks = []
    sample_debug = []
    max_tokens = 0
    for idx, (payload, metadata) in enumerate(zip(payloads, metadata_items)):
        xyz_gt_meter, valid_mask, debug = _build_one_sample(
            payload,
            metadata,
            pointmap_point_map_key=pointmap_point_map_key,
            use_geometry_confidence_mask=_as_bool(use_geometry_confidence_mask, True),
            pointmap_conf_threshold=float(pointmap_conf_threshold),
            sample_idx=idx,
        )
        sample_targets.append(xyz_gt_meter)
        sample_masks.append(valid_mask)
        sample_debug.append(debug)
        max_tokens = max(max_tokens, int(xyz_gt_meter.shape[0]))

    if not sample_targets:
        return torch.empty(0, 0, 3), torch.empty(0, 0, dtype=torch.bool), {"samples": []}

    dtype = sample_targets[0].dtype
    device = sample_targets[0].device
    batch_targets = torch.zeros(len(sample_targets), max_tokens, 3, dtype=dtype, device=device)
    batch_masks = torch.zeros(len(sample_targets), max_tokens, dtype=torch.bool, device=device)
    for idx, (xyz_gt_meter, valid_mask) in enumerate(zip(sample_targets, sample_masks)):
        n = int(xyz_gt_meter.shape[0])
        batch_targets[idx, :n] = xyz_gt_meter.to(device=device, dtype=dtype)
        batch_masks[idx, :n] = valid_mask.to(device=device)

    total_tokens = int(batch_masks.numel())
    total_valid = int(batch_masks.sum().item())
    debug = {
        "pointmap_point_map_key_requested": pointmap_point_map_key,
        "pointmap_conf_threshold": float(pointmap_conf_threshold),
        "num_samples": len(sample_targets),
        "num_total_visual_tokens": total_tokens,
        "num_valid_pointmap_tokens": total_valid,
        "valid_pointmap_token_ratio": float(total_valid / total_tokens) if total_tokens else 0.0,
        "samples": sample_debug,
    }
    if sample_debug:
        debug["pointmap_point_map_key_used"] = sample_debug[0].get("pointmap_target_key")
        debug["pointmap_target_space"] = sample_debug[0].get("pointmap_target_space")
    return batch_targets, batch_masks, debug
