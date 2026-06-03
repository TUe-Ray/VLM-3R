from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REFERENCE_POINT_MAP_KEYS = ("point_maps_ref", "pts3d_in_other_view")
CAMERA_POINT_MAP_KEYS = ("point_maps_cam", "pts3d_in_self_view")
GENERIC_POINT_MAP_KEYS = ("point_maps", "point_map", "points", "pts3d")
CONFIDENCE_KEYS = ("confidence", "conf", "depth_conf", "pts3d_conf")
DEPTH_KEYS = ("depth", "depth_map")


class BEVHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(hidden_dim), 2)

    def forward(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_hidden_states)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coalesce_tensor_sequence(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [_coalesce_tensor_sequence(item) for item in value if item is not None]
        tensors = [item for item in tensors if item is not None]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)
    return None


def _normalize_bev_point_map_key(key: Optional[str]) -> Tuple[Tuple[str, ...], str]:
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
        "cam": (CAMERA_POINT_MAP_KEYS, "camera"),
        "camera": (CAMERA_POINT_MAP_KEYS, "camera"),
        "self": (CAMERA_POINT_MAP_KEYS, "camera"),
        "point_maps_cam": (CAMERA_POINT_MAP_KEYS, "camera"),
        "pts3d_in_self_view": (("pts3d_in_self_view", "point_maps_cam"), "camera"),
        "point_maps": (("point_maps",), "generic"),
        "point_map": (("point_map",), "generic"),
        "points": (("points",), "generic"),
        "pts3d": (("pts3d",), "generic"),
    }
    if normalized not in aliases:
        raise ValueError(
            "bev_point_map_key must be one of ref/point_maps_ref/pts3d_in_other_view, "
            "cam/point_maps_cam/pts3d_in_self_view, or a generic point_maps alias; "
            f"got {key!r}"
        )
    return aliases[normalized]


def _find_tensor(payload: Any, keys: Iterable[str]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if isinstance(payload, torch.Tensor):
        return payload, "tensor"
    if not isinstance(payload, dict):
        return None, None
    for key in keys:
        value = _coalesce_tensor_sequence(payload.get(key))
        if value is not None:
            return value, key
    return None, None


def _point_map_to_fhwc(point_map: torch.Tensor) -> torch.Tensor:
    if point_map.dim() == 5:
        if point_map.shape[0] != 1:
            raise ValueError(
                "Expected a per-sample point map, got a rank-5 tensor with "
                f"batch={point_map.shape[0]}"
            )
        point_map = point_map[0]
    if point_map.dim() != 4:
        raise ValueError(
            "point_map must be [F,H,W,3], [F,3,H,W], or [1,F,H,W,3], "
            f"got {tuple(point_map.shape)}"
        )
    if point_map.shape[-1] == 3:
        return point_map.float()
    if point_map.shape[1] == 3:
        return point_map.permute(0, 2, 3, 1).float()
    raise ValueError(f"point_map must have 3 coordinate channels, got {tuple(point_map.shape)}")


def _scalar_map_to_fhw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.dim() == 5:
        if value.shape[0] != 1:
            raise ValueError(f"Expected per-sample {name}, got rank-5 tensor with batch={value.shape[0]}")
        value = value[0]
    if value.dim() == 4 and value.shape[-1] == 1:
        return value[..., 0].float()
    if value.dim() == 4 and value.shape[1] == 1:
        return value[:, 0].float()
    if value.dim() == 3:
        return value.float()
    raise ValueError(f"{name} must be [F,H,W], [F,H,W,1], or [F,1,H,W], got {tuple(value.shape)}")


def _shape_matches_fhw(value: torch.Tensor, frame_count: int, height: int, width: int) -> bool:
    return tuple(value.shape[:3]) == (int(frame_count), int(height), int(width))


def _tensor_to_list_per_sample(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, torch.Tensor):
        if value.dim() == 5:
            return [value[idx] for idx in range(value.shape[0])]
        return [value]
    raise TypeError(f"Unsupported point-map payload type: {type(value)}")


def _metadata_list(visual_metadata: Any) -> List[Dict[str, Any]]:
    if isinstance(visual_metadata, dict):
        return [visual_metadata]
    if isinstance(visual_metadata, (list, tuple)):
        return list(visual_metadata)
    raise TypeError(f"visual_metadata must be a dict or list of dicts, got {type(visual_metadata)}")


def _frame_grid_shape(metadata: Dict[str, Any], frame_id: int, token_count: Optional[int] = None) -> Tuple[int, int]:
    frame_order = [int(x) for x in metadata.get("frame_order", [])]
    shapes = metadata.get("visual_grid_shapes") or []
    if frame_order and shapes and int(frame_id) in frame_order:
        idx = frame_order.index(int(frame_id))
        shape = shapes[idx]
        return int(shape[0]), int(shape[1])
    if token_count is None:
        tokens_per_frame = metadata.get("tokens_per_frame")
        if isinstance(tokens_per_frame, list) and tokens_per_frame:
            token_count = int(tokens_per_frame[0])
    if token_count is None:
        raise ValueError(f"Cannot infer BEV grid shape for frame {frame_id}")
    side = int(math.isqrt(int(token_count)))
    if side * side != int(token_count):
        raise ValueError(f"Token count is not square for frame {frame_id}: {token_count}")
    return side, side


def _validate_visual_metadata(metadata: Dict[str, Any], sample_idx: int) -> None:
    visual_indices = metadata.get("visual_token_indices")
    frame_ids = metadata.get("visual_frame_ids")
    if not isinstance(visual_indices, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
        raise ValueError(f"visual_metadata[{sample_idx}] must contain tensor visual_token_indices and visual_frame_ids")
    if visual_indices.numel() != frame_ids.numel():
        raise ValueError(
            f"visual_metadata[{sample_idx}] visual index/frame-id length mismatch: "
            f"{visual_indices.numel()} vs {frame_ids.numel()}"
        )
    excluded_parts = []
    for key in (
        "newline_token_indices",
        "padding_token_indices",
        "answer_token_indices",
        "text_token_indices",
        "special_token_indices",
        "camera_prefix_token_indices",
    ):
        value = metadata.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            excluded_parts.append(value.to(device=visual_indices.device, dtype=torch.long))
    if excluded_parts:
        excluded = torch.cat(excluded_parts)
        if torch.isin(visual_indices.to(dtype=torch.long), excluded).any():
            raise ValueError(
                f"visual_metadata[{sample_idx}] visual_token_indices overlap excluded "
                "newline/padding/text/answer/special/camera-prefix tokens"
            )


def _build_dense_validity(
    payload: Any,
    target: torch.Tensor,
    target_key: str,
    target_space: str,
    use_geometry_confidence_mask: bool,
    bev_conf_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    frame_count, height, width, _ = target.shape
    finite_target = torch.isfinite(target).all(dim=-1)
    valid = finite_target.clone()
    debug: Dict[str, Any] = {
        "bev_target_key": target_key,
        "bev_target_space": target_space,
        "validity_depth_source": None,
        "confidence_source": None,
        "confidence_threshold": float(bev_conf_threshold),
        "validity_fallback_target_finite_only": False,
    }

    if target_space == "camera":
        valid = valid & torch.isfinite(target[..., 2]) & (target[..., 2] > 0)
        debug["validity_depth_source"] = target_key
    elif target_space == "reference":
        cam_tensor, cam_key = _find_tensor(payload, CAMERA_POINT_MAP_KEYS)
        cam_fhwc = None
        if cam_tensor is not None:
            cam_fhwc = _point_map_to_fhwc(cam_tensor)
        if cam_fhwc is not None and _shape_matches_fhw(cam_fhwc, frame_count, height, width):
            valid = valid & torch.isfinite(cam_fhwc[..., 2]) & (cam_fhwc[..., 2] > 0)
            debug["validity_depth_source"] = cam_key
        else:
            depth_tensor, depth_key = _find_tensor(payload, DEPTH_KEYS)
            if depth_tensor is not None:
                depth = _scalar_map_to_fhw(depth_tensor, name=depth_key or "depth")
                if _shape_matches_fhw(depth, frame_count, height, width):
                    valid = valid & torch.isfinite(depth) & (depth > 0)
                    debug["validity_depth_source"] = depth_key

    if use_geometry_confidence_mask:
        conf_tensor, conf_key = _find_tensor(payload, CONFIDENCE_KEYS)
        if conf_tensor is not None:
            conf = _scalar_map_to_fhw(conf_tensor, name=conf_key or "confidence")
            if _shape_matches_fhw(conf, frame_count, height, width):
                valid = valid & torch.isfinite(conf) & (conf > float(bev_conf_threshold))
                debug["confidence_source"] = conf_key

    debug["validity_fallback_target_finite_only"] = (
        debug["validity_depth_source"] is None and debug["confidence_source"] is None
    )
    return valid, debug


def _pool_bev_frame(
    target_frame: torch.Tensor,
    valid_frame: torch.Tensor,
    grid_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_tok, w_tok = int(grid_shape[0]), int(grid_shape[1])
    bev = target_frame[..., (0, 2)].permute(2, 0, 1).unsqueeze(0).float()
    mask = valid_frame.unsqueeze(0).unsqueeze(0).float()
    values = torch.where(mask.bool(), bev, torch.zeros_like(bev))
    pooled_values = F.adaptive_avg_pool2d(values, (h_tok, w_tok))[0]
    pooled_mask = F.adaptive_avg_pool2d(mask, (h_tok, w_tok))[0, 0]
    pooled_bev = pooled_values / pooled_mask.clamp_min(1e-8)
    pooled_bev = torch.where(
        (pooled_mask > 0).unsqueeze(0),
        pooled_bev,
        torch.zeros_like(pooled_bev),
    )
    return pooled_bev.permute(1, 2, 0).reshape(h_tok * w_tok, 2), pooled_mask.reshape(-1) > 0


def _build_one_sample(
    payload: Any,
    metadata: Dict[str, Any],
    *,
    bev_point_map_key: str,
    use_geometry_confidence_mask: bool,
    bev_conf_threshold: float,
    existing_geometry_mask: Optional[torch.Tensor],
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    _validate_visual_metadata(metadata, sample_idx)
    target_keys, target_space = _normalize_bev_point_map_key(bev_point_map_key)
    target_tensor, target_key = _find_tensor(payload, target_keys)
    if target_tensor is None:
        available = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise KeyError(f"Could not find BEV target {bev_point_map_key!r}. Available: {available}")
    target = _point_map_to_fhwc(target_tensor)
    valid_dense, validity_debug = _build_dense_validity(
        payload,
        target,
        target_key or str(bev_point_map_key),
        target_space,
        use_geometry_confidence_mask,
        bev_conf_threshold,
    )

    if existing_geometry_mask is not None:
        mask = existing_geometry_mask
        if mask.dim() == 4 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.dim() == 3 and _shape_matches_fhw(mask, target.shape[0], target.shape[1], target.shape[2]):
            valid_dense = valid_dense & mask.to(dtype=torch.bool, device=valid_dense.device)
            validity_debug["existing_geometry_mask_source"] = "dense_fhw"
        else:
            validity_debug["existing_geometry_mask_source"] = "provided_but_not_dense_fhw"

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
                f"Frame {frame_id} has {token_count} visual tokens but metadata grid "
                f"{grid_shape} implies {expected}"
            )
        if frame_id not in frame_to_source:
            raise ValueError(f"Frame id {frame_id} is not present in point-map frame_order/source frames")
        source_idx = frame_to_source[frame_id]
        if source_idx < 0 or source_idx >= int(target.shape[0]):
            raise ValueError(f"Frame id {frame_id} maps to out-of-range point-map frame {source_idx}")
        pooled_by_frame[frame_id] = _pool_bev_frame(target[source_idx], valid_dense[source_idx], grid_shape)
        frame_debug.append({
            "frame_id": frame_id,
            "source_frame_index": int(source_idx),
            "source_hw": [int(target.shape[1]), int(target.shape[2])],
            "target_grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
            "raw_visual_grid_shape": list((metadata.get("raw_visual_grid_shapes") or [grid_shape])[min(source_idx, len(metadata.get("raw_visual_grid_shapes") or [grid_shape]) - 1)]),
            "resizing_or_pooling_applied": [int(target.shape[1]), int(target.shape[2])] != [int(grid_shape[0]), int(grid_shape[1])],
            "num_visual_tokens": token_count,
        })

    cursors = {frame_id: 0 for frame_id in unique_frame_ids}
    bev_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    for frame_id_tensor in frame_ids:
        frame_id = int(frame_id_tensor.item())
        pooled_bev, pooled_valid = pooled_by_frame[frame_id]
        cursor = cursors[frame_id]
        if cursor >= pooled_bev.shape[0]:
            raise ValueError(f"Frame {frame_id} cursor exceeded pooled BEV token count")
        bev_rows.append(pooled_bev[cursor])
        mask_rows.append(pooled_valid[cursor])
        cursors[frame_id] = cursor + 1

    bev_gt = torch.stack(bev_rows, dim=0) if bev_rows else torch.empty(0, 2)
    valid_mask = torch.stack(mask_rows, dim=0).to(dtype=torch.bool) if mask_rows else torch.empty(0, dtype=torch.bool)
    debug = {
        **validity_debug,
        "num_visual_tokens": int(visual_indices.numel()),
        "num_valid_bev_tokens": int(valid_mask.sum().item()),
        "valid_bev_token_ratio": float(valid_mask.float().mean().item()) if valid_mask.numel() > 0 else 0.0,
        "source_num_frames": int(target.shape[0]),
        "frame_order": frame_order or list(range(int(target.shape[0]))),
        "pooling_method": "masked_adaptive_avg_pool2d",
        "pooling_note": "Dense point maps are pooled to visual_grid_shapes; raw H,W are not reshaped as token grids.",
        "frames": frame_debug,
    }
    return bev_gt, valid_mask, debug


def build_bev_targets_from_point_maps(
    point_map_payloads: Any,
    visual_metadata: Any,
    *,
    bev_point_map_key: str = "point_maps_ref",
    use_geometry_confidence_mask: bool = True,
    bev_conf_threshold: float = 0.0,
    existing_geometry_mask: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build per-visual-token BEV targets from CUT3R point-map sidecars.

    Target coordinates and validity are intentionally selected separately:
    reference-frame BEV targets use reference points for coordinates, but use
    camera-space depth or explicit depth for visibility validity when available.
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
        bev_gt, valid_mask, debug = _build_one_sample(
            payload,
            metadata,
            bev_point_map_key=bev_point_map_key,
            use_geometry_confidence_mask=_as_bool(use_geometry_confidence_mask, True),
            bev_conf_threshold=float(bev_conf_threshold),
            existing_geometry_mask=mask,
            sample_idx=idx,
        )
        sample_targets.append(bev_gt)
        sample_masks.append(valid_mask)
        sample_debug.append(debug)
        max_tokens = max(max_tokens, int(bev_gt.shape[0]))

    if not sample_targets:
        return torch.empty(0, 0, 2), torch.empty(0, 0, dtype=torch.bool), {"samples": []}

    dtype = sample_targets[0].dtype
    device = sample_targets[0].device
    batch_targets = torch.zeros(len(sample_targets), max_tokens, 2, dtype=dtype, device=device)
    batch_masks = torch.zeros(len(sample_targets), max_tokens, dtype=torch.bool, device=device)
    for idx, (bev_gt, valid_mask) in enumerate(zip(sample_targets, sample_masks)):
        n = int(bev_gt.shape[0])
        batch_targets[idx, :n] = bev_gt.to(device=device, dtype=dtype)
        batch_masks[idx, :n] = valid_mask.to(device=device)

    total_tokens = int(batch_masks.numel())
    total_valid = int(batch_masks.sum().item())
    debug = {
        "bev_point_map_key_requested": bev_point_map_key,
        "bev_conf_threshold": float(bev_conf_threshold),
        "num_samples": len(sample_targets),
        "num_total_visual_tokens": total_tokens,
        "num_valid_bev_tokens": total_valid,
        "valid_bev_token_ratio": float(total_valid / total_tokens) if total_tokens else 0.0,
        "samples": sample_debug,
    }
    return batch_targets, batch_masks, debug
