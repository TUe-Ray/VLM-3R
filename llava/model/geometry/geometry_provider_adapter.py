import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


POINT_MAP_ALIASES = (
    "point_map",
    "point_maps",
    "points",
    "pts3d",
    # Coordinate consistency is part of the GeoRoPE experiment contract:
    # train and eval must use the same coordinate source. For CUT3R sidecars,
    # point_maps_ref/pts3d_in_other_view is reference/anchor-frame, while
    # point_maps_cam/pts3d_in_self_view is per-frame camera coordinates.
    "point_maps_ref",
    "pts3d_in_other_view",
    "point_maps_cam",
    "pts3d_in_self_view",
)
DEPTH_ALIASES = ("depth", "depth_map")
CONFIDENCE_ALIASES = ("confidence", "conf", "depth_conf", "pts3d_conf")


def _normalize_point_map_key(key):
    if key in (None, ""):
        return None
    normalized = str(key).strip().lower()
    aliases = {
        "ref": ("point_maps_ref", "pts3d_in_other_view"),
        "reference": ("point_maps_ref", "pts3d_in_other_view"),
        "anchor": ("point_maps_ref", "pts3d_in_other_view"),
        "point_maps_ref": ("point_maps_ref", "pts3d_in_other_view"),
        "pts3d_in_other_view": ("pts3d_in_other_view", "point_maps_ref"),
        "cam": ("point_maps_cam", "pts3d_in_self_view"),
        "camera": ("point_maps_cam", "pts3d_in_self_view"),
        "self": ("point_maps_cam", "pts3d_in_self_view"),
        "point_maps_cam": ("point_maps_cam", "pts3d_in_self_view"),
        "pts3d_in_self_view": ("pts3d_in_self_view", "point_maps_cam"),
    }
    if normalized not in aliases:
        raise ValueError(
            "geometry_point_map_key must be one of ref/point_maps_ref/"
            "pts3d_in_other_view or cam/point_maps_cam/pts3d_in_self_view, "
            f"got {key!r}"
        )
    return aliases[normalized]


def _coalesce_tensor_sequence(value):
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


def canonicalize_geometry_outputs(*sources, point_maps=None, point_map_key=None) -> Dict[str, torch.Tensor]:
    canonical = {}
    if point_maps is not None:
        point_map = _coalesce_tensor_sequence(point_maps)
        if point_map is not None:
            canonical["point_map"] = point_map

    requested_point_map_aliases = _normalize_point_map_key(point_map_key)

    def visit(source):
        if source is None:
            return
        if isinstance(source, torch.Tensor):
            canonical.setdefault("point_map", source)
            return
        if isinstance(source, (list, tuple)):
            if all(isinstance(item, torch.Tensor) or item is None for item in source):
                tensor = _coalesce_tensor_sequence(source)
                if tensor is not None:
                    canonical.setdefault("point_map", tensor)
                return
            merged = {}
            for item in source:
                item_canonical = canonicalize_geometry_outputs(item, point_map_key=point_map_key)
                for key, value in item_canonical.items():
                    merged.setdefault(key, []).append(value)
            for key, values in merged.items():
                if key not in canonical:
                    canonical[key] = _coalesce_tensor_sequence(values)
            return
        if not isinstance(source, dict):
            return
        for out_key, aliases in (
            ("point_map", requested_point_map_aliases or POINT_MAP_ALIASES),
            ("depth", DEPTH_ALIASES),
            ("confidence", CONFIDENCE_ALIASES),
        ):
            for alias in aliases:
                if alias in source and source[alias] is not None:
                    tensor = _coalesce_tensor_sequence(source[alias])
                    if tensor is not None:
                        canonical.setdefault(out_key, tensor)
                    break

    for source in sources:
        visit(source)
    return canonical


class GeometryProviderAdapter(nn.Module):
    def __init__(
        self,
        mode: str = "spherical",
        max_abs: float = 10.0,
        fixed_scene_scale: float = 5.0,
        detach_geometry_targets: bool = True,
        use_geometry_confidence_mask: bool = True,
        point_map_key: Optional[str] = None,
    ):
        super().__init__()
        if mode not in {"depth", "xyz", "spherical"}:
            raise ValueError(f"Unexpected geometry_position_mode: {mode}")
        self.mode = mode
        self.max_abs = float(max_abs)
        self.fixed_scene_scale = float(fixed_scene_scale)
        self.detach_geometry_targets = bool(detach_geometry_targets)
        self.use_geometry_confidence_mask = bool(use_geometry_confidence_mask)
        self.point_map_key = point_map_key

    @staticmethod
    def _flatten_frame_dim(x: torch.Tensor, channel_last_dim: Optional[int] = None):
        if x is None:
            return None
        if x.dim() == 5:
            if channel_last_dim is not None and x.shape[-1] == channel_last_dim:
                return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
            if channel_last_dim == 1 and x.shape[-1] == 1:
                return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:4])
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        return x

    @staticmethod
    def _point_map_to_bchw(point_map: torch.Tensor):
        point_map = GeometryProviderAdapter._flatten_frame_dim(point_map, channel_last_dim=3)
        if point_map.dim() != 4:
            raise ValueError(f"point_map must be [B,H,W,3], [B,3,H,W], or [B,T,H,W,3], got {tuple(point_map.shape)}")
        if point_map.shape[-1] == 3:
            return point_map.permute(0, 3, 1, 2)
        if point_map.shape[1] == 3:
            return point_map
        raise ValueError(f"point_map must have 3 coordinate channels, got {tuple(point_map.shape)}")

    @staticmethod
    def _depth_to_bchw(depth: torch.Tensor):
        depth = GeometryProviderAdapter._flatten_frame_dim(depth, channel_last_dim=1)
        if depth.dim() == 3:
            return depth.unsqueeze(1)
        if depth.dim() == 4 and depth.shape[1] != 1 and depth.shape[-1] != 1:
            return depth.reshape(depth.shape[0] * depth.shape[1], 1, depth.shape[2], depth.shape[3])
        if depth.dim() == 4 and depth.shape[-1] == 1:
            return depth.permute(0, 3, 1, 2)
        if depth.dim() == 4 and depth.shape[1] == 1:
            return depth
        raise ValueError(f"depth must be [B,H,W], [B,H,W,1], [B,1,H,W], or [B,T,H,W], got {tuple(depth.shape)}")

    @staticmethod
    def _confidence_to_bchw(confidence: Optional[torch.Tensor]):
        if confidence is None:
            return None
        confidence = GeometryProviderAdapter._flatten_frame_dim(confidence, channel_last_dim=1)
        if confidence.dim() == 3:
            return confidence.unsqueeze(1)
        if confidence.dim() == 4 and confidence.shape[1] != 1 and confidence.shape[-1] != 1:
            return confidence.reshape(
                confidence.shape[0] * confidence.shape[1],
                1,
                confidence.shape[2],
                confidence.shape[3],
            )
        if confidence.dim() == 4 and confidence.shape[-1] == 1:
            return confidence.permute(0, 3, 1, 2)
        if confidence.dim() == 4 and confidence.shape[1] == 1:
            return confidence
        raise ValueError(f"confidence must be [B,H,W], [B,H,W,1], or [B,1,H,W], got {tuple(confidence.shape)}")

    @staticmethod
    def _resize_map(x: torch.Tensor, visual_grid_size: Tuple[int, int], mode: str):
        return F.interpolate(x.float(), size=visual_grid_size, mode=mode, align_corners=False if mode == "bilinear" else None)

    @staticmethod
    def _scene_scale_from_depth(depth: torch.Tensor, valid: torch.Tensor, fallback: float):
        flat_depth = depth.reshape(depth.shape[0], -1).float()
        flat_valid = valid.reshape(valid.shape[0], -1)
        scales = []
        for sample_depth, sample_valid in zip(flat_depth, flat_valid):
            values = sample_depth[sample_valid]
            values = values[torch.isfinite(values) & (values > 0)]
            if values.numel() == 0:
                scales.append(sample_depth.new_tensor(float(fallback)))
            else:
                scales.append(values.median().clamp_min(1e-6))
        return torch.stack(scales).view(-1, 1, 1)

    @staticmethod
    def _infer_visual_grid_size(num_tokens: int):
        side = int(math.isqrt(int(num_tokens)))
        if side * side != int(num_tokens):
            raise ValueError(
                "visual_grid_size is required when visual token count is not a perfect square, "
                f"got {num_tokens}"
            )
        return side, side

    def forward(
        self,
        geometry_outputs: Dict[str, torch.Tensor],
        visual_tokens: torch.Tensor,
        visual_grid_size: Optional[Tuple[int, int]] = None,
        num_frames: Optional[int] = None,
        spatial_merge_size: Optional[int] = None,
    ):
        del num_frames, spatial_merge_size
        if visual_tokens.dim() != 3:
            raise ValueError(f"visual_tokens must be [B,N,C], got {tuple(visual_tokens.shape)}")
        if visual_grid_size is None:
            visual_grid_size = self._infer_visual_grid_size(visual_tokens.shape[1])
        visual_grid_size = (int(visual_grid_size[0]), int(visual_grid_size[1]))
        expected_tokens = visual_grid_size[0] * visual_grid_size[1]
        if expected_tokens != visual_tokens.shape[1]:
            raise ValueError(
                f"visual_grid_size={visual_grid_size} implies {expected_tokens} tokens, "
                f"but visual_tokens has {visual_tokens.shape[1]}"
            )

        point_map = geometry_outputs.get("point_map")
        depth = geometry_outputs.get("depth")
        confidence = geometry_outputs.get("confidence")
        if point_map is None and depth is None:
            raise ValueError("GeometryProviderAdapter requires canonical point_map or depth.")

        point_grid = None
        if point_map is not None:
            point_grid = self._resize_map(self._point_map_to_bchw(point_map), visual_grid_size, mode="bilinear")
            depth_grid = point_grid[:, 2:3]
        else:
            depth_grid = self._resize_map(self._depth_to_bchw(depth), visual_grid_size, mode="bilinear")

        if depth is not None and point_map is not None:
            depth_grid = self._resize_map(self._depth_to_bchw(depth), visual_grid_size, mode="bilinear")

        finite_mask = torch.isfinite(depth_grid) & (depth_grid > 0)
        if point_grid is not None:
            finite_mask = finite_mask & torch.isfinite(point_grid).all(dim=1, keepdim=True)
        conf_grid = self._confidence_to_bchw(confidence)
        if conf_grid is not None and self.use_geometry_confidence_mask:
            conf_grid = self._resize_map(conf_grid, visual_grid_size, mode="nearest")
            finite_mask = finite_mask & torch.isfinite(conf_grid) & (conf_grid > 0)

        depth_grid = torch.nan_to_num(depth_grid.float(), nan=0.0, posinf=0.0, neginf=0.0)
        log_depth = torch.log1p(depth_grid.clamp_min(0.0))
        targets = {
            "depth": depth_grid.flatten(2).transpose(1, 2),
            "log_depth": log_depth.flatten(2).transpose(1, 2),
        }

        if point_grid is not None:
            point_grid = torch.nan_to_num(point_grid.float(), nan=0.0, posinf=0.0, neginf=0.0)
            xyz = point_grid.flatten(2).transpose(1, 2)
            x = point_grid[:, 0]
            y = point_grid[:, 1]
            z = point_grid[:, 2].clamp_min(0.0)
            radius_xz = torch.sqrt(x * x + z * z + 1e-6)
            radius = torch.sqrt(x * x + y * y + z * z + 1e-6)
            azimuth = torch.atan2(x, z)
            elevation = torch.atan2(y, radius_xz)
            log_distance = torch.log1p(radius)
            targets.update({
                "xyz": xyz,
                "azimuth": azimuth.flatten(1).unsqueeze(-1),
                "elevation": elevation.flatten(1).unsqueeze(-1),
                "log_distance": log_distance.flatten(1).unsqueeze(-1),
            })
        else:
            point_grid = None
            targets["log_distance"] = targets["log_depth"]

        if self.detach_geometry_targets:
            targets = {key: value.detach() for key, value in targets.items()}

        mask = finite_mask.flatten(2).squeeze(1)
        if self.mode == "depth":
            geometry_pos = targets["log_depth"]
        elif self.mode == "xyz":
            if point_grid is None:
                raise ValueError("geometry_position_mode='xyz' requires point_map geometry.")
            scale = self._scene_scale_from_depth(point_grid[:, 2], finite_mask[:, 0], self.fixed_scene_scale)
            xyz_norm = (point_grid / scale.unsqueeze(1)).clamp(min=-self.max_abs, max=self.max_abs)
            geometry_pos = xyz_norm.flatten(2).transpose(1, 2)
        else:
            if point_grid is None:
                raise ValueError("geometry_position_mode='spherical' requires point_map geometry.")
            geometry_pos = torch.cat(
                (targets["azimuth"], targets["elevation"], targets["log_distance"]),
                dim=-1,
            )

        return geometry_pos.to(device=visual_tokens.device), targets, mask.to(device=visual_tokens.device)
