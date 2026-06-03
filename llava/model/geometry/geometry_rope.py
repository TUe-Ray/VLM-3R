import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_GEO_ROPE_DIM_ALLOCATION = {
    "depth": [1],
    "xyz": [2, 1, 2],
    "spherical": [2, 1, 2],
}


class GeometryRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        mode: str = "spherical",
        theta: float = 10000.0,
        dim_allocation: Optional[Dict[str, Sequence[float]]] = None,
    ):
        super().__init__()
        if mode not in {"depth", "xyz", "spherical"}:
            raise ValueError(f"Unexpected geometry_position_mode: {mode}")
        self.head_dim = int(head_dim)
        self.mode = mode
        self.theta = float(theta)
        self.dim_allocation = dict(DEFAULT_GEO_ROPE_DIM_ALLOCATION)
        if dim_allocation:
            self.dim_allocation.update(dim_allocation)

        weights = self.dim_allocation[mode]
        self.group_dims = self._split_even_dims(self.head_dim, weights)
        self.rotary_dim = sum(self.group_dims)
        for idx, group_dim in enumerate(self.group_dims):
            inv_freq = self._build_inv_freq(group_dim)
            self.register_buffer(f"inv_freq_{idx}", inv_freq, persistent=False)

    @staticmethod
    def expected_pos_dim(mode: str) -> int:
        if mode == "depth":
            return 1
        if mode in {"xyz", "spherical"}:
            return 3
        raise ValueError(f"Unexpected geometry_position_mode: {mode}")

    @staticmethod
    def _split_even_dims(head_dim: int, weights: Sequence[float]):
        num_pairs = int(head_dim) // 2
        if num_pairs <= 0:
            return [0 for _ in weights]
        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError(f"Geometry-RoPE dim allocation must be positive, got {weights}")
        pair_counts = [int(math.floor(num_pairs * float(weight) / total_weight)) for weight in weights]
        pair_counts[-1] += num_pairs - sum(pair_counts)
        return [2 * count for count in pair_counts]

    def _build_inv_freq(self, group_dim: int):
        if group_dim <= 0:
            return torch.empty(0)
        return 1.0 / (self.theta ** (torch.arange(0, group_dim, 2).float() / group_dim))

    @staticmethod
    def _rotate_group(x_group: torch.Tensor, position_value: torch.Tensor, inv_freq: torch.Tensor):
        if x_group.shape[-1] == 0:
            return x_group
        inv_freq = inv_freq.to(device=x_group.device, dtype=position_value.dtype)
        position_value = torch.nan_to_num(position_value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        angle = position_value.unsqueeze(-1) * inv_freq
        cos = angle.cos().unsqueeze(1).to(dtype=x_group.dtype)
        sin = angle.sin().unsqueeze(1).to(dtype=x_group.dtype)

        x_even = x_group[..., 0::2]
        x_odd = x_group[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

    def _apply_one(self, x: torch.Tensor, geometry_pos: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"GeometryRoPE expects x as [B, H, N, D], got {tuple(x.shape)}")
        if geometry_pos.dim() != 3:
            raise ValueError(f"GeometryRoPE expects geometry_pos as [B, N, C], got {tuple(geometry_pos.shape)}")
        expected_pos_dim = self.expected_pos_dim(self.mode)
        if geometry_pos.shape[-1] != expected_pos_dim:
            raise ValueError(
                f"{self.mode} GeometryRoPE expects C_pos={expected_pos_dim}, got {geometry_pos.shape[-1]}"
            )
        if x.shape[0] != geometry_pos.shape[0] or x.shape[2] != geometry_pos.shape[1]:
            raise ValueError(
                "GeometryRoPE position shape must match x batch/token dims, "
                f"got x={tuple(x.shape)} and geometry_pos={tuple(geometry_pos.shape)}"
            )

        geometry_pos = torch.nan_to_num(geometry_pos.to(device=x.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
        rotated_groups = []
        start = 0
        for idx, group_dim in enumerate(self.group_dims):
            end = start + group_dim
            inv_freq = getattr(self, f"inv_freq_{idx}")
            rotated_groups.append(self._rotate_group(x[..., start:end], geometry_pos[..., idx], inv_freq))
            start = end
        if start < x.shape[-1]:
            rotated_groups.append(x[..., start:])
        return torch.cat(rotated_groups, dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, geometry_pos: torch.Tensor):
        if q.shape != k.shape:
            raise ValueError(f"GeometryRoPE expects q/k to share shape, got q={tuple(q.shape)} k={tuple(k.shape)}")
        return self._apply_one(q, geometry_pos), self._apply_one(k, geometry_pos)
