from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary_geometry_head import AuxiliaryGeometryHead
from .geometry_provider_adapter import GeometryProviderAdapter
from .geometry_rope import GeometryRoPE


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class GeometryAwareProjectionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mode: str = "spherical",
        dim_allocation: Optional[Dict[str, Sequence[float]]] = None,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.norm_attn = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.geometry_rope = GeometryRoPE(
            head_dim=self.head_dim,
            mode=mode,
            dim_allocation=dim_allocation,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gamma_attn = nn.Parameter(torch.tensor(float(gate_init)))

        ffn_hidden = int(hidden_size * float(mlp_ratio))
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.gamma_ffn = nn.Parameter(torch.tensor(float(gate_init)))

    def _to_heads(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        geometry_pos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        normed = self.norm_attn(visual_tokens)
        q = self._to_heads(self.q_proj(normed))
        k = self._to_heads(self.k_proj(normed))
        v = self._to_heads(self.v_proj(normed))
        q, k = self.geometry_rope(q, k, geometry_pos)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            key_mask = attention_mask.to(device=visual_tokens.device, dtype=torch.bool)
            all_invalid = ~key_mask.any(dim=-1)
            safe_key_mask = key_mask
            if all_invalid.any():
                safe_key_mask = key_mask.clone()
                safe_key_mask[all_invalid, :] = True
            attn = attn.masked_fill(~safe_key_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = F.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        attn_out = torch.matmul(attn, v)
        attn_out = self.out_proj(self._merge_heads(attn_out))
        visual_tokens = visual_tokens + self.gamma_attn.to(dtype=visual_tokens.dtype) * self.dropout(attn_out)

        ffn_out = self.ffn(self.norm_ffn(visual_tokens))
        visual_tokens = visual_tokens + self.gamma_ffn.to(dtype=visual_tokens.dtype) * self.dropout(ffn_out)
        return visual_tokens


class MetricGroundedGeometryProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = getattr(config, "mm_hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size")
        hidden_size = int(hidden_size)
        num_heads = int(getattr(config, "geometry_projection_num_heads", 16))
        mode = getattr(config, "geometry_position_mode", "spherical")
        num_layers = int(getattr(config, "num_geometry_projection_layers", 1))
        gate_init = float(getattr(config, "geometry_gate_init", 0.0))
        loss_type = getattr(config, "geometry_loss_type", "smooth_l1")
        target_names = getattr(config, "aux_geometry_targets", ["azimuth", "elevation", "log_distance"])
        if isinstance(target_names, str):
            target_names = [item.strip() for item in target_names.split(",") if item.strip()]
        dim_allocation = getattr(config, "geometry_rope_dim_allocation", None)

        self.hidden_size = hidden_size
        self.mode = mode
        self.use_auxiliary_geometry_head = _as_bool(getattr(config, "use_auxiliary_geometry_head", True), True)
        self.use_auxiliary_geometry_loss = _as_bool(getattr(config, "use_auxiliary_geometry_loss", True), True)
        self.lambda_geo = float(getattr(config, "lambda_geo", 0.1))
        self.adapter = GeometryProviderAdapter(
            mode=mode,
            max_abs=float(getattr(config, "geometry_position_max_abs", 10.0)),
            fixed_scene_scale=float(getattr(config, "geometry_fixed_scene_scale", 5.0)),
            detach_geometry_targets=_as_bool(getattr(config, "detach_geometry_targets", True), True),
            use_geometry_confidence_mask=_as_bool(getattr(config, "use_geometry_confidence_mask", True), True),
            point_map_key=(
                getattr(config, "geo_rope_point_map_key", None)
                or getattr(config, "geometry_point_map_key", None)
            ),
        )
        self.layers = nn.ModuleList([
            GeometryAwareProjectionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mode=mode,
                dim_allocation=dim_allocation,
                gate_init=gate_init,
                dropout_rate=float(getattr(config, "geometry_projection_dropout", 0.0)),
            )
            for _ in range(num_layers)
        ])
        self.aux_head = AuxiliaryGeometryHead(
            hidden_size,
            target_names=target_names,
            loss_type=loss_type,
            allow_missing_targets=_as_bool(getattr(config, "allow_missing_geometry_targets", False), False),
        )
        self.last_outputs = None

    def forward(
        self,
        visual_tokens: torch.Tensor,
        geometry_outputs: Dict[str, torch.Tensor],
        visual_grid_size: Optional[Tuple[int, int]],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[dict] = None,
        num_frames: Optional[int] = None,
        spatial_merge_size: Optional[int] = None,
    ):
        del labels
        geometry_pos, geometry_targets, geometry_mask = self.adapter(
            geometry_outputs=geometry_outputs,
            visual_tokens=visual_tokens,
            visual_grid_size=visual_grid_size,
            num_frames=num_frames,
            spatial_merge_size=spatial_merge_size,
        )
        geometry_mask_bool = geometry_mask.to(device=visual_tokens.device, dtype=torch.bool)
        geometry_pos = geometry_pos.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        geometry_pos = geometry_pos.masked_fill(~geometry_mask_bool[..., None], 0.0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=visual_tokens.device, dtype=torch.bool) & geometry_mask_bool
        else:
            attention_mask = geometry_mask_bool
        hidden = visual_tokens
        for layer in self.layers:
            hidden = layer(hidden, geometry_pos, attention_mask=attention_mask)

        geometry_predictions = {}
        loss_geo = None
        if self.use_auxiliary_geometry_head:
            geometry_predictions = self.aux_head(hidden)
            if self.use_auxiliary_geometry_loss:
                loss_geo = self.aux_head.compute_loss(geometry_predictions, geometry_targets, geometry_mask)

        outputs = {
            "refined_tokens": hidden,
            "geometry_predictions": geometry_predictions,
            "loss_geo": loss_geo,
            "geometry_pos": geometry_pos,
            "geometry_mask": geometry_mask_bool,
            "geometry_targets": geometry_targets,
        }
        self.last_outputs = outputs
        return outputs
