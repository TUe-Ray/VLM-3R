"""Lightweight SigLIP-to-SpatialStack residual predictors.

The classes in this module deliberately operate on pooled visual patch tokens
only.  They do not import, construct, or execute the language model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


DEFAULT_SOURCE_LAYERS: Tuple[int, ...] = (6, 9, 12)


def _parse_int_list(value, name: str) -> List[int]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (tuple, list)):
        items = list(value)
    else:
        items = [value]
    try:
        result = [int(item) for item in items]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integer values, got {value!r}.") from exc
    if not result:
        raise ValueError(f"{name} may not be empty.")
    return result


class TokenWiseResidualPredictor(nn.Module):
    """Shared per-token MLP with independent SpatialStack layer heads."""

    architecture_name = "token_mlp"

    def __init__(
        self,
        hidden_size: int = 3584,
        bottleneck_dim: int = 1024,
        source_layers: Sequence[int] = DEFAULT_SOURCE_LAYERS,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.proj_in = nn.Linear(self.hidden_size, self.bottleneck_dim)
        self.act = nn.GELU()
        self.heads = nn.ModuleDict(
            {str(layer): nn.Linear(self.bottleneck_dim, self.hidden_size) for layer in self.source_layers}
        )
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> Dict[str, object]:
        return {
            "predictor_type": self.architecture_name,
            "hidden_size": self.hidden_size,
            "bottleneck_dim": self.bottleneck_dim,
            "source_layers": list(self.source_layers),
        }

    def forward(
        self,
        visual_tokens: torch.Tensor,
        valid_frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        if visual_tokens.dim() != 4:
            raise ValueError(
                "visual_tokens must be [batch, frames, patches, hidden], got "
                f"{tuple(visual_tokens.shape)}."
            )
        if int(visual_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"Predictor hidden dim mismatch: got {int(visual_tokens.shape[-1])}, "
                f"expected {self.hidden_size}."
            )
        del valid_frame_mask  # Token-wise mode intentionally has no frame mixing.
        trunk = self.act(self.proj_in(self.norm(visual_tokens)))
        return {int(layer): self.heads[str(layer)](trunk) for layer in self.source_layers}


class TemporalResidualPredictor(nn.Module):
    """Temporal-only mixing at each fixed 14x14 patch location."""

    architecture_name = "temporal"

    def __init__(
        self,
        hidden_size: int = 3584,
        bottleneck_dim: int = 1024,
        temporal_hidden_dim: int = 512,
        temporal_num_layers: int = 2,
        temporal_num_heads: int = 8,
        temporal_ffn_dim: int = 2048,
        temporal_dropout: float = 0.0,
        temporal_max_frames: int = 128,
        source_layers: Sequence[int] = DEFAULT_SOURCE_LAYERS,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.temporal_num_layers = int(temporal_num_layers)
        self.temporal_num_heads = int(temporal_num_heads)
        self.temporal_ffn_dim = int(temporal_ffn_dim)
        self.temporal_dropout = float(temporal_dropout)
        self.temporal_max_frames = int(temporal_max_frames)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        if self.temporal_hidden_dim % self.temporal_num_heads:
            raise ValueError("temporal_hidden_dim must be divisible by temporal_num_heads.")
        self.norm = nn.LayerNorm(self.hidden_size)
        self.input_proj = nn.Linear(self.hidden_size, self.temporal_hidden_dim)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(self.temporal_max_frames, self.temporal_hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.temporal_hidden_dim,
            nhead=self.temporal_num_heads,
            dim_feedforward=self.temporal_ffn_dim,
            dropout=self.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.temporal_num_layers)
        self.proj_out = nn.Linear(self.temporal_hidden_dim, self.bottleneck_dim)
        self.act = nn.GELU()
        self.heads = nn.ModuleDict(
            {str(layer): nn.Linear(self.bottleneck_dim, self.hidden_size) for layer in self.source_layers}
        )
        nn.init.normal_(self.temporal_pos_embed, mean=0.0, std=0.02)
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> Dict[str, object]:
        return {
            "predictor_type": self.architecture_name,
            "hidden_size": self.hidden_size,
            "bottleneck_dim": self.bottleneck_dim,
            "temporal_hidden_dim": self.temporal_hidden_dim,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_num_heads": self.temporal_num_heads,
            "temporal_ffn_dim": self.temporal_ffn_dim,
            "temporal_dropout": self.temporal_dropout,
            "temporal_max_frames": self.temporal_max_frames,
            "source_layers": list(self.source_layers),
        }

    def forward(
        self,
        visual_tokens: torch.Tensor,
        valid_frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        if visual_tokens.dim() != 4:
            raise ValueError(
                "visual_tokens must be [batch, frames, patches, hidden], got "
                f"{tuple(visual_tokens.shape)}."
            )
        batch, frames, patches, hidden = visual_tokens.shape
        if int(hidden) != self.hidden_size:
            raise ValueError(f"Predictor hidden dim mismatch: got {hidden}, expected {self.hidden_size}.")
        if int(frames) > self.temporal_max_frames:
            raise ValueError(
                f"Temporal predictor received F={int(frames)}, but temporal_max_frames="
                f"{self.temporal_max_frames}."
            )
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, device=visual_tokens.device, dtype=torch.bool)
        if tuple(valid_frame_mask.shape) != (batch, frames):
            raise ValueError(
                "valid_frame_mask must be [batch, frames], got "
                f"{tuple(valid_frame_mask.shape)} for visual tokens {tuple(visual_tokens.shape)}."
            )
        x = self.input_proj(self.norm(visual_tokens))
        x = x.permute(0, 2, 1, 3).reshape(batch * patches, frames, self.temporal_hidden_dim)
        x = x + self.temporal_pos_embed[:frames].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        padding_mask = (~valid_frame_mask.bool()).unsqueeze(1).expand(batch, patches, frames)
        padding_mask = padding_mask.reshape(batch * patches, frames)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.reshape(batch, patches, frames, self.temporal_hidden_dim).permute(0, 2, 1, 3)
        trunk = self.act(self.proj_out(x))
        return {int(layer): self.heads[str(layer)](trunk) for layer in self.source_layers}


def _temporal_view(x, valid_frame_mask, pos_embed):
    """Convert [B,F,P,D] to same-location temporal batches."""
    batch, frames, patches, hidden = x.shape
    x = x.permute(0, 2, 1, 3).reshape(batch * patches, frames, hidden)
    x = x + pos_embed[:frames].to(device=x.device, dtype=x.dtype).unsqueeze(0)
    padding = (~valid_frame_mask.bool()).unsqueeze(1).expand(batch, patches, frames)
    return x, padding.reshape(batch * patches, frames)


def _token_view(x, batch, frames, patches):
    return x.reshape(batch, patches, frames, x.shape[-1]).permute(0, 2, 1, 3)


def _encoder(hidden, heads, ffn, dropout, layers):
    layer = nn.TransformerEncoderLayer(
        d_model=int(hidden), nhead=int(heads), dim_feedforward=int(ffn),
        dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=int(layers))


class SpatialDepthwiseResidualBlock(nn.Module):
    """Pre-LN [B,F,196,D] -> depthwise 3x3 -> GELU -> pointwise -> residual."""

    def __init__(self, hidden_dim: int, grid_size: int = 14):
        super().__init__()
        self.hidden_dim, self.grid_size = int(hidden_dim), int(grid_size)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.depthwise = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)

    def forward(self, x):
        batch, frames, patches, hidden = x.shape
        if patches != self.grid_size ** 2 or hidden != self.hidden_dim:
            raise ValueError(f"Spatial block expected [B,F,{self.grid_size ** 2},{self.hidden_dim}], got {tuple(x.shape)}.")
        residual = x
        x = self.norm(x)
        x = x.reshape(batch * frames, self.grid_size, self.grid_size, hidden).permute(0, 3, 1, 2)
        x = self.pointwise(self.activation(self.depthwise(x)))
        x = x.permute(0, 2, 3, 1).reshape(batch, frames, patches, hidden)
        return residual + x


class SpatialTemporalResidualPredictor(TemporalResidualPredictor):
    architecture_name = "spatial_temporal"

    def __init__(self, *args, spatial_num_blocks: int = 2, spatial_grid_size: int = 14, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_num_blocks, self.spatial_grid_size = int(spatial_num_blocks), int(spatial_grid_size)
        self.spatial_blocks = nn.ModuleList([
            SpatialDepthwiseResidualBlock(self.temporal_hidden_dim, self.spatial_grid_size)
            for _ in range(self.spatial_num_blocks)
        ])

    def architecture_config(self):
        result = super().architecture_config()
        result.update({"predictor_type": self.architecture_name, "spatial_num_blocks": self.spatial_num_blocks,
                       "spatial_grid_size": self.spatial_grid_size, "spatial_norm": "LayerNorm",
                       "spatial_depthwise_kernel": 3, "spatial_depthwise_padding": 1,
                       "spatial_depthwise_groups": self.temporal_hidden_dim, "spatial_activation": "GELU",
                       "spatial_pointwise_kernel": 1})
        return result

    def _encode(self, visual_tokens, valid_frame_mask=None):
        batch, frames, patches, hidden = visual_tokens.shape
        if hidden != self.hidden_size or patches != 196:
            raise ValueError(f"Spatial-temporal input must be [B,F,196,{self.hidden_size}], got {tuple(visual_tokens.shape)}.")
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, device=visual_tokens.device, dtype=torch.bool)
        if frames > self.temporal_max_frames or tuple(valid_frame_mask.shape) != (batch, frames):
            raise ValueError("Invalid frames or valid_frame_mask for spatial-temporal predictor.")
        x = self.input_proj(self.norm(visual_tokens))
        for block in self.spatial_blocks:
            x = block(x)
        x, padding = _temporal_view(x, valid_frame_mask, self.temporal_pos_embed)
        return self.encoder(x, src_key_padding_mask=padding), padding, batch, frames, patches

    def forward(self, visual_tokens, valid_frame_mask=None):
        x, _, batch, frames, patches = self._encode(visual_tokens, valid_frame_mask)
        x = _token_view(x, batch, frames, patches)
        trunk = self.act(self.proj_out(x))
        return {int(layer): self.heads[str(layer)](trunk) for layer in self.source_layers}


class TargetAdapterTemporalResidualPredictor(TemporalResidualPredictor):
    """Shared temporal encoder with target-specific one-layer refinements."""
    architecture_name = "target_adapter_temporal"

    def __init__(self, *args, shared_temporal_layers=1, adapter_num_layers=1, **kwargs):
        kwargs["temporal_num_layers"] = int(shared_temporal_layers)
        super().__init__(*args, **kwargs)
        self.shared_temporal_layers, self.adapter_num_layers = int(shared_temporal_layers), int(adapter_num_layers)
        # The inherited shared projection is replaced by target-specific ones.
        # Keeping the unused Linear registered would give it no gradients.
        self.proj_out = nn.Identity()
        self.adapters = nn.ModuleDict({str(source): _encoder(self.temporal_hidden_dim, self.temporal_num_heads, self.temporal_ffn_dim, self.temporal_dropout, self.adapter_num_layers) for source in self.source_layers})
        self.branch_proj_out = nn.ModuleDict({str(source): nn.Linear(self.temporal_hidden_dim, self.bottleneck_dim) for source in self.source_layers})

    def architecture_config(self):
        result = super().architecture_config()
        result.update({"predictor_type": self.architecture_name, "shared_temporal_layers": self.shared_temporal_layers, "adapter_num_layers": self.adapter_num_layers, "target_specific_projection": True})
        return result

    def forward(self, visual_tokens, valid_frame_mask=None):
        batch, frames, patches, hidden = visual_tokens.shape
        if hidden != self.hidden_size or patches != 196:
            raise ValueError("Target-adapter input shape mismatch.")
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, device=visual_tokens.device, dtype=torch.bool)
        x, padding = _temporal_view(self.input_proj(self.norm(visual_tokens)), valid_frame_mask, self.temporal_pos_embed)
        shared = self.encoder(x, src_key_padding_mask=padding)
        result = {}
        for source in self.source_layers:
            branch = _token_view(self.adapters[str(source)](shared, src_key_padding_mask=padding), batch, frames, patches)
            result[int(source)] = self.heads[str(source)](self.act(self.branch_proj_out[str(source)](branch)))
        return result

class LayerConditionedTemporalResidualPredictor(TemporalResidualPredictor):
    """Shared temporal encoding with learned layer embedding FiLM conditioning."""
    architecture_name = "layer_conditioned_temporal"

    def __init__(self, *args, conditioned_decoder_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditioned_decoder_layers = int(conditioned_decoder_layers)
        self.layer_embeddings = nn.Parameter(torch.zeros(len(self.source_layers), self.temporal_hidden_dim))
        self.film = nn.Linear(self.temporal_hidden_dim, self.temporal_hidden_dim * 2)
        self.conditioned_decoder = _encoder(self.temporal_hidden_dim, self.temporal_num_heads, self.temporal_ffn_dim, self.temporal_dropout, self.conditioned_decoder_layers)
        nn.init.normal_(self.layer_embeddings, mean=0.0, std=0.02)

    def architecture_config(self):
        result = super().architecture_config()
        result.update({"predictor_type": self.architecture_name, "conditioning": "FiLM", "conditioned_decoder_layers": self.conditioned_decoder_layers, "layer_embedding_count": len(self.source_layers)})
        return result

    def forward(self, visual_tokens, valid_frame_mask=None):
        batch, frames, patches, hidden = visual_tokens.shape
        if hidden != self.hidden_size or patches != 196:
            raise ValueError("Layer-conditioned input shape mismatch.")
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, device=visual_tokens.device, dtype=torch.bool)
        x, padding = _temporal_view(self.input_proj(self.norm(visual_tokens)), valid_frame_mask, self.temporal_pos_embed)
        shared = self.encoder(x, src_key_padding_mask=padding)
        result = {}
        for index, source in enumerate(self.source_layers):
            gamma, beta = self.film(self.layer_embeddings[index]).chunk(2, dim=-1)
            decoded = self.conditioned_decoder(shared * (1 + gamma.view(1, 1, -1)) + beta.view(1, 1, -1), src_key_padding_mask=padding)
            decoded = _token_view(decoded, batch, frames, patches)
            result[int(source)] = self.heads[str(source)](self.act(self.proj_out(decoded)))
        return result

class SpatialTemporalTargetAdapterResidualPredictor(SpatialTemporalResidualPredictor):
    """Spatial-temporal shared trunk with target-specific temporal adapters."""
    architecture_name = "spatial_temporal_target_adapter"

    def __init__(self, *args, adapter_num_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_num_layers = int(adapter_num_layers)
        # This variant also uses branch-specific output projections only.
        self.proj_out = nn.Identity()
        self.adapters = nn.ModuleDict({str(source): _encoder(self.temporal_hidden_dim, self.temporal_num_heads, self.temporal_ffn_dim, self.temporal_dropout, self.adapter_num_layers) for source in self.source_layers})
        self.branch_proj_out = nn.ModuleDict({str(source): nn.Linear(self.temporal_hidden_dim, self.bottleneck_dim) for source in self.source_layers})

    def architecture_config(self):
        result = super().architecture_config()
        result.update({"predictor_type": self.architecture_name, "adapter_num_layers": self.adapter_num_layers, "target_specific_projection": True})
        return result

    def forward(self, visual_tokens, valid_frame_mask=None):
        shared, padding, batch, frames, patches = self._encode(visual_tokens, valid_frame_mask)
        result = {}
        for source in self.source_layers:
            branch = _token_view(self.adapters[str(source)](shared, src_key_padding_mask=padding), batch, frames, patches)
            result[int(source)] = self.heads[str(source)](self.act(self.branch_proj_out[str(source)](branch)))
        return result

def build_residual_predictor(
    predictor_type: str,
    *,
    hidden_size: int = 3584,
    bottleneck_dim: int = 1024,
    temporal_hidden_dim: int = 512,
    temporal_num_layers: int = 2,
    temporal_num_heads: int = 8,
    temporal_ffn_dim: int = 2048,
    temporal_dropout: float = 0.0,
    temporal_max_frames: int = 128,
    spatial_num_blocks: int = 2,
    spatial_grid_size: int = 14,
    shared_temporal_layers: int = 1,
    adapter_num_layers: int = 1,
    conditioned_decoder_layers: int = 1,
    source_layers: Sequence[int] = DEFAULT_SOURCE_LAYERS,
) -> nn.Module:
    predictor_type = str(predictor_type).strip().lower()
    kwargs = {
        "hidden_size": int(hidden_size),
        "bottleneck_dim": int(bottleneck_dim),
        "source_layers": tuple(int(layer) for layer in source_layers),
    }
    if predictor_type == "token_mlp":
        return TokenWiseResidualPredictor(**kwargs)
    if predictor_type == "temporal":
        return TemporalResidualPredictor(
            **kwargs,
            temporal_hidden_dim=int(temporal_hidden_dim),
            temporal_num_layers=int(temporal_num_layers),
            temporal_num_heads=int(temporal_num_heads),
            temporal_ffn_dim=int(temporal_ffn_dim),
            temporal_dropout=float(temporal_dropout),
            temporal_max_frames=int(temporal_max_frames),
        )
    common = dict(**kwargs, temporal_hidden_dim=int(temporal_hidden_dim), temporal_num_layers=int(temporal_num_layers), temporal_num_heads=int(temporal_num_heads), temporal_ffn_dim=int(temporal_ffn_dim), temporal_dropout=float(temporal_dropout), temporal_max_frames=int(temporal_max_frames))
    if predictor_type == "spatial_temporal":
        return SpatialTemporalResidualPredictor(**common, spatial_num_blocks=int(spatial_num_blocks), spatial_grid_size=int(spatial_grid_size))
    if predictor_type == "target_adapter_temporal":
        return TargetAdapterTemporalResidualPredictor(**common, shared_temporal_layers=int(shared_temporal_layers), adapter_num_layers=int(adapter_num_layers))
    if predictor_type == "layer_conditioned_temporal":
        return LayerConditionedTemporalResidualPredictor(**common, conditioned_decoder_layers=int(conditioned_decoder_layers))
    if predictor_type == "spatial_temporal_target_adapter":
        return SpatialTemporalTargetAdapterResidualPredictor(**common, spatial_num_blocks=int(spatial_num_blocks), spatial_grid_size=int(spatial_grid_size), adapter_num_layers=int(adapter_num_layers))
    raise ValueError(f"Unknown residual predictor type {predictor_type!r}.")


def predictor_checkpoint_payload(
    predictor: nn.Module,
    **metadata,
) -> Dict[str, object]:
    if not hasattr(predictor, "architecture_config"):
        raise TypeError("Predictor must expose architecture_config() for checkpointing.")
    return {
        "format_version": 2,
        "architecture": predictor.architecture_config(),
        "predictor": {name: value.detach().cpu() for name, value in predictor.state_dict().items()},
        **metadata,
    }


def predictor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Stable tensor-state hash used only for checkpoint deduplication."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_residual_predictor_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_type: Optional[str] = None,
    expected_hidden_size: Optional[int] = None,
    expected_source_layers: Optional[Sequence[int]] = None,
) -> Tuple[nn.Module, Mapping[str, object]]:
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "architecture" not in checkpoint or "predictor" not in checkpoint:
        raise RuntimeError(f"Invalid residual predictor checkpoint: {path}")
    architecture = dict(checkpoint["architecture"])
    predictor_type = str(architecture.pop("predictor_type", ""))
    if expected_type is not None and predictor_type != str(expected_type):
        raise RuntimeError(
            f"Predictor checkpoint type mismatch: checkpoint={predictor_type!r}, requested={expected_type!r}."
        )
    if expected_hidden_size is not None and int(architecture.get("hidden_size", -1)) != int(expected_hidden_size):
        raise RuntimeError(
            "Predictor checkpoint hidden size mismatch: "
            f"checkpoint={architecture.get('hidden_size')}, expected={expected_hidden_size}."
        )
    layers = tuple(int(layer) for layer in architecture.get("source_layers", ()))
    if expected_source_layers is not None and layers != tuple(int(layer) for layer in expected_source_layers):
        raise RuntimeError(
            f"Predictor checkpoint source layers mismatch: checkpoint={layers}, "
            f"expected={tuple(int(layer) for layer in expected_source_layers)}."
        )
    predictor = build_residual_predictor(predictor_type, **architecture)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    return predictor, checkpoint


class PredictedSpatialStackResidualAdapter(nn.Module):
    """Turn predicted patch residuals into Qwen's existing full-sequence payload."""

    _EXCLUDED_METADATA_KEYS = (
        "newline_token_indices",
        "padding_token_indices",
        "answer_token_indices",
        "text_token_indices",
        "special_token_indices",
        "camera_prefix_token_indices",
        "cut3r_camera_token_indices",
        "spatial_bridge_token_indices",
    )

    def __init__(
        self,
        predictor: nn.Module,
        *,
        source_layers: Sequence[int],
        llm_layers: Sequence[int],
        gamma_layers: Optional[Sequence[float]] = None,
        control: str = "none",
    ):
        super().__init__()
        self.predictor = predictor
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.llm_layers = tuple(int(layer) for layer in llm_layers)
        if len(self.source_layers) != len(self.llm_layers):
            raise ValueError("source_layers and llm_layers must have identical lengths.")
        predictor_layers = tuple(int(layer) for layer in getattr(predictor, "source_layers", ()))
        if predictor_layers != self.source_layers:
            raise ValueError(
                f"Predictor source layers {predictor_layers} do not match configured layers {self.source_layers}."
            )
        self.gamma_layers = tuple(float(gamma) for gamma in (gamma_layers or [1.0] * len(self.llm_layers)))
        if len(self.gamma_layers) != len(self.llm_layers):
            raise ValueError("gamma_layers must have one value per configured layer.")
        self.control = self._normalize_control(control)
        self.last_debug: Dict[str, object] = {}

    @staticmethod
    def _normalize_control(control: str) -> str:
        normalized = str(control or "none").strip().lower()
        if normalized not in {"none", "zero"}:
            raise ValueError("predicted_residual_control must be 'none' or 'zero'.")
        return normalized

    def configure(self, *, gamma_layers: Optional[Sequence[float]] = None, control: Optional[str] = None) -> None:
        if gamma_layers is not None:
            values = tuple(float(gamma) for gamma in gamma_layers)
            if len(values) != len(self.llm_layers):
                raise ValueError("gamma_layers must have one value per configured layer.")
            self.gamma_layers = values
        if control is not None:
            self.control = self._normalize_control(control)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, config) -> "PredictedSpatialStackResidualAdapter":
        source_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_layers", "6,9,12"), "cut3r_spatialstack_layers")
        llm_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2"), "cut3r_spatialstack_llm_layers")
        expected_type = getattr(config, "residual_predictor_type", None)
        if str(expected_type or "").lower() == "auto":
            expected_type = None
        predictor, checkpoint = load_residual_predictor_checkpoint(
            checkpoint_path,
            expected_type=expected_type,
            expected_hidden_size=int(getattr(config, "hidden_size")),
            expected_source_layers=source_layers,
        )
        expected_teacher = getattr(config, "residual_predictor_teacher_checkpoint", None)
        recorded_teacher = checkpoint.get("teacher_checkpoint")
        if expected_teacher and recorded_teacher and str(expected_teacher) != str(recorded_teacher):
            raise RuntimeError(
                "Predictor checkpoint teacher mismatch: "
                f"predictor={recorded_teacher}, requested={expected_teacher}."
            )
        gammas = [
            float(getattr(config, f"predicted_residual_gamma_layer{index}", 1.0))
            for index in range(len(llm_layers))
        ]
        adapter = cls(
            predictor,
            source_layers=source_layers,
            llm_layers=llm_layers,
            gamma_layers=gammas,
            control=getattr(config, "predicted_residual_control", "none"),
        )
        adapter.checkpoint_metadata = {
            "path": str(checkpoint_path),
            "architecture": dict(checkpoint["architecture"]),
            "teacher_checkpoint": recorded_teacher,
        }
        return adapter

    @staticmethod
    def _metadata_items(visual_metadata) -> List[Mapping[str, object]]:
        if isinstance(visual_metadata, Mapping):
            return [visual_metadata]
        if isinstance(visual_metadata, (list, tuple)):
            return list(visual_metadata)
        raise RuntimeError(
            "Predicted SpatialStack residuals require visual metadata from the multimodal preparation path."
        )

    def _extract_sample_tokens(
        self,
        inputs_embeds: torch.Tensor,
        batch_index: int,
        metadata: Mapping[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_indices = metadata.get("visual_token_indices")
        frame_ids = metadata.get("visual_frame_ids")
        if not isinstance(visual_indices, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
            raise RuntimeError("Predicted SpatialStack metadata lacks visual_token_indices/visual_frame_ids.")
        device = inputs_embeds.device
        visual_indices = visual_indices.to(device=device, dtype=torch.long)
        frame_ids = frame_ids.to(device=device, dtype=torch.long)
        if visual_indices.numel() != frame_ids.numel() or visual_indices.numel() == 0:
            raise RuntimeError("Predicted SpatialStack visual index/frame metadata is empty or inconsistent.")
        if int(visual_indices.min()) < 0 or int(visual_indices.max()) >= int(inputs_embeds.shape[1]):
            raise RuntimeError("Predicted SpatialStack visual token indices are outside the multimodal sequence.")
        excluded = []
        for key in self._EXCLUDED_METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, torch.Tensor) and value.numel():
                excluded.append(value.to(device=device, dtype=torch.long))
        if excluded and torch.isin(visual_indices, torch.cat(excluded)).any():
            raise RuntimeError("Predicted SpatialStack visual positions overlap excluded sequence positions.")
        frame_order = metadata.get("frame_order")
        if frame_order is None:
            frame_order = list(dict.fromkeys(int(item) for item in frame_ids.detach().cpu().tolist()))
        else:
            frame_order = [int(item) for item in frame_order]
        frames = []
        ordered_indices = []
        for frame_id in frame_order:
            frame_positions = visual_indices[frame_ids == int(frame_id)]
            if int(frame_positions.numel()) != 196:
                raise RuntimeError(
                    "Predicted SpatialStack requires exactly 196 visual patch positions per frame; "
                    f"frame {frame_id} has {int(frame_positions.numel())}."
                )
            frames.append(inputs_embeds[batch_index].index_select(0, frame_positions))
            ordered_indices.append(frame_positions)
        return torch.stack(frames, dim=0), torch.cat(ordered_indices, dim=0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        visual_metadata,
    ) -> Dict[int, torch.Tensor]:
        if inputs_embeds.dim() != 3:
            raise ValueError(f"inputs_embeds must be [batch, sequence, hidden], got {tuple(inputs_embeds.shape)}.")
        metadata_items = self._metadata_items(visual_metadata)
        if len(metadata_items) != int(inputs_embeds.shape[0]):
            raise RuntimeError(
                "Predicted SpatialStack metadata batch mismatch: "
                f"metadata={len(metadata_items)}, embeds={int(inputs_embeds.shape[0])}."
            )
        residuals = {
            int(layer): torch.zeros_like(inputs_embeds)
            for layer in self.llm_layers
        }
        sample_debug = []
        for batch_index, metadata in enumerate(metadata_items):
            sample_tokens, ordered_indices = self._extract_sample_tokens(inputs_embeds, batch_index, metadata)
            frame_count = int(sample_tokens.shape[0])
            parameter = next(self.predictor.parameters(), None)
            if parameter is None:
                raise RuntimeError("Predicted SpatialStack predictor has no parameters.")
            if parameter.device != inputs_embeds.device:
                self.predictor.to(device=inputs_embeds.device)
                parameter = next(self.predictor.parameters())
            predictor_tokens = sample_tokens.to(dtype=parameter.dtype)
            valid_mask = torch.ones(1, frame_count, device=inputs_embeds.device, dtype=torch.bool)
            predictions = self.predictor(predictor_tokens.unsqueeze(0), valid_mask)
            for source_layer, llm_layer, gamma in zip(self.source_layers, self.llm_layers, self.gamma_layers):
                prediction = predictions[int(source_layer)][0]
                if self.control == "zero":
                    prediction = torch.zeros_like(prediction)
                prediction = prediction.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) * float(gamma)
                residuals[int(llm_layer)][batch_index].index_copy_(0, ordered_indices, prediction.reshape(-1, prediction.shape[-1]))
            sample_debug.append({
                "sample_index": int(batch_index),
                "frames": frame_count,
                "visual_patch_positions": int(ordered_indices.numel()),
            })
        self.last_debug = {
            "source": "predicted_siglip_residuals",
            "cut3r_called": False,
            "control": self.control,
            "source_layers": list(self.source_layers),
            "llm_layers": list(self.llm_layers),
            "gamma_layers": list(self.gamma_layers),
            "samples": sample_debug,
        }
        return residuals


class MeanSpatialStackResidualAdapter(PredictedSpatialStackResidualAdapter):
    """Inject a fixed training-split mean residual without a predictor forward."""

    def __init__(self, templates, *, source_layers, llm_layers, gamma_layers, artifact_metadata):
        nn.Module.__init__(self)
        self.predictor = None
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.llm_layers = tuple(int(layer) for layer in llm_layers)
        self.gamma_layers = tuple(float(gamma) for gamma in gamma_layers)
        self.control = "mean"
        self.last_debug = {}
        self.artifact_metadata = dict(artifact_metadata)
        if len(self.source_layers) != len(self.llm_layers) or len(self.llm_layers) != len(self.gamma_layers):
            raise RuntimeError("Mean residual layer mapping and gammas must have identical lengths.")
        for layer in self.source_layers:
            template = templates.get(layer, templates.get(str(layer)))
            if not isinstance(template, torch.Tensor) or template.dim() != 2 or tuple(template.shape)[0] != 196:
                raise RuntimeError(f"Mean residual template for layer {layer} must have shape [196, hidden].")
            self.register_buffer(f"mean_residual_{layer}", template.float().contiguous(), persistent=True)

    @classmethod
    def from_artifact(cls, artifact_path, config):
        artifact = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("mean_residuals"), Mapping):
            raise RuntimeError(f"Invalid mean residual artifact: {artifact_path}")
        source_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_layers", "6,9,12"), "cut3r_spatialstack_layers")
        llm_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2"), "cut3r_spatialstack_llm_layers")
        expected_mapping = {str(source): int(llm) for source, llm in zip(source_layers, llm_layers)}
        recorded_mapping = {str(key): int(value) for key, value in artifact.get("source_to_llm_mapping", {}).items()}
        if recorded_mapping != expected_mapping:
            raise RuntimeError(f"Mean residual mapping mismatch: artifact={recorded_mapping}, expected={expected_mapping}.")
        hidden_size = int(getattr(config, "hidden_size"))
        for layer in source_layers:
            template = artifact["mean_residuals"].get(layer, artifact["mean_residuals"].get(str(layer)))
            if not isinstance(template, torch.Tensor) or tuple(template.shape) != (196, hidden_size):
                raise RuntimeError(f"Mean residual layer-{layer} has invalid shape; expected [196,{hidden_size}].")
        gammas = [float(getattr(config, f"predicted_residual_gamma_layer{index}", 1.0)) for index in range(len(llm_layers))]
        return cls(
            artifact["mean_residuals"], source_layers=source_layers, llm_layers=llm_layers,
            gamma_layers=gammas,
            artifact_metadata={"path": str(artifact_path), "teacher_checkpoint": artifact.get("teacher_checkpoint"), "teacher_config_hash": artifact.get("teacher_config_hash")},
        )

    def forward(self, inputs_embeds, visual_metadata):
        residuals = {int(layer): torch.zeros_like(inputs_embeds) for layer in self.llm_layers}
        sample_debug = []
        for batch_index, metadata in enumerate(self._metadata_items(visual_metadata)):
            sample_tokens, ordered_indices = self._extract_sample_tokens(inputs_embeds, batch_index, metadata)
            frame_count = int(sample_tokens.shape[0])
            for source_layer, llm_layer, gamma in zip(self.source_layers, self.llm_layers, self.gamma_layers):
                template = getattr(self, f"mean_residual_{source_layer}").to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                prediction = template.unsqueeze(0).expand(frame_count, -1, -1) * float(gamma)
                residuals[int(llm_layer)][batch_index].index_copy_(0, ordered_indices, prediction.reshape(-1, prediction.shape[-1]))
            sample_debug.append({"sample_index": int(batch_index), "frames": frame_count, "visual_patch_positions": int(ordered_indices.numel())})
        self.last_debug = {"source": "mean_spatialstack_residuals", "cut3r_called": False, "control": "mean", "source_layers": list(self.source_layers), "llm_layers": list(self.llm_layers), "gamma_layers": list(self.gamma_layers), "samples": sample_debug}
        return residuals


class CalibratedSpatialStackResidualAdapter(PredictedSpatialStackResidualAdapter):
    """Scale residual predictions using a train-split least-squares artifact."""

    def __init__(self, *args, alphas: Mapping[int, torch.Tensor], artifact_metadata: Mapping[str, object], **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_metadata = dict(artifact_metadata)
        for layer in self.source_layers:
            alpha = alphas.get(layer, alphas.get(str(layer)))
            if not isinstance(alpha, torch.Tensor) or tuple(alpha.shape) not in {(), (196, 1)}:
                raise RuntimeError(f"Calibration alpha for layer {layer} must be scalar or [196,1].")
            self.register_buffer(f"calibration_alpha_{layer}", alpha.float().contiguous(), persistent=True)

    @classmethod
    def from_artifact(cls, artifact_path, checkpoint_path, config):
        artifact = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("alphas"), Mapping):
            raise RuntimeError(f"Invalid residual calibration artifact: {artifact_path}")
        source_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_layers", "6,9,12"), "cut3r_spatialstack_layers")
        llm_layers = _parse_int_list(getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2"), "cut3r_spatialstack_llm_layers")
        expected_mapping = {str(source): int(llm) for source, llm in zip(source_layers, llm_layers)}
        recorded_mapping = {str(key): int(value) for key, value in artifact.get("source_to_llm_mapping", {}).items()}
        if recorded_mapping != expected_mapping:
            raise RuntimeError(f"Calibration mapping mismatch: artifact={recorded_mapping}, expected={expected_mapping}.")
        expected_type = getattr(config, "residual_predictor_type", None)
        if str(expected_type or "").lower() == "auto":
            expected_type = None
        predictor, checkpoint = load_residual_predictor_checkpoint(
            checkpoint_path, expected_type=expected_type,
            expected_hidden_size=int(getattr(config, "hidden_size")), expected_source_layers=source_layers,
        )
        expected_hash = artifact.get("predictor_state_sha256")
        if expected_hash and str(expected_hash) != predictor_state_sha256(checkpoint["predictor"]):
            raise RuntimeError("Calibration artifact predictor-state SHA256 does not match residual predictor checkpoint.")
        gammas = [float(getattr(config, f"predicted_residual_gamma_layer{index}", 1.0)) for index in range(len(llm_layers))]
        return cls(
            predictor, source_layers=source_layers, llm_layers=llm_layers, gamma_layers=gammas,
            control="none", alphas=artifact["alphas"], artifact_metadata={"path": str(artifact_path)},
        )

    def forward(self, inputs_embeds, visual_metadata):
        residuals = super().forward(inputs_embeds, visual_metadata)
        for batch_index, metadata in enumerate(self._metadata_items(visual_metadata)):
            _, ordered_indices = self._extract_sample_tokens(inputs_embeds, batch_index, metadata)
            frames = int(ordered_indices.numel() // 196)
            for source_layer, llm_layer in zip(self.source_layers, self.llm_layers):
                alpha = getattr(self, f"calibration_alpha_{source_layer}").to(inputs_embeds.device, inputs_embeds.dtype)
                alpha = alpha.reshape(1, 1).expand(frames * 196, 1) if alpha.dim() == 0 else alpha.expand(frames, -1, -1).reshape(-1, 1)
                current = residuals[int(llm_layer)][batch_index].index_select(0, ordered_indices)
                residuals[int(llm_layer)][batch_index].index_copy_(0, ordered_indices, current * alpha)
        self.last_debug.update({"control": "calibrated", "calibration_artifact": self.artifact_metadata.get("path")})
        return residuals

def checkpoint_json_summary(checkpoint: Mapping[str, object]) -> str:
    """Small stable summary for logs without serialising predictor tensors."""
    return json.dumps({
        "architecture": checkpoint.get("architecture"),
        "teacher_checkpoint": checkpoint.get("teacher_checkpoint"),
        "teacher_config_hash": checkpoint.get("teacher_config_hash"),
    }, sort_keys=True)
