"""Raw SigLIP-to-CUT3R predictors and a differentiable frozen teacher path.

This module deliberately does not import a language model.  It is shared by
the offline distillation trainer and the online residual adapter so the exact
SpatialStack resize/projector/scale operation has one implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from llava.model.cut3r_spatialstack import Cut3RSpatialStackMerger


SOURCE_LAYERS: Tuple[int, ...] = (6, 9, 12)
QWEN_LAYERS: Tuple[int, ...] = (0, 1, 2)
RAW_SIGLIP_DIM = 1152
CUT3R_DIM = 768
GRID_SIZE = 27


def _load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw(x: torch.Tensor, valid_frame_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if x.dim() != 4 or tuple(x.shape[2:]) != (GRID_SIZE * GRID_SIZE, RAW_SIGLIP_DIM):
        raise ValueError(
            "Raw SigLIP tokens must be [B,F,729,1152], got "
            f"{tuple(x.shape)}."
        )
    batch, frames = x.shape[:2]
    if valid_frame_mask is None:
        return torch.ones(batch, frames, dtype=torch.bool, device=x.device)
    if tuple(valid_frame_mask.shape) != (batch, frames):
        raise ValueError(
            "valid_frame_mask must have shape [B,F], got "
            f"{tuple(valid_frame_mask.shape)} for {tuple(x.shape)}."
        )
    return valid_frame_mask.to(device=x.device, dtype=torch.bool)


class PatchCoordinateResampler(nn.Module):
    """Fixed, auditable 27x27 coordinate transform with no learned state.

    ``source_centers`` and ``target_centers`` are normalized pixel centres in
    [0, 1].  Identity is represented explicitly and avoids interpolation.
    """

    def __init__(
        self,
        source_centers: Optional[Sequence[float]] = None,
        target_centers: Optional[Sequence[float]] = None,
        *,
        status: str = "EXACT_PATCH_ALIGNMENT",
    ):
        super().__init__()
        self.status = str(status)
        source = torch.tensor(
            source_centers if source_centers is not None else [(i + 0.5) / GRID_SIZE for i in range(GRID_SIZE)],
            dtype=torch.float32,
        )
        target = torch.tensor(
            target_centers if target_centers is not None else [(i + 0.5) / GRID_SIZE for i in range(GRID_SIZE)],
            dtype=torch.float32,
        )
        if source.shape != (GRID_SIZE,) or target.shape != (GRID_SIZE,):
            raise ValueError("Patch coordinate resampling requires 27 source and 27 target centres.")
        if not torch.all(source[1:] > source[:-1]) or not torch.all(target[1:] > target[:-1]):
            raise ValueError("Patch centres must be strictly increasing.")
        self.register_buffer("source_centers", source, persistent=True)
        self.register_buffer("target_centers", target, persistent=True)
        self.identity = bool(torch.allclose(source, target, atol=0.0, rtol=0.0))

    def metadata(self) -> Dict[str, object]:
        payload = {
            "status": self.status,
            "source_centers": self.source_centers.cpu().tolist(),
            "target_centers": self.target_centers.cpu().tolist(),
            "identity": self.identity,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x
        _validate_raw(x, None)
        # Convert target normalized centres to fractional source-grid indices,
        # then to grid_sample's align_corners=True coordinate space.
        source = self.source_centers.to(device=x.device)
        target = self.target_centers.to(device=x.device)
        source_index = (target[:, None] - source[0]) / (source[-1] - source[0]) * (GRID_SIZE - 1)
        # The source grid is separable.  Keep exact border handling explicit.
        source_index = source_index[:, 0].clamp(0, GRID_SIZE - 1)
        coord = 2.0 * source_index / float(GRID_SIZE - 1) - 1.0
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        batch, frames, _, hidden = x.shape
        image = x.reshape(batch * frames, GRID_SIZE, GRID_SIZE, hidden).permute(0, 3, 1, 2)
        sampled = F.grid_sample(
            image.float(), grid.expand(batch * frames, -1, -1, -1), mode="bilinear",
            padding_mode="border", align_corners=True,
        ).to(dtype=x.dtype)
        return sampled.permute(0, 2, 3, 1).reshape(batch, frames, GRID_SIZE * GRID_SIZE, hidden)


class RawTokenMLPPredictor(nn.Module):
    architecture_name = "raw_cut3r_token_mlp"

    def __init__(
        self,
        hidden_dim: int = 1024,
        residual_blocks: int = 1,
        source_layers: Sequence[int] = SOURCE_LAYERS,
        output_init_std: float = 1e-3,
        alignment: Optional[PatchCoordinateResampler] = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual_blocks = int(residual_blocks)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        self.alignment = alignment or PatchCoordinateResampler()
        self.input_norm = nn.LayerNorm(RAW_SIGLIP_DIM)
        self.input_proj = nn.Linear(RAW_SIGLIP_DIM, self.hidden_dim)
        self.activation = nn.GELU()
        self.residual_norms = nn.ModuleList(nn.LayerNorm(self.hidden_dim) for _ in range(self.residual_blocks))
        self.residual_linears = nn.ModuleList(nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.residual_blocks))
        self.heads = nn.ModuleDict({str(layer): nn.Linear(self.hidden_dim, CUT3R_DIM) for layer in self.source_layers})
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> Dict[str, object]:
        return {
            "predictor_type": self.architecture_name,
            "hidden_dim": self.hidden_dim,
            "residual_blocks": self.residual_blocks,
            "source_layers": list(self.source_layers),
            "grid_size": GRID_SIZE,
            "input_dim": RAW_SIGLIP_DIM,
            "output_dim": CUT3R_DIM,
            "alignment": self.alignment.metadata(),
        }

    def forward(self, x: torch.Tensor, valid_frame_mask: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        mask = _validate_raw(x, valid_frame_mask)
        x = self.alignment(x)
        x = self.activation(self.input_proj(self.input_norm(x)))
        for norm, linear in zip(self.residual_norms, self.residual_linears):
            x = x + linear(self.activation(norm(x)))
        x = x * mask[:, :, None, None].to(dtype=x.dtype)
        return {layer: self.heads[str(layer)](x) * mask[:, :, None, None].to(dtype=x.dtype) for layer in self.source_layers}


class RawSpatialResidualBlock(nn.Module):
    """Exact pre-LN depthwise-3x3/GELU/pointwise spatial residual block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, hidden = x.shape
        if patches != GRID_SIZE * GRID_SIZE:
            raise ValueError(f"Spatial block needs 729 patches, got {patches}.")
        residual = x
        x = self.norm(x).reshape(batch * frames, GRID_SIZE, GRID_SIZE, hidden).permute(0, 3, 1, 2)
        x = self.pointwise(self.activation(self.depthwise(x)))
        x = x.permute(0, 2, 3, 1).reshape(batch, frames, patches, hidden)
        return residual + x


class RawSpatialTemporalPredictor(nn.Module):
    architecture_name = "raw_cut3r_spatial_temporal"

    def __init__(
        self,
        hidden_dim: int = CUT3R_DIM,
        spatial_blocks: int = 2,
        temporal_layers: int = 4,
        temporal_heads: int = 12,
        temporal_ffn_dim: int = 3072,
        temporal_dropout: float = 0.0,
        temporal_max_frames: int = 64,
        adapter_dim: int = 192,
        source_layers: Sequence[int] = SOURCE_LAYERS,
        output_init_std: float = 1e-3,
        alignment: Optional[PatchCoordinateResampler] = None,
    ):
        super().__init__()
        self.hidden_dim, self.spatial_blocks_count = int(hidden_dim), int(spatial_blocks)
        self.temporal_layers, self.temporal_heads = int(temporal_layers), int(temporal_heads)
        self.temporal_ffn_dim, self.temporal_dropout = int(temporal_ffn_dim), float(temporal_dropout)
        self.temporal_max_frames, self.adapter_dim = int(temporal_max_frames), int(adapter_dim)
        self.source_layers = tuple(int(layer) for layer in source_layers)
        if self.hidden_dim % self.temporal_heads:
            raise ValueError("hidden_dim must be divisible by temporal_heads.")
        self.alignment = alignment or PatchCoordinateResampler()
        self.input_norm = nn.LayerNorm(RAW_SIGLIP_DIM)
        self.input_proj = nn.Linear(RAW_SIGLIP_DIM, self.hidden_dim)
        self.spatial = nn.ModuleList(RawSpatialResidualBlock(self.hidden_dim) for _ in range(self.spatial_blocks_count))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(self.temporal_max_frames, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=self.temporal_heads, dim_feedforward=self.temporal_ffn_dim,
            dropout=self.temporal_dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=self.temporal_layers)
        self.adapter_norms = nn.ModuleDict({str(layer): nn.LayerNorm(self.hidden_dim) for layer in self.source_layers})
        self.adapter_down = nn.ModuleDict({str(layer): nn.Linear(self.hidden_dim, self.adapter_dim) for layer in self.source_layers})
        self.adapter_up = nn.ModuleDict({str(layer): nn.Linear(self.adapter_dim, self.hidden_dim) for layer in self.source_layers})
        self.heads = nn.ModuleDict({str(layer): nn.Linear(self.hidden_dim, CUT3R_DIM) for layer in self.source_layers})
        nn.init.normal_(self.temporal_pos_embed, mean=0.0, std=0.02)
        for head in self.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=float(output_init_std))
            nn.init.zeros_(head.bias)

    def architecture_config(self) -> Dict[str, object]:
        return {
            "predictor_type": self.architecture_name, "hidden_dim": self.hidden_dim,
            "spatial_blocks": self.spatial_blocks_count, "temporal_layers": self.temporal_layers,
            "temporal_heads": self.temporal_heads, "temporal_ffn_dim": self.temporal_ffn_dim,
            "temporal_dropout": self.temporal_dropout, "temporal_max_frames": self.temporal_max_frames,
            "adapter_type": "residual_bottleneck", "adapter_dim": self.adapter_dim,
            "source_layers": list(self.source_layers), "grid_size": GRID_SIZE,
            "input_dim": RAW_SIGLIP_DIM, "output_dim": CUT3R_DIM,
            "alignment": self.alignment.metadata(),
        }

    def forward(self, x: torch.Tensor, valid_frame_mask: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        mask = _validate_raw(x, valid_frame_mask)
        batch, frames = x.shape[:2]
        if frames > self.temporal_max_frames:
            raise ValueError(f"F={frames} exceeds temporal_max_frames={self.temporal_max_frames}.")
        x = self.input_proj(self.input_norm(self.alignment(x)))
        for block in self.spatial:
            x = block(x)
        temporal = x.permute(0, 2, 1, 3).reshape(batch * GRID_SIZE * GRID_SIZE, frames, self.hidden_dim)
        temporal = temporal + self.temporal_pos_embed[:frames].to(device=x.device, dtype=x.dtype).unsqueeze(0)
        padding = (~mask).unsqueeze(1).expand(batch, GRID_SIZE * GRID_SIZE, frames).reshape(-1, frames)
        temporal = self.temporal(temporal, src_key_padding_mask=padding)
        x = temporal.reshape(batch, GRID_SIZE * GRID_SIZE, frames, self.hidden_dim).permute(0, 2, 1, 3)
        mask_float = mask[:, :, None, None].to(dtype=x.dtype)
        result = {}
        for layer in self.source_layers:
            adapter = self.adapter_up[str(layer)](F.gelu(self.adapter_down[str(layer)](self.adapter_norms[str(layer)](x))))
            result[layer] = self.heads[str(layer)](x + adapter) * mask_float
        return result


def build_raw_cut3r_predictor(predictor_type: str, **kwargs) -> nn.Module:
    predictor_type = str(predictor_type).strip().lower()
    kwargs.pop("predictor_type", None)
    if predictor_type == RawTokenMLPPredictor.architecture_name:
        return RawTokenMLPPredictor(**kwargs)
    if predictor_type == RawSpatialTemporalPredictor.architecture_name:
        return RawSpatialTemporalPredictor(**kwargs)
    raise ValueError(f"Unknown raw CUT3R predictor type {predictor_type!r}.")


def raw_predictor_checkpoint_payload(predictor: nn.Module, **metadata) -> Dict[str, object]:
    if not hasattr(predictor, "architecture_config"):
        raise TypeError("Raw predictor must define architecture_config().")
    return {
        "format": "raw_siglip_cut3r_predictor_v1",
        "architecture": predictor.architecture_config(),
        "predictor": predictor.state_dict(),
        "trainable_parameter_count": sum(parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad),
        **metadata,
    }


def raw_predictor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Deterministic tensor-state identity used for evaluation deduplication."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_raw_predictor_checkpoint(path: str | Path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "raw_siglip_cut3r_predictor_v1":
        raise RuntimeError(f"Not a raw SigLIP/CUT3R checkpoint: {path}")
    architecture = dict(checkpoint["architecture"])
    alignment = architecture.pop("alignment", None)
    if alignment is not None:
        architecture["alignment"] = PatchCoordinateResampler(
            alignment["source_centers"], alignment["target_centers"], status=alignment.get("status", "EXACT_PATCH_ALIGNMENT")
        )
    predictor_type = architecture.pop("predictor_type")
    predictor = build_raw_cut3r_predictor(predictor_type, **architecture)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    return predictor, checkpoint


class FrozenSpatialStackPostprocessor(nn.Module):
    """Frozen oracle postprocessing which retains autograd through inputs."""

    def __init__(self, merger: Cut3RSpatialStackMerger, source_layers: Sequence[int] = SOURCE_LAYERS):
        super().__init__()
        self.merger = merger
        self.source_layers = tuple(int(layer) for layer in source_layers)
        for parameter in self.merger.parameters():
            parameter.requires_grad_(False)
        self.merger.eval()

    @classmethod
    def from_teacher_checkpoint(cls, checkpoint: str | Path, device=None, dtype=None):
        checkpoint = Path(checkpoint)
        config_path = checkpoint / "config.json"
        state_path = checkpoint / "non_lora_trainables.bin"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError("Teacher requires config.json and non_lora_trainables.bin.")
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        config = SimpleNamespace(**config_dict)
        merger = Cut3RSpatialStackMerger(config)
        raw = _load(state_path)
        marker = "cut3r_spatialstack_merger."
        state = {key.split(marker, 1)[1]: value for key, value in raw.items() if marker in key}
        if not state:
            raise RuntimeError(f"No SpatialStack merger state found in {state_path}.")
        merger.load_state_dict(state, strict=True)
        if device is not None or dtype is not None:
            merger.to(device=device, dtype=dtype)
        configured_layers = getattr(config, "cut3r_spatialstack_layers", SOURCE_LAYERS)
        if isinstance(configured_layers, str):
            configured_layers = [int(item.strip()) for item in configured_layers.split(",") if item.strip()]
        result = cls(merger, configured_layers)
        result.teacher_checkpoint = str(checkpoint)
        result.teacher_config_hash = _hash_file(config_path)
        return result

    def train(self, mode: bool = True):  # Frozen teacher must stay deterministic.
        super().train(False)
        return self

    def forward(self, features: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Postprocess predictions without ``no_grad`` so inputs receive grads."""
        result = {}
        for layer in self.source_layers:
            raw = features[int(layer)]
            if raw.dim() != 4 or tuple(raw.shape[2:]) != (729, CUT3R_DIM):
                raise ValueError(f"CUT3R layer {layer} must be [B,F,729,768], got {tuple(raw.shape)}.")
            batch, frames = raw.shape[:2]
            aligned = torch.stack(
                [self.merger.resize_square_grid(frame, 196) for frame in raw.reshape(batch * frames, 729, CUT3R_DIM)], dim=0
            )
            projected = self.merger.branches[str(layer)](aligned)
            result[int(layer)] = (projected * self.merger.residual_scale).reshape(batch, frames, 196, -1)
        return result

    @torch.no_grad()
    def targets(self, features: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {layer: value.detach() for layer, value in self.forward(features).items()}
