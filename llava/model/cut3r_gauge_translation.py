"""Trainable CUT3R global-translation gauge adapters.

This module is deliberately independent from the VLM and SpatialStack forward
paths.  It transforms the three CUT3R patch streams that SpatialStack retains
and, separately, the final CUT3R pose token used only for frozen-head training.
All geometry reductions in this file are performed in FP32.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LAYERS: Tuple[int, ...] = (6, 9, 12)


@dataclass(frozen=True)
class GaugeTranslationConfig:
    enabled: bool = False
    token_dim: int = 768
    trunk_hidden_dim: int = 128
    trunk_output_dim: int = 256
    patch_condition_dim: int = 256
    pose_condition_dim: int = 128
    patch_bottleneck_dim: int = 192
    pose_bottleneck_dim: int = 64
    use_pose_context_for_patch_adapter: bool = False

    def __post_init__(self) -> None:
        if self.use_pose_context_for_patch_adapter:
            raise ValueError(
                "Pose context for patch adapters is intentionally unsupported in v1; "
                "composition is defined on patches and delta conditioning only."
            )
        if self.pose_condition_dim not in (64, 128):
            raise ValueError("pose_condition_dim must be 64 or 128")
        for name, value in asdict(self).items():
            if name not in {"enabled", "use_pose_context_for_patch_adapter"} and int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


class GaugeTranslationOutput(NamedTuple):
    patch6: torch.Tensor
    patch9: torch.Tensor
    patch12: torch.Tensor
    pose12: torch.Tensor

    def patches(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.patch6, self.patch9, self.patch12


class _FiLMResidualAdapter(nn.Module):
    def __init__(self, token_dim: int, bottleneck_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.norm = nn.LayerNorm(self.token_dim)
        self.down = nn.Linear(self.token_dim, int(bottleneck_dim))
        self.film = nn.Linear(int(condition_dim), 2 * int(bottleneck_dim))
        self.up = nn.Linear(int(bottleneck_dim), self.token_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must be [B,F,N,D], got {tuple(tokens.shape)}")
        if int(tokens.shape[-1]) != self.token_dim:
            raise ValueError(f"token dim must be {self.token_dim}, got {int(tokens.shape[-1])}")
        if condition.ndim != 2 or int(condition.shape[0]) != int(tokens.shape[0]):
            raise ValueError(
                f"condition must be [B,C] with B={int(tokens.shape[0])}, got {tuple(condition.shape)}"
            )
        hidden = self.down(self.norm(tokens))
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        gamma = gamma[:, None, None, :].to(dtype=hidden.dtype)
        beta = beta[:, None, None, :].to(dtype=hidden.dtype)
        update = self.up(F.silu((1.0 + gamma) * hidden + beta))
        return tokens + update


class GaugeTranslationModel(nn.Module):
    """Patch-self-contained translation with separate patch/pose conditioning."""

    def __init__(self, config: GaugeTranslationConfig) -> None:
        super().__init__()
        self.config = config
        self.translation_trunk = nn.Sequential(
            nn.Linear(4, config.trunk_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.trunk_hidden_dim, config.trunk_output_dim),
            nn.SiLU(),
        )
        self.patch_conditioning_projection = nn.Sequential(
            nn.Linear(config.trunk_output_dim, config.patch_condition_dim), nn.SiLU()
        )
        self.pose_conditioning_projection = nn.Sequential(
            nn.Linear(config.trunk_output_dim, config.pose_condition_dim), nn.SiLU()
        )
        self.patch_adapters = nn.ModuleDict(
            {
                str(layer): _FiLMResidualAdapter(
                    config.token_dim, config.patch_bottleneck_dim, config.patch_condition_dim
                )
                for layer in LAYERS
            }
        )
        self.pose_adapter = _FiLMResidualAdapter(
            config.token_dim, config.pose_bottleneck_dim, config.pose_condition_dim
        )

    @staticmethod
    def conditioning_input(delta: torch.Tensor, scene_scale: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 2 or int(delta.shape[-1]) != 3:
            raise ValueError(f"delta must be [B,3], got {tuple(delta.shape)}")
        if scene_scale.ndim == 2 and int(scene_scale.shape[-1]) == 1:
            scene_scale = scene_scale[:, 0]
        if scene_scale.ndim != 1 or int(scene_scale.shape[0]) != int(delta.shape[0]):
            raise ValueError(f"scene_scale must be [B], got {tuple(scene_scale.shape)}")
        if not torch.isfinite(scene_scale).all() or torch.any(scene_scale <= 0):
            raise ValueError("scene_scale must be finite and positive")
        normalized = delta / scene_scale[:, None].to(device=delta.device, dtype=delta.dtype)
        return torch.cat([normalized, normalized.norm(dim=-1, keepdim=True)], dim=-1)

    def encode_conditions(
        self, delta: torch.Tensor, scene_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trunk = self.translation_trunk(self.conditioning_input(delta, scene_scale))
        return self.patch_conditioning_projection(trunk), self.pose_conditioning_projection(trunk)

    @staticmethod
    def _validate_shapes(
        patch6: torch.Tensor,
        patch9: torch.Tensor,
        patch12: torch.Tensor,
        pose12: torch.Tensor,
    ) -> None:
        if patch6.shape != patch9.shape or patch6.shape != patch12.shape:
            raise ValueError(
                f"patch streams must have identical shapes, got {patch6.shape}, {patch9.shape}, {patch12.shape}"
            )
        if patch6.ndim != 4 or int(patch6.shape[2]) != 729:
            raise ValueError(f"patch streams must be [B,F,729,D], got {tuple(patch6.shape)}")
        expected_pose = (int(patch6.shape[0]), int(patch6.shape[1]), 1, int(patch6.shape[3]))
        if tuple(pose12.shape) != expected_pose:
            raise ValueError(f"pose12 must have shape {expected_pose}, got {tuple(pose12.shape)}")

    def forward(
        self,
        patch6: torch.Tensor,
        patch9: torch.Tensor,
        patch12: torch.Tensor,
        pose12: torch.Tensor,
        delta: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> GaugeTranslationOutput:
        self._validate_shapes(patch6, patch9, patch12, pose12)
        patch_condition, pose_condition = self.encode_conditions(delta, scene_scale)
        return GaugeTranslationOutput(
            self.patch_adapters["6"](patch6, patch_condition),
            self.patch_adapters["9"](patch9, patch_condition),
            self.patch_adapters["12"](patch12, patch_condition),
            self.pose_adapter(pose12, pose_condition),
        )

    def transform_patches(
        self,
        patch6: torch.Tensor,
        patch9: torch.Tensor,
        patch12: torch.Tensor,
        delta: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if patch6.shape != patch9.shape or patch6.shape != patch12.shape:
            raise ValueError("patch streams must have identical shapes")
        condition, _ = self.encode_conditions(delta, scene_scale)
        return tuple(
            self.patch_adapters[str(layer)](tokens, condition)
            for layer, tokens in zip(LAYERS, (patch6, patch9, patch12))
        )

    def set_trainable_stage(self, stage: str) -> Sequence[str]:
        stage = str(stage).lower()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if stage == "a":
            modules: Iterable[nn.Module] = (
                self.translation_trunk,
                self.patch_conditioning_projection,
                self.patch_adapters,
            )
        elif stage == "b":
            modules = (self.pose_conditioning_projection, self.pose_adapter)
        elif stage == "c":
            modules = (self,)
        elif stage in {"frozen", "eval"}:
            modules = ()
        else:
            raise ValueError(f"unknown translator stage {stage!r}")
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]


class _ProbeBase(nn.Module):
    def __init__(self, token_dim: int = 768) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(token_dim) for _ in LAYERS])

    def fused(self, patches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(patches) != 3 or any(tensor.shape != patches[0].shape for tensor in patches[1:]):
            raise ValueError("probe requires three same-shaped patch streams")
        return torch.cat([norm(tensor) for norm, tensor in zip(self.norms, patches)], dim=-1)


class PatchGeometryTrainProbe(_ProbeBase):
    def __init__(self, token_dim: int = 768, hidden_dim: int = 128) -> None:
        super().__init__(token_dim)
        self.head = nn.Sequential(nn.Linear(3 * token_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def forward(self, patch6: torch.Tensor, patch9: torch.Tensor, patch12: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused((patch6, patch9, patch12)))


class PatchGeometryEvalProbe(_ProbeBase):
    def __init__(self, token_dim: int = 768) -> None:
        super().__init__(token_dim)
        self.head = nn.Linear(3 * token_dim, 3)

    def forward(self, patch6: torch.Tensor, patch9: torch.Tensor, patch12: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused((patch6, patch9, patch12)))


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def build_teacher_mask(
    reference_points: torch.Tensor,
    camera_points: torch.Tensor,
    reference_confidence: Optional[torch.Tensor] = None,
    self_confidence: Optional[torch.Tensor] = None,
    reference_confidence_threshold: float = 0.0,
    self_confidence_threshold: float = 0.0,
) -> torch.Tensor:
    with torch.autocast(device_type=reference_points.device.type, enabled=False):
        ref = reference_points.float()
        cam = camera_points.float()
        if ref.shape != cam.shape or int(ref.shape[-1]) != 3:
            raise ValueError(f"teacher point maps must have identical [...,3] shapes, got {ref.shape} and {cam.shape}")
        mask = torch.isfinite(ref).all(dim=-1) & torch.isfinite(cam).all(dim=-1) & (cam[..., 2] > 0)
        if reference_confidence is not None:
            conf = reference_confidence.float()
            mask &= torch.isfinite(conf) & (conf > float(reference_confidence_threshold))
        if self_confidence is not None:
            conf = self_confidence.float()
            mask &= torch.isfinite(conf) & (conf > float(self_confidence_threshold))
        return mask


def robust_scene_scale(reference_points: torch.Tensor, teacher_mask: torch.Tensor, minimum: float = 1e-3) -> torch.Tensor:
    with torch.autocast(device_type=reference_points.device.type, enabled=False):
        points = reference_points.float()[teacher_mask.bool()]
        if points.numel() == 0:
            raise ValueError("teacher mask contains no valid points")
        center = points.median(dim=0).values
        scale = torch.linalg.vector_norm(points - center, dim=-1).median()
        if not torch.isfinite(scale):
            raise ValueError("scene scale is non-finite")
        return scale.clamp_min(float(minimum))


def expected_patch_positions(height: int = 27, width: int = 27, device: Optional[torch.device] = None) -> torch.Tensor:
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    return torch.cartesian_prod(y, x)


def validate_patch_positions(pos: torch.Tensor, height: int = 27, width: int = 27) -> Dict[str, object]:
    positions = pos.detach().cpu()
    if positions.ndim == 3:
        frames = positions
    elif positions.ndim == 2:
        frames = positions.unsqueeze(0)
    else:
        raise ValueError(f"pos must be [F,N,2] or [N,2], got {tuple(pos.shape)}")
    expected = expected_patch_positions(height, width)
    if int(frames.shape[1]) != height * width or int(frames.shape[2]) != 2:
        raise ValueError(f"pos must contain {height*width} [y,x] pairs, got {tuple(frames.shape)}")
    integral = not torch.is_floating_point(frames) or torch.equal(frames, frames.round())
    per_frame_matches = [bool(torch.equal(frame.long(), expected)) for frame in frames]
    unique = [int(torch.unique(frame.long(), dim=0).shape[0]) for frame in frames]
    corners = {index: frames[0, index].long().tolist() for index in (0, 26, 702, 728)}
    transforms = {
        "transpose": expected[:, [1, 0]],
        "horizontal_flip": torch.stack([expected[:, 0], width - 1 - expected[:, 1]], dim=-1),
        "vertical_flip": torch.stack([height - 1 - expected[:, 0], expected[:, 1]], dim=-1),
        "both_flips": torch.stack([height - 1 - expected[:, 0], width - 1 - expected[:, 1]], dim=-1),
    }
    transform_matches = {
        name: bool(torch.equal(frames[0].long(), candidate)) for name, candidate in transforms.items()
    }
    passed = bool(integral and all(per_frame_matches) and all(count == height * width for count in unique))
    return {
        "passed": passed,
        "integral": bool(integral),
        "per_frame_row_major": per_frame_matches,
        "unique_counts": unique,
        "corners": corners,
        "alternative_transform_matches": transform_matches,
        "coordinate_convention": "[y,x]",
    }


def pool_points_by_positions(
    points: torch.Tensor,
    mask: torch.Tensor,
    pos: torch.Tensor,
    grid_shape: Tuple[int, int] = (27, 27),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked averaging using explicit token [y,x] cells."""
    with torch.autocast(device_type=points.device.type, enabled=False):
        pts = points.float()
        valid = mask.bool()
        positions = pos.long()
        if pts.ndim != 4 or int(pts.shape[-1]) != 3:
            raise ValueError(f"points must be [F,H,W,3], got {tuple(pts.shape)}")
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(int(pts.shape[0]), -1, -1)
        if int(positions.shape[0]) != int(pts.shape[0]):
            raise ValueError("position and point-map frame counts differ")
        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        height, width = int(pts.shape[1]), int(pts.shape[2])
        pooled, pooled_mask = [], []
        for frame_idx in range(int(pts.shape[0])):
            frame_values, frame_masks = [], []
            for y, x in positions[frame_idx].tolist():
                y0, y1 = math.floor(y * height / gh), math.floor((y + 1) * height / gh)
                x0, x1 = math.floor(x * width / gw), math.floor((x + 1) * width / gw)
                cell_mask = valid[frame_idx, y0:y1, x0:x1]
                if bool(cell_mask.any()):
                    frame_values.append(pts[frame_idx, y0:y1, x0:x1][cell_mask].mean(dim=0))
                    frame_masks.append(torch.tensor(True, device=pts.device))
                else:
                    frame_values.append(torch.zeros(3, dtype=torch.float32, device=pts.device))
                    frame_masks.append(torch.tensor(False, device=pts.device))
            pooled.append(torch.stack(frame_values))
            pooled_mask.append(torch.stack(frame_masks))
        return torch.stack(pooled), torch.stack(pooled_mask)


def pool_points_adaptive(
    points: torch.Tensor, mask: torch.Tensor, grid_shape: Tuple[int, int] = (27, 27)
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=points.device.type, enabled=False):
        pts = points.float().permute(0, 3, 1, 2)
        valid = mask.float().unsqueeze(1)
        values = F.adaptive_avg_pool2d(torch.where(valid.bool(), pts, torch.zeros_like(pts)), grid_shape)
        weights = F.adaptive_avg_pool2d(valid, grid_shape)
        pooled = (values / weights.clamp_min(1e-8)).permute(0, 2, 3, 1).reshape(points.shape[0], -1, 3)
        pooled_mask = weights[:, 0].reshape(points.shape[0], -1) > 0
        pooled = torch.where(pooled_mask.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return pooled, pooled_mask


def normalized_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scene_scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    beta: float = 0.05,
) -> torch.Tensor:
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.float()
        tgt = target.float()
        scale = scene_scale.float()
        while scale.ndim < pred.ndim:
            scale = scale.unsqueeze(-1)
        diff = (pred - tgt) / scale.clamp_min(1e-8)
        if mask is not None:
            expanded = mask.bool()
            while expanded.ndim < diff.ndim:
                expanded = expanded.unsqueeze(-1)
            if not torch.isfinite(diff[expanded.expand_as(diff)]).all():
                raise FloatingPointError("non-finite transformed geometry inside the fixed teacher mask")
            diff = diff[expanded.expand_as(diff)]
        elif not torch.isfinite(diff).all():
            raise FloatingPointError("non-finite normalized geometry")
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=float(beta), reduction="mean")


def quaternion_rotation_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=first.device.type, enabled=False):
        q1 = F.normalize(first.float(), dim=-1, eps=1e-8)
        q2 = F.normalize(second.float(), dim=-1, eps=1e-8)
        dot = (q1 * q2).sum(dim=-1).clamp(-1.0, 1.0)
        return (1.0 - dot.square()).mean()


def quaternion_geodesic_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=first.device.type, enabled=False):
        q1 = F.normalize(first.float(), dim=-1, eps=1e-8)
        q2 = F.normalize(second.float(), dim=-1, eps=1e-8)
        dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0 - 1e-7)
        return torch.rad2deg(2.0 * torch.acos(dot))


def relative_residual(original: torch.Tensor, transformed: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    with torch.autocast(device_type=original.device.type, enabled=False):
        numerator = (transformed.float() - original.float()).square().sum(dim=-1)
        denominator = original.float().square().sum(dim=-1).clamp_min(float(epsilon))
        return (numerator / denominator).mean()


def robust_translation_metrics(
    original: torch.Tensor,
    transformed: torch.Tensor,
    teacher_mask: torch.Tensor,
    target_delta: torch.Tensor,
    scene_scale: torch.Tensor,
) -> Dict[str, float]:
    with torch.autocast(device_type=original.device.type, enabled=False):
        base, moved, mask = original.float(), transformed.float(), teacher_mask.bool()
        selected = moved[mask]
        invalid_ratio = float((~torch.isfinite(selected).all(dim=-1)).float().mean().item()) if selected.numel() else 1.0
        if invalid_ratio > 0.0 or selected.numel() == 0:
            return {"valid": False, "invalid_ratio": invalid_ratio, "normalized_vector_error": float("inf")}
        differences = (moved - base)[mask]
        estimate = differences.median(dim=0).values
        delta = target_delta.float().reshape(-1, 3)[0]
        scale = float(scene_scale.float().reshape(-1)[0].item())
        delta_norm = float(delta.norm().item())
        estimate_norm = float(estimate.norm().item())
        cosine = float(F.cosine_similarity(estimate[None], delta[None], dim=-1, eps=1e-8).item()) if delta_norm > 0 else float("nan")
        residual = torch.linalg.vector_norm(differences - estimate, dim=-1)
        return {
            "valid": True,
            "invalid_ratio": 0.0,
            "estimate_x": float(estimate[0].item()),
            "estimate_y": float(estimate[1].item()),
            "estimate_z": float(estimate[2].item()),
            "cosine": cosine,
            "magnitude_ratio": estimate_norm / max(delta_norm, 1e-8),
            "normalized_vector_error": float(torch.linalg.vector_norm(estimate - delta).item()) / max(scale, 1e-8),
            "axis_error_x": float((estimate[0] - delta[0]).item()) / max(scale, 1e-8),
            "axis_error_y": float((estimate[1] - delta[1]).item()) / max(scale, 1e-8),
            "axis_error_z": float((estimate[2] - delta[2]).item()) / max(scale, 1e-8),
            "residual_median": float(residual.median().item()),
            "residual_p95": float(torch.quantile(residual, 0.95).item()),
        }


def sample_video_translations(
    batch_size: int,
    scene_scale: torch.Tensor,
    progress: float,
    *,
    zero_probability: float = 0.10,
    axis_probability: float = 0.20,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = scene_scale.device
    progress = max(0.0, min(1.0, float(progress) / 0.40))
    low = 0.02 + progress * (0.05 - 0.02)
    high = 0.15 + progress * (0.50 - 0.15)
    directions = torch.randn(batch_size, 3, device=device, generator=generator)
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    axis_draw = torch.rand(batch_size, device=device, generator=generator) < float(axis_probability)
    if bool(axis_draw.any()):
        axes = torch.randint(0, 3, (batch_size,), device=device, generator=generator)
        signs = torch.where(
            torch.rand(batch_size, device=device, generator=generator) < 0.5,
            -torch.ones(batch_size, device=device),
            torch.ones(batch_size, device=device),
        )
        axis_directions = torch.zeros_like(directions)
        axis_directions.scatter_(1, axes[:, None], signs[:, None])
        directions = torch.where(axis_draw[:, None], axis_directions, directions)
    alpha = low + (high - low) * torch.rand(batch_size, device=device, generator=generator)
    delta = directions * alpha[:, None] * scene_scale.reshape(batch_size, 1)
    zero_draw = torch.rand(batch_size, device=device, generator=generator) < float(zero_probability)
    return torch.where(zero_draw[:, None], torch.zeros_like(delta), delta)


def composite_checkpoint_score(metrics: Mapping[str, float]) -> float:
    return (
        0.35 * float(metrics["full_normalized_vector_error"])
        + 0.30 * float(metrics["q_eval_normalized_vector_error"])
        + 0.15 * float(metrics["normalized_self_drift"])
        + 0.10 * float(metrics["quaternion_rotation_loss"])
        + 0.10 * float(metrics["structural_consistency_error"])
    )


def checkpoint_feasibility(
    metrics: Mapping[str, float], thresholds: Optional[Mapping[str, float]] = None
) -> Tuple[bool, Sequence[str]]:
    limits = {
        "max_self_drift_ratio": 0.25,
        "max_pose_rotation_degrees": 5.0,
        "max_patch_residual_p95": 0.25,
        "max_pose_residual_p95": 0.25,
        "max_new_nonpositive_self_depth_ratio": 0.01,
        "max_confidence_relative_drop": 0.25,
    }
    limits.update(dict(thresholds or {}))
    failures = []
    checks = (
        (float(metrics.get("full_cosine", -1.0)) > 0.8, "full_cosine"),
        (0.5 <= float(metrics.get("full_magnitude_ratio", -1.0)) <= 1.5, "full_magnitude_ratio"),
        (float(metrics.get("q_eval_cosine", -1.0)) > 0.5, "q_eval_cosine"),
        (float(metrics.get("q_eval_magnitude_ratio", -1.0)) > 0.10, "q_eval_magnitude_ratio"),
        (float(metrics.get("normalized_self_drift", float("inf"))) <= limits["max_self_drift_ratio"], "self_drift"),
        (float(metrics.get("pose_rotation_degrees", float("inf"))) <= limits["max_pose_rotation_degrees"], "pose_rotation"),
        (float(metrics.get("patch_residual_p95", float("inf"))) <= limits["max_patch_residual_p95"], "patch_residual"),
        (float(metrics.get("pose_residual_p95", float("inf"))) <= limits["max_pose_residual_p95"], "pose_residual"),
        (float(metrics.get("invalid_output_ratio", 1.0)) == 0.0, "invalid_output"),
        (float(metrics.get("new_nonpositive_self_depth_ratio", 1.0)) <= limits["max_new_nonpositive_self_depth_ratio"], "self_depth"),
        (float(metrics.get("confidence_ref_relative_drop", float("inf"))) <= limits["max_confidence_relative_drop"], "reference_confidence"),
        (float(metrics.get("confidence_self_relative_drop", float("inf"))) <= limits["max_confidence_relative_drop"], "self_confidence"),
        (not bool(metrics.get("pose_dominated", True)), "pose_dominance"),
        (bool(metrics.get("structural_finite", False)), "structural_finite"),
    )
    failures.extend(name for passed, name in checks if not passed)
    return not failures, failures

