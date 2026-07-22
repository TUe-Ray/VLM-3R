"""Shared runtime helpers for CUT3R gauge-translation training."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CUT3R_ROOT = REPO_ROOT / "third_party" / "CUT3R"
for value in (str(REPO_ROOT), str(CUT3R_ROOT), str(CUT3R_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from dust3r.heads.postprocess import postprocess, postprocess_pose  # noqa: E402
from scripts.gauge_translation.standalone_model import validate_patch_positions  # noqa: E402
from src.dust3r.model import ARCroco3DStereo  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_torch(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


@dataclass
class GaugeSample:
    identifier: str
    patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    pose12: torch.Tensor
    dec0: torch.Tensor
    pos: torch.Tensor
    reference_points: torch.Tensor
    camera_points: torch.Tensor
    confidence_ref: torch.Tensor
    confidence_self: torch.Tensor
    teacher_mask: torch.Tensor
    pooled_ref_xyz: torch.Tensor
    pooled_valid: torch.Tensor
    scene_scale: torch.Tensor

    def to(self, device: torch.device, token_dtype: torch.dtype) -> "GaugeSample":
        return GaugeSample(
            self.identifier,
            tuple(value.unsqueeze(0).to(device=device, dtype=token_dtype) for value in self.patches),
            self.pose12.unsqueeze(0).to(device=device, dtype=token_dtype),
            self.dec0.to(device=device, dtype=token_dtype),
            self.pos.to(device=device),
            self.reference_points.to(device=device),
            self.camera_points.to(device=device),
            self.confidence_ref.to(device=device),
            self.confidence_self.to(device=device),
            self.teacher_mask.to(device=device),
            self.pooled_ref_xyz.unsqueeze(0).to(device=device),
            self.pooled_valid.unsqueeze(0).to(device=device),
            self.scene_scale.reshape(1).to(device=device),
        )


def _frame_selection(frame_count: int, max_frames: int, training: bool, generator: torch.Generator) -> torch.Tensor:
    count = min(int(frame_count), int(max_frames))
    if count <= 0:
        raise ValueError("sample contains no frames")
    if count == frame_count:
        return torch.arange(frame_count)
    if training:
        return torch.randperm(frame_count, generator=generator)[:count].sort().values
    return torch.linspace(0, frame_count - 1, count).round().long().unique()


def load_sample(
    record: Mapping[str, Any],
    device: torch.device,
    token_dtype: torch.dtype,
    max_frames: int,
    training: bool,
    seed: int,
) -> GaugeSample:
    layers = [load_torch(record[f"layer{layer}_path"]) for layer in (6, 9, 12)]
    context = load_torch(record["context_path"])
    pointmap = load_torch(record["pointmap_path"])
    frame_count = int(context["dec0"].shape[0])
    patches = tuple(payload["patch_tokens"][:frame_count] for payload in layers)
    pose12 = layers[2]["camera_tokens"][:frame_count]
    shapes = [tuple(value.shape) for value in (*patches, pose12, context["dec0"], context["pos"])]
    if any(int(shape[0]) != frame_count for shape in shapes):
        raise RuntimeError(f"frame mismatch for {record['id']}: {shapes}")
    if any(tuple(value.shape[1:]) != (729, 768) for value in patches):
        raise RuntimeError(f"bad patch shape for {record['id']}: {[tuple(value.shape) for value in patches]}")
    if tuple(pose12.shape[1:]) != (1, 768):
        raise RuntimeError(f"bad layer-12 pose shape for {record['id']}: {tuple(pose12.shape)}")
    alignment = context.get("metadata", {}).get("alignment", {})
    if not alignment.get("passed", False) or not validate_patch_positions(context["pos"])["passed"]:
        raise RuntimeError(f"alignment gate is absent or failed for {record['id']}")
    if context.get("metadata", {}).get("schema") != "cut3r_gauge_head_context_v1":
        raise RuntimeError(f"unsupported context schema for {record['id']}")
    generator = torch.Generator().manual_seed(int(seed))
    indices = _frame_selection(frame_count, max_frames, training, generator)
    sample = GaugeSample(
        str(record["id"]),
        tuple(value.index_select(0, indices) for value in patches),
        pose12.index_select(0, indices),
        context["dec0"].index_select(0, indices),
        context["pos"].index_select(0, indices),
        pointmap["point_maps_ref"].index_select(0, indices).float(),
        pointmap["point_maps_cam"].index_select(0, indices).float(),
        context["confidence_ref"].index_select(0, indices).float(),
        context["confidence_self"].index_select(0, indices).float(),
        context["teacher_mask"].index_select(0, indices).bool(),
        context["pooled_ref_xyz"].index_select(0, indices).float(),
        context["pooled_valid"].index_select(0, indices).bool(),
        context["scene_scale"].float(),
    )
    if not bool(sample.teacher_mask.any()) or not bool(sample.pooled_valid.any()):
        raise RuntimeError(f"empty teacher mask for {record['id']}")
    if not torch.isfinite(sample.scene_scale) or float(sample.scene_scale) <= 0:
        raise RuntimeError(f"invalid scene scale for {record['id']}")
    return sample.to(device, token_dtype)


class FrozenCut3RHead(nn.Module):
    """Frozen DPT self/cross and pose branches with gradients to latent inputs."""

    def __init__(self, checkpoint: Path, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        model = ARCroco3DStereo.from_pretrained(str(checkpoint))
        if type(model.downstream_head).__name__ != "DPTPts3dPose":
            raise RuntimeError(f"expected DPTPts3dPose, got {type(model.downstream_head).__name__}")
        self.head = model.downstream_head.eval().to(device=device, dtype=dtype)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        del model

    def decode_pose(self, pose_token: torch.Tensor) -> torch.Tensor:
        token = pose_token.reshape(-1, pose_token.shape[-1])
        raw = self.head.pose_head(token)
        return postprocess_pose(raw, self.head.pose_mode).reshape(*pose_token.shape[:-2], 7)

    def decode_self(
        self, dec0: torch.Tensor, patches: Sequence[torch.Tensor], image_size: tuple[int, int] = (432, 432)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = [value.reshape(-1, value.shape[-2], value.shape[-1]) for value in patches]
        raw = self.head.dpt_self([dec0, *flattened], image_size=image_size)
        result = postprocess(raw, self.head.depth_mode, self.head.conf_mode)
        return result["pts3d"], result["conf"]

    def decode_reference(
        self,
        dec0: torch.Tensor,
        patches: Sequence[torch.Tensor],
        pose_token: torch.Tensor,
        pos: torch.Tensor,
        image_size: tuple[int, int] = (432, 432),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = [value.reshape(-1, value.shape[-2], value.shape[-1]) for value in patches]
        pose = pose_token.reshape(-1, pose_token.shape[-1])
        conditioned = flattened[-1]
        for block in self.head.final_transform:
            conditioned = block(conditioned, pose, pos)
        raw = self.head.dpt_cross([dec0, flattened[0], flattened[1], conditioned], image_size=image_size)
        result = postprocess(raw, self.head.depth_mode, self.head.conf_mode)
        return result["pts3d"], result["conf"]


def parameter_grad_norm(module: nn.Module, prefixes: Sequence[str]) -> float:
    total = torch.zeros((), dtype=torch.float32, device=next(module.parameters()).device)
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and any(name.startswith(prefix) for prefix in prefixes):
            total += parameter.grad.detach().float().square().sum()
    return float(total.sqrt().item())


def resolve_stage_steps(value: Mapping[str, Any], train_videos: int, global_batch_size: int) -> tuple[int, float, int]:
    has_steps = value.get("steps") is not None
    has_epochs = value.get("epochs") is not None
    if has_steps == has_epochs:
        raise ValueError("each schedule must set exactly one of steps or epochs")
    steps_per_epoch = max(1, math.ceil(train_videos / global_batch_size))
    steps = int(value["steps"]) if has_steps else int(math.ceil(float(value["epochs"]) * steps_per_epoch))
    return steps, steps / steps_per_epoch, steps_per_epoch


def optimizer_schema(module: nn.Module, optimizer: torch.optim.Optimizer) -> list[dict]:
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    return [
        {"group": index, "lr": float(group["lr"]), "parameters": [names.get(id(p), "<external>") for p in group["params"]]}
        for index, group in enumerate(optimizer.param_groups)
    ]


def median_window(values: Sequence[float], first: bool, fraction: float = 0.2) -> float:
    if not values:
        return float("inf")
    count = max(1, int(math.ceil(len(values) * fraction)))
    selected = values[:count] if first else values[-count:]
    return float(np.median(selected))


def probe_gate(initial: Mapping[str, float], final: Mapping[str, float]) -> dict:
    failures = []
    for name in ("q_train", "q_eval"):
        error = float(final[f"{name}_normalized_error"])
        baseline = float(final["mean_predictor_normalized_error"])
        if not error <= 0.8 * baseline:
            failures.append(f"{name}_mean_baseline")
        if not error < float(initial[f"{name}_normalized_error"]):
            failures.append(f"{name}_initial_improvement")
        if not float(final[f"{name}_variance_ratio"]) >= 0.01:
            failures.append(f"{name}_variance")
        if not bool(final[f"{name}_finite"]):
            failures.append(f"{name}_finite")
    return {"passed": not failures, "failures": failures, "initial": dict(initial), "final": dict(final)}


def stage_a_gate(metrics: Mapping[str, float], history: Mapping[str, Sequence[float]]) -> dict:
    failures = []
    early, final = median_window(history["patch_eq"], True), median_window(history["patch_eq"], False)
    checks = (
        (final <= 0.8 * early, "patch_eq_decrease"),
        (float(metrics.get("q_eval_cosine", -1)) > 0.5, "q_eval_cosine"),
        (float(metrics.get("q_eval_magnitude_ratio", -1)) > 0.10, "q_eval_magnitude"),
        (float(metrics.get("patch_gradient_nonzero_fraction", 0)) >= 0.9, "patch_gradients"),
        (float(metrics.get("patch_change_median", 0)) > 1e-5, "patch_change_noise"),
        (float(metrics.get("patch_residual_p95", float("inf"))) < 0.25, "patch_change_bound"),
        (float(metrics.get("normalized_self_drift", float("inf"))) < 0.25, "self_drift"),
        (float(metrics.get("invalid_output_ratio", 1)) == 0, "invalid_output"),
        (float(metrics.get("new_nonpositive_self_depth_ratio", 1)) <= 0.01, "self_depth"),
    )
    failures.extend(name for passed, name in checks if not passed)
    return {"passed": not failures, "failures": failures, "early_patch_eq": early, "final_patch_eq": final, "metrics": dict(metrics)}


def stage_b_gate(metrics: Mapping[str, float], history: Mapping[str, Sequence[float]]) -> dict:
    failures = []
    early, final = median_window(history["pose_t"], True), median_window(history["pose_t"], False)
    checks = (
        (float(metrics.get("pose_head_cosine", -1)) > 0.5, "pose_head_cosine"),
        (float(metrics.get("pose_head_magnitude_ratio", -1)) > 0.10, "pose_head_magnitude"),
        (final <= 0.8 * early, "pose_loss_decrease"),
        (float(metrics.get("pose_head_rotation_degrees", float("inf"))) < 5.0, "pose_head_rotation"),
        (float(metrics.get("pose_change", 0)) > 1e-5, "pose_identity"),
        (float(metrics.get("pose_change", float("inf"))) < 0.25, "pose_change_bound"),
        (float(metrics.get("pose_gradient_nonzero_fraction", 0)) >= 0.9, "pose_gradients_nonzero"),
        (bool(metrics.get("pose_gradients_finite", False)), "pose_gradients_finite"),
        (bool(metrics.get("quaternion_sign_invariant", False)), "quaternion_sign"),
        (bool(metrics.get("pose_losses_finite", False)), "pose_finite"),
    )
    failures.extend(name for passed, name in checks if not passed)
    return {"passed": not failures, "failures": failures, "early_pose_t": early, "final_pose_t": final, "metrics": dict(metrics)}

