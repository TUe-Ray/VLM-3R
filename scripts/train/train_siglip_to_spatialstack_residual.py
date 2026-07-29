#!/usr/bin/env python3
"""Offline SigLIP-to-SpatialStack residual predictor training.

This is intentionally independent of LLaVA's SFT trainer: it loads only the
frozen mm_projector plus the three SpatialStack MLP branches and never creates
or forwards Qwen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from llava.model.cut3r_spatialstack import Cut3RSpatialStackMerger
from llava.model.llava_arch import pool_2d_visual_features
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.siglip_spatialstack_residual import (
    DEFAULT_SOURCE_LAYERS,
    PredictedSpatialStackResidualAdapter,
    build_residual_predictor,
    predictor_checkpoint_payload,
)

TEACHER_DEFAULT = "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963"


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def current_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; use bfloat16, float16, or float32.")


def parse_int_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = list(value)
    return [int(item) for item in values]


def normalise_key(value: str) -> str:
    key = str(value).replace("\\", "/").strip().lstrip("./")
    if key.endswith(".mp4"):
        key = key[:-4] + ".pt"
    key = key.replace("/videos/", "/")
    return key


def find_cache_files(root_spec: str | Path) -> Dict[str, Path]:
    """Discover .pt files from one root or ``dataset=path;...`` roots.

    Dataset labels become a stable key prefix, so the three FAST SigLIP cache
    roots align with the corresponding CUT3R layer roots without a manifest.
    """
    specs = [part.strip() for part in str(root_spec).split(";") if part.strip()]
    if not specs:
        raise ValueError("Cache root specification may not be empty.")
    files: Dict[str, Path] = {}
    for raw_spec in specs:
        label = None
        raw_path = raw_spec
        if "=" in raw_spec:
            label, raw_path = raw_spec.split("=", 1)
            label = label.strip().strip("/")
            if not label:
                raise ValueError(f"Invalid dataset cache root specification: {raw_spec!r}")
        root = Path(raw_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Cache root does not exist: {root}")
        discovered = sorted(root.rglob("*.pt"))
        if not discovered:
            raise FileNotFoundError(f"No .pt files found under cache root: {root}")
        for path in discovered:
            relative_key = path.relative_to(root).as_posix()
            key = f"{label}/{relative_key}" if label else relative_key
            if key in files:
                raise RuntimeError(f"Duplicate stable cache key {key!r} from {files[key]} and {path}.")
            files[key] = path
    return files


def read_key_list(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    result = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            result.add(normalise_key(line))
    return result


def keys_from_dataset_json(paths: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for raw_path in paths:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError(f"Dataset JSON must be a list or {{data: list}}: {raw_path}")
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for field in ("cache_key", "feature_key", "video", "video_path", "image"):
                if record.get(field):
                    result.add(normalise_key(record[field]))
                    break
    return result


def pick_tensor(value, kind: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return value, None
    if not isinstance(value, Mapping):
        raise TypeError(f"{kind} cache must contain a tensor or mapping, got {type(value).__name__}.")
    tensor = None
    names = ("features", "siglip_features", "patch_tokens", "tensor") if kind == "siglip" else ("patch_tokens", "features", "tensor")
    for name in names:
        if isinstance(value.get(name), torch.Tensor):
            tensor = value[name]
            break
    if tensor is None:
        tensors = [candidate for candidate in value.values() if isinstance(candidate, torch.Tensor)]
        if len(tensors) == 1:
            tensor = tensors[0]
    if tensor is None:
        raise TypeError(f"Could not find a tensor in {kind} cache mapping with keys={sorted(value)}.")
    mask = None
    for name in ("valid_frame_mask", "frame_valid_mask", "valid_frames"):
        if isinstance(value.get(name), torch.Tensor):
            mask = value[name].bool().flatten()
            break
    return tensor, mask


@dataclass(frozen=True)
class CacheSample:
    key: str
    siglip: Path
    cut3r: Mapping[int, Path]


class PairedResidualCache:
    """Bare-cache pairing by shared relative filename only."""

    def __init__(
        self,
        siglip_root: str,
        cut3r_root: str,
        layer_subdirs: Mapping[int, str],
        *,
        layer_roots: Optional[Mapping[int, str]] = None,
        train_keys: Optional[set[str]] = None,
        validation_keys: Optional[set[str]] = None,
        candidate_keys: Optional[set[str]] = None,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
    ):
        siglip_files = find_cache_files(Path(siglip_root))
        cut3r_files = {
            int(layer): find_cache_files(
                layer_roots[int(layer)] if layer_roots and layer_roots.get(int(layer)) else Path(cut3r_root) / subdir
            )
            for layer, subdir in layer_subdirs.items()
        }
        common = set(siglip_files)
        for files in cut3r_files.values():
            common &= set(files)
        if not common:
            raise RuntimeError("No SigLIP/CUT3R cache keys exist in all required roots.")
        samples = {
            key: CacheSample(key, siglip_files[key], {layer: files[key] for layer, files in cut3r_files.items()})
            for key in sorted(common)
        }
        if candidate_keys is not None:
            selected = self._resolve_requested_keys(candidate_keys, samples, "dataset JSON")
            samples = {key: samples[key] for key in sorted(selected)}
        train_keys = self._resolve_requested_keys(train_keys, samples, "train key list") if train_keys else None
        validation_keys = self._resolve_requested_keys(validation_keys, samples, "validation key list") if validation_keys else None
        if train_keys is not None or validation_keys is not None:
            if train_keys is None:
                train_keys = set(samples) - validation_keys
            if validation_keys is None:
                validation_keys = set(samples) - train_keys
        else:
            validation_keys = {
                key for key in samples
                if int(hashlib.sha256(f"{split_seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
                < float(validation_fraction)
            }
            if not validation_keys and len(samples) > 1:
                validation_keys = {sorted(samples)[0]}
            train_keys = set(samples) - validation_keys
        if not train_keys or not validation_keys:
            raise RuntimeError(
                f"Need non-empty train and validation cache splits; got {len(train_keys)} train and "
                f"{len(validation_keys)} validation keys."
            )
        self.samples = samples
        self.train_keys = sorted(train_keys)
        self.validation_keys = sorted(validation_keys)

    @staticmethod
    def _resolve_requested_keys(requested_keys: set[str], samples: Mapping[str, CacheSample], label: str) -> set[str]:
        """Map optional JSON/list paths to the paired relative cache key.

        Exact relative keys are preferred.  Existing dataset JSONs commonly
        contain absolute video paths, so a unique path suffix is also accepted;
        ambiguous basenames are rejected instead of guessing a sample.
        """
        available = set(samples)
        resolved = set()
        unresolved = []
        for requested in requested_keys:
            key = normalise_key(requested)
            if key in available:
                resolved.add(key)
                continue
            matches = [
                candidate for candidate in available
                if key.endswith("/" + candidate) or candidate.endswith("/" + key)
            ]
            if len(matches) == 1:
                resolved.add(matches[0])
            elif len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous {label} key {requested!r}; it matches cache keys {sorted(matches)[:8]}."
                )
            else:
                unresolved.append(str(requested))
        if unresolved:
            raise RuntimeError(
                f"{label} contains keys with no paired cache sample, for example {unresolved[:8]}."
            )
        return resolved

    @staticmethod
    def _check_shapes(siglip: torch.Tensor, cut3r: Mapping[int, torch.Tensor], key: str) -> int:
        if siglip.dim() != 3 or tuple(siglip.shape[1:]) != (729, 1152):
            raise RuntimeError(f"SigLIP cache shape mismatch for {key}: got {tuple(siglip.shape)}, expected [F,729,1152].")
        frames = int(siglip.shape[0])
        if frames <= 0:
            raise RuntimeError(f"SigLIP cache has no frames: {key}")
        for layer, tensor in cut3r.items():
            if tensor.dim() != 3 or tuple(tensor.shape[1:]) != (729, 768):
                raise RuntimeError(
                    f"CUT3R layer {layer} cache shape mismatch for {key}: got {tuple(tensor.shape)}, "
                    "expected [F,729,768]."
                )
            if int(tensor.shape[0]) != frames:
                raise RuntimeError(
                    f"Frame-count mismatch for {key}: SigLIP F={frames}, CUT3R layer {layer} F={int(tensor.shape[0])}."
                )
        return frames

    def load(self, key: str, *, strict: bool = False) -> Dict[str, object]:
        sample = self.samples[key]
        siglip, siglip_mask = pick_tensor(torch_load(sample.siglip), "siglip")
        cut3r = {}
        masks = [siglip_mask] if siglip_mask is not None else []
        for layer, path in sample.cut3r.items():
            tensor, mask = pick_tensor(torch_load(path), "cut3r")
            cut3r[int(layer)] = tensor
            if mask is not None:
                masks.append(mask)
        frames = self._check_shapes(siglip, cut3r, key)
        valid_mask = torch.ones(frames, dtype=torch.bool)
        for mask in masks:
            # Masks are optional metadata.  Legacy bare tensors have no mask and
            # use all stored frames; a malformed optional mask is only fatal in
            # strict/startup parity mode, never in the lightweight train loop.
            if int(mask.numel()) != frames:
                if strict:
                    raise RuntimeError(
                        f"Frame-mask length mismatch for {key}: got {int(mask.numel())}, expected {frames}."
                    )
                continue
            valid_mask &= mask.cpu()
        if strict:
            tensors = [siglip, *cut3r.values()]
            if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
                raise RuntimeError(f"Non-finite cached values found for {key}.")
        return {"key": key, "siglip": siglip, "cut3r": cut3r, "valid_mask": valid_mask}


def collate(samples: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    max_frames = max(int(sample["siglip"].shape[0]) for sample in samples)
    batch = len(samples)
    siglip = torch.zeros(batch, max_frames, 729, 1152, dtype=samples[0]["siglip"].dtype)
    cut3r = {layer: torch.zeros(batch, max_frames, 729, 768, dtype=samples[0]["cut3r"][layer].dtype) for layer in DEFAULT_SOURCE_LAYERS}
    mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    keys = []
    for index, sample in enumerate(samples):
        frames = int(sample["siglip"].shape[0])
        siglip[index, :frames] = sample["siglip"]
        for layer in DEFAULT_SOURCE_LAYERS:
            cut3r[layer][index, :frames] = sample["cut3r"][layer]
        mask[index, :frames] = sample["valid_mask"]
        keys.append(sample["key"])
    return {"keys": keys, "siglip": siglip, "cut3r": cut3r, "valid_mask": mask}


class FrozenSpatialStackTeacher(nn.Module):
    """The only frozen modules required for residual regression."""

    def __init__(self, checkpoint: str, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.checkpoint = Path(checkpoint)
        config_path = self.checkpoint / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Teacher config missing: {config_path}")
        self.config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        self.config = SimpleNamespace(**self.config_dict)
        self.device = device
        self.dtype = dtype
        self.source_layers = tuple(parse_int_list(getattr(self.config, "cut3r_spatialstack_layers", "6,9,12")))
        self.llm_layers = tuple(parse_int_list(getattr(self.config, "cut3r_spatialstack_llm_layers", "0,1,2")))
        if self.source_layers != DEFAULT_SOURCE_LAYERS or self.llm_layers != (0, 1, 2):
            raise RuntimeError(
                f"This experiment requires teacher mapping 6/9/12->0/1/2; checkpoint has "
                f"{self.source_layers}->{self.llm_layers}."
            )
        self.mm_projector = build_vision_projector(self.config)
        self._load_mm_projector()
        self.merger = Cut3RSpatialStackMerger(self.config)
        self._load_merger()
        self.mm_projector.to(device=device, dtype=dtype).eval()
        self.merger.to(device=device, dtype=dtype).eval()
        for module in (self.mm_projector, self.merger):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.config_hash = sha256_file(config_path)

    def _base_model_root(self) -> Path:
        adapter = self.checkpoint / "adapter_config.json"
        if not adapter.is_file():
            raise FileNotFoundError(f"Teacher adapter config missing: {adapter}")
        base = json.loads(adapter.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if not base:
            raise RuntimeError("Teacher adapter config has no base_model_name_or_path for mm_projector loading.")
        root = Path(base)
        if not root.is_dir():
            raise FileNotFoundError(f"Teacher base model missing: {root}")
        return root

    def _load_mm_projector(self) -> None:
        base_root = self._base_model_root()
        index_path = base_root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Expected safetensors index for frozen mm_projector: {index_path}")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load the frozen mm_projector.") from exc
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        prefix = "model.mm_projector."
        selected = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
        if not selected:
            raise RuntimeError(f"No mm_projector weights found in {index_path}")
        state = {}
        for key, shard in selected.items():
            with safe_open(str(base_root / shard), framework="pt", device="cpu") as handle:
                state[key[len(prefix):]] = handle.get_tensor(key)
        self.mm_projector.load_state_dict(state, strict=True)
        self.mm_projector_source = str(base_root)

    def _load_merger(self) -> None:
        state_path = self.checkpoint / "non_lora_trainables.bin"
        if not state_path.is_file():
            raise FileNotFoundError(f"Teacher SpatialStack weights missing: {state_path}")
        raw = torch_load(state_path)
        marker = "cut3r_spatialstack_merger."
        state = {key.split(marker, 1)[1]: value for key, value in raw.items() if marker in key}
        if not state:
            raise RuntimeError(f"No SpatialStack merger weights found in {state_path}")
        self.merger.load_state_dict(state, strict=True)

    @torch.no_grad()
    def inputs(self, siglip: torch.Tensor) -> torch.Tensor:
        batch, frames = siglip.shape[:2]
        projected = self.mm_projector(siglip.to(device=self.device, dtype=self.dtype).reshape(batch * frames, 729, 1152))
        pooled = pool_2d_visual_features(
            projected,
            num_patches_per_side=27,
            pool_mode=getattr(self.config, "mm_spatial_pool_mode", "bilinear"),
            stride=int(getattr(self.config, "mm_spatial_pool_stride", 2)),
        )
        if tuple(pooled.shape[1:]) != (196, int(getattr(self.config, "hidden_size"))):
            raise RuntimeError(f"Frozen SigLIP input pipeline returned {tuple(pooled.shape)}, expected [B*F,196,3584].")
        return pooled.reshape(batch, frames, 196, -1)

    @torch.no_grad()
    def targets(self, cut3r: Mapping[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        result = {}
        for layer in self.source_layers:
            raw = cut3r[int(layer)].to(device=self.device, dtype=self.dtype)
            batch, frames = raw.shape[:2]
            aligned = torch.stack(
                [self.merger.resize_square_grid(frame, 196) for frame in raw.reshape(batch * frames, 729, 768)],
                dim=0,
            )
            projected = self.merger.branches[str(layer)](aligned)
            # Match the oracle value immediately before Qwen residual injection.
            projected = projected * self.merger.residual_scale
            result[int(layer)] = projected.reshape(batch, frames, 196, -1)
        return result


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth_l1_weight: float,
    teacher_norm_eps: float,
):
    """FP32 regression metrics with a direction mask for null teacher residuals."""
    valid = valid_mask[:, :, None].expand(-1, -1, prediction.shape[2]).bool()
    prediction = prediction.float()
    target = target.float()
    target_norm = target.norm(dim=-1)
    pred_norm = prediction.norm(dim=-1)
    direction = valid & (target_norm > float(teacher_norm_eps))
    valid_weight = valid.sum().clamp_min(1).to(dtype=torch.float32)
    direction_weight = direction.sum().clamp_min(1).to(dtype=torch.float32)
    valid_float = valid.to(dtype=torch.float32)
    direction_float = direction.to(dtype=torch.float32)

    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    cosine_loss = ((1.0 - cosine) * direction_float).sum() / direction_weight
    smooth = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    smooth = (smooth * valid_float).sum() / valid_weight
    relative_l2 = (
        ((prediction - target).norm(dim=-1) / target_norm.clamp_min(float(teacher_norm_eps)))
        * direction_float
    ).sum() / direction_weight
    return {
        "loss": cosine_loss + float(smooth_l1_weight) * smooth,
        "cosine": (cosine * direction_float).sum() / direction_weight,
        "cosine_loss": cosine_loss,
        "smooth_l1": smooth,
        "relative_l2": relative_l2,
        "pred_norm": (pred_norm * valid_float).sum() / valid_weight,
        "teacher_norm": (target_norm * valid_float).sum() / valid_weight,
        "norm_ratio": ((pred_norm / target_norm.clamp_min(float(teacher_norm_eps))) * direction_float).sum()
        / direction_weight,
        "valid_weight": valid_weight,
        "direction_weight": direction_weight,
        "low_norm_excluded": (valid & ~direction).sum().to(dtype=torch.float32),
    }


def batch_metrics(predictor, teacher, batch, smooth_l1_weight, teacher_norm_eps):
    # Keep frozen teacher modules in BF16/FP16 if configured, but retain all
    # trainable predictor/AdamW weights in FP32.
    x = teacher.inputs(batch["siglip"])
    targets = teacher.targets(batch["cut3r"])
    prediction = predictor(x.float(), batch["valid_mask"].to(device=x.device))
    metrics = {}
    losses = []
    for layer in DEFAULT_SOURCE_LAYERS:
        values = regression_metrics(
            prediction[layer],
            targets[layer],
            batch["valid_mask"].to(device=x.device),
            smooth_l1_weight,
            teacher_norm_eps,
        )
        metrics[layer] = values
        losses.append(values["loss"])
    return torch.stack(losses).mean(), metrics, x, targets, prediction


def accumulate(totals, metrics):
    direction_metrics = {"cosine", "cosine_loss", "relative_l2", "norm_ratio"}
    valid_metrics = {"smooth_l1", "pred_norm", "teacher_norm"}
    for layer, values in metrics.items():
        valid_weight = float(values["valid_weight"].detach().cpu())
        direction_weight = float(values["direction_weight"].detach().cpu())
        totals[f"layer_{layer}_valid_weight"] += valid_weight
        totals[f"layer_{layer}_direction_weight"] += direction_weight
        totals[f"layer_{layer}_low_norm_excluded"] += float(values["low_norm_excluded"].detach().cpu())
        for name in direction_metrics:
            totals[f"layer_{layer}_{name}"] += float(values[name].detach().cpu()) * direction_weight
        for name in valid_metrics:
            totals[f"layer_{layer}_{name}"] += float(values[name].detach().cpu()) * valid_weight


def finalise(totals, smooth_l1_weight):
    result = {}
    layer_losses = []
    for layer in DEFAULT_SOURCE_LAYERS:
        valid_weight = max(totals[f"layer_{layer}_valid_weight"], 1.0)
        direction_weight = max(totals[f"layer_{layer}_direction_weight"], 1.0)
        for name in ("cosine", "cosine_loss", "relative_l2", "norm_ratio"):
            result[f"layer_{layer}_{name}"] = totals[f"layer_{layer}_{name}"] / direction_weight
        for name in ("smooth_l1", "pred_norm", "teacher_norm"):
            result[f"layer_{layer}_{name}"] = totals[f"layer_{layer}_{name}"] / valid_weight
        result[f"layer_{layer}_valid_tokens"] = valid_weight
        result[f"layer_{layer}_direction_tokens"] = direction_weight
        result[f"layer_{layer}_low_norm_excluded_fraction"] = (
            totals[f"layer_{layer}_low_norm_excluded"] / valid_weight
        )
        result[f"layer_{layer}_loss"] = (
            result[f"layer_{layer}_cosine_loss"] + float(smooth_l1_weight) * result[f"layer_{layer}_smooth_l1"]
        )
        layer_losses.append(result[f"layer_{layer}_loss"])
    result["loss"] = sum(layer_losses) / len(layer_losses)
    return result


def run_epoch(predictor, teacher, cache, keys, args, optimizer=None, scheduler=None):
    train = optimizer is not None
    predictor.train(train)
    totals = defaultdict(float)
    order = list(keys)
    if train:
        random.Random((args.seed + int(getattr(args, "_residual_epoch", 0))) if args.shuffle_each_epoch else 0).shuffle(order)
    for start in range(0, len(order), args.batch_size):
        selected = order[start:start + args.batch_size]
        batch = collate([cache.load(key, strict=False) for key in selected])
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _, _, _ = batch_metrics(
                predictor, teacher, batch, args.smooth_l1_weight, args.teacher_norm_eps
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite predictor loss for cache keys {selected}.")
            loss.backward()
            frozen_grads = [name for name, parameter in teacher.named_parameters() if parameter.grad is not None]
            if frozen_grads:
                raise RuntimeError(f"Frozen teacher parameters received gradients: {frozen_grads[:8]}")
            non_finite_grads = [
                name for name, parameter in predictor.named_parameters()
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if non_finite_grads:
                raise RuntimeError(f"Predictor received non-finite gradients: {non_finite_grads[:8]}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                _, metrics, _, _, _ = batch_metrics(
                    predictor, teacher, batch, args.smooth_l1_weight, args.teacher_norm_eps
                )
        accumulate(totals, metrics)
    return finalise(totals, args.smooth_l1_weight)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """PyTorch keeps optimizer tensors on the checkpoint load device by default."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def save_checkpoint(path: Path, predictor, optimizer, scheduler, epoch, step, teacher, args, train_metrics, validation_metrics):
    payload = predictor_checkpoint_payload(
        predictor,
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        epoch=int(epoch),
        step=int(step),
        teacher_checkpoint=str(teacher.checkpoint),
        teacher_config_hash=teacher.config_hash,
        teacher_mm_projector_source=teacher.mm_projector_source,
        source_to_llm_mapping={str(source): int(llm) for source, llm in zip(teacher.source_layers, teacher.llm_layers)},
        smooth_l1_weight=float(args.smooth_l1_weight),
        teacher_norm_eps=float(args.teacher_norm_eps),
        teacher_residual_scale=float(teacher.merger.residual_scale),
        predictor_parameter_dtype=str(next(predictor.parameters()).dtype),
        dtype=str(teacher.dtype),
        feature_shapes={"siglip": ["F", 729, 1152], "cut3r": ["F", 729, 768], "residual": ["F", 196, 3584]},
        frame_count_metadata="variable per sample; read from cached tensors at load time",
        valid_frame_mask_policy="absent mask means all stored frames are valid",
        git_commit=current_git_commit(),
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        split_seed=int(args.split_seed),
        validation_fraction=float(args.validation_fraction),
        cache_roots={
            "siglip": str(args.siglip_feature_cache),
            "cut3r": str(args.cut3r_feature_cache),
            "cut3r_layer_subdirs": {
                "6": str(args.cut3r_layer6_subdir),
                "9": str(args.cut3r_layer9_subdir),
                "12": str(args.cut3r_layer12_subdir),
            },
            "cut3r_layer_cache_overrides": {
                "6": args.cut3r_layer6_cache,
                "9": args.cut3r_layer9_cache,
                "12": args.cut3r_layer12_cache,
            },
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def startup_or_smoke(cache, teacher, predictor, args, smoke_only=False):
    requested = int(args.startup_check_samples)
    if requested < 0:
        raise ValueError("startup_check_samples must be non-negative.")
    if requested == 0 and not smoke_only:
        print(json.dumps({"startup_or_smoke": "skipped", "checked_samples": 0}, sort_keys=True))
        return
    count = min(len(cache.train_keys), max(1, requested))
    keys = cache.train_keys[:count]
    for key in keys:
        cache.load(key, strict=bool(args.strict_cache_checks) or smoke_only)
    batch = collate([cache.load(key, strict=smoke_only) for key in keys[: min(2, len(keys))]])
    loss, _, x, targets, predictions = batch_metrics(
        predictor, teacher, batch, args.smooth_l1_weight, args.teacher_norm_eps
    )
    if tuple(x.shape[2:]) != (196, 3584):
        raise RuntimeError(f"Offline frozen input shape mismatch: {tuple(x.shape)}")
    if any(tuple(targets[layer].shape[2:]) != (196, 3584) for layer in DEFAULT_SOURCE_LAYERS):
        raise RuntimeError("Offline frozen teacher target shape mismatch.")
    if any(tuple(predictions[layer].shape) != tuple(x.shape) for layer in DEFAULT_SOURCE_LAYERS):
        raise RuntimeError("Predictor output shape mismatch.")
    if smoke_only:
        predictor.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None for parameter in predictor.parameters() if parameter.requires_grad):
            raise RuntimeError("Smoke gradient check failed: a predictor parameter did not receive a gradient.")
        predictor.zero_grad(set_to_none=True)
        # Synthetic sequence with 196 patches plus 14 newline positions per frame.
        frames = int(x.shape[1])
        seq_len = frames * 210
        visual_indices = []
        frame_ids = []
        newline_indices = []
        for frame in range(frames):
            base = frame * 210
            for row in range(14):
                visual_indices.extend(base + row * 15 + col for col in range(14))
                frame_ids.extend([frame] * 14)
                newline_indices.append(base + row * 15 + 14)
        metadata = [{
            "visual_token_indices": torch.tensor(visual_indices),
            "visual_frame_ids": torch.tensor(frame_ids),
            "frame_order": list(range(frames)),
            "newline_token_indices": torch.tensor(newline_indices),
        }]
        adapter = PredictedSpatialStackResidualAdapter(
            predictor.eval(), source_layers=DEFAULT_SOURCE_LAYERS, llm_layers=(0, 1, 2)
        )
        embeds = torch.randn(1, seq_len, 3584, device=x.device, dtype=x.dtype)
        residuals = adapter(embeds, metadata)
        visual = torch.tensor(visual_indices, device=x.device)
        non_visual = torch.ones(seq_len, dtype=torch.bool, device=x.device)
        non_visual[visual] = False
        if any(residual[0, non_visual].abs().sum().item() != 0 for residual in residuals.values()):
            raise RuntimeError("Predicted residual smoke check wrote outside visual patch positions.")
        adapter.configure(control="zero")
        if any(residual.abs().sum().item() != 0 for residual in adapter(embeds, metadata).values()):
            raise RuntimeError("Zero predicted control smoke check is not zero.")
        if adapter.last_debug.get("cut3r_called") is not False:
            raise RuntimeError("Predicted residual smoke check did not prove CUT3R disablement.")
    print(json.dumps({"startup_or_smoke": "passed", "checked_samples": count, "loss": float(loss.detach().cpu())}, sort_keys=True))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--siglip_feature_cache", required=True)
    result.add_argument("--cut3r_feature_cache", required=True)
    result.add_argument("--cut3r_layer6_subdir", default="spatial_features_dec_6")
    result.add_argument("--cut3r_layer9_subdir", default="spatial_features_dec_9")
    result.add_argument("--cut3r_layer12_subdir", default="spatial_features")
    result.add_argument("--cut3r_layer6_cache", help="Optional root or dataset=path;... mapping for CUT3R layer 6.")
    result.add_argument("--cut3r_layer9_cache", help="Optional root or dataset=path;... mapping for CUT3R layer 9.")
    result.add_argument("--cut3r_layer12_cache", help="Optional root or dataset=path;... mapping for CUT3R layer 12.")
    result.add_argument("--teacher_checkpoint", default=TEACHER_DEFAULT)
    result.add_argument("--output_dir", required=True)
    result.add_argument("--train_key_list")
    result.add_argument("--validation_key_list")
    result.add_argument("--dataset_json", action="append", default=[])
    result.add_argument("--train_dataset_json", action="append", default=[])
    result.add_argument("--validation_dataset_json", action="append", default=[])
    result.add_argument("--validation_fraction", type=float, default=0.1)
    result.add_argument("--split_seed", type=int, default=42)
    result.add_argument("--startup_check_samples", type=int, default=8)
    result.add_argument("--strict_cache_checks", type=as_bool, default=False)
    result.add_argument("--run_parity_check", type=as_bool, default=False)
    result.add_argument("--smoke_only", action="store_true")
    result.add_argument("--residual_predictor_type", choices=("token_mlp", "temporal"), default="token_mlp")
    result.add_argument("--predictor_bottleneck_dim", type=int, default=1024)
    result.add_argument("--temporal_hidden_dim", type=int, default=512)
    result.add_argument("--temporal_num_layers", type=int, default=2)
    result.add_argument("--temporal_num_heads", type=int, default=8)
    result.add_argument("--temporal_ffn_dim", type=int, default=2048)
    result.add_argument("--temporal_dropout", type=float, default=0.0)
    result.add_argument("--temporal_max_frames", type=int, default=128)
    result.add_argument("--smooth_l1_weight", type=float, default=0.1)
    result.add_argument("--teacher_norm_eps", type=float, default=1e-6)
    result.add_argument("--learning_rate", type=float, default=1e-4)
    result.add_argument("--weight_decay", type=float, default=0.01)
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--batch_size", type=int, default=1)
    result.add_argument("--max_train_samples", type=int, default=0)
    result.add_argument("--max_validation_samples", type=int, default=0)
    result.add_argument("--warmup_ratio", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--resume")
    result.add_argument("--shuffle_each_epoch", type=as_bool, default=True)
    return result


def main():
    args = parser().parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if args.teacher_norm_eps <= 0:
        raise ValueError("teacher_norm_eps must be positive.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    candidate_keys = keys_from_dataset_json(args.dataset_json) if args.dataset_json else None
    train_keys = read_key_list(args.train_key_list)
    validation_keys = read_key_list(args.validation_key_list)
    if args.train_dataset_json:
        train_keys = (train_keys or set()) | keys_from_dataset_json(args.train_dataset_json)
    if args.validation_dataset_json:
        validation_keys = (validation_keys or set()) | keys_from_dataset_json(args.validation_dataset_json)
    cache = PairedResidualCache(
        args.siglip_feature_cache,
        args.cut3r_feature_cache,
        {6: args.cut3r_layer6_subdir, 9: args.cut3r_layer9_subdir, 12: args.cut3r_layer12_subdir},
        layer_roots={
            6: args.cut3r_layer6_cache,
            9: args.cut3r_layer9_cache,
            12: args.cut3r_layer12_cache,
        },
        train_keys=train_keys,
        validation_keys=validation_keys,
        candidate_keys=candidate_keys,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    if args.max_train_samples > 0:
        cache.train_keys = cache.train_keys[: args.max_train_samples]
    if args.max_validation_samples > 0:
        cache.validation_keys = cache.validation_keys[: args.max_validation_samples]
    if not cache.train_keys or not cache.validation_keys:
        raise RuntimeError("Configured smoke subset produced an empty train or validation split.")
    device = torch.device(args.device)
    teacher = FrozenSpatialStackTeacher(args.teacher_checkpoint, device, dtype_from_name(args.dtype))
    predictor = build_residual_predictor(
        args.residual_predictor_type,
        hidden_size=3584,
        bottleneck_dim=args.predictor_bottleneck_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_num_layers=args.temporal_num_layers,
        temporal_num_heads=args.temporal_num_heads,
        temporal_ffn_dim=args.temporal_ffn_dim,
        temporal_dropout=args.temporal_dropout,
        temporal_max_frames=args.temporal_max_frames,
    ).to(device=device)
    if any(parameter.dtype != torch.float32 for parameter in predictor.parameters()):
        raise RuntimeError("Predictor parameters must remain FP32 for AdamW updates.")
    frozen_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Frozen teacher unexpectedly has trainable parameters.")
    print(json.dumps({
        "cache_samples": len(cache.samples),
        "train_samples": len(cache.train_keys),
        "validation_samples": len(cache.validation_keys),
        "frozen_teacher_parameters": frozen_parameters,
        "trainable_predictor_parameters": trainable_parameters,
        "predictor_type": args.residual_predictor_type,
        "source_to_llm_mapping": {
            str(source): int(llm) for source, llm in zip(teacher.source_layers, teacher.llm_layers)
        },
        "trainable_predictor_parameter_names": [
            name for name, parameter in predictor.named_parameters() if parameter.requires_grad
        ],
        "frozen_teacher_parameter_names": [name for name, _ in teacher.named_parameters()],
    }, sort_keys=True), flush=True)
    startup_or_smoke(cache, teacher, predictor, args, smoke_only=args.smoke_only or args.run_parity_check)
    if args.smoke_only:
        return
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(cache.train_keys) / args.batch_size) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps)) * 0.5 * (1 + math.cos(math.pi * max(0, step - warmup_steps) / max(1, total_steps - warmup_steps))),
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        predictor.load_state_dict(checkpoint["predictor"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
    output = Path(args.output_dir)
    best_cosine, best_relative = -float("inf"), float("inf")
    for epoch in range(start_epoch, args.epochs):
        # Keep per-epoch shuffling deterministic while changing the order each epoch.
        args._residual_epoch = epoch
        train_metrics = run_epoch(predictor, teacher, cache, cache.train_keys, args, optimizer, scheduler)
        global_step += math.ceil(len(cache.train_keys) / args.batch_size)
        validation_metrics = run_epoch(predictor, teacher, cache, cache.validation_keys, args)
        record = {"epoch": epoch, "step": global_step, "train": train_metrics, "validation": validation_metrics}
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        save_checkpoint(output / "latest.pt", predictor, optimizer, scheduler, epoch, global_step, teacher, args, train_metrics, validation_metrics)
        if validation_metrics["layer_6_cosine"] + validation_metrics["layer_9_cosine"] + validation_metrics["layer_12_cosine"] > best_cosine:
            best_cosine = sum(validation_metrics[f"layer_{layer}_cosine"] for layer in DEFAULT_SOURCE_LAYERS)
            save_checkpoint(output / "best_validation_cosine.pt", predictor, optimizer, scheduler, epoch, global_step, teacher, args, train_metrics, validation_metrics)
        if sum(validation_metrics[f"layer_{layer}_relative_l2"] for layer in DEFAULT_SOURCE_LAYERS) < best_relative:
            best_relative = sum(validation_metrics[f"layer_{layer}_relative_l2"] for layer in DEFAULT_SOURCE_LAYERS)
            save_checkpoint(output / "best_validation_relative_l2.pt", predictor, optimizer, scheduler, epoch, global_step, teacher, args, train_metrics, validation_metrics)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
