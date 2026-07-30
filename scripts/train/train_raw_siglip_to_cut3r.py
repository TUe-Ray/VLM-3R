#!/usr/bin/env python3
"""Offline raw SigLIP -> CUT3R feature distillation with optional single-node DDP.

The data iterator deliberately operates on deterministic *global* groups of
four sample keys.  World-size one consumes each group as four accumulated
microbatches; world-size four assigns one key to each rank.  The final
incomplete group is dropped in both modes, avoiding DistributedSampler
padding and making optimizer-step inputs identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from llava.model.raw_siglip_cut3r import (
    CUT3R_DIM,
    GRID_SIZE,
    SOURCE_LAYERS,
    FrozenSpatialStackPostprocessor,
    build_raw_cut3r_predictor,
    raw_predictor_checkpoint_payload,
)

TEACHER_DEFAULT = "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/cut3r_spatialstack_45297963"


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return rank() == 0


def setup_distributed():
    size = world_size()
    if size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if size > 1:
        raise RuntimeError("DDP raw distillation requires CUDA/NCCL.")
    return torch.device("cpu")


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def discover(spec: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for entry in [item.strip() for item in str(spec).split(";") if item.strip()]:
        label, raw_root = entry.split("=", 1) if "=" in entry else ("", entry)
        root = Path(raw_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Cache root does not exist: {root}")
        for path in sorted(root.rglob("*.pt")):
            key = f"{label.strip('/')}/{path.relative_to(root).as_posix()}" if label else path.relative_to(root).as_posix()
            if key in result:
                raise RuntimeError(f"Duplicate cache key {key}: {result[key]} and {path}")
            result[key] = path
    if not result:
        raise RuntimeError(f"No .pt files found for cache spec {spec!r}")
    return result


def pick_tensor(value, kind: str):
    if isinstance(value, torch.Tensor):
        return value, None, {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{kind} cache must be a tensor/mapping, got {type(value).__name__}")
    names = ("features", "siglip_features", "patch_tokens", "tensor") if kind == "siglip" else ("patch_tokens", "features", "tensor")
    tensor = next((value[name] for name in names if isinstance(value.get(name), torch.Tensor)), None)
    if tensor is None:
        tensors = [candidate for candidate in value.values() if isinstance(candidate, torch.Tensor)]
        if len(tensors) == 1:
            tensor = tensors[0]
    if tensor is None:
        raise RuntimeError(f"Could not find {kind} tensor in cache mapping.")
    mask = next((value[name] for name in ("valid_frame_mask", "frame_valid_mask", "valid_frames") if isinstance(value.get(name), torch.Tensor)), None)
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return tensor, None if mask is None else mask.flatten().bool(), dict(metadata)


class RawCache:
    def __init__(
        self,
        siglip_spec: str,
        layer_specs: Mapping[int, str],
        seed: int,
        validation_fraction: float,
        *,
        alignment_artifact_verified: bool,
    ):
        self.siglip = discover(siglip_spec)
        self.targets = {layer: discover(spec) for layer, spec in layer_specs.items()}
        keys = set(self.siglip)
        for mapping in self.targets.values():
            keys &= set(mapping)
        if not keys:
            raise RuntimeError("No cache key appears in SigLIP and all three CUT3R roots.")
        self.keys = sorted(keys)
        self.train_keys = [
            key for key in self.keys
            if int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64 >= validation_fraction
        ]
        self.validation_keys = [key for key in self.keys if key not in set(self.train_keys)]
        if not self.train_keys or not self.validation_keys:
            raise RuntimeError("Hash split generated an empty training or validation set.")
        self.alignment_artifact_verified = bool(alignment_artifact_verified)

    @staticmethod
    def _done_record(path: Path) -> Mapping[str, object]:
        done_path = path.with_name(path.name + ".done.json")
        if not done_path.is_file():
            raise RuntimeError(
                f"Bare SigLIP cache requires extraction provenance at {done_path}; refusing {path}."
            )
        value = json.loads(done_path.read_text(encoding="utf-8"))
        frames = value.get("selected_frame_indices")
        if not isinstance(frames, list) or not frames:
            raise RuntimeError(f"SigLIP provenance has no selected_frame_indices: {done_path}")
        if not value.get("source_video"):
            raise RuntimeError(f"SigLIP provenance has no source_video: {done_path}")
        return value

    @staticmethod
    def _same_video(left: object, right: object) -> bool:
        try:
            return Path(str(left)).resolve() == Path(str(right)).resolve()
        except OSError:
            return str(left) == str(right)

    def load(self, key: str):
        siglip, siglip_mask, _ = pick_tensor(torch_load(self.siglip[key]), "siglip")
        siglip_provenance = self._done_record(self.siglip[key])
        targets, masks = {}, [siglip_mask] if siglip_mask is not None else []
        if siglip.dim() != 3 or tuple(siglip.shape[1:]) != (729, 1152):
            raise RuntimeError(f"SigLIP {key} has {tuple(siglip.shape)}, expected [F,729,1152].")
        frames = int(siglip.shape[0])
        selected_frames = [int(value) for value in siglip_provenance["selected_frame_indices"]]
        if len(selected_frames) != frames:
            raise RuntimeError(
                f"SigLIP provenance frame count differs from cached tensor for {key}: "
                f"{len(selected_frames)} != {frames}."
            )
        for layer in SOURCE_LAYERS:
            tensor, mask, metadata = pick_tensor(torch_load(self.targets[layer][key]), f"cut3r{layer}")
            if tensor.dim() != 3 or tuple(tensor.shape[1:]) != (729, CUT3R_DIM) or tensor.shape[0] != frames:
                raise RuntimeError(f"CUT3R layer {layer} {key} has incompatible shape {tuple(tensor.shape)}.")
            if not torch.isfinite(tensor).all() or mask is not None and mask.numel() != frames:
                raise RuntimeError(f"CUT3R layer {layer} {key} has non-finite values or malformed frame mask.")
            source_video = metadata.get("source_video")
            if source_video and not self._same_video(siglip_provenance["source_video"], source_video):
                raise RuntimeError(f"CUT3R layer {layer} source_video disagrees with SigLIP provenance for {key}.")
            recorded_frames = metadata.get("frame_indices")
            if recorded_frames is not None and [int(value) for value in recorded_frames] != selected_frames:
                raise RuntimeError(f"CUT3R layer {layer} frame order disagrees with SigLIP provenance for {key}.")
            if not source_video or recorded_frames is None:
                if not self.alignment_artifact_verified:
                    raise RuntimeError(
                        f"Bare CUT3R layer {layer} cache lacks complete provenance for {key}; "
                        "a verified alignment artifact is required."
                    )
            targets[layer] = tensor
            if mask is not None:
                masks.append(mask)
        if not torch.isfinite(siglip).all():
            raise RuntimeError(f"SigLIP {key} contains non-finite values.")
        valid = torch.ones(frames, dtype=torch.bool)
        for mask in masks:
            valid &= mask.cpu()
        return siglip, targets, valid


def measure(prediction, target, valid_mask, eps: float):
    """Return loss components plus globally reducible numerator/count tensors."""
    valid = valid_mask[:, None].expand(-1, prediction.shape[1])
    prediction, target = prediction.float(), target.float()
    target_norm, pred_norm = target.norm(dim=-1), prediction.norm(dim=-1)
    direction = valid & (target_norm > eps)
    valid_f, direction_f = valid.float(), direction.float()
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    relative_l2 = (prediction - target).norm(dim=-1) / target_norm.clamp_min(eps)
    smooth = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    values = {
        "cosine_loss_sum": ((1.0 - cosine) * direction_f).sum(),
        "relative_l2_sum": (relative_l2 * direction_f).sum(),
        "smooth_l1_sum": (smooth * valid_f).sum(),
        "pred_norm_sum": (pred_norm * valid_f).sum(),
        "teacher_norm_sum": (target_norm * valid_f).sum(),
        "norm_ratio_sum": ((pred_norm / target_norm.clamp_min(eps)) * direction_f).sum(),
        "valid_count": valid_f.sum(), "direction_count": direction_f.sum(),
    }
    return values


def weighted_loss(values, args, prefix: str):
    direction = values["direction_count"].clamp_min(1.0)
    valid = values["valid_count"].clamp_min(1.0)
    return (
        getattr(args, f"{prefix}_cosine_weight") * values["cosine_loss_sum"] / direction
        + getattr(args, f"{prefix}_relative_l2_weight") * values["relative_l2_sum"] / direction
        + getattr(args, f"{prefix}_smooth_l1_weight") * values["smooth_l1_sum"] / valid
    )


def sample_forward(model, postprocessor, cache, key, device, args):
    siglip, raw_targets, valid = cache.load(key)
    raw = siglip.unsqueeze(0).to(device=device, dtype=torch.float32)
    valid = valid.unsqueeze(0).to(device=device)
    raw_targets = {layer: value.unsqueeze(0).to(device=device, dtype=torch.float32) for layer, value in raw_targets.items()}
    autocast = bool(args.autocast and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        predicted_raw = model(raw, valid)
        predicted_residual = postprocessor(predicted_raw)  # must retain autograd
    target_residual = postprocessor.targets(raw_targets)
    totals = {"raw": {}, "residual": {}}
    loss = raw.new_zeros(())
    for layer in SOURCE_LAYERS:
        totals["raw"][layer] = measure(predicted_raw[layer][0], raw_targets[layer][0], valid[0], args.teacher_norm_eps)
        totals["residual"][layer] = measure(predicted_residual[layer][0], target_residual[layer][0], valid[0], args.teacher_norm_eps)
        loss = loss + args.raw_level_weight * weighted_loss(totals["raw"][layer], args, "raw")
        loss = loss + args.residual_level_weight * weighted_loss(totals["residual"][layer], args, "residual")
    return loss / len(SOURCE_LAYERS), totals


def merge_totals(totals, device):
    flat = []
    for level in ("raw", "residual"):
        for layer in SOURCE_LAYERS:
            for name in sorted(totals[level][layer]):
                flat.append(totals[level][layer][name].detach().float())
    vector = torch.stack(flat).to(device)
    all_reduce(vector)
    cursor = 0
    result = {"raw": {}, "residual": {}}
    for level in ("raw", "residual"):
        for layer in SOURCE_LAYERS:
            result[level][layer] = {}
            for name in sorted(totals[level][layer]):
                result[level][layer][name] = vector[cursor].cpu()
                cursor += 1
    return result


def fresh_totals(device):
    return {
        level: {layer: defaultdict(lambda: torch.zeros((), device=device)) for layer in SOURCE_LAYERS}
        for level in ("raw", "residual")
    }


def add_totals(destination, source):
    for level in destination:
        for layer in SOURCE_LAYERS:
            for key, value in source[level][layer].items():
                destination[level][layer][key] += value.detach()


def summarize(totals):
    output = {}
    for level in ("raw", "residual"):
        per_layer = {}
        for layer in SOURCE_LAYERS:
            values = totals[level][layer]
            direction, valid = values["direction_count"].item(), values["valid_count"].item()
            per_layer[str(layer)] = {
                "cosine_loss": float(values["cosine_loss_sum"] / max(direction, 1.0)),
                "relative_l2": float(values["relative_l2_sum"] / max(direction, 1.0)),
                "smooth_l1": float(values["smooth_l1_sum"] / max(valid, 1.0)),
                "norm_ratio": float(values["norm_ratio_sum"] / max(direction, 1.0)),
                "pred_norm": float(values["pred_norm_sum"] / max(valid, 1.0)),
                "teacher_norm": float(values["teacher_norm_sum"] / max(valid, 1.0)),
                "low_norm_excluded_fraction": float((valid - direction) / max(valid, 1.0)),
            }
        output[level] = {"per_layer": per_layer}
        for metric in ("cosine_loss", "relative_l2", "smooth_l1", "norm_ratio", "pred_norm", "teacher_norm", "low_norm_excluded_fraction"):
            output[level][metric] = sum(per_layer[str(layer)][metric] for layer in SOURCE_LAYERS) / len(SOURCE_LAYERS)
    return output


def ordered_groups(keys: Sequence[str], epoch: int, seed: int):
    order = list(keys)
    random.Random(seed + epoch).shuffle(order)
    retained = len(order) // 4 * 4
    return [order[i:i + 4] for i in range(0, retained, 4)], len(order) - retained


def save_checkpoint(path, predictor, optimizer, scheduler, epoch, args, metadata, metrics):
    payload = raw_predictor_checkpoint_payload(
        predictor, teacher_checkpoint=args.teacher_checkpoint, epoch=epoch, optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), metadata=metadata, metrics=metrics,
    )
    torch.save(payload, path)


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def loss_configuration(args) -> Dict[str, float]:
    return {
        name: float(getattr(args, name))
        for name in (
            "raw_cosine_weight", "raw_relative_l2_weight", "raw_smooth_l1_weight",
            "residual_cosine_weight", "residual_relative_l2_weight", "residual_smooth_l1_weight",
            "raw_level_weight", "residual_level_weight", "teacher_norm_eps",
        )
    }


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--siglip_feature_cache", required=True)
    p.add_argument("--cut3r_layer6_cache", required=True)
    p.add_argument("--cut3r_layer9_cache", required=True)
    p.add_argument("--cut3r_layer12_cache", required=True)
    p.add_argument("--teacher_checkpoint", default=TEACHER_DEFAULT)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alignment_report", required=True)
    p.add_argument("--predictor_type", choices=("raw_cut3r_token_mlp", "raw_cut3r_spatial_temporal"), required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation_fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--autocast", action="store_true")
    p.add_argument("--teacher_norm_eps", type=float, default=1e-6)
    p.add_argument("--raw_cosine_weight", type=float, default=1.0)
    p.add_argument("--raw_relative_l2_weight", type=float, default=0.25)
    p.add_argument("--raw_smooth_l1_weight", type=float, default=0.10)
    p.add_argument("--residual_cosine_weight", type=float, default=1.0)
    p.add_argument("--residual_relative_l2_weight", type=float, default=0.50)
    p.add_argument("--residual_smooth_l1_weight", type=float, default=0.10)
    p.add_argument("--raw_level_weight", type=float, default=1.0)
    p.add_argument("--residual_level_weight", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--residual_blocks", type=int, default=1)
    p.add_argument("--spatial_blocks", type=int, default=2)
    p.add_argument("--temporal_layers", type=int, default=4)
    p.add_argument("--temporal_heads", type=int, default=12)
    p.add_argument("--temporal_ffn_dim", type=int, default=3072)
    p.add_argument("--temporal_max_frames", type=int, default=64)
    p.add_argument("--adapter_dim", type=int, default=192)
    p.add_argument("--resume")
    p.add_argument("--max_train_samples", type=int)
    p.add_argument("--max_validation_samples", type=int)
    p.add_argument("--require_expected_split", action="store_true")
    return p


def main():
    args = parser().parse_args()
    device = setup_distributed()
    if world_size() not in (1, 4):
        raise RuntimeError("Raw experiment supports exactly world size 1 or 4.")
    report = json.loads(Path(args.alignment_report).read_text(encoding="utf-8"))
    frame_evidence = report.get("frame_identity_evidence", {})
    if report.get("status") == "ALIGNMENT_UNRESOLVED" or frame_evidence.get("status") != "verified":
        raise RuntimeError(
            "Raw distillation is blocked until the alignment report verifies paired source-video "
            "identity and exact frame order."
        )
    cache = RawCache(
        args.siglip_feature_cache,
        {6: args.cut3r_layer6_cache, 9: args.cut3r_layer9_cache, 12: args.cut3r_layer12_cache},
        args.seed,
        args.validation_fraction,
        alignment_artifact_verified=frame_evidence.get("status") == "verified",
    )
    if args.require_expected_split and (len(cache.train_keys), len(cache.validation_keys)) != (2198, 207):
        raise RuntimeError(f"Expected hash split 2198/207, got {len(cache.train_keys)}/{len(cache.validation_keys)}.")
    if args.max_train_samples:
        cache.train_keys = cache.train_keys[:args.max_train_samples]
    if args.max_validation_samples:
        cache.validation_keys = cache.validation_keys[:args.max_validation_samples]
    config = {"hidden_dim": args.hidden_dim, "residual_blocks": args.residual_blocks} if args.predictor_type.endswith("token_mlp") else {
        "hidden_dim": 768, "spatial_blocks": args.spatial_blocks, "temporal_layers": args.temporal_layers,
        "temporal_heads": args.temporal_heads, "temporal_ffn_dim": args.temporal_ffn_dim,
        "temporal_max_frames": args.temporal_max_frames, "adapter_dim": args.adapter_dim,
    }
    predictor = build_raw_cut3r_predictor(args.predictor_type, **config).to(device=device, dtype=torch.float32)
    postprocessor = FrozenSpatialStackPostprocessor.from_teacher_checkpoint(args.teacher_checkpoint, device=device, dtype=torch.float32)
    for parameter in postprocessor.parameters():
        if parameter.requires_grad:
            raise RuntimeError("Frozen SpatialStack postprocessor unexpectedly has trainable parameters.")
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    groups, dropped = ordered_groups(cache.train_keys, 0, args.seed)
    total_steps = len(groups) * args.epochs
    warmup_steps = int(total_steps * args.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(warmup_steps, 1)) if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1))))
    start_epoch = 0
    if args.resume:
        resume_info = [None]
        if is_rank0():
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint.get("format") != "raw_siglip_cut3r_predictor_v1":
                raise RuntimeError(f"Resume checkpoint is not a raw SigLIP/CUT3R predictor: {args.resume}")
            if checkpoint.get("architecture") != predictor.architecture_config():
                raise RuntimeError("Resume predictor architecture differs from the requested raw predictor.")
            resume_info[0] = {"epoch": int(checkpoint["epoch"]), "alignment_report_sha256": checkpoint.get("metadata", {}).get("run_metadata", {}).get("alignment", {}).get("report_sha256")}
        if dist.is_initialized():
            dist.broadcast_object_list(resume_info, src=0)
        if resume_info[0] is None:
            raise RuntimeError("Rank-safe resume state was not broadcast.")
        if resume_info[0]["alignment_report_sha256"] not in (None, canonical_sha256(report)):
            raise RuntimeError("Resume checkpoint alignment artifact differs from this launch.")
        barrier()
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        predictor.load_state_dict(checkpoint["predictor"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(resume_info[0]["epoch"]) + 1
    model = DDP(predictor, device_ids=[device.index], output_device=device.index) if world_size() > 1 else predictor
    output = Path(args.output_dir)
    if is_rank0():
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "distributed": {
                "world_size": world_size(), "rank_count": world_size(), "local_batch_size": 1,
                "gradient_accumulation": 4 if world_size() == 1 else 1, "effective_global_batch_size": 4,
                "sample_order_policy": "seeded global groups of four; rank i consumes group[i] in DDP",
                "retained_training_samples": len(groups) * 4, "dropped_incomplete_samples": dropped,
                "optimizer_steps_per_epoch": len(groups),
            },
            "split": {"seed": args.seed, "validation_fraction": args.validation_fraction,
                      "train": len(cache.train_keys), "validation": len(cache.validation_keys)},
            "alignment": {"report": str(Path(args.alignment_report).resolve()), "report_sha256": canonical_sha256(report),
                          "status": report.get("status"), "configuration": report.get("deterministic_resampling")},
            "teacher": {"checkpoint": args.teacher_checkpoint, "source_layers": list(SOURCE_LAYERS),
                        "qwen_layers": [0, 1, 2], "postprocessor": "resize_square_grid -> frozen branch -> residual_scale"},
            "predictor": {"architecture": predictor.architecture_config(),
                          "trainable_parameter_count": sum(p.numel() for p in predictor.parameters() if p.requires_grad)},
            "loss": loss_configuration(args), "git_revision": git_revision(),
            "precision": "fp32" if not args.autocast else "bf16_autocast_fp32_parameters",
            "resources": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK")},
        }
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    barrier()
    best = {"raw_relative_l2": float("inf"), "residual_relative_l2": float("inf"), "residual_cosine": float("inf")}
    for epoch in range(start_epoch, args.epochs):
        began = time.time()
        model.train()
        totals = fresh_totals(device)
        groups, _ = ordered_groups(cache.train_keys, epoch, args.seed)
        for group_index, group in enumerate(groups):
            optimizer.zero_grad(set_to_none=True)
            if world_size() == 1:
                for key in group:
                    loss, values = sample_forward(model, postprocessor, cache, key, device, args)
                    (loss / 4.0).backward()
                    add_totals(totals, values)
            else:
                loss, values = sample_forward(model, postprocessor, cache, group[rank()], device, args)
                loss.backward()
                add_totals(totals, values)
            if epoch == start_epoch and group_index == 0:
                missing_predictor_grads = [
                    name for name, parameter in predictor.named_parameters()
                    if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all() or not torch.count_nonzero(parameter.grad))
                ]
                if missing_predictor_grads:
                    raise RuntimeError(
                        "Full raw/residual objective failed to reach predictor parameters: "
                        f"{missing_predictor_grads[:8]}"
                    )
            frozen_grads = [name for name, parameter in postprocessor.named_parameters() if parameter.grad is not None]
            if frozen_grads:
                raise RuntimeError(f"Frozen postprocessor received gradients: {frozen_grads[:4]}")
            optimizer.step(); scheduler.step()
        train = summarize(merge_totals(totals, device))
        model.eval()
        validation_totals = fresh_totals(device)
        bare = model.module if isinstance(model, DDP) else model
        with torch.no_grad():
            for key in cache.validation_keys[rank()::world_size()]:
                _, values = sample_forward(bare, postprocessor, cache, key, device, args)
                add_totals(validation_totals, values)
        validation = summarize(merge_totals(validation_totals, device))
        if is_rank0():
            duration = time.time() - began
            record = {
                "epoch": epoch, "train": train, "validation": validation,
                "lr": optimizer.param_groups[0]["lr"], "duration_sec": duration,
                "throughput": {"training_samples_per_sec": (len(groups) * 4) / max(duration, 1e-8),
                               "optimizer_steps_per_sec": len(groups) / max(duration, 1e-8)},
            }
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            bare = model.module if isinstance(model, DDP) else model
            checkpoint_meta = {
                "run_metadata": metadata,
                "optimizer_step": (epoch + 1) * len(groups),
                "selection_metrics": {"raw_relative_l2": validation["raw"]["relative_l2"],
                                      "residual_relative_l2": validation["residual"]["relative_l2"],
                                      "residual_cosine_loss": validation["residual"]["cosine_loss"]},
            }
            save_checkpoint(output / "latest.pt", bare, optimizer, scheduler, epoch, args, checkpoint_meta, record)
            candidates = {
                "raw_relative_l2": validation["raw"]["relative_l2"],
                "residual_relative_l2": validation["residual"]["relative_l2"],
                "residual_cosine": validation["residual"]["cosine_loss"],
            }
            names = {"raw_relative_l2": "best_validation_raw_relative_l2.pt", "residual_relative_l2": "best_validation_residual_relative_l2.pt", "residual_cosine": "best_validation_residual_cosine.pt"}
            for metric, value in candidates.items():
                if value < best[metric]:
                    best[metric] = value
                    save_checkpoint(output / names[metric], bare, optimizer, scheduler, epoch, args, checkpoint_meta, record)
        barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
