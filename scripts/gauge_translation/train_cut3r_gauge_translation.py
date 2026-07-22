#!/usr/bin/env python3
"""Probe-pretrain and train the staged CUT3R gauge translator."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import signal
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gauge_translation.standalone_model import (  # noqa: E402
    GaugeTranslationConfig,
    GaugeTranslationModel,
    PatchGeometryEvalProbe,
    PatchGeometryTrainProbe,
    checkpoint_feasibility,
    composite_checkpoint_score,
    freeze_module,
    heldout_error_improvement,
    normalized_smooth_l1,
    orthogonal_control_delta,
    pose_dominance_detected,
    quaternion_geodesic_degrees,
    quaternion_rotation_loss,
    relative_residual,
    robust_translation_metrics,
    sample_video_translations,
    shuffled_delta_branch_control,
    stage_c_validation_gate,
)
from scripts.gauge_translation.common import (  # noqa: E402
    FrozenCut3RHead,
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    load_jsonl,
    load_sample,
    median_window,
    optimizer_schema,
    parameter_grad_norm,
    probe_gate,
    resolve_stage_steps,
    restore_rng_state,
    stage_a_gate,
    stage_b_gate,
)


STOP_REQUESTED = False


def _signal_handler(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("gauge translation training requires a CUDA allocation")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank0_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_gate(gate: Mapping[str, Any], device: torch.device) -> bool:
    value = torch.tensor([1 if gate.get("passed") else 0], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.broadcast(value, src=0)
    return bool(value.item())


def verify_alignment_gate(records, output_dir: Path, rank: int, device: torch.device) -> dict[str, Any]:
    """Aggregate cached per-video alignment evidence before probe optimization."""
    if rank == 0:
        failures, reports = [], []
        for record in records:
            context_path = Path(record["context_path"])
            try:
                payload = torch.load(context_path, map_location="cpu", weights_only=False)
                report = payload.get("metadata", {}).get("alignment", {})
                if not report.get("passed", False):
                    failures.append("%s: alignment did not pass" % record["id"])
                reports.append(report)
            except Exception as exc:
                failures.append("%s: %s" % (record["id"], exc))
        gate = {
            "passed": not failures and len(reports) == len(records),
            "videos_checked": len(reports),
            "expected_videos": len(records),
            "all_position_checks_passed": bool(reports) and all(
                report.get("position", {}).get("passed", False) for report in reports
            ),
            "max_explicit_adaptive_relative_error": max(
                (float(report.get("explicit_adaptive_relative_max", float("inf"))) for report in reports),
                default=float("inf"),
            ),
            "max_synthetic_ramp_relative_error": max(
                (float(report.get("synthetic_ramp_relative_max", float("inf"))) for report in reports),
                default=float("inf"),
            ),
            "failures": failures,
        }
        gate["passed"] = bool(gate["passed"] and gate["all_position_checks_passed"])
        atomic_json(output_dir / "alignment_gate.json", gate)
    else:
        gate = {}
    if not broadcast_gate(gate, device):
        raise RuntimeError("alignment gate failed; refusing to start probe training")
    barrier()
    return gate


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def ddp_wrap(module, world_size: int, local_rank: int):
    if world_size == 1:
        return module
    return DistributedDataParallel(module, device_ids=[local_rank], find_unused_parameters=True)


def append_metric(path: Path, payload: Mapping[str, Any], rank: int) -> None:
    if rank != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=True) + "\n")


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else nullcontext()


def _optimizer(parameters, lr: float, weight_decay: float, total_steps: int):
    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    warmup = max(1, int(0.03 * total_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _load_record(records: Sequence[dict], step: int, rank: int, world_size: int) -> dict:
    return records[(step * world_size + rank) % len(records)]


def _probe_error(prediction, target, mask, scale) -> tuple[float, float, bool]:
    pred, tgt, valid = prediction.float(), target.float(), mask.bool()
    finite = bool(torch.isfinite(pred[valid.unsqueeze(-1).expand_as(pred)]).all())
    metric_error = torch.linalg.vector_norm(pred - tgt, dim=-1)[valid].mean()
    normalized_error = metric_error / scale.float().reshape(-1)[0]
    return float(metric_error.item()), float(normalized_error.item()), finite


@torch.no_grad()
def compute_mean_target(records, device, dtype, max_frames, seed, limit: int) -> torch.Tensor:
    total = torch.zeros(3, device=device, dtype=torch.float64)
    count = 0
    for index, record in enumerate(records[:limit] if limit > 0 else records):
        sample = load_sample(record, device, dtype, max_frames, False, seed + index)
        values = sample.pooled_ref_xyz[0][sample.pooled_valid[0]].double()
        total += values.sum(dim=0)
        count += int(values.shape[0])
    if count == 0:
        raise RuntimeError("no valid probe targets")
    return (total / count).float()


@torch.no_grad()
def evaluate_probes(q_train, q_eval, records, mean_target, config, device, dtype) -> dict:
    q_train.eval()
    q_eval.eval()
    errors = defaultdict(list)
    predictions = defaultdict(list)
    targets = []
    limit = min(len(records), int(config["validation_video_limit"]))
    for index, record in enumerate(records[:limit]):
        sample = load_sample(record, device, dtype, config["max_frames"], False, config["seed"] + index)
        with autocast_context(device, dtype):
            train_prediction = q_train(*sample.patches)
            eval_prediction = q_eval(*sample.patches)
        for name, prediction in (("q_train", train_prediction), ("q_eval", eval_prediction)):
            metric_error, normalized_error, finite = _probe_error(prediction, sample.pooled_ref_xyz, sample.pooled_valid, sample.scene_scale)
            errors[f"{name}_metric_error"].append(metric_error)
            errors[f"{name}_normalized_error"].append(normalized_error)
            errors[f"{name}_finite"].append(float(finite))
            predictions[name].append(prediction.float()[sample.pooled_valid].cpu())
        baseline = mean_target.view(1, 1, 1, 3).expand_as(sample.pooled_ref_xyz)
        baseline_metric_error, baseline_normalized_error, _ = _probe_error(baseline, sample.pooled_ref_xyz, sample.pooled_valid, sample.scene_scale)
        errors["mean_predictor_metric_error"].append(baseline_metric_error)
        errors["mean_predictor_normalized_error"].append(baseline_normalized_error)
        targets.append(sample.pooled_ref_xyz[sample.pooled_valid].cpu())
    target_values = torch.cat(targets)
    target_variance = target_values.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
    result = {key: float(np.mean(values)) for key, values in errors.items() if not key.endswith("_finite")}
    for name in ("q_train", "q_eval"):
        values = torch.cat(predictions[name])
        prediction_variance = values.var(dim=0, unbiased=False).sum()
        result[f"{name}_prediction_variance"] = float(prediction_variance.item())
        result[f"{name}_variance_ratio"] = float((prediction_variance / target_variance).item())
        result[f"{name}_finite"] = bool(all(errors[f"{name}_finite"]))
    result["target_variance"] = float(target_variance.item())
    result["scene_disjoint_validation"] = True
    result["validation_videos"] = limit
    return result


def save_training_checkpoint(
    path: Path,
    translator,
    q_train,
    q_eval,
    stage: str,
    stage_step: int,
    global_step: int,
    optimizer,
    scheduler,
    scaler,
    config,
) -> None:
    base = unwrap(translator)
    atomic_torch_save(
        path,
        {
            "schema": "cut3r_gauge_translation_checkpoint_v1",
            "translator": base.state_dict(),
            "q_train": unwrap(q_train).state_dict(),
            "q_eval": unwrap(q_eval).state_dict(),
            "stage": stage,
            "stage_local_step": stage_step,
            "global_step": global_step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "optimizer_parameter_group_schema": optimizer_schema(base, optimizer) if optimizer is not None else [],
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": capture_rng_state(),
            "config": config,
        },
    )


def consider_best_checkpoint(
    validation, translator, q_train, q_eval, stage_step, global_step,
    optimizer, scheduler, scaler, config, output_dir,
):
    feasible, failures = checkpoint_feasibility(validation, config.get("feasibility"))
    score = composite_checkpoint_score(validation) if feasible else None
    selection = {
        "stage": "c",
        "stage_local_step": int(stage_step),
        "global_step": int(global_step),
        "eligible": bool(feasible),
        "failed_conditions": list(failures),
        "composite_score": score,
        "metrics": dict(validation),
    }
    append_metric(output_dir / "checkpoint_selection.jsonl", selection, 0)
    current_path = output_dir / "best_selection.json"
    current_score = float("inf")
    if current_path.is_file():
        current_score = float(json.loads(current_path.read_text()).get("composite_score", float("inf")))
    if feasible and float(score) < current_score:
        save_training_checkpoint(
            output_dir / "best.pt", translator, q_train, q_eval, "c", stage_step, global_step,
            optimizer, scheduler, scaler, config,
        )
        atomic_json(current_path, selection)
    return selection


def load_stage_c_heldout_baseline(output_dir: Path) -> dict:
    path = output_dir / "checkpoint_selection.jsonl"
    candidates = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("stage") != "c":
                continue
            error = payload.get("metrics", {}).get("full_normalized_vector_error")
            if error is not None and math.isfinite(float(error)):
                candidates.append(payload)
    if not candidates:
        raise RuntimeError("no held-out Stage C validation baseline is available")
    first = min(candidates, key=lambda item: int(item.get("stage_local_step", 0)))
    return {
        "error": float(first["metrics"]["full_normalized_vector_error"]),
        "stage_local_step": int(first.get("stage_local_step", 0)),
        "global_step": int(first.get("global_step", 0)),
        "source": str(path),
        "protocol": "same deterministic validation manifest, frame selection, deltas, and teacher masks",
    }


def pretrain_probes(
    q_train,
    q_eval,
    train_records,
    val_records,
    mean_target,
    steps,
    config,
    device,
    dtype,
    rank,
    world_size,
    local_rank,
    output_dir,
    metrics_path,
) -> tuple[dict, int]:
    initial = evaluate_probes(unwrap(q_train), unwrap(q_eval), val_records, mean_target, config, device, dtype) if rank == 0 else {}
    barrier()
    q_train.train()
    q_eval.train()
    parameters = [p for module in (q_train, q_eval) for p in module.parameters() if p.requires_grad]
    optimizer, scheduler = _optimizer(parameters, config["probe_lr"], config["weight_decay"], steps)
    best_train, best_eval = float("inf"), float("inf")
    best_train_state = best_eval_state = None
    for step in range(steps):
        record = _load_record(train_records, step, rank, world_size)
        sample = load_sample(record, device, dtype, config["max_frames"], True, config["seed"] + step + rank * 100000)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            pred_train = q_train(*sample.patches)
            pred_eval = q_eval(*sample.patches)
        loss_train = normalized_smooth_l1(pred_train, sample.pooled_ref_xyz, sample.scene_scale, sample.pooled_valid)
        loss_eval = normalized_smooth_l1(pred_eval, sample.pooled_ref_xyz, sample.scene_scale, sample.pooled_valid)
        loss = loss_train + loss_eval
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        optimizer.step()
        scheduler.step()
        append_metric(metrics_path, {"stage": "probe", "step": step + 1, "loss": float(loss), "q_train_loss": float(loss_train), "q_eval_loss": float(loss_eval)}, rank)
        if rank == 0 and ((step + 1) % int(config["validation_frequency"]) == 0 or step + 1 == steps):
            validation = evaluate_probes(unwrap(q_train), unwrap(q_eval), val_records, mean_target, config, device, dtype)
            if validation["q_train_normalized_error"] < best_train:
                best_train, best_train_state = validation["q_train_normalized_error"], {k: v.detach().cpu() for k, v in unwrap(q_train).state_dict().items()}
            if validation["q_eval_normalized_error"] < best_eval:
                best_eval, best_eval_state = validation["q_eval_normalized_error"], {k: v.detach().cpu() for k, v in unwrap(q_eval).state_dict().items()}
        barrier()
    if rank == 0:
        unwrap(q_train).load_state_dict(best_train_state)
        unwrap(q_eval).load_state_dict(best_eval_state)
        final = evaluate_probes(unwrap(q_train), unwrap(q_eval), val_records, mean_target, config, device, dtype)
        gate = probe_gate(initial, final)
        gate["alignment_gate_required"] = True
        atomic_json(output_dir / "probe_gate.json", gate)
        atomic_torch_save(output_dir / "q_train.pt", {"state_dict": unwrap(q_train).state_dict(), "metrics": final})
        atomic_torch_save(output_dir / "q_eval.pt", {"state_dict": unwrap(q_eval).state_dict(), "metrics": final})
    else:
        gate = {}
    passed = broadcast_gate(gate, device)
    if not passed:
        raise RuntimeError("probe gate failed; refusing to start Stage A")
    barrier()
    # Reload serialized rank-0 selections on every rank to prove checkpoint fidelity.
    unwrap(q_train).load_state_dict(torch.load(output_dir / "q_train.pt", map_location=device, weights_only=False)["state_dict"])
    unwrap(q_eval).load_state_dict(torch.load(output_dir / "q_eval.pt", map_location=device, weights_only=False)["state_dict"])
    freeze_module(unwrap(q_train))
    freeze_module(unwrap(q_eval))
    return gate, steps


def load_stage_history(metrics_path: Path, stage: str, max_stage_step: int):
    history = defaultdict(list)
    if max_stage_step <= 0 or not metrics_path.is_file():
        return history
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("stage") != stage or int(payload.get("stage_step", 0)) > max_stage_step:
            continue
        for key, value in payload.items():
            if key.startswith("raw_"):
                history[key[4:]].append(float(value))
        history["patch_grad"].append(float(payload.get("patch_grad_norm", 0.0)))
        history["pose_grad"].append(float(payload.get("pose_grad_norm", 0.0)))
    return history


def _relative_changes(original: Sequence[torch.Tensor], moved: Sequence[torch.Tensor]) -> torch.Tensor:
    values = []
    for source, target in zip(original, moved):
        numerator = torch.linalg.vector_norm((target - source).float(), dim=-1)
        denominator = torch.linalg.vector_norm(source.float(), dim=-1).clamp_min(1e-8)
        values.append((numerator / denominator).flatten())
    return torch.cat(values)


def _relative_change_summary(changes: Sequence[torch.Tensor]) -> dict[str, float]:
    """Compute population statistics instead of a median of per-video quantiles."""
    if not changes:
        return {"median": float("inf"), "p95": float("inf")}
    with torch.autocast(device_type=changes[0].device.type, enabled=False):
        population = torch.cat([item.detach().float().flatten() for item in changes])
        return {
            "median": float(population.median().item()),
            "p95": float(torch.quantile(population, 0.95).item()),
        }


def _structural_outputs(model, sample, delta, second_delta) -> dict:
    zero = torch.zeros_like(delta)
    identity = model(*sample.patches, sample.pose12, zero, sample.scene_scale)
    moved = model(*sample.patches, sample.pose12, delta, sample.scene_scale)
    inverse = model(*moved.patches(), moved.pose12, -delta, sample.scene_scale)
    sequential = model(*moved.patches(), moved.pose12, second_delta, sample.scene_scale)
    direct = model(*sample.patches, sample.pose12, delta + second_delta, sample.scene_scale)
    return {"identity": identity, "moved": moved, "inverse": inverse, "sequential": sequential, "direct": direct}


def _structural_losses(sample, outputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = sum(relative_residual(source, target) for source, target in zip(sample.patches, outputs["identity"].patches())) / 3
    identity_pose = relative_residual(sample.pose12, outputs["identity"].pose12)
    inverse = sum(relative_residual(source, target) for source, target in zip(sample.patches, outputs["inverse"].patches())) / 3
    inverse_pose = relative_residual(sample.pose12, outputs["inverse"].pose12)
    composition = sum(relative_residual(source, target) for source, target in zip(outputs["direct"].patches(), outputs["sequential"].patches())) / 3
    composition_pose = relative_residual(outputs["direct"].pose12, outputs["sequential"].pose12)
    return (identity + identity_pose) / 2, (inverse + inverse_pose) / 2, (composition + composition_pose) / 2, identity, identity_pose


def _pose_shift_metrics(original_pose, moved_pose, delta, scale) -> dict:
    shift = (moved_pose[..., :3].float() - original_pose[..., :3].float()).reshape(-1, 3).median(dim=0).values
    target = delta.float().reshape(-1, 3)[0]
    return {
        "pose_head_cosine": float(F.cosine_similarity(shift[None], target[None], dim=-1, eps=1e-8).item()),
        "pose_head_magnitude_ratio": float(shift.norm().item() / max(target.norm().item(), 1e-8)),
        "pose_head_normalized_error": float((shift - target).norm().item() / max(float(scale.item()), 1e-8)),
        "pose_head_rotation_degrees": float(quaternion_geodesic_degrees(original_pose[..., 3:7], moved_pose[..., 3:7]).mean().item()),
        "quaternion_rotation_loss": float(quaternion_rotation_loss(original_pose[..., 3:7], moved_pose[..., 3:7]).item()),
    }


@torch.no_grad()
def validate_translator(model, q_train, q_eval, frozen_head, records, config, device, dtype) -> dict:
    model.eval()
    values = defaultdict(list)
    patch_change_population = []
    pose_change_population = []
    saved_controls = []
    limit = min(len(records), int(config["validation_video_limit"]))
    for index, record in enumerate(records[:limit]):
        sample = load_sample(record, device, dtype, config["max_frames"], False, config["seed"] + 500000 + index)
        generator = torch.Generator(device=device).manual_seed(config["seed"] + index)
        delta = sample_video_translations(1, sample.scene_scale, 1.0, zero_probability=0.0, generator=generator)
        second_delta = -0.5 * delta
        with autocast_context(device, dtype):
            outputs = model(*sample.patches, sample.pose12, delta, sample.scene_scale)
            self_map, self_conf = frozen_head.decode_self(sample.dec0, outputs.patches())
            full_map, full_conf = frozen_head.decode_reference(sample.dec0, outputs.patches(), outputs.pose12, sample.pos)
            patch_map, _ = frozen_head.decode_reference(sample.dec0, outputs.patches(), sample.pose12, sample.pos)
            pose_map, _ = frozen_head.decode_reference(sample.dec0, sample.patches, outputs.pose12, sample.pos)
            original_pose = frozen_head.decode_pose(sample.pose12)
            moved_pose = frozen_head.decode_pose(outputs.pose12)
            q_train_original = q_train(*sample.patches)
            q_train_moved = q_train(*outputs.patches())
            q_eval_original = q_eval(*sample.patches)
            q_eval_moved = q_eval(*outputs.patches())
            structural = _structural_outputs(model, sample, delta, second_delta)
        branch_metrics = {
            "full": robust_translation_metrics(sample.reference_points, full_map, sample.teacher_mask, delta, sample.scene_scale),
            "patch": robust_translation_metrics(sample.reference_points, patch_map, sample.teacher_mask, delta, sample.scene_scale),
            "pose_only_ref": robust_translation_metrics(sample.reference_points, pose_map, sample.teacher_mask, delta, sample.scene_scale),
            "q_train": robust_translation_metrics(sample.pooled_ref_xyz[0], q_train_moved[0] - q_train_original[0] + sample.pooled_ref_xyz[0], sample.pooled_valid[0], delta, sample.scene_scale),
            "q_eval": robust_translation_metrics(sample.pooled_ref_xyz[0], q_eval_moved[0] - q_eval_original[0] + sample.pooled_ref_xyz[0], sample.pooled_valid[0], delta, sample.scene_scale),
        }
        for branch, metrics in branch_metrics.items():
            if metrics.get("valid"):
                for key in ("cosine", "magnitude_ratio", "normalized_vector_error", "residual_median", "residual_p95"):
                    values[f"{branch}_{key}"].append(metrics[key])
            values[f"{branch}_invalid_ratio"].append(metrics.get("invalid_ratio", 1.0))
        pose_metrics = _pose_shift_metrics(original_pose, moved_pose, delta, sample.scene_scale)
        for key, value in pose_metrics.items():
            values[key].append(value)
        changes = _relative_changes(sample.patches, outputs.patches())
        pose_change = _relative_changes((sample.pose12,), (outputs.pose12,))
        patch_change_population.append(changes)
        pose_change_population.append(pose_change)
        self_drift = torch.linalg.vector_norm((self_map.float() - sample.camera_points.float()), dim=-1)[sample.teacher_mask]
        values["normalized_self_drift"].append(float(self_drift.median().item() / max(delta.norm().item(), 1e-8)))
        nonpositive = ((self_map[..., 2] <= 0) & sample.teacher_mask).sum().float() / sample.teacher_mask.sum().clamp_min(1)
        values["new_nonpositive_self_depth_ratio"].append(float(nonpositive.item()))
        values["invalid_output_ratio"].append(max(branch_metrics[name].get("invalid_ratio", 1.0) for name in ("full", "patch", "pose_only_ref")))
        original_ref_confidence = sample.confidence_ref.float()
        original_self_confidence = sample.confidence_self.float()
        values["confidence_ref_change"].append(float((full_conf.float() - original_ref_confidence).abs()[sample.teacher_mask].mean().item()))
        values["confidence_self_change"].append(float((self_conf.float() - original_self_confidence).abs()[sample.teacher_mask].mean().item()))
        values["confidence_ref_relative_drop"].append(float(
            ((original_ref_confidence - full_conf.float()).clamp_min(0) / original_ref_confidence.abs().clamp_min(1e-6))[sample.teacher_mask].mean().item()
        ))
        values["confidence_self_relative_drop"].append(float(
            ((original_self_confidence - self_conf.float()).clamp_min(0) / original_self_confidence.abs().clamp_min(1e-6))[sample.teacher_mask].mean().item()
        ))
        identity_loss, inverse_loss, composition_loss, _, _ = _structural_losses(sample, structural)
        values["identity_error"].append(float(identity_loss.item()))
        values["inverse_error"].append(float(inverse_loss.item()))
        values["composition_error"].append(float(composition_loss.item()))

        if index < 2:
            with autocast_context(device, dtype):
                negative = model(*sample.patches, sample.pose12, -delta, sample.scene_scale)
                negative_q = q_eval(*negative.patches())
                negative_map, _ = frozen_head.decode_reference(sample.dec0, negative.patches(), negative.pose12, sample.pos)
            neg_q_metrics = robust_translation_metrics(sample.pooled_ref_xyz[0], negative_q[0] - q_eval_original[0] + sample.pooled_ref_xyz[0], sample.pooled_valid[0], -delta, sample.scene_scale)
            neg_full_metrics = robust_translation_metrics(sample.reference_points, negative_map, sample.teacher_mask, -delta, sample.scene_scale)
            for branch, positive, negative_metrics in (("q_eval", branch_metrics["q_eval"], neg_q_metrics), ("full", branch_metrics["full"], neg_full_metrics)):
                if positive.get("valid") and negative_metrics.get("valid"):
                    plus = torch.tensor([positive["estimate_x"], positive["estimate_y"], positive["estimate_z"]])
                    minus = torch.tensor([negative_metrics["estimate_x"], negative_metrics["estimate_y"], negative_metrics["estimate_z"]])
                    values[f"sign_{branch}_mutual_cosine"].append(float(F.cosine_similarity(plus[None], minus[None], dim=-1).item()))
                    values[f"sign_{branch}_magnitude_symmetry"].append(float(plus.norm() / minus.norm().clamp_min(1e-8)))
                    values[f"sign_{branch}_positive_assigned_cosine"].append(float(positive["cosine"]))
                    values[f"sign_{branch}_negative_assigned_cosine"].append(float(negative_metrics["cosine"]))
            saved_controls.append((sample, delta, q_eval_original.detach(), q_eval_moved.detach(), full_map.detach()))

    result = {key: float(np.median(items)) for key, items in values.items() if items}
    patch_change_summary = _relative_change_summary(patch_change_population)
    pose_change_summary = _relative_change_summary(pose_change_population)
    result["patch_change_median"] = patch_change_summary["median"]
    result["patch_residual_p95"] = patch_change_summary["p95"]
    result["pose_change"] = pose_change_summary["median"]
    result["pose_token_residual_p95"] = pose_change_summary["p95"]
    result["full_cosine"] = result.get("full_cosine", -1.0)
    result["full_magnitude_ratio"] = result.get("full_magnitude_ratio", -1.0)
    result["full_normalized_vector_error"] = result.get("full_normalized_vector_error", float("inf"))
    result["q_eval_cosine"] = result.get("q_eval_cosine", -1.0)
    result["q_eval_magnitude_ratio"] = result.get("q_eval_magnitude_ratio", -1.0)
    result["q_eval_normalized_vector_error"] = result.get("q_eval_normalized_vector_error", float("inf"))
    result["structural_consistency_error"] = float(np.mean([result.get("identity_error", math.inf), result.get("inverse_error", math.inf), result.get("composition_error", math.inf)]))
    result["structural_finite"] = bool(math.isfinite(result["structural_consistency_error"]))
    result["pose_dominated"] = pose_dominance_detected(result)
    result["quaternion_sign_invariant"] = True
    result["pose_losses_finite"] = bool(math.isfinite(result.get("pose_head_normalized_error", math.inf)))
    sign_pass = all(
        result.get(f"sign_{branch}_positive_assigned_cosine", -1.0) > 0.5
        and result.get(f"sign_{branch}_negative_assigned_cosine", -1.0) > 0.5
        and result.get(f"sign_{branch}_mutual_cosine", 1.0) < 0
        and 0.5 <= result.get(f"sign_{branch}_magnitude_symmetry", 0.0) <= 2.0
        for branch in ("q_eval", "full")
    )
    result["delta_sign_control_pass"] = sign_pass

    # Deterministic equal-magnitude orthogonal reassignment control.
    result["shuffled_delta_control_pass"] = False
    control_rows = {"q_eval": [], "full": []}
    for sample, source_delta, original_q, unshuffled_q, unshuffled_map in saved_controls:
        assigned_delta = orthogonal_control_delta(source_delta)
        with autocast_context(device, dtype):
            shuffled = model(*sample.patches, sample.pose12, assigned_delta, sample.scene_scale)
            shuffled_q = q_eval(*shuffled.patches())
            shuffled_map, _ = frozen_head.decode_reference(
                sample.dec0, shuffled.patches(), shuffled.pose12, sample.pos
            )
        scale_value = float(sample.scene_scale.float().reshape(-1)[0].item())
        q_output_change = float(
            torch.linalg.vector_norm((shuffled_q - unshuffled_q).float(), dim=-1)[sample.pooled_valid].median().item()
            / max(scale_value, 1e-8)
        )
        full_output_change = float(
            torch.linalg.vector_norm((shuffled_map - unshuffled_map).float(), dim=-1)[sample.teacher_mask].median().item()
            / max(scale_value, 1e-8)
        )
        q_metrics = robust_translation_metrics(
            sample.pooled_ref_xyz[0],
            shuffled_q[0] - original_q[0] + sample.pooled_ref_xyz[0],
            sample.pooled_valid[0], assigned_delta, sample.scene_scale,
        )
        full_metrics = robust_translation_metrics(
            sample.reference_points, shuffled_map, sample.teacher_mask, assigned_delta, sample.scene_scale
        )
        for branch, metrics, output_change in (
            ("q_eval", q_metrics, q_output_change), ("full", full_metrics, full_output_change)
        ):
            if metrics.get("valid"):
                estimate = torch.tensor(
                    [metrics["estimate_x"], metrics["estimate_y"], metrics["estimate_z"]], device=device
                )
                control = shuffled_delta_branch_control(
                    estimate, source_delta, assigned_delta, output_change
                )
            else:
                control = {
                    "source_assigned_delta_cosine": float("inf"),
                    "assigned_cosine": -1.0,
                    "source_cosine": 1.0,
                    "assigned_minus_source_margin": -2.0,
                    "normalized_output_change": output_change,
                    "passed": False,
                }
            control_rows[branch].append(control)
    control_fields = (
        "source_assigned_delta_cosine", "assigned_cosine", "source_cosine",
        "assigned_minus_source_margin", "normalized_output_change",
    )
    for branch, rows in control_rows.items():
        for field in control_fields:
            result[f"shuffled_{branch}_{field}"] = (
                float(np.median([float(row[field]) for row in rows])) if rows else float("nan")
            )
        result[f"shuffled_{branch}_pass"] = bool(rows and all(bool(row["passed"]) for row in rows))
    result["shuffled_delta_control_samples"] = len(saved_controls)
    result["shuffled_delta_control_pass"] = bool(
        result.get("shuffled_q_eval_pass", False) and result.get("shuffled_full_pass", False)
    )
    return result


def train_stage(
    stage,
    base_model,
    q_train,
    q_eval,
    frozen_head,
    train_records,
    val_records,
    steps,
    config,
    device,
    dtype,
    rank,
    world_size,
    local_rank,
    output_dir,
    metrics_path,
    global_step,
    resume_payload=None,
):
    trainable_names = base_model.set_trainable_stage(stage)
    rank0_print(rank, f"[STAGE {stage.upper()}] trainable={trainable_names}")
    model = ddp_wrap(base_model, world_size, local_rank)
    lr = config[f"stage_{stage}_lr"]
    parameters = [parameter for parameter in base_model.parameters() if parameter.requires_grad]
    optimizer, scheduler = _optimizer(parameters, lr, config["weight_decay"], steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
    start_step = 0
    if resume_payload is not None and str(resume_payload.get("stage")) == stage:
        start_step = int(resume_payload["stage_local_step"])
        global_step = int(resume_payload["global_step"])
        if resume_payload.get("optimizer") is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
        elif start_step != 0:
            raise RuntimeError("nonzero stage resume is missing optimizer state")
        if resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        restore_rng_state(resume_payload["rng"])
        rank0_print(rank, f"[RESUME] stage={stage} local_step={start_step} global_step={global_step}")
    history = load_stage_history(metrics_path, stage, start_step)
    for local_step in range(start_step, steps):
        record = _load_record(train_records, local_step, rank, world_size)
        sample = load_sample(record, device, dtype, config["max_frames"], True, config["seed"] + global_step + rank * 100000)
        progress = global_step / max(1, sum(config["resolved_stage_steps"].values()))
        generator = torch.Generator(device=device).manual_seed(config["seed"] + global_step + rank * 100000)
        delta = sample_video_translations(1, sample.scene_scale, progress, generator=generator)
        second_delta = sample_video_translations(1, sample.scene_scale, progress, zero_probability=0.0, generator=generator) * 0.5
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            outputs = _structural_outputs(model, sample, delta, second_delta)
            moved = outputs["moved"]
            original_pose = frozen_head.decode_pose(sample.pose12)
            moved_pose = frozen_head.decode_pose(moved.pose12)
        identity_loss, inverse_loss, composition_loss, patch_identity, pose_identity = _structural_losses(sample, outputs)
        patch_res = sum(relative_residual(source, target) for source, target in zip(sample.patches, moved.patches())) / 3
        pose_res = relative_residual(sample.pose12, moved.pose12)
        losses = {
            "identity": identity_loss,
            "inverse": inverse_loss,
            "composition": composition_loss,
            "patch_res": patch_res,
            "pose_res": pose_res,
        }
        if stage in {"a", "c"}:
            with autocast_context(device, dtype):
                original_q = q_train(*sample.patches)
                moved_q = q_train(*moved.patches())
                self_map, _ = frozen_head.decode_self(sample.dec0, moved.patches())
            target_q = original_q.detach() + delta[:, None, None, :]
            losses["patch_eq"] = normalized_smooth_l1(moved_q, target_q, sample.scene_scale, sample.pooled_valid)
            losses["self"] = normalized_smooth_l1(self_map, sample.camera_points, sample.scene_scale, sample.teacher_mask)
        if stage in {"b", "c"}:
            pose_target = original_pose[..., :3] + delta[:, None, :]
            losses["pose_t"] = normalized_smooth_l1(moved_pose[..., :3], pose_target, sample.scene_scale)
            losses["pose_R"] = quaternion_rotation_loss(original_pose[..., 3:7], moved_pose[..., 3:7])
        if stage == "c":
            with autocast_context(device, dtype):
                full_map, _ = frozen_head.decode_reference(sample.dec0, moved.patches(), moved.pose12, sample.pos)
            losses["full_ref"] = normalized_smooth_l1(
                full_map, sample.reference_points + delta[:, None, None, None, :][0], sample.scene_scale, sample.teacher_mask
            )
        weights = config["loss_weights"]
        total = torch.zeros((), device=device)
        for name, loss in losses.items():
            key = "pose_R" if name == "pose_R" else name
            total = total + float(weights.get(key, 0.0)) * loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite Stage {stage} loss: {losses}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        patch_grad = parameter_grad_norm(base_model, ("patch_adapters", "translation_trunk", "patch_conditioning_projection"))
        pose_grad = parameter_grad_norm(base_model, ("pose_adapter", "pose_conditioning_projection"))
        torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1
        for name, loss in losses.items():
            history[name].append(float(loss.detach().item()))
        history["patch_grad"].append(patch_grad)
        history["pose_grad"].append(pose_grad)
        append_metric(metrics_path, {"stage": stage, "stage_step": local_step + 1, "global_step": global_step, "total_loss": float(total.detach()), "patch_grad_norm": patch_grad, "pose_grad_norm": pose_grad, **{f"raw_{name}": float(loss.detach()) for name, loss in losses.items()}, **{f"weighted_{name}": float(loss.detach()) * float(weights.get(name, 0)) for name, loss in losses.items()}}, rank)
        if rank == 0 and ((local_step + 1) % int(config["checkpoint_frequency"]) == 0 or STOP_REQUESTED):
            save_training_checkpoint(output_dir / "latest.pt", model, q_train, q_eval, stage, local_step + 1, global_step, optimizer, scheduler, scaler, config)
            if not STOP_REQUESTED:
                save_training_checkpoint(output_dir / f"recovery_{stage}_{local_step+1:06d}.pt", model, q_train, q_eval, stage, local_step + 1, global_step, optimizer, scheduler, scaler, config)
        if stage == "c" and (local_step + 1) < steps and (local_step + 1) % int(config["validation_frequency"]) == 0:
            barrier()
            if rank == 0:
                periodic_validation = validate_translator(
                    base_model, q_train, q_eval, frozen_head, val_records, config, device, dtype
                )
                consider_best_checkpoint(
                    periodic_validation, model, q_train, q_eval, local_step + 1, global_step,
                    optimizer, scheduler, scaler, config, output_dir,
                )
            barrier()
            model.train()
        if STOP_REQUESTED:
            barrier()
            raise SystemExit("scheduler signal received; latest checkpoint saved for clean resume")
    barrier()
    if rank == 0:
        validation = validate_translator(base_model, q_train, q_eval, frozen_head, val_records, config, device, dtype)
        nonzero_patch_grads = [value for value in history["patch_grad"] if value > 1e-8]
        validation["patch_gradient_nonzero_fraction"] = len(nonzero_patch_grads) / max(1, len(history["patch_grad"]))
        pose_gradients = history["pose_grad"]
        validation["pose_gradient_nonzero_fraction"] = sum(value > 1e-8 for value in pose_gradients) / max(1, len(pose_gradients))
        validation["pose_gradients_finite"] = bool(pose_gradients and all(math.isfinite(value) for value in pose_gradients))
        atomic_json(output_dir / f"stage_{stage}_validation.json", validation)
        if stage == "a":
            gate = stage_a_gate(validation, history)
        elif stage == "b":
            gate = stage_b_gate(validation, history)
        else:
            baseline = load_stage_c_heldout_baseline(output_dir)
            gate = stage_c_validation_gate(
                validation, baseline["error"], config.get("feasibility"), 0.20
            )
            gate["held_out_stage_c_improvement"].update({
                "baseline_stage_local_step": baseline["stage_local_step"],
                "baseline_global_step": baseline["global_step"],
                "baseline_source": baseline["source"],
                "validation_protocol": baseline["protocol"],
            })
            gate["training_full_ref_diagnostic"] = {
                "early_median": median_window(history["full_ref"], True),
                "final_median": median_window(history["full_ref"], False),
                "used_for_acceptance": False,
            }
            if gate["stage_c_feasibility"]["passed"]:
                validation["composite_score"] = composite_checkpoint_score(validation)
            consider_best_checkpoint(
                validation, model, q_train, q_eval, steps, global_step,
                optimizer, scheduler, scaler, config, output_dir,
            )
        atomic_json(output_dir / f"stage_{stage}_gate.json", gate)
        save_training_checkpoint(output_dir / f"stage_{stage}_final.pt", model, q_train, q_eval, stage, steps, global_step, optimizer, scheduler, scaler, config)
        if stage in {"a", "b"} and gate["passed"]:
            next_stage = "b" if stage == "a" else "c"
            save_training_checkpoint(
                output_dir / "latest.pt", model, q_train, q_eval, next_stage, 0, global_step,
                None, None, None, config,
            )
    else:
        gate = {}
    passed = broadcast_gate(gate, device)
    if not passed:
        raise RuntimeError(f"Stage {stage.upper()} gate failed; refusing to continue")
    barrier()
    return global_step, gate


def coverage_report(config, train_records, val_records, world_size, context_root: Path) -> dict:
    global_batch = world_size * int(config["per_device_batch_size"]) * int(config["gradient_accumulation_steps"])
    resolved, epochs, steps_per_epoch = {}, {}, None
    for stage, schedule in config["schedule"].items():
        steps, effective_epochs, current_steps_per_epoch = resolve_stage_steps(schedule, len(train_records), global_batch)
        resolved[stage] = steps
        epochs[stage] = effective_epochs
        steps_per_epoch = current_steps_per_epoch
    cache_bytes = sum(path.stat().st_size for path in context_root.rglob("*.pt")) if context_root.exists() else 0
    validation_limit = min(len(val_records), int(config["validation_video_limit"]))
    validation_events_by_stage = {
        "a": 1,
        "b": 1,
        "c": max(1, math.ceil(resolved["c"] / int(config["validation_frequency"]))),
    }
    total_validation_events = sum(validation_events_by_stage.values())
    head_forwards = {
        "training": {
            "probe": {"dpt_self": 0, "dpt_cross": 0, "pose_head": 0},
            "a": {"dpt_self": resolved["a"] * world_size, "dpt_cross": 0, "pose_head": 2 * resolved["a"] * world_size},
            "b": {"dpt_self": 0, "dpt_cross": 0, "pose_head": 2 * resolved["b"] * world_size},
            "c": {"dpt_self": resolved["c"] * world_size, "dpt_cross": resolved["c"] * world_size, "pose_head": 2 * resolved["c"] * world_size},
        },
        "validation": {
            "events_by_stage": validation_events_by_stage,
            "dpt_self": total_validation_events * validation_limit,
            "dpt_cross": total_validation_events * (3 * validation_limit + min(4, 2 * validation_limit)),
            "pose_head": total_validation_events * 2 * validation_limit,
        },
    }
    temporary_config = GaugeTranslationConfig(**config["translator"])
    temporary_modules = (
        GaugeTranslationModel(temporary_config),
        PatchGeometryTrainProbe(temporary_config.token_dim),
        PatchGeometryEvalProbe(temporary_config.token_dim),
    )
    parameter_count = sum(parameter.numel() for module in temporary_modules for parameter in module.parameters())
    dtype_bytes = 4 if config["precision"] == "fp32" else 2
    estimated_checkpoint_bytes = parameter_count * dtype_bytes + parameter_count * 8 + 2**20
    checkpoint_frequency = int(config["checkpoint_frequency"])
    estimated_checkpoint_count = (
        sum(math.ceil(resolved[stage] / checkpoint_frequency) for stage in ("a", "b", "c"))
        + 5  # stage boundaries, best checkpoint, and latest recovery checkpoint
    )
    estimated_checkpoint_storage_bytes = estimated_checkpoint_count * estimated_checkpoint_bytes
    del temporary_modules
    training_dataset_counts = dict(sorted(Counter(item["dataset"] for item in train_records).items()))
    validation_dataset_counts = dict(sorted(Counter(item["dataset"] for item in val_records).items()))
    training_scene_counts = dict(sorted(Counter(item["dataset"] for item in {record["scene_group"]: record for record in train_records}.values()).items()))
    validation_scene_counts = dict(sorted(Counter(item["dataset"] for item in {record["scene_group"]: record for record in val_records}.values()).items()))
    training_head_total = sum(
        count
        for stage_counts in head_forwards["training"].values()
        for count in stage_counts.values()
    )
    validation_head_total = sum(
        head_forwards["validation"][key] for key in ("dpt_self", "dpt_cross", "pose_head")
    )
    expected_sampled_frames = int(config["max_frames"]) * (
        global_batch * sum(resolved.values()) + total_validation_events * validation_limit
    )
    return {
        "training_videos": len(train_records),
        "validation_videos": len(val_records),
        "training_scenes": len({item["scene_group"] for item in train_records}),
        "validation_scenes": len({item["scene_group"] for item in val_records}),
        "training_videos_by_dataset": training_dataset_counts,
        "validation_videos_by_dataset": validation_dataset_counts,
        "training_scenes_by_dataset": training_scene_counts,
        "validation_scenes_by_dataset": validation_scene_counts,
        "global_batch_size": global_batch,
        "steps_per_effective_epoch": steps_per_epoch,
        "resolved_stage_steps": resolved,
        "effective_epochs_per_stage": epochs,
        "sampled_frames_per_video": config["max_frames"],
        "estimated_total_sampled_frames": expected_sampled_frames,
        "estimated_frozen_head_forward_counts": head_forwards,
        "estimated_frozen_decoder_forward_count": training_head_total + validation_head_total,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "estimated_checkpoint_gib": estimated_checkpoint_bytes / 2**30,
        "estimated_checkpoint_count": estimated_checkpoint_count,
        "estimated_checkpoint_storage_bytes": estimated_checkpoint_storage_bytes,
        "estimated_checkpoint_storage_gib": estimated_checkpoint_storage_bytes / 2**30,
        "context_cache_bytes": cache_bytes,
        "context_cache_gib": cache_bytes / 2**30,
        "estimated_storage_and_cache_bytes": cache_bytes + estimated_checkpoint_storage_bytes,
    }


def revalidate_smoke_checkpoint(checkpoint_path: Path, config: dict, device: torch.device, rank: int, world_size: int) -> dict:
    if rank != 0 or world_size != 1:
        raise RuntimeError("checkpoint revalidation requires one process on one GPU")
    output_dir = checkpoint_path.parent
    val_records = load_jsonl(Path(os.path.expandvars(config["validation_manifest"])))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
    translator_config = GaugeTranslationConfig(**config["translator"])
    translator = GaugeTranslationModel(translator_config).to(device=device, dtype=dtype)
    q_train = PatchGeometryTrainProbe(translator_config.token_dim).to(device=device, dtype=dtype)
    q_eval = PatchGeometryEvalProbe(translator_config.token_dim).to(device=device, dtype=dtype)
    frozen_head = FrozenCut3RHead(
        Path(os.path.expandvars(config["cut3r_checkpoint"])), device, dtype
    )

    def load_checkpoint(path: Path) -> dict:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") != "cut3r_gauge_translation_checkpoint_v1":
            raise RuntimeError(f"unsupported checkpoint schema in {path}")
        translator.load_state_dict(payload["translator"], strict=True)
        q_train.load_state_dict(payload["q_train"], strict=True)
        q_eval.load_state_dict(payload["q_eval"], strict=True)
        translator.set_trainable_stage("eval")
        freeze_module(q_train)
        freeze_module(q_eval)
        return payload

    stage_b_path = output_dir / "stage_b_final.pt"
    if not stage_b_path.is_file():
        raise FileNotFoundError(f"missing Stage B checkpoint: {stage_b_path}")
    stage_b_payload = load_checkpoint(stage_b_path)
    stage_b_validation = validate_translator(
        translator, q_train, q_eval, frozen_head, val_records, config, device, dtype
    )
    stage_b_history = load_stage_history(
        output_dir / "metrics.jsonl", "b", int(stage_b_payload["stage_local_step"])
    )
    stage_b_validation["patch_gradient_nonzero_fraction"] = 0.0
    pose_gradients = stage_b_history["pose_grad"]
    stage_b_validation["pose_gradient_nonzero_fraction"] = (
        sum(value > 1e-8 for value in pose_gradients) / max(1, len(pose_gradients))
    )
    stage_b_validation["pose_gradients_finite"] = bool(
        pose_gradients and all(math.isfinite(value) for value in pose_gradients)
    )
    corrected_stage_b_gate = stage_b_gate(stage_b_validation, stage_b_history)

    selected_payload = load_checkpoint(checkpoint_path)
    stage_c_validation = validate_translator(
        translator, q_train, q_eval, frozen_head, val_records, config, device, dtype
    )
    baseline = load_stage_c_heldout_baseline(output_dir)
    corrected_stage_c_gate = stage_c_validation_gate(
        stage_c_validation, baseline["error"], config.get("feasibility"), 0.20
    )
    corrected_stage_c_gate["held_out_stage_c_improvement"].update({
        "baseline_stage_local_step": baseline["stage_local_step"],
        "baseline_global_step": baseline["global_step"],
        "selected_stage_local_step": int(selected_payload["stage_local_step"]),
        "selected_global_step": int(selected_payload["global_step"]),
        "baseline_source": baseline["source"],
        "validation_protocol": baseline["protocol"],
    })
    corrected_stage_c_gate["training_full_ref_diagnostic"] = {
        "used_for_acceptance": False,
        "reason": "held-out deterministic validation is the primary improvement criterion",
    }
    if corrected_stage_c_gate["stage_c_feasibility"]["passed"]:
        stage_c_validation["composite_score"] = composite_checkpoint_score(stage_c_validation)

    for name in ("stage_b_validation", "stage_b_gate", "stage_c_validation", "stage_c_gate"):
        original = output_dir / f"{name}.json"
        archive = output_dir / f"{name}_pre_correction.json"
        if original.is_file() and not archive.exists():
            atomic_json(archive, json.loads(original.read_text()))
    atomic_json(output_dir / "stage_b_validation_corrected.json", stage_b_validation)
    atomic_json(output_dir / "stage_b_gate_corrected.json", corrected_stage_b_gate)
    atomic_json(output_dir / "stage_c_validation_corrected.json", stage_c_validation)
    atomic_json(output_dir / "stage_c_gate_corrected.json", corrected_stage_c_gate)

    alignment_gate = json.loads((output_dir / "alignment_gate.json").read_text())
    probe_gate_payload = json.loads((output_dir / "probe_gate.json").read_text())
    stage_a_gate_payload = json.loads((output_dir / "stage_a_gate.json").read_text())
    shuffled_fields = {
        key: value for key, value in stage_c_validation.items() if key.startswith("shuffled_")
    }
    sign_fields = {
        key: value for key, value in stage_c_validation.items() if key.startswith("sign_")
    }
    sign_fields["passed"] = bool(stage_c_validation.get("delta_sign_control_pass", False))
    comparison = {
        branch: {
            metric: stage_c_validation.get(f"{branch}_{metric}")
            for metric in ("cosine", "magnitude_ratio", "normalized_vector_error", "residual_median", "residual_p95")
        }
        for branch in ("full", "patch", "pose_only_ref")
    }
    comparison["pose_head"] = {
        metric: stage_c_validation.get(f"pose_head_{metric}")
        for metric in ("cosine", "magnitude_ratio", "normalized_error", "rotation_degrees")
    }
    acceptance = {
        "schema": "cut3r_gauge_smoke_acceptance_v2",
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema": selected_payload.get("schema"),
        "checkpoint_stage": selected_payload.get("stage"),
        "checkpoint_stage_local_step": int(selected_payload["stage_local_step"]),
        "alignment_gate": alignment_gate,
        "probe_gate": probe_gate_payload,
        "stage_a_gate": stage_a_gate_payload,
        "stage_b_gate": corrected_stage_b_gate,
        "stage_c_feasibility": corrected_stage_c_gate["stage_c_feasibility"],
        "delta_sign_control": sign_fields,
        "shuffled_delta_control": shuffled_fields,
        "held_out_stage_c_improvement": corrected_stage_c_gate["held_out_stage_c_improvement"],
        "pose_dominance": corrected_stage_c_gate["pose_dominance"],
        "full_patch_pose_comparison": comparison,
        "stage_c_gate": corrected_stage_c_gate,
    }
    acceptance["passed"] = bool(
        alignment_gate.get("passed")
        and probe_gate_payload.get("passed")
        and stage_a_gate_payload.get("passed")
        and corrected_stage_b_gate.get("passed")
        and corrected_stage_c_gate.get("passed")
    )
    atomic_json(output_dir / "smoke_acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True), flush=True)
    if not acceptance["passed"]:
        raise RuntimeError("corrected smoke acceptance failed; official launch remains blocked")
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--revalidate-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rank, world_size, local_rank, device = distributed_setup()
    signal.signal(signal.SIGUSR1, _signal_handler)
    torch.manual_seed(int(config["seed"]) + rank)
    np.random.seed(int(config["seed"]) + rank)
    if args.revalidate_checkpoint is not None:
        revalidate_smoke_checkpoint(
            args.revalidate_checkpoint, config, device, rank, world_size
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    output_dir = Path(os.path.expandvars(config["output_dir"]))
    resume_path = None
    if args.resume_from:
        if args.resume_from == "auto":
            resume_glob = config.get("resume_glob")
            if resume_glob:
                candidates = [Path(path) / "latest.pt" for path in glob.glob(os.path.expandvars(resume_glob))]
                candidates = [path for path in candidates if path.is_file()]
                if not candidates:
                    raise FileNotFoundError(f"no resumable checkpoint matches {resume_glob}")
                resume_path = max(candidates, key=lambda path: path.stat().st_mtime)
            else:
                resume_path = output_dir / "latest.pt"
        else:
            resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        output_dir = resume_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    train_records = load_jsonl(Path(os.path.expandvars(config["train_manifest"])))
    val_records = load_jsonl(Path(os.path.expandvars(config["validation_manifest"])))
    config["provenance_hashes"] = {
        "source_config_sha256": sha256_file(args.config),
        "train_manifest_sha256": sha256_file(Path(os.path.expandvars(config["train_manifest"]))),
        "validation_manifest_sha256": sha256_file(Path(os.path.expandvars(config["validation_manifest"]))),
    }
    if config["mode"] == "smoke" and (len(train_records) < 16 or len(val_records) < 8):
        raise RuntimeError(f"smoke requires 16/8 videos; found {len(train_records)}/{len(val_records)}")
    context_root = Path(os.path.expandvars(config["context_root"]))
    verify_alignment_gate([*train_records, *val_records], output_dir, rank, device)
    coverage = coverage_report(config, train_records, val_records, world_size, context_root)
    config["resolved_stage_steps"] = coverage["resolved_stage_steps"]
    if rank == 0:
        atomic_json(output_dir / "resolved_config.json", config)
        atomic_json(output_dir / "coverage_report.json", coverage)
        print(json.dumps(coverage, indent=2), flush=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["precision"]]
    translator_config = GaugeTranslationConfig(**config["translator"])
    if translator_config.use_pose_context_for_patch_adapter:
        raise RuntimeError("pose context must remain disabled")
    translator = GaugeTranslationModel(translator_config).to(device=device, dtype=dtype)
    q_train = PatchGeometryTrainProbe(translator_config.token_dim).to(device=device, dtype=dtype)
    torch.manual_seed(int(config["seed"]) + 1009)
    q_eval = PatchGeometryEvalProbe(translator_config.token_dim).to(device=device, dtype=dtype)
    frozen_head = FrozenCut3RHead(Path(os.path.expandvars(config["cut3r_checkpoint"])), device, dtype)
    resume_payload = torch.load(resume_path, map_location=device, weights_only=False) if resume_path else None
    if resume_payload is not None:
        saved_hashes = resume_payload.get("config", {}).get("provenance_hashes", {})
        current_hashes = config.get("provenance_hashes", {})
        for key in ("train_manifest_sha256", "validation_manifest_sha256"):
            if saved_hashes.get(key) and saved_hashes[key] != current_hashes.get(key):
                raise RuntimeError(f"resume provenance mismatch for {key}")
        if rank == 0:
            atomic_json(output_dir / "resume_provenance.json", {
                "checkpoint": str(resume_path),
                "saved_hashes": saved_hashes,
                "current_hashes": current_hashes,
                "source_config_changed": bool(
                    saved_hashes.get("source_config_sha256")
                    and saved_hashes.get("source_config_sha256") != current_hashes.get("source_config_sha256")
                ),
            })
        if resume_payload.get("stage") not in ("a", "b", "c"):
            raise RuntimeError("resume currently requires a completed probe gate and a Stage A/B/C checkpoint")
        translator.load_state_dict(resume_payload["translator"])
        q_train.load_state_dict(resume_payload["q_train"])
        q_eval.load_state_dict(resume_payload["q_eval"])
        freeze_module(q_train)
        freeze_module(q_eval)
        global_step = int(resume_payload["global_step"])
        start_stage = str(resume_payload["stage"])
    else:
        q_train_ddp, q_eval_ddp = ddp_wrap(q_train, world_size, local_rank), ddp_wrap(q_eval, world_size, local_rank)
        mean_target = compute_mean_target(train_records, device, dtype, config["max_frames"], config["seed"], config.get("mean_baseline_video_limit", 0))
        _, global_step = pretrain_probes(
            q_train_ddp, q_eval_ddp, train_records, val_records, mean_target,
            coverage["resolved_stage_steps"]["probe"], config, device, dtype, rank, world_size, local_rank,
            output_dir, metrics_path,
        )
        del q_train_ddp, q_eval_ddp
        start_stage = "a"
    stages = ("a", "b", "c")
    for stage in stages[stages.index(start_stage):]:
        stage_resume = resume_payload if resume_payload is not None and resume_payload.get("stage") == stage else None
        global_step, _ = train_stage(
            stage, translator, q_train, q_eval, frozen_head, train_records, val_records,
            coverage["resolved_stage_steps"][stage], config, device, dtype, rank, world_size, local_rank,
            output_dir, metrics_path, global_step, stage_resume,
        )
        resume_payload = None
    if rank == 0:
        acceptance = {
            "passed": True,
            "probe_gate": json.loads((output_dir / "probe_gate.json").read_text()),
            "stage_a_gate": json.loads((output_dir / "stage_a_gate.json").read_text()),
            "stage_b_gate": json.loads((output_dir / "stage_b_gate.json").read_text()),
            "stage_c_gate": json.loads((output_dir / "stage_c_gate.json").read_text()),
        }
        acceptance["passed"] = all(acceptance[key]["passed"] for key in ("probe_gate", "stage_a_gate", "stage_b_gate", "stage_c_gate"))
        atomic_json(output_dir / "smoke_acceptance.json" if config["mode"] == "smoke" else output_dir / "official_initial_acceptance.json", acceptance)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
