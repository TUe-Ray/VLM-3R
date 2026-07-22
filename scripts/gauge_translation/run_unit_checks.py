#!/usr/bin/env python3
"""Dependency-free unit and shape checks for gauge translation."""

from __future__ import annotations

import json

import torch

from llava.model.cut3r_gauge_translation import (
    GaugeTranslationConfig,
    GaugeTranslationModel,
    PatchGeometryEvalProbe,
    PatchGeometryTrainProbe,
    build_teacher_mask,
    checkpoint_feasibility,
    expected_patch_positions,
    normalized_smooth_l1,
    pool_points_adaptive,
    pool_points_by_positions,
    quaternion_rotation_loss,
    robust_scene_scale,
    sample_video_translations,
    validate_patch_positions,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    results = {}
    config = GaugeTranslationConfig(
        enabled=True,
        token_dim=8,
        trunk_hidden_dim=16,
        trunk_output_dim=16,
        patch_condition_dim=16,
        pose_condition_dim=64,
        patch_bottleneck_dim=4,
        pose_bottleneck_dim=4,
        use_pose_context_for_patch_adapter=False,
    )
    model = GaugeTranslationModel(config)
    patches = tuple(torch.randn(2, 2, 729, 8) for _ in range(3))
    pose = torch.randn(2, 2, 1, 8)
    delta = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]])
    scale = torch.tensor([2.0, 3.0])
    output = model(*patches, pose, delta, scale)
    require(all(torch.equal(a, b) for a, b in zip(patches, output.patches())), "zero-init patch identity failed")
    require(torch.equal(pose, output.pose12), "zero-init pose identity failed")
    changed_pose = model(*patches, pose + 100, delta, scale)
    require(all(torch.equal(a, b) for a, b in zip(output.patches(), changed_pose.patches())), "patches depend on pose")
    results["translator_identity_and_pose_independence"] = True

    stage_a = model.set_trainable_stage("a")
    stage_b = model.set_trainable_stage("b")
    stage_c = model.set_trainable_stage("c")
    require(any(name.startswith("patch_adapters") for name in stage_a), "Stage A lacks patches")
    require(all(name.startswith(("pose_adapter", "pose_conditioning_projection")) for name in stage_b), "Stage B freeze set wrong")
    require(len(stage_c) == len(list(model.named_parameters())), "Stage C did not unfreeze all")
    results["stage_freezing"] = True

    q_train, q_eval = PatchGeometryTrainProbe(8, 4), PatchGeometryEvalProbe(8)
    require(q_train(*patches).shape == (2, 2, 729, 3), "Q_train shape failed")
    require(q_eval(*patches).shape == (2, 2, 729, 3), "Q_eval shape failed")
    results["probe_shapes"] = True

    positions = expected_patch_positions().unsqueeze(0)
    report = validate_patch_positions(positions)
    require(bool(report["passed"]), "row-major position gate failed")
    require(report["corners"] == {0: [0, 0], 26: [0, 26], 702: [26, 0], 728: [26, 26]}, "corner mapping failed")
    height = width = 432
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    point_map = torch.stack([yy, xx, yy * width + xx], dim=-1).float().unsqueeze(0)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    mask[:, :7, :5] = False
    explicit, explicit_mask = pool_points_by_positions(point_map, mask, positions)
    adaptive, adaptive_mask = pool_points_adaptive(point_map, mask)
    require(torch.equal(explicit_mask, adaptive_mask), "pool masks differ")
    require(torch.allclose(explicit, adaptive, atol=1e-5, rtol=3e-6), "position/adaptive pooling differs")
    results["alignment"] = report

    reference = torch.ones(1, 2, 2, 3)
    camera = torch.ones_like(reference)
    camera[0, 0, 0, 2] = -1
    confidence = torch.tensor([[[2.0, 0.5], [2.0, 2.0]]])
    teacher_mask = build_teacher_mask(reference, camera, confidence, confidence, 1.0, 1.0)
    require(teacher_mask.tolist() == [[[False, False], [True, True]]], "teacher mask definition failed")
    try:
        normalized_smooth_l1(torch.full_like(reference, torch.nan), reference, torch.tensor([1.0]), teacher_mask)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("invalid transformed teacher points did not fail")
    require(robust_scene_scale(point_map, mask).dtype == torch.float32, "scene scale not FP32")
    results["fixed_teacher_mask_and_fp32"] = True

    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
    require(float(quaternion_rotation_loss(quaternion, -quaternion)) == 0.0, "quaternion sign invariance failed")
    sampled = sample_video_translations(2, scale, 1.0, zero_probability=0.0, axis_probability=1.0, generator=torch.Generator().manual_seed(7))
    require(bool(torch.all((sampled != 0).sum(dim=-1) == 1)), "axis sampling failed")
    results["quaternion_and_delta_controls"] = True

    feasible = {
        "full_cosine": 0.9, "full_magnitude_ratio": 1.0,
        "q_eval_cosine": 0.7, "q_eval_magnitude_ratio": 0.2,
        "normalized_self_drift": 0.1, "pose_rotation_degrees": 1.0,
        "patch_residual_p95": 0.1, "pose_residual_p95": 0.1, "invalid_output_ratio": 0.0,
        "new_nonpositive_self_depth_ratio": 0.0, "confidence_ref_relative_drop": 0.0, "confidence_self_relative_drop": 0.0, "pose_dominated": False,
        "structural_finite": True,
    }
    require(checkpoint_feasibility(feasible)[0], "valid checkpoint rejected")
    feasible["pose_dominated"] = True
    require(not checkpoint_feasibility(feasible)[0], "pose-dominated checkpoint accepted")
    results["checkpoint_feasibility"] = True
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
