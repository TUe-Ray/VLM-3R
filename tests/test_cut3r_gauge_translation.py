import importlib.util
from pathlib import Path

import pytest
import torch

from scripts.gauge_translation.standalone_model import (
    GaugeTranslationConfig,
    GaugeTranslationModel,
    PatchGeometryEvalProbe,
    PatchGeometryTrainProbe,
    build_teacher_mask,
    checkpoint_feasibility,
    expected_patch_positions,
    heldout_error_improvement,
    normalized_smooth_l1,
    orthogonal_control_delta,
    pose_dominance_detected,
    pool_points_adaptive,
    pool_points_by_positions,
    quaternion_geodesic_degrees,
    quaternion_rotation_loss,
    robust_scene_scale,
    sample_video_translations,
    shuffled_delta_branch_control,
    stage_c_validation_gate,
    validate_patch_positions,
)
from scripts.gauge_translation.common import (
    capture_rng_state,
    probe_gate,
    resolve_stage_steps,
    restore_rng_state,
    stage_a_gate,
    stage_b_gate,
)
from scripts.gauge_translation.train_cut3r_gauge_translation import (
    _pose_shift_metrics,
    _relative_change_summary,
)


def make_model(token_dim=8):
    return GaugeTranslationModel(
        GaugeTranslationConfig(
            enabled=True,
            token_dim=token_dim,
            trunk_hidden_dim=16,
            trunk_output_dim=16,
            patch_condition_dim=16,
            pose_condition_dim=64,
            patch_bottleneck_dim=4,
            pose_bottleneck_dim=4,
            use_pose_context_for_patch_adapter=False,
        )
    )


def make_inputs(token_dim=8):
    patches = tuple(torch.randn(2, 3, 729, token_dim) for _ in range(3))
    pose = torch.randn(2, 3, 1, token_dim)
    delta = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]])
    scale = torch.tensor([2.0, 3.0])
    return patches, pose, delta, scale


def test_identity_initialization_shapes_and_pose_independence():
    model = make_model()
    patches, pose, delta, scale = make_inputs()
    output = model(*patches, pose, delta, scale)
    for source, transformed in zip(patches, output.patches()):
        assert transformed.shape == source.shape
        assert torch.equal(transformed, source)
    assert output.pose12.shape == pose.shape
    assert torch.equal(output.pose12, pose)
    changed_pose_output = model(*patches, pose + 100, delta, scale)
    for first, second in zip(output.patches(), changed_pose_output.patches()):
        assert torch.equal(first, second)


def test_pose_context_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        GaugeTranslationConfig(use_pose_context_for_patch_adapter=True)


def test_stage_trainable_sets_are_separate():
    model = make_model()
    stage_a = model.set_trainable_stage("a")
    assert any(name.startswith("translation_trunk") for name in stage_a)
    assert any(name.startswith("patch_conditioning_projection") for name in stage_a)
    assert not any(name.startswith("pose_conditioning_projection") for name in stage_a)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    stage_b = model.set_trainable_stage("b")
    assert all(name.startswith(("pose_conditioning_projection", "pose_adapter")) for name in stage_b)
    assert all(parameter.grad is None for parameter in model.parameters())
    stage_c = model.set_trainable_stage("c")
    assert len(stage_c) == len(list(model.named_parameters()))


def test_translation_is_per_video_and_broadcast_over_frames():
    model = make_model()
    with torch.no_grad():
        model.patch_adapters["6"].up.weight.fill_(0.01)
    patches, pose, delta, scale = make_inputs()
    output = model(*patches, pose, delta, scale)
    assert not torch.equal(output.patch6[0], patches[0][0])
    assert not torch.equal(output.patch6[1], patches[0][1])
    conditioning = model.conditioning_input(delta, scale)
    assert conditioning.shape == (2, 4)
    assert torch.allclose(conditioning[:, :3], delta / scale[:, None])


def test_probe_architectures_are_independent_and_shape_preserving():
    patches = tuple(torch.randn(1, 2, 729, 8) for _ in range(3))
    q_train = PatchGeometryTrainProbe(8, 4)
    q_eval = PatchGeometryEvalProbe(8)
    assert q_train(*patches).shape == (1, 2, 729, 3)
    assert q_eval(*patches).shape == (1, 2, 729, 3)
    assert type(q_train.head).__name__ != type(q_eval.head).__name__


def test_exact_position_convention_and_transform_detection():
    positions = expected_patch_positions()
    report = validate_patch_positions(positions)
    assert report["passed"]
    assert report["corners"] == {0: [0, 0], 26: [0, 26], 702: [26, 0], 728: [26, 26]}
    flipped = positions.clone()
    flipped[:, 1] = 26 - flipped[:, 1]
    report = validate_patch_positions(flipped)
    assert not report["passed"]
    assert report["alternative_transform_matches"]["horizontal_flip"]


def test_explicit_pooling_matches_adaptive_for_432_grid_and_masks():
    height = width = 432
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack([yy, xx, yy * width + xx], dim=-1).float().unsqueeze(0)
    mask = torch.ones(1, height, width, dtype=torch.bool)
    mask[:, 0:7, 0:5] = False
    positions = expected_patch_positions().unsqueeze(0)
    explicit, explicit_mask = pool_points_by_positions(points, mask, positions)
    adaptive, adaptive_mask = pool_points_adaptive(points, mask)
    assert torch.equal(explicit_mask, adaptive_mask)
    assert torch.allclose(explicit, adaptive, atol=1e-5, rtol=3e-6)


def test_teacher_mask_is_fixed_from_original_teacher():
    reference = torch.ones(1, 2, 2, 3)
    camera = torch.ones_like(reference)
    camera[0, 0, 0, 2] = -1
    reference[0, 1, 1] = torch.nan
    confidence = torch.tensor([[[2.0, 0.5], [2.0, 2.0]]])
    mask = build_teacher_mask(reference, camera, confidence, confidence, 1.0, 1.0)
    assert mask.tolist() == [[[False, False], [True, False]]]
    transformed = torch.full_like(reference, torch.nan)
    with pytest.raises(FloatingPointError):
        normalized_smooth_l1(transformed, reference, torch.tensor([1.0]), mask)


def test_scene_scale_is_fp32_positive_and_robust():
    points = torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]]]).half()
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    scale = robust_scene_scale(points, mask)
    assert scale.dtype == torch.float32
    assert float(scale) == pytest.approx(2.0)


def test_quaternion_loss_is_sign_invariant_and_fp32():
    first = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
    second = -first
    assert quaternion_rotation_loss(first, second).dtype == torch.float32
    assert float(quaternion_rotation_loss(first, second)) == pytest.approx(0.0)
    assert float(quaternion_geodesic_degrees(first, second)) < 0.1


def test_translation_sampling_shared_shape_zero_and_axis_controls():
    scale = torch.tensor([2.0, 3.0])
    zero = sample_video_translations(2, scale, 0.0, zero_probability=1.0)
    assert torch.equal(zero, torch.zeros_like(zero))
    generator = torch.Generator().manual_seed(7)
    axis = sample_video_translations(2, scale, 1.0, zero_probability=0.0, axis_probability=1.0, generator=generator)
    assert axis.shape == (2, 3)
    assert torch.all((axis != 0).sum(dim=-1) == 1)


def test_checkpoint_feasibility_rejects_pose_dominance_and_invalid_outputs():
    metrics = {
        "full_cosine": 0.9,
        "full_magnitude_ratio": 1.0,
        "q_eval_cosine": 0.7,
        "q_eval_magnitude_ratio": 0.2,
        "normalized_self_drift": 0.1,
        "pose_head_rotation_degrees": 1.0,
        "patch_residual_p95": 0.1,
        "pose_token_residual_p95": 0.1,
        "invalid_output_ratio": 0.0,
        "new_nonpositive_self_depth_ratio": 0.0,
        "confidence_ref_relative_drop": 0.0,
        "confidence_self_relative_drop": 0.0,
        "pose_dominated": False,
        "structural_finite": True,
    }
    assert checkpoint_feasibility(metrics)[0]
    metrics["pose_dominated"] = True
    assert "pose_dominance" in checkpoint_feasibility(metrics)[1]
    metrics["pose_dominated"] = False
    metrics["invalid_output_ratio"] = 0.01
    assert "invalid_output" in checkpoint_feasibility(metrics)[1]


def test_scene_grouping_keeps_scannet_recordings_together():
    path = Path(__file__).resolve().parents[1] / "scripts" / "gauge_translation" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("gauge_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.scene_group("scannet", "scene0303_00") == module.scene_group("scannet", "scene0303_01")
    records = [
        {"id": "a", "scene_group": "scannet:scene0001"},
        {"id": "b", "scene_group": "scannet:scene0001"},
        {"id": "c", "scene_group": "scannet:scene0002"},
    ]
    train, validation = module.split_records(records, seed=42, validation_fraction=0.5)
    assert {x["scene_group"] for x in train}.isdisjoint({x["scene_group"] for x in validation})


def test_schedule_resolution_is_exclusive_and_reports_effective_epochs():
    steps, epochs, steps_per_epoch = resolve_stage_steps(
        {"steps": 25, "epochs": None}, dataset_size=10, global_batch_size=2
    )
    assert (steps, epochs, steps_per_epoch) == (25, 5.0, 5)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_stage_steps({"steps": 25, "epochs": 2}, 10, 2)


def test_rng_checkpoint_state_round_trip():
    torch.manual_seed(77)
    state = capture_rng_state()
    expected = torch.rand(4)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)


def test_stage_gates_enforce_probe_patch_and_pose_requirements():
    initial = {"q_train_normalized_error": 2.0, "q_eval_normalized_error": 2.0}
    final = {
        "mean_predictor_normalized_error": 1.0,
        "q_train_normalized_error": 0.7, "q_eval_normalized_error": 0.7,
        "q_train_variance_ratio": 0.02, "q_eval_variance_ratio": 0.02,
        "q_train_finite": True, "q_eval_finite": True,
    }
    assert probe_gate(initial, final)["passed"]
    a_metrics = {
        "q_eval_cosine": 0.6, "q_eval_magnitude_ratio": 0.2,
        "patch_gradient_nonzero_fraction": 0.95, "patch_change_median": 1e-3,
        "patch_residual_p95": 0.1, "normalized_self_drift": 0.1,
        "invalid_output_ratio": 0.0, "new_nonpositive_self_depth_ratio": 0.0,
    }
    a_history = {"patch_eq": [1.0] * 10 + [0.5] * 10}
    assert stage_a_gate(a_metrics, a_history)["passed"]
    a_metrics["q_eval_magnitude_ratio"] = 0.05
    assert "q_eval_magnitude" in stage_a_gate(a_metrics, a_history)["failures"]
    b_metrics = {
        "pose_head_cosine": 0.6, "pose_head_magnitude_ratio": 0.2,
        "pose_head_rotation_degrees": 1.0, "pose_change": 0.01,
        "pose_gradient_nonzero_fraction": 0.95, "pose_gradients_finite": True,
        "quaternion_sign_invariant": True, "pose_losses_finite": True,
    }
    b_history = {"pose_t": [1.0] * 10 + [0.5] * 10}
    assert stage_b_gate(b_metrics, b_history)["passed"]


def test_orthogonal_control_handles_nearly_parallel_source_deltas_deterministically():
    sources = torch.tensor([[1.0, 1e-4, 0.0], [1.0, 1.1e-4, 0.0]])
    assert float(torch.nn.functional.cosine_similarity(sources[:1], sources[1:], dim=-1)) > 0.999
    assigned_first = orthogonal_control_delta(sources)
    assigned_second = orthogonal_control_delta(sources)
    assert torch.equal(assigned_first, assigned_second)
    source_assigned = torch.nn.functional.cosine_similarity(sources, assigned_first, dim=-1)
    assert torch.all(source_assigned.abs() < 1e-5)
    assert torch.allclose(assigned_first.norm(dim=-1), sources.norm(dim=-1))


def test_shuffled_control_passes_assigned_follower_and_rejects_source_follower():
    source = torch.tensor([[1.0, 0.0, 0.0]])
    assigned = orthogonal_control_delta(source)
    follows_assigned = shuffled_delta_branch_control(assigned[0], source, assigned, 0.1)
    ignores_assigned = shuffled_delta_branch_control(source[0], source, assigned, 0.1)
    assert follows_assigned["passed"]
    assert follows_assigned["assigned_minus_source_margin"] >= 0.2
    assert not ignores_assigned["passed"]


def _feasible_stage_c_metrics():
    return {
        "full_cosine": 0.99, "full_magnitude_ratio": 0.51,
        "full_normalized_vector_error": 0.1336,
        "q_eval_cosine": 0.97, "q_eval_magnitude_ratio": 0.99,
        "q_eval_normalized_vector_error": 0.08,
        "normalized_self_drift": 0.02, "pose_head_rotation_degrees": 1.3,
        "patch_residual_p95": 0.13, "pose_token_residual_p95": 0.07,
        "invalid_output_ratio": 0.0, "new_nonpositive_self_depth_ratio": 0.0,
        "confidence_ref_relative_drop": 0.01, "confidence_self_relative_drop": 0.01,
        "pose_dominated": False, "structural_finite": True,
        "delta_sign_control_pass": True, "shuffled_delta_control_pass": True,
    }


def test_stage_c_heldout_improvement_uses_validation_not_noisy_training_loss():
    improvement = heldout_error_improvement(0.1775, 0.1336)
    assert improvement["passed"]
    assert improvement["improvement_fraction"] == pytest.approx(1.0 - 0.1336 / 0.1775)
    metrics = _feasible_stage_c_metrics()
    metrics["training_full_ref_early"] = 0.0362
    metrics["training_full_ref_final"] = 0.0370
    gate = stage_c_validation_gate(metrics, 0.1775)
    assert gate["passed"]
    assert gate["held_out_stage_c_improvement"]["passed"]


def test_pose_metric_names_and_gate_consumers_are_separate():
    original = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    moved = original.clone()
    moved[..., 0] = 1.0
    pose_head = _pose_shift_metrics(original, moved, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([1.0]))
    assert "pose_head_cosine" in pose_head
    assert "pose_head_rotation_degrees" in pose_head
    assert "pose_only_ref_cosine" not in pose_head
    b_metrics = {
        "pose_head_cosine": 0.8, "pose_head_magnitude_ratio": 0.4,
        "pose_head_rotation_degrees": 1.0, "pose_change": 0.01,
        "pose_gradient_nonzero_fraction": 1.0, "pose_gradients_finite": True,
        "quaternion_sign_invariant": True, "pose_losses_finite": True,
        "pose_only_ref_cosine": -1.0, "pose_only_ref_magnitude_ratio": 2.0,
    }
    assert stage_b_gate(b_metrics, {"pose_t": [1.0] * 10 + [0.5] * 10})["passed"]
    assert pose_dominance_detected({
        "pose_only_ref_magnitude_ratio": 1.0, "q_eval_magnitude_ratio": 0.05,
        "pose_head_magnitude_ratio": 0.0,
    })
    assert not pose_dominance_detected({
        "pose_only_ref_magnitude_ratio": 0.2, "q_eval_magnitude_ratio": 0.05,
        "pose_head_magnitude_ratio": 1.2,
    })


def test_relative_change_summary_uses_population_p95_not_median_of_video_p95s():
    videos = [torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.tensor([0.0] * 100)]
    summary = _relative_change_summary(videos)
    population = torch.cat(videos)
    assert summary["median"] == pytest.approx(float(population.median()))
    assert summary["p95"] == pytest.approx(float(torch.quantile(population, 0.95)))
    per_video_p95_median = torch.tensor(
        [torch.quantile(video, 0.95) for video in videos]
    ).median()
    assert summary["p95"] != pytest.approx(float(per_video_p95_median))


def test_existing_translator_state_dict_remains_strictly_compatible():
    original = make_model()
    restored = make_model()
    restored.load_state_dict(original.state_dict(), strict=True)
