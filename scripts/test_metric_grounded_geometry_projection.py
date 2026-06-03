import math
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = REPO_ROOT / "llava" / "model" / "geometry"


def load_geometry_module(module_name):
    for package_name, package_path in (
        ("llava", REPO_ROOT / "llava"),
        ("llava.model", REPO_ROOT / "llava" / "model"),
        ("llava.model.geometry", GEOMETRY_ROOT),
    ):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            sys.modules[package_name] = package
    full_name = f"llava.model.geometry.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, GEOMETRY_ROOT / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


geometry_rope = load_geometry_module("geometry_rope")
geometry_provider_adapter = load_geometry_module("geometry_provider_adapter")
auxiliary_geometry_head = load_geometry_module("auxiliary_geometry_head")
geometry_aware_projection = load_geometry_module("geometry_aware_projection")

AuxiliaryGeometryHead = auxiliary_geometry_head.AuxiliaryGeometryHead
GeometryProviderAdapter = geometry_provider_adapter.GeometryProviderAdapter
GeometryRoPE = geometry_rope.GeometryRoPE
MetricGroundedGeometryProjection = geometry_aware_projection.MetricGroundedGeometryProjection
canonicalize_geometry_outputs = geometry_provider_adapter.canonicalize_geometry_outputs


def make_point_map(batch, height, width):
    y = torch.linspace(-1.0, 1.0, height).view(1, height, 1).expand(batch, height, width)
    x = torch.linspace(-1.0, 1.0, width).view(1, 1, width).expand(batch, height, width)
    z = torch.ones(batch, height, width) * 2.0
    return torch.stack((x, y, z), dim=-1)


def make_config(dtype=torch.float32):
    del dtype
    return SimpleNamespace(
        mm_hidden_size=1024,
        hidden_size=3584,
        geometry_projection_num_heads=16,
        geometry_position_mode="spherical",
        num_geometry_projection_layers=1,
        geometry_gate_init=0.0,
        use_auxiliary_geometry_head=True,
        aux_geometry_targets="azimuth,elevation,log_distance",
        lambda_geo=0.1,
        geometry_loss_type="smooth_l1",
        detach_geometry_targets=True,
        use_geometry_confidence_mask=True,
        geometry_projection_dropout=0.0,
        allow_missing_geometry_targets=False,
        use_auxiliary_geometry_loss=True,
    )


def test_canonicalize_aliases():
    point_map = make_point_map(2, 4, 4)
    depth = point_map[..., 2]
    confidence = torch.ones(2, 4, 4)
    out = canonicalize_geometry_outputs({
        "pts3d_in_other_view": point_map,
        "depth_map": depth,
        "pts3d_conf": confidence,
    })
    assert out["point_map"] is point_map
    assert out["depth"] is depth
    assert out["confidence"] is confidence


def test_adapter_shape_with_explicit_grid():
    visual_tokens = torch.randn(2, 128, 1024)
    point_map = make_point_map(2, 16, 16)
    adapter = GeometryProviderAdapter(mode="spherical")
    geometry_pos, targets, mask = adapter(
        {"point_map": point_map},
        visual_tokens=visual_tokens,
        visual_grid_size=(8, 16),
        num_frames=2,
    )
    assert geometry_pos.shape == (2, 128, 3)
    assert targets["azimuth"].shape == (2, 128, 1)
    assert mask.shape == (2, 128)
    assert mask.all()


def test_geometry_rope_shape_dtype_and_no_nan():
    q = torch.randn(2, 16, 128, 64)
    k = torch.randn(2, 16, 128, 64)
    geometry_pos = torch.randn(2, 128, 3)
    rope = GeometryRoPE(head_dim=64, mode="spherical")
    q_rot, k_rot = rope(q, k, geometry_pos)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert q_rot.dtype == q.dtype
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()


def test_zero_gate_identity_and_finite_loss():
    torch.manual_seed(7)
    cfg = make_config()
    module = MetricGroundedGeometryProjection(cfg)
    visual_tokens = torch.randn(2, 128, 1024)
    point_map = make_point_map(2, 16, 16)
    out = module(
        visual_tokens=visual_tokens,
        geometry_outputs={"point_map": point_map},
        visual_grid_size=(8, 16),
        num_frames=2,
    )
    assert out["refined_tokens"].shape == visual_tokens.shape
    assert torch.allclose(out["refined_tokens"], visual_tokens, atol=0.0, rtol=0.0)
    assert out["loss_geo"] is not None
    assert torch.isfinite(out["loss_geo"])


def test_masked_loss_and_azimuth_wraparound():
    head = AuxiliaryGeometryHead(4, target_names=["azimuth"], loss_type="mse")
    pred = {"azimuth": torch.tensor([[[math.pi - 0.01], [0.0]]])}
    target = {"azimuth": torch.tensor([[[-math.pi + 0.01], [100.0]]])}
    mask = torch.tensor([[True, False]])
    loss = head.compute_loss(pred, target, mask)
    assert loss < 0.001
    zero_loss = head.compute_loss(pred, target, torch.zeros_like(mask))
    assert zero_loss.item() == 0.0


def test_bf16_projection_if_supported():
    cfg = make_config()
    module = MetricGroundedGeometryProjection(cfg).to(dtype=torch.bfloat16)
    visual_tokens = torch.randn(1, 128, 1024, dtype=torch.bfloat16)
    point_map = make_point_map(1, 16, 16)
    out = module(
        visual_tokens=visual_tokens,
        geometry_outputs={"point_map": point_map},
        visual_grid_size=(8, 16),
        num_frames=1,
    )
    assert out["refined_tokens"].dtype == torch.bfloat16


def test_projection_uses_mm_hidden_size_not_llm_hidden_size():
    cfg = SimpleNamespace(
        hidden_size=3584,
        mm_hidden_size=1024,
        geometry_projection_num_heads=16,
        geometry_position_mode="spherical",
        num_geometry_projection_layers=1,
        geometry_gate_init=0.0,
        use_auxiliary_geometry_head=True,
        use_auxiliary_geometry_loss=True,
        aux_geometry_targets="azimuth,elevation,log_distance",
        lambda_geo=0.1,
        geometry_loss_type="smooth_l1",
        detach_geometry_targets=True,
        use_geometry_confidence_mask=True,
        geometry_projection_dropout=0.0,
    )
    module = MetricGroundedGeometryProjection(cfg)
    assert module.hidden_size == 1024
    assert module.layers[0].hidden_size == 1024


def test_invalid_geometry_zeroed_and_no_nan():
    cfg = make_config()
    module = MetricGroundedGeometryProjection(cfg)
    visual_tokens = torch.randn(1, 128, 1024)
    point_map = make_point_map(1, 16, 16)
    confidence = torch.ones(1, 16, 16)
    confidence[:, :8, :] = 0
    out = module(
        visual_tokens=visual_tokens,
        geometry_outputs={"point_map": point_map, "confidence": confidence},
        visual_grid_size=(8, 16),
        num_frames=1,
    )
    assert torch.isfinite(out["refined_tokens"]).all()
    invalid = ~out["geometry_mask"]
    assert invalid.any()
    assert torch.all(out["geometry_pos"][invalid] == 0)


def test_auxiliary_geometry_loss_can_be_disabled():
    cfg = make_config()
    cfg.use_auxiliary_geometry_loss = False
    module = MetricGroundedGeometryProjection(cfg)
    visual_tokens = torch.randn(1, 128, 1024)
    point_map = make_point_map(1, 16, 16)
    out = module(
        visual_tokens=visual_tokens,
        geometry_outputs={"point_map": point_map},
        visual_grid_size=(8, 16),
        num_frames=1,
    )
    assert out["geometry_predictions"]
    assert out["loss_geo"] is None


def test_all_invalid_geometry_does_not_nan():
    cfg = make_config()
    module = MetricGroundedGeometryProjection(cfg)
    visual_tokens = torch.randn(1, 128, 1024)
    point_map = make_point_map(1, 16, 16)
    confidence = torch.zeros(1, 16, 16)
    out = module(
        visual_tokens=visual_tokens,
        geometry_outputs={"point_map": point_map, "confidence": confidence},
        visual_grid_size=(8, 16),
        num_frames=1,
    )
    assert torch.isfinite(out["refined_tokens"]).all()
    assert out["geometry_mask"].sum() == 0


def test_depth_mode_uses_log_depth_not_log_radius():
    point_map = torch.zeros(1, 1, 1, 3)
    point_map[..., 0] = 3.0
    point_map[..., 1] = 0.0
    point_map[..., 2] = 4.0
    visual_tokens = torch.randn(1, 1, 8)
    adapter = GeometryProviderAdapter(mode="depth")
    geometry_pos, targets, mask = adapter(
        {"point_map": point_map},
        visual_tokens=visual_tokens,
        visual_grid_size=(1, 1),
    )
    assert mask.all()
    assert torch.allclose(targets["log_depth"], torch.log1p(torch.tensor([[[4.0]]])))
    assert torch.allclose(targets["log_distance"], torch.log1p(torch.tensor([[[5.0]]])))
    assert torch.allclose(geometry_pos, targets["log_depth"])


def test_missing_aux_targets_raise_by_default():
    head = AuxiliaryGeometryHead(
        hidden_size=4,
        target_names=["azimuth", "log_distance"],
        loss_type="smooth_l1",
        allow_missing_targets=False,
    )
    hidden = torch.randn(1, 2, 4)
    preds = head(hidden)
    targets = {"log_distance": torch.randn(1, 2, 1)}
    mask = torch.ones(1, 2, dtype=torch.bool)
    try:
        head.compute_loss(preds, targets, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing auxiliary geometry targets should raise by default.")


def test_missing_aux_targets_can_be_allowed():
    head = AuxiliaryGeometryHead(
        hidden_size=4,
        target_names=["azimuth", "log_distance"],
        loss_type="smooth_l1",
        allow_missing_targets=True,
    )
    hidden = torch.randn(1, 2, 4)
    preds = head(hidden)
    targets = {"log_distance": torch.randn(1, 2, 1)}
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = head.compute_loss(preds, targets, mask)
    assert torch.isfinite(loss)


def test_azimuth_wraparound_for_all_loss_types():
    for loss_type in ["smooth_l1", "mse", "l1"]:
        head = AuxiliaryGeometryHead(4, target_names=["azimuth"], loss_type=loss_type)
        pred = {"azimuth": torch.tensor([[[math.pi - 0.01]]])}
        target = {"azimuth": torch.tensor([[[-math.pi + 0.01]]])}
        mask = torch.tensor([[True]])
        loss = head.compute_loss(pred, target, mask)
        assert loss < 0.05


def main():
    test_canonicalize_aliases()
    test_adapter_shape_with_explicit_grid()
    test_geometry_rope_shape_dtype_and_no_nan()
    test_zero_gate_identity_and_finite_loss()
    test_masked_loss_and_azimuth_wraparound()
    test_bf16_projection_if_supported()
    test_projection_uses_mm_hidden_size_not_llm_hidden_size()
    test_invalid_geometry_zeroed_and_no_nan()
    test_auxiliary_geometry_loss_can_be_disabled()
    test_all_invalid_geometry_does_not_nan()
    test_depth_mode_uses_log_depth_not_log_radius()
    test_missing_aux_targets_raise_by_default()
    test_missing_aux_targets_can_be_allowed()
    test_azimuth_wraparound_for_all_loss_types()
    print("Metric-grounded geometry projection sanity checks passed.")


if __name__ == "__main__":
    main()
