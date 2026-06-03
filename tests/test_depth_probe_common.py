import importlib.util
import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON_PATH = REPO_ROOT / "scripts" / "probing" / "depth_probe_common.py"
TRAIN_PATH = REPO_ROOT / "scripts" / "probing" / "train_depth_probes.py"
sys.path.insert(0, str(COMMON_PATH.parent))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("_depth_probe_common", COMMON_PATH)
train_mod = load_module("_train_depth_probes", TRAIN_PATH)


def test_resolve_sidecar_path_uses_existing_fast_layout(tmp_path):
    sidecar = tmp_path / "scannet" / "spatial_features_points" / "scene0000_00.pt"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"x")
    resolved = common.resolve_sidecar_path(
        "scannet/videos/scene0000_00.mp4",
        tmp_path,
        "spatial_features_points",
    )
    assert resolved == sidecar


def test_frame_sample_indices_are_deterministic_and_sorted():
    first = common.frame_sample_indices(32, 2, 42, "video-a")
    second = common.frame_sample_indices(32, 2, 42, "video-a")
    other = common.frame_sample_indices(32, 2, 42, "video-b")
    assert first == second
    assert first == sorted(first)
    assert len(first) == 2
    assert first != other


def test_select_point_maps_prefers_camera_coordinates():
    payload = {
        "point_maps_ref": torch.ones(2, 4, 4, 3),
        "point_maps_cam": torch.full((2, 4, 4, 3), 2.0),
    }
    point_maps, key, mode = common.select_point_maps(payload)
    assert key == "point_maps_cam"
    assert mode == "camera_z"
    assert torch.equal(point_maps, payload["point_maps_cam"])


def test_reference_point_maps_require_explicit_euclidean_opt_in():
    payload = {"point_maps_ref": torch.ones(2, 4, 4, 3)}
    try:
        common.select_point_maps(payload)
    except ValueError as exc:
        assert "allow-euclidean-depth" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
    _point_maps, key, mode = common.select_point_maps(payload, allow_euclidean_depth=True)
    assert key == "point_maps_ref"
    assert mode == "euclidean"


def test_downsample_depth_uses_valid_average():
    depth = torch.tensor(
        [
            [1.0, 3.0, 0.0, 0.0],
            [5.0, 7.0, 2.0, 4.0],
            [0.0, 0.0, 10.0, 14.0],
            [0.0, 0.0, 18.0, 22.0],
        ]
    )
    pooled, valid = common.downsample_depth_to_grid(depth, (2, 2))
    expected = torch.tensor([[4.0, 3.0], [0.0, 16.0]])
    assert torch.allclose(pooled, expected)
    assert valid.tolist() == [[True, True], [False, True]]


def test_reshape_tokens_to_grid_checks_token_count():
    tokens = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    grid = common.reshape_tokens_to_grid(tokens, (2, 2))
    assert grid.shape == (2, 2, 3)
    try:
        common.reshape_tokens_to_grid(tokens, (1, 3))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_metric_values_depth_metrics():
    pred = torch.tensor([1.0, 2.0, 4.0])
    gt = torch.tensor([1.0, 4.0, 2.0])
    metrics = common.metric_values(pred, gt)
    assert abs(metrics["mae"] - (0.0 + 2.0 + 2.0) / 3.0) < 1e-6
    assert metrics["num_tokens"] == 3
    assert 0.0 <= metrics["delta125"] <= 1.0


def test_depth_probe_mlp_forward_shape():
    model = train_mod.DepthProbeMLP(5)
    out = model(torch.randn(7, 5))
    assert out.shape == (7,)
