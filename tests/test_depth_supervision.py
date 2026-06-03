import importlib.util
import pathlib
import sys
import types

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = REPO_ROOT / "llava" / "model" / "geometry"
BEV_PATH = GEOMETRY_ROOT / "bev_supervision.py"
DEPTH_PATH = GEOMETRY_ROOT / "depth_supervision.py"


def load_geometry_module(module_name, path):
    sys.modules.setdefault("llava", types.ModuleType("llava"))
    sys.modules["llava"].__path__ = [str(REPO_ROOT / "llava")]
    sys.modules.setdefault("llava.model", types.ModuleType("llava.model"))
    sys.modules["llava.model"].__path__ = [str(REPO_ROOT / "llava" / "model")]
    sys.modules.setdefault("llava.model.geometry", types.ModuleType("llava.model.geometry"))
    sys.modules["llava.model.geometry"].__path__ = [str(GEOMETRY_ROOT)]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


load_geometry_module("llava.model.geometry.bev_supervision", BEV_PATH)
depth_mod = load_geometry_module("llava.model.geometry.depth_supervision", DEPTH_PATH)
DepthHead = depth_mod.DepthHead
build_depth_targets_from_point_maps = depth_mod.build_depth_targets_from_point_maps


def _metadata(frame_ids, grid_shapes, visual_indices=None, **extra):
    frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.long)
    if visual_indices is None:
        visual_indices = list(range(len(frame_ids)))
    meta = {
        "visual_token_indices": torch.tensor(visual_indices, dtype=torch.long),
        "visual_frame_ids": frame_ids_tensor,
        "frame_order": sorted(set(int(x) for x in frame_ids)),
        "visual_grid_shapes": grid_shapes,
        "raw_visual_grid_shapes": grid_shapes,
        "newline_token_indices": torch.empty(0, dtype=torch.long),
        "padding_token_indices": torch.empty(0, dtype=torch.long),
        "answer_token_indices": torch.empty(0, dtype=torch.long),
        "text_token_indices": torch.empty(0, dtype=torch.long),
        "special_token_indices": torch.empty(0, dtype=torch.long),
        "camera_prefix_token_indices": torch.empty(0, dtype=torch.long),
    }
    meta.update(extra)
    return meta


def test_depth_head_forward_shape():
    head = DepthHead(5)
    out = head(torch.randn(2, 7, 5))
    assert out.shape == (2, 7)


def test_depth_targets_use_camera_z_log1p():
    cam = torch.zeros(1, 1, 2, 3)
    cam[0, 0, 0, 2] = 2.0
    cam[0, 0, 1, 2] = 8.0
    gt_log, mask, debug = build_depth_targets_from_point_maps(
        {"point_maps_cam": cam},
        _metadata([0, 0], [(1, 2)]),
    )
    assert torch.allclose(gt_log[0], torch.log1p(torch.tensor([2.0, 8.0])))
    assert mask.tolist() == [[True, True]]
    assert debug["samples"][0]["depth_target_key"] == "point_maps_cam"
    assert debug["samples"][0]["depth_target_space"] == "camera"


def test_invalid_zero_nan_and_too_far_depths_are_masked():
    cam = torch.zeros(1, 1, 4, 3)
    cam[0, 0, :, 2] = torch.tensor([2.0, 0.0, float("nan"), 25.0])
    gt_log, mask, _debug = build_depth_targets_from_point_maps(
        {"point_maps_cam": cam},
        _metadata([0, 0, 0, 0], [(1, 4)]),
        depth_max_gt=20.0,
    )
    assert mask.tolist() == [[True, False, False, False]]
    assert torch.isfinite(gt_log[mask]).all()


def test_tensor_payload_is_rejected_by_default_and_allowed_by_config():
    cam = torch.zeros(1, 1, 2, 3)
    cam[..., 2] = torch.tensor([[[1.0, 2.0]]])
    try:
        build_depth_targets_from_point_maps(
            cam,
            _metadata([0, 0], [(1, 2)]),
        )
    except ValueError as exc:
        assert "raw tensor payload" in str(exc)
    else:
        raise AssertionError("Expected raw tensor depth payload to be rejected by default")

    gt_log, mask, debug = build_depth_targets_from_point_maps(
        cam,
        _metadata([0, 0], [(1, 2)]),
        depth_allow_tensor_camera_assumed=True,
    )
    assert torch.allclose(gt_log[0], torch.log1p(torch.tensor([1.0, 2.0])))
    assert mask.tolist() == [[True, True]]
    assert debug["samples"][0]["depth_target_space"] == "camera_tensor_assumed"


def test_generic_point_map_key_requires_explicit_camera_assumption():
    cam = torch.zeros(1, 1, 2, 3)
    cam[..., 2] = torch.tensor([[[1.0, 2.0]]])
    try:
        build_depth_targets_from_point_maps(
            {"point_maps": cam},
            _metadata([0, 0], [(1, 2)]),
            depth_point_map_key="point_maps",
        )
    except ValueError as exc:
        assert "generic point-map" in str(exc)
    else:
        raise AssertionError("Expected generic point_maps key to be rejected by default")

    gt_log, mask, debug = build_depth_targets_from_point_maps(
        {"point_maps": cam},
        _metadata([0, 0], [(1, 2)]),
        depth_point_map_key="point_maps",
        depth_allow_generic_camera_assumed=True,
    )
    assert torch.allclose(gt_log[0], torch.log1p(torch.tensor([1.0, 2.0])))
    assert mask.tolist() == [[True, True]]
    assert debug["samples"][0]["depth_target_space"] == "generic_camera_assumed"


def test_confidence_threshold_is_applied():
    cam = torch.zeros(1, 1, 2, 3)
    cam[..., 2] = 3.0
    conf = torch.tensor([[[0.25, 0.75]]])
    _gt_log, mask, debug = build_depth_targets_from_point_maps(
        {"point_maps_cam": cam, "confidence": conf},
        _metadata([0, 0], [(1, 2)]),
        depth_conf_threshold=0.5,
    )
    assert mask.tolist() == [[False, True]]
    assert debug["samples"][0]["confidence_source"] == "confidence"


def test_multi_frame_depth_targets_follow_visual_metadata_order():
    cam = torch.zeros(2, 1, 2, 3)
    cam[0, 0, 0, 2] = 10.0
    cam[0, 0, 1, 2] = 11.0
    cam[1, 0, 0, 2] = 20.0
    cam[1, 0, 1, 2] = 21.0
    metadata = {
        "visual_token_indices": torch.tensor([5, 2, 6, 3], dtype=torch.long),
        "visual_frame_ids": torch.tensor([1, 0, 1, 0], dtype=torch.long),
        "frame_order": [0, 1],
        "visual_grid_shapes": [(1, 2), (1, 2)],
        "raw_visual_grid_shapes": [(1, 2), (1, 2)],
        "newline_token_indices": torch.empty(0, dtype=torch.long),
        "padding_token_indices": torch.empty(0, dtype=torch.long),
        "answer_token_indices": torch.empty(0, dtype=torch.long),
        "text_token_indices": torch.empty(0, dtype=torch.long),
        "special_token_indices": torch.empty(0, dtype=torch.long),
        "camera_prefix_token_indices": torch.empty(0, dtype=torch.long),
    }
    gt_log, mask, _debug = build_depth_targets_from_point_maps(
        {"point_maps_cam": cam},
        metadata,
        depth_max_gt=25.0,
    )
    expected_depth = torch.tensor([20.0, 10.0, 21.0, 11.0])
    assert torch.allclose(gt_log[0], torch.log1p(expected_depth))
    assert mask.tolist() == [[True, True, True, True]]


def test_reference_point_map_key_is_rejected_for_depth():
    ref = torch.ones(1, 1, 1, 3)
    try:
        build_depth_targets_from_point_maps(
            {"point_maps_ref": ref},
            _metadata([0], [(1, 1)]),
            depth_point_map_key="point_maps_ref",
        )
    except ValueError as exc:
        assert "camera-space point maps" in str(exc)
    else:
        raise AssertionError("Expected reference-frame depth key rejection")
