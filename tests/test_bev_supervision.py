import pathlib
import importlib.util

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BEV_PATH = REPO_ROOT / "llava" / "model" / "geometry" / "bev_supervision.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bev_mod = load_module("_bev_supervision", BEV_PATH)
BEVHead = bev_mod.BEVHead
build_bev_targets_from_point_maps = bev_mod.build_bev_targets_from_point_maps


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


def test_bev_head_forward_shape():
    head = BEVHead(5)
    out = head(torch.randn(2, 7, 5))
    assert out.shape == (2, 7, 2)


def test_reference_target_uses_camera_depth_validity_when_available():
    ref = torch.tensor(
        [
            [
                [[1.0, 9.0, -3.0], [2.0, 9.0, -4.0]],
            ]
        ]
    )
    cam = torch.tensor(
        [
            [
                [[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
            ]
        ]
    )
    gt, mask, debug = build_bev_targets_from_point_maps(
        {"point_maps_ref": ref, "point_maps_cam": cam},
        _metadata([0, 0], [(1, 2)]),
        bev_point_map_key="point_maps_ref",
    )
    assert torch.allclose(gt[0], torch.tensor([[1.0, -3.0], [0.0, 0.0]]))
    assert mask.tolist() == [[True, False]]
    assert debug["samples"][0]["validity_depth_source"] == "point_maps_cam"


def test_camera_target_uses_own_positive_depth():
    cam = torch.tensor(
        [
            [
                [[1.0, 0.0, 2.0], [2.0, 0.0, -1.0]],
            ]
        ]
    )
    _gt, mask, debug = build_bev_targets_from_point_maps(
        {"point_maps_cam": cam},
        _metadata([0, 0], [(1, 2)]),
        bev_point_map_key="point_maps_cam",
    )
    assert mask.tolist() == [[True, False]]
    assert debug["samples"][0]["bev_target_space"] == "camera"
    assert debug["samples"][0]["validity_depth_source"] == "point_maps_cam"


def test_confidence_and_finite_masks_are_applied():
    ref = torch.tensor(
        [
            [
                [[1.0, 0.0, 3.0], [float("nan"), 0.0, 4.0]],
                [[5.0, 0.0, 6.0], [7.0, 0.0, 8.0]],
            ]
        ]
    )
    cam = torch.ones_like(ref)
    cam[..., 2] = 1.0
    conf = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    _gt, mask, debug = build_bev_targets_from_point_maps(
        {"point_maps_ref": ref, "point_maps_cam": cam, "confidence": conf},
        _metadata([0, 0, 0, 0], [(2, 2)]),
        bev_point_map_key="point_maps_ref",
    )
    assert mask.tolist() == [[True, False, False, True]]
    assert debug["samples"][0]["confidence_source"] == "confidence"


def test_confidence_threshold_is_configurable():
    ref = torch.ones(1, 1, 2, 3)
    ref[..., 2] = 3.0
    cam = torch.ones_like(ref)
    cam[..., 2] = 1.0
    conf = torch.tensor([[[0.25, 0.75]]])
    _gt, mask, debug = build_bev_targets_from_point_maps(
        {"point_maps_ref": ref, "point_maps_cam": cam, "confidence": conf},
        _metadata([0, 0], [(1, 2)]),
        bev_point_map_key="point_maps_ref",
        bev_conf_threshold=0.5,
    )
    assert mask.tolist() == [[False, True]]
    assert debug["samples"][0]["confidence_threshold"] == 0.5


def test_masked_adaptive_pooling_averages_valid_points_only():
    ref = torch.zeros(1, 4, 4, 3)
    ref[..., 0] = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    ref[..., 2] = 10.0
    cam = torch.ones_like(ref)
    cam[..., 2] = 1.0
    cam[0, 0, 0, 2] = 0.0
    gt, mask, _debug = build_bev_targets_from_point_maps(
        {"point_maps_ref": ref, "point_maps_cam": cam},
        _metadata([0, 0, 0, 0], [(2, 2)]),
        bev_point_map_key="point_maps_ref",
    )
    expected_x = torch.tensor([(1.0 + 4.0 + 5.0) / 3.0, 4.5, 10.5, 12.5])
    assert torch.allclose(gt[0, :, 0], expected_x)
    assert torch.all(gt[0, :, 1] == 10.0)
    assert mask.tolist() == [[True, True, True, True]]


def test_multi_frame_targets_follow_visual_metadata_order_with_frame_cursors():
    points = torch.zeros(2, 1, 2, 3)
    points[0, 0, 0] = torch.tensor([10.0, 0.0, 100.0])
    points[0, 0, 1] = torch.tensor([11.0, 0.0, 101.0])
    points[1, 0, 0] = torch.tensor([20.0, 0.0, 200.0])
    points[1, 0, 1] = torch.tensor([21.0, 0.0, 201.0])
    cam = torch.ones_like(points)
    cam[..., 2] = 1.0
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
    gt, mask, _debug = build_bev_targets_from_point_maps(
        {"point_maps_ref": points, "point_maps_cam": cam},
        metadata,
        bev_point_map_key="point_maps_ref",
    )
    expected = torch.tensor([[20.0, 200.0], [10.0, 100.0], [21.0, 201.0], [11.0, 101.0]])
    assert torch.allclose(gt[0], expected)
    assert mask.tolist() == [[True, True, True, True]]


def test_visual_metadata_must_not_overlap_excluded_tokens():
    points = torch.ones(1, 1, 1, 3)
    points[..., 2] = 1.0
    metadata = _metadata([0], [(1, 1)], visual_indices=[3], newline_token_indices=torch.tensor([3]))
    try:
        build_bev_targets_from_point_maps(
            {"point_maps_cam": points},
            metadata,
            bev_point_map_key="point_maps_cam",
        )
    except ValueError as exc:
        assert "overlap excluded" in str(exc)
    else:
        raise AssertionError("Expected overlap validation failure")
