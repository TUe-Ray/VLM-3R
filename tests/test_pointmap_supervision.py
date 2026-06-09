import importlib.util
import pathlib

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = REPO_ROOT / "llava" / "model" / "geometry"


def load_geometry_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "llava.model.geometry"
    spec.loader.exec_module(module)
    return module


load_geometry_module("llava.model.geometry.bev_supervision", GEOMETRY_ROOT / "bev_supervision.py")
pointmap_mod = load_geometry_module("llava.model.geometry.pointmap_supervision", GEOMETRY_ROOT / "pointmap_supervision.py")
PointMapHead = pointmap_mod.PointMapHead
build_pointmap_targets_from_point_maps = pointmap_mod.build_pointmap_targets_from_point_maps


def _metadata(frame_ids, grid_shapes, visual_indices=None, **extra):
    if visual_indices is None:
        visual_indices = list(range(len(frame_ids)))
    meta = {
        "visual_token_indices": torch.tensor(visual_indices, dtype=torch.long),
        "visual_frame_ids": torch.tensor(frame_ids, dtype=torch.long),
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


def test_pointmap_head_forward_shape():
    head = PointMapHead(5)
    out = head(torch.randn(2, 7, 5))
    assert out.shape == (2, 7, 3)


def test_world_xyz_targets_follow_visual_metadata_order():
    points = torch.zeros(2, 1, 2, 3)
    points[0, 0, 0] = torch.tensor([10.0, 0.0, 100.0])
    points[0, 0, 1] = torch.tensor([11.0, 1.0, 101.0])
    points[1, 0, 0] = torch.tensor([20.0, 2.0, 200.0])
    points[1, 0, 1] = torch.tensor([21.0, 3.0, 201.0])
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
    gt, mask, debug = build_pointmap_targets_from_point_maps(
        {"point_maps_ref": points, "point_maps_cam": cam},
        metadata,
        pointmap_point_map_key="point_maps_ref",
    )
    expected = torch.tensor(
        [
            [20.0, 2.0, 200.0],
            [10.0, 0.0, 100.0],
            [21.0, 3.0, 201.0],
            [11.0, 1.0, 101.0],
        ]
    )
    assert torch.allclose(gt[0], expected)
    assert mask.tolist() == [[True, True, True, True]]
    assert debug["pointmap_target_space"] == "reference"


def test_camera_frame_key_is_rejected_for_pointmap_supervision():
    points = torch.ones(1, 1, 1, 3)
    try:
        build_pointmap_targets_from_point_maps(
            {"point_maps_cam": points},
            _metadata([0], [(1, 1)]),
            pointmap_point_map_key="point_maps_cam",
        )
    except ValueError as exc:
        assert "world/reference-frame" in str(exc)
    else:
        raise AssertionError("Expected camera-frame point-map key to be rejected")


def test_camera_prefix_overlap_with_visual_indices_is_rejected():
    points = torch.ones(1, 1, 1, 3)
    metadata = _metadata([0], [(1, 1)], visual_indices=[3], camera_prefix_token_indices=torch.tensor([3]))
    try:
        build_pointmap_targets_from_point_maps(
            {"point_maps_ref": points},
            metadata,
            pointmap_point_map_key="point_maps_ref",
        )
    except ValueError as exc:
        assert "overlap excluded" in str(exc)
    else:
        raise AssertionError("Expected overlap validation failure")
