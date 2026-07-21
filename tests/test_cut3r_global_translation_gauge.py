import importlib.util
import pathlib

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analysis" / "verify_cut3r_global_translation_gauge.py"
SPEC = importlib.util.spec_from_file_location("cut3r_global_translation_gauge", SCRIPT_PATH)
GAUGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GAUGE)


def test_known_c2w_translation_is_a_camera_geometry_gauge():
    sample = GAUGE.make_synthetic_sample()
    result = GAUGE.evaluate_sample(sample, (1.0, -0.5, 0.25), atol=1e-5, rtol=1e-5)

    assert result["correct_transform_pass"]
    assert result["point_only_control_detected"]
    assert result["correct_transform_camera_errors"]["max_abs"] <= 1e-5
    assert result["correct_transform_depth_errors"]["max_abs"] <= 1e-5
    assert result["point_only_camera_errors"]["max_abs"] > 0.1


def test_pose_representation_is_camera_to_world_for_known_point_cloud():
    sample = GAUGE.make_synthetic_sample()
    reconstructed = GAUGE.reference_to_camera(sample.world, sample.pose)
    assert torch.allclose(reconstructed, sample.camera, atol=1e-6, rtol=1e-6)
