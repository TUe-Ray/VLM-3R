import importlib.util
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/analysis/diagnose_cut3r_pointmap_pose_alignment.py"
SPEC = importlib.util.spec_from_file_location("cut3r_alignment_diagnostic", SCRIPT)
DIAG = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = DIAG
SPEC.loader.exec_module(DIAG)


def make_sample(with_confidence: bool = True):
    frames, height, width = 2, 3, 4
    cam = torch.randn(frames, height, width, 3)
    cam[..., 2] = cam[..., 2].abs() + 1.0
    angle = torch.tensor([0.0, 0.25])
    quaternion = torch.stack(
        (torch.cos(angle / 2), torch.zeros_like(angle), torch.sin(angle / 2), torch.zeros_like(angle)),
        dim=-1,
    )
    pose_encoding = torch.cat((torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]]), quaternion), dim=-1)
    pose = DIAG.pose_encoding_to_camera(pose_encoding)
    ref = DIAG.ref_from_camera(cam, pose)
    confidence = torch.arange(frames * height * width).reshape(frames, height, width).float()
    return DIAG.Sample(
        sample_id="synthetic",
        sidecar_path=Path("synthetic.pt"),
        point_maps_cam=cam,
        point_maps_ref=ref,
        pose_encoding=pose_encoding,
        camera_pose=pose,
        conf_self=confidence if with_confidence else None,
        conf_ref=confidence if with_confidence else None,
        metadata={},
    )


def test_official_pose_smoke_and_inverse_roundtrip():
    result = DIAG.run_pose_smoke_tests()
    assert result["passed"]
    assert result["official_inverse_max_abs"] < 1e-5
    assert result["intentional_wrong_direction_rmse"] > 0.1


def test_rigid_consistency_is_symmetric_and_confidence_changes_counts():
    result, maps = DIAG.evaluate_sample(make_sample(with_confidence=True), return_maps=True)
    assert maps["error_3d"].max() < 1e-5
    assert result["aggregate"]["all"]["error_3d"]["max"] < 1e-5
    assert result["aggregate"]["all"]["reference_direction_3d"]["max"] < 1e-5
    counts = [result["aggregate"][key]["valid_points"] for key in DIAG.CONFIDENCE_SWEEP]
    assert counts[0] > counts[1] > counts[2] > counts[3]


def test_missing_confidence_is_explicit_not_silently_uniform():
    result, _ = DIAG.evaluate_sample(make_sample(with_confidence=False))
    assert result["aggregate"]["all"]["available"]
    for key in ("top75", "top50", "top25"):
        assert not result["aggregate"][key]["available"]
        assert "absent" in result["aggregate"][key]["skip_reason"]


def test_pose_alignment_recovers_known_similarity():
    sample = make_sample()
    pred = sample.camera_pose
    rotation = DIAG.pose_encoding_to_camera(
        torch.tensor([[0.0, 0.0, 0.0, 0.9238795, 0.0, 0.0, 0.3826834]])
    )[0, :3, :3]
    scale = 2.5
    translation = torch.tensor([1.0, -2.0, 0.5])
    gt = torch.eye(4).repeat(pred.shape[0], 1, 1)
    gt[:, :3, :3] = rotation @ pred[:, :3, :3]
    gt[:, :3, 3] = scale * torch.einsum("ij,fj->fi", rotation, pred[:, :3, 3]) + translation
    aligned = DIAG.align_predicted_poses(pred, gt, torch.ones(pred.shape[0], dtype=torch.bool))
    assert torch.allclose(aligned["camera_pose_aligned"], gt, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["scale"], torch.tensor(scale), atol=1e-5)
