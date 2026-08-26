"""CPU contract test for paired geometry-perturbation probe materialization."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/probing/materialize_geometry_perturbation_probe_features.py"


class MaterializeGeometryPerturbationProbeFeaturesTest(unittest.TestCase):
    def test_materializes_paired_off_and_delta_with_preintervention_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "cache"
            pair_dir = root / "pairs" / "SS123"
            pair_dir.mkdir(parents=True)
            video = {
                "video_sample_id": "video-a",
                "video_path": "scannet/videos/scene0000_00.mp4",
                "split": "train",
                "frames": [{"frame_index": 0, "frame_sample_id": "video-a_f0000"}],
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"videos": [video]}), encoding="utf-8")
            (output_root / "metadata").mkdir(parents=True)
            torch.save({"visual_grid_shape": (14, 14)}, output_root / "metadata" / "frame_video-a_f0000.pt")
            normal_l0 = torch.zeros(196, 3584)
            normal_l1 = torch.ones(196, 3584)
            payload = {
                "schema_version": "frozen_probe_geometry_perturbation_features_v1",
                "model_label": "SS123",
                "video_id": "video-a",
                "video_path": video["video_path"],
                "split": "train",
                "selected_frames": [0],
                "hidden_state_indexing": "requested_L -> hidden_states[L + 1] (post-decoder-block L; includes injection at L)",
                "normal_by_layer": {"layer_0": {"0": normal_l0}, "layer_1": {"0": normal_l1}},
                "geometry_off_all_by_layer": {"layer_0": {"0": normal_l0.clone()}, "layer_1": {"0": torch.zeros_like(normal_l1)}},
            }
            torch.save(payload, pair_dir / "video_video-a.pt")
            subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--source-pair-root", str(root / "pairs"),
                    "--output-root", str(output_root), "--sample-indices", str(manifest),
                    "--model-label", "SS123", "--layers", "0,1", "--injection-layers", "1", "--delete-source",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            off = torch.load(output_root / "features" / "SS123__geometry_off" / "layer_1" / "frame_video-a_f0000.pt")
            delta = torch.load(output_root / "features" / "SS123__geometry_delta" / "layer_1" / "frame_video-a_f0000.pt")
            self.assertEqual(tuple(off.shape), (14, 14, 3584))
            self.assertEqual(tuple(delta.shape), (14, 14, 3584))
            self.assertTrue(torch.equal(off, torch.zeros_like(off)))
            self.assertTrue(torch.equal(delta, torch.ones_like(delta)))
            self.assertFalse((pair_dir / "video_video-a.pt").exists())


if __name__ == "__main__":
    unittest.main()
