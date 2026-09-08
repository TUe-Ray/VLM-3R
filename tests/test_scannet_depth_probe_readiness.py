"""CPU-only checks for ScanNet local depth-probe readiness plumbing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from depth_probe_common import hidden_state_for_layer, layer_feature_path, validate_llm_layers  # noqa: E402
from extract_depth_probe_features import save_frame_outputs, selected_frame_hidden_grids  # noqa: E402
from validate_scannet_depth_probe import postflight, write_parity_marker, write_report  # noqa: E402


class ScanNetDepthProbeReadinessTests(unittest.TestCase):
    def test_explicit_layer_validation_and_historical_indexing(self) -> None:
        self.assertEqual(validate_llm_layers([1, 2, 12, 18, 24], num_hidden_layers=28), [1, 2, 12, 18, 24])
        with self.assertRaises(ValueError):
            validate_llm_layers([1, 1])
        with self.assertRaises(ValueError):
            validate_llm_layers([28], num_hidden_layers=28)
        hidden_states = tuple(torch.full((1, 8, 2), float(index)) for index in range(29))
        self.assertEqual(float(hidden_state_for_layer(hidden_states, 6)[0, 0, 0]), 7.0)

    def test_multiple_layers_use_identical_visual_selection(self) -> None:
        metadata = {
            "visual_token_indices": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            "visual_frame_ids": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "frame_order": [0, 1],
            "visual_grid_shapes": [(2, 2), (2, 2)],
        }
        layer_one = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        layer_two = layer_one + 100
        selected_one = selected_frame_hidden_grids(layer_one, metadata, [0, 1])
        selected_two = selected_frame_hidden_grids(layer_two, metadata, [0, 1])
        self.assertEqual(set(selected_one), {0, 1})
        for frame_index in (0, 1):
            self.assertEqual(tuple(selected_one[frame_index].shape), (2, 2, 2))
            self.assertTrue(torch.equal(selected_two[frame_index] - selected_one[frame_index], torch.full((2, 2, 2), 100.0)))

    def test_layer_paths_and_provenance_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = layer_feature_path(root, "vlm3r_baseline", 1, "sample_f0007")
            second = layer_feature_path(root, "vlm3r_baseline", 2, "sample_f0007")
            self.assertNotEqual(first, second)
            save_frame_outputs(
                output_root=root,
                model_label="vlm3r_baseline",
                frame_record={"frame_sample_id": "sample_f0007", "video_sample_id": "sample"},
                llm_features={"layer_1": torch.ones(2, 2, 3), "layer_2": torch.zeros(2, 2, 3)},
                pre_llm_features={},
                gt_depth=torch.ones(2, 2),
                gt_valid=torch.ones(2, 2, dtype=torch.bool),
                metadata={"selected_frame_indices": [7, 22]},
                feature_provenance={"hidden_state_indexing": "requested_L -> hidden_states[L + 1]", "manifest_sha256": "test"},
                cache_dtype=torch.float16,
            )
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertTrue(torch.equal(torch.load(first, map_location="cpu"), torch.ones(2, 2, 3, dtype=torch.float16)))
            provenance = json.loads((first.parent / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["hidden_state_indexing"], "requested_L -> hidden_states[L + 1]")
            self.assertEqual(provenance["feature_level"], "layer_1")

    def test_parity_warning_does_not_create_pass_marker(self) -> None:
        provenance_root = Path("/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance")
        historical_metrics = json.loads((provenance_root / "baseline_L6" / "metrics.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preflight = root / "preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "assessment": "PASS",
                        "historical_reference": {"split_sha256": "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"},
                        "checkpoint_identity": {"test": "identity"},
                    }
                ),
                encoding="utf-8",
            )
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps(historical_metrics), encoding="utf-8")
            args = Namespace(provenance_root=provenance_root, new_metrics=metrics, preflight_report=preflight)
            passed = postflight(args)
            self.assertEqual(passed["assessment"], "PASS")
            report_path = write_report(passed, root, root / "postflight.json")
            marker = root / "baseline_l6_parity_pass.json"
            write_parity_marker(marker, passed, report_path)
            self.assertTrue(marker.is_file())

            warning_metrics = dict(historical_metrics)
            warning_metrics["absrel"] = historical_metrics["absrel"] * 1.06
            metrics.write_text(json.dumps(warning_metrics), encoding="utf-8")
            warning = postflight(args)
            self.assertEqual(warning["assessment"], "PASS_WITH_WARNING")
            with self.assertRaises(ValueError):
                write_parity_marker(root / "warning_marker.json", warning, report_path)

            failing_metrics = dict(historical_metrics)
            failing_metrics["mae"] = historical_metrics["mae"] * 1.06
            metrics.write_text(json.dumps(failing_metrics), encoding="utf-8")
            self.assertEqual(postflight(args)["assessment"], "FAIL")

    def test_smoke_and_full_namespaces_cannot_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            smoke = base / "smoke" / "baseline_l6" / "features" / "vlm3r_baseline" / "layer_6"
            full = base / "full" / "features" / "vlm3r_baseline" / "layer_6"
            self.assertNotEqual(smoke, full)
            smoke.mkdir(parents=True)
            (smoke / "frame_sample.pt").write_bytes(b"smoke")
            self.assertFalse((full / "frame_sample.pt").exists())


if __name__ == "__main__":
    unittest.main()
