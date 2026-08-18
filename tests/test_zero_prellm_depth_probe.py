"""CPU-only tests for zero-spatial pre-LLM capture and provenance contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from extract_depth_probe_features import (  # noqa: E402
    assert_zero_spatial_post_fusion_projector_capture,
    normalize_captured_video_tokens,
    register_pre_llm_hooks,
)
from validate_zero_prellm import validate_extraction_provenance  # noqa: E402


class _FakeVision(torch.nn.Module):
    def forward(self, value):
        return value


class _FakeFusion(torch.nn.Module):
    def forward(self, value):
        return value + 1


class _FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = _FakeVision()
        self.fusion = _FakeFusion()
        self.mm_projector = torch.nn.Linear(3, 4, bias=False)

    def get_fusion_block(self):
        return self.fusion


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _FakeBase()
        self.pool_calls = 0

    def get_model(self):
        return self.base

    def get_vision_tower(self):
        return self.base.vision

    def get_2dPool(self, value):
        self.pool_calls += 1
        return value[:, :196, :]


class ZeroPreLlmDepthProbeTests(unittest.TestCase):
    def test_zero_hooks_capture_post_fusion_projector_output(self):
        model = _FakeModel()
        captured = {}
        handles = register_pre_llm_hooks(
            model,
            "zero_spatial",
            ["siglip_output", "projected_features"],
            captured,
        )
        try:
            value = model.get_vision_tower()(torch.zeros(2, 729, 3))
            fused = model.get_model().get_fusion_block()(value)
            model.get_model().mm_projector(fused)
        finally:
            for handle in handles:
                handle.remove()
        self.assertIn("siglip_output", captured)
        self.assertIn("projected_features", captured)
        report = assert_zero_spatial_post_fusion_projector_capture(captured)
        self.assertEqual(report["assessment"], "PASS")
        self.assertTrue(report["fusion_output_equals_mm_projector_input"])
        self.assertEqual(tuple(captured["projected_features"].shape), (2, 729, 4))

    def test_zero_normalization_requires_model_pool(self):
        model = _FakeModel()
        normalized = normalize_captured_video_tokens(
            model,
            torch.randn(2, 729, 3),
            num_frames=2,
            target_grid_shape=(14, 14),
            require_model_pool=True,
        )
        self.assertEqual(tuple(normalized.shape), (2, 196, 3))
        self.assertEqual(model.pool_calls, 1)

    def test_zero_normalization_rejects_missing_pool(self):
        model = SimpleNamespace(get_2dPool=lambda value: (_ for _ in ()).throw(RuntimeError("unavailable")))
        with self.assertRaises(RuntimeError):
            normalize_captured_video_tokens(
                model,
                torch.randn(2, 729, 3),
                num_frames=2,
                target_grid_shape=(14, 14),
                require_model_pool=True,
            )

    def test_zero_provenance_requires_post_fusion_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_indices = root / "split.json"
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            sample_indices.write_text("{\"videos\": []}\n", encoding="utf-8")
            forward_root = root / "forward"
            target_root = root / "targets"
            feature_root = root / "features"
            payload = {
                "model_label": "zero_spatial",
                "requested_pre_llm_features": ["siglip_output", "projected_features"],
                "requested_llm_layers": [],
                "sample_indices_sha256": __import__("hashlib").sha256(sample_indices.read_bytes()).hexdigest(),
                "checkpoint_config_sha256": __import__("hashlib").sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
                "forward_frames_root": str(forward_root),
                "probe_targets_root": str(target_root),
                "feature_root": str(feature_root),
                "extraction_samples": [{"first_video_runtime_assertions": {"assessment": "PASS"}}],
                "zero_spatial_post_fusion_projector_contract": {
                    "projected_features": "mm_projector output after zero-spatial fusion path; verified by fusion output == mm_projector input"
                },
            }
            provenance = root / "extraction_provenance.json"
            provenance.write_text(json.dumps(payload), encoding="utf-8")
            report = validate_extraction_provenance(
                provenance,
                sample_indices=sample_indices,
                checkpoint=checkpoint,
                forward_root=forward_root,
                target_root=target_root,
                feature_root=feature_root,
            )
            self.assertEqual(report["assessment"], "PASS")


if __name__ == "__main__":
    unittest.main()
