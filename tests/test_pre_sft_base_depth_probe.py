"""CPU-only contract tests for the pre-SFT plain-base depth-probe control."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from depth_probe_common import layer_feature_path  # noqa: E402
from local_depth_probe_cache import (  # noqa: E402
    assert_pre_sft_base_vlm_forward_contract,
    pre_sft_projector_loading_evidence,
)
from validate_pre_sft_base_depth_probe import (  # noqa: E402
    MODEL_LABEL,
    attestation_path,
    run_identity,
    verify_smoke_attestation,
)


class _Base:
    def __init__(self, *, spatial=None, fusion=None):
        self.mm_projector = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.GELU(), torch.nn.Linear(2, 3))
        self._spatial = spatial
        self._fusion = fusion

    def get_spatial_tower(self):
        return self._spatial

    def get_fusion_block(self):
        return self._fusion


class LlavaQwenForCausalLM:
    def __init__(self, source: Path, *, spatial=None, fusion=None, missing=None):
        self._base = _Base(spatial=spatial, fusion=fusion)
        self._vision = object()
        self._pre_sft_source_path = str(source)
        self._pre_sft_loading_info = {
            "missing_keys": list(missing or []), "unexpected_keys": [], "mismatched_keys": [],
        }
        self.config = SimpleNamespace(
            use_cut3r_spatialstack=False,
            use_geometry_aware_projection=False,
            use_cut3r_camera_tokens=False,
            use_spatial_bridge_tokens=False,
            llm_visual_3d_rope_enable=False,
            use_bev_supervision=False,
            use_depth_supervision=False,
            use_pointmap_supervision=False,
        )

    def get_model(self):
        return self._base

    def get_vision_tower(self):
        return self._vision

    def get_spatial_tower(self):
        return self._base.get_spatial_tower()

    def get_fusion_block(self):
        return self._base.get_fusion_block()


class PreSftBaseDepthProbeTests(unittest.TestCase):
    def test_base_contract_requires_plain_projector_loaded_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            model = LlavaQwenForCausalLM(source)
            contract = assert_pre_sft_base_vlm_forward_contract(model, source)
            self.assertTrue(contract["no_vlm3r_sft_adapter_loaded"])
            self.assertEqual(contract["projector_loading_evidence"]["projector_missing_keys"], [])
            self.assertEqual(contract["projector_loading_evidence"]["projector_weight_source"], str(source))

    def test_contract_rejects_spatial_module_and_missing_projector_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            with self.assertRaises(RuntimeError):
                assert_pre_sft_base_vlm_forward_contract(LlavaQwenForCausalLM(source, spatial=object()), source)
            with self.assertRaises(RuntimeError):
                pre_sft_projector_loading_evidence(LlavaQwenForCausalLM(source, missing=["model.mm_projector.0.weight"]))

    def test_label_paths_do_not_collide_with_post_sft_conditions(self) -> None:
        root = Path("/tmp/pre_sft_probe_test")
        base = layer_feature_path(root, MODEL_LABEL, 6, "frame_id")
        baseline = layer_feature_path(root, "vlm3r_baseline", 6, "frame_id")
        zero = layer_feature_path(root, "zero_spatial", 6, "frame_id")
        self.assertNotEqual(base, baseline)
        self.assertNotEqual(base, zero)

    def test_stale_smoke_attestation_is_rejected(self) -> None:
        base = Path("/mnt/DATA_SSD/shaoruei/models/base/LLaVA-NeXT-Video-7B-Qwen2")
        siglip = Path("/mnt/DATA_SSD/shaoruei/models/base/siglip-so400m-patch14-384")
        split = Path("/home/shaoruei/probe_provenance/scannet_baseline_L6/scannet_baseline_L6_depth_provenance/splits/semantic_probe_scannet_final_usable_sample_indices.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                base_model=base, siglip=siglip, sample_indices=split, dtype="float16", device_map="auto",
                attn_implementation=None, smoke_root=root, output_root=root,
            )
            payload = {
                "assessment": "PASS", "identity": run_identity(args),
                "projected_grid_shape": [14, 14], "visual_tokens_per_selected_frame": 196,
            }
            path = attestation_path(root)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(verify_smoke_attestation(args)["assessment"], "PASS")
            args.dtype = "float32"
            self.assertEqual(verify_smoke_attestation(args)["assessment"], "FAIL")

    def test_runner_requires_readiness_and_does_not_chain_full_run(self) -> None:
        text = (REPO_ROOT / "scripts/probing/run_scannet_depth_layer_completion_local.sh").read_text(encoding="utf-8")
        block = text.split("  base-smoke)", 1)[1].split("  base-full)", 1)[0]
        self.assertLess(block.index("require_gpu"), block.index("extract_base_features"))
        self.assertNotIn("base-full", block)


if __name__ == "__main__":
    unittest.main()
