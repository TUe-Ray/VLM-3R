"""CPU-only integrity tests for the SpatialFocus SigLIP sidecar protocol."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts/extraction/extract_siglip_spatialfocus_features.py"
SPEC = importlib.util.spec_from_file_location("siglip_spatialfocus_extractor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


class SpatialFocusProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original_shape = extractor.EXPECTED_SHAPE
        self.original_dtype = extractor.EXPECTED_DTYPE
        # Keep this protocol test small; production constants are covered by
        # the extractor's strict runtime assertion.
        extractor.EXPECTED_SHAPE = (2, 3, 4)
        extractor.EXPECTED_DTYPE = "torch.bfloat16"
        self.entry = {
            "key": "dataset/sample.pt",
            "dataset": "dataset",
            "relative_output": "sample.pt",
            "frame_indices": [0, 1],
            "source_video": "/unused.mp4",
            "padding": {},
            "cut3r": {},
        }
        self.manifest = {
            "schema_version": 1,
            "contract": {"shape": [2, 3, 4]},
            "entries": [self.entry],
        }
        self.manifest["digest"] = extractor.digest(self.manifest)

    def tearDown(self) -> None:
        extractor.EXPECTED_SHAPE = self.original_shape
        extractor.EXPECTED_DTYPE = self.original_dtype
        self.temporary.cleanup()

    def test_atomic_publish_fast_resume_and_full_verify(self) -> None:
        feature = extractor.output_path(self.root, self.entry)
        tensor = torch.ones(extractor.EXPECTED_SHAPE, dtype=torch.bfloat16)
        extractor.publish(feature, tensor, self.entry, self.manifest, "metadata-digest", rank=3)

        self.assertTrue(extractor.fast_done(feature, self.entry, self.manifest))
        report = extractor.scan(self.manifest, self.root, verify=True, max_samples=0)
        self.assertEqual(report["completed"], [self.entry["key"]])
        self.assertFalse(report["missing"])
        self.assertFalse(report["corrupted"])

        marker = extractor.marker_path(feature)
        value = extractor.load_json(marker)
        value["bytes"] += 1
        extractor.atomic_json(marker, value)
        self.assertFalse(extractor.fast_done(feature, self.entry, self.manifest))

    def test_partition_rule_is_complete_and_disjoint(self) -> None:
        keys = [f"sample-{index}" for index in range(73)]
        for world_size in (8, 32):
            assigned = [key for rank in range(world_size) for index, key in enumerate(keys) if index % world_size == rank]
            self.assertEqual(sorted(assigned), sorted(keys))
            self.assertEqual(len(assigned), len(set(assigned)))

    def test_layer_subdirectories_share_dataset_prefixed_keys(self) -> None:
        for dataset in ("scannet", "scannetpp"):
            path = self.root / dataset / "spatial_features_dec_6" / "split" / "sample.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        files = extractor.find_cut3r_files(self.root, "spatial_features_dec_6")
        self.assertEqual(set(files), {"scannet/split/sample.pt", "scannetpp/split/sample.pt"})

    def test_historical_metadata_reconstructs_via_formal_sampler_path(self) -> None:
        roots = {}
        for layer, subdir in (("6", "spatial_features_dec_6"), ("9", "spatial_features_dec_9"), ("12", "spatial_features")):
            root = self.root / f"layer-{layer}"
            path = root / "scannet" / subdir / "sample.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"source_video": "/historical/video.mp4", "frames_upbound": 32, "video_fps": 1} if layer != "12" else {}
            torch.save({"patch_tokens": torch.zeros((32, 729, 768), dtype=torch.float16), "metadata": metadata}, path)
            roots[layer] = root

        original_sampler = extractor.formal_training_frame_indices
        extractor.formal_training_frame_indices = lambda source, metadata: list(range(32))
        try:
            manifest_path = self.root / "alignment.json"
            args = SimpleNamespace(
                manifest=str(manifest_path),
                cut3r_layer_root=[f"{layer}={root}" for layer, root in roots.items()],
                cut3r_subdir=["6=spatial_features_dec_6", "9=spatial_features_dec_9", "12=spatial_features"],
                alignment_layer="6",
                source_video_root=None,
                siglip_checkpoint="siglip-test", vision_select_feature="patch", dataset_root=None,
                pipeline_alignment_json=None,
            )
            extractor.command_build_manifest(args)
            manifest = extractor.load_manifest(manifest_path)
            self.assertEqual(manifest["entries"][0]["alignment_source"], "formal_spatialstack_sampler_reconstructed")
            self.assertEqual(manifest["entries"][0]["frame_indices"], list(range(32)))
        finally:
            extractor.formal_training_frame_indices = original_sampler


if __name__ == "__main__":
    unittest.main()
