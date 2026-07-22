import copy
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import HfArgumentParser

from llava.model.cut3r_token_only import Cut3RTokenOnlyProjector, extract_cut3r_patch_tokens
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    assert_cut3r_token_only_trainable_policy,
    find_all_linear_names,
)


class RaisingVisionTower(nn.Module):
    num_patches_per_side = 27

    def forward(self, _images):
        raise AssertionError("SigLIP vision tower must not run in CUT3R-token-only mode")


class CountingVisionTower(nn.Module):
    num_patches_per_side = 27

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return torch.ones(images.shape[0], 729, self.hidden_size, dtype=images.dtype)


class FakeBase(nn.Module):
    def __init__(self, config, hidden_size=8, vision=None):
        super().__init__()
        self.config = config
        self.cut3r_token_projector = Cut3RTokenOnlyProjector(768, hidden_size)
        self.vision_tower = vision or RaisingVisionTower()
        self.mm_projector = nn.Identity()
        self._cut3r_token_only_last_metrics = {}

    def get_cut3r_token_projector(self):
        return self.cut3r_token_projector

    def get_vision_tower(self):
        return self.vision_tower

    def get_spatial_tower(self):
        return None

    def get_fusion_block(self):
        return None


class EncodeHarness:
    encode_images = LlavaMetaForCausalLM.encode_images
    _encode_cut3r_only_sidecars = LlavaMetaForCausalLM._encode_cut3r_only_sidecars
    get_2dPool = LlavaMetaForCausalLM.get_2dPool
    add_token_per_grid = LlavaMetaForCausalLM.add_token_per_grid
    _split_prefix_tokens_for_square_grid = LlavaMetaForCausalLM._split_prefix_tokens_for_square_grid

    def __init__(self, config, base):
        self.config = config
        self._base = base
        self.model = SimpleNamespace(image_newline=nn.Parameter(torch.full((8,), -7.0)))

    def get_model(self):
        return self._base

    def get_vision_tower(self):
        return self._base.get_vision_tower()


class TinyLoraCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(8, 8, bias=False)
        self.language_model.register_parameter("lora_A", nn.Parameter(torch.ones(1, 8)))
        self.cut3r_token_projector = Cut3RTokenOnlyProjector(768, 8)
        self.vision_tower = nn.Linear(8, 8, bias=False)
        self.language_model.weight.requires_grad_(False)
        self.vision_tower.weight.requires_grad_(False)


class Cut3RTokenOnlyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_projector_shape_finite_and_update(self):
        projector = Cut3RTokenOnlyProjector(768, 32)
        tokens = torch.randn(2, 729, 768)
        before = projector.proj_out.weight.detach().clone()
        output = projector(tokens)
        self.assertEqual(tuple(output.shape), (2, 729, 32))
        self.assertTrue(torch.isfinite(output).all())
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, projector.proj_out.weight.detach()))

    def test_sidecar_rejections_and_batch_order(self):
        valid = {"patch_tokens": torch.zeros(2, 729, 768)}
        self.assertEqual(tuple(extract_cut3r_patch_tokens(valid, 2).shape), (2, 729, 768))
        with self.assertRaises(KeyError):
            extract_cut3r_patch_tokens({}, 2)
        with self.assertRaises(ValueError):
            extract_cut3r_patch_tokens(valid, 2, sidecar_key="camera_tokens")
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(1, 729, 768)}, 2)
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(2, 728, 768)}, 2)
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": torch.zeros(2, 729, 767)}, 2)
        bad = torch.zeros(2, 729, 768)
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            extract_cut3r_patch_tokens({"patch_tokens": bad}, 2)
        config = SimpleNamespace(
            visual_token_source="cut3r_only",
            cut3r_token_sidecar_key="patch_tokens",
            cut3r_token_feature_dim=768,
            cut3r_token_debug_telemetry=False,
        )
        harness = EncodeHarness(config, FakeBase(config))
        with self.assertRaises(RuntimeError):
            harness.encode_images(torch.empty(3, 3, 4, 4), spatial_features=[valid], split_sizes=[2, 1])

    def test_cut3r_source_bypasses_siglip_and_preserves_sample_frame_order(self):
        config = SimpleNamespace(
            visual_token_source="cut3r_only",
            cut3r_token_sidecar_key="patch_tokens",
            cut3r_token_feature_dim=768,
            cut3r_token_debug_telemetry=False,
        )
        base = FakeBase(config, hidden_size=8, vision=RaisingVisionTower())
        harness = EncodeHarness(config, base)
        first = torch.zeros(1, 729, 768)
        second = torch.ones(2, 729, 768)
        output = harness.encode_images(
            torch.empty(3, 3, 4, 4),
            spatial_features=[{"patch_tokens": first}, {"patch_tokens": second}],
            split_sizes=[1, 2],
        )
        expected = base.cut3r_token_projector(torch.cat((first, second), dim=0))
        self.assertTrue(torch.allclose(output, expected))
        self.assertEqual(base._cut3r_token_only_last_metrics["split_sizes"], [1, 2])
        self.assertTrue(base._cut3r_token_only_last_metrics["siglip_forward_bypassed"])

    def test_siglip_only_keeps_existing_vision_path(self):
        config = SimpleNamespace(visual_token_source="siglip_only", use_geometry_aware_projection=False)
        vision = CountingVisionTower(hidden_size=8)
        base = FakeBase(config, hidden_size=8, vision=vision)
        harness = EncodeHarness(config, base)
        output = harness.encode_images(torch.zeros(2, 3, 4, 4))
        self.assertEqual(vision.calls, 1)
        self.assertEqual(tuple(output.shape), (2, 729, 8))

    def test_pooling_and_newline_layout_is_frame_major(self):
        config = SimpleNamespace(mm_spatial_pool_mode="bilinear", mm_spatial_pool_stride=2, add_faster_video=False)
        base = FakeBase(config, hidden_size=8, vision=CountingVisionTower(hidden_size=8))
        harness = EncodeHarness(config, base)
        grid = torch.zeros(2, 729, 8)
        coordinates = torch.arange(729, dtype=torch.float32).view(1, 729, 1)
        grid[0] = coordinates
        grid[1] = coordinates + 10000.0
        pooled = harness.get_2dPool(grid, stride=2)
        self.assertEqual(tuple(pooled.shape), (2, 196, 8))
        self.assertLess(float(pooled[0].max()), 1000.0)
        self.assertGreater(float(pooled[1].min()), 9000.0)
        layout = harness.add_token_per_grid(pooled).view(2, 210, 8)
        self.assertEqual(tuple(layout.shape), (2, 210, 8))
        newline_positions = [row * 15 + 14 for row in range(14)]
        self.assertTrue(torch.all(layout[:, newline_positions] == -7.0))
        self.assertTrue(torch.all(layout[0, :14] < 1000.0))
        self.assertTrue(torch.all(layout[1, :14] > 9000.0))

    def test_lora_target_discovery_excludes_projector_and_freeze_policy(self):
        model = TinyLoraCandidate()
        targets = find_all_linear_names(model)
        self.assertIn("language_model", targets)
        self.assertFalse(any("cut3r_token_projector" in name for name in targets))
        groups = assert_cut3r_token_only_trainable_policy(model)
        self.assertGreater(groups["cut3r_projector"]["parameters"], 0)
        self.assertGreater(groups["llm_lora"]["parameters"], 0)
        self.assertEqual(groups["other_trainables"]["parameters"], 0)
        self.assertFalse(any("cut3r_token_projector" in name and "lora_" in name for name, _ in model.named_parameters()))

    def test_projector_checkpoint_round_trip(self):
        config = {"visual_token_source": "cut3r_only", "cut3r_token_feature_dim": 768}
        projector = Cut3RTokenOnlyProjector(768, 8).eval()
        tokens = torch.randn(1, 729, 768)
        expected = projector(tokens)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/non_lora_trainables.bin"
            torch.save({f"model.cut3r_token_projector.{key}": value for key, value in projector.state_dict().items()}, path)
            saved = torch.load(path, map_location="cpu")
            restored = Cut3RTokenOnlyProjector(config["cut3r_token_feature_dim"], 8).eval()
            state = {key.split("cut3r_token_projector.", 1)[1]: value for key, value in saved.items()}
            self.assertEqual(restored.load_state_dict(state, strict=True).missing_keys, [])
            self.assertTrue(torch.allclose(expected, restored(tokens)))

    def test_training_argument_parser_recognizes_cut3r_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
            model_args, _, training_args = parser.parse_args_into_dataclasses(
                args=[
                    "--visual_token_source", "cut3r_only",
                    "--cut3r_token_sidecar_key", "patch_tokens",
                    "--tune_cut3r_token_projector", "True",
                    "--output_dir", directory,
                ]
            )
        self.assertEqual(model_args.visual_token_source, "cut3r_only")
        self.assertTrue(model_args.tune_cut3r_token_projector)
        self.assertEqual(training_args.output_dir, directory)


if __name__ == "__main__":
    unittest.main()
