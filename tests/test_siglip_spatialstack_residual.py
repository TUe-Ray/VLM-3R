"""CPU-level regression tests for the lightweight residual path."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from llava.model.llava_arch import LlavaMetaForCausalLM, pool_2d_visual_features
from llava.model.siglip_spatialstack_residual import (
    PredictedSpatialStackResidualAdapter,
    TokenWiseResidualPredictor,
    build_residual_predictor,
    load_residual_predictor_checkpoint,
    predictor_checkpoint_payload,
)


def _metadata(frames: int):
    visual_indices = []
    frame_ids = []
    newline_indices = []
    for frame in range(frames):
        base = frame * 210
        for row in range(14):
            visual_indices.extend(base + row * 15 + col for col in range(14))
            frame_ids.extend([frame] * 14)
            newline_indices.append(base + row * 15 + 14)
    return [{
        "visual_token_indices": torch.tensor(visual_indices),
        "visual_frame_ids": torch.tensor(frame_ids),
        "frame_order": list(range(frames)),
        "newline_token_indices": torch.tensor(newline_indices),
    }]


def test_token_predictor_and_visual_only_residual_injection():
    predictor = TokenWiseResidualPredictor(hidden_size=8, bottleneck_dim=4)
    tokens = torch.randn(1, 2, 196, 8)
    output = predictor(tokens)
    assert set(output) == {6, 9, 12}
    assert all(value.shape == tokens.shape for value in output.values())

    adapter = PredictedSpatialStackResidualAdapter(
        predictor, source_layers=(6, 9, 12), llm_layers=(0, 1, 2)
    )
    embeds = torch.randn(1, 420, 8)
    residuals = adapter(embeds, _metadata(frames=2))
    visual = _metadata(2)[0]["visual_token_indices"]
    non_visual = torch.ones(420, dtype=torch.bool)
    non_visual[visual] = False
    assert all(torch.count_nonzero(value[0, non_visual]) == 0 for value in residuals.values())
    assert adapter.last_debug["cut3r_called"] is False

    adapter.configure(control="zero")
    assert all(torch.count_nonzero(value) == 0 for value in adapter(embeds, _metadata(2)).values())


def test_temporal_predictor_has_same_location_outputs_and_mask():
    predictor = build_residual_predictor(
        "temporal",
        hidden_size=8,
        bottleneck_dim=4,
        temporal_hidden_dim=8,
        temporal_num_layers=2,
        temporal_num_heads=2,
        temporal_ffn_dim=16,
        temporal_max_frames=4,
    )
    tokens = torch.randn(2, 3, 196, 8)
    output = predictor(tokens, torch.tensor([[True, True, False], [True, False, False]]))
    assert all(value.shape == tokens.shape for value in output.values())
    assert all(torch.isfinite(value).all() for value in output.values())


def test_predictor_small_subset_loss_decreases():
    torch.manual_seed(1)
    predictor = TokenWiseResidualPredictor(hidden_size=8, bottleneck_dim=8)
    tokens = torch.randn(1, 1, 196, 8)
    target = {layer: tokens * (0.15 + layer / 100.0) for layer in (6, 9, 12)}
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-2)

    def loss_value():
        prediction = predictor(tokens)
        return sum((prediction[layer] - target[layer]).square().mean() for layer in target)

    initial = float(loss_value().detach())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    assert float(loss_value().detach()) < initial


def test_shared_pool_matches_pre_refactor_oracle_semantics():
    class _Vision:
        num_patches_per_side = 27

    class _Owner:
        config = type("Config", (), {"mm_spatial_pool_mode": "bilinear"})()

        def get_vision_tower(self):
            return _Vision()

    def legacy_pool(image_feature, stride=2):
        # This is the pre-refactor get_2dPool behavior, kept as a compact oracle.
        height = width = 27
        frames, tokens, hidden = image_feature.shape
        prefix = image_feature[:, : tokens - height * width] if tokens > height * width else None
        image_feature = image_feature[:, tokens - height * width:].view(frames, height, width, hidden)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        image_feature = torch.nn.functional.interpolate(
            image_feature, size=(14, 14), mode="bilinear"
        )
        image_feature = image_feature.permute(0, 2, 3, 1).view(frames, -1, hidden)
        return torch.cat((prefix, image_feature), dim=1) if prefix is not None else image_feature

    features = torch.randn(2, 730, 8)
    expected = legacy_pool(features)
    torch.testing.assert_close(
        LlavaMetaForCausalLM.get_2dPool(_Owner(), features), expected
    )
    torch.testing.assert_close(
        pool_2d_visual_features(features, num_patches_per_side=27, pool_mode="bilinear"), expected
    )


def test_predictor_checkpoint_roundtrip(tmp_path: Path):
    predictor = TokenWiseResidualPredictor(hidden_size=8, bottleneck_dim=4)
    path = tmp_path / "predictor.pt"
    torch.save(predictor_checkpoint_payload(predictor, teacher_checkpoint="teacher"), path)
    restored, checkpoint = load_residual_predictor_checkpoint(
        path, expected_type="token_mlp", expected_hidden_size=8, expected_source_layers=(6, 9, 12)
    )
    tokens = torch.randn(1, 1, 196, 8)
    assert checkpoint["teacher_checkpoint"] == "teacher"
    for layer in (6, 9, 12):
        torch.testing.assert_close(predictor(tokens)[layer], restored(tokens)[layer])


def _trainer_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "residual_trainer", root / "scripts/train/train_siglip_to_spatialstack_residual.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_bare_cache_pairing_and_all_valid_default(tmp_path: Path):
    trainer = _trainer_module()
    siglip_root = tmp_path / "siglip"
    cut3r_root = tmp_path / "cut3r"
    for root in (siglip_root, cut3r_root / "l6", cut3r_root / "l9", cut3r_root / "l12"):
        root.mkdir(parents=True, exist_ok=True)
    key = "nested/sample.pt"
    for root, tensor in (
        (siglip_root, torch.randn(2, 729, 1152)),
        (cut3r_root / "l6", torch.randn(2, 729, 768)),
        (cut3r_root / "l9", torch.randn(2, 729, 768)),
        (cut3r_root / "l12", torch.randn(2, 729, 768)),
    ):
        (root / "nested").mkdir(exist_ok=True)
        torch.save(tensor, root / key)
        torch.save(tensor.clone(), root / "nested/second.pt")
    # Add one incomplete key: it must not be paired or required.
    torch.save(torch.randn(2, 729, 1152), siglip_root / "unpaired.pt")
    cache = trainer.PairedResidualCache(
        str(siglip_root), str(cut3r_root), {6: "l6", 9: "l9", 12: "l12"},
        validation_fraction=0.5, split_seed=7,
    )
    sample = cache.load(key)
    assert sample["valid_mask"].tolist() == [True, True]


def test_legacy_cache_frame_count_is_checked(tmp_path: Path):
    trainer = _trainer_module()
    siglip_root = tmp_path / "siglip"
    cut3r_root = tmp_path / "cut3r"
    for root in (siglip_root, cut3r_root / "l6", cut3r_root / "l9", cut3r_root / "l12"):
        root.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(2, 729, 1152), siglip_root / "sample.pt")
    for subdir, frames in (("l6", 2), ("l9", 1), ("l12", 2)):
        torch.save(torch.randn(frames, 729, 768), cut3r_root / subdir / "sample.pt")
    cache = trainer.PairedResidualCache(
        str(siglip_root), str(cut3r_root), {6: "l6", 9: "l9", 12: "l12"},
        train_keys={"sample.pt"}, validation_keys={"sample.pt"},
    )
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "Frame-count mismatch"):
        cache.load("sample.pt")


if __name__ == "__main__":
    test_token_predictor_and_visual_only_residual_injection()
    test_temporal_predictor_has_same_location_outputs_and_mask()
    test_predictor_small_subset_loss_decreases()
    test_shared_pool_matches_pre_refactor_oracle_semantics()
    with tempfile.TemporaryDirectory() as directory:
        test_predictor_checkpoint_roundtrip(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_legacy_bare_cache_pairing_and_all_valid_default(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_legacy_cache_frame_count_is_checked(Path(directory))
    print("siglip_spatialstack_residual CPU checks passed")
