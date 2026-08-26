"""CPU-only contracts for the common post-SFT probing feature set."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO_ROOT / "scripts" / "probing"
if str(PROBING_DIR) not in sys.path:
    sys.path.insert(0, str(PROBING_DIR))

from extract_depth_probe_features import (  # noqa: E402
    assert_first_post_sft_geometry_runtime,
    register_pre_llm_hooks,
    save_frame_outputs,
)
from scripts.probing.post_sft_geometry_probe_specs import (  # noqa: E402
    MODEL_SPECS,
    POST_SFT_DEPTH_LAYERS,
    effective_config,
)


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mm_projector = torch.nn.Linear(3, 4, bias=False)

    def get_fusion_block(self):
        raise AssertionError("pure visual 3D-RoPE must not use a fusion block")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _Base()

    def get_model(self):
        return self.base


class _Fusion(torch.nn.Module):
    def forward(self, query: torch.Tensor, _key_value: torch.Tensor):
        return query + 1, None


class _ExplicitBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = _Fusion()
        self.mm_projector = torch.nn.Linear(3, 4, bias=False)

    def get_fusion_block(self):
        return self.fusion


class _ExplicitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _ExplicitBase()

    def get_model(self):
        return self.base


def test_all_post_sft_configs_expose_requested_decoder_layers() -> None:
    assert POST_SFT_DEPTH_LAYERS[-1] == 27
    for spec in MODEL_SPECS.values():
        assert int(effective_config(spec)["num_hidden_layers"]) == 28


def test_pure_visual_fusion_output_is_projector_input() -> None:
    model = _Model()
    captured: dict[str, torch.Tensor] = {}
    handles = register_pre_llm_hooks(
        model,
        "visual_3d_rope",
        ["fusion_output", "projected_features"],
        captured,
        post_sft_architecture="visual_3d_rope",
    )
    value = torch.randn(2, 196, 3)
    try:
        projected = model.get_model().mm_projector(value)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(captured["fusion_output"], value)
    assert torch.equal(captured["projected_features"], projected)


def test_explicit_fusion_output_and_projector_output_are_both_captured() -> None:
    model = _ExplicitModel()
    captured: dict[str, torch.Tensor] = {}
    handles = register_pre_llm_hooks(
        model,
        "geo_rope_fusion",
        ["fusion_output", "projected_features"],
        captured,
        post_sft_architecture="geo_rope_fusion",
    )
    value = torch.randn(2, 196, 3)
    key_value = torch.randn(2, 8, 3)
    try:
        fused, _ = model.get_model().get_fusion_block()(value, key_value)
        projected = model.get_model().mm_projector(fused)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(captured["fusion_output"], fused)
    assert torch.equal(captured["projected_features"], projected)


def test_post_sft_runtime_contract_accepts_complete_14_level_layout() -> None:
    visual_tokens = 32 * 196
    report = assert_first_post_sft_geometry_runtime(
        architecture="geo_rope_fusion",
        hidden_states=[torch.empty((), device="meta") for _ in range(29)],
        inputs_embeds=torch.empty((1, visual_tokens + 10, 4), device="meta"),
        metadata={
            "visual_token_indices": torch.arange(visual_tokens),
            "visual_frame_ids": torch.arange(32).repeat_interleave(196),
        },
        selected_frames=[7, 22],
        model_forward_inputs={
            "spatial_features": True,
            "point_maps": False,
            "geometry_spatial_features": True,
        },
        geometry_point_map_shape=[32, 432, 432, 3],
        normalized_pre_llm={
            "fusion_output": torch.empty((32, 196, 1152), device="meta"),
            "projected_features": torch.empty((32, 196, 3584), device="meta"),
        },
        requested_pre_llm_features=["fusion_output", "projected_features"],
    )
    assert report["assessment"] == "PASS"
    assert report["normalized_pre_llm_shapes"]["fusion_output"] == [32, 196, 1152]
    assert report["normalized_pre_llm_shapes"]["projected_features"] == [32, 196, 3584]


def test_additive_resume_preserves_existing_features_and_refreshes_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fsid = "sample_f0007"
        existing = root / "features" / "geo_rope_fusion" / "layer_0" / f"frame_{fsid}.pt"
        existing.parent.mkdir(parents=True)
        original = torch.full((2, 2, 2), 7.0)
        torch.save(original, existing)
        (existing.parent / "provenance.json").write_text(
            '{"requested_llm_layers": [0]}\n', encoding="utf-8"
        )

        provenance = {
            "sample_indices_sha256": "split",
            "requested_llm_layers": [0, 12],
            "requested_pre_llm_features": ["fusion_output", "projected_features"],
        }
        save_frame_outputs(
            output_root=root,
            model_label="geo_rope_fusion",
            frame_record={"frame_sample_id": fsid},
            llm_features={
                "layer_0": torch.zeros((2, 2, 2)),
                "layer_12": torch.ones((2, 2, 2)),
            },
            pre_llm_features={
                "fusion_output": torch.ones((2, 2, 2)),
                "projected_features": torch.ones((2, 2, 2)),
            },
            gt_depth=torch.ones((2, 2)),
            gt_valid=torch.ones((2, 2), dtype=torch.bool),
            metadata={},
            feature_provenance=provenance,
            cache_dtype=torch.float16,
        )

        try:
            preserved = torch.load(existing, weights_only=True)
        except TypeError:
            preserved = torch.load(existing)
        assert torch.equal(preserved, original)
        assert (root / "features" / "geo_rope_fusion" / "layer_12" / f"frame_{fsid}.pt").is_file()
        refreshed = __import__("json").loads(
            (existing.parent / "provenance.json").read_text(encoding="utf-8")
        )
        assert refreshed["requested_llm_layers"] == [0, 12]
