from types import SimpleNamespace

import torch
import torch.nn as nn

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.multimodal_fusion_block.builder import PreProjectorAddFusion


class TinyVisionTower(nn.Module):
    def forward(self, images, return_raw_features=False):
        features = torch.ones(images.shape[0], 4, 4, dtype=images.dtype)
        return (features, features) if return_raw_features else features


class TinyCut3RSpatialTower(nn.Module):
    is_loaded = True


class RecordingProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, features):
        self.input = features.detach().clone()
        return torch.cat([features, features[..., :2]], dim=-1)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            spatial_tower="cut3r",
            spatial_tower_preextracted_only=True,
            fusion_block="pre_projector_add",
            pre_projector_add_source_layer=12,
            cut3r_spatialstack_feature_key="cut3r_dec_layers",
            zero_spatial_features=False,
        )
        self.vision_tower = TinyVisionTower()
        self.spatial_tower = TinyCut3RSpatialTower()
        self.fusion_block = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=False)
        self.mm_projector = RecordingProjector()

    def get_vision_tower(self):
        return self.vision_tower

    def get_spatial_tower(self):
        return self.spatial_tower

    def get_fusion_block(self):
        return self.fusion_block


class EncodeHarness:
    encode_images = LlavaMetaForCausalLM.encode_images

    def __init__(self):
        self._base = TinyBase()
        self.config = self._base.config

    def get_model(self):
        return self._base


def test_pre_projector_add_aligns_dec12_then_calls_mm_projector():
    torch.manual_seed(5)
    harness = EncodeHarness()
    dec12 = torch.randn(1, 9, 3)
    sidecar = {"cut3r_dec_layers": {"12": {"patch_tokens": dec12}}}
    output = harness.encode_images(torch.zeros(1, 3, 4, 4), spatial_features=[sidecar])

    assert output.shape == (1, 4, 6)
    assert torch.isfinite(output).all()
    assert harness._base.mm_projector.input.shape == (1, 4, 4)
    assert not torch.equal(harness._base.mm_projector.input, torch.ones(1, 4, 4))
    metrics = harness._base._last_pre_projector_add_metrics
    assert metrics["fusion_stage"] == "pre_mm_projector"
    assert metrics["cut3r_source_layer"] == 12
    assert metrics["raw_spatial_shape"] == [1, 9, 3]
    assert metrics["aligned_spatial_shape"] == [1, 4, 3]
    assert metrics["mm_projector_input_shape"] == [1, 4, 4]
    assert metrics["mm_projector_output_shape"] == [1, 4, 6]


def test_pre_projector_add_zero_init_is_identity_in_vision_space():
    fusion = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=True)
    clip = torch.randn(2, 4, 4)
    spatial = torch.randn(2, 9, 3)
    assert torch.equal(fusion(clip, spatial), clip)


def test_pre_projector_add_c1_scalars_preserve_native_default_and_emit_diagnostics():
    fusion = PreProjectorAddFusion(4, 3, source_layer=12, zero_init=False)
    clip = torch.randn(2, 4, 4)
    spatial = torch.randn(2, 9, 3)
    native = fusion(clip, spatial)
    fusion.set_c1_state(
        enabled=True,
        pre_gelu_scale=0.75,
        residual_gain=0.0,
        collect_diagnostics=True,
    )
    assert torch.equal(fusion(clip, spatial), clip)
    assert not torch.equal(native, clip)
    assert set(fusion._c1_last_diagnostics) == {
        "clip",
        "z_pre_raw",
        "z_pre",
        "delta_raw",
        "delta",
    }
    assert fusion._c1_last_diagnostics["delta"]["sum_sq"] == 0.0
