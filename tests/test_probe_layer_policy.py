from scripts.probing.depth_probe_common import LLM_LAYERS, SPATIALSTACK_LLM_LAYERS
from scripts.probing.post_sft_geometry_probe_specs import (
    POST_SFT_DEPTH_FEATURE_LEVELS,
    POST_SFT_DEPTH_LAYERS,
    POST_SFT_PRE_LLM_FEATURES,
)
from scripts.probing.probe_layer_policy import (
    COMMON_FULL_FEATURE_LEVELS,
    COMMON_PRE_LLM_FEATURES,
    COMMON_PROBE_LAYERS,
)


def test_repository_wide_full_probe_layers_are_complete_and_ordered():
    expected = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    assert COMMON_PROBE_LAYERS == expected
    assert tuple(LLM_LAYERS) == expected
    assert tuple(SPATIALSTACK_LLM_LAYERS) == expected


def test_post_sft_geometry_probe_layers_are_wrapper_local_and_ordered():
    expected = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    assert POST_SFT_DEPTH_LAYERS == expected
    assert POST_SFT_PRE_LLM_FEATURES == ("fusion_output", "projected_features")
    assert POST_SFT_DEPTH_FEATURE_LEVELS == POST_SFT_PRE_LLM_FEATURES + tuple(
        f"layer_{layer}" for layer in expected
    )


def test_canonical_full_probe_includes_pre_llm_features():
    assert COMMON_PRE_LLM_FEATURES == ("siglip_output", "projected_features")
    assert COMMON_FULL_FEATURE_LEVELS[:2] == COMMON_PRE_LLM_FEATURES
    assert tuple(COMMON_FULL_FEATURE_LEVELS[2:]) == tuple(f"layer_{layer}" for layer in COMMON_PROBE_LAYERS)
