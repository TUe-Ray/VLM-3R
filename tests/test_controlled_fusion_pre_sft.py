from types import SimpleNamespace

import torch.nn as nn

from llava.model.c1_structured_isometry import spatialstack_additive_branch
from llava.model.controlled_fusion_pre_sft import (
    CONTROLLED_FUSION_PRE_SFT_SPECS,
    controlled_fusion_artifact_metadata,
    controlled_fusion_spec_for_variant,
)
from scripts.diagnose_layerwise_spatial_hidden_scan import _parse_spatialstack_layers


def test_controlled_fusion_specs_encode_exact_snellius_topologies():
    specs = CONTROLLED_FUSION_PRE_SFT_SPECS
    assert tuple(specs) == ("B", "C", "D", "E", "H")
    assert specs["B"].architecture == "pre_projector_add"
    assert specs["C"].cut3r_source_layers == (12,)
    assert specs["C"].llm_injection_layers == (0,)
    assert specs["D"].fusion_type == "add"
    assert specs["E"].cut3r_source_layers == (12, 12, 12)
    assert specs["E"].llm_injection_layers == (0, 1, 2)
    assert specs["E"].projector_binding == "site_specific"
    assert specs["H"].fusion_type == "cross_attn"
    assert controlled_fusion_spec_for_variant("C1_CONTROLLED_H") is specs["H"]


def test_controlled_fusion_artifact_metadata_forbids_post_sft_state():
    metadata = controlled_fusion_artifact_metadata(CONTROLLED_FUSION_PRE_SFT_SPECS["E"])
    assert metadata["post_sft_checkpoint_loaded"] is False
    assert metadata["cut3r_source_layers"] == [12, 12, 12]
    assert metadata["projector_binding"] == "site_specific"


def test_spatialstack_source_parser_allows_repeats_only_when_explicit():
    assert _parse_spatialstack_layers(
        "12,12,12", default=(6, 9, 12), name="source", allow_repeated=True
    ) == [12, 12, 12]
    try:
        _parse_spatialstack_layers(
            "12,12,12", default=(6, 9, 12), name="source", allow_repeated=False
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Repeated source layers must be opt-in")


def test_site_specific_c1_resolution_keeps_repeated_dec12_projectors_independent():
    branches = nn.ModuleDict({str(layer): nn.Linear(2, 2) for layer in (0, 1, 2)})
    merger = SimpleNamespace(
        projector_binding="site_specific",
        layer_map={0: 12, 1: 12, 2: 12},
        branches=branches,
    )
    assert spatialstack_additive_branch(merger, 0) is branches["0"]
    assert spatialstack_additive_branch(merger, 1) is branches["1"]
    assert spatialstack_additive_branch(merger, 2) is branches["2"]
