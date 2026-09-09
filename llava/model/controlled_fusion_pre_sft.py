"""Exact pre-SFT topology definitions for controlled fusion B/C/D/E/H.

These definitions describe architecture only.  They never identify or load a
post-SFT checkpoint.  Keeping the topology in one importable table prevents
the C1 calibrator, proxy runner, and feature extractor from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledFusionPreSFTSpec:
    identifier: str
    pre_sft_variant: str
    architecture: str
    cut3r_source_layers: tuple[int, ...]
    llm_injection_layers: tuple[int, ...]
    fusion_type: str
    projector_binding: str
    display_name: str


CONTROLLED_FUSION_PRE_SFT_SPECS = {
    "B": ControlledFusionPreSFTSpec(
        identifier="B",
        pre_sft_variant="c1_controlled_b",
        architecture="pre_projector_add",
        cut3r_source_layers=(12,),
        llm_injection_layers=(),
        fusion_type="pre_projector_add",
        projector_binding="source_specific",
        display_name="B: pre-projector add, CUT3R dec12",
    ),
    "C": ControlledFusionPreSFTSpec(
        identifier="C",
        pre_sft_variant="c1_controlled_c",
        architecture="spatialstack_cross_attn_v1",
        cut3r_source_layers=(12,),
        llm_injection_layers=(0,),
        fusion_type="cross_attn",
        projector_binding="source_specific",
        display_name="C: cross-attention, CUT3R dec12 -> L0",
    ),
    "D": ControlledFusionPreSFTSpec(
        identifier="D",
        pre_sft_variant="c1_controlled_d",
        architecture="spatialstack_add",
        cut3r_source_layers=(12,),
        llm_injection_layers=(0,),
        fusion_type="add",
        projector_binding="source_specific",
        display_name="D: additive, CUT3R dec12 -> L0",
    ),
    "E": ControlledFusionPreSFTSpec(
        identifier="E",
        pre_sft_variant="c1_controlled_e",
        architecture="spatialstack_add",
        cut3r_source_layers=(12, 12, 12),
        llm_injection_layers=(0, 1, 2),
        fusion_type="add",
        projector_binding="site_specific",
        display_name="E: additive, CUT3R dec12 -> L0/L1/L2, site projectors",
    ),
    "H": ControlledFusionPreSFTSpec(
        identifier="H",
        pre_sft_variant="c1_controlled_h",
        architecture="spatialstack_cross_attn_v1",
        cut3r_source_layers=(12, 12, 12),
        llm_injection_layers=(0, 1, 2),
        fusion_type="cross_attn",
        projector_binding="source_specific",
        display_name="H: cross-attention, CUT3R dec12 -> L0/L1/L2",
    ),
}

CONTROLLED_FUSION_BY_PRE_SFT_VARIANT = {
    spec.pre_sft_variant: spec for spec in CONTROLLED_FUSION_PRE_SFT_SPECS.values()
}


def controlled_fusion_spec(identifier: str) -> ControlledFusionPreSFTSpec:
    key = str(identifier).strip().upper()
    try:
        return CONTROLLED_FUSION_PRE_SFT_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported controlled-fusion ID {identifier!r}; expected B/C/D/E/H."
        ) from exc


def controlled_fusion_spec_for_variant(variant: str) -> ControlledFusionPreSFTSpec | None:
    return CONTROLLED_FUSION_BY_PRE_SFT_VARIANT.get(str(variant).strip().lower())


def controlled_fusion_artifact_metadata(
    spec: ControlledFusionPreSFTSpec,
) -> dict[str, object]:
    return {
        "id": spec.identifier,
        "pre_sft_variant": spec.pre_sft_variant,
        "architecture": spec.architecture,
        "cut3r_source_layers": list(spec.cut3r_source_layers),
        "llm_injection_layers": list(spec.llm_injection_layers),
        "fusion_type": spec.fusion_type,
        "projector_binding": spec.projector_binding,
        "post_sft_checkpoint_loaded": False,
    }
