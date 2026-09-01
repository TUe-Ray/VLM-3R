#!/usr/bin/env python
"""Canonical layer policy for complete new-model depth probes.

Missing-layer completion jobs may intentionally pass a subset.  New-model
full probes must use ``COMMON_PROBE_LAYERS`` plus the common pre-LLM features.
New pre-SFT probes have the stricter ``PRE_SFT_*`` contract below.
"""

from __future__ import annotations


COMMON_PROBE_LAYERS = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
COMMON_PRE_LLM_FEATURES = ("siglip_output", "projected_features")
COMMON_PROBE_LAYER_LEVELS = tuple(f"layer_{layer}" for layer in COMMON_PROBE_LAYERS)
COMMON_FULL_FEATURE_LEVELS = COMMON_PRE_LLM_FEATURES + COMMON_PROBE_LAYER_LEVELS

# Required representation for every new pre-SFT depth probe.  Keep this as a
# separate policy because historical caches and intentional missing-layer
# completion jobs are allowed to remain partial.
PRE_SFT_PRE_LLM_FEATURES = ("siglip_output", "fusion_output", "projected_features")
PRE_SFT_FULL_FEATURE_LEVELS = PRE_SFT_PRE_LLM_FEATURES + COMMON_PROBE_LAYER_LEVELS


def common_layers_space() -> str:
    """Return the canonical layer list in shell-friendly space-separated form."""

    return " ".join(str(layer) for layer in COMMON_PROBE_LAYERS)


def common_layer_levels_csv() -> str:
    """Return the canonical LLM layer feature levels as CSV."""

    return ",".join(COMMON_PROBE_LAYER_LEVELS)


def common_full_feature_levels_csv() -> str:
    """Return pre-LLM features followed by every canonical LLM layer."""

    return ",".join(COMMON_FULL_FEATURE_LEVELS)


def pre_sft_full_feature_levels_csv() -> str:
    """Return the mandatory complete feature list for new pre-SFT probes."""

    return ",".join(PRE_SFT_FULL_FEATURE_LEVELS)
