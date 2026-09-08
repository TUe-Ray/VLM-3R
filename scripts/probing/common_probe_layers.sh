#!/usr/bin/env bash
# Shared full-probe policy for new models.
#
# Missing-layer completion wrappers intentionally do not source this file:
# they are allowed to request only the layers that are absent.

COMMON_PROBE_LAYERS=(0 1 2 3 6 9 12 15 18 21 24 27)
COMMON_PROBE_LAYERS_SPACE="${COMMON_PROBE_LAYERS[*]}"
COMMON_PROBE_LAYER_LEVELS_CSV="layer_0,layer_1,layer_2,layer_3,layer_6,layer_9,layer_12,layer_15,layer_18,layer_21,layer_24,layer_27"
COMMON_FULL_FEATURE_LEVELS_CSV="siglip_output,projected_features,${COMMON_PROBE_LAYER_LEVELS_CSV}"

# New pre-SFT probes additionally retain the fusion-block representation.
PRE_SFT_PRE_LLM_FEATURES_CSV="siglip_output,fusion_output,projected_features"
PRE_SFT_FULL_FEATURE_LEVELS_CSV="${PRE_SFT_PRE_LLM_FEATURES_CSV},${COMMON_PROBE_LAYER_LEVELS_CSV}"
