"""Small frozen EoMT runtime adapter used by post-SFT probing."""

from .eomt_extractor import EoMTExtractor
from .cache_consumers import (
    configure_selective_kv_gate,
    gate_cut3r_patch_tokens,
    object_tokens_from_cached,
    validate_cached_outputs,
)

__all__ = [
    "EoMTExtractor",
    "configure_selective_kv_gate",
    "gate_cut3r_patch_tokens",
    "object_tokens_from_cached",
    "validate_cached_outputs",
]
