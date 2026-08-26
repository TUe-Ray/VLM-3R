#!/usr/bin/env python
"""Verified specifications and runtime overlays for post-SFT geometry probes.

The four archived checkpoint directories migrated to mps-edu-06 contain the
trained adapter/non-LoRA weights, but not their small Hugging Face config
files.  This module reconstructs those files in an experiment-owned runtime
directory without modifying the checkpoint directories.  Every reconstruction
is guarded by an architecture-specific state-dict signature.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_ROOT = Path("/mnt/DATA_SSD/shaoruei/models")
BASE_MODEL = LOCAL_MODEL_ROOT / "base/LLaVA-NeXT-Video-7B-Qwen2"
SIGLIP_MODEL = LOCAL_MODEL_ROOT / "base/siglip-so400m-patch14-384"
CONFIG_TEMPLATE = LOCAL_MODEL_ROOT / "vlm3r_runs/Reproduction_2"
# Keep the four-checkpoint post-SFT comparison contract experiment-local.
POST_SFT_PRE_LLM_FEATURES = ("fusion_output", "projected_features")
POST_SFT_DEPTH_LAYERS = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27)
POST_SFT_DEPTH_FEATURE_LEVELS = POST_SFT_PRE_LLM_FEATURES + tuple(
    f"layer_{layer}" for layer in POST_SFT_DEPTH_LAYERS
)
SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"


@dataclass(frozen=True)
class ProbeModelSpec:
    key: str
    label: str
    checkpoint_name: str
    historical_checkpoint: str
    architecture: str
    source_revision: str
    config_overrides: dict[str, Any]
    required_non_lora_fragments: tuple[str, ...]
    forbidden_non_lora_fragments: tuple[str, ...]
    requires_cut3r_tokens: bool
    requires_full_point_maps: bool
    requires_eomt: bool
    point_map_key: str | None
    token_layout: str

    @property
    def checkpoint(self) -> Path:
        return LOCAL_MODEL_ROOT / self.checkpoint_name


_COMMON_CUT3R = {
    "spatial_tower": "cut3r",
    "mm_spatial_tower": "cut3r",
    "spatial_tower_select_feature": "all_tokens",
    "spatial_tower_select_layer": -1,
    "spatial_feature_dim": 768,
    "spatial_tower_preextracted_only": True,
    "zero_spatial_features": False,
}


MODEL_SPECS: dict[str, ProbeModelSpec] = {
    "eomt_object": ProbeModelSpec(
        key="eomt_object",
        label="eomt_object_tokens_post_sft",
        checkpoint_name="eomt_obj_text_phrase_100p_40403422",
        historical_checkpoint=(
            "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_eomt/"
            "eomt_obj_text_phrase_100p_40403422"
        ),
        architecture=(
            "CUT3R all-token cross-attention followed by EoMT-selected object-token "
            "insertion after each visual block; text-phrase object-info mode"
        ),
        source_revision="984d543/75a4a3e-era EoMT branch",
        config_overrides={
            **_COMMON_CUT3R,
            "fusion_block": "cross_attention",
            "mm_vision_select_layer": -2,
            "mm_eomt_enable_object_block": True,
            "mm_eomt_obj_info_mode": "text_phrase",
            "eomt_pool_top_k": -1,
            "eomt_pool_selection": "class_confidence",
            "eomt_pool_mask_area_threshold": 0.5,
            "eomt_pool_score_threshold": 0.8,
            "mm_eomt_object_block_position": "after_visual",
            "mm_eomt_object_block_max_objects": 8,
            "mm_eomt_object_block_max_per_frame": 2,
            "mm_eomt_selector_mode": "class_aware",
            "mm_eomt_selector_keep_stuff": False,
            "mm_eomt_selector_keep_things": True,
            "mm_eomt_selector_drop_no_object": True,
            "mm_eomt_selector_no_object_class_id": -1,
            "mm_eomt_selector_order": "word_match_then_frame_score",
            "mm_eomt_word_match_enable": True,
            "mm_eomt_word_match_source": "visible_grounded_words",
            "mm_eomt_word_match_mode": "hybrid_safe",
            "mm_eomt_word_match_no_match": "keep_masks",
            "mm_eomt_word_match_similarity_threshold": 0.86,
            "mm_eomt_use_object_type_embedding": False,
            "mm_eomt_obj_info_text": "Object information from the image:",
            "mm_eomt_obj_info_trainable": True,
            "post_sft_probe_architecture": "eomt_object",
        },
        required_non_lora_fragments=(".fusion_block.cross_attention.", ".mm_projector."),
        forbidden_non_lora_fragments=("geometry_aware_projection", "attn_query_proj", "rope_gate_"),
        requires_cut3r_tokens=True,
        requires_full_point_maps=False,
        requires_eomt=True,
        point_map_key=None,
        token_layout=(
            "32 x 196 ordinary frame-aligned visual tokens, with up to 8 EoMT object "
            "tokens inserted after the visual block. Primary probe selects only the "
            "ordinary visual-token indices."
        ),
    ),
    "eomt_selective": ProbeModelSpec(
        key="eomt_selective",
        label="eomt_selective_soft_word_match_zero3d_post_sft",
        checkpoint_name="cut3r_eomt_sel3dr2_wmzero_40416881",
        historical_checkpoint=(
            "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_eomt/"
            "cut3r_eomt_sel3dr2_wmzero_40416881"
        ),
        architecture=(
            "EoMT soft word-matched mask gate on CUT3R patch tokens, zero-3D empty "
            "fallback, followed by ordinary CUT3R all-token cross-attention"
        ),
        source_revision="75a4a3e",
        config_overrides={
            **_COMMON_CUT3R,
            "fusion_block": "cross_attention",
            "mm_vision_select_layer": -2,
            "mm_eomt_selective_3d_enable": True,
            "mm_eomt_selective_3d_gate_type": "soft",
            "mm_eomt_selective_3d_selector_mode": "confidence",
            "mm_eomt_selective_3d_score_threshold": 0.8,
            "mm_eomt_selective_3d_topk": -1,
            "mm_eomt_selective_3d_class_type": "things",
            "mm_eomt_selective_3d_merge_mode": "soft_max_union",
            "mm_eomt_selective_3d_word_match_enable": True,
            "mm_eomt_selective_3d_empty_fallback": "zero_3d",
            "mm_eomt_word_match_source": "visible_grounded_words",
            "mm_eomt_word_match_mode": "hybrid_safe",
            "mm_eomt_word_match_no_match": "keep_masks",
            "mm_eomt_word_match_similarity_threshold": 0.86,
            "post_sft_probe_architecture": "eomt_selective",
        },
        required_non_lora_fragments=(".fusion_block.cross_attention.", ".mm_projector."),
        forbidden_non_lora_fragments=("geometry_aware_projection", "attn_query_proj", "rope_gate_"),
        requires_cut3r_tokens=True,
        requires_full_point_maps=False,
        requires_eomt=True,
        point_map_key=None,
        token_layout=(
            "32 x 196 ordinary frame-aligned visual tokens. EoMT gates CUT3R patch K/V "
            "before fusion and adds no auxiliary tokens to the primary LLM sequence."
        ),
    ),
    "geo_rope_fusion": ProbeModelSpec(
        key="geo_rope_fusion",
        label="geo_rope_cut3r_kv_post_sft",
        checkpoint_name="rope_spherical_100p_40790070",
        historical_checkpoint=(
            "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_RoPE/"
            "rope_spherical_100p_40790070"
        ),
        architecture=(
            "svf_spherical_rope cross-attention: SigLIP 2D queries and CUT3R patch "
            "keys/values, with reference-frame spherical geometry rotating Q/K only"
        ),
        source_revision="71959a8",
        config_overrides={
            **_COMMON_CUT3R,
            "fusion_block": "svf_spherical_rope",
            "mm_vision_select_layer": -1,
            "geometry_rope_mode": "spherical",
            "geometry_rope_max_depth": 10.0,
            "geometry_rope_group_split": "2,1,2",
            "geometry_rope_log_stats": False,
            "geo_rope_fusion_mode": "spherical",
            "geo_rope_fusion_max_depth": 10.0,
            "geo_rope_fusion_group_split": "2,1,2",
            "geo_rope_gate_type": "scalar",
            "geo_rope_point_map_key": "point_maps_ref",
            "geometry_point_map_key": "point_maps_ref",
            "geo_rope_training_point_map_key": "point_maps_ref",
            "post_sft_probe_architecture": "geo_rope_fusion",
        },
        required_non_lora_fragments=(".fusion_block.attn_query_proj.", ".fusion_block.rope_gate_q", ".mm_projector."),
        forbidden_non_lora_fragments=("geometry_aware_projection", ".fusion_block.cross_attention."),
        requires_cut3r_tokens=True,
        requires_full_point_maps=True,
        requires_eomt=False,
        point_map_key="point_maps_ref",
        token_layout=(
            "32 x 196 ordinary frame-aligned visual tokens after geometry-aware "
            "CUT3R K/V cross-attention; geometry and camera tokens are not inserted."
        ),
    ),
    "visual_3d_rope": ProbeModelSpec(
        key="visual_3d_rope",
        label="pure_visual_spherical_3d_rope_post_sft",
        checkpoint_name="RoPE_Spherical_cut3r_100p_41520134",
        historical_checkpoint=(
            "/leonardo_work/EUHPC_D32_006/Train_Model/VLM3R/archived_RoPE/"
            "RoPE_Spherical_cut3r_100p_41520134"
        ),
        architecture=(
            "one-layer MetricGroundedGeometryProjection over SigLIP visual tokens; "
            "reference-frame spherical geometry rotates visual self-attention Q/K only, "
            "with no explicit CUT3R feature fusion"
        ),
        source_revision="75fa39f-era train_3D_RoPE_cut3r.sh",
        config_overrides={
            "fusion_block": None,
            # The LLaVA container still constructs a spatial-tower module;
            # the pure visual-token path does not consume its features, so a
            # sidecar-only CUT3R stub preserves construction without adding a
            # fusion branch.
            "spatial_tower": "cut3r",
            "mm_spatial_tower": "cut3r",
            "spatial_tower_preextracted_only": True,
            "mm_vision_select_layer": -2,
            "use_geometry_aware_projection": True,
            "spatial_encoder_type": "cut3r",
            "geometry_position_mode": "spherical",
            "geo_rope_point_map_key": "point_maps_ref",
            "geometry_point_map_key": "point_maps_ref",
            "geo_rope_training_point_map_key": "point_maps_ref",
            "num_geometry_projection_layers": 1,
            "geometry_projection_num_heads": 16,
            "use_auxiliary_geometry_head": True,
            "use_auxiliary_geometry_loss": True,
            "aux_geometry_targets": "azimuth,elevation,log_distance",
            "lambda_geo": 0.1,
            "geometry_loss_type": "smooth_l1",
            "detach_geometry_targets": True,
            "geometry_gate_init": 0.0,
            "use_geometry_confidence_mask": True,
            "allow_missing_geometry_targets": False,
            "geometry_position_max_abs": 10.0,
            "geometry_fixed_scene_scale": 5.0,
            "geometry_projection_dropout": 0.0,
            "post_sft_probe_architecture": "visual_3d_rope",
        },
        required_non_lora_fragments=(".geometry_aware_projection.layers.0.q_proj.", ".geometry_aware_projection.aux_head.", ".mm_projector."),
        forbidden_non_lora_fragments=(".fusion_block.",),
        requires_cut3r_tokens=False,
        requires_full_point_maps=True,
        requires_eomt=False,
        point_map_key="point_maps_ref",
        token_layout=(
            "32 x 196 ordinary frame-aligned visual tokens after spherical visual-token "
            "self-attention. Point maps supply positions only; no CUT3R/object/camera "
            "tokens enter the LLM sequence."
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state-dict mapping at {path}, got {type(payload).__name__}")
    return payload


def validate_checkpoint_signature(spec: ProbeModelSpec) -> dict[str, Any]:
    checkpoint = spec.checkpoint
    required_files = ("adapter_model.bin", "non_lora_trainables.bin")
    missing_files = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{spec.key} checkpoint is missing {missing_files}: {checkpoint}")

    non_lora = _load_state_dict(checkpoint / "non_lora_trainables.bin")
    adapter = _load_state_dict(checkpoint / "adapter_model.bin")
    keys = tuple(str(key) for key in non_lora)
    missing_fragments = [
        fragment for fragment in spec.required_non_lora_fragments if not any(fragment in key for key in keys)
    ]
    forbidden_hits = {
        fragment: sorted(key for key in keys if fragment in key)
        for fragment in spec.forbidden_non_lora_fragments
        if any(fragment in key for key in keys)
    }
    if missing_fragments or forbidden_hits:
        raise RuntimeError(
            f"{spec.key} architecture signature mismatch: missing={missing_fragments}, "
            f"forbidden_hits={forbidden_hits}"
        )
    lora_keys = [key for key in adapter if "lora_" in str(key)]
    if len(lora_keys) != 392:
        raise RuntimeError(f"{spec.key} expected 392 LoRA tensors, found {len(lora_keys)}")
    return {
        "checkpoint": str(checkpoint),
        "adapter_tensor_count": len(adapter),
        "lora_tensor_count": len(lora_keys),
        "non_lora_tensor_count": len(non_lora),
        "non_lora_keys": sorted(keys),
        "adapter_sha256": sha256_file(checkpoint / "adapter_model.bin"),
        "non_lora_sha256": sha256_file(checkpoint / "non_lora_trainables.bin"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def effective_config(spec: ProbeModelSpec) -> dict[str, Any]:
    config_path = spec.checkpoint / "config.json"
    if config_path.is_file():
        config = _read_json(config_path)
        source = "checkpoint"
    else:
        config = _read_json(CONFIG_TEMPLATE / "config.json")
        source = "reconstructed_from_Reproduction_2_template"
    config.update(deepcopy(spec.config_overrides))
    config.update(
        {
            "mm_vision_tower": str(SIGLIP_MODEL),
            "vision_tower": str(SIGLIP_MODEL),
            "post_sft_probe_config_source": source,
            "post_sft_probe_spec_key": spec.key,
            "post_sft_probe_source_revision": spec.source_revision,
        }
    )
    return config


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_runtime_overlay(spec: ProbeModelSpec, runtime_root: Path) -> Path:
    """Create a repeatable adapter directory while leaving source weights untouched."""
    signature = validate_checkpoint_signature(spec)
    runtime_dir = runtime_root / spec.checkpoint_name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_model.bin", "non_lora_trainables.bin"):
        source = (spec.checkpoint / filename).resolve()
        target = runtime_dir / filename
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

    adapter_config = _read_json(
        spec.checkpoint / "adapter_config.json"
        if (spec.checkpoint / "adapter_config.json").is_file()
        else CONFIG_TEMPLATE / "adapter_config.json"
    )
    adapter_config["base_model_name_or_path"] = str(BASE_MODEL)
    adapter_config["inference_mode"] = True
    generation_config_path = spec.checkpoint / "generation_config.json"
    if not generation_config_path.is_file():
        generation_config_path = CONFIG_TEMPLATE / "generation_config.json"

    config = effective_config(spec)
    _atomic_json(runtime_dir / "config.json", config)
    _atomic_json(runtime_dir / "adapter_config.json", adapter_config)
    _atomic_json(runtime_dir / "generation_config.json", _read_json(generation_config_path))
    _atomic_json(
        runtime_dir / "post_sft_probe_reconstruction.json",
        {
            "schema_version": "post_sft_geometry_checkpoint_reconstruction_v1",
            "spec": asdict(spec),
            "source_checkpoint": str(spec.checkpoint),
            "historical_checkpoint": spec.historical_checkpoint,
            "weights_are_symlinked_read_only_sources": True,
            "config_was_present_in_source": (spec.checkpoint / "config.json").is_file(),
            "adapter_config_was_present_in_source": (spec.checkpoint / "adapter_config.json").is_file(),
            "signature": signature,
            "effective_config_sha256": sha256_file(runtime_dir / "config.json"),
            "post_sft_probe_layers": list(POST_SFT_DEPTH_LAYERS),
            "post_sft_pre_llm_features": list(POST_SFT_PRE_LLM_FEATURES),
            "primary_probe_tokens": "ordinary frame-aligned visual tokens only",
        },
    )
    return runtime_dir


def iter_specs(keys: Iterable[str] | None = None) -> Iterable[ProbeModelSpec]:
    if keys is None:
        yield from MODEL_SPECS.values()
        return
    for key in keys:
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model key {key!r}; choose from {sorted(MODEL_SPECS)}")
        yield MODEL_SPECS[key]
