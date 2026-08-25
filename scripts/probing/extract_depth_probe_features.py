#!/usr/bin/env python
"""Extract cached frame-level features for the VLM-3R depth probing experiment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from depth_probe_common import (
    DEFAULT_DATA_YAML,
    DEFAULT_FAST_FEATURE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POINT_MAPS_SUBDIR,
    DEFAULT_SPATIAL_FEATURES_SUBDIR,
    FEATURE_PRESETS,
    MODEL_PRESETS,
    coerce_cache_dtype,
    depth_from_point_maps,
    downsample_depth_to_grid,
    feature_preset_for_model,
    frame_depth_metadata,
    grid_shape_for_frame,
    hidden_state_for_layer,
    layer_feature_path,
    llm_layers_for_model,
    load_frame_records,
    load_point_map_sidecar,
    parse_feature_names,
    parse_llm_layers,
    pre_llm_features_for_model,
    read_json,
    reshape_tokens_to_grid,
    resolve_sidecar_path,
    select_point_maps,
    torch_dtype_from_name,
    validate_llm_layers,
)
from local_depth_probe_cache import (
    assert_baseline_or_zero_spatial_forward_contract,
    assert_pre_sft_base_vlm_forward_contract,
    install_forward_frame_loader,
    load_selected_camera_depths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_layerwise_spatial_hidden_scan import load_model, make_data_args, move_to_device  # noqa: E402
from llava.model.c1_structured_isometry import apply_c1_calibration_artifact  # noqa: E402


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unavailable"

    status = run("git", "status", "--short")
    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_worktree_dirty": bool(status and status != "unavailable"),
    }


def model_placement_metadata(model: torch.nn.Module) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        values = [str(value) for value in device_map.values()]
        cpu_keys = sorted(str(key) for key, value in device_map.items() if str(value) in {"cpu", "disk"})
        gpu_keys = sorted(str(key) for key, value in device_map.items() if str(value).startswith("cuda") or str(value).isdigit())
    else:
        values, cpu_keys, gpu_keys = [], [], []
    backend = (
        getattr(model.config, "_attn_implementation", None)
        or getattr(model.config, "_attn_implementation_internal", None)
        or getattr(model.config, "attn_implementation", None)
    )
    effective_device_map = dict(device_map) if isinstance(device_map, dict) else device_map
    effective_vision_placement = getattr(model, "_pre_sft_vision_placement", None)
    if isinstance(effective_device_map, dict) and isinstance(effective_vision_placement, dict):
        effective_device = effective_vision_placement.get("vision_tower_effective_device")
        if effective_device:
            effective_device_map["model.vision_tower"] = effective_device
    return {
        "hf_device_map": device_map,
        "effective_hf_device_map": effective_device_map,
        "effective_placement_policy": getattr(model, "_pre_sft_placement_policy", None),
        "effective_vision_placement": effective_vision_placement,
        "cpu_offload_used": bool(cpu_keys),
        "cpu_or_disk_modules": cpu_keys,
        "gpu_modules": gpu_keys,
        "placement_values": sorted(set(values)),
        "attention_backend": backend,
    }


def module_dtype_metadata(module: Any) -> dict[str, Any]:
    """Describe materialized module parameter storage without inferring compute dtype."""
    if module is None or not hasattr(module, "parameters"):
        return {"parameter_dtypes": [], "parameter_devices": [], "meta_parameter_count": 0}
    parameters = list(module.parameters())
    materialized = [parameter for parameter in parameters if not parameter.is_meta]
    return {
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in materialized}),
        "parameter_devices": sorted({str(parameter.device) for parameter in materialized}),
        "meta_parameter_count": sum(parameter.is_meta for parameter in parameters),
    }


def dtype_name(value: Any) -> str | None:
    return str(value.dtype) if isinstance(value, torch.Tensor) else None


def summarize_runtime_dtypes(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Summarize observed forward dtypes; never substitute CLI configuration."""
    observed: dict[str, set[str]] = {}
    for sample in samples:
        for name, dtype in sample.get("runtime_dtypes", {}).items():
            if dtype is not None:
                observed.setdefault(name, set()).add(str(dtype))
    return {name: sorted(values) for name, values in sorted(observed.items())}


def base_module_device(module: Any, fallback: torch.device) -> torch.device:
    declared = getattr(module, "device", None)
    if isinstance(declared, torch.device) and declared.type != "meta":
        return declared
    for parameter in module.parameters():
        if not parameter.is_meta:
            return parameter.device
    return fallback


def base_move_value(value: Any, target: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        return value.to(device=target, dtype=dtype if dtype is not None and value.is_floating_point() else None)
    if isinstance(value, list):
        return [base_move_value(item, target, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(base_move_value(item, target, dtype) for item in value)
    return value


def json_ready_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu()
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def capture_hook(name: str, captured: dict[str, torch.Tensor]):
    def _hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if isinstance(item, torch.Tensor)]
            if len(tensors) != len(value):
                raise TypeError(f"{name} hook expected tensor/list of tensors, got {type(output)}")
            value = torch.cat(tensors, dim=0)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} hook expected tensor output, got {type(value)}")
        captured[name] = value.detach().cpu()

    return _hook


def capture_input_hook(name: str, captured: dict[str, torch.Tensor]):
    """Capture the first tensor entering a module, for ordering attestations."""

    def _hook(_module, inputs):
        if not inputs:
            raise TypeError(f"{name} hook received no positional inputs")
        value = inputs[0]
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if isinstance(item, torch.Tensor)]
            if len(tensors) != len(value):
                raise TypeError(f"{name} hook expected tensor/list of tensors, got {type(value)}")
            value = torch.cat(tensors, dim=0)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} hook expected tensor input, got {type(value)}")
        captured[name] = value.detach().cpu()

    return _hook


def register_pre_llm_hooks(
    model: torch.nn.Module,
    model_label: str,
    feature_names: list[str],
    captured: dict[str, torch.Tensor],
):
    if not feature_names:
        return []
    base = model.get_model()
    handles = []
    if "fusion_output" in feature_names:
        get_fusion_block = getattr(base, "get_fusion_block", None)
        if not callable(get_fusion_block):
            raise RuntimeError(f"{model_label} requested fusion_output, but base model has no get_fusion_block().")
        fusion_block = get_fusion_block()
        if fusion_block is None:
            # Additive SpatialStack injects residuals only inside the LLM,
            # so it has no pre-projector fusion module.  Its comparable
            # fusion_output is the raw visual tensor entering mm_projector.
            projector = getattr(base, "mm_projector", None)
            if projector is None:
                raise RuntimeError(
                    f"{model_label} requested fusion_output, but has neither a fusion block nor mm_projector."
                )
            handles.append(projector.register_forward_pre_hook(capture_input_hook("fusion_output", captured)))
        else:
            handles.append(fusion_block.register_forward_hook(capture_hook("fusion_output", captured)))
    if "projected_features" in feature_names:
        projector = getattr(base, "mm_projector", None)
        if projector is None:
            raise RuntimeError(f"{model_label} requested projected_features, but base model has no mm_projector.")
        if model_label == "zero_spatial":
            # In the verified zero-spatial cross-attention path, the fusion
            # block runs immediately before mm_projector.  Keep transient
            # captures so the saved projected feature is auditable as the
            # post-fusion projector output.
            get_fusion_block = getattr(base, "get_fusion_block", None)
            fusion_block = get_fusion_block() if callable(get_fusion_block) else None
            if fusion_block is None:
                raise RuntimeError(
                    "zero_spatial projected_features requires a configured fusion block "
                    "to establish the post-fusion projector contract."
                )
            handles.append(
                fusion_block.register_forward_hook(
                    capture_hook("_zero_spatial_fusion_output", captured)
                )
            )
            handles.append(
                projector.register_forward_pre_hook(
                    capture_input_hook("_zero_spatial_projector_input", captured)
                )
            )
        handles.append(projector.register_forward_hook(capture_hook("projected_features", captured)))
    if "siglip_output" in feature_names:
        if model_label not in {"pre_sft_base_vlm", "zero_spatial"}:
            raise RuntimeError(
                "siglip_output is defined only for pre_sft_base_vlm and zero_spatial conditions."
            )
        vision_tower = getattr(model, "get_vision_tower", lambda: None)()
        if vision_tower is None:
            raise RuntimeError(f"{model_label} requested siglip_output, but no vision tower is present.")
        # This hook is intentionally on the repo SigLipVisionTower wrapper,
        # not on its nested Hugging Face SigLIP module.  For zero_spatial it
        # captures the tensor before the fusion block and mm_projector.
        handles.append(vision_tower.register_forward_hook(capture_hook("siglip_output", captured)))
    return handles


_TEXT_EXCLUSION_KEYS = (
    "padding_token_indices",
    "newline_token_indices",
    "special_token_indices",
    "camera_prefix_token_indices",
    "cut3r_camera_token_indices",
    "spatial_bridge_token_indices",
    "eomt_object_token_indices",
)


def cleaned_text_token_indices(metadata: dict[str, Any], *, seq_len: int, device: torch.device) -> torch.Tensor:
    """Return prompt/query positions without structural or auxiliary tokens."""
    text = metadata.get("text_token_indices", torch.empty(0, dtype=torch.long, device=device))
    if not isinstance(text, torch.Tensor):
        raise TypeError("visual metadata text_token_indices must be a tensor")
    text = text.to(device=device, dtype=torch.long)
    excluded = []
    for key in _TEXT_EXCLUSION_KEYS:
        value = metadata.get(key)
        if isinstance(value, torch.Tensor) and value.numel():
            excluded.append(value.to(device=device, dtype=torch.long))
    if excluded:
        blocked = torch.unique(torch.cat(excluded))
        text = text[~torch.isin(text, blocked)]
    return torch.unique(text[(text >= 0) & (text < int(seq_len))])


def tensor_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return float("nan")
    return float(torch.sqrt(value.detach().float().square().mean()).item())


def geometry_on_off_rows(
    *,
    model: torch.nn.Module,
    input_ids: Any,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: Any,
    inputs_embeds: torch.Tensor,
    hidden_states_on: Any,
    metadata: dict[str, Any],
    llm_layers: list[int],
    pre_llm_features: dict[str, torch.Tensor],
    video_record: dict[str, Any],
    spatialstack_residuals_by_layer: Any,
    spatialstack_cross_attn_inputs_by_layer: Any,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Measure the native SpatialStack residual's forward-only influence."""
    seq_len = int(inputs_embeds.shape[1])
    device = inputs_embeds.device
    visual = metadata["visual_token_indices"].to(device=device, dtype=torch.long)
    text = cleaned_text_token_indices(metadata, seq_len=seq_len, device=device)
    rows: list[dict[str, Any]] = []
    for name, value in pre_llm_features.items():
        rows.append({
            "video_id": str(video_record.get("video_sample_id", video_record.get("video_path", ""))),
            "video_path": str(video_record.get("video_path", "")),
            "split": str(video_record.get("split", "")),
            "probe_point": name,
            "I_visual": 0.0,
            "I_text": None,
            "text_visual_transfer_ratio": None,
            "visual_token_count": int(value.shape[0] * value.shape[1]),
            "text_token_count": 0,
            "on_off_semantics": "pre_llm_before_spatialstack_injection",
        })
    on_selected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in llm_layers:
        hidden = hidden_state_for_layer(hidden_states_on, layer)
        on_selected[int(layer)] = (
            hidden[0, visual.to(device=hidden.device)].detach().float().cpu(),
            hidden[0, text.to(device=hidden.device)].detach().float().cpu(),
        )
    with torch.no_grad():
        outputs_off = model.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=False,
            output_attentions=False, output_hidden_states=True, return_dict=True,
            spatialstack_residuals_by_layer=None, spatialstack_cross_attn_inputs_by_layer=None,
        )
    hidden_states_off = outputs_off.hidden_states
    if hidden_states_off is None:
        raise RuntimeError("Geometry-OFF forward did not return hidden states")
    active_layers = sorted(
        int(layer) for payload in (spatialstack_residuals_by_layer, spatialstack_cross_attn_inputs_by_layer)
        if isinstance(payload, dict) for layer in payload
    )
    for layer in llm_layers:
        on_visual, on_text = on_selected[int(layer)]
        off_hidden = hidden_state_for_layer(hidden_states_off, layer)
        off_visual = off_hidden[0, visual.to(device=off_hidden.device)].detach().float().cpu()
        off_text = off_hidden[0, text.to(device=off_hidden.device)].detach().float().cpu()
        if tuple(on_visual.shape) != tuple(off_visual.shape) or tuple(on_text.shape) != tuple(off_text.shape):
            raise RuntimeError(f"Geometry ON/OFF shape mismatch at L{layer}")
        delta_visual = on_visual - off_visual
        delta_text = on_text - off_text
        delta_visual_rms = tensor_rms(delta_visual)
        delta_text_rms = tensor_rms(delta_text)
        on_visual_rms = tensor_rms(on_visual)
        on_text_rms = tensor_rms(on_text)
        if active_layers and int(layer) < active_layers[0] and delta_visual_rms > float(tolerance):
            raise RuntimeError(
                f"Geometry ON/OFF changed L{layer} before first injection L{active_layers[0]}: "
                f"RMS={delta_visual_rms} > tolerance={tolerance}"
            )
        rows.append({
            "video_id": str(video_record.get("video_sample_id", video_record.get("video_path", ""))),
            "video_path": str(video_record.get("video_path", "")),
            "split": str(video_record.get("split", "")),
            "probe_point": f"L{int(layer)}",
            "I_visual": delta_visual_rms / on_visual_rms if on_visual_rms > 0 else float("nan"),
            "I_text": delta_text_rms / on_text_rms if on_text.numel() and on_text_rms > 0 else None,
            "text_visual_transfer_ratio": delta_text_rms / delta_visual_rms if on_text.numel() and delta_visual_rms > 0 else None,
            "visual_token_count": int(on_visual.shape[0]),
            "text_token_count": int(on_text.shape[0]),
            "on_off_semantics": "same_prepared_input_native_spatialstack_payload_withheld",
        })
    return rows


def geometry_on_off_path(output_root: Path, model_label: str, video_record: dict[str, Any]) -> Path:
    video_id = str(video_record.get("video_sample_id", Path(str(video_record["video_path"])).stem))
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in video_id)
    return Path(output_root) / "geometry_on_off" / str(model_label) / f"video_{safe}.json"


def assert_zero_spatial_post_fusion_projector_capture(
    captured: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove that zero-spatial projected_features is the post-fusion projector output."""

    fusion = captured.get("_zero_spatial_fusion_output")
    projector_input = captured.get("_zero_spatial_projector_input")
    projected = captured.get("projected_features")
    missing = [
        name
        for name, value in (
            ("fusion_output", fusion),
            ("projector_input", projector_input),
            ("projected_features", projected),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            "zero_spatial post-fusion projected_features attestation missing captures: "
            + ", ".join(missing)
        )
    if tuple(fusion.shape) != tuple(projector_input.shape):
        raise RuntimeError(
            "zero_spatial fusion output/projector input shape mismatch: "
            f"fusion={tuple(fusion.shape)} projector_input={tuple(projector_input.shape)}"
        )
    if not torch.equal(fusion, projector_input):
        max_diff = float((fusion.float() - projector_input.float()).abs().max().item())
        raise RuntimeError(
            "zero_spatial projected_features was not produced from the captured fusion output: "
            f"max_input_difference={max_diff}"
        )
    return {
        "assessment": "PASS",
        "projected_features_definition": "mm_projector output after zero-spatial fusion path",
        "fusion_block_output_shape": list(fusion.shape),
        "mm_projector_input_shape": list(projector_input.shape),
        "projected_features_output_shape": list(projected.shape),
        "fusion_output_equals_mm_projector_input": True,
    }


def assert_first_base_video_runtime(
    *,
    captured: dict[str, torch.Tensor],
    normalized_pre_llm: dict[str, torch.Tensor],
    hidden_states: Any,
    metadata: dict[str, Any],
    selected_frames: list[int],
    num_frames: int,
    runtime_dtypes: dict[str, str | None],
    model_forward_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Fail before continuing a full run if the first base forward is wrong."""
    raw_siglip = captured.get("siglip_output")
    raw_projected = captured.get("projected_features")
    if raw_siglip is None or tuple(raw_siglip.shape) != (32, 729, 1152):
        raise RuntimeError(
            "First pre_sft_base_vlm video raw siglip_output assertion failed: "
            f"expected=[32,729,1152], observed={getattr(raw_siglip, 'shape', None)}"
        )
    if raw_projected is None:
        raise RuntimeError("First pre_sft_base_vlm video did not capture projected_features.")
    expected_normalized = {
        "siglip_output": (32, 196, 1152),
        "projected_features": (32, 196, 3584),
    }
    observed_normalized = {}
    for name, shape in expected_normalized.items():
        value = normalized_pre_llm.get(name)
        observed_normalized[name] = list(value.shape) if isinstance(value, torch.Tensor) else None
        if value is None or tuple(value.shape) != shape:
            raise RuntimeError(
                f"First pre_sft_base_vlm video normalized {name} assertion failed: "
                f"expected={list(shape)}, observed={observed_normalized[name]}"
            )
    if len(hidden_states) <= 7:
        raise RuntimeError(
            "First pre_sft_base_vlm video L6 assertion failed: hidden_states[7] is unavailable."
        )
    l6 = hidden_state_for_layer(hidden_states, 6)
    if l6 is not hidden_states[7]:
        raise RuntimeError("First pre_sft_base_vlm video L6 did not resolve to hidden_states[7].")

    frame_ids = metadata["visual_frame_ids"].detach().cpu().tolist()
    frame_counts = {frame: frame_ids.count(frame) for frame in sorted(set(frame_ids))}
    if int(num_frames) != 32 or sorted(frame_counts) != list(range(32)):
        raise RuntimeError(
            "First pre_sft_base_vlm video frame-order assertion failed: "
            f"num_frames={num_frames}, frame_counts={frame_counts}"
        )
    if len(set(frame_counts.values())) != 1 or next(iter(frame_counts.values()), 0) != 196:
        raise RuntimeError(
            "First pre_sft_base_vlm video selected-token alignment failed: "
            f"expected 196 visual tokens/frame, observed={frame_counts}"
        )
    if any(frame_ids[index] > frame_ids[index + 1] for index in range(len(frame_ids) - 1)):
        raise RuntimeError("First pre_sft_base_vlm video visual frame ordering is not monotonic.")
    if any(int(frame) not in frame_counts for frame in selected_frames):
        raise RuntimeError(
            "First pre_sft_base_vlm video selected target frames are absent from visual metadata: "
            f"selected={selected_frames}, available={sorted(frame_counts)}"
        )
    if any(value is not False for value in model_forward_inputs.values()):
        raise RuntimeError(
            "First pre_sft_base_vlm video consumed a forbidden spatial/geometry input: "
            f"{model_forward_inputs}"
        )

    required_fp16 = {
        "siglip_output_dtype",
        "vision_tower_forward_input_dtype",
        "vision_tower_forward_output_dtype",
        "mm_projector_forward_output_dtype",
        "projected_features_dtype",
        "llm_inputs_embeds_dtype",
        "llm_hidden_states_output_dtype",
        "layer_6_hidden_states_7_dtype",
    }
    non_fp16 = {
        name: runtime_dtypes.get(name)
        for name in sorted(required_fp16)
        if runtime_dtypes.get(name) != "torch.float16"
    }
    if non_fp16:
        raise RuntimeError(
            "First pre_sft_base_vlm video FP16 runtime assertion failed: "
            f"{non_fp16}"
        )
    return {
        "assessment": "PASS",
        "raw_siglip_output_shape": list(raw_siglip.shape),
        "normalized_siglip_output_shape": list(normalized_pre_llm["siglip_output"].shape),
        "normalized_projected_features_shape": list(normalized_pre_llm["projected_features"].shape),
        "l6_hidden_state_index": 7,
        "forward_num_frames": int(num_frames),
        "forward_frame_order": list(range(32)),
        "selected_target_frames": [int(frame) for frame in selected_frames],
        "visual_tokens_per_frame": frame_counts,
        "same_frame_ordering_and_selected_targets": True,
        "runtime_dtypes_fp16": True,
        "runtime_dtypes": dict(runtime_dtypes),
        "spatial_geometry_inputs_consumed": False,
        "model_forward_inputs": dict(model_forward_inputs),
    }


def assert_first_zero_spatial_pre_llm_video_runtime(
    *,
    captured: dict[str, torch.Tensor],
    normalized_pre_llm: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    selected_frames: list[int],
    num_frames: int,
    runtime_dtypes: dict[str, str | None],
    model_forward_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Fail before a full zero-spatial pre-LLM run if its capture contract is wrong."""

    raw_siglip = captured.get("siglip_output")
    raw_projected = captured.get("projected_features")
    if raw_siglip is None or tuple(raw_siglip.shape) != (32, 729, 1152):
        raise RuntimeError(
            "First zero_spatial video raw siglip_output assertion failed: "
            f"expected=[32,729,1152], observed={getattr(raw_siglip, 'shape', None)}"
        )
    if raw_projected is None or tuple(raw_projected.shape) != (32, 729, 3584):
        raise RuntimeError(
            "First zero_spatial video raw projected_features assertion failed: "
            f"expected=[32,729,3584], observed={getattr(raw_projected, 'shape', None)}"
        )
    expected_normalized = {
        "siglip_output": (32, 196, 1152),
        "projected_features": (32, 196, 3584),
    }
    observed_normalized = {}
    for name, shape in expected_normalized.items():
        value = normalized_pre_llm.get(name)
        observed_normalized[name] = list(value.shape) if isinstance(value, torch.Tensor) else None
        if value is None or tuple(value.shape) != shape:
            raise RuntimeError(
                f"First zero_spatial video normalized {name} assertion failed: "
                f"expected={list(shape)}, observed={observed_normalized[name]}"
            )
    post_fusion = assert_zero_spatial_post_fusion_projector_capture(captured)

    frame_ids = metadata["visual_frame_ids"].detach().cpu().tolist()
    frame_counts = {frame: frame_ids.count(frame) for frame in sorted(set(frame_ids))}
    if int(num_frames) != 32 or sorted(frame_counts) != list(range(32)):
        raise RuntimeError(
            "First zero_spatial video frame-order assertion failed: "
            f"num_frames={num_frames}, frame_counts={frame_counts}"
        )
    if len(set(frame_counts.values())) != 1 or next(iter(frame_counts.values()), 0) != 196:
        raise RuntimeError(
            "First zero_spatial video selected-token alignment failed: "
            f"expected 196 visual tokens/frame, observed={frame_counts}"
        )
    if any(frame_ids[index] > frame_ids[index + 1] for index in range(len(frame_ids) - 1)):
        raise RuntimeError("First zero_spatial video visual frame ordering is not monotonic.")
    if any(int(frame) not in frame_counts for frame in selected_frames):
        raise RuntimeError(
            "First zero_spatial video selected target frames are absent from visual metadata: "
            f"selected={selected_frames}, available={sorted(frame_counts)}"
        )
    expected_inputs = {
        "spatial_features": True,
        "point_maps": False,
        "geometry_spatial_features": False,
        "geometry_outputs": False,
    }
    if dict(model_forward_inputs) != expected_inputs:
        raise RuntimeError(
            "First zero_spatial video forward-input contract failed: "
            f"expected={expected_inputs}, observed={model_forward_inputs}"
        )
    required_fp16 = {
        "siglip_output_dtype",
        "vision_tower_forward_input_dtype",
        "vision_tower_forward_output_dtype",
        "mm_projector_forward_output_dtype",
        "projected_features_dtype",
    }
    non_fp16 = {
        name: runtime_dtypes.get(name)
        for name in sorted(required_fp16)
        if runtime_dtypes.get(name) != "torch.float16"
    }
    if non_fp16:
        raise RuntimeError(
            "First zero_spatial video FP16 runtime assertion failed: "
            f"{non_fp16}"
        )
    return {
        "assessment": "PASS",
        "raw_siglip_output_shape": list(raw_siglip.shape),
        "raw_projected_features_shape": list(raw_projected.shape),
        "normalized_siglip_output_shape": list(normalized_pre_llm["siglip_output"].shape),
        "normalized_projected_features_shape": list(normalized_pre_llm["projected_features"].shape),
        "normalization": "model.get_2dPool",
        "forward_num_frames": int(num_frames),
        "forward_frame_order": list(range(32)),
        "selected_target_frames": [int(frame) for frame in selected_frames],
        "visual_tokens_per_frame": frame_counts,
        "same_frame_ordering_and_selected_targets": True,
        "runtime_dtypes_fp16": True,
        "runtime_dtypes": dict(runtime_dtypes),
        "model_forward_inputs": dict(model_forward_inputs),
        "projected_features_contract": post_fusion,
    }


def assert_first_pre_sft_fusion_video_runtime(
    *,
    hidden_states: Any,
    metadata: dict[str, Any],
    selected_frames: list[int],
    model_forward_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Small first-forward contract for the sidecar-backed pre-SFT variants."""
    if not model_forward_inputs.get("spatial_features"):
        raise RuntimeError("pre_sft_fusion requires CUT3R spatial_features in the model forward.")
    if model_forward_inputs.get("point_maps") or model_forward_inputs.get("geometry_spatial_features"):
        raise RuntimeError(
            "pre_sft_fusion must use compact targets only for depth supervision, not model-forward geometry."
        )
    visual_frame_ids = metadata["visual_frame_ids"].detach().cpu().tolist()
    available = {int(value) for value in visual_frame_ids}
    missing = [int(frame) for frame in selected_frames if int(frame) not in available]
    if missing:
        raise RuntimeError(f"pre_sft_fusion selected frames are absent from visual metadata: {missing}")
    if len(hidden_states) <= 28:
        raise RuntimeError(f"pre_sft_fusion expected hidden states through L27, got {len(hidden_states)} states")
    return {
        "assessment": "PASS",
        "spatial_features_consumed": True,
        "compact_targets_excluded_from_model_forward": True,
        "selected_frames_present": True,
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
    }


def normalize_captured_video_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    num_frames: int,
    target_grid_shape: tuple[int, int],
    require_model_pool: bool = False,
) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim == 2:
        if tensor.shape[0] % int(num_frames) != 0:
            raise ValueError(f"Cannot unflatten captured tensor {tuple(tensor.shape)} into {num_frames} frames")
        tensor = tensor.reshape(int(num_frames), tensor.shape[0] // int(num_frames), tensor.shape[-1])
    if tensor.ndim != 3:
        raise ValueError(f"Expected captured tensor [F,N,D] or [F*N,D], got {tuple(tensor.shape)}")
    if tensor.shape[0] != int(num_frames):
        raise ValueError(f"Captured frame count mismatch: tensor={tuple(tensor.shape)} num_frames={num_frames}")

    target_tokens = int(target_grid_shape[0]) * int(target_grid_shape[1])
    pooled = None
    get_2d_pool = getattr(model, "get_2dPool", None)
    if callable(get_2d_pool):
        try:
            pooled = get_2d_pool(tensor)
        except Exception:
            pooled = None
    if pooled is not None and int(pooled.shape[1]) == target_tokens:
        return pooled.float()
    if require_model_pool:
        raise RuntimeError(
            "The requested representation must use model.get_2dPool, but pooling did not produce "
            f"the target grid {target_grid_shape} from {tuple(tensor.shape)}."
        )
    if int(tensor.shape[1]) == target_tokens:
        return tensor

    source_side = int(np.sqrt(int(tensor.shape[1])))
    if source_side * source_side != int(tensor.shape[1]):
        raise ValueError(
            f"Cannot resize non-square captured token grid {tuple(tensor.shape)} to {target_grid_shape}"
        )
    x = tensor.reshape(tensor.shape[0], source_side, source_side, tensor.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=target_grid_shape, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(tensor.shape[0], target_tokens, tensor.shape[-1]).contiguous()


def selected_frame_hidden_grids(
    hidden: torch.Tensor,
    metadata: dict[str, Any],
    selected_frames: list[int],
) -> dict[int, torch.Tensor]:
    visual_indices = metadata["visual_token_indices"].to(device=hidden.device)
    frame_ids = metadata["visual_frame_ids"].to(device=hidden.device)
    out: dict[int, torch.Tensor] = {}
    for frame_idx in selected_frames:
        indices = visual_indices[frame_ids == int(frame_idx)]
        if indices.numel() == 0:
            raise RuntimeError(f"No visual tokens found for selected frame {frame_idx}")
        grid_shape = grid_shape_for_frame(metadata, int(frame_idx), token_count=int(indices.numel()))
        out[int(frame_idx)] = reshape_tokens_to_grid(hidden[0, indices].detach().float().cpu(), grid_shape)
    return out


def save_frame_outputs(
    *,
    output_root: Path,
    model_label: str,
    frame_record: dict[str, Any],
    llm_features: dict[str, torch.Tensor],
    pre_llm_features: dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    metadata: dict[str, Any],
    feature_provenance: dict[str, Any],
    cache_dtype: torch.dtype,
) -> None:
    fsid = str(frame_record["frame_sample_id"])
    (output_root / "gt_depth").mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    gt_path = output_root / "gt_depth" / f"frame_{fsid}.pt"
    metadata_path = output_root / "metadata" / f"frame_{fsid}.pt"
    if not gt_path.exists():
        torch.save(gt_depth.float().cpu(), gt_path)

    if not metadata_path.exists():
        meta_payload = dict(frame_record)
        meta_payload.update(metadata)
        meta_payload["gt_valid_mask"] = gt_valid.cpu()
        meta_payload["gt_depth_map_downsampled"] = gt_depth.float().cpu()
        torch.save(meta_payload, metadata_path)

    for layer_name, value in llm_features.items():
        if not layer_name.startswith("layer_"):
            raise ValueError(f"Unexpected LLM feature name {layer_name!r}")
        layer = int(layer_name.removeprefix("layer_"))
        feature_path = layer_feature_path(output_root, model_label, layer, fsid)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(coerce_cache_dtype(value, cache_dtype), feature_path)
        provenance_path = feature_path.parent / "provenance.json"
        if not provenance_path.exists():
            payload = {**feature_provenance, "feature_level": layer_name, "feature_path_layout": "layer_per_directory_v1"}
            with provenance_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
    for feature_name, value in pre_llm_features.items():
        feature_dir = output_root / "features" / model_label / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        torch.save(coerce_cache_dtype(value, cache_dtype), feature_dir / f"frame_{fsid}.pt")
        provenance_path = feature_dir / "provenance.json"
        if not provenance_path.exists():
            payload = {**feature_provenance, "feature_level": feature_name, "feature_path_layout": "feature_per_directory_v1"}
            with provenance_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")


def output_complete(
    output_root: Path,
    model_label: str,
    frame_sample_id: str,
    pre_llm_feature_names: list[str],
    llm_layers: list[int],
) -> bool:
    if not (output_root / "gt_depth" / f"frame_{frame_sample_id}.pt").exists():
        return False
    if not (output_root / "metadata" / f"frame_{frame_sample_id}.pt").exists():
        return False
    for layer in llm_layers:
        if not layer_feature_path(output_root, model_label, layer, frame_sample_id).exists():
            return False
    for feature_name in pre_llm_feature_names:
        if not (output_root / "features" / model_label / feature_name / f"frame_{frame_sample_id}.pt").exists():
            return False
    return True


def build_dataset(args: argparse.Namespace, tokenizer: Any, image_processor: Any):
    from llava.train.train import DataCollatorForSupervisedDataset, LazySupervisedDataset

    data_args = make_data_args(args, image_processor)
    data_args.deterministic_data_order = True
    data_args.train_data_shuffle = False
    if args.model_loading_mode == "pre_sft_base_vlm":
        # The base control uses the same dataset/collator path but must never
        # cause LazySupervisedDataset to read a CUT3R sidecar.
        data_args.spatial_features_root = None
        data_args.spatial_features_subdir = None
        data_args.spatial_tower_type = None
        data_args.require_spatial_features = False
    else:
        data_args.spatial_features_root = args.feature_root
        data_args.spatial_features_subdir = args.spatial_features_subdir
        data_args.spatial_tower_type = "cut3r"
        data_args.require_spatial_features = True
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.train_data_json, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    by_video = {}
    for idx, item in enumerate(dataset.list_data_dict):
        video = item.get("video")
        if video is not None and video not in by_video:
            by_video[str(video)] = idx
    return dataset, collator, by_video


def extract_for_video(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    collator: Any,
    dataset: Any,
    dataset_index: int,
    video_record: dict[str, Any],
    selected_frames: list[int],
    captured: dict[str, torch.Tensor],
    cache_dtype: torch.dtype,
    device: torch.device,
    model_dtype: torch.dtype,
    assert_runtime: bool = False,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    item = dataset[dataset_index]

    if args.probe_targets_root:
        depths_by_frame, point_maps_path, point_payload = load_selected_camera_depths(
            Path(args.probe_targets_root), video_record, selected_frames
        )
        point_key = "point_maps_cam"
        depth_mode = "camera_z"
        # The compact target bundle has exactly two selected frames and must
        # never be inserted into the model's 32-frame forward input.
    else:
        point_maps_path = resolve_sidecar_path(
            str(video_record["video_path"]),
            Path(args.point_maps_root or args.feature_root),
            args.point_maps_subdir,
        )
        if point_maps_path is None:
            raise FileNotFoundError(f"Missing point-map sidecar for {video_record['video_path']}")
        point_payload = load_point_map_sidecar(point_maps_path)
        point_maps, point_key, depth_mode = select_point_maps(
            point_payload, allow_euclidean_depth=args.allow_euclidean_depth
        )
        depths = depth_from_point_maps(point_maps, depth_mode)
        depths_by_frame = {frame_index: depths[frame_index] for frame_index in selected_frames}
        if point_maps.shape[-1] == 3:
            model_point_maps = point_maps.permute(0, 3, 1, 2).contiguous()
        else:
            model_point_maps = point_maps.contiguous()
        item["point_maps"] = model_point_maps.float()

    batch = collator([item])
    if args.model_loading_mode == "pre_sft_base_vlm":
        forbidden = [key for key in ("spatial_features", "geometry_spatial_features", "point_maps") if key in batch]
        if forbidden:
            raise RuntimeError(f"pre_sft_base_vlm forward received forbidden spatial inputs: {forbidden}")
    model_forward_inputs = {
        "spatial_features": "spatial_features" in batch,
        "point_maps": "point_maps" in batch,
        "geometry_spatial_features": "geometry_spatial_features" in batch,
        "geometry_outputs": "geometry_outputs" in batch,
    }
    batch = move_to_device(batch, device, model_dtype)

    # Accelerate may place the vision/CUT3R towers on a different GPU from
    # the language-model input device. Keep the historical tensors unchanged,
    # but place each input beside the tower that consumes it.
    def module_device(module: Any, fallback: torch.device) -> torch.device:
        declared = getattr(module, "device", None)
        if isinstance(declared, torch.device) and declared.type != "meta":
            return declared
        for parameter in module.parameters():
            if not parameter.is_meta:
                return parameter.device
        return fallback

    def move_value(value: Any, target: torch.device, dtype: torch.dtype | None = None) -> Any:
        if torch.is_tensor(value):
            return value.to(device=target, dtype=dtype if dtype is not None and value.is_floating_point() else None)
        if isinstance(value, list):
            return [move_value(item, target, dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item, target, dtype) for item in value)
        return value

    vision_tower = getattr(model, "get_vision_tower", lambda: None)()
    if vision_tower is not None and "images" in batch:
        vision_dtype = model_dtype
        if args.model_loading_mode == "pre_sft_base_vlm":
            vision_dtype = getattr(vision_tower, "dtype", None)
            if not isinstance(vision_dtype, torch.dtype):
                raise RuntimeError("Could not determine the materialized vision-tower dtype.")
            if vision_dtype != model_dtype:
                raise RuntimeError(
                    "pre_sft_base_vlm requires explicit vision-tower FP16 materialization: "
                    f"requested={model_dtype}, observed={vision_dtype}"
                )
        batch["images"] = move_value(batch["images"], module_device(vision_tower, device), vision_dtype)
    spatial_tower = getattr(model, "get_spatial_tower", lambda: None)()
    if spatial_tower is not None:
        spatial_device = module_device(spatial_tower, device)
        for key in ("spatial_features", "point_maps"):
            if key in batch:
                batch[key] = move_value(batch[key], spatial_device)
    base_vision_tower = None
    if args.model_loading_mode == "pre_sft_base_vlm":
        base_vision_tower = getattr(model, "get_vision_tower", lambda: None)()
        if base_vision_tower is None:
            raise RuntimeError("pre_sft_base_vlm has no materialized vision tower at forward time.")
        if "images" in batch:
            base_vision_dtype = getattr(base_vision_tower, "dtype", None)
            if not isinstance(base_vision_dtype, torch.dtype):
                raise RuntimeError("Could not determine the materialized vision-tower dtype.")
            if base_vision_dtype != model_dtype:
                raise RuntimeError(
                    "pre_sft_base_vlm requires explicit vision-tower FP16 materialization: "
                    f"requested={model_dtype}, observed={base_vision_dtype}"
                )
            batch["images"] = base_move_value(
                batch["images"], base_module_device(base_vision_tower, device), base_vision_dtype
            )
    prepare_fn = getattr(model, "prepare_inputs_labels_for_multimodal", None)
    if prepare_fn is None:
        raise RuntimeError("Model does not expose prepare_inputs_labels_for_multimodal().")
    if "return_visual_metadata" not in inspect.signature(prepare_fn).parameters:
        raise RuntimeError("prepare_inputs_labels_for_multimodal() lacks return_visual_metadata support.")

    runtime_dtypes: dict[str, str | None] = {}
    if args.pre_llm_feature_names:
        if base_vision_tower is not None:
            runtime_dtypes["vision_tower_parameter_dtype"] = str(getattr(base_vision_tower, "dtype", None))
        if "images" in batch:
            images_value = batch["images"]
            if isinstance(images_value, (list, tuple)):
                runtime_dtypes["vision_tower_forward_input_dtype"] = dtype_name(images_value[0]) if images_value else None
            else:
                runtime_dtypes["vision_tower_forward_input_dtype"] = dtype_name(images_value)

    def vision_dtype_hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        runtime_dtypes["vision_tower_forward_output_dtype"] = dtype_name(value)

    captured.clear()
    vision_handle = (
        vision_tower.register_forward_hook(vision_dtype_hook)
        if args.pre_llm_feature_names and vision_tower is not None
        else None
    )
    try:
        with torch.no_grad():
            prepared = prepare_fn(
                input_ids=batch["input_ids"],
                position_ids=None,
                attention_mask=batch["attention_mask"],
                past_key_values=None,
                labels=None,
                images=batch["images"],
                spatial_features=None if args.model_loading_mode == "pre_sft_base_vlm" else batch.get("spatial_features"),
                point_maps=None if args.model_loading_mode == "pre_sft_base_vlm" else batch.get("point_maps"),
                modalities=batch.get("modalities"),
                image_sizes=batch.get("image_sizes"),
                return_visual_metadata=True,
            )
    finally:
        if vision_handle is not None:
            vision_handle.remove()
    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, _labels, visual_metadata = prepared
    if args.pre_llm_feature_names:
        runtime_dtypes["siglip_output_dtype"] = dtype_name(captured.get("siglip_output"))
        projected_dtype = dtype_name(captured.get("projected_features"))
        runtime_dtypes["mm_projector_forward_output_dtype"] = projected_dtype
        runtime_dtypes["projected_features_dtype"] = projected_dtype
        runtime_dtypes["llm_inputs_embeds_dtype"] = dtype_name(inputs_embeds)
    metadata = visual_metadata[0]
    visual_indices = metadata["visual_token_indices"]
    frame_ids = metadata["visual_frame_ids"]
    if visual_indices.numel() == 0:
        raise RuntimeError("No visual tokens returned by metadata")
    available_frames = {int(x) for x in frame_ids.detach().cpu().tolist()}
    missing = [idx for idx in selected_frames if int(idx) not in available_frames]
    if missing:
        raise RuntimeError(f"Selected frame ids not present in visual metadata: {missing}")

    spatialstack_residuals_by_layer = None
    spatialstack_cross_attn_inputs_by_layer = None
    use_cut3r_spatialstack = getattr(model.config, "use_cut3r_spatialstack", False)
    if isinstance(use_cut3r_spatialstack, str):
        use_cut3r_spatialstack = use_cut3r_spatialstack.lower() in {"1", "true", "yes", "y", "on"}
    if bool(use_cut3r_spatialstack):
        merger_getter = getattr(model.model, "get_cut3r_spatialstack_merger", None)
        merger = merger_getter() if callable(merger_getter) else None
        if merger is None:
            initializer = getattr(model.model, "initialize_cut3r_spatialstack_merger", None)
            if not callable(initializer):
                raise RuntimeError("use_cut3r_spatialstack=True, but model.model cannot initialize the merger.")
            merger = initializer(model.config)
        spatialstack_payload_by_layer = merger(
            batch.get("spatial_features"),
            visual_metadata,
            seq_len=int(inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        spatialstack_fusion_type = str(getattr(model.config, "cut3r_spatialstack_fusion_type", "add") or "add").strip().lower()
        if spatialstack_fusion_type == "cross_attn":
            spatialstack_cross_attn_inputs_by_layer = spatialstack_payload_by_layer
        else:
            spatialstack_residuals_by_layer = spatialstack_payload_by_layer

    if args.llm_layers:
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                spatialstack_residuals_by_layer=spatialstack_residuals_by_layer,
                spatialstack_cross_attn_inputs_by_layer=spatialstack_cross_attn_inputs_by_layer,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
    else:
        # Pre-LLM-only jobs need the vision/fusion/projector path but do not
        # need to execute the decoder or retain its hidden-state tuple.
        hidden_states = []
    if args.pre_llm_feature_names and hidden_states:
        runtime_dtypes["llm_hidden_states_output_dtype"] = dtype_name(hidden_states[-1])
        runtime_dtypes["layer_6_hidden_states_7_dtype"] = (
            dtype_name(hidden_states[7]) if len(hidden_states) > 7 else None
        )

    num_frames = int(max(available_frames)) + 1
    target_grid_shape = grid_shape_for_frame(
        metadata,
        int(selected_frames[0]),
        token_count=int((frame_ids == int(selected_frames[0])).sum().item()),
    )

    normalized_pre_llm: dict[str, torch.Tensor] = {}
    normalization_methods: dict[str, str] = {}
    for feature_name in args.pre_llm_feature_names:
        if feature_name not in captured:
            raise RuntimeError(f"{feature_name} hook did not capture an output")
        require_model_pool = (
            args.model_label == "zero_spatial"
            or args.model_label.startswith("cut3r_spatialstack")
            or args.model_loading_mode == "pre_sft_fusion"
        ) and feature_name in {
            "siglip_output",
            "fusion_output",
            "projected_features",
        }
        normalized_pre_llm[feature_name] = normalize_captured_video_tokens(
            model,
            captured[feature_name],
            num_frames=num_frames,
            target_grid_shape=target_grid_shape,
            require_model_pool=require_model_pool,
        )
        normalization_methods[feature_name] = (
            "model.get_2dPool" if require_model_pool else "model.get_2dPool_or_legacy_resize"
        )

    llm_by_layer: dict[str, dict[int, torch.Tensor]] = {}
    for layer in args.llm_layers:
        hidden_index = int(layer) + 1
        if hidden_index >= len(hidden_states):
            raise ValueError(f"Requested layer {layer}, but hidden_states length is {len(hidden_states)}")
        llm_by_layer[f"layer_{layer}"] = selected_frame_hidden_grids(
            hidden_state_for_layer(hidden_states, layer),
            metadata,
            selected_frames,
        )

    first_video_runtime_assertions = None
    if assert_runtime:
        if args.model_label == "zero_spatial":
            first_video_runtime_assertions = assert_first_zero_spatial_pre_llm_video_runtime(
                captured=captured,
                normalized_pre_llm=normalized_pre_llm,
                metadata=metadata,
                selected_frames=selected_frames,
                num_frames=num_frames,
                runtime_dtypes=runtime_dtypes,
                model_forward_inputs=model_forward_inputs,
            )
        elif args.model_loading_mode == "pre_sft_fusion":
            first_video_runtime_assertions = assert_first_pre_sft_fusion_video_runtime(
                hidden_states=hidden_states,
                metadata=metadata,
                selected_frames=selected_frames,
                model_forward_inputs=model_forward_inputs,
            )
        else:
            first_video_runtime_assertions = assert_first_base_video_runtime(
                captured=captured,
                normalized_pre_llm=normalized_pre_llm,
                hidden_states=hidden_states,
                metadata=metadata,
                selected_frames=selected_frames,
                num_frames=num_frames,
                runtime_dtypes=runtime_dtypes,
                model_forward_inputs=model_forward_inputs,
            )

    geometry_rows: list[dict[str, Any]] = []
    requested_on_off_split = getattr(args, "geometry_on_off_split", None)
    if requested_on_off_split and str(video_record.get("split", "")) == str(requested_on_off_split):
        if not bool(use_cut3r_spatialstack):
            raise RuntimeError("Geometry ON/OFF requires a SpatialStack-enabled model.")
        geometry_rows = geometry_on_off_rows(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            hidden_states_on=hidden_states,
            metadata=metadata,
            llm_layers=args.llm_layers,
            pre_llm_features=normalized_pre_llm,
            video_record=video_record,
            spatialstack_residuals_by_layer=spatialstack_residuals_by_layer,
            spatialstack_cross_attn_inputs_by_layer=spatialstack_cross_attn_inputs_by_layer,
            tolerance=float(args.geometry_on_off_tolerance),
        )
        on_off_path = geometry_on_off_path(output_root, args.model_label, video_record)
        on_off_path.parent.mkdir(parents=True, exist_ok=True)
        with on_off_path.open("w", encoding="utf-8") as f:
            json.dump(geometry_rows, f, indent=2, sort_keys=True)
            f.write("\n")

    for frame_record in video_record["frames"]:
        frame_idx = int(frame_record["frame_index"])
        grid_shape = grid_shape_for_frame(
            metadata,
            frame_idx,
            token_count=int((frame_ids == frame_idx).sum().item()),
        )
        gt_depth, gt_valid = downsample_depth_to_grid(depths_by_frame[frame_idx], grid_shape)
        depth_meta = frame_depth_metadata(gt_depth, gt_valid)
        selected_indices = metadata["visual_token_indices"][metadata["visual_frame_ids"] == frame_idx].detach().cpu()
        frame_metadata = {
            "sample_id": frame_record["frame_sample_id"],
            "model_label": args.model_label,
            "num_frames": 1,
            "source_video_num_frames": num_frames,
            "visual_grid_shape": tuple(int(x) for x in grid_shape),
            "visual_token_indices": selected_indices,
            "point_maps_path": str(point_maps_path),
            "point_map_key": point_key,
            "depth_mode": depth_mode,
            "requested_llm_layers": list(args.llm_layers),
            "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
            "pre_llm_normalization": dict(normalization_methods),
            "pre_llm_representation_definitions": dict(
                getattr(args, "pre_llm_representation_definitions", {})
            ),
            **depth_meta,
            "visual_metadata": json_ready_metadata(metadata),
        }
        llm_features = {layer_name: frames[frame_idx] for layer_name, frames in llm_by_layer.items()}
        pre_llm_features = {
            feature_name: reshape_tokens_to_grid(value[frame_idx].float().cpu(), grid_shape)
            for feature_name, value in normalized_pre_llm.items()
        }
        save_frame_outputs(
            output_root=output_root,
            model_label=args.model_label,
            frame_record={**video_record, **frame_record},
            llm_features=llm_features,
            pre_llm_features=pre_llm_features,
            gt_depth=gt_depth,
            gt_valid=gt_valid,
            metadata=frame_metadata,
            feature_provenance=args.feature_provenance,
            cache_dtype=cache_dtype,
        )
    captured.clear()
    return {
        "selected_frames": selected_frames,
        "source_video_num_frames": num_frames,
        "visual_grid_shapes": {
            str(frame_idx): list(grid_shape_for_frame(metadata, frame_idx, token_count=int((frame_ids == frame_idx).sum().item())))
            for frame_idx in selected_frames
        },
        "visual_tokens_per_selected_frame": {
            str(frame_idx): int((frame_ids == frame_idx).sum().item()) for frame_idx in selected_frames
        },
        "pre_llm_shapes": {name: list(value.shape) for name, value in normalized_pre_llm.items()},
        "llm_shapes": {name: list(frames[selected_frames[0]].shape) for name, frames in llm_by_layer.items()},
        "target_semantics": "point_maps_cam -> camera_z",
        "runtime_dtypes": runtime_dtypes,
        "model_forward_inputs": model_forward_inputs,
        "pre_llm_normalization": normalization_methods,
        "first_video_runtime_assertions": first_video_runtime_assertions,
        "spatialstack_insertion_stats": list(
            getattr(model.model, "_last_cut3r_spatialstack_injection_stats", [])
        ),
        "geometry_on_off_rows": geometry_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-loading-mode", choices=["adapter", "pre_sft_base_vlm", "pre_sft_fusion"], default="adapter")
    parser.add_argument(
        "--pre-sft-fusion-variant",
        choices=["ss_identity", "ss_zero", "vlm3r_native", "c1_ss_add", "c1_ss_cross_attn_v1", "c1_vlm3r"],
        default=None,
        help="Architecture attached to the plain base VLM in pre_sft_fusion mode.",
    )
    parser.add_argument(
        "--fusion-init-seed",
        type=int,
        default=None,
        help="Seed used only while constructing newly initialized fusion modules.",
    )
    parser.add_argument(
        "--c1-calibration-json",
        default=None,
        help="Frozen C1 scalar calibration artifact. Required for a c1_* fusion variant.",
    )
    parser.add_argument(
        "--common-model-init-seed",
        type=int,
        default=0,
        help="Fixed seed for incidental plain-base construction; independent of fusion initialization.",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--feature-preset", choices=FEATURE_PRESETS, default=None)
    parser.add_argument("--feature-levels", default=None, help="Comma-separated override, e.g. fusion_output,layer_0,layer_3")
    parser.add_argument("--llm-layers", default=None, help="Legacy comma-separated LLM layer indices to extract.")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Explicit LLM layers, e.g. --layers 1 2 12 18 24.")
    parser.add_argument("--pre-llm-features", default=None, help="Comma-separated pre-LLM hooks to extract.")
    parser.add_argument("--model-base", default="/leonardo_work/EUHPC_D32_006/FAST/hf_models/VLM3R/LLaVA-NeXT-Video-7B-Qwen2")
    parser.add_argument("--model-name", default="vlm-3r-llava-qwen2-lora")
    parser.add_argument("--sample-indices", default=str(DEFAULT_OUTPUT_ROOT / "sample_indices.json"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-data-json", "--data-yaml", dest="train_data_json", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--feature-root", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--spatial-features-subdir", default=DEFAULT_SPATIAL_FEATURES_SUBDIR)
    parser.add_argument("--point-maps-root", default=None)
    parser.add_argument("--point-maps-subdir", default=DEFAULT_POINT_MAPS_SUBDIR)
    parser.add_argument(
        "--forward-frames-root",
        default=None,
        help="Opt-in root of migrated forward_frames_32_v1 decoded RGB caches.",
    )
    parser.add_argument(
        "--probe-targets-root",
        default=None,
        help="Opt-in root of migrated probe_targets_2f_v1 compact camera-depth targets.",
    )
    parser.add_argument("--image-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--video-folder", default=str(DEFAULT_FAST_FEATURE_ROOT))
    parser.add_argument("--frames-upbound", type=int, default=32)
    parser.add_argument("--mm-spatial-pool-stride", type=int, default=2)
    parser.add_argument("--pool-mode", choices=["bilinear", "average", "max"], default="bilinear")
    parser.add_argument("--add-time-instruction", type=str2bool, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", choices=["auto", "cuda:0", "cpu"], default=None,
                        help="Accelerate placement map; use auto for TITAN V CPU offload.")
    parser.add_argument(
        "--pre-sft-gpu-weight-budget",
        default="5GiB",
        help="With pre_sft_base_vlm + device_map=auto, cap dispatched model weights to preserve TITAN-V forward headroom.",
    )
    parser.add_argument(
        "--pre-sft-cpu-offload-budget",
        default="45GiB",
        help="CPU memory budget used by the pre-SFT base auto-dispatch path.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--runtime-root", default=str(REPO_ROOT / ".offline_runtime"))
    parser.add_argument("--siglip-path", default=None)
    parser.add_argument("--cut3r-weights", default=None)
    parser.add_argument("--skip-spatial-tower-load", type=str2bool, default=None)
    parser.add_argument("--allow-euclidean-depth", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--assert-first-video",
        action="store_true",
        help="Fail before continuing unless the first forward matches the selected model's runtime contracts.",
    )
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--geometry-on-off-split",
        choices=["train", "val", "dev_eval", "confirmation"],
        default=None,
        help="Run the training-free native SpatialStack ON/OFF diagnostic only for this manifest split.",
    )
    parser.add_argument(
        "--geometry-on-off-tolerance",
        type=float,
        default=1e-6,
        help="Maximum RMS ON/OFF difference allowed before the first SpatialStack injection.",
    )
    args = parser.parse_args()

    if args.model_loading_mode == "pre_sft_base_vlm" and args.model_label != "pre_sft_base_vlm":
        parser.error("--model-loading-mode pre_sft_base_vlm requires --model-label pre_sft_base_vlm")
    if args.model_loading_mode == "pre_sft_fusion":
        if args.pre_sft_fusion_variant is None:
            parser.error("--model-loading-mode pre_sft_fusion requires --pre-sft-fusion-variant")
        c1_variant = str(args.pre_sft_fusion_variant).startswith("c1_")
        if not c1_variant and args.fusion_init_seed is None:
            parser.error("--model-loading-mode pre_sft_fusion requires --fusion-init-seed")
        if c1_variant and not args.c1_calibration_json:
            parser.error("C1 extraction requires --c1-calibration-json")
        if args.c1_calibration_json and not c1_variant:
            parser.error("--c1-calibration-json is valid only with a c1_* fusion variant")

    if bool(args.forward_frames_root) != bool(args.probe_targets_root):
        parser.error("--forward-frames-root and --probe-targets-root must be supplied together")
    if args.layers is not None and args.llm_layers is not None:
        parser.error("Use only one of --layers or --llm-layers")

    if args.model_path is None:
        if args.model_label not in MODEL_PRESETS:
            parser.error(f"--model-path is required for unknown model label {args.model_label!r}")
        args.model_path = MODEL_PRESETS[args.model_label]
    args.feature_preset = feature_preset_for_model(args.model_label, args.feature_preset)
    feature_level_override = parse_feature_names(args.feature_levels)
    if feature_level_override is not None:
        if args.layers is not None or args.llm_layers is not None:
            parser.error("--feature-levels cannot be combined with --layers or --llm-layers")
        pre_llm_override: list[str] = []
        llm_layer_override: list[int] = []
        for feature_level in feature_level_override:
            if feature_level.startswith("layer_"):
                llm_layer_override.append(int(feature_level.removeprefix("layer_")))
            else:
                pre_llm_override.append(feature_level)
    else:
        pre_llm_override = parse_feature_names(args.pre_llm_features)
        llm_layer_override = validate_llm_layers(args.layers) if args.layers is not None else parse_llm_layers(args.llm_layers)
    args.pre_llm_feature_names = pre_llm_features_for_model(
        args.model_label,
        args.feature_preset,
        pre_llm_override,
    )
    if feature_level_override is not None and not llm_layer_override:
        args.llm_layers = []
    else:
        args.llm_layers = validate_llm_layers(
            llm_layers_for_model(args.model_label, args.feature_preset, llm_layer_override)
        )
    args.spatial_feature_dir = args.feature_root
    args.zero_spatial_features = args.feature_preset == "zero_spatial"
    if args.skip_spatial_tower_load is None:
        args.skip_spatial_tower_load = True

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    try:
        torch._dynamo.disable()
    except Exception:
        pass

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "features" / args.model_label / "extraction_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_dtype = torch_dtype_from_name(args.dtype)
    cache_dtype = torch_dtype_from_name(args.cache_dtype)

    print(f"[INFO] Loading model {args.model_label}: {args.model_path}")
    print(
        f"[INFO] Feature preset={args.feature_preset} pre_llm={args.pre_llm_feature_names} "
        f"llm_layers={args.llm_layers}",
        flush=True,
    )
    tokenizer, model, image_processor = load_model(args, device, model_dtype)
    c1_artifact = None
    if args.c1_calibration_json:
        calibration_path = Path(args.c1_calibration_json).resolve()
        with calibration_path.open("r", encoding="utf-8") as f:
            c1_artifact = json.load(f)
        apply_c1_calibration_artifact(model, c1_artifact)
    configured_layers = getattr(model.config, "num_hidden_layers", None)
    if args.llm_layers:
        args.llm_layers = validate_llm_layers(args.llm_layers, num_hidden_layers=configured_layers)
    args.feature_provenance = {
        "model_label": args.model_label,
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_config_sha256": sha256_file(Path(args.model_path) / "config.json"),
        "sample_indices": str(Path(args.sample_indices).resolve()),
        "sample_indices_sha256": sha256_file(Path(args.sample_indices)),
        "requested_llm_layers": list(args.llm_layers),
        "requested_pre_llm_features": list(args.pre_llm_feature_names),
        "requested_feature_levels": list(args.pre_llm_feature_names) + [f"layer_{layer}" for layer in args.llm_layers],
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
        "forward_frames_root": str(Path(args.forward_frames_root).resolve()) if args.forward_frames_root else None,
        "probe_targets_root": str(Path(args.probe_targets_root).resolve()) if args.probe_targets_root else None,
        "feature_root": str(Path(args.feature_root).resolve()) if args.feature_root else None,
        "model_loading_mode": args.model_loading_mode,
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
        "first_video_runtime_assertions_required": bool(args.assert_first_video),
        "seed": args.seed,
        "command": [sys.executable, *sys.argv],
        **git_metadata(),
    }
    no_pre_llm_fusion_module = (
        args.model_loading_mode == "pre_sft_fusion"
        and getattr(model.get_model(), "get_fusion_block", lambda: None)() is None
    )
    args.pre_llm_representation_definitions = {
        "siglip_output": "SigLipVisionTower.forward output before fusion and mm_projector",
        "fusion_output": (
            "mm_projector input; additive SpatialStack has no pre-projector fusion module"
            if no_pre_llm_fusion_module
            else "configured fusion-block output immediately before mm_projector"
        ),
        "projected_features": "mm_projector output after the model's configured fusion path",
    }
    if args.model_loading_mode == "pre_sft_fusion":
        args.feature_provenance.update(
            {
                "experiment_variant": args.pre_sft_fusion_variant,
                "fusion_init_seed": int(args.fusion_init_seed or 0),
                "common_model_init_seed": int(args.common_model_init_seed),
                "spatialstack_output_init": (
                    "identity" if args.pre_sft_fusion_variant == "ss_identity"
                    else "zero" if args.pre_sft_fusion_variant == "ss_zero" else None
                ),
                "shared_llm_layers": list(args.llm_layers),
                "c1_calibration_json": str(Path(args.c1_calibration_json).resolve()) if args.c1_calibration_json else None,
                "c1_calibration_sha256": sha256_file(Path(args.c1_calibration_json)) if args.c1_calibration_json else None,
                "c1_artifact_architecture": c1_artifact.get("architecture") if c1_artifact else None,
            }
        )
    if args.pre_llm_feature_names:
        args.feature_provenance["pre_llm_representation_definitions"] = dict(
            args.pre_llm_representation_definitions
        )
    if args.model_label == "zero_spatial" and args.pre_llm_feature_names:
        args.feature_provenance["zero_spatial_post_fusion_projector_contract"] = {
            "fusion_block": str(getattr(model.config, "fusion_block", "")),
            "spatial_tower": str(
                getattr(model.config, "spatial_tower", getattr(model.config, "mm_spatial_tower", ""))
            ),
            "siglip_output": "SigLipVisionTower.forward output before zero-spatial fusion and mm_projector",
            "projected_features": (
                "mm_projector output after zero-spatial fusion path; verified by fusion output == mm_projector input"
            ),
        }
    if args.model_loading_mode == "adapter":
        args.feature_provenance["adapter_config_sha256"] = sha256_file(Path(args.model_path) / "adapter_config.json")
    else:
        args.feature_provenance.update(
            {
                "base_model_path": str(Path(args.model_path).resolve()),
                "base_model_config_sha256": sha256_file(Path(args.model_path) / "config.json"),
                "siglip_path": str(Path(args.siglip_path).resolve()) if args.siglip_path else None,
                "siglip_config_sha256": sha256_file(Path(args.siglip_path) / "config.json") if args.siglip_path else None,
                "no_vlm3r_sft_adapter_loaded": True,
                "no_cut3r_or_spatial_sidecar_usage": args.model_loading_mode == "pre_sft_base_vlm",
                "target_semantics": "point_maps_cam -> camera_z",
            }
        )
    if args.zero_spatial_features:
        model.config.zero_spatial_features = True
    if args.forward_frames_root:
        if args.model_loading_mode == "pre_sft_base_vlm":
            args.feature_provenance["base_forward_contract"] = assert_pre_sft_base_vlm_forward_contract(
                model, Path(args.model_path)
            )
            args.feature_provenance["placement"] = model_placement_metadata(model)
            base_model = model.get_model()
            args.feature_provenance["materialized_parameter_dtypes"] = {
                "llm_decoder_layers": module_dtype_metadata(getattr(base_model, "layers", None)),
                "vision_tower": module_dtype_metadata(model.get_vision_tower()),
                "mm_projector": module_dtype_metadata(getattr(base_model, "mm_projector", None)),
            }
        elif args.model_loading_mode == "adapter":
            assert_baseline_or_zero_spatial_forward_contract(model)
        if args.model_loading_mode in {"pre_sft_base_vlm", "pre_sft_fusion"}:
            args.feature_provenance["placement"] = model_placement_metadata(model)
        install_forward_frame_loader(Path(args.forward_frames_root))
    model.eval()

    print("[INFO] Building dataset")
    dataset, collator, by_video = build_dataset(args, tokenizer, image_processor)
    if args.shard_count < 1:
        raise ValueError(f"--shard-count must be >= 1, got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError(
            f"--shard-index must be in [0, shard_count), got {args.shard_index}/{args.shard_count}"
        )

    sample_payload = read_json(Path(args.sample_indices))
    all_videos = list(sample_payload.get("videos", []))
    if args.limit_videos is not None:
        all_videos = all_videos[: int(args.limit_videos)]
    videos = [
        video
        for global_index, video in enumerate(all_videos)
        if global_index % int(args.shard_count) == int(args.shard_index)
    ]
    print(
        f"[INFO] Shard {args.shard_index}/{args.shard_count}: "
        f"{len(videos)}/{len(all_videos)} videos selected from fixed sample_indices",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    captured: dict[str, torch.Tensor] = {}
    extraction_samples: list[dict[str, Any]] = []
    first_runtime_video_seen = False
    fusion_init_diagnostic_logged = False
    handles = register_pre_llm_hooks(model, args.model_label, args.pre_llm_feature_names, captured)
    try:
        with log_path.open("a", encoding="utf-8") as log_f:
            for idx, video in enumerate(videos):
                video_path = str(video["video_path"])
                selected_frames = [int(frame["frame_index"]) for frame in video["frames"]]
                if args.resume and all(
                    output_complete(
                        output_root,
                        args.model_label,
                        str(frame["frame_sample_id"]),
                        args.pre_llm_feature_names,
                        args.llm_layers,
                    )
                    for frame in video["frames"]
                ) and (
                    args.geometry_on_off_split is None
                    or str(video.get("split", "")) != str(args.geometry_on_off_split)
                    or geometry_on_off_path(output_root, args.model_label, video).is_file()
                ):
                    print(f"[SKIP] {idx + 1}/{len(videos)} {video_path} already complete")
                    continue
                if video_path not in by_video:
                    payload = {"ok": False, "video_path": video_path, "error": "video not found in dataset"}
                    print(json.dumps(payload), file=log_f, flush=True)
                    print(f"[WARN] {payload}", file=sys.stderr)
                    continue
                try:
                    print(f"[INFO] {idx + 1}/{len(videos)} extracting {video_path} frames={selected_frames}")
                    first_attempt = bool(args.assert_first_video and not first_runtime_video_seen)
                    sample = extract_for_video(
                        args=args,
                        model=model,
                        collator=collator,
                        dataset=dataset,
                        dataset_index=by_video[video_path],
                        video_record=video,
                        selected_frames=selected_frames,
                        captured=captured,
                        cache_dtype=cache_dtype,
                        device=device,
                        model_dtype=model_dtype,
                        assert_runtime=first_attempt,
                    )
                    extraction_samples.append(sample)
                    if first_attempt:
                        first_runtime_video_seen = True
                        print(
                            f"[ASSERTION PASS] first {args.model_loading_mode} video runtime contract: "
                            + json.dumps(sample.get("first_video_runtime_assertions"), sort_keys=True),
                            flush=True,
                        )
                    if (
                        args.model_loading_mode == "pre_sft_fusion"
                        and sample["spatialstack_insertion_stats"]
                        and not fusion_init_diagnostic_logged
                    ):
                        print(
                            "[SPATIALSTACK_INIT] "
                            + json.dumps(sample["spatialstack_insertion_stats"], sort_keys=True),
                            flush=True,
                        )
                        fusion_init_diagnostic_logged = True
                    print(json.dumps({"ok": True, "video_path": video_path, "frames": selected_frames}), file=log_f, flush=True)
                except Exception as exc:
                    payload = {"ok": False, "video_path": video_path, "frames": selected_frames, "error": str(exc)}
                    print(json.dumps(payload), file=log_f, flush=True)
                    print(f"[ERROR] {payload}", file=sys.stderr)
                    if args.assert_first_video and not first_runtime_video_seen:
                        raise RuntimeError(
                            "Fail-fast first video runtime assertion/forward failure: "
                            f"{video_path}: {exc}"
                        ) from exc
    finally:
        for handle in handles:
            handle.remove()
    if args.assert_first_video and not first_runtime_video_seen:
        raise RuntimeError("--assert-first-video was requested, but no video completed a forward pass.")
    if args.model_loading_mode in {"pre_sft_base_vlm", "pre_sft_fusion"}:
        if device.type == "cuda":
            args.feature_provenance["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            args.feature_provenance["cuda_peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    if args.model_loading_mode in {"pre_sft_base_vlm", "pre_sft_fusion"} or args.assert_first_video or args.pre_llm_feature_names:
        args.feature_provenance["extraction_samples"] = extraction_samples
        args.feature_provenance["runtime_dtype_summary"] = summarize_runtime_dtypes(extraction_samples)
        run_provenance = output_root / "features" / args.model_label / "extraction_provenance.json"
        with run_provenance.open("w", encoding="utf-8") as f:
            json.dump(args.feature_provenance, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
