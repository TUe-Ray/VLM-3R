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
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} hook expected tensor output, got {type(value)}")
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
            raise RuntimeError(f"{model_label} requested fusion_output, but get_fusion_block() returned None.")
        handles.append(fusion_block.register_forward_hook(capture_hook("fusion_output", captured)))
    if "projected_features" in feature_names:
        projector = getattr(base, "mm_projector", None)
        if projector is None:
            raise RuntimeError(f"{model_label} requested projected_features, but base model has no mm_projector.")
        handles.append(projector.register_forward_hook(capture_hook("projected_features", captured)))
    return handles


def normalize_captured_video_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    num_frames: int,
    target_grid_shape: tuple[int, int],
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
    if int(tensor.shape[1]) == target_tokens:
        return tensor

    pooled = None
    get_2d_pool = getattr(model, "get_2dPool", None)
    if callable(get_2d_pool):
        try:
            pooled = get_2d_pool(tensor)
        except Exception:
            pooled = None
    if pooled is not None and int(pooled.shape[1]) == target_tokens:
        return pooled.float()

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
    batch = move_to_device(batch, device, model_dtype)
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
    if args.model_loading_mode == "pre_sft_base_vlm":
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
        base_vision_tower.register_forward_hook(vision_dtype_hook)
        if args.model_loading_mode == "pre_sft_base_vlm" and base_vision_tower is not None
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
    if args.model_loading_mode == "pre_sft_base_vlm":
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
    if args.model_loading_mode == "pre_sft_base_vlm":
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
    for feature_name in args.pre_llm_feature_names:
        if feature_name not in captured:
            raise RuntimeError(f"{feature_name} hook did not capture an output")
        normalized_pre_llm[feature_name] = normalize_captured_video_tokens(
            model,
            captured[feature_name],
            num_frames=num_frames,
            target_grid_shape=target_grid_shape,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-loading-mode", choices=["adapter", "pre_sft_base_vlm"], default="adapter")
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
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_loading_mode == "pre_sft_base_vlm" and args.model_label != "pre_sft_base_vlm":
        parser.error("--model-loading-mode pre_sft_base_vlm requires --model-label pre_sft_base_vlm")

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
    args.llm_layers = validate_llm_layers(llm_layers_for_model(args.model_label, args.feature_preset, llm_layer_override))
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
    configured_layers = getattr(model.config, "num_hidden_layers", None)
    args.llm_layers = validate_llm_layers(args.llm_layers, num_hidden_layers=configured_layers)
    args.feature_provenance = {
        "model_label": args.model_label,
        "model_path": str(Path(args.model_path).resolve()),
        "checkpoint_config_sha256": sha256_file(Path(args.model_path) / "config.json"),
        "sample_indices": str(Path(args.sample_indices).resolve()),
        "sample_indices_sha256": sha256_file(Path(args.sample_indices)),
        "requested_llm_layers": list(args.llm_layers),
        "hidden_state_indexing": "requested_L -> hidden_states[L + 1]",
        "forward_frames_root": str(Path(args.forward_frames_root).resolve()) if args.forward_frames_root else None,
        "probe_targets_root": str(Path(args.probe_targets_root).resolve()) if args.probe_targets_root else None,
        "model_loading_mode": args.model_loading_mode,
        "dtype": args.dtype,
        "cache_dtype": args.cache_dtype,
        "seed": args.seed,
        "command": [sys.executable, *sys.argv],
        **git_metadata(),
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
                "no_cut3r_or_spatial_sidecar_usage": True,
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
        else:
            assert_baseline_or_zero_spatial_forward_contract(model)
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
                    extraction_samples.append(extract_for_video(
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
                    ))
                    print(json.dumps({"ok": True, "video_path": video_path, "frames": selected_frames}), file=log_f, flush=True)
                except Exception as exc:
                    payload = {"ok": False, "video_path": video_path, "frames": selected_frames, "error": str(exc)}
                    print(json.dumps(payload), file=log_f, flush=True)
                    print(f"[ERROR] {payload}", file=sys.stderr)
    finally:
        for handle in handles:
            handle.remove()
    if args.model_loading_mode == "pre_sft_base_vlm":
        if device.type == "cuda":
            args.feature_provenance["cuda_peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            args.feature_provenance["cuda_peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        args.feature_provenance["extraction_samples"] = extraction_samples
        args.feature_provenance["runtime_dtype_summary"] = summarize_runtime_dtypes(extraction_samples)
        run_provenance = output_root / "features" / args.model_label / "extraction_provenance.json"
        with run_provenance.open("w", encoding="utf-8") as f:
            json.dump(args.feature_provenance, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
