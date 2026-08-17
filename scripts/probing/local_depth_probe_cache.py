"""Local migrated-input helpers for the ScanNet depth-probe experiment.

This module is intentionally opt-in.  It preserves the existing
``LazySupervisedDataset`` path and replaces only its MP4 decoder with the
lossless 32-frame RGB cache created from the historical preprocessing path.
Compact two-frame targets are loaded separately and are never supplied to the
model as forward geometry.
"""

from __future__ import annotations

import sys
import types
import importlib.machinery
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORWARD_CACHE_SCHEMA = "forward_frames_32_v1"


def _video_identity(video_path: str) -> tuple[str, str, str]:
    """Return dataset, scene id, and canonical relative video path."""
    path = Path(video_path)
    parts = path.parts
    for dataset in ("scannet", "scannetpp", "arkitscenes"):
        try:
            index = parts.index(dataset)
        except ValueError:
            continue
        if index + 2 < len(parts) and parts[index + 1] == "videos":
            filename = parts[index + 2]
            if Path(filename).suffix.lower() != ".mp4":
                raise ValueError(f"Expected an MP4 name in {video_path!r}")
            return dataset, Path(filename).stem, f"{dataset}/videos/{filename}"
    raise ValueError(f"Could not resolve dataset/video identity from {video_path!r}")


def forward_cache_path(forward_frames_root: Path, video_path: str) -> Path:
    dataset, scene_id, _ = _video_identity(video_path)
    return Path(forward_frames_root) / "frames" / dataset / f"{scene_id}.pt"


def compact_target_path(probe_targets_root: Path, video_path: str) -> Path:
    dataset, scene_id, _ = _video_identity(video_path)
    return Path(probe_targets_root) / "targets" / dataset / "spatial_features_points" / f"{scene_id}.pt"


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_forward_frames(forward_frames_root: Path, video_path: str) -> dict[str, Any]:
    """Load and validate one lossless 32-frame RGB cache payload."""
    path = forward_cache_path(forward_frames_root, video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing 32-frame cache for {video_path}: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected cache dict at {path}, got {type(payload)}")
    dataset, scene_id, canonical_video = _video_identity(video_path)
    if payload.get("schema_version") != FORWARD_CACHE_SCHEMA:
        raise ValueError(f"{path}: unexpected schema {payload.get('schema_version')!r}")
    if payload.get("dataset") != dataset or payload.get("scene_id") != scene_id:
        raise ValueError(f"{path}: cache identity does not match {video_path}")
    if payload.get("source_video_relative_path") != canonical_video:
        raise ValueError(f"{path}: source video mismatch for {video_path}")
    frames = payload.get("frames_rgb_uint8")
    source_indices = payload.get("source_frame_indices")
    if not isinstance(frames, torch.Tensor) or frames.dtype != torch.uint8 or frames.ndim != 4:
        raise ValueError(f"{path}: expected uint8 [32,H,W,3] frames")
    if frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise ValueError(f"{path}: expected 32 RGB frames, got {tuple(frames.shape)}")
    if not isinstance(source_indices, torch.Tensor) or tuple(source_indices.shape) != (32,):
        raise ValueError(f"{path}: expected 32 source frame indices")
    if not bool(torch.all(source_indices[1:] >= source_indices[:-1])):
        raise ValueError(f"{path}: source frame indices are not ordered")
    return payload


def install_forward_frame_loader(forward_frames_root: Path) -> None:
    """Patch only the dataset module's decoder for this process.

    ``LazySupervisedDataset`` imported ``process_video_with_decord`` into its
    module namespace, so patching that bound symbol leaves the processor,
    prompts, collator, and all later model input handling unchanged.
    """
    # ``llava.train.train`` imports DeepSpeed even though the extraction path
    # only needs its dataset/collator definitions.  The migrated environment
    # has a working PyTorch CUDA runtime but no CUDA toolkit (``CUDA_HOME``),
    # so importing DeepSpeed would fail while probing its optional build
    # extensions.  Keep this workaround local to the cache adapter; training
    # entry points continue to use the real dependency.
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    if CUDA_HOME is None and "deepspeed" not in sys.modules:
        deepspeed_stub = types.ModuleType("deepspeed")
        deepspeed_stub.__path__ = []
        # Transformers checks ``find_spec`` while importing its Trainer
        # helpers.  A real-looking spec lets that optional integration report
        # availability from package metadata without importing DeepSpeed's
        # CUDA-op probes.
        deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec(
            "deepspeed", loader=None, is_package=True
        )
        sys.modules["deepspeed"] = deepspeed_stub
    from llava.train import train as train_module
    from llava import utils as utils_module

    cache_root = Path(forward_frames_root)

    def process_cached_video(video_file: str, _data_args: Any):
        payload = load_forward_frames(cache_root, video_file)
        frames = payload["frames_rgb_uint8"].contiguous().numpy()
        if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
            raise TypeError(f"Cached RGB conversion failed for {video_file}")
        video_time = float(payload["video_time_seconds"])
        frame_time = str(payload["frame_time_string"])
        return frames, video_time, frame_time, 32

    # The dataset holds a direct import, while a few legacy call paths resolve
    # it through llava.utils. Patch both names in this process; neither path
    # may invoke an MP4 decoder for the migrated cache.
    train_module.process_video_with_decord = process_cached_video
    utils_module.process_video_with_decord = process_cached_video


def load_selected_camera_depths(
    probe_targets_root: Path,
    video_record: dict[str, Any],
    selected_frames: list[int],
) -> tuple[dict[int, torch.Tensor], Path, dict[str, Any]]:
    """Load compact camera-space targets and bind their order to manifest IDs."""
    video_path = str(video_record["video_path"])
    path = compact_target_path(Path(probe_targets_root), video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing compact probe target for {video_path}: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected target dict at {path}, got {type(payload)}")
    dataset, scene_id, canonical_video = _video_identity(video_path)
    if payload.get("dataset") != dataset:
        raise ValueError(f"{path}: target dataset does not match manifest video {video_path}")
    if payload.get("source_video_relative_path") != canonical_video:
        raise ValueError(f"{path}: target video does not match manifest video {video_path}")
    source_sidecar = payload.get("source_sidecar_relative_path")
    if not isinstance(source_sidecar, str) or Path(source_sidecar).name != f"{scene_id}.pt":
        raise ValueError(f"{path}: target sidecar provenance does not match {video_path}")
    selected = payload.get("selected_frame_indices")
    point_maps = payload.get("point_maps_cam")
    if not isinstance(selected, torch.Tensor) or selected.ndim != 1:
        raise ValueError(f"{path}: missing selected_frame_indices tensor")
    if not isinstance(point_maps, torch.Tensor) or point_maps.ndim != 4:
        raise ValueError(f"{path}: missing point_maps_cam [2,H,W,3] tensor")
    target_indices = [int(value) for value in selected.detach().cpu().tolist()]
    if target_indices != [int(value) for value in selected_frames]:
        raise ValueError(
            f"{path}: selected target indices {target_indices} do not match manifest {selected_frames}"
        )
    if point_maps.shape[0] != len(selected_frames) or point_maps.shape[-1] != 3:
        raise ValueError(f"{path}: invalid point_maps_cam shape {tuple(point_maps.shape)}")
    # Keep camera-space depth construction identical to the historical probe:
    # select the camera point-map representation then delegate z-depth to the
    # shared probe helper.  The compact bundle is target-only and is never
    # attached to the model-forward batch.
    from depth_probe_common import depth_from_point_maps, select_point_maps

    selected_maps, point_key, depth_mode = select_point_maps(payload, allow_euclidean_depth=False)
    if point_key != "point_maps_cam" or depth_mode != "camera_z":
        raise ValueError(f"{path}: expected camera-space point_maps_cam target, got {point_key}/{depth_mode}")
    depths = depth_from_point_maps(selected_maps, depth_mode)
    return {frame_index: depths[index] for index, frame_index in enumerate(target_indices)}, path, payload


def assert_baseline_or_zero_spatial_forward_contract(model: torch.nn.Module) -> None:
    """Reject checkpoints that need full point-map geometry at model forward."""
    config = model.config
    fusion_block = str(getattr(config, "fusion_block", "") or "").strip().lower()
    spatial_tower = str(getattr(config, "spatial_tower", getattr(config, "mm_spatial_tower", "")) or "").lower()
    if fusion_block != "cross_attention" or "cut3r" not in spatial_tower:
        raise RuntimeError(
            "The local compact-target adapter supports only the verified CUT3R cross_attention "
            f"forward contract, got fusion_block={fusion_block!r}, spatial_tower={spatial_tower!r}."
        )


def pre_sft_projector_loading_evidence(model: torch.nn.Module) -> dict[str, Any]:
    """Return auditable evidence that the base checkpoint restored mm_projector."""
    base = model.get_model()
    projector = getattr(base, "mm_projector", None)
    if projector is None:
        raise RuntimeError("pre_sft_base_vlm has no mm_projector.")
    loading_info = getattr(model, "_pre_sft_loading_info", None)
    if not isinstance(loading_info, dict):
        raise RuntimeError("pre_sft_base_vlm loader did not retain Hugging Face loading metadata.")
    missing = [str(key) for key in loading_info.get("missing_keys", [])]
    unexpected = [str(key) for key in loading_info.get("unexpected_keys", [])]
    mismatched = [str(item) for item in loading_info.get("mismatched_keys", [])]
    projector_names = [f"model.mm_projector.{name}" for name, _ in projector.named_parameters()]
    projector_missing = [key for key in missing if "mm_projector" in key]
    projector_mismatched = [key for key in mismatched if "mm_projector" in key]
    if projector_missing or projector_mismatched:
        raise RuntimeError(
            "Plain base checkpoint did not fully restore pretrained mm_projector weights: "
            f"missing={projector_missing}, mismatched={projector_mismatched}"
        )
    return {
        "projector_class": f"{projector.__class__.__module__}.{projector.__class__.__name__}",
        "projector_parameter_names": projector_names,
        "projector_weight_source": str(getattr(model, "_pre_sft_source_path", "")),
        "projector_missing_keys": projector_missing,
        "projector_mismatched_keys": projector_mismatched,
        "all_missing_keys": missing,
        "all_unexpected_keys": unexpected,
        "all_mismatched_keys": mismatched,
    }


def assert_pre_sft_base_vlm_forward_contract(model: torch.nn.Module, base_model_path: Path) -> dict[str, Any]:
    """Reject any VLM-3R/SFT architecture from the plain-base control."""
    source = Path(str(getattr(model, "_pre_sft_source_path", ""))).resolve()
    expected = Path(base_model_path).resolve()
    if source != expected:
        raise RuntimeError(f"Plain base source mismatch: loaded={source}, expected={expected}")
    if any((expected / name).exists() for name in ("adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin")):
        raise RuntimeError(f"Plain base source unexpectedly contains adapter artifacts: {expected}")
    if model.__class__.__name__ != "LlavaQwenForCausalLM":
        raise RuntimeError(f"Expected LlavaQwenForCausalLM, got {model.__class__.__name__}")
    base = model.get_model()
    if getattr(base, "mm_projector", None) is None or model.get_vision_tower() is None:
        raise RuntimeError("Plain base forward requires normal SigLIP vision tower and mm_projector.")
    if model.get_spatial_tower() is not None or model.get_fusion_block() is not None:
        raise RuntimeError("pre_sft_base_vlm must not contain a spatial tower or fusion block.")
    forbidden_flags = (
        "use_cut3r_spatialstack",
        "use_geometry_aware_projection",
        "use_cut3r_camera_tokens",
        "use_spatial_bridge_tokens",
        "llm_visual_3d_rope_enable",
        "use_bev_supervision",
        "use_depth_supervision",
        "use_pointmap_supervision",
    )
    enabled = [name for name in forbidden_flags if bool(getattr(model.config, name, False))]
    if enabled:
        raise RuntimeError(f"pre_sft_base_vlm unexpectedly enables VLM-3R spatial options: {enabled}")
    evidence = pre_sft_projector_loading_evidence(model)
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "base_model_path": str(expected),
        "vision_tower_class": f"{model.get_vision_tower().__class__.__module__}.{model.get_vision_tower().__class__.__name__}",
        "no_vlm3r_sft_adapter_loaded": True,
        "no_cut3r_or_spatial_sidecar_usage": True,
        "projector_loading_evidence": evidence,
    }
