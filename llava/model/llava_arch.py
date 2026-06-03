#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_spatial_encoder.builder import build_spatial_tower
from .multimodal_fusion_block.builder import build_multimodal_fusion_block
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .geometry import MetricGroundedGeometryProjection, canonicalize_geometry_outputs
from .pi3x_decoded_features import Pi3XDecodedFeatures
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from einops import rearrange

import torch.nn.functional as F
import numpy as np
import cv2  # OpenCV for resizing and writing images
import matplotlib.cm as cm # For colormaps
import os


def _as_long_tensor(values, device):
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)
    return torch.tensor(values, device=device, dtype=torch.long)


def _empty_long(device):
    return torch.empty(0, device=device, dtype=torch.long)


def _as_bool_config(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _geometry_projection_hidden_size(module):
    if module is None:
        return None
    if hasattr(module, "hidden_size"):
        return int(module.hidden_size)
    if hasattr(module, "layers") and len(module.layers) > 0:
        return int(module.layers[0].hidden_size)
    return None


def pool_point_map_to_tokens(point_maps, target_num_tokens):
    """
    Pool dense point maps to a square token grid.

    Args:
        point_maps: [B, H, W, 3] or [B, 3, H, W]
        target_num_tokens: square token count N
    Returns:
        pos_tokens: [B, N, 3]
    """
    if not isinstance(point_maps, torch.Tensor):
        raise TypeError(f"point_maps must be a torch.Tensor, got {type(point_maps)}")
    if point_maps.dim() != 4:
        raise ValueError(f"point_maps must be rank 4, got shape {tuple(point_maps.shape)}")

    side = int(math.isqrt(int(target_num_tokens)))
    if side * side != int(target_num_tokens):
        raise ValueError(f"target_num_tokens must be square, got {target_num_tokens}")

    if point_maps.shape[-1] == 3:
        pm = point_maps.permute(0, 3, 1, 2)
    elif point_maps.shape[1] == 3:
        pm = point_maps
    else:
        raise ValueError(f"point_maps must be [B,H,W,3] or [B,3,H,W], got {tuple(point_maps.shape)}")

    pm = pm.float()
    pm = F.interpolate(pm, size=(side, side), mode="bilinear", align_corners=False)
    return pm.permute(0, 2, 3, 1).reshape(pm.shape[0], side * side, 3)


def _coalesce_point_maps(point_maps):
    if point_maps is None:
        return None
    if isinstance(point_maps, torch.Tensor):
        return point_maps
    if isinstance(point_maps, (list, tuple)):
        if len(point_maps) == 0:
            return None
        tensors = []
        for item in point_maps:
            if item is None:
                continue
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"point_maps entries must be torch.Tensor, got {type(item)}")
            tensors.append(item)
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)
    raise TypeError(f"point_maps must be a tensor or list/tuple of tensors, got {type(point_maps)}")


def _probe_log_index(config, attr_name, limit=5):
    count = int(getattr(config, attr_name, 0))
    setattr(config, attr_name, count + 1)
    return count if count < int(limit) else None


def _maybe_shuffle_geometry_point_maps(config, geometry_point_maps, training):
    if training or not _as_bool_config(getattr(config, "probe_geometry_shuffle", False)):
        return geometry_point_maps

    frame_count = int(geometry_point_maps.shape[0])
    log_index = _probe_log_index(config, "_probe_geometry_shuffle_log_count", limit=5)
    if frame_count <= 1:
        if log_index is not None:
            rank0_print("[Probe G-Shuffle] Skip shuffle because F <= 1")
        return geometry_point_maps

    mode = str(getattr(config, "probe_geometry_shuffle_mode", "cyclic_shift"))
    device = geometry_point_maps.device
    frame_ids = torch.arange(frame_count, device=device)

    if mode == "cyclic_shift":
        shift = int(getattr(config, "probe_geometry_shuffle_shift", 1))
        shift = shift % frame_count
        if shift == 0:
            shift = 1
        perm = torch.roll(frame_ids, shifts=shift, dims=0)
    elif mode == "reverse":
        perm = torch.arange(frame_count - 1, -1, -1, device=device)
    elif mode == "random_derange":
        seed = int(getattr(config, "probe_geometry_shuffle_seed", 0))
        generator_device = device if device.type == "cuda" else "cpu"
        g = torch.Generator(device=generator_device)
        g.manual_seed(seed)
        perm = torch.randperm(frame_count, generator=g, device=device)
        if torch.equal(perm, frame_ids):
            perm = torch.roll(perm, shifts=1, dims=0)
    else:
        raise ValueError(f"Unknown probe_geometry_shuffle_mode: {mode}")

    if log_index is not None:
        rank0_print(
            "[Probe G-Shuffle] "
            f"sample_log_index={log_index}, mode={mode}, F={frame_count}, "
            f"perm={perm.detach().cpu().tolist()}"
        )
    elif int(getattr(config, "_probe_geometry_shuffle_log_count", 0)) == 6:
        rank0_print("[Probe G-Shuffle] Further per-sample permutation logs suppressed.")

    return geometry_point_maps[perm]


def _spatial_frame_swap_permutation(frame_count, device, mode, seed):
    frame_ids = torch.arange(frame_count, device=device)

    if mode == "cyclic_shift":
        shift = seed % frame_count
        if shift == 0:
            shift = 1
        return torch.roll(frame_ids, shifts=shift, dims=0)
    if mode == "reverse":
        return torch.arange(frame_count - 1, -1, -1, device=device)

    generator_device = device if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(frame_count, generator=generator, device=device)
    if mode == "random_permutation":
        return perm
    if mode != "random_derange":
        raise ValueError(f"Unknown probe_spatial_feature_frame_swap_mode: {mode}")

    for _ in range(16):
        if not torch.any(perm == frame_ids):
            return perm
        perm = torch.randperm(frame_count, generator=generator, device=device)

    fixed_idx = torch.nonzero(perm == frame_ids, as_tuple=False).flatten()
    if int(fixed_idx.numel()) == frame_count:
        return torch.roll(perm, shifts=1, dims=0)
    if int(fixed_idx.numel()) == 1:
        idx = fixed_idx[0]
        swap_idx = (idx + 1) % frame_count
        tmp = perm[idx].clone()
        perm[idx] = perm[swap_idx]
        perm[swap_idx] = tmp
    elif int(fixed_idx.numel()) > 1:
        perm[fixed_idx] = torch.roll(perm[fixed_idx], shifts=1, dims=0)
    return perm


def _maybe_swap_spatial_frame_features(config, camera_tokens, patch_tokens, camera_pose, training):
    if training or not _as_bool_config(getattr(config, "probe_spatial_feature_frame_swap", False), False):
        return camera_tokens, patch_tokens, camera_pose

    if camera_tokens is None or patch_tokens is None:
        raise RuntimeError("Spatial frame-swap probe requires both camera_tokens and patch_tokens.")

    frame_count = int(camera_tokens.shape[0])
    if int(patch_tokens.shape[0]) != frame_count:
        raise RuntimeError(
            "Spatial frame-swap probe requires camera_tokens and patch_tokens to have the same "
            f"frame count, got {tuple(camera_tokens.shape)} and {tuple(patch_tokens.shape)}."
        )
    if camera_pose is not None and int(camera_pose.shape[0]) != frame_count:
        raise RuntimeError(
            "Spatial frame-swap probe requires camera_pose to match token frame count, "
            f"got camera_pose={tuple(camera_pose.shape)} and F={frame_count}."
        )

    log_index = _probe_log_index(config, "_probe_spatial_feature_frame_swap_log_count", limit=5)
    if frame_count <= 1:
        if log_index is not None:
            rank0_print("[Probe Spatial Frame Swap] Skip swap because F <= 1")
        return camera_tokens, patch_tokens, camera_pose

    sample_index = int(getattr(config, "_probe_spatial_feature_frame_swap_count", 0))
    setattr(config, "_probe_spatial_feature_frame_swap_count", sample_index + 1)
    base_seed = int(getattr(config, "probe_spatial_feature_frame_swap_seed", 0) or 0)
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    seed = base_seed + sample_index + rank * 1000003
    mode = str(getattr(config, "probe_spatial_feature_frame_swap_mode", "random_derange") or "random_derange")
    perm = _spatial_frame_swap_permutation(frame_count, camera_tokens.device, mode, seed)

    if log_index is not None:
        rank0_print(
            "[Probe Spatial Frame Swap] "
            f"sample_log_index={log_index}, sample_index={sample_index}, "
            f"mode={mode}, seed={seed}, F={frame_count}, "
            f"perm={perm.detach().cpu().tolist()}"
        )
    elif int(getattr(config, "_probe_spatial_feature_frame_swap_log_count", 0)) == 6:
        rank0_print("[Probe Spatial Frame Swap] Further per-sample permutation logs suppressed.")

    camera_tokens = camera_tokens.index_select(0, perm)
    patch_tokens = patch_tokens.index_select(0, perm.to(device=patch_tokens.device))
    if camera_pose is not None:
        camera_pose = camera_pose.index_select(0, perm.to(device=camera_pose.device))
    return camera_tokens, patch_tokens, camera_pose


def _maybe_apply_cross_frame_geo_rope_probe(config, fusion_block, image_features, patch_tokens, pos_clip, pos_spatial, dtype, training):
    if training:
        return None

    window = int(getattr(config, "probe_cross_frame_window", 0) or 0)
    if window <= 0:
        return None

    mode = str(getattr(config, "probe_cross_frame_mode", "sliding_window"))
    if mode != "sliding_window":
        raise ValueError(f"Unknown probe_cross_frame_mode: {mode}")

    include_self = _as_bool_config(getattr(config, "probe_cross_frame_include_self", True), default=True)
    frame_count = int(image_features.shape[0])
    original_n_spatial = int(patch_tokens.shape[1])
    outputs = []
    frame_windows = []

    patch_tokens = patch_tokens.to(device=image_features.device, dtype=dtype)
    pos_clip = pos_clip.to(device=image_features.device, dtype=image_features.dtype)
    pos_spatial = pos_spatial.to(device=image_features.device, dtype=image_features.dtype)

    for frame_idx in range(frame_count):
        start = max(0, frame_idx - window)
        end = min(frame_count, frame_idx + window + 1)
        frame_ids = list(range(start, end))
        if not include_self:
            frame_ids = [idx for idx in frame_ids if idx != frame_idx]
        if len(frame_ids) == 0:
            frame_ids = [frame_idx]

        q_tokens = image_features[frame_idx:frame_idx + 1]
        q_pos = pos_clip[frame_idx:frame_idx + 1]
        kv_tokens = patch_tokens[frame_ids].reshape(1, -1, patch_tokens.shape[-1])
        kv_pos = pos_spatial[frame_ids].reshape(1, -1, pos_spatial.shape[-1])

        out_f, _ = fusion_block(q_tokens, kv_tokens, q_pos, kv_pos)
        outputs.append(out_f)
        frame_windows.append((frame_idx, frame_ids, int(kv_tokens.shape[1])))

    log_index = _probe_log_index(config, "_probe_cross_frame_log_count", limit=3)
    if log_index is not None:
        rank0_print(
            "[Probe X-Frame] "
            f"sample_log_index={log_index}, mode={mode}, window={window}, "
            f"include_self={include_self}, F={frame_count}, "
            f"original_N_spatial={original_n_spatial}"
        )
        max_logged_frames = min(len(frame_windows), 8)
        for frame_idx, frame_ids, kv_count in frame_windows[:max_logged_frames]:
            rank0_print(
                "[Probe X-Frame] "
                f"query_frame={frame_idx}, kv_frame_ids={frame_ids}, expanded_N_spatial={kv_count}"
            )
        if len(frame_windows) > max_logged_frames:
            rank0_print("[Probe X-Frame] Additional query-frame window logs suppressed for this sample.")
    elif int(getattr(config, "_probe_cross_frame_log_count", 0)) == 4:
        rank0_print("[Probe X-Frame] Further per-sample window logs suppressed.")

    return torch.cat(outputs, dim=0), None


def _maybe_apply_intra_frame_pos_shuffle_probe(config, pos_clip, pos_spatial, training):
    if training or not _as_bool_config(getattr(config, "probe_intra_frame_pos_shuffle", False)):
        return pos_clip, pos_spatial

    frame_count = int(pos_clip.shape[0])
    log_index = _probe_log_index(config, "_probe_intra_frame_pos_shuffle_log_count", limit=3)
    for frame_idx in range(frame_count):
        perm_clip = torch.randperm(pos_clip.shape[1], device=pos_clip.device)
        perm_spatial = torch.randperm(pos_spatial.shape[1], device=pos_spatial.device)
        pos_clip[frame_idx] = pos_clip[frame_idx, perm_clip]
        pos_spatial[frame_idx] = pos_spatial[frame_idx, perm_spatial]

        if log_index is not None and frame_idx == 0:
            rank0_print(
                "[Probe Intra-Frame Pos Shuffle] "
                f"sample_log_index={log_index}, F={frame_count}, "
                f"N_clip={pos_clip.shape[1]}, N_spatial={pos_spatial.shape[1]}"
            )
    if log_index is None and int(getattr(config, "_probe_intra_frame_pos_shuffle_log_count", 0)) == 4:
        rank0_print("[Probe Intra-Frame Pos Shuffle] Further per-sample logs suppressed.")

    return pos_clip, pos_spatial


def _normalize_point_map_key(key):
    if key in (None, ""):
        return None
    normalized = str(key).strip().lower()
    aliases = {
        "ref": "point_maps_ref",
        "reference": "point_maps_ref",
        "anchor": "point_maps_ref",
        "point_maps_ref": "point_maps_ref",
        "pts3d_in_other_view": "point_maps_ref",
        "cam": "point_maps_cam",
        "camera": "point_maps_cam",
        "self": "point_maps_cam",
        "point_maps_cam": "point_maps_cam",
        "pts3d_in_self_view": "point_maps_cam",
    }
    if normalized not in aliases:
        raise ValueError(
            "geo_rope_point_map_key must be one of ref/point_maps_ref/"
            "pts3d_in_other_view or cam/point_maps_cam/pts3d_in_self_view, "
            f"got {key!r}"
        )
    return aliases[normalized]


def _point_map_key_candidates(key):
    normalized = _normalize_point_map_key(key)
    if normalized == "point_maps_ref":
        return ("point_maps_ref", "pts3d_in_other_view")
    if normalized == "point_maps_cam":
        return ("point_maps_cam", "pts3d_in_self_view")
    return ()


def _llm_visual_3d_rope_point_map_key(config):
    return (
        getattr(config, "llm_visual_3d_rope_geometry_source", None)
        or getattr(config, "geo_rope_point_map_key", None)
        or getattr(config, "geometry_point_map_key", None)
        or "point_maps_ref"
    )


def _resolve_llm_geo_point_maps(config, loaded_spatial_features=None, point_maps=None):
    requested_key = _llm_visual_3d_rope_point_map_key(config)
    point_map_keys = _point_map_key_candidates(requested_key)
    if isinstance(loaded_spatial_features, dict):
        for point_key in point_map_keys:
            if point_key in loaded_spatial_features:
                return _coalesce_point_maps(loaded_spatial_features[point_key]), point_key
        raise RuntimeError(
            "LLM visual 3D RoPE requested geometry source "
            f"{requested_key!r}, but none of {list(point_map_keys)} were found in the CUT3R sidecar. "
            "Do not silently fall back to camera coordinates."
        )
    if point_maps is not None:
        return _coalesce_point_maps(point_maps), "point_maps_argument"
    return None, None


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            
            # create spatial tower and fusion block
            if hasattr(config, "spatial_tower"):
                preextracted_only = _as_bool_config(
                    getattr(config, "spatial_tower_preextracted_only", False),
                    False,
                )
                # Sidecar-only CUT3R jobs need the tower wrapper/config for fusion
                # routing, but the heavy encoder weights are never used.
                self.spatial_tower = build_spatial_tower(config, delay_load=preextracted_only)
            if hasattr(config, "fusion_block"):
                self.fusion_block = build_multimodal_fusion_block(config, delay_load=delay_load)
            self.geometry_aware_projection = None

            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            if _as_bool_config(getattr(config, "use_geometry_aware_projection", False), False):
                self.initialize_geometry_aware_projection(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_spatial_tower(self):
        spatial_tower = getattr(self, "spatial_tower", None)
        if type(spatial_tower) is list:
            spatial_tower = spatial_tower[0]
        return spatial_tower

    def get_fusion_block(self):
        fusion_block = getattr(self, "fusion_block", None)
        if type(fusion_block) is list:
            fusion_block = fusion_block[0]
        return fusion_block

    def get_geometry_aware_projection(self):
        geometry_aware_projection = getattr(self, "geometry_aware_projection", None)
        if type(geometry_aware_projection) is list:
            geometry_aware_projection = geometry_aware_projection[0]
        return geometry_aware_projection

    def get_pi3x_geometry_tower(self):
        pi3x_geometry_tower = getattr(self, "pi3x_geometry_tower", None)
        if type(pi3x_geometry_tower) is list:
            pi3x_geometry_tower = pi3x_geometry_tower[0]
        return pi3x_geometry_tower

    def initialize_pi3x_geometry_tower(self, model_args=None, fsdp=None, device=None, dtype=None):
        existing = self.get_pi3x_geometry_tower()
        if existing is not None:
            if hasattr(existing, "load_model") and not getattr(existing, "is_loaded", True):
                existing.load_model()
            if device is not None or dtype is not None:
                existing.to(device=device, dtype=dtype)
            return existing

        class _Pi3xGeometryTowerConfig:
            pass

        cfg = _Pi3xGeometryTowerConfig()
        cfg.spatial_tower = "pi3x"
        cfg.mm_spatial_tower = "pi3x"
        cfg.mm_tunable_parts = ""
        cfg.pi3x_weights_path = (
            getattr(model_args, "pi3x_weights_path", None)
            if model_args is not None
            else getattr(self.config, "pi3x_weights_path", None)
        ) or "/leonardo_scratch/fast/EUHPC_D32_006/hf_models/Pi3X"
        cfg.pi3x_input_size = (
            getattr(model_args, "pi3x_input_size", None)
            if model_args is not None
            else getattr(self.config, "pi3x_input_size", 518)
        ) or 518

        tower = build_spatial_tower(cfg, delay_load=False)
        tower.requires_grad_(False)
        if device is not None or dtype is not None:
            tower.to(device=device, dtype=dtype)

        if fsdp is not None and len(fsdp) > 0:
            self.pi3x_geometry_tower = [tower]
        else:
            self.pi3x_geometry_tower = tower
        return tower

    def initialize_spatial_tower(self, model_args, fsdp=None):
        cli_spatial_tower = model_args.spatial_tower
        self.config.mm_spatial_tower = cli_spatial_tower
        preextracted_only = _as_bool_config(
            getattr(model_args, "spatial_tower_preextracted_only", getattr(self.config, "spatial_tower_preextracted_only", False)),
            False,
        )
        self.config.spatial_tower_preextracted_only = preextracted_only

        if self.get_spatial_tower() is None:
            # Sidecar-only jobs intentionally keep the tower in cfg-only mode;
            # forward must use pre-extracted camera_tokens/patch_tokens.
            spatial_tower = build_spatial_tower(model_args, delay_load=preextracted_only)

            if hasattr(spatial_tower.config, "to_dict"):
                cfg_dict = spatial_tower.config.to_dict()
            else:
                cfg_dict = dict(spatial_tower.config)

            for k, v in cfg_dict.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.spatial_tower = [spatial_tower]
            else:
                self.spatial_tower = spatial_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                spatial_tower = self.spatial_tower[0]
            else:
                spatial_tower = self.spatial_tower

            if hasattr(spatial_tower, "spatial_tower_name"):
                spatial_tower.spatial_tower_name = cli_spatial_tower
            elif hasattr(spatial_tower, "tower_name"):
                spatial_tower.tower_name = cli_spatial_tower
            elif hasattr(spatial_tower, "model_name"):
                spatial_tower.model_name = cli_spatial_tower
            print("[DEBUG] cli spatial_tower =", model_args.spatial_tower)
            print("[DEBUG] existing spatial tower =", spatial_tower)
            print("[DEBUG] spatial_tower_name =", getattr(spatial_tower, "spatial_tower_name", None))
            print("[DEBUG] tower_name =", getattr(spatial_tower, "tower_name", None))
            print("[DEBUG] model_name =", getattr(spatial_tower, "model_name", None))
            if preextracted_only:
                rank0_print(
                    "[SPATIAL] spatial_tower_preextracted_only=True; leaving spatial tower unloaded "
                    "and expecting pre-extracted token sidecars."
                )
                return
            spatial_tower.load_model()

    def initialize_geometry_aware_projection(self, model_args, fsdp=None):
        for attr in (
            "use_geometry_aware_projection",
            "spatial_encoder_type",
            "geometry_position_mode",
            "num_geometry_projection_layers",
            "use_auxiliary_geometry_head",
            "use_auxiliary_geometry_loss",
            "aux_geometry_targets",
            "lambda_geo",
            "geometry_loss_type",
            "detach_geometry_targets",
            "geometry_gate_init",
            "use_geometry_confidence_mask",
            "geometry_projection_num_heads",
            "geometry_position_max_abs",
            "geometry_fixed_scene_scale",
            "allow_missing_geometry_targets",
            "geometry_projection_dropout",
        ):
            if hasattr(model_args, attr):
                setattr(self.config, attr, getattr(model_args, attr))
        if getattr(self.config, "mm_hidden_size", None) is None:
            raise RuntimeError("initialize_geometry_aware_projection requires config.mm_hidden_size from the vision tower.")
        existing = self.get_geometry_aware_projection()
        expected_hidden = int(getattr(self.config, "mm_hidden_size"))
        needs_rebuild = (
            existing is None
            or _geometry_projection_hidden_size(existing) != expected_hidden
        )
        if needs_rebuild:
            module = MetricGroundedGeometryProjection(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.geometry_aware_projection = [module]
            else:
                self.geometry_aware_projection = module
        else:
            for p in existing.parameters():
                p.requires_grad = True

    def initialize_fusion_block(self, model_args, fsdp=None):
        requested_fusion_block = getattr(model_args, "fusion_block", None)
        if requested_fusion_block is not None:
            self.config.fusion_block = requested_fusion_block

        def _expected_class_name(fusion_block_type):
            if fusion_block_type in ["cross_attention", "svf_baseline"]:
                return "CrossAttentionFusion"
            if fusion_block_type == "svf_patch_only":
                return "CrossAttentionFusion"
            if fusion_block_type == "svf_patch_only_geo_rope_eval":
                return "PostHocGeoRoPEPatchCrossAttention"
            if fusion_block_type in ["svf_geo_rope_fusion", "svf_geo_rope_fusion_forced", "svf_geo_rope_fusion_per_head_gate",
                                      "svf_depth_geo_rope_fusion", "svf_xyz_geo_rope_fusion", "svf_spherical_geo_rope_fusion",
                                      "svf_depth_rope", "svf_xyz_rope", "svf_spherical_rope"]:
                return "GeoRoPEFusionCrossAttention"
            if fusion_block_type == "svf_patch_cam_concat":
                return "PatchCrossAttentionCameraConcatFusion"
            if fusion_block_type == "svf_geometry_bridge":
                return "GeometryBridgeFusion"
            if fusion_block_type == "svf_pose_geometry_bridge":
                return "GeometryBridgeFusion"
            if fusion_block_type == "svf_cat_feat":
                return "SvfCatFeatFusion"
            if fusion_block_type == "svf_pose_prepend":
                return "SvfPosePrependFusion"
            if fusion_block_type == "cross_attention_with_mlp":
                return "CrossAttentionFusionWithMLP"
            if fusion_block_type == "mlp_after_clip_proj":
                return "MLPFusion"
            if fusion_block_type == "transformer":
                return "TransformerFusion"
            if fusion_block_type == "concat_mlp":
                return "ConcatMLPFusion"
            if fusion_block_type == "concat_self_attention":
                return "ConcatSelfAttentionFusion"
            if fusion_block_type == "llava_3d_fusion_block":
                return "llava_3d_fusion_block"
            if fusion_block_type == "video_3d_llm_fusion_block":
                return "video_3d_llm_fusion_block"
            if isinstance(fusion_block_type, str) and fusion_block_type.endswith("_layer_cross_attention"):
                return "MultiLayerCrossAttentionFusion"
            return None

        existing_fusion_block = self.get_fusion_block()
        current_type = getattr(self.config, "fusion_block", None)
        expected_cls = _expected_class_name(current_type)
        needs_rebuild = (
            existing_fusion_block is None
            or (expected_cls is not None and existing_fusion_block.__class__.__name__ != expected_cls)
        )

        # Build/rebuild to keep the instantiated module aligned with config.fusion_block.
        if needs_rebuild:
            self.fusion_block = build_multimodal_fusion_block(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.fusion_block.parameters():
                p.requires_grad = True

        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.fusion_block.load_state_dict(get_w(mm_projector_weights, "fusion_block"), strict=False)
            rank0_print(f"Loaded fusion block weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vt = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vt = self.vision_tower

            vt.vision_tower_name = model_args.vision_tower
            print("[DEBUG] cli vision_tower =", model_args.vision_tower)
            print("[DEBUG] existing vision tower name =", getattr(vt, "vision_tower_name", None))
            vt.load_model()

            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        current_vt = self.get_vision_tower()
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", current_vt.hidden_size)
        if _as_bool_config(getattr(self.config, "use_geometry_aware_projection", False), False):
            self.initialize_geometry_aware_projection(model_args, fsdp=fsdp)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_spatial_tower(self):
        return self.get_model().get_spatial_tower()

    def get_fusion_block(self):
        return self.get_model().get_fusion_block()

    def get_geometry_aware_projection(self):
        return self.get_model().get_geometry_aware_projection()

    def initialize_spatial_rank_head(self, output_dim=256, device=None, dtype=None):
        hidden_size = int(getattr(self.config, "hidden_size"))
        if getattr(self, "spatial_rank_head", None) is None:
            self.spatial_rank_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, int(output_dim), bias=True),
            )
        if device is not None or dtype is not None:
            self.spatial_rank_head.to(device=device, dtype=dtype)
        for param in self.spatial_rank_head.parameters():
            param.requires_grad = True
        self.config.spatial_rank_projection_dim = int(output_dim)
        self._spatial_rank_last_metrics = {}
        self._spatial_rank_debug_printed = False
        self._spatial_rank_grad_checked = False
        return self.spatial_rank_head

    def initialize_bev_head(self, device=None, dtype=None):
        from .geometry import BEVHead

        hidden_size = int(getattr(self.config, "hidden_size"))
        if getattr(self, "bev_head", None) is None:
            self.bev_head = BEVHead(hidden_size)
        if device is not None or dtype is not None:
            self.bev_head.to(device=device, dtype=dtype)
        for param in self.bev_head.parameters():
            param.requires_grad = True
        self.config.has_bev_head = True
        return self.bev_head

    def initialize_depth_head(self, device=None, dtype=None):
        from .geometry import DepthHead

        hidden_size = int(getattr(self.config, "hidden_size"))
        if getattr(self, "depth_head", None) is None:
            self.depth_head = DepthHead(hidden_size)
        if device is not None or dtype is not None:
            self.depth_head.to(device=device, dtype=dtype)
        for param in self.depth_head.parameters():
            param.requires_grad = True
        self.config.has_depth_head = True
        return self.depth_head

    def _build_grid_metadata(
        self,
        num_frames,
        grid_side,
        device,
        raw_grid_side=None,
        prefix_len=0,
        frame_indices=None,
    ):
        frame_indices = list(range(num_frames)) if frame_indices is None else list(frame_indices)
        per_frame_len = int(prefix_len) + int(grid_side) * (int(grid_side) + 1)
        visual_positions = []
        frame_ids = []
        newline_positions = []
        prefix_positions = []

        for local_frame_idx, frame_id in enumerate(frame_indices):
            frame_offset = local_frame_idx * per_frame_len
            for prefix_pos in range(int(prefix_len)):
                prefix_positions.append(frame_offset + prefix_pos)
            grid_offset = frame_offset + int(prefix_len)
            for row in range(int(grid_side)):
                row_offset = grid_offset + row * (int(grid_side) + 1)
                for col in range(int(grid_side)):
                    visual_positions.append(row_offset + col)
                    frame_ids.append(int(frame_id))
                newline_positions.append(row_offset + int(grid_side))

        return {
            "visual_token_indices": _as_long_tensor(visual_positions, device),
            "visual_frame_ids": _as_long_tensor(frame_ids, device),
            "frame_order": [int(x) for x in frame_indices],
            "visual_grid_shapes": [(int(grid_side), int(grid_side)) for _ in frame_indices],
            "raw_visual_grid_shapes": [
                (int(raw_grid_side or grid_side), int(raw_grid_side or grid_side))
                for _ in frame_indices
            ],
            "newline_token_indices": _as_long_tensor(newline_positions, device),
            "camera_prefix_token_indices": _as_long_tensor(prefix_positions, device),
            "tokens_per_frame": [int(grid_side) * int(grid_side) for _ in frame_indices],
            "sequence_length": int(num_frames) * per_frame_len,
            "layout": "grid_with_newline",
        }

    def _build_flat_frame_metadata(
        self,
        num_frames,
        tokens_per_frame,
        device,
        grid_side=None,
        raw_grid_side=None,
        prefix_len=0,
        frame_indices=None,
    ):
        frame_indices = list(range(num_frames)) if frame_indices is None else list(frame_indices)
        real_tokens_per_frame = int(tokens_per_frame) - int(prefix_len)
        if real_tokens_per_frame <= 0:
            raise ValueError(f"Invalid flat visual layout: tokens_per_frame={tokens_per_frame}, prefix_len={prefix_len}")
        visual_positions = []
        frame_ids = []
        prefix_positions = []
        for local_frame_idx, frame_id in enumerate(frame_indices):
            frame_offset = local_frame_idx * int(tokens_per_frame)
            for prefix_pos in range(int(prefix_len)):
                prefix_positions.append(frame_offset + prefix_pos)
            for token_pos in range(real_tokens_per_frame):
                visual_positions.append(frame_offset + int(prefix_len) + token_pos)
                frame_ids.append(int(frame_id))
        grid_side = grid_side or int(math.isqrt(real_tokens_per_frame))
        return {
            "visual_token_indices": _as_long_tensor(visual_positions, device),
            "visual_frame_ids": _as_long_tensor(frame_ids, device),
            "frame_order": [int(x) for x in frame_indices],
            "visual_grid_shapes": [(int(grid_side), int(grid_side)) for _ in frame_indices],
            "raw_visual_grid_shapes": [
                (int(raw_grid_side or grid_side), int(raw_grid_side or grid_side))
                for _ in frame_indices
            ],
            "newline_token_indices": _empty_long(device),
            "camera_prefix_token_indices": _as_long_tensor(prefix_positions, device),
            "tokens_per_frame": [int(real_tokens_per_frame) for _ in frame_indices],
            "sequence_length": int(num_frames) * int(tokens_per_frame),
            "layout": "flat_frames",
        }

    def _shift_metadata(self, metadata, offset, max_len=None, padding_positions=None, answer_positions=None, text_positions=None):
        if metadata is None:
            metadata = {}
        shifted = {}
        visual_keep_mask = None
        tensor_keys = {
            "visual_token_indices",
            "visual_frame_ids",
            "newline_token_indices",
            "camera_prefix_token_indices",
            "padding_token_indices",
            "answer_token_indices",
            "text_token_indices",
            "special_token_indices",
        }
        for key, value in metadata.items():
            if key in tensor_keys and isinstance(value, torch.Tensor):
                if key == "visual_frame_ids":
                    continue
                else:
                    cur = value + int(offset)
                    if max_len is not None:
                        keep = cur < int(max_len)
                        if key == "visual_token_indices":
                            visual_keep_mask = keep
                        cur = cur[keep]
                    shifted[key] = cur
            else:
                shifted[key] = value
        if "visual_frame_ids" in metadata and isinstance(metadata["visual_frame_ids"], torch.Tensor):
            frame_ids = metadata["visual_frame_ids"].clone()
            if visual_keep_mask is not None:
                frame_ids = frame_ids[visual_keep_mask]
            shifted["visual_frame_ids"] = frame_ids
        if "visual_token_positions_3d" in metadata and isinstance(metadata["visual_token_positions_3d"], torch.Tensor):
            positions_3d = metadata["visual_token_positions_3d"].clone()
            if visual_keep_mask is not None:
                positions_3d = positions_3d[visual_keep_mask]
            shifted["visual_token_positions_3d"] = positions_3d
        device = None
        for value in shifted.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            device = self.device
        shifted.setdefault("visual_token_indices", _empty_long(device))
        shifted.setdefault("visual_frame_ids", _empty_long(device))
        shifted["padding_token_indices"] = _as_long_tensor(padding_positions or [], device)
        shifted["answer_token_indices"] = _as_long_tensor(answer_positions or [], device)
        shifted["text_token_indices"] = _as_long_tensor(text_positions or [], device)
        shifted.setdefault("special_token_indices", _empty_long(device))
        shifted.setdefault("newline_token_indices", _empty_long(device))
        shifted.setdefault("camera_prefix_token_indices", _empty_long(device))
        return shifted

    def _attach_llm_geo_positions_to_metadata(self, metadata, geometry_point_maps, geometry_source):
        if metadata is None or geometry_point_maps is None:
            return metadata
        visual_indices = metadata.get("visual_token_indices", None)
        frame_ids = metadata.get("visual_frame_ids", None)
        if not isinstance(visual_indices, torch.Tensor) or visual_indices.numel() == 0:
            return metadata
        if not isinstance(frame_ids, torch.Tensor) or frame_ids.numel() != visual_indices.numel():
            raise RuntimeError("LLM visual 3D RoPE metadata requires visual_frame_ids aligned to visual_token_indices.")

        tokens_per_frame = metadata.get("tokens_per_frame", None)
        if not tokens_per_frame:
            raise RuntimeError("LLM visual 3D RoPE metadata requires tokens_per_frame.")
        unique_counts = sorted({int(value) for value in tokens_per_frame})
        if len(unique_counts) != 1:
            raise RuntimeError(f"LLM visual 3D RoPE expects equal grid tokens per frame, got {tokens_per_frame}")
        target_tokens = unique_counts[0]

        geometry_point_maps = geometry_point_maps.to(device=visual_indices.device)
        if int(geometry_point_maps.shape[0]) < len(tokens_per_frame):
            raise RuntimeError(
                "LLM visual 3D RoPE point-map frame count is smaller than visual frames: "
                f"point_maps={tuple(geometry_point_maps.shape)}, tokens_per_frame={tokens_per_frame}"
            )
        pooled = pool_point_map_to_tokens(geometry_point_maps[: len(tokens_per_frame)], target_tokens)

        frame_order = metadata.get("frame_order", list(range(len(tokens_per_frame))))
        frame_to_local = {int(frame_id): local_idx for local_idx, frame_id in enumerate(frame_order)}
        frame_offsets = {int(frame_id): 0 for frame_id in frame_ids.detach().cpu().tolist()}
        positions = []
        for frame_id in frame_ids.detach().cpu().tolist():
            local_frame_id = int(frame_id)
            if local_frame_id not in frame_to_local:
                raise RuntimeError(f"LLM visual 3D RoPE frame id {local_frame_id} missing from frame_order={frame_order}")
            token_offset = frame_offsets[local_frame_id]
            if token_offset >= target_tokens:
                raise RuntimeError(
                    "LLM visual 3D RoPE visual token layout has more tokens for a frame than the pooled point map."
                )
            positions.append(pooled[frame_to_local[local_frame_id], token_offset])
            frame_offsets[local_frame_id] = token_offset + 1

        positions = torch.stack(positions, dim=0).to(device=visual_indices.device)
        if positions.shape[0] != visual_indices.numel():
            raise RuntimeError(
                "LLM visual 3D RoPE position count must match visual token count, "
                f"got positions={positions.shape[0]} visual={visual_indices.numel()}"
            )
        metadata["visual_token_positions_3d"] = positions
        metadata["llm_geo_source"] = geometry_source
        return metadata

    def _build_llm_geo_tensors(self, visual_metadata_padded, batch_size, max_len, dtype, device):
        llm_geo_pos = torch.zeros((batch_size, max_len, 3), dtype=dtype, device=device)
        llm_geo_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        debug_samples = []

        for batch_idx, metadata in enumerate(visual_metadata_padded):
            visual_indices = metadata.get("visual_token_indices", _empty_long(device)).to(device=device)
            positions_3d = metadata.get("visual_token_positions_3d", None)
            if not isinstance(positions_3d, torch.Tensor) or visual_indices.numel() == 0:
                debug_samples.append({
                    "sample_id": int(batch_idx),
                    "seq_len": int(max_len),
                    "num_visual_tokens": int(visual_indices.numel()),
                    "num_frames": int(len(metadata.get("tokens_per_frame", []) or [])),
                    "llm_geo_mask_count": 0,
                    "visual_token_indices_head": visual_indices[:16].detach().cpu().tolist(),
                    "first_3d_positions": [],
                    "geometry_source": metadata.get("llm_geo_source", None),
                })
                continue
            positions_3d = positions_3d.to(device=device, dtype=dtype)
            if positions_3d.shape[0] != visual_indices.numel():
                raise RuntimeError(
                    "LLM visual 3D RoPE dense metadata mismatch: "
                    f"positions={positions_3d.shape[0]} visual_indices={visual_indices.numel()}"
                )
            keep = visual_indices < int(max_len)
            visual_indices = visual_indices[keep]
            positions_3d = positions_3d[keep]
            finite = torch.isfinite(positions_3d).all(dim=-1)
            visual_indices = visual_indices[finite]
            positions_3d = positions_3d[finite]
            if visual_indices.numel() > 0:
                llm_geo_pos[batch_idx, visual_indices] = positions_3d
                llm_geo_mask[batch_idx, visual_indices] = True

            def overlap_count(key):
                values = metadata.get(key, _empty_long(device))
                if not isinstance(values, torch.Tensor) or values.numel() == 0 or visual_indices.numel() == 0:
                    return 0
                return int(torch.isin(visual_indices, values.to(device=device)).sum().item())

            debug_samples.append({
                "sample_id": int(batch_idx),
                "seq_len": int(max_len),
                "num_visual_tokens": int(metadata.get("visual_token_indices", _empty_long(device)).numel()),
                "num_frames": int(len(metadata.get("tokens_per_frame", []) or [])),
                "llm_geo_mask_count": int(llm_geo_mask[batch_idx].sum().item()),
                "visual_token_indices_head": visual_indices[:16].detach().cpu().tolist(),
                "newline_overlap": overlap_count("newline_token_indices"),
                "prefix_overlap": overlap_count("camera_prefix_token_indices"),
                "padding_overlap": overlap_count("padding_token_indices"),
                "text_overlap": overlap_count("text_token_indices"),
                "first_3d_positions": positions_3d[:5].detach().float().cpu().tolist(),
                "geometry_source": metadata.get("llm_geo_source", None),
            })

        debug = {
            "seq_len": int(max_len),
            "num_valid_geo_tokens": int(llm_geo_mask.sum().item()),
            "samples": debug_samples,
            "shuffle_enabled": False,
            "shuffle_seed": int(getattr(self.config, "llm_visual_3d_rope_shuffle_seed", 0) or 0),
            "shuffle_mode": getattr(self.config, "llm_visual_3d_rope_shuffle_mode", "intra_sample_token_shuffle"),
        }

        if (
            (not self.training)
            and _as_bool_config(getattr(self.config, "llm_visual_3d_rope_shuffle", False), False)
        ):
            mode = str(getattr(self.config, "llm_visual_3d_rope_shuffle_mode", "intra_sample_token_shuffle"))
            if mode != "intra_sample_token_shuffle":
                raise ValueError(f"Unsupported llm_visual_3d_rope_shuffle_mode: {mode}")
            seed = int(getattr(self.config, "llm_visual_3d_rope_shuffle_seed", 0) or 0)
            debug["shuffle_enabled"] = True
            debug["shuffle_permutations"] = []
            for batch_idx in range(batch_size):
                idx = torch.nonzero(llm_geo_mask[batch_idx], as_tuple=False).flatten()
                if idx.numel() <= 1:
                    continue
                generator_device = device if device.type == "cuda" else "cpu"
                generator = torch.Generator(device=generator_device)
                generator.manual_seed(seed + batch_idx)
                perm_order = torch.randperm(idx.numel(), generator=generator, device=device)
                if torch.equal(perm_order, torch.arange(idx.numel(), device=device)):
                    perm_order = torch.roll(perm_order, shifts=1, dims=0)
                original = llm_geo_pos[batch_idx, idx].clone()
                llm_geo_pos[batch_idx, idx] = original[perm_order]
                debug["shuffle_permutations"].append({
                    "sample_id": int(batch_idx),
                    "token_indices_head": idx[:8].detach().cpu().tolist(),
                    "permuted_from_head": idx[perm_order[:8]].detach().cpu().tolist(),
                })

        return llm_geo_pos, llm_geo_mask, debug

    def _pool_cut3r_teacher_to_student_grid(self, teacher_tokens, target_tokens, pool_mode):
        token_count = int(teacher_tokens.shape[0])
        if token_count != 729:
            raise ValueError(f"CUT3R teacher must have 729 tokens before pooling, got {token_count}")
        teacher_grid = teacher_tokens.view(27, 27, -1)
        valid_grid = torch.ones(27, 27, dtype=torch.bool, device=teacher_tokens.device)

        if target_tokens == 729:
            return teacher_tokens, torch.ones(729, dtype=torch.bool, device=teacher_tokens.device)
        if target_tokens != 196:
            raise ValueError(
                f"Unsupported H_1 visual token count per frame: {target_tokens}. "
                f"CUT3R teacher raw tokens={token_count}"
            )

        pool_mode = str(pool_mode or "bilinear").lower()
        if pool_mode == "bilinear":
            pooled = F.interpolate(
                teacher_grid.permute(2, 0, 1).unsqueeze(0).float(),
                size=(14, 14),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).to(dtype=teacher_tokens.dtype)
            return pooled.reshape(196, -1), torch.ones(196, dtype=torch.bool, device=teacher_tokens.device)

        pad_grid = F.pad(teacher_grid.permute(2, 0, 1), (0, 1, 0, 1), value=0.0)
        pad_valid = F.pad(valid_grid[None].float(), (0, 1, 0, 1), value=0.0)
        if pool_mode == "average":
            valid_counts = F.avg_pool2d(pad_valid, kernel_size=2, stride=2) * 4.0
            summed = F.avg_pool2d(pad_grid.float(), kernel_size=2, stride=2) * 4.0
            pooled = summed / valid_counts.clamp_min(1.0)
            valid = valid_counts[0] > 0
        elif pool_mode == "max":
            masked = pad_grid.float().masked_fill(pad_valid.bool().expand_as(pad_grid) == 0, -torch.finfo(torch.float32).max)
            pooled = F.max_pool2d(masked, kernel_size=2, stride=2)
            valid = F.max_pool2d(pad_valid, kernel_size=2, stride=2)[0] > 0
        else:
            raise ValueError(f"Unsupported mm_spatial_pool_mode for CUT3R target matching: {pool_mode}")
        return pooled.permute(1, 2, 0).reshape(196, -1).to(dtype=teacher_tokens.dtype), valid.reshape(-1)

    def _get_spatial_feature_for_batch(self, spatial_features, batch_idx):
        if spatial_features is None:
            raise ValueError("spatial_rank_loss_enable=True requires CUT3R spatial_features with patch_tokens.")
        if isinstance(spatial_features, (list, tuple)):
            if len(spatial_features) == 1:
                return spatial_features[0]
            return spatial_features[batch_idx]
        if isinstance(spatial_features, dict):
            return spatial_features
        raise ValueError(f"Unsupported spatial_features type for ranking loss: {type(spatial_features)}")

    def _sample_spatial_rank_triplets(self, teacher_features, anchors_per_frame, positive_top_percent, negative_bottom_percent):
        num_tokens = int(teacher_features.shape[0])
        if num_tokens < 3:
            raise ValueError(f"Need at least 3 visual tokens for ranking loss, got {num_tokens}")
        teacher_norm = F.normalize(teacher_features.float(), dim=-1)
        teacher_sim = teacher_norm @ teacher_norm.T
        anchor_count = min(int(anchors_per_frame), num_tokens)
        anchors = torch.randperm(num_tokens, device=teacher_features.device)[:anchor_count]
        pos_k = max(1, int(math.ceil(num_tokens * float(positive_top_percent) / 100.0)))
        neg_k = max(1, int(math.ceil(num_tokens * float(negative_bottom_percent) / 100.0)))
        positives = []
        negatives = []
        teacher_pos = []
        teacher_neg = []

        for anchor in anchors:
            row = teacher_sim[anchor].clone()
            row[anchor] = -float("inf")
            pos_pool = torch.topk(row, k=min(pos_k, num_tokens - 1), largest=True).indices
            row_for_neg = teacher_sim[anchor].clone()
            row_for_neg[anchor] = float("inf")
            neg_pool = torch.topk(row_for_neg, k=min(neg_k, num_tokens - 1), largest=False).indices
            pos = pos_pool[torch.randint(pos_pool.numel(), (1,), device=teacher_features.device)]
            neg = neg_pool[torch.randint(neg_pool.numel(), (1,), device=teacher_features.device)]
            positives.append(pos)
            negatives.append(neg)
            teacher_pos.append(teacher_sim[anchor, pos])
            teacher_neg.append(teacher_sim[anchor, neg])

        positives = torch.cat(positives)
        negatives = torch.cat(negatives)
        teacher_pos = torch.cat(teacher_pos)
        teacher_neg = torch.cat(teacher_neg)
        return anchors, positives, negatives, teacher_pos, teacher_neg

    def _run_spatial_rank_grad_checks(self, rank_loss):
        if getattr(self, "_spatial_rank_grad_checked", False):
            return
        self._spatial_rank_grad_checked = True

        groups = {"p_geo": [], "block0_lora": [], "block1plus_lora": [], "upstream": []}
        frozen_p_geo = bool(getattr(self.config, "freeze_spatial_rank_head", False))
        if frozen_p_geo and getattr(self, "spatial_rank_head", None) is not None:
            assert all(not param.requires_grad for param in self.spatial_rank_head.parameters()), (
                "freeze_spatial_rank_head=True but at least one P_geo parameter is trainable."
            )
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "spatial_rank_head" in name:
                groups["p_geo"].append((name, param))
            elif "lora_" in name and re.search(r"\.layers\.0\.", name):
                groups["block0_lora"].append((name, param))
            elif "lora_" in name and re.search(r"\.layers\.(\d+)\.", name):
                layer_idx = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
                if layer_idx >= 1:
                    groups["block1plus_lora"].append((name, param))
            elif any(key in name for key in ("mm_projector", "fusion_block")):
                groups["upstream"].append((name, param))

        check_params = []
        check_names = []
        for key in ("p_geo", "block0_lora", "upstream", "block1plus_lora"):
            for name, param in groups[key][:4]:
                check_names.append((key, name))
                check_params.append(param)
        if not check_params:
            return

        checkpointing_enabled = any(
            bool(getattr(module, "gradient_checkpointing", False))
            for module in self.modules()
        )
        if checkpointing_enabled:
            rank0_print(
                "[SPATIAL_RANK] Debug gradient probes skipped because "
                "gradient checkpointing is enabled; normal backward still "
                "propagates L_rank."
            )
            return

        grads = torch.autograd.grad(rank_loss, check_params, retain_graph=True, allow_unused=True)
        by_group = {key: [] for key in groups}
        for (key, name), grad in zip(check_names, grads):
            by_group[key].append((name, grad))

        required_gradient_groups = ("block0_lora", "upstream") if frozen_p_geo else ("p_geo", "block0_lora", "upstream")
        for key in required_gradient_groups:
            if groups[key]:
                assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for _, grad in by_group[key]), (
                    f"Expected rank-loss gradient for {key}, but none was found."
                )
        if groups["block1plus_lora"]:
            assert all(grad is None for _, grad in by_group["block1plus_lora"]), (
                "L_rank should not directly produce gradients for block1+ LoRA parameters."
            )

    def compute_spatial_ranking_loss(
        self,
        h1,
        visual_metadata,
        spatial_features,
        debug_checks=False,
    ):
        if getattr(self, "spatial_rank_head", None) is None:
            raise RuntimeError("spatial_rank_loss_enable=True but spatial_rank_head/P_geo is not initialized.")
        if h1 is None:
            raise RuntimeError("H1 is required for spatial ranking loss.")
        assert h1.requires_grad, "Captured H1 must require gradients for spatial ranking loss."

        device = h1.device
        dtype = h1.dtype
        margin = float(getattr(self.config, "spatial_rank_margin", 0.2))
        lambda_sim = float(getattr(self.config, "lambda_sim", 0.01))
        anchors_per_frame = int(getattr(self.config, "anchors_per_frame", 128))
        positive_top_percent = float(getattr(self.config, "positive_top_percent", 10.0))
        negative_bottom_percent = float(getattr(self.config, "negative_bottom_percent", 30.0))
        pool_mode = getattr(self.config, "mm_spatial_pool_mode", "bilinear")

        rank_losses = []
        metrics = {
            "spatial_rank_anchors": 0,
            "spatial_rank_valid_visual_tokens": 0,
            "spatial_rank_h1_token_count": 0,
            "spatial_rank_cut3r_token_count": 0,
        }
        teacher_pos_values = []
        teacher_neg_values = []
        student_pos_values = []
        student_neg_values = []
        accuracies = []

        for batch_idx, metadata in enumerate(visual_metadata or []):
            visual_indices = metadata["visual_token_indices"].to(device=device)
            frame_ids = metadata["visual_frame_ids"].to(device=device)
            if visual_indices.numel() == 0:
                continue

            excluded = torch.cat([
                metadata.get("newline_token_indices", _empty_long(device)).to(device=device),
                metadata.get("padding_token_indices", _empty_long(device)).to(device=device),
                metadata.get("answer_token_indices", _empty_long(device)).to(device=device),
                metadata.get("text_token_indices", _empty_long(device)).to(device=device),
                metadata.get("special_token_indices", _empty_long(device)).to(device=device),
                metadata.get("camera_prefix_token_indices", _empty_long(device)).to(device=device),
            ])
            if excluded.numel() > 0:
                overlap = torch.isin(visual_indices, excluded)
                assert not overlap.any(), "H_1^vis metadata includes text/newline/padding/special/prefix tokens."

            sf = self._get_spatial_feature_for_batch(spatial_features, batch_idx)
            if not isinstance(sf, dict) or "patch_tokens" not in sf:
                raise ValueError("CUT3R teacher target must come from spatial_features['patch_tokens'].")
            patch_tokens = sf["patch_tokens"].to(device=device)
            assert not patch_tokens.requires_grad, "CUT3R teacher tokens must be detached/frozen."
            if patch_tokens.dim() == 4 and patch_tokens.shape[0] == 1:
                patch_tokens = patch_tokens[0]
            if patch_tokens.dim() != 3:
                raise ValueError(f"Expected CUT3R patch_tokens shape (frames, tokens, dim), got {tuple(patch_tokens.shape)}")

            frame_order = list(metadata.get("frame_order", []))
            assert len(frame_order) == int(patch_tokens.shape[0]), (
                f"Sampled visual frame count {len(frame_order)} != CUT3R teacher frame count {patch_tokens.shape[0]}"
            )
            if "frame_indices" in sf:
                sf_frame_indices = [int(x) for x in sf["frame_indices"]]
                assert sf_frame_indices == frame_order, f"Frame index mismatch: visual={frame_order}, CUT3R={sf_frame_indices}"
            else:
                assert frame_order == list(range(len(frame_order))), f"Unexpected visual frame order without teacher indices: {frame_order}"

            if debug_checks and not getattr(self, "_spatial_rank_debug_printed", False):
                rank0_print(
                    "[SPATIAL_RANK] first batch shapes: "
                    f"H_1^vis={tuple(h1[batch_idx, visual_indices].shape)}, "
                    f"CUT3R_patch_tokens={tuple(patch_tokens.shape)}, "
                    f"tokens_per_frame={metadata.get('tokens_per_frame')}, "
                    f"visual_grid_shape={metadata.get('visual_grid_shapes')}, "
                    f"num_frames={len(frame_order)}"
                )

            for local_frame_idx, frame_id in enumerate(frame_order):
                frame_mask = frame_ids == int(frame_id)
                frame_indices = visual_indices[frame_mask]
                h1_frame = h1[batch_idx, frame_indices]
                teacher_frame, teacher_valid = self._pool_cut3r_teacher_to_student_grid(
                    patch_tokens[local_frame_idx].detach(),
                    int(h1_frame.shape[0]),
                    pool_mode,
                )
                teacher_valid = teacher_valid.to(device=device)
                assert int(teacher_valid.sum().item()) == int(h1_frame.shape[0]), (
                    f"CUT3R valid mask count {teacher_valid.sum().item()} != H1_vis token count {h1_frame.shape[0]}"
                )
                teacher_frame = teacher_frame[teacher_valid]
                assert int(h1_frame.shape[0]) == int(teacher_frame.shape[0]), (
                    f"H1_vis token count {h1_frame.shape[0]} != CUT3R target token count {teacher_frame.shape[0]}"
                )

                anchors, positives, negatives, teacher_pos, teacher_neg = self._sample_spatial_rank_triplets(
                    teacher_frame,
                    anchors_per_frame,
                    positive_top_percent,
                    negative_bottom_percent,
                )
                assert teacher_pos.mean() > teacher_neg.mean(), (
                    "Teacher positive similarity must be greater than teacher negative similarity."
                )
                z = F.normalize(self.spatial_rank_head(h1_frame.to(dtype=dtype)), dim=-1)
                sim_pos = (z[anchors] * z[positives]).sum(dim=-1)
                sim_neg = (z[anchors] * z[negatives]).sum(dim=-1)
                rank_loss = F.relu(margin - sim_pos + sim_neg).mean()
                assert torch.isfinite(rank_loss), "L_rank is not finite."
                rank_losses.append(rank_loss)

                metrics["spatial_rank_anchors"] += int(anchors.numel())
                metrics["spatial_rank_valid_visual_tokens"] += int(h1_frame.shape[0])
                metrics["spatial_rank_h1_token_count"] += int(h1_frame.shape[0])
                metrics["spatial_rank_cut3r_token_count"] += int(teacher_frame.shape[0])
                teacher_pos_values.append(teacher_pos.mean())
                teacher_neg_values.append(teacher_neg.mean())
                student_pos_values.append(sim_pos.mean())
                student_neg_values.append(sim_neg.mean())
                accuracies.append((sim_pos > sim_neg).float().mean())

        if not rank_losses:
            zero = h1.sum() * 0.0
            return zero, {"spatial_rank_loss": 0.0, "spatial_rank_weighted_loss": 0.0}

        rank_loss = torch.stack(rank_losses).mean()
        if debug_checks:
            self._run_spatial_rank_grad_checks(rank_loss)
        metrics.update({
            "spatial_rank_loss": float(rank_loss.detach().float().item()),
            "spatial_rank_weighted_loss": float((rank_loss.detach().float() * lambda_sim).item()),
            "spatial_rank_teacher_sim_pos": float(torch.stack(teacher_pos_values).mean().detach().float().item()),
            "spatial_rank_teacher_sim_neg": float(torch.stack(teacher_neg_values).mean().detach().float().item()),
            "spatial_rank_student_sim_pos": float(torch.stack(student_pos_values).mean().detach().float().item()),
            "spatial_rank_student_sim_neg": float(torch.stack(student_neg_values).mean().detach().float().item()),
            "spatial_rank_accuracy": float(torch.stack(accuracies).mean().detach().float().item()),
            "spatial_rank_head_frozen": bool(getattr(self.config, "freeze_spatial_rank_head", False)),
        })
        self._spatial_rank_debug_printed = True
        return rank_loss, metrics

    def _split_prefix_tokens_for_square_grid(self, image_feature):
        num_tokens = image_feature.shape[1]
        side = int(math.isqrt(num_tokens))
        if side * side == num_tokens:
            return None, image_feature, side

        # Some fusion blocks prepend non-grid tokens (for example, camera tokens).
        for prefix_len in range(1, num_tokens):
            remaining = num_tokens - prefix_len
            side = int(math.isqrt(remaining))
            if side * side == remaining:
                prefix_tokens = image_feature[:, :prefix_len, :]
                grid_tokens = image_feature[:, prefix_len:, :]
                return prefix_tokens, grid_tokens, side

        raise ValueError(
            f"Cannot split tokens into prefix + square grid, got shape {tuple(image_feature.shape)}"
        )

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        expected_grid_tokens = height * width
        if num_tokens < expected_grid_tokens:
            raise ValueError(
                f"Insufficient tokens for {height}x{width} grid: got {num_tokens}"
            )

        prefix_tokens = None
        if num_tokens > expected_grid_tokens:
            prefix_len = num_tokens - expected_grid_tokens
            prefix_tokens = image_feature[:, :prefix_len, :]
            image_feature = image_feature[:, prefix_len:, :]

        image_feature = image_feature.view(num_frames, height, width, num_dim)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        if prefix_tokens is not None:
            image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
        return image_feature

    def _geometry_visual_grid_size(self, grid_token_count):
        vision_tower = self.get_vision_tower()
        side = getattr(vision_tower, "num_patches_per_side", None)
        if side is not None and int(side) * int(side) == int(grid_token_count):
            return int(side), int(side)
        side = int(math.isqrt(int(grid_token_count)))
        if side * side == int(grid_token_count):
            return side, side
        raise ValueError(
            "Geometry-aware projection needs explicit visual_grid_size for non-square visual tokens, "
            f"got {grid_token_count}"
        )

    def _split_geometry_visual_tokens(self, image_features):
        vision_tower = self.get_vision_tower()
        side = getattr(vision_tower, "num_patches_per_side", None)
        if side is not None:
            grid_tokens = int(side) * int(side)
            if image_features.shape[1] == grid_tokens:
                return None, image_features
            if image_features.shape[1] > grid_tokens:
                prefix_len = image_features.shape[1] - grid_tokens
                return image_features[:, :prefix_len, :], image_features[:, prefix_len:, :]
        prefix_tokens, grid_tokens, _ = self._split_prefix_tokens_for_square_grid(image_features)
        return prefix_tokens, grid_tokens

    def _canonical_geometry_outputs_for_projection(self, geometry_outputs, point_maps, spatial_features, device):
        pi3x_geometry = self._decode_pi3x_geometry_outputs_for_projection(spatial_features, device)
        point_map_key = (
            getattr(self.get_model().config, "geo_rope_point_map_key", None)
            or getattr(self.get_model().config, "geometry_point_map_key", None)
        )
        canonical = canonicalize_geometry_outputs(
            geometry_outputs,
            pi3x_geometry,
            spatial_features,
            point_maps=point_maps,
            point_map_key=point_map_key,
        )
        moved = {}
        for key, value in canonical.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device=device)
        return moved

    def _runtime_geometry_outputs_for_projection(self, images):
        spatial_tower = self.get_model().get_spatial_tower()
        if spatial_tower is None:
            return None

        if hasattr(spatial_tower, "load_model") and not getattr(spatial_tower, "is_loaded", True):
            spatial_tower.load_model()
        if isinstance(images, torch.Tensor):
            if images.is_floating_point():
                spatial_tower.to(device=images.device, dtype=images.dtype)
            else:
                spatial_tower.to(device=images.device)

        with torch.no_grad():
            outputs = spatial_tower(images)

        canonical = canonicalize_geometry_outputs(outputs)
        if canonical:
            return canonical

        spatial_tower_name = str(
            getattr(self.config, "spatial_tower", "")
            or getattr(self.config, "mm_spatial_tower", "")
            or getattr(spatial_tower, "spatial_tower_name", "")
        )
        raise RuntimeError(
            "Runtime geometry-aware projection could not obtain point_map/depth from "
            f"spatial_tower={spatial_tower_name!r}. Use spatial_tower='cut3r_points' "
            "for on-the-fly CUT3R point-map prediction, or provide pre-extracted "
            "CUT3R point-map sidecars as spatial_features."
        )

    def _decode_pi3x_geometry_outputs_for_projection(self, spatial_features, device, force=False):
        spatial_encoder_type = str(getattr(self.config, "spatial_encoder_type", "") or "").lower()
        spatial_tower_name = str(getattr(self.config, "spatial_tower", "") or getattr(self.config, "mm_spatial_tower", "") or "").lower()
        if not force and "pi3" not in spatial_encoder_type and "pi3" not in spatial_tower_name:
            return None
        if spatial_features is None:
            return None

        if isinstance(spatial_features, (list, tuple)):
            feature_items = [item for item in spatial_features if item is not None]
        else:
            feature_items = [spatial_features]
        decoded_items = [
            item for item in feature_items
            if isinstance(item, dict)
            and isinstance(item.get("frames"), dict)
            and "decoded_features" in item["frames"]
        ]
        if not decoded_items:
            return None

        if force:
            target_dtype = torch.float32
            spatial_tower = self.get_model().initialize_pi3x_geometry_tower(
                device=device,
                dtype=target_dtype,
            )
        else:
            spatial_tower = self.get_model().get_spatial_tower()
        if spatial_tower is None:
            raise RuntimeError(
                "PI3X Geometry-RoPE from decoded_features requires --spatial_tower pi3x "
                "or geometry_spatial_tower_type=pi3x so the frozen PI3X point/camera/conf heads are available."
            )
        if hasattr(spatial_tower, "load_model") and not getattr(spatial_tower, "is_loaded", True):
            spatial_tower.load_model()
        pi3_model = getattr(spatial_tower, "pi3_model", None)
        if pi3_model is None:
            raise RuntimeError("PI3X spatial tower does not expose pi3_model for geometry decoding.")

        decoded_geometry = [
            self._decode_single_pi3x_geometry_output(item, pi3_model, device)
            for item in decoded_items
        ]
        merged = {}
        for key in decoded_geometry[0].keys():
            values = [item[key] for item in decoded_geometry if key in item and item[key] is not None]
            if not values:
                continue
            merged[key] = torch.cat(values, dim=0) if len(values) > 1 else values[0]
        return merged

    @staticmethod
    def _module_param_dtype(module, fallback):
        for param in module.parameters():
            return param.dtype
        return fallback

    @staticmethod
    def _module_param_device(module, fallback):
        for param in module.parameters():
            return param.device
        return fallback

    def _decode_single_pi3x_geometry_output(self, spatial_feature, pi3_model, device):
        sf = Pi3XDecodedFeatures.from_loaded(spatial_feature)
        decoded_features = sf.decoded_features.to(device=device)
        num_frames = int(decoded_features.shape[0])
        input_h = input_w = int(sf.input_size)
        patch_h = patch_w = int(sf.input_size // sf.patch_size)
        token_count = int(decoded_features.shape[1])
        expected_tokens = int(patch_h * patch_w + sf.patch_start_idx)
        if token_count != expected_tokens:
            raise RuntimeError(
                "PI3X decoded feature token count does not match stored input/patch metadata: "
                f"tokens={token_count}, expected={expected_tokens}, input_size={sf.input_size}, patch_size={sf.patch_size}"
            )

        pos = sf.decoded_pos_template.to(device=device)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0).expand(num_frames, -1, -1)
        elif pos.dim() == 3 and pos.shape[0] != num_frames:
            pos = pos[:1].expand(num_frames, -1, -1)
        if pos.dtype not in (torch.int32, torch.int64):
            pos = pos.to(dtype=torch.int64)

        head_dtype = self._module_param_dtype(pi3_model.point_decoder, torch.float32)
        if head_dtype == torch.bfloat16:
            pi3_model.to(dtype=torch.float32)
            head_dtype = torch.float32
        hidden = decoded_features.to(dtype=head_dtype)
        pos = pos.to(device=hidden.device).contiguous()
        patch_start_idx = int(sf.patch_start_idx)

        with torch.no_grad():
            autocast_device = hidden.device.type if hidden.device.type in {"cuda", "cpu"} else "cuda"
            with torch.amp.autocast(device_type=autocast_device, enabled=False):
                point_hidden = pi3_model.point_decoder(hidden, xpos=pos)
                camera_hidden = pi3_model.camera_decoder(hidden, xpos=pos)
                conf_hidden = pi3_model.conf_decoder(hidden, xpos=pos)

                point_tokens = point_hidden[:, patch_start_idx:]
                conf_tokens = conf_hidden[:, patch_start_idx:]
                camera_tokens = camera_hidden[:, patch_start_idx:]

                if hasattr(pi3_model, "_chunked_conv_head"):
                    point_tokens = point_tokens.to(dtype=self._module_param_dtype(pi3_model.point_head, point_tokens.dtype)).contiguous()
                    xy, z = pi3_model._chunked_conv_head(pi3_model.point_head, point_tokens, patch_h, patch_w)
                    xy = xy.float().permute(0, 2, 3, 1).reshape(1, num_frames, input_h, input_w, -1)
                    z = z.float().permute(0, 2, 3, 1).reshape(1, num_frames, input_h, input_w, -1)
                    z = torch.exp(z.clamp(max=15.0))

                    conf_tokens = conf_tokens.to(dtype=self._module_param_dtype(pi3_model.conf_head, conf_tokens.dtype)).contiguous()
                    conf = pi3_model._chunked_conv_head(pi3_model.conf_head, conf_tokens, patch_h, patch_w)[0]
                    conf = conf.float().permute(0, 2, 3, 1).reshape(1, num_frames, input_h, input_w, -1)
                else:
                    point_tokens = point_tokens.to(dtype=self._module_param_dtype(pi3_model.point_head, point_tokens.dtype)).contiguous()
                    ret = pi3_model.point_head([point_tokens], (input_h, input_w)).float().reshape(
                        1, num_frames, input_h, input_w, -1
                    )
                    xy, z = ret.split([2, 1], dim=-1)
                    z = torch.exp(z)

                    conf_tokens = conf_tokens.to(dtype=self._module_param_dtype(pi3_model.conf_head, conf_tokens.dtype)).contiguous()
                    conf = pi3_model.conf_head([conf_tokens], (input_h, input_w)).float().reshape(
                        1, num_frames, input_h, input_w, -1
                    )

                local_points = torch.cat([xy * z, z], dim=-1)

                camera_tokens = camera_tokens.to(dtype=self._module_param_dtype(pi3_model.camera_head, camera_tokens.dtype)).contiguous()
                camera_poses = pi3_model.camera_head(camera_tokens, patch_h, patch_w).float().reshape(1, num_frames, 4, 4)

                ones = torch.ones_like(local_points[..., :1])
                homogeneous_local_points = torch.cat([local_points, ones], dim=-1)
                points = torch.einsum("bnij,bnhwj->bnhwi", camera_poses, homogeneous_local_points)[..., :3]

                metric = None
                if all(hasattr(pi3_model, attr) for attr in ("metric_decoder", "metric_head", "metric_token")):
                    pos_hw = pos.reshape(1, num_frames * token_count, -1)
                    metric_hidden = pi3_model.metric_decoder(
                        pi3_model.metric_token.repeat(1, 1, 1).to(device=hidden.device, dtype=hidden.dtype),
                        hidden.reshape(1, num_frames * token_count, -1),
                        xpos=pos_hw[:, 0:1],
                        ypos=pos_hw,
                    )
                    metric_hidden = metric_hidden.to(dtype=self._module_param_dtype(pi3_model.metric_head, metric_hidden.dtype))
                    metric = pi3_model.metric_head(metric_hidden).reshape(1).float().exp()
                    points = points * metric.view(1, 1, 1, 1, 1)
                    local_points = local_points * metric.view(1, 1, 1, 1, 1)

        return {
            "point_map": points.squeeze(0).detach(),
            "points": points.squeeze(0).detach(),
            "local_points": local_points.squeeze(0).detach(),
            "depth": local_points.squeeze(0)[..., 2].detach(),
            "confidence": conf.squeeze(0).detach(),
            "camera_pose": camera_poses.squeeze(0).detach(),
            "metric": None if metric is None else metric.detach(),
        }

    def _apply_geometry_aware_projection(
        self,
        image_features,
        geometry_outputs=None,
        point_maps=None,
        spatial_features=None,
        split_sizes=None,
    ):
        module = self.get_model().get_geometry_aware_projection()
        if module is None:
            raise RuntimeError("use_geometry_aware_projection=True but geometry_aware_projection is not initialized.")
        canonical_geometry = self._canonical_geometry_outputs_for_projection(
            geometry_outputs=geometry_outputs,
            point_maps=point_maps,
            spatial_features=spatial_features,
            device=image_features.device,
        )
        if not canonical_geometry:
            raise RuntimeError(
                "use_geometry_aware_projection=True requires explicit geometry_outputs, legacy point_maps, "
                "or spatial_features containing point_map/depth/confidence aliases."
            )

        prefix_tokens, grid_tokens = self._split_geometry_visual_tokens(image_features)
        expected_frames = int(grid_tokens.shape[0])
        for key, value in canonical_geometry.items():
            if not isinstance(value, torch.Tensor):
                continue
            geo_frames = int(value.shape[0])
            if value.dim() == 5:
                geo_frames = int(value.shape[0] * value.shape[1])
            elif key in {"depth", "confidence"} and value.dim() == 4 and value.shape[1] != 1 and value.shape[-1] != 1:
                geo_frames = int(value.shape[0] * value.shape[1])
            if geo_frames != expected_frames:
                raise RuntimeError(
                    f"Geometry-aware projection frame count mismatch for {key}: "
                    f"geometry has {geo_frames} frames but visual grid tokens have {expected_frames}. "
                    f"geometry shape={tuple(value.shape)}, grid_tokens shape={tuple(grid_tokens.shape)}"
                )
        visual_grid_size = self._geometry_visual_grid_size(grid_tokens.shape[1])
        geo_out = module(
            visual_tokens=grid_tokens,
            geometry_outputs=canonical_geometry,
            visual_grid_size=visual_grid_size,
            num_frames=int(sum(split_sizes)) if split_sizes is not None else int(grid_tokens.shape[0]),
            spatial_merge_size=getattr(self.config, "spatial_merge_size", None),
        )
        refined_tokens = geo_out["refined_tokens"]
        if prefix_tokens is not None:
            refined_tokens = torch.cat((prefix_tokens, refined_tokens), dim=1)
        self.get_model()._last_geometry_projection_outputs = geo_out
        self.get_model()._last_geometry_projection_metrics = {
            "loss_geo": None if geo_out.get("loss_geo") is None else float(geo_out["loss_geo"].detach().float().item()),
            "valid_geometry_tokens": int(geo_out["geometry_mask"].detach().bool().sum().item()),
        }
        return refined_tokens

    # def encode_images(self, images):
    #     # vision features
    #     image_features = self.get_model().get_vision_tower()(images)
    #     # set brance
    #     if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
    #         # spatial features
    #         spatial_encoder_type = self.get_model().config.spatial_tower
    #         if spatial_encoder_type == "cut3r":
    #             # Scale up image by 16/14 before passing to spatial tower
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "vggt":
    #             images_scaled = nn.functional.interpolate(images, size=(378, 378), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         elif spatial_encoder_type == "cut3r_points":
    #             images_scaled = nn.functional.interpolate(images, size=(432, 432), mode='bilinear')
    #             images_for_spatial_tower = images_scaled.unsqueeze(1) ## FIXME: the first dimension is the number of frames in one batch
    #             image_spatial_features = self.get_model().get_spatial_tower()(images_for_spatial_tower)
    #         else:
    #             raise ValueError(f"Unexpected spatial encoder type: {spatial_encoder_type}")
            
    #         fusion_block_type = self.get_model().config.fusion_block
            
    #         # Handle special case for mlp2x_gelu_cat first
    #         if fusion_block_type == "mlp2x_gelu_cat":
    #             image_features = torch.cat((image_features, image_spatial_features), dim=-1)
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #         # Handle special case for mlp2x_gelu
    #         elif fusion_block_type == "mlp2x_gelu":
    #             image_features = self.get_model().get_fusion_block()(image_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         # Handle all other fusion types that follow the same pattern
    #         elif fusion_block_type in ["cross_attention", "mlp", "transformer"]:
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         elif fusion_block_type == "llava_3d_fusion_block":
    #             image_features = self.get_model().get_fusion_block()(image_features, image_spatial_features)
    #             image_features = self.get_model().mm_projector(image_features)
    #         else:
    #             raise ValueError(f"Unexpected fusion block type: {fusion_block_type}")
    #     else:
    #         # project features
    #         image_features = self.get_model().mm_projector(image_features)

    #     return image_features

    def encode_images(self, images, spatial_features=None, geometry_spatial_features=None, point_maps=None, geometry_outputs=None, split_sizes=None):
        # vision features
        image_features = self.get_model().get_vision_tower()(images)
        self.get_model()._last_geometry_projection_outputs = None
        self.get_model()._last_geometry_projection_metrics = None

        use_geometry_projection = _as_bool_config(
            getattr(self.get_model().config, "use_geometry_aware_projection", False),
            False,
        )
        if use_geometry_projection:
            zero_spatial_features = getattr(self.get_model().config, "zero_spatial_features", False)
            if isinstance(zero_spatial_features, str):
                zero_spatial_features = zero_spatial_features.lower() in {"1", "true", "yes", "y", "on"}
            if zero_spatial_features:
                return self.get_model().mm_projector(image_features)
            if geometry_outputs is None and point_maps is None and spatial_features is None:
                geometry_outputs = self._runtime_geometry_outputs_for_projection(images)
            image_features = self._apply_geometry_aware_projection(
                image_features=image_features,
                geometry_outputs=geometry_outputs,
                point_maps=point_maps,
                spatial_features=spatial_features,
                split_sizes=split_sizes,
            )
            return self.get_model().mm_projector(image_features)

        # fuse with spatial features
        if self.get_model().get_spatial_tower() is not None and self.get_model().get_fusion_block() is not None:
            spatial_encoder_type = self.get_model().config.spatial_tower
            fusion_block_type = self.get_model().config.fusion_block
            spatial_tower = self.get_model().get_spatial_tower()
            preextracted_only = _as_bool_config(
                getattr(self.get_model().config, "spatial_tower_preextracted_only", False),
                False,
            )

            def ensure_spatial_tower_loaded():
                if spatial_tower is None:
                    return
                if getattr(spatial_tower, "is_loaded", True):
                    return
                if preextracted_only:
                    raise RuntimeError(
                        "spatial_tower_preextracted_only=True disables runtime CUT3R tower loading. "
                        "Forward requires pre-extracted spatial_features with camera_tokens and "
                        "patch_tokens; use SPATIAL_FEATURES_SUBDIR=spatial_features and pass "
                        "point-map sidecars separately as geometry_spatial_features."
                    )
                load_model = getattr(spatial_tower, "load_model", None)
                if callable(load_model):
                    load_model()
                    spatial_tower.to(device=images.device, dtype=image_features.dtype)

            zero_spatial_features = getattr(self.get_model().config, "zero_spatial_features", False)
            if isinstance(zero_spatial_features, str):
                zero_spatial_features = zero_spatial_features.lower() in {"1", "true", "yes", "y", "on"}

            # Keep zero-spatial ablation simple and deterministic: pure SigLIP path.
            # This bypasses spatial tower and fusion block entirely.
            if zero_spatial_features:
                return self.get_model().mm_projector(image_features)

            if spatial_encoder_type.endswith("points"):
                ensure_spatial_tower_loaded()
                points = spatial_tower(images)
                image_features = self.get_model().get_fusion_block()(image_features, points)
                image_features = self.get_model().mm_projector(image_features)
            
            else:
                cfg_spatial_tower_name = str(spatial_encoder_type or "").lower()
                runtime_spatial_tower_name = str(
                    getattr(spatial_tower, "spatial_tower_name", None)
                    or getattr(spatial_tower, "tower_name", None)
                    or getattr(spatial_tower, "model_name", None)
                    or ""
                ).lower()
                spatial_tower_module = ""
                spatial_tower_class_name = ""
                if spatial_tower is not None:
                    spatial_tower_module = getattr(spatial_tower.__class__, "__module__", "").lower()
                    spatial_tower_class_name = spatial_tower.__class__.__name__.lower()

                is_cut3r_spatial = any(
                    "cut3r" in value
                    for value in (
                        cfg_spatial_tower_name,
                        runtime_spatial_tower_name,
                        spatial_tower_module,
                        spatial_tower_class_name,
                    )
                )
                is_pi3x_spatial = any(
                    "pi3x" in value
                    for value in (
                        cfg_spatial_tower_name,
                        runtime_spatial_tower_name,
                        spatial_tower_module,
                        spatial_tower_class_name,
                    )
                )
                is_vggt_spatial = any(
                    "vggt" in value
                    for value in (
                        cfg_spatial_tower_name,
                        runtime_spatial_tower_name,
                        spatial_tower_module,
                        spatial_tower_class_name,
                    )
                )
                loaded_spatial_features = spatial_features[0] if spatial_features is not None else None
                has_token_pair_features = (
                    isinstance(loaded_spatial_features, dict)
                    and "camera_tokens" in loaded_spatial_features
                    and "patch_tokens" in loaded_spatial_features
                )

                _sf = None
                camera_pose = None

                if spatial_features is not None and has_token_pair_features and not is_pi3x_spatial:
                    camera_tokens = loaded_spatial_features["camera_tokens"]
                    patch_tokens = loaded_spatial_features["patch_tokens"]
                elif spatial_features is not None and is_pi3x_spatial:
                    ensure_spatial_tower_loaded()
                    _sf = Pi3XDecodedFeatures.from_loaded(loaded_spatial_features)
                    if _sf.is_new_schema():
                        # Camera tokens must be computed from the stored decoded_features
                        # by running pi3.camera_decoder (lightweight head, no re-encoding).
                        _spatial_tower = spatial_tower
                        _cam_dec = getattr(_spatial_tower, "camera_decoder", None)
                        if _cam_dec is None:
                            raise RuntimeError(
                                "Pi3X spatial tower must be loaded to compute camera_tokens "
                                "from decoded_features. Ensure spatial_tower is not None."
                            )
                        _sf.compute_camera_tokens(
                            _cam_dec,
                            device=images.device,
                            dtype=self.dtype,
                        )
                        # svf_pose_prepend needs camera_head to get the 12-value pose.
                        if fusion_block_type == 'svf_pose_prepend':
                            _cam_head = getattr(_spatial_tower, "camera_head", None)
                            if _cam_head is None:
                                raise RuntimeError(
                                    "svf_pose_prepend requires pi3.camera_head. "
                                    "Ensure Pi3X spatial tower is loaded."
                                )
                            _patch_h = _patch_w = _sf.input_size // _sf.patch_size
                            _sf.compute_camera_pose(_cam_head, _patch_h, _patch_w, device=images.device)
                            camera_pose = _sf.camera_pose
                    camera_tokens, patch_tokens = _sf.camera_tokens, _sf.patch_tokens
                else:
                    ensure_spatial_tower_loaded()
                    if is_vggt_spatial and split_sizes is not None:
                        if images.ndim != 4:
                            raise RuntimeError(
                                "VGGT grouped runtime path expects flattened 4D images, "
                                f"got shape {tuple(images.shape)}"
                            )
                        if sum(split_sizes) != images.shape[0]:
                            raise RuntimeError(
                                "VGGT split_sizes must sum to the flattened image count, "
                                f"got split_sizes={split_sizes} and images={tuple(images.shape)}"
                            )
                        camera_token_chunks = []
                        patch_token_chunks = []
                        for image_chunk in torch.split(images, split_sizes, dim=0):
                            chunk_camera_tokens, chunk_patch_tokens = spatial_tower(image_chunk)
                            camera_token_chunks.append(chunk_camera_tokens)
                            patch_token_chunks.append(chunk_patch_tokens)
                        camera_tokens = torch.cat(camera_token_chunks, dim=0)
                        patch_tokens = torch.cat(patch_token_chunks, dim=0)
                    else:
                        camera_tokens, patch_tokens = spatial_tower(images)
                    # Runtime path parity for svf_pose_prepend:
                    # compute camera_pose from camera_head using runtime camera_tokens.
                    if fusion_block_type == 'svf_pose_prepend':
                        _spatial_tower = spatial_tower
                        _cam_head = getattr(_spatial_tower, "camera_head", None)
                        if _cam_head is None:
                            raise RuntimeError(
                                "svf_pose_prepend requires pi3.camera_head in runtime path. "
                                "Ensure Pi3X spatial tower is loaded."
                            )

                        patch_token_num = int(patch_tokens.shape[1])
                        patch_side = int(math.isqrt(patch_token_num))
                        if patch_side * patch_side != patch_token_num:
                            raise RuntimeError(
                                f"svf_pose_prepend runtime path expects square patch grid, got {patch_token_num} tokens"
                            )

                        cam_tokens_for_pose = camera_tokens
                        cam_head_param = next(_cam_head.parameters(), None)
                        if cam_head_param is not None and cam_tokens_for_pose.dtype != cam_head_param.dtype:
                            cam_tokens_for_pose = cam_tokens_for_pose.to(dtype=cam_head_param.dtype)

                        with torch.no_grad():
                            pose_4x4 = _cam_head(cam_tokens_for_pose, patch_side, patch_side)
                        camera_pose = pose_4x4[:, :3, :].reshape(pose_4x4.shape[0], 12)

                camera_tokens, patch_tokens, camera_pose = _maybe_swap_spatial_frame_features(
                    self.get_model().config,
                    camera_tokens,
                    patch_tokens,
                    camera_pose,
                    self.training,
                )
                
                if fusion_block_type in ['cross_attention', 'svf_baseline']:
                    # Build spatial KV tokens.
                    if fusion_block_type == 'svf_baseline':
                        # Ablation-1: Q=2D, KV=[camera, patch], output=2D+cross_attn.
                        if camera_tokens.shape[-1] != patch_tokens.shape[-1]:
                            raise ValueError(
                                "svf_baseline requires camera_tokens and patch_tokens to share the same "
                                f"feature dimension, got {camera_tokens.shape[-1]} and {patch_tokens.shape[-1]}. "
                                "Add a camera projection layer before using svf_baseline."
                            )
                        final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(self.dtype)
                    else:
                        # Legacy cross_attention keeps runtime-selectable spatial token composition.
                        spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                        spatial_tower_select_feature_list = [
                            feature.strip()
                            for feature in spatial_tower_select_feature.split(",")
                            if feature.strip()
                        ]
                        selected_tokens = []
                        for spatial_tower_select_feature in spatial_tower_select_feature_list:
                            if spatial_tower_select_feature == "camera_tokens":
                                selected_tokens.append(camera_tokens)
                            elif spatial_tower_select_feature == "patch_tokens":
                                selected_tokens.append(patch_tokens)
                            elif spatial_tower_select_feature in ["all", "all_tokens"]:
                                selected_tokens = [camera_tokens, patch_tokens]
                            else:
                                raise ValueError(f"Unexpected spatial_tower_select_feature: {spatial_tower_select_feature}")
                        if not selected_tokens:
                            raise ValueError("spatial_tower_select_feature must select at least one token stream.")
                        selected_dims = {int(tokens.shape[-1]) for tokens in selected_tokens}
                        if len(selected_dims) != 1:
                            raise ValueError(
                                "cross_attention requires selected spatial tokens to share the same "
                                f"feature dimension, got {sorted(selected_dims)} from "
                                f"spatial_tower_select_feature='{spatial_tower_select_feature}'."
                            )
                        final_image_features = torch.cat(selected_tokens, dim=1).to(self.dtype)

                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, final_image_features)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_patch_cam_concat':
                    # Ablation-2: Q=2D, KV=patch only, then prepend projected camera tokens.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        patch_tokens.to(self.dtype),
                        camera_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_geometry_bridge':
                    # Ablation-3: camera->patch attention builds geometry-aware tokens, then 2D queries those tokens.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)
                
                elif fusion_block_type == 'svf_patch_only':
                    # Baseline: 2D cross-attends patch_tokens only.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type in ['svf_patch_only_geo_rope_eval',
                                           'svf_geo_rope_fusion', 'svf_geo_rope_fusion_forced', 'svf_geo_rope_fusion_per_head_gate',
                                           'svf_depth_geo_rope_fusion', 'svf_xyz_geo_rope_fusion', 'svf_spherical_geo_rope_fusion',
                                           'svf_depth_rope', 'svf_xyz_rope', 'svf_spherical_rope']:
                    geometry_point_maps = None
                    if geometry_spatial_features is not None:
                        loaded_geometry_features = (
                            geometry_spatial_features[0]
                            if isinstance(geometry_spatial_features, (list, tuple)) and len(geometry_spatial_features) > 0
                            else geometry_spatial_features
                        )
                        has_cut3r_point_maps = (
                            isinstance(loaded_geometry_features, dict)
                            and any(
                                key in loaded_geometry_features
                                for key in (
                                    "point_maps_ref",
                                    "pts3d_in_other_view",
                                    "point_maps_cam",
                                    "pts3d_in_self_view",
                                    "point_maps",
                                    "point_map",
                                    "points",
                                    "pts3d",
                                )
                            )
                        )
                        if has_cut3r_point_maps:
                            geometry_point_maps, _geometry_source = _resolve_llm_geo_point_maps(
                                self.get_model().config,
                                loaded_spatial_features=loaded_geometry_features,
                            )
                        else:
                            pi3x_geometry = self._decode_pi3x_geometry_outputs_for_projection(
                                geometry_spatial_features,
                                device=image_features.device,
                                force=True,
                            )
                            if pi3x_geometry is not None:
                                geometry_point_maps = _coalesce_point_maps(
                                    pi3x_geometry.get("point_map", pi3x_geometry.get("points"))
                                )
                    if geometry_point_maps is None:
                        geometry_point_maps = _coalesce_point_maps(point_maps)
                    if geometry_point_maps is None and isinstance(loaded_spatial_features, dict):
                        # Coordinate consistency rule:
                        # For CUT3R point-map sidecars, point_maps_ref /
                        # pts3d_in_other_view are reference/anchor-frame
                        # coordinates, while point_maps_cam /
                        # pts3d_in_self_view are per-frame camera coordinates.
                        # Train and eval for the same checkpoint must select
                        # the same coordinate source; do not add eval-only
                        # aliases that silently change this priority.
                        requested_point_map_key = (
                            getattr(self.get_model().config, "geo_rope_point_map_key", None)
                            or getattr(self.get_model().config, "geometry_point_map_key", None)
                        )
                        if requested_point_map_key:
                            point_map_keys = _point_map_key_candidates(requested_point_map_key)
                            missing_keys = ", ".join(point_map_keys)
                            for point_key in point_map_keys:
                                if point_key in loaded_spatial_features:
                                    geometry_point_maps = _coalesce_point_maps(loaded_spatial_features[point_key])
                                    break
                            if geometry_point_maps is None:
                                raise RuntimeError(
                                    f"{fusion_block_type} requested geo_rope_point_map_key="
                                    f"{requested_point_map_key!r}, but none of [{missing_keys}] "
                                    "were found in the loaded CUT3R sidecar."
                                )
                        else:
                            point_map_keys = (
                                "point_maps",
                                "point_map",
                                "points",
                                "pts3d",
                                "point_maps_ref",
                                "pts3d_in_other_view",
                                "point_maps_cam",
                                "pts3d_in_self_view",
                            )
                            for point_key in point_map_keys:
                                if point_key in loaded_spatial_features:
                                    geometry_point_maps = _coalesce_point_maps(loaded_spatial_features[point_key])
                                    break
                    if geometry_point_maps is None:
                        loaded_keys = (
                            sorted(loaded_spatial_features.keys())
                            if isinstance(loaded_spatial_features, dict)
                            else None
                        )
                        requested_point_map_key = (
                            getattr(self.get_model().config, "geo_rope_point_map_key", None)
                            or getattr(self.get_model().config, "geometry_point_map_key", None)
                        )
                        raise RuntimeError(
                            f"{fusion_block_type} requires point_maps from CUT3R or "
                            "geometry_spatial_features decoded from PI3X/CUT3R. Expected one of: "
                            "geometry_spatial_features, point_maps, point_maps_ref, "
                            "point_maps_cam, pts3d_in_other_view, pts3d_in_self_view. "
                            f"Debug: requested_point_map_key={requested_point_map_key!r}, "
                            f"spatial_features_type={type(spatial_features).__name__}, "
                            f"loaded_spatial_features_type={type(loaded_spatial_features).__name__}, "
                            f"loaded_spatial_features_keys={loaded_keys}, "
                            f"point_maps_type={type(point_maps).__name__}, "
                            f"geometry_spatial_features_type={type(geometry_spatial_features).__name__}, "
                            f"modalities={locals().get('modalities', None)}, "
                            f"split_sizes={split_sizes}."
                        )

                    if geometry_point_maps.shape[0] != image_features.shape[0]:
                        raise RuntimeError(
                            f"{fusion_block_type} point_maps batch/frame count must match image_features, "
                            f"got point_maps={tuple(geometry_point_maps.shape)} and "
                            f"image_features={tuple(image_features.shape)}"
                        )

                    geometry_point_maps = _maybe_shuffle_geometry_point_maps(
                        self.get_model().config,
                        geometry_point_maps,
                        self.training,
                    )
                    pos_clip = pool_point_map_to_tokens(geometry_point_maps, image_features.shape[1])
                    pos_spatial = pool_point_map_to_tokens(geometry_point_maps, patch_tokens.shape[1])
                    pos_clip, pos_spatial = _maybe_apply_intra_frame_pos_shuffle_probe(
                        self.get_model().config,
                        pos_clip,
                        pos_spatial,
                        self.training,
                    )
                    cross_frame_result = _maybe_apply_cross_frame_geo_rope_probe(
                        self.get_model().config,
                        self.get_model().get_fusion_block(),
                        image_features,
                        patch_tokens,
                        pos_clip,
                        pos_spatial,
                        self.dtype,
                        self.training,
                    )
                    if cross_frame_result is not None:
                        image_features, attn_weights = cross_frame_result
                    else:
                        image_features, attn_weights = self.get_model().get_fusion_block()(
                            image_features,
                            patch_tokens.to(self.dtype),
                            pos_clip.to(device=image_features.device, dtype=image_features.dtype),
                            pos_spatial.to(device=image_features.device, dtype=image_features.dtype),
                        )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_cat_feat':
                    # Comparison 1: feature-dim concat [camera‖patch] as single KV stream.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_geometry_bridge':
                    # Comparison 2: camera tokens (from camera_decoder branch) query
                    # patch tokens to build geometry-aware tokens, then 2D queries them.
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_tokens.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'svf_pose_prepend':
                    # Comparison 3: 1 pose token (12-value camera matrix → Linear(12,d_clip))
                    # prepended to 2D-fused sequence. camera_pose = (F, 12).
                    if camera_pose is None:
                        raise RuntimeError(
                            "svf_pose_prepend: camera_pose not computed. "
                            "Check that the preextracted branch ran compute_camera_pose()."
                        )
                    image_features, attn_weights = self.get_model().get_fusion_block()(
                        image_features,
                        camera_pose.to(self.dtype),
                        patch_tokens.to(self.dtype),
                    )
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'cross_attention_with_mlp':
                    image_features, attn_weights = self.get_model().get_fusion_block()(image_features, patch_tokens)
                    image_features = self.get_model().mm_projector(image_features)

                elif fusion_block_type == 'transformer':
                    spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
                    if spatial_tower_select_feature in ["all", "all_tokens"]:
                        final_image_features = torch.cat((camera_tokens, patch_tokens), dim=1).to(self.dtype)
                        image_features = self.get_model().get_fusion_block()(image_features, final_image_features)
                        image_features = self.get_model().mm_projector(image_features)

                elif (fusion_block_type == 'mlp_after_clip_proj' 
                      or fusion_block_type == 'concat_mlp'
                      or fusion_block_type == 'concat_self_attention'):

                    image_features = self.get_model().mm_projector(image_features)
                    image_features = self.get_model().get_fusion_block()(image_features, patch_tokens)

                else:
                    raise ValueError(f"Unexpected fusion block type: {fusion_block_type}")

        elif self.get_model().get_spatial_tower() is None and self.get_model().get_fusion_block() is not None:
            assert point_maps is not None
            image_features = self.get_model().mm_projector(image_features)
            image_features = self.get_model().get_fusion_block()(image_features, point_maps[0]) # FIXME: point_maps is a list of tensors, each tensor is a point map for one image

        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        prefix_tokens, image_feature, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            if prefix_tokens is not None:
                image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
        image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
        image_feature = image_feature.flatten(1, 2)
        if prefix_tokens is not None:
            image_feature = torch.cat((prefix_tokens, image_feature), dim=1)
        image_feature = image_feature.flatten(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, spatial_features=None, point_maps=None, modalities=["image"], image_sizes=None, return_visual_metadata=False, geometry_outputs=None, geometry_spatial_features=None, return_llm_geo_metadata=False):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            result = (input_ids, position_ids, attention_mask, past_key_values, None, labels)
            extras = []
            if return_visual_metadata:
                extras.append(None)
            if return_llm_geo_metadata:
                extras.extend([None, None, {"skip_reason": "no_multimodal_prefill"}])
            if extras:
                return result + tuple(extras)
            return result

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(
                concat_images,
                spatial_features=spatial_features,
                geometry_spatial_features=geometry_spatial_features,
                point_maps=point_maps,
                geometry_outputs=geometry_outputs,
                split_sizes=split_sizes,
            )
            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is None:
            #         camera_tokens, patch_tokens = self.encode_spatial_features(concat_images)
            #     else:
            #         camera_tokens, patch_tokens = spatial_features[0]["camera_tokens"], spatial_features[0]["patch_tokens"]
            #     # fuse with spatial features
            #     spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "patch_tokens")
            #     spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
            #     final_image_features = []
            #     for spatial_tower_select_feature in spatial_tower_select_feature_list:
            #         if spatial_tower_select_feature == "camera_tokens":
            #             final_image_features.append(camera_tokens)
            #         elif spatial_tower_select_feature == "patch_tokens":
            #             final_image_features.append(patch_tokens)
            #     final_image_features = torch.cat(final_image_features, dim=1)
            #     encoded_image_features = self.get_model().get_fusion_block()(encoded_image_features, final_image_features)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            llm_geo_point_maps_by_image = []
            if return_llm_geo_metadata:
                for geo_idx in range(len(images_list)):
                    loaded_spatial_features = None
                    if isinstance(spatial_features, (list, tuple)) and geo_idx < len(spatial_features):
                        loaded_spatial_features = spatial_features[geo_idx]
                    elif isinstance(spatial_features, dict):
                        loaded_spatial_features = spatial_features
                    geo_maps, geo_source = _resolve_llm_geo_point_maps(
                        self.get_model().config,
                        loaded_spatial_features=loaded_spatial_features,
                        point_maps=point_maps,
                    )
                    if geo_maps is not None and int(geo_maps.shape[0]) != int(images_list[geo_idx].shape[0]):
                        raise RuntimeError(
                            "LLM visual 3D RoPE point-map frame count must match sampled frames, "
                            f"got point_maps={tuple(geo_maps.shape)} images={tuple(images_list[geo_idx].shape)}"
                        )
                    llm_geo_point_maps_by_image.append((geo_maps, geo_source))
            image_features = []
            image_feature_layouts = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    raw_prefix, raw_grid_tokens, raw_side = self._split_prefix_tokens_for_square_grid(image_feat)
                    prefix_len = 0 if raw_prefix is None else int(raw_prefix.shape[1])
                    pooled_feat = self.get_2dPool(image_feat)
                    _, pooled_grid_tokens, pooled_side = self._split_prefix_tokens_for_square_grid(pooled_feat)
                    image_features.append(pooled_feat)
                    image_feature_layouts.append({
                        "modality": "video",
                        "num_frames": int(pooled_feat.shape[0]),
                        "raw_grid_side": int(raw_side),
                        "grid_side": int(pooled_side),
                        "prefix_len": int(prefix_len),
                    })
                else:
                    image_features.append(image_feat)
                    image_feature_layouts.append({"modality": "image"})
            
            # if self.get_model().get_spatial_tower() is not None:
            #     if spatial_features is not None:
            #         encoded_camera_tokens = spatial_features[0]["camera_tokens"] ## FIXME: spatial_features is a list of dicts, each dict contains camera_tokens and patch_tokens
            #         encoded_patch_tokens = spatial_features[0]["patch_tokens"]
            #         # fusion block
            #         encoded_camera_tokens, encoded_patch_tokens = self.get_model().get_fusion_block()(encoded_camera_tokens, encoded_patch_tokens)
            #     else:
            #         encoded_camera_tokens, encoded_patch_tokens = self.encode_spatial_features(concat_images)
            #     camera_tokens = torch.split(encoded_camera_tokens, split_sizes)
            #     encoded_patch_tokens = torch.split(encoded_patch_tokens, split_sizes)
            #     # split and merge
            #     patch_tokens = []
            #     # pool patch tokens
            #     for idx, patch_token in enumerate(encoded_patch_tokens):
            #         if idx in video_idx_in_batch:
            #             patch_tokens.append(self.get_2dPool(patch_token))
            #         else:
            #             patch_tokens.append(patch_token)

            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                new_image_feature_metadata = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            layout = image_feature_layouts[image_idx]
                            prefix_tokens, _, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
                            prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
                            metadata = self._build_grid_metadata(
                                num_frames=int(image_feature.shape[0]),
                                grid_side=int(resize_h),
                                raw_grid_side=int(layout.get("raw_grid_side", resize_h)),
                                prefix_len=prefix_len,
                                device=image_feature.device,
                            )
                            if return_llm_geo_metadata and image_idx < len(llm_geo_point_maps_by_image):
                                geo_maps, geo_source = llm_geo_point_maps_by_image[image_idx]
                                metadata = self._attach_llm_geo_positions_to_metadata(metadata, geo_maps, geo_source)
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                            new_image_feature_metadata.append(metadata)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            layout = image_feature_layouts[image_idx]
                            prefix_tokens, _, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
                            prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
                            metadata = self._build_flat_frame_metadata(
                                num_frames=int(image_feature.shape[0]),
                                tokens_per_frame=int(image_feature.shape[1]) + 1,
                                grid_side=int(resize_h),
                                raw_grid_side=int(layout.get("raw_grid_side", resize_h)),
                                prefix_len=prefix_len,
                                device=image_feature.device,
                            )
                            frame_len = int(image_feature.shape[1]) + 1
                            newline_positions = torch.tensor(
                                [frame_idx * frame_len + frame_len - 1 for frame_idx in range(int(image_feature.shape[0]))],
                                device=image_feature.device,
                                dtype=torch.long,
                            )
                            keep_visual = ~torch.isin(metadata["visual_token_indices"], newline_positions)
                            metadata["visual_token_indices"] = metadata["visual_token_indices"][keep_visual]
                            metadata["visual_frame_ids"] = metadata["visual_frame_ids"][keep_visual]
                            metadata["newline_token_indices"] = newline_positions
                            metadata["tokens_per_frame"] = [int(image_feature.shape[1]) - prefix_len for _ in range(int(image_feature.shape[0]))]
                            if return_llm_geo_metadata and image_idx < len(llm_geo_point_maps_by_image):
                                geo_maps, geo_source = llm_geo_point_maps_by_image[image_idx]
                                metadata = self._attach_llm_geo_positions_to_metadata(metadata, geo_maps, geo_source)
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            new_image_feature_metadata.append(metadata)
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            layout = image_feature_layouts[image_idx]
                            prefix_tokens, _, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
                            prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
                            metadata = self._build_flat_frame_metadata(
                                num_frames=int(image_feature.shape[0]),
                                tokens_per_frame=int(image_feature.shape[1]),
                                grid_side=int(resize_h),
                                raw_grid_side=int(layout.get("raw_grid_side", resize_h)),
                                prefix_len=prefix_len,
                                device=image_feature.device,
                            )
                            if return_llm_geo_metadata and image_idx < len(llm_geo_point_maps_by_image):
                                geo_maps, geo_source = llm_geo_point_maps_by_image[image_idx]
                                metadata = self._attach_llm_geo_positions_to_metadata(metadata, geo_maps, geo_source)
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                                metadata["newline_token_indices"] = torch.cat((
                                    metadata["newline_token_indices"],
                                    torch.tensor([image_feature.shape[0] - 1], device=image_feature.device, dtype=torch.long),
                                ))
                            new_image_features.append(image_feature)      
                            new_image_feature_metadata.append(metadata)
                        elif mm_newline_position == "no_token":
                            layout = image_feature_layouts[image_idx]
                            prefix_tokens, _, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
                            prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
                            metadata = self._build_flat_frame_metadata(
                                num_frames=int(image_feature.shape[0]),
                                tokens_per_frame=int(image_feature.shape[1]),
                                grid_side=int(resize_h),
                                raw_grid_side=int(layout.get("raw_grid_side", resize_h)),
                                prefix_len=prefix_len,
                                device=image_feature.device,
                            )
                            if return_llm_geo_metadata and image_idx < len(llm_geo_point_maps_by_image):
                                geo_maps, geo_source = llm_geo_point_maps_by_image[image_idx]
                                metadata = self._attach_llm_geo_positions_to_metadata(metadata, geo_maps, geo_source)
                            new_image_features.append(image_feature.flatten(0, 1))
                            new_image_feature_metadata.append(metadata)
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                        new_image_feature_metadata.append({"visual_token_indices": _empty_long(image_feature.device)})
                    else:  # single image operations
                        # For single images, apply the same grid-wise newline logic
                        # as used for video frames to maintain consistency.
                        prefix_tokens, _, resize_h = self._split_prefix_tokens_for_square_grid(image_feature)
                        prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
                        metadata = self._build_grid_metadata(
                            num_frames=int(image_feature.shape[0]),
                            grid_side=int(resize_h),
                            raw_grid_side=int(resize_h),
                            prefix_len=prefix_len,
                            device=image_feature.device,
                        )
                        if return_llm_geo_metadata and image_idx < len(llm_geo_point_maps_by_image):
                            geo_maps, geo_source = llm_geo_point_maps_by_image[image_idx]
                            metadata = self._attach_llm_geo_positions_to_metadata(metadata, geo_maps, geo_source)
                        image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)
                        new_image_feature_metadata.append(metadata)
                image_features = new_image_features
                image_feature_metadata = new_image_feature_metadata
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            split_sizes = [1] * images.shape[0] if getattr(images, "ndim", 0) == 4 else None
            image_features = self.encode_images(
                images,
                spatial_features=spatial_features,
                geometry_spatial_features=geometry_spatial_features,
                point_maps=point_maps,
                geometry_outputs=geometry_outputs,
                split_sizes=split_sizes,
            )
            image_feature_metadata = [{"visual_token_indices": _empty_long(image_features.device)}]

        if "image_feature_metadata" not in locals():
            image_feature_metadata = []
            for image_feature in image_features:
                image_feature_metadata.append({"visual_token_indices": _empty_long(image_feature.device)})

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        unpadded_visual_metadata = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                # Handle cases with no image tokens if necessary.
                # Original code appends empty features, adapt if needed.
                cur_image_features = image_features[cur_image_idx]
                # Also get corresponding spatial features
                # if self.get_model().get_spatial_tower() is not None:
                #     cur_camera_tokens = camera_tokens[cur_image_idx]
                #     cur_patch_tokens = patch_tokens[cur_image_idx]

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)

                # Concatenate text embeds, visual embeds [0:0], and spatial embeds [0:0]?
                # This part of original code seems odd (using [0:0]), clarify its purpose.
                # Assuming you want to append actual features if available, otherwise skip.
                embeds_to_concat = [cur_input_embeds_1]
                # if cur_image_features is not None and cur_image_features.numel() > 0:
                #     embeds_to_concat.append(cur_image_features[0:0]) # Original behavior

                cur_input_embeds = torch.cat(embeds_to_concat, dim=0)

                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                unpadded_visual_metadata.append({
                    "visual_token_indices": _empty_long(cur_input_embeds.device),
                    "visual_frame_ids": _empty_long(cur_input_embeds.device),
                })
                cur_image_idx += 1 # Increment even if no image token? Check original logic intent.
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_visual_metadata = None

            for i in range(num_images + 1):
                # Append text embeddings and labels
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                # If this segment was followed by an image token, insert features
                if i < num_images:
                    try:
                        # Get the visual/fused features for the current image
                        cur_image_features = image_features[cur_image_idx]
                        # Get the spatial features for the current image
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx]
                    except IndexError:
                         # Fallback logic from original code
                        cur_image_features = image_features[cur_image_idx - 1]
                        # if self.get_model().get_spatial_tower() is not None:
                        #     cur_camera_tokens = camera_tokens[cur_image_idx - 1]
                        #     cur_patch_tokens = patch_tokens[cur_image_idx - 1]

                    cur_image_idx += 1

                    # Prepare combined features (visual + spatial)
                    features_to_insert = []
                    if cur_image_features is not None and cur_image_features.shape[0] > 0:
                        features_to_insert.append(cur_image_features)
                    # spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", None)
                    # if self.get_model().get_spatial_tower() is not None and spatial_tower_select_feature is not None:
                    #     spatial_feature_flags = spatial_tower_select_feature.split(",")
                        
                    #     if cur_camera_tokens is not None and cur_camera_tokens.shape[0] > 0 and "camera_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_camera_tokens.flatten(0, 1))
                    #     if cur_patch_tokens is not None and cur_patch_tokens.shape[0] > 0 and "patch_tokens" in spatial_feature_flags:
                    #         features_to_insert.append(cur_patch_tokens.flatten(0, 1))

                    if features_to_insert:
                        combined_features = torch.cat(features_to_insert, dim=0)
                        insert_start = sum(x.shape[0] for x in cur_new_input_embeds)
                        if cur_visual_metadata is None:
                            cur_visual_metadata = self._shift_metadata(
                                image_feature_metadata[cur_image_idx - 1],
                                offset=insert_start,
                            )
                        cur_new_input_embeds.append(combined_features)
                        # Add IGNORE_INDEX labels for the entire combined feature length
                        cur_new_labels.append(torch.full((combined_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if cur_visual_metadata is None:
                cur_visual_metadata = {
                    "visual_token_indices": _empty_long(cur_new_input_embeds.device),
                    "visual_frame_ids": _empty_long(cur_new_input_embeds.device),
                }
            unpadded_visual_metadata.append(cur_visual_metadata)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        visual_metadata_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                pad_positions = list(range(max_len - cur_len))
                answer_positions = (torch.where(cur_new_labels != IGNORE_INDEX)[0] + (max_len - cur_len)).tolist()
                visual_base = unpadded_visual_metadata[i]
                text_mask = torch.ones(cur_len, dtype=torch.bool, device=cur_new_embed.device)
                if "visual_token_indices" in visual_base and visual_base["visual_token_indices"].numel() > 0:
                    valid_visual = visual_base["visual_token_indices"].to(cur_new_embed.device)
                    valid_visual = valid_visual[valid_visual < cur_len]
                    text_mask[valid_visual] = False
                text_positions = (torch.where(text_mask)[0] + (max_len - cur_len)).tolist()
                visual_metadata_padded.append(self._shift_metadata(
                    visual_base,
                    offset=max_len - cur_len,
                    max_len=max_len,
                    padding_positions=pad_positions,
                    answer_positions=answer_positions,
                    text_positions=text_positions,
                ))
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                pad_positions = list(range(cur_len, max_len))
                answer_positions = torch.where(cur_new_labels != IGNORE_INDEX)[0].tolist()
                visual_base = unpadded_visual_metadata[i]
                text_mask = torch.ones(cur_len, dtype=torch.bool, device=cur_new_embed.device)
                if "visual_token_indices" in visual_base and visual_base["visual_token_indices"].numel() > 0:
                    valid_visual = visual_base["visual_token_indices"].to(cur_new_embed.device)
                    valid_visual = valid_visual[valid_visual < cur_len]
                    text_mask[valid_visual] = False
                text_positions = torch.where(text_mask)[0].tolist()
                visual_metadata_padded.append(self._shift_metadata(
                    visual_base,
                    offset=0,
                    max_len=max_len,
                    padding_positions=pad_positions,
                    answer_positions=answer_positions,
                    text_positions=text_positions,
                ))

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        result = (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)
        extras = []
        if return_visual_metadata:
            extras.append(visual_metadata_padded)
        if return_llm_geo_metadata:
            llm_geo_pos, llm_geo_mask, llm_geo_debug = self._build_llm_geo_tensors(
                visual_metadata_padded,
                batch_size=batch_size,
                max_len=max_len,
                dtype=new_input_embeds.dtype,
                device=new_input_embeds.device,
            )
            extras.extend([llm_geo_pos, llm_geo_mask, llm_geo_debug])
        if extras:
            return result + tuple(extras)
        return result

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
