import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from datetime import timedelta
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

import sys
_default_repo_root = str(Path(__file__).resolve().parents[3])
_repo_root = os.environ.get("VLM3R_CODE_ROOT", _default_repo_root)
if _repo_root not in sys.path:
    sys.path = [_repo_root] + sys.path
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.model.cut3r_token_only import assert_cut3r_token_projector_checkpoint_values
    from llava.cut3r_token_sidecar_manifest import (
        normalize_cut3r_token_manifest_policy,
        load_cut3r_token_sidecar_manifest,
        validate_cut3r_token_sidecar_manifest_entry,
    )
except ImportError:
    eval_logger.debug("LLaVA-Video is not installed. Please install LLaVA-Video to use this model.")

try:
    from llava.model.language_model.llava_qwen import LlavaQwenConfig

    AutoConfig.register("llava_qwen", LlavaQwenConfig)
except ImportError:
    eval_logger.debug("No Qwen for llava vid")

from llava.model.language_model.llava_llama import LlavaConfig

AutoConfig.register("llava_llama", LlavaConfig)


def _str_to_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _default_attn_implementation():
    version = torch.__version__.split("+", 1)[0]
    try:
        major, minor, *_ = (int(part) for part in version.split("."))
    except ValueError:
        return "sdpa"
    return "sdpa" if (major, minor) >= (2, 1) else "eager"


def _normalize_geo_rope_point_map_key(value):
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
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
            f"got {value!r}"
        )
    return aliases[normalized]


def _infer_legacy_training_point_map_key(config):
    configured = (
        getattr(config, "geo_rope_training_point_map_key", None)
        or getattr(config, "geometry_training_point_map_key", None)
        or getattr(config, "geo_rope_point_map_key", None)
        or getattr(config, "geometry_point_map_key", None)
    )
    normalized = _normalize_geo_rope_point_map_key(configured)
    if normalized is not None:
        return normalized

    geometry_tower_type = str(getattr(config, "geometry_spatial_tower_type", "") or "").lower()
    geometry_subdir = str(getattr(config, "geometry_spatial_features_subdir", "") or "").lower()
    spatial_subdir = str(getattr(config, "spatial_features_subdir", "") or "").lower()
    if "cut3r" in geometry_tower_type or "spatial_features_points" in geometry_subdir or "spatial_features_points" in spatial_subdir:
        # Legacy CUT3R point-map checkpoints did not store this field. The
        # model-side priority selected point_maps_ref before point_maps_cam.
        return "point_maps_ref"
    return None


def _validate_eval_point_map_key(config, requested_eval_key, mutate_config=True):
    eval_key = _normalize_geo_rope_point_map_key(
        requested_eval_key
        or getattr(config, "geo_rope_point_map_key", None)
        or getattr(config, "geometry_point_map_key", None)
    )
    if eval_key is None:
        return None

    train_key = _infer_legacy_training_point_map_key(config)
    if train_key is not None and train_key != eval_key:
        raise RuntimeError(
            "GeoRoPE point-map coordinate mismatch: checkpoint training used "
            f"{train_key}, but this eval requested {eval_key}. Use the same "
            "coordinate source for train and eval, or evaluate a checkpoint "
            "trained with the requested source."
        )
    if mutate_config:
        setattr(config, "geo_rope_point_map_key", eval_key)
        if train_key is not None:
            setattr(config, "geo_rope_training_point_map_key", train_key)
    return eval_key


def _format_gate_value(value: torch.Tensor) -> str:
    flat = value.detach().float().cpu().reshape(-1)
    if flat.numel() == 1:
        return f"{flat.item():.4f}"
    return "[" + ", ".join(f"{x:.4f}" for x in flat.tolist()) + "]"


def _force_geo_rope_gates_zero(model: torch.nn.Module, checkpoint_path: str) -> None:
    found = []
    print("[Gate0 Ablation] FORCE_GEO_ROPE_GATE_ZERO=True", flush=True)
    print(f"[Gate0 Ablation] checkpoint path: {checkpoint_path}", flush=True)

    config = getattr(model, "config", None)
    geometry_rope_mode = (
        getattr(config, "geo_rope_fusion_mode", None)
        or getattr(config, "geometry_rope_mode", None)
    )
    geometry_rope_max_depth = (
        getattr(config, "geo_rope_fusion_max_depth", None)
        or getattr(config, "geometry_rope_max_depth", None)
    )
    print(f"[Gate0 Ablation] geometry_rope_mode: {geometry_rope_mode}", flush=True)
    print(f"[Gate0 Ablation] geometry_rope_max_depth: {geometry_rope_max_depth}", flush=True)

    for module_name, module in model.named_modules():
        gate_q = getattr(module, "geo_rope_fusion_gate_q", None)
        gate_k = getattr(module, "geo_rope_fusion_gate_k", None)
        if gate_q is None and gate_k is None:
            continue
        if not isinstance(gate_q, torch.Tensor) or not isinstance(gate_k, torch.Tensor):
            raise RuntimeError(
                f"[Gate0 Ablation] Module {module_name} has incomplete/non-tensor GeoRoPE gates: "
                f"gate_q={type(gate_q)}, gate_k={type(gate_k)}"
            )

        found.append(module_name)
        print(f"[Gate0 Ablation] Found module: {module_name}", flush=True)
        print(f"gate_q before: {_format_gate_value(gate_q)}", flush=True)
        print(f"gate_k before: {_format_gate_value(gate_k)}", flush=True)
        with torch.no_grad():
            gate_q.zero_()
            gate_k.zero_()
        print(f"gate_q after: {_format_gate_value(gate_q)}", flush=True)
        print(f"gate_k after: {_format_gate_value(gate_k)}", flush=True)

    if not found:
        raise RuntimeError(
            "[Gate0 Ablation] FORCE_GEO_ROPE_GATE_ZERO=True but no modules with "
            "geo_rope_fusion_gate_q and geo_rope_fusion_gate_k were found."
        )

    print(f"[Gate0 Ablation] Zeroed GeoRoPE Q/K gates in {len(found)} module(s).", flush=True)


@register_model("vlm_3r")
class Vlm3r(lmms):
    """
    Vlm3r Model
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/VLM-3R-7B-Qwen2",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=_default_attn_implementation(),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        max_frames_num: int = 3,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "bilinear",
        mm_newline_position: str = "grid",
        mm_pooling_position: str = "after",
        overwrite: bool = True,
        video_decode_backend: str = "pyav",
        delay_load: bool = False,
        tie_weights: bool = True,
        model_name: str = None,
        model_base: str = None,
        zero_spatial_features: Union[bool, str] = False,
        zero_visual_patch_embeddings: Union[bool, str] = False,
        visual_token_source: str = None,
        cut3r_token_sidecar_key: str = None,
        cut3r_token_feature_dim: Optional[Union[int, str]] = None,
        cut3r_token_projector_layernorm: Optional[Union[bool, str]] = None,
        spatial_tower: str = None,
        spatial_feature_dim: Optional[Union[int, str]] = None,
        spatial_tower_select_feature: str = None,
        fusion_block: str = None,
        geometry_rope_mode: str = None,
        geometry_rope_max_depth: Optional[Union[float, str]] = None,
        geometry_rope_group_split: str = None,
        geometry_rope_log_stats: Union[bool, str] = False,
        geo_rope_point_map_key: str = None,
        force_geo_rope_gate_zero: Union[bool, str] = False,
        probe_geometry_shuffle: Union[bool, str] = False,
        probe_geometry_shuffle_mode: str = "cyclic_shift",
        probe_geometry_shuffle_shift: Optional[Union[int, str]] = 1,
        probe_geometry_shuffle_seed: Optional[Union[int, str]] = 0,
        probe_spatial_feature_frame_swap: Union[bool, str] = False,
        probe_spatial_feature_frame_swap_mode: str = "random_derange",
        probe_spatial_feature_frame_swap_seed: Optional[Union[int, str]] = 0,
        probe_cross_frame_window: Optional[Union[int, str]] = 0,
        probe_cross_frame_include_self: Union[bool, str] = True,
        probe_cross_frame_mode: str = "sliding_window",
        probe_intra_frame_pos_shuffle: Union[bool, str] = False,
        llm_visual_3d_rope_enable: Union[bool, str] = False,
        llm_visual_3d_rope_alpha: Optional[Union[float, str]] = 1.0,
        llm_visual_3d_rope_mode: str = "spherical",
        llm_visual_3d_rope_group_split: str = "2,1,2",
        llm_visual_3d_rope_max_depth: Optional[Union[float, str]] = 10.0,
        llm_visual_3d_rope_layers: str = "all",
        llm_visual_3d_rope_geometry_source: str = "point_maps_ref",
        llm_visual_3d_rope_shuffle: Union[bool, str] = False,
        llm_visual_3d_rope_shuffle_mode: str = "intra_sample_token_shuffle",
        llm_visual_3d_rope_shuffle_seed: Optional[Union[int, str]] = 0,
        llm_visual_3d_rope_log_stats: Union[bool, str] = True,
        llm_visual_3d_rope_log_layers: str = "first_middle_last",
        llm_visual_3d_rope_force_eager_attention: Union[bool, str] = True,
        llm_visual_3d_rope_stats_path: Optional[str] = None,
        cut3r_spatialstack_residual_scale: Optional[Union[float, str]] = None,
        cut3r_spatialstack_projector_type: Optional[str] = None,
        cut3r_spatialstack_merge_size: Optional[Union[int, str]] = None,
        cut3r_spatialstack_projector_hidden_dim: Optional[Union[int, str]] = None,
        cut3r_spatialstack_llm_layers: Optional[str] = None,
        cut3r_spatialstack_preagg_enable: Optional[Union[bool, str]] = None,
        cut3r_spatialstack_preagg_layers: Optional[str] = None,
        cut3r_spatialstack_preagg_type: Optional[str] = None,
        cut3r_spatialstack_preagg_projector_sharing: Optional[str] = None,
        cut3r_spatialstack_preagg_use_layer_gamma: Optional[Union[bool, str]] = None,
        cut3r_spatialstack_preagg_layer_gamma_init: Optional[Union[float, str]] = None,
        cut3r_spatialstack_frame_shuffle: Union[bool, str] = False,
        cut3r_spatialstack_frame_shuffle_mode: str = "random_derange",
        cut3r_spatialstack_frame_shuffle_seed: Optional[Union[int, str]] = 0,
        cut3r_spatialstack_token_shuffle: Union[bool, str] = False,
        cut3r_token_manifest_policy: str = "warn",
        cut3r_spatialstack_token_shuffle_mode: str = "random_derange",
        cut3r_token_sidecar_manifest: str = None,
        cut3r_spatialstack_token_shuffle_seed: Optional[Union[int, str]] = 0,
        cut3r_spatialstack_per_frame_token_mean: Union[bool, str] = False,
        residual_predictor_type: str = None,
        residual_predictor_checkpoint: str = None,
        use_predicted_spatialstack_residuals: Union[bool, str] = False,
        predicted_residual_gamma_layer0: Optional[Union[float, str]] = 1.0,
        predicted_residual_gamma_layer1: Optional[Union[float, str]] = 1.0,
        predicted_residual_gamma_layer2: Optional[Union[float, str]] = 1.0,
        predicted_residual_control: str = "none",
        mean_residual_artifact: str = None,
        spatialstack_residual_mode: str = None,
        spatialstack_residual_beta: Optional[Union[float, str]] = None,
        expected_key_manifest: str = None,
        expected_key_manifest_sha256: str = None,
        evaluation_telemetry_dir: str = None,
        disable_cut3r_spatialstack: Union[bool, str] = False,
        spatial_features_root: str = None,
        spatial_features_subdir: str = "spatial_features_points",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = get_model_name_from_path(pretrained)
        # Run directories such as checkpoint-2 do not encode the architecture.
        # Recover the repository Qwen/LLaVA LoRA loader from checkpoint config.
        try:
            checkpoint_architectures = AutoConfig.from_pretrained(pretrained).architectures or []
        except Exception as err:
            eval_logger.warning("[CFG] Cannot determine checkpoint architecture for loader selection: {}", err)
            checkpoint_architectures = []
        if "LlavaQwenForCausalLM" in checkpoint_architectures:
            loader_hints = self.model_name.lower()
            missing_hints = [hint for hint in ("llava", "qwen", "lora") if hint not in loader_hints]
            if missing_hints:
                self.model_name = f"{self.model_name}-{'-'.join(missing_hints)}"
                eval_logger.info("[CFG] Resolved Qwen/LLaVA adapter loader name: {}", self.model_name)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = _str_to_bool(overwrite)
        self.mm_resampler_type = mm_resampler_type
        self.mm_spatial_pool_stride = int(mm_spatial_pool_stride)
        self.mm_spatial_pool_out_channels = int(mm_spatial_pool_out_channels)
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.max_frames_num = int(max_frames_num)
        self.mm_resampler_location = mm_pooling_position
        self.mm_newline_position = mm_newline_position
        self.delay_load = delay_load
        self.attn_implementation = attn_implementation
        self.zero_spatial_features = _str_to_bool(zero_spatial_features)
        self.zero_visual_patch_embeddings = _str_to_bool(zero_visual_patch_embeddings)
        self.visual_token_source = visual_token_source or None
        self.cut3r_token_sidecar_key = cut3r_token_sidecar_key or None
        self.cut3r_token_feature_dim = int(cut3r_token_feature_dim) if cut3r_token_feature_dim not in (None, "") else None
        self.cut3r_token_projector_layernorm = (
            _str_to_bool(cut3r_token_projector_layernorm)
            if cut3r_token_projector_layernorm not in (None, "")
            else None
        )
        self.spatial_tower = spatial_tower or None
        self.spatial_feature_dim = int(spatial_feature_dim) if spatial_feature_dim not in (None, "") else None
        self.spatial_tower_select_feature = spatial_tower_select_feature or None
        self.fusion_block = fusion_block or None
        self.geometry_rope_mode = geometry_rope_mode or None
        self.geometry_rope_max_depth = float(geometry_rope_max_depth) if geometry_rope_max_depth not in (None, "") else None
        self.geometry_rope_group_split = geometry_rope_group_split or None
        self.geometry_rope_log_stats = _str_to_bool(geometry_rope_log_stats)
        self.geo_rope_point_map_key = _normalize_geo_rope_point_map_key(geo_rope_point_map_key)
        self.force_geo_rope_gate_zero = _str_to_bool(force_geo_rope_gate_zero)
        self.probe_geometry_shuffle = _str_to_bool(probe_geometry_shuffle)
        self.probe_geometry_shuffle_mode = probe_geometry_shuffle_mode or "cyclic_shift"
        self.probe_geometry_shuffle_shift = int(probe_geometry_shuffle_shift or 0)
        self.probe_geometry_shuffle_seed = int(probe_geometry_shuffle_seed or 0)
        self.probe_spatial_feature_frame_swap = _str_to_bool(probe_spatial_feature_frame_swap)
        self.probe_spatial_feature_frame_swap_mode = probe_spatial_feature_frame_swap_mode or "random_derange"
        self.probe_spatial_feature_frame_swap_seed = int(probe_spatial_feature_frame_swap_seed or 0)
        self.probe_cross_frame_window = int(probe_cross_frame_window or 0)
        self.probe_cross_frame_include_self = _str_to_bool(probe_cross_frame_include_self)
        self.probe_cross_frame_mode = probe_cross_frame_mode or "sliding_window"
        self.probe_intra_frame_pos_shuffle = _str_to_bool(probe_intra_frame_pos_shuffle)
        self.llm_visual_3d_rope_enable = _str_to_bool(llm_visual_3d_rope_enable)
        self.llm_visual_3d_rope_alpha = float(llm_visual_3d_rope_alpha)
        self.llm_visual_3d_rope_mode = llm_visual_3d_rope_mode or "spherical"
        self.llm_visual_3d_rope_group_split = (llm_visual_3d_rope_group_split or "2,1,2").replace("|", ",").replace(";", ",")
        self.llm_visual_3d_rope_max_depth = float(llm_visual_3d_rope_max_depth)
        self.llm_visual_3d_rope_layers = llm_visual_3d_rope_layers or "all"
        self.llm_visual_3d_rope_geometry_source = _normalize_geo_rope_point_map_key(llm_visual_3d_rope_geometry_source)
        self.llm_visual_3d_rope_shuffle = _str_to_bool(llm_visual_3d_rope_shuffle)
        self.llm_visual_3d_rope_shuffle_mode = llm_visual_3d_rope_shuffle_mode or "intra_sample_token_shuffle"
        self.llm_visual_3d_rope_shuffle_seed = int(llm_visual_3d_rope_shuffle_seed or 0)
        self.llm_visual_3d_rope_log_stats = _str_to_bool(llm_visual_3d_rope_log_stats)
        self.llm_visual_3d_rope_log_layers = llm_visual_3d_rope_log_layers or "first_middle_last"
        self.llm_visual_3d_rope_force_eager_attention = _str_to_bool(llm_visual_3d_rope_force_eager_attention)
        self.cut3r_spatialstack_residual_scale = (
            None if cut3r_spatialstack_residual_scale in (None, "")
            else float(cut3r_spatialstack_residual_scale)
        )
        self.cut3r_spatialstack_projector_type = cut3r_spatialstack_projector_type or None
        self.cut3r_spatialstack_merge_size = (
            int(cut3r_spatialstack_merge_size) if cut3r_spatialstack_merge_size not in (None, "") else None
        )
        self.cut3r_spatialstack_projector_hidden_dim = (
            int(cut3r_spatialstack_projector_hidden_dim)
            if cut3r_spatialstack_projector_hidden_dim not in (None, "")
            else None
        )
        self.cut3r_spatialstack_llm_layers = cut3r_spatialstack_llm_layers or None
        self.cut3r_spatialstack_preagg_enable = (
            _str_to_bool(cut3r_spatialstack_preagg_enable)
            if cut3r_spatialstack_preagg_enable not in (None, "")
            else None
        )
        self.cut3r_spatialstack_preagg_layers = cut3r_spatialstack_preagg_layers or None
        self.cut3r_spatialstack_preagg_type = cut3r_spatialstack_preagg_type or None
        self.cut3r_spatialstack_preagg_projector_sharing = cut3r_spatialstack_preagg_projector_sharing or None
        self.cut3r_spatialstack_preagg_use_layer_gamma = (
            _str_to_bool(cut3r_spatialstack_preagg_use_layer_gamma)
            if cut3r_spatialstack_preagg_use_layer_gamma not in (None, "")
            else None
        )
        self.cut3r_spatialstack_preagg_layer_gamma_init = (
            float(cut3r_spatialstack_preagg_layer_gamma_init)
            if cut3r_spatialstack_preagg_layer_gamma_init not in (None, "")
            else None
        )
        self.cut3r_spatialstack_frame_shuffle = _str_to_bool(cut3r_spatialstack_frame_shuffle)
        self.cut3r_spatialstack_frame_shuffle_mode = cut3r_spatialstack_frame_shuffle_mode or "random_derange"
        self.cut3r_spatialstack_frame_shuffle_seed = int(cut3r_spatialstack_frame_shuffle_seed or 0)
        self.cut3r_spatialstack_token_shuffle = _str_to_bool(cut3r_spatialstack_token_shuffle)
        self.cut3r_spatialstack_token_shuffle_mode = cut3r_spatialstack_token_shuffle_mode or "random_derange"
        self.cut3r_spatialstack_token_shuffle_seed = int(cut3r_spatialstack_token_shuffle_seed or 0)
        self.cut3r_spatialstack_per_frame_token_mean = _str_to_bool(cut3r_spatialstack_per_frame_token_mean)
        self.residual_predictor_type = residual_predictor_type or None
        self.residual_predictor_checkpoint = residual_predictor_checkpoint or None
        self.use_predicted_spatialstack_residuals = _str_to_bool(use_predicted_spatialstack_residuals)
        self.predicted_residual_gamma_layers = (
            float(predicted_residual_gamma_layer0),
            float(predicted_residual_gamma_layer1),
            float(predicted_residual_gamma_layer2),
        )
        self.mean_residual_artifact = mean_residual_artifact or None
        self.spatialstack_residual_mode = str(spatialstack_residual_mode or "").strip().lower()
        if self.spatialstack_residual_mode not in {"", "oracle_replay", "oracle_replay_parity", "interpolate"}:
            raise ValueError(f"Unsupported spatialstack_residual_mode={self.spatialstack_residual_mode!r}.")
        self.spatialstack_residual_beta = None if spatialstack_residual_beta in (None, "") else float(spatialstack_residual_beta)
        if self.spatialstack_residual_mode == "interpolate" and self.spatialstack_residual_beta is None:
            raise ValueError("spatialstack_residual_mode=interpolate requires spatialstack_residual_beta.")
        self._teacher_residual_scale_provenance = {}
        if self.spatialstack_residual_mode:
            predictor_path = Path(self.residual_predictor_checkpoint or "")
            if not predictor_path.is_file():
                raise RuntimeError(
                    "Experiment residual modes require residual_predictor_checkpoint so the "
                    "teacher residual scale can be loaded from recorded predictor provenance."
                )
            predictor_metadata = torch.load(str(predictor_path), map_location="cpu", weights_only=False)
            if not isinstance(predictor_metadata, Mapping):
                raise RuntimeError(f"Invalid residual predictor provenance: {predictor_path}")
            recorded_scale = predictor_metadata.get("teacher_residual_scale")
            recorded_config_hash = str(predictor_metadata.get("teacher_config_hash") or "")
            recorded_teacher_checkpoint = str(predictor_metadata.get("teacher_checkpoint") or "").strip()
            config_path = Path(recorded_teacher_checkpoint) / "config.json"
            if recorded_scale is None or not recorded_teacher_checkpoint or not config_path.is_file():
                raise RuntimeError(
                    "Experiment residual mode requires teacher_residual_scale and a teacher config "
                    f"for provenance validation: predictor={predictor_path}, config={config_path}."
                )
            actual_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if not recorded_config_hash or actual_config_hash != recorded_config_hash:
                raise RuntimeError(
                    "Teacher config hash does not match the predictor's recorded provenance: "
                    f"actual={actual_config_hash}, recorded={recorded_config_hash}."
                )
            recorded_scale = float(recorded_scale)
            if not math.isfinite(recorded_scale):
                raise RuntimeError(f"Recorded teacher residual scale is nonfinite: {recorded_scale!r}.")
            if (
                self.cut3r_spatialstack_residual_scale is not None
                and not math.isclose(self.cut3r_spatialstack_residual_scale, recorded_scale, rel_tol=0.0, abs_tol=0.0)
            ):
                raise RuntimeError(
                    "Explicit cut3r_spatialstack_residual_scale conflicts with predictor-recorded "
                    f"teacher value: explicit={self.cut3r_spatialstack_residual_scale}, recorded={recorded_scale}."
                )
            self.cut3r_spatialstack_residual_scale = recorded_scale
            self._teacher_residual_scale_provenance = {
                "predictor_checkpoint": str(predictor_path.resolve()),
                "teacher_checkpoint": recorded_teacher_checkpoint,
                "teacher_config_sha256": actual_config_hash,
                "recorded_teacher_residual_scale": recorded_scale,
                "source": "residual_predictor_checkpoint_metadata",
            }
        elif self.cut3r_spatialstack_residual_scale is None:
            self.cut3r_spatialstack_residual_scale = 1.0
        self.expected_key_manifest = Path(expected_key_manifest).resolve() if expected_key_manifest else None
        self.expected_key_manifest_sha256 = str(expected_key_manifest_sha256 or "").strip() or None
        self.evaluation_telemetry_dir = Path(evaluation_telemetry_dir).resolve() if evaluation_telemetry_dir else None
        self.expected_key_manifest_sha256 = str(expected_key_manifest_sha256 or "").strip() or None
        self.evaluation_telemetry_dir = Path(evaluation_telemetry_dir).resolve() if evaluation_telemetry_dir else None
        self._expected_key_by_doc_id = {}
        if self.expected_key_manifest is not None:
            if not self.expected_key_manifest.is_file():
                raise FileNotFoundError(f"Expected-key manifest is missing: {self.expected_key_manifest}")
            manifest_hash = hashlib.sha256(self.expected_key_manifest.read_bytes()).hexdigest()
            if not self.expected_key_manifest_sha256:
                raise RuntimeError("expected_key_manifest_sha256 is required whenever expected_key_manifest is set.")
            if manifest_hash != self.expected_key_manifest_sha256:
                raise RuntimeError(f"Expected-key manifest SHA256 mismatch: expected={self.expected_key_manifest_sha256}, actual={manifest_hash}.")
            for line in self.expected_key_manifest.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                doc_id = int(item["doc_id"])
                if doc_id in self._expected_key_by_doc_id:
                    raise RuntimeError(f"Expected-key manifest has duplicate doc_id={doc_id}.")
                self._expected_key_by_doc_id[doc_id] = item
            if len(self._expected_key_by_doc_id) != 5130:
                raise RuntimeError(f"Expected-key manifest must contain 5130 keys, got {len(self._expected_key_by_doc_id)}.")
            if len({item["canonical_key"] for item in self._expected_key_by_doc_id.values()}) != 5130:
                raise RuntimeError("Expected-key manifest has canonical-key collisions.")
        self.predicted_residual_control = predicted_residual_control or "none"
        self.disable_cut3r_spatialstack = _str_to_bool(disable_cut3r_spatialstack)
        self._predicted_spatialstack_sidecar_load_attempts = 0
        stats_path = llm_visual_3d_rope_stats_path or os.environ.get("LLM_VISUAL_3D_ROPE_STATS_PATH", "")
        self.llm_visual_3d_rope_stats_path = Path(stats_path) if stats_path else None
        self._llm_visual_3d_rope_eval_counter = 0
        if self.llm_visual_3d_rope_enable and self.llm_visual_3d_rope_force_eager_attention:
            self.attn_implementation = "eager"
        elif self.llm_visual_3d_rope_enable and self.attn_implementation != "eager":
            raise RuntimeError("LLM visual-token 3D RoPE eval requires attn_implementation=eager.")
        self.spatial_features_root = Path(spatial_features_root) if spatial_features_root not in (None, "") else None
        self.spatial_features_subdir = spatial_features_subdir or "spatial_features_points"
        self.cut3r_token_manifest_policy = normalize_cut3r_token_manifest_policy(cut3r_token_manifest_policy)
        self.cut3r_token_sidecar_manifest = load_cut3r_token_sidecar_manifest(cut3r_token_sidecar_manifest, policy=self.cut3r_token_manifest_policy, warning_callback=eval_logger.warning)
        self._spatial_layer_specs = self._split_spatial_layer_specs(self.spatial_features_subdir)
        preserved_config = {}
        if not self.overwrite:
            checkpoint_config = AutoConfig.from_pretrained(self.pretrained).to_dict()
            preserved_prefixes = (
                "cut3r_",
                "visual_token_",
                "use_cut3r_",
                "llm_visual_3d_rope_",
                "geometry_",
                "geo_rope_",
                "spatial_",
                "use_geometry_",
                "use_auxiliary_",
                "use_bev_",
                "use_depth_",
                "use_pointmap_",
                "pointmap_",
                "use_spatial_bridge_",
                "num_spatial_bridge_",
                "lambda_",
            )
            preserved_config = {
                key: value
                for key, value in checkpoint_config.items()
                if key == "fusion_block"
                or key.startswith(preserved_prefixes)
            }

        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = self.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
            overwrite_config["mm_pooling_position"] = self.mm_resampler_location
            overwrite_config["mm_newline_position"] = self.mm_newline_position
            overwrite_config["add_faster_video"] = False
            overwrite_config["delay_load"] = self.delay_load
            overwrite_config["zero_spatial_features"] = self.zero_spatial_features
            overwrite_config["zero_visual_patch_embeddings"] = self.zero_visual_patch_embeddings
            if self.visual_token_source is not None:
                overwrite_config["visual_token_source"] = self.visual_token_source
            if self.cut3r_token_sidecar_key is not None:
                overwrite_config["cut3r_token_sidecar_key"] = self.cut3r_token_sidecar_key
            if self.cut3r_token_feature_dim is not None:
                overwrite_config["cut3r_token_feature_dim"] = self.cut3r_token_feature_dim
            if self.cut3r_token_projector_layernorm is not None:
                overwrite_config["cut3r_token_projector_layernorm"] = self.cut3r_token_projector_layernorm
            if self.spatial_tower is not None:
                overwrite_config["spatial_tower"] = self.spatial_tower
            if self.spatial_feature_dim is not None:
                overwrite_config["spatial_feature_dim"] = self.spatial_feature_dim
            if self.spatial_tower_select_feature is not None:
                overwrite_config["spatial_tower_select_feature"] = self.spatial_tower_select_feature
            if self.fusion_block is not None:
                overwrite_config["fusion_block"] = self.fusion_block
            if self.geometry_rope_mode is not None:
                overwrite_config["geometry_rope_mode"] = self.geometry_rope_mode
            if self.geometry_rope_max_depth is not None:
                overwrite_config["geometry_rope_max_depth"] = self.geometry_rope_max_depth
            if self.geometry_rope_group_split is not None:
                overwrite_config["geometry_rope_group_split"] = self.geometry_rope_group_split
            overwrite_config["geometry_rope_log_stats"] = self.geometry_rope_log_stats
            if self.geo_rope_point_map_key is not None:
                overwrite_config["geo_rope_point_map_key"] = self.geo_rope_point_map_key
                overwrite_config["geometry_point_map_key"] = self.geo_rope_point_map_key
            overwrite_config["probe_geometry_shuffle"] = self.probe_geometry_shuffle
            overwrite_config["probe_geometry_shuffle_mode"] = self.probe_geometry_shuffle_mode
            overwrite_config["probe_geometry_shuffle_shift"] = self.probe_geometry_shuffle_shift
            overwrite_config["probe_geometry_shuffle_seed"] = self.probe_geometry_shuffle_seed
            overwrite_config["probe_spatial_feature_frame_swap"] = self.probe_spatial_feature_frame_swap
            overwrite_config["probe_spatial_feature_frame_swap_mode"] = self.probe_spatial_feature_frame_swap_mode
            overwrite_config["probe_spatial_feature_frame_swap_seed"] = self.probe_spatial_feature_frame_swap_seed
            overwrite_config["probe_cross_frame_window"] = self.probe_cross_frame_window
            overwrite_config["probe_cross_frame_include_self"] = self.probe_cross_frame_include_self
            overwrite_config["probe_cross_frame_mode"] = self.probe_cross_frame_mode
            overwrite_config["probe_intra_frame_pos_shuffle"] = self.probe_intra_frame_pos_shuffle
            overwrite_config["llm_visual_3d_rope_enable"] = self.llm_visual_3d_rope_enable
            overwrite_config["llm_visual_3d_rope_alpha"] = self.llm_visual_3d_rope_alpha
            overwrite_config["llm_visual_3d_rope_mode"] = self.llm_visual_3d_rope_mode
            overwrite_config["llm_visual_3d_rope_group_split"] = self.llm_visual_3d_rope_group_split
            overwrite_config["llm_visual_3d_rope_max_depth"] = self.llm_visual_3d_rope_max_depth
            overwrite_config["llm_visual_3d_rope_layers"] = self.llm_visual_3d_rope_layers
            overwrite_config["llm_visual_3d_rope_geometry_source"] = self.llm_visual_3d_rope_geometry_source
            overwrite_config["llm_visual_3d_rope_shuffle"] = self.llm_visual_3d_rope_shuffle
            overwrite_config["llm_visual_3d_rope_shuffle_mode"] = self.llm_visual_3d_rope_shuffle_mode
            overwrite_config["llm_visual_3d_rope_shuffle_seed"] = self.llm_visual_3d_rope_shuffle_seed
            overwrite_config["llm_visual_3d_rope_log_stats"] = self.llm_visual_3d_rope_log_stats
            overwrite_config["llm_visual_3d_rope_log_layers"] = self.llm_visual_3d_rope_log_layers
            overwrite_config["llm_visual_3d_rope_force_eager_attention"] = self.llm_visual_3d_rope_force_eager_attention
            overwrite_config["cut3r_spatialstack_residual_scale"] = self.cut3r_spatialstack_residual_scale
            if self.cut3r_spatialstack_projector_type is not None:
                overwrite_config["cut3r_spatialstack_projector_type"] = self.cut3r_spatialstack_projector_type
            if self.cut3r_spatialstack_merge_size is not None:
                overwrite_config["cut3r_spatialstack_merge_size"] = self.cut3r_spatialstack_merge_size
            if self.cut3r_spatialstack_projector_hidden_dim is not None:
                overwrite_config["cut3r_spatialstack_projector_hidden_dim"] = self.cut3r_spatialstack_projector_hidden_dim
            if self.cut3r_spatialstack_llm_layers is not None:
                overwrite_config["cut3r_spatialstack_llm_layers"] = self.cut3r_spatialstack_llm_layers
            if self.cut3r_spatialstack_preagg_enable is not None:
                overwrite_config["cut3r_spatialstack_preagg_enable"] = self.cut3r_spatialstack_preagg_enable
            if self.cut3r_spatialstack_preagg_layers is not None:
                overwrite_config["cut3r_spatialstack_preagg_layers"] = self.cut3r_spatialstack_preagg_layers
            if self.cut3r_spatialstack_preagg_type is not None:
                overwrite_config["cut3r_spatialstack_preagg_type"] = self.cut3r_spatialstack_preagg_type
            if self.cut3r_spatialstack_preagg_projector_sharing is not None:
                overwrite_config["cut3r_spatialstack_preagg_projector_sharing"] = self.cut3r_spatialstack_preagg_projector_sharing
            if self.cut3r_spatialstack_preagg_use_layer_gamma is not None:
                overwrite_config["cut3r_spatialstack_preagg_use_layer_gamma"] = self.cut3r_spatialstack_preagg_use_layer_gamma
            if self.cut3r_spatialstack_preagg_layer_gamma_init is not None:
                overwrite_config["cut3r_spatialstack_preagg_layer_gamma_init"] = self.cut3r_spatialstack_preagg_layer_gamma_init
            overwrite_config["cut3r_spatialstack_frame_shuffle"] = self.cut3r_spatialstack_frame_shuffle
            overwrite_config["cut3r_spatialstack_frame_shuffle_mode"] = self.cut3r_spatialstack_frame_shuffle_mode
            overwrite_config["cut3r_spatialstack_frame_shuffle_seed"] = self.cut3r_spatialstack_frame_shuffle_seed
            overwrite_config["cut3r_spatialstack_token_shuffle"] = self.cut3r_spatialstack_token_shuffle
            overwrite_config["cut3r_spatialstack_token_shuffle_mode"] = self.cut3r_spatialstack_token_shuffle_mode
            overwrite_config["cut3r_spatialstack_token_shuffle_seed"] = self.cut3r_spatialstack_token_shuffle_seed
            overwrite_config["cut3r_spatialstack_per_frame_token_mean"] = self.cut3r_spatialstack_per_frame_token_mean
            if self.spatialstack_residual_mode:
                overwrite_config["spatialstack_residual_mode"] = self.spatialstack_residual_mode
                overwrite_config["spatialstack_residual_beta"] = self.spatialstack_residual_beta
            if self.use_predicted_spatialstack_residuals or self.spatialstack_residual_mode == "interpolate":
                # Keep use_cut3r_spatialstack intact while loading the checkpoint's non-LoRA
                # branch weights; disable it immediately after model construction below.
                overwrite_config["use_predicted_spatialstack_residuals"] = True
                overwrite_config["residual_predictor_type"] = self.residual_predictor_type
                overwrite_config["residual_predictor_checkpoint"] = self.residual_predictor_checkpoint
                overwrite_config["predicted_residual_gamma_layer0"] = self.predicted_residual_gamma_layers[0]
                overwrite_config["predicted_residual_gamma_layer1"] = self.predicted_residual_gamma_layers[1]
                overwrite_config["mean_residual_artifact"] = self.mean_residual_artifact
                overwrite_config["predicted_residual_gamma_layer2"] = self.predicted_residual_gamma_layers[2]
                overwrite_config["predicted_residual_control"] = self.predicted_residual_control
            if self.disable_cut3r_spatialstack:
                # Keep the oracle branch enabled through checkpoint loading, then
                # turn it off after construction so its saved non-LoRA state stays compatible.
                overwrite_config["disable_cut3r_spatialstack"] = True
            if self.llm_visual_3d_rope_enable:
                overwrite_config["geo_rope_point_map_key"] = self.llm_visual_3d_rope_geometry_source
                overwrite_config["geometry_point_map_key"] = self.llm_visual_3d_rope_geometry_source
                overwrite_config["_attn_implementation"] = "eager"
                overwrite_config["_attn_implementation_internal"] = "eager"
                overwrite_config["attn_implementation"] = "eager"
            # overwrite_config["attn_implementation"] = attn_implementation

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)
            architectures = getattr(cfg_pretrained, "architectures", None) or []
            model_architecture = architectures[0] if len(architectures) > 0 else None

            # Some LoRA/PEFT checkpoints do not persist `architectures` in config.json.
            # Fall back to model_base for architecture-specific branching when needed.
            if model_architecture is None and model_base is not None:
                try:
                    cfg_base = AutoConfig.from_pretrained(model_base)
                    base_architectures = getattr(cfg_base, "architectures", None) or []
                    if len(base_architectures) > 0:
                        model_architecture = base_architectures[0]
                        eval_logger.info(
                            "[CFG] Missing architectures in pretrained config; fallback to model_base architecture={}.",
                            model_architecture,
                        )
                except Exception as err:
                    eval_logger.warning("[CFG] Failed to load model_base config for architecture fallback: {}", err)

            if model_architecture == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
                if "224" in cfg_pretrained.mm_vision_tower:
                    least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            if "v1.5" in pretrained:  # A hardcode solution here to load v1.5 model, otherwise it will use LlavaConfig from hf transformers
                from llavavid.model.language_model.llava_llama import (
                    LlavaConfig,
                    LlavaLlamaForCausalLM,
                )
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
                cfg_pretrained = LlavaConfig.from_pretrained(pretrained)
                if overwrite_config is not None:
                    eval_logger.log(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                kwargs["torch_dtype"] = torch.float16
                self._model = LlavaLlamaForCausalLM.from_pretrained(pretrained, low_cpu_mem_usage=True, config=cfg_pretrained, device_map=self.device_map, **kwargs)
                vision_tower = self._model.get_vision_tower()
                if not vision_tower.is_loaded:
                    vision_tower.load_model(device_map=self.device_map)
                if self.device_map != "auto":
                    vision_tower.to(device="cuda", dtype=torch.float16)
                self._image_processor = vision_tower.image_processor

                if hasattr(self._model.config, "max_sequence_length"):
                    self._max_length = self._model.config.max_sequence_length
                else:
                    self._max_length = 2048
            else:
                self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                    pretrained,
                    model_base,
                    self.model_name,
                    device_map=self.device_map,
                    attn_implementation=self.attn_implementation,
                    overwrite_config=overwrite_config,
                )
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                model_base,
                self.model_name,
                device_map=self.device_map,
                attn_implementation=self.attn_implementation,
            )

        # HF `device_map="cuda:<rank>"` does not reliably place custom
        # modules with newly initialized or non-LoRA-loaded weights. Keep the
        # small VLM-3R heads aligned with the rank-local model before
        # Accelerate wraps it; otherwise their LayerNorm weights can remain on
        # CPU while visual/spatial features are already on CUDA.
        def _move_custom_module(module, name):
            if isinstance(module, list):
                module = module[0] if module else None
            if module is None:
                return
            param = next(module.parameters(), None)
            if param is None:
                return
            if param.device != self._device:
                eval_logger.info(
                    "[DEVICE][EVAL] moving {} from {} to {}",
                    name,
                    param.device,
                    self._device,
                )
                module.to(device=self._device)

        base_model = self._model.get_model() if hasattr(self._model, "get_model") else getattr(self._model, "model", None)
        if base_model is not None:
            get_fusion_block = getattr(base_model, "get_fusion_block", None)
            _move_custom_module(get_fusion_block() if callable(get_fusion_block) else getattr(base_model, "fusion_block", None), "model.fusion_block")
            _move_custom_module(getattr(base_model, "mm_projector", None), "model.mm_projector")
            _move_custom_module(getattr(base_model, "vision_resampler", None), "model.vision_resampler")
            _move_custom_module(getattr(base_model, "geometry_aware_projection", None), "model.geometry_aware_projection")
            get_spatialstack = getattr(base_model, "get_cut3r_spatialstack_merger", None)
            _move_custom_module(
                get_spatialstack() if callable(get_spatialstack) else getattr(base_model, "cut3r_spatialstack", None),
                "model.cut3r_spatialstack",
            )
            get_dual_path = getattr(base_model, "get_cut3r_dual_path", None)
            _move_custom_module(
                get_dual_path() if callable(get_dual_path) else getattr(base_model, "cut3r_dual_path", None),
                "model.cut3r_dual_path",
            )
            get_cut3r_projector = getattr(base_model, "get_cut3r_token_projector", None)
            _move_custom_module(
                get_cut3r_projector() if callable(get_cut3r_projector) else getattr(base_model, "cut3r_token_projector", None),
                "model.cut3r_token_projector",
            )
        _move_custom_module(getattr(self._model, "bev_head", None), "bev_head")

        if self.force_geo_rope_gate_zero:
            _force_geo_rope_gates_zero(self._model, self.pretrained)

        self._config = self._model.config
        self.disable_cut3r_spatialstack = bool(
            self.disable_cut3r_spatialstack
            or _str_to_bool(getattr(self._config, "disable_cut3r_spatialstack", False))
        )
        self.use_predicted_spatialstack_residuals = bool(
            self.use_predicted_spatialstack_residuals
            or self.spatialstack_residual_mode == "interpolate"
            or _str_to_bool(getattr(self._config, "use_predicted_spatialstack_residuals", False))
        )
        if self.use_predicted_spatialstack_residuals:
            if base_model is None or not hasattr(base_model, "initialize_predicted_spatialstack_residual_predictor"):
                raise RuntimeError("Loaded VLM does not support predicted SpatialStack residuals.")
            checkpoint_path = self.residual_predictor_checkpoint or getattr(
                self._config, "residual_predictor_checkpoint", None
            )
            mean_control = str(self.predicted_residual_control).strip().lower() == "mean"
            mean_artifact = self.mean_residual_artifact or getattr(
                self._config, "mean_residual_artifact", None
            )
            if not checkpoint_path and not mean_control:
                raise RuntimeError(
                    "Predicted SpatialStack evaluation requires residual_predictor_checkpoint."
                )
            if mean_control and not mean_artifact:
                raise RuntimeError(
                    "Mean SpatialStack evaluation requires mean_residual_artifact."
                )
            self.residual_predictor_checkpoint = str(checkpoint_path) if checkpoint_path else None
            self.mean_residual_artifact = str(mean_artifact) if mean_artifact else None
            self.residual_predictor_type = self.residual_predictor_type or getattr(
                self._config, "residual_predictor_type", None
            )
            setattr(self._config, "use_predicted_spatialstack_residuals", True)
            if checkpoint_path:
                setattr(self._config, "residual_predictor_checkpoint", str(checkpoint_path))
            setattr(self._config, "residual_predictor_type", self.residual_predictor_type or getattr(self._config, "residual_predictor_type", None))
            setattr(self._config, "predicted_residual_gamma_layer0", self.predicted_residual_gamma_layers[0])
            setattr(self._config, "predicted_residual_gamma_layer1", self.predicted_residual_gamma_layers[1])
            setattr(self._config, "predicted_residual_gamma_layer2", self.predicted_residual_gamma_layers[2])
            setattr(self._config, "predicted_residual_control", self.predicted_residual_control)
            setattr(self._config, "mean_residual_artifact", self.mean_residual_artifact)
            predictor_adapter = base_model.initialize_predicted_spatialstack_residual_predictor(
                checkpoint_path, self._config
            )
            model_parameter = next(self._model.parameters(), None)
            predictor_dtype = model_parameter.dtype if model_parameter is not None and model_parameter.is_floating_point() else None
            if predictor_dtype is None:
                predictor_adapter.to(device=self._device)
            else:
                predictor_adapter.to(device=self._device, dtype=predictor_dtype)
            predictor_adapter.eval()
            # In interpolation mode the frozen oracle merger remains active.
            if self.spatialstack_residual_mode != "interpolate":
                setattr(self._config, "use_cut3r_spatialstack", False)
            self._predicted_spatialstack_predictor_parameters = sum(
                parameter.numel() for parameter in predictor_adapter.parameters()
            )
            eval_logger.info(
                "[PREDICTED_SPATIALSTACK][EVAL] checkpoint={}, type={}, parameters={}, control={}, CUT3R_disabled=true",
                checkpoint_path or self.mean_residual_artifact,
                getattr(self._config, "residual_predictor_type", None),
                self._predicted_spatialstack_predictor_parameters,
                self.predicted_residual_control,
            )
        if self.disable_cut3r_spatialstack:
            setattr(self._config, "use_cut3r_spatialstack", False)
            eval_logger.info("[GEOMETRY_OFF][EVAL] CUT3R SpatialStack residual branch disabled.")
        self._validate_cut3r_token_only_checkpoint(base_model)
        self.geo_rope_point_map_key = _validate_eval_point_map_key(
            self._config,
            self.geo_rope_point_map_key,
            mutate_config=self.overwrite,
        )
        if self.overwrite:
            setattr(self._config, "zero_spatial_features", self.zero_spatial_features)
            setattr(self._config, "zero_visual_patch_embeddings", self.zero_visual_patch_embeddings)
            setattr(self._config, "probe_geometry_shuffle", self.probe_geometry_shuffle)
            setattr(self._config, "probe_geometry_shuffle_mode", self.probe_geometry_shuffle_mode)
            setattr(self._config, "probe_geometry_shuffle_shift", self.probe_geometry_shuffle_shift)
            setattr(self._config, "probe_geometry_shuffle_seed", self.probe_geometry_shuffle_seed)
            setattr(self._config, "probe_spatial_feature_frame_swap", self.probe_spatial_feature_frame_swap)
            setattr(self._config, "probe_spatial_feature_frame_swap_mode", self.probe_spatial_feature_frame_swap_mode)
            setattr(self._config, "probe_spatial_feature_frame_swap_seed", self.probe_spatial_feature_frame_swap_seed)
            setattr(self._config, "probe_cross_frame_window", self.probe_cross_frame_window)
            setattr(self._config, "probe_cross_frame_include_self", self.probe_cross_frame_include_self)
            setattr(self._config, "probe_cross_frame_mode", self.probe_cross_frame_mode)
            setattr(self._config, "probe_intra_frame_pos_shuffle", self.probe_intra_frame_pos_shuffle)
            setattr(self._config, "llm_visual_3d_rope_enable", self.llm_visual_3d_rope_enable)
            setattr(self._config, "llm_visual_3d_rope_alpha", self.llm_visual_3d_rope_alpha)
            setattr(self._config, "llm_visual_3d_rope_mode", self.llm_visual_3d_rope_mode)
            setattr(self._config, "llm_visual_3d_rope_group_split", self.llm_visual_3d_rope_group_split)
            setattr(self._config, "llm_visual_3d_rope_max_depth", self.llm_visual_3d_rope_max_depth)
            setattr(self._config, "llm_visual_3d_rope_layers", self.llm_visual_3d_rope_layers)
            setattr(self._config, "llm_visual_3d_rope_geometry_source", self.llm_visual_3d_rope_geometry_source)
            setattr(self._config, "llm_visual_3d_rope_shuffle", self.llm_visual_3d_rope_shuffle)
            setattr(self._config, "llm_visual_3d_rope_shuffle_mode", self.llm_visual_3d_rope_shuffle_mode)
            setattr(self._config, "llm_visual_3d_rope_shuffle_seed", self.llm_visual_3d_rope_shuffle_seed)
            setattr(self._config, "llm_visual_3d_rope_log_stats", self.llm_visual_3d_rope_log_stats)
            setattr(self._config, "llm_visual_3d_rope_log_layers", self.llm_visual_3d_rope_log_layers)
            setattr(self._config, "llm_visual_3d_rope_force_eager_attention", self.llm_visual_3d_rope_force_eager_attention)
            setattr(self._config, "cut3r_spatialstack_residual_scale", self.cut3r_spatialstack_residual_scale)
            if self.cut3r_spatialstack_projector_type is not None:
                setattr(self._config, "cut3r_spatialstack_projector_type", self.cut3r_spatialstack_projector_type)
            if self.cut3r_spatialstack_merge_size is not None:
                setattr(self._config, "cut3r_spatialstack_merge_size", self.cut3r_spatialstack_merge_size)
            if self.cut3r_spatialstack_projector_hidden_dim is not None:
                setattr(self._config, "cut3r_spatialstack_projector_hidden_dim", self.cut3r_spatialstack_projector_hidden_dim)
            if self.cut3r_spatialstack_llm_layers is not None:
                setattr(self._config, "cut3r_spatialstack_llm_layers", self.cut3r_spatialstack_llm_layers)
            if self.cut3r_spatialstack_preagg_enable is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_enable", self.cut3r_spatialstack_preagg_enable)
            if self.cut3r_spatialstack_preagg_layers is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_layers", self.cut3r_spatialstack_preagg_layers)
            if self.cut3r_spatialstack_preagg_type is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_type", self.cut3r_spatialstack_preagg_type)
            if self.cut3r_spatialstack_preagg_projector_sharing is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_projector_sharing", self.cut3r_spatialstack_preagg_projector_sharing)
            if self.cut3r_spatialstack_preagg_use_layer_gamma is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_use_layer_gamma", self.cut3r_spatialstack_preagg_use_layer_gamma)
            if self.cut3r_spatialstack_preagg_layer_gamma_init is not None:
                setattr(self._config, "cut3r_spatialstack_preagg_layer_gamma_init", self.cut3r_spatialstack_preagg_layer_gamma_init)
            setattr(self._config, "cut3r_spatialstack_frame_shuffle", self.cut3r_spatialstack_frame_shuffle)
            setattr(self._config, "cut3r_spatialstack_frame_shuffle_mode", self.cut3r_spatialstack_frame_shuffle_mode)
            setattr(self._config, "cut3r_spatialstack_frame_shuffle_seed", self.cut3r_spatialstack_frame_shuffle_seed)
            setattr(self._config, "cut3r_spatialstack_token_shuffle_mode", self.cut3r_spatialstack_token_shuffle_mode)
            setattr(self._config, "cut3r_spatialstack_token_shuffle", self.cut3r_spatialstack_token_shuffle)
            setattr(self._config, "cut3r_spatialstack_token_shuffle_seed", self.cut3r_spatialstack_token_shuffle_seed)
            setattr(self._config, "cut3r_spatialstack_per_frame_token_mean", self.cut3r_spatialstack_per_frame_token_mean)
            if self.use_predicted_spatialstack_residuals:
                setattr(self._config, "use_predicted_spatialstack_residuals", True)
                setattr(self._config, "residual_predictor_type", self.residual_predictor_type)
                setattr(self._config, "residual_predictor_checkpoint", self.residual_predictor_checkpoint)
                setattr(self._config, "predicted_residual_gamma_layer0", self.predicted_residual_gamma_layers[0])
                setattr(self._config, "predicted_residual_gamma_layer1", self.predicted_residual_gamma_layers[1])
                setattr(self._config, "predicted_residual_gamma_layer2", self.predicted_residual_gamma_layers[2])
                setattr(self._config, "predicted_residual_control", self.predicted_residual_control)
                setattr(self._config, "mean_residual_artifact", self.mean_residual_artifact)
        else:
            self.zero_spatial_features = _str_to_bool(
                getattr(self._config, "zero_spatial_features", self.zero_spatial_features)
            )
            self.zero_visual_patch_embeddings = _str_to_bool(
                getattr(self._config, "zero_visual_patch_embeddings", self.zero_visual_patch_embeddings)
            )
            # This inference-only intervention is intentionally allowed with
            # overwrite=False: every checkpoint experiment setting remains loaded
            # verbatim, while this opt-in runtime flag is supplied by the evaluator.
            setattr(
                self._config,
                "zero_visual_patch_embeddings",
                self.zero_visual_patch_embeddings,
            )
            self.llm_visual_3d_rope_enable = _str_to_bool(
                getattr(self._config, "llm_visual_3d_rope_enable", self.llm_visual_3d_rope_enable)
            )
            self.cut3r_spatialstack_residual_scale = float(
                getattr(self._config, "cut3r_spatialstack_residual_scale", self.cut3r_spatialstack_residual_scale)
            )
            self.cut3r_spatialstack_projector_type = getattr(
                self._config, "cut3r_spatialstack_projector_type", self.cut3r_spatialstack_projector_type
            )
            self.cut3r_spatialstack_llm_layers = getattr(
                self._config, "cut3r_spatialstack_llm_layers", self.cut3r_spatialstack_llm_layers
            )
            self.cut3r_spatialstack_frame_shuffle = _str_to_bool(
                getattr(self._config, "cut3r_spatialstack_frame_shuffle", self.cut3r_spatialstack_frame_shuffle)
            )
            self.cut3r_spatialstack_frame_shuffle_mode = getattr(
                self._config, "cut3r_spatialstack_frame_shuffle_mode", self.cut3r_spatialstack_frame_shuffle_mode
            )
            self.cut3r_spatialstack_frame_shuffle_seed = int(
                getattr(self._config, "cut3r_spatialstack_frame_shuffle_seed", self.cut3r_spatialstack_frame_shuffle_seed)
            )
            self.cut3r_spatialstack_token_shuffle = _str_to_bool(
                getattr(self._config, "cut3r_spatialstack_token_shuffle", self.cut3r_spatialstack_token_shuffle)
            )
            self.cut3r_spatialstack_token_shuffle_mode = getattr(
                self._config, "cut3r_spatialstack_token_shuffle_mode", self.cut3r_spatialstack_token_shuffle_mode
            )
            self.cut3r_spatialstack_token_shuffle_seed = int(
                getattr(self._config, "cut3r_spatialstack_token_shuffle_seed", self.cut3r_spatialstack_token_shuffle_seed)
            )
            loaded_config = self._config.to_dict()
            changed_config = {
                key: (expected, loaded_config.get(key))
                for key, expected in preserved_config.items()
                if loaded_config.get(key) != expected
            }
            if changed_config:
                details = ", ".join(
                    f"{key}: checkpoint={expected!r}, loaded={actual!r}"
                    for key, (expected, actual) in sorted(changed_config.items())
                )
                raise RuntimeError(
                    "overwrite=False must preserve checkpoint experiment config, but values changed: "
                    + details
                )
        if self.llm_visual_3d_rope_enable:
            setattr(self._config, "_attn_implementation", "eager")
            setattr(self._config, "_attn_implementation_internal", "eager")
            setattr(self._config, "attn_implementation", "eager")
            base_model = self._model.get_model() if hasattr(self._model, "get_model") else getattr(self._model, "model", None)
            first_attn = None
            if base_model is not None and getattr(base_model, "layers", None):
                first_attn = getattr(base_model.layers[0], "self_attn", None)
            if first_attn is None or first_attn.__class__.__name__ != "Qwen2Visual3DRopeAttention":
                raise RuntimeError(
                    "LLM visual-token 3D RoPE eval requires Qwen2Visual3DRopeAttention eager layers, "
                    f"got {first_attn.__class__.__name__ if first_attn is not None else None}."
                )
        resolved_attn_implementation = getattr(self._config, "_attn_implementation", None)
        if resolved_attn_implementation is None:
            resolved_attn_implementation = getattr(self._config, "attn_implementation", None)
        eval_logger.info(
            "[ATTN][EVAL] requested_attn_implementation={}, resolved_attn_implementation={}",
            self.attn_implementation,
            resolved_attn_implementation,
        )
        eval_logger.info("[ABLATION][EVAL] zero_spatial_features={}", self.zero_spatial_features)
        eval_logger.info("[ABLATION][EVAL] zero_visual_patch_embeddings={}", self.zero_visual_patch_embeddings)
        eval_logger.info("[ABLATION][EVAL] force_geo_rope_gate_zero={}", self.force_geo_rope_gate_zero)
        eval_logger.info(
            "[PROBE][EVAL] geometry_shuffle={}, mode={}, shift={}, seed={}",
            self.probe_geometry_shuffle,
            self.probe_geometry_shuffle_mode,
            self.probe_geometry_shuffle_shift,
            self.probe_geometry_shuffle_seed,
        )
        eval_logger.info(
            "[PROBE][EVAL] spatial_feature_frame_swap={}, mode={}, seed={}",
            self.probe_spatial_feature_frame_swap,
            self.probe_spatial_feature_frame_swap_mode,
            self.probe_spatial_feature_frame_swap_seed,
        )
        eval_logger.info(
            "[PROBE][EVAL] cross_frame_window={}, include_self={}, mode={}",
            self.probe_cross_frame_window,
            self.probe_cross_frame_include_self,
            self.probe_cross_frame_mode,
        )
        eval_logger.info("[PROBE][EVAL] intra_frame_pos_shuffle={}", self.probe_intra_frame_pos_shuffle)
        eval_logger.info(
            "[SPATIALSTACK][EVAL] residual_scale={}; llm_layers={}; projector_type={}, merge_size={}, projector_hidden_dim={}; preagg_enable={}, preagg_layers={}, preagg_type={}, preagg_projector_sharing={}, preagg_use_layer_gamma={}, preagg_layer_gamma_init={}; frame_shuffle={}, mode={}, seed={}; token_shuffle={}, mode={}, seed={}",
            self.cut3r_spatialstack_residual_scale,
            getattr(self._config, "cut3r_spatialstack_llm_layers", None),
            getattr(self._config, "cut3r_spatialstack_projector_type", "token_mlp"),
            getattr(self._config, "cut3r_spatialstack_merge_size", 2),
            getattr(self._config, "cut3r_spatialstack_projector_hidden_dim", 4096),
            getattr(self._config, "cut3r_spatialstack_preagg_enable", False),
            getattr(self._config, "cut3r_spatialstack_preagg_layers", "6,9,12"),
            getattr(self._config, "cut3r_spatialstack_preagg_type", "weighted_sum"),
            getattr(self._config, "cut3r_spatialstack_preagg_projector_sharing", "shared"),
            getattr(self._config, "cut3r_spatialstack_preagg_use_layer_gamma", True),
            getattr(self._config, "cut3r_spatialstack_preagg_layer_gamma_init", 1.0),
            self.cut3r_spatialstack_frame_shuffle,
            self.cut3r_spatialstack_frame_shuffle_mode,
            self.cut3r_spatialstack_frame_shuffle_seed,
            self.cut3r_spatialstack_token_shuffle,
            self.cut3r_spatialstack_token_shuffle_mode,
            self.cut3r_spatialstack_token_shuffle_seed,
        )
        eval_logger.info(
            "[ROPE][EVAL] geo_rope_point_map_key={}, training_point_map_key={}",
            getattr(self._config, "geo_rope_point_map_key", None),
            getattr(self._config, "geo_rope_training_point_map_key", None),
        )
        eval_logger.info(
            "[ROPE][EVAL] fusion_block={}, geometry_rope_mode={}, group_split={}, max_depth={}, log_stats={}, eval_lambda={}",
            getattr(self._config, "fusion_block", None),
            getattr(self._config, "geo_rope_fusion_mode", None) or getattr(self._config, "geometry_rope_mode", None),
            getattr(self._config, "geo_rope_fusion_group_split", None) or getattr(self._config, "geometry_rope_group_split", None),
            getattr(self._config, "geo_rope_fusion_max_depth", None) or getattr(self._config, "geometry_rope_max_depth", None),
            getattr(self._config, "geometry_rope_log_stats", None),
            getattr(self._config, "geo_rope_fusion_eval_lambda", None),
        )
        eval_logger.info(
            "[SPATIAL][EVAL] spatial_tower={}, spatial_features_root={}, spatial_features_subdir={}",
            getattr(self._config, "spatial_tower", None),
            self.spatial_features_root,
            self.spatial_features_subdir,
        )
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _is_cut3r_token_only(self):
        return str(getattr(self._config, "visual_token_source", "siglip_only") or "siglip_only").lower() == "cut3r_only"

    def _uses_predicted_spatialstack_residuals(self):
        return bool(
            self.use_predicted_spatialstack_residuals
            or _str_to_bool(getattr(self._config, "use_predicted_spatialstack_residuals", False))
        )

    def _skips_spatial_sidecars(self):
        return (self._uses_predicted_spatialstack_residuals() and self.spatialstack_residual_mode != "interpolate") or self.disable_cut3r_spatialstack

    def _install_cut3r_token_only_siglip_bypass_guard(self, base_model):
        if getattr(self, "_cut3r_siglip_forward_guard", None) is not None:
            return
        getter = getattr(self._model, "get_vision_tower", None)
        vision_tower = getter() if callable(getter) else None
        if vision_tower is None:
            getter = getattr(base_model, "get_vision_tower", None)
            vision_tower = getter() if callable(getter) else getattr(base_model, "vision_tower", None)
        if vision_tower is None or not hasattr(vision_tower, "forward"):
            raise RuntimeError("CUT3R-token-only evaluator cannot install the required SigLIP forward guard.")
        original_forward = vision_tower.forward
        guard = {"calls": 0, "original_forward": original_forward}

        def guarded_forward(*args, **kwargs):
            guard["calls"] += 1
            raise RuntimeError("SigLIP forward ran during CUT3R-token-only evaluation.")

        vision_tower.forward = guarded_forward
        self._cut3r_siglip_forward_guard = guard

    def _validate_cut3r_token_only_checkpoint(self, base_model):
        source = str(getattr(self._config, "visual_token_source", "siglip_only") or "siglip_only").lower()
        if self.visual_token_source is not None and source != self.visual_token_source.lower():
            raise RuntimeError(f"Evaluator requested visual_token_source={self.visual_token_source!r}, checkpoint records {source!r}.")
        if source != "cut3r_only":
            return
        forbidden_flags = (
            "use_cut3r_spatialstack", "use_cut3r_camera_tokens", "use_geometry_aware_projection",
            "llm_visual_3d_rope_enable", "use_spatial_bridge_tokens", "add_faster_video",
            "use_bev_supervision", "use_depth_supervision", "use_pointmap_supervision",
        )
        active = [name for name in forbidden_flags if _str_to_bool(getattr(self._config, name, False))]
        fusion = getattr(self._config, "fusion_block", None)
        if fusion not in (None, "", "none", "None"):
            active.append("fusion_block")
        if active:
            raise RuntimeError("CUT3R-token-only evaluator checkpoint enables incompatible paths: " + ", ".join(active))
        if str(getattr(self._config, "cut3r_token_sidecar_key", "patch_tokens")) != "patch_tokens" or int(getattr(self._config, "cut3r_token_feature_dim", 768)) != 768:
            raise RuntimeError("CUT3R-token-only evaluator requires final patch_tokens [F,729,768].")
        projector_getter = getattr(base_model, "get_cut3r_token_projector", None)
        projector = projector_getter() if callable(projector_getter) else None
        if projector is None:
            raise RuntimeError("CUT3R-token-only checkpoint did not reconstruct cut3r_token_projector.")
        state_path = Path(self.pretrained) / "non_lora_trainables.bin"
        if not state_path.is_file():
            raise RuntimeError(f"CUT3R-token-only checkpoint is missing projector state: {state_path}")
        raw_state = torch.load(str(state_path), map_location="cpu")
        keys = assert_cut3r_token_projector_checkpoint_values(projector, raw_state)
        self._cut3r_projector_checkpoint_values_verified = True
        self._install_cut3r_token_only_siglip_bypass_guard(base_model)
        parameter = next(projector.parameters())
        eval_logger.info(
            "[CUT3R_TOKEN_ONLY][EVAL] projector_device={}, projector_dtype={}, keys={}, checkpoint_values_verified=true",
            parameter.device, parameter.dtype, keys,
        )

    def _validate_cut3r_token_only_sidecar(self, video_path, sidecar, sidecar_path, selected_frame_indices):
        if not self._is_cut3r_token_only():
            return
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"CUT3R-token-only VSI sidecar for {video_path} must be a dict.")
        tokens = sidecar.get("patch_tokens")
        expected_shape = (len(selected_frame_indices), 729, 768)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected_shape:
            raise RuntimeError(f"CUT3R-token-only VSI sidecar shape mismatch for {video_path}: got {tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else None}, expected {expected_shape}.")
        if not torch.isfinite(tokens).all():
            raise RuntimeError(f"CUT3R-token-only VSI sidecar contains non-finite patch_tokens: {video_path}")
        frame_indices = self._sidecar_frame_indices(sidecar)
        source = "embedded_metadata" if frame_indices is not None else None
        if frame_indices is None:
            frame_indices = validate_cut3r_token_sidecar_manifest_entry(self.cut3r_token_sidecar_manifest, video_path=video_path, sidecar_path=sidecar_path, patch_tokens=tokens, selected_frame_indices=selected_frame_indices, video_fps=1, frames_upbound=self.max_frames_num, force_sample=True, policy=self.cut3r_token_manifest_policy, warning_callback=eval_logger.warning)
            source = "manifest_verified" if frame_indices is not None else "deterministic_legacy_fallback"
        if frame_indices is not None and list(frame_indices) != list(selected_frame_indices):
            raise RuntimeError(f"CUT3R-token-only VSI sidecar frame order mismatch for {video_path}: sidecar={frame_indices}, sampler={selected_frame_indices}")
        if frame_indices is None:
            frame_indices = list(selected_frame_indices)
            eval_logger.warning("[CUT3R_TOKEN_ONLY][MANIFEST][WARN] provenance=deterministic_legacy_fallback video={} sidecar_shape={} selected_frame_indices={} sampler={{'video_fps': 1, 'frames_upbound': {}, 'force_sample': true}}", video_path, tuple(tokens.shape), frame_indices, self.max_frames_num)
        sidecar["_cut3r_token_only_provenance"] = {"source": source, "selected_frame_indices": list(map(int, frame_indices)), "sampling": {"video_fps": 1, "frames_upbound": int(self.max_frames_num), "force_sample": True}}

    def _write_cut3r_token_only_eval_telemetry(self, mode, spatial_features, logits):
        if not self._is_cut3r_token_only():
            return
        path_value = os.environ.get("CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_PATH", "")
        if not path_value:
            return
        base_model = self._model.get_model() if hasattr(self._model, "get_model") else getattr(self._model, "model", None)
        projector_getter = getattr(base_model, "get_cut3r_token_projector", None)
        projector = projector_getter() if callable(projector_getter) else None
        parameter = next(projector.parameters()) if projector is not None else None
        guard = getattr(self, "_cut3r_siglip_forward_guard", None)
        model_metrics = dict(getattr(base_model, "_cut3r_token_only_last_metrics", {})) if base_model is not None else {}
        siglip_forward_bypassed = bool(
            isinstance(guard, dict)
            and int(guard.get("calls", -1)) == 0
            and model_metrics.get("source") == "cut3r_only"
            and model_metrics.get("siglip_forward_bypassed") is True
        )
        if not siglip_forward_bypassed:
            raise RuntimeError(
                "CUT3R-token-only evaluator could not prove SigLIP bypass "
                f"(guard_calls={None if not isinstance(guard, dict) else guard.get('calls')}, metrics={model_metrics})."
            )
        finite_logits = bool(logits is not None and torch.isfinite(logits).all().item())
        payload = {
            "checkpoint": self.pretrained,
            "mode": mode,
            "checkpoint_reloaded": bool(projector is not None and getattr(self, "_cut3r_projector_checkpoint_values_verified", False)),
            "resumed_forward_passed": finite_logits,
            "evaluator_preflight_passed": bool(finite_logits and siglip_forward_bypassed),
            "visual_token_source": getattr(self._config, "visual_token_source", None),
            "siglip_forward_bypassed": siglip_forward_bypassed,
            "siglip_forward_calls": int(guard["calls"]),
            "projector_checkpoint_values_verified": bool(getattr(self, "_cut3r_projector_checkpoint_values_verified", False)),
            "projector_device": str(parameter.device) if parameter is not None else None,
            "projector_dtype": str(parameter.dtype) if parameter is not None else None,
            "sidecar_shapes": [list(item.get("patch_tokens").shape) for item in (spatial_features or []) if isinstance(item, dict) and isinstance(item.get("patch_tokens"), torch.Tensor)],
        }
        output = Path(path_value)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload["report_path"] = str(output)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        eval_logger.info("[CUT3R_TOKEN_ONLY][EVAL_TELEMETRY] {}", json.dumps(payload, sort_keys=True))

    def load_video(self, video_path, max_frames_num, return_indices=False):
        from types import SimpleNamespace
        from llava.utils import process_video_with_decord

        sampler_args = SimpleNamespace(video_fps=1, frames_upbound=int(max_frames_num), force_sample=True)
        frames, _, _, _, frame_indices = process_video_with_decord(video_path, sampler_args, return_indices=True)
        return (frames, frame_indices) if return_indices else frames

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            spatial_features = []
            for visual in visuals:
                video, selected_frame_indices = self.load_video(visual, self.max_frames_num, return_indices=True)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)
                if self._skips_spatial_sidecars():
                    sidecar = None
                else:
                    sidecar = self._load_spatial_sidecar(visual, selected_frame_indices=selected_frame_indices)
                if sidecar is not None:
                    spatial_features.append(sidecar)
            spatial_features = spatial_features if len(spatial_features) > 0 else None

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, spatial_features=spatial_features, modalities="video")
            self._write_llm_visual_3d_rope_eval_stats("loglikelihood")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            self._write_cut3r_token_only_eval_telemetry("loglikelihood", spatial_features, logits)
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _jsonable(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().float().item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): self._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _collect_llm_visual_3d_rope_eval_stats(self):
        base = self.model.get_model() if hasattr(self.model, "get_model") else self.model
        prefill_stats = getattr(self.model, "_last_llm_visual_3d_rope_prefill_stats", None)
        prefill_metadata = getattr(self.model, "_last_llm_geo_prefill_debug", None)
        decode_stats = getattr(self.model, "_last_llm_visual_3d_rope_decode_stats", None)
        decode_metadata = getattr(self.model, "_last_llm_geo_decode_debug", None)
        stats = prefill_stats or getattr(base, "_last_llm_visual_3d_rope_stats", None)
        metadata = prefill_metadata or getattr(self.model, "_last_llm_geo_debug", None)
        if not stats:
            return None
        non_skipped = [item for item in stats if not item.get("skipped", False)]
        aggregate = {
            "num_logged_layers": len(stats),
            "num_active_layers": len(non_skipped),
            "attention_delta_mean_abs": 0.0,
            "visual_visual_logits_delta_mean_abs": 0.0,
            "num_valid_geo_tokens": 0,
        }
        if non_skipped:
            aggregate["attention_delta_mean_abs"] = float(
                sum(float(item.get("attention_delta_mean_abs", 0.0) or 0.0) for item in non_skipped) / len(non_skipped)
            )
            aggregate["visual_visual_logits_delta_mean_abs"] = float(
                sum(float(item.get("visual_visual_logits_delta_mean_abs", 0.0) or 0.0) for item in non_skipped)
                / len(non_skipped)
            )
        for item in stats:
            aggregate["num_valid_geo_tokens"] = max(
                aggregate["num_valid_geo_tokens"],
                int(item.get("num_valid_geo_tokens", 0) or 0),
            )
        return {
            "aggregate": aggregate,
            "layers": self._jsonable(stats),
            "metadata": self._jsonable(metadata),
            "decode_layers": self._jsonable(decode_stats),
            "decode_metadata": self._jsonable(decode_metadata),
        }

    def _write_llm_visual_3d_rope_eval_stats(self, stage):
        if not self.llm_visual_3d_rope_enable or not self.llm_visual_3d_rope_log_stats or self.rank != 0:
            return
        stats = self._collect_llm_visual_3d_rope_eval_stats()
        if not stats:
            return
        payload = {
            "event": "eval",
            "counter": int(self._llm_visual_3d_rope_eval_counter),
            "stage": stage,
            "alpha": self.llm_visual_3d_rope_alpha,
            "shuffle_enabled": self.llm_visual_3d_rope_shuffle,
            "shuffle_mode": self.llm_visual_3d_rope_shuffle_mode,
            "shuffle_seed": self.llm_visual_3d_rope_shuffle_seed,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if torch.cuda.is_available() else 0,
            **stats,
        }
        self._llm_visual_3d_rope_eval_counter += 1
        line = json.dumps(payload, sort_keys=True)
        eval_logger.info(f"[LLM_VISUAL_3D_ROPE_STATS] {line}")
        if self.llm_visual_3d_rope_stats_path is None:
            return
        try:
            self.llm_visual_3d_rope_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.llm_visual_3d_rope_stats_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            eval_logger.warning(f"Failed to write LLM visual 3D RoPE stats: {exc}")

    def _requires_geometry_rope_sidecar(self):
        if getattr(self, "llm_visual_3d_rope_enable", False):
            return True
        fusion_block = self.fusion_block or getattr(self._config, "fusion_block", None)
        return fusion_block in {"svf_3d_rope", "svf_depth_rope", "svf_xyz_rope", "svf_spherical_rope"}

    @staticmethod
    def _try_parse_spatial_layer_key(value, infer_from_path=False):
        if value is None:
            return None
        text = str(value).strip().strip("/\\")
        if not text:
            return None

        def _parse_token(token):
            token = str(token).strip().lower()
            if token.startswith("decoder"):
                token = token[len("decoder") :].strip("_-")
            if token.startswith("dec"):
                token = token[len("dec") :].strip("_-")
            if token.startswith("layer"):
                token = token[len("layer") :].strip("_-")
            if token.startswith("m") and token[1:].isdigit():
                token = "-" + token[1:]
            try:
                return str(int(token))
            except ValueError:
                return None

        parsed = _parse_token(text)
        if parsed is not None or not infer_from_path:
            return parsed
        basename = os.path.basename(text)
        match = re.search(r"(?:^|[_-])(?:decoder|dec|layer)?[_-]?(m?-?\d+)$", basename.lower())
        if match:
            return _parse_token(match.group(1))
        return None

    @classmethod
    def _split_spatial_layer_specs(cls, features_subdir):
        if not isinstance(features_subdir, str):
            return None
        parts = [part.strip() for part in features_subdir.replace(";", ",").split(",") if part.strip()]
        if len(parts) <= 1 and (not parts or not any(sep in parts[0] for sep in (":", "="))):
            return None
        specs = []
        for part in parts:
            colon_pieces = [piece.strip() for piece in part.split(":")]
            if len(colon_pieces) >= 3 and cls._try_parse_spatial_layer_key(colon_pieces[0], infer_from_path=False) is not None:
                layer_key = cls._try_parse_spatial_layer_key(colon_pieces[0], infer_from_path=False)
                layer_root = ":".join(colon_pieces[1:-1]).strip()
                layer_subdir = colon_pieces[-1]
            elif ":" in part:
                left, right = [piece.strip() for piece in part.split(":", 1)]
                layer_root = None
                left_key = cls._try_parse_spatial_layer_key(left, infer_from_path=False)
                right_key = cls._try_parse_spatial_layer_key(right, infer_from_path=False)
                if left_key is not None:
                    layer_key, layer_subdir = left_key, right
                elif right_key is not None:
                    layer_key, layer_subdir = right_key, left
                else:
                    inferred_key = cls._try_parse_spatial_layer_key(left, infer_from_path=True)
                    if inferred_key is None:
                        inferred_key = cls._try_parse_spatial_layer_key(right, infer_from_path=True)
                        layer_subdir = left
                    else:
                        layer_subdir = right
                    if inferred_key is None:
                        raise ValueError(f"Cannot infer CUT3R decoder layer from spatial_features_subdir spec: {part!r}")
                    layer_key = inferred_key
            elif "=" in part:
                left, right = [piece.strip() for piece in part.split("=", 1)]
                layer_root = None
                left_key = cls._try_parse_spatial_layer_key(left, infer_from_path=False)
                right_key = cls._try_parse_spatial_layer_key(right, infer_from_path=False)
                if left_key is not None:
                    layer_key, layer_subdir = left_key, right
                elif right_key is not None:
                    layer_key, layer_subdir = right_key, left
                else:
                    raise ValueError(f"Cannot infer CUT3R decoder layer from spatial_features_subdir spec: {part!r}")
            else:
                layer_root = None
                layer_key = cls._try_parse_spatial_layer_key(part, infer_from_path=True)
                if layer_key is None:
                    raise ValueError(
                        "Comma-separated spatial_features_subdir entries must include a decoder layer, "
                        f"for example '6:{part}'."
                    )
                layer_subdir = part
            if not layer_subdir:
                raise ValueError(f"Empty spatial feature subdir in spec: {part!r}")
            if any(existing_layer == layer_key for existing_layer, _, _ in specs):
                raise ValueError(f"Duplicate CUT3R decoder layer {layer_key} in spatial_features_subdir={features_subdir!r}")
            specs.append((layer_key, layer_root, layer_subdir))
        return specs

    @staticmethod
    def _sidecar_frame_indices(sidecar):
        if not isinstance(sidecar, dict):
            return None
        for key in ("frame_indices", "frame_order"):
            if key in sidecar:
                value = sidecar[key]
                if isinstance(value, torch.Tensor):
                    return [int(x) for x in value.detach().cpu().flatten().tolist()]
                return [int(x) for x in value]
        metadata = sidecar.get("metadata")
        if isinstance(metadata, dict):
            for key in ("frame_indices", "frame_order"):
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, torch.Tensor):
                        return [int(x) for x in value.detach().cpu().flatten().tolist()]
                    return [int(x) for x in value]
        return None

    def _spatial_sidecar_candidates(self, video_path, features_root=None, features_subdir=None):
        features_root = Path(features_root) if features_root not in (None, "") else self.spatial_features_root
        features_subdir = features_subdir or self.spatial_features_subdir
        if features_root is None:
            return []

        video_path = Path(video_path)
        candidates = []
        datasets = ("scannetpp", "scannet", "arkitscenes")

        for dataset in datasets:
            if dataset not in video_path.parts:
                continue

            dataset_idx = video_path.parts.index(dataset)
            tail_parts = video_path.parts[dataset_idx + 1 :]
            if len(tail_parts) > 0 and tail_parts[0] == "videos":
                tail_parts = tail_parts[1:]

            if len(tail_parts) == 0:
                continue

            rel_path = Path(dataset) / features_subdir / Path(*tail_parts)
            candidates.append((features_root / rel_path).with_suffix(".pt"))

        return candidates

    def _load_single_spatial_sidecar(self, video_path, features_root=None, features_subdir=None):
        features_root = Path(features_root) if features_root not in (None, "") else self.spatial_features_root
        if features_root is None:
            if self._requires_geometry_rope_sidecar():
                raise RuntimeError("Geometry-RoPE eval requires spatial_features_root for CUT3R point-map sidecars.")
            return None, None

        candidates = self._spatial_sidecar_candidates(video_path, features_root=features_root, features_subdir=features_subdir)
        for candidate in candidates:
            if not candidate.is_file():
                continue

            sidecar = torch.load(str(candidate), map_location="cpu")
            return sidecar, candidate

        if self._requires_geometry_rope_sidecar():
            pretty = ", ".join(str(path) for path in candidates) or "<no candidates>"
            raise FileNotFoundError(f"Missing CUT3R point-map sidecar for {video_path}. Tried: {pretty}")
        return None, None

    def _compose_layered_spatial_sidecar(self, video_path):
        combined_layers = {}
        loaded_paths = {}
        metadata = {}
        reference_frame_indices = None
        missing = []

        for layer_key, layer_root, layer_subdir in self._spatial_layer_specs:
            effective_root = layer_root or self.spatial_features_root
            sidecar, sidecar_path = self._load_single_spatial_sidecar(
                video_path,
                features_root=effective_root,
                features_subdir=layer_subdir,
            )
            if sidecar is None:
                missing.append(f"layer {layer_key}: root={effective_root}, subdir={layer_subdir}")
                continue
            if not isinstance(sidecar, dict):
                raise RuntimeError(
                    f"Layered spatial_features sidecar for layer {layer_key} must be a dict, "
                    f"got {type(sidecar).__name__} from {sidecar_path}."
                )
            frame_indices = self._sidecar_frame_indices(sidecar)
            if reference_frame_indices is None:
                reference_frame_indices = frame_indices
            elif frame_indices is not None and reference_frame_indices != frame_indices:
                raise RuntimeError(
                    f"Layered spatial_features frame_indices mismatch for {video_path}, layer {layer_key}: "
                    f"{frame_indices} != {reference_frame_indices}."
                )

            if "cut3r_dec_layers" in sidecar:
                layer_payloads = sidecar["cut3r_dec_layers"]
                if not isinstance(layer_payloads, dict):
                    raise RuntimeError(f"cut3r_dec_layers in {sidecar_path} must be a dict.")
                if layer_key not in layer_payloads and int(layer_key) not in layer_payloads:
                    raise RuntimeError(
                        f"Sidecar {sidecar_path} does not contain requested decoder layer {layer_key}; "
                        f"available keys={sorted(str(k) for k in layer_payloads.keys())}."
                    )
                payload = layer_payloads.get(layer_key, layer_payloads.get(int(layer_key)))
            elif "patch_tokens" in sidecar:
                payload = {"patch_tokens": sidecar["patch_tokens"]}
                if "camera_tokens" in sidecar:
                    payload["camera_tokens"] = sidecar["camera_tokens"]
            else:
                raise RuntimeError(
                    f"Layered spatial_features sidecar {sidecar_path} must contain 'patch_tokens' "
                    "or 'cut3r_dec_layers'."
                )

            combined_layers[layer_key] = payload
            loaded_paths[layer_key] = str(sidecar_path)
            if not metadata and isinstance(sidecar.get("metadata"), dict):
                metadata = dict(sidecar["metadata"])
            for key in ("frame_indices", "frame_order"):
                if key in sidecar and key not in metadata:
                    metadata[key] = sidecar[key]

        if missing:
            raise FileNotFoundError(
                "Missing layered spatial_features sidecar(s) for "
                f"{video_path}: " + "; ".join(missing)
            )
        metadata = dict(metadata)
        metadata["layer_subdir_paths"] = loaded_paths
        combined = {"cut3r_dec_layers": combined_layers, "metadata": metadata}
        for key in ("frame_indices", "frame_order"):
            if key in metadata:
                combined[key] = metadata[key]
        return combined

    def _load_spatial_sidecar(self, video_path, selected_frame_indices=None):
        if self._uses_predicted_spatialstack_residuals() and self.spatialstack_residual_mode != "interpolate":
            self._predicted_spatialstack_sidecar_load_attempts += 1
            raise RuntimeError("CUT3R sidecar loading is forbidden in predicted SpatialStack residual mode.")
        if self.disable_cut3r_spatialstack:
            return None
        if self._spatial_layer_specs:
            return self._compose_layered_spatial_sidecar(video_path)
        loaded = self._load_single_spatial_sidecar(video_path)
        if loaded is None:
            return None
        sidecar, sidecar_path = loaded
        if sidecar is not None:
            if selected_frame_indices is not None:
                self._validate_cut3r_token_only_sidecar(video_path, sidecar, sidecar_path, selected_frame_indices)
            return sidecar
        return None

    def _canonical_expected_key(self, task, split, doc_id, doc):
        if not self._expected_key_by_doc_id:
            return None
        item = self._expected_key_by_doc_id.get(int(doc_id))
        if item is None:
            raise RuntimeError(f"Evaluation doc_id={doc_id} is absent from frozen expected-key manifest.")
        row_json = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        row_hash = hashlib.sha256(row_json.encode("utf-8")).hexdigest()
        if str(item["task"]) != str(task) or str(item["split"]) != str(split) or row_hash != item["row_sha256"]:
            raise RuntimeError(f"Frozen-manifest mismatch for doc_id={doc_id}; task input/order has changed.")
        return str(item["canonical_key"])

    def _write_experiment_telemetry(self, payload):
        if self.evaluation_telemetry_dir is None:
            return
        rank = int(getattr(self, "rank", 0))
        directory = self.evaluation_telemetry_dir / f"rank_{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "samples.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")

    def _experiment_spatialstack_debug(self):
        model = self.model
        debug = getattr(model, "_last_cut3r_spatialstack_debug", None)
        if debug is None:
            debug = getattr(getattr(model, "module", None), "_last_cut3r_spatialstack_debug", None)
        return dict(debug or {})

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Encode, pad, and truncate contexts for this batch. Validate the
            # frozen source row before generation so rank sharding cannot hide
            # an input-order or task-version change.
            source_doc = self.task_dict[task][split][doc_id]
            canonical_key = self._canonical_expected_key(task, split, doc_id, source_doc)
            visuals = [doc_to_visual(source_doc)]
            preflight_video = os.environ.get("CUT3R_TOKEN_ONLY_EVAL_PREFLIGHT_VIDEO", "")
            if preflight_video:
                preflight_path = Path(preflight_video).resolve()
                if not preflight_path.is_file():
                    raise RuntimeError(f"CUT3R-token-only evaluator preflight video is missing: {preflight_path}")
                if not self._is_cut3r_token_only():
                    raise RuntimeError("CUT3R-token-only evaluator preflight video override requires visual_token_source=cut3r_only.")
                visuals = [str(preflight_path)]
                eval_logger.info("[CUT3R_TOKEN_ONLY][EVAL_PREFLIGHT] using verified manifest video override: {}", preflight_path)
            if visuals != [None]:
                visuals = self.flatten(visuals)
                videos = []
                spatial_features = []
                try:
                    for visual in visuals:
                        if self._is_cut3r_token_only() or self.video_decode_backend == "decord":
                            video, selected_frame_indices = self.load_video(visual, self.max_frames_num, return_indices=True)
                        elif self.video_decode_backend == "pyav":
                            video = read_video_pyav(visual, num_frm=self.max_frames_num)
                            selected_frame_indices = None
                        # video = self.load_video(visual, self.max_frames_num)
                        video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                        videos.append(video)
                        if self._skips_spatial_sidecars():
                            sidecar = None
                        else:
                            sidecar = self._load_spatial_sidecar(visual, selected_frame_indices=selected_frame_indices)
                        if sidecar is not None:
                            spatial_features.append(sidecar)
                except Exception as e:
                    eval_logger.info(f"{e}")
                    eval_logger.info(f"Video {visuals} can not load, check the source")
                    video_path = "\n".join(visuals)
                    res.append(f"Video {video_path} can not load, check the source")
                    pbar.update(1)
                    continue
                spatial_features = spatial_features if len(spatial_features) > 0 else None

                qs = contexts
                if self.model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN * len(videos) + "\n" + qs
            else:
                videos = None
                spatial_features = None
                qs = contexts

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if "llama_3" in self.conv_template:
                pad_token_ids = 0  # lmms-lab/llama3-llava-8b is trained on this pad token id. You may need to customize this for other models.
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = contexts

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=videos,
                    spatial_features=spatial_features,
                    attention_mask=attention_masks,
                    modalities=["video" for _ in videos] if videos is not None else None,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                self._write_llm_visual_3d_rope_eval_stats("generate")
                self._write_cut3r_token_only_eval_telemetry("generate", spatial_features, output_ids)
                # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, use_cache=True, stopping_criteria=[stopping_criteria])

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # inputs = self.tokenizer.batch_decode(input_ids % self.tokenizer.vocab_size, skip_special_tokens=True)[0].strip()
            # print(inputs, outputs)
            self._write_experiment_telemetry({
                "canonical_key": canonical_key, "doc_id": int(doc_id), "task": str(task), "split": str(split),
                "rank": int(getattr(self, "rank", 0)), "answer": outputs,
                "generated_token_ids": output_ids.detach().cpu().reshape(-1).tolist(),
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if torch.cuda.is_available() else 0,
                "spatialstack_payload_provenance": self._experiment_spatialstack_debug(),
                "teacher_residual_scale_provenance": self._teacher_residual_scale_provenance,
            })
            res.append(outputs)
            pbar.update(1)
        return res
