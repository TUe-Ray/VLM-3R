#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.geometry import build_bev_targets_from_point_maps, build_depth_targets_from_point_maps
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.model.language_model.llm_visual_3d_rope import (
    clear_qwen2_visual_3d_rope_context,
    collect_qwen2_visual_3d_rope_stats,
    install_qwen2_visual_3d_rope_attention,
    qwen2_visual_3d_rope_requires_eager,
)

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


def _as_bool_config(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        install_qwen2_visual_3d_rope_attention()
        super(LlavaQwenModel, self).__init__(config)

    def _decoder_layer_forward_with_llm_geo(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        llm_geo_pos,
        llm_geo_mask,
    ):
        attn = getattr(decoder_layer, "self_attn", None)
        if attn is not None:
            attn._llm_visual_3d_rope_pos = llm_geo_pos
            attn._llm_visual_3d_rope_mask = llm_geo_mask
            attn.last_llm_visual_3d_rope_stats = None
        try:
            return decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        finally:
            if attn is not None:
                for attr in ("_llm_visual_3d_rope_pos", "_llm_visual_3d_rope_mask"):
                    if hasattr(attn, attr):
                        delattr(attn, attr)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        llm_geo_pos: Optional[torch.FloatTensor] = None,
        llm_geo_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if qwen2_visual_3d_rope_requires_eager(self.config):
            raise RuntimeError("LLM visual-token 3D RoPE requires Qwen2 eager attention; disable FlashAttention/SDPA.")
        try:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                use_cache = False

            past_key_values_length = 0
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            else:
                use_legacy_cache = False

            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
                is_padding_right = attention_mask[:, -1].sum().item() != batch_size
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'. "
                        "Use left padding for batched generation."
                    )

            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self._attn_implementation == "sdpa" and not output_attentions:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.config.sliding_window,
                )

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        self._decoder_layer_forward_with_llm_geo,
                        decoder_layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        llm_geo_pos,
                        llm_geo_mask,
                    )
                else:
                    layer_outputs = self._decoder_layer_forward_with_llm_geo(
                        decoder_layer,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        llm_geo_pos,
                        llm_geo_mask,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

            if not return_dict:
                outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            else:
                outputs = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )
            self._last_llm_visual_3d_rope_stats = collect_qwen2_visual_3d_rope_stats(self)
            return outputs
        finally:
            clear_qwen2_visual_3d_rope_context(self)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        install_qwen2_visual_3d_rope_attention()
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        if getattr(config, "spatial_rank_projection_dim", None) is not None:
            self.initialize_spatial_rank_head(output_dim=int(config.spatial_rank_projection_dim))
        use_bev_supervision = getattr(config, "use_bev_supervision", False)
        if isinstance(use_bev_supervision, str):
            use_bev_supervision = use_bev_supervision.lower() in {"1", "true", "yes", "y", "on"}
        if use_bev_supervision:
            self.initialize_bev_head()
        use_depth_supervision = getattr(config, "use_depth_supervision", False)
        if isinstance(use_depth_supervision, str):
            use_depth_supervision = use_depth_supervision.lower() in {"1", "true", "yes", "y", "on"}
        if use_depth_supervision:
            self.initialize_depth_head()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 创建模型实例
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # 加载自定义权重
        if model.get_spatial_tower() is not None:
            zero_spatial_features = getattr(model.config, "zero_spatial_features", False)
            if isinstance(zero_spatial_features, str):
                zero_spatial_features = zero_spatial_features.lower() in {"1", "true", "yes", "y", "on"}
            preextracted_only = getattr(model.config, "spatial_tower_preextracted_only", False)
            if isinstance(preextracted_only, str):
                preextracted_only = preextracted_only.lower() in {"1", "true", "yes", "y", "on"}

            # Vision-only ablation: keep spatial tower wrapper but skip heavy weight loading.
            if not zero_spatial_features and not preextracted_only:
                model.get_spatial_tower().is_loaded = False
                model.get_spatial_tower().load_model()
                model.get_spatial_tower().is_loaded = True
                model.get_spatial_tower().to(kwargs.get("torch_dtype", torch.float16))

        return model

    def get_model(self):
        return self.model

    @staticmethod
    def _as_bool_config(value, default=False):
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _metadata_items(visual_metadata):
        if isinstance(visual_metadata, dict):
            return [visual_metadata]
        if isinstance(visual_metadata, (list, tuple)):
            return list(visual_metadata)
        raise RuntimeError(
            "Auxiliary visual-token supervision requires visual metadata from prepare_inputs_labels_for_multimodal(); "
            f"got {type(visual_metadata).__name__}."
        )

    @staticmethod
    def _total_visual_tokens_from_metadata(visual_metadata):
        total = 0
        for metadata in LlavaQwenForCausalLM._metadata_items(visual_metadata):
            indices = metadata.get("visual_token_indices") if isinstance(metadata, dict) else None
            if not isinstance(indices, torch.Tensor):
                raise RuntimeError("Auxiliary visual metadata is missing tensor visual_token_indices.")
            total += int(indices.numel())
        return total

    def _select_aux_hidden_states(self, outputs, captured_final_hidden=None, *, source="llm_output", aux_name="aux"):
        source = str(source or "llm_output")
        if source == "llm_output":
            if captured_final_hidden is not None:
                return captured_final_hidden
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                return hidden_states[-1]
            raise RuntimeError(
                f"{aux_name}_head_source='llm_output' could not capture final sequence hidden states. "
                "Expected the final-hidden hook on the base LLM to fire; if this model class "
                "does not expose a hookable base model output, pass output_hidden_states=True "
                "explicitly for debugging or add a model-specific final-hidden capture path."
            )
        raise NotImplementedError(
            f"{aux_name}_head_source={source!r} is not wired safely in this training path yet; "
            f"use {aux_name}_head_source='llm_output'."
        )

    def _select_bev_hidden_states(self, outputs, captured_final_hidden=None):
        source = str(getattr(self.config, "bev_head_source", "llm_output") or "llm_output")
        return self._select_aux_hidden_states(
            outputs,
            captured_final_hidden=captured_final_hidden,
            source=source,
            aux_name="bev",
        )

    def _gather_aux_visual_hidden(self, sequence_hidden_states, visual_metadata, *, aux_name="aux"):
        metadata_items = self._metadata_items(visual_metadata)
        if len(metadata_items) != int(sequence_hidden_states.shape[0]):
            raise RuntimeError(
                f"{aux_name} metadata batch size mismatch: "
                f"hidden batch={int(sequence_hidden_states.shape[0])}, metadata batch={len(metadata_items)}."
            )

        batch_size, seq_len, hidden_dim = sequence_hidden_states.shape
        lengths = []
        for batch_idx, metadata in enumerate(metadata_items):
            indices = metadata.get("visual_token_indices") if isinstance(metadata, dict) else None
            if not isinstance(indices, torch.Tensor):
                raise RuntimeError(f"{aux_name} metadata[{batch_idx}] is missing tensor visual_token_indices.")
            lengths.append(int(indices.numel()))

        max_tokens = max(lengths) if lengths else 0
        gathered = sequence_hidden_states.new_zeros(batch_size, max_tokens, hidden_dim)
        for batch_idx, (metadata, token_count) in enumerate(zip(metadata_items, lengths)):
            if token_count == 0:
                continue
            indices = metadata["visual_token_indices"].to(device=sequence_hidden_states.device, dtype=torch.long)
            if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(seq_len):
                raise RuntimeError(
                    f"{aux_name} visual_token_indices for sample {batch_idx} are outside the LLM sequence: "
                    f"min={int(indices.min().item())}, max={int(indices.max().item())}, seq_len={int(seq_len)}."
                )
            gathered[batch_idx, :token_count] = sequence_hidden_states[batch_idx].index_select(0, indices)
        return gathered

    def _gather_bev_visual_hidden(self, sequence_hidden_states, visual_metadata):
        return self._gather_aux_visual_hidden(sequence_hidden_states, visual_metadata, aux_name="BEV")

    @staticmethod
    def _point_map_payload_available(candidate, point_map_keys):
        if candidate is None:
            return False
        if isinstance(candidate, dict):
            return any(candidate.get(key) is not None for key in point_map_keys)
        if isinstance(candidate, (list, tuple)):
            return len(candidate) > 0 and any(LlavaQwenForCausalLM._point_map_payload_available(item, point_map_keys) for item in candidate)
        return isinstance(candidate, torch.Tensor)

    @staticmethod
    def _bev_payload_available(candidate):
        point_map_keys = (
            "point_maps_ref",
            "pts3d_in_other_view",
            "point_maps_cam",
            "pts3d_in_self_view",
            "point_maps",
            "point_map",
            "points",
            "pts3d",
        )
        return LlavaQwenForCausalLM._point_map_payload_available(candidate, point_map_keys)

    @staticmethod
    def _depth_payload_available(candidate, *, allow_generic=False, allow_tensor=False):
        if candidate is None:
            return False
        if isinstance(candidate, torch.Tensor):
            return bool(allow_tensor)
        point_map_keys = (
            "point_maps_cam",
            "pts3d_in_self_view",
        )
        if allow_generic:
            point_map_keys = point_map_keys + (
                "point_maps",
                "point_map",
                "points",
                "pts3d",
            )
        if isinstance(candidate, dict):
            return any(candidate.get(key) is not None for key in point_map_keys)
        if isinstance(candidate, (list, tuple)):
            return len(candidate) > 0 and any(
                LlavaQwenForCausalLM._depth_payload_available(
                    item,
                    allow_generic=allow_generic,
                    allow_tensor=allow_tensor,
                )
                for item in candidate
            )
        return False

    def _select_bev_point_map_payloads(self, spatial_features, point_maps, geometry_spatial_features):
        for candidate in (spatial_features, point_maps, geometry_spatial_features):
            if self._bev_payload_available(candidate):
                return candidate
        return None

    def _select_depth_point_map_payloads(self, spatial_features, point_maps, geometry_spatial_features):
        allow_generic = self._as_bool_config(getattr(self.config, "depth_allow_generic_camera_assumed", False), False)
        allow_tensor = self._as_bool_config(getattr(self.config, "depth_allow_tensor_camera_assumed", False), False)
        for candidate in (geometry_spatial_features, point_maps, spatial_features):
            if self._depth_payload_available(candidate, allow_generic=allow_generic, allow_tensor=allow_tensor):
                return candidate
        return None

    def _shuffle_depth_targets(self, depth_gt_log, depth_valid_mask, visual_metadata):
        if not self._as_bool_config(getattr(self.config, "depth_shuffle_target", False), False):
            return depth_gt_log, depth_valid_mask, False, str(getattr(self.config, "depth_shuffle_mode", "none") or "none")

        mode = str(getattr(self.config, "depth_shuffle_mode", "frame_shuffle") or "frame_shuffle")
        mode = mode.strip().lower()
        if mode in {"batch", "batch_shuffle"}:
            if depth_gt_log.shape[0] <= 1:
                return depth_gt_log, depth_valid_mask, False, "batch_shuffle"
            perm = torch.randperm(depth_gt_log.shape[0], device=depth_gt_log.device)
            if torch.equal(perm, torch.arange(depth_gt_log.shape[0], device=depth_gt_log.device)):
                perm = torch.roll(perm, shifts=1, dims=0)
            return depth_gt_log.index_select(0, perm), depth_valid_mask.index_select(0, perm), True, "batch_shuffle"

        metadata_items = self._metadata_items(visual_metadata)
        shuffled_gt = depth_gt_log.clone()
        shuffled_mask = depth_valid_mask.clone()
        applied = False

        if mode in {"intra_sample_token_shuffle", "token_shuffle"}:
            for batch_idx, metadata in enumerate(metadata_items):
                indices = metadata.get("visual_token_indices") if isinstance(metadata, dict) else None
                token_count = int(indices.numel()) if isinstance(indices, torch.Tensor) else 0
                token_count = min(token_count, int(depth_gt_log.shape[1]))
                if token_count <= 1:
                    continue
                perm = torch.randperm(token_count, device=depth_gt_log.device)
                if torch.equal(perm, torch.arange(token_count, device=depth_gt_log.device)):
                    perm = torch.roll(perm, shifts=1, dims=0)
                shuffled_gt[batch_idx, :token_count] = depth_gt_log[batch_idx, :token_count].index_select(0, perm)
                shuffled_mask[batch_idx, :token_count] = depth_valid_mask[batch_idx, :token_count].index_select(0, perm)
                applied = True
            return shuffled_gt, shuffled_mask, applied, "intra_sample_token_shuffle"

        if mode in {"frame_shuffle", "intra_sample_frame_shuffle"}:
            for batch_idx, metadata in enumerate(metadata_items):
                frame_ids = metadata.get("visual_frame_ids") if isinstance(metadata, dict) else None
                if not isinstance(frame_ids, torch.Tensor) or frame_ids.numel() <= 1:
                    continue
                frame_ids_cpu = frame_ids.detach().cpu().to(dtype=torch.long)
                unique_frames = list(dict.fromkeys(int(x) for x in frame_ids_cpu.tolist()))
                if len(unique_frames) <= 1:
                    continue
                perm = torch.randperm(len(unique_frames), device=depth_gt_log.device)
                if torch.equal(perm, torch.arange(len(unique_frames), device=depth_gt_log.device)):
                    perm = torch.roll(perm, shifts=1, dims=0)
                source_frames = [unique_frames[int(idx)] for idx in perm.detach().cpu().tolist()]
                frame_ids_device = frame_ids.to(device=depth_gt_log.device, dtype=torch.long)
                for dst_frame, src_frame in zip(unique_frames, source_frames):
                    dst_pos = torch.nonzero(frame_ids_device == int(dst_frame), as_tuple=False).flatten()
                    src_pos = torch.nonzero(frame_ids_device == int(src_frame), as_tuple=False).flatten()
                    dst_pos = dst_pos[dst_pos < depth_gt_log.shape[1]]
                    src_pos = src_pos[src_pos < depth_gt_log.shape[1]]
                    if dst_pos.numel() != src_pos.numel():
                        raise RuntimeError(
                            "depth_shuffle_mode='frame_shuffle' requires equal visual-token counts per frame, "
                            f"got frame {dst_frame}: {dst_pos.numel()} and frame {src_frame}: {src_pos.numel()}."
                        )
                    if dst_pos.numel() == 0:
                        continue
                    shuffled_gt[batch_idx, dst_pos] = depth_gt_log[batch_idx, src_pos]
                    shuffled_mask[batch_idx, dst_pos] = depth_valid_mask[batch_idx, src_pos]
                    applied = True
            return shuffled_gt, shuffled_mask, applied, "frame_shuffle"

        raise ValueError(
            "Unsupported depth_shuffle_mode. Expected batch_shuffle, "
            f"intra_sample_token_shuffle, or frame_shuffle; got {mode!r}."
        )

    def _compute_bev_supervision_loss(
        self,
        outputs,
        visual_metadata,
        spatial_features,
        point_maps,
        geometry_spatial_features,
        ce_loss,
        final_sequence_hidden=None,
    ):
        sequence_hidden = self._select_bev_hidden_states(outputs, captured_final_hidden=final_sequence_hidden)
        visual_hidden = self._gather_bev_visual_hidden(sequence_hidden, visual_metadata)
        payloads = self._select_bev_point_map_payloads(spatial_features, point_maps, geometry_spatial_features)
        if payloads is None:
            raise RuntimeError(
                "use_bev_supervision=True requires CUT3R point-map sidecars in spatial_features, "
                "point_maps, or geometry_spatial_features. Expected keys such as "
                "point_maps_ref/point_maps_cam."
            )

        bev_gt_meter, bev_valid_mask, bev_debug = build_bev_targets_from_point_maps(
            payloads,
            visual_metadata,
            bev_point_map_key=str(getattr(self.config, "bev_point_map_key", "point_maps_ref")),
            use_geometry_confidence_mask=self._as_bool_config(
                getattr(self.config, "use_geometry_confidence_mask", True),
                True,
            ),
            bev_conf_threshold=float(getattr(self.config, "bev_conf_threshold", 0.0)),
        )
        bev_gt_meter = bev_gt_meter.to(device=visual_hidden.device, dtype=visual_hidden.dtype)
        bev_valid_mask = bev_valid_mask.to(device=visual_hidden.device, dtype=torch.bool)

        if visual_hidden.shape[:2] != bev_gt_meter.shape[:2] or bev_gt_meter.shape[:2] != bev_valid_mask.shape[:2]:
            raise RuntimeError(
                "BEV visual-token alignment mismatch. "
                f"visual_hidden[:2]={tuple(visual_hidden.shape[:2])}, "
                f"bev_gt[:2]={tuple(bev_gt_meter.shape[:2])}, "
                f"bev_valid_mask[:2]={tuple(bev_valid_mask.shape[:2])}. "
                "Likely causes: visual_grid_shapes differ from CUT3R patch pooling, "
                "visual_token_indices include non-visual tokens, or frame order differs."
            )

        shuffle_applied = False
        if self._as_bool_config(getattr(self.config, "bev_shuffle_target", False), False) and bev_gt_meter.shape[0] > 1:
            perm = torch.randperm(bev_gt_meter.shape[0], device=bev_gt_meter.device)
            bev_gt_meter = bev_gt_meter.index_select(0, perm)
            bev_valid_mask = bev_valid_mask.index_select(0, perm)
            shuffle_applied = True

        bev_head = getattr(self, "bev_head", None)
        if bev_head is None:
            bev_head = self.initialize_bev_head(device=visual_hidden.device, dtype=visual_hidden.dtype)

        bev_input = visual_hidden.detach() if self._as_bool_config(getattr(self.config, "bev_detach_hidden", False), False) else visual_hidden
        bev_pred_norm = bev_head(bev_input)
        coord_scale = float(getattr(self.config, "bev_coord_scale", 10.0))
        if coord_scale <= 0:
            raise ValueError(f"bev_coord_scale must be positive, got {coord_scale}")
        bev_gt_norm = bev_gt_meter / coord_scale

        finite_mask = torch.isfinite(bev_gt_norm).all(dim=-1) & torch.isfinite(bev_pred_norm).all(dim=-1)
        valid_mask = bev_valid_mask & finite_mask
        num_valid = int(valid_mask.detach().sum().item())
        num_total = self._total_visual_tokens_from_metadata(visual_metadata)
        total_for_ratio = max(int(num_total), 1)

        if num_valid == 0:
            loss_bev = ce_loss.new_zeros(())
            bev_mae_meter = ce_loss.new_zeros(())
        else:
            loss_bev = F.smooth_l1_loss(bev_pred_norm[valid_mask].float(), bev_gt_norm[valid_mask].float())
            bev_pred_meter = bev_pred_norm * coord_scale
            bev_mae_meter = (bev_pred_meter[valid_mask].float() - bev_gt_meter[valid_mask].float()).abs().mean()

        metrics = {
            "loss_ce": float(ce_loss.detach().float().item()),
            "loss_bev": float(loss_bev.detach().float().item()),
            "lambda_bev_times_loss_bev": float((loss_bev.detach().float() * float(getattr(self.config, "lambda_bev", 0.05))).item()),
            "bev_mae_meter": float(bev_mae_meter.detach().float().item()),
            "valid_bev_token_ratio": float(num_valid / total_for_ratio),
            "num_valid_bev_tokens": float(num_valid),
            "num_total_bev_tokens": float(num_total),
            "bev_point_map_key": str(getattr(self.config, "bev_point_map_key", "point_maps_ref")),
            "bev_head_source": str(getattr(self.config, "bev_head_source", "llm_output")),
            "bev_detach_hidden": float(self._as_bool_config(getattr(self.config, "bev_detach_hidden", False), False)),
            "bev_shuffle_target": float(self._as_bool_config(getattr(self.config, "bev_shuffle_target", False), False)),
            "bev_shuffle_applied": float(shuffle_applied),
        }
        if isinstance(bev_debug, dict):
            metrics["bev_debug_valid_ratio_from_builder"] = float(bev_debug.get("valid_bev_token_ratio", 0.0) or 0.0)
        return loss_bev, metrics

    def _compute_depth_supervision_loss(
        self,
        outputs,
        visual_metadata,
        spatial_features,
        point_maps,
        geometry_spatial_features,
        ce_loss,
        final_sequence_hidden=None,
    ):
        source = str(getattr(self.config, "depth_head_source", "llm_output") or "llm_output")
        sequence_hidden = self._select_aux_hidden_states(
            outputs,
            captured_final_hidden=final_sequence_hidden,
            source=source,
            aux_name="depth",
        )
        visual_hidden = self._gather_aux_visual_hidden(sequence_hidden, visual_metadata, aux_name="Depth")
        payloads = self._select_depth_point_map_payloads(spatial_features, point_maps, geometry_spatial_features)
        if payloads is None:
            raise RuntimeError(
                "use_depth_supervision=True requires CUT3R camera-space point-map sidecars in "
                "geometry_spatial_features, point_maps, or spatial_features if it contains point maps. "
                "Expected point_maps_cam or pts3d_in_self_view; point_maps_ref z is not camera depth."
            )

        depth_gt_log, depth_valid_mask, depth_debug = build_depth_targets_from_point_maps(
            payloads,
            visual_metadata,
            depth_point_map_key=str(getattr(self.config, "depth_point_map_key", "point_maps_cam")),
            use_geometry_confidence_mask=self._as_bool_config(
                getattr(self.config, "use_geometry_confidence_mask", True),
                True,
            ),
            depth_conf_threshold=float(getattr(self.config, "depth_conf_threshold", 0.0)),
            depth_max_gt=float(getattr(self.config, "depth_max_gt", 20.0)),
            depth_allow_generic_camera_assumed=self._as_bool_config(
                getattr(self.config, "depth_allow_generic_camera_assumed", False),
                False,
            ),
            depth_allow_tensor_camera_assumed=self._as_bool_config(
                getattr(self.config, "depth_allow_tensor_camera_assumed", False),
                False,
            ),
        )
        depth_gt_log = depth_gt_log.to(device=visual_hidden.device, dtype=visual_hidden.dtype)
        depth_valid_mask = depth_valid_mask.to(device=visual_hidden.device, dtype=torch.bool)

        if visual_hidden.shape[:2] != depth_gt_log.shape[:2] or depth_gt_log.shape[:2] != depth_valid_mask.shape[:2]:
            raise RuntimeError(
                "Depth visual-token alignment mismatch. "
                f"visual_hidden[:2]={tuple(visual_hidden.shape[:2])}, "
                f"depth_gt_log[:2]={tuple(depth_gt_log.shape[:2])}, "
                f"depth_valid_mask[:2]={tuple(depth_valid_mask.shape[:2])}. "
                "Likely causes: visual_grid_shapes differ from CUT3R patch pooling, "
                "visual_token_indices include non-visual tokens, or frame order differs."
            )

        depth_gt_log, depth_valid_mask, shuffle_applied, shuffle_mode_used = self._shuffle_depth_targets(
            depth_gt_log,
            depth_valid_mask,
            visual_metadata,
        )

        depth_head = getattr(self, "depth_head", None)
        if depth_head is None:
            depth_head = self.initialize_depth_head(device=visual_hidden.device, dtype=visual_hidden.dtype)

        depth_input = (
            visual_hidden.detach()
            if self._as_bool_config(getattr(self.config, "depth_detach_hidden", False), False)
            else visual_hidden
        )
        depth_pred_log = depth_head(depth_input)

        finite_mask = torch.isfinite(depth_gt_log) & torch.isfinite(depth_pred_log)
        valid_mask = depth_valid_mask & finite_mask
        num_valid = int(valid_mask.detach().sum().item())
        num_total = self._total_visual_tokens_from_metadata(visual_metadata)
        total_for_ratio = max(int(num_total), 1)

        if num_valid == 0:
            loss_depth = ce_loss.new_zeros(())
            depth_mae_meter = ce_loss.new_zeros(())
        else:
            loss_depth = F.smooth_l1_loss(depth_pred_log[valid_mask].float(), depth_gt_log[valid_mask].float())
            depth_pred_meter = torch.expm1(depth_pred_log[valid_mask].float())
            depth_gt_meter = torch.expm1(depth_gt_log[valid_mask].float())
            depth_mae_meter = (depth_pred_meter - depth_gt_meter).abs().mean()

        metrics = {
            "loss_ce": float(ce_loss.detach().float().item()),
            "loss_depth": float(loss_depth.detach().float().item()),
            "lambda_depth_times_loss_depth": float((loss_depth.detach().float() * float(getattr(self.config, "lambda_depth", 0.05))).item()),
            "depth_mae_meter": float(depth_mae_meter.detach().float().item()),
            "valid_depth_token_ratio": float(num_valid / total_for_ratio),
            "num_valid_depth_tokens": float(num_valid),
            "num_total_depth_tokens": float(num_total),
            "depth_point_map_key": str(getattr(self.config, "depth_point_map_key", "point_maps_cam")),
            "depth_head_source": source,
            "depth_detach_hidden": float(self._as_bool_config(getattr(self.config, "depth_detach_hidden", False), False)),
            "depth_shuffle_target": float(self._as_bool_config(getattr(self.config, "depth_shuffle_target", False), False)),
            "depth_shuffle_mode": shuffle_mode_used,
            "depth_shuffle_applied": float(shuffle_applied),
            "depth_max_gt": float(getattr(self.config, "depth_max_gt", 20.0)),
            "depth_conf_threshold": float(getattr(self.config, "depth_conf_threshold", 0.0)),
            "depth_allow_generic_camera_assumed": float(self._as_bool_config(getattr(self.config, "depth_allow_generic_camera_assumed", False), False)),
            "depth_allow_tensor_camera_assumed": float(self._as_bool_config(getattr(self.config, "depth_allow_tensor_camera_assumed", False), False)),
        }
        if isinstance(depth_debug, dict):
            metrics["depth_debug_valid_ratio_from_builder"] = float(depth_debug.get("valid_depth_token_ratio", 0.0) or 0.0)
            metrics["depth_point_map_key_used"] = str(depth_debug.get("depth_point_map_key_used", ""))
            metrics["depth_target_space"] = str(depth_debug.get("depth_target_space", ""))
        return loss_depth, metrics

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        spatial_features: Optional[Dict[str, torch.FloatTensor]] = None,
        geometry_spatial_features: Optional[Dict[str, torch.FloatTensor]] = None,
        point_maps: Optional[torch.FloatTensor] = None,
        geometry_outputs: Optional[Dict[str, torch.FloatTensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        return_visual_metadata: Optional[bool] = False,
        llm_geo_pos: Optional[torch.FloatTensor] = None,
        llm_geo_mask: Optional[torch.BoolTensor] = None,
        llm_geo_debug: Optional[Dict] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        self.get_model()._last_geometry_projection_outputs = None
        self.get_model()._last_geometry_projection_metrics = None
        self._geometry_projection_last_metrics = {}
        self._bev_last_metrics = {}
        self._depth_last_metrics = {}
        metadata_requested = bool(return_visual_metadata)
        input_embeds_provided = inputs_embeds is not None
        spatial_rank_enabled = bool(
            self.training
            and getattr(self.config, "spatial_rank_loss_enable", False)
            and labels is not None
            and not input_embeds_provided
        )
        bev_loss_enabled = bool(
            self.training
            and not dpo_forward
            and labels is not None
            and not input_embeds_provided
            and self._as_bool_config(getattr(self.config, "use_bev_supervision", False), False)
        )
        depth_loss_enabled = bool(
            self.training
            and not dpo_forward
            and labels is not None
            and not input_embeds_provided
            and self._as_bool_config(getattr(self.config, "use_depth_supervision", False), False)
        )
        aux_loss_enabled = bev_loss_enabled or depth_loss_enabled
        original_output_hidden_states = output_hidden_states
        if spatial_rank_enabled:
            output_hidden_states = True
            return_dict = True
        elif aux_loss_enabled:
            return_dict = True
        elif metadata_requested:
            output_hidden_states = True
            return_dict = True

        visual_metadata = None
        llm_geo_debug_info = llm_geo_debug
        llm_visual_3d_rope_enabled = _as_bool_config(getattr(self.config, "llm_visual_3d_rope_enable", False), False)
        should_prepare_multimodal = (
            inputs_embeds is None
            and images is not None
            and input_ids is not None
            and input_ids.shape[1] != 1
        )
        if should_prepare_multimodal:
            prepared = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                spatial_features,
                point_maps,
                modalities,
                image_sizes,
                return_visual_metadata=spatial_rank_enabled or metadata_requested or aux_loss_enabled,
                return_llm_geo_metadata=llm_visual_3d_rope_enabled,
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
            visual_metadata_requested = spatial_rank_enabled or metadata_requested or aux_loss_enabled
            if visual_metadata_requested and llm_visual_3d_rope_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    visual_metadata,
                    llm_geo_pos,
                    llm_geo_mask,
                    llm_geo_debug_info,
                ) = prepared
            elif visual_metadata_requested:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_metadata) = prepared
            elif llm_visual_3d_rope_enabled:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    llm_geo_pos,
                    llm_geo_mask,
                    llm_geo_debug_info,
                ) = prepared
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = prepared
        elif llm_visual_3d_rope_enabled and llm_geo_debug_info is None:
            llm_geo_debug_info = {"skip_reason": "text_only_or_cached_decode"}

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                llm_geo_pos=llm_geo_pos,
                llm_geo_mask=llm_geo_mask,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            h1_holder = {}
            aux_final_hidden_holder = {}
            hook_handle = None
            aux_hook_handle = None
            if spatial_rank_enabled:
                first_block = self.model.layers[0]

                def capture_h1(_module, _inputs, output):
                    h1_holder["h1"] = output[0] if isinstance(output, (tuple, list)) else output

                hook_handle = first_block.register_forward_hook(capture_h1)
            aux_final_hidden_needed = (
                (bev_loss_enabled and str(getattr(self.config, "bev_head_source", "llm_output") or "llm_output") == "llm_output")
                or (depth_loss_enabled and str(getattr(self.config, "depth_head_source", "llm_output") or "llm_output") == "llm_output")
            )
            if aux_final_hidden_needed:
                def capture_final_hidden(_module, _inputs, output):
                    if isinstance(output, (tuple, list)):
                        hidden = output[0]
                    elif hasattr(output, "last_hidden_state"):
                        hidden = output.last_hidden_state
                    else:
                        hidden = output
                    aux_final_hidden_holder["hidden"] = hidden

                aux_hook_handle = self.model.register_forward_hook(capture_final_hidden)
            try:
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    llm_geo_pos=llm_geo_pos,
                    llm_geo_mask=llm_geo_mask,
                )
                if llm_visual_3d_rope_enabled:
                    current_stats = getattr(self.get_model(), "_last_llm_visual_3d_rope_stats", None)
                    if current_stats:
                        has_active_geo = any(
                            (not item.get("skipped", False)) and int(item.get("num_valid_geo_tokens", 0) or 0) > 0
                            for item in current_stats
                        )
                        if has_active_geo:
                            self._last_llm_visual_3d_rope_prefill_stats = current_stats
                            self._last_llm_geo_prefill_debug = llm_geo_debug_info
                        else:
                            self._last_llm_visual_3d_rope_decode_stats = current_stats
                            self._last_llm_geo_decode_debug = llm_geo_debug_info
                hidden_states = model_outputs[0]
                logits = self.lm_head(hidden_states)
                logits = logits.float()

                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                if not return_dict:
                    output = (logits,) + model_outputs[1:]
                    outputs = (loss,) + output if loss is not None else output
                else:
                    outputs = CausalLMOutputWithPast(
                        loss=loss,
                        logits=logits,
                        past_key_values=model_outputs.past_key_values,
                        hidden_states=model_outputs.hidden_states,
                        attentions=model_outputs.attentions,
                    )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
                if aux_hook_handle is not None:
                    aux_hook_handle.remove()
            if metadata_requested:
                self._last_visual_metadata = visual_metadata
            if llm_visual_3d_rope_enabled:
                self._last_llm_geo_debug = llm_geo_debug_info
            geometry_projection_outputs = getattr(self.get_model(), "_last_geometry_projection_outputs", None)
            use_geometry_projection = getattr(self.config, "use_geometry_aware_projection", False)
            if isinstance(use_geometry_projection, str):
                use_geometry_projection = use_geometry_projection.lower() in {"1", "true", "yes", "y", "on"}
            use_auxiliary_geometry_head = getattr(self.config, "use_auxiliary_geometry_head", True)
            if isinstance(use_auxiliary_geometry_head, str):
                use_auxiliary_geometry_head = use_auxiliary_geometry_head.lower() in {"1", "true", "yes", "y", "on"}
            use_auxiliary_geometry_loss = getattr(self.config, "use_auxiliary_geometry_loss", True)
            if isinstance(use_auxiliary_geometry_loss, str):
                use_auxiliary_geometry_loss = use_auxiliary_geometry_loss.lower() in {"1", "true", "yes", "y", "on"}
            geometry_loss_enabled = bool(
                self.training
                and labels is not None
                and geometry_projection_outputs is not None
                and geometry_projection_outputs.get("loss_geo") is not None
                and use_geometry_projection
                and use_auxiliary_geometry_head
                and use_auxiliary_geometry_loss
            )
            if not spatial_rank_enabled and not geometry_loss_enabled and not aux_loss_enabled:
                return outputs

            ce_loss = outputs.loss
            if ce_loss is None:
                raise RuntimeError("Auxiliary BEV/depth/spatial/geometry losses require labels so CE loss is available.")

            total_loss = ce_loss
            if geometry_loss_enabled:
                loss_geo = geometry_projection_outputs["loss_geo"]
                lambda_geo = float(getattr(self.config, "lambda_geo", 0.1))
                total_loss = total_loss + lambda_geo * loss_geo
                self._geometry_projection_last_metrics = {
                    "geometry_loss_lm": float(ce_loss.detach().float().item()),
                    "geometry_loss_geo": float(loss_geo.detach().float().item()),
                    "geometry_loss_total": float(total_loss.detach().float().item()),
                    "lambda_geo": lambda_geo,
                }

            if bev_loss_enabled:
                loss_bev, bev_metrics = self._compute_bev_supervision_loss(
                    outputs,
                    visual_metadata,
                    spatial_features,
                    point_maps,
                    geometry_spatial_features,
                    ce_loss,
                    final_sequence_hidden=aux_final_hidden_holder.get("hidden"),
                )
                lambda_bev = float(getattr(self.config, "lambda_bev", 0.05))
                total_loss = total_loss + lambda_bev * loss_bev
                bev_metrics["lambda_bev_times_loss_bev"] = float((loss_bev.detach().float() * lambda_bev).item())
                bev_metrics["loss_total"] = float(total_loss.detach().float().item())
                bev_metrics["lambda_bev"] = lambda_bev
                self._bev_last_metrics = bev_metrics

            if depth_loss_enabled:
                loss_depth, depth_metrics = self._compute_depth_supervision_loss(
                    outputs,
                    visual_metadata,
                    spatial_features,
                    point_maps,
                    geometry_spatial_features,
                    ce_loss,
                    final_sequence_hidden=aux_final_hidden_holder.get("hidden"),
                )
                lambda_depth = float(getattr(self.config, "lambda_depth", 0.05))
                total_loss = total_loss + lambda_depth * loss_depth
                depth_metrics["lambda_depth_times_loss_depth"] = float((loss_depth.detach().float() * lambda_depth).item())
                depth_metrics["loss_total"] = float(total_loss.detach().float().item())
                depth_metrics["lambda_depth"] = lambda_depth
                self._depth_last_metrics = depth_metrics
                if bev_loss_enabled and self._bev_last_metrics:
                    self._bev_last_metrics["loss_total"] = float(total_loss.detach().float().item())

            if not spatial_rank_enabled:
                if geometry_loss_enabled:
                    self._geometry_projection_last_metrics["geometry_loss_total"] = float(total_loss.detach().float().item())
                return CausalLMOutputWithPast(
                    loss=total_loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states if original_output_hidden_states else None,
                    attentions=outputs.attentions,
                )

            h1 = h1_holder.get("h1", None)
            if h1 is None:
                raise RuntimeError("Spatial ranking loss could not capture H1 from self.model.layers[0].")
            rank_loss, rank_metrics = self.compute_spatial_ranking_loss(
                h1,
                visual_metadata,
                spatial_features,
                debug_checks=bool(getattr(self.config, "spatial_rank_debug_checks", False)),
            )
            lambda_sim = float(getattr(self.config, "lambda_sim", 0.01))
            total_loss = total_loss + lambda_sim * rank_loss
            self._spatial_rank_last_metrics = dict(rank_metrics)
            self._spatial_rank_last_metrics.update({
                "spatial_rank_ce_loss": float(ce_loss.detach().float().item()),
                "spatial_rank_total_loss": float(total_loss.detach().float().item()),
                "spatial_rank_lambda": lambda_sim,
            })
            if geometry_loss_enabled:
                self._spatial_rank_last_metrics.update({
                    "geometry_loss_geo": float(loss_geo.detach().float().item()),
                    "geometry_loss_weighted": float((loss_geo.detach().float() * lambda_geo).item()),
                    "lambda_geo": lambda_geo,
                })
            if bev_loss_enabled and self._bev_last_metrics:
                self._bev_last_metrics["loss_total"] = float(total_loss.detach().float().item())
                self._spatial_rank_last_metrics.update({
                    "bev_loss_bev": self._bev_last_metrics.get("loss_bev", 0.0),
                    "bev_loss_weighted": self._bev_last_metrics.get("lambda_bev_times_loss_bev", 0.0),
                    "lambda_bev": self._bev_last_metrics.get("lambda_bev", 0.0),
                })
            if depth_loss_enabled and self._depth_last_metrics:
                self._depth_last_metrics["loss_total"] = float(total_loss.detach().float().item())
                self._spatial_rank_last_metrics.update({
                    "depth_loss_depth": self._depth_last_metrics.get("loss_depth", 0.0),
                    "depth_loss_weighted": self._depth_last_metrics.get("lambda_depth_times_loss_depth", 0.0),
                    "lambda_depth": self._depth_last_metrics.get("lambda_depth", 0.0),
                })

            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states if original_output_hidden_states else None,
                attentions=outputs.attentions,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
        geometry_spatial_features: Optional[torch.Tensor] = None,
        point_maps: Optional[torch.Tensor] = None,
        geometry_outputs: Optional[Dict[str, torch.Tensor]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        llm_visual_3d_rope_enabled = _as_bool_config(getattr(self.config, "llm_visual_3d_rope_enable", False), False)
        if llm_visual_3d_rope_enabled:
            for attr in (
                "_last_llm_visual_3d_rope_prefill_stats",
                "_last_llm_geo_prefill_debug",
                "_last_llm_visual_3d_rope_decode_stats",
                "_last_llm_geo_decode_debug",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
        llm_geo_pos = None
        llm_geo_mask = None
        llm_geo_debug = None
        if images is not None:
            prepared = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                spatial_features,
                point_maps,
                modalities,
                image_sizes=image_sizes,
                return_llm_geo_metadata=llm_visual_3d_rope_enabled,
                geometry_outputs=geometry_outputs,
                geometry_spatial_features=geometry_spatial_features,
            )
            if llm_visual_3d_rope_enabled:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _, llm_geo_pos, llm_geo_mask, llm_geo_debug) = prepared
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = prepared
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        try:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                llm_geo_pos=llm_geo_pos,
                llm_geo_mask=llm_geo_mask,
                llm_geo_debug=llm_geo_debug,
                **kwargs,
            )
        finally:
            clear_qwen2_visual_3d_rope_context(self.model)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        geometry_outputs = kwargs.pop("geometry_outputs", None)
        geometry_spatial_features = kwargs.pop("geometry_spatial_features", None)
        llm_geo_pos = kwargs.pop("llm_geo_pos", None)
        llm_geo_mask = kwargs.pop("llm_geo_mask", None)
        llm_geo_debug = kwargs.pop("llm_geo_debug", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if geometry_outputs is not None:
            inputs["geometry_outputs"] = geometry_outputs
        if geometry_spatial_features is not None:
            inputs["geometry_spatial_features"] = geometry_spatial_features
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if llm_geo_pos is not None:
            inputs["llm_geo_pos"] = llm_geo_pos
        if llm_geo_mask is not None:
            inputs["llm_geo_mask"] = llm_geo_mask
        if llm_geo_debug is not None:
            inputs["llm_geo_debug"] = llm_geo_debug
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
