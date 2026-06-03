import math
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.qwen2.modeling_qwen2 import (
    QWEN2_ATTENTION_CLASSES,
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from llava.model.multimodal_fusion_block.builder import GeoRoPEFusionRotary


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_layer_set(value, num_layers):
    if value is None or str(value).strip().lower() == "all":
        return set(range(int(num_layers)))
    layers = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return {layer for layer in layers if 0 <= layer < int(num_layers)}


def _log_layer_set(config):
    value = getattr(config, "llm_visual_3d_rope_log_layers", "first_middle_last")
    value = "first_middle_last" if value in (None, "") else str(value).strip().lower()
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if value == "all":
        return set(range(num_layers))
    if value == "first_middle_last":
        if num_layers <= 0:
            return set()
        return {0, num_layers // 2, num_layers - 1}
    return _parse_layer_set(value, num_layers)


def _jsonable_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().float().item())
        return None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return value


class Qwen2Visual3DRopeAttention(Qwen2Attention):
    """Qwen2 eager attention with optional visual-token-only 3D RoPE logit replacement."""

    def _get_geo_rope(self):
        mode = str(getattr(self.config, "llm_visual_3d_rope_mode", "spherical"))
        group_split = getattr(self.config, "llm_visual_3d_rope_group_split", "2,1,2")
        group_split = str(group_split).replace("|", ",").replace(";", ",")
        max_depth = float(getattr(self.config, "llm_visual_3d_rope_max_depth", 10.0))
        cache_key = (self.head_dim, mode, str(group_split), max_depth)
        if getattr(self, "_llm_visual_3d_rope_cache_key", None) != cache_key:
            self._llm_visual_3d_rope = GeoRoPEFusionRotary(
                head_dim=self.head_dim,
                mode=mode,
                max_depth=max_depth,
                group_split=group_split,
            )
            self._llm_visual_3d_rope_cache_key = cache_key
        return self._llm_visual_3d_rope

    def _layer_enabled(self):
        if not _as_bool(getattr(self.config, "llm_visual_3d_rope_enable", False)):
            return False
        num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        return int(self.layer_idx or 0) in _parse_layer_set(
            getattr(self.config, "llm_visual_3d_rope_layers", "all"),
            num_layers,
        )

    def _stats_enabled(self):
        if not _as_bool(getattr(self.config, "llm_visual_3d_rope_log_stats", True), True):
            return False
        return int(self.layer_idx or 0) in _log_layer_set(self.config)

    def _empty_stats(self, reason):
        if self._stats_enabled():
            self.last_llm_visual_3d_rope_stats = {
                "layer_idx": int(self.layer_idx or 0),
                "enabled": bool(_as_bool(getattr(self.config, "llm_visual_3d_rope_enable", False))),
                "skipped": True,
                "skip_reason": reason,
                "alpha": float(getattr(self.config, "llm_visual_3d_rope_alpha", 1.0)),
            }

    @staticmethod
    def _mean(x):
        if x.numel() == 0:
            return 0.0
        return float(x.detach().float().mean().item())

    @staticmethod
    def _mean_abs(x):
        if x.numel() == 0:
            return 0.0
        return float(x.detach().float().abs().mean().item())

    @staticmethod
    def _mean_finite(x):
        if x.numel() == 0:
            return 0.0
        x = x.detach().float()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return 0.0
        return float(x.mean().item())

    @staticmethod
    def _mean_abs_finite(x):
        if x.numel() == 0:
            return 0.0
        x = x.detach().float()
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return 0.0
        return float(x.abs().mean().item())

    def _maybe_replace_visual_visual_logits(
        self,
        attn_weights,
        query_states_raw,
        key_states_raw,
        llm_geo_pos,
        llm_geo_mask,
        kv_seq_len,
    ):
        if not self._layer_enabled():
            self._empty_stats("disabled_or_layer_not_selected")
            return attn_weights

        alpha = float(getattr(self.config, "llm_visual_3d_rope_alpha", 1.0))
        if alpha == 0.0:
            self._empty_stats("alpha_zero")
            return attn_weights

        if llm_geo_pos is None or llm_geo_mask is None:
            self._empty_stats("missing_geometry_metadata")
            return attn_weights

        bsz, num_heads, q_len, _ = query_states_raw.shape
        if llm_geo_mask.dim() != 2 or llm_geo_pos.dim() != 3:
            self._empty_stats("bad_geometry_rank")
            return attn_weights

        full_seq_len = int(llm_geo_mask.shape[1])
        if q_len != full_seq_len or int(kv_seq_len) != full_seq_len:
            # Generation decode steps have q_len=1 and cached text queries; they are safely skipped.
            self._empty_stats("non_prefill_or_cached_decode")
            return attn_weights

        if query_states_raw.shape[-2] != full_seq_len or key_states_raw.shape[-2] != full_seq_len:
            self._empty_stats("raw_qk_not_full_sequence")
            return attn_weights

        llm_geo_pos = llm_geo_pos.to(device=query_states_raw.device)
        llm_geo_mask = llm_geo_mask.to(device=query_states_raw.device, dtype=torch.bool)
        key_states_raw = repeat_kv(key_states_raw, self.num_key_value_groups)
        if key_states_raw.shape[1] != num_heads:
            self._empty_stats("gqa_repeat_failed")
            return attn_weights

        rope = self._get_geo_rope()
        valid_counts = []
        first_indices = None
        delta_abs_sum = 0.0
        delta_count = 0
        one_d_sum = 0.0
        one_d_count = 0
        three_d_sum = 0.0
        three_d_count = 0
        chunk_size = int(getattr(self.config, "llm_visual_3d_rope_chunk_size", 256) or 256)
        chunk_size = max(chunk_size, 1)

        for batch_idx in range(bsz):
            idx = torch.nonzero(llm_geo_mask[batch_idx], as_tuple=False).flatten()
            if idx.numel() == 0:
                valid_counts.append(0)
                continue
            valid_counts.append(int(idx.numel()))
            if first_indices is None:
                first_indices = idx[:16].detach().cpu().tolist()

            q_vis = query_states_raw[batch_idx : batch_idx + 1, :, idx, :]
            k_vis = key_states_raw[batch_idx : batch_idx + 1, :, idx, :]
            pos_vis = llm_geo_pos[batch_idx : batch_idx + 1, idx, :]
            q_vis_3d = rope(q_vis, pos_vis)
            k_vis_3d = rope(k_vis, pos_vis)
            k_vis_3d_t = k_vis_3d.float().transpose(2, 3)

            for start in range(0, int(idx.numel()), chunk_size):
                end = min(start + chunk_size, int(idx.numel()))
                row_idx = idx[start:end]
                q_chunk = q_vis_3d[:, :, start:end, :].float()
                logits_3d = torch.matmul(q_chunk, k_vis_3d_t) / math.sqrt(self.head_dim)
                logits_3d = logits_3d.squeeze(0)

                logits_1d = attn_weights[batch_idx, :, row_idx[:, None], idx[None, :]]
                if alpha == 1.0:
                    updated = logits_3d
                else:
                    updated = (1.0 - alpha) * logits_1d.float() + alpha * logits_3d
                if attn_weights.dtype in (torch.float16, torch.bfloat16):
                    dtype_info = torch.finfo(attn_weights.dtype)
                    updated = torch.nan_to_num(
                        updated,
                        nan=0.0,
                        posinf=float(dtype_info.max),
                        neginf=float(dtype_info.min),
                    ).clamp(min=float(dtype_info.min), max=float(dtype_info.max))
                attn_weights[batch_idx, :, row_idx[:, None], idx[None, :]] = updated.to(attn_weights.dtype)

                if self._stats_enabled():
                    one_d_float = logits_1d.detach().float()
                    three_d_float = logits_3d.detach().float()
                    delta = three_d_float - one_d_float

                    finite_delta = torch.isfinite(delta)
                    if finite_delta.any():
                        delta_abs_sum += float(delta[finite_delta].abs().sum().item())
                        delta_count += int(finite_delta.sum().item())

                    finite_one = torch.isfinite(one_d_float)
                    if finite_one.any():
                        one_d_sum += float(one_d_float[finite_one].sum().item())
                        one_d_count += int(finite_one.sum().item())

                    finite_three = torch.isfinite(three_d_float)
                    if finite_three.any():
                        three_d_sum += float(three_d_float[finite_three].sum().item())
                        three_d_count += int(finite_three.sum().item())

        if self._stats_enabled():
            delta_mean_abs = float(delta_abs_sum / delta_count) if delta_count > 0 else 0.0
            one_d_mean = float(one_d_sum / one_d_count) if one_d_count > 0 else 0.0
            three_d_mean = float(three_d_sum / three_d_count) if three_d_count > 0 else 0.0
            pos_valid = llm_geo_pos[llm_geo_mask]
            self.last_llm_visual_3d_rope_stats = {
                "layer_idx": int(self.layer_idx or 0),
                "enabled": True,
                "skipped": False,
                "alpha": alpha,
                "mode": str(getattr(self.config, "llm_visual_3d_rope_mode", "spherical")),
                "group_split": str(getattr(self.config, "llm_visual_3d_rope_group_split", "2,1,2")),
                "max_depth": float(getattr(self.config, "llm_visual_3d_rope_max_depth", 10.0)),
                "num_total_tokens": int(full_seq_len),
                "num_valid_geo_tokens": int(sum(valid_counts)),
                "valid_geo_tokens_per_sample": valid_counts,
                "first_visual_token_indices": first_indices or [],
                "visual_visual_logits_1d_mean": one_d_mean,
                "visual_visual_logits_3d_mean": three_d_mean,
                "visual_visual_logits_delta_mean_abs": delta_mean_abs,
                "attention_delta_mean_abs": delta_mean_abs,
                "geo_pos_mean": self._mean(pos_valid) if pos_valid.numel() else 0.0,
                "geo_pos_std": float(pos_valid.detach().float().std(unbiased=False).item()) if pos_valid.numel() else 0.0,
                "geo_pos_min": float(pos_valid.detach().float().min().item()) if pos_valid.numel() else 0.0,
                "geo_pos_max": float(pos_valid.detach().float().max().item()) if pos_valid.numel() else 0.0,
                "geo_pos_nan_count": int(torch.isnan(pos_valid).sum().item()) if pos_valid.numel() else 0,
                "geo_pos_inf_count": int(torch.isinf(pos_valid).sum().item()) if pos_valid.numel() else 0,
                "replacement_before_causal_mask": True,
            }
        return attn_weights

    def _record_mask_violation(self, attn_probs, attention_mask):
        if not self._stats_enabled() or not hasattr(self, "last_llm_visual_3d_rope_stats"):
            return
        # Avoid full [B,H,Q,K] diagnostic masks in training. They can add
        # multi-GiB temporary allocations and are not part of the model path.
        head_max_probs = attn_probs.detach().amax(dim=1, keepdim=True)
        prob_tensor_for_checks = head_max_probs if self.training else attn_probs.detach()
        if attention_mask is None:
            self.last_llm_visual_3d_rope_stats["masked_attention_prob_max"] = 0.0
            self.last_llm_visual_3d_rope_stats["attention_prob_nan_count"] = int(
                torch.isnan(prob_tensor_for_checks).sum().item()
            )
            self.last_llm_visual_3d_rope_stats["attention_prob_inf_count"] = int(
                torch.isinf(prob_tensor_for_checks).sum().item()
            )
            self.last_llm_visual_3d_rope_stats["masked_attention_prob_nonfinite_count"] = 0
            return
        mask = attention_mask < -1e4
        if not mask.any():
            self.last_llm_visual_3d_rope_stats["masked_attention_prob_max"] = 0.0
            self.last_llm_visual_3d_rope_stats["attention_prob_nan_count"] = int(
                torch.isnan(prob_tensor_for_checks).sum().item()
            )
            self.last_llm_visual_3d_rope_stats["attention_prob_inf_count"] = int(
                torch.isinf(prob_tensor_for_checks).sum().item()
            )
            self.last_llm_visual_3d_rope_stats["masked_attention_prob_nonfinite_count"] = 0
            return
        # Keep this validation cheap: expanding the mask over all attention
        # heads can allocate tens of GiB for long VLM sequences during
        # training. A max over heads preserves the causal-mask violation signal
        # while reducing the diagnostic tensor from [B,H,Q,K] to [B,1,Q,K].
        masked_probs = head_max_probs.masked_select(mask)
        nonfinite_count = int((~torch.isfinite(masked_probs)).sum().item()) if masked_probs.numel() else 0
        finite_masked_probs = masked_probs.float()
        finite_masked_probs = finite_masked_probs[torch.isfinite(finite_masked_probs)]
        self.last_llm_visual_3d_rope_stats["masked_attention_prob_max"] = (
            float(finite_masked_probs.max().item()) if finite_masked_probs.numel() else 0.0
        )
        self.last_llm_visual_3d_rope_stats["attention_prob_nan_count"] = int(
            torch.isnan(prob_tensor_for_checks).sum().item()
        )
        self.last_llm_visual_3d_rope_stats["attention_prob_inf_count"] = int(
            torch.isinf(prob_tensor_for_checks).sum().item()
        )
        self.last_llm_visual_3d_rope_stats["masked_attention_prob_nonfinite_count"] = nonfinite_count

    def _sanitize_attention_logits(self, attn_weights, attention_mask):
        if not torch.is_floating_point(attn_weights):
            return attn_weights

        if self.training:
            if self._stats_enabled() and hasattr(self, "last_llm_visual_3d_rope_stats"):
                sample = attn_weights.detach()[..., : min(attn_weights.shape[-2], 128), : min(attn_weights.shape[-1], 128)]
                self.last_llm_visual_3d_rope_stats["attention_logit_nan_count"] = int(torch.isnan(sample).sum().item())
                self.last_llm_visual_3d_rope_stats["attention_logit_inf_count"] = int(torch.isinf(sample).sum().item())
                self.last_llm_visual_3d_rope_stats["attention_logits_sanitized"] = "training_inplace"

            torch.nan_to_num_(attn_weights, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
            if attention_mask is not None:
                mask = attention_mask < -1e4
                if mask.any():
                    attn_weights.masked_fill_(mask, -1.0e9)
            return attn_weights

        finite = torch.isfinite(attn_weights)
        nonfinite_count = int((~finite).sum().item())
        if self._stats_enabled() and hasattr(self, "last_llm_visual_3d_rope_stats"):
            self.last_llm_visual_3d_rope_stats["attention_logit_nan_count"] = int(torch.isnan(attn_weights).sum().item())
            self.last_llm_visual_3d_rope_stats["attention_logit_inf_count"] = int(torch.isinf(attn_weights).sum().item())
            self.last_llm_visual_3d_rope_stats["attention_logits_sanitized"] = bool(nonfinite_count > 0)

        if nonfinite_count == 0:
            return attn_weights

        sanitized = torch.nan_to_num(attn_weights.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        if attention_mask is not None:
            mask = attention_mask < -1e4
            if mask.any():
                expanded = mask.expand(sanitized.shape[0], sanitized.shape[1], sanitized.shape[2], sanitized.shape[3])
                sanitized = sanitized.masked_fill(expanded, -1.0e9)
        return sanitized

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            import warnings

            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states_raw = query_states
        key_states_raw = key_states

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = self._maybe_replace_visual_visual_logits(
            attn_weights=attn_weights,
            query_states_raw=query_states_raw,
            key_states_raw=key_states_raw,
            llm_geo_pos=getattr(self, "_llm_visual_3d_rope_pos", None),
            llm_geo_mask=getattr(self, "_llm_visual_3d_rope_mask", None),
            kv_seq_len=kv_seq_len,
        )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = self._sanitize_attention_logits(attn_weights, attention_mask)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._record_mask_violation(attn_weights, attention_mask)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def install_qwen2_visual_3d_rope_attention():
    if QWEN2_ATTENTION_CLASSES.get("eager") is not Qwen2Visual3DRopeAttention:
        QWEN2_ATTENTION_CLASSES["eager"] = Qwen2Visual3DRopeAttention


def qwen2_visual_3d_rope_requires_eager(config):
    return _as_bool(getattr(config, "llm_visual_3d_rope_enable", False)) and str(
        getattr(config, "_attn_implementation", getattr(config, "attn_implementation", "eager"))
    ) != "eager"


def attach_qwen2_visual_3d_rope_context(model, llm_geo_pos, llm_geo_mask):
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn._llm_visual_3d_rope_pos = llm_geo_pos
        attn._llm_visual_3d_rope_mask = llm_geo_mask
        attn.last_llm_visual_3d_rope_stats = None


def clear_qwen2_visual_3d_rope_context(model):
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for attr in ("_llm_visual_3d_rope_pos", "_llm_visual_3d_rope_mask"):
            if hasattr(attn, attr):
                delattr(attn, attr)


def collect_qwen2_visual_3d_rope_stats(model):
    stats = []
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        item = getattr(attn, "last_llm_visual_3d_rope_stats", None)
        if item:
            stats.append({key: _jsonable_scalar(value) for key, value in dict(item).items()})
    return stats
