"""Question-aware dense CUT3R side path.

The module intentionally has no transformers dependency.  It owns the
configuration validation, dense-token bookkeeping, hybrid masks, writeback,
and generation cache; the Qwen integration supplies cloned decoder blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.memory_audit import record_cuda_memory


def _enabled(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_ints(value, name: str) -> List[int]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or comma-separated string, got {value!r}.")
    result = [int(item) for item in value]
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _as_nonnegative_int(value, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _empty_long(device):
    return torch.empty(0, dtype=torch.long, device=device)


def _rank0_print(message: str) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if int(torch.distributed.get_rank()) != 0:
            return
    print(message, flush=True)


def is_trainable_downstream_lora_parameter(name: str) -> bool:
    """Keep dual-path's protected canonical layers 0--2 frozen.

    PEFT prefixes vary (for example ``base_model.model.model``), so a layer
    index may occur after arbitrary module prefixes. Parameters outside the
    decoder-layer naming convention retain the existing trainable policy.
    """

    match = re.search(r"(?:model\.)?layers\.(\d+)\.", str(name))
    return match is None or int(match.group(1)) >= 3


def prepare_merged_peft_donor_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    label: str,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Map a merged donor's bare linear tensors into PEFT-wrapped targets.

    ``merge_and_unload`` removes a donor's ``lora_*`` factor tensors and
    folds their contribution into the bare ``*.weight`` matrices.  A spatial
    block cloned from the canonical model can still expose those same base
    matrices as ``*.base_layer.weight``.  The function maps only those exact
    base tensors, leaves target-only zero-impact LoRA factors intact, and
    makes every other state mismatch explicit.
    """

    prepared: Dict[str, torch.Tensor] = {}
    used_source_keys = set()
    retained_lora_keys: List[str] = []
    missing_keys: List[str] = []
    shape_mismatches: List[str] = []
    for target_key, target_tensor in target_state.items():
        candidates = [target_key]
        if ".base_layer." in target_key:
            candidates.append(target_key.replace(".base_layer.", "."))
        source_key = next((key for key in candidates if key in source_state), None)
        if source_key is None:
            if ".lora_" in target_key:
                retained_lora_keys.append(target_key)
                continue
            missing_keys.append(target_key)
            continue
        source_tensor = source_state[source_key]
        # ZeRO-3 exposes partitioned parameters as shape-[0] placeholders
        # through ``state_dict()`` even inside its gather context.  Their
        # logical shape is supplied by the donor during ``load_state_dict``;
        # compare shapes only when the target is materialized.
        if target_tensor.numel() and tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                f"{target_key}: donor={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
            continue
        prepared[target_key] = source_tensor
        used_source_keys.add(source_key)
    unexpected_keys = sorted(set(source_state) - used_source_keys)
    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            f"{label} donor-state mismatch: missing={sorted(missing_keys)}, "
            f"unexpected={unexpected_keys}, shape_mismatched={shape_mismatches}."
        )
    return prepared, retained_lora_keys


@dataclass
class DualPathSpatialCache:
    """Static spatial evidence that travels with the autoregressive KV cache."""

    states: torch.Tensor                 # [batch, spatial_tokens, hidden]
    valid_mask: torch.Tensor             # [batch, spatial_tokens]
    frame_ids: torch.Tensor              # [batch, spatial_tokens], -1 for invalid
    source_batch_ids: torch.Tensor       # [batch]
    prefill_complete: bool = True

    def _index(self, index: torch.LongTensor) -> "DualPathSpatialCache":
        index = index.to(device=self.states.device, dtype=torch.long)
        return DualPathSpatialCache(
            states=self.states.index_select(0, index),
            valid_mask=self.valid_mask.index_select(0, index),
            frame_ids=self.frame_ids.index_select(0, index),
            source_batch_ids=self.source_batch_ids.index_select(0, index),
            prefill_complete=self.prefill_complete,
        )

    def batch_repeat_interleave(self, repeats: int) -> "DualPathSpatialCache":
        if int(repeats) <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}.")
        index = torch.arange(self.states.shape[0], device=self.states.device).repeat_interleave(int(repeats))
        return self._index(index)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "DualPathSpatialCache":
        return self._index(beam_idx)

    def to(self, *, device=None, dtype=None) -> "DualPathSpatialCache":
        return DualPathSpatialCache(
            states=self.states.to(device=device, dtype=dtype or self.states.dtype),
            valid_mask=self.valid_mask.to(device=device),
            frame_ids=self.frame_ids.to(device=device),
            source_batch_ids=self.source_batch_ids.to(device=device),
            prefill_complete=self.prefill_complete,
        )

    def validate(self, batch_size: int, device, dtype) -> None:
        if self.states.ndim != 3 or self.valid_mask.shape != self.states.shape[:2] or self.frame_ids.shape != self.states.shape[:2]:
            raise RuntimeError("Invalid dual-path spatial cache tensor shapes.")
        if self.states.shape[0] != int(batch_size):
            raise RuntimeError(
                f"Dual-path cache batch mismatch: cache={self.states.shape[0]}, active={batch_size}."
            )
        if self.states.device != device or self.valid_mask.device != device or self.frame_ids.device != device:
            raise RuntimeError("Dual-path cache is on a different device from the active decode batch.")
        if self.states.dtype != dtype:
            raise RuntimeError(
                f"Dual-path cache dtype mismatch: cache={self.states.dtype}, active={dtype}."
            )


def build_hybrid_attention_allow_mask(
    text_valid: torch.Tensor,
    spatial_valid: torch.Tensor,
    spatial_frame_ids: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Return boolean [B,L,L] query/key visibility for the hybrid branch.

    This compact reference mask is used in unit tests and small debugging
    examples. Production frame-local execution is block-structured and does
    not materialize this mask at full video token counts.
    """
    if mode not in {"frame_local", "global"}:
        raise ValueError(f"spatial_attention_mode must be frame_local or global, got {mode!r}.")
    if text_valid.ndim != 2 or spatial_valid.ndim != 2 or spatial_frame_ids.shape != spatial_valid.shape:
        raise ValueError("Hybrid-mask inputs must be [batch,tokens].")
    batch, text_len = text_valid.shape
    spatial_len = spatial_valid.shape[1]
    allow = torch.zeros((batch, text_len + spatial_len, text_len + spatial_len), dtype=torch.bool, device=text_valid.device)
    causal = torch.tril(torch.ones((text_len, text_len), dtype=torch.bool, device=text_valid.device))
    allow[:, :text_len, :text_len] = causal.unsqueeze(0) & text_valid[:, None, :] & text_valid[:, :, None]
    # Spatial queries see every valid prompt token. Text queries never see a spatial key.
    allow[:, text_len:, :text_len] = spatial_valid[:, :, None] & text_valid[:, None, :]
    if mode == "global":
        spatial_allow = spatial_valid[:, :, None] & spatial_valid[:, None, :]
    else:
        spatial_allow = (
            spatial_valid[:, :, None]
            & spatial_valid[:, None, :]
            & (spatial_frame_ids[:, :, None] == spatial_frame_ids[:, None, :])
        )
    allow[:, text_len:, text_len:] = spatial_allow
    return allow


def build_writeback_allow_mask(
    query_valid: torch.Tensor,
    query_is_visual: torch.Tensor,
    query_frame_ids: torch.Tensor,
    spatial_valid: torch.Tensor,
    spatial_frame_ids: torch.Tensor,
    visibility: str,
) -> torch.Tensor:
    if visibility not in {"frame_local", "global"}:
        raise ValueError(f"writeback_visibility must be frame_local or global, got {visibility!r}.")
    allow = query_valid[:, :, None] & spatial_valid[:, None, :]
    if visibility == "frame_local":
        visual = query_is_visual[:, :, None]
        same_frame = query_frame_ids[:, :, None] == spatial_frame_ids[:, None, :]
        allow = allow & (~visual | same_frame)
    return allow


class DenseCut3RProjector(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.proj_in = nn.Linear(feature_dim, hidden_size)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(self.norm(tokens))))


class DualPathWriteback(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, output_init_std: float = 1e-5):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("writeback num_heads must divide hidden_size.")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.o_proj.bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allow: torch.Tensor, *, allow_is_private: bool) -> torch.Tensor:
        query_has_memory = allow.any(dim=-1)
        # The caller-created chunk mask has no aliases and may be updated in
        # place. The public ``forward`` keeps its historical non-mutating
        # contract for callers that retain an ``allow`` tensor.
        safe_allow = allow if allow_is_private else allow.clone()
        # SDPA returns NaN for an all-masked query row on PyTorch 2.1. Give
        # excluded queries a temporary key, then remove their attention output
        # exactly before the output projection.
        safe_allow[..., 0] |= ~query_has_memory
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_allow[:, None], dropout_p=0.0)
        return attended * query_has_memory[:, None, :, None].to(dtype=attended.dtype)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        if allow.shape != (queries.shape[0], queries.shape[1], memory.shape[1]):
            raise ValueError("Writeback visibility mask shape mismatch.")
        q, k, v = self._heads(self.q_proj(queries)), self._heads(self.k_proj(memory)), self._heads(self.v_proj(memory))
        attended = self._masked_attention(q, k, v, allow, allow_is_private=False)
        attended = attended.transpose(1, 2).reshape_as(queries)
        return self.o_proj(attended)

    def _forward_chunk_from_preprojected_kv(
        self,
        queries: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        allow: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one exact writeback query chunk against shared projected K/V.

        ``allow`` must remain immutable when this helper is checkpointed: the
        backward recomputation receives the same tensor.  The public masked
        attention helper therefore takes its private copy here.
        """

        q = self._heads(self.q_proj(queries))
        attended = self._masked_attention(q, key_states, value_states, allow, allow_is_private=False)
        attended = attended.transpose(1, 2).reshape_as(queries)
        return self.o_proj(attended)

    def forward_chunked(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        query_valid: torch.Tensor,
        query_is_visual: torch.Tensor,
        query_frame_ids: torch.Tensor,
        spatial_valid: torch.Tensor,
        spatial_frame_ids: torch.Tensor,
        visibility: str,
        query_chunk_size: int,
        checkpoint_chunks: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Exact writeback with bounded query-side mask and activation memory.

        Checkpointing a chunk prevents its dense SDPA backward state from being
        retained for every query chunk.  Projected spatial K/V remain shared,
        so this trades only the chunk-local exact attention computation for
        backward-time recomputation.
        """

        if query_chunk_size <= 0:
            raise ValueError("forward_chunked requires a positive query_chunk_size.")
        if query_valid.shape != queries.shape[:2] or query_is_visual.shape != queries.shape[:2] or query_frame_ids.shape != queries.shape[:2]:
            raise ValueError("Writeback query metadata shape mismatch.")
        if spatial_valid.shape != memory.shape[:2] or spatial_frame_ids.shape != memory.shape[:2]:
            raise ValueError("Writeback spatial metadata shape mismatch.")

        k, v = self._heads(self.k_proj(memory)), self._heads(self.v_proj(memory))
        output = queries.new_empty(queries.shape)
        allowed_entries = 0
        total_entries = int(queries.shape[0] * queries.shape[1] * memory.shape[1])
        for start in range(0, queries.shape[1], int(query_chunk_size)):
            stop = min(start + int(query_chunk_size), queries.shape[1])
            allow = build_writeback_allow_mask(
                query_valid[:, start:stop],
                query_is_visual[:, start:stop],
                query_frame_ids[:, start:stop],
                spatial_valid,
                spatial_frame_ids,
                visibility,
            )
            allowed_entries += int(allow.sum(dtype=torch.int64).item())
            query_chunk = queries[:, start:stop]
            if checkpoint_chunks:
                from torch.utils.checkpoint import checkpoint

                chunk_output = checkpoint(
                    self._forward_chunk_from_preprojected_kv,
                    query_chunk,
                    k,
                    v,
                    allow,
                    use_reentrant=False,
                )
            else:
                chunk_output = self._forward_chunk_from_preprojected_kv(query_chunk, k, v, allow)
            output[:, start:stop] = chunk_output
        valid_ratio = float(allowed_entries) / float(total_entries) if total_entries else 0.0
        return output, valid_ratio


class Cut3RDualPathSpatialBranch(nn.Module):
    """Owns dense projectors, cloned decoder blocks, and writeback weights."""

    def __init__(self, config, decoder_blocks: Optional[Sequence[nn.Module]] = None):
        super().__init__()
        self.source_layers = _as_ints(getattr(config, "spatial_source_layers", [0, 1, 2]), "spatial_source_layers")
        self.num_layers = int(getattr(config, "spatial_num_layers", 3))
        if self.num_layers != 3 or len(self.source_layers) != 3:
            raise ValueError("The first dual-path experiment requires exactly three source layers and three spatial blocks.")
        self.cut3r_layers = _as_ints(getattr(config, "cut3r_spatialstack_layers", "6,9,12"), "cut3r_spatialstack_layers")
        if self.cut3r_layers != [6, 9, 12]:
            raise ValueError("Dual-path SpatialStack mapping must remain CUT3R [6,9,12].")
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.feature_dim = int(getattr(config, "cut3r_spatialstack_feature_dim", getattr(config, "spatial_feature_dim", 768)) or 768)
        self.attention_mode = str(getattr(config, "spatial_attention_mode", "frame_local"))
        self.query_scope = str(getattr(config, "writeback_query_scope", "all_tokens"))
        self.writeback_visibility = str(getattr(config, "writeback_visibility", "frame_local"))
        self.gradient_checkpointing = _enabled(getattr(config, "dual_path_gradient_checkpointing", True))
        self.spatial_mlp_chunk_size = _as_nonnegative_int(
            getattr(config, "spatial_mlp_chunk_size", 1024), "spatial_mlp_chunk_size"
        )
        self.writeback_query_chunk_size = _as_nonnegative_int(
            getattr(config, "writeback_query_chunk_size", 512), "writeback_query_chunk_size"
        )
        self.projectors = nn.ModuleDict({str(layer): DenseCut3RProjector(self.feature_dim, self.hidden_size) for layer in self.cut3r_layers})
        self.blocks = nn.ModuleList(list(decoder_blocks or []))
        self.writeback = DualPathWriteback(
            self.hidden_size,
            int(getattr(config, "num_attention_heads", 1)),
            float(getattr(config, "writeback_output_init_std", 1e-5)),
        )
        self.last_debug: Dict[str, object] = {}
        self._checkpointing_logged = False

    @staticmethod
    def _payload(sidecar: dict, layer: int) -> torch.Tensor:
        payloads = sidecar.get("cut3r_dec_layers", {})
        payload = payloads.get(str(layer), payloads.get(layer)) if isinstance(payloads, dict) else None
        if isinstance(payload, dict):
            payload = payload.get("patch_tokens")
        if payload is None and layer == 12:
            payload = sidecar.get("patch_tokens")
        if not isinstance(payload, torch.Tensor) or payload.ndim != 3:
            raise RuntimeError(f"Missing dense CUT3R layer-{layer} patch_tokens [frames,patches,dim].")
        return payload

    def extract_projected_levels(self, spatial_features, *, device, dtype, raw_layer12: bool = False) -> Dict[int, torch.Tensor]:
        items = [spatial_features] if isinstance(spatial_features, dict) else list(spatial_features or [])
        if not items:
            raise RuntimeError("Dual-path spatial branch requires CUT3R sidecars.")
        result = {layer: [] for layer in self.cut3r_layers}
        selected = [12] if raw_layer12 else self.cut3r_layers
        for sidecar in items:
            for layer in selected:
                tokens = self._payload(sidecar, layer).to(device=device, dtype=dtype)
                record_cuda_memory(f"after_cut3r_feature_level_{layer}")
                if tokens.shape[-1] != self.feature_dim:
                    raise RuntimeError(f"CUT3R layer-{layer} feature dim mismatch: {tokens.shape[-1]} != {self.feature_dim}.")
                projector = self.projectors[str(layer)]
                if self.training and self.gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    projected = checkpoint(projector, tokens, use_reentrant=False)
                else:
                    projected = projector(tokens)
                result[layer].append(projected)
                record_cuda_memory(f"after_spatial_projector_level_{layer}")
        return {layer: torch.stack(values, dim=0) for layer, values in result.items() if values}

    def mature_frame_local(
        self,
        levels: Dict[int, torch.Tensor],
        prompt_embeddings: Sequence[torch.Tensor],
        prompt_position_ids: Sequence[torch.Tensor],
        spatial_position_ids: Sequence[torch.Tensor],
        *,
        raw_layer12: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run cloned blocks one frame at a time without a video-sized mask.

        Qwen decoder blocks accept the standard additive [B,1,L,L] mask.  At
        most one frame's 729 patches is present in a call; global execution is
        intentionally delegated to the target-GPU SDPA preflight integration.
        """
        if raw_layer12:
            return [levels[12][sample] for sample in range(levels[12].shape[0])], list(prompt_embeddings)
        if len(self.blocks) != 3:
            raise RuntimeError("Dual-path branch does not have three cloned decoder blocks.")
        spatial_by_sample = [levels[6][sample] for sample in range(levels[6].shape[0])]
        prompts = list(prompt_embeddings)
        if self.training and self.gradient_checkpointing and not self._checkpointing_logged:
            _rank0_print(
                "[DUAL_PATH][CHECKPOINT] enabled custom projectors and spatial blocks "
                f"(mlp_chunk_size={self.spatial_mlp_chunk_size}, writeback_query_chunk_size={self.writeback_query_chunk_size})."
            )
            self._checkpointing_logged = True
        for block_index, source_layer in enumerate((6, 9, 12)):
            if block_index:
                spatial_by_sample = [
                    spatial + levels[source_layer][sample] for sample, spatial in enumerate(spatial_by_sample)
                ]
            updated_samples = []
            updated_prompts = []
            for sample, spatial_frames in enumerate(spatial_by_sample):
                prompt = prompts[sample]
                prompt_pos = prompt_position_ids[sample]
                if self.attention_mode == "global":
                    spatial = spatial_frames.flatten(0, 1)
                    positions = torch.cat((prompt_pos, spatial_position_ids[sample].flatten()), dim=0)
                    hidden = self._run_spatial_block(self.blocks[block_index], prompt, spatial, positions)
                    updated_samples.append(hidden[prompt.shape[0]:].view_as(spatial_frames))
                    updated_prompts.append(hidden[:prompt.shape[0]])
                    record_cuda_memory(f"after_spatial_block_{block_index}")
                    continue
                frame_outputs = []
                first_prompt = None
                for frame in range(spatial_frames.shape[0]):
                    spatial = spatial_frames[frame]
                    positions = torch.cat((prompt_pos, spatial_position_ids[sample][frame]), dim=0)
                    hidden = self._run_spatial_block(self.blocks[block_index], prompt, spatial, positions)
                    if first_prompt is None:
                        first_prompt = hidden[:prompt.shape[0]]
                    frame_outputs.append(hidden[prompt.shape[0]:])
                updated_samples.append(torch.stack(frame_outputs, dim=0))
                updated_prompts.append(first_prompt)
                record_cuda_memory(f"after_spatial_block_{block_index}")
            spatial_by_sample, prompts = updated_samples, updated_prompts
        return spatial_by_sample, prompts

    def _run_spatial_block(self, block: nn.Module, text: torch.Tensor, spatial: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                lambda current_text, current_spatial: self._run_split_qwen_block(
                    block, current_text, current_spatial, position_ids, self.spatial_mlp_chunk_size
                ),
                text,
                spatial,
                use_reentrant=False,
            )
        return self._run_split_qwen_block(block, text, spatial, position_ids, self.spatial_mlp_chunk_size)

    @staticmethod
    def _run_tokenwise_mlp(mlp: nn.Module, hidden: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Run a tokenwise Qwen SwiGLU MLP in exact sequence chunks."""

        if chunk_size <= 0 or hidden.shape[1] <= int(chunk_size):
            return mlp(hidden)
        return torch.cat(
            [mlp(hidden[:, start:min(start + int(chunk_size), hidden.shape[1])]) for start in range(0, hidden.shape[1], int(chunk_size))],
            dim=1,
        )

    @staticmethod
    def _run_split_qwen_block(
        block: nn.Module,
        text: torch.Tensor,
        spatial: torch.Tensor,
        position_ids: torch.Tensor,
        mlp_chunk_size: int = 0,
    ) -> torch.Tensor:
        """Execute Qwen attention without a dense video-sized 4D mask.

        Text queries are evaluated causally against text keys. Spatial queries
        use the same Q/K/V projections but attend to all text plus the supplied
        spatial set.  Frame-local callers supply one frame; global callers
        supply all frames and rely on the selected SDPA memory-efficient kernel.
        """
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        except ImportError as exc:  # pragma: no cover - exercised on training env
            raise RuntimeError("Dual-path Qwen execution requires transformers Qwen2 attention helpers.") from exc
        hidden = torch.cat((text, spatial), dim=0).unsqueeze(0)
        residual = hidden
        normed = block.input_layernorm(hidden)
        attn = block.self_attn
        batch, length, _ = normed.shape
        heads = int(attn.num_heads)
        kv_heads = int(attn.num_key_value_heads)
        head_dim = int(attn.head_dim)
        q = attn.q_proj(normed).view(batch, length, heads, head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        rotary_seq_len = max(length, int(position_ids.max().item()) + 1)
        cos, sin = attn.rotary_emb(v, seq_len=rotary_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids.unsqueeze(0))
        groups = heads // kv_heads
        k, v = repeat_kv(k, groups), repeat_kv(v, groups)
        text_len = text.shape[0]
        text_out = F.scaled_dot_product_attention(q[:, :, :text_len], k[:, :, :text_len], v[:, :, :text_len], is_causal=True)
        spatial_out = F.scaled_dot_product_attention(q[:, :, text_len:], k, v, is_causal=False)
        attended = torch.cat((text_out, spatial_out), dim=2).transpose(1, 2).reshape(batch, length, -1)
        hidden = residual + attn.o_proj(attended)
        residual = hidden
        hidden = residual + Cut3RDualPathSpatialBranch._run_tokenwise_mlp(
            block.mlp, block.post_attention_layernorm(hidden), mlp_chunk_size
        )
        return hidden[0]

    def apply_writeback(
        self,
        canonical_hidden: torch.Tensor,
        spatial_cache: DualPathSpatialCache,
        query_mask: torch.Tensor,
        query_is_visual: torch.Tensor,
        query_frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        spatial_cache.validate(canonical_hidden.shape[0], canonical_hidden.device, canonical_hidden.dtype)
        if self.query_scope == "text_only":
            query_mask = query_mask & ~query_is_visual
        elif self.query_scope != "all_tokens":
            raise ValueError(f"writeback_query_scope must be text_only or all_tokens, got {self.query_scope!r}.")
        query_chunk_size = int(getattr(self, "writeback_query_chunk_size", 0))
        if query_chunk_size > 0:
            delta, valid_ratio = self.writeback.forward_chunked(
                canonical_hidden,
                spatial_cache.states,
                query_mask,
                query_is_visual,
                query_frame_ids,
                spatial_cache.valid_mask,
                spatial_cache.frame_ids,
                self.writeback_visibility,
                query_chunk_size,
                checkpoint_chunks=bool(self.training and self.gradient_checkpointing),
            )
        else:
            allow = build_writeback_allow_mask(
                query_mask, query_is_visual, query_frame_ids, spatial_cache.valid_mask, spatial_cache.frame_ids, self.writeback_visibility
            )
            delta = self.writeback(canonical_hidden, spatial_cache.states, allow)
            valid_ratio = float(allow.sum(dtype=torch.int64).item()) / float(allow.numel()) if allow.numel() else 0.0
        delta = delta * query_mask.unsqueeze(-1).to(dtype=delta.dtype)
        record_cuda_memory("after_writeback")
        self.last_debug = {
            "writeback_residual_norm": float(torch.linalg.vector_norm(delta.detach(), dtype=torch.float32).item()),
            "canonical_hidden_norm": float(torch.linalg.vector_norm(canonical_hidden.detach(), dtype=torch.float32).item()),
            "writeback_valid_ratio": valid_ratio,
            "writeback_query_chunk_size": query_chunk_size,
        }
        return canonical_hidden + delta
