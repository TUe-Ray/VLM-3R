import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
except Exception:
    Qwen2RMSNorm = None


def _as_bool_config(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_optional_int_config(value, name):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null"}:
            return None
        value = value.strip()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or None, got {value!r}.") from exc


def _resolve_additive_output_init(value, *, zero_init: bool) -> str:
    """Resolve the explicit additive output-projection initialization mode.

    ``cut3r_spatialstack_zero_init`` predates the scoring-only identity mode.
    Keeping ``value=None`` tied to that flag preserves every existing
    checkpoint/training configuration, including the uncommon native-init
    ``zero_init=False`` setup.
    """
    if value is None:
        return "zero" if zero_init else "native"
    mode = str(value).strip().lower()
    if mode not in {"zero", "identity"}:
        raise ValueError(
            "cut3r_spatialstack_output_init must be 'zero' or 'identity' when set, "
            f"got {value!r}."
        )
    return mode


def _initialize_additive_output_projection(proj_out: nn.Linear, mode: str) -> None:
    """Apply the scoring-only terminal projection initialization.

    An exact identity is meaningful only for a square projection.  Refuse a
    rectangular projector instead of silently selecting an arbitrary partial
    identity for a configuration that this experiment does not define.
    """
    if mode == "native":
        return
    if mode == "zero":
        nn.init.zeros_(proj_out.weight)
        nn.init.zeros_(proj_out.bias)
        return
    if mode == "identity":
        if proj_out.weight.shape[0] != proj_out.weight.shape[1]:
            raise ValueError(
                "cut3r_spatialstack_output_init='identity' requires a square additive proj_out, "
                f"got weight shape={tuple(proj_out.weight.shape)}."
            )
        nn.init.eye_(proj_out.weight)
        nn.init.zeros_(proj_out.bias)
        return
    raise AssertionError(f"Unhandled additive output initialization mode: {mode}")


def _parse_int_list(value, name):
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    try:
        parsed = [int(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers, got {value!r}.") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one layer.")
    return parsed


def _empty_long(device):
    return torch.empty(0, dtype=torch.long, device=device)


def _rank0_print(message: str):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if int(torch.distributed.get_rank()) != 0:
            return
    print(message, flush=True)


def _distributed_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _seeded_permutation(count: int, device, mode: str, seed: int) -> torch.Tensor:
    ids = torch.arange(count, device=device)
    if count <= 1:
        return ids
    if mode == "cyclic_shift":
        shift = int(seed) % count
        if shift == 0:
            shift = 1
        return torch.roll(ids, shifts=shift, dims=0)
    if mode == "reverse":
        return torch.arange(count - 1, -1, -1, device=device)

    generator_device = device if getattr(device, "type", "cpu") == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(count, generator=generator, device=device)
    if mode == "random_permutation":
        return perm
    if mode != "random_derange":
        raise ValueError(f"Unknown CUT3R SpatialStack shuffle mode: {mode}")

    for _ in range(16):
        if not torch.any(perm == ids):
            return perm
        perm = torch.randperm(count, generator=generator, device=device)
    fixed = torch.nonzero(perm == ids, as_tuple=False).flatten()
    if int(fixed.numel()) == count:
        return torch.roll(perm, shifts=1, dims=0)
    if int(fixed.numel()) == 1:
        idx = fixed[0]
        swap_idx = (idx + 1) % count
        tmp = perm[idx].clone()
        perm[idx] = perm[swap_idx]
        perm[swap_idx] = tmp
    elif int(fixed.numel()) > 1:
        perm[fixed] = torch.roll(perm[fixed], shifts=1, dims=0)
    return perm


def _build_norm(hidden_size: int, norm_type: str = "qwen_rmsnorm") -> nn.Module:
    norm_type = str(norm_type or "qwen_rmsnorm").strip().lower()
    if norm_type in {"qwen_rmsnorm", "rmsnorm", "qwen2_rmsnorm"} and Qwen2RMSNorm is not None:
        return Qwen2RMSNorm(int(hidden_size), eps=1e-6)
    if norm_type in {"qwen_rmsnorm", "rmsnorm", "qwen2_rmsnorm", "layernorm", "layer_norm"}:
        return nn.LayerNorm(int(hidden_size))
    raise ValueError(
        "cut3r_spatialstack_cross_attn_norm_type must be 'qwen_rmsnorm' or 'layernorm', "
        f"got {norm_type!r}."
    )


def _sincos_1d(length: int, dim: int, device, dtype) -> torch.Tensor:
    length = int(length)
    dim = int(dim)
    if dim <= 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    pair_dim = dim // 2
    if pair_dim <= 0:
        return torch.zeros(length, dim, device=device, dtype=dtype)
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    omega = torch.arange(pair_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(pair_dim, 1)))
    angles = positions * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim].to(dtype=dtype)


def _sincos_2d(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    height = int(height)
    width = int(width)
    dim = int(dim)
    if height <= 0 or width <= 0:
        raise ValueError(f"2D positional grid must be positive, got {(height, width)}.")
    dim_h = dim // 2
    dim_w = dim - dim_h
    emb_h = _sincos_1d(height, dim_h, device, dtype)
    emb_w = _sincos_1d(width, dim_w, device, dtype)
    pos_h = emb_h[:, None, :].expand(height, width, dim_h)
    pos_w = emb_w[None, :, :].expand(height, width, dim_w)
    return torch.cat([pos_h, pos_w], dim=-1).reshape(height * width, dim)


class Cut3RSpatialStackBranch(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        zero_init: bool = True,
        output_init: Optional[str] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(int(feature_dim))
        self.proj_in = nn.Linear(int(feature_dim), int(hidden_size))
        self.act = nn.GELU()
        self.proj_out = nn.Linear(int(hidden_size), int(hidden_size))
        self.output_init = _resolve_additive_output_init(output_init, zero_init=zero_init)
        _initialize_additive_output_projection(self.proj_out, self.output_init)
        # C1 buffers are deliberately non-persistent: calibration artifacts are
        # portable JSON and are applied explicitly at inference time.
        self.register_buffer("c1_enabled", torch.tensor(False), persistent=False)
        self.register_buffer("c1_pre_gelu_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("c1_residual_gain", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def set_c1_state(
        self,
        *,
        enabled: bool = True,
        pre_gelu_scale: Optional[float] = None,
        residual_gain: Optional[float] = None,
    ) -> None:
        self.c1_enabled.fill_(bool(enabled))
        if pre_gelu_scale is not None:
            self.c1_pre_gelu_scale.fill_(float(pre_gelu_scale))
        if residual_gain is not None:
            self.c1_residual_gain.fill_(float(residual_gain))

    def c1_components(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw pre-GELU, raw delta, and gain-scaled delta for C1."""
        z_pre = self.proj_in(self.norm(tokens))
        scale = self.c1_pre_gelu_scale.to(device=z_pre.device, dtype=z_pre.dtype)
        delta_raw = self.proj_out(self.act(scale * z_pre))
        gain = self.c1_residual_gain.to(device=delta_raw.device, dtype=delta_raw.dtype)
        return z_pre, delta_raw, gain * delta_raw

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if bool(self.c1_enabled.item()):
            return self.c1_components(tokens)[2]
        return self.proj_out(self.act(self.proj_in(self.norm(tokens))))


class Cut3RSpatialStackMergeBranch(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        merge_size: int = 2,
        projector_hidden_dim: int = 4096,
        zero_init: bool = True,
        output_init: Optional[str] = None,
    ):
        super().__init__()
        self.merge_size = int(merge_size)
        merged_dim = int(feature_dim) * self.merge_size * self.merge_size
        self.norm = nn.LayerNorm(merged_dim)
        self.proj_in = nn.Linear(merged_dim, int(projector_hidden_dim))
        self.act = nn.GELU()
        self.proj_out = nn.Linear(int(projector_hidden_dim), int(hidden_size))
        self.output_init = _resolve_additive_output_init(output_init, zero_init=zero_init)
        _initialize_additive_output_projection(self.proj_out, self.output_init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(self.norm(tokens))))


class Cut3RSpatialStackPreAggregator(nn.Module):
    """Aggregate multiple CUT3R decoder levels before SpatialStack projection."""

    def __init__(self, source_layers: List[int], feature_dim: int, preagg_type: str = "weighted_sum"):
        super().__init__()
        self.source_layers = [int(layer) for layer in source_layers]
        self.feature_dim = int(feature_dim)
        self.preagg_type = str(preagg_type or "weighted_sum").strip().lower()
        if self.preagg_type not in {"weighted_sum", "concat_linear"}:
            raise ValueError(
                "cut3r_spatialstack_preagg_type must be 'weighted_sum' or 'concat_linear', "
                f"got {self.preagg_type!r}."
            )
        self.norms = nn.ModuleDict(
            {str(layer): nn.LayerNorm(self.feature_dim) for layer in self.source_layers}
        )
        if self.preagg_type == "weighted_sum":
            self.scalar_logits = nn.Parameter(torch.zeros(len(self.source_layers), dtype=torch.float32))
            self.concat_proj = None
        else:
            self.scalar_logits = None
            self.concat_proj = nn.Linear(len(self.source_layers) * self.feature_dim, self.feature_dim)

    def forward(self, layer_features: Dict[int, torch.Tensor]) -> torch.Tensor:
        missing = [layer for layer in self.source_layers if int(layer) not in layer_features]
        if missing:
            raise RuntimeError(
                "CUT3R SpatialStack pre-aggregation source layers are missing: "
                f"missing={missing}, available={sorted(int(k) for k in layer_features.keys())}."
            )
        features = [layer_features[int(layer)] for layer in self.source_layers]
        shapes = [tuple(feature.shape) for feature in features]
        if any(shape != shapes[0] for shape in shapes[1:]):
            shape_by_layer = {
                int(layer): tuple(layer_features[int(layer)].shape)
                for layer in self.source_layers
            }
            raise RuntimeError(
                "CUT3R SpatialStack pre-aggregation requires identical feature shapes; "
                f"got {shape_by_layer}."
            )
        if int(features[0].shape[-1]) != self.feature_dim:
            raise RuntimeError(
                "CUT3R SpatialStack pre-aggregation feature dim mismatch: "
                f"got {int(features[0].shape[-1])}, expected {self.feature_dim}."
            )
        normed = [
            self.norms[str(layer)](feature)
            for layer, feature in zip(self.source_layers, features)
        ]
        if self.preagg_type == "weighted_sum":
            weights = F.softmax(self.scalar_logits.float(), dim=0).to(device=normed[0].device, dtype=normed[0].dtype)
            stacked = torch.stack(normed, dim=0)
            return (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return self.concat_proj(torch.cat(normed, dim=-1))

    def debug_info(self) -> dict:
        info = {
            "preagg_layers": list(self.source_layers),
            "preagg_type": self.preagg_type,
            "feature_dim": int(self.feature_dim),
        }
        if self.preagg_type == "weighted_sum":
            weights = F.softmax(self.scalar_logits.detach().float(), dim=0)
            info["raw_scalar_logits"] = [float(x) for x in self.scalar_logits.detach().float().cpu().tolist()]
            info["softmax_weights"] = {
                f"preagg_weight_dec{layer}": float(weight)
                for layer, weight in zip(self.source_layers, weights.cpu().tolist())
            }
        else:
            info["concat_input_dim"] = int(len(self.source_layers) * self.feature_dim)
            info["aggregation_output_dim"] = int(self.feature_dim)
            info["aggregation_weight_norm"] = float(self.concat_proj.weight.detach().float().norm().item())
        return info


class Cut3RCameraTokenProjector(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, init_scale: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(int(feature_dim))
        self.proj_in = nn.Linear(int(feature_dim), int(hidden_size))
        self.act = nn.GELU()
        self.proj_out = nn.Linear(int(hidden_size), int(hidden_size))
        self.gamma = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.proj_out(self.act(self.proj_in(self.norm(tokens))))
        return projected * self.gamma.to(device=projected.device, dtype=projected.dtype)


class Cut3RSpatialStackCrossAttentionBlock(nn.Module):
    """Cross-attend LLM visual states to one aligned set of CUT3R geometry tokens."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"cut3r_spatialstack_cross_attn_heads must be positive, got {self.num_heads}.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "cut3r_spatialstack_cross_attn_heads must divide hidden_size, "
                f"got hidden_size={self.hidden_size}, heads={self.num_heads}."
            )
        self.head_dim = self.hidden_size // self.num_heads
        self.visual_norm = nn.LayerNorm(self.hidden_size)
        self.geometry_norm = nn.LayerNorm(self.feature_dim)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.feature_dim, self.hidden_size)
        self.v_proj = nn.Linear(self.feature_dim, self.hidden_size)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        self.register_buffer("c1_enabled", torch.tensor(False), persistent=False)
        self.register_buffer("c1_qk_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("c1_residual_gain", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self._c1_collect_diagnostics = False
        self._c1_last_diagnostics = None

    def set_c1_state(
        self,
        *,
        enabled: bool = True,
        qk_scale: Optional[float] = None,
        residual_gain: Optional[float] = None,
        collect_diagnostics: Optional[bool] = None,
    ) -> None:
        self.c1_enabled.fill_(bool(enabled))
        if qk_scale is not None:
            self.c1_qk_scale.fill_(float(qk_scale))
        if residual_gain is not None:
            self.c1_residual_gain.fill_(float(residual_gain))
        if collect_diagnostics is not None:
            self._c1_collect_diagnostics = bool(collect_diagnostics)

    @staticmethod
    def _c1_moments(value: torch.Tensor) -> dict:
        value = value.detach().float()
        return {
            "count": int(value.numel()),
            "sum": float(value.sum().item()),
            "sum_sq": float(value.square().sum().item()),
        }

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, visual_hidden: torch.Tensor, geometry_tokens: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if visual_hidden.dim() == 2:
            visual_hidden = visual_hidden.unsqueeze(0)
            squeeze_batch = True
        if geometry_tokens.dim() == 2:
            geometry_tokens = geometry_tokens.unsqueeze(0)
        if visual_hidden.dim() != 3:
            raise ValueError(f"visual_hidden must be [tokens,hidden] or [batch,tokens,hidden], got {tuple(visual_hidden.shape)}.")
        if geometry_tokens.dim() != 3:
            raise ValueError(
                "geometry_tokens must be [tokens,dim] or [batch,tokens,dim], "
                f"got {tuple(geometry_tokens.shape)}."
            )
        if int(visual_hidden.shape[0]) != int(geometry_tokens.shape[0]):
            raise ValueError(
                "visual_hidden and geometry_tokens batch size mismatch: "
                f"{int(visual_hidden.shape[0])} vs {int(geometry_tokens.shape[0])}."
            )
        if int(geometry_tokens.shape[1]) == 0:
            raise ValueError("CUT3R SpatialStack cross-attn requires at least one geometry token.")
        if int(visual_hidden.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"visual_hidden dim mismatch: got {int(visual_hidden.shape[-1])}, expected {self.hidden_size}."
            )
        if int(geometry_tokens.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"geometry token dim mismatch: got {int(geometry_tokens.shape[-1])}, expected {self.feature_dim}."
            )

        visual_normed = self.visual_norm(visual_hidden)
        geometry_normed = self.geometry_norm(geometry_tokens)
        q_raw = self._split_heads(self.q_proj(visual_normed))
        k_raw = self._split_heads(self.k_proj(geometry_normed))
        v = self._split_heads(self.v_proj(geometry_normed))
        if bool(self.c1_enabled.item()):
            qk_scale = self.c1_qk_scale.to(device=q_raw.device, dtype=q_raw.dtype)
            q = qk_scale * q_raw
            k = qk_scale * k_raw
        else:
            q, k = q_raw, k_raw
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=attn_scores.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(visual_hidden.shape[0], visual_hidden.shape[1], self.hidden_size)
        delta_raw = self.out_proj(attended)
        if bool(self.c1_enabled.item()):
            delta = self.c1_residual_gain.to(device=delta_raw.device, dtype=delta_raw.dtype) * delta_raw
            if self._c1_collect_diagnostics:
                raw_logits = torch.matmul(q_raw, k_raw.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
                self._c1_last_diagnostics = {
                    "q": self._c1_moments(q_raw),
                    "k": self._c1_moments(k_raw),
                    "v": self._c1_moments(v),
                    "raw_logits": self._c1_moments(raw_logits),
                    "calibrated_logits": self._c1_moments(attn_scores),
                    "delta_raw": self._c1_moments(delta_raw),
                    "delta": self._c1_moments(delta),
                    "q_shape": list(q_raw.shape),
                    "k_shape": list(k_raw.shape),
                    "v_shape": list(v.shape),
                    "logit_shape": list(attn_scores.shape),
                }
        else:
            delta = delta_raw
        return delta.squeeze(0) if squeeze_batch else delta


class Cut3RSpatialStackCrossAttentionBlockV2(nn.Module):
    """Original-style camera-aware CUT3R SpatialStack cross-attention block."""

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        *,
        patch_align: str = "resize",
        merge_size: int = 2,
        projector_hidden_dim: int = 4096,
        dropout: float = 0.0,
        use_camera_tokens: bool = True,
        use_mlp: bool = True,
        norm_type: str = "qwen_rmsnorm",
        pos_embed: str = "sincos2d",
        gamma_attn_init: float = 0.05,
        gamma_mlp_init: float = 0.05,
        gamma_learnable: bool = True,
        force_zero_gamma_at_eval: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.patch_align = str(patch_align or "resize").strip().lower()
        self.merge_size = int(merge_size)
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.use_camera_tokens = bool(use_camera_tokens)
        self.use_mlp = bool(use_mlp)
        self.force_zero_gamma_at_eval = bool(force_zero_gamma_at_eval)
        self.norm_type = str(norm_type or "qwen_rmsnorm").strip().lower()
        self.pos_embed = str(pos_embed or "none").strip().lower()
        if self.patch_align not in {"resize", "merge"}:
            raise ValueError(
                "cut3r_spatialstack_cross_attn_patch_align must be 'resize' or 'merge', "
                f"got {self.patch_align!r}."
            )
        if self.pos_embed not in {"none", "false", "off", "sincos2d"}:
            raise ValueError(
                "cut3r_spatialstack_cross_attn_pos_embed must be 'sincos2d' or 'none', "
                f"got {self.pos_embed!r}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"cut3r_spatialstack_cross_attn_heads must be positive, got {self.num_heads}.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "cut3r_spatialstack_cross_attn_heads must divide hidden_size, "
                f"got hidden_size={self.hidden_size}, heads={self.num_heads}."
            )
        patch_dim = self.feature_dim
        if self.patch_align == "merge":
            patch_dim = self.feature_dim * self.merge_size * self.merge_size
        self.patch_input_dim = int(patch_dim)
        self.camera_norm = _build_norm(self.feature_dim, self.norm_type)
        self.patch_norm = _build_norm(self.patch_input_dim, self.norm_type)
        self.camera_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(self.projector_hidden_dim, self.hidden_size),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_input_dim, self.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(self.projector_hidden_dim, self.hidden_size),
        )
        self.q_norm = _build_norm(self.hidden_size, self.norm_type)
        self.kv_norm = _build_norm(self.hidden_size, self.norm_type)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.mlp_norm = _build_norm(self.hidden_size, self.norm_type)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        gamma_attn = torch.tensor(float(gamma_attn_init), dtype=torch.float32)
        gamma_mlp = torch.tensor(float(gamma_mlp_init), dtype=torch.float32)
        if gamma_learnable:
            self.gamma_attn = nn.Parameter(gamma_attn)
            self.gamma_mlp = nn.Parameter(gamma_mlp)
        else:
            self.register_buffer("gamma_attn", gamma_attn)
            self.register_buffer("gamma_mlp", gamma_mlp)
        self.camera_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size, dtype=torch.float32))

    def _effective_gammas(self, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.force_zero_gamma_at_eval and not self.training:
            zero = self.gamma_attn.new_zeros(()).to(device=device, dtype=dtype)
            return zero, zero
        return (
            self.gamma_attn.to(device=device, dtype=dtype),
            self.gamma_mlp.to(device=device, dtype=dtype),
        )

    def _add_pos(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        visual_grid_shape: Optional[Tuple[int, int]],
        geometry_grid_shape: Optional[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_embed in {"none", "false", "off"}:
            return q, kv
        if visual_grid_shape is None:
            visual_side = int(math.isqrt(int(q.shape[1])))
            if visual_side * visual_side != int(q.shape[1]):
                raise RuntimeError(
                    "CUT3R SpatialStack cross_attn_v2 sincos2d needs visual_grid_shape for non-square "
                    f"visual token count {int(q.shape[1])}."
                )
            visual_grid_shape = (visual_side, visual_side)
        if geometry_grid_shape is None:
            patch_count = int(kv.shape[1]) - (1 if self.use_camera_tokens else 0)
            geo_side = int(math.isqrt(patch_count))
            if geo_side * geo_side != patch_count:
                raise RuntimeError(
                    "CUT3R SpatialStack cross_attn_v2 sincos2d needs geometry_grid_shape for non-square "
                    f"geometry patch token count {patch_count}."
                )
            geometry_grid_shape = (geo_side, geo_side)
        visual_pos = _sincos_2d(
            int(visual_grid_shape[0]),
            int(visual_grid_shape[1]),
            self.hidden_size,
            q.device,
            q.dtype,
        )
        geo_pos = _sincos_2d(
            int(geometry_grid_shape[0]),
            int(geometry_grid_shape[1]),
            self.hidden_size,
            kv.device,
            kv.dtype,
        )
        if int(visual_pos.shape[0]) != int(q.shape[1]):
            raise RuntimeError(
                "CUT3R SpatialStack cross_attn_v2 visual positional shape mismatch: "
                f"pos={tuple(visual_pos.shape)}, q={tuple(q.shape)}."
            )
        patch_count = int(kv.shape[1]) - (1 if self.use_camera_tokens else 0)
        if int(geo_pos.shape[0]) != patch_count:
            raise RuntimeError(
                "CUT3R SpatialStack cross_attn_v2 geometry positional shape mismatch: "
                f"pos={tuple(geo_pos.shape)}, kv={tuple(kv.shape)}."
            )
        q = q + visual_pos.unsqueeze(0)
        if self.use_camera_tokens:
            camera_pos = self.camera_pos.to(device=kv.device, dtype=kv.dtype).expand(kv.shape[0], -1, -1)
            kv_pos = torch.cat([camera_pos, geo_pos.unsqueeze(0).expand(kv.shape[0], -1, -1)], dim=1)
        else:
            kv_pos = geo_pos.unsqueeze(0).expand(kv.shape[0], -1, -1)
        return q, kv + kv_pos

    @staticmethod
    def _attention_stats(
        attn_weights: torch.Tensor,
        *,
        visual_grid_shape: Optional[Tuple[int, int]],
        has_camera: bool,
    ) -> dict:
        weights = attn_weights.detach().float()
        if weights.dim() == 4:
            weights = weights.mean(dim=1)
        eps = 1e-12
        stats = {
            "attention_entropy": float((-(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1)).mean().item()),
        }
        patch_offset = 1 if has_camera else 0
        if has_camera:
            stats["camera_attention_mass"] = float(weights[..., 0].mean().item())
        else:
            stats["camera_attention_mass"] = 0.0
        patch_weights = weights[..., patch_offset:]
        stats["patch_attention_mass"] = float(patch_weights.sum(dim=-1).mean().item())
        query_count = int(weights.shape[1])
        patch_count = int(patch_weights.shape[-1])
        diag_count = min(query_count, patch_count)
        if diag_count > 0:
            q_idx = torch.arange(diag_count, device=weights.device)
            stats["diagonal_attention_mass"] = float(weights[:, q_idx, patch_offset + q_idx].mean().item())
        else:
            stats["diagonal_attention_mass"] = 0.0
        local_values = []
        if visual_grid_shape is not None and patch_count == query_count:
            grid_h, grid_w = int(visual_grid_shape[0]), int(visual_grid_shape[1])
            if grid_h * grid_w == query_count:
                for token_idx in range(query_count):
                    row = token_idx // grid_w
                    col = token_idx % grid_w
                    patch_indices = []
                    for rr in range(max(0, row - 1), min(grid_h, row + 2)):
                        for cc in range(max(0, col - 1), min(grid_w, col + 2)):
                            patch_indices.append(patch_offset + rr * grid_w + cc)
                    local_values.append(weights[:, token_idx, patch_indices].sum(dim=-1))
        if local_values:
            stats["local_3x3_attention_mass"] = float(torch.stack(local_values, dim=1).mean().item())
        else:
            stats["local_3x3_attention_mass"] = 0.0
        return stats

    def forward(
        self,
        visual_hidden: torch.Tensor,
        patch_tokens: torch.Tensor,
        camera_tokens: Optional[torch.Tensor] = None,
        *,
        visual_grid_shape: Optional[Tuple[int, int]] = None,
        geometry_grid_shape: Optional[Tuple[int, int]] = None,
        return_stats: bool = False,
    ):
        squeeze_batch = False
        if visual_hidden.dim() == 2:
            visual_hidden = visual_hidden.unsqueeze(0)
            squeeze_batch = True
        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(0)
        if camera_tokens is not None and camera_tokens.dim() == 2:
            camera_tokens = camera_tokens.unsqueeze(0)
        if visual_hidden.dim() != 3:
            raise ValueError(f"visual_hidden must be [tokens,hidden] or [batch,tokens,hidden], got {tuple(visual_hidden.shape)}.")
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be [tokens,dim] or [batch,tokens,dim], got {tuple(patch_tokens.shape)}.")
        if int(visual_hidden.shape[0]) != int(patch_tokens.shape[0]):
            raise ValueError(
                "visual_hidden and patch_tokens batch size mismatch: "
                f"{int(visual_hidden.shape[0])} vs {int(patch_tokens.shape[0])}."
            )
        if int(visual_hidden.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"visual_hidden dim mismatch: got {int(visual_hidden.shape[-1])}, expected {self.hidden_size}."
            )
        if int(patch_tokens.shape[-1]) != self.patch_input_dim:
            raise ValueError(
                "CUT3R cross_attn_v2 patch token dim mismatch after alignment: "
                f"got {int(patch_tokens.shape[-1])}, expected {self.patch_input_dim}."
            )
        if self.use_camera_tokens:
            if camera_tokens is None:
                raise RuntimeError("CUT3R cross_attn_v2 requires camera_tokens, but none were provided.")
            if camera_tokens.dim() != 3:
                raise ValueError(f"camera_tokens must be [batch,tokens,dim], got {tuple(camera_tokens.shape)}.")
            if int(camera_tokens.shape[0]) != int(visual_hidden.shape[0]):
                raise ValueError(
                    "visual_hidden and camera_tokens batch size mismatch: "
                    f"{int(visual_hidden.shape[0])} vs {int(camera_tokens.shape[0])}."
                )
            if int(camera_tokens.shape[1]) != 1:
                raise ValueError(f"CUT3R cross_attn_v2 expects one camera token per frame, got {int(camera_tokens.shape[1])}.")
            if int(camera_tokens.shape[-1]) != self.feature_dim:
                raise ValueError(
                    f"camera token dim mismatch: got {int(camera_tokens.shape[-1])}, expected {self.feature_dim}."
                )
        patch_lang = self.patch_proj(self.patch_norm(patch_tokens))
        if self.use_camera_tokens:
            camera_lang = self.camera_proj(self.camera_norm(camera_tokens))
            geo_memory = torch.cat([camera_lang, patch_lang], dim=1)
        else:
            geo_memory = patch_lang
        q = self.q_norm(visual_hidden)
        kv = self.kv_norm(geo_memory)
        q, k = self._add_pos(
            q,
            kv,
            visual_grid_shape=visual_grid_shape,
            geometry_grid_shape=geometry_grid_shape,
        )
        attn_out, attn_weights = self.cross_attention(
            query=q,
            key=k,
            value=kv,
            need_weights=return_stats,
            average_attn_weights=False,
        )
        attn_out = self.dropout(attn_out)
        gamma_attn, gamma_mlp = self._effective_gammas(visual_hidden.device, visual_hidden.dtype)
        x = visual_hidden + gamma_attn * attn_out
        mlp_out = self.ffn(self.mlp_norm(x)) if self.use_mlp else torch.zeros_like(x)
        x = x + gamma_mlp * mlp_out
        delta = x - visual_hidden
        if not return_stats:
            return delta.squeeze(0) if squeeze_batch else delta
        stats = self._attention_stats(
            attn_weights,
            visual_grid_shape=visual_grid_shape,
            has_camera=self.use_camera_tokens,
        )
        hidden_norm = visual_hidden.detach().float().norm().item()
        stats.update(
            {
                "geo_memory_shape": list(geo_memory.shape),
                "cross_attn_v2_force_zero_gamma_at_eval": bool(self.force_zero_gamma_at_eval),
                "learned_gamma_attn": float(self.gamma_attn.detach().float().item()),
                "learned_gamma_mlp": float(self.gamma_mlp.detach().float().item()),
                "effective_gamma_attn": float(gamma_attn.detach().float().item()),
                "effective_gamma_mlp": float(gamma_mlp.detach().float().item()),
                "gamma_attn": float(self.gamma_attn.detach().float().item()),
                "gamma_mlp": float(self.gamma_mlp.detach().float().item()),
                "attn_out_norm": float(attn_out.detach().float().norm().item()),
                "mlp_out_norm": float(mlp_out.detach().float().norm().item()),
                "delta_norm": float(delta.detach().float().norm().item()),
                "hidden_norm": float(hidden_norm),
                "attn_out_to_hidden_norm": float(attn_out.detach().float().norm().item() / max(hidden_norm, 1e-12)),
                "mlp_out_to_hidden_norm": float(mlp_out.detach().float().norm().item() / max(hidden_norm, 1e-12)),
                "delta_to_hidden_norm": float(delta.detach().float().norm().item() / max(hidden_norm, 1e-12)),
            }
        )
        return (delta.squeeze(0) if squeeze_batch else delta), stats


class Cut3RSpatialStackMerger(nn.Module):
    """Build dense LLM residuals from pre-extracted CUT3R decoder-layer sidecars."""

    EXCLUDED_METADATA_KEYS = (
        "newline_token_indices",
        "padding_token_indices",
        "answer_token_indices",
        "text_token_indices",
        "special_token_indices",
        "camera_prefix_token_indices",
        "spatial_bridge_token_indices",
    )

    def __init__(self, config):
        super().__init__()
        self.cut3r_layers = _parse_int_list(
            getattr(config, "cut3r_spatialstack_layers", "6,9,12"),
            "cut3r_spatialstack_layers",
        )
        self.llm_layers = _parse_int_list(
            getattr(config, "cut3r_spatialstack_llm_layers", "0,1,2"),
            "cut3r_spatialstack_llm_layers",
        )
        self.preagg_enable = _as_bool_config(
            getattr(config, "cut3r_spatialstack_preagg_enable", False),
            False,
        )
        self.preagg_layers = _parse_int_list(
            getattr(config, "cut3r_spatialstack_preagg_layers", "6,9,12"),
            "cut3r_spatialstack_preagg_layers",
        )
        self.preagg_type = str(
            getattr(config, "cut3r_spatialstack_preagg_type", "weighted_sum") or "weighted_sum"
        ).strip().lower()
        if self.preagg_type not in {"weighted_sum", "concat_linear"}:
            raise ValueError(
                "cut3r_spatialstack_preagg_type must be 'weighted_sum' or 'concat_linear', "
                f"got {self.preagg_type!r}."
            )
        self.preagg_projector_sharing = str(
            getattr(config, "cut3r_spatialstack_preagg_projector_sharing", "shared") or "shared"
        ).strip().lower()
        if self.preagg_projector_sharing not in {"shared", "layer_specific"}:
            raise ValueError(
                "cut3r_spatialstack_preagg_projector_sharing must be 'shared' or 'layer_specific', "
                f"got {self.preagg_projector_sharing!r}."
            )
        self.preagg_log_weights = _as_bool_config(
            getattr(config, "cut3r_spatialstack_preagg_log_weights", True),
            True,
        )
        self.preagg_output_layer_key = str(
            getattr(config, "cut3r_spatialstack_preagg_output_layer_key", "preagg") or "preagg"
        )
        self.preagg_use_layer_gamma = _as_bool_config(
            getattr(config, "cut3r_spatialstack_preagg_use_layer_gamma", True),
            True,
        )
        self.preagg_layer_gamma_init = float(
            getattr(config, "cut3r_spatialstack_preagg_layer_gamma_init", 1.0)
        )
        if not self.preagg_enable and len(self.cut3r_layers) != len(self.llm_layers):
            raise ValueError(
                "cut3r_spatialstack_layers and cut3r_spatialstack_llm_layers must have the same length, "
                f"got {self.cut3r_layers} and {self.llm_layers}."
            )
        self.fusion_type = str(getattr(config, "cut3r_spatialstack_fusion_type", "add") or "add").strip().lower()
        if self.fusion_type not in {"add", "cross_attn", "cross_attn_v2"}:
            raise ValueError(
                "cut3r_spatialstack_fusion_type must be 'add', 'cross_attn', or 'cross_attn_v2', "
                f"got {self.fusion_type!r}."
            )
        if self.preagg_enable and self.fusion_type != "add":
            raise ValueError(
                "CUT3R SpatialStack pre-aggregation is only supported with "
                f"cut3r_spatialstack_fusion_type='add', got {self.fusion_type!r}."
            )
        feature_dim = getattr(config, "cut3r_spatialstack_feature_dim", None)
        if feature_dim is None:
            feature_dim = getattr(config, "spatial_feature_dim", None)
        if feature_dim is None:
            raise ValueError(
                "use_cut3r_spatialstack=True requires cut3r_spatialstack_feature_dim "
                "or spatial_feature_dim so trainable merger parameters exist before optimizer creation."
            )
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.feature_key = str(getattr(config, "cut3r_spatialstack_feature_key", "cut3r_dec_layers"))
        self.zero_init = _as_bool_config(getattr(config, "cut3r_spatialstack_zero_init", True), True)
        self.output_init = _resolve_additive_output_init(
            getattr(config, "cut3r_spatialstack_output_init", None),
            zero_init=self.zero_init,
        )
        self.log_first_n = int(getattr(config, "cut3r_spatialstack_log_first_n", 3) or 0)
        self.projector_type = str(
            getattr(config, "cut3r_spatialstack_projector_type", "token_mlp") or "token_mlp"
        ).strip().lower()
        if self.projector_type not in {"token_mlp", "merge_mlp"}:
            raise ValueError(
                "cut3r_spatialstack_projector_type must be 'token_mlp' or 'merge_mlp', "
                f"got {self.projector_type!r}."
            )
        if self.fusion_type != "add" and self.projector_type != "token_mlp":
            raise ValueError(
                "cut3r_spatialstack_projector_type='merge_mlp' is only supported with "
                f"cut3r_spatialstack_fusion_type='add', got {self.fusion_type!r}."
            )
        self.merge_size = int(getattr(config, "cut3r_spatialstack_merge_size", 2) or 2)
        if self.merge_size <= 0:
            raise ValueError(f"cut3r_spatialstack_merge_size must be positive, got {self.merge_size}.")
        self.projector_hidden_dim = int(getattr(config, "cut3r_spatialstack_projector_hidden_dim", 4096) or 4096)
        if self.projector_hidden_dim <= 0:
            raise ValueError(
                "cut3r_spatialstack_projector_hidden_dim must be positive, "
                f"got {self.projector_hidden_dim}."
            )
        default_heads = getattr(config, "num_attention_heads", 1)
        self.cross_attn_heads = _as_optional_int_config(
            getattr(config, "cut3r_spatialstack_cross_attn_heads", None),
            "cut3r_spatialstack_cross_attn_heads",
        )
        if self.cross_attn_heads is None:
            self.cross_attn_heads = int(default_heads)
        self.cross_attn_dropout = float(getattr(config, "cut3r_spatialstack_cross_attn_dropout", 0.0) or 0.0)
        self.cross_attn_zero_init = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_zero_init", True),
            True,
        )
        self.cross_attn_same_frame_only = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_same_frame_only", True),
            True,
        )
        if self.fusion_type == "cross_attn_v2" and not self.cross_attn_same_frame_only:
            raise ValueError("cut3r_spatialstack_fusion_type='cross_attn_v2' requires same-frame cross-attention.")
        self.cross_attn_impl = str(
            getattr(config, "cut3r_spatialstack_cross_attn_impl", "torch_mha") or "torch_mha"
        ).strip().lower()
        if self.cross_attn_impl != "torch_mha":
            raise ValueError(
                "cut3r_spatialstack_cross_attn_impl currently supports only 'torch_mha', "
                f"got {self.cross_attn_impl!r}."
            )
        self.cross_attn_patch_align = str(
            getattr(config, "cut3r_spatialstack_cross_attn_patch_align", "resize") or "resize"
        ).strip().lower()
        if self.cross_attn_patch_align not in {"resize", "merge"}:
            raise ValueError(
                "cut3r_spatialstack_cross_attn_patch_align must be 'resize' or 'merge', "
                f"got {self.cross_attn_patch_align!r}."
            )
        self.cross_attn_use_camera_tokens = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_use_camera_tokens", self.fusion_type == "cross_attn_v2"),
            self.fusion_type == "cross_attn_v2",
        )
        self.require_camera_tokens = _as_bool_config(
            getattr(config, "cut3r_spatialstack_require_camera_tokens", self.cross_attn_use_camera_tokens),
            self.cross_attn_use_camera_tokens,
        )
        if self.require_camera_tokens and not self.cross_attn_use_camera_tokens:
            raise ValueError(
                "cut3r_spatialstack_require_camera_tokens=True requires "
                "cut3r_spatialstack_cross_attn_use_camera_tokens=True."
            )
        self.cross_attn_use_mlp = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_use_mlp", True),
            True,
        )
        self.cross_attn_norm_type = str(
            getattr(config, "cut3r_spatialstack_cross_attn_norm_type", "qwen_rmsnorm") or "qwen_rmsnorm"
        ).strip().lower()
        self.cross_attn_pos_embed = str(
            getattr(config, "cut3r_spatialstack_cross_attn_pos_embed", "sincos2d") or "sincos2d"
        ).strip().lower()
        self.cross_attn_gamma_attn_init = float(
            getattr(config, "cut3r_spatialstack_cross_attn_gamma_attn_init", 0.05)
        )
        self.cross_attn_gamma_mlp_init = float(
            getattr(config, "cut3r_spatialstack_cross_attn_gamma_mlp_init", 0.05)
        )
        self.cross_attn_gamma_learnable = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_gamma_learnable", True),
            True,
        )
        self.cross_attn_v2_force_zero_gamma_at_eval = _as_bool_config(
            getattr(config, "cut3r_spatialstack_cross_attn_v2_force_zero_gamma_at_eval", False),
            False,
        )
        self.residual_scale = float(getattr(config, "cut3r_spatialstack_residual_scale", 1.0))
        self.frame_shuffle = _as_bool_config(getattr(config, "cut3r_spatialstack_frame_shuffle", False), False)
        self.frame_shuffle_mode = str(getattr(config, "cut3r_spatialstack_frame_shuffle_mode", "random_derange") or "random_derange")
        self.frame_shuffle_seed = int(getattr(config, "cut3r_spatialstack_frame_shuffle_seed", 0) or 0)
        self.token_shuffle = _as_bool_config(getattr(config, "cut3r_spatialstack_token_shuffle", False), False)
        self.token_shuffle_mode = str(getattr(config, "cut3r_spatialstack_token_shuffle_mode", "random_derange") or "random_derange")
        self.token_shuffle_seed = int(getattr(config, "cut3r_spatialstack_token_shuffle_seed", 0) or 0)
        source_layers = self.preagg_layers if self.preagg_enable else self.cut3r_layers
        self.layer_map = (
            {}
            if self.preagg_enable
            else {int(llm_layer): int(cut3r_layer) for cut3r_layer, llm_layer in zip(self.cut3r_layers, self.llm_layers)}
        )
        self.preaggregator = None
        self.branches = nn.ModuleDict()
        self.preagg_layer_gammas = nn.ParameterDict()
        self.cross_attn_blocks = nn.ModuleDict()
        if self.fusion_type == "add":
            if self.preagg_enable:
                self.preaggregator = Cut3RSpatialStackPreAggregator(
                    source_layers,
                    self.feature_dim,
                    preagg_type=self.preagg_type,
                )
                if self.preagg_projector_sharing == "shared":
                    self.branches = nn.ModuleDict({"shared": self._build_projector_branch()})
                else:
                    self.branches = nn.ModuleDict(
                        {str(llm_layer): self._build_projector_branch() for llm_layer in self.llm_layers}
                    )
                if self.preagg_use_layer_gamma:
                    self.preagg_layer_gammas = nn.ParameterDict(
                        {
                            str(llm_layer): nn.Parameter(
                                torch.tensor(float(self.preagg_layer_gamma_init), dtype=torch.float32)
                            )
                            for llm_layer in self.llm_layers
                        }
                    )
            else:
                self.branches = nn.ModuleDict(
                    {str(cut3r_layer): self._build_projector_branch() for cut3r_layer in self.cut3r_layers}
                )
        elif self.fusion_type == "cross_attn":
            self.cross_attn_blocks = nn.ModuleDict(
                {
                    str(llm_layer): Cut3RSpatialStackCrossAttentionBlock(
                        self.feature_dim,
                        self.hidden_size,
                        num_heads=self.cross_attn_heads,
                        dropout=self.cross_attn_dropout,
                        zero_init=self.cross_attn_zero_init,
                    )
                    for llm_layer in self.llm_layers
                }
            )
        else:
            self.cross_attn_blocks = nn.ModuleDict(
                {
                    str(llm_layer): Cut3RSpatialStackCrossAttentionBlockV2(
                        self.feature_dim,
                        self.hidden_size,
                        num_heads=self.cross_attn_heads,
                        patch_align=self.cross_attn_patch_align,
                        merge_size=self.merge_size,
                        projector_hidden_dim=self.projector_hidden_dim,
                        dropout=self.cross_attn_dropout,
                        use_camera_tokens=self.cross_attn_use_camera_tokens,
                        use_mlp=self.cross_attn_use_mlp,
                        norm_type=self.cross_attn_norm_type,
                        pos_embed=self.cross_attn_pos_embed,
                        gamma_attn_init=self.cross_attn_gamma_attn_init,
                        gamma_mlp_init=self.cross_attn_gamma_mlp_init,
                        gamma_learnable=self.cross_attn_gamma_learnable,
                        force_zero_gamma_at_eval=self.cross_attn_v2_force_zero_gamma_at_eval,
                    )
                    for llm_layer in self.llm_layers
                }
            )
        self.last_debug = {}
        self._shuffle_sample_count = 0
        self._frame_shuffle_log_count = 0
        self._token_shuffle_log_count = 0
        self._cross_attn_log_count = 0

    def _build_projector_branch(self) -> nn.Module:
        if self.projector_type == "merge_mlp":
            return Cut3RSpatialStackMergeBranch(
                self.feature_dim,
                self.hidden_size,
                merge_size=self.merge_size,
                projector_hidden_dim=self.projector_hidden_dim,
                zero_init=self.zero_init,
                output_init=self.output_init,
            )
        return Cut3RSpatialStackBranch(
            self.feature_dim,
            self.hidden_size,
            zero_init=self.zero_init,
            output_init=self.output_init,
        )

    @staticmethod
    def resize_grid(tokens: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"CUT3R frame tokens must be [tokens, dim], got {tuple(tokens.shape)}.")
        source_tokens = int(tokens.shape[0])
        target_h = int(target_h)
        target_w = int(target_w)
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Target visual grid must be positive, got {(target_h, target_w)}.")
        source_side = int(math.isqrt(source_tokens))
        if source_side * source_side != source_tokens:
            raise ValueError(
                "CUT3R source token count must be a square grid to align SpatialStack features, "
                f"got {source_tokens}."
            )
        if source_side == target_h and source_side == target_w:
            return tokens
        grid = tokens.reshape(source_side, source_side, tokens.shape[-1]).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            grid.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized[0].permute(1, 2, 0).reshape(target_h * target_w, tokens.shape[-1]).to(dtype=tokens.dtype)

    @staticmethod
    def resize_square_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        target_tokens = int(target_tokens)
        target_side = int(math.isqrt(target_tokens))
        if target_side * target_side != target_tokens:
            raise ValueError(f"Target visual token count must be a square grid, got {target_tokens}.")
        return Cut3RSpatialStackMerger.resize_grid(tokens, target_side, target_side)

    def merge_frame_grid(self, tokens: torch.Tensor, target_grid_shape: Tuple[int, int]) -> torch.Tensor:
        if target_grid_shape is None:
            raise RuntimeError(
                "CUT3R SpatialStack merge_mlp requires visual_grid_shapes metadata for every visual frame."
            )
        target_h, target_w = int(target_grid_shape[0]), int(target_grid_shape[1])
        merge_size = int(self.merge_size)
        high_res = self.resize_grid(tokens, target_h * merge_size, target_w * merge_size)
        channels = int(high_res.shape[-1])
        grouped = high_res.reshape(target_h, merge_size, target_w, merge_size, channels)
        grouped = grouped.permute(0, 2, 1, 3, 4).contiguous()
        return grouped.reshape(target_h * target_w, merge_size * merge_size * channels)

    @staticmethod
    def _metadata_items(visual_metadata):
        if isinstance(visual_metadata, dict):
            return [visual_metadata]
        if isinstance(visual_metadata, (list, tuple)):
            return list(visual_metadata)
        raise RuntimeError(
            "CUT3R SpatialStack requires visual_metadata from prepare_inputs_labels_for_multimodal(); "
            f"got {type(visual_metadata).__name__}."
        )

    @staticmethod
    def _feature_items(spatial_features, batch_size: int) -> List[dict]:
        if spatial_features is None:
            raise RuntimeError("use_cut3r_spatialstack=True requires pre-extracted CUT3R spatial_features sidecars.")
        if isinstance(spatial_features, dict):
            if batch_size != 1:
                raise RuntimeError(
                    "A single spatial_features dict can only be used with batch_size=1 for CUT3R SpatialStack; "
                    f"got batch_size={batch_size}."
                )
            return [spatial_features]
        if isinstance(spatial_features, (list, tuple)):
            if len(spatial_features) != batch_size:
                raise RuntimeError(
                    "CUT3R SpatialStack spatial_features batch mismatch: "
                    f"features={len(spatial_features)}, visual_metadata={batch_size}."
                )
            return list(spatial_features)
        raise RuntimeError(f"Unsupported spatial_features type for CUT3R SpatialStack: {type(spatial_features).__name__}.")

    def _extract_layer_tokens(self, sidecar: dict, cut3r_layer: int) -> torch.Tensor:
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"CUT3R SpatialStack sidecar must be a dict, got {type(sidecar).__name__}.")
        layer_key = str(int(cut3r_layer))
        if self.feature_key in sidecar:
            layer_payloads = sidecar[self.feature_key]
            if not isinstance(layer_payloads, dict):
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar[{self.feature_key!r}] must be a dict keyed by decoder layer."
                )
            if layer_key not in layer_payloads and int(cut3r_layer) not in layer_payloads:
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar is missing decoder layer {cut3r_layer}; "
                    f"available keys={sorted(str(k) for k in layer_payloads.keys())}."
                )
            payload = layer_payloads.get(layer_key, layer_payloads.get(int(cut3r_layer)))
            if isinstance(payload, dict):
                if "patch_tokens" not in payload:
                    raise RuntimeError(f"CUT3R decoder layer {cut3r_layer} payload lacks 'patch_tokens'.")
                tokens = payload["patch_tokens"]
            else:
                tokens = payload
        elif "patch_tokens" in sidecar:
            selected_layers = self.preagg_layers if self.preagg_enable else self.cut3r_layers
            if len(selected_layers) != 1:
                raise RuntimeError(
                    "Legacy CUT3R sidecar schema with top-level 'patch_tokens' is only valid when exactly "
                    f"one CUT3R source layer is configured; got {selected_layers}."
                )
            tokens = sidecar["patch_tokens"]
        else:
            raise RuntimeError(
                f"CUT3R SpatialStack sidecar must contain {self.feature_key!r} or legacy 'patch_tokens'; "
                f"got keys={sorted(sidecar.keys())}."
            )
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"CUT3R layer {cut3r_layer} patch tokens must be a tensor, got {type(tokens).__name__}.")
        if tokens.dim() == 4 and int(tokens.shape[0]) == 1:
            tokens = tokens[0]
        if tokens.dim() != 3:
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} patch tokens must be [frames,tokens,dim], got {tuple(tokens.shape)}."
            )
        if int(tokens.shape[-1]) != self.feature_dim:
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} feature dim mismatch: sidecar dim={int(tokens.shape[-1])}, "
                f"configured cut3r_spatialstack_feature_dim={self.feature_dim}."
            )
        return tokens.detach()

    def _extract_layer_camera_tokens(self, sidecar: dict, cut3r_layer: int) -> torch.Tensor:
        if not isinstance(sidecar, dict):
            raise RuntimeError(f"CUT3R camera-token sidecar must be a dict, got {type(sidecar).__name__}.")
        layer_key = str(int(cut3r_layer))
        tokens = None
        if self.feature_key in sidecar:
            layer_payloads = sidecar[self.feature_key]
            if not isinstance(layer_payloads, dict):
                raise RuntimeError(
                    f"CUT3R camera-token sidecar[{self.feature_key!r}] must be a dict keyed by decoder layer."
                )
            if layer_key not in layer_payloads and int(cut3r_layer) not in layer_payloads:
                raise RuntimeError(
                    f"CUT3R camera-token sidecar is missing decoder layer {cut3r_layer}; "
                    f"available keys={sorted(str(k) for k in layer_payloads.keys())}."
                )
            payload = layer_payloads.get(layer_key, layer_payloads.get(int(cut3r_layer)))
            if not isinstance(payload, dict) or "camera_tokens" not in payload:
                raise RuntimeError(
                    f"CUT3R decoder layer {cut3r_layer} payload lacks 'camera_tokens'. "
                    "Re-extract sidecars with scripts/extraction/extract_cut3r_layer_features.py."
                )
            tokens = payload["camera_tokens"]
        elif "camera_tokens" in sidecar:
            selected_layers = self.preagg_layers if self.preagg_enable else self.cut3r_layers
            if len(selected_layers) != 1:
                raise RuntimeError(
                    "Legacy CUT3R sidecar schema with top-level 'camera_tokens' is only valid when exactly "
                    f"one CUT3R source layer is configured; got {selected_layers}."
                )
            tokens = sidecar["camera_tokens"]
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} camera_tokens must be a tensor, got {type(tokens).__name__}."
            )
        if tokens.dim() == 4 and int(tokens.shape[0]) == 1:
            tokens = tokens[0]
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.dim() != 3:
            raise RuntimeError(
                f"CUT3R layer {cut3r_layer} camera_tokens must be [frames,tokens,dim], got {tuple(tokens.shape)}."
            )
        if int(tokens.shape[1]) != 1:
            raise RuntimeError(
                f"CUT3R Design 1 expects exactly one camera token per frame, got {int(tokens.shape[1])}."
            )
        if int(tokens.shape[-1]) != self.feature_dim:
            raise RuntimeError(
                f"CUT3R camera token dim mismatch: sidecar dim={int(tokens.shape[-1])}, "
                f"configured cut3r_spatialstack_feature_dim={self.feature_dim}."
            )
        return tokens.detach()

    @staticmethod
    def _sidecar_frame_indices(sidecar: dict) -> Optional[List[int]]:
        for key in ("frame_indices", "frame_order"):
            if key in sidecar:
                value = sidecar[key]
                if isinstance(value, torch.Tensor):
                    return [int(x) for x in value.detach().cpu().flatten().tolist()]
                return [int(x) for x in value]
        metadata = sidecar.get("metadata") if isinstance(sidecar, dict) else None
        if isinstance(metadata, dict):
            for key in ("frame_indices", "frame_order"):
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, torch.Tensor):
                        return [int(x) for x in value.detach().cpu().flatten().tolist()]
                    return [int(x) for x in value]
        return None

    @staticmethod
    def _grid_shape_at(metadata: dict, local_frame_idx: int) -> Optional[Tuple[int, int]]:
        shapes = metadata.get("visual_grid_shapes", None)
        if not isinstance(shapes, (list, tuple)) or local_frame_idx >= len(shapes):
            return None
        shape = shapes[local_frame_idx]
        if isinstance(shape, torch.Tensor):
            shape = shape.detach().cpu().flatten().tolist()
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            return None
        return int(shape[0]), int(shape[1])

    def _validate_visual_metadata(self, metadata: dict, batch_idx: int, device) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        visual_indices = metadata.get("visual_token_indices", None)
        frame_ids = metadata.get("visual_frame_ids", None)
        if not isinstance(visual_indices, torch.Tensor) or not isinstance(frame_ids, torch.Tensor):
            raise RuntimeError(f"CUT3R SpatialStack metadata[{batch_idx}] is missing visual_token_indices/visual_frame_ids.")
        visual_indices = visual_indices.to(device=device, dtype=torch.long)
        frame_ids = frame_ids.to(device=device, dtype=torch.long)
        if visual_indices.numel() != frame_ids.numel():
            raise RuntimeError(
                f"CUT3R SpatialStack metadata[{batch_idx}] visual_token_indices and visual_frame_ids length mismatch: "
                f"{visual_indices.numel()} vs {frame_ids.numel()}."
            )
        excluded = []
        for key in self.EXCLUDED_METADATA_KEYS:
            value = metadata.get(key, _empty_long(device))
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                excluded.append(value.to(device=device, dtype=torch.long))
        if excluded and visual_indices.numel() > 0:
            excluded_indices = torch.cat(excluded)
            overlap = torch.isin(visual_indices, excluded_indices)
            if bool(overlap.any().item()):
                bad = visual_indices[overlap][:16].detach().cpu().tolist()
                raise RuntimeError(
                    f"CUT3R SpatialStack metadata[{batch_idx}] visual tokens overlap excluded token positions: {bad}."
                )
        frame_order = metadata.get("frame_order", None)
        if frame_order is None:
            frame_order = list(dict.fromkeys(int(x) for x in frame_ids.detach().cpu().tolist()))
        else:
            frame_order = [int(x) for x in frame_order]
        return visual_indices, frame_ids, frame_order

    def _sample_seed(self, base_seed: int, sample_index: int, *, layer: int = 0, frame: int = 0) -> int:
        return int(base_seed) + int(sample_index) * 1009 + int(layer) * 9176 + int(frame) * 131 + _distributed_rank() * 1000003

    def _frame_source_order(self, frame_count: int, device, sample_index: int):
        if not self.frame_shuffle or frame_count <= 1:
            return list(range(frame_count)), None
        seed = self._sample_seed(self.frame_shuffle_seed, sample_index)
        perm = _seeded_permutation(frame_count, device, self.frame_shuffle_mode, seed)
        if self.log_first_n > 0 and self._frame_shuffle_log_count < self.log_first_n:
            _rank0_print(
                "[CUT3R SpatialStack Frame Shuffle] "
                f"sample_index={sample_index}, mode={self.frame_shuffle_mode}, seed={seed}, "
                f"F={frame_count}, source_frame_for_visual_frame={perm.detach().cpu().tolist()}"
            )
        elif self.log_first_n > 0 and self._frame_shuffle_log_count == self.log_first_n:
            _rank0_print("[CUT3R SpatialStack Frame Shuffle] Further per-sample logs suppressed.")
        self._frame_shuffle_log_count += 1
        return [int(x) for x in perm.detach().cpu().tolist()], perm.detach().cpu().tolist()

    def _maybe_shuffle_frame_tokens(self, tokens: torch.Tensor, sample_index: int, frame_idx: int) -> Tuple[torch.Tensor, Optional[List[int]]]:
        if not self.token_shuffle or int(tokens.shape[0]) <= 1:
            return tokens, None
        seed = self._sample_seed(self.token_shuffle_seed, sample_index, frame=frame_idx)
        perm = _seeded_permutation(int(tokens.shape[0]), tokens.device, self.token_shuffle_mode, seed)
        if self.log_first_n > 0 and self._token_shuffle_log_count < self.log_first_n:
            _rank0_print(
                "[CUT3R SpatialStack Token Shuffle] "
                f"sample_index={sample_index}, frame_idx={frame_idx}, mode={self.token_shuffle_mode}, "
                f"seed={seed}, N={int(tokens.shape[0])}, perm={perm.detach().cpu().tolist()}"
            )
        elif self.log_first_n > 0 and self._token_shuffle_log_count == self.log_first_n:
            _rank0_print("[CUT3R SpatialStack Token Shuffle] Further per-frame logs suppressed.")
        self._token_shuffle_log_count += 1
        return tokens.index_select(0, perm), perm.detach().cpu().tolist()

    def _ensure_module_dtype(self, device, dtype):
        param = next(self.parameters(), None)
        if param is not None and (param.device != device or param.dtype != dtype):
            self.to(device=device, dtype=dtype)

    def forward(
        self,
        spatial_features,
        visual_metadata,
        *,
        seq_len: int,
        device,
        dtype,
    ) -> Dict[int, torch.Tensor]:
        if self.fusion_type in {"cross_attn", "cross_attn_v2"}:
            return self.prepare_cross_attn_inputs(
                spatial_features,
                visual_metadata,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )

        metadata_items = self._metadata_items(visual_metadata)
        batch_size = len(metadata_items)
        feature_items = self._feature_items(spatial_features, batch_size)
        self._ensure_module_dtype(device, dtype)

        residuals = {
            int(llm_layer): torch.zeros(
                batch_size,
                int(seq_len),
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for llm_layer in self.llm_layers
        }
        debug = {
            "fusion_type": "add",
            "projector_type": self.projector_type,
            "selected_cut3r_layers": list(self.cut3r_layers),
            "selected_llm_layers": list(self.llm_layers),
            "preagg_enable": bool(self.preagg_enable),
            "preagg_layers": list(self.preagg_layers),
            "preagg_type": self.preagg_type,
            "preagg_projector_sharing": self.preagg_projector_sharing,
            "preagg_use_layer_gamma": bool(self.preagg_use_layer_gamma),
            "preagg_layer_gamma_init": float(self.preagg_layer_gamma_init),
            "feature_dim": int(self.feature_dim),
            "hidden_size": int(self.hidden_size),
            "merge_size": int(self.merge_size),
            "projector_hidden_dim": int(self.projector_hidden_dim),
            "zero_init": bool(self.zero_init),
            "output_init": self.output_init,
            "residual_scale": float(self.residual_scale),
            "samples": [],
            "layers": {},
        }

        for batch_idx, (metadata, sidecar) in enumerate(zip(metadata_items, feature_items)):
            visual_indices, frame_ids, frame_order = self._validate_visual_metadata(metadata, batch_idx, device)
            if int(visual_indices.numel()) == 0:
                continue
            if int(visual_indices.min().item()) < 0 or int(visual_indices.max().item()) >= int(seq_len):
                bad_min = int(visual_indices.min().item())
                bad_max = int(visual_indices.max().item())
                raise RuntimeError(
                    f"CUT3R SpatialStack metadata[{batch_idx}] visual_token_indices out of bounds for "
                    f"seq_len={int(seq_len)}: min={bad_min}, max={bad_max}."
                )
            sidecar_frame_indices = self._sidecar_frame_indices(sidecar)
            sidecar_frame_lookup = None
            if sidecar_frame_indices is not None and sidecar_frame_indices != frame_order:
                sidecar_frame_lookup = {int(frame_id): idx for idx, frame_id in enumerate(sidecar_frame_indices)}
                missing_frame_ids = [int(frame_id) for frame_id in frame_order if int(frame_id) not in sidecar_frame_lookup]
                if missing_frame_ids:
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame_indices mismatch for sample {batch_idx}: "
                        f"visual frame_order={frame_order}, sidecar frame_indices={sidecar_frame_indices}; "
                        f"missing visual frames={missing_frame_ids}."
                    )
            if sidecar_frame_indices is None and frame_order != list(range(len(frame_order))):
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar for sample {batch_idx} lacks frame_indices/frame_order, "
                    f"but visual frame_order={frame_order} is not the unambiguous default order."
                )
            should_debug_sample = self.log_first_n < 0 or len(debug["samples"]) < self.log_first_n
            if should_debug_sample:
                debug["samples"].append(
                    {
                        "sample_id": int(batch_idx),
                        "visual_token_count": int(visual_indices.numel()),
                        "frame_order": list(frame_order),
                    }
                )
            sample_index = int(self._shuffle_sample_count)
            if self.frame_shuffle or self.token_shuffle:
                self._shuffle_sample_count += 1
            frame_source_order, frame_shuffle_perm = self._frame_source_order(len(frame_order), device, sample_index)
            if should_debug_sample and frame_shuffle_perm is not None:
                debug["samples"][-1]["cut3r_spatialstack_frame_shuffle_perm"] = frame_shuffle_perm

            if self.preagg_enable:
                layer_features = {
                    int(cut3r_layer): self._extract_layer_tokens(sidecar, cut3r_layer).to(device=device, dtype=dtype)
                    for cut3r_layer in self.preagg_layers
                }
                aggregated_tokens = self.preaggregator(layer_features)
                preagg_debug = self.preaggregator.debug_info()
                debug["preagg"] = preagg_debug
                sidecar_frame_count = int(aggregated_tokens.shape[0])
                if sidecar_frame_lookup is not None:
                    token_frame_indices = [sidecar_frame_lookup[int(frame_id)] for frame_id in frame_order]
                elif sidecar_frame_count == len(frame_order):
                    token_frame_indices = list(range(len(frame_order)))
                elif frame_order and max(int(frame_id) for frame_id in frame_order) < sidecar_frame_count:
                    token_frame_indices = [int(frame_id) for frame_id in frame_order]
                else:
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame count mismatch for sample {batch_idx}, pre-aggregated layers "
                        f"{self.preagg_layers}: sidecar frames={sidecar_frame_count}, visual frame_order={frame_order}."
                    )
                aligned_frames = []
                aligned_indices = []
                raw_counts = []
                aligned_counts = []
                target_grid_shapes = []
                token_shuffle_perms = []
                for local_frame_idx, frame_id in enumerate(frame_order):
                    frame_mask = frame_ids == int(frame_id)
                    frame_visual_indices = visual_indices[frame_mask]
                    target_count = int(frame_visual_indices.numel())
                    if target_count == 0:
                        continue
                    grid_shape = self._grid_shape_at(metadata, local_frame_idx)
                    if grid_shape is not None:
                        grid_h, grid_w = grid_shape
                        if self.projector_type == "token_mlp" and grid_h != grid_w:
                            raise RuntimeError(
                                f"CUT3R SpatialStack requires square visual grids for sample {batch_idx}, "
                                f"frame {frame_id}; got visual_grid_shapes[{local_frame_idx}]={grid_shape}."
                            )
                        if grid_h * grid_w != target_count:
                            raise RuntimeError(
                                f"CUT3R SpatialStack visual token count mismatch for sample {batch_idx}, "
                                f"frame {frame_id}: visual_grid_shape={grid_shape} implies {grid_h * grid_w} "
                                f"tokens, but visual metadata has {target_count} positions."
                            )
                    source_local_frame_idx = int(frame_source_order[local_frame_idx])
                    raw_frame_tokens = aggregated_tokens[token_frame_indices[source_local_frame_idx]]
                    if self.projector_type == "merge_mlp":
                        if grid_shape is None:
                            raise RuntimeError(
                                "CUT3R SpatialStack merge_mlp requires visual_grid_shapes metadata; "
                                f"missing sample {batch_idx}, frame {frame_id}."
                            )
                        aligned = self.merge_frame_grid(raw_frame_tokens, grid_shape)
                    else:
                        aligned = self.resize_square_grid(raw_frame_tokens, target_count)
                    aligned, token_perm = self._maybe_shuffle_frame_tokens(aligned, sample_index, local_frame_idx)
                    if int(aligned.shape[0]) != target_count:
                        raise RuntimeError(
                            f"CUT3R SpatialStack aligned token count mismatch for sample {batch_idx}, "
                            f"frame {frame_id}, projector_type={self.projector_type}: "
                            f"aligned={int(aligned.shape[0])}, visual={target_count}."
                        )
                    aligned_frames.append(aligned)
                    aligned_indices.append(frame_visual_indices)
                    raw_counts.append(int(raw_frame_tokens.shape[0]))
                    aligned_counts.append(int(aligned.shape[0]))
                    target_grid_shapes.append(tuple(int(x) for x in grid_shape) if grid_shape is not None else None)
                    if token_perm is not None and len(token_shuffle_perms) < 3:
                        token_shuffle_perms.append(
                            {
                                "frame_id": int(frame_id),
                                "perm": token_perm,
                            }
                        )
                if not aligned_frames:
                    continue
                aligned_tokens = torch.cat(aligned_frames, dim=0)
                target_indices = torch.cat(aligned_indices, dim=0)
                debug.setdefault("preagg_input_layer_shapes", {}).update(
                    {str(layer): list(tensor.shape) for layer, tensor in layer_features.items()}
                )
                debug["preagg_aggregated_feature_shape"] = list(aggregated_tokens.shape)
                residual_norms_for_log = {}
                gamma_for_log = {}
                if self.preagg_projector_sharing == "shared":
                    shared_projected = self.branches["shared"](aligned_tokens)
                    for llm_layer in self.llm_layers:
                        projected = shared_projected
                        gamma_value = None
                        if self.preagg_use_layer_gamma:
                            gamma = self.preagg_layer_gammas[str(int(llm_layer))].to(
                                device=projected.device,
                                dtype=projected.dtype,
                            )
                            gamma_value = float(gamma.detach().float().item())
                            projected = projected * gamma
                        projected = projected * self.residual_scale
                        residuals[int(llm_layer)][batch_idx, target_indices] = projected
                        residual_norms_for_log[int(llm_layer)] = float(projected.detach().float().norm().item())
                        if gamma_value is not None:
                            gamma_for_log[int(llm_layer)] = gamma_value
                        if should_debug_sample:
                            debug["layers"].setdefault(str(llm_layer), []).append(
                                {
                                    "sample_id": int(batch_idx),
                                    "cut3r_layer": self.preagg_output_layer_key,
                                    "preagg_layers": list(self.preagg_layers),
                                    "raw_token_counts": raw_counts,
                                    "aligned_token_counts": aligned_counts,
                                    "target_grid_shapes": target_grid_shapes,
                                    "residual_norm": float(projected.detach().float().norm().item()),
                                    "frame_source_order": list(frame_source_order),
                                    "token_shuffle_perms": token_shuffle_perms,
                                    "projector_sharing": "shared",
                                    "gamma": gamma_value,
                                }
                            )
                else:
                    for llm_layer in self.llm_layers:
                        projected = self.branches[str(int(llm_layer))](aligned_tokens)
                        gamma_value = None
                        if self.preagg_use_layer_gamma:
                            gamma = self.preagg_layer_gammas[str(int(llm_layer))].to(
                                device=projected.device,
                                dtype=projected.dtype,
                            )
                            gamma_value = float(gamma.detach().float().item())
                            projected = projected * gamma
                        projected = projected * self.residual_scale
                        residuals[int(llm_layer)][batch_idx, target_indices] = projected
                        residual_norms_for_log[int(llm_layer)] = float(projected.detach().float().norm().item())
                        if gamma_value is not None:
                            gamma_for_log[int(llm_layer)] = gamma_value
                        if should_debug_sample:
                            debug["layers"].setdefault(str(llm_layer), []).append(
                                {
                                    "sample_id": int(batch_idx),
                                    "cut3r_layer": self.preagg_output_layer_key,
                                    "preagg_layers": list(self.preagg_layers),
                                    "raw_token_counts": raw_counts,
                                    "aligned_token_counts": aligned_counts,
                                    "target_grid_shapes": target_grid_shapes,
                                    "residual_norm": float(projected.detach().float().norm().item()),
                                    "frame_source_order": list(frame_source_order),
                                    "token_shuffle_perms": token_shuffle_perms,
                                    "projector_sharing": "layer_specific",
                                    "gamma": gamma_value,
                                }
                            )
                if should_debug_sample and self.log_first_n != 0:
                    extra = ""
                    if self.preagg_type == "weighted_sum" and self.preagg_log_weights:
                        extra = (
                            f", raw_scalar_logits={preagg_debug.get('raw_scalar_logits')}, "
                            f"softmax_weights={preagg_debug.get('softmax_weights')}"
                        )
                    elif self.preagg_type == "concat_linear":
                        extra = (
                            f", concat_input_dim={preagg_debug.get('concat_input_dim')}, "
                            f"aggregation_output_dim={preagg_debug.get('aggregation_output_dim')}, "
                            f"aggregation_weight_norm={preagg_debug.get('aggregation_weight_norm'):.6f}"
                        )
                    _rank0_print(
                        "[CUT3R SpatialStack PreAgg] "
                        f"preagg_enable={self.preagg_enable}, "
                        f"preagg_layers={self.preagg_layers}, "
                        f"preagg_type={self.preagg_type}, "
                        f"preagg_projector_sharing={self.preagg_projector_sharing}, "
                        f"selected_target_llm_layers={self.llm_layers}, "
                        f"projector_type={self.projector_type}, "
                        f"feature_dim={self.feature_dim}, hidden_size={self.hidden_size}, "
                        f"input_layer_feature_shapes={debug.get('preagg_input_layer_shapes')}, "
                        f"aggregated_feature_shape={list(aggregated_tokens.shape)}, "
                        f"visual_token_count={int(visual_indices.numel())}, "
                        f"target_grid_shapes={target_grid_shapes}, "
                        f"residual_norms={residual_norms_for_log}, "
                        f"gammas={gamma_for_log}"
                        f"{extra}"
                    )
                continue

            for llm_layer, cut3r_layer in self.layer_map.items():
                patch_tokens = self._extract_layer_tokens(sidecar, cut3r_layer).to(device=device, dtype=dtype)
                sidecar_frame_count = int(patch_tokens.shape[0])
                if sidecar_frame_lookup is not None:
                    token_frame_indices = [sidecar_frame_lookup[int(frame_id)] for frame_id in frame_order]
                elif sidecar_frame_count == len(frame_order):
                    token_frame_indices = list(range(len(frame_order)))
                elif frame_order and max(int(frame_id) for frame_id in frame_order) < sidecar_frame_count:
                    token_frame_indices = [int(frame_id) for frame_id in frame_order]
                else:
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame count mismatch for sample {batch_idx}, layer {cut3r_layer}: "
                        f"sidecar frames={sidecar_frame_count}, visual frame_order={frame_order}."
                    )
                aligned_frames = []
                aligned_indices = []
                raw_counts = []
                aligned_counts = []
                target_grid_shapes = []
                token_shuffle_perms = []
                for local_frame_idx, frame_id in enumerate(frame_order):
                    frame_mask = frame_ids == int(frame_id)
                    frame_visual_indices = visual_indices[frame_mask]
                    target_count = int(frame_visual_indices.numel())
                    if target_count == 0:
                        continue
                    grid_shape = self._grid_shape_at(metadata, local_frame_idx)
                    if grid_shape is not None:
                        grid_h, grid_w = grid_shape
                        if self.projector_type == "token_mlp" and grid_h != grid_w:
                            raise RuntimeError(
                                f"CUT3R SpatialStack requires square visual grids for sample {batch_idx}, "
                                f"frame {frame_id}; got visual_grid_shapes[{local_frame_idx}]={grid_shape}."
                            )
                        if grid_h * grid_w != target_count:
                            raise RuntimeError(
                                f"CUT3R SpatialStack visual token count mismatch for sample {batch_idx}, "
                                f"frame {frame_id}: visual_grid_shape={grid_shape} implies {grid_h * grid_w} "
                                f"tokens, but visual metadata has {target_count} positions."
                            )
                    source_local_frame_idx = int(frame_source_order[local_frame_idx])
                    raw_frame_tokens = patch_tokens[token_frame_indices[source_local_frame_idx]]
                    if self.projector_type == "merge_mlp":
                        if grid_shape is None:
                            raise RuntimeError(
                                "CUT3R SpatialStack merge_mlp requires visual_grid_shapes metadata; "
                                f"missing sample {batch_idx}, frame {frame_id}."
                            )
                        aligned = self.merge_frame_grid(raw_frame_tokens, grid_shape)
                    else:
                        aligned = self.resize_square_grid(raw_frame_tokens, target_count)
                    aligned, token_perm = self._maybe_shuffle_frame_tokens(aligned, sample_index, local_frame_idx)
                    if int(aligned.shape[0]) != target_count:
                        raise RuntimeError(
                            f"CUT3R SpatialStack aligned token count mismatch for sample {batch_idx}, "
                            f"frame {frame_id}, projector_type={self.projector_type}: "
                            f"aligned={int(aligned.shape[0])}, visual={target_count}."
                        )
                    aligned_frames.append(aligned)
                    aligned_indices.append(frame_visual_indices)
                    raw_counts.append(int(raw_frame_tokens.shape[0]))
                    aligned_counts.append(int(aligned.shape[0]))
                    target_grid_shapes.append(tuple(int(x) for x in grid_shape) if grid_shape is not None else None)
                    if token_perm is not None and len(token_shuffle_perms) < 3:
                        token_shuffle_perms.append(
                            {
                                "frame_id": int(frame_id),
                                "perm": token_perm,
                            }
                        )
                if not aligned_frames:
                    continue
                aligned_tokens = torch.cat(aligned_frames, dim=0)
                target_indices = torch.cat(aligned_indices, dim=0)
                projected = self.branches[str(cut3r_layer)](aligned_tokens)
                projected = projected * self.residual_scale
                residuals[int(llm_layer)][batch_idx, target_indices] = projected
                if should_debug_sample:
                    debug["layers"].setdefault(str(llm_layer), []).append(
                        {
                            "sample_id": int(batch_idx),
                            "cut3r_layer": int(cut3r_layer),
                            "raw_token_counts": raw_counts,
                            "aligned_token_counts": aligned_counts,
                            "target_grid_shapes": target_grid_shapes,
                            "residual_norm": float(projected.detach().float().norm().item()),
                            "frame_source_order": list(frame_source_order),
                            "token_shuffle_perms": token_shuffle_perms,
                        }
                    )

        self.last_debug = debug
        return residuals

    def prepare_cross_attn_inputs(
        self,
        spatial_features,
        visual_metadata,
        *,
        seq_len: int,
        device,
        dtype,
    ) -> Dict[int, dict]:
        metadata_items = self._metadata_items(visual_metadata)
        batch_size = len(metadata_items)
        feature_items = self._feature_items(spatial_features, batch_size)
        self._ensure_module_dtype(device, dtype)

        prepared = {
            int(llm_layer): {
                "cut3r_layer": int(cut3r_layer),
                "same_frame_only": bool(self.cross_attn_same_frame_only),
                "frames": [],
                "selected_cut3r_layers": list(self.cut3r_layers),
                "selected_llm_layers": list(self.llm_layers),
            }
            for llm_layer, cut3r_layer in self.layer_map.items()
        }
        debug = {
            "fusion_type": self.fusion_type,
            "selected_cut3r_layers": list(self.cut3r_layers),
            "selected_llm_layers": list(self.llm_layers),
            "feature_dim": int(self.feature_dim),
            "hidden_size": int(self.hidden_size),
            "same_frame_only": bool(self.cross_attn_same_frame_only),
            "cross_attn_heads": int(self.cross_attn_heads),
            "cross_attn_dropout": float(self.cross_attn_dropout),
            "cross_attn_zero_init": bool(self.cross_attn_zero_init),
            "cross_attn_impl": self.cross_attn_impl,
            "patch_align": self.cross_attn_patch_align,
            "use_camera_tokens": bool(self.cross_attn_use_camera_tokens),
            "require_camera_tokens": bool(self.require_camera_tokens),
            "cross_attn_use_mlp": bool(self.cross_attn_use_mlp),
            "cross_attn_norm_type": self.cross_attn_norm_type,
            "cross_attn_pos_embed": self.cross_attn_pos_embed,
            "cross_attn_gamma_attn_init": float(self.cross_attn_gamma_attn_init),
            "cross_attn_gamma_mlp_init": float(self.cross_attn_gamma_mlp_init),
            "cross_attn_v2_force_zero_gamma_at_eval": bool(self.cross_attn_v2_force_zero_gamma_at_eval),
            "samples": [],
            "layers": {},
        }

        for batch_idx, (metadata, sidecar) in enumerate(zip(metadata_items, feature_items)):
            visual_indices, frame_ids, frame_order = self._validate_visual_metadata(metadata, batch_idx, device)
            if int(visual_indices.numel()) == 0:
                continue
            if int(visual_indices.min().item()) < 0 or int(visual_indices.max().item()) >= int(seq_len):
                bad_min = int(visual_indices.min().item())
                bad_max = int(visual_indices.max().item())
                raise RuntimeError(
                    f"CUT3R SpatialStack metadata[{batch_idx}] visual_token_indices out of bounds for "
                    f"seq_len={int(seq_len)}: min={bad_min}, max={bad_max}."
                )
            sidecar_frame_indices = self._sidecar_frame_indices(sidecar)
            sidecar_frame_lookup = None
            if sidecar_frame_indices is not None and sidecar_frame_indices != frame_order:
                sidecar_frame_lookup = {int(frame_id): idx for idx, frame_id in enumerate(sidecar_frame_indices)}
                missing_frame_ids = [int(frame_id) for frame_id in frame_order if int(frame_id) not in sidecar_frame_lookup]
                if missing_frame_ids:
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame_indices mismatch for sample {batch_idx}: "
                        f"visual frame_order={frame_order}, sidecar frame_indices={sidecar_frame_indices}; "
                        f"missing visual frames={missing_frame_ids}."
                    )
            if sidecar_frame_indices is None and frame_order != list(range(len(frame_order))):
                raise RuntimeError(
                    f"CUT3R SpatialStack sidecar for sample {batch_idx} lacks frame_indices/frame_order, "
                    f"but visual frame_order={frame_order} is not the unambiguous default order."
                )
            should_debug_sample = self.log_first_n < 0 or len(debug["samples"]) < self.log_first_n
            if should_debug_sample:
                debug["samples"].append(
                    {
                        "sample_id": int(batch_idx),
                        "visual_token_count": int(visual_indices.numel()),
                        "frame_order": list(frame_order),
                    }
                )
            sample_index = int(self._shuffle_sample_count)
            if self.frame_shuffle or self.token_shuffle:
                self._shuffle_sample_count += 1
            frame_source_order, frame_shuffle_perm = self._frame_source_order(len(frame_order), device, sample_index)
            if should_debug_sample and frame_shuffle_perm is not None:
                debug["samples"][-1]["cut3r_spatialstack_frame_shuffle_perm"] = frame_shuffle_perm

            for llm_layer, cut3r_layer in self.layer_map.items():
                patch_tokens = self._extract_layer_tokens(sidecar, cut3r_layer).to(device=device, dtype=dtype)
                camera_tokens = None
                if self.fusion_type == "cross_attn_v2" and self.cross_attn_use_camera_tokens:
                    camera_tokens = self._extract_layer_camera_tokens(sidecar, cut3r_layer).to(device=device, dtype=dtype)
                sidecar_frame_count = int(patch_tokens.shape[0])
                if camera_tokens is not None and int(camera_tokens.shape[0]) != sidecar_frame_count:
                    raise RuntimeError(
                        f"CUT3R SpatialStack camera/patch frame count mismatch for layer {cut3r_layer}: "
                        f"camera={int(camera_tokens.shape[0])}, patch={sidecar_frame_count}."
                    )
                if sidecar_frame_lookup is not None:
                    token_frame_indices = [sidecar_frame_lookup[int(frame_id)] for frame_id in frame_order]
                elif sidecar_frame_count == len(frame_order):
                    token_frame_indices = list(range(len(frame_order)))
                elif frame_order and max(int(frame_id) for frame_id in frame_order) < sidecar_frame_count:
                    token_frame_indices = [int(frame_id) for frame_id in frame_order]
                else:
                    raise RuntimeError(
                        f"CUT3R SpatialStack frame count mismatch for sample {batch_idx}, layer {cut3r_layer}: "
                        f"sidecar frames={sidecar_frame_count}, visual frame_order={frame_order}."
                    )

                frame_entries = []
                raw_counts = []
                aligned_counts = []
                visual_counts = []
                target_grid_shapes = []
                token_shuffle_perms = []
                for local_frame_idx, frame_id in enumerate(frame_order):
                    frame_mask = frame_ids == int(frame_id)
                    frame_visual_indices = visual_indices[frame_mask]
                    target_count = int(frame_visual_indices.numel())
                    if target_count == 0:
                        continue
                    grid_shape = self._grid_shape_at(metadata, local_frame_idx)
                    if grid_shape is not None:
                        grid_h, grid_w = grid_shape
                        if self.fusion_type == "cross_attn" and grid_h != grid_w:
                            raise RuntimeError(
                                f"CUT3R SpatialStack requires square visual grids for sample {batch_idx}, "
                                f"frame {frame_id}; got visual_grid_shapes[{local_frame_idx}]={grid_shape}."
                            )
                        if grid_h * grid_w != target_count:
                            raise RuntimeError(
                                f"CUT3R SpatialStack visual token count mismatch for sample {batch_idx}, "
                                f"frame {frame_id}: visual_grid_shape={grid_shape} implies {grid_h * grid_w} "
                                f"tokens, but visual metadata has {target_count} positions."
                            )
                    source_local_frame_idx = int(frame_source_order[local_frame_idx])
                    source_frame_idx = token_frame_indices[source_local_frame_idx]
                    raw_frame_tokens = patch_tokens[source_frame_idx]
                    if self.fusion_type == "cross_attn_v2" and self.cross_attn_patch_align == "merge":
                        if grid_shape is None:
                            raise RuntimeError(
                                "CUT3R SpatialStack cross_attn_v2 merge alignment requires visual_grid_shapes metadata; "
                                f"missing sample {batch_idx}, frame {frame_id}."
                            )
                        aligned = self.merge_frame_grid(raw_frame_tokens, grid_shape)
                    elif grid_shape is not None:
                        aligned = self.resize_grid(raw_frame_tokens, int(grid_shape[0]), int(grid_shape[1]))
                    else:
                        aligned = self.resize_square_grid(raw_frame_tokens, target_count)
                    aligned, token_perm = self._maybe_shuffle_frame_tokens(aligned, sample_index, local_frame_idx)
                    if int(aligned.shape[0]) != target_count:
                        raise RuntimeError(
                            f"CUT3R SpatialStack cross-attn aligned token count mismatch for sample {batch_idx}, "
                            f"frame {frame_id}, fusion_type={self.fusion_type}, patch_align={self.cross_attn_patch_align}: "
                            f"aligned={int(aligned.shape[0])}, visual={target_count}."
                        )
                    frame_entry = {
                        "batch_idx": int(batch_idx),
                        "frame_id": int(frame_id),
                        "visual_indices": frame_visual_indices.detach(),
                        "geometry_tokens": aligned.detach(),
                    }
                    if self.fusion_type == "cross_attn_v2":
                        if camera_tokens is not None:
                            frame_entry["camera_tokens"] = camera_tokens[source_frame_idx].detach()
                        elif self.require_camera_tokens:
                            raise RuntimeError(
                                f"CUT3R SpatialStack cross_attn_v2 requires camera_tokens for decoder layer {cut3r_layer}."
                            )
                        frame_entry["visual_grid_shape"] = tuple(int(x) for x in grid_shape) if grid_shape is not None else None
                        frame_entry["geometry_grid_shape"] = tuple(int(x) for x in grid_shape) if grid_shape is not None else None
                        frame_entry["patch_align"] = self.cross_attn_patch_align
                    frame_entries.append(
                        frame_entry
                    )
                    raw_counts.append(int(raw_frame_tokens.shape[0]))
                    aligned_counts.append(int(aligned.shape[0]))
                    visual_counts.append(int(target_count))
                    target_grid_shapes.append(tuple(int(x) for x in grid_shape) if grid_shape is not None else None)
                    if token_perm is not None and len(token_shuffle_perms) < 3:
                        token_shuffle_perms.append(
                            {
                                "frame_id": int(frame_id),
                                "perm": token_perm,
                            }
                        )
                if not frame_entries:
                    continue
                if self.cross_attn_same_frame_only:
                    prepared[int(llm_layer)]["frames"].extend(frame_entries)
                else:
                    prepared[int(llm_layer)]["frames"].append(
                        {
                            "batch_idx": int(batch_idx),
                            "frame_id": "all",
                            "visual_indices": torch.cat([entry["visual_indices"] for entry in frame_entries], dim=0).detach(),
                            "geometry_tokens": torch.cat([entry["geometry_tokens"] for entry in frame_entries], dim=0).detach(),
                        }
                    )
                if should_debug_sample:
                    debug["layers"].setdefault(str(llm_layer), []).append(
                        {
                            "sample_id": int(batch_idx),
                            "cut3r_layer": int(cut3r_layer),
                            "raw_token_counts": raw_counts,
                            "aligned_token_counts": aligned_counts,
                            "visual_token_counts": visual_counts,
                            "target_grid_shapes": target_grid_shapes,
                            "frame_source_order": list(frame_source_order),
                            "token_shuffle_perms": token_shuffle_perms,
                            "same_frame_only": bool(self.cross_attn_same_frame_only),
                            "patch_align": self.cross_attn_patch_align,
                            "camera_tokens_present": bool(camera_tokens is not None),
                            "camera_token_shape": list(camera_tokens.shape) if camera_tokens is not None else None,
                            "patch_token_shape": list(patch_tokens.shape),
                        }
                    )

        self.last_debug = debug
        return prepared

    def apply_cross_attn_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_inputs: dict,
        *,
        cached_decode_skip_count: int = 0,
        collect_stats: bool = True,
    ):
        if self.fusion_type not in {"cross_attn", "cross_attn_v2"}:
            raise RuntimeError(
                "CUT3R SpatialStack cross-attn update requested while fusion_type is not "
                "'cross_attn' or 'cross_attn_v2'."
            )
        block_key = str(int(layer_idx))
        if block_key not in self.cross_attn_blocks:
            raise RuntimeError(f"CUT3R SpatialStack has no cross-attn block for LLM layer {int(layer_idx)}.")
        block = self.cross_attn_blocks[block_key]
        frames = layer_inputs.get("frames", []) if isinstance(layer_inputs, dict) else []
        if not frames:
            return hidden_states, None

        updated = hidden_states.clone()
        output_deltas = [] if collect_stats else None
        visual_counts = [] if collect_stats else None
        geometry_counts = [] if collect_stats else None
        frame_ids = [] if collect_stats else None
        v2_block_stats = [] if collect_stats and self.fusion_type == "cross_attn_v2" else None
        before_norm = hidden_states.detach().float().norm().item() if collect_stats else None
        grouped_entries = {}
        for entry in frames:
            batch_idx = int(entry["batch_idx"])
            visual_indices = entry["visual_indices"].to(device=hidden_states.device, dtype=torch.long)
            geometry_tokens = entry["geometry_tokens"].to(device=hidden_states.device, dtype=hidden_states.dtype)
            if int(visual_indices.numel()) == 0 or int(geometry_tokens.shape[0]) == 0:
                continue
            if batch_idx < 0 or batch_idx >= int(hidden_states.shape[0]):
                raise RuntimeError(
                    f"CUT3R SpatialStack cross-attn batch_idx out of bounds at LLM layer {int(layer_idx)}: "
                    f"batch_idx={batch_idx}, batch_size={int(hidden_states.shape[0])}."
                )
            if self.fusion_type == "cross_attn_v2":
                camera_tokens = entry.get("camera_tokens", None)
                if self.cross_attn_use_camera_tokens:
                    if camera_tokens is None:
                        raise RuntimeError(
                            "CUT3R SpatialStack cross_attn_v2 payload is missing camera_tokens; "
                            "check sidecar extraction and cut3r_spatialstack_require_camera_tokens."
                        )
                    camera_tokens = camera_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype)
                visual_grid_shape = entry.get("visual_grid_shape", None)
                geometry_grid_shape = entry.get("geometry_grid_shape", None)
                key = (
                    int(visual_indices.numel()),
                    int(geometry_tokens.shape[0]),
                    int(geometry_tokens.shape[-1]),
                    tuple(visual_grid_shape) if visual_grid_shape is not None else None,
                    tuple(geometry_grid_shape) if geometry_grid_shape is not None else None,
                    1 if camera_tokens is not None else 0,
                )
            else:
                camera_tokens = None
                visual_grid_shape = None
                geometry_grid_shape = None
                key = (int(visual_indices.numel()), int(geometry_tokens.shape[0]))
            grouped_entries.setdefault(key, []).append(
                {
                    "batch_idx": batch_idx,
                    "visual_indices": visual_indices,
                    "geometry_tokens": geometry_tokens,
                    "camera_tokens": camera_tokens,
                    "visual_grid_shape": visual_grid_shape,
                    "geometry_grid_shape": geometry_grid_shape,
                    "frame_id": entry.get("frame_id"),
                }
            )

        applied_any = False
        for group in grouped_entries.values():
            visual_hidden = torch.stack(
                [hidden_states[entry["batch_idx"], entry["visual_indices"]] for entry in group],
                dim=0,
            )
            geometry_tokens = torch.stack([entry["geometry_tokens"] for entry in group], dim=0)
            if self.fusion_type == "cross_attn_v2":
                camera_tokens = None
                if self.cross_attn_use_camera_tokens:
                    camera_tokens = torch.stack([entry["camera_tokens"] for entry in group], dim=0)
                first_entry = group[0]
                block_result = block(
                    visual_hidden,
                    geometry_tokens,
                    camera_tokens,
                    visual_grid_shape=first_entry.get("visual_grid_shape"),
                    geometry_grid_shape=first_entry.get("geometry_grid_shape"),
                    return_stats=collect_stats,
                )
                if collect_stats:
                    deltas, block_stats = block_result
                    v2_block_stats.append(block_stats)
                else:
                    deltas = block_result
            else:
                deltas = block(visual_hidden, geometry_tokens)
            for row_idx, entry in enumerate(group):
                updated[entry["batch_idx"], entry["visual_indices"]] = visual_hidden[row_idx] + deltas[row_idx]
            applied_any = True
            if collect_stats:
                output_deltas.append(deltas.detach().float().reshape(-1, deltas.shape[-1]))
                for entry in group:
                    visual_counts.append(int(entry["visual_indices"].numel()))
                    geometry_counts.append(int(entry["geometry_tokens"].shape[0]))
                    frame_ids.append(entry["frame_id"])

        if not applied_any:
            return hidden_states, None
        if not collect_stats:
            return updated, None
        output_norm = torch.cat(output_deltas, dim=0).norm().item()
        stat = {
            "fusion_type": self.fusion_type,
            "layer_idx": int(layer_idx),
            "cut3r_layer": int(layer_inputs.get("cut3r_layer", -1)),
            "use_cut3r_spatialstack": True,
            "selected_cut3r_layers": list(layer_inputs.get("selected_cut3r_layers", self.cut3r_layers)),
            "selected_llm_layers": list(layer_inputs.get("selected_llm_layers", self.llm_layers)),
            "same_frame_only": bool(layer_inputs.get("same_frame_only", self.cross_attn_same_frame_only)),
            "frame_ids": frame_ids,
            "visual_tokens_per_frame": visual_counts,
            "geometry_tokens_per_frame": geometry_counts,
            "cross_attn_output_norm": float(output_norm),
            "hidden_norm_before": float(before_norm),
            "hidden_norm_after": float(updated.detach().float().norm().item()),
            "output_projection_zero_initialized": bool(self.cross_attn_zero_init),
            "cached_decode_skip_count": int(cached_decode_skip_count),
        }
        if self.fusion_type == "cross_attn_v2" and v2_block_stats:
            numeric_keys = [
                "camera_attention_mass",
                "patch_attention_mass",
                "attention_entropy",
                "diagonal_attention_mass",
                "local_3x3_attention_mass",
                "cross_attn_v2_force_zero_gamma_at_eval",
                "learned_gamma_attn",
                "learned_gamma_mlp",
                "effective_gamma_attn",
                "effective_gamma_mlp",
                "gamma_attn",
                "gamma_mlp",
                "attn_out_norm",
                "mlp_out_norm",
                "delta_norm",
                "hidden_norm",
                "attn_out_to_hidden_norm",
                "mlp_out_to_hidden_norm",
                "delta_to_hidden_norm",
            ]
            for key in numeric_keys:
                values = [float(item[key]) for item in v2_block_stats if key in item]
                if values:
                    stat[key] = float(sum(values) / len(values))
            stat.update(
                {
                    "patch_align": self.cross_attn_patch_align,
                    "use_camera_tokens": bool(self.cross_attn_use_camera_tokens),
                    "require_camera_tokens": bool(self.require_camera_tokens),
                    "cross_attn_impl": self.cross_attn_impl,
                    "cross_attn_norm_type": self.cross_attn_norm_type,
                    "cross_attn_pos_embed": self.cross_attn_pos_embed,
                    "cross_attn_use_mlp": bool(self.cross_attn_use_mlp),
                    "cross_attn_v2_force_zero_gamma_at_eval": bool(self.cross_attn_v2_force_zero_gamma_at_eval),
                    "output_projection_zero_initialized": False,
                    "geo_memory_shapes": [
                        item.get("geo_memory_shape")
                        for item in v2_block_stats
                        if "geo_memory_shape" in item
                    ],
                    "final_delta_norm": float(output_norm),
                }
            )
        self._maybe_log_cross_attn_stat(stat)
        return updated, stat

    def _maybe_log_cross_attn_stat(self, stat: dict):
        if self.log_first_n == 0:
            return
        if self.log_first_n > 0 and self._cross_attn_log_count >= self.log_first_n:
            if self._cross_attn_log_count == self.log_first_n:
                _rank0_print("[CUT3R SpatialStack CrossAttn] Further per-layer logs suppressed.")
            self._cross_attn_log_count += 1
            return
        _rank0_print(
            "[CUT3R SpatialStack CrossAttn] "
            f"use_cut3r_spatialstack={stat['use_cut3r_spatialstack']}, "
            f"fusion_type={stat['fusion_type']}, "
            f"llm_layer={stat['layer_idx']}, cut3r_layer={stat['cut3r_layer']}, "
            f"selected_cut3r_layers={stat['selected_cut3r_layers']}, "
            f"selected_llm_layers={stat['selected_llm_layers']}, "
            f"same_frame_only={stat['same_frame_only']}, "
            f"visual_tokens_per_frame={stat['visual_tokens_per_frame']}, "
            f"geometry_tokens_per_frame={stat['geometry_tokens_per_frame']}, "
            f"cross_attn_output_norm={stat['cross_attn_output_norm']:.6f}, "
            f"hidden_norm_before={stat['hidden_norm_before']:.6f}, "
            f"hidden_norm_after={stat['hidden_norm_after']:.6f}, "
            f"output_projection_zero_initialized={stat['output_projection_zero_initialized']}, "
            f"cached_decode_skip_count={stat['cached_decode_skip_count']}"
            + (
                f", patch_align={stat.get('patch_align')}, "
                f"use_camera_tokens={stat.get('use_camera_tokens')}, "
                f"cross_attn_v2_force_zero_gamma_at_eval={stat.get('cross_attn_v2_force_zero_gamma_at_eval')}, "
                f"learned_gamma_attn={stat.get('learned_gamma_attn'):.6f}, "
                f"learned_gamma_mlp={stat.get('learned_gamma_mlp'):.6f}, "
                f"effective_gamma_attn={stat.get('effective_gamma_attn'):.6f}, "
                f"effective_gamma_mlp={stat.get('effective_gamma_mlp'):.6f}, "
                f"camera_attention_mass={stat.get('camera_attention_mass'):.6f}, "
                f"patch_attention_mass={stat.get('patch_attention_mass'):.6f}, "
                f"attention_entropy={stat.get('attention_entropy'):.6f}, "
                f"diagonal_attention_mass={stat.get('diagonal_attention_mass'):.6f}, "
                f"local_3x3_attention_mass={stat.get('local_3x3_attention_mass'):.6f}, "
                f"final_delta_norm={stat.get('final_delta_norm'):.6f}, "
                f"geo_memory_shapes={stat.get('geo_memory_shapes')}"
                if stat.get("fusion_type") == "cross_attn_v2"
                else ""
            )
        )
        self._cross_attn_log_count += 1
