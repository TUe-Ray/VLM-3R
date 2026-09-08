"""Deterministic structured isometries for C1 pre-SFT fusion probing.

The routines in this module intentionally never inspect or mutate any random
number-generator state.  They implement the C1 matrix family directly from
normalized Sylvester Hadamard blocks and a fixed perfect shuffle.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


SCHEME_NAME = "structured_hadamard_perfect_shuffle"
SCHEME_VERSION = "c1_structured_isometry_v1"
QK_BASIS_MODES = ("shared_canonical", "role_offset")

_SQUARE_CACHE: dict[int, torch.Tensor] = {}
_RECTANGULAR_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _validate_dimension(dimension: int) -> int:
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError(f"Structured isometry dimension must be positive, got {dimension}.")
    return dimension


def structured_block_size(dimension: int) -> int:
    """Largest power-of-two block (at most 512) dividing ``dimension``."""
    dimension = _validate_dimension(dimension)
    block = 512
    while block > 1 and dimension % block:
        block //= 2
    if dimension % block:
        raise ValueError(
            f"Dimension {dimension} cannot use the C1 Hadamard block construction."
        )
    return block


def perfect_shuffle_indices(dimension: int) -> torch.Tensor:
    """Return the fixed C1 perfect-shuffle output ordering on CPU."""
    dimension = _validate_dimension(dimension)
    block = structured_block_size(dimension)
    num_blocks = dimension // block
    return torch.arange(dimension, dtype=torch.long).reshape(num_blocks, block).transpose(0, 1).reshape(-1)


def _block_hadamard_right(values: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply blockdiag(H/sqrt(H)) on the final dimension of ``values``."""
    if values.dtype != torch.float32:
        raise TypeError(f"C1 Hadamard construction requires float32, got {values.dtype}.")
    total = int(values.shape[-1])
    if total % block_size:
        raise ValueError(f"Last dimension {total} is not divisible by block size {block_size}.")
    num_blocks = total // block_size
    prefix = values.shape[:-1]
    result = values.reshape(*prefix, num_blocks, block_size)
    width = 1
    while width < block_size:
        result = result.reshape(*prefix, num_blocks, block_size // (2 * width), 2, width)
        left, right = result.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape(*prefix, num_blocks, block_size)
        width *= 2
    return result.reshape(*prefix, total) * (1.0 / math.sqrt(float(block_size)))


def _perfect_shuffle_right(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Apply the fixed channel shuffle to row vectors in ``values``."""
    block = structured_block_size(dimension)
    num_blocks = dimension // block
    return values.reshape(*values.shape[:-1], num_blocks, block).transpose(-2, -1).reshape(*values.shape[:-1], dimension)


def canonical_square(dimension: int) -> torch.Tensor:
    """Return ``U_d = B_d @ P_d @ B_d`` as a cached CPU float32 tensor."""
    dimension = _validate_dimension(dimension)
    cached = _SQUARE_CACHE.get(dimension)
    if cached is not None:
        return cached
    block = structured_block_size(dimension)
    # Starting from I and transforming row vectors materializes exactly B P B
    # without a cubic dense matrix multiplication.
    matrix = torch.eye(dimension, dtype=torch.float32)
    matrix = _block_hadamard_right(matrix, block)
    matrix = _perfect_shuffle_right(matrix, dimension)
    matrix = _block_hadamard_right(matrix, block).contiguous()
    _SQUARE_CACHE[dimension] = matrix
    return matrix


def canonical_linear_weight(d_in: int, d_out: int) -> torch.Tensor:
    """Return a deterministic semi-isometry shaped for ``nn.Linear.weight``.

    The returned tensor has shape ``[d_out, d_in]``.  Expansion satisfies
    ``W.T @ W = I`` and contraction satisfies ``W @ W.T = I`` in FP32.
    """
    d_in = _validate_dimension(d_in)
    d_out = _validate_dimension(d_out)
    key = (d_in, d_out)
    cached = _RECTANGULAR_CACHE.get(key)
    if cached is not None:
        return cached
    if d_in == d_out:
        weight = canonical_square(d_in)
    elif d_out > d_in:
        weight = canonical_square(d_out)[:, :d_in] @ canonical_square(d_in).transpose(0, 1)
    else:
        weight = canonical_square(d_out) @ canonical_square(d_in)[:d_out, :]
    weight = weight.contiguous()
    _RECTANGULAR_CACHE[key] = weight
    return weight


def role_offset_weight(weight: torch.Tensor) -> torch.Tensor:
    """Apply the optional deterministic output-channel offset used only for K."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a linear weight, got shape {tuple(weight.shape)}.")
    return weight.index_select(0, perfect_shuffle_indices(int(weight.shape[0])))


def validate_qk_basis_mode(mode: str) -> str:
    mode = str(mode or "shared_canonical").strip().lower()
    if mode not in QK_BASIS_MODES:
        raise ValueError(f"qk_basis_mode must be one of {QK_BASIS_MODES}, got {mode!r}.")
    return mode


def copy_linear_weight(linear: nn.Linear, weight: torch.Tensor, *, transpose: bool = False) -> None:
    expected = (int(linear.out_features), int(linear.in_features))
    value = weight.transpose(0, 1) if transpose else weight
    if tuple(value.shape) != expected:
        raise ValueError(
            f"Canonical weight shape mismatch for {linear}: expected {expected}, got {tuple(value.shape)}."
        )
    with torch.no_grad():
        linear.weight.copy_(value.to(device=linear.weight.device, dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


def set_norm_defaults(module: nn.Module) -> None:
    """Set affine normalization parameters to the C1 canonical defaults."""
    with torch.no_grad():
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if isinstance(weight, torch.Tensor):
            weight.fill_(1.0)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def set_mha_identity(mha: nn.MultiheadAttention) -> None:
    """Set a standard packed MHA to exact identity Q/K/V and output maps."""
    if not getattr(mha, "_qkv_same_embed_dim", False):
        raise ValueError("C1 supports only packed same-dimension nn.MultiheadAttention.")
    dim = int(mha.embed_dim)
    if tuple(mha.in_proj_weight.shape) != (3 * dim, dim):
        raise ValueError(f"Unexpected MHA in_proj_weight shape {tuple(mha.in_proj_weight.shape)}.")
    with torch.no_grad():
        identity = torch.eye(dim, device=mha.in_proj_weight.device, dtype=mha.in_proj_weight.dtype)
        mha.in_proj_weight.copy_(torch.cat((identity, identity, identity), dim=0))
        if mha.in_proj_bias is not None:
            mha.in_proj_bias.zero_()
        mha.out_proj.weight.copy_(identity.to(dtype=mha.out_proj.weight.dtype))
        if mha.out_proj.bias is not None:
            mha.out_proj.bias.zero_()


def matrix_scheme_metadata(*, dimensions: list[int] | tuple[int, ...], qk_basis_mode: str) -> dict[str, Any]:
    return {
        "scheme": SCHEME_NAME,
        "version": SCHEME_VERSION,
        "qk_basis_mode": validate_qk_basis_mode(qk_basis_mode),
        "dimensions": {
            str(int(dimension)): {
                "hadamard_block_size": structured_block_size(int(dimension)),
                "num_blocks": int(dimension) // structured_block_size(int(dimension)),
            }
            for dimension in sorted({int(value) for value in dimensions})
        },
    }


def apply_spatialstack_c1(merger: nn.Module, *, qk_basis_mode: str = "shared_canonical") -> None:
    """Replace a native SpatialStack V1/additive module with C1 weights.

    The underlying modules and their injection topology are intentionally
    untouched.  Only from-scratch affine parameters and non-trainable C1
    scalar buffers are set here.  Callers set calibrated scalar values later.
    """
    mode = validate_qk_basis_mode(qk_basis_mode)
    fusion_type = str(getattr(merger, "fusion_type", "")).lower()
    feature_dim = int(getattr(merger, "feature_dim", -1))
    hidden_size = int(getattr(merger, "hidden_size", -1))
    if feature_dim != 768 or hidden_size != 3584:
        raise ValueError(
            "C1 SpatialStack is defined for CUT3R=768 and LLM hidden=3584, "
            f"got feature_dim={feature_dim}, hidden_size={hidden_size}."
        )
    if fusion_type == "add":
        if bool(getattr(merger, "preagg_enable", False)):
            raise ValueError("C1 additive SpatialStack does not define pre-aggregation.")
        if str(getattr(merger, "projector_type", "")) != "token_mlp":
            raise ValueError("C1 additive SpatialStack requires projector_type='token_mlp'.")
        in_weight = canonical_linear_weight(feature_dim, hidden_size)
        out_weight = canonical_square(hidden_size).transpose(0, 1).contiguous()
        for branch in merger.branches.values():
            if not isinstance(branch, nn.Module) or not hasattr(branch, "proj_in"):
                raise ValueError("C1 additive branch is missing the native token-MLP projections.")
            copy_linear_weight(branch.proj_in, in_weight)
            copy_linear_weight(branch.proj_out, out_weight)
            set_norm_defaults(branch.norm)
            branch.set_c1_state(enabled=True, pre_gelu_scale=1.0, residual_gain=0.0)
        return
    if fusion_type != "cross_attn":
        raise ValueError(
            "C1 only supports SpatialStack additive or V1 cross_attn; "
            f"got fusion_type={fusion_type!r}."
        )
    if bool(getattr(merger, "cross_attn_use_camera_tokens", False)):
        raise ValueError("C1 SS cross-attention V1 must not use camera tokens.")
    if bool(getattr(merger, "cross_attn_use_mlp", False)):
        raise ValueError("C1 SS cross-attention V1 must not contain a branch FFN.")
    q_weight = canonical_square(hidden_size)
    kv_weight = canonical_linear_weight(feature_dim, hidden_size)
    k_weight = role_offset_weight(kv_weight) if mode == "role_offset" else kv_weight
    out_weight = q_weight.transpose(0, 1).contiguous()
    for block in merger.cross_attn_blocks.values():
        if int(getattr(block, "num_heads", -1)) != 28 or int(getattr(block, "head_dim", -1)) != 128:
            raise ValueError(
                "C1 SS cross-attention V1 requires 28 heads with head_dim 128, "
                f"got heads={getattr(block, 'num_heads', None)}, head_dim={getattr(block, 'head_dim', None)}."
            )
        copy_linear_weight(block.q_proj, q_weight)
        copy_linear_weight(block.k_proj, k_weight)
        copy_linear_weight(block.v_proj, kv_weight)
        copy_linear_weight(block.out_proj, out_weight)
        set_norm_defaults(block.visual_norm)
        set_norm_defaults(block.geometry_norm)
        block.set_c1_state(enabled=True, qk_scale=1.0, residual_gain=0.0)


def apply_vlm3r_c1(fusion: nn.Module, *, qk_basis_mode: str = "shared_canonical") -> None:
    """Replace all VLM3R from-scratch fusion affine maps with C1 maps."""
    mode = validate_qk_basis_mode(qk_basis_mode)
    # The native class deliberately does not expose dimension attributes;
    # infer them from the concrete affine transforms so this validates the
    # active implementation instead of duplicating a config convention.
    clip_dim = int(fusion.clip_query_proj.in_features)
    spatial_dim = int(fusion.spatial_encoder_key_proj.in_features)
    attn_dim = int(fusion.clip_query_proj.out_features)
    if (clip_dim, spatial_dim, attn_dim) != (1152, 768, 1152):
        raise ValueError(
            "C1 VLM3R is defined for visual=1152, CUT3R=768, attention=1152, "
            f"got {(clip_dim, spatial_dim, attn_dim)}."
        )
    q_weight = canonical_square(attn_dim)
    kv_weight = canonical_linear_weight(spatial_dim, attn_dim)
    k_weight = role_offset_weight(kv_weight) if mode == "role_offset" else kv_weight
    copy_linear_weight(fusion.clip_query_proj, q_weight)
    copy_linear_weight(fusion.spatial_encoder_key_proj, k_weight)
    copy_linear_weight(fusion.spatial_encoder_value_proj, kv_weight)
    set_mha_identity(fusion.cross_attention)
    copy_linear_weight(fusion.out_proj, q_weight.transpose(0, 1).contiguous())
    set_norm_defaults(fusion.clip_norm)
    set_norm_defaults(fusion.spatial_encoder_norm)
    set_norm_defaults(fusion.out_norm)
    fusion.set_c1_state(enabled=True, qk_scale=1.0, residual_gain=0.0)


def apply_geo_rope_fusion_c1(
    fusion: nn.Module,
    *,
    qk_basis_mode: str = "shared_canonical",
    qk_scale: float = 1.0,
    residual_gain: float = 0.0,
    gate_q: float = 1.0,
    gate_k: float = 1.0,
) -> None:
    """Canonicalize ``GeoRoPEFusionCrossAttention`` from the C1 VLM3R maps.

    The GeoRoPE implementation replaces ``nn.MultiheadAttention`` with four
    explicit affine maps.  At zero GeoRoPE gates these maps are algebraically
    equivalent to the packed C1 MHA: Q/K carry the frozen C1 logit scale, V/O
    are identities, and the outer branch retains the frozen C1 residual gain.
    This function deliberately sets no learned post-SFT values.
    """
    mode = validate_qk_basis_mode(qk_basis_mode)
    required = (
        "clip_query_proj", "spatial_encoder_key_proj", "spatial_encoder_value_proj",
        "attn_query_proj", "attn_key_proj", "attn_value_proj", "attn_out_proj",
        "out_proj", "clip_norm", "spatial_encoder_norm", "out_norm",
        "geo_rope_fusion_gate_q", "geo_rope_fusion_gate_k", "set_c1_state",
    )
    missing = [name for name in required if not hasattr(fusion, name)]
    if missing:
        raise ValueError(f"C1 GeoRoPE fusion topology is incomplete: missing {missing}.")
    clip_dim = int(fusion.clip_query_proj.in_features)
    spatial_dim = int(fusion.spatial_encoder_key_proj.in_features)
    attn_dim = int(fusion.clip_query_proj.out_features)
    if (clip_dim, spatial_dim, attn_dim) != (1152, 768, 1152):
        raise ValueError(
            "C1 GeoRoPE fusion is defined for visual=1152, CUT3R=768, attention=1152, "
            f"got {(clip_dim, spatial_dim, attn_dim)}."
        )
    if int(getattr(fusion, "num_heads", -1)) != 18 or int(getattr(fusion, "head_dim", -1)) != 64:
        raise ValueError(
            "C1 GeoRoPE fusion requires 18 heads with head_dim=64, "
            f"got heads={getattr(fusion, 'num_heads', None)}, head_dim={getattr(fusion, 'head_dim', None)}."
        )
    q_weight = canonical_square(attn_dim)
    kv_weight = canonical_linear_weight(spatial_dim, attn_dim)
    k_weight = role_offset_weight(kv_weight) if mode == "role_offset" else kv_weight
    copy_linear_weight(fusion.clip_query_proj, q_weight)
    copy_linear_weight(fusion.spatial_encoder_key_proj, k_weight)
    copy_linear_weight(fusion.spatial_encoder_value_proj, kv_weight)
    identity = torch.eye(attn_dim, dtype=torch.float32)
    copy_linear_weight(fusion.attn_query_proj, float(qk_scale) * identity)
    copy_linear_weight(fusion.attn_key_proj, float(qk_scale) * identity)
    copy_linear_weight(fusion.attn_value_proj, identity)
    copy_linear_weight(fusion.attn_out_proj, identity)
    copy_linear_weight(fusion.out_proj, q_weight.transpose(0, 1).contiguous())
    set_norm_defaults(fusion.clip_norm)
    set_norm_defaults(fusion.spatial_encoder_norm)
    set_norm_defaults(fusion.out_norm)
    with torch.no_grad():
        fusion.geo_rope_fusion_gate_q.fill_(float(gate_q))
        fusion.geo_rope_fusion_gate_k.fill_(float(gate_k))
    fusion.set_c1_state(enabled=True, residual_gain=float(residual_gain))


def apply_visual_geo_rope_c1(module: nn.Module, *, qk_basis_mode: str = "shared_canonical") -> None:
    """Canonicalize the representation-changing visual GeoRoPE block only.

    Auxiliary geometry heads are intentionally outside this function: they do
    not feed the frozen forward representation.  Each visual projection is a
    deterministic C1 structured map; the two native residual gates are left
    for the single-site unlabeled calibration.
    """
    mode = validate_qk_basis_mode(qk_basis_mode)
    layers = getattr(module, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != 1:
        raise ValueError("C1 Visual GeoRoPE requires exactly one geometry projection layer.")
    layer = layers[0]
    required = ("q_proj", "k_proj", "v_proj", "out_proj", "norm_attn", "norm_ffn", "ffn", "gamma_attn", "gamma_ffn")
    missing = [name for name in required if not hasattr(layer, name)]
    if missing:
        raise ValueError(f"C1 Visual GeoRoPE topology is incomplete: missing {missing}.")
    hidden = int(layer.q_proj.in_features)
    if hidden != 1152 or int(layer.q_proj.out_features) != hidden:
        raise ValueError(f"C1 Visual GeoRoPE requires 1152-dimensional visual maps, got {hidden}.")
    if int(getattr(layer, "num_heads", -1)) != 16 or int(getattr(layer, "head_dim", -1)) != 72:
        raise ValueError(
            "C1 Visual GeoRoPE requires 16 heads with head_dim=72, "
            f"got heads={getattr(layer, 'num_heads', None)}, head_dim={getattr(layer, 'head_dim', None)}."
        )
    if not isinstance(layer.ffn, nn.Sequential) or len(layer.ffn) != 3:
        raise ValueError("C1 Visual GeoRoPE requires the native two-linear GELU FFN.")
    ffn_in, ffn_out = layer.ffn[0], layer.ffn[2]
    if not isinstance(ffn_in, nn.Linear) or not isinstance(ffn_out, nn.Linear):
        raise ValueError("C1 Visual GeoRoPE requires linear FFN endpoints.")
    q_weight = canonical_square(hidden)
    k_weight = role_offset_weight(q_weight) if mode == "role_offset" else q_weight
    copy_linear_weight(layer.q_proj, q_weight)
    copy_linear_weight(layer.k_proj, k_weight)
    copy_linear_weight(layer.v_proj, q_weight)
    copy_linear_weight(layer.out_proj, q_weight.transpose(0, 1).contiguous())
    copy_linear_weight(ffn_in, canonical_linear_weight(hidden, int(ffn_in.out_features)))
    copy_linear_weight(ffn_out, canonical_linear_weight(int(ffn_out.in_features), hidden))
    set_norm_defaults(layer.norm_attn)
    set_norm_defaults(layer.norm_ffn)


def apply_c1_calibration_artifact(model: nn.Module, artifact: dict[str, Any]) -> None:
    """Regenerate C1 weights and load fixed scalar calibration values.

    Artifacts deliberately contain no dense matrix values.  This makes a
    mismatched topology fail early instead of accepting accidental trained or
    random fusion weights.
    """
    version = artifact.get("canonicalization_scheme_version")
    if version != SCHEME_VERSION:
        raise ValueError(
            f"C1 calibration artifact scheme mismatch: expected {SCHEME_VERSION!r}, got {version!r}."
        )
    architecture = str(artifact.get("architecture", "")).strip().lower()
    qk_basis_mode = validate_qk_basis_mode(artifact.get("qk_basis_mode", "shared_canonical"))
    base = model.get_model()
    if architecture in {"spatialstack_add", "spatialstack_cross_attn_v1"}:
        merger = base.get_cut3r_spatialstack_merger()
        if merger is None:
            raise ValueError("C1 artifact expects a SpatialStack merger, but the loaded model has none.")
        expected_fusion = "add" if architecture == "spatialstack_add" else "cross_attn"
        if str(getattr(merger, "fusion_type", "")).lower() != expected_fusion:
            raise ValueError(
                f"C1 artifact architecture={architecture!r} requires fusion_type={expected_fusion!r}, "
                f"got {getattr(merger, 'fusion_type', None)!r}."
            )
        apply_spatialstack_c1(merger, qk_basis_mode=qk_basis_mode)
        layer_values = artifact.get("layers", {})
        if architecture == "spatialstack_add":
            # Additive branches are keyed by CUT3R decoder source layer
            # (e.g. 6/9/12), whereas C1 artifacts are intentionally keyed by
            # the LLM injection site (0/1/2).  Calibrated scalar values belong
            # to the latter because r0 and residual ratios are measured at
            # their LLM injection locations.
            module_by_injection = {
                str(llm_layer): merger.branches[str(cut3r_layer)]
                for llm_layer, cut3r_layer in merger.layer_map.items()
            }
        else:
            module_by_injection = {str(layer_key): module for layer_key, module in merger.cross_attn_blocks.items()}
        for layer_key, module in module_by_injection.items():
            values = layer_values.get(str(layer_key))
            if not isinstance(values, dict):
                raise ValueError(f"C1 artifact lacks scalar values for injection layer {layer_key}.")
            if architecture == "spatialstack_add":
                module.set_c1_state(
                    enabled=True,
                    pre_gelu_scale=float(values["s_pre"]),
                    residual_gain=float(values["residual_gain"]),
                )
            else:
                module.set_c1_state(
                    enabled=True,
                    qk_scale=float(values["s_qk"]),
                    residual_gain=float(values["residual_gain"]),
                )
        return
    if architecture == "vlm3r":
        fusion = base.get_fusion_block()
        values = artifact.get("vlm3r", artifact)
        # GeoRoPE fusion is a topology-preserving expansion of VLM3R MHA.
        # The ordinary C1 artifact supplies its Q/K and residual reference;
        # its later architecture-specific calibration may override only the
        # outer gain, never the deterministic maps.
        if fusion is not None and hasattr(fusion, "geo_rope_fusion_gate_q"):
            apply_geo_rope_fusion_c1(
                fusion,
                qk_basis_mode=qk_basis_mode,
                qk_scale=float(values["s_qk"]),
                residual_gain=float(values["lambda"]),
                gate_q=float(fusion.geo_rope_fusion_gate_q.detach().float().item()),
                gate_k=float(fusion.geo_rope_fusion_gate_k.detach().float().item()),
            )
            return
        if fusion is None and base.get_geometry_aware_projection() is not None:
            apply_visual_geo_rope_c1(base.get_geometry_aware_projection(), qk_basis_mode=qk_basis_mode)
            return
        if fusion is None:
            raise ValueError("C1 artifact expects a VLM3R fusion block, but the loaded model has none.")
        apply_vlm3r_c1(fusion, qk_basis_mode=qk_basis_mode)
        fusion.set_c1_state(
            enabled=True,
            qk_scale=float(values["s_qk"]),
            residual_gain=float(values["lambda"]),
        )
        return
    raise ValueError(f"Unsupported C1 artifact architecture: {architecture!r}.")


def apply_geometry_c1_calibration_artifact(model: nn.Module, artifact: dict[str, Any]) -> None:
    """Apply a solved single-site C1 activation without changing dense maps."""
    if artifact.get("schema_version") != "c1_geometry_activation_v1":
        raise ValueError("Unsupported geometry C1 activation artifact schema.")
    architecture = str(artifact.get("architecture", "")).strip().lower()
    activation = artifact.get("activation")
    if not isinstance(activation, dict):
        raise ValueError("Geometry C1 activation artifact lacks an activation object.")
    base = model.get_model()
    if architecture == "geo_rope_fusion":
        fusion = base.get_fusion_block()
        if fusion is None or not hasattr(fusion, "geo_rope_fusion_gate_q"):
            raise ValueError("GeoRoPE C1 activation artifact requires GeoRoPE fusion.")
        gate_q = float(activation.get("gate_q", float("nan")))
        gate_k = float(activation.get("gate_k", float("nan")))
        lam = float(activation.get("lambda_geo", float("nan")))
        if not (math.isfinite(gate_q) and math.isfinite(gate_k) and math.isfinite(lam)):
            raise ValueError("GeoRoPE C1 activation contains non-finite gate or lambda values.")
        if gate_q != 1.0 or gate_k != 1.0:
            raise ValueError("Official C1 GeoRoPE requires fixed full gates g_q=g_k=1.")
        with torch.no_grad():
            fusion.geo_rope_fusion_gate_q.fill_(gate_q)
            fusion.geo_rope_fusion_gate_k.fill_(gate_k)
        fusion.set_c1_state(enabled=True, residual_gain=lam)
        return
    if architecture == "visual_geo_rope":
        module = base.get_geometry_aware_projection()
        if module is None or not isinstance(getattr(module, "layers", None), nn.ModuleList) or len(module.layers) != 1:
            raise ValueError("Visual GeoRoPE C1 activation artifact requires one geometry projection layer.")
        gamma = float(activation.get("gamma_c1", float("nan")))
        if not math.isfinite(gamma):
            raise ValueError("Visual GeoRoPE C1 activation contains a non-finite gamma.")
        with torch.no_grad():
            module.layers[0].gamma_attn.fill_(gamma)
            module.layers[0].gamma_ffn.fill_(gamma)
        return
    raise ValueError(f"Unsupported geometry C1 architecture: {architecture!r}.")
