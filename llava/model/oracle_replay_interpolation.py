"""Small, eval-only helpers for oracle replay and residual interpolation.

The helpers deliberately allocate new payload tensors for every construction.
They make it possible to prove that replay parity is value parity, not an
aliasing artefact from reusing a previously constructed residual dictionary.
"""

from __future__ import annotations

import copy
from typing import Dict, Mapping, Sequence

import torch


def effective_residual_scales(merger) -> Dict[int, float]:
    """Return the loaded effective scale for every configured LLM layer.

    ``Cut3RSpatialStackMerger`` owns the authoritative value after loading the
    teacher checkpoint/config.  This function intentionally never supplies a
    numeric default: absence is a checkpoint incompatibility.
    """
    if merger is None or not hasattr(merger, "residual_scale"):
        raise RuntimeError("Teacher SpatialStack merger does not expose a loaded residual_scale.")
    value = getattr(merger, "residual_scale")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"Teacher residual_scale must be scalar, got {tuple(value.shape)}.")
        value = value.detach().float().item()
    value = float(value)
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError(f"Teacher residual_scale is nonfinite: {value!r}.")
    layers = tuple(int(layer) for layer in getattr(merger, "llm_layers", ()))
    if not layers:
        raise RuntimeError("Teacher SpatialStack merger has no configured LLM layers.")
    return {layer: value for layer in layers}


def _clone_features(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_features(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_features(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_features(item) for item in value)
    return copy.deepcopy(value)


def _payload_storage_ids(payload: Mapping[int, torch.Tensor]) -> Dict[int, int]:
    result = {}
    for layer, tensor in payload.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Residual payload for layer {layer} is not a tensor.")
        result[int(layer)] = int(tensor.untyped_storage().data_ptr())
    return result


def build_oracle_payload(merger, spatial_features, visual_metadata, *, seq_len: int, device, dtype):
    """Build a fresh oracle payload and return loaded-scale provenance."""
    payload = merger(
        _clone_features(spatial_features),
        _clone_features(visual_metadata),
        seq_len=int(seq_len), device=device, dtype=dtype,
    )
    if isinstance(payload, Mapping):
        items = payload.items()
    elif hasattr(payload, "_payloads"):
        items = payload._payloads.items()
    else:
        raise RuntimeError("Oracle SpatialStack merger did not return dense residual payloads.")
    copied = {int(layer): tensor.clone() for layer, tensor in items}
    return copied, {"effective_residual_scales": effective_residual_scales(merger), "storage": _payload_storage_ids(copied)}


def build_independent_oracle_payloads(merger, spatial_features, visual_metadata, *, seq_len: int, device, dtype):
    """Build two independent payloads for original-vs-replay tensor parity."""
    oracle, oracle_info = build_oracle_payload(
        merger, spatial_features, visual_metadata, seq_len=seq_len, device=device, dtype=dtype
    )
    replay, replay_info = build_oracle_payload(
        merger, spatial_features, visual_metadata, seq_len=seq_len, device=device, dtype=dtype
    )
    if set(oracle) != set(replay):
        raise RuntimeError(f"Oracle/replay layer mismatch: {sorted(oracle)} vs {sorted(replay)}.")
    aliases = []
    for layer in oracle:
        if oracle[layer] is replay[layer] or oracle_info["storage"][layer] == replay_info["storage"][layer]:
            aliases.append(int(layer))
    if aliases:
        raise RuntimeError(f"Oracle and replay payloads unexpectedly share tensor storage for layers {aliases}.")
    return oracle, replay, {"oracle": oracle_info, "replay": replay_info}


def interpolate_payloads(teacher: Mapping[int, torch.Tensor], predicted: Mapping[int, torch.Tensor], beta: float):
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"spatialstack_residual_beta must be in [0, 1], got {beta}.")
    if set(teacher) != set(predicted):
        raise RuntimeError(f"Teacher/predicted layer mismatch: {sorted(teacher)} vs {sorted(predicted)}.")
    result = {}
    for layer, target in teacher.items():
        prediction = predicted[int(layer)]
        if tuple(target.shape) != tuple(prediction.shape):
            raise RuntimeError(f"Residual shape mismatch at layer {layer}: {tuple(target.shape)} vs {tuple(prediction.shape)}.")
        if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Nonfinite residual values at layer {layer}.")
        if beta == 0.0:
            result[int(layer)] = target
        elif beta == 1.0:
            result[int(layer)] = prediction
        else:
            result[int(layer)] = ((1.0 - beta) * target.float() + beta * prediction.float()).to(dtype=target.dtype)
    return result
