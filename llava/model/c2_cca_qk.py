"""Forward-only C2 CCA-QK calibration primitives for SpatialStack V1.

C2 deliberately composes affine Q/K maps on top of C1.  It never touches V or
the output projection and has no trainable state.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .c1_structured_isometry import SCHEME_VERSION, apply_spatialstack_c1, validate_qk_basis_mode


ARTIFACT_SCHEMA = "c2_cca_qk_spatialstack_v1"
_EPS64 = torch.finfo(torch.float64).eps


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{name} must be finite and positive, got {value}.")
    return value


def _linear_output_to_heads(value: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    if value.ndim != 3 or int(value.shape[-1]) != int(heads) * int(head_dim):
        raise RuntimeError(
            "C2 expected projected Q/K values shaped [batch,tokens,heads*head_dim], got "
            f"{tuple(value.shape)} for heads={heads}, head_dim={head_dim}."
        )
    return value.detach().reshape(-1, int(heads), int(head_dim))


@dataclass
class _Moment:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        self.count += int(value.numel())
        self.total += float(value.sum().item())
        self.total_sq += float(value.square().sum().item())

    def std(self, name: str) -> float:
        if self.count <= 1:
            raise RuntimeError(f"C2 {name} needs at least two logits.")
        variance = max(self.total_sq / self.count - (self.total / self.count) ** 2, 0.0)
        return _finite_positive(math.sqrt(variance), name)


class PairedQKObserver:
    """Observe native V1 Q/K outputs without changing the forward function."""

    def __init__(self, block: nn.Module, *, collect_cca: bool, collect_logits: bool, chunk_size: int = 1024):
        self.block = block
        self.heads = int(getattr(block, "num_heads"))
        self.head_dim = int(getattr(block, "head_dim"))
        self.collect_cca = bool(collect_cca)
        self.collect_logits = bool(collect_logits)
        self.chunk_size = int(chunk_size)
        self._pending_q: torch.Tensor | None = None
        self.handles = []
        self.pair_count = 0
        self.pre_q_moment = _Moment()
        self.logit_moment = _Moment()
        self.count = torch.zeros(self.heads, dtype=torch.long)
        self.sum_x = torch.zeros(self.heads, self.head_dim, dtype=torch.float64)
        self.sum_y = torch.zeros_like(self.sum_x)
        self.sum_xx = torch.zeros(self.heads, self.head_dim, self.head_dim, dtype=torch.float64)
        self.sum_yy = torch.zeros_like(self.sum_xx)
        self.sum_xy = torch.zeros_like(self.sum_xx)

    def __enter__(self):
        self.handles = [
            self.block.q_proj.register_forward_pre_hook(self._on_q_input),
            self.block.q_proj.register_forward_hook(self._on_q),
            self.block.k_proj.register_forward_hook(self._on_k),
        ]
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self._pending_q is not None:
            raise RuntimeError("C2 saw a q_proj output without its matching k_proj output.")

    def _on_q_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            self.pre_q_moment.add(inputs[0])

    def _on_q(self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if self._pending_q is not None:
            raise RuntimeError("C2 expected q_proj/k_proj calls to be paired serially.")
        self._pending_q = output.detach()

    def _on_k(self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        if self._pending_q is None:
            raise RuntimeError("C2 saw k_proj before q_proj.")
        q = self._pending_q
        self._pending_q = None
        k = output.detach()
        qh = _linear_output_to_heads(q, self.heads, self.head_dim)
        kh = _linear_output_to_heads(k, self.heads, self.head_dim)
        if tuple(qh.shape) != tuple(kh.shape):
            raise RuntimeError(
                "C2 requires same-frame/same-position Q/K pairs; native projected shapes differ: "
                f"q={tuple(qh.shape)}, k={tuple(kh.shape)}."
            )
        self.pair_count += int(qh.shape[0])
        if self.collect_cca:
            self._accumulate_cca(qh, kh)
        if self.collect_logits:
            self._accumulate_logits(q, k)

    def _accumulate_cca(self, qh: torch.Tensor, kh: torch.Tensor) -> None:
        # Float64 matrix products run in bounded chunks on the device that owns
        # the active fusion block; only sufficient statistics move to CPU.
        for start in range(0, int(qh.shape[0]), self.chunk_size):
            end = min(start + self.chunk_size, int(qh.shape[0]))
            x = qh[start:end].permute(1, 0, 2).to(dtype=torch.float64)
            y = kh[start:end].permute(1, 0, 2).to(dtype=torch.float64)
            n = int(x.shape[1])
            self.count += n
            self.sum_x += x.sum(dim=1).cpu()
            self.sum_y += y.sum(dim=1).cpu()
            self.sum_xx += torch.matmul(x.transpose(1, 2), x).cpu()
            self.sum_yy += torch.matmul(y.transpose(1, 2), y).cpu()
            self.sum_xy += torch.matmul(x.transpose(1, 2), y).cpu()

    def _accumulate_logits(self, q_raw: torch.Tensor, k_raw: torch.Tensor) -> None:
        # Each native V1 call contains a batch of frames.  Attention logits are
        # all patch-to-patch pairs within those frames, matching C1's statistic.
        q = q_raw.reshape(q_raw.shape[0], q_raw.shape[1], self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_raw.reshape(k_raw.shape[0], k_raw.shape[1], self.heads, self.head_dim).permute(0, 2, 1, 3)
        for start in range(0, int(q.shape[2]), self.chunk_size):
            logits = torch.matmul(q[:, :, start : start + self.chunk_size], k.transpose(-2, -1))
            logits = logits / math.sqrt(float(self.head_dim))
            self.logit_moment.add(logits)

    def fit_cca(self, ridge_relative: float) -> dict[str, torch.Tensor]:
        ridge_relative = _finite_positive(ridge_relative, "C2 CCA ridge_relative")
        if not bool(torch.all(self.count > self.head_dim).item()):
            raise RuntimeError(
                f"C2 needs more than d_head={self.head_dim} paired observations per head, got {self.count.tolist()}."
            )
        means_x = self.sum_x / self.count[:, None]
        means_y = self.sum_y / self.count[:, None]
        denom = (self.count - 1).to(dtype=torch.float64)[:, None, None]
        cov_xx = (self.sum_xx - self.count[:, None, None] * means_x[:, :, None] * means_x[:, None, :]) / denom
        cov_yy = (self.sum_yy - self.count[:, None, None] * means_y[:, :, None] * means_y[:, None, :]) / denom
        cov_xy = (self.sum_xy - self.count[:, None, None] * means_x[:, :, None] * means_y[:, None, :]) / denom
        eye = torch.eye(self.head_dim, dtype=torch.float64).expand(self.heads, -1, -1)
        ridge_x = ridge_relative * cov_xx.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / self.head_dim
        ridge_y = ridge_relative * cov_yy.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / self.head_dim
        if not bool(torch.isfinite(ridge_x).all() and torch.isfinite(ridge_y).all()):
            raise RuntimeError("C2 CCA ridge values are non-finite.")
        reg_x = cov_xx + ridge_x[:, None, None] * eye
        reg_y = cov_yy + ridge_y[:, None, None] * eye
        inv_x = _symmetric_inverse_sqrt(reg_x, "SigmaXX")
        inv_y = _symmetric_inverse_sqrt(reg_y, "SigmaYY")
        t = inv_x @ cov_xy @ inv_y
        u, rho, vh = torch.linalg.svd(t, full_matrices=False)
        v = vh.transpose(-2, -1)
        _canonicalize_joint_signs(u, v)
        if not bool(torch.isfinite(rho).all()) or bool((rho < -1e-8).any()) or bool((rho > 1.0 + 1e-5).any()):
            raise RuntimeError(f"C2 canonical correlations are numerically invalid: {rho.flatten().tolist()[:8]}")
        return {
            "mu_q": means_x,
            "mu_k": means_y,
            "a": inv_x @ u,
            "b": inv_y @ v,
            "canonical_correlations": rho,
            "ridge_x": ridge_x,
            "ridge_y": ridge_y,
            "pair_count_per_head": self.count.clone(),
        }


def _symmetric_inverse_sqrt(value: torch.Tensor, name: str) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(value)
    max_eigen = values.amax(dim=-1, keepdim=True)
    floor = torch.maximum(max_eigen * 1e-12, torch.full_like(max_eigen, _EPS64))
    if not bool(torch.isfinite(values).all()) or bool((max_eigen <= 0).any()):
        raise RuntimeError(f"C2 {name} eigendecomposition is unusable.")
    adjusted = values.clamp_min(floor)
    return (vectors * adjusted.rsqrt().unsqueeze(-2)) @ vectors.transpose(-2, -1)


def _canonicalize_joint_signs(u: torch.Tensor, v: torch.Tensor) -> None:
    for head in range(int(u.shape[0])):
        for component in range(int(u.shape[-1])):
            column = u[head, :, component]
            sign = 1.0 if float(column[column.abs().argmax()].item()) >= 0.0 else -1.0
            u[head, :, component].mul_(sign)
            v[head, :, component].mul_(sign)


def compose_c2_qk(block: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Compose CCA affine maps with the current deterministic C1 Q/K maps."""
    heads, dim = int(block.num_heads), int(block.head_dim)
    for name, transform_key, mean_key in (("q_proj", "a", "mu_q"), ("k_proj", "b", "mu_k")):
        linear = getattr(block, name)
        transform = state[transform_key].to(dtype=torch.float64)
        mean = state[mean_key].to(dtype=torch.float64)
        if tuple(transform.shape) != (heads, dim, dim) or tuple(mean.shape) != (heads, dim):
            raise ValueError(f"C2 {name} state does not match heads={heads}, head_dim={dim}.")
        base_weight = linear.weight.detach().to(dtype=torch.float64).reshape(heads, dim, linear.in_features)
        weight = torch.matmul(transform.transpose(-2, -1), base_weight).reshape_as(linear.weight)
        bias = torch.matmul(-mean.unsqueeze(1), transform).squeeze(1).reshape(-1)
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise RuntimeError(f"C2 composed {name} parameters are non-finite.")
        with torch.no_grad():
            linear.weight.copy_(weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
            linear.bias.copy_(bias.to(device=linear.bias.device, dtype=linear.bias.dtype))


def c1_stat_contract(reference: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if reference.get("canonicalization_scheme_version") != SCHEME_VERSION:
        raise ValueError("C2 requires a compatible C1 structured-isometry artifact.")
    if str(reference.get("architecture", "")).lower() != "spatialstack_cross_attn_v1":
        raise ValueError("C2 only extends C1 architecture='spatialstack_cross_attn_v1'.")
    qk = reference.get("qk_logit_calibration")
    residual = reference.get("residual_calibration")
    if not isinstance(qk, dict) or not isinstance(residual, dict):
        raise ValueError("C2 requires a regenerated C1 artifact with explicit statistic contracts.")
    if qk.get("statistic") != "population_std_over_all_same_frame_attention_logits" or qk.get("variance") != "population_E_x2_minus_E_x_squared":
        raise ValueError(f"Unsupported C1 QK statistic contract: {qk}")
    if qk.get("qk_scale_application") != "multiply_both_q_and_k":
        raise ValueError(f"Unsupported C1 QK scale contract: {qk}")
    _finite_positive(qk.get("target_std"), "C1 target QK logit std")
    expected_residual = {
        "sample_statistic": "rms_delta_over_rms_pre_injection_hidden_at_visual_tokens",
        "sample_aggregation": "median_over_samples",
        "target": "base_artifact_r0",
        "target_scope": "per_injection_site",
    }
    if any(residual.get(key) != value for key, value in expected_residual.items()):
        raise ValueError(f"Unsupported C1 residual statistic contract: {residual}")
    _finite_positive(reference.get("r0"), "C1 residual target r0")
    return qk, residual


def apply_c2_calibration_artifact(model: nn.Module, artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != ARTIFACT_SCHEMA or not bool(artifact.get("complete", False)):
        raise ValueError("C2 artifact is missing, incompatible, or an incomplete smoke artifact.")
    reference = artifact.get("c1_reference")
    if not isinstance(reference, dict):
        raise ValueError("C2 artifact lacks its C1 statistic reference.")
    c1_stat_contract(reference)
    base = model.get_model()
    merger = base.get_cut3r_spatialstack_merger()
    if merger is None or str(getattr(merger, "fusion_type", "")).lower() != "cross_attn":
        raise ValueError("C2 artifact requires a SpatialStack V1 cross-attention merger.")
    if int(getattr(merger, "cross_attn_heads", -1)) != 28:
        raise ValueError("C2 artifact requires the C1 28-head V1 topology.")
    identifier = artifact.get("model_identifier")
    if not isinstance(identifier, dict):
        raise ValueError("C2 artifact lacks its model/topology compatibility identifier.")
    expected_topology = {
        "fusion_type": str(getattr(merger, "fusion_type", "")),
        "hidden_size": int(getattr(merger, "hidden_size", -1)),
        "feature_dim": int(getattr(merger, "feature_dim", -1)),
        "cross_attn_heads": int(getattr(merger, "cross_attn_heads", -1)),
        "llm_layers": [int(value) for value in getattr(merger, "llm_layers", [])],
        "cut3r_layers": [int(value) for value in getattr(merger, "cut3r_layers", [])],
    }
    if any(identifier.get(key) != value for key, value in expected_topology.items()):
        raise ValueError(
            "C2 artifact topology does not match the loaded SpatialStack merger: "
            f"artifact={identifier}, loaded={expected_topology}."
        )
    expected_config_hash = identifier.get("model_config_sha256")
    source_path = getattr(model, "_pre_sft_source_path", None)
    if not isinstance(expected_config_hash, str) or not source_path:
        raise ValueError("C2 artifact/model lacks a base-model config identity.")
    loaded_config = Path(source_path).resolve() / "config.json"
    if not loaded_config.is_file() or _sha256_file(loaded_config) != expected_config_hash:
        raise ValueError("C2 artifact was calibrated for a different base-model config.json.")
    qk_mode = validate_qk_basis_mode(reference.get("qk_basis_mode", "shared_canonical"))
    apply_spatialstack_c1(merger, qk_basis_mode=qk_mode)
    layers = artifact.get("layers")
    if not isinstance(layers, dict) or set(layers) != set(merger.cross_attn_blocks.keys()):
        raise ValueError("C2 artifact injection-layer IDs do not match the loaded merger.")
    for layer, block in merger.cross_attn_blocks.items():
        values = layers[layer]
        if not isinstance(values, dict):
            raise ValueError(f"C2 artifact lacks state for layer {layer}.")
        v_before = block.v_proj.state_dict()
        o_before = block.out_proj.state_dict()
        compose_c2_qk(block, values)
        if any(not torch.equal(value, block.v_proj.state_dict()[key]) for key, value in v_before.items()):
            raise RuntimeError("C2 unexpectedly modified V.")
        if any(not torch.equal(value, block.out_proj.state_dict()[key]) for key, value in o_before.items()):
            raise RuntimeError("C2 unexpectedly modified O.")
        block.set_c1_state(
            enabled=True,
            qk_scale=float(values["qk_scale"]),
            residual_gain=float(values["residual_gain"]),
            collect_diagnostics=False,
        )
        for parameter in block.parameters():
            parameter.requires_grad_(False)
